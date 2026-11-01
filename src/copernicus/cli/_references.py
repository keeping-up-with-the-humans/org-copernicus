"""The ``references`` commands: write reference texts for the poles of a rubric."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from functools import partial
from pathlib import Path
from typing import Annotated, Any

import typer

from copernicus.cli._backends import BackendKind, build_backend, placeholder_reference_responder
from copernicus.cli._code import BACKEND_OPTION_HELP
from copernicus.cli._logging import Verbose, cli_logging
from copernicus.cli._options import cli_errors, parse_backend_options
from copernicus.cli._output import check_table_output, format_rows
from copernicus.errors import BackendError, SettingsError
from copernicus.llm import Backend
from copernicus.references import (
    DeferredWriter,
    GeneratorSettings,
    ReferenceSet,
    TokenEstimate,
    estimate_tokens,
    generate_references,
)
from copernicus.references.estimate import COMPLETION_TOKENS_PER_TEXT
from copernicus.references.generator import Writer
from copernicus.rubric import Rubric, load_rubric

logger = logging.getLogger(__name__)

__all__ = ["format_estimate", "format_reference_summary", "references_app"]

DEFAULTS = GeneratorSettings()
BACKEND_HELP = (
    "Where the writer models run. 'fake' answers with placeholder sentences and is for "
    "testing the pipeline only."
)
CONCURRENCY_HELP = (
    "First attempts sent to a writer together; retries go one at a time. The openai backend "
    "sends them in parallel; vLLM generates them in one engine call."
)
DRY_RUN_HELP = (
    "Print the number of calls and the token estimate and stop. No backend is built, so "
    "backend options are not checked."
)
RESUME_HELP = (
    "Resume file (.parquet or .csv): every finished call is written to it at once, and a "
    "rerun with the same file makes only the calls it lacks."
)
PRICE_HELP = "US dollars per million {kind} tokens; with both prices the estimate shows a cost."

references_app = typer.Typer(
    help="Write reference texts for the poles of a rubric.", no_args_is_help=True
)


def format_reference_summary(reference_set: ReferenceSet, out: Path) -> str:
    """Format what a generation run produced as a short aligned table.

    Args:
        reference_set: The generated set.
        out: The file it was saved to.

    Returns:
        Rows for the number of texts, the texts per cell, the missing texts, the texts the
        leakage screen excluded, the repeated texts, whether the poles are balanced (with
        each problem on a line of its own), the attempts and the prompt and completion
        tokens of all writers, and the file.
    """
    summary = reference_set.summary()
    per_cell = set(summary["texts_per_cell"].values())
    spread = str(per_cell.pop()) if len(per_cell) == 1 else f"{min(per_cell)} to {max(per_cell)}"
    balance = summary["balance"]
    usage = summary["usage"].values()
    rows = [
        ("texts", str(len(reference_set.frame))),
        ("per cell", spread),
        ("shortfalls", str(sum(summary["shortfalls"].values()))),
        ("screened out", str(summary["screened_out"]["total"])),
        ("duplicates", str(summary["duplicates"])),
        ("balanced", "yes" if balance["balanced"] and balance["complete"] else "no"),
        *(("problem", problem) for problem in balance["problems"]),
        ("attempts", str(sum(each["attempts"] for each in usage))),
        ("prompt tokens", str(sum(each["prompt_tokens"] for each in usage))),
        ("completion tokens", str(sum(each["completion_tokens"] for each in usage))),
        ("file", str(out)),
    ]
    return format_rows(rows)


def format_estimate(
    estimate: TokenEstimate, price_input: float | None, price_output: float | None
) -> str:
    """Format the plan of a run: its calls, its estimated tokens, and their cost if priced.

    Args:
        estimate: The token estimate of the run.
        price_input: US dollars per million prompt tokens, or ``None``.
        price_output: US dollars per million completion tokens, or ``None``.

    Returns:
        One ``label: value`` line each for the planned calls and the prompt and completion
        tokens, and one for the cost when both prices are given.

    Raises:
        SettingsError: If only one of the two prices is given.
    """
    if (price_input is None) != (price_output is None):
        raise SettingsError("give both --price-input and --price-output, or neither")
    lines = [
        f"planned calls: {estimate.calls}",
        f"estimated prompt tokens: {estimate.prompt_tokens} ({estimate.counter})",
        f"estimated completion tokens: {estimate.completion_tokens} "
        f"({COMPLETION_TOKENS_PER_TEXT} per requested text)",
    ]
    if price_input is not None and price_output is not None:
        lines.append(f"estimated cost (USD): {estimate.cost(price_input, price_output):.4f}")
    return "\n".join(lines)


def _writers(
    kind: BackendKind,
    models: list[str],
    rubric: Rubric,
    settings: GeneratorSettings,
    *,
    base_url: str | None,
    timeout: float | None,
    options: Mapping[str, Any],
) -> list[Writer]:
    """Build the writers of a run; vLLM writers are deferred, so one engine is loaded at a time.

    The concurrency of the run becomes the parallel requests of an OpenAI-compatible writer,
    unless ``--backend-option concurrency=...`` sets them.
    """
    arguments = dict(options)
    if kind is BackendKind.OPENAI and "concurrency" not in arguments:
        arguments["concurrency"] = settings.concurrency
    responder = placeholder_reference_responder(settings.texts_per_call)

    def build(name: str) -> Backend:
        return build_backend(
            kind,
            model=name,
            base_url=base_url,
            timeout=timeout,
            options=arguments,
            rubric=rubric,
            fake_script=None,
            fake_responder=responder,
        )

    if kind is BackendKind.VLLM:
        return [DeferredWriter(name, partial(build, name)) for name in models]
    return [build(name) for name in models]


@references_app.command("generate")
def generate(
    rubric: Annotated[
        Path, typer.Option("--rubric", exists=True, dir_okay=False, help="Rubric JSON file.")
    ],
    out: Annotated[
        Path, typer.Option("--out", help="Reference texts, .parquet or .csv (and a .json beside).")
    ],
    model: Annotated[
        list[str], typer.Option("--model", help="Writer model name; repeat for several writers.")
    ],
    backend: Annotated[
        BackendKind, typer.Option("--backend", help=BACKEND_HELP)
    ] = BackendKind.OPENAI,
    base_url: Annotated[
        str | None, typer.Option("--base-url", help="OpenAI-compatible endpoint (openai).")
    ] = None,
    style: Annotated[
        str, typer.Option("--style", help="'neutral' leaves the medium open; 'register' names it.")
    ] = DEFAULTS.style,
    register_description: Annotated[
        str | None,
        typer.Option(
            "--register-description",
            help="Instruction sentence naming the format of the texts; required with "
            "--style register, refused otherwise.",
        ),
    ] = None,
    texts_per_call: Annotated[
        int, typer.Option("--texts-per-call", help="Texts requested, and required, per call.")
    ] = DEFAULTS.texts_per_call,
    calls_per_cell: Annotated[
        int, typer.Option("--calls-per-cell", help="Calls per writer and cell.")
    ] = DEFAULTS.calls_per_cell,
    seed: Annotated[int, typer.Option("--seed", help="Base seed of the calls.")] = DEFAULTS.seed,
    temperature: Annotated[
        float, typer.Option("--temperature", help="Sampling temperature, 0 to 2.")
    ] = DEFAULTS.temperature,
    max_new_tokens: Annotated[
        int, typer.Option("--max-new-tokens", help="Upper bound on the tokens of an answer.")
    ] = DEFAULTS.max_new_tokens,
    max_retries: Annotated[
        int, typer.Option("--max-retries", help="Attempts per call, counting the first.")
    ] = DEFAULTS.max_retries,
    concurrency: Annotated[
        int, typer.Option("--concurrency", help=CONCURRENCY_HELP)
    ] = DEFAULTS.concurrency,
    resume: Annotated[Path | None, typer.Option("--resume", help=RESUME_HELP)] = None,
    price_input: Annotated[
        float | None, typer.Option("--price-input", help=PRICE_HELP.format(kind="prompt"))
    ] = None,
    price_output: Annotated[
        float | None, typer.Option("--price-output", help=PRICE_HELP.format(kind="completion"))
    ] = None,
    no_screen: Annotated[
        bool, typer.Option("--no-screen", help="Skip the leakage screen of the texts.")
    ] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run", help=DRY_RUN_HELP)] = False,
    timeout: Annotated[
        float | None,
        typer.Option("--timeout", help="Seconds before an HTTP request times out (openai)."),
    ] = None,
    backend_option: Annotated[
        list[str] | None, typer.Option("--backend-option", help=BACKEND_OPTION_HELP)
    ] = None,
    verbose: Verbose = False,
) -> None:
    """Write reference texts for every pole of the construct and of each attribute.

    The number of calls and the estimated tokens (and their cost, with both prices) are
    printed first, and --dry-run stops there. vLLM writers are loaded one after the other.
    """
    options = parse_backend_options(backend_option or [])
    with cli_logging(verbose), cli_errors():
        check_table_output(out)
        if resume is not None and resume.resolve() == out.resolve():
            raise SettingsError("--resume must name another file than --out")
        loaded_rubric = load_rubric(rubric)
        settings = GeneratorSettings(
            style=style,  # type: ignore[arg-type]
            register_description=register_description,
            texts_per_call=texts_per_call,
            calls_per_cell=calls_per_cell,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            seed=seed,
            max_retries=max_retries,
            concurrency=concurrency,
        )
        estimate = estimate_tokens(loaded_rubric, settings, n_writers=len(model))
        typer.echo(format_estimate(estimate, price_input, price_output))
        if dry_run:
            return
        writers = _writers(
            backend,
            model,
            loaded_rubric,
            settings,
            base_url=base_url,
            timeout=timeout,
            options=options,
        )
        try:
            reference_set = generate_references(
                loaded_rubric, writers, settings, screen=not no_screen, resume=resume
            )
        except BackendError:
            # The calls finished before the failure are in the resume file; say where.
            if resume is not None:
                typer.echo(format_rows([("resume file", str(resume))]))
            raise
        reference_set.save(out)
        logger.info("wrote %d reference texts to %s", len(reference_set.frame), out)
        typer.echo(format_reference_summary(reference_set, out))
