"""The ``code`` commands: run the two-stage or the one-stage LLM coder on a text table."""

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Literal

import typer

from copernicus.cli._backends import BackendKind, build_backend
from copernicus.cli._logging import Verbose, cli_logging
from copernicus.cli._options import (
    cli_errors,
    parse_backend_options,
    parse_id_list,
    parse_range,
    parse_temperature,
)
from copernicus.cli._output import format_rows
from copernicus.coders import CodingSettings, CodingSummary, code_texts
from copernicus.coders.store_cells import FAILURE_STATUSES
from copernicus.data import TextTable
from copernicus.errors import BackendError, RunFailureCeilingError
from copernicus.rubric import load_rubric

logger = logging.getLogger(__name__)

__all__ = ["code_app", "format_summary"]

STATUS_ORDER = ("ok", "no_evidence", "too_short", "extraction_failed", "scoring_failed")
CONCURRENCY_HELP = (
    "Attribute evaluations in flight. The openai backend gains throughput from more; "
    "the vLLM backend runs one call at a time, so keep 1 with vLLM."
)
BACKEND_OPTION_HELP = (
    "Constructor argument of the backend as KEY=JSON; repeatable. Example for vllm: "
    "--backend-option 'chat_template_kwargs={\"enable_thinking\": false}' switches thinking "
    'off. Further examples: max_model_len=8192, quantization="awq", extra_body={"top_k": 20}, '
    'or capabilities={"reasoning_effort": "low"} for openai.'
)

code_app = typer.Typer(
    help="Code texts with a rubric into a resumable run store.", no_args_is_help=True
)


def format_summary(summary: CodingSummary) -> str:
    """Format what a coding invocation did as a short aligned table.

    Args:
        summary: The summary returned by :func:`~copernicus.coders.code_texts`, or the
            one a :class:`~copernicus.errors.RunFailureCeilingError` carries.

    Returns:
        One ``label  value`` row per line: texts, runs, attribute calls, the statuses
        with a non-zero count, resumed and retried cells, the failed attribute results the
        store held for the scope when there are any (``stored failed``), tokens, seconds,
        and the store path.
    """
    stored_failed = sum(summary.stored_status_counts[status] for status in FAILURE_STATUSES)
    rows = [
        ("texts", str(summary.n_texts)),
        ("runs", str(summary.n_runs)),
        ("attribute calls", str(summary.n_attribute_calls)),
        *(
            (status, str(summary.status_counts[status]))
            for status in STATUS_ORDER
            if summary.status_counts.get(status, 0)
        ),
        ("resumed", str(summary.resumed)),
        ("retried", str(summary.retried)),
        *((("stored failed", str(stored_failed)),) if stored_failed else ()),
        ("prompt tokens", str(summary.prompt_tokens)),
        ("completion tokens", str(summary.completion_tokens)),
        ("seconds", f"{summary.seconds:.1f}"),
        ("store", str(summary.store_path)),
    ]
    return format_rows(rows)


def _command(method: Literal["two_stage", "one_stage"]) -> Callable[..., None]:
    """Build the ``code`` command of one method."""
    label = method.replace("_", "-")

    def run(
        texts: Annotated[
            Path,
            typer.Option(
                "--texts", exists=True, dir_okay=False, help="CSV, parquet, or JSONL file."
            ),
        ],
        rubric: Annotated[
            Path, typer.Option("--rubric", exists=True, dir_okay=False, help="Rubric JSON file.")
        ],
        out: Annotated[Path, typer.Option("--out", help="Run store (SQLite); resumed if present.")],
        backend: Annotated[
            BackendKind, typer.Option("--backend", help="Where the model runs.")
        ] = BackendKind.OPENAI,
        model: Annotated[
            str | None, typer.Option("--model", help="Model name (openai and vllm).")
        ] = None,
        base_url: Annotated[
            str | None, typer.Option("--base-url", help="OpenAI-compatible endpoint (openai).")
        ] = None,
        fake_script: Annotated[
            Path | None,
            typer.Option(
                "--fake-script",
                exists=True,
                dir_okay=False,
                help="JSON list of answers for --backend fake with a non-toy rubric.",
            ),
        ] = None,
        runs: Annotated[int, typer.Option("--runs", help="Repeated codings of each text.")] = 2,
        temperature: Annotated[
            str, typer.Option("--temperature", help="'auto' or a number in [0, 2].")
        ] = "auto",
        seed: Annotated[int, typer.Option("--seed", help="Seed of wording and sampling.")] = 42,
        text_range: Annotated[
            str | None,
            typer.Option("--range", help="Code only texts A:B, by position in text-id order."),
        ] = None,
        concurrency: Annotated[int, typer.Option("--concurrency", help=CONCURRENCY_HELP)] = 1,
        timeout: Annotated[
            float | None,
            typer.Option("--timeout", help="Seconds before an HTTP request times out (openai)."),
        ] = None,
        backend_option: Annotated[
            list[str] | None, typer.Option("--backend-option", help=BACKEND_OPTION_HELP)
        ] = None,
        enable: Annotated[
            str | None, typer.Option("--enable", help="Comma list of wording variants to pin.")
        ] = None,
        disable: Annotated[
            str | None, typer.Option("--disable", help="Comma list of wording variants to drop.")
        ] = None,
        no_shuffle: Annotated[
            bool, typer.Option("--no-shuffle", help="Keep the marker order of the rubric.")
        ] = False,
        max_failure_share: Annotated[
            float,
            typer.Option(
                "--max-failure-share",
                help=(
                    "Largest share of failed attribute results, judged on this run's calls "
                    "after each batch and on all requested texts and runs, stored ones "
                    "included, at the end."
                ),
            ),
        ] = 0.10,
        retry_failed: Annotated[
            bool,
            typer.Option(
                "--retry-failed", help="Code again the stored cells with a failed attribute."
            ),
        ] = False,
        text_id_column: Annotated[
            str, typer.Option("--text-id-column", help="Column holding the text ids.")
        ] = "text_id",
        text_column: Annotated[
            str, typer.Option("--text-column", help="Column holding the texts.")
        ] = "text",
        group_column: Annotated[
            str | None, typer.Option("--group-column", help="Column holding the group ids.")
        ] = None,
        verbose: Verbose = False,
    ) -> None:
        parsed_temperature = parse_temperature(temperature)
        parsed_range = parse_range(text_range) if text_range is not None else None
        options = parse_backend_options(backend_option or [])
        with cli_logging(verbose), cli_errors():
            loaded_rubric = load_rubric(rubric)
            columns = {"text_id": text_id_column, "text": text_column}
            if group_column is not None:
                columns["group_id"] = group_column
            table = TextTable.read(texts, **columns)
            settings = CodingSettings(
                method=method,
                runs=runs,
                temperature=parsed_temperature,
                seed=seed,
                wording_enable=parse_id_list(enable) if enable is not None else "all",
                wording_disable=parse_id_list(disable),
                shuffle_markers=not no_shuffle,
                concurrency=concurrency,
                max_failure_share=max_failure_share,
                retry_failed=retry_failed,
                text_range=parsed_range,
            )
            llm = build_backend(
                backend,
                model=model,
                base_url=base_url,
                timeout=timeout,
                options=options,
                rubric=loaded_rubric,
                fake_script=fake_script,
            )
            logger.info("coding %d texts (%s) into %s", len(table), method, out)
            try:
                summary = code_texts(table, loaded_rubric, llm, out, settings)
            except RunFailureCeilingError as error:
                typer.echo(format_summary(error.summary))
                raise
            except BackendError:
                # The cells finished before the failure are stored; say where.
                typer.echo(format_rows([("store", str(out))]))
                raise
            logger.info(
                "coded %d texts with %d attribute calls, %d cells resumed, %d coded again",
                summary.n_texts,
                summary.n_attribute_calls,
                summary.resumed,
                summary.retried,
            )
            typer.echo(format_summary(summary))

    run.__doc__ = f"Code texts with the {label} LLM coder."
    run.__name__ = f"code_{method}"
    return run


code_app.command("two-stage", help="Code texts with the two-stage LLM coder.")(
    _command("two_stage")
)
code_app.command("one-stage", help="Code texts with the one-stage LLM coder.")(
    _command("one_stage")
)
