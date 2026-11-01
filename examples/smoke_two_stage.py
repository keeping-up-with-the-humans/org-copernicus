"""Code the toy corpus with a real model, within a budget, and check the result.

The script estimates the cost of the run before it makes a call, keeps a running total of
what earlier runs spent in ``OUT_DIR/spend.json``, and refuses to start when the estimate
plus the running total reaches the ceiling you give it. It then codes the first ``--texts``
texts of the toy corpus with the two-stage LLM coder, writes the run store and the exports,
counts how many calls and passages came out well, builds the validity panel of the construct
score against the gold value, prints a report, and adds the spend of the run to the ledger.

The API key is read from the ``OPENAI_API_KEY`` environment variable by the backend. The
script never prints, logs or writes it, never reads a file for it, and never writes a request
body: ``spend.json`` holds the backend kind, numbers and timestamps only. ``--fake`` replaces
the model with the fake backend, so the whole script can be tried without a key, a network or
a cost: its estimate counts characters / 4 instead of loading tiktoken's encoding, and its
ledger entry records the backend ``fake`` with a cost of 0.

Run it with ``python examples/smoke_two_stage.py --out DIR --price-input P --price-output Q
--ceiling C``, where the prices are US dollars per million tokens; add ``--dry-run`` to print
the estimate and stop; a dry run whose estimate plus the running total reaches the ceiling
exits with status 1, like the run it stands for.

Exit status: 0 after a dry run or a run; 1 when the run is refused, and also when the run
finished but some answers carried no token counts, because the spend is then understated (the
number is in the ledger entry as ``answers_without_usage`` and the data are kept).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd
from smoke_support import (
    SPEND_FILE,
    MeteredBackend,
    acceptance_metrics,
    append_spend,
    estimate_run,
    estimated_cost,
    format_acceptance,
    format_estimate,
    read_total,
    spend_entry,
)

import copernicus as cop
from copernicus.errors import SettingsError
from copernicus.llm import Backend

logger = logging.getLogger(__name__)

__all__ = ["main"]

GOLD = "gold_constructiveness"
CONSTRUCT = "construct"
REPORT_FILE = "smoke_report.txt"
SUMMARY_FILE = "smoke_summary.json"
DEFAULT_MODEL = "gpt-4.1-mini"
DEFAULT_RUNS = 2
DEFAULT_CONCURRENCY = 4
DEFAULT_N_BOOT = 200
# The SDK logs every request through its HTTP client: httpx2 from openai 3 on, httpx before.
QUIET_LOGGERS = ("openai", "httpx", "httpx2", "httpcore")
FAILURE_EXIT_CODE = 1


def parse_arguments(arguments: Sequence[str] | None) -> argparse.Namespace:
    """Read and check the command line.

    Args:
        arguments: Command-line arguments; ``None`` reads ``sys.argv``.

    Returns:
        The options. Exits with the usage message when an option is missing or out of range.
    """
    parser = argparse.ArgumentParser(description="Code the toy corpus with a real model.")
    parser.add_argument("--out", type=Path, required=True, help="folder for the run and the ledger")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="model name sent to the endpoint")
    parser.add_argument("--texts", type=int, help="code the first N texts (default: all)")
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS, help="codings per text")
    parser.add_argument(
        "--price-input", type=float, required=True, help="USD per million input tokens"
    )
    parser.add_argument(
        "--price-output", type=float, required=True, help="USD per million output tokens"
    )
    parser.add_argument(
        "--ceiling", type=float, required=True, help="USD; refuse when estimate + total reaches it"
    )
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT, help="panel bootstrap draws")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the estimate and stop; exits 1 when the run would be refused",
    )
    parser.add_argument("--fake", action="store_true", help="use the fake backend, not the model")
    options = parser.parse_args(arguments)
    checks = (
        ("--texts", options.texts is None or options.texts >= 1),
        ("--runs", options.runs >= 1),
        ("--concurrency", options.concurrency >= 1),
        ("--n-boot", options.n_boot >= 1),
        ("--price-input", options.price_input >= 0),
        ("--price-output", options.price_output >= 0),
        ("--ceiling", options.ceiling > 0),
    )
    for name, valid in checks:
        if not valid:
            parser.error(f"{name} is out of range")
    return options


def coding_settings(options: argparse.Namespace) -> cop.CodingSettings:
    """Build the settings of the run: its runs, its concurrency and the first texts."""
    return cop.CodingSettings(
        runs=options.runs,
        concurrency=options.concurrency,
        text_range=None if options.texts is None else (0, options.texts),
    )


def first_texts(table: cop.TextTable, n_texts: int | None) -> list[Any]:
    """Return the ids of the first texts in text-id order (ids compared as strings).

    Args:
        table: The texts.
        n_texts: How many to take; ``None`` takes all.

    Returns:
        The ids, the same texts that ``CodingSettings(text_range=(0, n_texts))`` codes.

    Raises:
        SettingsError: If the corpus holds fewer texts than asked for.
    """
    ordered = sorted(table.ids(), key=str)
    if n_texts is not None and n_texts > len(ordered):
        raise SettingsError(f"--texts {n_texts} exceeds the {len(ordered)} texts of the corpus")
    return ordered if n_texts is None else ordered[:n_texts]


def build_backend(options: argparse.Namespace, rubric: cop.Rubric) -> Backend:
    """Build the fake backend, or the OpenAI backend that reads its key from the environment."""
    if options.fake:
        return cop.FakeBackend(cop.cue_responder(rubric, cop.assets.load_toy_cues()))
    return cop.OpenAICompatibleBackend(options.model)


def code_and_record(
    options: argparse.Namespace, table: cop.TextTable, rubric: cop.Rubric
) -> tuple[Path, dict[str, Any], float]:
    """Code the texts into the run store and add the spend to the ledger.

    The tokens of every answer are counted as the backend returns it, and the spend is
    recorded when the coding ends, whether it ends normally or in an error, so the answers of
    a run that stops are recorded too. A call that raises returns no usage, so its tokens are
    not metered and the recorded spend can fall short of the provider's bill by such calls.
    When the ledger cannot be written while an error is in flight, the ledger failure is
    raised from that error, so both appear in the traceback.

    Args:
        options: The command-line options.
        table: The texts.
        rubric: The rubric.

    Returns:
        The path of the run store, the ledger entry of this run, and the running total in
        US dollars.

    Raises:
        OSError: If the ledger cannot be written; raised from the coding error when there was one.
    """
    options.out.mkdir(parents=True, exist_ok=True)
    n_texts = len(first_texts(table, options.texts))
    store = options.out / f"two_stage_runs{options.runs}.sqlite"
    backend = MeteredBackend(build_backend(options, rubric))

    def record() -> tuple[dict[str, Any], float]:
        entry = spend_entry(
            options,
            backend.info().backend,
            n_texts,
            backend.prompt_tokens,
            backend.completion_tokens,
            backend.answers_without_usage,
        )
        return entry, append_spend(options.out / SPEND_FILE, entry)

    try:
        cop.code_texts(table, rubric, backend, store, coding_settings(options))
    except BaseException as failure:
        try:
            record()
        except Exception as ledger_error:
            raise ledger_error from failure
        raise
    spend, total = record()
    return store, spend, total


def export_and_measure(
    options: argparse.Namespace, store: Path, spend: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Write the exports and the summary of the run store, and measure the acceptance.

    Args:
        options: The command-line options.
        store: The run store.
        spend: The ledger entry of the run.

    Returns:
        The long score table and the acceptance metrics.
    """
    scores = cop.export_scores(store)
    evidence = cop.export_evidence(store)
    scores.to_csv(options.out / "two_stage_scores.csv", index=False)
    cop.export_text_scores(store).to_csv(options.out / "two_stage_text_scores.csv", index=False)
    evidence.to_csv(options.out / "two_stage_evidence.csv", index=False)
    metrics = acceptance_metrics(scores, evidence)
    (options.out / SUMMARY_FILE).write_text(
        json.dumps({"metrics": metrics, "spend": spend}, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return scores, metrics


def main(arguments: Sequence[str] | None = None) -> int:
    """Print the estimate, refuse when over the ceiling, and otherwise run and report.

    Args:
        arguments: Command-line arguments; ``None`` reads ``sys.argv``.

    Returns:
        The exit code: 0 after a dry run or a run; 1 when the estimate plus the running total
        reaches the ceiling (also in a dry run) and nothing was run, or when the run finished
        but some answers reported no token counts.
    """
    options = parse_arguments(arguments)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    for name in QUIET_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
    rubric = cop.load_rubric(cop.rubric.loader.toy_rubric_path())
    table = cop.TextTable.read(cop.assets.toy_corpus_path(), group_id="group_id")
    settings = coding_settings(options)
    stages = estimate_run(
        table, rubric, first_texts(table, options.texts), settings, not options.fake
    )
    total = read_total(options.out / SPEND_FILE)
    print(format_estimate(stages, options, total, settings))
    if estimated_cost(stages, options) + total >= options.ceiling:
        print(
            f"refused: the estimate plus the running total reaches the ceiling of USD "
            f"{options.ceiling}; nothing was run",
            file=sys.stderr,
        )
        return FAILURE_EXIT_CODE
    if options.dry_run:
        return 0
    store, spend, total = code_and_record(options, table, rubric)
    scores, metrics = export_and_measure(options, store, spend)
    acceptance = format_acceptance(metrics, spend, total)
    print(acceptance)
    panel = cop.validity_panel(
        scores,
        table.frame,
        table,
        targets={CONSTRUCT: GOLD},
        settings=cop.PanelSettings(n_boot=options.n_boot),
    )
    section = f"== validity panel (construct score against the gold value) ==\n{panel.report()}"
    (options.out / REPORT_FILE).write_text(f"{acceptance}\n\n{section}\n", encoding="utf-8")
    print(f"\n{section}")
    if spend["answers_without_usage"] > 0:
        print(
            f"failed: {spend['answers_without_usage']} answers carried no usage (token counts), "
            f"so the spend in {SPEND_FILE} is understated; the data are kept",
            file=sys.stderr,
        )
        return FAILURE_EXIT_CODE
    return 0


if __name__ == "__main__":
    sys.exit(main())
