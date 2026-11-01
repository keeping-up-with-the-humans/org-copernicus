"""The ``validity`` command: the validity panel of coder scores against gold values."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Annotated, Any

import pandas as pd
import typer

from copernicus.cli._embedding_options import (
    GroupColumn,
    TextColumn,
    TextIdColumn,
    TextsOption,
    read_table,
)
from copernicus.cli._inputs import TEXT_ID, GoldOption, parse_pairs, read_gold
from copernicus.cli._logging import Verbose, cli_logging
from copernicus.cli._options import cli_errors
from copernicus.cli._output import (
    check_table_output,
    format_rows,
    write_description,
    write_report,
    write_table,
)
from copernicus.coders import export_scores
from copernicus.data import read_frame
from copernicus.errors import SettingsError
from copernicus.similarity.provenance import library_stamp
from copernicus.validity import PanelSettings, ValidityPanel, validity_panel

logger = logging.getLogger(__name__)

__all__ = ["validity_app"]

DEFAULTS = PanelSettings()
ANALYSIS = "validity panel"
SCORE_ID_COLUMNS = (TEXT_ID, "attribute_id", "target_id", "target_level")
validity_app = typer.Typer(
    help="Validity of coder scores against gold values.", no_args_is_help=True
)


def _scores_frame(store: Path | None, scores: Path | None) -> pd.DataFrame:
    """Return the scores of the one source given: a run store or a scores file."""
    if store is not None and scores is None:
        return export_scores(store)
    if scores is not None and store is None:
        return read_frame(scores, protected_columns=SCORE_ID_COLUMNS)
    raise SettingsError("give exactly one of --store and --scores")


def _reliability_number(text: str) -> float:
    """Read one reliability value of ``--reliability``."""
    try:
        return float(text)
    except ValueError as error:
        raise SettingsError(f"--reliability expects a number, got {text!r}") from error


def parse_reliability(values: Sequence[str]) -> float | dict[str, float] | None:
    """Parse ``--reliability``: one number for every target, or ``COLUMN=R`` per gold column.

    Args:
        values: The values as given on the command line, possibly none.

    Returns:
        ``None`` without a value, the number of a single ``R``, or the mapping from gold
        column to reliability of the ``COLUMN=R`` values.

    Raises:
        SettingsError: If numbers and ``COLUMN=R`` values are mixed, several numbers are
            given, a value is not a number, or a column is named twice.
    """
    per_column = [value for value in values if "=" in value]
    if per_column and len(per_column) != len(values):
        raise SettingsError(
            "--reliability takes one number R for every target or COLUMN=R per gold "
            "column, not both"
        )
    if per_column:
        pairs = parse_pairs(values, "--reliability", "COLUMN=R")
        return {column: _reliability_number(number) for column, number in pairs}
    if len(values) > 1:
        raise SettingsError(
            "--reliability takes one number for every target; name the gold column of each "
            "value as COLUMN=R to give several"
        )
    return _reliability_number(values[0]) if values else None


def _settings_mapping(settings: PanelSettings) -> dict[str, Any]:
    """The settings as JSON values; ``items``, which the command line never sets, is left out."""
    reliability = settings.reliability
    return {
        "covariate": settings.covariate,
        "covariate_scale": settings.covariate_scale,
        "n_boot": settings.n_boot,
        "seed": settings.seed,
        "direction": settings.direction,
        "reliability": dict(reliability) if isinstance(reliability, Mapping) else reliability,
    }


def _description(
    result: ValidityPanel,
    targets: Mapping[str, str],
    inputs: Mapping[str, str],
    text_columns: Mapping[str, str | None],
) -> dict[str, Any]:
    """Describe a panel run for the file written beside the panel."""
    return {
        "analysis": ANALYSIS,
        "settings": _settings_mapping(result.settings),
        "targets": dict(targets),
        "inputs": dict(inputs),
        "text_columns": dict(text_columns),
        "n_rows": len(result.frame),
        "n_used": result.n_used,
        "n_dropped_missing": result.n_dropped_missing,
        **library_stamp(),
    }


@validity_app.command("panel")
def panel(
    texts: TextsOption,
    gold: GoldOption,
    out: Annotated[Path, typer.Option("--out", help="The panel, .csv or .parquet.")],
    target: Annotated[
        list[str] | None,
        typer.Option(
            "--target",
            help="KEY=COLUMN: the score target KEY (an attribute id, 'construct', or a "
            "similarity target id) against the gold column COLUMN; repeatable, at least one.",
        ),
    ] = None,
    store: Annotated[
        Path | None,
        typer.Option(
            "--store", exists=True, dir_okay=False, help="Run store to export scores from."
        ),
    ] = None,
    scores: Annotated[
        Path | None,
        typer.Option(
            "--scores",
            exists=True,
            dir_okay=False,
            help="Scores file (CSV or parquet), for example similarity or hybrid output.",
        ),
    ] = None,
    covariate: Annotated[
        str | None,
        typer.Option(
            "--covariate",
            help="Column of the gold or text file for the covariate-linked error; "
            "default log(n_chars).",
        ),
    ] = DEFAULTS.covariate,
    covariate_scale: Annotated[
        str,
        typer.Option(
            "--covariate-scale",
            help=f"'standardized' or 'raw'; default {DEFAULTS.covariate_scale}.",
        ),
    ] = DEFAULTS.covariate_scale,
    direction: Annotated[
        str,
        typer.Option(
            "--direction",
            help=f"'positive', 'negative', or 'auto' (flip when the correlation is "
            f"negative); default {DEFAULTS.direction}.",
        ),
    ] = DEFAULTS.direction,
    reliability: Annotated[
        list[str] | None,
        typer.Option(
            "--reliability",
            help="Reliability of the gold values in (0, 1]: R for every target (only when "
            "all targets share one gold instrument), or COLUMN=R for the targets of the gold "
            "column COLUMN; repeatable in that form.",
        ),
    ] = None,
    n_boot: Annotated[
        int,
        typer.Option("--n-boot", help=f"Bootstrap replicates; default {DEFAULTS.n_boot}."),
    ] = DEFAULTS.n_boot,
    seed: Annotated[
        int, typer.Option("--seed", help=f"Seed of the bootstrap; default {DEFAULTS.seed}.")
    ] = DEFAULTS.seed,
    report: Annotated[
        Path | None,
        typer.Option("--report", help="Also write the plain-text report to this file."),
    ] = None,
    text_id_column: TextIdColumn = TEXT_ID,
    text_column: TextColumn = "text",
    group_column: GroupColumn = None,
    verbose: Verbose = False,
) -> None:
    """Compute the validity panel of scores against gold values.

    The scores come from a run store (--store) or from a scores file (--scores).

    The panel has one row per --target and goes to --out, its description to <out>.json.

    The one-page report is printed, and written to --report as well when that is given.
    """
    with cli_logging(verbose), cli_errors():
        check_table_output(out)
        if report is not None and not report.parent.is_dir():
            raise SettingsError(f"folder of the report file does not exist: {report.parent}")
        pairs = parse_pairs(target or [], "--target", "KEY=COLUMN")
        if not pairs:
            raise SettingsError("give at least one --target KEY=COLUMN")
        options: dict[str, Any] = {
            "covariate": covariate,
            "covariate_scale": covariate_scale,
            "n_boot": n_boot,
            "seed": seed,
            "direction": direction,
            "reliability": parse_reliability(reliability or []),
        }
        settings = PanelSettings(**options)
        score_frame = _scores_frame(store, scores)
        table = read_table(texts, text_id_column, text_column, group_column)
        logger.info("validity panel for %d targets on %d score rows", len(pairs), len(score_frame))
        result = validity_panel(
            score_frame, read_gold(gold), table, targets=dict(pairs), settings=settings
        )
        write_table(result.frame, out)
        source = {"store": str(store)} if store is not None else {"scores": str(scores)}
        inputs = {**source, "texts": str(texts), "gold": str(gold)}
        columns = {TEXT_ID: text_id_column, "text": text_column, "group_id": group_column}
        description = write_description(out, _description(result, dict(pairs), inputs, columns))
        text = result.report().rstrip()
        typer.echo(text)
        rows = [("rows", str(len(result.frame))), ("file", str(out))]
        if report is not None:
            write_report(report, text + "\n")
            rows.append(("report file", str(report)))
        rows.append(("description file", str(description)))
        typer.echo("\n" + format_rows(rows))
