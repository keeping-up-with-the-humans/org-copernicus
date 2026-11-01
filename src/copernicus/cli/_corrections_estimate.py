"""The ``corrections estimate`` command: one corrected estimate from a human-coded sample."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated, Any

import typer

from copernicus._io import json_ready, write_json
from copernicus.cli._correction_inputs import (
    COVARIATE,
    CovariateOption,
    GoldColumnOption,
    ScoreColumnsOption,
    ScoresOption,
    bind_to_draw,
    load_correction_data,
)
from copernicus.cli._embedding_options import (
    GroupColumn,
    TextColumn,
    TextIdColumn,
    TextsOption,
    library_default,
)
from copernicus.cli._estimate_kinds import (
    KINDS,
    WEAK_INSTRUMENT_FLOOR,
    EstimateOptions,
    check_options,
    run_estimate,
)
from copernicus.cli._inputs import TEXT_ID, GoldOption
from copernicus.cli._logging import Verbose, cli_logging
from copernicus.cli._options import cli_errors
from copernicus.cli._output import format_rows
from copernicus.corrections import Estimate, moment_subtraction, simex
from copernicus.errors import SettingsError
from copernicus.similarity.provenance import library_stamp

logger = logging.getLogger(__name__)

__all__ = ["estimate"]

ANALYSIS = "corrected estimate"
SUFFIX_JSON = ".json"
N_BOOT_HELP = (
    "Bootstrap resamples of the standard error; default the estimator's own "
    f"({library_default(moment_subtraction, 'n_boot')} for moment subtraction and ORIV, "
    f"{library_default(simex, 'n_boot')} for SIMEX)."
)


def _estimate_rows(found: Estimate, out: Path) -> list[tuple[str, str]]:
    """The printed summary of an estimate."""
    return [
        ("method", found.method),
        ("estimate", f"{found.point:.4g} (SE {found.se:.4g})"),
        ("95% interval", f"[{found.ci_low:.4g}, {found.ci_high:.4g}]"),
        ("labelled texts", str(found.n_labelled)),
        ("texts", str(found.n_total)),
        ("file", str(out)),
    ]


def estimate(
    kind: Annotated[str, typer.Option("--kind", help=f"One of {', '.join(KINDS)}.")],
    scores: ScoresOption,
    texts: TextsOption,
    gold: GoldOption,
    gold_column: GoldColumnOption,
    score_columns: ScoreColumnsOption,
    out: Annotated[
        Path, typer.Option("--out", help="The estimate with its details, a .json file.")
    ],
    draw: Annotated[
        Path | None,
        typer.Option(
            "--draw",
            exists=True,
            dir_okay=False,
            help="The draw file of corrections sample, <out>.json. It binds the coded sample to "
            "the texts file: a texts file whose texts or groups changed after the draw is "
            "refused, every drawn text must be coded, and pi comes from the draw. Pass it "
            "whenever the sample came from corrections sample.",
        ),
    ] = None,
    pi_column: Annotated[
        str | None,
        typer.Option(
            "--pi-column",
            help="Column of the gold file with the inclusion probability of each coded text, as "
            "the sample command writes it; the prediction-powered kinds need it or --draw. "
            "With --draw it must equal the draw's pi.",
        ),
    ] = None,
    prediction_column: Annotated[
        str | None,
        typer.Option(
            "--prediction-column",
            help="Column with the machine measurement: the prediction of a prediction-powered "
            "kind, the score of moment subtraction and SIMEX; default the first score column.",
        ),
    ] = None,
    power_tuning: Annotated[
        bool, typer.Option("--power-tuning", help="Power-tune a mean or a difference.")
    ] = False,
    group_column_split: Annotated[
        str | None,
        typer.Option(
            "--group-column-split",
            help="Boolean or 0/1 column of the texts file: a difference is the mean where it is "
            "true minus the mean where it is false.",
        ),
    ] = None,
    long_short: Annotated[
        bool,
        typer.Option(
            "--long-short",
            help="A difference of the texts above the covariate's median minus the rest.",
        ),
    ] = False,
    formula: Annotated[
        str | None,
        typer.Option(
            "--formula",
            help="Right-hand side of the regression of the construct, such as '1 + log_chars'.",
        ),
    ] = None,
    outcome: Annotated[
        str | None,
        typer.Option(
            "--outcome", help="Numeric column of the texts file regressed on the construct."
        ),
    ] = None,
    covariates: Annotated[
        str | None,
        typer.Option(
            "--covariates", help="Right-hand side of the covariates beside the construct."
        ),
    ] = None,
    coefficient: Annotated[
        str | None,
        typer.Option(
            "--coefficient",
            help="The reported coefficient: a column name or a position (default 1, the column "
            "after the intercept; for ols-predictor the construct's slope).",
        ),
    ] = None,
    cluster: Annotated[
        bool | None,
        typer.Option(
            "--cluster/--no-cluster",
            help="Sum the errors of the texts of a group (a drawn group is one cluster) or treat "
            "the texts as independent; default: cluster when a group holds several coded texts. "
            "A clustered difference needs coded texts from two groups on each side.",
        ),
    ] = None,
    score_a: Annotated[
        str | None, typer.Option("--score-a", help="First measurement of ORIV.")
    ] = None,
    score_b: Annotated[
        str | None, typer.Option("--score-b", help="Second measurement of ORIV.")
    ] = None,
    weak_instrument_floor: Annotated[
        float | None,
        typer.Option(
            "--weak-instrument-floor",
            help="Smallest absolute correlation of ORIV's two measurements, in (0, 1); default "
            f"{WEAK_INSTRUMENT_FLOOR}.",
        ),
    ] = None,
    variant: Annotated[
        str | None,
        typer.Option("--variant", help="'ppi' (default) or 'ludwig' for ols-predictor."),
    ] = None,
    seed: Annotated[
        int | None,
        typer.Option("--seed", help="Seed of the bootstrap; default the estimator's own (42)."),
    ] = None,
    n_boot: Annotated[int | None, typer.Option("--n-boot", help=N_BOOT_HELP)] = None,
    covariate: CovariateOption = COVARIATE,
    text_id_column: TextIdColumn = TEXT_ID,
    text_column: TextColumn = "text",
    group_column: GroupColumn = None,
    verbose: Verbose = False,
) -> None:
    """Estimate a mean, a difference or a coefficient of the construct from a human sample.

    The scores, texts and gold files are joined by text id; every further column of the texts
    file can be read by --formula, --outcome, --covariates and --group-column-split.

    The draw file of corrections sample binds the coded sample to the texts file: with --draw,
    a texts file changed after the draw is refused. Without it, the prediction-powered kinds
    take pi from --pi-column and check only that the coded texts cover whole groups, so a
    regrouped texts file goes unnoticed; the other kinds check nothing of the draw.

    The estimate is printed and written with its details to --out (.json).
    """
    with cli_logging(verbose), cli_errors():
        if out.suffix != SUFFIX_JSON:
            raise SettingsError(f"--out must be a .json file, got {out.suffix!r}")
        if not out.parent.is_dir():
            raise SettingsError(f"folder of the output file does not exist: {out.parent}")
        options = EstimateOptions(
            prediction_column=prediction_column,
            power_tuning=power_tuning,
            group_column_split=group_column_split,
            long_short=long_short,
            formula=formula,
            outcome=outcome,
            covariates=covariates,
            coefficient=coefficient,
            cluster=cluster,
            score_a=score_a,
            score_b=score_b,
            weak_instrument_floor=weak_instrument_floor,
            variant=variant,
            seed=seed,
            n_boot=n_boot,
        )
        check_options(kind, options, pi_column, has_draw=draw is not None)
        given = options.given()
        paths = {"scores": scores, "texts": texts, "gold": gold}
        columns = (text_id_column, text_column, group_column)
        data = load_correction_data(
            paths,
            gold_column,
            score_columns,
            covariate,
            columns,
            pi_column=pi_column if draw is None else None,
            carry_table_columns=True,
        )
        if draw is not None:
            data = bind_to_draw(data, gold, draw, pi_column)
        found = run_estimate(kind, data, options)
        content: dict[str, Any] = {
            "analysis": ANALYSIS,
            "kind": kind,
            "estimate": {
                "point": found.point,
                "se": found.se,
                "ci_low": found.ci_low,
                "ci_high": found.ci_high,
                "n_labelled": found.n_labelled,
                "n_total": found.n_total,
                "method": found.method,
                "details": json_ready(found.details),
            },
            "options": {name.lstrip("-").replace("-", "_"): value for name, value in given.items()}
            | {"pi_column": pi_column, "covariate": covariate},
            "inputs": {name: str(path) for name, path in paths.items()}
            | {
                "draw": None if draw is None else str(draw),
                "gold_column": gold_column,
                "score_columns": list(data.score_columns),
            },
            "text_columns": {
                TEXT_ID: text_id_column,
                "text": text_column,
                "group_id": group_column,
            },
            **library_stamp(),
        }
        write_json(out, json_ready(content))
        logger.info("wrote the %s estimate to %s", kind, out)
        typer.echo(format_rows(_estimate_rows(found, out)))
