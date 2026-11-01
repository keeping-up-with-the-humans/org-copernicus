"""The kinds of ``corrections estimate``: which options each reads, and the estimator it runs.

Every kind refuses an option it does not read and names the options it needs, so a mistyped
command fails before any data are read. Options that are not given keep the estimator's own
defaults; without ``--cluster`` or ``--no-cluster`` a prediction-powered kind clusters its
standard error by group whenever a group holds more than one coded text.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, fields
from typing import Literal

import numpy as np
import pandas as pd

from copernicus.cli._correction_inputs import INTEGER
from copernicus.cli._embedding_options import library_default
from copernicus.corrections import (
    CorrectionData,
    Estimate,
    group_difference_ppi,
    long_short_split,
    mean_ppi,
    moment_subtraction,
    ols_outcome_ppi,
    ols_predictor_ppi,
    oriv,
    simex,
)
from copernicus.corrections._whole_groups import require_whole_groups
from copernicus.errors import ConfigurationError, SettingsError

logger = logging.getLogger(__name__)

__all__ = ["KINDS", "WEAK_INSTRUMENT_FLOOR", "EstimateOptions", "check_options", "run_estimate"]

KINDS = (
    "mean",
    "difference",
    "ols-outcome",
    "ols-predictor",
    "moment-subtraction",
    "simex",
    "oriv",
)
REQUIRED: dict[str, tuple[str, ...]] = {
    "mean": (),
    "difference": (),
    "ols-outcome": ("--formula",),
    "ols-predictor": ("--outcome",),
    "moment-subtraction": ("--outcome",),
    "simex": ("--outcome",),
    "oriv": ("--outcome", "--score-a", "--score-b"),
}
PPI_OPTIONS = ("--prediction-column", "--power-tuning", "--cluster")
PPI_KINDS = ("mean", "difference", "ols-outcome", "ols-predictor")
BOOT_OPTIONS = ("--seed", "--n-boot")
ALLOWED: dict[str, tuple[str, ...]] = {
    "mean": PPI_OPTIONS,
    "difference": (*PPI_OPTIONS, "--group-column-split", "--long-short"),
    "ols-outcome": ("--formula", "--prediction-column", "--coefficient", "--cluster"),
    "ols-predictor": (
        "--outcome",
        "--covariates",
        "--prediction-column",
        "--coefficient",
        "--variant",
        "--cluster",
    ),
    "moment-subtraction": ("--outcome", "--prediction-column", *BOOT_OPTIONS),
    "simex": ("--outcome", "--prediction-column", *BOOT_OPTIONS),
    "oriv": ("--outcome", "--score-a", "--score-b", "--weak-instrument-floor", *BOOT_OPTIONS),
}
CONSTRUCT_SLOPE = library_default(ols_predictor_ppi, "coefficient")
FIRST_AFTER_INTERCEPT = library_default(ols_outcome_ppi, "coefficient")
PREDICTOR_VARIANTS: dict[str, Literal["ppi", "ludwig"]] = {"ppi": "ppi", "ludwig": "ludwig"}
DEFAULT_VARIANT = library_default(ols_predictor_ppi, "variant")
WEAK_INSTRUMENT_FLOOR = library_default(oriv, "weak_instrument_floor")


@dataclass(frozen=True)
class EstimateOptions:
    """The options of ``corrections estimate`` that depend on the kind; unset is the default.

    Attributes:
        prediction_column: The machine prediction (PPI) or the score column (moment
            subtraction, SIMEX); the first score column when unset.
        power_tuning: Whether a PPI mean or difference is power-tuned.
        group_column_split: A boolean or 0/1 column of the texts file for a difference.
        long_short: Whether a difference is long minus short texts (covariate median split).
        formula: The right-hand side of a regression with the construct as outcome.
        outcome: The outcome column of a regression with the construct as predictor.
        covariates: The covariates of that regression, a right-hand side.
        coefficient: The reported coefficient, a column name or a position.
        cluster: Whether a PPI estimate's errors are clustered by group (``--cluster``) or not
            (``--no-cluster``); ``None`` clusters when a group holds more than one coded text.
        score_a: The first measurement of ORIV.
        score_b: The second measurement of ORIV.
        weak_instrument_floor: The smallest absolute correlation of ORIV's two proxies.
        variant: ``ppi`` or ``ludwig`` for the predictor regression.
        seed: The seed of the bootstrap (and of the SIMEX noise).
        n_boot: The bootstrap resamples.
    """

    prediction_column: str | None = None
    power_tuning: bool = False
    group_column_split: str | None = None
    long_short: bool = False
    formula: str | None = None
    outcome: str | None = None
    covariates: str | None = None
    coefficient: str | None = None
    cluster: bool | None = None
    score_a: str | None = None
    score_b: str | None = None
    weak_instrument_floor: float | None = None
    variant: str | None = None
    seed: int | None = None
    n_boot: int | None = None

    def given(self) -> dict[str, object]:
        """The options set on the command line (those not at their default), by their names."""
        return {
            f"--{item.name.replace('_', '-')}": getattr(self, item.name)
            for item in fields(self)
            if getattr(self, item.name) is not item.default
        }


def check_options(
    kind: str, options: EstimateOptions, pi_column: str | None, *, has_draw: bool
) -> None:
    """Refuse a command whose options do not fit its kind, before any data are read.

    Args:
        kind: The ``--kind``.
        options: The kind-dependent options.
        pi_column: The ``--pi-column``, or ``None``.
        has_draw: Whether ``--draw`` is given, which supplies ``pi`` from the draw.

    Raises:
        SettingsError: If the kind is unknown, an option it needs is missing, an option it does
            not read is given, a difference names no split or two, the predictor regression
            gets a ``--variant`` other than ``ppi`` and ``ludwig`` or a ``--coefficient`` that
            is not a position, or a prediction-powered kind has neither ``--draw`` nor
            ``--pi-column``.
    """
    if kind not in KINDS:
        raise SettingsError(f"--kind must be one of {', '.join(KINDS)}; got {kind!r}")
    given = options.given()
    missing = [name for name in REQUIRED[kind] if name not in given]
    if missing:
        raise SettingsError(f"--kind {kind} needs {', '.join(missing)}")
    stray = [name for name in given if name not in ALLOWED[kind]]
    if stray:
        verb = "does" if len(stray) == 1 else "do"
        raise SettingsError(f"{', '.join(stray)} {verb} not apply to --kind {kind}")
    if kind == "difference" and ("--group-column-split" in given) == ("--long-short" in given):
        raise SettingsError("--kind difference needs one of --group-column-split or --long-short")
    if kind == "ols-predictor":
        _check_predictor(options)
    if kind in PPI_KINDS and pi_column is None and not has_draw:
        raise SettingsError(
            f"--kind {kind} needs --draw or --pi-column: the draw file of corrections sample, "
            "which binds the coded sample to the texts file and gives pi, or the column of the "
            "gold file with the inclusion probability of each coded text (pi)"
        )


def _check_predictor(options: EstimateOptions) -> None:
    """Refuse a predictor-regression variant or coefficient that the estimator cannot take."""
    if options.variant is not None and options.variant not in PREDICTOR_VARIANTS:
        raise SettingsError(f"--variant must be ppi or ludwig; got {options.variant!r}")
    if options.coefficient is not None and not INTEGER.fullmatch(options.coefficient):
        raise SettingsError(
            "--coefficient of --kind ols-predictor is a position in [1, construct, covariates]: "
            f"1 is the construct's slope; got {options.coefficient!r}"
        )


def _needed(value: str | None, option: str) -> str:
    """The value of an option the kind requires (checked by ``check_options``)."""
    if value is None:
        raise SettingsError(f"{option} is required")
    return value


def _column(data: CorrectionData, name: str, option: str) -> pd.Series:
    """A column of the correction data that an option names."""
    if name not in data.frame.columns:
        raise ConfigurationError(
            f"{option} {name!r} is not a column of the scores or texts file; columns: "
            f"{', '.join(str(column) for column in data.frame.columns)}"
        )
    return data.frame[name]


def _group_mask(data: CorrectionData, name: str) -> np.ndarray:
    """The two-level grouping a boolean or 0/1 column of the texts file defines."""
    values = _column(data, name, "--group-column-split")
    if pd.api.types.is_bool_dtype(values):
        return values.to_numpy(dtype=bool)
    if pd.api.types.is_numeric_dtype(values) and bool(values.isin([0, 1]).all()):
        return values.to_numpy() == 1
    raise ConfigurationError(
        f"--group-column-split {name!r} must be boolean or 0/1 on every text: the difference is "
        "the mean where it is true minus the mean where it is false"
    )


def _prediction(data: CorrectionData, name: str | None) -> pd.Series | None:
    """The prediction a PPI estimate uses: a named column, or None for the first score column."""
    if name is None:
        return None
    values = _column(data, name, "--prediction-column").to_numpy()
    return pd.Series(values, index=data.ids, name=name)


def _reported(text: str | None) -> str | int | None:
    """The coefficient ``--coefficient`` names: a position for an integer, else a column name."""
    if text is None:
        return None
    return int(text) if INTEGER.fullmatch(text) else text


def run_estimate(kind: str, data: CorrectionData, options: EstimateOptions) -> Estimate:
    """The estimate of the kind, for options passed by ``check_options``.

    Options that are unset keep the estimator's defaults. A prediction-powered kind first
    checks that the coded texts cover whole groups, as ``corrections sample`` draws them, since
    its ``pi`` holds only then.

    Raises:
        ConfigurationError: If a prediction-powered kind gets coded texts that leave texts of
            their groups uncoded.
    """
    if kind in PPI_KINDS:
        require_whole_groups(data)
    boot = {
        name: value
        for name, value in (("seed", options.seed), ("n_boot", options.n_boot))
        if value is not None
    }
    if kind in ("moment-subtraction", "simex"):
        method = moment_subtraction if kind == "moment-subtraction" else simex
        outcome = _needed(options.outcome, "--outcome")
        return method(data, outcome, score_column=options.prediction_column, **boot)
    if kind == "oriv":
        score_a, score_b = (
            _needed(options.score_a, "--score-a"),
            _needed(options.score_b, "--score-b"),
        )
        floor = options.weak_instrument_floor
        return oriv(
            data,
            _needed(options.outcome, "--outcome"),
            score_a=score_a,
            score_b=score_b,
            weak_instrument_floor=WEAK_INSTRUMENT_FLOOR if floor is None else floor,
            **boot,
        )
    prediction = _prediction(data, options.prediction_column)
    cluster = options.cluster
    if kind == "mean":
        return mean_ppi(
            data, prediction=prediction, power_tuning=options.power_tuning, cluster=cluster
        )
    if kind == "difference":
        split = options.group_column_split
        group = long_short_split(data) if split is None else _group_mask(data, split)
        return group_difference_ppi(
            data, group, prediction=prediction, power_tuning=options.power_tuning, cluster=cluster
        )
    if kind == "ols-outcome":
        position = _reported(options.coefficient)
        return ols_outcome_ppi(
            data,
            _needed(options.formula, "--formula"),
            prediction=prediction,
            coefficient=FIRST_AFTER_INTERCEPT if position is None else position,
            cluster=cluster,
        )
    chosen = DEFAULT_VARIANT if options.variant is None else options.variant
    return ols_predictor_ppi(
        data,
        _needed(options.outcome, "--outcome"),
        options.covariates,
        prediction=prediction,
        variant=PREDICTOR_VARIANTS[chosen],
        coefficient=CONSTRUCT_SLOPE if options.coefficient is None else int(options.coefficient),
        cluster=cluster,
    )
