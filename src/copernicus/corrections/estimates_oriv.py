"""Instrumented coding (ORIV): two measurements of the construct, each the other's instrument.

When two measurements ``A`` and ``B`` of the construct have errors that are independent of each
other and of the construct, the slope of an outcome on the construct is identified by
instrumenting one measurement with the other. Equation (6) of Gillen, Snowberg and Yariv uses
both directions at once: the outcome stacked twice, ``(y; y)``, on ``(W_A; W_B)`` with an
intercept per block, instrumented by the block-diagonal ``[W_B, 0; 0, W_A]``, estimated as one
two-stage least squares.

The two measurements are first put on the construct's scale by the reverse calibration on the
labelled rows (see ``_proxy``). That is the only step that reads gold values; two measurements
already on one scale would not need it. The standard error is the standard deviation over a
stratified bootstrap that refits both calibration lines on every resample (texts are resampled,
not groups), so the sampling error of the scale step is in the interval. The 2SLS sandwich
clustered by text (rows ``i`` and ``N + i`` are the same text), which treats the calibration as
known and is too narrow with a small human sample, is reported as ``sandwich_se``. A resample on
which the estimate is undefined (no calibration line, a correlation below the floor, a singular
system) is counted, as for the other bootstrapped estimators.

The estimate needs a strong first stage: a correlation of the two proxies below the floor
``weak_instrument_floor`` (default ``WEAK_INSTRUMENT_FLOOR``, 0.20) in absolute value is refused,
on the data and on every bootstrap resample. ``shared_error_correlation`` reports the
correlation of the two proxies' residuals after regressing each on the gold values over the
labelled rows; ORIV assumes it is zero, and a clearly positive value means a shared error that
biases the estimate.
"""

from __future__ import annotations

import logging
import numbers
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from copernicus.corrections._estimate_inputs import outcome_values, require_labelled, score_values
from copernicus.corrections._proxy import (
    MIN_FINITE_BOOTS,
    MIN_LABELLED,
    ReverseCalibration,
    bootstrap_se,
    check_count,
    reverse_calibrate,
    sample_covariance,
    sample_variance,
    stratified_bootstrap,
)
from copernicus.corrections.data import CorrectionData
from copernicus.corrections.estimates import Estimate, make_estimate
from copernicus.corrections.families._registry import FamilySpec, register
from copernicus.errors import EstimatorError, SettingsError
from copernicus.validity.metrics import is_constant

logger = logging.getLogger(__name__)

__all__ = ["WEAK_INSTRUMENT_FLOOR", "StackedFit", "oriv", "stacked_2sls"]

WEAK_INSTRUMENT_FLOOR = 0.20
SLOPE_COLUMN = 2
# Rows of the first-stage coefficients: W_B instruments the first block, W_A the second.
INSTRUMENT_B_ROW = 2
INSTRUMENT_A_ROW = 3
ESTIMATE_ERRORS = (
    "attenuation of the association from two machine measurements; the scale step uses the "
    "labelled rows"
)


@dataclass(frozen=True)
class StackedFit:
    """The stacked two-stage least squares of equation (6).

    Attributes:
        slope: The coefficient of the stacked measurement.
        se: Its standard error, clustered by text.
        first_stage_a_on_b: The first-stage coefficient of ``W_B`` for ``W_A``.
        first_stage_b_on_a: The first-stage coefficient of ``W_A`` for ``W_B``.
    """

    slope: float
    se: float
    first_stage_a_on_b: float
    first_stage_b_on_a: float


def stacked_2sls(outcome: np.ndarray, proxy_a: np.ndarray, proxy_b: np.ndarray) -> StackedFit:
    """Estimate equation (6): ``(y; y)`` on ``(W_A; W_B)``, instrumented by ``(W_B; W_A)``.

    Args:
        outcome: The outcome of every text.
        proxy_a: The first measurement of every text.
        proxy_b: The second measurement of every text.

    Returns:
        The slope, its text-clustered standard error and the two first-stage coefficients.

    Raises:
        EstimatorError: If the system is singular: the instruments are collinear (a constant
            measurement) or carry no first-stage signal.
    """
    n = outcome.size
    zeros, ones = np.zeros(n), np.ones(n)
    stacked_outcome = np.concatenate([outcome, outcome])
    design = np.column_stack(
        [
            np.concatenate([ones, zeros]),
            np.concatenate([zeros, ones]),
            np.concatenate([proxy_a, proxy_b]),
        ]
    )
    instruments = np.column_stack(
        [
            design[:, 0],
            design[:, 1],
            np.concatenate([proxy_b, zeros]),
            np.concatenate([zeros, proxy_a]),
        ]
    )
    if int(np.linalg.matrix_rank(instruments)) < instruments.shape[1]:
        raise EstimatorError(
            "the 2SLS system is singular: the instruments are collinear (a measurement is constant)"
        )
    first_stage = np.linalg.solve(instruments.T @ instruments, instruments.T @ design)
    projected = instruments @ first_stage
    cross = projected.T @ design
    if int(np.linalg.matrix_rank(cross)) < cross.shape[0]:
        raise EstimatorError(
            "the 2SLS system is singular: the instruments carry no first-stage signal"
        )
    beta = np.linalg.solve(cross, projected.T @ stacked_outcome)
    bread = np.linalg.inv(cross)
    residual = stacked_outcome - design @ beta
    scores = projected[:n] * residual[:n, None] + projected[n:] * residual[n:, None]
    covariance = bread @ (scores.T @ scores) @ bread.T
    # The sandwich is positive semi-definite by construction; the clip only removes rounding
    # below zero.
    se = float(np.sqrt(max(float(covariance[SLOPE_COLUMN, SLOPE_COLUMN]), 0.0)))
    return StackedFit(
        slope=float(beta[SLOPE_COLUMN]),
        se=se,
        first_stage_a_on_b=float(first_stage[INSTRUMENT_B_ROW, SLOPE_COLUMN]),
        first_stage_b_on_a=float(first_stage[INSTRUMENT_A_ROW, SLOPE_COLUMN]),
    )


def _residual(values: np.ndarray, gold: np.ndarray) -> np.ndarray:
    """The residual of the least-squares line of the values on the gold values."""
    slope = sample_covariance(values, gold) / sample_variance(gold)
    return values - (values.mean() + slope * (gold - gold.mean()))


def _residual_correlation(
    proxy_a: np.ndarray, proxy_b: np.ndarray, labelled: np.ndarray, gold: np.ndarray
) -> float:
    """Correlation of the two proxies' residuals on gold over the labelled rows.

    NaN when a proxy has no residual variance there (it equals a line of the gold values), so
    no error correlation is defined.
    """
    gold_labelled = gold[labelled]
    residual_a = _residual(proxy_a[labelled], gold_labelled)
    residual_b = _residual(proxy_b[labelled], gold_labelled)
    if is_constant(residual_a) or is_constant(residual_b):
        return float("nan")
    return float(np.corrcoef(residual_a, residual_b)[0, 1])


def _first_stage_f(correlation: float, n: int) -> float:
    """The first-stage F statistic of one block with one instrument: ``(n - 2) r^2 / (1 - r^2)``."""
    squared = correlation * correlation
    return float("inf") if squared >= 1.0 else (n - 2) * squared / (1.0 - squared)


@dataclass(frozen=True, eq=False)
class _OrivFit:
    """Equation (6) on the two proxies, with their calibration lines and correlation."""

    fit: StackedFit
    calibration_a: ReverseCalibration
    calibration_b: ReverseCalibration
    correlation: float


def _checked_floor(floor: float) -> float:
    """The weak-instrument floor: a real number strictly between 0 and 1."""
    if (
        isinstance(floor, bool)
        or not isinstance(floor, numbers.Real)
        or not 0.0 < float(floor) < 1.0
    ):
        raise SettingsError(
            f"weak_instrument_floor must be a number strictly between 0 and 1; got {floor!r}"
        )
    return float(floor)


def _oriv_fit(
    values_a: np.ndarray,
    values_b: np.ndarray,
    response: np.ndarray,
    labelled: np.ndarray,
    gold: np.ndarray,
    floor: float,
) -> _OrivFit:
    """Reverse-calibrate both measurements on the labelled rows and estimate equation (6)."""
    calibration_a = reverse_calibrate(values_a, labelled, gold)
    calibration_b = reverse_calibrate(values_b, labelled, gold)
    correlation = float(np.corrcoef(calibration_a.proxy, calibration_b.proxy)[0, 1])
    if abs(correlation) < floor:
        raise EstimatorError(
            f"weak instruments: the two proxies correlate at {correlation:.3f}, below the floor "
            f"of {floor:.2f} in absolute value"
        )
    return _OrivFit(
        fit=stacked_2sls(response, calibration_a.proxy, calibration_b.proxy),
        calibration_a=calibration_a,
        calibration_b=calibration_b,
        correlation=correlation,
    )


def oriv(
    data: CorrectionData,
    outcome: str,
    *,
    score_a: str,
    score_b: str,
    n_boot: int = 200,
    seed: int = 42,
    weak_instrument_floor: float = WEAK_INSTRUMENT_FLOOR,
) -> Estimate:
    """The slope of an outcome on the construct by instrumenting one measurement with another.

    Args:
        data: The correction data with gold on a random sample of at least three labelled rows
            (used only to put both measurements on the construct's scale).
        outcome: A numeric column of the data, finite on every text.
        score_a: The first measurement, a score column of the data.
        score_b: The second measurement, another score column whose error is independent of the
            first's.
        n_boot: Number of bootstrap resamples, at least ``MIN_FINITE_BOOTS`` (50).
        seed: Seed of the bootstrap, an integer of at least 0.
        weak_instrument_floor: The smallest absolute correlation of the two proxies accepted,
            strictly between 0 and 1.

    Returns:
        The estimate, with method ``oriv``, the bootstrap standard error, and details
        ``score_a``, ``score_b``, ``outcome``, ``instrument_correlation`` (the correlation of
        the two proxies over every text), ``shared_error_correlation`` (NaN when a proxy has no
        residual on the labelled rows), ``first_stage`` (read-only: ``a_on_b``, ``b_on_a`` and
        ``f_statistic``), ``calibration_a`` and ``calibration_b`` (the intercept and slope of
        each reverse calibration), ``weak_instrument_floor`` (the floor used), ``sandwich_se``
        (the 2SLS sandwich clustered by text), ``n_boot``, ``n_boot_undefined`` and
        ``n_boot_defined``.

    Raises:
        SettingsError: If the two measurements are the same column, or ``n_boot``, ``seed`` or
            ``weak_instrument_floor`` is outside its range.
        ConfigurationError: If the outcome or a measurement does not match the data.
        EstimatorError: If fewer than three texts are labelled, a reverse calibration has no
            line, the proxies' correlation is below the floor in absolute value, the 2SLS
            system is singular, or fewer than ``MIN_FINITE_BOOTS`` resamples have a defined
            estimate.
    """
    if score_a == score_b:
        raise SettingsError("score_a and score_b must differ: ORIV needs two measurements")
    n_boot = check_count("n_boot", n_boot, MIN_FINITE_BOOTS)
    seed = check_count("seed", seed, 0)
    floor = _checked_floor(weak_instrument_floor)
    response = outcome_values(data, outcome)
    values_a, name_a = score_values(data, score_a)
    values_b, name_b = score_values(data, score_b)
    require_labelled(data.n_labelled, MIN_LABELLED, "ORIV")
    labelled, gold = data.labelled_mask, data.gold
    estimate = _oriv_fit(values_a, values_b, response, labelled, gold, floor)

    def resample_slope(rows: np.ndarray, sample: np.ndarray) -> float:
        return _oriv_fit(
            values_a[rows], values_b[rows], response[rows], sample, gold[rows], floor
        ).fit.slope

    draws = stratified_bootstrap(
        resample_slope, labelled, n_boot=n_boot, rng=np.random.default_rng(seed)
    )
    se = bootstrap_se(draws, n_boot)
    fit = estimate.fit
    logger.info(
        "ORIV slope %.4g (SE %.3g, sandwich %.3g), instrument correlation %.3f",
        fit.slope,
        se,
        fit.se,
        estimate.correlation,
    )
    return make_estimate(
        fit.slope,
        se,
        n_labelled=int(labelled.sum()),
        n_total=data.n,
        method="oriv",
        details={
            "score_a": name_a,
            "score_b": name_b,
            "outcome": outcome,
            "instrument_correlation": estimate.correlation,
            "shared_error_correlation": _residual_correlation(
                estimate.calibration_a.proxy, estimate.calibration_b.proxy, labelled, gold
            ),
            "first_stage": MappingProxyType(
                {
                    "a_on_b": fit.first_stage_a_on_b,
                    "b_on_a": fit.first_stage_b_on_a,
                    "f_statistic": _first_stage_f(estimate.correlation, data.n),
                }
            ),
            "calibration_a": (estimate.calibration_a.intercept, estimate.calibration_a.slope),
            "calibration_b": (estimate.calibration_b.intercept, estimate.calibration_b.slope),
            "weak_instrument_floor": floor,
            "sandwich_se": fit.se,
            "n_boot": n_boot,
            "n_boot_undefined": draws.n_undefined,
            "n_boot_defined": n_boot - draws.n_undefined,
        },
    )


SPEC = register(
    FamilySpec(
        family="instrumented_coding",
        variants=("oriv",),
        target="D",
        level="estimate",
        errors_addressed=(),
        estimate_errors=ESTIMATE_ERRORS,
        factory=None,
    )
)
