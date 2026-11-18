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
system) is counted, as for the other bootstrapped estimators, and the standard error needs
``min_defined`` (50) defined resamples: with fewer, ``below_floor="refuse"`` refuses the
estimate and ``below_floor="point"`` returns its point with the standard error and the interval
NaN.

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
    bootstrap_standard_error,
    check_bootstrap,
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
RESIDUAL_OK = "ok"
RESIDUAL_CONSTANT = "constant_residual"
"""The status of a residual correlation that is NaN because a proxy's residual is constant."""
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

    @property
    def se_exactly_zero(self) -> bool:
        """Whether the computed standard error equals 0."""
        return self.se == 0.0


def stacked_2sls(outcome: np.ndarray, proxy_a: np.ndarray, proxy_b: np.ndarray) -> StackedFit:
    """Estimate equation (6): ``(y; y)`` on ``(W_A; W_B)``, instrumented by ``(W_B; W_A)``.

    Args:
        outcome: The outcome of every text.
        proxy_a: The first measurement of every text.
        proxy_b: The second measurement of every text.

    Returns:
        The slope, its text-clustered standard error and the two first-stage coefficients. The
        standard error comes from each text's score row transformed by the inverse of the
        system's matrix (a solve, then the sum of squares), so it cannot come out negative.

    Raises:
        EstimatorError: If the instrument matrix is rank-deficient (a measurement that does not
            vary, or two measurements whose sizes differ by many orders of magnitude; the
            message gives the rank) or the instruments carry no first-stage signal.
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
    rank = int(np.linalg.matrix_rank(instruments))
    if rank < instruments.shape[1]:
        raise EstimatorError(
            "the stacked two-stage system is rank-deficient: its instrument matrix has rank "
            f"{rank} of {instruments.shape[1]}; check for a measurement that does not vary on the "
            "texts, and for two measurements whose sizes differ by many orders of magnitude, "
            "which you rescale to comparable sizes"
        )
    first_stage = np.linalg.solve(instruments.T @ instruments, instruments.T @ design)
    projected = instruments @ first_stage
    cross = projected.T @ design
    if int(np.linalg.matrix_rank(cross)) < cross.shape[0]:
        raise EstimatorError(
            "the 2SLS system is singular: the instruments carry no first-stage signal"
        )
    beta = np.linalg.solve(cross, projected.T @ stacked_outcome)
    residual = stacked_outcome - design @ beta
    scores = projected[:n] * residual[:n, None] + projected[n:] * residual[n:, None]
    # The sandwich inverse(cross) S'S inverse(cross)' as the Gram matrix of the transformed score
    # rows: the slope's variance is a sum of squares and cannot come out negative.
    transformed = np.linalg.solve(cross, scores.T).T[:, SLOPE_COLUMN]
    se = float(np.sqrt(transformed @ transformed))
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
) -> tuple[float, str]:
    """Correlation of the two proxies' residuals on gold over the labelled rows, and its status.

    NaN with the status ``constant_residual`` when the residual of a proxy is constant there by
    ``is_constant`` (the proxy is a line of the gold values), so no error correlation is
    defined; otherwise the status is ``ok``.
    """
    gold_labelled = gold[labelled]
    residual_a = _residual(proxy_a[labelled], gold_labelled)
    residual_b = _residual(proxy_b[labelled], gold_labelled)
    if is_constant(residual_a) or is_constant(residual_b):
        return float("nan"), RESIDUAL_CONSTANT
    return float(np.corrcoef(residual_a, residual_b)[0, 1]), RESIDUAL_OK


def _first_stage_f(correlation: float, n: int) -> float:
    """The first-stage F statistic of one block with one instrument: ``(n - 2) r^2 / (1 - r^2)``.

    Infinite when ``r^2`` is 1 or more, as for two proxies on one line; rounding can leave
    their ``r^2`` just below 1, and the statistic is then very large.
    """
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
    min_defined: int = MIN_FINITE_BOOTS,
    below_floor: str = "refuse",
) -> Estimate:
    """The slope of an outcome on the construct by instrumenting one measurement with another.

    Args:
        data: The correction data with gold on a random sample of at least three labelled rows
            (used only to put both measurements on the construct's scale).
        outcome: A numeric column of the data, finite on every text.
        score_a: The first measurement, a score column of the data.
        score_b: The second measurement, another score column whose error is independent of the
            first's.
        n_boot: Number of bootstrap resamples, an integer of at least ``min_defined``.
        seed: Seed of the bootstrap, an integer of at least 0.
        weak_instrument_floor: The smallest absolute correlation of the two proxies accepted,
            strictly between 0 and 1.
        min_defined: The fewest resamples with a defined estimate from which the standard
            error is computed, an integer from 2 to ``n_boot``; by default ``MIN_FINITE_BOOTS``
            (50).
        below_floor: What happens when fewer resamples have a defined estimate: ``"refuse"``
            raises ``EstimatorError``; ``"point"`` returns the point with ``se``, ``ci_low``
            and ``ci_high`` NaN and the details' ``se_status`` ``too_few_defined``, and logs
            one warning that names the counts; ``sandwich_se`` is reported either way.

    Returns:
        The estimate, with method ``oriv``, the bootstrap standard error, and details
        ``score_a``, ``score_b``, ``outcome``, ``instrument_correlation`` (the correlation of
        the two proxies over every text), ``shared_error_correlation`` with
        ``shared_error_correlation_status`` (``ok``; ``constant_residual`` when the residual of
        a proxy on the gold values is constant on the labelled rows by ``is_constant``, so that
        no correlation is defined and the value is NaN), ``first_stage`` (read-only:
        ``a_on_b``, ``b_on_a`` and ``f_statistic``, ``(n - 2) r^2 / (1 - r^2)`` of the proxies'
        correlation ``r`` over the ``n`` texts, infinite when ``r^2`` is 1 or more),
        ``calibration_a`` and ``calibration_b`` (the intercept and slope of
        each reverse calibration), ``weak_instrument_floor`` (the floor used), ``sandwich_se``
        (the 2SLS sandwich clustered by text) with ``sandwich_se_exactly_zero`` (True when that
        sandwich is exactly 0, as for an outcome without any residual), ``n_boot``,
        ``n_boot_undefined``, ``n_boot_defined``, ``min_defined`` (the floor that applied),
        ``below_floor``,
        ``se_status`` (``ok`` when the standard error was computed, ``too_few_defined`` when
        fewer than ``min_defined`` resamples were defined and the point was kept without it)
        and ``se_exactly_zero`` (True when the bootstrap standard error is exactly 0, as when
        the defined estimates of the resamples are all equal; ``se_status`` is then ``ok``; see
        ``make_estimate``).

    Raises:
        SettingsError: If the two measurements are the same column, or ``n_boot``, ``seed``,
            ``weak_instrument_floor``, ``min_defined`` or ``below_floor`` is outside its range.
        ConfigurationError: If the outcome or a measurement does not match the data.
        EstimatorError: If fewer than three texts are labelled, a reverse calibration has no
            line, the proxies' correlation is below the floor in absolute value, the stacked
            system is rank-deficient or its instruments carry no first-stage signal, or fewer
            than ``min_defined`` resamples have a defined estimate and ``below_floor`` is
            ``"refuse"``.
    """
    if score_a == score_b:
        raise SettingsError("score_a and score_b must differ: ORIV needs two measurements")
    bootstrap = check_bootstrap(n_boot, min_defined, below_floor)
    n_boot = bootstrap.n_boot
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
    standard_error = bootstrap_standard_error(draws, bootstrap, method="oriv", log=logger)
    se = standard_error.se
    fit = estimate.fit
    shared, shared_status = _residual_correlation(
        estimate.calibration_a.proxy, estimate.calibration_b.proxy, labelled, gold
    )
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
            "shared_error_correlation": shared,
            "shared_error_correlation_status": shared_status,
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
            "sandwich_se_exactly_zero": fit.se_exactly_zero,
            "n_boot": n_boot,
            "n_boot_undefined": draws.n_undefined,
            "n_boot_defined": n_boot - draws.n_undefined,
            **standard_error.details(),
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
