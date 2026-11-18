"""The classical proxy of the construct and the stratified bootstrap of the classical estimators.

Reverse calibration: on the labelled rows, the regression of the score on the gold value,
``s = c + d g + u`` with ``d = cov(s, g) / var(g)`` and ``c = mean(s) - d mean(g)``, gives the
proxy ``W = (s - c) / d = g + u / d`` on every text. It measures the construct on the construct's
own scale, and its error is uncorrelated with the gold value on the labelled rows by
construction. Moments are sample moments (divisor ``n - 1``) throughout. The line is flat, and
there is no proxy, when its rise over the labelled rows, ``|d|`` times the range of the gold
values, is at most ``1e-12`` of the range of the score there: the rule for a constant column
(``is_constant``) applied to ``d (g - mean(g))`` in units of the score's range, which does not
depend on the units of the score or of the gold values. A covariance that is zero up to
rounding (a correlation of the order of ``1e-16``) falls below it; a correlation of a few times
``1e-12`` lies above it.

Stratified bootstrap: each resample draws the labelled rows and the unlabelled rows separately,
with replacement and each to its own count, so every resample keeps the size of the human sample
and of the corpus. An estimate that is undefined on a resample (it raises ``EstimatorError``) is
counted and left out; the standard error is the standard deviation of the defined estimates and
needs at least ``min_defined`` of them, ``MIN_FINITE_BOOTS`` (50) unless the estimator is given
another floor. With fewer, ``below_floor="refuse"`` refuses the estimate, and
``below_floor="point"`` keeps its point without a standard error, logs one warning and records
the status ``too_few_defined``. Texts are resampled, not groups, so the bootstrap treats the
texts as independent.
"""

from __future__ import annotations

import logging
import numbers
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from copernicus.errors import EstimatorError, SettingsError
from copernicus.validity.metrics import is_constant

logger = logging.getLogger(__name__)

__all__ = [
    "BELOW_FLOOR_RULES",
    "MIN_DEFINED_LOWEST",
    "MIN_FINITE_BOOTS",
    "MIN_LABELLED",
    "POINT",
    "REFUSE",
    "SE_OK",
    "SE_TOO_FEW_DEFINED",
    "BootstrapDraws",
    "BootstrapSE",
    "BootstrapSettings",
    "ReverseCalibration",
    "bootstrap_se",
    "bootstrap_standard_error",
    "check_bootstrap",
    "check_count",
    "reverse_calibrate",
    "sample_covariance",
    "sample_variance",
    "stratified_bootstrap",
]

MIN_FINITE_BOOTS = 50
# A standard deviation with divisor n - 1 needs two values.
MIN_DEFINED_LOWEST = 2
MIN_LABELLED = 3
REFUSE = "refuse"
POINT = "point"
BELOW_FLOOR_RULES = (REFUSE, POINT)
SE_OK = "ok"
SE_TOO_FEW_DEFINED = "too_few_defined"
_NO_LINE = (
    "the score does not covary with gold on the labelled rows: there is no reverse calibration "
    "line and no proxy"
)


def sample_covariance(first: np.ndarray, second: np.ndarray) -> float:
    """The sample covariance (divisor ``n - 1``)."""
    return float(np.cov(first, second, ddof=1)[0, 1])


def sample_variance(values: np.ndarray) -> float:
    """The sample variance (divisor ``n - 1``)."""
    return float(np.var(values, ddof=1))


def check_count(name: str, value: int, minimum: int) -> int:
    """An integer setting of at least ``minimum``.

    Args:
        name: The setting's name, for the message.
        value: The value given.
        minimum: The smallest admissible value.

    Returns:
        The value as an ``int``.

    Raises:
        SettingsError: If the value is not an integer or is below ``minimum``.
    """
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < minimum:
        raise SettingsError(f"{name} must be an integer of at least {minimum}; got {value!r}")
    return int(value)


@dataclass(frozen=True)
class BootstrapSettings:
    """The checked bootstrap settings of a classical estimator.

    Attributes:
        n_boot: The number of resamples drawn, at least ``min_defined``.
        min_defined: The fewest resamples with a defined estimate from which the standard
            error is computed, at least 2.
        below_floor: What happens with fewer: ``"refuse"`` refuses the estimate, ``"point"``
            keeps its point without a standard error.
    """

    n_boot: int
    min_defined: int
    below_floor: str


def check_bootstrap(n_boot: int, min_defined: int, below_floor: str) -> BootstrapSettings:
    """Check the bootstrap settings of a classical estimator.

    ``min_defined`` is checked first and ``n_boot`` against it, so under the default floor of
    50 ``n_boot`` must be at least 50.

    Args:
        n_boot: Number of resamples.
        min_defined: The fewest resamples with a defined estimate that a standard error needs.
        below_floor: ``"refuse"`` or ``"point"``.

    Returns:
        The settings.

    Raises:
        SettingsError: If ``min_defined`` is not an integer of at least 2, ``n_boot`` is not
            an integer of at least ``min_defined``, or ``below_floor`` is neither ``"refuse"``
            nor ``"point"``.
    """
    floor = check_count("min_defined", min_defined, MIN_DEFINED_LOWEST)
    count = check_count("n_boot", n_boot, floor)
    if not isinstance(below_floor, str) or below_floor not in BELOW_FLOOR_RULES:
        raise SettingsError(f"below_floor must be 'refuse' or 'point'; got {below_floor!r}")
    return BootstrapSettings(n_boot=count, min_defined=floor, below_floor=below_floor)


@dataclass(frozen=True, eq=False)
class ReverseCalibration:
    """The proxy of the construct from the regression of the score on the gold values.

    Attributes:
        proxy: ``W = (s - intercept) / slope`` for every text.
        intercept: ``c``, the score at a gold value of zero.
        slope: ``d``, the change of the score per unit of gold.
    """

    proxy: np.ndarray
    intercept: float
    slope: float


def reverse_calibrate(
    score: np.ndarray, labelled: np.ndarray, gold: np.ndarray
) -> ReverseCalibration:
    """Put a score on the construct's scale with the regression of the score on gold.

    Args:
        score: The score of every text.
        labelled: Boolean mask of the labelled rows.
        gold: The gold values, read on the labelled rows only.

    Returns:
        The proxy and the line.

    Raises:
        EstimatorError: If the gold values are constant on the labelled rows, or the score is
            constant there or does not covary with the gold values: the line's rise over the
            labelled rows is at most ``1e-12`` of the score's range there (no line, so no
            proxy).
    """
    score_labelled, gold_labelled = score[labelled], gold[labelled]
    if is_constant(gold_labelled):
        raise EstimatorError(
            "the reverse calibration needs gold values that vary on the labelled rows; they "
            "are constant"
        )
    if is_constant(score_labelled):
        raise EstimatorError(_NO_LINE)
    slope = sample_covariance(score_labelled, gold_labelled) / sample_variance(gold_labelled)
    if is_constant(slope * (gold_labelled - gold_labelled.mean()) / float(np.ptp(score_labelled))):
        raise EstimatorError(_NO_LINE)
    intercept = float(score_labelled.mean() - slope * gold_labelled.mean())
    return ReverseCalibration(proxy=(score - intercept) / slope, intercept=intercept, slope=slope)


@dataclass(frozen=True, eq=False)
class BootstrapDraws:
    """The defined estimates of a stratified bootstrap.

    Attributes:
        values: The estimates of the resamples on which the estimate is defined.
        n_undefined: The number of resamples on which it is not.
    """

    values: np.ndarray
    n_undefined: int


def stratified_bootstrap(
    estimate: Callable[[np.ndarray, np.ndarray], float],
    labelled: np.ndarray,
    *,
    n_boot: int,
    rng: np.random.Generator,
) -> BootstrapDraws:
    """Re-estimate on resamples that redraw the labelled and the unlabelled rows separately.

    Args:
        estimate: The estimate on a resample, called with the resample's row positions (into
            the original data; the labelled block first) and its labelled mask. It raises
            ``EstimatorError`` where the estimate is undefined.
        labelled: Boolean mask of the labelled rows of the data.
        n_boot: Number of resamples.
        rng: Generator of the resamples (and of any randomness inside ``estimate``).

    Returns:
        The defined estimates and the count of undefined ones.
    """
    labelled_rows = np.flatnonzero(labelled)
    other_rows = np.flatnonzero(~labelled)
    in_sample = np.zeros(labelled.size, dtype=bool)
    in_sample[: labelled_rows.size] = True
    values: list[float] = []
    undefined = 0
    for _ in range(n_boot):
        rows = np.concatenate(
            [
                rng.choice(labelled_rows, size=labelled_rows.size, replace=True),
                rng.choice(other_rows, size=other_rows.size, replace=True),
            ]
        )
        # The one failure tolerated per resample: an estimate that its own rules leave
        # undefined on this resample is counted, and the floor of the standard error
        # (bootstrap_standard_error) decides what too many of them mean.
        try:
            values.append(estimate(rows, in_sample))
        except EstimatorError as error:
            undefined += 1
            logger.debug("bootstrap resample undefined: %s", error)
    return BootstrapDraws(values=np.asarray(values, dtype=float), n_undefined=undefined)


def bootstrap_se(
    draws: BootstrapDraws, n_boot: int, *, min_defined: int = MIN_FINITE_BOOTS
) -> float:
    """The standard deviation (divisor ``n - 1``) of the defined bootstrap estimates.

    Args:
        draws: The bootstrap's defined estimates.
        n_boot: The number of resamples drawn, for the message.
        min_defined: The fewest defined estimates the standard error needs, at least 2.

    Returns:
        The standard error.

    Raises:
        SettingsError: If ``min_defined`` is not an integer of at least 2.
        EstimatorError: If fewer than ``min_defined`` resamples have a defined estimate.
    """
    floor = check_count("min_defined", min_defined, MIN_DEFINED_LOWEST)
    defined = int(draws.values.size)
    if defined < floor:
        raise EstimatorError(
            f"only {defined} of {n_boot} bootstrap resamples have a defined estimate; the "
            f"standard error needs at least {floor}"
        )
    return float(np.std(draws.values, ddof=1))


@dataclass(frozen=True, eq=False)
class BootstrapSE:
    """The standard error of a stratified bootstrap and the floor that decided it.

    Attributes:
        se: The standard deviation (divisor ``n - 1``) of the defined estimates; NaN when
            fewer than ``settings.min_defined`` resamples have a defined estimate, which
            only ``below_floor="point"`` returns.
        n_defined: The number of resamples with a defined estimate.
        settings: The bootstrap settings that applied.
    """

    se: float
    n_defined: int
    settings: BootstrapSettings

    @property
    def status(self) -> str:
        """``ok`` when the standard error was computed, ``too_few_defined`` when it was not."""
        return SE_OK if self.n_defined >= self.settings.min_defined else SE_TOO_FEW_DEFINED

    def details(self) -> dict[str, object]:
        """The details keys of every bootstrapped estimate: the floor, the rule, the status."""
        return {
            "min_defined": self.settings.min_defined,
            "below_floor": self.settings.below_floor,
            "se_status": self.status,
        }


def bootstrap_standard_error(
    draws: BootstrapDraws, settings: BootstrapSettings, *, method: str, log: logging.Logger
) -> BootstrapSE:
    """The standard error of the defined estimates, refused or left out below the floor.

    Args:
        draws: The bootstrap's defined estimates.
        settings: The checked bootstrap settings.
        method: The estimator's name, which opens the warning.
        log: The estimator's module logger, which logs the warning.

    Returns:
        The standard error with its counts and settings. When fewer than
        ``settings.min_defined`` resamples are defined and ``below_floor`` is ``"point"``, the
        standard error is NaN, its status is ``too_few_defined``, and one warning that names
        the counts is logged.

    Raises:
        EstimatorError: If fewer than ``settings.min_defined`` resamples are defined and
            ``below_floor`` is ``"refuse"``.
    """
    defined = int(draws.values.size)
    if defined >= settings.min_defined or settings.below_floor == REFUSE:
        se = bootstrap_se(draws, settings.n_boot, min_defined=settings.min_defined)
        return BootstrapSE(se=se, n_defined=defined, settings=settings)
    log.warning(
        "%s: only %d of %d bootstrap resamples have a defined estimate, fewer than the %d that "
        "a standard error needs (min_defined), so the estimate keeps its point and has no "
        "standard error and no interval (below_floor='point')",
        method,
        defined,
        settings.n_boot,
        settings.min_defined,
    )
    return BootstrapSE(se=float("nan"), n_defined=defined, settings=settings)
