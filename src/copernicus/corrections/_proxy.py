"""The classical proxy of the construct and the stratified bootstrap of the classical estimators.

Reverse calibration: on the labelled rows, the regression of the score on the gold value,
``s = c + d g + u`` with ``d = cov(s, g) / var(g)`` and ``c = mean(s) - d mean(g)``, gives the
proxy ``W = (s - c) / d = g + u / d`` on every text. It measures the construct on the construct's
own scale, and its error is uncorrelated with the gold value on the labelled rows by
construction. Moments are sample moments (divisor ``n - 1``) throughout.

Stratified bootstrap: each resample draws the labelled rows and the unlabelled rows separately,
with replacement and each to its own count, so every resample keeps the size of the human sample
and of the corpus. An estimate that is undefined on a resample (it raises ``EstimatorError``) is
counted and left out; the standard error is the standard deviation of the defined estimates and
needs at least ``MIN_FINITE_BOOTS`` of them. Texts are resampled, not groups, so the bootstrap
treats the texts as independent.
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
    "MIN_FINITE_BOOTS",
    "MIN_LABELLED",
    "BootstrapDraws",
    "ReverseCalibration",
    "bootstrap_se",
    "check_count",
    "reverse_calibrate",
    "sample_covariance",
    "sample_variance",
    "stratified_bootstrap",
]

MIN_FINITE_BOOTS = 50
MIN_LABELLED = 3


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
            constant there or does not covary with the gold values (no line, so no proxy).
    """
    score_labelled, gold_labelled = score[labelled], gold[labelled]
    if is_constant(gold_labelled):
        raise EstimatorError(
            "the reverse calibration needs gold values that vary on the labelled rows; they "
            "are constant"
        )
    covariance = sample_covariance(score_labelled, gold_labelled)
    if is_constant(score_labelled) or covariance == 0.0:
        raise EstimatorError(
            "the score does not covary with gold on the labelled rows: there is no reverse "
            "calibration line and no proxy"
        )
    slope = covariance / sample_variance(gold_labelled)
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
        # undefined on this resample is counted, and bootstrap_se refuses too many of them.
        try:
            values.append(estimate(rows, in_sample))
        except EstimatorError as error:
            undefined += 1
            logger.debug("bootstrap resample undefined: %s", error)
    return BootstrapDraws(values=np.asarray(values, dtype=float), n_undefined=undefined)


def bootstrap_se(draws: BootstrapDraws, n_boot: int) -> float:
    """The standard deviation (divisor ``n - 1``) of the defined bootstrap estimates.

    Args:
        draws: The bootstrap's defined estimates.
        n_boot: The number of resamples drawn, for the message.

    Returns:
        The standard error.

    Raises:
        EstimatorError: If fewer than ``MIN_FINITE_BOOTS`` resamples have a defined estimate.
    """
    defined = int(draws.values.size)
    if defined < MIN_FINITE_BOOTS:
        raise EstimatorError(
            f"only {defined} of {n_boot} bootstrap resamples have a defined estimate; the "
            f"standard error needs at least {MIN_FINITE_BOOTS}"
        )
    return float(np.std(draws.values, ddof=1))
