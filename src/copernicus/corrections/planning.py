"""The size planner: how many human-coded texts a corrected estimate is worth.

``plan_sample_size`` turns the correlation of the machine score with the human value into the
effective-size factor of a corrected estimate, and ``plan_table`` lays the factor out over
candidate human and machine sample sizes. The planner reads no data; the draw of the human
sample itself lives in ``sampling``.
"""

from __future__ import annotations

import logging
import numbers
from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

from copernicus.errors import SettingsError

logger = logging.getLogger(__name__)

__all__ = ["SizePlan", "plan_sample_size", "plan_table"]

SIZE_PLAN_COLUMNS = ("rho", "n_human", "n_machine", "factor", "limit", "uncorrected_equivalent")


@dataclass(frozen=True)
class SizePlan:
    """Effective-size factor of a corrected estimate for one design.

    Attributes:
        rho: Correlation between the machine score and the human value.
        n_human: Number of human-coded texts.
        n_machine: Number of machine-scored texts.
        factor: ``1 / ((1 - rho^2) + rho^2 * n_human / n_machine)``; the human sample
            counts as ``factor * n_human`` human-coded texts for an estimator that uses the
            score at its best (see `plan_sample_size`).
        limit: ``1 / (1 - rho^2)``; the factor as ``n_machine`` grows without bound.
        uncorrected_equivalent: ``1 / rho^2``; human texts per machine-scored text that
            a plain estimate from the machine score is worth.
    """

    rho: float
    n_human: int
    n_machine: int
    factor: float
    limit: float
    uncorrected_equivalent: float


def plan_sample_size(rho: float, n_human: int, n_machine: int) -> SizePlan:
    """Effective-size factor of a corrected estimate from a human sample and many machine scores.

    The factor is the ratio of the variance of the mean of ``n_human`` human values to the
    variance of a prediction-powered mean whose prediction enters with the weight
    ``cov(gold, score) / var(score)``. That holds for the power-tuned estimator,
    ``mean_ppi(data, power_tuning=True)``, and for a prediction put on the gold scale by the
    least-squares line, the ``calibrated`` rectifier of ``pseudo_values``. The default
    ``mean_ppi(data)`` adds the raw score with the weight 1, which reaches the factor only
    when the score already lies on the gold scale with that slope; a score of another spread
    is worth fewer human texts, so a study sized by this factor should estimate with power
    tuning. The texts are counted as independent; under whole-group draws of correlated
    texts the effective size is smaller.

    Args:
        rho: Correlation of the machine score with the human value, in ``(0, 1)``.
        n_human: Number of human-coded texts, at least 2.
        n_machine: Number of machine-scored texts, at least ``n_human``.

    Returns:
        The factor ``1 / ((1 - rho^2) + rho^2 * n_human / n_machine)``, its limit
        ``1 / (1 - rho^2)``, and the uncorrected equivalent ``1 / rho^2``.

    Raises:
        SettingsError: If an argument is outside its range or a size is not an integer.
    """
    if not 0.0 < rho < 1.0:
        raise SettingsError(f"rho must lie strictly between 0 and 1, got {rho!r}")
    for name, count in (("n_human", n_human), ("n_machine", n_machine)):
        if isinstance(count, bool) or not isinstance(count, numbers.Integral):
            raise SettingsError(f"{name} must be an integer, got {count!r}")
    if n_human < 2:
        raise SettingsError(f"n_human must be at least 2, got {n_human}")
    if n_machine < n_human:
        raise SettingsError(f"n_machine must be at least n_human; got {n_machine} < {n_human}")
    rho_squared = rho * rho
    return SizePlan(
        rho=rho,
        n_human=int(n_human),
        n_machine=int(n_machine),
        factor=1.0 / ((1.0 - rho_squared) + rho_squared * n_human / n_machine),
        limit=1.0 / (1.0 - rho_squared),
        uncorrected_equivalent=1.0 / rho_squared,
    )


def plan_table(
    rho: float, n_human_values: Sequence[int], n_machine_values: Sequence[int]
) -> pd.DataFrame:
    """A size-plan table with one row per pair of human and machine sample sizes.

    Args:
        rho: Correlation of the machine score with the human value.
        n_human_values: Candidate numbers of human-coded texts (the outer order).
        n_machine_values: Candidate numbers of machine-scored texts (the inner order).

    Returns:
        A frame with the columns of ``SizePlan``.

    Raises:
        SettingsError: If a list is empty or a pair is outside the range of
            ``plan_sample_size``.
    """
    if len(n_human_values) == 0 or len(n_machine_values) == 0:
        raise SettingsError("n_human_values and n_machine_values must not be empty")
    plans = [
        plan_sample_size(rho, n_human, n_machine)
        for n_human in n_human_values
        for n_machine in n_machine_values
    ]
    return pd.DataFrame(
        {name: [getattr(plan, name) for plan in plans] for name in SIZE_PLAN_COLUMNS}
    )
