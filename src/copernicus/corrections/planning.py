"""The size planner: how many human-coded texts a corrected estimate is worth.

``plan_sample_size`` turns the Pearson correlation of the machine score with the human value
into the effective-size factor of a corrected estimate, and ``plan_table`` lays the factor out
over candidate human and machine sample sizes. The planner reads no data; the draw of the human
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
        rho: Pearson correlation between the machine score and the human value (see
            `plan_sample_size` for the correlation that a regression coefficient needs).
        n_human: Number of human-coded texts.
        n_machine: Number of machine-scored texts.
        factor: ``1 / ((1 - rho^2) + rho^2 * n_human / n_machine)``; the multiplier of the
            human sample, which counts as ``factor * n_human`` human-coded texts for an
            estimator that uses the score at its best (see `plan_sample_size`).
        limit: ``1 / (1 - rho^2)``; the upper limit of the multiplier, which ``factor``
            approaches as ``n_machine`` grows.
        uncorrected_equivalent: ``1 / rho^2``; about the number of machine-scored texts that,
            used without correction, give a test of a small association with a covariate the
            power of one human-coded text. This holds for independent texts whose human value,
            score and covariate are jointly normal, with a score that is a line in the human
            value plus an error independent of the human value and of the covariate. With an
            error whose size depends on the covariate the number can differ.
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
    ``mean_ppi(data)`` adds the raw score with the weight 1. When some scored texts have no
    human value, the weight 1 reaches the factor only when ``cov(gold, score) / var(score)`` is
    1, and with another slope it is worth fewer human texts; the spreads of score and gold alone
    do not show that slope. When every scored text is human-coded, every weight gives the mean
    of the human values. So a study sized by this factor should estimate with power tuning.

    The planner counts the texts as independent and does not model the dependence of texts
    within a group. What matters is the dependence, within a group, of what each text
    contributes to the estimate: a positive correlation of these contributions lowers the
    effective sample size below that of independent texts, and a negative one raises it. Check
    the design with ``repeated_study``.

    ``rho`` is the Pearson correlation of score and human value, computed from the data; a rank
    correlation, such as the rank validity of the validity panel, is another number. For a mean
    that Pearson correlation enters. For a coefficient of a regression with the construct as
    the outcome (``ols_outcome_ppi``), call a residual what the linear fit on the regressors
    leaves unexplained, of the human value or of the score. The same large-sample factor then
    holds with the partial correlation, the correlation of the two residuals, when the
    prediction is the least-squares fit of the human value on the score and the regressors and
    when the mean squares of the two residuals and their mean product, at given values of the
    regressors, do not change with the regressors. Jointly normal human value, score and
    regressors satisfy this condition on the residuals. It is not enough that each residual is,
    on its own, unrelated to the regressors. When the size of the score's error depends on a
    regressor, the corrected coefficient can be less precise than the human-coded sample alone,
    even with that prediction. The raw score as the prediction can also make the coefficient
    worth fewer texts than the human-coded sample alone. ``repeated_study`` checks a design for
    the mean and the long-short difference; for a coefficient, draw human samples the same way
    from a corpus with known human values and compare the estimates with the coefficient of all
    of them.

    The factor is a large-sample number: it treats the weight of the score as known. Estimating
    the weight on the human-coded texts, as power tuning does, lowers the realised factor. For
    independent texts with a jointly normal score and human value, of which a random subset of
    more than three is human-coded and gives the weight, the power-tuned estimate of the mean
    of the population the texts come from falls short of the plan by at most
    ``1 / (n_human - 2)`` of it, an upper value approached when the human-coded texts are a
    small share of the corpus.

    Args:
        rho: Pearson correlation of the machine score with the human value, in ``(0, 1)``.
        n_human: Number of human-coded texts, at least 2.
        n_machine: Number of machine-scored texts, at least ``n_human``.

    Returns:
        The factor ``1 / ((1 - rho^2) + rho^2 * n_human / n_machine)``, its upper limit
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
        rho: Pearson correlation of the machine score with the human value.
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
