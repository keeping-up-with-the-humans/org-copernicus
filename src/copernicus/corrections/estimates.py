"""Estimates of a statistic of the construct from a human-coded sample and machine scores.

Every estimator returns an ``Estimate``: the point, its standard error, the 95 percent interval
``point +/- 1.96 SE``, the sample sizes, the method and a read-only mapping of what else it
computed.

This module holds the prediction-powered (PPI) estimates of a mean and of the difference of two
group means. With a prediction ``f`` of every text (the first score column unless another is
given; it must already be on the gold scale), the gold values ``y`` of the labelled texts, and
``theta`` the mean or the difference of the two group means::

    point = theta(y, lab) + (theta(f, all) - theta(f, lab))
    SE = sqrt(se(f - y, lab)^2 + se(f, all)^2)

with ``se(mean) = sd / sqrt(n)`` and ``se(difference) = sqrt(var_a / n_a + var_b / n_b)``
(sample moments, divisor ``n - 1``). With ``power_tuning`` the machine terms are scaled by
``lambda_star = cov(y, f) / var(f)`` on the labelled rows (population moments, 1 for a constant
prediction; the helper of ``pseudo_values``)::

    point = theta(y, lab) + lambda_star * (theta(f, all) - theta(f, lab))
    SE = sqrt(se(lambda_star * f - y, lab)^2 + lambda_star^2 * se(f, all)^2)

The formulas weight every labelled text alike, so they need a sample in which every text had
the same inclusion probability, such as the draw of ``draw_human_sample`` attached with
``attach_draw``; a labelled sample without ``pi`` is refused. That draw takes whole groups, so by
default (``cluster=None``) the standard error sums the contributions of the texts within each
group whenever a group holds more than one labelled text (the formula is in ``_clustering``,
and a clustered difference needs labelled texts from two groups on each side); with one
labelled text per group it is the formula above. When every text is labelled, the estimates
equal the gold estimates exactly. Under a whole-group draw with groups of unequal sizes the
points are ratio estimators with a small-sample bias; ``mean_ppi`` explains it and names an
exactly unbiased mean.

The registry family ``ludwig_moment`` lists these estimators (variants ``mean`` and
``difference``) together with the coefficients of ``estimates_ols``.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
import pandas as pd

from copernicus.corrections._clustering import (
    Clustering,
    contributions,
    mean_covariance,
    resolve_cluster,
)
from copernicus.corrections._estimate_inputs import (
    equal_probability_sample,
    group_mask,
    prediction_values,
    require_labelled,
)
from copernicus.corrections.data import CorrectionData
from copernicus.corrections.families._registry import FamilySpec, register
from copernicus.corrections.families.pseudo import tuning_coefficient
from copernicus.errors import EstimatorError

logger = logging.getLogger(__name__)

__all__ = [
    "CI_LEVEL",
    "Z",
    "Estimate",
    "group_difference_ppi",
    "long_short_split",
    "make_estimate",
    "mean_ppi",
    "naive_se",
    "plugin_value",
]

CI_LEVEL = 0.95
Z = 1.96
MIN_LABELLED_PER_GROUP = 2
FAMILY = "ludwig_moment"
VARIANTS = ("mean", "difference", "ols_outcome", "ols_predictor")
ESTIMATE_ERRORS = (
    "bias and coverage of the construct's coefficient as predictor (D) or of means and "
    "differences (C)"
)


@dataclass(frozen=True, eq=False)
class Estimate:
    """An estimate with its standard error and 95 percent interval.

    Attributes:
        point: The estimate.
        se: Its standard error.
        ci_low: ``point - Z * se``.
        ci_high: ``point + Z * se``.
        n_labelled: Number of human-coded texts the estimate used.
        n_total: Number of texts in the data.
        method: The estimator, such as ``ppi_mean`` or ``simex``.
        details: What else the estimator computed, as a read-only mapping (each estimator
            documents its keys). The mapping cannot be changed, and an estimate cannot be
            pickled because of it; copy it with ``dict(estimate.details)`` to keep it.
    """

    point: float
    se: float
    ci_low: float
    ci_high: float
    n_labelled: int
    n_total: int
    method: str
    details: Mapping[str, object]

    def __post_init__(self) -> None:
        """Freeze the details."""
        object.__setattr__(self, "details", MappingProxyType(dict(self.details)))


def make_estimate(
    point: float,
    se: float,
    *,
    n_labelled: int,
    n_total: int,
    method: str,
    details: Mapping[str, object],
) -> Estimate:
    """An estimate with the normal 95 percent interval ``point +/- Z * se``.

    Args:
        point: The estimate.
        se: Its standard error.
        n_labelled: Number of human-coded texts used.
        n_total: Number of texts in the data.
        method: The estimator's name.
        details: What else the estimator computed.

    Returns:
        The estimate.
    """
    return Estimate(
        point=float(point),
        se=float(se),
        ci_low=float(point - Z * se),
        ci_high=float(point + Z * se),
        n_labelled=int(n_labelled),
        n_total=int(n_total),
        method=method,
        details=details,
    )


def _parts(values: np.ndarray, split: np.ndarray | None, minimum: int) -> list[np.ndarray]:
    """The values, or the values of the two groups, each with at least ``minimum`` entries."""
    parts = [values] if split is None else [values[split], values[~split]]
    sizes = [int(part.size) for part in parts]
    if min(sizes) < minimum:
        raise EstimatorError(
            f"the estimand needs at least {minimum} values in each group; the groups hold {sizes}"
        )
    return parts


def plugin_value(values: np.ndarray, split: np.ndarray | None = None) -> float:
    """The mean of the values, or the mean where ``split`` is True minus the mean elsewhere.

    Args:
        values: One value per text.
        split: ``None`` for the mean, or a boolean mask of the first group.

    Returns:
        The plug-in value of the estimand.

    Raises:
        EstimatorError: If a group is empty.
    """
    parts = _parts(values, split, 1)
    if split is None:
        return float(np.mean(parts[0]))
    return float(np.mean(parts[0]) - np.mean(parts[1]))


def naive_se(values: np.ndarray, split: np.ndarray | None = None) -> float:
    """The standard error of ``plugin_value`` when the values are independent draws.

    Args:
        values: One value per text.
        split: ``None`` for the mean, or a boolean mask of the first group.

    Returns:
        ``sd / sqrt(n)``, or ``sqrt(var_a / n_a + var_b / n_b)``, with sample moments.

    Raises:
        EstimatorError: If a group holds fewer than two values.
    """
    parts = _parts(values, split, 2)
    return float(np.sqrt(sum(np.var(part, ddof=1) / part.size for part in parts)))


def long_short_split(data: CorrectionData) -> np.ndarray:
    """The texts whose covariate lies strictly above its median: the long half.

    With the default covariate (the log of the character count) the mask marks the longer
    texts; texts at the median count as short.

    Args:
        data: The correction data.

    Returns:
        A boolean array in data order.
    """
    covariate = data.covariate
    return np.asarray(covariate > np.median(covariate), dtype=bool)


def _term_se(values: np.ndarray, split: np.ndarray | None, groups: np.ndarray | None) -> float:
    """The standard error of one term: of independent texts, or summed within the groups."""
    if groups is None:
        return naive_se(values, split)
    _parts(values, split, MIN_LABELLED_PER_GROUP)
    psi = contributions(values, split)[:, None]
    return float(np.sqrt(mean_covariance(psi, groups)[0, 0]))


def _ppi(
    data: CorrectionData,
    values: np.ndarray,
    name: str,
    split: np.ndarray | None,
    power_tuning: bool,
    clustering: Clustering,
    method: str,
) -> tuple[float, float, dict[str, object]]:
    """The point, the standard error and the details of a PPI mean or difference."""
    labelled = data.labelled_mask
    gold = data.gold[labelled]
    predicted = values[labelled]
    labelled_split = None if split is None else split[labelled]
    lambda_star = tuning_coefficient(predicted, gold) if power_tuning else 1.0
    plugin = plugin_value(values, split)
    predicted_value = plugin_value(predicted, labelled_split)
    gold_value = plugin_value(gold, labelled_split)
    point = gold_value + lambda_star * (plugin - predicted_value)
    se = float(
        np.hypot(
            _term_se(lambda_star * predicted - gold, labelled_split, clustering.labelled_groups),
            lambda_star * _term_se(values, split, clustering.groups),
        )
    )
    logger.debug(
        "%s: %.4g (SE %.3g) from %d labelled of %d texts",
        method,
        point,
        se,
        int(labelled.sum()),
        data.n,
    )
    details: dict[str, object] = {
        "prediction": name,
        "power_tuning": power_tuning,
        "lambda_star": lambda_star,
        "plugin": plugin,
        "rectifier": lambda_star * predicted_value - gold_value,
        **clustering.details(),
    }
    return point, se, details


def mean_ppi(
    data: CorrectionData,
    *,
    prediction: pd.Series | None = None,
    power_tuning: bool = False,
    cluster: bool | None = None,
) -> Estimate:
    """The prediction-powered mean of the construct.

    Under a whole-group draw with groups of unequal sizes the point is a ratio estimator: it
    divides by the number of texts drawn, which varies with the groups drawn, so it carries a
    small-sample bias of the order of one over the number of drawn groups.
    ``pseudo_values(data).mean_rectified`` is the Horvitz-Thompson mean of the same draw:
    exactly unbiased over all possible draws, usually more variable, and without a standard
    error.

    Args:
        data: The correction data with gold on the labelled rows and the same ``pi`` on every
            labelled row (``draw_human_sample`` and ``attach_draw`` supply it).
        prediction: The prediction of every text, indexed by text id (ids compared as strings),
            on the gold scale; ``None`` for the first score column, which is never chosen on
            gold values.
        power_tuning: Whether to scale the machine terms by ``lambda_star``.
        cluster: Whether the standard error sums the contributions of the texts within each
            group; ``None`` does so when a group holds more than one labelled text (as in a
            whole-group draw), ``True`` and ``False`` (numpy booleans too) force the choice.

    Returns:
        The estimate, with method ``ppi_mean`` (``ppi_mean_tuned`` with power tuning) and
        details ``prediction`` (the name of the prediction), ``power_tuning``, ``lambda_star``
        (1 without power tuning), ``plugin`` (the mean of the prediction over every text),
        ``rectifier`` (``lambda_star`` times the prediction's mean on the labelled rows minus
        the gold mean there), so that ``point = lambda_star * plugin - rectifier``, ``cluster``
        (as given), ``clustered`` (the choice made) and, when clustered, ``n_clusters`` and
        ``n_clusters_labelled`` (the groups of every text and of the labelled texts).

    Raises:
        SettingsError: If ``cluster`` is not a boolean or ``None``.
        ConfigurationError: If the prediction does not cover every text with a finite value.
        EstimatorError: If fewer than two texts are labelled, ``pi`` is missing, not finite
            or not the same on the labelled rows, or the estimate clusters and the texts or
            the labelled texts hold fewer than two groups.
    """
    method = "ppi_mean_tuned" if power_tuning else "ppi_mean"
    require_labelled(data.n_labelled, MIN_LABELLED_PER_GROUP, "the prediction-powered mean")
    labelled = equal_probability_sample(data, "the prediction-powered mean")
    clustering = resolve_cluster(data, labelled, cluster)
    values, name = prediction_values(data, prediction)
    point, se, details = _ppi(data, values, name, None, power_tuning, clustering, method)
    return make_estimate(
        point,
        se,
        n_labelled=int(labelled.sum()),
        n_total=data.n,
        method=method,
        details=details,
    )


def group_difference_ppi(
    data: CorrectionData,
    group: pd.Series | np.ndarray,
    *,
    prediction: pd.Series | None = None,
    power_tuning: bool = False,
    cluster: bool | None = None,
) -> Estimate:
    """The prediction-powered difference between the mean of a group and that of the rest.

    Args:
        data: The correction data with gold on the labelled rows and the same ``pi`` on every
            labelled row.
        group: The two-level grouping: a boolean array in data order, or a boolean series
            indexed by text id. The difference is the mean where it is True minus the mean
            where it is False; ``long_short_split`` gives the long-minus-short difference.
        prediction: As for ``mean_ppi``.
        power_tuning: Whether to scale the machine terms by ``lambda_star``, computed on all
            labelled rows.
        cluster: As for ``mean_ppi``; the texts of a group on both sides of the grouping are
            summed together. A clustered difference, by the automatic choice as well, needs
            labelled texts from at least two groups on each side: a side whose labelled texts
            lie in one group would add none of its variation between groups to the standard
            error, so it is refused (``cluster=False`` treats the texts as independent).

    Returns:
        The estimate, with method ``ppi_difference`` (``ppi_difference_tuned``) and the details
        of ``mean_ppi`` plus ``n_group`` (texts in the group), ``n_labelled_group`` and
        ``n_labelled_other`` (labelled texts in the group and outside it).

    Raises:
        SettingsError: If ``cluster`` is not a boolean or ``None``.
        ConfigurationError: If the grouping or the prediction does not match the data.
        EstimatorError: If a group holds fewer than two labelled texts, ``pi`` is missing, not
            finite or not the same on the labelled rows, or the estimate clusters and the texts,
            the labelled texts or the labelled texts of one side hold fewer than two groups.
    """
    method = "ppi_difference_tuned" if power_tuning else "ppi_difference"
    mask = group_mask(data, group)
    labelled = data.labelled_mask
    counts = (int((labelled & mask).sum()), int((labelled & ~mask).sum()))
    require_labelled(
        min(counts), MIN_LABELLED_PER_GROUP, "the prediction-powered difference", " in each group"
    )
    equal_probability_sample(data, "the prediction-powered difference")
    clustering = resolve_cluster(data, labelled, cluster, split=mask)
    values, name = prediction_values(data, prediction)
    point, se, details = _ppi(data, values, name, mask, power_tuning, clustering, method)
    details.update(n_group=int(mask.sum()), n_labelled_group=counts[0], n_labelled_other=counts[1])
    return make_estimate(
        point,
        se,
        n_labelled=int(labelled.sum()),
        n_total=data.n,
        method=method,
        details=details,
    )


SPEC = register(
    FamilySpec(
        family=FAMILY,
        variants=VARIANTS,
        target="C / D",
        level="estimate",
        errors_addressed=(),
        estimate_errors=ESTIMATE_ERRORS,
        factory=None,
    )
)
