"""Unit-level test-fold metrics and their weighted aggregation over folds.

Predictions and gold are averaged within the analysis unit inside each test fold; metrics
are computed on those unit means; fold values are then weighted by the fold's unit count.
Binary gold adds AUC and Somers' D (``2 * AUC - 1``).

``mae_01`` is the mean absolute error after min-max scaling both prediction and gold with
the training draw's gold range (the ridge predicts on the gold scale, so one common scale
keeps level information and stays free of test-fold leakage); ``mae_native`` is the unscaled
error.

Each fold also gets a diagnostic ``status``: ``ok``, ``constant_gold``, ``constant_pred``,
``single_class``, ``too_few_units`` or ``nonbinary_unit_gold``. The status is a label only;
it never gates an individual metric, whose eligibility is whether its own value is finite.
In particular AUC and Somers' D are computed whenever the fold's unit-mean gold takes the
two class values of a binary gold with both present, even on a constant prediction (AUC
0.5, Somers' D 0); only a single-class fold leaves them undefined. The larger class value
is the positive class, as in the validity panel, so any two-valued coding (0 and 1, 1 and 2,
-1 and 1) gives the same AUC. ``nonbinary_unit_gold`` flags a unit-mean gold value outside
the two class values, which means within-unit averaging mixed classes.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from copernicus.errors import ConfigurationError, EstimatorError
from copernicus.validity.metrics import auc_from_ranks, is_constant

logger = logging.getLogger(__name__)

__all__ = [
    "METRIC_KEYS",
    "STATUS_KEYS",
    "aggregate_units",
    "classes_are_separable",
    "fold_metrics",
    "weighted_over_folds",
]

METRIC_KEYS: tuple[str, ...] = (
    "spearman",
    "spearman_abs",
    "pearson",
    "mae_01",
    "mae_native",
    "auc",
    "somers_d",
)
"""The metrics computed for each fold of the baseline."""

STATUS_KEYS: tuple[str, ...] = (
    "ok",
    "constant_gold",
    "constant_pred",
    "single_class",
    "too_few_units",
    "nonbinary_unit_gold",
)
"""The statuses of a fold's metrics; the results table counts the folds of each."""

MIN_UNITS = 3
CLASS_TOLERANCE = 1e-12


def aggregate_units(
    pred: np.ndarray, gold: np.ndarray, unit_id: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean prediction and gold per analysis unit.

    Args:
        pred: Prediction per row.
        gold: Gold per row.
        unit_id: Analysis unit of each row.

    Returns:
        The sorted unit ids, the mean prediction per unit and the mean gold per unit.

    Raises:
        EstimatorError: If a prediction is not finite.
    """
    frame = pd.DataFrame(
        {
            "unit": unit_id,
            "pred": np.asarray(pred, dtype=float),
            "gold": np.asarray(gold, dtype=float),
        }
    )
    if not np.isfinite(frame["pred"]).all():
        raise EstimatorError("aggregate_units: non-finite prediction")
    means = frame.groupby("unit", sort=True).mean()
    return means.index.to_numpy(), means["pred"].to_numpy(), means["gold"].to_numpy()


def classes_are_separable(first: float, second: float) -> bool:
    """Tell whether no unit mean can match both of two class values.

    A unit mean matches a class value up to ``CLASS_TOLERANCE`` times the larger of 1 and the
    value's magnitude; two values closer than the two tolerances together (gold that differs
    by floating-point noise only, such as 0.3 and 0.1 + 0.2) cannot be told apart.
    """
    reach = CLASS_TOLERANCE * (max(1.0, abs(first)) + max(1.0, abs(second)))
    return bool(abs(first - second) > reach)


def _class_values(classes: tuple[float, ...]) -> tuple[float, float]:
    """The smaller and the larger of two distinct finite class values.

    Raises:
        ConfigurationError: If there are not two distinct finite numbers, or the two
            cannot be told apart by the class match.
    """
    values = [float(value) for value in classes]
    if len(values) != 2 or not np.isfinite(values).all() or values[0] == values[1]:
        raise ConfigurationError(f"classes must be two distinct finite numbers; got {classes!r}")
    if not classes_are_separable(values[0], values[1]):
        raise ConfigurationError(
            f"classes {values[0]!r} and {values[1]!r} cannot be told apart: they differ by "
            f"less than the {CLASS_TOLERANCE:g} tolerance of the class match"
        )
    return min(values), max(values)


def _matches(gold: np.ndarray, value: float) -> np.ndarray:
    """Whether each unit mean equals a class value up to the rounding of the mean."""
    return np.asarray(np.abs(gold - value) <= CLASS_TOLERANCE * max(1.0, abs(value)))


def fold_metrics(
    pred_u: np.ndarray,
    gold_u: np.ndarray,
    classes: tuple[float, float] | None,
    train_gold: np.ndarray,
) -> dict[str, float | str]:
    """Metrics on the unit means of one test fold; undefined values are NaN.

    AUC and Somers' D are computed before the constant-prediction and constant-gold early
    return, from their own eligibility rule (a binary gold, every unit-mean value one of the
    two class values up to the rounding of the mean, both classes present), so a constant
    predictor on two-class gold still contributes AUC 0.5 and Somers' D 0; a single-class
    fold leaves both undefined but keeps its error metrics.

    Args:
        pred_u: Mean prediction per unit.
        gold_u: Mean gold per unit.
        classes: The two values of a binary gold, in any order; the larger is the positive
            class. ``None`` for a gold that is not binary.
        train_gold: Gold of the training draw, whose range scales ``mae_01``; ``mae_01`` is
            NaN when that range is zero.

    Returns:
        The metric values (see ``METRIC_KEYS``), ``n_units`` and a ``status`` (see
        ``STATUS_KEYS``).

    Raises:
        ConfigurationError: If ``classes`` is not two distinct finite numbers that the
            class match can tell apart.
    """
    pred = np.asarray(pred_u, dtype=float)
    gold = np.asarray(gold_u, dtype=float)
    out: dict[str, float | str] = {key: float("nan") for key in METRIC_KEYS}
    out["n_units"] = float(pred.shape[0])
    low, high = float(np.min(train_gold)), float(np.max(train_gold))
    if not is_constant(np.asarray(train_gold, dtype=float)):
        out["mae_01"] = float(
            np.mean(np.abs((pred - low) / (high - low) - (gold - low) / (high - low)))
        )
    out["mae_native"] = float(np.mean(np.abs(pred - gold)))

    too_few = pred.shape[0] < MIN_UNITS
    constant_gold = is_constant(gold)
    constant_pred = is_constant(pred)

    status = "ok"
    if constant_gold:
        status = "constant_gold"
    if constant_pred:
        status = "constant_pred"

    if classes is not None:
        low, high = _class_values(classes)
        is_low, is_high = _matches(gold, low), _matches(gold, high)
        if bool((is_low | is_high).all()):
            if not (is_low.any() and is_high.any()):
                status = "single_class"
            else:
                auc = auc_from_ranks(pred, is_high)
                out["auc"] = auc
                out["somers_d"] = 2.0 * auc - 1.0
        else:
            status = "nonbinary_unit_gold"

    if too_few:
        status = "too_few_units"

    out["status"] = status
    if too_few or constant_gold or constant_pred:
        logger.debug("fold correlations not computed: status %s", status)
        return out

    rho = float(spearmanr(pred, gold).statistic)
    out["spearman"] = rho
    out["spearman_abs"] = abs(rho)
    out["pearson"] = float(pearsonr(pred, gold).statistic)
    return out


def weighted_over_folds(per_fold: Sequence[Mapping[str, float | str]]) -> dict[str, float]:
    """Unit-count-weighted mean of every metric over the folds where it is finite.

    ``<metric>`` is the weighted mean over the folds on which the metric is finite,
    ``<metric>_n_folds`` the number of those folds and ``<metric>_unit_weight`` their total
    unit count. ``n_folds_<status>`` (one per ``STATUS_KEYS`` entry, 0 when absent) counts
    the folds by diagnostic status; the counts gate no metric.

    Args:
        per_fold: One ``fold_metrics`` result per test fold.

    Returns:
        The aggregated values.
    """
    out: dict[str, float] = {}
    for key in METRIC_KEYS:
        numerator = denominator = 0.0
        n_finite = 0
        for fold in per_fold:
            value = float(fold[key])
            if np.isfinite(value):
                weight = float(fold["n_units"])
                numerator += weight * value
                denominator += weight
                n_finite += 1
        out[key] = numerator / denominator if denominator > 0 else float("nan")
        out[f"{key}_n_folds"] = float(n_finite)
        out[f"{key}_unit_weight"] = denominator
    out["n_test_units_total"] = float(sum(float(fold["n_units"]) for fold in per_fold))
    for status in STATUS_KEYS:
        out[f"n_folds_{status}"] = float(sum(1 for fold in per_fold if fold["status"] == status))
    return out
