"""The metrics of one fold of the correction battery and their aggregation over folds.

Every metric is computed on the held-out texts of a fold that carry the user's gold value:

* rank validity: the absolute Spearman correlation of the prediction with the gold value, and
  the signed Spearman correlation beside it, which shows a correction that reverses the order
  (``association``);
* MAE: the mean absolute difference between the 0-1 values of the prediction and of the gold
  value, each scaled with the range of its values on the drawn (training) texts by
  ``scale_test``, so no information from the held-out texts enters the scaling;
* native MAE: the mean absolute difference between the prediction and the gold value on the
  gold's own scale (the raw score enters as it stands);
* covariate-linked error: the slope of the 0-1 difference on the covariate
  (``covariate_slope``), per standard deviation or per unit of the covariate.

Because each 0-1 scale is the variable's own training range, any increasing affine correction
of the raw score (a calibration line, for example) has exactly the raw score's 0-1 values, so
its MAE and covariate-linked gains are zero by construction; the native MAE shows what such a
correction does to the level and the spread. The raw baseline is scored the same way.

Over the folds of a size, each metric is the mean of the fold values weighted by the number of
held-out texts with a gold value, taken over the folds where both the raw and the corrected
value are defined (a fold where the prediction or the raw score is constant on the held-out
texts has no rank validity; a covariate that is constant there has no slope). The gains come
from those means: corrected minus raw rank validity, raw minus corrected MAE and native MAE,
and raw minus corrected absolute covariate-linked error. The folds left out of a metric are
named in each fold's provenance (``dropped_from``) with the two association statuses.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from copernicus.corrections.scaling import scale_test
from copernicus.validity.anatomy import covariate_slope
from copernicus.validity.metrics import association, is_constant

logger = logging.getLogger(__name__)

__all__ = [
    "CONSTANT_TRAINING_RANGE",
    "ERROR_METRICS",
    "METRICS",
    "NON_FINITE_PREDICTIONS",
    "FoldMetrics",
    "FoldOutcome",
    "aggregate",
    "dropped_metrics",
    "fold_metrics",
    "prediction_problem",
    "rank_metrics",
]

CONSTANT_TRAINING_RANGE = "constant_training_range"
NON_FINITE_PREDICTIONS = "non_finite_predictions"
METRICS = ("rank_validity", "spearman", "mae", "mae_native", "covariate_error")
ERROR_METRICS = ("mae", "mae_native", "covariate_error")


@dataclass(frozen=True)
class FoldMetrics:
    """The scores of one prediction on the held-out texts of one fold.

    Attributes:
        n_test: Number of held-out texts with a gold value.
        rank_validity: Absolute Spearman correlation with the gold values (NaN when the
            association is undefined; see ``status``).
        spearman: The signed Spearman correlation (NaN where ``rank_validity`` is).
        mae: Mean absolute difference of the train-scaled 0-1 values; NaN for a rank-only
            prediction.
        mae_native: Mean absolute difference from the gold values on their own scale; NaN for
            a rank-only prediction.
        covariate_error: Slope of the train-scaled difference on the covariate; NaN for a
            rank-only prediction or a constant covariate.
        status: The status of the association, ``ok`` or why it is undefined.
        rank_only: Whether only the rank metrics were scored (the correction ensemble).
    """

    n_test: int
    rank_validity: float
    spearman: float
    mae: float
    mae_native: float
    covariate_error: float
    status: str
    rank_only: bool = False


@dataclass(frozen=True)
class FoldOutcome:
    """One entry's scores on one fold, beside the raw baseline of that fold.

    Attributes:
        fold: The fold index.
        n_labelled: Number of drawn texts the entry was fitted on.
        raw_column: The raw score column chosen on the drawn texts.
        raw: The raw baseline's scores.
        corrected: The entry's scores.
        provenance: The fold's draw, both scores and what was fitted, as plain values.
    """

    fold: int
    n_labelled: int
    raw_column: str
    raw: FoldMetrics
    corrected: FoldMetrics
    provenance: Mapping[str, object]


def prediction_problem(train_pred: np.ndarray, test_pred: np.ndarray) -> str:
    """Why predictions cannot be scored on a fold, or ``""`` when they can.

    Args:
        train_pred: The predictions on the drawn texts.
        test_pred: The predictions on the held-out texts.

    Returns:
        ``non_finite_predictions: ...`` when a prediction is not finite,
        ``constant_training_range: ...`` when the predictions on the drawn texts do not vary
        (they then have no 0-1 scale), else ``""``.
    """
    bad = int(
        np.count_nonzero(~np.isfinite(train_pred)) + np.count_nonzero(~np.isfinite(test_pred))
    )
    if bad:
        return (
            f"{NON_FINITE_PREDICTIONS}: {bad} of the {train_pred.size + test_pred.size} "
            "predictions on the drawn and held-out texts are not finite"
        )
    if is_constant(train_pred):
        return (
            f"{CONSTANT_TRAINING_RANGE}: the predictions on the {train_pred.size} drawn texts "
            f"all equal {float(train_pred[0])!r}, so they have no range to scale with"
        )
    return ""


def fold_metrics(
    train_pred: np.ndarray,
    train_gold: np.ndarray,
    test_pred: np.ndarray,
    test_gold: np.ndarray,
    test_covariate: np.ndarray,
    scale: str,
) -> FoldMetrics:
    """Rank validity, Spearman, both MAEs and covariate-linked error on held-out texts.

    Args:
        train_pred: The finite, varying predictions on the drawn texts.
        train_gold: The varying gold values of the drawn texts.
        test_pred: The finite predictions on the held-out texts with a gold value.
        test_gold: Their gold values.
        test_covariate: Their covariate values.
        scale: ``standardized`` or ``raw``, the scale of the covariate slope.

    Returns:
        The fold's scores.
    """
    found = association(test_pred, test_gold)
    pred01, gold01 = scale_test(train_pred, train_gold, test_pred, test_gold)
    difference = pred01 - gold01
    return FoldMetrics(
        n_test=int(test_gold.size),
        rank_validity=found.rank_validity,
        spearman=found.spearman,
        mae=float(np.mean(np.abs(difference))),
        mae_native=float(np.mean(np.abs(test_pred - test_gold))),
        covariate_error=covariate_slope(difference, test_covariate, scale),
        status=found.status,
    )


def rank_metrics(test_pred: np.ndarray, test_gold: np.ndarray) -> FoldMetrics:
    """The rank metrics of a rank-only prediction, with NaN for the error metrics.

    Args:
        test_pred: The predictions on the held-out texts with a gold value.
        test_gold: Their gold values.

    Returns:
        The fold's scores, marked ``rank_only``.
    """
    found = association(test_pred, test_gold)
    nan = float("nan")
    return FoldMetrics(
        n_test=int(test_gold.size),
        rank_validity=found.rank_validity,
        spearman=found.spearman,
        mae=nan,
        mae_native=nan,
        covariate_error=nan,
        status=found.status,
        rank_only=True,
    )


def dropped_metrics(raw: FoldMetrics, corrected: FoldMetrics) -> list[str]:
    """The metrics a fold leaves out because its raw or corrected value is undefined.

    The error metrics of a rank-only prediction are not scored at all, so they are not
    listed.

    Args:
        raw: The raw baseline's scores on the fold.
        corrected: The entry's scores on the fold.

    Returns:
        The names among ``METRICS`` whose raw or corrected value is not finite.
    """
    scored = [metric for metric in METRICS if not (corrected.rank_only and metric in ERROR_METRICS)]
    return [
        metric
        for metric in scored
        if not (np.isfinite(getattr(raw, metric)) and np.isfinite(getattr(corrected, metric)))
    ]


def aggregate(outcomes: Sequence[FoldOutcome]) -> dict[str, float | int]:
    """The fold-weighted means of the raw and corrected metrics, and the gains.

    Each metric is averaged over the folds where its raw and corrected values are both finite,
    weighted by their numbers of held-out texts with a gold value; with no such fold it is NaN.

    Args:
        outcomes: One entry's outcomes on the folds where it was scored, at least one.

    Returns:
        ``n_folds_used`` (the folds of the rank validity), then for every metric of ``METRICS``
        its ``<metric>_raw`` and ``<metric>_corrected`` means, and ``gain_rank_validity``,
        ``gain_mae``, ``gain_mae_native`` and ``gain_covariate_error``.
    """
    means: dict[str, float] = {}
    used: dict[str, int] = {}
    for metric in METRICS:
        values = np.array(
            [
                (
                    outcome.corrected.n_test,
                    getattr(outcome.raw, metric),
                    getattr(outcome.corrected, metric),
                )
                for outcome in outcomes
            ],
            dtype=float,
        )
        usable = values[np.isfinite(values[:, 1]) & np.isfinite(values[:, 2])]
        used[metric] = int(usable.shape[0])
        weights = usable[:, 0]
        for column, side in ((1, "raw"), (2, "corrected")):
            means[f"{metric}_{side}"] = (
                float(np.sum(weights * usable[:, column]) / weights.sum())
                if used[metric]
                else float("nan")
            )
    return {
        "n_folds_used": used["rank_validity"],
        **means,
        "gain_rank_validity": means["rank_validity_corrected"] - means["rank_validity_raw"],
        "gain_mae": means["mae_raw"] - means["mae_corrected"],
        "gain_mae_native": means["mae_native_raw"] - means["mae_native_corrected"],
        "gain_covariate_error": abs(means["covariate_error_raw"])
        - abs(means["covariate_error_corrected"]),
    }
