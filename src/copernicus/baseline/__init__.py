"""NLP-ML baseline: ridge on embeddings with exact grouped leave-one-group-out tuning.

The modules here hold the ridge design, the grouped folds with whole-group draws, the
unit-level fold metrics, and the runner that fits the baseline over seeds, training sizes and
encoder sets; importing this package does not import scikit-learn.
"""

from copernicus.baseline.folds import (
    FLOOR_UNITS,
    K_FOLDS,
    SIZES,
    DrawReceipt,
    draw_groups,
    grouped_kfold,
    rng_for,
    size_skip_reason,
)
from copernicus.baseline.metrics import (
    METRIC_KEYS,
    STATUS_KEYS,
    aggregate_units,
    fold_metrics,
    weighted_over_folds,
)
from copernicus.baseline.ridge import (
    ALPHAS,
    RidgeDesign,
    RidgeFit,
    Scaler,
    apply_scaler,
    fit_scaler,
    logo_ridge_fit,
    predict,
)
from copernicus.baseline.runner import (
    BaselineResult,
    BaselineSettings,
    curves,
    run_ridge_baseline,
)

__all__ = [
    "ALPHAS",
    "FLOOR_UNITS",
    "K_FOLDS",
    "METRIC_KEYS",
    "SIZES",
    "STATUS_KEYS",
    "BaselineResult",
    "BaselineSettings",
    "DrawReceipt",
    "RidgeDesign",
    "RidgeFit",
    "Scaler",
    "aggregate_units",
    "apply_scaler",
    "curves",
    "draw_groups",
    "fit_scaler",
    "fold_metrics",
    "grouped_kfold",
    "logo_ridge_fit",
    "predict",
    "rng_for",
    "run_ridge_baseline",
    "size_skip_reason",
    "weighted_over_folds",
]
