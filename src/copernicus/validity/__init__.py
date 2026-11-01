"""Validity of a score against a gold value: association, score errors, ceiling and the panel.

The modules are pure numpy and scipy over arrays and frames; importing this package does
not import scikit-learn.
"""

from copernicus.validity.anatomy import (
    ScoreErrors,
    constant_score_errors,
    covariate_slope,
    replicate_noise,
    score_errors,
)
from copernicus.validity.bootstrap import (
    BootstrapCI,
    VectorMetric,
    bootstrap_many,
    cluster_bootstrap_ci,
)
from copernicus.validity.metrics import (
    Association,
    AssociationStatus,
    association,
    auc_from_ranks,
    finite_pairs,
    is_binary,
    is_constant,
    minmax01,
)
from copernicus.validity.panel import PanelSettings, ValidityPanel, validity_panel
from copernicus.validity.reliability import cronbach_alpha, reliability_ceiling, share_of_ceiling
from copernicus.validity.report import render_report

__all__ = [
    "Association",
    "AssociationStatus",
    "BootstrapCI",
    "PanelSettings",
    "ScoreErrors",
    "ValidityPanel",
    "VectorMetric",
    "association",
    "auc_from_ranks",
    "bootstrap_many",
    "cluster_bootstrap_ci",
    "constant_score_errors",
    "covariate_slope",
    "cronbach_alpha",
    "finite_pairs",
    "is_binary",
    "is_constant",
    "minmax01",
    "reliability_ceiling",
    "render_report",
    "replicate_noise",
    "score_errors",
    "share_of_ceiling",
    "validity_panel",
]
