"""The correction families behind one interface.

``_registry`` comes first so that every family module can register itself at import.
"""

from copernicus.corrections.families._registry import (
    REGISTRY,
    FamilySpec,
    Fitted,
    ScoreCorrection,
    get_correction,
    list_corrections,
    register,
)
from copernicus.corrections.families.calibration import (
    LinePredictor,
    RegressionCalibration,
)
from copernicus.corrections.families.classical_score import ClassicalScore
from copernicus.corrections.families.embedding_augmented import EmbeddingAugmented
from copernicus.corrections.families.ensemble import CorrectionEnsemble, EnsembleResult
from copernicus.corrections.families.imputation import (
    MultipleImputation,
    PooledEstimate,
    rubin_pool,
)
from copernicus.corrections.families.latent import LatentPooling
from copernicus.corrections.families.marginal import MarginalMatching
from copernicus.corrections.families.misclassification import MisclassificationMatrix
from copernicus.corrections.families.pseudo import PseudoValues, pseudo_values
from copernicus.corrections.families.stacking import SupervisedStacking
from copernicus.corrections.families.transport import TransportResult, blind_transport

__all__ = [
    "REGISTRY",
    "ClassicalScore",
    "CorrectionEnsemble",
    "EmbeddingAugmented",
    "EnsembleResult",
    "FamilySpec",
    "Fitted",
    "LatentPooling",
    "LinePredictor",
    "MarginalMatching",
    "MisclassificationMatrix",
    "MultipleImputation",
    "PooledEstimate",
    "PseudoValues",
    "RegressionCalibration",
    "ScoreCorrection",
    "SupervisedStacking",
    "TransportResult",
    "blind_transport",
    "get_correction",
    "list_corrections",
    "pseudo_values",
    "register",
    "rubin_pool",
]
