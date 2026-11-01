"""Supervised stacking: a penalised linear learner over the scores of every configuration.

The features of a text are all its score columns, the covariate and the spread of the
scores: the row-wise population standard deviation of the score columns standardised on the
training rows (the standardisation the design applies to them), so that the units of a column
do not change the predictions; the research took the spread of the raw score columns. Each
feature is standardised on the training rows with the population standard deviation; a
constant feature keeps standard deviation 1, so it standardises to zeros and gets no weight.
A penalised linear regression
maps the features to the gold value. The penalty type is the variant, and its strength is
chosen on the training rows alone by validation that never splits a group:

* ``ridge``: the penalty in ``ALPHAS`` (``logspace(-2, 8, 41)``) with the smallest exact
  grouped leave-one-group-out error, from the baseline's ``RidgeDesign``.
* ``lasso``: ``LassoCV`` over a path of 12 penalties (``max_iter = 1500``, ``tol = 1e-3``)
  validated on ``min(5, groups)`` grouped inner splits.
* ``enet``: ``ElasticNetCV`` with ``l1_ratio`` 0.5 or 0.9 on splits of the same kind.

The inner splits of ``lasso`` and ``enet`` come from a seed drawn from the fit's generator
and recorded as ``inner_seed``; ``ridge`` needs no split but draws and records the seed too.
The fitted stacker keeps the penalty, the intercept and the coefficients on the standardised
features, and predicts ``intercept + standardised features @ coefficients`` for every
variant (for ``ridge`` this is the baseline's ``predict``); it keeps no scikit-learn object.

Unlike the calibration of one column, stacking combines the information of several
columns, so it can rank the texts better than the best of them. It raises the ceiling of
the calibration shortfall (by the added information), corrects level displacement, and
addresses covariate-linked error (the covariate is one of its inputs). It needs at least two
score columns and two groups among the labelled rows. scikit-learn is imported inside the
fit; from scikit-learn 1.7 on the number of penalties of the path is passed as ``alphas``,
before as ``n_alphas`` (the same path of penalties either way).
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import ClassVar, Literal

import numpy as np
import pandas as pd

from copernicus.baseline.folds import rng_for
from copernicus.baseline.ridge import ALPHAS, RidgeDesign
from copernicus.corrections.crossfit import inner_k, inner_splits
from copernicus.corrections.data import CorrectionData
from copernicus.corrections.families._factor import standardize_apply, standardize_fit
from copernicus.corrections.families._fitted import (
    grouped_training,
    labelled_group_shortage,
    require_columns,
    row_positions,
    score_column_shortage,
)
from copernicus.corrections.families._registry import FamilySpec, register
from copernicus.errors import ConfigurationError, EstimatorError, NotApplicableError, SettingsError
from copernicus.validity.metrics import is_constant

logger = logging.getLogger(__name__)

__all__ = ["StackingFitted", "SupervisedStacking", "stacking_features"]

FAMILY = "supervised_stacking"
METHOD = "supervised stacking"
VARIANTS = ("ridge", "lasso", "enet")
ERRORS_ADDRESSED = ("calibration_shortfall", "level_displacement", "covariate_linked_error")
SPLIT_KEY = "stacking"
N_ALPHAS = 12
MAX_ITER = 1500
TOLERANCE = 1e-3
ENET_L1_RATIOS = (0.5, 0.9)
LASSO_L1_RATIO = 1.0
RIDGE_L1_RATIO = 0.0
ALPHA_SELECTION_LOGO = "grouped_logo"
ALPHA_SELECTION_KFOLD = "grouped_kfold"
SPREAD_FEATURE = "score_spread"
INTEGER_ALPHAS_FROM = (1, 7)


def stacking_features(
    data: CorrectionData,
    columns: tuple[str, ...],
    positions: np.ndarray,
    centre: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    """The score columns, the covariate and the spread of the standardised scores.

    Args:
        data: The correction data.
        columns: The score columns, in the order of the features.
        positions: Row positions into the data's frame.
        centre: The training mean of each score column.
        scale: The training population standard deviation of each score column (1 for a
            constant one), as ``standardize_fit`` returns them.

    Returns:
        One row per position with ``len(columns) + 2`` features: the raw scores, the covariate
        and the row-wise population SD of ``(scores - centre) / scale``.
    """
    scores = data.frame[list(columns)].to_numpy(dtype=float)[positions]
    spread = standardize_apply(scores, centre, scale).std(axis=1)
    return np.column_stack([scores, data.covariate[positions], spread])


def _alpha_count_keyword() -> str:
    """The keyword under which the installed scikit-learn takes the number of penalties."""
    import sklearn

    match = re.match(r"(\d+)\.(\d+)", sklearn.__version__)
    if match is None:
        raise ConfigurationError(
            f"cannot read the scikit-learn version {sklearn.__version__!r}; the keyword for the "
            "number of penalties of the lasso and elastic net path depends on it"
        )
    version = (int(match.group(1)), int(match.group(2)))
    return "alphas" if version >= INTEGER_ALPHAS_FROM else "n_alphas"


@dataclass(frozen=True, eq=False)
class _Learner:
    """One fitted learner on the standardised design, with how its penalty was chosen."""

    alpha: float
    l1_ratio: float
    alpha_selection: str
    k_inner_folds: int
    intercept: float
    coefficients: np.ndarray


def _fit_ridge(design: np.ndarray, gold: np.ndarray, groups: np.ndarray) -> _Learner:
    """Ridge at the penalty with the smallest exact grouped leave-one-group-out error."""
    fit = RidgeDesign(design, gold, groups).fit(ALPHAS)
    return _Learner(
        alpha=float(fit.alpha),
        l1_ratio=RIDGE_L1_RATIO,
        alpha_selection=ALPHA_SELECTION_LOGO,
        k_inner_folds=int(fit.n_groups),
        intercept=float(fit.intercept),
        coefficients=np.asarray(fit.coef, dtype=float),
    )


def _fit_coordinate_descent(
    variant: str, design: np.ndarray, gold: np.ndarray, groups: np.ndarray, seed: int
) -> _Learner:
    """Lasso or elastic net with the penalty validated on grouped inner splits."""
    from sklearn.linear_model import ElasticNetCV, LassoCV

    n_groups = int(np.unique(groups).size)
    splits = inner_splits(
        groups, inner_k(n_groups), rng_for(seed, SPLIT_KEY, variant, "inner_splits")
    )
    settings: dict[str, object] = {
        "cv": splits,
        "random_state": seed,
        "max_iter": MAX_ITER,
        "tol": TOLERANCE,
        _alpha_count_keyword(): N_ALPHAS,
    }
    if variant == "lasso":
        model = LassoCV(**settings)
    else:
        model = ElasticNetCV(l1_ratio=list(ENET_L1_RATIOS), **settings)
    model.fit(design, gold)
    return _Learner(
        alpha=float(model.alpha_),
        l1_ratio=LASSO_L1_RATIO if variant == "lasso" else float(model.l1_ratio_),
        alpha_selection=ALPHA_SELECTION_KFOLD,
        k_inner_folds=len(splits),
        intercept=float(model.intercept_),
        coefficients=np.asarray(model.coef_, dtype=float),
    )


@dataclass(frozen=True, eq=False)
class StackingFitted:
    """A fitted stacker.

    Attributes:
        columns: The score columns the features read, in feature order.
        mean: Training mean of each feature.
        sd: Training population standard deviation of each feature (1 for a constant one).
        alpha: The chosen penalty.
        intercept: The unpenalised intercept.
        coefficients: The coefficient of each standardised feature.
        provenance: What was fitted; stored as a read-only mapping.
    """

    columns: tuple[str, ...]
    mean: np.ndarray
    sd: np.ndarray
    alpha: float
    intercept: float
    coefficients: np.ndarray
    provenance: Mapping[str, object]

    def __post_init__(self) -> None:
        """Freeze the provenance."""
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))

    def predict(self, data: CorrectionData, ids: np.ndarray) -> pd.Series:
        """The stacked prediction of the named rows.

        Args:
            data: The correction data holding the rows.
            ids: Text ids of the rows.

        Returns:
            Floats on the gold scale indexed by the ids (empty for no id).

        Raises:
            ConfigurationError: If the data lack a score column of the fit (the message names
                it) or an id is not in the data.
        """
        require_columns(data, self.columns)
        positions = row_positions(data, ids)
        n_scores = len(self.columns)
        features = stacking_features(
            data, self.columns, positions, self.mean[:n_scores], self.sd[:n_scores]
        )
        design = standardize_apply(features, self.mean, self.sd)
        return pd.Series(
            self.intercept + design @ self.coefficients,
            index=pd.Index(np.asarray(ids).astype(str)),
            dtype=float,
        )


@dataclass(frozen=True)
class SupervisedStacking:
    """A penalised linear learner over all score columns, the covariate and their spread.

    Attributes:
        variant: ``ridge``, ``lasso`` or ``enet``.
    """

    variant: str
    family: ClassVar[str] = FAMILY
    target: ClassVar[Literal["A", "B"]] = "A"

    def __post_init__(self) -> None:
        """Refuse an unknown variant.

        Raises:
            SettingsError: If the variant is not ``ridge``, ``lasso`` or ``enet``.
        """
        if self.variant not in VARIANTS:
            raise SettingsError(
                f"unknown variant {self.variant!r} of family {FAMILY!r}; variants: "
                f"{', '.join(VARIANTS)}"
            )

    def applies(self, data: CorrectionData) -> tuple[bool, str]:
        """Whether the data hold enough score columns and labelled groups; ``fit`` checks both.

        Args:
            data: The correction data.

        Returns:
            ``(True, "")``, or ``(False, reason)`` with fewer than two score columns or fewer
            than two groups among the labelled rows.
        """
        shortage = score_column_shortage(data, METHOD) or labelled_group_shortage(data, METHOD)
        return (False, shortage) if shortage else (True, "")

    def fit(
        self, data: CorrectionData, train_ids: np.ndarray, *, rng: np.random.Generator
    ) -> StackingFitted:
        """Standardise the features on the training rows and fit the penalised learner.

        Args:
            data: The correction data.
            train_ids: Ids of the labelled rows to learn from.
            rng: Generator from which the seed of the inner splits is drawn.

        Returns:
            The fitted stacker; its provenance holds ``penalty_alpha``, ``l1_ratio``,
            ``alpha_selection`` (``grouped_logo`` or ``grouped_kfold``), ``k_inner_folds``,
            ``n_inner_groups``, ``n_features``, ``inner_seed``, ``feature_names``,
            ``coefficients`` (on the standardised features), ``intercept`` and
            ``score_column = "all"``.

        Raises:
            NotApplicableError: If the data have fewer than two score columns (with the
                reason ``applies`` gives) or the training rows hold fewer than two groups.
            ConfigurationError: If an id is unknown, repeated or has no gold value.
            EstimatorError: If the gold values of the training rows are constant, or every
                feature is constant on them.
        """
        shortage = score_column_shortage(data, METHOD)
        if shortage:
            raise NotApplicableError(f"{FAMILY}/{self.variant} does not apply: {shortage}")
        columns = data.score_columns
        rows = grouped_training(data, train_ids, rng, METHOD, label=f"{FAMILY}/{self.variant}")
        if is_constant(rows.gold):
            raise EstimatorError(
                "the training gold values are constant: there is nothing for the stacker to learn"
            )
        scores = data.frame[list(columns)].to_numpy(dtype=float)[rows.positions]
        features = stacking_features(data, columns, rows.positions, *standardize_fit(scores))
        if all(is_constant(features[:, index]) for index in range(features.shape[1])):
            raise EstimatorError(
                "every feature is constant on the training rows (the score columns, the "
                "covariate and their spread): the stacker has nothing to combine"
            )
        mean, sd = standardize_fit(features)
        design = standardize_apply(features, mean, sd)
        if self.variant == "ridge":
            learner = _fit_ridge(design, rows.gold, rows.groups)
        else:
            learner = _fit_coordinate_descent(
                self.variant, design, rows.gold, rows.groups, rows.inner_seed
            )
        logger.debug(
            "stacking %s: penalty %.4g over %d features", self.variant, learner.alpha, len(mean)
        )
        return StackingFitted(
            columns=columns,
            mean=mean,
            sd=sd,
            alpha=learner.alpha,
            intercept=learner.intercept,
            coefficients=learner.coefficients,
            provenance={
                "family": FAMILY,
                "variant": self.variant,
                "score_column": "all",
                "penalty_alpha": learner.alpha,
                "l1_ratio": learner.l1_ratio,
                "alpha_selection": learner.alpha_selection,
                "k_inner_folds": learner.k_inner_folds,
                "n_inner_groups": rows.n_groups,
                "n_features": int(features.shape[1]),
                "inner_seed": rows.inner_seed,
                "feature_names": [*columns, data.covariate_column, SPREAD_FEATURE],
                "coefficients": [float(value) for value in learner.coefficients],
                "intercept": learner.intercept,
            },
        )


SPEC = register(
    FamilySpec(
        family=FAMILY,
        variants=VARIANTS,
        target="A",
        level="score",
        errors_addressed=ERRORS_ADDRESSED,
        estimate_errors="none",
        factory=SupervisedStacking,
    )
)
