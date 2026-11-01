"""The grouped nested penalty search of embedding-augmented calibration, and the outer fits.

The groups of the training rows are split into grouped folds and each fold is deleted in
turn: the variant's pipeline is refitted on the retained rows and the squared error of the
deleted rows is scored at every penalty of ``ALPHAS``. For the two variants that read a
score, the score column is selected again on the retained rows of each deletion, and for
``score_anchored_residual`` the calibration line is refitted there before the residuals are
formed; ``embeddings_only`` has nothing to select. ``choose_penalty`` takes the penalty with
the smallest error pooled over all deleted rows, and the variant is refitted on all training
rows at that penalty, with the score column selected on all of them.

``embeddings_plus_score`` appends the score column standardised with its mean and population
standard deviation over all training rows. These two numbers use no gold value, so they stay
the same in every deletion, and the fit stores them to standardise the score of new rows.

Cost: every variant decomposes the retained design of every deletion afresh, and the variants do
not share these decompositions, although ``embeddings_only`` and ``score_anchored_residual``
decompose the same matrices. In the battery, which fits all three variants on every fold, this
is the largest cost: about 3.7 seconds per fold at 2,000 drawn texts and 768 dimensions.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import partial

import numpy as np
import pandas as pd

from copernicus.baseline.ridge import ALPHAS
from copernicus.corrections.data import best_score_column
from copernicus.corrections.families._ridge_path import (
    GroupDeletion,
    PenaltyChoice,
    RidgeCoefficients,
    choose_penalty,
    ridge_coefficients,
    ridge_path_predictions,
    squared_error,
)
from copernicus.corrections.families.calibration import fit_ols_line

logger = logging.getLogger(__name__)

__all__ = [
    "POLICY_GROUPED",
    "POLICY_RESELECTED",
    "Predictor",
    "VariantFit",
    "fit_variant",
    "penalty_provenance",
]

POLICY_GROUPED = "grouped_deletions"
POLICY_RESELECTED = "grouped_deletions_reselected"

Predictor = Callable[[np.ndarray, np.ndarray], np.ndarray]
Splits = list[tuple[np.ndarray, np.ndarray]]


@dataclass(frozen=True, eq=False)
class _ReselectedDeletion:
    """A deletion with the score column selected on its retained rows alone.

    ``score`` holds that column over all training rows: the held rows are predicted with
    it, never fitted on.
    """

    retained: np.ndarray
    held: np.ndarray
    column: str
    score: np.ndarray


def _reselected(
    scores: pd.DataFrame, gold: np.ndarray, splits: Splits
) -> list[_ReselectedDeletion]:
    """The deletions of the score variants, each with its own score column."""
    deletions = []
    for retained, held in splits:
        column = best_score_column(scores.iloc[retained].reset_index(drop=True), gold[retained])
        deletions.append(
            _ReselectedDeletion(
                retained=retained,
                held=held,
                column=column,
                score=scores[column].to_numpy(dtype=float),
            )
        )
    return deletions


def _embeddings_only_error(
    design: np.ndarray, gold: np.ndarray, deletion: GroupDeletion
) -> np.ndarray:
    predictions = ridge_path_predictions(
        design[deletion.retained], gold[deletion.retained], design[deletion.held], ALPHAS
    )
    return squared_error(predictions, gold[deletion.held])


def _score_scale(score: np.ndarray) -> tuple[float, float]:
    """Mean and population standard deviation of a score column over all training rows.

    No guard for a zero spread: the column comes from ``best_score_column``, which skips a
    column that is constant on the rows it selects on; those are the training rows or the
    retained part of them, so the column varies over the training rows.
    """
    return float(score.mean()), float(score.std())


def _plus_score_error(
    design: np.ndarray, gold: np.ndarray, deletion: _ReselectedDeletion
) -> np.ndarray:
    center, spread = _score_scale(deletion.score)
    features = np.column_stack([design, (deletion.score - center) / spread])
    predictions = ridge_path_predictions(
        features[deletion.retained], gold[deletion.retained], features[deletion.held], ALPHAS
    )
    return squared_error(predictions, gold[deletion.held])


def _residual_error(
    design: np.ndarray, gold: np.ndarray, deletion: _ReselectedDeletion
) -> np.ndarray:
    line = fit_ols_line(deletion.score[deletion.retained], gold[deletion.retained], deletion.column)
    target = gold - line.predict(deletion.score)
    predictions = ridge_path_predictions(
        design[deletion.retained], target[deletion.retained], design[deletion.held], ALPHAS
    )
    return squared_error(
        predictions + line.predict(deletion.score[deletion.held]), gold[deletion.held]
    )


@dataclass(frozen=True, eq=False)
class VariantFit:
    """The outer model of one variant at the chosen penalty.

    Attributes:
        choice: The penalty the deletions chose, with its error trace.
        ridge: The ridge at that penalty on all training rows.
        predictor: Maps the standardised embeddings and the matrix of ``score_columns`` to
            the prediction.
        score_columns: The score column the variant reads (none for ``embeddings_only``).
        coefficients: The scalar coefficients, for the provenance.
    """

    choice: PenaltyChoice
    ridge: RidgeCoefficients
    predictor: Predictor
    score_columns: tuple[str, ...]
    coefficients: Mapping[str, float]


def _outer_score(scores: pd.DataFrame, gold: np.ndarray) -> tuple[str, np.ndarray]:
    """The best score column of all training rows and its values."""
    column = best_score_column(scores, gold)
    return column, scores[column].to_numpy(dtype=float)


def _fit_embeddings_only(
    design: np.ndarray, gold: np.ndarray, scores: pd.DataFrame, splits: Splits, n_groups: int
) -> VariantFit:
    deletions = [GroupDeletion(retained=retained, held=held) for retained, held in splits]
    choice = choose_penalty(
        deletions, partial(_embeddings_only_error, design, gold), ALPHAS, POLICY_GROUPED, n_groups
    )
    ridge = ridge_coefficients(design, gold, choice.alpha)
    return VariantFit(
        choice=choice,
        ridge=ridge,
        predictor=lambda embeddings, score_matrix: ridge.predict(embeddings),
        score_columns=(),
        coefficients={"ridge_intercept": ridge.intercept},
    )


def _fit_plus_score(
    design: np.ndarray, gold: np.ndarray, scores: pd.DataFrame, splits: Splits, n_groups: int
) -> VariantFit:
    choice = choose_penalty(
        _reselected(scores, gold, splits),
        partial(_plus_score_error, design, gold),
        ALPHAS,
        POLICY_RESELECTED,
        n_groups,
    )
    column, score = _outer_score(scores, gold)
    center, spread = _score_scale(score)
    ridge = ridge_coefficients(
        np.column_stack([design, (score - center) / spread]), gold, choice.alpha
    )
    return VariantFit(
        choice=choice,
        ridge=ridge,
        predictor=lambda embeddings, score_matrix: ridge.predict(
            np.column_stack([embeddings, (score_matrix - center) / spread])
        ),
        score_columns=(column,),
        coefficients={
            "ridge_intercept": ridge.intercept,
            "score_coefficient": float(ridge.weights[-1]),
            "score_mean": center,
            "score_sd": spread,
        },
    )


def _fit_anchored_residual(
    design: np.ndarray, gold: np.ndarray, scores: pd.DataFrame, splits: Splits, n_groups: int
) -> VariantFit:
    choice = choose_penalty(
        _reselected(scores, gold, splits),
        partial(_residual_error, design, gold),
        ALPHAS,
        POLICY_RESELECTED,
        n_groups,
    )
    column, score = _outer_score(scores, gold)
    line = fit_ols_line(score, gold, column)
    ridge = ridge_coefficients(design, gold - line.predict(score), choice.alpha)
    return VariantFit(
        choice=choice,
        ridge=ridge,
        predictor=lambda embeddings, score_matrix: (
            line.predict(score_matrix[:, 0]) + ridge.predict(embeddings)
        ),
        score_columns=(column,),
        coefficients={
            "ridge_intercept": ridge.intercept,
            "calibration_intercept": line.intercept,
            "calibration_slope": line.slope,
        },
    )


_FITTERS: dict[str, Callable[[np.ndarray, np.ndarray, pd.DataFrame, Splits, int], VariantFit]] = {
    "embeddings_only": _fit_embeddings_only,
    "embeddings_plus_score": _fit_plus_score,
    "score_anchored_residual": _fit_anchored_residual,
}


def fit_variant(
    variant: str,
    design: np.ndarray,
    gold: np.ndarray,
    scores: pd.DataFrame,
    splits: Splits,
    n_groups: int,
) -> VariantFit:
    """Choose a variant's penalty on the grouped deletions and refit it at that penalty.

    Args:
        variant: ``embeddings_only``, ``embeddings_plus_score`` or
            ``score_anchored_residual``.
        design: The standardised embeddings of the training rows.
        gold: Their gold values.
        scores: Their score columns, rows in the order of ``design``.
        splits: The grouped ``(retained, held)`` position pairs of the deletions.
        n_groups: The number of groups in the training rows.

    Returns:
        The outer model with the chosen penalty.

    Raises:
        EstimatorError: If no score column is usable on all training rows or on the
            retained rows of a deletion, where the variant reads a score.
    """
    fitted = _FITTERS[variant](design, gold, scores, splits, n_groups)
    logger.debug("%s: penalty %.4g by %s", variant, fitted.choice.alpha, fitted.choice.policy)
    return fitted


def penalty_provenance(choice: PenaltyChoice) -> dict[str, object]:
    """The penalty fields of a fit: the choice, its policy and how often it sits on an edge.

    Args:
        choice: The penalty choice.

    Returns:
        ``penalty_alpha``, ``penalty_selection``, ``k_inner_folds``, ``n_inner_groups``,
        ``penalty_at_grid_edge``, ``penalty_at_lower_edge``, ``penalty_at_upper_edge``,
        ``penalty_deletion_share_lower`` and ``penalty_deletion_share_upper``.
    """
    return {
        "penalty_alpha": choice.alpha,
        "penalty_selection": choice.policy,
        "k_inner_folds": choice.n_deletions,
        "n_inner_groups": choice.n_groups,
        "penalty_at_grid_edge": choice.at_grid_edge,
        "penalty_at_lower_edge": choice.at_lower_edge,
        "penalty_at_upper_edge": choice.at_upper_edge,
        "penalty_deletion_share_lower": choice.deletion_edge_share(upper=False),
        "penalty_deletion_share_upper": choice.deletion_edge_share(upper=True),
    }
