"""The cross-fitted predictor behind the ``crossfit`` pseudo-value rectifier.

The groups of every row are split into ``k`` inner folds. For each fold the rows of the other
folds that carry a gold value are the permitted rows. A group-level half of the permitted
groups chooses the score column (the best signed Spearman correlation with the gold values on
that half), and the line ``gold ~ 1 + score + covariate`` is fitted on all the permitted rows
and predicts every row of the fold. A row is therefore never predicted by a model that saw
its own gold value or the gold value of any text of its group.

The line of a fold is the solution that ``numpy.linalg.lstsq`` with ``rcond=None`` returns: it
leaves out the directions of the design below its cutoff and returns the coefficients of smallest
norm of what remains. A fold whose permitted rows give the design ``[1, score, covariate]`` a
numerical rank below 3, as ``lstsq`` returns it, is named in the result of ``crossfit_ghat``,
which logs one warning per call that names such folds; within ``repeated_study`` that warning is
counted instead, and the study sums such warnings up per size (``_open_cases``). The rank is below
3 for a design that is rank-deficient in exact arithmetic, whose rows many lines fit equally well,
and also for a design of full rank with a direction below the cutoff.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from copernicus.baseline.folds import rng_for
from copernicus.corrections._open_cases import warn_rank_deficient_folds
from copernicus.corrections.crossfit import inner_splits
from copernicus.corrections.data import CorrectionData, best_score_column
from copernicus.errors import EstimatorError

logger = logging.getLogger(__name__)

__all__ = [
    "FOLD_TOO_THIN",
    "INNER_GROUPS_BELOW_K",
    "NESTED_SELECTION_HALF_DEGENERATE",
    "CrossfitPredictions",
    "crossfit_ghat",
]

INNER_GROUPS_BELOW_K = "inner_groups_below_k"
FOLD_TOO_THIN = "fold_too_thin"
NESTED_SELECTION_HALF_DEGENERATE = "nested_selection_half_degenerate"
DESIGN_PARAMETERS = 3
MIN_PERMITTED_ROWS = DESIGN_PARAMETERS + 1
MIN_SELECTION_GROUPS = 2


@dataclass(frozen=True, eq=False)
class CrossfitPredictions:
    """The out-of-fold prediction of every row, with what each fold chose and fitted.

    Attributes:
        ghat: The prediction of every row, in data order.
        columns: The score column each fold chose, in fold order.
        rank_deficient_folds: The folds, in order, whose design ``[1, score, covariate]`` has a
            numerical rank below 3 on their permitted rows, as ``numpy.linalg.lstsq`` with
            ``rcond=None`` returns it; empty when that rank is 3 for every fold.
    """

    ghat: np.ndarray
    columns: tuple[str, ...]
    rank_deficient_folds: tuple[int, ...]


def _selection_half(
    permitted: np.ndarray, groups: np.ndarray, rng: np.random.Generator, fold: int
) -> np.ndarray:
    """The permitted rows of a random half of the permitted groups.

    Args:
        permitted: Row positions of the fold's permitted rows.
        groups: Group id of every row of the data.
        rng: The generator that also drew the inner folds.
        fold: The fold's index, for the message.

    Returns:
        Row positions of every permitted row of the drawn groups.

    Raises:
        EstimatorError: With the reason ``nested_selection_half_degenerate`` if the
            permitted rows hold fewer than two groups.
    """
    permitted_groups = np.unique(groups[permitted])
    if permitted_groups.size < MIN_SELECTION_GROUPS:
        raise EstimatorError(
            f"{NESTED_SELECTION_HALF_DEGENERATE}: fold {fold} permits "
            f"{permitted_groups.size} group(s); a group-level half needs at least "
            f"{MIN_SELECTION_GROUPS}"
        )
    half = max(permitted_groups.size // 2, 1)
    chosen = rng.choice(permitted_groups, size=half, replace=False)
    return permitted[np.isin(groups[permitted], chosen)]


def crossfit_ghat(data: CorrectionData, *, k: int, seed: int) -> CrossfitPredictions:
    """Out-of-fold predictions of the gold value for every row.

    The inner folds and the selection halves come from one generator,
    ``rng_for(seed, "pseudo", "crossfit", k)``, so the result is a function of the data,
    ``k`` and ``seed``. A fold whose design ``[1, score, covariate]`` has a numerical rank below
    3 on its permitted rows, as ``numpy.linalg.lstsq`` with ``rcond=None`` returns it, keeps the
    coefficients that ``lstsq`` returns (see the module docstring); such folds are named in the
    result, and one warning that names them is logged.

    Args:
        data: The correction data; the labelled rows carry the gold values.
        k: Number of inner folds, at least 2.
        seed: Seed of the generator.

    Returns:
        The predictions, one per row in data order, the score column each fold chose and the
        folds whose design has a numerical rank below 3.

    Raises:
        EstimatorError: With the reason ``inner_groups_below_k`` (fewer groups than ``k``),
            ``fold_too_thin`` (a fold permits fewer than four labelled rows), or
            ``nested_selection_half_degenerate`` (a fold permits fewer than two groups, or
            no score column is usable on the selection half).
    """
    groups = data.groups
    n_groups = int(np.unique(groups).size)
    if n_groups < k:
        raise EstimatorError(
            f"{INNER_GROUPS_BELOW_K}: {k} inner folds need at least {k} groups; "
            f"the data hold {n_groups}"
        )
    rng = rng_for(seed, "pseudo", "crossfit", k)
    folds = inner_splits(groups, k, rng)
    labelled = data.labelled_mask
    gold = data.gold
    scores = data.scores
    covariate = data.covariate
    ghat = np.full(data.n, np.nan)
    chosen_columns: list[str] = []
    deficient: list[tuple[int, int]] = []
    for fold, (train, test) in enumerate(folds):
        permitted = train[labelled[train]]
        if permitted.size < MIN_PERMITTED_ROWS:
            raise EstimatorError(
                f"{FOLD_TOO_THIN}: fold {fold} permits {permitted.size} labelled rows; "
                f"at least {MIN_PERMITTED_ROWS} are needed for the {DESIGN_PARAMETERS} "
                "coefficients"
            )
        selection = _selection_half(permitted, groups, rng, fold)
        column = _select_column(scores, gold, selection, fold)
        score = scores[column].to_numpy(dtype=float)
        design = np.column_stack([np.ones(data.n), score, covariate])
        beta, _, rank, _ = np.linalg.lstsq(design[permitted], gold[permitted], rcond=None)
        if rank < DESIGN_PARAMETERS:
            deficient.append((fold, int(rank)))
        ghat[test] = design[test] @ beta
        chosen_columns.append(column)
    if deficient:
        warn_rank_deficient_folds(logger, deficient, k, DESIGN_PARAMETERS)
    logger.debug(
        "cross-fitted predictions over %d folds, columns %s", k, sorted(set(chosen_columns))
    )
    return CrossfitPredictions(
        ghat=ghat,
        columns=tuple(chosen_columns),
        rank_deficient_folds=tuple(fold for fold, _ in deficient),
    )


def _select_column(scores: pd.DataFrame, gold: np.ndarray, selection: np.ndarray, fold: int) -> str:
    """The best score column on the selection half, or the degenerate-half refusal."""
    try:
        return best_score_column(scores.iloc[selection].reset_index(drop=True), gold[selection])
    except EstimatorError as error:
        raise EstimatorError(
            f"{NESTED_SELECTION_HALF_DEGENERATE}: no score column is usable on the "
            f"{selection.size} selection rows of fold {fold} ({error})"
        ) from error
