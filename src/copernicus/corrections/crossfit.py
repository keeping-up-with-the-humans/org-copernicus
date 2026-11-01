"""Grouped folds, the grouped holdout split, inner splits, and cross-fitted predictions.

A correction is evaluated and applied with cross-fitting: the texts are split into grouped
folds, each fold is predicted by a correction fitted on the labelled texts of the other
folds, so the corrected value of every text comes from a fit that never saw that text's gold
value. A group (the persons or conversations that must not straddle a split) always lies
whole on one side of a split.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from copernicus.baseline.folds import grouped_kfold, rng_for
from copernicus.corrections.data import CorrectionData
from copernicus.errors import ConfigurationError, EstimatorError, NotApplicableError, SettingsError

if TYPE_CHECKING:
    # Type-only: the family modules use the inner splits of this module, so a runtime
    # import of the families here would make the two import each other.
    from copernicus.corrections.families._registry import ScoreCorrection

logger = logging.getLogger(__name__)

__all__ = [
    "HOLDOUT_BAND",
    "HOLDOUT_SHARE",
    "INNER_FOLDS",
    "MIN_TRAIN",
    "cross_fit_predictions",
    "grouped_folds",
    "holdout_split",
    "inner_k",
    "inner_splits",
]

MIN_TRAIN = 10
"""The fewest labelled training texts each fold of ``cross_fit_predictions`` needs."""
INNER_FOLDS = 5
"""The inner folds for choosing a penalty in a training set; ``inner_k`` makes one fold per
group when the set has fewer groups."""
HOLDOUT_SHARE = 0.2
"""The default target share of the rows on the test side of ``holdout_split`` (an 80/20 split)."""
HOLDOUT_BAND = (0.75, 0.85)
"""The default lowest and highest admissible share of the rows on the training side of
``holdout_split``; a split that cannot reach the band is undefined."""
MIN_GROUPS = 2


def grouped_folds(
    groups: np.ndarray, k: int, rng: np.random.Generator
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Split rows into ``k`` grouped folds as ``(train_pos, test_pos)`` position arrays.

    Args:
        groups: Group label of each row.
        k: Number of folds.
        rng: Generator for the permutation of the groups.

    Returns:
        One ``(train_pos, test_pos)`` pair per fold; the test positions of all folds
        partition the rows and no group is split between train and test.

    Raises:
        ConfigurationError: If ``k`` is below 2 or there are fewer groups than folds.
    """
    return grouped_kfold(np.asarray(groups), k, rng)


def holdout_split(
    groups: np.ndarray,
    rng: np.random.Generator,
    share: float = HOLDOUT_SHARE,
    band: tuple[float, float] = HOLDOUT_BAND,
) -> tuple[np.ndarray, np.ndarray]:
    """One grouped train/test split whose training side is about ``1 - share`` of the rows.

    The groups are permuted once and added to the training side in that order. Of the
    ``G - 1`` prefixes that leave at least one group for the test side, the one whose row
    share is nearest ``1 - share`` is kept; a tie goes to the smaller training side.

    Args:
        groups: Group label of each row.
        rng: Generator for the permutation of the groups.
        share: Target share of the rows on the test side, in ``(0, 1)``.
        band: Lowest and highest admissible share of the rows on the training side.

    Returns:
        ``(train_pos, test_pos)`` position arrays.

    Raises:
        SettingsError: If ``share`` is outside ``(0, 1)`` or ``band`` is not an ordered pair
            inside ``[0, 1]``.
        ConfigurationError: If there are fewer than two groups.
        EstimatorError: If the realised training share lies outside ``band`` (the message
            names the share, the band, and the group and row counts).
    """
    low, high = band
    if not 0.0 < share < 1.0:
        raise SettingsError(f"share must lie in (0, 1); got {share!r}")
    if not 0.0 <= low <= high <= 1.0:
        raise SettingsError(f"band must be an ordered pair inside [0, 1]; got {band!r}")
    labels = np.asarray(groups)
    unique, counts = np.unique(labels, return_counts=True)
    n_groups, n_rows = int(unique.size), int(labels.size)
    if n_groups < MIN_GROUPS:
        raise ConfigurationError(
            f"a grouped holdout needs at least two groups; got {n_groups} in {n_rows} rows"
        )
    order = rng.permutation(n_groups)
    train_shares = np.cumsum(counts[order])[:-1] / n_rows
    best = int(np.argmin(np.abs(train_shares - (1.0 - share))))
    realised = float(train_shares[best])
    if not low <= realised <= high:
        raise EstimatorError(
            f"the nearest whole-group training share {realised:.4f} lies outside the band "
            f"[{low:.2f}, {high:.2f}] over {n_groups} groups and {n_rows} rows"
        )
    train_mask = np.isin(labels, unique[order[: best + 1]])
    positions = np.arange(n_rows)
    return positions[train_mask], positions[~train_mask]


def inner_k(n_groups: int) -> int:
    """Number of inner folds for a training set with ``n_groups`` groups.

    Args:
        n_groups: Groups in the training set.

    Returns:
        ``min(5, n_groups)``.

    Raises:
        ConfigurationError: If there is no group.
    """
    if n_groups < 1:
        raise ConfigurationError("inner folds need at least one group")
    return min(INNER_FOLDS, n_groups)


def inner_splits(
    groups: np.ndarray, k: int, rng: np.random.Generator
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Grouped folds inside a training set, for choosing a penalty.

    Args:
        groups: Group label of each training row.
        k: Number of inner folds, usually ``inner_k`` of the group count.
        rng: Generator for the permutation of the groups.

    Returns:
        One ``(train_pos, test_pos)`` pair per fold.

    Raises:
        ConfigurationError: If there are fewer than two groups, ``k`` is below 2, or there
            are fewer groups than folds.
    """
    n_groups = int(np.unique(np.asarray(groups)).size)
    if n_groups < MIN_GROUPS:
        raise ConfigurationError(
            f"inner splits need at least two groups; got {n_groups} in {len(groups)} rows"
        )
    return grouped_folds(groups, k, rng)


def _training_data(data: CorrectionData, rows: np.ndarray) -> CorrectionData:
    """The data a fold's correction sees: the gold values of its training rows only."""
    keep = np.zeros(data.n, dtype=bool)
    keep[rows] = True
    frame = data.frame.copy()
    frame[data.gold_column] = np.where(keep, data.gold, np.nan)
    return replace(data, frame=frame)


def cross_fit_predictions(
    correction: ScoreCorrection, data: CorrectionData, *, k: int = 5, seed: int = 42
) -> pd.Series:
    """The corrected value of every text from a fit that never saw that text's gold.

    The groups of all texts are split into ``k`` grouped folds. For each fold the
    correction is checked, fitted and applied on data that carry the gold values of the
    labelled texts outside the fold only, and predicts every text of the fold, labelled or
    not. ``applies`` is checked on the whole data first and then on each fold's training
    rows, so a floor (such as the misclassification matrix's per-class minimum) holds for
    every fit.

    Args:
        correction: The score-level correction.
        data: The correction data.
        k: Number of folds.
        seed: Seed of the folds and of the fits.

    Returns:
        Floats on the gold scale indexed by the data's text ids, in data order.

    Raises:
        NotApplicableError: If the correction does not apply to the data, or to the training
            rows of a fold (the message names the fold).
        ConfigurationError: If there are fewer groups than folds or ``k`` is below 2.
        EstimatorError: If a fold has fewer than ``MIN_TRAIN`` labelled training rows, or
            a fit fails.
    """
    name = f"{correction.family}/{correction.variant}"
    applies, reason = correction.applies(data)
    if not applies:
        raise NotApplicableError(f"{name} does not apply: {reason}")
    ids = data.ids
    labelled = data.labelled_mask
    predictions = np.full(data.n, np.nan)
    folds = grouped_folds(data.groups, k, rng_for(seed, "crossfit", k))
    for fold_index, (train_pos, test_pos) in enumerate(folds):
        train_labelled = train_pos[labelled[train_pos]]
        if train_labelled.size < MIN_TRAIN:
            raise EstimatorError(
                f"fold {fold_index} has {train_labelled.size} labelled training rows; "
                f"at least {MIN_TRAIN} are needed"
            )
        training = _training_data(data, train_labelled)
        applies, reason = correction.applies(training)
        if not applies:
            raise NotApplicableError(
                f"{name} does not apply to the training rows of fold {fold_index}: {reason}"
            )
        fitted = correction.fit(training, ids[train_labelled], rng=rng_for(seed, "fit", fold_index))
        predicted = fitted.predict(training, ids[test_pos])
        if predicted.index.tolist() != ids[test_pos].tolist():
            raise EstimatorError(
                f"{name}: predictions of fold {fold_index} are not indexed by the ids asked for"
            )
        predictions[test_pos] = predicted.to_numpy(dtype=float)
    logger.info("cross-fitted %s over %d folds (%d texts)", name, k, data.n)
    return pd.Series(predictions, index=pd.Index(ids))
