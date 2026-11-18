"""The draws of each (seed, size) of a baseline run, and the splits that a run records.

``plan_draws`` decides for one (seed, size) whether it is fitted or skipped, with which reason,
and which draws it consults: those of the keyed generators, or the checked draws that the user
supplied. ``quota_or_pool_reason`` and ``pool_floor_reason`` are the parts of the skip rule that
the folds decide before any draw. ``splits_from_positions`` turns the folds and the draws that a
run consulted back into ``BaselineSplits``. Positions follow the order of the analysis rows.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping

import numpy as np
import pandas as pd

from copernicus.baseline.folds import FULL, draw_groups, rng_for, size_skip_reason
from copernicus.baseline.runner_inputs import AlignedInputs
from copernicus.baseline.runner_rows import size_label
from copernicus.baseline.splits import (
    DRAW_COLUMNS,
    INTEGER_COLUMNS,
    BaselineSplits,
    Draws,
    Folds,
    integer_array,
)

logger = logging.getLogger(__name__)

__all__ = ["plan_draws", "pool_floor_reason", "quota_or_pool_reason", "splits_from_positions"]


def _smallest_pool(aligned: AlignedInputs, folds: Folds) -> int:
    """The number of analysis units of the smallest training pool of the folds."""
    return min(int(np.unique(aligned.unit_id[train]).shape[0]) for train, _ in folds)


def quota_or_pool_reason(
    aligned: AlignedInputs, folds: Folds, size: int | str, n_units: int
) -> str:
    """Why a size is skipped before any draw is made, or an empty string.

    Args:
        aligned: The aligned inputs.
        folds: Training and test positions of each fold.
        size: The training size.
        n_units: Number of analysis units of all analysis rows.

    Returns:
        ``quota_exceeds_half_units(...)`` when the quota exceeds half the units,
        ``pool_below_quota:min_pool_units=<n>`` when some fold's training pool holds fewer
        units than the quota, and ``""`` otherwise and for ``"full"``.
    """
    reason = size_skip_reason(size, n_units)
    if reason or size == FULL:
        return reason
    smallest_pool = _smallest_pool(aligned, folds)
    if smallest_pool < int(size):
        return f"pool_below_quota:min_pool_units={smallest_pool}"
    return ""


def pool_floor_reason(aligned: AlignedInputs, folds: Folds, floor_units: int) -> str:
    """Why no draw from these folds can reach the floor, or an empty string.

    A draw holds at most the units of its fold's training pool, so a pool with fewer units than
    the floor puts every draw from it below the floor.

    Args:
        aligned: The aligned inputs.
        folds: Training and test positions of each fold.
        floor_units: The fewest units a draw may hold.

    Returns:
        ``below_floor(<floor>):min_pool_units=<n>`` when the smallest pool holds fewer units
        than the floor, and ``""`` otherwise.
    """
    smallest_pool = _smallest_pool(aligned, folds)
    if smallest_pool < floor_units:
        return f"below_floor({floor_units}):min_pool_units={smallest_pool}"
    return ""


def plan_draws(
    aligned: AlignedInputs,
    folds: Folds,
    *,
    seed: int,
    size: int | str,
    n_units: int,
    floor_units: int,
    gold_column: str,
    supplied: Mapping[tuple[int, str], Draws] | None,
    pool_skips: Mapping[tuple[int, str], str],
) -> tuple[str, Draws]:
    """The skip reason of one (seed, size), empty when it is fitted, and the draws it consulted.

    A size is skipped without a draw when its quota exceeds half the units or some fold's
    training pool holds fewer units than the quota, and, for supplied splits without a draw of
    that size, when its pools lie below the floor (``pool_skips``). Otherwise the draws are
    consulted, and the size is skipped when some draw holds fewer units than the floor; the
    draws are then returned with that reason too, so that a run given them reaches the same
    skip.

    Args:
        aligned: The aligned inputs.
        folds: Training and test positions of each fold.
        seed: The seed, as the settings give it.
        size: The training size.
        n_units: Number of analysis units of all analysis rows.
        floor_units: The fewest units a draw may hold.
        gold_column: The name of the gold column, part of the generator key.
        supplied: The checked supplied draws of every (seed, size label) that is not skipped
            before a draw, or ``None`` to draw from the keyed generators.
        pool_skips: The reasons of the supplied (seed, size label) skipped for their pools.

    Returns:
        The reason (``""`` when fitted) and the consulted draws (empty when none was).
    """
    reason = quota_or_pool_reason(aligned, folds, size, n_units)
    if reason:
        return reason, []
    key = (int(seed), size_label(size))
    if supplied is None:
        draws = [
            draw_groups(
                train,
                aligned.group_id,
                aligned.unit_id,
                size,
                rng_for(seed, gold_column, size, fold_index, "draw"),
            )
            for fold_index, (train, _) in enumerate(folds)
        ]
    elif key in pool_skips:
        return pool_skips[key], []
    else:
        draws = supplied[key]
    low = min(receipt.actual_units for _, receipt in draws)
    if low < floor_units:
        return f"below_floor({floor_units}):min_draw_units={low}", draws
    return "", draws


def _seed_column(seed: int, n_rows: int) -> np.ndarray:
    """The seed on every row: int64 when it fits in 64 signed bits, otherwise Python integers."""
    if integer_array([int(seed)]).dtype == np.int64:
        return np.full(n_rows, seed, dtype=np.int64)
    column = np.empty(n_rows, dtype=object)
    column[:] = int(seed)
    return column


def splits_from_positions(
    text_ids: np.ndarray, folds: Mapping[int, Folds], draws: Mapping[tuple[int, str], Draws]
) -> BaselineSplits:
    """The folds and the draws that a run consulted, held as row positions, as splits.

    Args:
        text_ids: The text id of every analysis row.
        folds: The folds of every seed, in the order of the settings.
        draws: The consulted draws of every (seed, size label), in the order of the run.

    Returns:
        The splits: the folds seed by seed, each text in the order of the rows; the draws
        seed by seed, size by size and fold by fold, each draw's texts in that order.
    """
    n_rows = text_ids.shape[0]
    fold_frames = []
    for seed, parts in folds.items():
        fold_of = np.empty(n_rows, dtype=np.int64)
        for fold, (_, test) in enumerate(parts):
            fold_of[test] = fold
        fold_frames.append(
            pd.DataFrame({"seed": _seed_column(seed, n_rows), "text_id": text_ids, "fold": fold_of})
        )
    draw_frames = [
        pd.DataFrame(
            {
                "seed": _seed_column(seed, drawn.shape[0]),
                "size": np.full(drawn.shape[0], label),
                "fold": np.full(drawn.shape[0], fold, dtype=np.int64),
                "text_id": text_ids[drawn],
            }
        )
        for (seed, label), parts in draws.items()
        for fold, (drawn, _) in enumerate(parts)
    ]
    empty = pd.DataFrame(
        {
            column: np.empty(0, dtype=np.int64 if column in INTEGER_COLUMNS else str)
            for column in DRAW_COLUMNS
        }
    )
    return BaselineSplits(
        folds=pd.concat(fold_frames, ignore_index=True),
        draws=pd.concat(draw_frames, ignore_index=True) if draw_frames else empty,
    )
