"""Folds and training draws of the baseline as tables, from the keyed generators or by hand."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from baseline_support import GOLD, World

from copernicus.baseline import BaselineResult, BaselineSplits
from copernicus.baseline.folds import draw_groups, grouped_kfold, rng_for

Folds = list[tuple[np.ndarray, np.ndarray]]
Draws = dict[tuple[int, str], list[np.ndarray]]


def assert_same_tables(first: BaselineResult, second: BaselineResult) -> None:
    """The three tables are equal, the floating-point columns in every bit."""
    for name in ("results", "oof_predictions", "skipped"):
        mine, other = getattr(first, name), getattr(second, name)
        pd.testing.assert_frame_equal(mine, other, check_exact=True)
        for column in mine.columns:
            if mine[column].dtype == np.float64:
                bits = mine[column].to_numpy().view(np.int64)
                assert np.array_equal(bits, other[column].to_numpy().view(np.int64)), column


def assert_same_splits(first: BaselineSplits, second: BaselineSplits) -> None:
    """The folds and the draws of two splits are equal tables."""
    pd.testing.assert_frame_equal(first.folds, second.folds)
    assert first.draws is not None and second.draws is not None
    pd.testing.assert_frame_equal(first.draws, second.draws)


def keyed_folds(
    world: World, seeds: Sequence[int], k: int = 5, gold_column: str = GOLD
) -> dict[int, Folds]:
    """The folds that ``grouped_kfold`` gives each seed under the documented generator key."""
    groups = np.asarray(world.groups)
    return {seed: grouped_kfold(groups, k, rng_for(seed, gold_column, "folds")) for seed in seeds}


def folds_from_labels(labels: np.ndarray, k: int) -> Folds:
    """Training and test positions of each fold, from the fold label of every row."""
    positions = np.arange(labels.shape[0])
    return [(positions[labels != fold], positions[labels == fold]) for fold in range(k)]


def fold_labels(folds: Folds, n_rows: int) -> np.ndarray:
    """The test fold of every row."""
    labels = np.full(n_rows, -1, dtype=np.int64)
    for index, (_, test) in enumerate(folds):
        labels[test] = index
    return labels


def fold_table(world: World, folds: Mapping[int, Folds]) -> pd.DataFrame:
    """The folds as a table of ``seed``, ``text_id`` and ``fold``: seed by seed, texts in order."""
    ids = np.asarray(world.ids)
    frames = [
        pd.DataFrame(
            {
                "seed": np.full(ids.shape[0], seed, dtype=np.int64),
                "text_id": ids,
                "fold": fold_labels(parts, ids.shape[0]),
            }
        )
        for seed, parts in folds.items()
    ]
    return pd.concat(frames, ignore_index=True)


def keyed_draws(
    world: World,
    folds: Mapping[int, Folds],
    sizes: Sequence[int | str],
    gold_column: str = GOLD,
    units: np.ndarray | None = None,
) -> Draws:
    """The draws that ``draw_groups`` gives each seed, size and fold under the documented key.

    The analysis unit is the text unless ``units`` gives one per text.
    """
    groups = np.asarray(world.groups)
    unit_of = np.asarray(world.ids) if units is None else units
    return {
        (seed, str(size)): [
            draw_groups(
                train, groups, unit_of, size, rng_for(seed, gold_column, size, index, "draw")
            )[0]
            for index, (train, _) in enumerate(parts)
        ]
        for seed, parts in folds.items()
        for size in sizes
    }


def draw_table(world: World, draws: Draws) -> pd.DataFrame:
    """The draws as a table of ``seed``, ``size``, ``fold`` and ``text_id``, in the given order."""
    ids = np.asarray(world.ids)
    frames = [
        pd.DataFrame(
            {
                "seed": np.full(drawn.shape[0], seed, dtype=np.int64),
                "size": np.full(drawn.shape[0], label),
                "fold": np.full(drawn.shape[0], index, dtype=np.int64),
                "text_id": ids[drawn],
            }
        )
        for (seed, label), parts in draws.items()
        for index, drawn in enumerate(parts)
    ]
    return pd.concat(frames, ignore_index=True)


def whole_groups_until(
    world: World, train: np.ndarray, quota: int, order: Sequence[str]
) -> np.ndarray:
    """A draw of whole groups of the pool, taken in the given order until ``quota`` texts."""
    groups = np.asarray(world.groups)
    chosen: list[str] = []
    count = 0
    for group in order:
        members = train[groups[train] == group]
        if members.size == 0:
            continue
        chosen.append(group)
        count += members.size
        if count >= quota:
            break
    return np.sort(train[np.isin(groups[train], chosen)])
