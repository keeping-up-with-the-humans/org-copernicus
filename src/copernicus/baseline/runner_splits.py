"""Supplied splits of the baseline, checked against the analysis rows and the settings.

``check_splits`` turns the tables of a ``BaselineSplits`` into the training and test positions of
every fold, and into the drawn positions of every fold at every (seed, size) whose draws the
baseline consults. It checks every guarantee that the baseline gives for its own folds and draws
first, and refuses the first one that does not hold. A count size for which the splits hold no
draw is skipped without one when its pools already lie below the floor. Positions follow the
order of the analysis rows, as those of ``grouped_kfold`` and ``draw_groups`` do.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from copernicus.baseline.draw_plan import pool_floor_reason, quota_or_pool_reason
from copernicus.baseline.folds import FULL, DrawReceipt
from copernicus.baseline.runner_inputs import AlignedInputs
from copernicus.baseline.runner_rows import size_label
from copernicus.baseline.splits import BaselineSplits, Draws, Folds, listed
from copernicus.errors import ConfigurationError

logger = logging.getLogger(__name__)

__all__ = ["SuppliedSplits", "check_splits"]

NO_DRAW = np.empty(0, dtype=object)


@dataclass(frozen=True)
class SuppliedSplits:
    """Supplied splits as positions of the analysis rows.

    Attributes:
        folds: The training and test positions of every fold, by seed.
        draws: The drawn positions of every fold with their receipt, by seed and size label,
            for every (seed, size) whose draws the baseline consults; ``None`` when the splits
            hold no draws and the baseline draws them itself.
        pool_skips: The reason ``below_floor(<floor>):min_pool_units=<n>`` of every count
            size, by seed and size label, for which the splits hold no draw and whose smallest
            pool lies below the floor.
    """

    folds: Mapping[int, Folds]
    draws: Mapping[tuple[int, str], Draws] | None
    pool_skips: Mapping[tuple[int, str], str]


def _first_point(rows: pd.DataFrame) -> str:
    """The first (seed, size, fold) of offending draw rows, with up to ten of its texts."""
    first = rows.iloc[0]
    point = rows[
        (rows["seed"] == first["seed"])
        & (rows["size"] == first["size"])
        & (rows["fold"] == first["fold"])
    ]
    return (
        f"the first is seed {first['seed']} size {first['size']} fold {first['fold']}, with the "
        f"texts {listed(point['text_id'].tolist())}"
    )


def _check_seeds(folds: pd.DataFrame, seeds: Sequence[int]) -> None:
    """Refuse folds whose seeds are not exactly the seeds of the settings."""
    given = sorted(set(folds["seed"].tolist()))
    asked = [int(seed) for seed in seeds]
    if set(given) == set(asked):
        return
    parts = []
    missing = [seed for seed in asked if seed not in given]
    if missing:
        parts.append(f"missing: {listed(missing)}")
    extra = [seed for seed in given if seed not in asked]
    if extra:
        parts.append(f"not asked for: {listed(extra)}")
    raise ConfigurationError(
        f"the seeds of the splits' folds ({listed(given) or 'none'}) are not the seeds of the "
        f"settings ({listed(asked)}); {'; '.join(parts)}"
    )


def _check_draw_labels(
    draws: pd.DataFrame, seeds: Sequence[int], sizes: Sequence[int | str], k_folds: int
) -> None:
    """Refuse draws for a seed, a size or a fold that the settings do not ask for.

    The message names the values, the first offending (seed, size, fold) and up to ten of its
    texts.
    """
    asked = [int(seed) for seed in seeds]
    extra_seeds = sorted(set(draws["seed"].tolist()) - set(asked))
    if extra_seeds:
        raise ConfigurationError(
            f"the splits' draws hold seeds that the settings do not ask for: "
            f"{listed(extra_seeds)}; the settings ask for {listed(asked)}; "
            f"{_first_point(draws[draws['seed'].isin(extra_seeds)])}"
        )
    labels = [size_label(size) for size in sizes]
    extra_sizes = sorted(
        set(draws["size"].tolist()) - set(labels),
        key=lambda label: (label == FULL, int(label) if label != FULL else 0),
    )
    if extra_sizes:
        raise ConfigurationError(
            f"the splits' draws hold sizes that the settings do not ask for: "
            f"{listed(extra_sizes)}; the settings ask for {listed(labels)}; "
            f"{_first_point(draws[draws['size'].isin(extra_sizes)])}"
        )
    outside = draws[(draws["fold"] < 0) | (draws["fold"] >= k_folds)]
    if not outside.empty:
        raise ConfigurationError(
            f"the splits' draws hold fold labels outside 0 to {k_folds - 1}: "
            f"{listed(sorted(set(outside['fold'].tolist())))}; {_first_point(outside)}"
        )


def _seed_folds(
    table: pd.DataFrame, seed: int, aligned: AlignedInputs, index: pd.Index, k_folds: int
) -> Folds:
    """The folds of one seed as positions, after every check that the folds must pass."""
    rows = table[table["seed"] == seed]
    ids = rows["text_id"].to_numpy(dtype=object)
    shown = [
        f"{text} (fold {label})"
        for text, label in zip(ids, rows["fold"].tolist(), strict=True)
        if not 0 <= label < k_folds
    ]
    if shown:
        raise ConfigurationError(
            f"seed {seed}: fold labels must lie from 0 to {k_folds - 1} (k_folds {k_folds}), "
            f"but these texts have another: {listed(shown)}"
        )
    labels = rows["fold"].to_numpy(dtype=np.int64)
    positions = index.get_indexer(ids)
    unknown = ids[positions < 0].tolist()
    if unknown:
        raise ConfigurationError(
            f"seed {seed}: the folds hold texts that are not analysis texts (rows of the gold "
            f"table): {listed(unknown)}"
        )
    counts = np.bincount(positions, minlength=aligned.text_ids.shape[0])
    repeated = aligned.text_ids[counts > 1].tolist()
    if repeated:
        raise ConfigurationError(
            f"seed {seed}: the folds hold the texts {listed(repeated)} in more than one row; "
            f"every analysis text has exactly one fold"
        )
    missing = aligned.text_ids[counts == 0].tolist()
    if missing:
        raise ConfigurationError(
            f"seed {seed}: the folds lack the analysis texts {listed(missing)}; every analysis "
            f"text has exactly one fold"
        )
    fold_of = np.empty(aligned.text_ids.shape[0], dtype=np.int64)
    fold_of[positions] = labels
    empty = [fold for fold in range(k_folds) if not (fold_of == fold).any()]
    if empty:
        named = f"fold {empty[0]} holds" if len(empty) == 1 else f"the folds {listed(empty)} hold"
        raise ConfigurationError(
            f"seed {seed}: {named} no text; every fold from 0 to {k_folds - 1} must hold texts"
        )
    spans = pd.Series(fold_of).groupby(aligned.group_id, sort=False).unique()
    straddling = [
        f"{group} (folds {', '.join(str(fold) for fold in sorted(folds))})"
        for group, folds in spans.items()
        if len(folds) > 1
    ]
    if straddling:
        raise ConfigurationError(
            f"seed {seed}: the groups {listed(straddling)} have texts in more than one fold; a "
            f"group never straddles train and test"
        )
    all_positions = np.arange(aligned.text_ids.shape[0])
    return [
        (all_positions[fold_of != fold], all_positions[fold_of == fold]) for fold in range(k_folds)
    ]


def _partial_groups(drawn: np.ndarray, in_pool: np.ndarray, groups: np.ndarray) -> list[str]:
    """The groups of the pool that a draw holds only part of, as ``group (k of n texts)``."""
    in_draw = np.zeros(groups.shape[0], dtype=bool)
    in_draw[drawn] = True
    pool = pd.DataFrame({"group": groups[in_pool], "drawn": in_draw[in_pool]})
    counts = pool.groupby("group", sort=False)["drawn"].agg(["sum", "size"])
    partial = counts[(counts["sum"] > 0) & (counts["sum"] < counts["size"])]
    return [
        f"{group} ({int(taken)} of {int(held)} texts)"
        for group, taken, held in zip(partial.index, partial["sum"], partial["size"], strict=True)
    ]


def _checked_draw(
    ids: np.ndarray,
    where: str,
    quota: int | None,
    train: np.ndarray,
    aligned: AlignedInputs,
    index: pd.Index,
) -> np.ndarray:
    """The positions of one supplied draw, after every check that a draw must pass."""
    positions = index.get_indexer(ids)
    unknown = ids[positions < 0].tolist()
    if unknown:
        raise ConfigurationError(
            f"{where}: the draw holds texts that are not analysis texts: {listed(unknown)}"
        )
    in_pool = np.zeros(aligned.text_ids.shape[0], dtype=bool)
    in_pool[train] = True
    outside = np.sort(positions[~in_pool[positions]])
    if outside.size:
        raise ConfigurationError(
            f"{where}: the draw holds texts outside the fold's training pool, from its test "
            f"fold: {listed(aligned.text_ids[outside].tolist())}"
        )
    drawn = np.sort(positions)
    if quota is None:
        lacking = np.setdiff1d(train, drawn)
        if lacking.size:
            raise ConfigurationError(
                f"{where}: a draw of size full must be the whole training pool; it lacks the "
                f"texts {listed(aligned.text_ids[lacking].tolist())}"
            )
        return drawn
    partial = _partial_groups(drawn, in_pool, aligned.group_id)
    if partial:
        raise ConfigurationError(
            f"{where}: a draw takes whole groups (every text of the training pool that belongs "
            f"to a drawn group), but this one holds part of the groups {listed(partial)}"
        )
    units = int(np.unique(aligned.unit_id[drawn]).shape[0])
    pool_units = int(np.unique(aligned.unit_id[train]).shape[0])
    if units < quota and units < pool_units:
        raise ConfigurationError(
            f"{where}: the draw holds {units} units, fewer than the size names ({quota}), "
            f"although the fold's training pool holds {pool_units}; no size is fitted on fewer "
            f"units than it names; the draw holds the texts "
            f"{listed(aligned.text_ids[drawn].tolist())}"
        )
    return drawn


def _receipt(
    size: int | str, drawn: np.ndarray, train: np.ndarray, aligned: AlignedInputs
) -> DrawReceipt:
    """The receipt of a supplied draw, computed as ``draw_groups`` computes its own."""
    return DrawReceipt(
        size=size,
        requested_units=0 if size == FULL else int(size),
        actual_units=int(np.unique(aligned.unit_id[drawn]).shape[0]),
        actual_rows=int(drawn.shape[0]),
        actual_groups=int(np.unique(aligned.group_id[drawn]).shape[0]),
        pool_units=int(np.unique(aligned.unit_id[train]).shape[0]),
        pool_rows=int(train.shape[0]),
        pool_groups=int(np.unique(aligned.group_id[train]).shape[0]),
    )


def _point_draws(
    part: pd.DataFrame | None,
    seed: int,
    size: int | str,
    folds: Folds,
    aligned: AlignedInputs,
    index: pd.Index,
) -> Draws:
    """The checked draws of every fold of one (seed, size)."""
    quota = None if size == FULL else int(size)
    draws: Draws = []
    for fold, (train, _) in enumerate(folds):
        where = f"seed {seed} size {size_label(size)} fold {fold}"
        ids = (
            NO_DRAW if part is None else part.loc[part["fold"] == fold, "text_id"].to_numpy(object)
        )
        if ids.size == 0 and quota is not None:
            raise ConfigurationError(
                f"{where}: the splits hold no draw for this fold; give the draw of every fold of "
                f"every seed and size whose draws the run consults (to fit them, or to check "
                f"them against the floor), or pass draws=None and the library draws them"
            )
        drawn = train if ids.size == 0 else _checked_draw(ids, where, quota, train, aligned, index)
        draws.append((drawn, _receipt(size, drawn, train, aligned)))
    return draws


def check_splits(
    splits: BaselineSplits,
    aligned: AlignedInputs,
    *,
    seeds: Sequence[int],
    sizes: Sequence[int | str],
    k_folds: int,
    floor_units: int,
) -> SuppliedSplits:
    """Check supplied splits against the analysis rows and the settings; return positions.

    The folds of every seed are checked first, then the draws of every (seed, size) whose draws
    the baseline consults: those of a size that is skipped because its quota exceeds half the
    units or some fold's pool lies below it are not needed, and supplied ones are not used (a
    warning says so). A count size for which the splits hold no draw at all needs none when
    its smallest pool holds fewer units than the floor, as no draw could reach the floor: it is
    recorded in ``pool_skips``. A draw of size ``"full"`` that the splits leave out is the
    whole pool.

    Args:
        splits: The supplied splits.
        aligned: The aligned inputs; their rows are the analysis texts.
        seeds: The seeds of the settings.
        sizes: The sizes of the settings.
        k_folds: The number of folds of the settings.
        floor_units: The floor of the settings.

    Returns:
        The folds by seed and, when the splits hold draws, the checked draws with their
        receipts and the skips that the pools decide, by seed and size label.

    Raises:
        ConfigurationError: If the supplied splits break a guarantee of the baseline; the
            message names the seed, the size and fold where they apply, and the first ten
            offending ids (see ``run_ridge_baseline``).
    """
    _check_seeds(splits.folds, seeds)
    if splits.draws is not None:
        _check_draw_labels(splits.draws, seeds, sizes, k_folds)
    index = pd.Index(aligned.text_ids)
    folds = {
        int(seed): _seed_folds(splits.folds, int(seed), aligned, index, k_folds) for seed in seeds
    }
    if splits.draws is None:
        return SuppliedSplits(folds=folds, draws=None, pool_skips={})
    n_units = int(np.unique(aligned.unit_id).shape[0])
    by_point = {
        (int(seed), str(label)): part
        for (seed, label), part in splits.draws.groupby(["seed", "size"], sort=False)
    }
    draws: dict[tuple[int, str], Draws] = {}
    pool_skips: dict[tuple[int, str], str] = {}
    for seed in [int(value) for value in seeds]:
        for size in sizes:
            label = size_label(size)
            part = by_point.get((seed, label))
            reason = quota_or_pool_reason(aligned, folds[seed], size, n_units)
            if reason and part is not None:
                logger.warning(
                    "seed %s size %s: the supplied draws are not used, because the size is "
                    "skipped: %s",
                    seed,
                    label,
                    reason,
                )
            if reason:
                continue
            bound = ""
            if part is None and size != FULL:
                bound = pool_floor_reason(aligned, folds[seed], floor_units)
            if bound:
                pool_skips[(seed, label)] = bound
            else:
                draws[(seed, label)] = _point_draws(part, seed, size, folds[seed], aligned, index)
    return SuppliedSplits(folds=folds, draws=draws, pool_skips=pool_skips)
