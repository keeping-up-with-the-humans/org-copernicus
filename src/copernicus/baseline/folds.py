"""Grouped outer folds and whole-group draws of training data.

* ``grouped_kfold`` partitions the groups (the clusters that must not straddle a split,
  such as persons or conversations) into ``k`` test folds; a group is never split across
  train and test.
* ``draw_groups`` selects whole groups from a fold's training pool until the requested
  number of analysis units is reached (bounded overshoot, recorded in the ``DrawReceipt``);
  ``"full"`` takes the whole training pool.
* Feasibility: a requested quota above half the cohort's units is a recorded skip, and the
  caller compares an actual draw with ``FLOOR_UNITS`` units.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass

import numpy as np

from copernicus.errors import ConfigurationError, EstimatorError, SettingsError

logger = logging.getLogger(__name__)

__all__ = [
    "FLOOR_UNITS",
    "K_FOLDS",
    "SIZES",
    "DrawReceipt",
    "draw_groups",
    "grouped_kfold",
    "rng_for",
    "size_skip_reason",
]

SIZES: tuple[int | str, ...] = (50, 100, 150, 200, 250, 500, 1000, "full")
"""The default training sizes, in analysis units; ``"full"`` takes the whole training pool."""
FLOOR_UNITS = 50
"""The default floor: the fewest analysis units a training draw may hold."""
K_FOLDS = 5
"""The default number of grouped outer folds."""
FULL = "full"


def rng_for(*parts: object) -> np.random.Generator:
    """Deterministic generator keyed by the string parts joined with ``::`` (md5).

    The seed is the first 16 hex digits of the md5 digest, read as an integer.

    Args:
        *parts: Values identifying the draw, such as a run name, a seed and a fold index.

    Returns:
        A fresh generator that is the same for the same parts.
    """
    digest = hashlib.md5("::".join(str(part) for part in parts).encode()).hexdigest()
    return np.random.default_rng(int(digest[:16], 16))


def _quota(size: int | str) -> int | None:
    """Requested unit quota of a size; ``None`` for ``"full"``."""
    if size == FULL:
        return None
    if isinstance(size, bool) or not isinstance(size, int | np.integer) or size < 1:
        raise SettingsError(f"size must be a positive integer or {FULL!r}; got {size!r}")
    return int(size)


def size_skip_reason(size: int | str, n_units: int) -> str:
    """Why a size is not admissible for a cohort, or an empty string when it is.

    Args:
        size: Requested number of analysis units, or ``"full"``.
        n_units: Number of analysis units in the cohort.

    Returns:
        ``""`` for ``"full"`` or a quota of at most half the units, otherwise
        ``quota_exceeds_half_units(quota=<q>,units=<n>)``.

    Raises:
        SettingsError: If ``size`` is neither a positive integer nor ``"full"``.
    """
    quota = _quota(size)
    if quota is not None and quota > n_units // 2:
        return f"quota_exceeds_half_units(quota={quota},units={n_units})"
    return ""


def grouped_kfold(
    group_id: np.ndarray, k: int, rng: np.random.Generator
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Split rows into ``k`` grouped folds as ``(train_pos, test_pos)`` position arrays.

    Groups are randomly permuted and split into ``k`` nearly equal sets of groups; every
    row lands in exactly one test fold together with its whole group.

    Args:
        group_id: Group label of each row.
        k: Number of folds.
        rng: Generator for the permutation of the groups.

    Returns:
        One ``(train_pos, test_pos)`` pair per fold.

    Raises:
        ConfigurationError: If ``k`` is below 2 or there are fewer groups than folds.
        EstimatorError: If the test folds do not partition the rows.
    """
    if k < 2:
        raise ConfigurationError(f"grouped_kfold needs k of at least 2; got k={k}")
    groups = np.unique(group_id)
    if groups.shape[0] < k:
        raise ConfigurationError(
            f"grouped_kfold needs at least k groups; got {groups.shape[0]} groups for k={k} "
            f"in {group_id.shape[0]} rows"
        )
    permuted = groups[rng.permutation(groups.shape[0])]
    all_positions = np.arange(group_id.shape[0])
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for test_groups in np.array_split(permuted, k):
        test_mask = np.isin(group_id, test_groups)
        folds.append((all_positions[~test_mask], all_positions[test_mask]))
    covered = np.concatenate([test for _, test in folds])
    if np.unique(covered).shape[0] != group_id.shape[0] or covered.shape[0] != group_id.shape[0]:
        raise EstimatorError("grouped_kfold: test folds do not partition the rows")
    return folds


@dataclass(frozen=True)
class DrawReceipt:
    """What a draw asked for and what it selected.

    Attributes:
        size: The requested size, an integer quota or ``"full"``.
        requested_units: The unit quota; 0 for ``"full"``.
        actual_units: Analysis units in the drawn rows.
        actual_rows: Drawn rows.
        actual_groups: Groups in the drawn rows.
        pool_units: Analysis units in the training pool.
        pool_rows: Rows in the training pool.
        pool_groups: Groups in the training pool.
    """

    size: int | str
    requested_units: int
    actual_units: int
    actual_rows: int
    actual_groups: int
    pool_units: int
    pool_rows: int
    pool_groups: int

    @property
    def overshoot(self) -> int:
        """Units drawn beyond the quota; 0 for ``"full"``."""
        return self.actual_units - self.requested_units if self.requested_units > 0 else 0


def draw_groups(
    train_pos: np.ndarray,
    group_id: np.ndarray,
    unit_id: np.ndarray,
    size: int | str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, DrawReceipt]:
    """Draw whole groups from a training pool until the size's unit quota is reached.

    Groups are taken in random order and the draw stops at the first group that brings the
    number of distinct units to the quota, so the draw can overshoot by less than one
    group. ``"full"`` returns the whole pool. The caller decides whether ``actual_units``
    meets the floor.

    Args:
        train_pos: Row positions of the training pool.
        group_id: Group label of every row.
        unit_id: Analysis unit of every row.
        size: Requested number of analysis units, or ``"full"``.
        rng: Generator for the group order.

    Returns:
        The sorted drawn positions and the receipt.

    Raises:
        SettingsError: If ``size`` is neither a positive integer nor ``"full"``.
        EstimatorError: If the draw stops below the quota while groups remain.
    """
    quota = _quota(size)
    pool_groups = np.unique(group_id[train_pos])
    pool_units = int(np.unique(unit_id[train_pos]).shape[0])
    if quota is None:
        drawn = np.sort(train_pos)
        requested = 0
    else:
        order = pool_groups[rng.permutation(pool_groups.shape[0])]
        group_of = group_id[train_pos]
        unit_of = unit_id[train_pos]
        chosen_groups = []
        units_seen: set = set()
        for group in order:
            chosen_groups.append(group)
            units_seen.update(unit_of[group_of == group].tolist())
            if len(units_seen) >= quota:
                break
        drawn = np.sort(train_pos[np.isin(group_of, np.array(chosen_groups))])
        requested = quota
    receipt = DrawReceipt(
        size=size,
        requested_units=requested,
        actual_units=int(np.unique(unit_id[drawn]).shape[0]),
        actual_rows=int(drawn.shape[0]),
        actual_groups=int(np.unique(group_id[drawn]).shape[0]),
        pool_units=pool_units,
        pool_rows=int(train_pos.shape[0]),
        pool_groups=int(pool_groups.shape[0]),
    )
    if quota is not None and receipt.actual_units < quota and receipt.actual_units < pool_units:
        raise EstimatorError(
            f"draw_groups: stopped below quota with groups remaining "
            f"(drew {receipt.actual_units} of {quota} units from {pool_units} in the pool)"
        )
    logger.debug(
        "draw %r: %d units in %d groups", size, receipt.actual_units, receipt.actual_groups
    )
    return drawn, receipt
