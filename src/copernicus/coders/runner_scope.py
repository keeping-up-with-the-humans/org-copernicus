"""The scope of one coding invocation and what the store already holds of it.

The scope is the texts of the text range times the runs 1 to ``runs``. Every cell of the
scope is coded, coded again, or kept from the store, and a kept cell must have been coded on
the text the table now holds under its id.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import get_args

from copernicus.coders.store_cells import StoredCell, listed_ids
from copernicus.coders.two_stage_types import Status
from copernicus.errors import StoreError

logger = logging.getLogger(__name__)

__all__ = ["CHANGED_TEXTS", "Scope", "changed_text_ids", "plan_scope"]

CHANGED_TEXTS = (
    "the texts changed since the store was written: a store binds every text id to the text "
    "it was coded on; code the changed texts into a new store, or restore the texts"
)


@dataclass(frozen=True)
class Scope:
    """The cells of one invocation and what becomes of each.

    Attributes:
        pending: Per text with a cell to code, ``(text id, original id, runs)``, in the
            order of the texts.
        retry: The ``(text_id, run_number)`` among ``pending`` that the store holds with a
            failed attribute and that are coded again.
        resumed: Stored cells of the scope that are kept.
        stored_status_counts: Number of attribute results per status, for every status,
            over the kept cells.
    """

    pending: tuple[tuple[str, object, tuple[int, ...]], ...]
    retry: frozenset[tuple[str, int]]
    resumed: int
    stored_status_counts: Mapping[str, int]


def changed_text_ids(
    text_ids: Sequence[str],
    hashes: Mapping[str, str],
    cells: Mapping[tuple[str, int], StoredCell],
) -> list[str]:
    """Return the ids among ``text_ids`` whose stored cells were coded on another text.

    Every stored cell of an id counts, whatever its run, so a store never holds two texts
    under one id.

    Args:
        text_ids: The ids to check, in the order the result keeps.
        hashes: The text hash of the current text of each id.
        cells: The stored cells.

    Returns:
        The ids that have a stored cell whose text hash differs from ``hashes``.
    """
    stored: dict[str, set[str]] = {}
    for (text_id, _), cell in cells.items():
        stored.setdefault(text_id, set()).add(cell.text_hash)
    return [
        text_id
        for text_id in text_ids
        if text_id in stored and stored[text_id] != {hashes[text_id]}
    ]


def plan_scope(
    texts: Sequence[tuple[str, object]],
    hashes: Mapping[str, str],
    runs: int,
    cells: Mapping[tuple[str, int], StoredCell],
    retry_failed: bool,
    store_path: Path,
) -> Scope:
    """Sort the cells of the scope into those to code, to code again, and to keep.

    Args:
        texts: ``(text id as string, original id)`` of the texts of the range, in text-id
            order.
        hashes: The text hash of the text of each id, as the coder receives it.
        runs: The runs 1 to ``runs`` belong to the scope.
        cells: The cells the store holds.
        retry_failed: Whether a stored cell with a failed attribute is coded again.
        store_path: Location of the store, for the message.

    Returns:
        The scope.

    Raises:
        StoreError: If a text of the range has a stored cell that was coded on another text;
            the message names the first ten such ids and is raised before anything is coded.
    """
    changed = changed_text_ids([text_id for text_id, _ in texts], hashes, cells)
    if changed:
        noun = "text id" if len(changed) == 1 else "text ids"
        raise StoreError(
            f"{CHANGED_TEXTS} (store {store_path}; {len(changed)} changed {noun}: "
            f"{listed_ids(changed)})"
        )
    pending: list[tuple[str, object, tuple[int, ...]]] = []
    retry: set[tuple[str, int]] = set()
    stored: Counter[str] = Counter()
    resumed = 0
    for text_id, raw_id in texts:
        to_code: list[int] = []
        for run in range(1, runs + 1):
            key = (text_id, run)
            if key not in cells:
                to_code.append(run)
            elif retry_failed and cells[key].failed:
                to_code.append(run)
                retry.add(key)
            else:
                resumed += 1
                stored.update(cells[key].statuses)
        if to_code:
            pending.append((text_id, raw_id, tuple(to_code)))
    counts = {status: stored[status] for status in get_args(Status)}
    logger.info(
        "scope of %d texts and %d runs: %d cells kept, %d coded again",
        len(texts),
        runs,
        resumed,
        len(retry),
    )
    return Scope(tuple(pending), frozenset(retry), resumed, counts)
