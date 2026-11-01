"""The cells of a run store as resuming and the failure ceiling read them.

A cell is one text in one run. For each stored cell the store holds the text hash of the text
it was coded on and the status of each attribute result: resuming compares the hash with the
text to code, and the failure ceiling counts the statuses of the cells it keeps.
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, get_args

from copernicus.coders.store_schema import attribute_column, quote
from copernicus.coders.two_stage_types import EXTRACTION_FAILED, SCORING_FAILED, Status
from copernicus.errors import StoreError

logger = logging.getLogger(__name__)

__all__ = [
    "FAILURE_STATUSES",
    "MAX_IDS_IN_MESSAGE",
    "StoredCell",
    "delete_failed_cells",
    "listed_ids",
    "read_cells",
]

FAILURE_STATUSES = (EXTRACTION_FAILED, SCORING_FAILED)
MAX_IDS_IN_MESSAGE = 10
CELL_CONDITION = '"text_id" = ? AND "run_number" = ? AND "model_name" = ? AND "temperature" = ?'

Key = tuple[str, int]


@dataclass(frozen=True)
class StoredCell:
    """What the run store holds for one text in one run.

    Attributes:
        text_hash: The text hash of the text the cell was coded on.
        statuses: The status of each attribute result, in rubric order.
    """

    text_hash: str
    statuses: tuple[str, ...]

    @property
    def failed(self) -> bool:
        """Whether an attribute result ended in ``extraction_failed`` or ``scoring_failed``."""
        return any(status in FAILURE_STATUSES for status in self.statuses)


def listed_ids(ids: Sequence[str]) -> str:
    """Return the first ten ids, quoted, with the number of the others."""
    shown = ", ".join(repr(text_id) for text_id in ids[:MAX_IDS_IN_MESSAGE])
    rest = len(ids) - MAX_IDS_IN_MESSAGE
    return shown + (f" and {rest} more" if rest > 0 else "")


def _select(attribute_ids: Sequence[str]) -> str:
    """Return the query of the key, the text hash, and the attribute statuses of cells."""
    statuses = ", ".join(quote(attribute_column(a, "status")) for a in attribute_ids)
    return f'SELECT "text_id", "run_number", "text_hash", {statuses} FROM "results"'


def _cell(row: Sequence[Any]) -> tuple[Key, StoredCell]:
    """Turn a row of `_select` into the key and the cell, refusing a status the library lacks."""
    text_id, run_number, text_hash, *statuses = row
    unknown = sorted(set(statuses) - set(get_args(Status)))
    if unknown:
        raise StoreError(
            f"text_id={text_id!r}, run_number={run_number}: unknown statuses {unknown}"
        )
    return (text_id, run_number), StoredCell(text_hash, tuple(statuses))


def read_cells(
    connection: sqlite3.Connection,
    attribute_ids: Sequence[str],
    model_name: str,
    temperature: float,
    keys: Iterable[Key] | None = None,
) -> dict[Key, StoredCell]:
    """Return the stored cells of one model and temperature.

    Args:
        connection: An open connection to a store.
        attribute_ids: The attribute ids of the rubric, in rubric order.
        model_name: Model name of the run.
        temperature: Temperature of the run.
        keys: The ``(text_id, run_number)`` to read; ``None`` reads every cell.

    Returns:
        The cell of each key that is stored.

    Raises:
        StoreError: If a stored status is not one the library writes.
    """
    query = _select(attribute_ids)
    if keys is None:
        rows = connection.execute(
            f'{query} WHERE "model_name" = ? AND "temperature" = ?', (model_name, temperature)
        ).fetchall()
    else:
        found = (
            connection.execute(
                f"{query} WHERE {CELL_CONDITION}", (text_id, run, model_name, temperature)
            ).fetchone()
            for text_id, run in keys
        )
        rows = [row for row in found if row is not None]
    return dict(_cell(row) for row in rows)


def delete_failed_cells(
    connection: sqlite3.Connection,
    keys: Iterable[Key],
    attribute_ids: Sequence[str],
    model_name: str,
    temperature: float,
) -> None:
    """Delete the stored cells among ``keys``, each of which must hold a failed attribute.

    The results row and the evidence of every stored cell among ``keys`` are deleted inside
    the caller's transaction; keys that are not stored are passed over.

    Args:
        connection: A connection with an open transaction that holds the write lock, so no
            other writer changes a cell between its check and its deletion.
        keys: The ``(text_id, run_number)`` of the cells.
        attribute_ids: The attribute ids of the rubric, in rubric order.
        model_name: Model name of the run.
        temperature: Temperature of the run.

    Raises:
        StoreError: If a stored cell among ``keys`` has no attribute result in
            ``extraction_failed`` or ``scoring_failed``; nothing is deleted then.
    """
    stored = read_cells(connection, attribute_ids, model_name, temperature, keys)
    kept = sorted(key for key, cell in stored.items() if not cell.failed)
    if kept:
        text_id, run_number = kept[0]
        raise StoreError(
            f"text_id={text_id!r}, run_number={run_number}: the stored cell has no failed "
            f"attribute, so it is not replaced ({len(kept)} such cell(s) in the batch)"
        )
    values = [(text_id, run, model_name, temperature) for text_id, run in stored]
    for table in ("evidence", "results"):
        connection.executemany(f"DELETE FROM {quote(table)} WHERE {CELL_CONDITION}", values)
