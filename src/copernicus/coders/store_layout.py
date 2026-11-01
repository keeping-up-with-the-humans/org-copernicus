"""The columns of the tables of a store file, and the check that they are the library's.

SQLite reads a double-quoted name that matches no column as a string literal, so a reader that
selected a column the file lacks would return the column's name as data. Every reader of a
store therefore compares the columns of ``results`` and ``evidence`` with the layout of
`copernicus.coders.store_schema` before it reads a row (`check_layout`): with the columns that
the rubric of a run implies, or with those that the ids of the store's own ``run_info`` imply.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from collections.abc import Sequence
from pathlib import Path

from copernicus.coders.store_schema import (
    EVIDENCE_COLUMNS,
    TABLES,
    layout_columns,
    quote,
    read_run_info,
)
from copernicus.errors import StoreError

logger = logging.getLogger(__name__)

__all__ = ["EARLIER_LAYOUT", "LAYOUT_KEYS", "check_layout", "count_rows", "table_columns"]

EARLIER_LAYOUT = (
    "the run store {path} was made by an earlier version of the library, or was changed by hand; "
    "code into a new store"
)
# The run_info entries whose ids, in rubric order, imply the attribute and marker columns.
LAYOUT_KEYS = ("attribute_ids", "marker_ids")


def table_columns(connection: sqlite3.Connection) -> dict[str, list[str]]:
    """Return the column names of every table of a store, read with ``PRAGMA table_info``.

    Args:
        connection: An open connection to a store.

    Returns:
        The column names of each table of `TABLES`, in column order; a table the file lacks
        has none.
    """
    return {
        table: [row[1] for row in connection.execute(f"PRAGMA main.table_info({quote(table)})")]
        for table in TABLES
    }


def check_layout(
    connection: sqlite3.Connection, path: Path, result_names: Sequence[str] | None = None
) -> None:
    """Raise if the tables of a store lack the columns this version of the library writes.

    Every column of ``results`` and of ``evidence`` is compared, so that no column is read that
    the file lacks.

    Args:
        connection: An open connection to the store.
        path: Location of the store, for the message.
        result_names: Every column of ``results`` that the rubric of the run implies, as
            `copernicus.coders.store_schema.result_columns` names them; ``None`` takes the
            columns that the ids of the store's own ``run_info`` imply (``attribute_ids`` and
            ``marker_ids``), which needs no rubric.

    Raises:
        StoreError: If the file is no run store (it has no ``run_info`` with the ids), or a
            compared list of columns differs: the store was made by an earlier version of the
            library, or changed by hand, and has to be coded anew.
    """
    if result_names is not None:
        wanted = list(result_names)
    else:
        info = read_run_info(connection)
        missing = [key for key in LAYOUT_KEYS if key not in info]
        if missing:
            raise StoreError(f"file is not a run store: its run_info lacks {missing}: {path}")
        attribute_ids, marker_ids = (json.loads(info[key]) for key in LAYOUT_KEYS)
        wanted = [name for name, _ in layout_columns(attribute_ids, marker_ids)]
    evidence = [name for name, _ in EVIDENCE_COLUMNS]
    columns = table_columns(connection)
    if columns["results"] != wanted or columns["evidence"] != evidence:
        raise StoreError(EARLIER_LAYOUT.format(path=path))


def count_rows(connection: sqlite3.Connection) -> dict[str, int]:
    """Return the number of rows of each table of a store.

    Args:
        connection: An open connection to a store.

    Returns:
        Row count per table name in `TABLES`.
    """
    return {
        table: int(connection.execute(f"SELECT COUNT(*) FROM {quote(table)}").fetchone()[0])
        for table in TABLES
    }
