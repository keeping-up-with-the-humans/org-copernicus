"""Table layout, identifier checks, and connections of the run store.

One store file holds the results of one run: a ``results`` table with a row per
text and run, an ``evidence`` table with a row per returned passage, ``wording``
and ``shuffles`` tables with the prompt variation that was used, and a
``run_info`` table of key/value provenance. Every table and column name is
double-quoted in the SQL the library writes, so the ids of a rubric can be used
as column names once they follow the rule of
`copernicus.rubric.identifiers.identifier_problems`, which rubric validation
applies too. SQLite reads a double-quoted name that matches no column as a string
literal, so a store whose columns differ from this layout is refused before it is
read (`copernicus.coders.store_layout.check_layout`).
"""

from __future__ import annotations

import dataclasses
import logging
import sqlite3
from collections.abc import Sequence
from contextlib import closing
from pathlib import Path

from copernicus.coders.evidence import EvidenceRow
from copernicus.errors import StoreError
from copernicus.rubric.identifiers import identifier_problems
from copernicus.rubric.model import Rubric

logger = logging.getLogger(__name__)

__all__ = [
    "ATTRIBUTE_FIELDS",
    "BUSY_TIMEOUT_MS",
    "EVIDENCE_COLUMNS",
    "FIXED_RESULT_COLUMNS",
    "MARKER_FIELDS",
    "SKIP_KEY",
    "TABLES",
    "attribute_column",
    "create_statements",
    "is_blank",
    "layout_columns",
    "marker_column",
    "open_read_only",
    "open_read_write",
    "quote",
    "read_run_info",
    "result_columns",
    "validate_identifiers",
]

BUSY_TIMEOUT_MS = 30000
NOT_A_DATABASE = "file is not a database"
TABLES = ("results", "evidence", "wording", "shuffles", "run_info")

FIXED_RESULT_COLUMNS = (
    ("text_id", "TEXT NOT NULL"),
    ("text_key", "INTEGER NOT NULL"),
    ("text_hash", "TEXT NOT NULL"),
    ("run_number", "INTEGER NOT NULL"),
    ("model_name", "TEXT NOT NULL"),
    ("temperature", "REAL NOT NULL"),
    ("seed", "INTEGER NOT NULL"),
    ("role_id", "TEXT NOT NULL"),
    ("task_id", "TEXT NOT NULL"),
    ("level_id", "TEXT NOT NULL"),
    ("shuffle_seed", "INTEGER NOT NULL"),
    ("seconds", "REAL"),
    ("prompt_tokens", "INTEGER"),
    ("completion_tokens", "INTEGER"),
    ("created_at", "TEXT NOT NULL"),
    ("library_version", "TEXT NOT NULL"),
    ("temperature_applied", "REAL"),
    ("top_p_applied", "REAL"),
)
ATTRIBUTE_FIELDS = (
    ("score", "INTEGER"),
    ("status", "TEXT NOT NULL"),
    ("explanation", "TEXT"),
    ("stage1_attempt", "INTEGER"),
    ("stage2_attempt", "INTEGER"),
    ("stage1_temperature_applied", "REAL"),
)
MARKER_FIELDS = (("evaluable", "INTEGER"), ("justification", "TEXT"))
SKIP_KEY = ("text_id", "run_number", "model_name", "temperature")
_EVIDENCE_TYPES = {
    "attribute_id": "TEXT NOT NULL",
    "marker_id": "TEXT NOT NULL",
    "quote_index": "INTEGER NOT NULL",
    "quote_raw": "TEXT NOT NULL",
    "quote_norm": "TEXT NOT NULL",
    "verified": "INTEGER NOT NULL",
    "match_score": "REAL NOT NULL",
    "char_start": "INTEGER",
    "char_end": "INTEGER",
    "duplicate_of": "INTEGER",
    "used_in_stage2": "INTEGER NOT NULL",
}
EVIDENCE_COLUMNS = tuple(
    [(name, dict(FIXED_RESULT_COLUMNS)[name]) for name in SKIP_KEY]
    + [(f.name, _EVIDENCE_TYPES[f.name]) for f in dataclasses.fields(EvidenceRow)]
)


def quote(name: str) -> str:
    """Return an SQL identifier in double quotes."""
    return '"' + name.replace('"', '""') + '"'


def attribute_column(attribute_id: str, field: str) -> str:
    """Return the results column of one attribute field, e.g. ``tone__score``."""
    return f"{attribute_id}__{field}"


def marker_column(marker_id: str, field: str) -> str:
    """Return the results column of one marker field, e.g. ``tone_respect__evaluable``."""
    return f"{marker_id}__{field}"


def validate_identifiers(rubric: Rubric) -> None:
    """Check that the rubric ids can serve as column name prefixes.

    The rule is `copernicus.rubric.identifiers.identifier_problems`, the one
    rubric validation applies; a rubric built without validation is checked here.

    Args:
        rubric: The rubric whose attribute and marker ids become columns.

    Raises:
        StoreError: If an id does not match ``[A-Za-z_][A-Za-z0-9_]*``, if two
            ids are equal when compared without regard to case (SQLite column
            names are case-insensitive), if an id equals the name of a fixed
            column or of a column the exports add, or if an id is ``construct`` or
            ends in ``__rollup``, the names of the generated targets; the message
            names every such id.
    """
    ids = [a.attribute_id for a in rubric.attributes]
    ids += [m.marker_id for a in rubric.attributes for m in a.markers]
    problems = identifier_problems(ids)
    if problems:
        raise StoreError("; ".join(message for _, message in problems))


def result_columns(rubric: Rubric) -> list[tuple[str, str]]:
    """Return the name and SQL declaration of every column of ``results``.

    Args:
        rubric: The rubric that defines the attribute and marker columns.

    Returns:
        The fixed columns, then the columns of `ATTRIBUTE_FIELDS` for each attribute,
        then the two columns of each marker, in rubric order.
    """
    markers = [marker.marker_id for attribute in rubric.attributes for marker in attribute.markers]
    return layout_columns([attribute.attribute_id for attribute in rubric.attributes], markers)


def layout_columns(
    attribute_ids: Sequence[str], marker_ids: Sequence[str]
) -> list[tuple[str, str]]:
    """Return the name and SQL declaration of every column of ``results`` for these ids.

    The ``run_info`` of a store records the ids of its rubric in rubric order, so the columns of
    a store follow from its ``run_info`` alone.

    Args:
        attribute_ids: The attribute ids, in rubric order.
        marker_ids: The marker ids, in rubric order (the markers of the first attribute first).

    Returns:
        The fixed columns, then the columns of `ATTRIBUTE_FIELDS` for each attribute id, then
        the columns of `MARKER_FIELDS` for each marker id.
    """
    columns = list(FIXED_RESULT_COLUMNS)
    for attribute_id in attribute_ids:
        for field, declaration in ATTRIBUTE_FIELDS:
            columns.append((attribute_column(attribute_id, field), declaration))
    for marker_id in marker_ids:
        for field, declaration in MARKER_FIELDS:
            columns.append((marker_column(marker_id, field), declaration))
    return columns


def _create_table(name: str, lines: list[str]) -> str:
    """Return a CREATE TABLE statement from column and constraint lines."""
    body = ",\n".join(f"    {line}" for line in lines)
    return f"CREATE TABLE {quote(name)} (\n{body}\n)"


def create_statements(rubric: Rubric) -> list[str]:
    """Return the CREATE statements of all tables of a store.

    Args:
        rubric: The rubric that defines the columns of ``results``.

    Returns:
        One statement per table, in the order of `TABLES`.
    """
    skip_key = ", ".join(quote(c) for c in SKIP_KEY)
    results = [f"{quote(name)} {declaration}" for name, declaration in result_columns(rubric)]
    results.append(f"UNIQUE ({skip_key})")
    evidence = [f"{quote(name)} {declaration}" for name, declaration in EVIDENCE_COLUMNS]
    evidence_key = ", ".join(
        quote(c) for c in (*SKIP_KEY, "attribute_id", "marker_id", "quote_index")
    )
    evidence.append(f"PRIMARY KEY ({evidence_key})")
    return [
        _create_table("results", results),
        _create_table("evidence", evidence),
        _create_table(
            "wording",
            [
                '"dimension" TEXT NOT NULL',
                '"variant_id" TEXT NOT NULL',
                '"column_name" TEXT NOT NULL',
                '"text" TEXT NOT NULL',
                'PRIMARY KEY ("dimension", "variant_id", "column_name")',
            ],
        ),
        _create_table(
            "shuffles",
            [
                '"shuffle_seed" INTEGER NOT NULL',
                '"attribute_id" TEXT NOT NULL',
                '"marker_order_json" TEXT NOT NULL',
                'PRIMARY KEY ("shuffle_seed", "attribute_id")',
            ],
        ),
        _create_table(
            "run_info",
            ['"key" TEXT PRIMARY KEY', '"value" TEXT NOT NULL'],
        ),
    ]


def open_read_write(path: Path) -> sqlite3.Connection:
    """Open a store file for writing in WAL mode, creating it if absent.

    Args:
        path: Location of the SQLite file.

    Returns:
        A connection with a 30 second busy timeout and a WAL journal; it opens
        transactions implicitly before writes.

    Raises:
        StoreError: If the folder of the file does not exist.
    """
    if not path.parent.is_dir():
        raise StoreError(f"folder of the store does not exist: {path.parent}")
    connection = sqlite3.connect(path.resolve().as_uri(), uri=True, timeout=BUSY_TIMEOUT_MS / 1000)
    try:
        connection.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS}")
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
    except BaseException:
        connection.close()
        raise
    return connection


def open_read_only(path: Path) -> sqlite3.Connection:
    """Open an existing store file without the ability to change it.

    Args:
        path: Location of the SQLite file.

    Returns:
        A read-only connection.

    Raises:
        StoreError: If the file does not exist.
    """
    if not path.is_file():
        raise StoreError(f"store does not exist: {path}")
    connection = sqlite3.connect(
        f"{path.resolve().as_uri()}?mode=ro", uri=True, timeout=BUSY_TIMEOUT_MS / 1000
    )
    connection.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS}")
    return connection


def _first_row(connection: sqlite3.Connection, statement: str) -> tuple[object, ...] | None:
    """Run a query on a store file and return its first row.

    Args:
        connection: An open connection to a candidate store file.
        statement: The query to run.

    Returns:
        The first row, or ``None`` if the query returns none.

    Raises:
        StoreError: If the file is not a SQLite database at all. Other database
            errors, such as a locked database, propagate unchanged.
    """
    try:
        return connection.execute(statement).fetchone()
    except sqlite3.DatabaseError as error:
        if type(error) is not sqlite3.DatabaseError or NOT_A_DATABASE not in str(error):
            raise
        location = connection.execute("PRAGMA database_list").fetchone()[2]
        raise StoreError(f"file is not a run store: {location}") from error


def is_blank(path: Path) -> bool:
    """Return whether a file holds no tables, so a store can be created in it.

    Args:
        path: Location of the SQLite file.

    Returns:
        ``True`` if the file is missing, empty, or a database without tables
        (for example the leftover of a creation that failed and was rolled back).

    Raises:
        StoreError: If the file is not a SQLite database.
    """
    if not path.is_file() or path.stat().st_size == 0:
        return True
    with closing(open_read_only(path)) as connection:
        found = _first_row(connection, "SELECT 1 FROM sqlite_master WHERE type = 'table'")
    return found is None


def read_run_info(connection: sqlite3.Connection) -> dict[str, str]:
    """Return the ``run_info`` table as a dictionary.

    Args:
        connection: An open connection to a store.

    Returns:
        Key to value, in key order.

    Raises:
        StoreError: If the file is no SQLite database or has no ``run_info`` table.
    """
    exists = _first_row(
        connection, "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'run_info'"
    )
    if exists is None:
        raise StoreError("file is not a run store: it has no run_info table")
    rows = connection.execute('SELECT "key", "value" FROM "run_info" ORDER BY "key"').fetchall()
    return {key: value for key, value in rows}
