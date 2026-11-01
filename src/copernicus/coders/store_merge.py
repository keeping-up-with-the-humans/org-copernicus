"""Fold shard stores of one run into a single store."""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Mapping, Sequence
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

from copernicus.coders.store_cells import listed_ids
from copernicus.coders.store_layout import check_layout, count_rows, table_columns
from copernicus.coders.store_schema import (
    SKIP_KEY,
    is_blank,
    open_read_only,
    open_read_write,
    quote,
    read_run_info,
)
from copernicus.errors import ConfigurationError, StoreError

logger = logging.getLogger(__name__)

__all__ = ["merge_stores"]

IGNORED_KEY = "library_version"
IGNORE_TABLES = ("wording", "shuffles")
TEXTS_DIFFER = (
    "the stores hold different texts under the same text ids: a store binds every text id to "
    "the text it was coded on, so codings of different texts cannot be merged"
)
# Timing, accounting, and provenance columns do not make two codings of a cell differ.
NON_CODING_COLUMNS = (
    "created_at",
    "library_version",
    "seconds",
    "prompt_tokens",
    "completion_tokens",
)


def _check_agreement(reference: Mapping[str, str], other: Mapping[str, str], label: str) -> None:
    """Raise unless two ``run_info`` mappings agree on every key but the library version."""
    for key in sorted((set(reference) | set(other)) - {IGNORED_KEY}):
        if key not in other:
            raise StoreError(f"run_info missing {key} in {label}")
        if key not in reference:
            raise StoreError(f"run_info missing {key} in the reference store")
        if reference[key] != other[key]:
            raise StoreError(
                f"run_info mismatch on {key}: stored {reference[key]}, requested {other[key]}"
                f" (in {label})"
            )
    if reference.get(IGNORED_KEY) != other.get(IGNORED_KEY):
        logger.warning(
            "%s differs: %s in the reference store, %s in %s",
            IGNORED_KEY,
            reference.get(IGNORED_KEY),
            other.get(IGNORED_KEY),
            label,
        )


@dataclass(frozen=True)
class _Store:
    """What the merge reads of a store before anything is written.

    Attributes:
        info: The ``run_info`` of the store.
        columns: The column names of each table.
        schema: The CREATE statements of the store.
        texts: The text hashes stored under each text id.
    """

    info: dict[str, str]
    columns: dict[str, list[str]]
    schema: list[str]
    texts: dict[str, set[str]]


def _read_store(path: Path) -> _Store:
    """Return the run_info, columns per table, CREATE statements, and text hashes of a store.

    Raises:
        StoreError: If the file is not a run store or was made by an earlier version of the
            library.
    """
    with closing(open_read_only(path)) as connection:
        info = read_run_info(connection)
        check_layout(connection, path)
        columns = table_columns(connection)
        rows = connection.execute(
            "SELECT sql FROM sqlite_master WHERE sql IS NOT NULL AND name NOT LIKE 'sqlite_%' "
            "ORDER BY CASE type WHEN 'table' THEN 0 ELSE 1 END, rowid"
        ).fetchall()
        texts: dict[str, set[str]] = {}
        for text_id, text_hash in connection.execute(
            'SELECT DISTINCT "text_id", "text_hash" FROM "results"'
        ):
            texts.setdefault(text_id, set()).add(text_hash)
    return _Store(info, columns, [sql for (sql,) in rows], texts)


def _check_texts(stores: Sequence[tuple[Path, _Store]]) -> None:
    """Raise if the stores hold more than one text hash under one text id.

    Raises:
        StoreError: Naming the stores and the first ten such ids, in id order.
    """
    seen: dict[str, set[str]] = {}
    for _, store in stores:
        for text_id, hashes in store.texts.items():
            seen.setdefault(text_id, set()).update(hashes)
    conflicting = sorted(text_id for text_id, hashes in seen.items() if len(hashes) > 1)
    if conflicting:
        noun = "text id" if len(conflicting) == 1 else "text ids"
        names = ", ".join(str(path) for path, _ in stores)
        raise StoreError(
            f"{TEXTS_DIFFER} (stores {names}; {len(conflicting)} {noun}: {listed_ids(conflicting)})"
        )


def merge_stores(sources: Sequence[str | Path], target: str | Path) -> dict[str, int]:
    """Fold shard stores into one store.

    Results and evidence are merged per cell, where a cell is one
    ``(text_id, run_number, model_name, temperature)``: a cell that the target
    already holds keeps its existing results row and evidence, and the source's
    rows for it are dropped, so the evidence of two different codings of one cell
    is never mixed. A dropped cell whose results row differs from the kept one is
    a conflict: it is logged as a warning and counted. ``wording``, ``shuffles``,
    and ``run_info`` are merged with ``INSERT OR IGNORE`` (``run_info`` from the
    first source only). Merging the same shards again leaves the target unchanged.
    A store binds every text id to the text it was coded on, so the stores must hold
    one text hash per text id between them: shards coded on the same texts merge, and
    stores coded on different texts under one id are refused. All sources are checked
    before the target is created or written. Each source is checked again, against the
    target, inside the transaction that copies it, on the same snapshot of the source
    the copy reads, so that a source or target changed after the first check cannot
    bring a second text under one id into the target; when this check refuses a source,
    the sources copied before it stay in the target, each of them complete. The target
    is created with the schema of the first source when it does not exist.

    Args:
        sources: Shard stores of one run, a sequence (such as a list) whose items are each
            a ``str`` or a ``Path``; a single path is refused, not read letter by letter.
        target: Store that receives the rows, a ``str`` or a ``Path``; may already hold
            rows of the run.

    Returns:
        The row count of each table of the target after the merge, plus
        ``conflicting_cells``: the number of cells, summed over the sources of
        this call, that were dropped because the target held a different result.

    Raises:
        ConfigurationError: If ``sources`` is a single ``str`` or ``Path`` instead of a
            sequence of them.
        StoreError: If no source is given, a source does not exist, a store was made by
            an earlier version of the library or changed by hand, the target is one of the
            sources, two stores disagree on any ``run_info`` key except ``library_version``
            (which is logged), the tables of two stores have different columns, or the
            sources and the target hold different text hashes under one text id (the first ten such
            ids are named), found before anything is written or, when a store changed
            after the first check, inside the transaction that copies a source, in which
            case the sources copied before it stay in the target.
    """
    if isinstance(sources, str | Path):
        raise ConfigurationError(
            "sources must be a sequence of store paths, such as a list; got the single path "
            f"{str(sources)!r} (a single store needs no merge; to fold it into the target, "
            f"pass [{str(sources)!r}])"
        )
    return _merge_paths([Path(source) for source in sources], Path(target))


def _merge_paths(sources: Sequence[Path], target: Path) -> dict[str, int]:
    """``merge_stores`` on paths: check every source, then fold them into the target."""
    if not sources:
        raise StoreError("merge needs at least one source store")
    resolved_target = target.resolve()
    stores = []
    for source in sources:
        if source.resolve() == resolved_target:
            raise StoreError(f"target {target} is also a source")
        stores.append(_read_store(source))
    target_exists = not is_blank(target)
    reference = _read_store(target) if target_exists else stores[0]
    reference_label = str(target) if target_exists else str(sources[0])
    for source, store in zip(sources, stores, strict=True):
        _check_agreement(reference.info, store.info, str(source))
        if store.columns != reference.columns:
            raise StoreError(f"tables of {source} have other columns than {reference_label}")
    labelled = list(zip(sources, stores, strict=True))
    if target_exists:
        labelled.append((target, reference))
    _check_texts(labelled)

    schema = [] if target_exists else stores[0].schema
    connection = open_read_write(target)
    connection.isolation_level = None
    try:
        conflicts = 0
        for index, source in enumerate(sources):
            statements = schema if index == 0 else []
            columns = stores[index].columns
            conflicts += _merge_one(connection, source, columns, statements, index == 0)
        counts = count_rows(connection)
    finally:
        connection.close()
    logger.info(
        "merged %d stores into %s: %s, %d conflicting cells",
        len(sources),
        target,
        counts,
        conflicts,
    )
    return {**counts, "conflicting_cells": conflicts}


def _merge_one(
    connection: sqlite3.Connection,
    source: Path,
    columns: Mapping[str, list[str]],
    schema: Sequence[str],
    include_run_info: bool,
) -> int:
    """Copy the rows of one source into ``main`` in one transaction.

    Before any row is copied, the transaction checks that the source and the target hold one
    text hash per text id between them, on the snapshot of the source that the copy reads.

    Args:
        connection: Autocommit connection to the target.
        source: The shard to copy from.
        columns: Column names per table.
        schema: CREATE statements to run first, in the same transaction, when the
            target is new; a failed merge then leaves no tables behind.
        include_run_info: Whether to copy ``run_info`` too.

    Returns:
        The number of conflicting cells of this source.

    Raises:
        StoreError: If the source and the target hold different text hashes under one text
            id, or if the rows of a new cell cannot be inserted; the transaction is rolled
            back, so the target keeps none of the rows of this source.
    """
    key = ", ".join(quote(name) for name in SKIP_KEY)
    connection.execute("ATTACH DATABASE ? AS src", (f"{source.resolve().as_uri()}?mode=ro",))
    try:
        with connection:
            connection.execute("BEGIN IMMEDIATE")
            for statement in schema:
                connection.execute(statement)
            _check_copied_texts(connection)
            connection.execute(
                f"CREATE TEMP TABLE new_cells AS SELECT {key} FROM src.results "
                f"EXCEPT SELECT {key} FROM main.results"
            )
            conflicts = _log_conflicts(connection, columns["results"])
            for table in ("results", "evidence"):
                listed = ", ".join(quote(name) for name in columns[table])
                connection.execute(
                    f"INSERT INTO main.{quote(table)} ({listed}) "
                    f"SELECT {listed} FROM src.{quote(table)} "
                    f"WHERE ({key}) IN (SELECT {key} FROM temp.new_cells)"
                )
            connection.execute("DROP TABLE temp.new_cells")
            tables = IGNORE_TABLES + (("run_info",) if include_run_info else ())
            for table in tables:
                listed = ", ".join(quote(name) for name in columns[table])
                connection.execute(
                    f"INSERT OR IGNORE INTO main.{quote(table)} ({listed}) "
                    f"SELECT {listed} FROM src.{quote(table)}"
                )
    except sqlite3.IntegrityError as exc:
        raise StoreError(f"merging {source} failed on a row of a new cell: {exc}") from exc
    finally:
        connection.execute("DETACH DATABASE src")
    return conflicts


def _check_copied_texts(connection: sqlite3.Connection) -> None:
    """Raise if the attached source and the target hold more than one text hash under one id.

    It runs in the transaction that copies the source, before any row is copied. SQLite reads
    the attached source from one snapshot for the whole transaction, and the write lock of the
    transaction keeps other writers off the target, so the check judges the rows the copy
    reads, whatever a writer changed since the first check.

    Raises:
        StoreError: Naming the source and the target by their resolved paths and the first ten
            such ids, in id order, and saying that the stores copied before stay in the target.
    """
    rows = connection.execute(
        'SELECT "text_id" FROM (SELECT "text_id", "text_hash" FROM src."results" '
        'UNION SELECT "text_id", "text_hash" FROM main."results") '
        'GROUP BY "text_id" HAVING COUNT(*) > 1 ORDER BY "text_id"'
    ).fetchall()
    conflicting = [text_id for (text_id,) in rows]
    if conflicting:
        noun = "text id" if len(conflicting) == 1 else "text ids"
        paths = {name: path for _, name, path in connection.execute("PRAGMA database_list")}
        raise StoreError(
            f"{TEXTS_DIFFER} (source {paths['src']}, checked again against the target "
            f"{paths['main']} inside the transaction that copies it; {len(conflicting)} {noun}: "
            f"{listed_ids(conflicting)}). The stores copied before this one stay in the target, "
            "each complete, and merging them again changes nothing, as a merge is idempotent."
        )


def _log_conflicts(connection: sqlite3.Connection, names: Sequence[str]) -> int:
    """Warn about each cell of ``src`` that ``main`` holds with a different coding.

    ``EXCEPT`` compares whole rows with NULLs equal and, unlike a chain of column
    comparisons, does not grow the expression tree with the number of columns.
    """
    key = ", ".join(quote(name) for name in SKIP_KEY)
    skipped = (*SKIP_KEY, *NON_CODING_COLUMNS)
    coding = ", ".join(quote(name) for name in names if name not in skipped)
    rows = connection.execute(
        f"SELECT {key}, {coding} FROM src.results "
        f"WHERE ({key}) IN (SELECT {key} FROM main.results) "
        f"EXCEPT SELECT {key}, {coding} FROM main.results"
    ).fetchall()
    for text_id, run_number, model_name, temperature, *_ in rows:
        logger.warning(
            "kept the stored result of text_id=%r, run_number=%s, model_name=%r, "
            "temperature=%r; the source holds a different one",
            text_id,
            run_number,
            model_name,
            temperature,
        )
    return len(rows)
