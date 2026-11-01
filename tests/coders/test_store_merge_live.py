"""A merge checks the texts of each store again, on the snapshot of the source that it copies.

A writer may add rows to a source or to the target after the merge has checked the stores and
before it copies them. These tests run such a writer, the public runner, at fixed points of the
merge and read what the target holds afterwards.
"""

from __future__ import annotations

import sqlite3
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from contextlib import closing
from dataclasses import replace
from pathlib import Path

import pytest

from copernicus.coders import CodingSettings, code_texts, export_scores, merge_stores, store_merge
from copernicus.coders.store_schema import TABLES
from copernicus.data import TextTable
from copernicus.errors import StoreError
from copernicus.llm import FakeBackend
from copernicus.rubric.model import Rubric

Coder = Callable[..., None]
Rows = dict[str, Counter[tuple[object, ...]]]
TEXTS_DIFFER = (
    "the stores hold different texts under the same text ids: a store binds every text id to "
    "the text it was coded on, so codings of different texts cannot be merged"
)
IN_THE_COPY = "inside the transaction that copies it"
KEPT = (
    "The stores copied before this one stay in the target, each complete, and merging them "
    "again changes nothing, as a merge is idempotent."
)
NEW_TEXT = "The package arrived on Tuesday near the eastern gate."
SETTINGS = CodingSettings(runs=1, temperature=0.3)


@pytest.fixture
def code(toy_rubric: Rubric, cue_backend: Callable[[], FakeBackend]) -> Coder:
    """Return a function that codes a table into a store through the runner, with ``runs`` runs."""

    def run(table: TextTable, store: Path, runs: int = 1) -> None:
        code_texts(table, toy_rubric, cue_backend(), store, replace(SETTINGS, runs=runs))

    return run


def revised(table: TextTable) -> TextTable:
    """Return the table with another text under each of its ids."""
    frame = table.frame.copy()
    frame["text"] = NEW_TEXT
    return TextTable.from_frame(frame, group_id="group_id")


def write_before_copy(
    monkeypatch: pytest.MonkeyPatch, store: Path, write: Callable[[], None]
) -> None:
    """Make the merge run ``write`` after its first check, just before it copies ``store``."""
    copy = store_merge._merge_one

    def interleaved(
        connection: sqlite3.Connection,
        source: Path,
        columns: Mapping[str, list[str]],
        schema: Sequence[str],
        include_run_info: bool,
    ) -> int:
        if source == store:
            write()
        return copy(connection, source, columns, schema, include_run_info)

    monkeypatch.setattr(store_merge, "_merge_one", interleaved)


def rows_of(path: Path) -> Rows:
    """Return the rows of each table of a store, as a multiset of rows per table."""
    with closing(sqlite3.connect(path)) as connection:
        return {table: Counter(connection.execute(f'SELECT * FROM "{table}"')) for table in TABLES}


def hashes_per_id(path: Path) -> dict[str, int]:
    """Return the number of distinct text hashes that a store holds under each text id."""
    with closing(sqlite3.connect(path)) as connection:
        rows = connection.execute(
            'SELECT "text_id", COUNT(DISTINCT "text_hash") FROM "results" GROUP BY "text_id"'
        ).fetchall()
    return dict(rows)


def cells(path: Path) -> set[tuple[str, int]]:
    """Return the ``(text_id, run_number)`` of every results row of a store."""
    with closing(sqlite3.connect(path)) as connection:
        return set(connection.execute('SELECT "text_id", "run_number" FROM "results"'))


def test_a_source_that_changed_after_the_preflight_is_refused_inside_the_copy(
    toy_corpus: TextTable, code: Coder, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first, second = tmp_path / "first.sqlite", tmp_path / "second.sqlite"
    target = tmp_path / "merged.sqlite"
    code(toy_corpus.subset(["t052"]), first)
    code(toy_corpus.subset(["t053"]), second)
    changed = revised(toy_corpus.subset(["t052"]))
    # Legal for the second source, which holds no t052 when the merge checks it first.
    write_before_copy(monkeypatch, second, lambda: code(changed, second, runs=2))
    with pytest.raises(StoreError, match=TEXTS_DIFFER) as refused:
        merge_stores([first, second], target)
    message = str(refused.value)
    assert str(second) in message and str(target) in message and IN_THE_COPY in message
    assert "1 text id: 't052')" in message and KEPT in message
    assert rows_of(target) == rows_of(first)
    # As the message says: merging the store copied before again changes nothing.
    merge_stores([first], target)
    assert rows_of(target) == rows_of(first)
    assert hashes_per_id(target) == {"t052": 1}
    scores = export_scores(target)
    assert scores.loc[scores["text_id"] == "t052", "text_hash"].nunique() == 1


def test_a_source_that_gained_the_same_text_after_the_preflight_is_copied(
    toy_corpus: TextTable, code: Coder, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first, second = tmp_path / "first.sqlite", tmp_path / "second.sqlite"
    target = tmp_path / "merged.sqlite"
    same = toy_corpus.subset(["t052"])
    code(same, first)
    code(toy_corpus.subset(["t053"]), second)
    write_before_copy(monkeypatch, second, lambda: code(same, second, runs=2))
    counts = merge_stores([first, second], target)
    assert counts["results"] == 3 and counts["conflicting_cells"] == 0
    assert cells(target) == {("t052", 1), ("t052", 2), ("t053", 1)}
    assert hashes_per_id(target) == {"t052": 1, "t053": 1}


def test_a_target_that_changed_after_the_preflight_is_refused_inside_the_copy(
    toy_corpus: TextTable, code: Coder, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Paths relative to the working folder, which the refusal names in their resolved form.
    monkeypatch.chdir(tmp_path)
    source, target = Path("source.sqlite"), Path("merged.sqlite")
    # Two runs, so that run 2 is a new cell for the target and would be copied.
    code(toy_corpus.subset(["t052"]), source, runs=2)
    code(toy_corpus.subset(["t053"]), target)
    before_copy: Rows = {}

    def revise_the_target() -> None:
        # Legal for the target, which holds no t052 when the merge checks it first.
        code(revised(toy_corpus.subset(["t052"])), target)
        before_copy.update(rows_of(target))

    write_before_copy(monkeypatch, source, revise_the_target)
    with pytest.raises(StoreError, match=TEXTS_DIFFER) as refused:
        merge_stores([source], target)
    message = str(refused.value)
    named = f"source {tmp_path / source}, checked again against the target {tmp_path / target} "
    assert named in message and IN_THE_COPY in message and "1 text id: 't052')" in message
    assert rows_of(target) == before_copy
    assert hashes_per_id(target) == {"t052": 1, "t053": 1}


def test_the_copy_reads_the_snapshot_the_check_saw(
    toy_corpus: TextTable, code: Coder, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first, second = tmp_path / "first.sqlite", tmp_path / "second.sqlite"
    target = tmp_path / "merged.sqlite"
    code(toy_corpus.subset(["t052"]), first)
    code(toy_corpus.subset(["t053"]), second)
    merge_stores([first], target)
    held, snapshot = rows_of(target), rows_of(second)
    log_conflicts = store_merge._log_conflicts

    def write_after_the_check(connection: sqlite3.Connection, names: Sequence[str]) -> int:
        # Inside the transaction of the copy: in WAL mode the writer commits while the merge
        # holds its read snapshot of the source.
        code(revised(toy_corpus.subset(["t052"])), second, runs=2)
        return log_conflicts(connection, names)

    with monkeypatch.context() as patch:
        patch.setattr(store_merge, "_log_conflicts", write_after_the_check)
        counts = merge_stores([second], target)
    assert cells(second) == {("t052", 1), ("t052", 2), ("t053", 1)}
    # Read from the source as it is now, the late cell (t052, 1) would be a conflict and the
    # late cell (t052, 2) would bring the rows of a new shuffle seed.
    assert counts["conflicting_cells"] == 0
    merged = rows_of(target)
    for table in ("results", "evidence"):
        assert merged[table] == held[table] + snapshot[table]
    for table in ("wording", "shuffles", "run_info"):
        assert merged[table] == held[table] | snapshot[table]
    assert hashes_per_id(target) == {"t052": 1, "t053": 1}
    with pytest.raises(StoreError, match=TEXTS_DIFFER) as refused:
        merge_stores([second], target)
    assert IN_THE_COPY not in str(refused.value)
    assert rows_of(target) == merged


def test_a_commit_right_after_the_check_stays_out_of_the_copy(
    toy_corpus: TextTable, code: Coder, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first, second = tmp_path / "first.sqlite", tmp_path / "second.sqlite"
    target = tmp_path / "merged.sqlite"
    code(toy_corpus.subset(["t052"]), first)
    code(toy_corpus.subset(["t053"]), second)
    merge_stores([first], target)
    held, snapshot = rows_of(target), rows_of(second)
    check = store_merge._check_copied_texts

    def write_after_the_check(connection: sqlite3.Connection) -> None:
        check(connection)
        # Between the check and the selection of the cells to copy: in WAL mode the writer
        # commits while the merge holds its read snapshot of the source.
        code(revised(toy_corpus.subset(["t052"])), second, runs=2)

    with monkeypatch.context() as patch:
        patch.setattr(store_merge, "_check_copied_texts", write_after_the_check)
        counts = merge_stores([second], target)
    assert cells(second) == {("t052", 1), ("t052", 2), ("t053", 1)}
    # Read from the source as it is now, the late cell (t052, 2) would be selected and copied,
    # with its evidence and shuffles, and the late cell (t052, 1) would be a conflict.
    merged = rows_of(target)
    for table in ("results", "evidence"):
        assert merged[table] == held[table] + snapshot[table]
    for table in ("wording", "shuffles", "run_info"):
        assert merged[table] == held[table] | snapshot[table]
    assert counts["conflicting_cells"] == 0
    assert hashes_per_id(target) == {"t052": 1, "t053": 1}
    with pytest.raises(StoreError, match=TEXTS_DIFFER) as refused:
        merge_stores([second], target)
    assert IN_THE_COPY not in str(refused.value)
    assert rows_of(target) == merged
