"""The run store reads its cells and replaces a cell with a failed attribute in one transaction."""

from __future__ import annotations

import dataclasses
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest
from store_support import N_PASSAGES, SCORES, make_row, open_store, text_of

from copernicus.coders import StoredCell, hash_text
from copernicus.errors import StoreError
from copernicus.rubric.model import Rubric

FAILED_TONE = {**SCORES, "tone": None}


def tone_scores(path: Path) -> list[tuple[str, int | None, str]]:
    with closing(sqlite3.connect(path)) as connection:
        return connection.execute(
            'SELECT "text_id", "tone__score", "tone__status" FROM "results" ORDER BY "text_id"'
        ).fetchall()


def test_the_store_reports_the_text_hash_and_the_statuses_of_every_cell(
    toy_rubric: Rubric, tmp_path: Path
) -> None:
    with open_store(tmp_path / "run.sqlite", toy_rubric) as store:
        store.insert(
            [
                make_row(toy_rubric, "a", 1, SCORES),
                make_row(toy_rubric, "b", 2, FAILED_TONE, text="the text of b"),
            ]
        )
        cells = store.completed_cells()
        assert store.completed_keys() == set(cells) == {("a", 1), ("b", 2)}
    assert cells[("a", 1)] == StoredCell(hash_text(text_of("a")), ("ok", "ok", "ok"))
    assert cells[("b", 2)] == StoredCell(hash_text("the text of b"), ("ok", "scoring_failed", "ok"))
    assert not cells[("a", 1)].failed and cells[("b", 2)].failed


def test_a_stored_status_the_library_does_not_write_is_refused(
    toy_rubric: Rubric, tmp_path: Path
) -> None:
    path = tmp_path / "run.sqlite"
    with open_store(path, toy_rubric) as store:
        store.insert([make_row(toy_rubric, "a", 1, SCORES)])
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute('UPDATE "results" SET "tone__status" = \'pending\'')
    with open_store(path, toy_rubric) as store:
        with pytest.raises(StoreError, match=r"text_id='a', run_number=1: unknown statuses"):
            store.completed_cells()


def test_a_cell_with_a_failed_attribute_is_replaced_with_its_evidence(
    toy_rubric: Rubric, tmp_path: Path
) -> None:
    path = tmp_path / "run.sqlite"
    with open_store(path, toy_rubric) as store:
        store.insert(
            [make_row(toy_rubric, "a", 1, FAILED_TONE, 3), make_row(toy_rubric, "b", 1, SCORES)]
        )
        with pytest.raises(StoreError, match="already stored"):
            store.insert([make_row(toy_rubric, "a", 1, SCORES)])
        store.insert(
            [make_row(toy_rubric, "a", 1, SCORES), make_row(toy_rubric, "c", 1, SCORES)],
            replace_failed=True,
        )
        counts = store.counts()
        cells = store.completed_cells()
    assert counts["results"] == 3 and counts["evidence"] == 3 * len(toy_rubric.attributes)
    assert not any(cell.failed for cell in cells.values())
    assert tone_scores(path) == [("a", 5, "ok"), ("b", 5, "ok"), ("c", 5, "ok")]


def test_a_cell_without_a_failed_attribute_is_never_replaced(
    toy_rubric: Rubric, tmp_path: Path
) -> None:
    path = tmp_path / "run.sqlite"
    with open_store(path, toy_rubric) as store:
        store.insert([make_row(toy_rubric, "a", 1, SCORES, N_PASSAGES)])
        before = store.counts()
        batch = [make_row(toy_rubric, "c", 1, SCORES), make_row(toy_rubric, "a", 1, SCORES)]
        with pytest.raises(StoreError, match="text_id='a', run_number=1.*no failed attribute"):
            store.insert(batch, replace_failed=True)
        assert store.counts() == before


def test_a_replacement_that_fails_keeps_the_stored_cell_and_its_evidence(
    toy_rubric: Rubric, tmp_path: Path
) -> None:
    path = tmp_path / "run.sqlite"
    with open_store(path, toy_rubric) as store:
        store.insert([make_row(toy_rubric, "a", 1, FAILED_TONE, 3)])
        before = store.counts()
        row = make_row(toy_rubric, "a", 1, SCORES)
        no_role = dataclasses.replace(
            row, assignment=dataclasses.replace(row.assignment, role_id=None)
        )
        with pytest.raises(StoreError, match="constraint failed while storing text_id='a'"):
            store.insert([no_role], replace_failed=True)
        assert store.counts() == before
        assert store.completed_cells()[("a", 1)].failed
    assert tone_scores(path) == [("a", None, "scoring_failed")]
