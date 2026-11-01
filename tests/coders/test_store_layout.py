"""A run store with the table layout of an earlier library version is refused, not used."""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Callable
from contextlib import closing
from pathlib import Path

import pytest
from store_support import SCORES, make_row, make_shard, open_store

from copernicus.coders import (
    CodingSettings,
    code_texts,
    export_construct_scores,
    export_evidence,
    export_provenance,
    export_scores,
    export_text_scores,
    merge_stores,
)
from copernicus.data import TextTable
from copernicus.errors import StoreError
from copernicus.llm import FakeBackend
from copernicus.rubric.model import Rubric

EARLIER = (
    "was made by an earlier version of the library, or was changed by hand; code into a new store"
)


def alter(path: Path, statement: str) -> None:
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute(statement)


def legacy_message(path: Path) -> str:
    return re.escape(f"the run store {path} {EARLIER}")


@pytest.mark.parametrize(
    "statement",
    [
        'ALTER TABLE "results" DROP COLUMN "text_hash"',
        'ALTER TABLE "results" RENAME COLUMN "seconds" TO "elapsed"',
        'ALTER TABLE "evidence" DROP COLUMN "match_score"',
        'ALTER TABLE "results" DROP COLUMN "tone__explanation"',
        'ALTER TABLE "results" DROP COLUMN "tone__stage1_temperature_applied"',
    ],
)
def test_a_store_whose_columns_differ_from_the_schema_is_refused(
    toy_rubric: Rubric, tmp_path: Path, statement: str
) -> None:
    path = tmp_path / "old.sqlite"
    with open_store(path, toy_rubric) as store:
        store.insert([make_row(toy_rubric, "a", 1, SCORES)])
    alter(path, statement)
    with pytest.raises(StoreError, match=legacy_message(path)):
        open_store(path, toy_rubric)


def test_a_store_without_the_text_hash_is_refused_by_the_runner_the_merge_and_the_exports(
    toy_rubric: Rubric, toy_corpus: TextTable, cue_backend: Callable[[], FakeBackend],
    tmp_path: Path,
) -> None:  # fmt: skip
    path = tmp_path / "old.sqlite"
    table = toy_corpus.subset(["t001"])
    code_texts(table, toy_rubric, cue_backend(), path, CodingSettings(runs=1))
    alter(path, 'ALTER TABLE "results" DROP COLUMN "text_hash"')
    backend = cue_backend()
    with pytest.raises(StoreError, match=legacy_message(path)):
        code_texts(table, toy_rubric, backend, path, CodingSettings(runs=1))
    assert backend.calls == []
    for export in (export_scores, export_text_scores, export_evidence):
        with pytest.raises(StoreError, match=legacy_message(path)):
            export(path)
    assert export_provenance(path)["model_name"] == "fake"
    target = tmp_path / "merged.sqlite"
    with pytest.raises(StoreError, match=legacy_message(path)):
        merge_stores([path], target)
    assert not target.exists()


@pytest.mark.parametrize(
    "column", ["tone__explanation", "tone__stage1_temperature_applied", "tone_blame__justification"]
)
def test_every_export_and_the_merge_refuse_a_store_that_lacks_a_column_of_its_rubric(
    toy_rubric: Rubric, tmp_path: Path, column: str
) -> None:
    """SQLite reads a missing double-quoted column as a string literal, so it is never read."""
    path = make_shard(toy_rubric, tmp_path / "edited.sqlite", ["a"])
    alter(path, f'ALTER TABLE "results" DROP COLUMN "{column}"')
    for export in (export_scores, export_construct_scores, export_text_scores, export_evidence):
        with pytest.raises(StoreError, match=legacy_message(path)):
            export(path)
    assert export_provenance(path)["rubric_id"] == toy_rubric.rubric_id
    target = tmp_path / "merged.sqlite"
    with pytest.raises(StoreError, match=legacy_message(path)):
        merge_stores([path], target)
    assert not target.exists()


def test_a_file_whose_run_info_lacks_the_ids_of_its_columns_is_no_run_store(
    toy_rubric: Rubric, tmp_path: Path
) -> None:
    path = make_shard(toy_rubric, tmp_path / "edited.sqlite", ["a"])
    alter(path, """DELETE FROM "run_info" WHERE "key" IN ('attribute_ids', 'marker_ids')""")
    for export in (export_scores, export_construct_scores, export_text_scores):
        with pytest.raises(StoreError, match="is not a run store"):
            export(path)
    with pytest.raises(StoreError, match=r"its run_info lacks \['attribute_ids', 'marker_ids'\]"):
        export_scores(path)


def test_a_merge_into_a_target_of_the_earlier_layout_is_refused(
    toy_rubric: Rubric, tmp_path: Path
) -> None:
    target = make_shard(toy_rubric, tmp_path / "target.sqlite", ["a"])
    alter(target, 'ALTER TABLE "results" DROP COLUMN "text_hash"')
    source = make_shard(toy_rubric, tmp_path / "s1.sqlite", ["b"])
    with pytest.raises(StoreError, match=legacy_message(target)):
        merge_stores([source], target)
