import dataclasses
import logging
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest
from store_support import (
    N_PASSAGES,
    SCORES,
    make_identity,
    make_row,
    make_shard,
    open_store,
    table_counts,
)

from copernicus.coders import merge_stores, store_merge
from copernicus.errors import ConfigurationError, StoreError
from copernicus.rubric.model import Rubric

N_ATTRIBUTES = 3
TEXTS_DIFFER = (
    "the stores hold different texts under the same text ids: a store binds every text id to "
    "the text it was coded on, so codings of different texts cannot be merged"
)


def test_merge_is_idempotent_and_refuses_drift(toy_rubric: Rubric, tmp_path: Path) -> None:
    one = make_shard(toy_rubric, tmp_path / "s1.sqlite", ["a", "b"])
    two = make_shard(toy_rubric, tmp_path / "s2.sqlite", ["b", "c"])
    target = tmp_path / "merged.sqlite"
    counts = merge_stores([one, two], target)
    shard = table_counts(one)
    assert counts == {
        "results": 3,
        "evidence": 3 * N_ATTRIBUTES * N_PASSAGES,
        "wording": shard["wording"],
        "shuffles": 0,
        "run_info": shard["run_info"],
        "conflicting_cells": 0,  # cell "b" is coded identically in both shards
    }
    assert merge_stores([one, two], target) == counts
    assert merge_stores([two], target) == counts
    with open_store(target, toy_rubric) as merged:
        assert merged.completed_keys() == {("a", 1), ("b", 1), ("c", 1)}
    drift = make_shard(toy_rubric, tmp_path / "s3.sqlite", ["d"], model_name="model-b")
    other = tmp_path / "other.sqlite"
    with pytest.raises(StoreError, match="run_info mismatch on model_name"):
        merge_stores([one, drift], other)
    assert not other.exists()
    with pytest.raises(StoreError, match="run_info mismatch on model_name"):
        merge_stores([drift], target)
    assert merge_stores([one, two], target) == counts


def test_overlapping_cells_keep_the_first_coding_and_its_evidence(
    toy_rubric: Rubric, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    first, second = tmp_path / "s1.sqlite", tmp_path / "s2.sqlite"
    with open_store(first, toy_rubric) as store:
        store.insert([make_row(toy_rubric, "a", 1, SCORES, 1)])
    with open_store(second, toy_rubric) as store:
        store.insert([make_row(toy_rubric, "a", 1, {**SCORES, "tone": 1}, 3)])
    target = tmp_path / "merged.sqlite"
    with caplog.at_level(logging.WARNING, logger="copernicus.coders.store_merge"):
        counts = merge_stores([first, second], target)
    assert counts["results"] == 1 and counts["conflicting_cells"] == 1
    assert counts["evidence"] == N_ATTRIBUTES * 1
    assert "text_id='a', run_number=1" in caplog.text
    with sqlite3.connect(target) as connection:
        assert connection.execute('SELECT "tone__score" FROM "results"').fetchone() == (5,)
        quotes = connection.execute('SELECT DISTINCT "quote_index" FROM "evidence"').fetchall()
    assert quotes == [(0,)]
    again = merge_stores([first, second, first], target)
    assert again == counts
    assert merge_stores([first], tmp_path / "solo.sqlite")["conflicting_cells"] == 0


def test_shards_that_hold_another_text_under_one_id_are_refused_before_writing(
    toy_rubric: Rubric, tmp_path: Path
) -> None:
    one = make_shard(toy_rubric, tmp_path / "s1.sqlite", ["a", "b"])
    two = make_shard(toy_rubric, tmp_path / "s2.sqlite", ["b", "c"], contents={"b": "a new b"})
    target = tmp_path / "merged.sqlite"
    with pytest.raises(StoreError, match=TEXTS_DIFFER) as refused:
        merge_stores([one, two], target)
    message = str(refused.value)
    assert "1 text id: 'b'" in message and "'a'" not in message and "'c'" not in message
    assert not target.exists()
    same = make_shard(toy_rubric, tmp_path / "s3.sqlite", ["b", "c"])
    assert merge_stores([one, same], target)["results"] == 3


def test_a_source_that_holds_another_text_than_the_target_is_refused(
    toy_rubric: Rubric, tmp_path: Path
) -> None:
    target = tmp_path / "merged.sqlite"
    merge_stores([make_shard(toy_rubric, tmp_path / "s1.sqlite", ["a", "b"])], target)
    before = table_counts(target)
    changed = make_shard(
        toy_rubric, tmp_path / "s2.sqlite", ["a", "d"], contents={"a": "a revised a"}
    )
    with pytest.raises(StoreError, match=TEXTS_DIFFER) as refused:
        merge_stores([changed], target)
    assert "1 text id: 'a'" in str(refused.value) and str(target) in str(refused.value)
    assert table_counts(target) == before


def test_the_refusal_names_the_first_ten_ids_that_hold_two_texts(
    toy_rubric: Rubric, tmp_path: Path
) -> None:
    ids = [f"t{number:02d}" for number in range(12)]
    one = make_shard(toy_rubric, tmp_path / "s1.sqlite", ids)
    two = make_shard(toy_rubric, tmp_path / "s2.sqlite", ids, contents={i: f"new {i}" for i in ids})
    with pytest.raises(StoreError, match=TEXTS_DIFFER) as refused:
        merge_stores([one, two], tmp_path / "m.sqlite")
    message = str(refused.value)
    assert "12 text ids: " in message and "and 2 more" in message
    assert all(f"'{i}'" in message for i in ids[:10])
    assert "'t10'" not in message and "'t11'" not in message


def test_merging_the_same_shard_twice_changes_nothing(toy_rubric: Rubric, tmp_path: Path) -> None:
    one = make_shard(toy_rubric, tmp_path / "s1.sqlite", ["a", "b"])
    target = tmp_path / "merged.sqlite"
    first = merge_stores([one], target)
    assert merge_stores([one], target) == first
    assert first["conflicting_cells"] == 0 and first["results"] == 2


def test_merge_warns_about_library_version_but_not_other_settings(
    toy_rubric: Rubric, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    one = make_shard(toy_rubric, tmp_path / "s1.sqlite", ["a"])
    two = make_shard(toy_rubric, tmp_path / "s2.sqlite", ["b"])
    with sqlite3.connect(two) as connection:
        connection.execute("UPDATE run_info SET value = '9.9.9' WHERE key = 'library_version'")
    with caplog.at_level(logging.WARNING, logger="copernicus.coders.store_merge"):
        assert merge_stores([one, two], tmp_path / "m.sqlite")["results"] == 2
    assert "library_version differs" in caplog.text
    with sqlite3.connect(two) as connection:
        connection.execute("UPDATE run_info SET value = '1' WHERE key = 'min_chars'")
    with pytest.raises(StoreError, match="mismatch on min_chars"):
        merge_stores([one, two], tmp_path / "m2.sqlite")
    assert not (tmp_path / "m2.sqlite").exists()


def test_merge_refuses_other_columns_before_writing(toy_rubric: Rubric, tmp_path: Path) -> None:
    one = make_shard(toy_rubric, tmp_path / "s1.sqlite", ["a"])
    two = make_shard(toy_rubric, tmp_path / "s2.sqlite", ["b"])
    with sqlite3.connect(two) as connection:
        connection.execute('ALTER TABLE "wording" ADD COLUMN "extra" TEXT')
    target = tmp_path / "m.sqlite"
    with pytest.raises(StoreError, match="other columns"):
        merge_stores([one, two], target)
    assert not target.exists()
    # A column that results or evidence lack or add fails the check of the layout first.
    three = make_shard(toy_rubric, tmp_path / "s3.sqlite", ["c"])
    with sqlite3.connect(three) as connection:
        connection.execute('ALTER TABLE "results" ADD COLUMN "extra" TEXT')
    with pytest.raises(StoreError, match="s3.sqlite was made by an earlier version"):
        merge_stores([one, three], target)
    assert not target.exists()


def test_merge_refuses_bad_arguments(toy_rubric: Rubric, tmp_path: Path) -> None:
    one = make_shard(toy_rubric, tmp_path / "s1.sqlite", ["a"])
    with pytest.raises(StoreError, match="also a source"):
        merge_stores([one], tmp_path / "." / "s1.sqlite")
    with pytest.raises(StoreError, match="does not exist"):
        merge_stores([one, tmp_path / "missing.sqlite"], tmp_path / "m.sqlite")
    with pytest.raises(StoreError, match="at least one"):
        merge_stores([], tmp_path / "m.sqlite")
    assert not (tmp_path / "m.sqlite").exists()


def test_merge_refuses_a_single_path_in_place_of_a_sequence(
    toy_rubric: Rubric, tmp_path: Path
) -> None:
    one = make_shard(toy_rubric, tmp_path / "s1.sqlite", ["a"])
    target = tmp_path / "m.sqlite"
    for single in (str(one), one):
        with pytest.raises(ConfigurationError, match="sequence of store paths") as refusal:
            merge_stores(single, target)  # type: ignore[arg-type]
        assert repr(str(one)) in str(refusal.value) and "a single store" in str(refusal.value)
    assert not target.exists()
    assert merge_stores((one,), target)["results"] == 1


def test_merge_into_a_store_of_another_run_is_refused(toy_rubric: Rubric, tmp_path: Path) -> None:
    one = make_shard(toy_rubric, tmp_path / "s1.sqlite", ["a"])
    target = tmp_path / "t.sqlite"
    open_store(target, toy_rubric, make_identity(toy_rubric, model_name="model-b")).close()
    with pytest.raises(StoreError, match="mismatch on model_name"):
        merge_stores([one], target)


def wide_rubric(rubric: Rubric, n_attributes: int) -> Rubric:
    template = rubric.attributes[0]
    marker = template.markers[0]
    attributes = tuple(
        dataclasses.replace(
            template,
            attribute_id=f"attr{i}",
            markers=(dataclasses.replace(marker, marker_id=f"mark{i}"),),
        )
        for i in range(n_attributes)
    )
    return dataclasses.replace(rubric, attributes=attributes)


def test_merge_handles_a_rubric_with_hundreds_of_attributes(
    toy_rubric: Rubric, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    wide = wide_rubric(toy_rubric, 220)
    scores = {a.attribute_id: 3 for a in wide.attributes}
    first, second = tmp_path / "s1.sqlite", tmp_path / "s2.sqlite"
    with open_store(first, wide) as store:
        store.insert([make_row(wide, "a", 1, scores), make_row(wide, "b", 1, scores)])
    with open_store(second, wide) as store:
        store.insert(
            [make_row(wide, "b", 1, {**scores, "attr219": 5}), make_row(wide, "c", 1, scores)]
        )
    with caplog.at_level(logging.WARNING, logger="copernicus.coders.store_merge"):
        counts = merge_stores([first, second], tmp_path / "m.sqlite")
    assert counts["results"] == 3 and counts["conflicting_cells"] == 1
    assert counts["evidence"] == 3 * 220
    assert "text_id='b'" in caplog.text


def test_failed_merge_leaves_no_half_created_target(
    toy_rubric: Rubric, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    one = make_shard(toy_rubric, tmp_path / "s1.sqlite", ["a"])
    target = tmp_path / "m.sqlite"

    def fail(*_: object) -> int:
        raise sqlite3.OperationalError("injected failure")

    with monkeypatch.context() as patch:
        patch.setattr(store_merge, "_log_conflicts", fail)
        with pytest.raises(sqlite3.OperationalError, match="injected"):
            merge_stores([one], target)
    with sqlite3.connect(target) as connection:
        assert connection.execute("SELECT COUNT(*) FROM sqlite_master").fetchone() == (0,)
    assert merge_stores([one], target)["results"] == 1


def test_an_orphan_evidence_row_fails_the_merge_loudly(toy_rubric: Rubric, tmp_path: Path) -> None:
    one = make_shard(toy_rubric, tmp_path / "s1.sqlite", ["a"])
    target = tmp_path / "m.sqlite"
    merge_stores([one], target)
    two = make_shard(toy_rubric, tmp_path / "s2.sqlite", ["b"])
    with closing(sqlite3.connect(two)) as source:
        orphan = source.execute('SELECT * FROM "evidence" LIMIT 1').fetchone()
    with closing(sqlite3.connect(target)) as connection, connection:
        marks = ", ".join("?" * len(orphan))
        connection.execute(f'INSERT INTO "evidence" VALUES ({marks})', orphan)
    with pytest.raises(StoreError, match="failed on a row of a new cell"):
        merge_stores([two], target)
    with closing(sqlite3.connect(target)) as connection:
        assert connection.execute('SELECT COUNT(*) FROM "results"').fetchone() == (1,)


def test_the_store_and_the_merge_take_paths_as_text(toy_rubric: Rubric, tmp_path: Path) -> None:
    one = make_shard(toy_rubric, tmp_path / "s1.sqlite", ["a"])
    two = make_shard(toy_rubric, tmp_path / "s2.sqlite", ["b"])
    target = tmp_path / "merged.sqlite"
    counts = merge_stores([str(one), str(two)], str(target))
    assert counts["results"] == 2 and target.is_file()
    with open_store(str(target), toy_rubric) as merged:
        assert merged.path == target and isinstance(merged.path, Path)
        assert merged.completed_keys() == {("a", 1), ("b", 1)}
