"""The rows, transactions and tables of a run store; its identity is in test_store_identity."""

import dataclasses
import json
import sqlite3
from datetime import datetime
from pathlib import Path

import pytest
from store_support import (
    MODEL,
    N_PASSAGES,
    SCORES,
    TEMPERATURE,
    evidence_for,
    make_result,
    make_row,
    open_store,
    text_of,
)

import copernicus
from copernicus.coders import hash_text, load_wording_pool
from copernicus.coders import store as store_module
from copernicus.errors import StoreError
from copernicus.rubric.model import Rubric


def two_rows(rubric: Rubric) -> list:
    return [
        make_row(rubric, "a", 1, SCORES, N_PASSAGES),
        make_row(rubric, "b", 1, SCORES, N_PASSAGES),
    ]


def table_names(path: Path) -> list[str]:
    with sqlite3.connect(path) as connection:
        rows = connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
    return sorted(name for (name,) in rows)


def test_insert_is_one_transaction_and_resume_skips(toy_rubric: Rubric, tmp_path: Path) -> None:
    path = tmp_path / "run.sqlite"
    with open_store(path, toy_rubric) as store:
        store.insert(two_rows(toy_rubric))
    n_evidence = 2 * len(toy_rubric.attributes) * N_PASSAGES
    with open_store(path, toy_rubric) as store:
        assert store.completed_keys() == {("a", 1), ("b", 1)}
        counts = store.counts()
    assert counts == {
        "results": 2,
        "evidence": n_evidence,
        "wording": 0,
        "shuffles": 0,
        "run_info": counts["run_info"],
    }
    assert counts["run_info"] > 20
    with open_store(path, toy_rubric) as store:
        batch = [
            make_row(toy_rubric, "c", 1, SCORES, N_PASSAGES),
            make_row(toy_rubric, "a", 1, SCORES),
        ]
        with pytest.raises(StoreError, match="already stored"):
            store.insert(batch)
        assert store.completed_keys() == {("a", 1), ("b", 1)}
        assert store.counts() == counts


def test_failed_insert_writes_nothing(toy_rubric: Rubric, tmp_path: Path) -> None:
    bad = dataclasses.replace(
        make_row(toy_rubric, "b", 1, SCORES),
        attributes={"tone": make_result(toy_rubric, "tone", 3)},
    )
    with open_store(tmp_path / "run.sqlite", toy_rubric) as store:
        with pytest.raises(StoreError, match="missing.*reasoning"):
            store.insert([make_row(toy_rubric, "a", 1, SCORES), bad])
        assert store.counts()["results"] == 0


def test_duplicate_key_rolls_back_the_whole_batch(toy_rubric: Rubric, tmp_path: Path) -> None:
    with open_store(tmp_path / "run.sqlite", toy_rubric) as store:
        store.insert([make_row(toy_rubric, "a", 1, SCORES)])
        batch = [
            make_row(toy_rubric, "c", 1, SCORES, N_PASSAGES),
            make_row(toy_rubric, "a", 1, SCORES),
        ]
        with pytest.raises(StoreError, match="text_id='a'.*run_number=1"):
            store.insert(batch)
        counts = store.counts()
    assert counts["results"] == 1
    assert counts["evidence"] == len(toy_rubric.attributes)


def test_same_text_in_another_run_is_a_new_key(toy_rubric: Rubric, tmp_path: Path) -> None:
    with open_store(tmp_path / "run.sqlite", toy_rubric) as store:
        store.insert([make_row(toy_rubric, "a", 1, SCORES), make_row(toy_rubric, "a", 2, SCORES)])
        assert store.completed_keys() == {("a", 1), ("a", 2)}


def test_schema_and_pragmas(toy_rubric: Rubric, tmp_path: Path) -> None:
    path = tmp_path / "run.sqlite"
    open_store(path, toy_rubric).close()
    assert table_names(path) == ["evidence", "results", "run_info", "shuffles", "wording"]
    with sqlite3.connect(path) as connection:
        assert connection.execute("PRAGMA journal_mode").fetchone() == ("wal",)
        columns = [r[1] for r in connection.execute('PRAGMA table_info("results")')]
        evidence = [r[1] for r in connection.execute('PRAGMA table_info("evidence")')]
    assert columns[:5] == ["text_id", "text_key", "text_hash", "run_number", "model_name"]
    for name in ("tone__score", "tone__stage2_attempt", "tone_blame__evaluable"):
        assert name in columns
    assert "nope__score" not in columns
    assert columns[15:18] == ["library_version", "temperature_applied", "top_p_applied"]
    assert columns[18:24] == [
        "specificity__score",
        "specificity__status",
        "specificity__explanation",
        "specificity__stage1_attempt",
        "specificity__stage2_attempt",
        "specificity__stage1_temperature_applied",
    ]
    assert len(columns) == 18 + 6 * 3 + 2 * 9
    assert evidence[:6] == [
        "text_id",
        "run_number",
        "model_name",
        "temperature",
        "attribute_id",
        "marker_id",
    ]
    assert len(evidence) == 15


def test_stored_cells(toy_rubric: Rubric, tmp_path: Path) -> None:
    path = tmp_path / "run.sqlite"
    scores = {"specificity": 4, "tone": None, "reasoning": 1}
    with open_store(path, toy_rubric) as store:
        store.insert([make_row(toy_rubric, "a", 3, scores)])
    with sqlite3.connect(path) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute('SELECT * FROM "results"').fetchone()
    assert (row["model_name"], row["temperature"], row["seed"], row["run_number"]) == (
        MODEL,
        TEMPERATURE,
        11,
        3,
    )
    assert (row["role_id"], row["shuffle_seed"]) == ("role_baseline", 7)
    assert row["text_hash"] == hash_text(text_of("a"))
    assert row["library_version"] == copernicus.__version__
    assert row["tone__score"] is None and row["tone__status"] == "scoring_failed"
    assert row["tone__stage2_attempt"] is None and row["specificity__stage2_attempt"] == 0
    assert row["reasoning__status"] == "no_evidence" and row["reasoning__score"] == 1
    assert row["spec_examples__evaluable"] == 1 and row["spec_location__evaluable"] is None
    assert row["spec_examples__justification"] == "raw passage 0"
    assert row["spec_location__justification"] is None
    created = datetime.fromisoformat(row["created_at"])
    assert created.utcoffset().total_seconds() == 0 and created.microsecond == 0  # type: ignore[union-attr]


def test_evaluable_false_is_stored_as_zero(toy_rubric: Rubric, tmp_path: Path) -> None:
    path = tmp_path / "run.sqlite"
    row = make_row(toy_rubric, "a", 1, SCORES)
    result = dataclasses.replace(row.attributes["tone"], evaluable={"tone_respect": False})
    row = dataclasses.replace(row, attributes={**row.attributes, "tone": result})
    with open_store(path, toy_rubric) as store:
        store.insert([row])
    with sqlite3.connect(path) as connection:
        assert connection.execute('SELECT "tone_respect__evaluable" FROM "results"').fetchone() == (
            0,
        )


def test_rows_must_fit_the_rubric(toy_rubric: Rubric, tmp_path: Path) -> None:
    row = make_row(toy_rubric, "a", 1, SCORES)
    wrong_id = dataclasses.replace(row.attributes["tone"], attribute_id="reasoning")
    stray_marker = dataclasses.replace(row.attributes["tone"], evaluable={"nope": True})
    foreign = dataclasses.replace(evidence_for(toy_rubric, "tone", 1)[0], attribute_id="reasoning")
    stray_evidence = dataclasses.replace(row.attributes["tone"], evidence=(foreign,))
    cases = [
        ("belongs to 'reasoning'", wrong_id),
        ("unknown markers \\['nope'\\]", stray_marker),
        ("evidence belongs to another attribute", stray_evidence),
    ]
    with open_store(tmp_path / "run.sqlite", toy_rubric) as store:
        for message, result in cases:
            bad = dataclasses.replace(row, attributes={**row.attributes, "tone": result})
            with pytest.raises(StoreError, match=message):
                store.insert([bad])
        assert store.counts()["results"] == 0


def test_record_wording_and_shuffle_are_idempotent(toy_rubric: Rubric, tmp_path: Path) -> None:
    pool = load_wording_pool()
    with open_store(tmp_path / "run.sqlite", toy_rubric) as store:
        store.record_wording(pool)
        first = store.counts()["wording"]
        store.record_wording(pool)
        store.record_shuffle(7, "tone", ["b", "a"])
        store.record_shuffle(7, "tone", ["a", "b"])
        counts = store.counts()
        with pytest.raises(StoreError, match="unknown attribute 'nope'"):
            store.record_shuffle(7, "nope", [])
    expected = sum(len(w.texts) for d in (pool.role, pool.task, pool.level) for w in d)
    assert first == counts["wording"] == expected
    assert counts["shuffles"] == 1
    with sqlite3.connect(tmp_path / "run.sqlite") as connection:
        order = connection.execute('SELECT "marker_order_json" FROM "shuffles"').fetchone()[0]
    assert json.loads(order) == ["a", "b"]


def test_text_id_must_be_a_string(toy_rubric: Rubric) -> None:
    with pytest.raises(TypeError, match="text_id must be a str, got 5"):
        dataclasses.replace(make_row(toy_rubric, "a", 1, SCORES), text_id=5)  # type: ignore[arg-type]


@pytest.mark.parametrize("digest", ["", "abc", "A" * 64, "g" * 64, hash_text("x") + "0"])
def test_a_row_whose_text_hash_is_no_sha256_hex_digest_is_refused(
    toy_rubric: Rubric, tmp_path: Path, digest: str
) -> None:
    row = dataclasses.replace(make_row(toy_rubric, "a", 1, SCORES), text_hash=digest)
    with open_store(tmp_path / "run.sqlite", toy_rubric) as store:
        with pytest.raises(StoreError, match="text_id='a': text_hash must be 64 lowercase"):
            store.insert([make_row(toy_rubric, "b", 1, SCORES), row])
        assert store.counts()["results"] == 0


def test_other_constraint_failures_are_not_reported_as_duplicates(
    toy_rubric: Rubric, tmp_path: Path
) -> None:
    row = make_row(toy_rubric, "a", 1, SCORES)
    no_role = dataclasses.replace(row, assignment=dataclasses.replace(row.assignment, role_id=None))
    twice = evidence_for(toy_rubric, "tone", 1) * 2
    repeated = dataclasses.replace(row.attributes["tone"], evidence=twice)
    doubled = dataclasses.replace(row, attributes={**row.attributes, "tone": repeated})
    with open_store(tmp_path / "run.sqlite", toy_rubric) as store:
        with pytest.raises(
            StoreError, match="constraint failed while storing text_id='a'.*NOT NULL"
        ):
            store.insert([no_role])  # type: ignore[arg-type]
        with pytest.raises(StoreError, match="constraint failed while storing.*evidence"):
            store.insert([doubled])
        assert store.counts()["results"] == 0


def test_failed_creation_leaves_a_file_that_can_be_created_again(
    toy_rubric: Rubric, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "run.sqlite"
    real = store_module.create_statements
    monkeypatch.setattr(
        store_module,
        "create_statements",
        lambda rubric: [*real(rubric), "SELECT no_such_function()"],
    )
    with pytest.raises(sqlite3.OperationalError, match="no_such_function"):
        open_store(path, toy_rubric)
    monkeypatch.undo()
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM sqlite_master").fetchone() == (0,)
    with open_store(path, toy_rubric) as store:
        store.insert([make_row(toy_rubric, "a", 1, SCORES)])
        assert store.counts()["results"] == 1


def test_close_is_idempotent(toy_rubric: Rubric, tmp_path: Path) -> None:
    store = open_store(tmp_path / "run.sqlite", toy_rubric)
    with store:
        store.close()
    store.close()
