import math
import sqlite3
from pathlib import Path

import pandas as pd
import pytest
from store_support import MODEL, TEMPERATURE, make_row, open_store, text_of

import copernicus.coders.results as results_module
from copernicus.coders import (
    construct_scores,
    export_construct_scores,
    export_evidence,
    export_provenance,
    export_scores,
    export_text_scores,
    hash_text,
)
from copernicus.coders.store_schema import is_blank, read_run_info
from copernicus.errors import StoreError
from copernicus.rubric.model import Rubric

ATTRIBUTES = ["specificity", "tone", "reasoning"]
SCORE_COLUMNS = [
    "text_id", "run_number", "attribute_id", "score", "status", "explanation", "stage1_attempt",
    "stage2_attempt", "stage1_temperature_applied", "role_id", "task_id", "level_id",
    "model_name", "temperature", "seed", "shuffle_seed", "text_key", "text_hash",
    "library_version", "temperature_applied", "top_p_applied",
]  # fmt: skip


@pytest.fixture
def store_path(toy_rubric: Rubric, tmp_path: Path) -> Path:
    """A store with text t10 (two runs, one failed cell in run 2) and text t2 (one run)."""
    path = tmp_path / "run.sqlite"
    rows = [
        make_row(toy_rubric, "t2", 1, {"specificity": 3, "tone": 3, "reasoning": 3}),
        make_row(toy_rubric, "t10", 2, {"specificity": 5, "tone": None, "reasoning": 4}, 2),
        make_row(toy_rubric, "t10", 1, {"specificity": 4, "tone": 5, "reasoning": 1}, 0),
    ]
    with open_store(path, toy_rubric) as store:
        store.insert(rows)
    return path


def test_exports_shapes(store_path: Path) -> None:
    scores = export_scores(store_path)
    assert list(scores.columns) == SCORE_COLUMNS
    assert len(scores) == 9
    assert list(scores["text_id"].iloc[:6]) == ["t10"] * 6
    assert list(scores["attribute_id"].iloc[:3]) == ATTRIBUTES
    assert list(scores["run_number"].iloc[:6]) == [1, 1, 1, 2, 2, 2]

    construct = export_construct_scores(store_path).set_index(["text_id", "run_number"])
    # a no_evidence attribute (score 1) counts like any numeric score
    assert construct.loc[("t10", 1), "construct_score"] == pytest.approx((4 + 5 + 1) / 3)
    assert construct.loc[("t10", 1), "n_attributes_scored"] == 3
    # a scoring_failed attribute is excluded and reflected in n_attributes_scored
    assert construct.loc[("t10", 2), "construct_score"] == pytest.approx((5 + 4) / 2)
    assert construct.loc[("t10", 2), "n_attributes_scored"] == 2
    assert construct.loc[("t2", 1), "construct_score"] == 3

    texts = export_text_scores(store_path).set_index("text_id")
    assert list(texts.columns) == ["construct_score", *ATTRIBUTES, "n_runs"]
    assert list(texts.index) == ["t10", "t2"]
    assert texts.loc["t10", "n_runs"] == 2 and texts.loc["t2", "n_runs"] == 1
    assert texts.loc["t10", "specificity"] == pytest.approx(4.5)
    assert texts.loc["t10", "tone"] == 5  # the failed cell of run 2 is excluded
    assert texts.loc["t10", "reasoning"] == pytest.approx(2.5)
    assert texts.loc["t10", "construct_score"] == pytest.approx(((4 + 5 + 1) / 3 + 4.5) / 2)


def test_export_scores_dtypes_and_missing(store_path: Path) -> None:
    scores = export_scores(store_path)
    assert str(scores["score"].dtype) == "Int64"
    assert str(scores["stage1_attempt"].dtype) == "Int64"
    assert str(scores["stage2_attempt"].dtype) == "Int64"
    assert scores["run_number"].dtype == "int64" and scores["seed"].dtype == "int64"
    assert scores["temperature"].dtype == "float64"
    failed = scores[scores["status"] == "scoring_failed"].iloc[0]
    assert pd.isna(failed["score"]) and pd.isna(failed["stage2_attempt"])
    assert failed["stage1_attempt"] == 0
    first = scores.iloc[0]
    assert (first["model_name"], first["temperature"], first["seed"]) == (MODEL, TEMPERATURE, 11)
    assert (
        first["role_id"] == "role_baseline" and first["explanation"] == "explanation of specificity"
    )
    assert scores["shuffle_seed"].dtype == "int64" and scores["text_key"].dtype == "int64"
    assert (first["shuffle_seed"], first["text_key"]) == (7, len("t10"))
    assert scores["text_hash"].dtype == object
    hashes = {text_id: hash_text(text_of(text_id)) for text_id in ("t10", "t2")}
    assert scores["text_hash"].tolist() == [hashes[text_id] for text_id in scores["text_id"]]
    assert scores["temperature_applied"].dtype == "float64"
    assert scores["top_p_applied"].dtype == "float64"
    assert scores["stage1_temperature_applied"].dtype == "float64"
    applied = ["temperature_applied", "top_p_applied", "stage1_temperature_applied"]
    assert scores[applied].isna().all().all()


def test_construct_score_is_missing_when_no_attribute_scored(
    toy_rubric: Rubric, tmp_path: Path
) -> None:
    path = tmp_path / "run.sqlite"
    with open_store(path, toy_rubric) as store:
        store.insert([make_row(toy_rubric, "x", 1, dict.fromkeys(ATTRIBUTES))])
    construct = export_construct_scores(path)
    assert math.isnan(construct.loc[0, "construct_score"])
    assert construct.loc[0, "n_attributes_scored"] == 0
    assert math.isnan(export_text_scores(path).loc[0, "construct_score"])


def test_empty_store_exports_empty_typed_tables(toy_rubric: Rubric, tmp_path: Path) -> None:
    path = tmp_path / "run.sqlite"
    open_store(path, toy_rubric).close()
    scores = export_scores(path)
    assert list(scores.columns) == SCORE_COLUMNS and scores.empty
    assert str(scores["score"].dtype) == "Int64"
    assert export_construct_scores(path).empty
    texts = export_text_scores(path)
    assert list(texts.columns) == ["text_id", "construct_score", *ATTRIBUTES, "n_runs"]
    assert texts.empty
    assert export_evidence(path).empty


def test_export_evidence(store_path: Path) -> None:
    evidence = export_evidence(store_path)
    assert len(evidence) == 3 * 1 + 3 * 2 + 0
    assert list(evidence.columns[:6]) == [
        "text_id", "run_number", "model_name", "temperature", "attribute_id", "marker_id",
    ]  # fmt: skip
    assert len(evidence.columns) == 15
    assert (
        str(evidence["char_start"].dtype) == "Int64"
        and str(evidence["duplicate_of"].dtype) == "Int64"
    )
    assert evidence["verified"].dtype == "int64" and evidence["match_score"].dtype == "float64"
    assert evidence["quote_index"].dtype == "int64"
    t2 = evidence[evidence["text_id"] == "t2"]
    assert list(t2["attribute_id"]) == ATTRIBUTES  # rubric order, not alphabetical
    second = evidence[(evidence["text_id"] == "t10") & (evidence["run_number"] == 2)]
    assert list(second["quote_index"]) == [0, 1] * 3
    assert pd.isna(second["char_start"].iloc[0]) and second["char_start"].iloc[1] == 10


def test_export_provenance(store_path: Path) -> None:
    info = export_provenance(store_path)
    assert info["method"] == "two_stage" and info["model_name"] == MODEL
    assert info["temperature"] == "0.3" and info["seed"] == "11"
    assert info["rubric_id"] == "constructive_feedback_v1"
    assert info["attribute_ids"] == '["specificity", "tone", "reasoning"]'
    assert info["verify_threshold"] == "88.0" and "library_version" in info


def test_exports_work_while_a_run_is_writing(toy_rubric: Rubric, tmp_path: Path) -> None:
    path = tmp_path / "run.sqlite"
    with open_store(path, toy_rubric) as store:
        store.insert([make_row(toy_rubric, "a", 1, dict.fromkeys(ATTRIBUTES, 2))])
        assert len(export_scores(path)) == 3
        store.insert([make_row(toy_rubric, "b", 1, dict.fromkeys(ATTRIBUTES, 2))])
        assert len(export_scores(path)) == 6


def test_exports_reject_missing_and_foreign_files(tmp_path: Path) -> None:
    with pytest.raises(StoreError, match="store does not exist"):
        export_scores(tmp_path / "missing.sqlite")
    other = tmp_path / "other.sqlite"
    with sqlite3.connect(other) as connection:
        connection.execute("CREATE TABLE t (x INTEGER)")
    with pytest.raises(StoreError, match="not a run store"):
        export_provenance(other)
    text_file = tmp_path / "texts.csv"
    text_file.write_text("text_id,text\nt1,hello\n", encoding="utf-8")
    with pytest.raises(StoreError, match="not a run store") as raised:
        export_provenance(text_file)
    assert str(text_file.resolve()) in str(raised.value)
    with pytest.raises(StoreError, match="not a run store"):
        is_blank(text_file)


class LockedConnection:
    """A connection stand-in whose statements fail as if another writer held the lock."""

    def execute(self, statement: str) -> None:
        raise sqlite3.OperationalError("database is locked")


def test_a_locked_database_is_not_reported_as_a_foreign_file() -> None:
    with pytest.raises(sqlite3.OperationalError, match="database is locked"):
        read_run_info(LockedConnection())  # type: ignore[arg-type]


def test_export_scores_reads_one_snapshot_with_one_select(
    store_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    statements: list[str] = []
    real_open = results_module.open_read_only

    def traced(path: Path) -> sqlite3.Connection:
        connection = real_open(path)
        connection.set_trace_callback(statements.append)
        return connection

    monkeypatch.setattr(results_module, "open_read_only", traced)
    export_scores(store_path)
    selects = [
        s for s in statements if s.lstrip().upper().startswith("SELECT") and '"results"' in s
    ]
    assert len(selects) == 1
    statements.clear()
    export_text_scores(store_path)
    reads = [s for s in statements if 'FROM "results"' in s]
    assert len(reads) == 1


def test_construct_scores_is_the_mean_of_the_numeric_scores_per_text_and_run():
    numeric = pd.DataFrame(
        {
            "text_id": ["a", "a", "a", "b", "b"],
            "run_number": [1, 1, 1, 1, 1],
            "attribute_id": ["x", "y", "z", "x", "y"],
            "value": [1.0, 3.0, float("nan"), float("nan"), float("nan")],
        }
    )
    result = construct_scores(numeric).set_index("text_id")
    assert result.loc["a", "construct_score"] == 2.0
    assert result.loc["a", "n_attributes_scored"] == 2
    assert math.isnan(result.loc["b", "construct_score"])
    assert result.loc["b", "n_attributes_scored"] == 0


def test_the_exports_take_the_store_path_as_text(store_path: Path) -> None:
    text = str(store_path)
    pd.testing.assert_frame_equal(export_scores(text), export_scores(store_path))
    pd.testing.assert_frame_equal(
        export_construct_scores(text), export_construct_scores(store_path)
    )
    pd.testing.assert_frame_equal(export_text_scores(text), export_text_scores(store_path))
    pd.testing.assert_frame_equal(export_evidence(text), export_evidence(store_path))
    assert export_provenance(text) == export_provenance(store_path)
