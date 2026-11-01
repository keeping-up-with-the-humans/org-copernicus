"""Tests for the coding runner."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest
from scipy.stats import spearmanr

import copernicus.coders as coders
from copernicus.assets import load_toy_cues
from copernicus.coders import (
    CodingSettings,
    OneStageConfig,
    TwoStageConfig,
    code_texts,
    export_scores,
    export_text_scores,
    resolve_temperature,
    runner_jobs,
)
from copernicus.data import TextTable
from copernicus.errors import (
    PromptContractError,
    RunFailureCeilingError,
    SettingsError,
    StoreError,
    TextTableError,
)
from copernicus.llm import FakeBackend, cue_responder
from copernicus.rubric.model import Rubric

Factory = Callable[[], FakeBackend]
CUES = load_toy_cues()


def test_code_texts_end_to_end_two_runs(
    toy_rubric: Rubric, toy_corpus: TextTable, tmp_path: Path
) -> None:
    store = tmp_path / "run.sqlite"
    backend = FakeBackend(cue_responder(toy_rubric, CUES))
    s = code_texts(toy_corpus, toy_rubric, backend, store, CodingSettings())
    assert s.n_texts == 54 and s.n_runs == 2 and s.n_attribute_calls == 54 * 2 * 3
    assert s.resumed == 0 and s.store_path == store
    assert sum(s.status_counts.values()) == s.n_attribute_calls
    assert s.status_counts["ok"] > 0 and s.status_counts["no_evidence"] > 0
    assert s.status_counts["extraction_failed"] == 0
    assert s.prompt_tokens > 0 and s.completion_tokens > 0 and s.seconds >= 0
    scores = export_text_scores(store)
    assert len(scores) == 54 and scores.construct_score.between(1, 5).all()
    gold = toy_corpus.frame.set_index("text_id").loc[scores.text_id, "gold_constructiveness"]
    assert spearmanr(scores.construct_score, gold.to_numpy()).correlation >= 0.5


def test_stored_run_records_wording_shuffles_and_identity(
    toy_rubric: Rubric, toy_corpus: TextTable, cue_backend: Factory, tmp_path: Path
) -> None:
    store = tmp_path / "run.sqlite"
    subset = toy_corpus.subset(["t001", "t002"])
    code_texts(subset, toy_rubric, cue_backend(), store, CodingSettings(runs=2, seed=7))
    with sqlite3.connect(store) as connection:
        info = dict(connection.execute("SELECT key, value FROM run_info").fetchall())
        orders = connection.execute("SELECT marker_order_json FROM shuffles").fetchall()
    assert info["method"] == "two_stage" and info["seed"] == "7"
    assert info["temperature"] == "0.3" and info["shuffle_markers"] == "True"
    assert info["model_name"] == "fake"
    assert "runs" not in info and "concurrency" not in info
    assert len(orders) == 2 * 2 * 3
    for (order_json,) in orders:
        assert len(json.loads(order_json)) == 3
    scores = export_scores(store)
    assert set(scores.run_number) == {1, 2} and set(scores.seed) == {7}


def test_resume_skips_completed_and_range_shards(
    toy_rubric: Rubric, toy_corpus: TextTable, cue_backend: Factory, tmp_path: Path
) -> None:
    p = tmp_path / "r.sqlite"
    first = code_texts(
        toy_corpus, toy_rubric, cue_backend(), p, CodingSettings(runs=1, text_range=(0, 20))
    )
    assert first.n_texts == 20 and first.resumed == 0
    s2 = code_texts(toy_corpus, toy_rubric, cue_backend(), p, CodingSettings(runs=1))
    assert s2.resumed == 20 and s2.n_texts == 34
    s3 = code_texts(toy_corpus, toy_rubric, cue_backend(), p, CodingSettings(runs=1))
    assert s3.resumed == 54 and s3.n_texts == 0 and s3.n_attribute_calls == 0
    assert len(export_scores(p)) == 54 * 3


def test_resume_may_extend_the_number_of_runs(
    toy_rubric: Rubric, toy_corpus: TextTable, cue_backend: Factory, tmp_path: Path
) -> None:
    p = tmp_path / "r.sqlite"
    subset = toy_corpus.subset(["t001", "t002", "t003"])
    code_texts(subset, toy_rubric, cue_backend(), p, CodingSettings(runs=1, temperature=0.3))
    s = code_texts(subset, toy_rubric, cue_backend(), p, CodingSettings(runs=2, temperature=0.3))
    assert s.resumed == 3 and s.n_texts == 3 and s.n_attribute_calls == 3 * 3
    assert export_scores(p).groupby("run_number").size().tolist() == [9, 9]


def test_extending_the_runs_under_auto_temperature_changes_the_identity(
    toy_rubric: Rubric, toy_corpus: TextTable, cue_backend: Factory, tmp_path: Path
) -> None:
    p = tmp_path / "r.sqlite"
    subset = toy_corpus.subset(["t001"])
    code_texts(subset, toy_rubric, cue_backend(), p, CodingSettings(runs=1))
    with pytest.raises(StoreError, match="pass an explicit temperature") as error:
        code_texts(subset, toy_rubric, cue_backend(), p, CodingSettings(runs=2))
    assert "resolves to 0.0 for one run and 0.3 otherwise" in str(error.value)
    assert isinstance(error.value.__cause__, StoreError)
    with pytest.raises(StoreError) as plain:
        code_texts(subset, toy_rubric, cue_backend(), p, CodingSettings(runs=2, temperature=0.5))
    assert "explicit temperature" not in str(plain.value)


def test_text_range_selects_positions_in_text_id_order(
    toy_rubric: Rubric, toy_corpus: TextTable, cue_backend: Factory, tmp_path: Path
) -> None:
    p = tmp_path / "r.sqlite"
    code_texts(toy_corpus, toy_rubric, cue_backend(), p, CodingSettings(runs=1, text_range=(2, 5)))
    assert sorted(set(export_scores(p).text_id)) == ["t003", "t004", "t005"]


def test_integer_ids_are_stored_as_strings_in_string_order(
    toy_rubric: Rubric, cue_backend: Factory, tmp_path: Path
) -> None:
    frame = pd.DataFrame({"text_id": [10, 9, 100], "text": ["Thank you for the draft today."] * 3})
    table = TextTable.from_frame(frame)
    p = tmp_path / "r.sqlite"
    code_texts(table, toy_rubric, cue_backend(), p, CodingSettings(runs=1, text_range=(0, 2)))
    assert sorted(set(export_scores(p).text_id)) == ["10", "100"]
    s = code_texts(table, toy_rubric, cue_backend(), p, CodingSettings(runs=1))
    assert s.resumed == 2 and s.n_texts == 1


def test_failure_ceiling_raises_after_writing(
    toy_rubric: Rubric, toy_corpus: TextTable, tmp_path: Path
) -> None:
    store = tmp_path / "f.sqlite"
    with pytest.raises(RunFailureCeilingError) as error:
        code_texts(
            toy_corpus.subset(["t001", "t002"]), toy_rubric, FakeBackend(lambda r: ""), store,
            CodingSettings(runs=1),
        )  # fmt: skip
    assert "6 of 6" in str(error.value) and str(store) in str(error.value)
    summary = error.value.summary
    assert (summary.n_texts, summary.n_attribute_calls, summary.store_path) == (2, 6, store)
    assert summary.status_counts["extraction_failed"] == 6
    assert sum(summary.stored_status_counts.values()) == 0
    assert (summary.resumed, summary.retried) == (0, 0)
    scores = export_scores(store)
    assert len(scores) == 2 * 3 and scores.status.eq("extraction_failed").all()


def test_failures_within_the_ceiling_do_not_raise(
    toy_rubric: Rubric, toy_corpus: TextTable, tmp_path: Path
) -> None:
    backend = FakeBackend(cue_responder(toy_rubric, CUES))
    subset = toy_corpus.subset(["t001", "t002"])
    s = code_texts(
        subset, toy_rubric, backend, tmp_path / "ok.sqlite",
        CodingSettings(runs=1, max_failure_share=0.0),
    )  # fmt: skip
    assert s.status_counts["extraction_failed"] == 0


def test_auto_temperature_rule() -> None:
    assert resolve_temperature("auto", 1) == 0.0 and resolve_temperature("auto", 2) == 0.3
    assert resolve_temperature(0.7, 1) == 0.7


def test_temperature_reaches_the_backend_and_the_identity(
    toy_rubric: Rubric, toy_corpus: TextTable, tmp_path: Path
) -> None:
    backend = FakeBackend(cue_responder(toy_rubric, CUES))
    code_texts(
        toy_corpus.subset(["t001"]), toy_rubric, backend, tmp_path / "r.sqlite",
        CodingSettings(runs=1),
    )  # fmt: skip
    assert {call.temperature for call in backend.calls} == {0.0}
    assert {call.seed for call in backend.calls} == {42 + 1}


def test_one_stage_method_runs(toy_rubric: Rubric, toy_corpus: TextTable, tmp_path: Path) -> None:
    store = tmp_path / "o.sqlite"
    backend = FakeBackend(cue_responder(toy_rubric, CUES))
    s = code_texts(
        toy_corpus, toy_rubric, backend, store, CodingSettings(method="one_stage", runs=1)
    )
    assert s.n_texts == 54 and s.n_attribute_calls == 54 * 3 and len(backend.calls) == 54 * 3
    assert export_scores(store).status.isin(["ok"]).all()
    gold = toy_corpus.frame.set_index("text_id").gold_constructiveness
    scores = export_text_scores(store)
    assert spearmanr(scores.construct_score, gold.loc[scores.text_id]).correlation >= 0.5


def test_resume_with_another_method_or_shuffle_setting_is_refused(
    toy_rubric: Rubric, toy_corpus: TextTable, cue_backend: Factory, tmp_path: Path
) -> None:
    p = tmp_path / "r.sqlite"
    subset = toy_corpus.subset(["t001"])
    code_texts(subset, toy_rubric, cue_backend(), p, CodingSettings(runs=1))
    with pytest.raises(StoreError):
        code_texts(subset, toy_rubric, cue_backend(), p, CodingSettings(runs=1, method="one_stage"))
    with pytest.raises(StoreError):
        code_texts(
            subset, toy_rubric, cue_backend(), p, CodingSettings(runs=1, shuffle_markers=False)
        )
    with pytest.raises(StoreError):
        code_texts(
            subset, toy_rubric, cue_backend(), p,
            CodingSettings(runs=1, two_stage=TwoStageConfig(min_chars=30)),
        )  # fmt: skip


def test_wording_selection_pins_the_rotation(
    toy_rubric: Rubric, toy_corpus: TextTable, cue_backend: Factory, tmp_path: Path
) -> None:
    p = tmp_path / "r.sqlite"
    pinned = ["role_baseline", "task_baseline", "level_independent"]
    code_texts(
        toy_corpus.subset(["t001", "t002", "t003"]), toy_rubric, cue_backend(), p,
        CodingSettings(runs=1, wording_enable=pinned),
    )  # fmt: skip
    scores = export_scores(p)
    assert set(scores.role_id) == {"role_baseline"} and set(scores.level_id) == {
        "level_independent"
    }
    with sqlite3.connect(p) as connection:
        stored = connection.execute("SELECT DISTINCT dimension, variant_id FROM wording").fetchall()
    assert sorted(stored) == [
        ("level", "level_independent"),
        ("role", "role_baseline"),
        ("task", "task_baseline"),
    ]


def test_ids_that_collide_as_strings_are_refused(
    toy_rubric: Rubric, cue_backend: Factory, tmp_path: Path
) -> None:
    frame = pd.DataFrame({"text_id": [1, "1"], "text": ["Thank you for the draft today."] * 2})
    with pytest.raises(TextTableError, match="distinct"):
        code_texts(
            TextTable.from_frame(frame), toy_rubric, cue_backend(), tmp_path / "r.sqlite",
            CodingSettings(runs=1),
        )  # fmt: skip


def test_unknown_wording_variant_is_refused(
    toy_rubric: Rubric, toy_corpus: TextTable, cue_backend: Factory, tmp_path: Path
) -> None:
    with pytest.raises(PromptContractError, match="nope"):
        code_texts(
            toy_corpus, toy_rubric, cue_backend(), tmp_path / "r.sqlite",
            CodingSettings(wording_enable=["nope"]),
        )  # fmt: skip


@pytest.mark.parametrize(
    "overrides, field",
    [
        ({"method": "three_stage"}, "method"),
        ({"runs": 0}, "runs"),
        ({"temperature": 2.5}, "temperature"),
        ({"temperature": -0.1}, "temperature"),
        ({"temperature": "warm"}, "temperature"),
        ({"concurrency": 0}, "concurrency"),
        ({"save_every": 0}, "save_every"),
        ({"max_failure_share": 1.5}, "max_failure_share"),
        ({"max_failure_share": -0.1}, "max_failure_share"),
        ({"text_range": (-1, 3)}, "text_range"),
        ({"text_range": (4, 3)}, "text_range"),
    ],
)
def test_settings_are_validated(overrides: dict[str, object], field: str) -> None:
    with pytest.raises(SettingsError, match=field) as error:
        CodingSettings(**overrides)  # type: ignore[arg-type]
    assert isinstance(error.value, ValueError)


def test_default_settings_hold_the_default_configs() -> None:
    settings = CodingSettings()
    assert settings.two_stage == TwoStageConfig() and settings.one_stage == OneStageConfig()
    assert settings.runs == 2 and settings.method == "two_stage" and settings.temperature == "auto"


def test_code_texts_takes_the_store_path_as_text(
    toy_rubric: Rubric, toy_corpus: TextTable, cue_backend: Factory, tmp_path: Path
) -> None:
    store = tmp_path / "run.sqlite"
    subset = toy_corpus.subset(["t001", "t002"])
    first = code_texts(subset, toy_rubric, cue_backend(), str(store), CodingSettings(runs=1))
    assert first.store_path == store and isinstance(first.store_path, Path)
    assert first.n_texts == 2 and store.is_file()
    again = code_texts(subset, toy_rubric, cue_backend(), str(store), CodingSettings(runs=1))
    assert again.resumed == 2 and again.n_texts == 0


def test_the_cell_planner_is_public_in_the_coders_package() -> None:
    assert coders.plan_cell is runner_jobs.plan_cell
    assert coders.CellPlan is runner_jobs.CellPlan
    assert {"plan_cell", "CellPlan"} <= set(coders.__all__)
