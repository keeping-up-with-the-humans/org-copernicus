"""Tests for the sampling a backend applied, from the coders through the store to the exports."""

from __future__ import annotations

import dataclasses
import json
import sqlite3
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from store_support import SCORES, make_row, open_store
from two_stage_support import (
    DOC,
    OK_SCORE,
    CountingBackend,
    evidence_json,
    make_ctx,
    names_of,
    no_evidence_json,
)

from copernicus.assets import load_toy_cues
from copernicus.coders import (
    AttributeResult,
    CodingSettings,
    OneStageConfig,
    TwoStageConfig,
    code_texts,
    evaluate_attribute,
    evaluate_attribute_one_stage,
    export_provenance,
    export_scores,
    stage1_attempt_temperature,
)
from copernicus.data import TextTable
from copernicus.errors import StoreError
from copernicus.llm import (
    FakeBackend,
    GenerationRequest,
    GenerationResult,
    OpenAICompatibleBackend,
    ScriptedResponder,
    cue_responder,
)
from copernicus.llm.fake import STAGE_ONE_MARKER
from copernicus.rubric.model import Rubric

ONE_STAGE_OK = json.dumps({"dimension_result": {"result": 3, "explanation": "fits level 3"}})
UNPARSEABLE = ["not json", "still not json", "never json"]


def two_stage(backend: FakeBackend, rubric: Rubric, text: str = DOC) -> AttributeResult:
    return evaluate_attribute(
        backend,
        text,
        make_ctx(rubric),
        TwoStageConfig(),
        temperature_stage1=0.5,
        temperature_stage2=0.2,
        seed=1,
    )


def one_stage(backend: FakeBackend, rubric: Rubric, temperature: float, text: str = DOC) -> Any:
    return evaluate_attribute_one_stage(
        backend, text, make_ctx(rubric), OneStageConfig(), temperature=temperature, seed=1
    )


def http_backend(model: str, rubric: Rubric, **options: Any) -> OpenAICompatibleBackend:
    """An HTTP backend whose stub endpoint answers with the cue responder of the toy rubric."""
    responder = cue_responder(rubric, load_toy_cues())

    def create(**kwargs: Any) -> SimpleNamespace:
        system, user = (message["content"] for message in kwargs["messages"])
        content = responder(GenerationRequest(system, user, 1, None, None))
        message = SimpleNamespace(content=content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=None)

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return OpenAICompatibleBackend(model, client=client, **options)


def with_applied(result: AttributeResult, temperature: float | None, top_p: float | None) -> Any:
    return dataclasses.replace(result, temperature_applied=temperature, top_p_applied=top_p)


def stored_applied(path: Path) -> list[tuple[str, float | None, float | None]]:
    with sqlite3.connect(path) as connection:
        return connection.execute(
            'SELECT "text_id", "temperature_applied", "top_p_applied" FROM "results" ORDER BY 1'
        ).fetchall()


def test_two_stage_result_carries_the_sampling_of_the_scoring_call(toy_rubric: Rubric) -> None:
    names = names_of(toy_rubric)
    ok = two_stage(FakeBackend(ScriptedResponder([evidence_json(names), OK_SCORE])), toy_rubric)
    assert (ok.status, ok.temperature_applied, ok.top_p_applied) == ("ok", 0.2, 0.9)
    failed_backend = FakeBackend(ScriptedResponder([evidence_json(names), *UNPARSEABLE]))
    failed = two_stage(failed_backend, toy_rubric)
    assert (failed.status, failed.temperature_applied, failed.top_p_applied) == (
        "scoring_failed",
        0.2,
        0.9,
    )


@pytest.mark.parametrize(
    ("answers", "text", "status"),
    [
        ("no_evidence", DOC, "no_evidence"),
        ("unparseable", DOC, "extraction_failed"),
        ("none", "too short", "too_short"),
    ],
)
def test_two_stage_result_without_a_scoring_call_has_no_applied_sampling(
    toy_rubric: Rubric, answers: str, text: str, status: str
) -> None:
    scripts = {
        "no_evidence": [no_evidence_json(names_of(toy_rubric))],
        "unparseable": UNPARSEABLE,
        "none": [],
    }
    result = two_stage(FakeBackend(ScriptedResponder(scripts[answers])), toy_rubric, text)
    assert (result.status, result.temperature_applied, result.top_p_applied) == (
        status,
        None,
        None,
    )


class HalvingBackend(FakeBackend):
    """A fake backend that applies, and reports, half the temperature a request asks for."""

    def generate(self, request: GenerationRequest) -> GenerationResult:
        generated = super().generate(request)
        applied = None if request.temperature is None else request.temperature / 2
        return dataclasses.replace(generated, temperature_applied=applied)


def test_the_accepted_extraction_attempt_records_the_temperature_the_backend_reports(
    toy_rubric: Rubric,
) -> None:
    names = names_of(toy_rubric)
    answers = ["not json", evidence_json(names), OK_SCORE]
    raised = stage1_attempt_temperature(0.5, 1, 0.3, 1.0)
    retried = two_stage(FakeBackend(ScriptedResponder(answers)), toy_rubric)
    assert (retried.status, retried.stage1_attempt, raised) == ("ok", 1, 0.8)
    assert (retried.stage1_temperature_applied, retried.temperature_applied) == (raised, 0.2)
    halved = two_stage(HalvingBackend(ScriptedResponder(answers)), toy_rubric)
    assert (halved.stage1_attempt, halved.stage1_temperature_applied) == (1, raised / 2)
    first = two_stage(FakeBackend(ScriptedResponder([no_evidence_json(names)])), toy_rubric)
    assert (first.status, first.stage1_attempt, first.stage1_temperature_applied) == (
        "no_evidence",
        0,
        0.5,
    )
    silent = two_stage(CountingBackend([evidence_json(names), OK_SCORE]), toy_rubric)
    assert (silent.stage1_attempt, silent.stage1_temperature_applied) == (0, None)


@pytest.mark.parametrize(
    ("answers", "text", "status"),
    [(UNPARSEABLE, DOC, "extraction_failed"), ([], "too short", "too_short")],
)
def test_no_extraction_temperature_is_recorded_without_an_accepted_attempt(
    toy_rubric: Rubric, answers: list[str], text: str, status: str
) -> None:
    result = two_stage(FakeBackend(ScriptedResponder(answers)), toy_rubric, text)
    assert (result.status, result.stage1_attempt, result.stage1_temperature_applied) == (
        status,
        None,
        None,
    )


def test_a_run_stores_and_exports_the_temperature_of_each_accepted_extraction_attempt(
    toy_rubric: Rubric, toy_corpus: TextTable, tmp_path: Path
) -> None:
    cues = cue_responder(toy_rubric, load_toy_cues())

    def first_extraction_fails(request: GenerationRequest) -> str:
        if STAGE_ONE_MARKER in request.system and request.temperature == 0.3:
            return "not json"
        return cues(request)

    store = tmp_path / "retried.sqlite"
    subset = toy_corpus.subset(["t001", "t050"])
    code_texts(subset, toy_rubric, FakeBackend(first_extraction_fails), store, CodingSettings())
    raised = stage1_attempt_temperature(0.3, 1, 0.3, 1.0)
    with sqlite3.connect(store) as connection:
        stored = connection.execute('SELECT "tone__stage1_temperature_applied" FROM "results"')
        assert stored.fetchall() == [(raised,)] * 4
    scores = export_scores(store)
    assert scores.stage1_attempt.eq(1).all() and scores.stage1_temperature_applied.eq(raised).all()
    scored = scores[scores.status == "ok"]
    assert not scored.empty and scored.temperature_applied.eq(0.3).all()


def test_the_one_stage_coder_records_no_extraction_temperature(
    toy_rubric: Rubric, toy_corpus: TextTable, tmp_path: Path
) -> None:
    result = one_stage(FakeBackend(ScriptedResponder([ONE_STAGE_OK])), toy_rubric, 0.4)
    assert (result.stage1_attempt, result.stage1_temperature_applied) == (None, None)
    assert result.temperature_applied == 0.4
    store = tmp_path / "one_stage.sqlite"
    backend = FakeBackend(cue_responder(toy_rubric, load_toy_cues()))
    settings = CodingSettings(runs=1, method="one_stage")
    code_texts(toy_corpus.subset(["t001", "t050"]), toy_rubric, backend, store, settings)
    scores = export_scores(store)
    assert scores.stage1_temperature_applied.isna().all()
    assert scores.temperature_applied.eq(0.0).all()


def test_one_stage_result_carries_the_sampling_of_its_call(toy_rubric: Rubric) -> None:
    ok = one_stage(FakeBackend(ScriptedResponder([ONE_STAGE_OK])), toy_rubric, 0.4)
    assert (ok.status, ok.temperature_applied, ok.top_p_applied) == ("ok", 0.4, 0.9)
    greedy = one_stage(FakeBackend(ScriptedResponder([ONE_STAGE_OK])), toy_rubric, 0.0)
    assert (greedy.temperature_applied, greedy.top_p_applied) == (0.0, None)
    failed = one_stage(FakeBackend(ScriptedResponder(UNPARSEABLE)), toy_rubric, 0.4)
    assert (failed.status, failed.temperature_applied, failed.top_p_applied) == (
        "scoring_failed",
        0.4,
        0.9,
    )
    short = one_stage(FakeBackend(ScriptedResponder([])), toy_rubric, 0.4, text="too short")
    assert (short.status, short.temperature_applied, short.top_p_applied) == (
        "too_short",
        None,
        None,
    )


def test_store_keeps_the_applied_sampling_of_each_row(toy_rubric: Rubric, tmp_path: Path) -> None:
    row = make_row(toy_rubric, "a", 1, SCORES)
    results: Mapping[str, AttributeResult] = {
        attribute_id: with_applied(result, 0.3, 0.9) if attribute_id != "tone" else result
        for attribute_id, result in row.attributes.items()
    }
    mixed = dataclasses.replace(row, attributes=results)
    silent = make_row(toy_rubric, "b", 1, SCORES)
    path = tmp_path / "run.sqlite"
    with open_store(path, toy_rubric) as store:
        store.insert([mixed, silent])
    assert stored_applied(path) == [("a", 0.3, 0.9), ("b", None, None)]
    scores = export_scores(path).set_index(["text_id", "attribute_id"])
    assert scores.loc[("a", "tone"), "temperature_applied"] == 0.3
    assert scores.loc[("a", "tone"), "top_p_applied"] == 0.9
    assert scores.loc[("b", "tone"), ["temperature_applied", "top_p_applied"]].isna().all()


def test_store_refuses_a_row_whose_attributes_disagree_on_the_sampling(
    toy_rubric: Rubric, tmp_path: Path
) -> None:
    row = make_row(toy_rubric, "a", 1, SCORES)
    results = dict(row.attributes)
    results["tone"] = with_applied(results["tone"], 0.3, 0.9)
    results["reasoning"] = with_applied(results["reasoning"], 0.3, None)
    with open_store(tmp_path / "run.sqlite", toy_rubric) as store:
        with pytest.raises(StoreError, match="text_id='a'.*applied sampling"):
            store.insert([dataclasses.replace(row, attributes=results)])
        assert store.counts()["results"] == 0


def test_a_reasoning_model_run_records_that_no_temperature_was_applied(
    toy_rubric: Rubric, toy_corpus: TextTable, tmp_path: Path
) -> None:
    store = tmp_path / "reasoning.sqlite"
    subset = toy_corpus.subset(["t001", "t050"])
    code_texts(subset, toy_rubric, http_backend("gpt-5-mini", toy_rubric), store, CodingSettings())
    info = export_provenance(store)
    assert info["temperature"] == "0.3" and info["backend_kind"] == "openai_compatible"
    assert info["backend_supports_temperature"] == "False"
    assert info["backend_endpoint"] == "https://api.openai.com"
    scores = export_scores(store)
    assert scores.status.eq("ok").any()
    assert scores.temperature_applied.isna().all() and scores.top_p_applied.isna().all()
    assert scores.stage1_attempt.eq(0).all() and scores.stage1_temperature_applied.isna().all()


def test_a_sampling_model_run_records_the_applied_temperature(
    toy_rubric: Rubric, toy_corpus: TextTable, tmp_path: Path
) -> None:
    store = tmp_path / "sampling.sqlite"
    backend = http_backend("gpt-4.1-mini", toy_rubric)
    code_texts(toy_corpus.subset(["t050"]), toy_rubric, backend, store, CodingSettings())
    scores = export_scores(store)
    assert scores.temperature_applied.eq(0.3).all() and scores.top_p_applied.eq(0.9).all()
    assert scores.stage1_temperature_applied.eq(0.3).all()
    assert export_provenance(store)["backend_supports_temperature"] == "True"


def test_resume_with_a_backend_that_generates_differently_is_refused(
    toy_rubric: Rubric, toy_corpus: TextTable, tmp_path: Path
) -> None:
    store = tmp_path / "run.sqlite"
    subset = toy_corpus.subset(["t001"])
    settings = CodingSettings(runs=1)
    code_texts(subset, toy_rubric, http_backend("gpt-4.1-mini", toy_rubric), store, settings)
    thinking_off = http_backend(
        "gpt-4.1-mini", toy_rubric, extra_body={"chat_template_kwargs": {"enable_thinking": False}}
    )
    with pytest.raises(StoreError, match="mismatch on backend_extra_body"):
        code_texts(subset, toy_rubric, thinking_off, store, settings)
    elsewhere = http_backend("gpt-4.1-mini", toy_rubric, base_url="http://localhost:8000/v1")
    with pytest.raises(StoreError, match="mismatch on backend_endpoint"):
        code_texts(subset, toy_rubric, elsewhere, store, settings)


class LooseProvenance(FakeBackend):
    """A fake backend whose provenance holds a number instead of a string."""

    def provenance(self) -> dict[str, Any]:
        return {**super().provenance(), "window": 4096}


def test_a_provenance_value_that_is_no_string_is_refused(
    toy_rubric: Rubric, toy_corpus: TextTable, tmp_path: Path
) -> None:
    backend = LooseProvenance(cue_responder(toy_rubric, load_toy_cues()))
    with pytest.raises(TypeError, match="'window'"):
        code_texts(
            toy_corpus.subset(["t001"]), toy_rubric, backend, tmp_path / "r.sqlite",
            CodingSettings(runs=1),
        )  # fmt: skip
    assert not (tmp_path / "r.sqlite").exists()
