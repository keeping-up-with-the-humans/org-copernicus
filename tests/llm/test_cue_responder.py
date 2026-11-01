"""Tests for the deterministic cue-based responder of the fake backend."""

from __future__ import annotations

import json
from collections.abc import Callable

import pytest

from copernicus.coders import (
    PromptContext,
    load_wording_pool,
    render_extraction_prompt,
    render_one_stage_prompt,
    render_scoring_prompt,
    select,
    user_message_for_text,
)
from copernicus.data import TextTable
from copernicus.errors import BackendError
from copernicus.llm import FakeBackend, GenerationRequest, cue_responder
from copernicus.rubric.model import Rubric

PINNED = ["role_baseline", "task_baseline", "level_independent"]
TEXT = (
    "Thank you for the draft. The section on costs is thin, for example the second line "
    "is empty. I suggest a table! Why not?"
)


def make_ctx(rubric: Rubric, attribute_index: int) -> PromptContext:
    pool = select(load_wording_pool(), enable=PINNED)
    attribute = rubric.attributes[attribute_index]
    order = [marker.name for marker in attribute.markers]
    return PromptContext(
        rubric,
        attribute,
        order,
        pool.get("role", "role_baseline"),
        pool.get("task", "task_baseline"),
        pool.get("level", "level_independent"),
    )


def ask(backend: FakeBackend, system: str, text: str) -> dict[str, object]:
    request = GenerationRequest(system, user_message_for_text(text), 100, 0.3, 1)
    answer = json.loads(backend.generate(request).text)
    assert isinstance(answer, dict)
    return answer


def test_cue_responder_quotes_are_verbatim_sentences(
    toy_rubric: Rubric, toy_corpus: TextTable, cue_backend: Callable[[], FakeBackend]
) -> None:
    backend = cue_backend()
    text = toy_corpus.text_of("t050")
    out = ask(backend, render_extraction_prompt(make_ctx(toy_rubric, 0)), text)
    assert any(entry["quotes"] for entry in out.values())  # type: ignore[index]
    for entry in out.values():
        for quote in entry["quotes"]:  # type: ignore[index]
            assert quote in text


def test_extraction_quotes_whole_sentences_with_their_punctuation(
    toy_rubric: Rubric, cue_backend: Callable[[], FakeBackend]
) -> None:
    out = ask(cue_backend(), render_extraction_prompt(make_ctx(toy_rubric, 0)), TEXT)
    quotes = {name: entry["quotes"] for name, entry in out.items()}  # type: ignore[index]
    assert quotes["Concrete examples"] == [
        "The section on costs is thin, for example the second line is empty."
    ]
    assert quotes["Actionable next step"] == ["I suggest a table!"]
    assert out["Actionable next step"]["evaluable"] is True  # type: ignore[index]


def test_id_keyed_and_name_keyed_cues_answer_alike(
    toy_rubric: Rubric, cues: dict[str, list[str]]
) -> None:
    names = {m.marker_id: m.name for a in toy_rubric.attributes for m in a.markers}
    by_id = FakeBackend(cue_responder(toy_rubric, cues))
    by_name = FakeBackend(cue_responder(toy_rubric, {names[k]: v for k, v in cues.items()}))
    for index in range(len(toy_rubric.attributes)):
        ctx = make_ctx(toy_rubric, index)
        for system in (render_extraction_prompt(ctx), render_one_stage_prompt(ctx)):
            assert ask(by_id, system, TEXT) == ask(by_name, system, TEXT)


def test_scoring_prompt_is_answered_from_the_passage_count(
    toy_rubric: Rubric, cue_backend: Callable[[], FakeBackend]
) -> None:
    ctx = make_ctx(toy_rubric, 0)
    system = render_scoring_prompt(ctx, list(ctx.marker_order))
    user = 'Extracted Evidence:\n- "One."\n- "Two."\n- "Three."\n'
    request = GenerationRequest(system, user, 100, 0.0, 1)
    answer = json.loads(cue_backend().generate(request).text)
    assert answer["result"] == 2 and "3 passages" in answer["explanation"]


def test_one_stage_prompt_names_its_attribute(
    toy_rubric: Rubric, cues: dict[str, list[str]]
) -> None:
    backend = FakeBackend(cue_responder(toy_rubric, cues))
    text = "Thank you for this. However, the draft is long."
    tone = ask(backend, render_one_stage_prompt(make_ctx(toy_rubric, 1)), text)
    specificity = ask(backend, render_one_stage_prompt(make_ctx(toy_rubric, 0)), text)
    assert tone["dimension_result"]["explanation"] == "2 cue words"  # type: ignore[index]
    assert specificity["dimension_result"]["explanation"] == "0 cue words"  # type: ignore[index]
    forced = FakeBackend(cue_responder(toy_rubric, cues, one_stage_level=lambda n: 5))
    assert ask(forced, render_one_stage_prompt(make_ctx(toy_rubric, 1)), text) == {
        "dimension_result": {"result": 5, "explanation": "2 cue words"}
    }


def test_unknown_cue_key_is_rejected(toy_rubric: Rubric, cues: dict[str, list[str]]) -> None:
    with pytest.raises(ValueError, match="'not_a_marker'"):
        cue_responder(toy_rubric, {**cues, "not_a_marker": ["x"]})


def test_missing_cues_for_a_marker_are_rejected(
    toy_rubric: Rubric, cues: dict[str, list[str]]
) -> None:
    del cues["tone_blame"]
    with pytest.raises(ValueError, match="tone_blame"):
        cue_responder(toy_rubric, cues)


def test_empty_cue_list_is_rejected(toy_rubric: Rubric, cues: dict[str, list[str]]) -> None:
    cues["tone_blame"] = []
    with pytest.raises(ValueError, match="tone_blame"):
        cue_responder(toy_rubric, cues)


def test_a_bare_string_is_not_a_cue_list(toy_rubric: Rubric, cues: dict[str, object]) -> None:
    cues["tone_blame"] = "sloppy"
    with pytest.raises(ValueError, match="tone_blame"):
        cue_responder(toy_rubric, cues)  # type: ignore[arg-type]


def test_marker_without_a_cue_hit_is_not_evaluable(
    toy_rubric: Rubric, cue_backend: Callable[[], FakeBackend]
) -> None:
    out = ask(cue_backend(), render_extraction_prompt(make_ctx(toy_rubric, 0)), "Plain words only.")
    assert all(entry == {"quotes": [], "evaluable": False} for entry in out.values())


def test_unknown_prompt_is_rejected(toy_rubric: Rubric, cues: dict[str, list[str]]) -> None:
    responder = cue_responder(toy_rubric, cues)
    with pytest.raises(BackendError, match="classify"):
        responder(GenerationRequest("something else", "u", 10, 0.0, 1))


def test_one_stage_prompt_without_a_known_attribute_is_rejected(
    toy_rubric: Rubric, cues: dict[str, list[str]]
) -> None:
    responder = cue_responder(toy_rubric, cues)
    system = (
        "You operate as a single integrated evaluator.\nYou are evaluating the attribute: Wit\n"
    )
    with pytest.raises(BackendError, match="attribute"):
        responder(GenerationRequest(system, user_message_for_text(TEXT), 10, 0.0, 1))
    with pytest.raises(BackendError, match="attribute"):
        responder(
            GenerationRequest(
                "single integrated evaluator", user_message_for_text(TEXT), 10, 0.0, 1
            )
        )


def test_user_message_without_a_text_is_rejected(
    toy_rubric: Rubric, cues: dict[str, list[str]]
) -> None:
    responder = cue_responder(toy_rubric, cues)
    system = render_extraction_prompt(make_ctx(toy_rubric, 0))
    with pytest.raises(BackendError, match="text"):
        responder(GenerationRequest(system, "no prefix here", 10, 0.0, 1))
