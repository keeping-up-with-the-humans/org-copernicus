"""Tests for the reference-text generator."""

from __future__ import annotations

import hashlib
import json
import logging

import pytest
from references_support import (
    clean_responder,
    seed_responder,
    texts_json,
    writer,
)

from copernicus.errors import BackendError, ConfigurationError
from copernicus.llm.base import GenerationRequest
from copernicus.references import (
    GeneratorSettings,
    build_cells,
    generate_references,
    plan_calls,
)
from copernicus.references.prompts import PROMPT_VERSION, system_prompt, user_prompt
from copernicus.rubric.model import Rubric

CELL_COUNT = 24


def test_a_writer_produces_ten_texts_per_cell_with_balanced_polarities(
    toy_rubric: Rubric,
) -> None:
    result = generate_references(toy_rubric, [writer(clean_responder())], GeneratorSettings())
    frame = result.frame
    assert len(frame) == CELL_COUNT * 10
    assert list(frame.columns) == [
        "rubric_id",
        "target",
        "polarity",
        "level",
        "style",
        "writer_model",
        "text",
        "call_index",
        "attempt",
        "seed",
        "prompt_tokens",
        "completion_tokens",
        "temperature_applied",
        "top_p_applied",
        "prompt_hash",
        "prompt_version",
        "screened_out",
        "screen_reason",
    ]
    counts = frame.groupby(["target", "polarity"]).size().unstack()
    assert (counts["positive"] == counts["negative"]).all()
    assert (frame.groupby(["target", "polarity", "level"]).size() == 10).all()
    assert frame["writer_model"].unique().tolist() == ["writer-a"]
    assert frame["rubric_id"].unique().tolist() == [toy_rubric.rubric_id]
    assert frame["style"].unique().tolist() == ["neutral"]
    assert frame["prompt_version"].unique().tolist() == [PROMPT_VERSION]
    assert frame["call_index"].unique().tolist() == [0]
    assert frame["screened_out"].dtype == bool
    assert not frame["screened_out"].any()
    assert result.shortfalls == ()
    result.assert_balanced()


def test_each_cell_row_records_the_hash_of_its_prompts(toy_rubric: Rubric) -> None:
    result = generate_references(toy_rubric, [writer(clean_responder())], GeneratorSettings())
    cell = build_cells(toy_rubric)[0]
    system = system_prompt("neutral", 10, None)
    user = user_prompt(cell, toy_rubric, 10)
    expected = hashlib.sha256((system + "\n" + user).encode("utf-8")).hexdigest()[:16]
    first = result.frame.iloc[0]
    assert (first["target"], first["level"], first["polarity"]) == ("construct", "top", "positive")
    assert first["prompt_hash"] == expected
    assert result.frame["prompt_hash"].nunique() == CELL_COUNT


def test_requests_carry_the_prompts_and_the_settings(toy_rubric: Rubric) -> None:
    backend = writer(clean_responder(3))
    settings = GeneratorSettings(texts_per_call=3, temperature=0.5, max_new_tokens=999)
    generate_references(toy_rubric, [backend], settings)
    first = backend.calls[0]
    assert first.system == system_prompt("neutral", 3, None)
    assert first.user == user_prompt(build_cells(toy_rubric)[0], toy_rubric, 3)
    assert (first.temperature, first.max_new_tokens) == (0.5, 999)


def test_the_register_style_uses_the_register_prompt(toy_rubric: Rubric) -> None:
    backend = writer(clean_responder())
    settings = GeneratorSettings(style="register", register_description="short posts")
    result = generate_references(toy_rubric, [backend], settings)
    assert backend.calls[0].system == system_prompt("register", 10, "short posts")
    assert result.frame["style"].unique().tolist() == ["register"]


def test_several_writers_each_write_every_cell_and_label_their_rows(toy_rubric: Rubric) -> None:
    writers = [writer(clean_responder(), "model-x"), writer(clean_responder(), "model-y")]
    result = generate_references(toy_rubric, writers, GeneratorSettings(calls_per_cell=2))
    assert len(result.frame) == 2 * CELL_COUNT * 2 * 10
    assert result.frame.groupby("writer_model").size().to_dict() == {
        "model-x": CELL_COUNT * 20,
        "model-y": CELL_COUNT * 20,
    }
    assert sorted(result.frame["call_index"].unique().tolist()) == [0, 1]
    result.assert_balanced()


def test_plan_calls_is_cells_times_writers_times_calls_per_cell(toy_rubric: Rubric) -> None:
    writers = [writer(clean_responder(), "a"), writer(clean_responder(), "b")]
    assert plan_calls(toy_rubric, writers[:1], GeneratorSettings()) == CELL_COUNT
    assert plan_calls(toy_rubric, writers, GeneratorSettings(calls_per_cell=3)) == CELL_COUNT * 6
    with pytest.raises(ConfigurationError, match="writer"):
        plan_calls(toy_rubric, [], GeneratorSettings())


def test_no_writer_is_a_configuration_error(toy_rubric: Rubric) -> None:
    with pytest.raises(ConfigurationError, match="writer"):
        generate_references(toy_rubric, [], GeneratorSettings())


def test_two_writers_with_one_model_name_are_a_configuration_error(toy_rubric: Rubric) -> None:
    writers = [writer(clean_responder(), "same"), writer(clean_responder(), "same")]
    with pytest.raises(ConfigurationError, match="same"):
        generate_references(toy_rubric, writers, GeneratorSettings())


def test_seeds_follow_the_call_and_cell_indices_and_retries_add_one_hundred_thousand(
    toy_rubric: Rubric,
) -> None:
    backend = writer(seed_responder({1043: "no json here"}))
    generate_references(toy_rubric, [backend], GeneratorSettings(calls_per_cell=2))
    seeds = [call.seed for call in backend.calls]
    # cell 0 calls 0 and 1, cell 1 calls 0 and 1 (which fails once and is retried), cell 2 ...
    assert seeds[:6] == [42, 1042, 43, 1043, 101043, 44]
    assert len(seeds) == CELL_COUNT * 2 + 1
    assert len(set(seeds)) == len(seeds)


def test_a_text_that_is_not_json_counts_as_a_failed_attempt(toy_rubric: Rubric) -> None:
    backend = writer(seed_responder({42: "not json", 100042: "still nothing"}))
    result = generate_references(toy_rubric, [backend], GeneratorSettings())
    assert [call.seed for call in backend.calls[:4]] == [42, 100042, 200042, 43]
    assert result.shortfalls == ()
    assert len(result.frame) == CELL_COUNT * 10


def test_nine_texts_trigger_three_attempts_and_a_shortfall_of_one(toy_rubric: Rubric) -> None:
    backend = writer(clean_responder(9))
    result = generate_references(toy_rubric, [backend], GeneratorSettings())
    assert len(backend.calls) == CELL_COUNT * 3
    assert len(result.frame) == CELL_COUNT * 9
    assert len(result.shortfalls) == CELL_COUNT
    assert {(s.expected, s.received) for s in result.shortfalls} == {(10, 9)}
    assert sum(result.summary()["shortfalls"].values()) == CELL_COUNT


def test_the_last_attempt_decides_what_a_short_cell_keeps(toy_rubric: Rubric) -> None:
    answers = {
        42: texts_json(9, "first"),
        100042: texts_json(9, "second"),
        200042: texts_json(8, "third"),
    }
    result = generate_references(toy_rubric, [writer(seed_responder(answers))], GeneratorSettings())
    first_cell = result.frame[
        (result.frame["target"] == "construct")
        & (result.frame["level"] == "top")
        & (result.frame["polarity"] == "positive")
    ]
    assert len(first_cell) == 8
    assert first_cell["text"].str.startswith("Note third").all()
    assert [(s.target, s.expected, s.received) for s in result.shortfalls] == [("construct", 10, 8)]


def test_a_cell_whose_last_attempt_is_unusable_keeps_nothing_and_records_the_shortfall(
    toy_rubric: Rubric,
) -> None:
    backend = writer(seed_responder({42: "nothing useful"}))
    result = generate_references(toy_rubric, [backend], GeneratorSettings(max_retries=1))
    assert len(backend.calls) == CELL_COUNT
    assert len(result.frame) == (CELL_COUNT - 1) * 10
    (shortfall,) = result.shortfalls
    assert (shortfall.target, shortfall.level, shortfall.polarity) == (
        "construct",
        "top",
        "positive",
    )
    assert (shortfall.expected, shortfall.received) == (10, 0)
    with pytest.raises(ConfigurationError, match="construct"):
        result.assert_balanced()


def test_objects_without_a_text_make_an_answer_invalid_and_are_dropped_at_the_end(
    toy_rubric: Rubric,
) -> None:
    broken = '[{"text": "a fine note"}, {"text": ""}, {"note": "no text key"}, "bare string"]'
    backend = writer(seed_responder({42: broken, 100042: broken, 200042: broken}))
    result = generate_references(toy_rubric, [backend], GeneratorSettings())
    assert len(backend.calls) == CELL_COUNT + 2
    assert (result.frame["text"] == "a fine note").sum() == 1
    assert [(s.expected, s.received) for s in result.shortfalls] == [(10, 1)]


def test_extra_keys_are_ignored_and_a_surplus_is_cut_on_the_first_attempt(
    toy_rubric: Rubric,
) -> None:
    extra = texts_json(10, "x", extra_keys=True)
    surplus = texts_json(12, "y")
    answers = {42: extra, 43: surplus}
    backend = writer(seed_responder(answers))
    result = generate_references(toy_rubric, [backend], GeneratorSettings())
    sizes = result.frame.groupby(["target", "polarity", "level"]).size()
    assert (sizes == 10).all()
    assert result.shortfalls == ()
    assert len(backend.calls) == CELL_COUNT
    frame = result.frame
    second = frame[(frame["target"] == "construct") & (frame["polarity"] == "negative")]
    second = second[second["level"] == "top"]
    assert second["text"].tolist() == [
        f"Note y-{index}: I went for a long walk today." for index in range(10)
    ]


def test_a_truncated_answer_keeps_its_complete_texts(toy_rubric: Rubric) -> None:
    truncated = texts_json(4, "cut")[:-1] + ', {"text": "Note cut-4: I went'
    backend = writer(seed_responder({42: truncated}))
    result = generate_references(toy_rubric, [backend], GeneratorSettings(max_retries=1))
    first = result.frame[result.frame["call_index"] == 0].head(4)
    assert first["text"].str.startswith("Note cut-").all()
    assert [(s.target, s.expected, s.received) for s in result.shortfalls] == [("construct", 10, 4)]


def test_ten_complete_texts_without_the_closing_bracket_are_accepted(toy_rubric: Rubric) -> None:
    unclosed = texts_json(10, "open")[:-1]
    backend = writer(seed_responder({42: unclosed}))
    result = generate_references(toy_rubric, [backend], GeneratorSettings())
    assert len(backend.calls) == CELL_COUNT and result.shortfalls == ()


def test_every_failed_attempt_is_logged_with_its_reason_and_an_excerpt(
    toy_rubric: Rubric, caplog: pytest.LogCaptureFixture
) -> None:
    backend = writer(seed_responder({42: "garbled output"}))
    with caplog.at_level(logging.INFO, logger="copernicus.references.answers"):
        generate_references(toy_rubric, [backend], GeneratorSettings())
    failures = [r.getMessage() for r in caplog.records if "failed" in r.getMessage()]
    assert len(failures) == 1
    assert "attempt 1/3" in failures[0]
    assert "garbled output" in failures[0]
    assert "json_parse_failure" in failures[0]


def test_a_transport_failure_propagates(toy_rubric: Rubric) -> None:
    def broken(request: object) -> str:
        raise BackendError("endpoint unreachable")

    with pytest.raises(BackendError, match="unreachable"):
        generate_references(toy_rubric, [writer(broken)], GeneratorSettings())


def leaking_responder(request: GenerationRequest) -> str:
    """Answer positive cells cleanly and negative cells with texts that name two markers."""
    if "active LOW end" in request.user:
        leaking = [
            {"text": f"Note {index}: all I could think about was the Tone."} for index in range(10)
        ]
        return json.dumps(leaking)
    return texts_json(10, request.seed)


def test_texts_that_name_the_construct_are_screened_out_but_stay_in_the_frame(
    toy_rubric: Rubric,
) -> None:
    result = generate_references(toy_rubric, [writer(leaking_responder)], GeneratorSettings())
    assert len(result.frame) == CELL_COUNT * 10
    negative = result.frame["polarity"] == "negative"
    assert result.frame.loc[negative, "screened_out"].all()
    assert not result.frame.loc[~negative, "screened_out"].any()
    assert (result.frame.loc[negative, "screen_reason"] == "names construct term 'tone'").all()
    assert len(result.kept()) == CELL_COUNT // 2 * 10
    with pytest.raises(ConfigurationError) as error:
        result.assert_balanced()
    assert "writer-a" in str(error.value)
    assert "construct" in str(error.value)


def test_screening_can_be_switched_off(toy_rubric: Rubric) -> None:
    result = generate_references(
        toy_rubric, [writer(leaking_responder)], GeneratorSettings(), screen=False
    )
    assert not result.frame["screened_out"].any()
    assert (result.frame["screen_reason"] == "").all()
    result.assert_balanced()


def test_a_bare_array_behind_prose_still_fills_the_cell(toy_rubric: Rubric) -> None:
    def prose_first(request: GenerationRequest) -> str:
        return "Here are the texts you asked for:\n" + texts_json(10, request.seed)

    backend = writer(prose_first)
    result = generate_references(toy_rubric, [backend], GeneratorSettings())
    assert len(backend.calls) == CELL_COUNT
    assert len(result.frame) == CELL_COUNT * 10
    assert result.shortfalls == ()


def test_a_register_description_without_a_full_stop_warns_once_per_run(
    toy_rubric: Rubric, caplog: pytest.LogCaptureFixture
) -> None:
    settings = GeneratorSettings(style="register", register_description="short posts")
    writers = [writer(clean_responder(), "model-x"), writer(clean_responder(), "model-y")]
    with caplog.at_level(logging.WARNING, logger="copernicus.references"):
        generate_references(toy_rubric, writers, settings)
    assert len([r for r in caplog.records if "short posts" in r.getMessage()]) == 1
