import json
import logging

import pytest
from two_stage_support import (
    DOC,
    OK_SCORE,
    PASSAGE,
    CountingBackend,
    evidence_json,
    make_ctx,
    names_of,
    no_evidence_json,
    run,
)

from copernicus.coders import (
    TwoStageConfig,
    evaluate_attribute,
    render_extraction_prompt,
    render_scoring_prompt,
    stage1_attempt_temperature,
    validate_scoring_output,
)
from copernicus.errors import BackendError
from copernicus.llm.fake import FakeBackend, ScriptedResponder
from copernicus.rubric.model import Rubric


def test_full_path_ok_records_attempts_and_evidence(toy_rubric: Rubric) -> None:
    names = names_of(toy_rubric)
    b = FakeBackend(ScriptedResponder([evidence_json(names), OK_SCORE]))
    ctx = make_ctx(toy_rubric)
    r = run(b, toy_rubric, seed=43)
    assert r.status == "ok" and r.score == 4 and r.stage1_attempt == 0 and r.stage2_attempt == 0
    assert r.attribute_id == toy_rubric.attributes[0].attribute_id
    assert r.explanation == "clear evidence"
    assert len(r.evidence) == 3 and b.calls[1].user.startswith("The scoring rubric:")
    assert "Text to analyze" not in b.calls[1].user and b.calls[0].seed == 43
    assert b.calls[0].user == f"Text to analyze:\n{DOC}"
    assert (b.calls[0].max_new_tokens, b.calls[1].max_new_tokens) == (8192, 2048)
    assert isinstance(r.prompt_tokens, int) and r.prompt_tokens > 0
    assert isinstance(r.completion_tokens, int) and r.completion_tokens > 0
    assert r.seconds >= 0
    assert b.calls[0].system == render_extraction_prompt(ctx)
    assert b.calls[1].system == render_scoring_prompt(ctx, names)
    assert all(r.evaluable.values()) and len(r.evaluable) == 3
    assert set(r.justification.values()) == {PASSAGE}


def test_token_counts_treat_none_as_zero(toy_rubric: Rubric) -> None:
    names = names_of(toy_rubric)
    b = CountingBackend([evidence_json(names), OK_SCORE])
    r = run(b, toy_rubric)
    assert r.status == "ok" and r.prompt_tokens == 0 and r.completion_tokens == 0


def test_token_counts_are_summed_over_every_call(toy_rubric: Rubric) -> None:
    names = names_of(toy_rubric)
    b = FakeBackend(ScriptedResponder(["not json", evidence_json(names), OK_SCORE]))
    r = run(b, toy_rubric)
    assert r.stage1_attempt == 1
    assert r.prompt_tokens == len(b.calls[0].system.split()) + len(b.calls[0].user.split()) + len(
        b.calls[1].system.split()
    ) + len(b.calls[1].user.split()) + len(b.calls[2].system.split()) + len(b.calls[2].user.split())


def test_no_evidence_scores_one_without_stage2(toy_rubric: Rubric) -> None:
    b = FakeBackend(ScriptedResponder([no_evidence_json(names_of(toy_rubric))]))
    r = run(b, toy_rubric)
    assert r.status == "no_evidence" and r.score == 1 and len(b.calls) == 1
    assert r.explanation == "no_markers_evaluable"
    assert r.stage1_attempt == 0 and r.stage2_attempt is None
    assert r.evidence == () and not any(r.evaluable.values()) and len(r.evaluable) == 3


def test_no_evidence_score_none_when_configured(toy_rubric: Rubric) -> None:
    b = FakeBackend(ScriptedResponder([no_evidence_json(names_of(toy_rubric))]))
    r = run(b, toy_rubric, TwoStageConfig(no_evidence_score=None))
    assert r.status == "no_evidence" and r.score is None and len(b.calls) == 1


def test_hallucinated_only_evidence_is_stored_but_not_scored(toy_rubric: Rubric) -> None:
    names = names_of(toy_rubric)
    stage1 = json.dumps(
        {
            names[0]: {"evaluable": True, "quotes": ["this passage is invented wholesale"]},
            **{n: {"evaluable": False, "quotes": []} for n in names[1:]},
        }
    )
    b = FakeBackend(ScriptedResponder([stage1]))
    r = run(b, toy_rubric)
    assert r.status == "no_evidence" and len(r.evidence) == 1 and r.evidence[0].verified == 0
    assert len(b.calls) == 1


def test_stage1_failure_is_none_never_zero_and_temperature_escalates(toy_rubric: Rubric) -> None:
    b = FakeBackend(ScriptedResponder(["not json", "{}", "[]"]))
    r = run(b, toy_rubric, seed=10)
    assert r.status == "extraction_failed" and r.score is None and r.stage1_attempt is None
    assert [c.temperature for c in b.calls] == [0.3, 0.6, 0.9]
    assert [c.seed for c in b.calls] == [10, 11, 12]
    assert r.evidence == () and r.evaluable == {} and r.justification == {}
    assert r.stage2_attempt is None
    assert r.explanation.startswith("extraction_failed: top-level type list")


def test_stage1_retry_succeeds_on_a_later_attempt(toy_rubric: Rubric) -> None:
    names = names_of(toy_rubric)
    b = FakeBackend(ScriptedResponder(["oops", evidence_json(names), OK_SCORE]))
    r = run(b, toy_rubric, seed=5)
    assert r.status == "ok" and r.stage1_attempt == 1 and r.stage2_attempt == 0
    assert [c.seed for c in b.calls] == [5, 6, 5]
    assert [c.temperature for c in b.calls] == [0.3, 0.6, 0.3]


def test_stage2_out_of_range_retries_then_scoring_failed(toy_rubric: Rubric) -> None:
    stage1 = evidence_json(names_of(toy_rubric), "Everyone agreed.")
    b = FakeBackend(
        ScriptedResponder([stage1, '{"result": 7}', '{"result": "high"}', '{"result": 0}'])
    )
    r = run(b, toy_rubric)
    assert r.status == "scoring_failed" and r.score is None
    assert r.stage1_attempt == 0 and r.stage2_attempt is None
    assert [c.temperature for c in b.calls[1:]] == [0.3, 0.3, 0.3]
    assert [c.seed for c in b.calls[1:]] == [1, 2, 3]
    assert r.explanation == "scoring_failed: 'result' out of range 1..5: 0"
    assert len(r.evidence) == 3 and len(r.evaluable) == 3 and len(r.justification) == 3


def test_stage2_retry_succeeds_and_records_the_attempt(toy_rubric: Rubric) -> None:
    stage1 = evidence_json(names_of(toy_rubric))
    b = FakeBackend(ScriptedResponder([stage1, "no json here", OK_SCORE]))
    r = run(b, toy_rubric, seed=20)
    assert r.status == "ok" and r.score == 4 and r.stage2_attempt == 1
    assert [c.seed for c in b.calls[1:]] == [20, 21]


def test_stage2_unparseable_output_gives_json_parse_failure(toy_rubric: Rubric) -> None:
    stage1 = evidence_json(names_of(toy_rubric))
    b = FakeBackend(ScriptedResponder([stage1, "", "", ""]))
    r = run(b, toy_rubric)
    assert r.status == "scoring_failed" and r.explanation == "scoring_failed: json_parse_failure"


def test_too_short_text_skipped_without_calls(toy_rubric: Rubric) -> None:
    b = FakeBackend(ScriptedResponder([]))
    r = evaluate_attribute(
        b,
        "  too short  ",
        make_ctx(toy_rubric),
        TwoStageConfig(),
        temperature_stage1=0.3,
        temperature_stage2=0.3,
        seed=1,
    )
    assert r.status == "too_short" and r.score is None and r.explanation == "too_short"
    assert b.calls == [] and r.stage1_attempt is None and r.stage2_attempt is None
    assert r.evidence == () and r.evaluable == {} and r.justification == {}
    assert r.prompt_tokens == 0 and r.completion_tokens == 0


def test_min_chars_counts_the_stripped_text(toy_rubric: Rubric) -> None:
    padded = " " * 40 + "short" + " " * 40
    b = FakeBackend(ScriptedResponder([]))
    r = evaluate_attribute(
        b,
        padded,
        make_ctx(toy_rubric),
        TwoStageConfig(min_chars=20),
        temperature_stage1=None,
        temperature_stage2=None,
        seed=1,
    )
    assert r.status == "too_short" and b.calls == []


def test_wrapped_marker_analysis_accepted(toy_rubric: Rubric) -> None:
    stage1 = json.dumps({"marker_analysis": json.loads(evidence_json(names_of(toy_rubric)))})
    b = FakeBackend(ScriptedResponder([stage1, OK_SCORE]))
    r = run(b, toy_rubric)
    assert r.status == "ok" and r.stage1_attempt == 0 and len(r.evidence) == 3


def test_labelled_stage2_groups_by_marker(toy_rubric: Rubric) -> None:
    names = names_of(toy_rubric)
    stage1 = json.dumps(
        {
            names[0]: {"evaluable": True, "quotes": ["Everyone agreed."]},
            **{n: {"evaluable": False, "quotes": []} for n in names[1:]},
        }
    )
    b = FakeBackend(ScriptedResponder([stage1, OK_SCORE]))
    r = run(b, toy_rubric, TwoStageConfig(stage2_evidence="labelled"))
    assert r.status == "ok"
    assert f'Marker: {names[0]}\n- "Everyone agreed."' in b.calls[1].user
    unlabelled = FakeBackend(ScriptedResponder([stage1, OK_SCORE]))
    run(unlabelled, toy_rubric)
    assert "Marker: " not in unlabelled.calls[1].user


def test_empty_output_counts_as_parse_failure(toy_rubric: Rubric) -> None:
    b = FakeBackend(ScriptedResponder(["", "", ""]))
    r = run(b, toy_rubric)
    assert r.status == "extraction_failed"
    assert r.explanation == "extraction_failed: json_parse_failure"
    assert r.score is None and len(b.calls) == 3


def test_max_retries_bounds_the_calls(toy_rubric: Rubric) -> None:
    b = FakeBackend(ScriptedResponder(["x", "y", "z", "w"]))
    r = run(b, toy_rubric, TwoStageConfig(max_retries=2))
    assert r.status == "extraction_failed" and len(b.calls) == 2


def test_backend_error_from_generate_propagates(toy_rubric: Rubric) -> None:
    b = FakeBackend(ScriptedResponder([]))
    with pytest.raises(BackendError, match="exhausted"):
        run(b, toy_rubric)


def test_every_failed_attempt_is_logged_with_reason_and_excerpt(
    toy_rubric: Rubric, caplog: pytest.LogCaptureFixture
) -> None:
    b = FakeBackend(ScriptedResponder(["not json at all", "{}", "[]"]))
    with caplog.at_level(logging.INFO, logger="copernicus.coders.stage_runner"):
        run(b, toy_rubric, seed=7)
    messages = [rec.getMessage() for rec in caplog.records]
    assert len(messages) == 3
    attribute_id = toy_rubric.attributes[0].attribute_id
    assert all(attribute_id in m and "extraction" in m for m in messages)
    assert "attempt 1/3" in messages[0] and "json_parse_failure" in messages[0]
    assert "seed=7" in messages[0] and "'not json at all'" in messages[0]
    assert "attempt 3/3" in messages[2] and "seed=9" in messages[2]


def test_long_raw_output_is_logged_as_head_and_tail(
    toy_rubric: Rubric, caplog: pytest.LogCaptureFixture
) -> None:
    raw = "a" * 400 + "b" * 400
    b = FakeBackend(ScriptedResponder([raw, raw, raw]))
    with caplog.at_level(logging.INFO, logger="copernicus.coders.stage_runner"):
        run(b, toy_rubric)
    first = caplog.records[0].getMessage()
    assert repr("a" * 300) in first and repr("b" * 200) in first and "[300 chars]" in first


def test_second_attribute_uses_its_own_markers(toy_rubric: Rubric) -> None:
    names = names_of(toy_rubric, 1)
    b = FakeBackend(ScriptedResponder([evidence_json(names), OK_SCORE]))
    r = evaluate_attribute(
        b,
        DOC,
        make_ctx(toy_rubric, 1),
        TwoStageConfig(),
        temperature_stage1=None,
        temperature_stage2=None,
        seed=1,
    )
    assert r.status == "ok" and r.attribute_id == toy_rubric.attributes[1].attribute_id
    assert b.calls[0].temperature is None and b.calls[1].temperature is None


@pytest.mark.parametrize(
    "base,attempt,expected",
    [
        (0.3, 0, 0.3),
        (None, 0, None),
        (0.3, 1, 0.6),
        (0.3, 2, 0.9),
        (0.3, 3, 1.0),
        (None, 1, 0.3),
        (None, 2, 0.6),
        (0.7, -1, 0.7),
    ],
)
def test_stage1_attempt_temperature(
    base: float | None, attempt: int, expected: float | None
) -> None:
    assert stage1_attempt_temperature(base, attempt, 0.3, 1.0) == expected


def test_stage1_attempt_temperature_rounds_float_noise() -> None:
    assert stage1_attempt_temperature(0.1, 2, 0.1, 1.0) == 0.3


@pytest.mark.parametrize(
    "parsed,ok",
    [
        ({"result": 3, "explanation": "x"}, True),
        ({"result": "4"}, True),
        ({"result": " 5 "}, True),
        ({"result": 4.0}, True),
        ({"result": True}, False),
        ({"result": 6}, False),
        ({"result": 0}, False),
        ({"result": 3.5}, False),
        ({"result": "high"}, False),
        ({"result": 3, "explanation": 7}, False),
        ({}, False),
        ([3], False),
        (None, False),
    ],
)
def test_validate_scoring_output(parsed: object, ok: bool) -> None:
    assert validate_scoring_output(parsed)[0] is ok


def test_validate_scoring_output_returns_value_and_explanation() -> None:
    assert validate_scoring_output({"result": "4", "explanation": "x"}) == (True, "", 4, "x")
    assert validate_scoring_output({"result": 2}) == (True, "", 2, "")
    assert validate_scoring_output({"result": 2, "explanation": None}) == (True, "", 2, "")
    valid, reason, value, explanation = validate_scoring_output({"result": 9})
    assert (valid, value, explanation) == (False, None, "") and "out of range" in reason
    assert validate_scoring_output([3])[1] == "top-level type list is not a JSON object"
    assert validate_scoring_output({})[1] == "missing 'result'"
