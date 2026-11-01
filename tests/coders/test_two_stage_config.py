import json
import logging

import pytest
from two_stage_support import (
    OK_SCORE,
    PASSAGE,
    CountingBackend,
    evidence_json,
    make_ctx,
    names_of,
    run,
)

from copernicus.coders import AttributeResult, TwoStageConfig, evaluate_attribute
from copernicus.errors import BackendError, CopernicusError, SettingsError
from copernicus.llm.base import GenerationRequest
from copernicus.llm.fake import FakeBackend, ScriptedResponder
from copernicus.rubric.model import Rubric

FUZZY_QUOTE = "We decided to paint the gardn shed"


def test_default_config_is_valid() -> None:
    config = TwoStageConfig()
    assert config.max_retries == 3 and config.no_evidence_score == 1


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_retries", 0),
        ("max_retries", -1),
        ("no_evidence_score", 0),
        ("no_evidence_score", 6),
        ("no_evidence_score", 2.5),
        ("no_evidence_score", True),
        ("no_evidence_score", "2"),
        ("stage2_evidence", "Labelled"),
        ("stage2_evidence", ""),
        ("stage1_max_new_tokens", 0),
        ("stage2_max_new_tokens", 0),
        ("min_chars", -1),
        ("min_fuzzy_words", 0),
        ("verify_threshold", -0.1),
        ("verify_threshold", 100.1),
        ("retry_temperature_step", -0.1),
        ("retry_temperature_cap", -0.5),
    ],
)
def test_invalid_setting_raises_and_names_field_and_value(field: str, value: object) -> None:
    with pytest.raises(SettingsError, match=f"{field} must be .*got {value!r}") as error:
        TwoStageConfig(**{field: value})  # type: ignore[arg-type]
    assert isinstance(error.value, CopernicusError) and isinstance(error.value, ValueError)


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_retries", 1),
        ("no_evidence_score", None),
        ("no_evidence_score", 5),
        ("stage2_evidence", "labelled"),
        ("min_chars", 0),
        ("verify_threshold", 0.0),
        ("verify_threshold", 100.0),
        ("retry_temperature_step", 0.0),
        ("retry_temperature_cap", 0.0),
    ],
)
def test_boundary_settings_are_accepted(field: str, value: object) -> None:
    assert getattr(TwoStageConfig(**{field: value}), field) == value  # type: ignore[arg-type]


def _result(**overrides: object) -> AttributeResult:
    fields: dict[str, object] = {
        "attribute_id": "a",
        "score": 3,
        "status": "ok",
        "explanation": "",
        "stage1_attempt": 0,
        "stage2_attempt": 0,
        "evidence": (),
        "evaluable": {},
        "justification": {},
        "seconds": 0.0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
    }
    fields.update(overrides)
    return AttributeResult(**fields)  # type: ignore[arg-type]


@pytest.mark.parametrize("score", [0, 6, -1])
def test_attribute_result_rejects_an_out_of_range_score(score: int) -> None:
    with pytest.raises(ValueError, match="1..5"):
        _result(score=score)


@pytest.mark.parametrize("score", [True, False, 3.0, 2.5, "3"])
def test_attribute_result_rejects_a_bool_or_non_integer_score(score: object) -> None:
    with pytest.raises(ValueError, match="integer"):
        _result(score=score)


def test_attribute_result_rejects_ok_without_a_score() -> None:
    with pytest.raises(ValueError, match="ok"):
        _result(score=None)


@pytest.mark.parametrize("status", ["too_short", "extraction_failed", "scoring_failed"])
def test_attribute_result_rejects_a_score_on_a_failed_status(status: str) -> None:
    with pytest.raises(ValueError, match=f"'{status}' requires score None"):
        _result(status=status, score=3)
    assert _result(status=status, score=None).score is None


def test_attribute_result_accepts_a_score_for_no_evidence() -> None:
    assert _result(score=None, status="no_evidence").score is None
    assert _result(score=1, status="no_evidence").score == 1


def test_verify_threshold_reaches_the_verification(toy_rubric: Rubric) -> None:
    stage1 = evidence_json(names_of(toy_rubric), FUZZY_QUOTE)
    lenient = run(FakeBackend(ScriptedResponder([stage1, OK_SCORE])), toy_rubric)
    assert lenient.status == "ok" and lenient.evidence[0].verified == 2
    strict_backend = FakeBackend(ScriptedResponder([stage1]))
    strict = run(strict_backend, toy_rubric, TwoStageConfig(verify_threshold=100.0))
    assert strict.status == "no_evidence" and len(strict_backend.calls) == 1
    assert strict.evidence[0].verified == 0


def test_min_fuzzy_words_reaches_the_verification(toy_rubric: Rubric) -> None:
    stage1 = evidence_json(names_of(toy_rubric), FUZZY_QUOTE)
    backend = FakeBackend(ScriptedResponder([stage1]))
    result = run(backend, toy_rubric, TwoStageConfig(min_fuzzy_words=99))
    assert result.status == "no_evidence" and result.evidence[0].verified == 0


def test_transport_error_during_scoring_propagates(toy_rubric: Rubric) -> None:
    answers = [evidence_json(names_of(toy_rubric))]

    def responder(request: GenerationRequest) -> str:
        if answers:
            return answers.pop()
        raise BackendError("connection reset", attempts=1)

    with pytest.raises(BackendError, match="connection reset"):
        run(FakeBackend(responder), toy_rubric)


def test_labelled_mode_shows_a_group_per_marker_to_rate(toy_rubric: Rubric) -> None:
    names = names_of(toy_rubric)
    stage1 = json.dumps(
        {
            names[0]: {"evaluable": True, "quotes": ["Everyone agreed."]},
            names[1]: {"evaluable": True, "quotes": [PASSAGE]},
            names[2]: {"evaluable": False, "quotes": []},
        }
    )
    backend = FakeBackend(ScriptedResponder([stage1, OK_SCORE]))
    result = run(backend, toy_rubric, TwoStageConfig(stage2_evidence="labelled"))
    user = backend.calls[1].user
    assert result.status == "ok"
    assert f'Marker: {names[0]}\n- "Everyone agreed."' in user
    assert f'Marker: {names[1]}\n- "{PASSAGE}"' in user
    assert f"Marker: {names[2]}" not in user


def test_token_counts_sum_over_every_call(toy_rubric: Rubric) -> None:
    answers = ["not json", evidence_json(names_of(toy_rubric)), OK_SCORE]
    backend = CountingBackend(answers, prompt_tokens=10, completion_tokens=7)
    result = run(backend, toy_rubric)
    assert result.status == "ok" and len(backend.calls) == 3
    assert (result.prompt_tokens, result.completion_tokens) == (30, 21)


def test_tokens_of_a_failed_extraction_are_counted(toy_rubric: Rubric) -> None:
    backend = CountingBackend(["x", "y", "z"], prompt_tokens=5, completion_tokens=2)
    result = run(backend, toy_rubric)
    assert result.status == "extraction_failed"
    assert (result.prompt_tokens, result.completion_tokens) == (15, 6)


def test_min_chars_boundary_counts_stripped_characters(toy_rubric: Rubric) -> None:
    ctx = make_ctx(toy_rubric)
    config = TwoStageConfig(min_chars=20)
    nineteen = "x" * 19
    twenty = "x" * 20
    short_backend = CountingBackend([])
    short = evaluate_attribute(
        short_backend,
        f"  {nineteen}  ",
        ctx,
        config,
        temperature_stage1=None,
        temperature_stage2=None,
        seed=1,
    )
    assert short.status == "too_short" and short_backend.calls == []
    long_backend = CountingBackend(["not json"] * 3)
    long = evaluate_attribute(
        long_backend,
        f"  {twenty}  ",
        ctx,
        config,
        temperature_stage1=None,
        temperature_stage2=None,
        seed=1,
    )
    assert long.status == "extraction_failed" and len(long_backend.calls) == 3


def test_a_repaired_parse_leaves_a_trace_in_the_log(
    toy_rubric: Rubric, caplog: pytest.LogCaptureFixture
) -> None:
    fenced = "```json\n" + evidence_json(names_of(toy_rubric)) + "\n```"
    backend = FakeBackend(ScriptedResponder([fenced, OK_SCORE]))
    with caplog.at_level(logging.INFO, logger="copernicus.coders.stage_runner"):
        result = run(backend, toy_rubric)
    assert result.status == "ok"
    messages = [record.getMessage() for record in caplog.records]
    assert len(messages) == 1
    attribute_id = toy_rubric.attributes[0].attribute_id
    assert attribute_id in messages[0] and "extraction stage" in messages[0]
    assert "attempt 1/3" in messages[0] and "strategy fence" in messages[0]


def test_a_direct_parse_is_not_logged(toy_rubric: Rubric, caplog: pytest.LogCaptureFixture) -> None:
    backend = FakeBackend(ScriptedResponder([evidence_json(names_of(toy_rubric)), OK_SCORE]))
    with caplog.at_level(logging.INFO, logger="copernicus.coders.stage_runner"):
        run(backend, toy_rubric)
    assert caplog.records == []
