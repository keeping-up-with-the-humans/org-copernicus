import json
import logging
from typing import Any

import pytest
from two_stage_support import DOC, CountingBackend, make_ctx

from copernicus.coders import (
    OneStageConfig,
    evaluate_attribute_one_stage,
    render_one_stage_prompt,
    validate_one_stage_output,
)
from copernicus.errors import BackendError, SettingsError
from copernicus.llm.fake import FakeBackend, ScriptedResponder
from copernicus.rubric.model import Rubric

TEMPERATURE = 0.3


def wrapped(result: Any, **extra: Any) -> str:
    return json.dumps({"dimension_result": {"result": result, **extra}})


def run_one_stage(
    backend: Any,
    rubric: Rubric,
    config: OneStageConfig | None = None,
    text: str = DOC,
    seed: int = 10,
) -> Any:
    return evaluate_attribute_one_stage(
        backend,
        text,
        make_ctx(rubric),
        config if config is not None else OneStageConfig(),
        temperature=TEMPERATURE,
        seed=seed,
    )


def test_ok_path_sends_one_call_with_the_one_stage_prompt(toy_rubric: Rubric) -> None:
    backend = FakeBackend(ScriptedResponder([wrapped(3, explanation="partly there")]))
    result = run_one_stage(backend, toy_rubric)
    assert result.status == "ok" and result.score == 3
    assert result.explanation == "partly there"
    assert result.attribute_id == toy_rubric.attributes[0].attribute_id
    assert len(backend.calls) == 1
    call = backend.calls[0]
    assert call.system == render_one_stage_prompt(make_ctx(toy_rubric))
    assert "single integrated evaluator" in call.system
    assert call.user.startswith("Text to analyze:") and call.user == f"Text to analyze:\n{DOC}"
    assert (call.max_new_tokens, call.temperature, call.seed) == (2048, TEMPERATURE, 10)


def test_ok_path_records_the_call_as_the_scoring_call(toy_rubric: Rubric) -> None:
    backend = FakeBackend(ScriptedResponder(["not json", wrapped(4)]))
    result = run_one_stage(backend, toy_rubric)
    assert result.status == "ok" and result.score == 4
    assert result.stage1_attempt is None and result.stage2_attempt == 1
    assert result.evidence == () and result.evaluable == {} and result.justification == {}
    assert result.seconds >= 0


def test_a_missing_explanation_becomes_the_empty_string(toy_rubric: Rubric) -> None:
    backend = FakeBackend(ScriptedResponder([wrapped(2)]))
    result = run_one_stage(backend, toy_rubric)
    assert result.status == "ok" and result.score == 2 and result.explanation == ""


def test_out_of_range_three_times_fails_with_the_last_reason(toy_rubric: Rubric) -> None:
    backend = FakeBackend(ScriptedResponder([wrapped(7), wrapped("high"), wrapped(0)]))
    result = run_one_stage(backend, toy_rubric, seed=40)
    assert result.status == "scoring_failed" and result.score is None
    assert result.explanation == "scoring_failed: 'result' out of range 1..5: 0"
    assert result.stage1_attempt is None and result.stage2_attempt is None
    assert result.evidence == () and result.evaluable == {} and result.justification == {}
    assert [call.seed for call in backend.calls] == [40, 41, 42]
    assert [call.temperature for call in backend.calls] == [TEMPERATURE] * 3


def test_unparseable_output_fails_with_the_parse_reason(toy_rubric: Rubric) -> None:
    backend = FakeBackend(ScriptedResponder(["x", "y", "z"]))
    result = run_one_stage(backend, toy_rubric)
    assert result.status == "scoring_failed"
    assert result.explanation == "scoring_failed: json_parse_failure"


def test_a_result_without_the_wrapper_is_rejected(toy_rubric: Rubric) -> None:
    backend = FakeBackend(ScriptedResponder([json.dumps({"result": 3})] * 3))
    result = run_one_stage(backend, toy_rubric)
    assert result.status == "scoring_failed" and result.score is None
    assert result.explanation == "scoring_failed: missing or non-object dimension_result"
    assert len(backend.calls) == 3


def test_max_retries_bounds_the_calls(toy_rubric: Rubric) -> None:
    backend = FakeBackend(ScriptedResponder([wrapped(9)] * 5))
    result = run_one_stage(backend, toy_rubric, OneStageConfig(max_retries=2))
    assert result.status == "scoring_failed" and len(backend.calls) == 2


def test_max_new_tokens_comes_from_the_config(toy_rubric: Rubric) -> None:
    backend = FakeBackend(ScriptedResponder([wrapped(3)]))
    run_one_stage(backend, toy_rubric, OneStageConfig(max_new_tokens=512))
    assert backend.calls[0].max_new_tokens == 512


def test_a_none_temperature_stays_none_on_every_attempt(toy_rubric: Rubric) -> None:
    backend = FakeBackend(ScriptedResponder([wrapped(9), wrapped(3)]))
    result = evaluate_attribute_one_stage(
        backend, DOC, make_ctx(toy_rubric), OneStageConfig(), temperature=None, seed=1
    )
    assert result.status == "ok"
    assert [call.temperature for call in backend.calls] == [None, None]


def test_too_short_text_is_not_sent(toy_rubric: Rubric) -> None:
    backend = FakeBackend(ScriptedResponder([]))
    result = run_one_stage(backend, toy_rubric, text="  too short  ")
    assert result.status == "too_short" and result.score is None
    assert result.explanation == "too_short" and backend.calls == []
    assert result.stage1_attempt is None and result.stage2_attempt is None
    assert result.evidence == () and result.evaluable == {} and result.justification == {}
    assert result.prompt_tokens == 0 and result.completion_tokens == 0


def test_min_chars_counts_the_stripped_text(toy_rubric: Rubric) -> None:
    config = OneStageConfig(min_chars=5)
    short = run_one_stage(FakeBackend(ScriptedResponder([])), toy_rubric, config, text="  abcd  ")
    assert short.status == "too_short"
    backend = FakeBackend(ScriptedResponder([wrapped(3)]))
    assert run_one_stage(backend, toy_rubric, config, text="  abcde  ").status == "ok"


def test_backend_error_from_generate_propagates(toy_rubric: Rubric) -> None:
    backend = FakeBackend(ScriptedResponder([]))
    with pytest.raises(BackendError, match="exhausted"):
        run_one_stage(backend, toy_rubric)


def test_token_counts_are_summed_and_none_counts_as_zero(toy_rubric: Rubric) -> None:
    counted = FakeBackend(ScriptedResponder(["nope", wrapped(3)]))
    result = run_one_stage(counted, toy_rubric)
    expected_prompt = sum(len(c.system.split()) + len(c.user.split()) for c in counted.calls)
    assert result.prompt_tokens == expected_prompt and result.completion_tokens > 0
    silent = CountingBackend(["nope", wrapped(3)])
    assert run_one_stage(silent, toy_rubric).prompt_tokens == 0


def test_every_failed_attempt_is_logged(
    toy_rubric: Rubric, caplog: pytest.LogCaptureFixture
) -> None:
    backend = FakeBackend(ScriptedResponder(["not json at all", wrapped(7), wrapped(3)]))
    with caplog.at_level(logging.INFO, logger="copernicus.coders"):
        run_one_stage(backend, toy_rubric, seed=7)
    messages = [record.getMessage() for record in caplog.records]
    assert len(messages) == 2
    attribute_id = toy_rubric.attributes[0].attribute_id
    assert all(attribute_id in message for message in messages)
    assert "attempt 1/3" in messages[0] and "json_parse_failure" in messages[0]
    assert "seed=7" in messages[0] and "'not json at all'" in messages[0]
    assert "attempt 2/3" in messages[1] and "out of range" in messages[1]
    assert "seed=8" in messages[1]


def test_a_repaired_parse_is_logged(toy_rubric: Rubric, caplog: pytest.LogCaptureFixture) -> None:
    backend = FakeBackend(ScriptedResponder(["```json\n" + wrapped(3) + "\n```"]))
    with caplog.at_level(logging.INFO, logger="copernicus.coders"):
        result = run_one_stage(backend, toy_rubric)
    assert result.status == "ok"
    messages = [record.getMessage() for record in caplog.records]
    assert len(messages) == 1 and "strategy fence" in messages[0]


@pytest.mark.parametrize(
    ("parsed", "valid", "reason", "score", "explanation"),
    [
        ({"dimension_result": {"result": 3}}, True, "", 3, ""),
        ({"dimension_result": {"result": "4", "explanation": "why"}}, True, "", 4, "why"),
        ({"dimension_result": {"result": 4.0}}, True, "", 4, ""),
        ({"dimension_result": {"result": 2, "explanation": None}}, True, "", 2, ""),
        ({"dimension_result": {"result": True}}, False, "'result' is a boolean", None, ""),
        (
            {"dimension_result": {"result": 6}},
            False,
            "'result' out of range 1..5: 6",
            None,
            "",
        ),
        ({"dimension_result": {}}, False, "missing 'result'", None, ""),
        (
            {"dimension_result": {"result": 3, "explanation": 5}},
            False,
            "'explanation' is not a string",
            None,
            "",
        ),
        ({"result": 3}, False, "missing or non-object dimension_result", None, ""),
        ({"dimension_result": 3}, False, "missing or non-object dimension_result", None, ""),
        ({"dimension_result": [3]}, False, "missing or non-object dimension_result", None, ""),
        ([3], False, "top-level type list is not a JSON object", None, ""),
        ("3", False, "top-level type str is not a JSON object", None, ""),
    ],
)
def test_validate_one_stage_output(
    parsed: Any, valid: bool, reason: str, score: int | None, explanation: str
) -> None:
    assert validate_one_stage_output(parsed) == (valid, reason, score, explanation)


def test_config_defaults_are_accepted() -> None:
    config = OneStageConfig()
    assert (config.max_retries, config.max_new_tokens, config.min_chars) == (3, 2048, 20)


@pytest.mark.parametrize(
    ("field", "value"), [("max_retries", 0), ("max_new_tokens", 0), ("min_chars", -1)]
)
def test_config_rejects_out_of_range_settings(field: str, value: int) -> None:
    with pytest.raises(SettingsError, match=f"{field} must be .*{value}"):
        OneStageConfig(**{field: value})


def test_config_is_frozen() -> None:
    config = OneStageConfig()
    with pytest.raises(AttributeError):
        config.max_retries = 5  # type: ignore[misc]
