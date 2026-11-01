"""Tests for the backend protocol helpers and the fake backend."""

from __future__ import annotations

import pytest

from copernicus.errors import BackendError
from copernicus.llm import (
    Backend,
    BackendInfo,
    FakeBackend,
    GenerationRequest,
    ScriptedResponder,
    capabilities_for_model,
)


def test_capabilities_table() -> None:
    assert capabilities_for_model("gpt-4.1-mini").supports_temperature is True
    caps = capabilities_for_model("gpt-5-mini")
    assert caps.supports_temperature is False
    assert caps.max_tokens_parameter == "max_completion_tokens"


@pytest.mark.parametrize("model", ["gpt-5", "GPT-5-mini", "o1-preview", "o3", "o4-mini"])
def test_capabilities_restricted_models(model: str) -> None:
    caps = capabilities_for_model(model)
    assert caps.supports_temperature is False
    assert caps.supports_top_p is False
    assert caps.supports_seed is True
    assert caps.max_tokens_parameter == "max_completion_tokens"


def test_capabilities_defaults_for_other_models() -> None:
    caps = capabilities_for_model("llama-3.1-8b")
    assert caps.supports_temperature and caps.supports_top_p and caps.supports_seed
    assert caps.max_tokens_parameter == "max_tokens"
    assert caps.reasoning_effort is None


def test_scripted_fake_records_calls_and_exhausts() -> None:
    backend = FakeBackend(ScriptedResponder(["one"]))
    result = backend.generate(GenerationRequest("s", "u", 10, 0.3, 42))
    assert result.text == "one"
    assert backend.calls[0].seed == 42
    with pytest.raises(BackendError) as error:
        backend.generate(GenerationRequest("s", "u", 10, 0.3, 43))
    assert error.value.attempts == 1


def test_fake_result_fields() -> None:
    backend = FakeBackend(ScriptedResponder(["a b c"]))
    result = backend.generate(GenerationRequest("sys one", "user two three", 10, 0.3, 7))
    assert result.prompt_tokens == 5
    assert result.completion_tokens == 3
    assert result.temperature_applied == 0.3
    assert result.top_p_applied == 0.9
    assert result.seed_applied == 7


def test_fake_greedy_request_has_no_top_p() -> None:
    backend = FakeBackend(ScriptedResponder(["x"]))
    result = backend.generate(GenerationRequest("s", "u", 10, None, None))
    assert result.temperature_applied is None
    assert result.top_p_applied is None
    assert result.seed_applied is None


def test_fake_batch_keeps_order_and_info() -> None:
    backend = FakeBackend(ScriptedResponder(["a", "b"]), model_name="m")
    requests = [GenerationRequest("s", "u1", 5, 0.0, 1), GenerationRequest("s", "u2", 5, 0.0, 2)]
    results = backend.generate_batch(requests)
    assert [r.text for r in results] == ["a", "b"]
    assert backend.calls == requests
    assert backend.info() == BackendInfo("m", "fake", None, None)


def test_fake_backend_satisfies_protocol() -> None:
    backend: Backend = FakeBackend(ScriptedResponder([]))
    assert backend.info().backend == "fake"


def test_fake_provenance_names_kind_model_and_capabilities() -> None:
    backend = FakeBackend(ScriptedResponder([]), model_name="m")
    assert backend.provenance() == {
        "kind": "fake",
        "model": "m",
        "supports_temperature": "True",
        "supports_top_p": "True",
        "supports_seed": "True",
        "max_tokens_parameter": "max_tokens",
        "reasoning_effort": "none",
        "top_p": "0.9",
    }
