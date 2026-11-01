"""Tests for the OpenAI-compatible backend."""

from __future__ import annotations

import hashlib
import logging
from types import SimpleNamespace

import pytest
from openai_support import StubClient, connection_error, rate_limit_error, status_error

from copernicus.errors import BackendError, SettingsError
from copernicus.llm import (
    Backend,
    BackendInfo,
    GenerationRequest,
    ModelCapabilities,
    OpenAICompatibleBackend,
)


def test_parameters_follow_capabilities(stub_client: StubClient) -> None:
    b = OpenAICompatibleBackend("gpt-4.1-mini", api_key="k", client=stub_client)
    r = b.generate(GenerationRequest("s", "u", 50, 0.3, 7))
    kw = stub_client.last_kwargs
    assert kw["temperature"] == 0.3 and kw["top_p"] == 0.9 and kw["seed"] == 7
    assert kw["max_tokens"] == 50
    assert kw["model"] == "gpt-4.1-mini"
    assert kw["messages"] == [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
    ]
    assert (r.temperature_applied, r.top_p_applied, r.seed_applied) == (0.3, 0.9, 7)
    assert (r.text, r.prompt_tokens, r.completion_tokens) == ('{"ok": true}', 12, 3)


def test_reasoning_model_omits_sampling_and_uses_completion_budget(
    stub_client: StubClient,
) -> None:
    b = OpenAICompatibleBackend("gpt-5-mini", api_key="k", client=stub_client)
    r = b.generate(GenerationRequest("s", "u", 50, 0.3, 7))
    kw = stub_client.last_kwargs
    assert "temperature" not in kw and "top_p" not in kw and kw["max_completion_tokens"] == 50
    assert "max_tokens" not in kw
    assert r.temperature_applied is None and r.top_p_applied is None
    assert r.seed_applied == 7


def test_greedy_sends_temperature_zero_without_top_p(stub_client: StubClient) -> None:
    b = OpenAICompatibleBackend("gpt-4.1-mini", api_key="k", client=stub_client)
    r = b.generate(GenerationRequest("s", "u", 50, 0.0, None))
    kw = stub_client.last_kwargs
    assert kw["temperature"] == 0.0 and "top_p" not in kw and "seed" not in kw
    assert (r.temperature_applied, r.top_p_applied, r.seed_applied) == (0.0, None, None)


def test_none_temperature_omits_sampling(stub_client: StubClient) -> None:
    b = OpenAICompatibleBackend("gpt-4.1-mini", api_key="k", client=stub_client)
    r = b.generate(GenerationRequest("s", "u", 50, None, 1))
    assert "temperature" not in stub_client.last_kwargs and "top_p" not in stub_client.last_kwargs
    assert r.temperature_applied is None and r.top_p_applied is None


def test_reasoning_effort_and_extra_body(stub_client: StubClient) -> None:
    caps = ModelCapabilities(supports_temperature=False, reasoning_effort="low")
    b = OpenAICompatibleBackend(
        "m", api_key="k", client=stub_client, capabilities=caps, extra_body={"top_k": 5}
    )
    b.generate(GenerationRequest("s", "u", 50, 0.3, None))
    kw = stub_client.last_kwargs
    assert kw["reasoning_effort"] == "low" and kw["extra_body"] == {"top_k": 5}
    assert "seed" not in kw


def test_missing_usage_and_empty_content() -> None:
    client = StubClient(
        respond=lambda kw: SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=None))], usage=None
        )
    )
    b = OpenAICompatibleBackend("m", api_key="k", client=client)
    r = b.generate(GenerationRequest("s", "u", 5, 0.0, None))
    assert r.text == "" and r.prompt_tokens is None and r.completion_tokens is None


def test_transport_error_retried_then_backend_error(
    stub_client_failing: StubClient, no_sleep: list[float]
) -> None:
    b = OpenAICompatibleBackend("m", api_key="k", client=stub_client_failing)
    with pytest.raises(BackendError) as excinfo:
        b.generate(GenerationRequest("s", "u", 5, 0.0, None))
    assert excinfo.value.attempts == 3
    assert len(stub_client_failing.calls) == 3
    assert no_sleep == [0.5, 1.0]


def test_transport_error_recovers(no_sleep: list[float]) -> None:
    client = StubClient(failures=[connection_error(), status_error(503)])
    b = OpenAICompatibleBackend("m", api_key="k", client=client)
    r = b.generate(GenerationRequest("s", "u", 5, 0.0, None))
    assert r.text == '{"ok": true}' and len(client.calls) == 3


def test_rate_limit_is_retried(no_sleep: list[float]) -> None:
    client = StubClient(failures=[rate_limit_error(), status_error(429)])
    b = OpenAICompatibleBackend("m", api_key="k", client=client)
    assert b.generate(GenerationRequest("s", "u", 5, 0.0, None)).text == '{"ok": true}'
    assert len(client.calls) == 3 and no_sleep == [0.5, 1.0]


def test_empty_choices_count_as_an_empty_answer(caplog: pytest.LogCaptureFixture) -> None:
    client = StubClient(respond=lambda kw: SimpleNamespace(choices=[], usage=None))
    b = OpenAICompatibleBackend("m", api_key="k", client=client)
    with caplog.at_level(logging.WARNING, logger="copernicus.llm.openai_compatible"):
        result = b.generate(GenerationRequest("s", "u", 5, 0.0, None))
    assert result.text == "" and result.prompt_tokens is None
    assert "no choices" in caplog.text


@pytest.mark.parametrize(
    ("field", "value"), [("transport_retries", 0), ("concurrency", 0), ("timeout", 0.0)]
)
def test_out_of_range_settings_are_refused(
    stub_client: StubClient, field: str, value: float
) -> None:
    with pytest.raises(SettingsError, match=f"{field} must be"):
        OpenAICompatibleBackend("m", api_key="k", client=stub_client, **{field: value})  # type: ignore[arg-type]


def test_client_error_is_not_retried(no_sleep: list[float]) -> None:
    client = StubClient(failures=[status_error(400)])
    b = OpenAICompatibleBackend("m", api_key="k", client=client)
    with pytest.raises(BackendError) as excinfo:
        b.generate(GenerationRequest("s", "u", 5, 0.0, None))
    assert excinfo.value.attempts == 1 and len(client.calls) == 1 and no_sleep == []


def test_missing_key_is_loud(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(BackendError, match="no API key"):
        OpenAICompatibleBackend("m")


def test_environment_key_is_used_to_build_the_client(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    b = OpenAICompatibleBackend("m", base_url="http://localhost:1/v1")
    assert b.info() == BackendInfo("m", "openai_compatible", None, None)


def test_batch_preserves_order() -> None:
    client = StubClient(
        respond=lambda kw: SimpleNamespace(
            choices=[
                SimpleNamespace(message=SimpleNamespace(content=kw["messages"][1]["content"]))
            ],
            usage=None,
        )
    )
    b = OpenAICompatibleBackend("m", api_key="k", client=client, concurrency=3)
    requests = [GenerationRequest("s", f"u{i}", 5, 0.0, None) for i in range(10)]
    assert [r.text for r in b.generate_batch(requests)] == [f"u{i}" for i in range(10)]


def test_batch_propagates_errors(no_sleep: list[float]) -> None:
    client = StubClient(failures=[status_error(400)])
    b = OpenAICompatibleBackend("m", api_key="k", client=client, concurrency=1)
    with pytest.raises(BackendError):
        b.generate_batch([GenerationRequest("s", "u", 5, 0.0, None)] * 2)


def test_satisfies_backend_protocol(stub_client: StubClient) -> None:
    backend: Backend = OpenAICompatibleBackend("m", api_key="k", client=stub_client)
    assert backend.info().backend == "openai_compatible"


def test_provenance_records_endpoint_capabilities_and_body(stub_client: StubClient) -> None:
    backend = OpenAICompatibleBackend(
        "gpt-5-mini",
        base_url="https://user:secret@LLM-host:8443/v1/?key=abc",
        api_key="k",
        client=stub_client,
        extra_body={"top_k": 5, "chat_template_kwargs": {"enable_thinking": False}},
    )
    provenance = backend.provenance()
    assert provenance == {
        "kind": "openai_compatible",
        "endpoint": "https://llm-host:8443",
        "endpoint_path_digest": hashlib.sha256(b"/v1").hexdigest()[:16],
        "model": "gpt-5-mini",
        "supports_temperature": "False",
        "supports_top_p": "False",
        "supports_seed": "True",
        "max_tokens_parameter": "max_completion_tokens",
        "reasoning_effort": "none",
        "top_p": "0.9",
        "extra_body": '{"chat_template_kwargs":{"enable_thinking":false},"top_k":5}',
    }
    assert not any(secret in value for value in provenance.values() for secret in ("secret", "abc"))


def test_provenance_of_the_default_endpoint_and_overrides(
    stub_client: StubClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    caps = ModelCapabilities(supports_seed=False, reasoning_effort="low")
    injected = OpenAICompatibleBackend("m", client=stub_client, capabilities=caps)
    assert injected.provenance()["endpoint"] == "https://api.openai.com"
    assert injected.provenance()["endpoint_path_digest"] == hashlib.sha256(b"/v1").hexdigest()[:16]
    assert injected.provenance()["reasoning_effort"] == "low"
    assert injected.provenance()["supports_seed"] == "False"
    assert injected.provenance()["extra_body"] == "{}"
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    built = OpenAICompatibleBackend("m")
    assert built.provenance()["endpoint"] == "https://api.openai.com"
    # The client ends its own base URL with a slash ("/v1/"); the digest drops trailing
    # slashes, so the built client and an injected one describe the same deployment.
    digest = built.provenance()["endpoint_path_digest"]
    assert digest == injected.provenance()["endpoint_path_digest"]
    assert digest == hashlib.sha256(b"/v1").hexdigest()[:16]
    local = OpenAICompatibleBackend("m", base_url="http://[::1]:8000/v1")
    assert local.provenance()["endpoint"] == "http://[::1]:8000"
    assert local.provenance()["endpoint_path_digest"] == hashlib.sha256(b"/v1").hexdigest()[:16]


@pytest.mark.parametrize("base_url", ["localhost:8000", "/v1", "http://host:port/v1"])
def test_base_url_without_a_usable_host_is_refused(stub_client: StubClient, base_url: str) -> None:
    with pytest.raises(BackendError, match="base_url") as error:
        OpenAICompatibleBackend("m", base_url=base_url, client=stub_client)
    assert base_url not in str(error.value)


def test_base_url_that_cannot_be_parsed_is_a_backend_error(stub_client: StubClient) -> None:
    with pytest.raises(BackendError, match="invalid base_url") as error:
        OpenAICompatibleBackend("m", base_url="http://[invalid", client=stub_client)
    assert "[invalid" not in str(error.value)
