"""Tests for the in-process vLLM backend, run against an injected fake ``vllm`` module."""

from __future__ import annotations

import logging
import re
import sys
import threading
import time
from types import SimpleNamespace
from typing import Any

import pytest
from vllm_support import COMMIT, FakeLLM, install_fake_vllm

from copernicus.errors import BackendError
from copernicus.llm import Backend, BackendInfo, GenerationRequest, VLLMBackend, clamp_max_tokens

OTHER_COMMIT = "fedcba9876543210fedcba9876543210fedcba98"
VLLM_HINT = (
    'install the vllm extra: pip install "org-copernicus[vllm]", '
    'or pip install ".[vllm]" from the package\'s root folder'
)
NO_COMMIT = (
    "pins no commit: set revision to the 40-character commit hash whose code you have reviewed"
)


@pytest.fixture()
def fake_vllm(monkeypatch: pytest.MonkeyPatch) -> type[FakeLLM]:
    """Install a fake ``vllm`` module and return its ``LLM`` class."""
    return install_fake_vllm(monkeypatch)


def test_clamp_max_tokens() -> None:
    assert clamp_max_tokens(8192, 1000, 32768) == 8192
    assert clamp_max_tokens(8192, 32000, 32768) == 760
    assert clamp_max_tokens(8192, 32768, 32768) == 1


def test_sampling_params_and_dtype_for_awq(fake_vllm: type[FakeLLM]) -> None:
    backend = VLLMBackend("org/Llama-3-70B-AWQ")
    kwargs = fake_vllm.instances[0].kwargs
    assert kwargs["quantization"] == "awq" and kwargs["dtype"] == "float16"
    assert kwargs["model"] == "org/Llama-3-70B-AWQ" and kwargs["trust_remote_code"] is False
    assert kwargs["max_model_len"] == 32768 and kwargs["tensor_parallel_size"] == 1
    assert kwargs["gpu_memory_utilization"] == 0.90
    result = backend.generate(GenerationRequest("sys", "two words", 100, 0.3, 7))
    prompts, params = fake_vllm.instances[0].calls[0]
    assert prompts == ["sys\ntwo words"]
    assert params[0].kwargs == {"max_tokens": 100, "temperature": 0.3, "top_p": 0.9, "seed": 7}
    assert result.text == '{"ok": true}'
    assert (result.prompt_tokens, result.completion_tokens) == (3, 3)
    assert (result.temperature_applied, result.top_p_applied, result.seed_applied) == (0.3, 0.9, 7)
    assert backend.info() == BackendInfo("org/Llama-3-70B-AWQ", "vllm", "awq", 32768)


def test_greedy_sampling_has_no_top_p_or_seed(fake_vllm: type[FakeLLM]) -> None:
    backend = VLLMBackend("plain-model")
    for temperature in (0.0, None):
        result = backend.generate(GenerationRequest("s", "u", 50, temperature, None))
        params = fake_vllm.instances[0].calls[-1][1]
        assert params[0].kwargs == {"max_tokens": 50, "temperature": 0.0}
        assert (result.temperature_applied, result.top_p_applied, result.seed_applied) == (
            0.0,
            None,
            None,
        )


def test_quantization_inference_and_overrides(fake_vllm: type[FakeLLM]) -> None:
    assert VLLMBackend("a/model-GPTQ-int4").info().quantization == "gptq"
    assert fake_vllm.instances[-1].kwargs["dtype"] == "float16"
    assert VLLMBackend("a/model-fp8").info().quantization == "fp8"
    assert "dtype" not in fake_vllm.instances[-1].kwargs
    plain = VLLMBackend("a/model")
    assert plain.info().quantization is None
    assert "quantization" not in fake_vllm.instances[-1].kwargs
    explicit = VLLMBackend("a/model", quantization="awq", dtype="bfloat16")
    assert explicit.info().quantization == "awq"
    assert fake_vllm.instances[-1].kwargs["dtype"] == "bfloat16"


def test_engine_options_and_chat_template_kwargs(fake_vllm: type[FakeLLM]) -> None:
    backend = VLLMBackend(
        "m",
        kv_cache_dtype="fp8",
        chat_template_kwargs={"enable_thinking": False},
        engine_kwargs={"enforce_eager": True, "max_model_len": 4096},
    )
    kwargs = fake_vllm.instances[0].kwargs
    assert kwargs["kv_cache_dtype"] == "fp8" and kwargs["enforce_eager"] is True
    assert kwargs["max_model_len"] == 4096
    assert "chat_template_kwargs" not in kwargs
    backend.generate(GenerationRequest("s", "u", 5, 0.0, None))
    assert fake_vllm.instances[0].tokenizer.template_kwargs == {"enable_thinking": False}


def test_the_log_names_the_engine_options_but_never_their_values(
    fake_vllm: type[FakeLLM], caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO, logger="copernicus.llm.vllm_backend"):
        VLLMBackend("m", engine_kwargs={"hf_token": "hf_SECRETVALUE", "enforce_eager": True})
    assert "hf_token" in caplog.text and "enforce_eager" in caplog.text
    assert "hf_SECRETVALUE" not in caplog.text


def test_over_long_prompt_raises(fake_vllm: type[FakeLLM]) -> None:
    backend = VLLMBackend("m")
    request = GenerationRequest("s", " ".join(["w"] * 40000), 10, 0.0, None)
    with pytest.raises(BackendError, match=r"prompt of 40001 tokens exceeds the context window"):
        backend.generate(request)
    assert fake_vllm.instances[0].calls == []


def test_prompt_budget_is_clamped_to_the_window(fake_vllm: type[FakeLLM]) -> None:
    backend = VLLMBackend("m", max_model_len=100)
    backend.generate(GenerationRequest("s", " ".join(["w"] * 49), 8192, 0.0, None))
    assert fake_vllm.instances[0].calls[0][1][0].kwargs["max_tokens"] == 100 - 50 - 8


def test_batch_uses_per_request_seeds(fake_vllm: type[FakeLLM]) -> None:
    backend = VLLMBackend("m", max_model_len=100)
    requests = [
        GenerationRequest("s", "a b", 8192, 0.5, 1),
        GenerationRequest("s", "c d e f", 20, 0.5, 2),
        GenerationRequest("s", "g", 20, 0.0, None),
    ]
    results = backend.generate_batch(requests)
    assert len(fake_vllm.instances[0].calls) == 1
    prompts, params_list = fake_vllm.instances[0].calls[0]
    assert prompts == ["s\na b", "s\nc d e f", "s\ng"]
    assert [p.kwargs for p in params_list] == [
        {"max_tokens": 100 - 3 - 8, "temperature": 0.5, "top_p": 0.9, "seed": 1},
        {"max_tokens": 20, "temperature": 0.5, "top_p": 0.9, "seed": 2},
        {"max_tokens": 20, "temperature": 0.0},
    ]
    assert [r.seed_applied for r in results] == [1, 2, None]
    assert all(r.text == '{"ok": true}' for r in results)


def test_batch_with_an_over_long_prompt_generates_nothing(fake_vllm: type[FakeLLM]) -> None:
    backend = VLLMBackend("m", max_model_len=50)
    requests = [
        GenerationRequest("s", "short", 10, 0.0, None),
        GenerationRequest("s", " ".join(["w"] * 60), 10, 0.0, None),
    ]
    with pytest.raises(BackendError, match=r"request 1: prompt of 61 tokens"):
        backend.generate_batch(requests)
    assert fake_vllm.instances[0].calls == []


def test_missing_token_ids_leave_completion_tokens_unknown(
    fake_vllm: type[FakeLLM], monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = VLLMBackend("m")
    monkeypatch.setattr(
        fake_vllm,
        "generate",
        lambda self, prompts, params: [SimpleNamespace(outputs=[SimpleNamespace(text="x")])],
    )
    result = backend.generate(GenerationRequest("s", "u", 5, 0.0, None))
    assert result.completion_tokens is None and result.text == "x"


def test_concurrent_calls_never_overlap_in_the_engine(
    fake_vllm: type[FakeLLM], monkeypatch: pytest.MonkeyPatch
) -> None:
    active = 0
    overlaps: list[int] = []
    guard = threading.Lock()
    original = FakeLLM.generate

    def slow_generate(self: FakeLLM, prompts: list[str], params: Any) -> list[SimpleNamespace]:
        nonlocal active
        with guard:
            active += 1
            overlaps.append(active)
        time.sleep(0.02)
        try:
            return original(self, prompts, params)
        finally:
            with guard:
                active -= 1

    monkeypatch.setattr(FakeLLM, "generate", slow_generate)
    backend = VLLMBackend("m")
    threads = [
        threading.Thread(
            target=backend.generate, args=(GenerationRequest("s", f"u{i}", 5, 0.0, 1),)
        )
        for i in range(6)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(overlaps) == 6 and max(overlaps) == 1


def test_satisfies_backend_protocol(fake_vllm: type[FakeLLM]) -> None:
    backend: Backend = VLLMBackend("m")
    assert backend.info().backend == "vllm"


def test_remote_code_runs_only_when_asked(fake_vllm: type[FakeLLM]) -> None:
    assert VLLMBackend("m").provenance()["trust_remote_code"] == "False"
    trusted = VLLMBackend("m", trust_remote_code=True, engine_kwargs={"revision": COMMIT})
    kwargs = fake_vllm.instances[-1].kwargs
    assert kwargs["trust_remote_code"] is True and kwargs["revision"] == COMMIT
    assert trusted.provenance()["trust_remote_code"] == "True"


def test_remote_code_requires_a_pinned_revision(fake_vllm: type[FakeLLM]) -> None:
    with pytest.raises(BackendError, match=NO_COMMIT):
        VLLMBackend("m", trust_remote_code=True)
    with pytest.raises(BackendError, match=NO_COMMIT):
        VLLMBackend("m", trust_remote_code=True, engine_kwargs={"revision": None})
    assert fake_vllm.instances == []
    VLLMBackend("m", trust_remote_code=True, engine_kwargs={"revision": COMMIT})
    assert len(fake_vllm.instances) == 1


def test_an_empty_revision_counts_as_unpinned(fake_vllm: type[FakeLLM]) -> None:
    for revision in ("", "   "):
        with pytest.raises(BackendError, match=NO_COMMIT):
            VLLMBackend("m", trust_remote_code=True, engine_kwargs={"revision": revision})
    assert fake_vllm.instances == []


@pytest.mark.parametrize(
    "revision",
    ["main", COMMIT[:7], "v1.0", COMMIT.upper(), COMMIT + "0"],
    ids=["branch", "short-hash", "tag", "upper-case", "41-characters"],
)
def test_remote_code_refuses_a_branch_a_tag_or_a_short_hash(
    fake_vllm: type[FakeLLM], revision: str
) -> None:
    with pytest.raises(BackendError, match=NO_COMMIT):
        VLLMBackend("m", trust_remote_code=True, engine_kwargs={"revision": revision})
    assert fake_vllm.instances == []


def test_without_remote_code_a_branch_may_be_loaded(fake_vllm: type[FakeLLM]) -> None:
    VLLMBackend("m", engine_kwargs={"revision": "main"})
    assert fake_vllm.instances[-1].kwargs["revision"] == "main"


def test_the_tokenizer_is_pinned_with_the_model_when_remote_code_runs(
    fake_vllm: type[FakeLLM],
) -> None:
    VLLMBackend("m", trust_remote_code=True, engine_kwargs={"revision": COMMIT})
    assert fake_vllm.instances[-1].kwargs["tokenizer_revision"] == COMMIT
    VLLMBackend(
        "m",
        trust_remote_code=True,
        engine_kwargs={"revision": COMMIT, "tokenizer_revision": OTHER_COMMIT},
    )
    assert fake_vllm.instances[-1].kwargs["tokenizer_revision"] == OTHER_COMMIT


@pytest.mark.parametrize("tokenizer_revision", ["main", OTHER_COMMIT[:7]])
def test_a_tokenizer_revision_set_under_remote_code_must_be_a_commit_hash(
    fake_vllm: type[FakeLLM], tokenizer_revision: str
) -> None:
    with pytest.raises(BackendError, match="tokenizer_revision"):
        VLLMBackend(
            "m",
            trust_remote_code=True,
            engine_kwargs={"revision": COMMIT, "tokenizer_revision": tokenizer_revision},
        )
    assert fake_vllm.instances == []


def test_a_blank_tokenizer_revision_counts_as_absent_under_remote_code(
    fake_vllm: type[FakeLLM],
) -> None:
    for tokenizer_revision in (None, "", "   "):
        VLLMBackend(
            "m",
            trust_remote_code=True,
            engine_kwargs={"revision": COMMIT, "tokenizer_revision": tokenizer_revision},
        )
        assert fake_vllm.instances[-1].kwargs["tokenizer_revision"] == COMMIT


def test_no_tokenizer_revision_is_added_without_remote_code(fake_vllm: type[FakeLLM]) -> None:
    VLLMBackend("m", engine_kwargs={"revision": "abc"})
    assert "tokenizer_revision" not in fake_vllm.instances[-1].kwargs


def test_missing_vllm_is_explained(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "vllm", None)
    with pytest.raises(BackendError, match=re.escape(VLLM_HINT)):
        VLLMBackend("m")
