"""A fake ``vllm`` module for the vLLM backend tests: the engine records what it is given."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

COMMIT = "0123456789abcdef0123456789abcdef01234567"


class FakeTokenizer:
    """Joins message contents with newlines; one token per whitespace-separated word."""

    def __init__(self) -> None:
        self.template_kwargs: dict[str, Any] = {}

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        tokenize: bool = False,
        add_generation_prompt: bool = True,
        **kwargs: Any,
    ) -> str:
        assert tokenize is False and add_generation_prompt is True
        self.template_kwargs = kwargs
        return "\n".join(message["content"] for message in messages)

    def encode(self, text: str) -> list[int]:
        return list(range(len(text.split())))


class FakeSamplingParams:
    """Stores its keyword arguments as attributes."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        for name, value in kwargs.items():
            setattr(self, name, value)


class FakeLLM:
    """Records constructor keyword arguments and every ``generate`` call."""

    instances: list[FakeLLM] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.tokenizer = FakeTokenizer()
        self.calls: list[tuple[list[str], Any]] = []
        FakeLLM.instances.append(self)

    def get_tokenizer(self) -> FakeTokenizer:
        return self.tokenizer

    def generate(self, prompts: list[str], params: Any) -> list[SimpleNamespace]:
        self.calls.append((prompts, params))
        return [
            SimpleNamespace(outputs=[SimpleNamespace(text=' {"ok": true} ', token_ids=[1, 2, 3])])
            for _ in prompts
        ]


def install_fake_vllm(monkeypatch: pytest.MonkeyPatch) -> type[FakeLLM]:
    """Install a fake ``vllm`` module and return its ``LLM`` class, with no instances yet."""
    FakeLLM.instances = []
    module = ModuleType("vllm")
    module.LLM = FakeLLM  # type: ignore[attr-defined]
    module.SamplingParams = FakeSamplingParams  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "vllm", module)
    return FakeLLM
