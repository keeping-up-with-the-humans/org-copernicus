"""Shared parts of the smoke-script tests: the loaded script, its command line and fake backends.

The name differs from ``examples/smoke_support.py``, which the script imports and which the
test run must not shadow.
"""

from __future__ import annotations

import importlib.util
import re
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest

import copernicus as cop
from copernicus.llm import FakeBackend, GenerationRequest, GenerationResult, Responder

LIBRARY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = LIBRARY_ROOT / "examples" / "smoke_two_stage.py"
KEY_SENTINEL = "sk-test-sentinel-0123456789"
PRICES = ["--price-input", "0.40", "--price-output", "1.60"]
ENTRY_FIELDS = [
    "timestamp",
    "backend",
    "texts",
    "runs",
    "prompt_tokens",
    "completion_tokens",
    "answers_without_usage",
    "price_input",
    "price_output",
    "cost_usd",
]

NUMBER_FIELDS = ENTRY_FIELDS[2:]
HTTP_LOGGERS = ("openai", "httpx", "httpx2", "httpcore")


def load_script() -> ModuleType:
    """Load ``examples/smoke_two_stage.py`` as a module; ``smoke_support`` must be importable."""
    specification = importlib.util.spec_from_file_location("smoke_two_stage_script", SCRIPT)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def script_fixture() -> Callable[..., ModuleType]:
    """Return the ``script`` fixture; each test module binds it by ``script = script_fixture()``."""

    @pytest.fixture
    def script(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
        """Return the script, with a key in the environment that the tests must never see leak."""
        monkeypatch.setenv("OPENAI_API_KEY", KEY_SENTINEL)
        monkeypatch.syspath_prepend(str(SCRIPT.parent))
        return load_script()

    return script


def arguments(out: Path, *more: str, ceiling: str = "20") -> list[str]:
    """Build a command line with the fake backend and the given prices."""
    return ["--out", str(out), *PRICES, "--ceiling", ceiling, "--fake", *more]


def printed_estimate(text: str) -> float:
    """Read the estimated cost from the line that states the formula."""
    match = re.search(
        r"estimate \(USD\) = \((\d+) x ([\d.]+) \+ (\d+) x ([\d.]+)\) / 1000000 = ([\d.]+)", text
    )
    assert match is not None, text
    prompt, price_input, completion, price_output, cost = match.groups()
    recomputed = (int(prompt) * float(price_input) + int(completion) * float(price_output)) / 1e6
    assert float(cost) == pytest.approx(recomputed, abs=5e-5)
    return float(cost)


def fake_usage(fake: FakeBackend) -> tuple[int, int]:
    """Return the prompt and completion tokens that the fake backend reported over its calls."""
    prompt = sum(len(call.system.split()) + len(call.user.split()) for call in fake.calls)
    return prompt, sum(len(fake.responses[index].split()) for index in range(len(fake.calls)))


class RecordingFake(FakeBackend):
    """A fake backend that remembers the text of each answer, to add up the tokens it reported."""

    def __init__(self, responder: Responder) -> None:
        super().__init__(responder)
        self.responses: list[str] = []

    def generate(self, request: GenerationRequest) -> GenerationResult:
        result = super().generate(request)
        self.responses.append(result.text)
        return result


def toy_responder() -> Responder:
    """Return the cue responder of the toy rubric."""
    rubric = cop.load_rubric(cop.rubric.loader.toy_rubric_path())
    return cop.cue_responder(rubric, cop.assets.load_toy_cues())
