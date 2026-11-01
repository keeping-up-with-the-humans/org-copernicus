"""The metering of the smoke-test script: batches, missing usage numbers and ledger failures."""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pandas as pd
import pytest
from script_rules import check_script_rules
from smoke_test_support import (
    SCRIPT,
    arguments,
    script_fixture,
    toy_responder,
)

import copernicus as cop
from copernicus.errors import BackendError
from copernicus.llm import (
    FakeBackend,
    GenerationRequest,
    GenerationResult,
    Responder,
)

script = script_fixture()


class SpyBackend(FakeBackend):
    """A fake backend that records how it was called and can leave out usage numbers."""

    def __init__(self, responder: Responder, *, drop: str = "") -> None:
        super().__init__(responder)
        self.batch_calls = 0
        self.single_calls = 0
        self.drop = drop

    def generate(self, request: GenerationRequest) -> GenerationResult:
        self.single_calls += 1
        result = super().generate(request)
        if self.drop == "both":
            return dataclasses.replace(result, prompt_tokens=None, completion_tokens=None)
        if self.drop == "completion":
            return dataclasses.replace(result, completion_tokens=None)
        return result

    def generate_batch(self, requests: Any) -> list[GenerationResult]:
        self.batch_calls += 1
        return super().generate_batch(requests)


def one_stage_request() -> GenerationRequest:
    """Return a request that the cue responder answers as a one-stage call."""
    rubric = cop.load_rubric(cop.rubric.loader.toy_rubric_path())
    attribute = rubric.attributes[0].name
    return GenerationRequest(
        f"single integrated evaluator, evaluating the attribute: {attribute}\n",
        "Text to analyze:\nthank you for the clear report",
        10,
        0.0,
        1,
    )


def test_the_metered_backend_answers_a_batch_one_request_at_a_time_and_meters_each(
    script: ModuleType,
) -> None:
    spy = SpyBackend(toy_responder())
    metered = script.MeteredBackend(spy)
    request = one_stage_request()
    results = metered.generate_batch([request, request])
    assert (spy.batch_calls, spy.single_calls, len(results)) == (0, 2, 2)
    assert metered.prompt_tokens == sum(r.prompt_tokens for r in results)
    assert metered.completion_tokens == sum(r.completion_tokens for r in results)
    assert metered.answers_without_usage == 0


def test_a_batch_that_fails_part_way_keeps_the_answers_before_the_failure_metered(
    script: ModuleType,
) -> None:
    answer = toy_responder()
    calls = 0

    def responder(request: GenerationRequest) -> str:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise BackendError("the endpoint refused", attempts=1)
        return answer(request)

    metered = script.MeteredBackend(FakeBackend(responder))
    request = one_stage_request()
    with pytest.raises(BackendError, match="the endpoint refused"):
        metered.generate_batch([request, request, request])
    assert calls == 2
    assert metered.prompt_tokens > 0 and metered.completion_tokens > 0


def test_answers_without_usage_are_counted_recorded_and_fail_the_run_loudly(
    script: ModuleType,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spy = SpyBackend(toy_responder(), drop="both")
    monkeypatch.setattr(script, "build_backend", lambda options, rubric: spy)
    out = tmp_path / "smoke"
    status = script.main(arguments(out, "--texts", "12", "--runs", "1", "--n-boot", "20"))
    captured = capsys.readouterr()
    assert status == 1
    assert spy.single_calls > 0
    entry = json.loads((out / "spend.json").read_text(encoding="utf-8"))["entries"][0]
    assert entry["answers_without_usage"] == spy.single_calls
    assert entry["prompt_tokens"] == entry["completion_tokens"] == 0
    assert f"answers without usage: {spy.single_calls}" in captured.out
    assert str(spy.single_calls) in captured.err and "usage" in captured.err
    assert (out / "smoke_report.txt").stat().st_size > 0
    assert len(pd.read_csv(out / "two_stage_text_scores.csv")) == 12


def test_an_answer_missing_only_one_count_is_counted_once_and_keeps_the_other(
    script: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spy = SpyBackend(toy_responder(), drop="completion")
    monkeypatch.setattr(script, "build_backend", lambda options, rubric: spy)
    out = tmp_path / "smoke"
    assert script.main(arguments(out, "--texts", "12", "--runs", "1", "--n-boot", "20")) == 1
    entry = json.loads((out / "spend.json").read_text(encoding="utf-8"))["entries"][0]
    assert entry["answers_without_usage"] == spy.single_calls
    assert entry["prompt_tokens"] > 0 and entry["completion_tokens"] == 0


def test_a_run_with_complete_usage_reports_zero_answers_without_usage(
    script: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "smoke"
    assert script.main(arguments(out, "--texts", "12", "--runs", "1", "--n-boot", "20")) == 0
    assert "answers without usage: 0" in capsys.readouterr().out
    entry = json.loads((out / "spend.json").read_text(encoding="utf-8"))["entries"][0]
    assert entry["answers_without_usage"] == 0


def test_a_ledger_failure_during_an_error_is_raised_from_the_original_error(
    script: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def responder(request: GenerationRequest) -> str:
        raise BackendError("the endpoint refused", attempts=1)

    monkeypatch.setattr(script, "build_backend", lambda options, rubric: FakeBackend(responder))

    def broken_ledger(path: Path, entry: dict[str, Any]) -> float:
        raise OSError("disk full")

    monkeypatch.setattr(script, "append_spend", broken_ledger)
    with pytest.raises(OSError, match="disk full") as stopped:
        script.main(arguments(tmp_path / "smoke", "--texts", "12", "--runs", "1"))
    cause = stopped.value.__cause__
    assert isinstance(cause, BackendError) and "the endpoint refused" in str(cause)


def test_a_ledger_failure_without_another_error_is_raised_plainly(
    script: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken_ledger(path: Path, entry: dict[str, Any]) -> float:
        raise OSError("disk full")

    monkeypatch.setattr(script, "append_spend", broken_ledger)
    with pytest.raises(OSError, match="disk full") as stopped:
        script.main(arguments(tmp_path / "smoke", "--texts", "12", "--runs", "1"))
    assert stopped.value.__cause__ is None
    assert sys.exc_info()[1] is None


def test_the_script_and_its_support_module_use_only_public_names_and_stay_short() -> None:
    check_script_rules(SCRIPT.parent, ("smoke_two_stage.py", "smoke_support.py"))
