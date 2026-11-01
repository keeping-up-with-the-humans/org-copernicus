"""The smoke-test script estimates, refuses, runs and keeps a ledger, here on the fake model."""

from __future__ import annotations

import dataclasses
import json
import logging
import math
import re
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pandas as pd
import pytest
from smoke_test_support import (
    ENTRY_FIELDS,
    HTTP_LOGGERS,
    KEY_SENTINEL,
    NUMBER_FIELDS,
    PRICES,
    RecordingFake,
    arguments,
    fake_usage,
    printed_estimate,
    script_fixture,
    toy_responder,
)

import copernicus as cop
from copernicus.coders import VERIFIED_EXACT, VERIFIED_NONE
from copernicus.errors import BackendError, RunFailureCeilingError, SettingsError
from copernicus.llm import GenerationRequest

script = script_fixture()


def test_the_estimate_is_printed_with_its_formula_and_the_dry_run_writes_nothing(
    script: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "smoke"
    assert script.main(arguments(out, "--texts", "12", "--runs", "1", "--dry-run")) == 0
    printed = capsys.readouterr().out
    assert printed_estimate(printed) > 0
    for line in ("stage 1 (extraction)", "stage 2 (scoring)", "min(8192, 1.5 x text tokens + 200)"):
        assert line in printed
    assert "running total in spend.json (USD): 0.0000" in printed
    assert "token counter: characters / 4" in printed
    assert not out.exists()


def test_a_fake_run_counts_characters_and_a_real_one_tiktoken_where_it_is_installed(
    script: ModuleType,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded: list[str] = []

    def encoding(name: str) -> SimpleNamespace:
        loaded.append(name)
        return SimpleNamespace(encode=lambda text: text.split())

    monkeypatch.setitem(sys.modules, "tiktoken", SimpleNamespace(get_encoding=encoding))
    assert script.main(arguments(tmp_path / "fake", "--texts", "6", "--dry-run")) == 0
    assert "token counter: characters / 4" in capsys.readouterr().out and loaded == []
    real = ["--out", str(tmp_path / "real"), *PRICES, "--ceiling", "20", "--texts", "6"]
    assert script.main([*real, "--dry-run"]) == 0
    assert "token counter: tiktoken o200k_base" in capsys.readouterr().out and loaded


def test_the_printed_formula_names_the_configured_budget_of_the_extraction_answer(
    script: ModuleType,
) -> None:
    rubric = cop.load_rubric(cop.rubric.loader.toy_rubric_path())
    table = cop.TextTable.read(cop.assets.toy_corpus_path(), group_id="group_id")
    options = script.parse_arguments(["--out", "x", *PRICES, "--ceiling", "20", "--runs", "1"])
    settings = dataclasses.replace(
        script.coding_settings(options),
        two_stage=cop.TwoStageConfig(stage1_max_new_tokens=4096),
    )
    stages = script.estimate_run(table, rubric, script.first_texts(table, 6), settings, False)
    printed = script.format_estimate(stages, options, 0.0, settings)
    assert "completion per call = min(4096, 1.5 x text tokens + 200)" in printed
    assert "completion = 300" in printed


def test_the_http_loggers_stay_quiet_after_the_logging_setup(
    script: ModuleType, tmp_path: Path
) -> None:
    previous = {name: logging.getLogger(name).level for name in HTTP_LOGGERS}
    assert set(HTTP_LOGGERS) <= set(script.QUIET_LOGGERS)
    script.main(arguments(tmp_path / "smoke", "--texts", "6", "--runs", "1", "--dry-run"))
    levels = {name: logging.getLogger(name).level for name in HTTP_LOGGERS}
    for name, level in previous.items():
        logging.getLogger(name).setLevel(level)
    assert levels == dict.fromkeys(HTTP_LOGGERS, logging.WARNING)


def test_the_estimate_matches_the_stage_estimates_and_grows_with_texts_and_runs(
    script: ModuleType,
) -> None:
    rubric = cop.load_rubric(cop.rubric.loader.toy_rubric_path())
    table = cop.TextTable.read(cop.assets.toy_corpus_path(), group_id="group_id")

    def cost(n_texts: int, runs: int) -> float:
        options = script.parse_arguments(
            ["--out", "x", *PRICES, "--ceiling", "20", "--runs", str(runs)]
        )
        ids = script.first_texts(table, n_texts)
        settings = script.coding_settings(options)
        stages = script.estimate_run(table, rubric, ids, settings, not options.fake)
        assert stages[0].calls == stages[1].calls == n_texts * runs * len(rubric.attributes)
        return float(script.estimated_cost(stages, options))

    assert cost(12, 2) == pytest.approx(2 * cost(12, 1), rel=0.2)
    assert cost(24, 1) > cost(12, 1) > 0


def test_a_run_over_the_ceiling_is_refused_and_nothing_is_run(
    script: ModuleType,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*_: Any) -> None:
        raise AssertionError("no backend may be built after a refusal")

    monkeypatch.setattr(script, "build_backend", forbidden)
    out = tmp_path / "smoke"
    assert script.main(arguments(out, "--texts", "12", "--ceiling", "0.0001")) == 1
    captured = capsys.readouterr()
    assert "refused" in captured.err and "nothing was run" in captured.err
    assert not out.exists()


def test_the_running_total_counts_against_the_ceiling(
    script: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "smoke"
    assert script.main(arguments(out, "--texts", "12", "--runs", "1", "--dry-run")) == 0
    estimate = printed_estimate(capsys.readouterr().out)
    out.mkdir()
    ledger = {"entries": [{"cost_usd": 20.0 - estimate / 2}], "total_usd": 20.0 - estimate / 2}
    (out / "spend.json").write_text(json.dumps(ledger), encoding="utf-8")
    assert script.main(arguments(out, "--texts", "12", "--runs", "1")) == 1
    assert "refused" in capsys.readouterr().err
    assert json.loads((out / "spend.json").read_text(encoding="utf-8")) == ledger
    assert not list(out.glob("*.sqlite"))


def test_a_fake_run_writes_the_store_the_exports_the_report_and_the_ledger(
    script: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "smoke"
    assert script.main(arguments(out, "--texts", "12", "--runs", "1", "--n-boot", "20")) == 0
    printed = capsys.readouterr().out
    for name in (
        "two_stage_runs1.sqlite",
        "two_stage_scores.csv",
        "two_stage_text_scores.csv",
        "two_stage_evidence.csv",
        "smoke_report.txt",
        "smoke_summary.json",
        "spend.json",
    ):
        assert (out / name).stat().st_size > 0, name
    report = (out / "smoke_report.txt").read_text(encoding="utf-8")
    assert report.strip() in printed
    for heading in ("== spend ==", "== acceptance ==", "== validity panel"):
        assert heading in report
    assert len(pd.read_csv(out / "two_stage_text_scores.csv")) == 12
    ledger = json.loads((out / "spend.json").read_text(encoding="utf-8"))
    assert len(ledger["entries"]) == 1
    entry = ledger["entries"][0]
    assert list(entry) == ENTRY_FIELDS
    assert entry["texts"] == 12 and entry["runs"] == 1
    assert entry["backend"] == "fake" and entry["prompt_tokens"] > 0
    assert entry["cost_usd"] == 0.0 and ledger["total_usd"] == 0.0
    summary = json.loads((out / "smoke_summary.json").read_text(encoding="utf-8"))
    assert summary["metrics"]["n_scores_equal_to_zero"] == 0
    assert summary["metrics"]["share_not_extraction_failed"] == 1.0
    assert summary["spend"] == entry


def test_a_second_run_adds_to_the_running_total(
    script: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "smoke"
    out.mkdir()
    earlier = {"backend": "openai_compatible", "cost_usd": 0.25}
    (out / "spend.json").write_text(
        json.dumps({"entries": [earlier], "total_usd": 0.25}), encoding="utf-8"
    )
    assert script.main(arguments(out, "--texts", "15", "--runs", "2", "--n-boot", "20")) == 0
    ledger = json.loads((out / "spend.json").read_text(encoding="utf-8"))
    assert len(ledger["entries"]) == 2 and ledger["entries"][0] == earlier
    assert ledger["entries"][1]["cost_usd"] == 0.0
    assert ledger["total_usd"] == pytest.approx(0.25)
    assert "running total (USD): 0.2500" in capsys.readouterr().out
    assert len(pd.read_csv(out / "two_stage_text_scores.csv")) == 15


def test_a_paid_backend_is_charged_at_the_prices_and_a_fake_one_costs_nothing(
    script: ModuleType,
) -> None:
    options = script.parse_arguments(["--out", "x", *PRICES, "--ceiling", "20", "--runs", "2"])
    paid = script.spend_entry(options, "openai_compatible", 12, 1_000_000, 500_000, 0)
    assert list(paid) == ENTRY_FIELDS
    assert paid["backend"] == "openai_compatible" and paid["runs"] == 2
    assert paid["cost_usd"] == pytest.approx(0.40 + 0.80)
    free = script.spend_entry(options, "fake", 12, 1_000_000, 500_000, 0)
    assert free["backend"] == "fake" and free["cost_usd"] == 0.0
    assert free["prompt_tokens"] == 1_000_000


def test_a_dry_run_after_a_run_reads_the_running_total(
    script: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "smoke"
    script.main(arguments(out, "--texts", "12", "--runs", "1", "--n-boot", "20"))
    total = json.loads((out / "spend.json").read_text(encoding="utf-8"))["total_usd"]
    capsys.readouterr()
    script.main(arguments(out, "--texts", "12", "--runs", "1", "--dry-run"))
    assert f"running total in spend.json (USD): {total:.4f}" in capsys.readouterr().out
    assert len(json.loads((out / "spend.json").read_text(encoding="utf-8"))["entries"]) == 1


def test_the_key_never_reaches_the_output_a_log_line_or_a_file(
    script: ModuleType,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)
    out = tmp_path / "smoke"
    script.main(arguments(out, "--texts", "12", "--runs", "1", "--n-boot", "20"))
    captured = capsys.readouterr()
    assert KEY_SENTINEL not in captured.out + captured.err
    assert len(caplog.records) > 0
    assert all(KEY_SENTINEL not in record.getMessage() for record in caplog.records)
    assert KEY_SENTINEL not in caplog.text
    for path in out.iterdir():
        assert KEY_SENTINEL.encode() not in path.read_bytes(), path.name
    ledger = json.loads((out / "spend.json").read_text(encoding="utf-8"))
    for entry in ledger["entries"]:
        assert all(isinstance(value, (int, float, str)) for value in entry.values())
        assert all(isinstance(entry[name], (int, float)) for name in NUMBER_FIELDS)
        assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\+00:00", entry["timestamp"])


def test_the_real_backend_takes_its_key_from_the_environment_and_fails_without_it(
    script: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    rubric = cop.load_rubric(cop.rubric.loader.toy_rubric_path())
    options = script.parse_arguments(["--out", str(tmp_path), *PRICES, "--ceiling", "20"])
    backend = script.build_backend(options, rubric)
    assert backend.info().model_name == "gpt-4.1-mini"
    assert KEY_SENTINEL not in json.dumps(backend.provenance())
    assert KEY_SENTINEL not in caplog.text
    monkeypatch.delenv("OPENAI_API_KEY")
    out = tmp_path / "smoke"
    with pytest.raises(BackendError, match="no API key"):
        script.main(["--out", str(out), *PRICES, "--ceiling", "20", "--texts", "6"])
    assert not (out / "spend.json").exists()


def test_the_ledger_holds_the_tokens_the_backend_reported(
    script: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = RecordingFake(toy_responder())
    monkeypatch.setattr(script, "build_backend", lambda options, rubric: fake)
    out = tmp_path / "smoke"
    script.main(arguments(out, "--texts", "12", "--runs", "1", "--n-boot", "20"))
    entry = json.loads((out / "spend.json").read_text(encoding="utf-8"))["entries"][0]
    assert (entry["prompt_tokens"], entry["completion_tokens"]) == fake_usage(fake)
    assert fake.calls


def test_a_run_that_stops_on_failures_still_records_its_spend(
    script: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = RecordingFake(lambda request: "not json")
    monkeypatch.setattr(script, "build_backend", lambda options, rubric: fake)
    out = tmp_path / "smoke"
    with pytest.raises(RunFailureCeilingError):
        script.main(arguments(out, "--texts", "12", "--runs", "1"))
    ledger = json.loads((out / "spend.json").read_text(encoding="utf-8"))
    entry = ledger["entries"][0]
    assert len(ledger["entries"]) == 1 and entry["prompt_tokens"] > 0
    assert (entry["prompt_tokens"], entry["completion_tokens"]) == fake_usage(fake)


def test_a_backend_failure_in_the_middle_records_what_was_spent_before_it(
    script: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    answer = toy_responder()
    answered = 0

    def responder(request: GenerationRequest) -> str:
        nonlocal answered
        if answered >= 20:
            raise BackendError("the endpoint refused", attempts=1)
        answered += 1
        return answer(request)

    fake = RecordingFake(responder)
    monkeypatch.setattr(script, "build_backend", lambda options, rubric: fake)
    out = tmp_path / "smoke"
    with pytest.raises(BackendError, match="the endpoint refused"):
        script.main(arguments(out, "--texts", "12", "--runs", "1"))
    entry = json.loads((out / "spend.json").read_text(encoding="utf-8"))["entries"][0]
    assert len(fake.responses) == 20
    assert entry["completion_tokens"] == sum(len(text.split()) for text in fake.responses)
    assert entry["prompt_tokens"] > 0


def test_more_texts_than_the_corpus_holds_is_an_error(script: ModuleType, tmp_path: Path) -> None:
    with pytest.raises(SettingsError, match="exceeds the 54 texts"):
        script.main(arguments(tmp_path / "smoke", "--texts", "55", "--dry-run"))


@pytest.mark.parametrize(
    "missing", ["--price-input", "--price-output", "--ceiling", "--out"], ids=str
)
def test_prices_ceiling_and_folder_are_required(script: ModuleType, missing: str) -> None:
    full = ["--out", "x", "--price-input", "1", "--price-output", "2", "--ceiling", "20"]
    position = full.index(missing)
    with pytest.raises(SystemExit) as stopped:
        script.parse_arguments(full[:position] + full[position + 2 :])
    assert stopped.value.code == 2


def test_the_acceptance_metrics_count_failures_unverified_passages_and_zero_scores(
    script: ModuleType,
) -> None:
    scores = pd.DataFrame(
        {
            "status": ["ok", "ok", "no_evidence", "extraction_failed"],
            "score": pd.array([3, 0, 1, None], dtype="Int64"),
        }
    )
    evidence = pd.DataFrame(
        {"verified": [VERIFIED_EXACT, VERIFIED_NONE, VERIFIED_EXACT, VERIFIED_NONE]}
    )
    metrics = script.acceptance_metrics(scores, evidence)
    assert metrics["share_not_extraction_failed"] == 0.75
    assert metrics["share_passages_verified"] == 0.5
    assert metrics["n_scores_equal_to_zero"] == 1
    assert metrics["status_counts"] == {
        "ok": 2,
        "no_evidence": 1,
        "too_short": 0,
        "extraction_failed": 1,
        "scoring_failed": 0,
    }
    assert script.acceptance_metrics(scores, evidence.iloc[0:0])["share_passages_verified"] is None


def test_the_refusal_boundary_is_at_equality_of_estimate_plus_total_and_ceiling(
    script: ModuleType, tmp_path: Path
) -> None:
    rubric = cop.load_rubric(cop.rubric.loader.toy_rubric_path())
    table = cop.TextTable.read(cop.assets.toy_corpus_path(), group_id="group_id")
    base = ["--out", "x", *PRICES, "--ceiling", "20", "--texts", "12", "--runs", "1"]
    options = script.parse_arguments(base)
    stages = script.estimate_run(
        table, rubric, script.first_texts(table, 12), script.coding_settings(options), False
    )
    total = 5.0
    ceiling = script.estimated_cost(stages, options) + total
    out = tmp_path / "smoke"
    out.mkdir()
    (out / "spend.json").write_text(
        json.dumps({"entries": [{"cost_usd": total}], "total_usd": total}), encoding="utf-8"
    )
    dry = ("--texts", "12", "--runs", "1", "--dry-run")
    assert script.main(arguments(out, *dry, ceiling=repr(ceiling))) == 1
    above = math.nextafter(ceiling, math.inf)
    assert script.main(arguments(out, *dry, ceiling=repr(above))) == 0


def test_a_dry_run_over_the_ceiling_exits_with_one(
    script: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "smoke"
    command = arguments(out, "--texts", "12", "--runs", "1", "--ceiling", "0.0001", "--dry-run")
    assert script.main(command) == 1
    captured = capsys.readouterr()
    assert "estimate (USD)" in captured.out and "refused" in captured.err
    assert not out.exists()
