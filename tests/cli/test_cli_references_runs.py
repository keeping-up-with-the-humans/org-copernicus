"""Tests of the run options of ``references generate``: estimate, settings, resume, writers."""

from __future__ import annotations

import json
import weakref
from pathlib import Path
from typing import Any

import pytest
from cli_support import summary_value
from embedding_support import TOY_CALLS, references_args
from typer.testing import CliRunner

from copernicus.cli import _references
from copernicus.cli._backends import BackendKind, placeholder_reference_responder
from copernicus.cli.app import app
from copernicus.llm import Backend, FakeBackend
from copernicus.references import GeneratorSettings, ReferenceSet, estimate_tokens
from copernicus.rubric.loader import load_rubric, toy_rubric_path


def test_the_dry_run_prints_the_token_estimate(runner: CliRunner, tmp_path: Path) -> None:
    result = runner.invoke(app, references_args(tmp_path / "r.parquet", "--dry-run"))
    assert result.exit_code == 0, result.stderr
    estimate = estimate_tokens(load_rubric(toy_rubric_path()), GeneratorSettings())
    lines = result.stdout.splitlines()
    assert lines[1] == f"estimated prompt tokens: {estimate.prompt_tokens} (characters / 4)"
    assert lines[2] == (
        f"estimated completion tokens: {estimate.completion_tokens} (120 per requested text)"
    )


def test_prices_turn_the_estimate_into_dollars(runner: CliRunner, tmp_path: Path) -> None:
    prices = ("--price-input", "2.5", "--price-output", "10")
    result = runner.invoke(app, references_args(tmp_path / "r.parquet", "--dry-run", *prices))
    assert result.exit_code == 0, result.stderr
    estimate = estimate_tokens(load_rubric(toy_rubric_path()), GeneratorSettings())
    assert result.stdout.splitlines()[3] == (
        f"estimated cost (USD): {estimate.cost(2.5, 10.0):.4f}"
    )
    one = runner.invoke(app, references_args(tmp_path / "r.parquet", "--price-input", "2.5"))
    assert one.exit_code == 1 and "--price-output" in one.stderr


def test_the_run_options_reach_the_settings(runner: CliRunner, tmp_path: Path) -> None:
    out = tmp_path / "references.parquet"
    options = ("--temperature", "0.5", "--max-new-tokens", "999", "--max-retries", "2")
    result = runner.invoke(app, references_args(out, *options, "--concurrency", "3"))
    assert result.exit_code == 0, result.stderr
    settings = ReferenceSet.load(out).settings
    assert (settings.temperature, settings.max_new_tokens) == (0.5, 999)
    assert (settings.max_retries, settings.concurrency) == (2, 3)
    assert int(summary_value(result.stdout, "prompt tokens")) > 0
    assert int(summary_value(result.stdout, "completion tokens")) > 0
    assert summary_value(result.stdout, "attempts") == str(TOY_CALLS)


def test_a_resume_file_is_written_and_a_rerun_reads_it(runner: CliRunner, tmp_path: Path) -> None:
    out, resume = tmp_path / "references.parquet", tmp_path / "progress.parquet"
    first = runner.invoke(app, references_args(out, "--resume", str(resume)))
    assert first.exit_code == 0, first.stderr
    description = json.loads((tmp_path / "progress.parquet.json").read_text(encoding="utf-8"))
    assert len(description["completed"]) == TOY_CALLS
    again = runner.invoke(app, references_args(tmp_path / "again.parquet", "--resume", str(resume)))
    assert again.exit_code == 0, again.stderr
    assert ReferenceSet.load(tmp_path / "again.parquet").frame.equals(ReferenceSet.load(out).frame)


def test_the_resume_file_must_differ_from_the_output(runner: CliRunner, tmp_path: Path) -> None:
    out = tmp_path / "references.parquet"
    result = runner.invoke(app, references_args(out, "--resume", str(out)))
    assert result.exit_code == 1 and "--resume" in result.stderr and not out.exists()


def test_vllm_writers_are_built_one_at_a_time(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    built: list[tuple[str, weakref.ref[Backend]]] = []

    def fake_build(kind: BackendKind, *, model: str | None, **options: Any) -> Backend:
        assert kind is BackendKind.VLLM and model is not None
        assert all(reference() is None for _, reference in built), "an earlier engine is alive"
        backend = FakeBackend(placeholder_reference_responder(10), model_name=model)
        built.append((model, weakref.ref(backend)))
        return backend

    monkeypatch.setattr(_references, "build_backend", fake_build)
    args = references_args(tmp_path / "r.parquet", "--model", "writer-b")
    args[args.index("--backend") + 1] = "vllm"
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.stderr
    assert [name for name, _ in built] == ["writer-a", "writer-b"]


def test_an_api_key_cannot_be_given_as_a_backend_option(runner: CliRunner, tmp_path: Path) -> None:
    out = tmp_path / "r.parquet"
    result = runner.invoke(app, references_args(out, "--backend-option", 'api_key="secret"'))
    assert result.exit_code == 1
    assert result.stderr.startswith("error:") and "api_key" in result.stderr
    assert "secret" not in result.stderr and "secret" not in result.stdout and not out.exists()


def test_the_dry_run_help_says_that_no_backend_is_built(runner: CliRunner) -> None:
    result = runner.invoke(app, ["references", "generate", "--help"], env={"COLUMNS": "200"})
    flat = " ".join(result.stdout.replace("│", " ").split())
    assert "backend options are not checked" in flat
