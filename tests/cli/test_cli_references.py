"""Tests of the ``references generate`` command."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from cli_support import summary_value
from embedding_support import TOY_CALLS, TOY_TEXTS_PER_CELL, references_args
from typer.testing import CliRunner

from copernicus.cli._backends import placeholder_reference_responder
from copernicus.cli.app import app
from copernicus.llm import GenerationRequest
from copernicus.references import ReferenceSet
from copernicus.rubric.loader import toy_rubric_path

REGISTER_DESCRIPTION = "Write each text as a short note to a neighbour, in plain wording."


def test_dry_run_prints_the_planned_calls_and_writes_nothing(
    runner: CliRunner, tmp_path: Path
) -> None:
    out = tmp_path / "references.parquet"
    result = runner.invoke(app, references_args(out, "--dry-run"))
    assert result.exit_code == 0, result.stderr
    assert result.stdout.splitlines()[0] == f"planned calls: {TOY_CALLS}"
    assert len(result.stdout.splitlines()) == 3
    assert not out.exists()


def test_dry_run_counts_models_and_calls_per_cell(runner: CliRunner, tmp_path: Path) -> None:
    args = references_args(tmp_path / "r.parquet", "--model", "writer-b", "--calls-per-cell", "3")
    result = runner.invoke(app, [*args, "--dry-run"])
    assert result.exit_code == 0, result.stderr
    assert result.stdout.splitlines()[0] == f"planned calls: {TOY_CALLS * 2 * 3}"


def test_dry_run_builds_no_real_backend(runner: CliRunner, tmp_path: Path) -> None:
    args = [
        *("references", "generate", "--rubric", str(toy_rubric_path()), "--model", "some-model"),
        *("--out", str(tmp_path / "r.parquet"), "--dry-run"),
    ]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.stderr
    assert result.stdout.splitlines()[0] == f"planned calls: {TOY_CALLS}"


def test_generate_with_the_fake_backend_writes_the_table_and_prints_the_summary(
    runner: CliRunner, tmp_path: Path
) -> None:
    out = tmp_path / "references.parquet"
    result = runner.invoke(app, references_args(out))
    assert result.exit_code == 0, result.stderr
    assert result.stdout.splitlines()[0] == f"planned calls: {TOY_CALLS}"
    frame = pd.read_parquet(out)
    assert len(frame) == TOY_CALLS * TOY_TEXTS_PER_CELL
    assert (tmp_path / "references.parquet.json").is_file()
    assert not frame["screened_out"].any()
    assert summary_value(result.stdout, "texts") == str(len(frame))
    assert summary_value(result.stdout, "per cell") == str(TOY_TEXTS_PER_CELL)
    assert summary_value(result.stdout, "screened out") == "0"
    assert summary_value(result.stdout, "shortfalls") == "0"
    assert summary_value(result.stdout, "balanced") == "yes"
    assert summary_value(result.stdout, "file") == str(out)


def test_generate_writes_csv_by_suffix(runner: CliRunner, tmp_path: Path) -> None:
    out = tmp_path / "references.csv"
    result = runner.invoke(app, references_args(out))
    assert result.exit_code == 0, result.stderr
    assert len(ReferenceSet.load(out).frame) == TOY_CALLS * TOY_TEXTS_PER_CELL


def test_generate_passes_the_settings_and_every_model(runner: CliRunner, tmp_path: Path) -> None:
    out = tmp_path / "references.parquet"
    args = references_args(
        out, *("--model", "writer-b", "--texts-per-call", "3", "--calls-per-cell", "2"),
        *("--seed", "7", "--no-screen"),
    )  # fmt: skip
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.stderr
    loaded = ReferenceSet.load(out)
    assert loaded.settings.seed == 7 and loaded.settings.texts_per_call == 3
    assert loaded.settings.calls_per_cell == 2
    assert set(loaded.frame["writer_model"]) == {"writer-a", "writer-b"}
    assert len(loaded.frame) == TOY_CALLS * 2 * 2 * 3
    assert not loaded.frame["screened_out"].any()


def test_register_style_writes_the_style_into_the_table(runner: CliRunner, tmp_path: Path) -> None:
    out = tmp_path / "references.parquet"
    args = references_args(
        out, "--style", "register", "--register-description", REGISTER_DESCRIPTION
    )
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.stderr
    assert set(pd.read_parquet(out)["style"]) == {"register"}


def test_register_style_without_a_description_is_an_error(
    runner: CliRunner, tmp_path: Path
) -> None:
    out = tmp_path / "references.parquet"
    result = runner.invoke(app, references_args(out, "--style", "register"))
    assert result.exit_code == 1
    assert result.stderr.startswith("error:") and "register_description" in result.stderr
    assert not out.exists()


def test_a_description_without_the_register_style_is_an_error(
    runner: CliRunner, tmp_path: Path
) -> None:
    args = references_args(tmp_path / "r.parquet", "--register-description", REGISTER_DESCRIPTION)
    result = runner.invoke(app, args)
    assert result.exit_code == 1
    assert result.stderr.startswith("error:")


def test_a_wrong_output_suffix_is_an_error_before_any_call(
    runner: CliRunner, tmp_path: Path
) -> None:
    result = runner.invoke(app, references_args(tmp_path / "references.txt"))
    assert result.exit_code == 1
    assert result.stderr.startswith("error:") and ".parquet" in result.stderr
    assert "planned calls" not in result.stdout


def test_a_missing_output_folder_is_an_error(runner: CliRunner, tmp_path: Path) -> None:
    result = runner.invoke(app, references_args(tmp_path / "absent" / "references.parquet"))
    assert result.exit_code == 1
    assert result.stderr.startswith("error:") and "absent" in result.stderr


def test_the_openai_backend_without_a_key_names_the_variable_not_a_value(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    args = references_args(tmp_path / "references.parquet")
    args[args.index("--backend") + 1] = "openai"
    result = runner.invoke(app, args)
    assert result.exit_code == 1
    assert result.stderr.startswith("error:")


def test_verbose_logs_the_run_on_standard_error(runner: CliRunner, tmp_path: Path) -> None:
    result = runner.invoke(app, references_args(tmp_path / "references.parquet", "-v"))
    assert result.exit_code == 0, result.stderr
    assert "generating reference texts" in result.stderr
    assert "generating reference texts" not in result.stdout


def test_placeholder_responder_answers_with_distinct_texts_per_cell() -> None:
    responder = placeholder_reference_responder(4)

    def request(user: str) -> GenerationRequest:
        return GenerationRequest(system="s", user=user, max_new_tokens=10, temperature=0.9, seed=1)

    first = json.loads(responder(request("HIGH end of target a")))
    second = json.loads(responder(request("LOW end of target a")))
    assert [list(item) for item in first] == [["text"]] * 4
    texts = {item["text"] for item in [*first, *second]}
    assert len(texts) == 8
