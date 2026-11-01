"""Tests for the validation and coding commands of the command-line interface."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from cli_support import (
    code_args,
    minimal_rubric_json,
    option_defaults,
    scripted_run,
    summary_value,
    with_backend,
)
from typer.testing import CliRunner

from copernicus.cli import app
from copernicus.coders import export_provenance, export_scores, export_text_scores
from copernicus.llm.openai_compatible import API_KEY_VARIABLE
from copernicus.rubric.loader import toy_rubric_path

N_TEXTS = 6
N_ATTRIBUTES = 3
EXTRACTION = {"Concrete examples": {"quotes": ["The example in section two"], "evaluable": True}}
GOOD_ANSWERS = [json.dumps(EXTRACTION), json.dumps({"result": 4, "explanation": "clear"})]


def test_rubric_validate_toy_rubric(runner: CliRunner) -> None:
    result = runner.invoke(app, ["rubric", "validate", str(toy_rubric_path())])
    assert result.exit_code == 0, result.output
    assert result.stdout.startswith("valid: constructive_feedback_v1 (3 attributes, ")
    assert result.stdout.strip().endswith("markers)")


def test_rubric_validate_accepts_sources_for_the_rubric_alone_or_for_each_attribute(
    runner: CliRunner, tmp_path: Path
) -> None:
    data = json.loads(toy_rubric_path().read_text(encoding="utf-8"))
    for attribute in data["attributes"]:
        del attribute["literature_sources"]
    rubric_only = tmp_path / "rubric_sources_only.json"
    rubric_only.write_text(json.dumps(data), encoding="utf-8")
    for path in (toy_rubric_path(), rubric_only):
        result = runner.invoke(app, ["rubric", "validate", str(path)])
        assert result.exit_code == 0, result.output
        assert result.stdout == "valid: constructive_feedback_v1 (3 attributes, 9 markers)\n"


def test_rubric_validate_broken_file_exits_one(runner: CliRunner, tmp_path: Path) -> None:
    broken = tmp_path / "broken.json"
    broken.write_text(json.dumps({"rubric_id": "x"}), encoding="utf-8")
    result = runner.invoke(app, ["rubric", "validate", str(broken)])
    assert result.exit_code == 1
    assert result.stderr.startswith("error:")
    assert "valid:" not in result.stdout


def test_code_two_stage_with_fake_backend_creates_store(
    runner: CliRunner, texts_csv: Path, tmp_path: Path
) -> None:
    store = tmp_path / "run.sqlite"
    result = runner.invoke(app, code_args("two-stage", texts_csv, store))
    assert result.exit_code == 0, result.output
    assert store.exists()
    lines = result.stdout.splitlines()
    assert lines[0] == f"{'texts':<18}{N_TEXTS}" and lines[-1] == f"{'store':<18}{store}"
    assert summary_value(result.stdout, "store") == str(store)
    assert summary_value(result.stdout, "texts") == str(N_TEXTS)
    assert summary_value(result.stdout, "runs") == "2"
    assert summary_value(result.stdout, "attribute calls") == str(N_TEXTS * 2 * N_ATTRIBUTES)
    assert summary_value(result.stdout, "resumed") == "0"
    assert int(summary_value(result.stdout, "ok")) > 0
    assert "extraction_failed" not in result.stdout
    assert len(export_text_scores(store)) == N_TEXTS


def test_code_one_stage_with_fake_backend(
    runner: CliRunner, texts_csv: Path, tmp_path: Path
) -> None:
    store = tmp_path / "one.sqlite"
    result = runner.invoke(app, code_args("one-stage", texts_csv, store, "--runs", "1"))
    assert result.exit_code == 0, result.output
    assert export_provenance(store)["method"] == "one_stage"
    assert summary_value(result.stdout, "attribute calls") == str(N_TEXTS * N_ATTRIBUTES)


def test_range_codes_only_the_selected_texts(
    runner: CliRunner, texts_csv: Path, tmp_path: Path
) -> None:
    store = tmp_path / "shard.sqlite"
    result = runner.invoke(app, code_args("two-stage", texts_csv, store, "--range", "2:4"))
    assert result.exit_code == 0, result.output
    assert summary_value(result.stdout, "texts") == "2"
    assert len(export_text_scores(store)) == 2


def test_second_invocation_resumes_every_cell(
    runner: CliRunner, texts_csv: Path, tmp_path: Path
) -> None:
    store = tmp_path / "run.sqlite"
    args = code_args("two-stage", texts_csv, store)
    assert runner.invoke(app, args).exit_code == 0
    second = runner.invoke(app, args)
    assert second.exit_code == 0, second.output
    assert summary_value(second.stdout, "resumed") == str(N_TEXTS * 2)
    assert summary_value(second.stdout, "texts") == "0"


def test_changed_settings_against_existing_store_is_an_error(
    runner: CliRunner, texts_csv: Path, tmp_path: Path
) -> None:
    store = tmp_path / "run.sqlite"
    assert runner.invoke(app, code_args("two-stage", texts_csv, store)).exit_code == 0
    result = runner.invoke(app, code_args("two-stage", texts_csv, store, "--seed", "7"))
    assert result.exit_code == 1
    assert result.stderr.startswith("error:")


def test_fake_backend_with_other_rubric_needs_a_script(
    runner: CliRunner, texts_csv: Path, tmp_path: Path
) -> None:
    rubric = tmp_path / "rubric.json"
    rubric.write_text(minimal_rubric_json(), encoding="utf-8")
    args = code_args("two-stage", texts_csv, tmp_path / "x.sqlite")
    args[args.index("--rubric") + 1] = str(rubric)
    result = runner.invoke(app, args)
    assert result.exit_code == 1
    assert result.stderr.startswith("error:") and "--fake-script" in result.stderr


def test_fake_script_codes_with_a_non_toy_rubric(runner: CliRunner, tmp_path: Path) -> None:
    rubric = tmp_path / "rubric.json"
    rubric.write_text(minimal_rubric_json(), encoding="utf-8")
    texts = tmp_path / "one.csv"
    texts.write_text(
        "text_id,text\nt1,The example in section two shows the problem clearly.\n", encoding="utf-8"
    )
    extraction = {
        "Concrete examples": {"quotes": ["The example in section two"], "evaluable": True}
    }
    scoring = {"result": 4, "explanation": "clear"}
    script = tmp_path / "script.json"
    script.write_text(json.dumps([json.dumps(extraction), json.dumps(scoring)]), encoding="utf-8")
    store = tmp_path / "scripted.sqlite"
    result = runner.invoke(
        app,
        [
            *("code", "two-stage", "--texts", str(texts), "--rubric", str(rubric)),
            *("--out", str(store), "--backend", "fake", "--fake-script", str(script)),
            *("--runs", "1"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert summary_value(result.stdout, "ok") == "1"
    assert export_text_scores(store).construct_score.tolist() == [4.0]


def test_failure_ceiling_reports_error_after_the_rows_are_written(
    runner: CliRunner, tmp_path: Path
) -> None:
    rubric = tmp_path / "rubric.json"
    rubric.write_text(minimal_rubric_json(), encoding="utf-8")
    texts = tmp_path / "one.csv"
    texts.write_text(
        "text_id,text,group_id\nt1,The example in section two shows it.,g1\n", encoding="utf-8"
    )
    script = tmp_path / "garbage.json"
    script.write_text(json.dumps(["not json"] * 9), encoding="utf-8")
    store = tmp_path / "failed.sqlite"
    result = runner.invoke(
        app,
        [
            *("code", "two-stage", "--texts", str(texts), "--rubric", str(rubric)),
            *("--out", str(store), "--backend", "fake", "--fake-script", str(script)),
            *("--runs", "1", "--max-failure-share", "0"),
        ],
    )
    assert result.exit_code == 1
    assert result.stderr.startswith("error:")
    assert store.exists()
    assert export_scores(store).status.tolist() == ["extraction_failed"]


def test_failed_cells_are_coded_again_only_with_retry_failed(
    runner: CliRunner, tmp_path: Path
) -> None:
    failing, store = scripted_run(tmp_path, ["not json"] * 9)
    assert runner.invoke(app, failing).exit_code == 1
    plain, _ = scripted_run(tmp_path, GOOD_ANSWERS)
    kept = runner.invoke(app, plain)
    assert kept.exit_code == 1
    assert "1 of 1 attribute results in the requested scope failed, 1 of them stored" in kept.stderr
    assert summary_value(kept.stdout, "stored failed") == "1"
    assert summary_value(kept.stdout, "resumed") == "1"
    assert export_scores(store).status.tolist() == ["extraction_failed"]
    retrying, _ = scripted_run(tmp_path, GOOD_ANSWERS, "--retry-failed")
    retried = runner.invoke(app, retrying)
    assert retried.exit_code == 0, retried.output
    assert summary_value(retried.stdout, "retried") == "1"
    assert summary_value(retried.stdout, "ok") == "1"
    assert "stored failed" not in retried.stdout
    assert export_scores(store).status.tolist() == ["ok"]


def test_both_code_commands_take_retry_failed() -> None:
    for command in ("two-stage", "one-stage"):
        assert option_defaults("code", command)["retry_failed"] is False


def test_range_beyond_the_table_codes_every_text(
    runner: CliRunner, texts_csv: Path, tmp_path: Path
) -> None:
    result = runner.invoke(
        app, code_args("two-stage", texts_csv, tmp_path / "all.sqlite", "--range", "0:999")
    )
    assert result.exit_code == 0, result.output
    assert summary_value(result.stdout, "texts") == str(N_TEXTS)


def test_code_into_a_file_that_is_no_store_is_an_error(runner: CliRunner, texts_csv: Path) -> None:
    before = texts_csv.read_bytes()
    result = runner.invoke(app, code_args("two-stage", texts_csv, texts_csv))
    assert result.exit_code == 1
    assert result.stderr.startswith("error: file is not a run store")
    assert texts_csv.read_bytes() == before


def test_unknown_backend_is_a_usage_error(
    runner: CliRunner, texts_csv: Path, tmp_path: Path
) -> None:
    result = runner.invoke(
        app, with_backend(code_args("two-stage", texts_csv, tmp_path / "x.sqlite"), "opnai")
    )
    assert result.exit_code == 2
    assert not (tmp_path / "x.sqlite").exists()


def test_temperature_option_is_stored(runner: CliRunner, texts_csv: Path, tmp_path: Path) -> None:
    store = tmp_path / "hot.sqlite"
    result = runner.invoke(app, code_args("two-stage", texts_csv, store, "--temperature", "0.7"))
    assert result.exit_code == 0, result.output
    assert float(export_provenance(store)["temperature"]) == pytest.approx(0.7)


def test_no_shuffle_is_stored(runner: CliRunner, texts_csv: Path, tmp_path: Path) -> None:
    store = tmp_path / "ordered.sqlite"
    result = runner.invoke(app, code_args("two-stage", texts_csv, store, "--no-shuffle"))
    assert result.exit_code == 0, result.output
    assert export_provenance(store)["shuffle_markers"] == "False"


def test_unknown_wording_variant_is_reported_as_error(
    runner: CliRunner, texts_csv: Path, tmp_path: Path
) -> None:
    args = code_args("two-stage", texts_csv, tmp_path / "x.sqlite", "--enable", "no_such_variant")
    result = runner.invoke(app, args)
    assert result.exit_code == 1
    assert result.stderr.startswith("error:") and "no_such_variant" in result.stderr


def test_unparsable_temperature_is_a_usage_error(
    runner: CliRunner, texts_csv: Path, tmp_path: Path
) -> None:
    args = code_args("two-stage", texts_csv, tmp_path / "x.sqlite", "--temperature", "nonsense")
    result = runner.invoke(app, args)
    assert result.exit_code == 2
    assert not (tmp_path / "x.sqlite").exists()


@pytest.mark.parametrize("bad_range", ["3:1", "3", "a:b", "-1:2"])
def test_bad_range_is_rejected(
    runner: CliRunner, texts_csv: Path, tmp_path: Path, bad_range: str
) -> None:
    args = code_args("two-stage", texts_csv, tmp_path / "x.sqlite", "--range", bad_range)
    result = runner.invoke(app, args)
    assert result.exit_code != 0
    assert not (tmp_path / "x.sqlite").exists()


def test_out_of_range_setting_is_reported_as_error(
    runner: CliRunner, texts_csv: Path, tmp_path: Path
) -> None:
    args = code_args("two-stage", texts_csv, tmp_path / "x.sqlite", "--runs", "0")
    result = runner.invoke(app, args)
    assert result.exit_code == 1
    assert result.stderr.startswith("error:") and "runs" in result.stderr


def test_openai_backend_without_key_is_an_error(
    runner: CliRunner, texts_csv: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(API_KEY_VARIABLE, raising=False)
    args = code_args("two-stage", texts_csv, tmp_path / "x.sqlite", "--model", "gpt-4o-mini")
    result = runner.invoke(app, with_backend(args, "openai"))
    assert result.exit_code == 1
    assert result.stderr.strip() == "error: no API key: pass api_key or set OPENAI_API_KEY"


def test_openai_backend_needs_a_model(
    runner: CliRunner, texts_csv: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(API_KEY_VARIABLE, "unused")
    args = with_backend(code_args("two-stage", texts_csv, tmp_path / "x.sqlite"), "openai")
    result = runner.invoke(app, args)
    assert result.exit_code == 1 and "--model" in result.stderr


def test_vllm_backend_reports_missing_install(
    runner: CliRunner, texts_csv: Path, tmp_path: Path
) -> None:
    args = with_backend(
        code_args("two-stage", texts_csv, tmp_path / "x.sqlite", "--model", "some/model"), "vllm"
    )
    result = runner.invoke(app, args)
    assert result.exit_code == 1
    assert "vLLM is not installed" in result.stderr


def test_fake_script_that_is_not_utf8_text_is_refused(
    runner: CliRunner, texts_csv: Path, tmp_path: Path
) -> None:
    script = tmp_path / "latin1.json"
    script.write_bytes('["café"]'.encode("latin-1"))
    args = code_args("two-stage", texts_csv, tmp_path / "x.sqlite", "--fake-script", str(script))
    result = runner.invoke(app, args)
    assert result.exit_code == 1
    assert result.stderr.startswith("error:") and "not UTF-8 text" in result.stderr
    assert "Traceback" not in result.stderr
