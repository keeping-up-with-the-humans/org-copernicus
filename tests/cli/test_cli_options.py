"""Tests for logging, summaries on error, and backend options of the command-line interface."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from cli_support import code_args, scripted_run, summary_value, with_backend
from typer.testing import CliRunner

from copernicus.cli import app
from copernicus.cli._backends import BackendKind, build_backend
from copernicus.coders import export_provenance
from copernicus.errors import SettingsError
from copernicus.llm.openai_compatible import API_KEY_VARIABLE
from copernicus.rubric import load_rubric, toy_rubric_path

THINKING_OFF = 'chat_template_kwargs={"enable_thinking": false}'
COMMIT = "0123456789abcdef0123456789abcdef01234567"
EXTRACTION = json.dumps(
    {"Concrete examples": {"quotes": ["The example in section two"], "evaluable": True}}
)


def build(kind: BackendKind, model: str | None, **options: Any) -> Any:
    return build_backend(
        kind,
        model=model,
        base_url=None,
        timeout=None,
        options=options,
        rubric=load_rubric(toy_rubric_path()),
        fake_script=None,
    )


def test_verbose_shows_progress_on_standard_error(
    runner: CliRunner, texts_csv: Path, tmp_path: Path
) -> None:
    quiet = runner.invoke(app, code_args("two-stage", texts_csv, tmp_path / "q.sqlite"))
    assert quiet.exit_code == 0, quiet.output
    assert quiet.stderr == ""
    loud = runner.invoke(app, code_args("two-stage", texts_csv, tmp_path / "l.sqlite", "-v"))
    assert loud.exit_code == 0, loud.output
    assert "INFO copernicus.coders.runner: coded 6 of 6 texts" in loud.stderr
    assert summary_value(loud.stdout, "texts") == "6"
    package_logger = logging.getLogger("copernicus")
    assert package_logger.handlers == [] and package_logger.level == logging.NOTSET
    validated = runner.invoke(app, ["rubric", "validate", str(toy_rubric_path()), "--verbose"])
    assert validated.exit_code == 0 and "validated rubric" in validated.stderr


def test_a_ceiling_error_prints_the_summary_before_the_error(
    runner: CliRunner, tmp_path: Path
) -> None:
    args, store = scripted_run(tmp_path, ["not json"] * 3, "--max-failure-share", "0")
    result = runner.invoke(app, args)
    assert result.exit_code == 1
    assert summary_value(result.stdout, "extraction_failed") == "1"
    assert summary_value(result.stdout, "store") == str(store)
    assert result.stderr.startswith(
        "error: 1 of 1 attribute results in the requested scope failed, 0 of them stored "
        "before this invocation"
    )


def test_a_backend_error_names_the_store_the_text_and_the_attribute(
    runner: CliRunner, tmp_path: Path
) -> None:
    args, store = scripted_run(tmp_path, [EXTRACTION])
    result = runner.invoke(app, args)
    assert result.exit_code == 1
    assert result.stdout.strip() == f"{'store':<18}{store}"
    assert result.stderr.startswith(
        "error: text 't1', attribute 'specificity': scripted responses exhausted"
    )


def test_a_backend_option_reaches_the_backend_constructor(
    runner: CliRunner, texts_csv: Path, tmp_path: Path
) -> None:
    store = tmp_path / "named.sqlite"
    args = code_args("one-stage", texts_csv, store, "--backend-option", 'model_name="mock"')
    result = runner.invoke(app, [*args, "--runs", "1"])
    assert result.exit_code == 0, result.output
    assert export_provenance(store)["model_name"] == "mock"
    assert export_provenance(store)["backend_model"] == "mock"


@pytest.mark.parametrize(
    ("option", "fragment"),
    [
        ('colour="red"', "colour"),
        ('api_key="secret"', "api_key"),
        ("timeout=5", "timeout"),
    ],
)
def test_a_backend_option_the_backend_does_not_take_is_an_error(
    runner: CliRunner, texts_csv: Path, tmp_path: Path, option: str, fragment: str
) -> None:
    store = tmp_path / "x.sqlite"
    args = code_args("two-stage", texts_csv, store, "--backend-option", option)
    result = runner.invoke(app, args)
    assert result.exit_code == 1
    assert result.stderr.startswith("error:") and fragment in result.stderr
    assert "secret" not in result.stderr and not store.exists()


@pytest.mark.parametrize("option", ["no_equals_sign", "x=not json", "1x=2", "=3"])
def test_a_malformed_backend_option_is_a_usage_error(
    runner: CliRunner, texts_csv: Path, tmp_path: Path, option: str
) -> None:
    store = tmp_path / "x.sqlite"
    args = code_args("two-stage", texts_csv, store, "--backend-option", option)
    result = runner.invoke(app, args)
    assert result.exit_code == 2 and not store.exists()


def test_timeout_reaches_the_openai_backend_only(
    runner: CliRunner, texts_csv: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(API_KEY_VARIABLE, "unused")
    args = code_args("two-stage", texts_csv, tmp_path / "x.sqlite", "--model", "m")
    refused = runner.invoke(app, with_backend([*args, "--timeout", "0"], "openai"))
    assert refused.exit_code == 1 and "timeout must be" in refused.stderr
    fake_args = code_args("two-stage", texts_csv, tmp_path / "y.sqlite", "--timeout", "5")
    fake = runner.invoke(app, fake_args)
    assert fake.exit_code == 1 and "--timeout applies to --backend openai only" in fake.stderr


def test_openai_options_reach_the_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(API_KEY_VARIABLE, "unused")
    extra_body = {"chat_template_kwargs": {"enable_thinking": False}}
    backend = build(
        BackendKind.OPENAI,
        "gpt-5-mini",
        extra_body=extra_body,
        capabilities={"reasoning_effort": "low"},
        transport_retries=5,
    )
    provenance = backend.provenance()
    assert provenance["extra_body"] == '{"chat_template_kwargs":{"enable_thinking":false}}'
    assert provenance["reasoning_effort"] == "low"
    assert provenance["supports_temperature"] == "False"


@pytest.mark.parametrize("capabilities", [{"colour": 1}, 5])
def test_capabilities_must_be_an_object_of_capability_fields(
    monkeypatch: pytest.MonkeyPatch, capabilities: object
) -> None:
    monkeypatch.setenv(API_KEY_VARIABLE, "unused")
    with pytest.raises(SettingsError, match="capabilities"):
        build(BackendKind.OPENAI, "m", capabilities=capabilities)


def test_vllm_options_reach_the_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    class Engine:
        def __init__(self, **kwargs: Any) -> None:
            seen.update(kwargs)

        def get_tokenizer(self) -> None:
            return None

    module = ModuleType("vllm")
    module.LLM = Engine  # type: ignore[attr-defined]
    module.SamplingParams = dict  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "vllm", module)
    options = {"max_model_len": 4096, "quantization": "awq", "trust_remote_code": True}
    thinking_off = {"enable_thinking": False}
    backend = build(
        BackendKind.VLLM,
        "m",
        chat_template_kwargs=thinking_off,
        engine_kwargs={"revision": COMMIT},
        **options,
    )
    assert {key: seen[key] for key in options} == options
    assert seen["revision"] == seen["tokenizer_revision"] == COMMIT
    assert backend.provenance()["chat_template_kwargs"] == '{"enable_thinking":false}'


def test_an_unreadable_text_file_is_an_error_not_a_traceback(
    runner: CliRunner, tmp_path: Path
) -> None:
    texts = tmp_path / "broken.csv"
    texts.write_bytes(b'text_id,text\nt1,"never closed\n')
    result = runner.invoke(app, code_args("two-stage", texts, tmp_path / "x.sqlite"))
    assert result.exit_code == 1
    assert result.stderr.startswith("error: cannot read 'broken.csv'")


def test_a_rubric_file_that_is_not_utf8_is_an_error_not_a_traceback(
    runner: CliRunner, tmp_path: Path
) -> None:
    rubric = tmp_path / "latin.json"
    rubric.write_bytes(b'{"rubric_id": "caf\xe9"}')
    result = runner.invoke(app, ["rubric", "validate", str(rubric)])
    assert result.exit_code == 1 and result.stderr.startswith("error:")


def test_help_documents_concurrency_with_vllm_and_the_thinking_switch(runner: CliRunner) -> None:
    result = runner.invoke(app, ["code", "two-stage", "--help"], env={"COLUMNS": "200"})
    assert result.exit_code == 0
    assert "keep 1 with vLLM" in result.stdout
    assert all(token in result.stdout for token in THINKING_OFF.split())
