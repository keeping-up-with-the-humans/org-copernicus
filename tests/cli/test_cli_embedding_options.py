"""Tests of the options of the ``embed`` and ``hybrid`` commands and of their defaults."""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import typer
from embedding_support import references_args
from typer.testing import CliRunner

from copernicus.cli._embedding_options import make_encoder
from copernicus.cli.app import app
from copernicus.embeddings import DEFAULT_ENCODER, ENCODERS, Encoder
from copernicus.errors import SettingsError
from copernicus.hybrid import score_hybrid
from copernicus.references import GeneratorSettings

GENERATOR_OPTIONS = (
    "style",
    "texts_per_call",
    "calls_per_cell",
    "seed",
    "temperature",
    "max_new_tokens",
    "max_retries",
    "concurrency",
)


def command_defaults(group: str, name: str) -> dict[str, object]:
    """Return the default of every parameter of one command of the application."""
    root = typer.main.get_command(app)
    command = root.commands[group].commands[name]  # type: ignore[attr-defined]
    return {param.name: param.default for param in command.params}


def test_the_option_defaults_are_the_defaults_of_the_library() -> None:
    embed_init = inspect.signature(Encoder.__init__).parameters
    embed_call = inspect.signature(Encoder.embed).parameters
    texts = command_defaults("embed", "texts")
    assert texts["encoder"] == DEFAULT_ENCODER
    assert texts["batch_size"] == embed_init["batch_size"].default
    assert texts["checkpoint_every"] == embed_call["checkpoint_every"].default
    hybrid = command_defaults("hybrid", "score")
    signature = inspect.signature(score_hybrid).parameters
    for name in ("granularity", "run_number", "min_words", "checkpoint_every"):
        assert hybrid[name] == signature[name].default, name
    assert hybrid["batch_size"] == embed_init["batch_size"].default
    generate = command_defaults("references", "generate")
    for name in GENERATOR_OPTIONS:
        assert generate[name] == getattr(GeneratorSettings(), name), name


@pytest.mark.parametrize(
    "options",
    [("--pooling", "cls"), ("--max-tokens", "64")],
)
def test_pooling_and_window_options_cannot_change_a_registered_encoder(
    runner: CliRunner, texts_csv: Path, tmp_path: Path, options: tuple[str, str]
) -> None:
    args = [
        *("embed", "texts", "--texts", str(texts_csv), "--encoder", "minilm"),
        *("--out", str(tmp_path / "set"), *options),
    ]
    result = runner.invoke(app, args)
    assert result.exit_code == 1
    assert result.stderr.startswith("error:") and "minilm" in result.stderr
    assert not list(tmp_path.glob("set*"))


def test_the_checkpoint_interval_is_checked(
    runner: CliRunner, texts_csv: Path, tmp_path: Path
) -> None:
    args = [
        *("embed", "texts", "--texts", str(texts_csv), "--encoder", "minilm"),
        *("--out", str(tmp_path / "set"), "--checkpoint-every", "0"),
    ]
    result = runner.invoke(app, args)
    assert result.exit_code == 1 and "checkpoint_every" in result.stderr


def test_the_reference_embedding_command_refuses_a_contradicting_window(
    runner: CliRunner, tmp_path: Path
) -> None:
    references = tmp_path / "references.parquet"
    assert runner.invoke(app, references_args(references)).exit_code == 0
    args = [
        *("embed", "references", "--references", str(references)),
        *("--encoder", "minilm", "--out", str(tmp_path / "set"), "--max-tokens", "64"),
    ]
    result = runner.invoke(app, args)
    assert result.exit_code == 1 and "minilm" in result.stderr
    assert not list(tmp_path.glob("set*"))


def test_the_revision_option_pins_the_encoder_of_every_embedding_command() -> None:
    pinned = make_encoder("minilm", None, None, "cpu", 4, "0123abcd")
    assert pinned.spec.revision == "0123abcd"
    assert pinned.spec.hf_id == ENCODERS["minilm"].hf_id
    assert make_encoder("minilm", None, None, "cpu", 4, None).spec is ENCODERS["minilm"]
    for group, name in (("embed", "texts"), ("embed", "references"), ("hybrid", "score")):
        assert command_defaults(group, name)["revision"] is None
        result = CliRunner().invoke(app, [group, name, "--help"], terminal_width=200)
        assert "--revision" in result.output


BLANK_REVISION = (
    "--revision needs a commit, branch or tag; leave it out to use the registered revision"
)


@pytest.mark.parametrize("blank", ["", "   "])
def test_a_blank_revision_is_refused_before_the_encoder_is_built(
    runner: CliRunner, texts_csv: Path, tmp_path: Path, blank: str
) -> None:
    args = [
        *("embed", "texts", "--texts", str(texts_csv), "--encoder", "minilm"),
        *("--out", str(tmp_path / "set"), "--revision", blank),
    ]
    result = runner.invoke(app, args)
    assert result.exit_code == 1
    assert result.stderr.startswith("error:") and BLANK_REVISION in result.stderr
    assert not list(tmp_path.glob("set*"))
    with pytest.raises(SettingsError, match=BLANK_REVISION):
        make_encoder("minilm", None, None, "cpu", 4, blank)


def test_an_encoder_that_runs_remote_code_refuses_a_branch_as_its_revision(
    runner: CliRunner, texts_csv: Path, tmp_path: Path
) -> None:
    args = [
        *("embed", "texts", "--texts", str(texts_csv), "--encoder", "gte_qwen2_7b"),
        *("--out", str(tmp_path / "set"), "--revision", "main"),
    ]
    result = runner.invoke(app, args)
    assert result.exit_code == 1
    assert result.stderr.startswith("error:") and "pins no commit" in result.stderr
    assert not list(tmp_path.glob("set*"))
