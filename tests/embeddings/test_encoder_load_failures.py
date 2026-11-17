"""Tests that a failed load of an encoder says what was being loaded, against fake modules."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from fake_libraries import (
    FAKE_TORCH_VERSION,
    GTE,
    GTE_REVISION,
    TRANSFORMERS_4,
    TRANSFORMERS_5,
    FakeLibraries,
    install,
)
from typer.testing import CliRunner

from copernicus.cli.app import app
from copernicus.embeddings import ENCODERS, Encoder, EncoderSpec
from copernicus.errors import ConfigurationError

pytest.importorskip("torch")
pytestmark = pytest.mark.embed

ROPE_ERROR = "'Qwen2Config' object has no attribute 'rope_theta'"
REVIEWED_COMMIT = "0123456789abcdef0123456789abcdef01234567"


def test_a_model_that_fails_inside_its_own_code_names_what_was_being_loaded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    remote = EncoderSpec(
        "mine", "org/mine", "last_token", 64, trust_remote_code=True, revision=REVIEWED_COMMIT
    )
    original = AttributeError(ROPE_ERROR)
    libraries = FakeLibraries(transformers_version=TRANSFORMERS_5, model_error=original)
    install(monkeypatch, libraries)
    with pytest.raises(ConfigurationError) as failed:
        Encoder(remote)._load()
    message = str(failed.value)
    expected = (
        "encoder 'mine' could not load its model",
        "org/mine",
        f"revision {REVIEWED_COMMIT}",
        "the repository's own code allowed",
        f"transformers {TRANSFORMERS_5}",
        f"torch {FAKE_TORCH_VERSION}",
        f"AttributeError: {ROPE_ERROR}",
    )
    for part in expected:
        assert part in message, part
    assert failed.value.__cause__ is original
    assert len(libraries.tokenizer_calls) == 1 and len(libraries.model_calls) == 1


def test_a_tokenizer_that_fails_to_load_is_named_and_the_model_is_not_loaded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = OSError("the files are not in the cache and the hub cannot be reached")
    libraries = FakeLibraries(tokenizer_error=original)
    install(monkeypatch, libraries)
    with pytest.raises(ConfigurationError) as failed:
        Encoder("minilm")._load()
    message = str(failed.value)
    expected = (
        "encoder 'minilm' could not load its tokenizer",
        ENCODERS["minilm"].hf_id,
        "the default branch, no revision pinned",
        "the repository's own code not allowed",
        "transformers 9.99.0",
        f"torch {FAKE_TORCH_VERSION}",
        "OSError: the files are not in the cache",
    )
    for part in expected:
        assert part in message, part
    assert failed.value.__cause__ is original
    assert len(libraries.tokenizer_calls) == 1 and libraries.model_calls == []


def test_a_failed_load_of_the_registered_large_encoder_names_its_pinned_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = RuntimeError("weights do not fit the configuration")
    libraries = FakeLibraries(transformers_version=TRANSFORMERS_4, model_error=original)
    install(monkeypatch, libraries)
    with pytest.raises(ConfigurationError) as failed:
        Encoder(GTE)._load()
    message = str(failed.value)
    expected = (
        f"encoder {GTE!r} could not load its model",
        f"revision {GTE_REVISION}",
        "the repository's own code allowed",
        f"transformers {TRANSFORMERS_4}",
        "RuntimeError: weights do not fit the configuration",
    )
    for part in expected:
        assert part in message, part
    assert failed.value.__cause__ is original


def test_a_failed_load_keeps_no_runtime_so_a_second_call_loads_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    libraries = FakeLibraries(model_error=ValueError("broken shard"))
    install(monkeypatch, libraries)
    encoder = Encoder("minilm")
    with pytest.raises(ConfigurationError, match="broken shard"):
        encoder._load()
    libraries.model_error = None
    runtime = encoder._load()
    assert runtime.model is libraries.model
    assert len(libraries.tokenizer_calls) == 2 and len(libraries.model_calls) == 2


def test_an_interruption_during_a_load_is_not_turned_into_a_library_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install(monkeypatch, FakeLibraries(tokenizer_error=KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        Encoder("minilm")._load()


def test_the_embed_command_ends_a_failed_load_with_the_message_and_exit_code_1(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    texts = tmp_path / "texts.csv"
    table = pd.DataFrame({"text_id": ["a"], "text": ["One short text."], "group_id": ["g"]})
    table.to_csv(texts, index=False)
    install(monkeypatch, FakeLibraries(model_error=OSError("no such model in the cache")))
    args = ["embed", "texts", "--texts", str(texts), "--out", str(tmp_path / "set")]
    options = ["--encoder", "minilm", "--device", "cpu", "--group-column", "group_id"]
    result = CliRunner().invoke(app, [*args, *options])
    assert result.exit_code == 1 and isinstance(result.exception, SystemExit)
    flat = " ".join(result.stderr.split())
    assert flat.startswith("error: encoder 'minilm' could not load its model")
    assert "OSError: no such model in the cache" in flat
    assert not (tmp_path / "set.npz").exists()
