"""Tests for the encoder's chunking, pooling, weighting and checkpoints, on a fake model."""

from __future__ import annotations

import importlib
import re
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fake_runtime import (
    CLS_ID,
    FAKE_VERSIONS,
    SEP_ID,
    BatchTokenizer,
    TableModel,
    fake_runtime,
    torch,
)

from copernicus.data import TextTable
from copernicus.embeddings import EmbeddingSet, Encoder, EncoderSpec
from copernicus.errors import ConfigurationError, SettingsError

pytestmark = pytest.mark.embed
EMBED_HINT = (
    'install the embed extra: pip install "org-copernicus[embed]", '
    'or pip install ".[embed]" from the package\'s root folder'
)
SPEC = EncoderSpec("fake", "org/fake", "mean", 6)
TEXTS = ["a b c", " ".join(f"w{i}" for i in range(10)), "d e", " ".join(f"v{i}" for i in range(9))]
IDS = [10, 11, 12, 13]
CHUNKS_PER_TEXT = [1, 3, 1, 3]
ARRAYS = ("ids", "vectors", "chunk_vectors", "chunk_owner", "chunk_weight")


def encoder_with(
    monkeypatch: pytest.MonkeyPatch, model: TableModel | None = None, **options: Any
) -> tuple[Encoder, Any]:
    """Return an encoder whose runtime is the fake one, and that runtime."""
    runtime = fake_runtime(model, options.pop("pooling", "mean"))
    monkeypatch.setattr(Encoder, "_load", lambda self: runtime)
    return Encoder(options.pop("spec", SPEC), **options), runtime


def token_vectors(model: TableModel, tokenizer: BatchTokenizer, text: str) -> np.ndarray:
    """Vectors of the attended tokens of one window: its words wrapped in two special tokens."""
    ids = [CLS_ID, *tokenizer.encode(text, add_special_tokens=False), SEP_ID]
    return model.table[ids].to(torch.float64).numpy()


def unit(vector: np.ndarray) -> np.ndarray:
    return vector / np.linalg.norm(vector)


def test_texts_are_chunked_and_weighted_by_their_token_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = TableModel()
    encoder, runtime = encoder_with(monkeypatch, model, batch_size=3)
    result = encoder.embed(TEXTS, IDS)
    assert result.ids.tolist() == ["10", "11", "12", "13"] and result.encoder == "fake"
    assert np.bincount(result.chunk_owner).tolist() == CHUNKS_PER_TEXT
    assert result.chunk_owner.tolist() == sorted(result.chunk_owner.tolist())
    assert result.chunk_weight[result.chunk_owner == 1].tolist() == [6.0, 6.0, 4.0]
    assert result.vectors.dtype == np.float32 and result.chunk_vectors.dtype == np.float32
    assert result.vectors.shape == (4, 4) and result.chunk_vectors.shape == (8, 4)
    assert np.allclose(np.linalg.norm(result.vectors, axis=1), 1.0, atol=1e-6)
    assert np.allclose(np.linalg.norm(result.chunk_vectors, axis=1), 1.0, atol=1e-6)
    windows = ["w0 w1 w2 w3", "w4 w5 w6 w7", "w8 w9"]
    raw = [token_vectors(model, runtime.tokenizer, window).mean(axis=0) for window in windows]
    assert np.allclose(result.chunk_vectors[1:4], [unit(vector) for vector in raw], atol=1e-6)
    every_token = np.concatenate([token_vectors(model, runtime.tokenizer, w) for w in windows])
    assert np.allclose(result.vectors[1], unit(every_token.mean(axis=0)), atol=1e-6)
    assert not np.allclose(result.vectors[1], result.chunk_vectors[1], atol=1e-3)


def test_batches_are_filled_in_order_of_length(monkeypatch: pytest.MonkeyPatch) -> None:
    model = TableModel()
    encoder, _ = encoder_with(monkeypatch, model, batch_size=3)
    encoder.embed(TEXTS, IDS)
    assert model.batches == [3, 3, 2]


def test_the_batch_size_does_not_change_the_vectors(monkeypatch: pytest.MonkeyPatch) -> None:
    sets = [encoder_with(monkeypatch, batch_size=size)[0].embed(TEXTS, IDS) for size in (1, 8)]
    assert np.array_equal(sets[0].ids, sets[1].ids)
    for name in ARRAYS[1:]:
        assert np.allclose(getattr(sets[0], name), getattr(sets[1], name), atol=1e-6), name


@pytest.mark.parametrize("pooling", ["cls", "last_token"])
def test_the_pooling_of_the_spec_is_used(monkeypatch: pytest.MonkeyPatch, pooling: str) -> None:
    spec = EncoderSpec("fake", "org/fake", pooling, 6)  # type: ignore[arg-type]
    model = TableModel()
    encoder, runtime = encoder_with(monkeypatch, model, spec=spec, pooling=pooling)
    result = encoder.embed(["a b c"], ["x"])
    expected = model.table[CLS_ID if pooling == "cls" else SEP_ID].numpy()
    assert np.allclose(result.vectors[0], unit(expected), atol=1e-6)


def test_ids_are_stored_as_strings_and_the_sidecar_holds_identity_and_versions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = encoder_with(monkeypatch)[0].embed(["a b"], [7])
    assert result.ids.dtype.kind == "U" and result.ids.tolist() == ["7"]
    assert result.sidecar == {
        "hf_id": "org/fake",
        "revision": "unpinned",
        "pooling": "mean",
        "max_tokens": 6,
        "dtype": None,
        "trust_remote_code": False,
        "padding_side": "right",
        "special_tokens_reserved": 2,
        **FAKE_VERSIONS,
    }


def test_checkpoint_resume_gives_the_same_set_and_skips_finished_chunks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    reference = encoder_with(monkeypatch, batch_size=1)[0].embed(TEXTS, IDS)
    checkpoint = tmp_path / "state"
    dying, _ = encoder_with(monkeypatch, TableModel(fail_after_calls=2), batch_size=1)
    with pytest.raises(RuntimeError, match="model killed"):
        dying.embed(TEXTS, IDS, checkpoint=checkpoint, checkpoint_every=2)
    assert (tmp_path / "state.npz").is_file() and not list(tmp_path.glob("*.partial"))
    model = TableModel()
    resumed = encoder_with(monkeypatch, model, batch_size=1)[0].embed(
        TEXTS, IDS, checkpoint=checkpoint, checkpoint_every=2
    )
    assert len(model.batches) == sum(CHUNKS_PER_TEXT) - 2
    for name in ARRAYS:
        assert np.array_equal(getattr(resumed, name), getattr(reference, name)), name


def test_a_checkpoint_run_without_interruption_equals_a_run_without_checkpoint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    plain = encoder_with(monkeypatch, batch_size=2)[0].embed(TEXTS, IDS)
    saved = encoder_with(monkeypatch, batch_size=2)[0].embed(
        TEXTS, IDS, checkpoint=tmp_path / "state", checkpoint_every=3
    )
    assert list(tmp_path.iterdir()) == []
    for name in ARRAYS:
        assert np.array_equal(getattr(plain, name), getattr(saved, name)), name


def test_a_checkpoint_of_other_texts_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    dying, _ = encoder_with(monkeypatch, TableModel(fail_after_calls=2), batch_size=1)
    with pytest.raises(RuntimeError, match="model killed"):
        dying.embed(TEXTS, IDS, checkpoint=tmp_path / "state", checkpoint_every=2)
    encoder, _ = encoder_with(monkeypatch, batch_size=1)
    with pytest.raises(ConfigurationError, match="other texts or another encoder"):
        encoder.embed(TEXTS[:3], IDS[:3], checkpoint=tmp_path / "state")
    assert (tmp_path / "state.npz").is_file()


def test_a_checkpoint_of_another_commit_of_the_model_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    dying, _ = encoder_with(monkeypatch, TableModel(fail_after_calls=2), batch_size=1)
    with pytest.raises(RuntimeError, match="model killed"):
        dying.embed(TEXTS, IDS, checkpoint=tmp_path / "state", checkpoint_every=2)
    moved = replace(fake_runtime(), versions={**FAKE_VERSIONS, "resolved_revision": "moved"})
    monkeypatch.setattr(Encoder, "_load", lambda self: moved)
    with pytest.raises(ConfigurationError, match="commit"):
        Encoder(SPEC, batch_size=1).embed(TEXTS, IDS, checkpoint=tmp_path / "state")
    assert (tmp_path / "state.npz").is_file()


def test_a_completed_run_deletes_its_checkpoint_so_the_path_can_be_reused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    encoder, _ = encoder_with(monkeypatch, batch_size=1)
    encoder.embed(TEXTS, IDS, checkpoint=tmp_path / "state", checkpoint_every=2)
    assert list(tmp_path.iterdir()) == []
    other = encoder.embed(TEXTS[:3], IDS[:3], checkpoint=tmp_path / "state", checkpoint_every=2)
    assert len(other) == 3 and list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("side", ["left", "right"])
def test_last_token_pooling_reads_the_last_attended_word_on_either_padding_side(
    monkeypatch: pytest.MonkeyPatch, side: str
) -> None:
    spec = EncoderSpec("fake", "org/fake", "last_token", 6, special_tokens_reserved=0)
    model = TableModel()
    tokenizer = BatchTokenizer(padding_side=side, wrap=False)
    runtime = fake_runtime(model, "last_token", tokenizer)
    monkeypatch.setattr(Encoder, "_load", lambda self: runtime)
    result = Encoder(spec, batch_size=2).embed(["a b c d", "e f", "g"], ["x", "y", "z"])
    for row, word in enumerate(["d", "f", "g"]):
        expected = model.table[tokenizer.vocabulary[word]].numpy()
        assert np.allclose(result.vectors[row], unit(expected), atol=1e-6), word


def test_inputs_that_cannot_be_embedded_are_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    encoder, _ = encoder_with(monkeypatch)
    with pytest.raises(ConfigurationError, match="no texts"):
        encoder.embed([], [])
    with pytest.raises(ConfigurationError, match="3 ids for 2 texts"):
        encoder.embed(["a", "b"], [1, 2, 3])
    with pytest.raises(ConfigurationError, match="unique"):
        encoder.embed(["a", "b"], [1, "1"])
    with pytest.raises(SettingsError, match="checkpoint_every"):
        encoder.embed(["a"], [1], checkpoint_every=0)
    with pytest.raises(SettingsError, match="batch_size"):
        Encoder(SPEC, batch_size=0)


def test_empty_and_whitespace_texts_are_refused_naming_their_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    encoder, _ = encoder_with(monkeypatch)
    with pytest.raises(ConfigurationError, match=r"2 texts .*'b', 'c'") as error:
        encoder.embed(["a b", "", " \n\t ", "d"], ["a", "b", "c", "d"])
    assert "'a'" not in str(error.value) and "'d'" not in str(error.value)
    many = [""] * 12
    with pytest.raises(ConfigurationError, match="and 2 more") as error:
        encoder.embed(many, [f"t{index}" for index in range(12)])
    assert "'t9'" in str(error.value) and "'t10'" not in str(error.value)


def test_embed_table_keeps_the_other_columns_as_metadata(
    monkeypatch: pytest.MonkeyPatch, toy_corpus: TextTable
) -> None:
    table = TextTable.from_frame(toy_corpus.frame.head(3))
    encoder, _ = encoder_with(monkeypatch)
    result = encoder.embed_table(table)
    assert result.ids.tolist() == table.ids() and len(result) == 3
    assert result.metadata is not None and "text" not in result.metadata.columns
    assert result.metadata.text_id.tolist() == table.ids()
    assert result.metadata.group_id.tolist() == list(table.frame.group_id)
    assert isinstance(result, EmbeddingSet)


def test_a_missing_torch_is_explained(monkeypatch: pytest.MonkeyPatch) -> None:
    real = importlib.import_module

    def without_torch(name: str, *args: Any) -> Any:
        if name == "torch":
            raise ImportError("No module named 'torch'", name="torch")
        return real(name, *args)

    monkeypatch.setattr("copernicus.embeddings.encoder.importlib.import_module", without_torch)
    with pytest.raises(
        ConfigurationError, match=f"torch is not installed; {re.escape(EMBED_HINT)}"
    ):
        Encoder(SPEC).embed(["a"], [1])


def test_a_dtype_that_torch_does_not_know_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    spec = EncoderSpec("fake", "org/fake", "mean", 6, dtype="float99")
    with pytest.raises(ConfigurationError, match="float99"):
        Encoder(spec).embed(["a"], [1])
