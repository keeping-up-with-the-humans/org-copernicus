"""Tests of the reference embeddings: pole sets made from the kept reference texts."""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from embeddings_support import needs_minilm

from copernicus import __version__
from copernicus.embeddings import EmbeddingSet, Encoder, EncoderSpec
from copernicus.errors import ConfigurationError
from copernicus.references import GeneratorSettings, ReferenceSet, build_cells
from copernicus.references.reference_set import FRAME_COLUMNS
from copernicus.rubric.model import Rubric
from copernicus.similarity import ReferenceEmbeddings, embed_references

TEXTS_PER_CELL = 2
STUB_VERSIONS = {
    "resolved_revision": "abc123def",
    "torch_version": "9.9.9",
    "transformers_version": "8.8.8",
}


def toy_reference_set(rubric: Rubric, writers: tuple[str, ...] = ("writer-a",)) -> ReferenceSet:
    """Build a reference set by hand: two texts per cell and writer, nothing screened out."""
    rows: list[dict[str, Any]] = []
    for writer in writers:
        for cell in build_cells(rubric):
            for number in range(TEXTS_PER_CELL):
                rows.append(
                    {
                        "rubric_id": rubric.rubric_id,
                        "target": cell.target,
                        "polarity": cell.polarity,
                        "level": cell.level,
                        "style": "neutral",
                        "writer_model": writer,
                        "text": (
                            f"A {cell.polarity} note about {cell.target} at the {cell.level} "
                            f"level, number {number}, from {writer}."
                        ),
                        "call_index": number,
                        "attempt": 0,
                        "seed": 42 + number,
                        "prompt_tokens": 100.0,
                        "completion_tokens": 50.0,
                        "temperature_applied": 0.9,
                        "top_p_applied": None,
                        "prompt_hash": "hash",
                        "prompt_version": "v1",
                        "screened_out": False,
                        "screen_reason": "",
                    }
                )
    frame = pd.DataFrame(rows, columns=list(FRAME_COLUMNS))
    return ReferenceSet.for_rubric(frame, rubric, GeneratorSettings())


class StubEncoder:
    """An encoder stand-in: gives each text the given number of chunks of random vectors."""

    def __init__(self, chunks_per_text: list[int], dimension: int = 5) -> None:
        self.spec = EncoderSpec("stub", "org/stub", "mean", 8, revision="abc123")
        self.chunks_per_text = chunks_per_text
        self.dimension = dimension
        self.seen: tuple[list[str], list[Any]] | None = None

    def embed(self, texts: list[str], ids: list[Any]) -> EmbeddingSet:
        self.seen = (list(texts), list(ids))
        generator = np.random.default_rng(11)
        owner = np.repeat(np.arange(len(texts)), self.chunks_per_text).astype(np.int64)
        chunks = generator.normal(size=(len(owner), self.dimension))
        chunks /= np.linalg.norm(chunks, axis=1, keepdims=True)
        sums = np.zeros((len(texts), self.dimension))
        np.add.at(sums, owner, chunks)
        sums /= np.linalg.norm(sums, axis=1, keepdims=True)
        return EmbeddingSet(
            ids=np.asarray([str(text_id) for text_id in ids]),
            vectors=sums.astype(np.float32),
            chunk_vectors=chunks.astype(np.float32),
            chunk_owner=owner,
            chunk_weight=np.ones(len(owner)),
            encoder=self.spec.name,
            metadata=None,
            sidecar={**self.spec.identity(), **STUB_VERSIONS},
        )


def stub_embeddings(
    rubric: Rubric, writers: tuple[str, ...] = ("writer-a",)
) -> ReferenceEmbeddings:
    """Embed the hand-built reference set with the stub encoder, one chunk per text."""
    refset = toy_reference_set(rubric, writers)
    encoder: Any = StubEncoder([1] * len(refset.kept()))
    return embed_references(refset, encoder)


def drop_rows(refs: ReferenceEmbeddings, mask: np.ndarray) -> ReferenceEmbeddings:
    """Return the reference embeddings without the rows where ``mask`` is true."""
    keep = ~mask
    return replace(
        refs,
        vectors=refs.vectors[keep],
        target=refs.target[keep],
        polarity=refs.polarity[keep],
        level=refs.level[keep],
        style=refs.style[keep],
        writer_model=refs.writer_model[keep],
    )


@needs_minilm
def test_minilm_reference_embeddings_cover_every_pole(toy_rubric: Rubric) -> None:
    encoder = Encoder("minilm", device="cpu")
    refs = embed_references(toy_reference_set(toy_rubric), encoder)
    assert refs.vectors.shape == (48, 384) and refs.vectors.dtype == np.float32
    assert np.allclose(np.linalg.norm(refs.vectors, axis=1), 1.0, atol=1e-4)
    assert refs.encoder == "minilm" and refs.rubric_id == toy_rubric.rubric_id
    assert refs.pole_indices("construct", "positive").size == 6
    refs.assert_complete(toy_rubric)
    broken = drop_rows(refs, (refs.target == "construct") & (refs.polarity == "negative"))
    with pytest.raises(ConfigurationError, match=r"negative.*construct") as raised:
        broken.assert_complete(toy_rubric)
    assert "construct" in str(raised.value) and "negative" in str(raised.value)


@needs_minilm
def test_minilm_reference_embeddings_survive_a_round_trip(
    toy_rubric: Rubric, tmp_path: Path
) -> None:
    refs = embed_references(toy_reference_set(toy_rubric), Encoder("minilm", device="cpu"))
    array_path, sidecar_path = refs.save(tmp_path / "poles")
    assert array_path.name == "poles.npz" and sidecar_path.name == "poles.json"
    loaded = ReferenceEmbeddings.load(tmp_path / "poles", expect_encoder="minilm")
    assert np.array_equal(loaded.vectors, refs.vectors)
    for name in ("target", "polarity", "level", "style", "writer_model"):
        assert getattr(loaded, name).tolist() == getattr(refs, name).tolist()
    assert loaded.rubric_id == refs.rubric_id and loaded.encoder == "minilm"
    loaded.assert_complete(toy_rubric)


def test_the_frame_order_and_positional_ids_are_used(toy_rubric: Rubric) -> None:
    refset = toy_reference_set(toy_rubric)
    kept = refset.kept()
    screened = refset.frame.copy()
    screened.loc[screened.index[3], "screened_out"] = True
    half_screened = ReferenceSet.for_rubric(screened, toy_rubric, GeneratorSettings())
    encoder = StubEncoder([1] * 47)
    refs = embed_references(half_screened, encoder)  # type: ignore[arg-type]
    assert encoder.seen is not None
    texts, ids = encoder.seen
    assert texts == list(half_screened.kept()["text"]) and ids == [str(i) for i in range(47)]
    assert len(kept) == 48 and refs.vectors.shape == (47, 5)
    assert refs.target.tolist() == list(half_screened.kept()["target"])
    assert refs.sidecar["prompt_versions"] == ["v1"] and refs.sidecar["hf_id"] == "org/stub"
    assert refs.sidecar["rubric_id"] == toy_rubric.rubric_id and refs.sidecar["n_references"] == 47
    assert refs.sidecar["rubric_hash"] == toy_rubric.content_hash() == refs.rubric_hash


def test_an_empty_kept_frame_is_refused(toy_rubric: Rubric) -> None:
    frame = toy_reference_set(toy_rubric).frame.assign(screened_out=True)
    refset = ReferenceSet.for_rubric(frame, toy_rubric, GeneratorSettings())
    with pytest.raises(ConfigurationError, match="no reference texts"):
        embed_references(refset, StubEncoder([]))  # type: ignore[arg-type]


def test_multi_chunk_references_use_the_text_vector_and_are_counted(
    toy_rubric: Rubric, caplog: pytest.LogCaptureFixture
) -> None:
    refset = toy_reference_set(toy_rubric)
    chunks = [1] * 48
    chunks[2], chunks[30] = 3, 2
    encoder = StubEncoder(chunks)
    with caplog.at_level(logging.WARNING, logger="copernicus.similarity.references"):
        refs = embed_references(refset, encoder)  # type: ignore[arg-type]
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1 and "2 reference texts" in warnings[0].getMessage()
    expected = encoder.embed(list(refset.kept()["text"]), list(range(48))).vectors
    assert np.array_equal(refs.vectors, expected)


def test_single_chunk_references_log_no_warning(
    toy_rubric: Rubric, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING):
        stub_embeddings(toy_rubric)
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_pole_indices_span_levels_and_writers(toy_rubric: Rubric) -> None:
    refs = stub_embeddings(toy_rubric, ("writer-a", "writer-b"))
    indices = refs.pole_indices("construct", "negative")
    assert indices.size == 12 and indices.dtype.kind == "i"
    assert set(refs.writer_model[indices]) == {"writer-a", "writer-b"}
    assert set(refs.level[indices]) == {"top", "mid", "low"}
    assert set(refs.polarity[indices]) == {"negative"} and set(refs.target[indices]) == {
        "construct"
    }
    with pytest.raises(ConfigurationError, match="unknown target 'nonsense'"):
        refs.pole_indices("nonsense", "positive")
    with pytest.raises(ConfigurationError, match="unknown polarity 'neutral'"):
        refs.pole_indices("construct", "neutral")


def test_writers_targets_and_subset_writers(toy_rubric: Rubric) -> None:
    refs = stub_embeddings(toy_rubric, ("writer-b", "writer-a"))
    assert refs.writers() == ["writer-a", "writer-b"]
    assert refs.targets() == sorted(["construct", *(a.attribute_id for a in toy_rubric.attributes)])
    only = refs.subset_writers(["writer-b"])
    assert only.writers() == ["writer-b"] and only.vectors.shape[0] == 48
    assert np.array_equal(only.vectors, refs.vectors[refs.writer_model == "writer-b"])
    assert only.sidecar["n_references"] == 48 and only.rubric_id == refs.rubric_id
    assert only.rubric_hash == refs.rubric_hash == toy_rubric.content_hash()
    only.assert_complete(toy_rubric)
    with pytest.raises(ConfigurationError, match="unknown writer model 'writer-c'"):
        refs.subset_writers(["writer-a", "writer-c"])


def test_assert_complete_names_every_missing_pole(toy_rubric: Rubric) -> None:
    refs = stub_embeddings(toy_rubric)
    attribute = toy_rubric.attributes[0].attribute_id
    gone = ((refs.target == "construct") & (refs.polarity == "negative")) | (
        (refs.target == attribute) & (refs.polarity == "positive")
    )
    with pytest.raises(ConfigurationError) as raised:
        drop_rows(refs, gone).assert_complete(toy_rubric)
    message = str(raised.value)
    assert f"rubric {toy_rubric.rubric_id} lack" in message
    assert "the negative pole of target construct" in message
    assert f"the positive pole of target {attribute}" in message


def test_assert_complete_refuses_a_missing_target_and_another_rubric(toy_rubric: Rubric) -> None:
    refs = stub_embeddings(toy_rubric)
    attribute = toy_rubric.attributes[-1].attribute_id
    with pytest.raises(ConfigurationError, match=f"target {attribute}"):
        drop_rows(refs, refs.target == attribute).assert_complete(toy_rubric)
    other = replace(refs, rubric_id="other-rubric")
    with pytest.raises(ConfigurationError, match="other-rubric.*not for rubric"):
        other.assert_complete(toy_rubric)


def test_assert_complete_refuses_the_rubric_once_its_content_changed(toy_rubric: Rubric) -> None:
    refs = stub_embeddings(toy_rubric)
    changed = replace(toy_rubric, concept_definition=toy_rubric.concept_definition + " More.")
    expected = (
        f"the reference embeddings were generated for rubric {toy_rubric.rubric_id} with hash "
        f"{toy_rubric.content_hash()}, but the rubric now has hash {changed.content_hash()}: the "
        "references describe the earlier definition; regenerate them (or rebuild a hand-written "
        "set with ReferenceSet.for_rubric)"
    )
    for targets in (None, ["tone"]):
        with pytest.raises(ConfigurationError) as refused:
            refs.assert_complete(changed, targets)
        assert str(refused.value) == expected


def test_assert_complete_for_some_targets_needs_only_their_poles(toy_rubric: Rubric) -> None:
    refs = stub_embeddings(toy_rubric)
    without_construct = drop_rows(refs, refs.target == "construct")
    without_construct.assert_complete(toy_rubric, ["tone", "reasoning"])
    with pytest.raises(ConfigurationError, match="target construct"):
        without_construct.assert_complete(toy_rubric)
    with pytest.raises(ConfigurationError, match="the positive pole of target tone"):
        drop_rows(refs, (refs.target == "tone") & (refs.polarity == "positive")).assert_complete(
            toy_rubric, ["tone"]
        )


def test_the_rubric_hash_reaches_the_description_and_comes_back(
    toy_rubric: Rubric, tmp_path: Path
) -> None:
    refs = stub_embeddings(toy_rubric)
    _, sidecar_path = refs.save(tmp_path / "poles")
    description = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert description["rubric_hash"] == toy_rubric.content_hash()
    assert ReferenceEmbeddings.load(tmp_path / "poles").rubric_hash == toy_rubric.content_hash()


def test_load_refuses_a_description_without_a_rubric_hash_or_with_a_malformed_one(
    toy_rubric: Rubric, tmp_path: Path
) -> None:
    _, sidecar_path = stub_embeddings(toy_rubric).save(tmp_path / "poles")
    description = json.loads(sidecar_path.read_text(encoding="utf-8"))
    del description["rubric_hash"]
    sidecar_path.write_text(json.dumps(description), encoding="utf-8")
    with pytest.raises(ConfigurationError) as refused:
        ReferenceEmbeddings.load(tmp_path / "poles")
    assert str(refused.value) == (
        f"reference embeddings {tmp_path / 'poles'}: poles.json has no rubric_hash: it was "
        f"written by an earlier version of copernicus than this one ({__version__}); embed the "
        "reference set again (embed_references, or copernicus embed references), which makes "
        "no writer call"
    )
    sidecar_path.write_text(json.dumps({**description, "rubric_hash": "0" * 64}), encoding="utf-8")
    with pytest.raises(ConfigurationError, match=r"poles\.json: rubric_hash must be the content"):
        ReferenceEmbeddings.load(tmp_path / "poles")


@pytest.mark.parametrize("value", ["", "4BDEE2EFB58B0003", "0" * 64, None])
def test_construction_refuses_a_sidecar_without_the_content_hash_of_a_rubric(
    toy_rubric: Rubric, value: object
) -> None:
    refs = stub_embeddings(toy_rubric)
    bare = {key: entry for key, entry in refs.sidecar.items() if key != "rubric_hash"}
    with pytest.raises(ConfigurationError, match="lacks rubric_hash"):
        replace(refs, sidecar=bare)
    with pytest.raises(ConfigurationError, match="rubric_hash must be the content hash"):
        replace(refs, sidecar={**bare, "rubric_hash": value})


def test_load_refuses_another_encoder_and_missing_files(toy_rubric: Rubric, tmp_path: Path) -> None:
    refs = stub_embeddings(toy_rubric)
    refs.save(tmp_path / "poles")
    expected = (
        f"reference embeddings {tmp_path / 'poles'} were made with encoder stub, expected gte"
    )
    with pytest.raises(ConfigurationError) as raised:
        ReferenceEmbeddings.load(tmp_path / "poles", expect_encoder="gte")
    assert str(raised.value) == expected
    assert ReferenceEmbeddings.load(tmp_path / "poles", expect_encoder="stub").encoder == "stub"
    (tmp_path / "poles.json").unlink()
    with pytest.raises(ConfigurationError, match=r"poles\.json does not exist"):
        ReferenceEmbeddings.load(tmp_path / "poles")
    with pytest.raises(ConfigurationError, match=r"absent\.npz does not exist"):
        ReferenceEmbeddings.load(tmp_path / "absent")


def test_the_sidecar_records_identity_counts_and_versions(
    toy_rubric: Rubric, tmp_path: Path
) -> None:
    refs = stub_embeddings(toy_rubric)
    _, sidecar_path = refs.save(tmp_path / "poles")
    description = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert description["encoder"] == "stub" and description["hf_id"] == "org/stub"
    assert description["pooling"] == "mean" and description["max_tokens"] == 8
    assert description["dtype"] is None and description["rubric_id"] == toy_rubric.rubric_id
    assert description["n_references"] == 48 and description["prompt_versions"] == ["v1"]
    assert {"library_version", "created_at"} <= description.keys()
    assert (description["revision"], description["trust_remote_code"]) == ("abc123", False)
    assert (description["padding_side"], description["special_tokens_reserved"]) == ("right", 2)
    assert {key: description[key] for key in STUB_VERSIONS} == STUB_VERSIONS
    loaded = ReferenceEmbeddings.load(tmp_path / "poles")
    assert loaded.sidecar["n_references"] == 48
    assert not list(tmp_path.glob("*.partial"))


def test_save_refuses_a_sidecar_without_the_encoder_identity(
    toy_rubric: Rubric, tmp_path: Path
) -> None:
    binding = {"rubric_hash": toy_rubric.content_hash(), "bound_with": "for_rubric"}
    sidecar = {"prompt_versions": ["v1"], **binding}
    refs = replace(stub_embeddings(toy_rubric), sidecar=sidecar)
    with pytest.raises(ConfigurationError, match="lacks"):
        refs.save(tmp_path / "poles")
    assert not list(tmp_path.iterdir())


def test_construction_checks_lengths_norms_and_labels(toy_rubric: Rubric) -> None:
    refs = stub_embeddings(toy_rubric)
    with pytest.raises(ConfigurationError, match="47 level"):
        replace(refs, level=refs.level[:-1])
    with pytest.raises(ConfigurationError, match="unit length"):
        replace(refs, vectors=refs.vectors * 2.0)
    with pytest.raises(ConfigurationError, match="float32"):
        replace(refs, vectors=refs.vectors.astype(np.float64))
    with pytest.raises(ConfigurationError, match="two-dimensional"):
        replace(refs, vectors=refs.vectors[:, 0])
    odd_polarity = np.asarray(["neutral", *refs.polarity.tolist()[1:]])
    with pytest.raises(ConfigurationError, match="polarity.*neutral"):
        replace(refs, polarity=odd_polarity)
    odd_level = np.asarray(["high", *refs.level.tolist()[1:]])
    with pytest.raises(ConfigurationError, match="level.*high"):
        replace(refs, level=odd_level)
    with pytest.raises(ConfigurationError, match="no references"):
        drop_rows(refs, np.ones(48, dtype=bool))
