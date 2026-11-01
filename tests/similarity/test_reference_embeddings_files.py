"""Tests of the files of reference embeddings: the sidecar names the arrays it describes.

The arrays are written first and the sidecar last, each atomically. A save stopped between the
two writes leaves new vectors beside the old sidecar, as many when a rubric rewritten under the
same ids keeps its cells. The sidecar records the digest of its arrays (``arrays_hash``), so the
next load refuses such a pair instead of reading the new vectors under the old rubric hash.
"""

from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from hybrid_support import StubEncoder
from references_support import clean_responder, fail_write, texts_json, writer

import copernicus.similarity.references as reference_vectors
from copernicus import __version__
from copernicus._digests import arrays_hash
from copernicus.errors import ConfigurationError
from copernicus.references import GeneratorSettings, generate_references
from copernicus.rubric.model import Rubric
from copernicus.similarity import ReferenceEmbeddings, embed_references
from copernicus.similarity.references import LABEL_NAMES

SETTINGS = GeneratorSettings(texts_per_call=1, max_retries=1)
CELL_COUNT = 24
DIGEST = re.compile("[0-9a-f]{64}")


def arrays_of(refs: ReferenceEmbeddings) -> dict[str, np.ndarray]:
    """Return the arrays of reference embeddings in the order that ``save`` writes them."""
    return {"vectors": refs.vectors} | {name: getattr(refs, name) for name in LABEL_NAMES}


def sidecar_of(stem: Path) -> dict[str, object]:
    """Read the sidecar ``<stem>.json``."""
    return json.loads(stem.with_name(stem.name + ".json").read_text(encoding="utf-8"))


def rewrite_sidecar(stem: Path, drop: tuple[str, ...] = (), **values: object) -> None:
    """Drop keys from the sidecar ``<stem>.json``, or set them, as another version would."""
    kept = {key: value for key, value in sidecar_of(stem).items() if key not in drop}
    stem.with_name(stem.name + ".json").write_text(json.dumps({**kept, **values}), encoding="utf-8")


def files_of(folder: Path) -> dict[str, bytes]:
    """Return the content of every file of a folder, by name."""
    return {path.name: path.read_bytes() for path in folder.iterdir()}


@pytest.fixture
def old_refs(toy_rubric: Rubric) -> ReferenceEmbeddings:
    """The embeddings of one text per cell, written for the toy rubric."""
    refset = generate_references(toy_rubric, [writer(clean_responder(1))], SETTINGS)
    return embed_references(refset, StubEncoder())


@pytest.fixture
def revised_refs(rewritten_toy_rubric: Rubric) -> ReferenceEmbeddings:
    """The embeddings of other texts, written for the toy rubric rewritten under the same ids."""
    revised = writer(lambda request: texts_json(1, f"revised-{request.seed}"))
    refset = generate_references(rewritten_toy_rubric, [revised], SETTINGS)
    return embed_references(refset, StubEncoder())


def test_the_sidecar_names_the_arrays_it_was_written_with(
    old_refs: ReferenceEmbeddings, tmp_path: Path
) -> None:
    stem = tmp_path / "poles"
    old_refs.save(stem)
    recorded = sidecar_of(stem)["arrays_hash"]
    assert isinstance(recorded, str) and DIGEST.fullmatch(recorded)
    assert recorded == arrays_hash(arrays_of(old_refs))
    loaded = ReferenceEmbeddings.load(stem)
    for name, array in arrays_of(old_refs).items():
        assert np.array_equal(getattr(loaded, name), array), name
    assert loaded.sidecar["arrays_hash"] == recorded == arrays_hash(arrays_of(loaded))
    assert "arrays_hash" not in old_refs.sidecar


def test_an_overwrite_stopped_before_its_sidecar_is_refused_by_the_next_load(
    old_refs: ReferenceEmbeddings,
    revised_refs: ReferenceEmbeddings,
    toy_rubric: Rubric,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stem = tmp_path / "poles"
    old_refs.save(stem)
    assert len(revised_refs) == len(old_refs) == CELL_COUNT
    fail_write(monkeypatch, reference_vectors, "write_json")
    with pytest.raises(OSError, match="disk full"):
        revised_refs.save(stem)
    with np.load(tmp_path / "poles.npz") as arrays:
        assert np.array_equal(arrays["vectors"], revised_refs.vectors)
    assert sidecar_of(stem)["rubric_hash"] == toy_rubric.content_hash()
    for expected in (None, "stub"):
        with pytest.raises(ConfigurationError, match="do not belong together") as refused:
            ReferenceEmbeddings.load(stem, expect_encoder=expected)
        message = str(refused.value)
        assert message.startswith(f"{tmp_path / 'poles.npz'} and poles.json do not belong together")
        assert "a save stopped between its two writes leaves such a pair" in message


def test_a_failed_array_write_keeps_the_old_embeddings_and_their_digest(
    old_refs: ReferenceEmbeddings,
    revised_refs: ReferenceEmbeddings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stem = tmp_path / "poles"
    old_refs.save(stem)
    before = files_of(tmp_path)
    fail_write(monkeypatch, reference_vectors, "write_atomically")
    with pytest.raises(OSError, match="disk full"):
        revised_refs.save(stem)
    assert files_of(tmp_path) == before
    loaded = ReferenceEmbeddings.load(stem)
    assert np.array_equal(loaded.vectors, old_refs.vectors)
    assert loaded.rubric_hash == old_refs.rubric_hash
    digest = arrays_hash(arrays_of(loaded))
    assert digest == sidecar_of(stem)["arrays_hash"] == arrays_hash(arrays_of(old_refs))


def test_a_sidecar_without_an_arrays_hash_is_refused_as_an_earlier_version(
    old_refs: ReferenceEmbeddings, tmp_path: Path
) -> None:
    stem = tmp_path / "poles"
    old_refs.save(stem)
    rewrite_sidecar(stem, drop=("arrays_hash",))
    with pytest.raises(ConfigurationError) as refused:
        ReferenceEmbeddings.load(stem)
    assert str(refused.value) == (
        f"reference embeddings {stem}: poles.json has no arrays_hash: it was written by an "
        f"earlier version of copernicus than this one ({__version__}); embed the reference set "
        "again (embed_references, or copernicus embed references), which makes no writer call"
    )


@pytest.mark.parametrize("value", ["", "A" * 64, "0" * 65, "z" * 64, 7, None])
def test_a_malformed_arrays_hash_is_refused_naming_the_key(
    old_refs: ReferenceEmbeddings, tmp_path: Path, value: object
) -> None:
    stem = tmp_path / "poles"
    old_refs.save(stem)
    rewrite_sidecar(stem, arrays_hash=value)
    with pytest.raises(ConfigurationError, match=r"poles\.json: arrays_hash must be") as refused:
        ReferenceEmbeddings.load(stem)
    assert repr(value) in str(refused.value)


def test_a_saved_subset_of_writers_loads_with_the_digest_of_its_own_arrays(
    toy_rubric: Rubric, tmp_path: Path
) -> None:
    writers = [writer(clean_responder(1), name) for name in ("writer-a", "writer-b")]
    refs = embed_references(generate_references(toy_rubric, writers, SETTINGS), StubEncoder())
    refs.save(tmp_path / "both")
    loaded = ReferenceEmbeddings.load(tmp_path / "both")
    only = loaded.subset_writers(["writer-b"])
    assert "arrays_hash" not in only.sidecar
    assert loaded.sidecar["arrays_hash"] == sidecar_of(tmp_path / "both")["arrays_hash"]
    only.save(tmp_path / "only")
    assert sidecar_of(tmp_path / "only")["arrays_hash"] == arrays_hash(arrays_of(only))
    again = ReferenceEmbeddings.load(tmp_path / "only")
    assert again.writers() == ["writer-b"] and np.array_equal(again.vectors, only.vectors)
    assert again.rubric_hash == toy_rubric.content_hash() and len(again) == CELL_COUNT


def test_save_records_the_digest_of_the_arrays_it_writes_not_the_one_it_read(
    old_refs: ReferenceEmbeddings, tmp_path: Path
) -> None:
    stem = tmp_path / "poles"
    old_refs.save(stem)
    loaded = ReferenceEmbeddings.load(stem)
    flipped = replace(loaded, vectors=-loaded.vectors)
    assert flipped.sidecar["arrays_hash"] == sidecar_of(stem)["arrays_hash"]
    flipped.save(stem)
    again = ReferenceEmbeddings.load(stem)
    assert np.array_equal(again.vectors, -old_refs.vectors)
    digest = arrays_hash(arrays_of(flipped))
    assert sidecar_of(stem)["arrays_hash"] == digest == again.sidecar["arrays_hash"]
