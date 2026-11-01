"""Tests for the embedding set and its files."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest
from embeddings_support import TOY_IDENTITY, TOY_VERSIONS, small_set
from references_support import fail_write

import copernicus.embeddings.sets as embedding_sets
from copernicus import __version__
from copernicus._digests import arrays_hash
from copernicus.embeddings import EmbeddingSet
from copernicus.errors import ConfigurationError

ARRAYS = ("ids", "vectors", "chunk_vectors", "chunk_owner", "chunk_weight")
DIGEST = re.compile("[0-9a-f]{64}")


def arrays_of(embedded: EmbeddingSet) -> dict[str, np.ndarray]:
    """Return the arrays of an embedding set in the order that ``save`` writes them."""
    return {name: getattr(embedded, name) for name in ARRAYS}


def sidecar_of(stem: Path) -> dict[str, object]:
    """Read the sidecar ``<stem>.json``."""
    return json.loads(stem.with_name(stem.name + ".json").read_text(encoding="utf-8"))


def files_of(folder: Path) -> dict[str, bytes]:
    """Return the content of every file of a folder, by name."""
    return {path.name: path.read_bytes() for path in folder.iterdir()}


def test_embedding_set_roundtrip_and_encoder_check(tmp_path: Path) -> None:
    original = small_set()
    array_path, sidecar_path = original.save(tmp_path / "toy_set")
    assert (array_path.name, sidecar_path.name) == ("toy_set.npz", "toy_set.json")
    loaded = EmbeddingSet.load(tmp_path / "toy_set", expect_encoder="toy")
    for name in ARRAYS:
        assert np.array_equal(getattr(loaded, name), getattr(original, name)), name
        assert getattr(loaded, name).dtype == getattr(original, name).dtype
    assert loaded.encoder == "toy"
    assert loaded.metadata is not None and original.metadata is not None
    assert loaded.metadata.equals(original.metadata)
    with pytest.raises(ConfigurationError) as error:
        EmbeddingSet.load(tmp_path / "toy_set", expect_encoder="other")
    assert (
        str(error.value)
        == f"embedding set {tmp_path / 'toy_set'} was made with encoder toy, expected other"
    )


def test_the_sidecar_describes_the_encoder_the_counts_and_the_library(tmp_path: Path) -> None:
    small_set().save(tmp_path / "s")
    description = json.loads((tmp_path / "s.json").read_text(encoding="utf-8"))
    assert {
        key: description[key] for key in description if key not in ("metadata", "created_at")
    } == {
        "encoder": "toy",
        **TOY_IDENTITY,
        **TOY_VERSIONS,
        "n_texts": 3,
        "n_chunks": 4,
        "library_version": __version__,
        "arrays_hash": arrays_hash(arrays_of(small_set())),
    }
    assert description["created_at"].endswith("+00:00")
    assert description["metadata"] == [
        {"text_id": "t0", "group_id": "a"},
        {"text_id": "t1", "group_id": "b"},
        {"text_id": "t2", "group_id": "c"},
    ]
    loaded = EmbeddingSet.load(tmp_path / "s")
    assert loaded.sidecar["n_chunks"] == 4 and loaded.sidecar["hf_id"] == "org/toy"


def test_a_set_without_metadata_writes_none_and_loads_none(tmp_path: Path) -> None:
    small_set(with_metadata=False).save(tmp_path / "s")
    assert "metadata" not in json.loads((tmp_path / "s.json").read_text(encoding="utf-8"))
    assert EmbeddingSet.load(tmp_path / "s").metadata is None


def test_a_missing_file_is_a_configuration_error(tmp_path: Path) -> None:
    small_set().save(tmp_path / "s")
    (tmp_path / "s.json").unlink()
    with pytest.raises(ConfigurationError, match=r"s\.json does not exist"):
        EmbeddingSet.load(tmp_path / "s")
    with pytest.raises(ConfigurationError, match=r"nothing\.npz does not exist"):
        EmbeddingSet.load(tmp_path / "nothing")


def test_a_sidecar_that_disagrees_with_the_arrays_is_refused(tmp_path: Path) -> None:
    small_set().save(tmp_path / "s")
    sidecar = json.loads((tmp_path / "s.json").read_text(encoding="utf-8"))
    sidecar["n_texts"] = 99
    (tmp_path / "s.json").write_text(json.dumps(sidecar), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="counts 99 texts"):
        EmbeddingSet.load(tmp_path / "s")


def test_saving_needs_the_identity_of_the_encoder(tmp_path: Path) -> None:
    from dataclasses import replace

    bare = replace(small_set(), sidecar={})
    with pytest.raises(ConfigurationError, match="hf_id"):
        bare.save(tmp_path / "s")
    no_revision = {key: value for key, value in TOY_IDENTITY.items() if key != "revision"}
    with pytest.raises(ConfigurationError, match=r"\['revision'\]"):
        replace(small_set(), sidecar=no_revision).save(tmp_path / "s")
    assert not list(tmp_path.iterdir())


def test_the_versions_are_written_when_known_and_left_out_otherwise(tmp_path: Path) -> None:
    from dataclasses import replace

    replace(small_set(), sidecar=dict(TOY_IDENTITY)).save(tmp_path / "s")
    description = json.loads((tmp_path / "s.json").read_text(encoding="utf-8"))
    assert not set(TOY_VERSIONS) & description.keys()
    assert {key: description[key] for key in TOY_IDENTITY} == TOY_IDENTITY


def test_inconsistent_arrays_are_refused() -> None:
    from dataclasses import replace

    base = small_set()
    with pytest.raises(ConfigurationError, match="every text needs at least one chunk"):
        replace(base, chunk_owner=np.array([0, 0, 0, 1], dtype=np.int64))
    with pytest.raises(ConfigurationError, match="ids are not unique"):
        replace(base, ids=np.asarray(["t0", "t0", "t2"]))
    with pytest.raises(ConfigurationError, match="metadata rows"):
        replace(base, metadata=base.metadata.head(2))  # type: ignore[union-attr]
    with pytest.raises(ConfigurationError, match="differ in length"):
        replace(base, chunk_weight=base.chunk_weight[:3])


def test_subset_keeps_the_given_texts_in_the_given_order_with_their_chunks() -> None:
    original = small_set()
    picked = original.subset(["t2", "t1"])
    assert picked.ids.tolist() == ["t2", "t1"] and len(picked) == 2
    assert np.array_equal(picked.vectors, original.vectors[[2, 1]])
    assert picked.chunk_owner.tolist() == [0, 1, 1]
    assert np.array_equal(picked.chunk_vectors, original.chunk_vectors[[3, 1, 2]])
    assert picked.chunk_weight.tolist() == [4.0, 2.0, 3.0]
    assert picked.metadata is not None and picked.metadata.text_id.tolist() == ["t2", "t1"]
    assert picked.metadata.index.tolist() == [0, 1]
    assert (picked.sidecar["n_texts"], picked.sidecar["n_chunks"]) == (2, 3)
    assert len(original) == 3 and original.sidecar.get("n_texts") is None


def test_subset_by_integer_ids_matches_string_ids_and_rejects_unknown_or_repeated_ids() -> None:
    original = small_set()
    assert original.subset(["t0"]).ids.tolist() == ["t0"]
    with pytest.raises(ConfigurationError, match="1 ids are not in the embedding set: 'nope'"):
        original.subset(["t0", "nope"])
    with pytest.raises(ConfigurationError, match="unique"):
        original.subset(["t0", "t0"])


def test_first_chunk_vectors_are_the_first_chunk_of_each_text() -> None:
    original = small_set()
    assert np.array_equal(original.first_chunk_vectors(), original.chunk_vectors[[0, 1, 3]])
    assert np.array_equal(
        original.subset(["t1", "t0"]).first_chunk_vectors(), original.chunk_vectors[[1, 0]]
    )


def test_a_sidecar_that_counts_other_chunks_than_the_arrays_hold_is_refused(tmp_path: Path) -> None:
    small_set().save(tmp_path / "s")
    sidecar = json.loads((tmp_path / "s.json").read_text(encoding="utf-8"))
    sidecar["n_chunks"] = 5
    (tmp_path / "s.json").write_text(json.dumps(sidecar), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="counts 5 chunks"):
        EmbeddingSet.load(tmp_path / "s")
    del sidecar["n_chunks"]
    (tmp_path / "s.json").write_text(json.dumps(sidecar), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="lacks"):
        EmbeddingSet.load(tmp_path / "s")


def test_saving_leaves_no_partial_file(tmp_path: Path) -> None:
    small_set().save(tmp_path / "s")
    assert sorted(path.name for path in tmp_path.iterdir()) == ["s.json", "s.npz"]


def test_integer_ids_become_strings_and_survive_a_round_trip(tmp_path: Path) -> None:
    from dataclasses import replace

    numbered = replace(small_set(), ids=np.arange(3))
    assert numbered.ids.dtype.kind == "U" and numbered.ids.tolist() == ["0", "1", "2"]
    numbered.save(tmp_path / "s")
    loaded = EmbeddingSet.load(tmp_path / "s")
    assert loaded.ids.tolist() == ["0", "1", "2"] and loaded.subset(["2"]).ids.tolist() == ["2"]


def test_object_and_list_ids_become_string_arrays_and_survive_a_round_trip(
    tmp_path: Path,
) -> None:
    from dataclasses import replace

    objects = replace(small_set(), ids=np.asarray(["a", "b", "c"], dtype=object))
    listed = replace(small_set(), ids=["a", "b", "c"])  # type: ignore[arg-type]
    for each in (objects, listed):
        assert each.ids.dtype.kind == "U" and each.ids.tolist() == ["a", "b", "c"]
    objects.save(tmp_path / "s")
    assert EmbeddingSet.load(tmp_path / "s").ids.tolist() == ["a", "b", "c"]


def test_ids_that_are_not_one_dimensional_strings_or_integers_are_refused() -> None:
    from dataclasses import replace

    with pytest.raises(ConfigurationError, match="one-dimensional"):
        replace(small_set(), ids=np.asarray([["t0"], ["t1"], ["t2"]]))
    with pytest.raises(ConfigurationError, match="None"):
        replace(small_set(), ids=np.asarray(["t0", None, "t2"], dtype=object))


@pytest.mark.parametrize("content", ["{not json", "[1, 2]"])
def test_a_malformed_sidecar_is_a_configuration_error_naming_the_file(
    tmp_path: Path, content: str
) -> None:
    small_set().save(tmp_path / "s")
    (tmp_path / "s.json").write_text(content, encoding="utf-8")
    with pytest.raises(ConfigurationError, match=r"s\.json"):
        EmbeddingSet.load(tmp_path / "s")


def test_a_failed_save_keeps_the_earlier_files_and_leaves_no_partial_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    small_set().save(tmp_path / "s")
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}

    def broken(handle: object, **arrays: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr("copernicus.embeddings.sets.np.savez", broken)
    with pytest.raises(OSError, match="disk full"):
        small_set(encoder="other").save(tmp_path / "s")
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == before


def test_the_sidecar_names_the_arrays_it_was_written_with_and_keeps_the_digest_on_load(
    tmp_path: Path,
) -> None:
    original = small_set()
    original.save(tmp_path / "s")
    recorded = sidecar_of(tmp_path / "s")["arrays_hash"]
    assert isinstance(recorded, str) and DIGEST.fullmatch(recorded)
    assert recorded == arrays_hash(arrays_of(original))
    loaded = EmbeddingSet.load(tmp_path / "s")
    assert loaded.sidecar["arrays_hash"] == recorded == arrays_hash(arrays_of(loaded))
    assert loaded.metadata is not None and original.metadata is not None
    assert loaded.metadata.equals(original.metadata)
    assert "arrays_hash" not in original.sidecar


def test_an_overwrite_stopped_before_its_sidecar_is_refused_by_the_next_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dataclasses import replace

    original = small_set()
    other = replace(
        small_set(encoder="other"),
        vectors=-original.vectors,
        chunk_vectors=-original.chunk_vectors,
        sidecar={**TOY_IDENTITY, "hf_id": "org/other"},
    )
    original.save(tmp_path / "s")
    fail_write(monkeypatch, embedding_sets, "write_json")
    with pytest.raises(OSError, match="disk full"):
        other.save(tmp_path / "s")
    with np.load(tmp_path / "s.npz") as arrays:
        assert np.array_equal(arrays["vectors"], other.vectors)
    assert sidecar_of(tmp_path / "s")["encoder"] == "toy"
    for expected in (None, "toy"):
        with pytest.raises(ConfigurationError, match="do not belong together") as refused:
            EmbeddingSet.load(tmp_path / "s", expect_encoder=expected)
        message = str(refused.value)
        assert message.startswith(f"{tmp_path / 's.npz'} and s.json do not belong together")
        assert "a save stopped between its two writes leaves such a pair" in message


def test_a_failed_array_write_keeps_the_old_set_and_its_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = small_set()
    original.save(tmp_path / "s")
    before = files_of(tmp_path)
    fail_write(monkeypatch, embedding_sets, "write_atomically")
    with pytest.raises(OSError, match="disk full"):
        small_set(encoder="other").save(tmp_path / "s")
    assert files_of(tmp_path) == before
    loaded = EmbeddingSet.load(tmp_path / "s", expect_encoder="toy")
    for name in ARRAYS:
        assert np.array_equal(getattr(loaded, name), getattr(original, name)), name
    assert loaded.sidecar["arrays_hash"] == arrays_hash(arrays_of(original))


def test_a_sidecar_without_an_arrays_hash_is_refused_as_an_earlier_version(tmp_path: Path) -> None:
    small_set().save(tmp_path / "s")
    description = sidecar_of(tmp_path / "s")
    del description["arrays_hash"]
    (tmp_path / "s.json").write_text(json.dumps(description), encoding="utf-8")
    with pytest.raises(ConfigurationError) as refused:
        EmbeddingSet.load(tmp_path / "s")
    assert str(refused.value) == (
        f"embedding set {tmp_path / 's'}: s.json has no arrays_hash: it was written by an "
        f"earlier version of copernicus than this one ({__version__}); embed the texts again "
        "(Encoder.embed, or copernicus embed texts)"
    )


@pytest.mark.parametrize("value", ["", "A" * 64, "0" * 63, 7, None])
def test_a_malformed_arrays_hash_is_refused_naming_the_key(tmp_path: Path, value: object) -> None:
    small_set().save(tmp_path / "s")
    description = sidecar_of(tmp_path / "s")
    (tmp_path / "s.json").write_text(json.dumps({**description, "arrays_hash": value}))
    with pytest.raises(ConfigurationError, match=r"s\.json: arrays_hash must be") as refused:
        EmbeddingSet.load(tmp_path / "s")
    assert repr(value) in str(refused.value)


def test_a_saved_subset_loads_with_the_digest_of_its_own_arrays(tmp_path: Path) -> None:
    small_set().save(tmp_path / "s")
    picked = EmbeddingSet.load(tmp_path / "s").subset(["t2", "t1"])
    assert "arrays_hash" not in picked.sidecar
    picked.save(tmp_path / "picked")
    assert sidecar_of(tmp_path / "picked")["arrays_hash"] == arrays_hash(arrays_of(picked))
    again = EmbeddingSet.load(tmp_path / "picked")
    assert again.ids.tolist() == ["t2", "t1"]
    assert np.array_equal(again.chunk_vectors, picked.chunk_vectors)


def test_save_records_the_digest_of_the_arrays_it_writes_not_the_one_it_read(
    tmp_path: Path,
) -> None:
    from dataclasses import replace

    small_set().save(tmp_path / "s")
    loaded = EmbeddingSet.load(tmp_path / "s")
    flipped = replace(loaded, vectors=-loaded.vectors)
    assert flipped.sidecar["arrays_hash"] == sidecar_of(tmp_path / "s")["arrays_hash"]
    flipped.save(tmp_path / "s")
    again = EmbeddingSet.load(tmp_path / "s")
    assert np.array_equal(again.vectors, -small_set().vectors)
    digest = arrays_hash(arrays_of(flipped))
    assert sidecar_of(tmp_path / "s")["arrays_hash"] == digest == again.sidecar["arrays_hash"]
