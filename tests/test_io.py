"""Tests of the file and array helpers shared by the containers with a JSON sidecar."""

from __future__ import annotations

import json
from pathlib import Path
from types import MappingProxyType
from typing import BinaryIO

import numpy as np
import pytest

from copernicus._io import (
    IDENTITY_KEYS,
    VERSION_KEYS,
    as_unicode_1d,
    check_same_encoder,
    encoder_identity,
    encoder_versions,
    json_ready,
    read_sidecar,
    stem_paths,
    write_atomically,
    write_json,
)
from copernicus.errors import ConfigurationError

IDENTITY = {
    "hf_id": "org/x",
    "revision": "unpinned",
    "pooling": "mean",
    "max_tokens": 8,
    "dtype": None,
    "trust_remote_code": False,
    "padding_side": "right",
    "special_tokens_reserved": 2,
}


def test_stem_paths_add_the_two_suffixes(tmp_path: Path) -> None:
    assert stem_paths(tmp_path / "set.v1") == (tmp_path / "set.v1.npz", tmp_path / "set.v1.json")


def test_a_failed_write_keeps_the_old_file_and_leaves_no_partial_file(tmp_path: Path) -> None:
    target = tmp_path / "file.bin"
    target.write_bytes(b"old")

    def broken(handle: BinaryIO) -> None:
        handle.write(b"half")
        raise OSError("disk full")

    with pytest.raises(OSError, match="disk full"):
        write_atomically(target, broken)
    assert target.read_bytes() == b"old"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["file.bin"]


def test_write_json_writes_indented_utf8_with_a_final_newline(tmp_path: Path) -> None:
    target = tmp_path / "description.json"
    write_json(target, {"name": "Größe", "n": 2})
    text = target.read_text(encoding="utf-8")
    assert text.endswith("}\n") and "Größe" in text and '\n  "n": 2' in text
    assert json.loads(text) == {"name": "Größe", "n": 2}


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (b"{not json", "is not valid JSON"),
        (b"[1, 2]", "does not hold a JSON object"),
        (b"\xff\xfe{}", "is not UTF-8 text"),
    ],
)
def test_a_malformed_sidecar_is_a_configuration_error_naming_the_file(
    tmp_path: Path, content: bytes, message: str
) -> None:
    path = tmp_path / "set.json"
    path.write_bytes(content)
    with pytest.raises(ConfigurationError, match=message) as error:
        read_sidecar(path)
    assert str(path) in str(error.value)


def test_the_identity_keys_are_the_settings_that_decide_the_vectors() -> None:
    assert IDENTITY_KEYS == tuple(IDENTITY)
    assert VERSION_KEYS == ("resolved_revision", "torch_version", "transformers_version")


def test_encoder_identity_returns_the_keys_in_order_and_names_missing_ones() -> None:
    sidecar = {**dict(reversed(list(IDENTITY.items()))), "other": 1}
    assert list(encoder_identity(sidecar, "embedding set")) == list(IDENTITY_KEYS)
    lacking = {key: value for key, value in IDENTITY.items() if key != "pooling"}
    with pytest.raises(ConfigurationError, match=r"embedding set lacks \['pooling'\]"):
        encoder_identity(lacking, "embedding set")


def test_encoder_versions_keep_the_known_version_keys_only() -> None:
    assert encoder_versions({**IDENTITY, "torch_version": "2.5"}) == {"torch_version": "2.5"}
    assert encoder_versions(IDENTITY) == {}


def test_two_sidecars_of_one_encoder_pass_whatever_their_versions() -> None:
    check_same_encoder(
        {**IDENTITY, "torch_version": "1"}, {**IDENTITY, "torch_version": "2"}, "texts", "refs"
    )


def test_sidecars_that_differ_in_an_identity_key_are_refused_naming_each_key() -> None:
    other = {**IDENTITY, "hf_id": "org/y", "padding_side": "left"}
    with pytest.raises(ConfigurationError) as error:
        check_same_encoder(IDENTITY, other, "texts", "reference embeddings")
    message = str(error.value)
    assert "texts" in message and "reference embeddings" in message
    assert "hf_id ('org/x' and 'org/y')" in message
    assert "padding_side ('right' and 'left')" in message and "pooling" not in message


def test_a_sidecar_without_an_identity_key_cannot_be_compared() -> None:
    lacking = {key: value for key, value in IDENTITY.items() if key != "revision"}
    with pytest.raises(ConfigurationError, match=r"references lacks \['revision'\]"):
        check_same_encoder(IDENTITY, lacking, "texts", "references")


def test_sidecars_loaded_from_different_commits_are_refused_naming_both_commits() -> None:
    texts = {**IDENTITY, "resolved_revision": "aaa111"}
    references = {**IDENTITY, "resolved_revision": "bbb222"}
    with pytest.raises(ConfigurationError) as error:
        check_same_encoder(texts, references, "texts", "reference embeddings")
    message = str(error.value)
    assert "texts" in message and "reference embeddings" in message and "org/x" in message
    assert "different commits" in message and "'aaa111' and 'bbb222'" in message


@pytest.mark.parametrize(
    "other",
    [{"resolved_revision": "aaa111"}, {"resolved_revision": "unknown"}, {}],
    ids=["same commit", "unknown commit", "no commit"],
)
def test_the_commits_are_compared_only_when_both_sidecars_record_one(
    other: dict[str, str],
) -> None:
    recorded = {**IDENTITY, "resolved_revision": "aaa111"}
    check_same_encoder(recorded, {**IDENTITY, **other}, "texts", "references")
    check_same_encoder({**IDENTITY, **other}, recorded, "texts", "references")


def test_integer_ids_become_strings() -> None:
    converted = as_unicode_1d(np.arange(3), "ids")
    assert converted.dtype.kind == "U" and converted.tolist() == ["0", "1", "2"]


def test_lists_and_object_arrays_of_strings_or_integers_become_string_arrays() -> None:
    from_list = as_unicode_1d(["a", 7, np.int64(8)], "ids")
    assert from_list.dtype.kind == "U" and from_list.tolist() == ["a", "7", "8"]
    objects = np.asarray(["x", "y"], dtype=object)
    assert as_unicode_1d(objects, "labels").tolist() == ["x", "y"]
    assert as_unicode_1d(objects, "labels").dtype.kind == "U"


def test_a_string_array_is_returned_as_it_is_and_an_empty_one_stays_empty() -> None:
    strings = np.asarray(["a", "b"])
    assert as_unicode_1d(strings, "ids") is strings
    empty = as_unicode_1d([], "ids")
    assert empty.dtype.kind == "U" and empty.shape == (0,)


@pytest.mark.parametrize(
    ("values", "message"),
    [
        (np.asarray([["a"], ["b"]]), "one-dimensional"),
        ("abc", "one-dimensional"),
        (np.asarray(["a", None], dtype=object), "None"),
        (["a", 1.5], "1.5"),
        ([True, False], "True"),
        (np.asarray([1.0, 2.0]), "float64"),
        (np.asarray([b"a"]), "dtype"),
        (np.asarray([["a"], ["b", "c"]], dtype=object), "neither|strings or integers"),
    ],
)
def test_values_that_are_not_strings_or_integers_are_refused(values: object, message: str) -> None:
    with pytest.raises(ConfigurationError, match=message):
        as_unicode_1d(values, "ids")


def test_json_ready_turns_numpy_values_and_mappings_into_plain_values() -> None:
    value = {
        "count": np.int64(3),
        "share": np.float32(0.5),
        "flag": np.bool_(True),
        "vector": np.array([1.0, 2.0]),
        "nested": MappingProxyType({"pair": (1, "a"), 4: [np.int32(2)]}),
    }
    plain = json_ready(value)
    assert plain == {
        "count": 3,
        "share": 0.5,
        "flag": True,
        "vector": [1.0, 2.0],
        "nested": {"pair": [1, "a"], "4": [2]},
    }
    assert type(plain["count"]) is int and type(plain["flag"]) is bool
    assert json.loads(json.dumps(plain)) == plain


def test_json_ready_writes_non_finite_numbers_as_null() -> None:
    provenance = [
        {"fold": 0, "mae": float("nan"), "slope": np.float64("inf")},
        {"nested": MappingProxyType({"values": (np.float32("nan"), -np.inf, 1.5)})},
    ]
    plain = json_ready(provenance)
    assert plain == [
        {"fold": 0, "mae": None, "slope": None},
        {"nested": {"values": [None, None, 1.5]}},
    ]
    assert json.loads(json.dumps(plain, allow_nan=False)) == plain
