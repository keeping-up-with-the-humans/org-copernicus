"""Tests of the content digests that a sidecar records of its payload: tables and arrays.

Each digest is computed again here from the canonical text or bytes that its docstring
documents, with hashlib alone, so that a change of the format fails these tests.
"""

from __future__ import annotations

import hashlib
import struct

import numpy as np
import pandas as pd
import pytest

from copernicus._digests import ARRAYS_HASH_KEY, TABLE_HASH_KEY, arrays_hash, table_hash
from copernicus.errors import ConfigurationError


def sha256(content: bytes) -> str:
    """Return the SHA-256 digest of bytes as lowercase hexadecimal characters."""
    return hashlib.sha256(content).hexdigest()


def test_the_keys_are_the_names_that_the_descriptions_use() -> None:
    assert (TABLE_HASH_KEY, ARRAYS_HASH_KEY) == ("table_hash", "arrays_hash")


def test_the_table_hash_is_the_digest_of_the_documented_text() -> None:
    frame = pd.DataFrame(
        {
            "text": ["a b", None, "zwölf"],
            "count": [3, -1, 0],
            "share": [0.1 + 0.2, float("nan"), 2.0],
            "flag": [True, False, True],
            "left_out": ["x", "y", "z"],
        }
    )
    canonical = "true\ta b\t3\t0.30000000000000004\nfalse\tnull\t-1\tnull\ntrue\tzwölf\t0\t2.0"
    digest = table_hash(frame, ["flag", "text", "count", "share"])
    assert digest == sha256(canonical.encode("utf-8"))
    assert len(digest) == 64 and digest == digest.lower()


def test_numpy_scalars_and_every_missing_value_take_the_text_of_their_python_value() -> None:
    values = [np.int32(7), np.float32(0.1), np.bool_(False), pd.NA, None, np.nan, np.str_("ok")]
    frame = pd.DataFrame({"value": pd.Series(values, dtype=object)})
    canonical = "7\n0.10000000149011612\nfalse\nnull\nnull\nnull\nok"
    assert table_hash(frame, ["value"]) == sha256(canonical.encode("utf-8"))


def test_a_bool_column_is_not_hashed_as_integers() -> None:
    flags = pd.DataFrame({"value": [True, False]})
    numbers = pd.DataFrame({"value": [1, 0]})
    assert table_hash(flags, ["value"]) == sha256(b"true\nfalse")
    assert table_hash(numbers, ["value"]) == sha256(b"1\n0")


def test_a_table_without_rows_has_the_digest_of_the_empty_text() -> None:
    assert table_hash(pd.DataFrame({"value": []}), ["value"]) == sha256(b"")


@pytest.mark.parametrize("value", [pd.Timestamp("2026-01-01"), b"bytes", [1, 2], 1 + 2j])
def test_a_value_of_another_type_is_refused_naming_the_column_and_the_type(value: object) -> None:
    frame = pd.DataFrame({"fine": ["a"], "odd": pd.Series([value], dtype=object)})
    with pytest.raises(ConfigurationError) as refused:
        table_hash(frame, ["fine", "odd"])
    assert "'odd'" in str(refused.value) and type(value).__name__ in str(refused.value)


def test_the_arrays_hash_is_the_digest_of_the_documented_bytes() -> None:
    arrays = {
        "ids": np.asarray(["a", "bc"]),
        "vectors": np.asarray([[1.0, 2.0]], dtype=np.float32),
    }
    ids = "a".encode("utf-32-le") + b"\0" * 4 + "bc".encode("utf-32-le")
    vectors = struct.pack("<2f", 1.0, 2.0)
    expected = sha256(b"ids\0<U2\0" + b"2\0" + ids + b"vectors\0<f4\0" + b"1,2\0" + vectors)
    assert arrays_hash(arrays) == expected


def test_the_arrays_hash_follows_the_names_the_dtype_and_the_shape_but_not_the_layout() -> None:
    grid = np.arange(6, dtype=np.int64).reshape(2, 3)
    flags = np.asarray([True])
    assert arrays_hash({"a": grid, "b": flags}) != arrays_hash({"b": flags, "a": grid})
    assert arrays_hash({"a": np.asfortranarray(grid)}) == arrays_hash({"a": grid})
    assert arrays_hash({"a": grid.astype(np.int32)}) != arrays_hash({"a": grid})
    assert arrays_hash({"a": grid.reshape(3, 2)}) != arrays_hash({"a": grid})
    assert arrays_hash({"c": grid}) != arrays_hash({"a": grid})
