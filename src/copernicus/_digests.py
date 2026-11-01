"""Content digests that a sidecar records of its payload, so that a load can tell they belong.

A payload file and the JSON sidecar beside it are written one after the other, each atomically.
A save that stops between the two writes leaves a new payload beside the old sidecar, and their
counts and ids can agree. The sidecar therefore names the content of its payload: a reference
set file records ``table_hash``, the digest of its table (`table_hash`), and reference
embeddings and an embedding set record ``arrays_hash``, the digest of their arrays
(`arrays_hash`). A load computes the digest of what it read and refuses a payload that the
sidecar does not name.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from copernicus._io import stem_paths
from copernicus._version import __version__
from copernicus.errors import ConfigurationError

__all__ = [
    "ARRAYS_HASH_KEY",
    "TABLE_HASH_KEY",
    "arrays_hash",
    "check_arrays_hash",
    "digest_problem",
    "table_hash",
    "without_arrays_hash",
]

# The keys of the digests in the descriptions of the payloads.
TABLE_HASH_KEY = "table_hash"
ARRAYS_HASH_KEY = "arrays_hash"
DIGEST_LENGTH = 64
DIGEST_PATTERN = re.compile(rf"[0-9a-f]{{{DIGEST_LENGTH}}}")
MISSING_TEXT = "null"
VALUE_SEPARATOR = "\t"
ROW_SEPARATOR = "\n"
NAME_END = b"\0"


def _value_text(value: object, column: str) -> str:
    """Return the canonical text of one value of a table, as `table_hash` documents it."""
    if value is None or value is pd.NA:
        return MISSING_TEXT
    if isinstance(value, (bool, np.bool_)):
        return "true" if value else "false"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return MISSING_TEXT if math.isnan(value) else repr(float(value))
    if isinstance(value, str):
        return value
    raise ConfigurationError(
        f"cannot compute the table hash: column {column!r} holds a value of type "
        f"{type(value).__name__}; the hash covers strings, integers, floats, booleans and "
        "missing values"
    )


def table_hash(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    """Return the SHA-256 digest of the rows of a table, restricted to some of its columns.

    The digest is taken over a canonical text, which another reader can build again: for
    each row of ``frame``, in frame order, the values of ``columns`` in the given order, each
    turned into text and joined with a tab; the rows joined with a newline, without a final
    one; the whole encoded as UTF-8. A value becomes ``null`` when it is missing (``None``,
    NaN or ``pandas.NA``); ``true`` or ``false`` when it is a bool, Python's or numpy's (a
    bool is told apart before the integers, since Python counts it as one);
    ``str(int(value))``, such as ``-1``, when it is an integer, Python's or numpy's;
    ``repr(float(value))``, the shortest text that reads back as the same float, when it is
    a float; and the string itself when it is a string. The index of the frame does not
    enter, and a table without rows has the digest of the empty text.

    Args:
        frame: The table; it holds every column of ``columns``.
        columns: The columns whose values enter, in this order.

    Returns:
        64 lowercase hexadecimal characters.

    Raises:
        ConfigurationError: Naming the column and the type, if a value is of another type,
            such as a timestamp, bytes or a list.
    """
    texts = [[_value_text(value, name) for value in frame[name].tolist()] for name in columns]
    rows = (VALUE_SEPARATOR.join(values) for values in zip(*texts, strict=True))
    return hashlib.sha256(ROW_SEPARATOR.join(rows).encode("utf-8")).hexdigest()


def arrays_hash(arrays: Mapping[str, np.ndarray]) -> str:
    """Return the SHA-256 digest of named arrays, in the order of the mapping.

    The digest is taken over bytes that another reader can build again: for each name, in
    the order of the mapping, the name in UTF-8, a NUL byte, ``array.dtype.str`` (such as
    ``<f4``), a NUL byte, the shape as decimal numbers joined with commas, a NUL byte, and
    the bytes of ``np.ascontiguousarray(array)``, the values in C order. An array of another
    memory layout has the digest of its C-ordered copy; another name, order of the names,
    dtype or shape gives another digest.

    Args:
        arrays: The arrays by name, as a file holds them.

    Returns:
        64 lowercase hexadecimal characters.
    """
    digest = hashlib.sha256()
    for name, value in arrays.items():
        array = np.asarray(value)
        shape = ",".join(str(size) for size in array.shape)
        digest.update(name.encode("utf-8") + NAME_END + array.dtype.str.encode("ascii") + NAME_END)
        digest.update(shape.encode("ascii") + NAME_END)
        digest.update(np.ascontiguousarray(array).data)
    return digest.hexdigest()


def digest_problem(key: str, value: object) -> str | None:
    """Say why a recorded digest is malformed, or return ``None`` when it has the right form.

    Args:
        key: The key that records the digest, named in the message.
        value: The value recorded under the key.

    Returns:
        ``None`` for a string of 64 lowercase hexadecimal characters, the form of the digests
        of this module; otherwise the reason.
    """
    if isinstance(value, str) and DIGEST_PATTERN.fullmatch(value) is not None:
        return None
    return (
        f"{key} must be a SHA-256 digest, {DIGEST_LENGTH} lowercase hexadecimal characters, "
        f"got {value!r}"
    )


def check_arrays_hash(
    description: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    stem: str | Path,
    owner: str,
    remedy: str,
) -> None:
    """Refuse arrays that the description of their file does not name.

    Args:
        description: The description read from ``<stem>.json``.
        arrays: The arrays read from ``<stem>.npz``, in the order they are written.
        stem: The path of the files without suffix, named in the messages.
        owner: What the files hold, such as ``"embedding set"``, named in the messages.
        remedy: How the reader makes the files again, for the messages.

    Raises:
        ConfigurationError: If the description records no ``arrays_hash`` (it was written by
            an earlier version), a value that is not a digest (naming the key), or the digest
            of other arrays (naming both files and both digests), as a save stopped between
            its two writes leaves.
    """
    array_path, sidecar_path = stem_paths(stem)
    if ARRAYS_HASH_KEY not in description:
        raise ConfigurationError(
            f"{owner} {stem}: {sidecar_path.name} has no {ARRAYS_HASH_KEY}: it was written by "
            f"an earlier version of copernicus than this one ({__version__}); {remedy}"
        )
    recorded = description[ARRAYS_HASH_KEY]
    problem = digest_problem(ARRAYS_HASH_KEY, recorded)
    if problem is not None:
        raise ConfigurationError(f"{owner} {stem}: {sidecar_path.name}: {problem}")
    actual = arrays_hash(arrays)
    if actual != recorded:
        raise ConfigurationError(
            f"{array_path} and {sidecar_path.name} do not belong together: the arrays have hash "
            f"{actual}, the description records {recorded}; a save stopped between its two "
            f"writes leaves such a pair: save the {owner} again, or {remedy}"
        )


def without_arrays_hash(sidecar: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy of a sidecar without ``arrays_hash``, for an object of other arrays.

    A subset holds other arrays than the file its sidecar was read from; its own save
    records the digest of the arrays it writes.

    Args:
        sidecar: The sidecar of the object the subset was taken from.

    Returns:
        The other keys of the sidecar.
    """
    return {key: value for key, value in sidecar.items() if key != ARRAYS_HASH_KEY}
