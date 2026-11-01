"""File and array helpers shared by the containers that are saved with a JSON sidecar."""

from __future__ import annotations

import json
import logging
import math
import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np

from copernicus.errors import ConfigurationError

logger = logging.getLogger(__name__)

__all__ = [
    "IDENTITY_KEYS",
    "LOADED_COMMIT_KEY",
    "UNKNOWN_REVISION",
    "VERSION_KEYS",
    "as_unicode_1d",
    "check_same_encoder",
    "encoder_identity",
    "encoder_versions",
    "json_ready",
    "read_sidecar",
    "stem_paths",
    "write_atomically",
    "write_json",
]

# The settings that decide the vectors; texts and references must agree on every one.
IDENTITY_KEYS = (
    "hf_id",
    "revision",
    "pooling",
    "max_tokens",
    "dtype",
    "trust_remote_code",
    "padding_side",
    "special_tokens_reserved",
)
# The version key of the commit the weights were loaded from, and its value when the model
# names none.
LOADED_COMMIT_KEY = "resolved_revision"
UNKNOWN_REVISION = "unknown"
# Recorded with every set. The loaded commit is compared when both descriptions record one
# other than "unknown" (see check_same_encoder); the library versions are never compared. A
# model repository gets a new commit even when only its documentation changes, so sets embedded
# from its default branch before and after such a commit must be re-embedded, or embedded with
# one pinned revision, to be scored together.
VERSION_KEYS = (LOADED_COMMIT_KEY, "torch_version", "transformers_version")
PARTIAL_SUFFIX = ".partial"


def stem_paths(stem: str | Path) -> tuple[Path, Path]:
    """Return the array file ``<stem>.npz`` and the sidecar file ``<stem>.json`` of a stem."""
    base = Path(stem)
    return base.with_name(base.name + ".npz"), base.with_name(base.name + ".json")


def write_atomically(target: Path, write: Callable[[BinaryIO], object]) -> None:
    """Write a file through a partial file, so a failed write leaves the old file intact.

    Args:
        target: The file to write.
        write: Writes the content to the open binary handle it is given.

    Raises:
        OSError: If the partial file cannot be written or moved into place; the partial
            file is removed before the error propagates.
    """
    partial = target.with_name(target.name + PARTIAL_SUFFIX)
    try:
        with partial.open("wb") as handle:
            write(handle)
        os.replace(partial, target)
    finally:
        partial.unlink(missing_ok=True)


def write_json(target: Path, content: Mapping[str, Any]) -> None:
    """Write a mapping as indented UTF-8 JSON with a final newline, atomically.

    Args:
        target: The file to write.
        content: A mapping of JSON values.
    """
    text = json.dumps(dict(content), indent=2, ensure_ascii=False) + "\n"
    write_atomically(target, lambda handle: handle.write(text.encode("utf-8")))


def json_ready(value: Any) -> Any:
    """Turn a value into strict JSON values, as ``json.dumps`` needs them.

    Mappings (read-only ``MappingProxyType`` views included) become dicts with string keys,
    tuples, lists and numpy arrays become lists, numpy scalars become the Python number or
    bool they hold, and a float that is not finite (NaN, infinity) becomes ``None``, written
    as ``null``, at every depth; the output is then valid JSON for any reader. Any other value
    is returned as it is.

    Args:
        value: The value, such as the provenance of a fitted correction.

    Returns:
        The value built from dicts, lists, Python scalars and ``None``.
    """
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.generic):
        return json_ready(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def read_sidecar(path: Path) -> dict[str, Any]:
    """Read a JSON sidecar that must hold one object.

    Args:
        path: The sidecar file; it must exist.

    Returns:
        The object of the file.

    Raises:
        ConfigurationError: Naming the file, if it is not UTF-8 text, not valid JSON, or
            holds something other than a JSON object.
    """
    try:
        content = json.loads(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError as error:
        raise ConfigurationError(f"{path} is not UTF-8 text") from error
    except json.JSONDecodeError as error:
        raise ConfigurationError(f"{path} is not valid JSON: {error}") from error
    if not isinstance(content, dict):
        raise ConfigurationError(f"{path} does not hold a JSON object")
    return content


def encoder_identity(sidecar: Mapping[str, Any], owner: str) -> dict[str, Any]:
    """Return the keys of a sidecar that identify the encoder, in their documented order.

    Args:
        sidecar: The description of an embedding set or of reference embeddings.
        owner: What the sidecar belongs to, named in the error message.

    Returns:
        The value of every key of ``IDENTITY_KEYS``.

    Raises:
        ConfigurationError: If the sidecar lacks one of the keys.
    """
    missing = [key for key in IDENTITY_KEYS if key not in sidecar]
    if missing:
        raise ConfigurationError(f"the sidecar of the {owner} lacks {missing}")
    return {key: sidecar[key] for key in IDENTITY_KEYS}


def encoder_versions(sidecar: Mapping[str, Any]) -> dict[str, Any]:
    """Return the version keys a sidecar holds; a set built by hand may hold none.

    Args:
        sidecar: The description of an embedding set or of reference embeddings.

    Returns:
        The keys of ``VERSION_KEYS`` that the sidecar holds, with their values.
    """
    return {key: sidecar[key] for key in VERSION_KEYS if key in sidecar}


def _loaded_commit(description: Mapping[str, Any]) -> Any:
    """Return the loaded commit a description records, or ``None`` for none or ``"unknown"``."""
    commit = description.get(LOADED_COMMIT_KEY)
    return None if commit == UNKNOWN_REVISION else commit


def check_same_encoder(
    first: Mapping[str, Any], second: Mapping[str, Any], first_owner: str, second_owner: str
) -> None:
    """Refuse two sidecars that do not describe the same encoder settings and weights.

    The identity keys are compared, and so is the loaded commit, ``resolved_revision``,
    when both sidecars record one other than ``"unknown"``. A description that records
    none, such as the identity of an encoder specification before its model is loaded, is
    compared on the identity keys alone. The library versions may differ. A model
    repository gets a new commit even when only its documentation changes, so sets
    embedded from its default branch before and after such a commit are refused together;
    re-embed one of them, or embed both with the same pinned revision.

    Args:
        first: The first description.
        second: The second description.
        first_owner: What the first description belongs to, for the message.
        second_owner: What the second description belongs to, for the message.

    Raises:
        ConfigurationError: If a description lacks an identity key; naming every identity
            key whose values differ, with both values; or naming both loaded commits when
            they differ.
    """
    one = encoder_identity(first, first_owner)
    two = encoder_identity(second, second_owner)
    differing = [
        f"{key} ({one[key]!r} and {two[key]!r})" for key in IDENTITY_KEYS if one[key] != two[key]
    ]
    if differing:
        raise ConfigurationError(
            f"the {first_owner} and the {second_owner} were embedded with different encoder "
            f"settings: " + ", ".join(differing)
        )
    commits = (_loaded_commit(first), _loaded_commit(second))
    if None not in commits and commits[0] != commits[1]:
        raise ConfigurationError(
            f"the {first_owner} and the {second_owner} were embedded from different commits "
            f"of {one['hf_id']}: {commits[0]!r} and {commits[1]!r}; a model repository gets a "
            "new commit even when only its documentation changes, so embed both from one "
            "commit, for example with the same pinned revision"
        )


def _is_label(value: object) -> bool:
    """Tell whether a value can stand for an id or a label: a string or an integer."""
    if isinstance(value, (bool, np.bool_)):
        return False
    return isinstance(value, (str, int, np.integer))


def as_unicode_1d(values: Any, name: str) -> np.ndarray:
    """Return ids or labels as a one-dimensional array of strings.

    Strings stay as they are and integers become their decimal strings, whether they come
    in a list, a string or integer array, or an object array. Anything else is refused, so
    that an array saved to an npz file never holds Python objects.

    Args:
        values: The ids or labels.
        name: Name of the field, used in the error message.

    Returns:
        A one-dimensional array of dtype ``<U``; the input itself when it already is one.

    Raises:
        ConfigurationError: If the values are not one-dimensional, or a value is neither a
            string nor an integer (for example ``None``, a float, bytes, or a list).
    """
    array = values if isinstance(values, np.ndarray) else np.asarray(values, dtype=object)
    if array.ndim != 1:
        raise ConfigurationError(f"{name} must be one-dimensional, got shape {array.shape}")
    if array.dtype.kind == "U":
        return array
    if array.size == 0:
        return np.asarray([], dtype=str)
    if array.dtype.kind in "iu":
        return array.astype(str)
    if array.dtype.kind == "O":
        refused = [value for value in array.tolist() if not _is_label(value)]
        if refused:
            raise ConfigurationError(
                f"{name} must hold strings or integers; {len(refused)} values do not, "
                f"for example {refused[0]!r}"
            )
        return np.asarray([str(value) for value in array.tolist()], dtype=str)
    raise ConfigurationError(f"{name} must hold strings or integers, got dtype {array.dtype}")
