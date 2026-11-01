"""Access to data files shipped inside the installed package."""

from __future__ import annotations

import json
from collections.abc import Collection, Mapping
from importlib.resources import files
from pathlib import Path

__all__ = [
    "asset_path",
    "load_toy_cues",
    "toy_corpus_path",
    "toy_cues_path",
    "validate_toy_cues",
]

TOY_CORPUS_ASSET = "assets/examples/toy_corpus.csv"
TOY_CUES_ASSET = "assets/examples/toy_cues.json"


def asset_path(relative: str) -> Path:
    """Resolve a file shipped inside the installed ``copernicus`` package.

    Args:
        relative: Path of the file relative to the package root.

    Returns:
        Absolute path of the existing file inside the package.

    Raises:
        FileNotFoundError: If no such file exists or the path leaves the package;
            the message holds the full requested path.
    """
    package_root = Path(str(files("copernicus"))).resolve()
    resolved = (package_root / relative).resolve()
    if not resolved.is_relative_to(package_root):
        raise FileNotFoundError(f"Package asset outside the package: {package_root / relative}")
    if not resolved.exists():
        raise FileNotFoundError(f"Package asset not found: {resolved}")
    return resolved


def toy_corpus_path() -> Path:
    """Return the path of the hand-written example corpus shipped with the package.

    Returns:
        Path of ``toy_corpus.csv``: 54 short feedback texts by 18 writers with hand-assigned
        gold scores.
    """
    return asset_path(TOY_CORPUS_ASSET)


def toy_cues_path() -> Path:
    """Return the path of the cue-word table of the example rubric.

    Returns:
        Path of ``toy_cues.json``, a JSON object that maps each marker id of the toy
        rubric to the cue words the fake backend looks for.
    """
    return asset_path(TOY_CUES_ASSET)


def validate_toy_cues(data: object, marker_ids: Collection[str]) -> dict[str, list[str]]:
    """Check a cue table against the keys a rubric allows.

    This is the one validator of cue tables; the cue-based fake backend uses it too.

    Args:
        data: The parsed JSON value or a mapping.
        marker_ids: The keys that may appear (marker ids, and marker names where the
            caller accepts them).

    Returns:
        The cue table as a dictionary from key to a list of cue words.

    Raises:
        ValueError: If ``data`` is not an object, a key is not allowed, or a value is
            not a non-empty list of non-empty strings (a bare string is rejected); the
            message names the key.
    """
    if not isinstance(data, Mapping):
        raise ValueError(f"cue table must be a JSON object, got {type(data).__name__}")
    for key, words in data.items():
        if key not in marker_ids:
            raise ValueError(f"cue key {key!r} is not a marker of the rubric")
        if (
            not isinstance(words, (list, tuple))
            or not words
            or not all(isinstance(word, str) and word for word in words)
        ):
            raise ValueError(f"cue words of {key!r} must be a non-empty list of non-empty strings")
    return {key: list(words) for key, words in data.items()}


def load_toy_cues() -> dict[str, list[str]]:
    """Load the cue-word table of the toy rubric.

    Returns:
        The cue words per marker id of the toy rubric.

    Raises:
        ValueError: If the table does not fit the toy rubric.
    """
    # Imported here because the rubric loader itself imports this module.
    from copernicus.rubric.loader import load_rubric, toy_rubric_path

    rubric = load_rubric(toy_rubric_path())
    marker_ids = {m.marker_id for attribute in rubric.attributes for m in attribute.markers}
    data = json.loads(toy_cues_path().read_text(encoding="utf-8"))
    return validate_toy_cues(data, marker_ids)
