"""Tests for the example assets that ship with the package."""

from __future__ import annotations

import pytest

from copernicus.assets import (
    load_toy_cues,
    toy_corpus_path,
    toy_cues_path,
    validate_toy_cues,
)
from copernicus.rubric.model import Rubric

MARKER_IDS = ["a", "b"]


def test_asset_paths_exist() -> None:
    assert toy_corpus_path().name == "toy_corpus.csv" and toy_corpus_path().exists()
    assert toy_cues_path().name == "toy_cues.json" and toy_cues_path().exists()


def test_toy_cues_cover_every_marker_of_the_toy_rubric(toy_rubric: Rubric) -> None:
    cues = load_toy_cues()
    marker_ids = {m.marker_id for a in toy_rubric.attributes for m in a.markers}
    assert set(cues) == marker_ids
    assert cues["spec_examples"] == ["for example", "such as", "paragraph"]


def test_valid_cues_pass() -> None:
    assert validate_toy_cues({"a": ["x"]}, MARKER_IDS) == {"a": ["x"]}


@pytest.mark.parametrize(
    "data, fragment",
    [
        (["a"], "JSON object"),
        ({"zzz": ["x"]}, "'zzz'"),
        ({"a": []}, "'a'"),
        ({"a": "x"}, "'a'"),
        ({"a": ["x", ""]}, "'a'"),
        ({"a": ["x", 3]}, "'a'"),
    ],
)
def test_invalid_cues_are_rejected(data: object, fragment: str) -> None:
    with pytest.raises(ValueError, match=fragment):
        validate_toy_cues(data, MARKER_IDS)
