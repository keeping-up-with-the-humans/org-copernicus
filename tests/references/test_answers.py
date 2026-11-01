"""Tests for reading the texts out of a writer's answer."""

from __future__ import annotations

from copernicus.references.answers import REASON_NOT_AN_ARRAY, read_answer

TEXTS = [{"text": "alpha"}, {"text": "beta"}]


def test_an_array_behind_an_unmatched_bracket_gives_its_texts() -> None:
    raw = 'Here you go [2 texts:\n[{"text": "alpha"}, {"text": "beta"}]'
    assert read_answer(raw, 2, "call") == (["alpha", "beta"], None)


def test_an_object_that_wraps_the_array_gives_its_texts() -> None:
    raw = '{"texts": [{"text": "alpha"}, {"text": "beta"}]}'
    assert read_answer(raw, 2, "call") == (["alpha", "beta"], None)


def test_an_object_without_a_list_of_objects_is_not_an_array() -> None:
    assert read_answer('{"text": "alpha"}', 1, "call") == ([], REASON_NOT_AN_ARRAY)
