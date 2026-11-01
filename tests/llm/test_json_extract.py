"""Tests for the robust JSON extraction."""

from __future__ import annotations

import json

import pytest

from copernicus.errors import BackendError, SettingsError
from copernicus.llm import extract_json

PASSAGES = [
    "None of the figures on page four have axis labels, so the reader cannot tell the units.",
    "It is True that the summary is clear.",
    "False starts are fine.",
    "The list [1, 2,] ends with a comma,}",
    'He said "None", then left.',
    "A path C:\\True\\None stays.",
]


@pytest.mark.parametrize(
    ("text", "strategy"),
    [
        ('{"a": 1}', "direct"),
        ('```json\n{"a": 1}\n```', "fence"),
        ('Sure: {"a": 1} done', "balanced"),
        ('{"q": "he said \\\'hi\\\' and "x" more"}', "string_escapes"),
        ("{'a': True,}", "common_fixes"),
    ],
)
def test_extract_json_strategies(text: str, strategy: str) -> None:
    parsed, used = extract_json(text)
    assert isinstance(parsed, dict)
    assert used == strategy


def test_extract_json_nothing_raises_with_excerpt() -> None:
    with pytest.raises(BackendError) as error:
        extract_json("no json here")
    assert error.value.last_output == "no json here"
    assert error.value.attempts == 0


def test_extract_json_excerpt_is_truncated() -> None:
    with pytest.raises(BackendError) as error:
        extract_json("x" * 900)
    assert error.value.last_output == "x" * 500


def test_extract_json_empty_text_raises() -> None:
    with pytest.raises(BackendError):
        extract_json("   ")


def test_extract_json_balanced_brackets_list() -> None:
    parsed, used = extract_json("Result: [1, 2, 3] end")
    assert parsed == [1, 2, 3]
    assert used == "balanced"


def test_extract_json_repairs_inside_surrounding_text() -> None:
    parsed, used = extract_json("Here: {'a': True,} ok")
    assert parsed == {"a": True}
    assert used == "common_fixes"


def test_extract_json_string_escapes_keeps_content() -> None:
    parsed, _ = extract_json('{"q": "he said \\\'hi\\\' and "x" more"}')
    assert parsed == {"q": "he said 'hi' and \"x\" more"}


def test_common_fixes_leave_the_passages_of_an_extraction_unchanged() -> None:
    quotes = ", ".join(json.dumps(passage) for passage in PASSAGES[:2])
    raw = '{"Concrete examples": {"evaluable": True, "quotes": [' + quotes + "]},}"
    parsed, strategy = extract_json(raw)
    assert strategy == "common_fixes"
    assert parsed == {"Concrete examples": {"evaluable": True, "quotes": PASSAGES[:2]}}


@pytest.mark.parametrize("passage", PASSAGES)
def test_literals_and_trailing_commas_inside_strings_survive(passage: str) -> None:
    raw = '{"quotes": [' + json.dumps(passage) + ',], "ok": True, "gap": None, "no": False,}'
    parsed, strategy = extract_json(raw)
    assert strategy == "common_fixes"
    assert parsed == {"quotes": [passage], "ok": True, "gap": None, "no": False}


def test_single_quoted_output_is_fixed_outside_its_strings_only() -> None:
    parsed, strategy = extract_json("{'flag': True, 'quote': 'None of it, True,]',}")
    assert strategy == "common_fixes"
    assert parsed == {"flag": True, "quote": "None of it, True,]"}


def test_string_escape_repair_and_common_fixes_together_keep_the_passage() -> None:
    raw = '{"q": "he said "None" twice, True,]", "ok": True, "gap": None,}'
    parsed, strategy = extract_json(raw)
    assert strategy == "string_escapes"
    assert parsed == {"q": 'he said "None" twice, True,]', "ok": True, "gap": None}


PREAMBLE_ARRAY = 'Here are the texts:\n[{"text": "a"}, {"text": "b"}]'


def test_a_bare_array_behind_prose_is_read_as_the_array_when_one_is_expected() -> None:
    parsed, strategy = extract_json(PREAMBLE_ARRAY, expect="array")
    assert parsed == [{"text": "a"}, {"text": "b"}]
    assert strategy == "balanced"


def test_the_default_and_the_object_expectation_keep_the_first_object_first() -> None:
    assert extract_json(PREAMBLE_ARRAY) == ({"text": "a"}, "balanced")
    assert extract_json(PREAMBLE_ARRAY, expect="object") == ({"text": "a"}, "balanced")


def test_an_array_expectation_still_reads_objects_fences_and_direct_arrays() -> None:
    assert extract_json('Sure: {"a": 1} done', expect="array") == ({"a": 1}, "balanced")
    assert extract_json("```json\n[1, 2]\n```", expect="array") == ([1, 2], "fence")
    assert extract_json("[1, 2]", expect="array") == ([1, 2], "direct")
    assert extract_json("Items: [1, 2,] ok", expect="array") == ([1, 2], "common_fixes")


TRUNCATED_ARRAY = '[{"text": "a"}, {"text": "b, with [brackets] and {braces}"}, {"text": "c'
SALVAGED = [{"text": "a"}, {"text": "b, with [brackets] and {braces}"}]


def test_a_bracketed_word_before_the_array_does_not_hide_it() -> None:
    text = 'Here are the texts [as requested]:\n[{"text": "a"}, {"text": "b"}]'
    assert extract_json(text, expect="array") == ([{"text": "a"}, {"text": "b"}], "balanced")


def test_bracket_spans_are_tried_in_order_until_one_parses_as_a_list() -> None:
    text = 'First [see notes], then {"note": 1}, then [{"text": "a"}] and [2]'
    assert extract_json(text, expect="array") == ([{"text": "a"}], "balanced")


def test_the_complete_objects_of_a_truncated_array_are_salvaged() -> None:
    assert extract_json(TRUNCATED_ARRAY, expect="array") == (SALVAGED, "truncated_array")


def test_a_truncated_array_behind_prose_and_an_unclosed_fence_is_salvaged() -> None:
    text = "Sure, here you go:\n```json\n" + TRUNCATED_ARRAY
    assert extract_json(text, expect="array") == (SALVAGED, "truncated_array")


def test_the_salvage_keeps_the_complete_top_level_objects_only() -> None:
    text = '[{"text": "a", "tags": [{"x": 1}, {"y": 2}]}, 5, "s", {"text": "b"}, {"text": "c", "t'
    parsed, strategy = extract_json(text, expect="array")
    assert strategy == "truncated_array"
    assert parsed == [{"text": "a", "tags": [{"x": 1}, {"y": 2}]}, {"text": "b"}]


def test_an_unclosed_bracket_in_the_preamble_does_not_hide_the_truncated_array() -> None:
    text = 'Notes [draft\n[{"text": "a"}, {"text": "b'
    assert extract_json(text, expect="array") == ([{"text": "a"}], "truncated_array")


def test_an_open_array_without_a_complete_object_yields_nothing() -> None:
    with pytest.raises(BackendError):
        extract_json('[1, 2, {"text": "cut', expect="array")


def test_without_an_array_expectation_a_truncated_array_is_read_as_its_first_object() -> None:
    assert extract_json(TRUNCATED_ARRAY) == ({"text": "a"}, "balanced")
    assert extract_json(TRUNCATED_ARRAY, expect="object") == ({"text": "a"}, "balanced")


def test_an_unknown_expectation_is_a_settings_error() -> None:
    with pytest.raises(SettingsError, match="expect"):
        extract_json("[1]", expect="list")  # type: ignore[arg-type]


UNMATCHED_PREAMBLE = 'Here you go [10 texts:\n[{"text": "a"}, {"text": "b [x] {y}"}]\nEnjoy.'


def test_a_complete_array_behind_an_unmatched_bracket_in_the_preamble_is_found() -> None:
    parsed, strategy = extract_json(UNMATCHED_PREAMBLE, expect="array")
    assert parsed == [{"text": "a"}, {"text": "b [x] {y}"}]
    assert strategy == "array_after_open_bracket"


def test_an_array_that_needs_the_common_fixes_is_found_behind_an_unmatched_bracket() -> None:
    text = 'Sure [as asked:\n[{"text": "a"}, {"text": "b"},]'
    assert extract_json(text, expect="array") == (
        [{"text": "a"}, {"text": "b"}],
        "array_after_open_bracket",
    )


def test_an_unmatched_bracket_followed_by_prose_only_is_still_refused() -> None:
    with pytest.raises(BackendError):
        extract_json("Here you go [10 texts: I could not write them.", expect="array")


def test_a_list_inside_a_cut_off_object_is_not_read_as_the_array() -> None:
    text = '[{"text": "a", "sub": [{"text": "inner"}]'
    with pytest.raises(BackendError):
        extract_json(text, expect="array")


def test_the_unmatched_bracket_strategy_comes_after_the_truncated_salvage() -> None:
    text = 'Notes [draft\n[{"text": "a"}, {"text": "b"}, {"text": "c'
    assert extract_json(text, expect="array") == (
        [{"text": "a"}, {"text": "b"}],
        "truncated_array",
    )


def test_a_single_key_object_that_wraps_the_array_is_unwrapped() -> None:
    raw = '{"texts": [{"text": "a"}, {"text": "b"}]}'
    assert extract_json(raw, expect="array") == ([{"text": "a"}, {"text": "b"}], "wrapped_array")


def test_an_object_with_one_list_of_objects_among_other_values_is_unwrapped() -> None:
    raw = '```json\n{"count": 2, "tags": [1, 2], "items": [{"text": "a"}]}\n```'
    assert extract_json(raw, expect="array") == ([{"text": "a"}], "wrapped_array")


@pytest.mark.parametrize(
    "raw",
    [
        '{"a": [{"text": "x"}], "b": [{"text": "y"}]}',
        '{"text": "alone"}',
        '{"texts": []}',
        '{"texts": [1, 2]}',
        '{"texts": [{"text": "x"}, 3]}',
    ],
)
def test_an_object_without_exactly_one_list_of_objects_stays_an_object(raw: str) -> None:
    value, strategy = extract_json(raw, expect="array")
    assert isinstance(value, dict)
    assert strategy == "direct"


def test_unwrapping_applies_to_the_array_expectation_only() -> None:
    raw = '{"texts": [{"text": "a"}]}'
    assert extract_json(raw) == ({"texts": [{"text": "a"}]}, "direct")
    assert extract_json(raw, expect="object") == ({"texts": [{"text": "a"}]}, "direct")


def test_an_object_with_a_string_value_of_its_own_is_not_unwrapped() -> None:
    raw = '{"text": "outer", "items": [{"text": "inner"}]}'
    value, strategy = extract_json(raw, expect="array")
    assert value == {"text": "outer", "items": [{"text": "inner"}]}
    assert strategy == "direct"


def test_a_scalar_value_that_is_not_a_string_does_not_block_unwrapping() -> None:
    raw = '{"count": 1, "ok": true, "note": null, "items": [{"text": "a"}]}'
    assert extract_json(raw, expect="array") == ([{"text": "a"}], "wrapped_array")


def test_a_wrapping_object_in_a_truncated_array_is_unwrapped() -> None:
    raw = 'Here [x:\n{"texts":[{"text":"a"}]}'
    assert extract_json(raw, expect="array") == ([{"text": "a"}], "wrapped_array")


def test_a_truncated_array_of_one_record_with_a_string_value_stays_a_list() -> None:
    raw = 'Here [x:\n{"text": "outer", "items": [{"text": "inner"}]}'
    assert extract_json(raw, expect="array") == (
        [{"text": "outer", "items": [{"text": "inner"}]}],
        "truncated_array",
    )


def test_a_truncated_array_of_several_objects_is_not_unwrapped() -> None:
    raw = '[{"texts": [{"text": "a"}]}, {"texts": [{"text": "b"}]}, {"te'
    assert extract_json(raw, expect="array") == (
        [{"texts": [{"text": "a"}]}, {"texts": [{"text": "b"}]}],
        "truncated_array",
    )


def test_a_salvaged_record_followed_by_a_cut_off_object_is_not_unwrapped() -> None:
    raw = '[{"id":1,"spans":[{"s":"a"}]}, {"id":2,"spans":[{"s'
    assert extract_json(raw, expect="array") == (
        [{"id": 1, "spans": [{"s": "a"}]}],
        "truncated_array",
    )


def test_a_brace_inside_a_string_after_the_salvaged_wrapper_does_not_block_unwrapping() -> None:
    raw = 'Here [x:\n{"texts":[{"text":"a"}]}, "note { open'
    assert extract_json(raw, expect="array") == ([{"text": "a"}], "wrapped_array")
