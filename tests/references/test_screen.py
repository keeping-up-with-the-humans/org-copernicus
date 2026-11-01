"""Tests for the leakage screen of reference texts."""

from __future__ import annotations

import pandas as pd

from copernicus.references import leakage_terms, screen_report, screen_texts
from copernicus.references.reference_set import FRAME_COLUMNS
from copernicus.rubric.model import Attribute, Marker, Rubric


def frame_of(texts: list[str]) -> pd.DataFrame:
    """Return a minimal frame holding the given texts."""
    return pd.DataFrame({"text": texts, "target": "construct"})


def test_the_terms_are_the_concept_attributes_and_markers_casefolded(toy_rubric: Rubric) -> None:
    terms = leakage_terms(toy_rubric)
    assert toy_rubric.concept.casefold() in terms
    for attribute in toy_rubric.attributes:
        assert attribute.name.casefold() in terms
        for marker in attribute.markers:
            assert marker.name.casefold() in terms
    assert all(term == term.casefold() for term in terms)
    assert len(terms) == len(set(terms)) == 1 + 3 + 9


def test_the_terms_are_longest_first_deduplicated_and_at_least_four_characters(
    toy_rubric: Rubric,
) -> None:
    terms = leakage_terms(toy_rubric)
    assert [len(term) for term in terms] == sorted((len(term) for term in terms), reverse=True)
    assert min(len(term) for term in terms) >= 4


def test_short_and_repeated_terms_are_dropped() -> None:
    marker = Marker("m1", "Tone", "d", ())
    short = Marker("m2", "Fit", "d", ())
    repeat = Marker("m3", "TONE", "d", ())
    attribute = Attribute("a", "Tone", "d", (), (marker, short, repeat))
    rubric = Rubric("r", "Warmth", "defn", (), (attribute,), {})
    assert leakage_terms(rubric) == ["warmth", "tone"]


def test_a_text_naming_the_construct_is_flagged_and_a_clean_text_is_not(
    toy_rubric: Rubric,
) -> None:
    terms = leakage_terms(toy_rubric)
    frame = frame_of(
        [
            "This is truly CONSTRUCTIVENESS of written feedback at work.",
            "I went for a long walk and bought bread.",
        ]
    )
    screened = screen_texts(frame, terms)
    assert screened["screened_out"].tolist() == [True, False]
    assert screened["screen_reason"].tolist() == [
        "names construct term 'constructiveness of written feedback'",
        "",
    ]
    assert screened["screened_out"].dtype == bool
    assert "screened_out" not in frame.columns


def test_matching_respects_word_boundaries_and_casefolds() -> None:
    frame = frame_of(["The Tone was fine.", "A stone wall.", "Tones differ.", "my tone-deaf aunt"])
    screened = screen_texts(frame, ["tone"])
    assert screened["screened_out"].tolist() == [True, False, False, True]


def test_the_first_term_in_the_list_gives_the_reason() -> None:
    screened = screen_texts(
        frame_of(["respectful wording and tone"]), ["respectful wording", "tone"]
    )
    assert screened["screen_reason"].tolist() == ["names construct term 'respectful wording'"]


def test_a_multiword_term_matches_across_whitespace_and_regex_characters_are_literal() -> None:
    frame = frame_of(["respectful\n  wording here", "plan (b) is fine", "planxbx"])
    screened = screen_texts(frame, ["respectful wording", "plan (b)"])
    assert screened["screened_out"].tolist() == [True, True, False]


def test_an_empty_frame_and_no_terms_are_handled() -> None:
    empty = screen_texts(frame_of([]), ["tone"])
    assert len(empty) == 0
    assert list(empty.columns[-2:]) == ["screened_out", "screen_reason"]
    unscreened = screen_texts(frame_of(["tone"]), [])
    assert unscreened["screened_out"].tolist() == [False]


def test_the_report_counts_hits_per_term_and_the_share_screened_out() -> None:
    frame = frame_of(["tone one", "tone two", "reasoning here", "nothing", "plain"])
    report = screen_report(screen_texts(frame, ["reasoning", "tone"]))
    assert report == {
        "n_texts": 5,
        "n_screened_out": 3,
        "share_screened_out": 0.6,
        "by_term": {"tone": 2, "reasoning": 1},
    }


def test_the_report_of_an_empty_frame_has_a_zero_share() -> None:
    report = screen_report(screen_texts(frame_of([]), ["tone"]))
    assert report["share_screened_out"] == 0.0
    assert report["by_term"] == {}


def test_the_frame_columns_end_with_the_screen_columns() -> None:
    assert FRAME_COLUMNS == (
        "rubric_id",
        "target",
        "polarity",
        "level",
        "style",
        "writer_model",
        "text",
        "call_index",
        "attempt",
        "seed",
        "prompt_tokens",
        "completion_tokens",
        "temperature_applied",
        "top_p_applied",
        "prompt_hash",
        "prompt_version",
        "screened_out",
        "screen_reason",
    )
