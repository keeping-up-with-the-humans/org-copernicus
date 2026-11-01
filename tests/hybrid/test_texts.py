"""Tests of the evidence texts: which verified passages are joined, and when raw text stands in."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest
from hybrid_support import evidence_frame

import copernicus.hybrid.texts as texts_module
from copernicus.assets import load_toy_cues
from copernicus.coders import CodingSettings, code_texts, export_evidence, export_scores, normalize
from copernicus.data import TextTable
from copernicus.errors import ConfigurationError, EvidenceContractError, SettingsError
from copernicus.hybrid import HybridTexts, evidence_texts
from copernicus.llm import FakeBackend, cue_responder
from copernicus.rubric.model import Rubric

FALLBACKS = {"", "short", "no_evidence"}
SUMMARY_KEYS = {
    "n_texts",
    "n_short",
    "n_no_evidence",
    "n_evidence",
    "mean_words_evidence",
    "n_passages_embedded",
}
LOW_THRESHOLD = 10
PASSAGE_ORDER = ["char_start", "char_end", "attribute_id", "marker_id", "quote_index"]


def expected_lines(evidence: pd.DataFrame, text_id: str, markers: set[str] | None) -> list[str]:
    """Recompute the passages of one text from the raw evidence table, in the paper's order.

    Passages are ordered by start, end, attribute id, marker id, and passage index.
    """
    rows = evidence[
        (evidence.text_id == text_id) & (evidence.run_number == 1) & (evidence.verified > 0)
    ]
    rows = rows[rows.duplicate_of.isna()]
    if markers is not None:
        rows = rows[rows.marker_id.isin(markers)]
    rows = rows.sort_values(PASSAGE_ORDER, kind="stable")
    seen: set[str] = set()
    lines = []
    for quote in rows.quote_norm:
        if normalize(quote) not in seen:
            seen.add(normalize(quote))
            lines.append(quote)
    return lines


def test_construct_texts_join_the_verified_passages_in_document_order(
    coded_store: Path, rubric: Rubric, table: TextTable
) -> None:
    result = evidence_texts(coded_store, rubric, table, granularity="construct", min_words=1)
    assert isinstance(result, HybridTexts) and result.attribute_id is None
    assert result.ids == [str(text_id) for text_id in table.ids()]
    assert len(result.texts) == len(result.fallback) == len(table)
    assert set(result.fallback) <= FALLBACKS and "" in result.fallback
    evidence = export_evidence(coded_store)
    for text_id, text, fallback in zip(result.ids, result.texts, result.fallback, strict=True):
        source = table.text_of(text_id)
        if fallback == "":
            lines = text.split("\n")
            assert lines == expected_lines(evidence, text_id, None)
            cursor = 0
            for line in lines:
                position = source.find(line, cursor)
                assert position >= 0
                cursor = position
        else:
            assert text == source and fallback == "no_evidence"
            assert expected_lines(evidence, text_id, None) == []


def test_construct_summary_counts_are_consistent(
    coded_store: Path, rubric: Rubric, table: TextTable
) -> None:
    result = evidence_texts(coded_store, rubric, table, granularity="construct", min_words=1)
    summary = result.summary
    assert set(summary) == SUMMARY_KEYS and summary["n_texts"] == len(table)
    assert summary["n_short"] == 0 and summary["n_evidence"] > 0
    assert summary["n_short"] + summary["n_no_evidence"] + summary["n_evidence"] == len(table)
    assert summary["n_no_evidence"] == result.fallback.count("no_evidence")
    lines = [t.split("\n") for t, f in zip(result.texts, result.fallback, strict=True) if f == ""]
    assert summary["n_passages_embedded"] == sum(map(len, lines))
    words = [len(" ".join(each).split()) for each in lines]
    assert summary["mean_words_evidence"] == pytest.approx(sum(words) / len(words))


def test_texts_below_min_words_stay_raw_and_are_marked_short(
    coded_store: Path, rubric: Rubric, table: TextTable
) -> None:
    everything = evidence_texts(coded_store, rubric, table, granularity="construct", min_words=1000)
    assert everything.fallback == ["short"] * len(table)
    assert everything.texts == [table.text_of(text_id) for text_id in table.ids()]
    assert everything.texts == [" ".join(text.split()) for text in everything.texts]
    assert everything.summary["n_short"] == len(table) and everything.summary["n_evidence"] == 0
    assert everything.summary["mean_words_evidence"] == 0.0
    default = evidence_texts(coded_store, rubric, table, granularity="construct")
    short = [int(n) < 40 for n in table.frame["n_words"]]
    assert [f == "short" for f in default.fallback] == short and any(short) and not all(short)


def test_a_ten_word_text_is_short_and_keeps_its_raw_text(
    rubric: Rubric, table: TextTable, tmp_path: Path
) -> None:
    frame = table.subset(table.ids()[:3]).frame.copy()
    ten_words = "one two three four five six seven eight nine ten"
    frame.loc[0, "text"] = ten_words
    short = TextTable.from_frame(frame)
    store = tmp_path / "ten_words.sqlite"
    backend = FakeBackend(cue_responder(rubric, load_toy_cues()))
    code_texts(short, rubric, backend, store, CodingSettings(runs=1))
    result = evidence_texts(store, rubric, short, granularity="construct", min_words=11)
    assert result.fallback[0] == "short" and result.texts[0] == ten_words
    longer = evidence_texts(store, rubric, short, granularity="construct", min_words=10)
    assert longer.fallback[0] == "no_evidence" and longer.texts[0] == ten_words


def test_a_text_that_changed_since_the_coding_is_refused(
    coded_store: Path, rubric: Rubric, table: TextTable
) -> None:
    frame = table.frame.copy()
    for row in range(12):
        frame.loc[row, "text"] = f"A replaced text, number {row}, that the store never saw."
    changed = TextTable.from_frame(frame)
    with pytest.raises(ConfigurationError) as construct:
        evidence_texts(coded_store, rubric, changed, granularity="construct")
    with pytest.raises(ConfigurationError) as attribute:
        evidence_texts(coded_store, rubric, changed, granularity="attribute")
    ids = [str(text_id) for text_id in table.ids()]
    for refused in (construct, attribute):
        message = str(refused.value)
        assert f"12 of {len(table)} texts differ" in message and "and 2 more" in message
        assert all(repr(text_id) in message for text_id in ids[:10])
        assert not any(repr(text_id) in message for text_id in ids[10:])
        assert "a store binds every text id to the text it was coded on" in message


def test_attribute_texts_use_only_the_markers_of_the_attribute(
    coded_store: Path, rubric: Rubric, table: TextTable
) -> None:
    result = evidence_texts(coded_store, rubric, table, granularity="attribute", min_words=1)
    assert isinstance(result, dict)
    assert list(result) == [attribute.attribute_id for attribute in rubric.attributes]
    evidence = export_evidence(coded_store)
    for attribute in rubric.attributes:
        mine = result[attribute.attribute_id]
        assert (
            mine.attribute_id == attribute.attribute_id
            and mine.ids == result[rubric.attributes[0].attribute_id].ids
        )
        markers = {marker.marker_id for marker in attribute.markers}
        for text_id, text, fallback in zip(mine.ids, mine.texts, mine.fallback, strict=True):
            lines = expected_lines(evidence, text_id, markers)
            assert (text.split("\n"), fallback) == (
                (lines, "") if lines else ([table.text_of(text_id)], "no_evidence")
            )
        counts = mine.summary
        assert counts["n_short"] + counts["n_no_evidence"] + counts["n_evidence"] == len(table)
    texts_per_attribute = [each.summary["n_evidence"] for each in result.values()]
    construct = evidence_texts(coded_store, rubric, table, granularity="construct", min_words=1)
    assert all(count <= construct.summary["n_evidence"] for count in texts_per_attribute)
    assert max(texts_per_attribute) < construct.summary["n_evidence"]


def test_a_text_without_evidence_for_one_attribute_falls_back_there_only(
    coded_store: Path, rubric: Rubric, table: TextTable
) -> None:
    result = evidence_texts(coded_store, rubric, table, granularity="attribute", min_words=1)
    assert isinstance(result, dict)
    patterns = {tuple(each.fallback[row] for each in result.values()) for row in range(len(table))}
    assert any("" in pattern and "no_evidence" in pattern for pattern in patterns)


def hand_made_evidence(table: TextTable, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Replace the exported evidence by a hand-made table for the first three texts."""
    first, second, third = (str(text_id) for text_id in table.ids()[:3])
    rows = [
        {"text_id": first, "attribute_id": "tone", "marker_id": "tone_balance"}
        | {"quote_norm": "Second sentence here.", "char_start": 30},
        {"text_id": first, "attribute_id": "tone", "marker_id": "tone_respect"}
        | {"quote_norm": "First sentence here.", "char_start": 0},
        {"text_id": first, "attribute_id": "tone", "marker_id": "tone_blame"}
        | {"quote_norm": "first  sentence HERE.", "char_start": 0},
        {"text_id": first, "attribute_id": "reasoning", "marker_id": "reas_why"}
        | {"quote_norm": "Unverified.", "verified": 0, "char_start": 10},
        {"text_id": first, "attribute_id": "reasoning", "marker_id": "reas_why"}
        | {"quote_norm": "Repeated.", "char_start": 12, "duplicate_of": 0, "quote_index": 1},
        {"text_id": second, "attribute_id": "reasoning", "marker_id": "reas_why"}
        | {"quote_norm": "Because of this.", "char_start": 5},
    ]
    monkeypatch.setattr(texts_module, "export_evidence", lambda path: evidence_frame(rows))
    return [first, second, third]


def test_passages_are_ordered_filtered_and_deduplicated_across_markers(
    coded_store: Path, rubric: Rubric, table: TextTable, monkeypatch: pytest.MonkeyPatch
) -> None:
    first, second, third = hand_made_evidence(table, monkeypatch)
    sub = table.subset([first, second, third])
    result = evidence_texts(coded_store, rubric, sub, granularity="construct", min_words=1)
    # Two passages share start and end; marker tone_blame sorts before tone_respect, so its
    # spelling survives the deduplication.
    assert result.texts[0] == "first  sentence HERE.\nSecond sentence here."
    assert result.texts[1] == "Because of this."
    assert result.texts[2] == sub.text_of(third) and result.fallback == ["", "", "no_evidence"]
    assert result.summary["n_passages_embedded"] == 3
    by_attribute = evidence_texts(coded_store, rubric, sub, granularity="attribute", min_words=1)
    assert isinstance(by_attribute, dict)
    assert by_attribute["tone"].texts[0] == "first  sentence HERE.\nSecond sentence here."
    assert by_attribute["tone"].fallback == ["", "no_evidence", "no_evidence"]
    assert by_attribute["reasoning"].fallback == ["no_evidence", "", "no_evidence"]
    assert by_attribute["reasoning"].texts[1] == "Because of this."
    assert by_attribute["specificity"].fallback == ["no_evidence"] * 3


def test_passages_with_the_same_start_are_ordered_by_their_end_then_by_their_ids(
    coded_store: Path, rubric: Rubric, table: TextTable, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = str(table.ids()[0])
    base = {"text_id": first, "attribute_id": "specificity"}
    rows = [
        base | {"marker_id": "spec_examples", "char_start": 10, "char_end": 56}
        | {"quote_norm": "I would move the summary up and cut the intro."},
        base | {"marker_id": "spec_location", "char_start": 10, "char_end": 37}
        | {"quote_norm": "I would move the summary up."},
        base | {"marker_id": "spec_actionable", "char_start": 10, "char_end": 56}
        | {"quote_norm": "I would move the summary up and cut the intro!"},
    ]  # fmt: skip
    monkeypatch.setattr(texts_module, "export_evidence", lambda path: evidence_frame(rows))
    result = evidence_texts(
        coded_store, rubric, table.subset([first]), granularity="construct", min_words=1
    )
    assert result.texts == [
        "I would move the summary up.\n"
        "I would move the summary up and cut the intro!\n"
        "I would move the summary up and cut the intro."
    ]


def test_a_verified_passage_without_offsets_or_words_is_refused(
    coded_store: Path, rubric: Rubric, table: TextTable, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = str(table.ids()[0])
    base = {"text_id": first, "attribute_id": "tone", "marker_id": "tone_respect"}
    bad_rows = [
        [base | {"quote_norm": "No offsets.", "char_start": pd.NA}],
        [base | {"quote_norm": "   "}],
    ]
    for rows in bad_rows:
        monkeypatch.setattr(
            texts_module, "export_evidence", lambda path, rows=rows: evidence_frame(rows)
        )
        with pytest.raises(EvidenceContractError, match=first):
            evidence_texts(coded_store, rubric, table, granularity="construct", min_words=1)


@pytest.mark.parametrize("offset", ["char_start", "char_end"])
def test_a_verified_passage_without_its_start_or_end_offset_is_refused(
    coded_store: Path,
    rubric: Rubric,
    table: TextTable,
    monkeypatch: pytest.MonkeyPatch,
    offset: str,
) -> None:
    first = str(table.ids()[0])
    rows = [
        {"text_id": first, "attribute_id": "tone", "marker_id": "tone_respect"}
        | {"quote_norm": "Placed at one end only.", offset: pd.NA}
    ]
    monkeypatch.setattr(texts_module, "export_evidence", lambda path: evidence_frame(rows))
    with pytest.raises(EvidenceContractError, match=f"text {first} has no match offsets"):
        evidence_texts(coded_store, rubric, table, granularity="construct", min_words=1)


def test_a_rubric_that_differs_from_the_stored_one_is_refused(
    coded_store: Path, rubric: Rubric, table: TextTable
) -> None:
    other = replace(rubric, concept_definition=rubric.concept_definition + " Changed.")
    with pytest.raises(ConfigurationError) as caught:
        evidence_texts(coded_store, other, table, granularity="construct")
    assert rubric.content_hash() in str(caught.value) and other.content_hash() in str(caught.value)


def test_a_one_stage_store_has_no_evidence_to_use(
    one_stage_store: Path, rubric: Rubric, table: TextTable
) -> None:
    small = table.subset(table.ids()[:3])
    with pytest.raises(ConfigurationError, match="one_stage"):
        evidence_texts(one_stage_store, rubric, small, granularity="construct")


def test_texts_absent_from_the_run_are_listed(
    coded_store: Path, rubric: Rubric, table: TextTable
) -> None:
    frame = table.frame.drop(columns=["n_chars", "n_words"]).copy()
    frame.loc[len(frame)] = {**frame.iloc[0].to_dict(), "text_id": "not_in_the_store"}
    with pytest.raises(ConfigurationError, match="not_in_the_store"):
        evidence_texts(coded_store, rubric, TextTable.from_frame(frame), granularity="construct")
    with pytest.raises(ConfigurationError) as caught:
        evidence_texts(coded_store, rubric, table, granularity="construct", run_number=2)
    message = str(caught.value)
    ids = [str(text_id) for text_id in table.ids()]
    assert all(text_id in message for text_id in ids[:10])
    assert ids[10] not in message and str(len(ids)) in message


def test_two_labels_for_one_text_and_run_are_refused(
    coded_store: Path, rubric: Rubric, table: TextTable, monkeypatch: pytest.MonkeyPatch
) -> None:
    scores = export_scores(coded_store)
    other = scores.assign(model_name="another-model")
    monkeypatch.setattr(
        texts_module, "export_scores", lambda path: pd.concat([scores, other], ignore_index=True)
    )
    with pytest.raises(ConfigurationError, match="another-model"):
        evidence_texts(coded_store, rubric, table, granularity="construct")


def test_settings_outside_their_range_are_refused(
    coded_store: Path, rubric: Rubric, table: TextTable
) -> None:
    with pytest.raises(SettingsError, match="granularity"):
        evidence_texts(coded_store, rubric, table, granularity="marker")  # type: ignore[call-overload]
    with pytest.raises(SettingsError, match="min_words"):
        evidence_texts(coded_store, rubric, table, granularity="construct", min_words=-1)
    with pytest.raises(SettingsError, match="run_number"):
        evidence_texts(coded_store, rubric, table, granularity="construct", run_number=0)
