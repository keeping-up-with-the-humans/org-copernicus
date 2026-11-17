"""A text table keeps each text as given, and the coders send it with its line breaks.

An earlier version of the library collapsed every run of whitespace in a text to one space
before it sent the text and hashed it. A store of that version therefore holds, under the id of
a text with a line break, a tab, a run of spaces or other whitespace between its words, the hash
of the collapsed text. Such a store keeps working for every other text, and refuses these texts
on resume, on merge and in the hybrid coder, because their prompt has changed.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest
from hybrid_support import STUB_DIMENSION, StubEncoder, unit_references

from copernicus.coders import (
    CodingSettings,
    code_texts,
    export_evidence,
    export_scores,
    merge_stores,
)
from copernicus.data import TextTable
from copernicus.errors import ConfigurationError, StoreError
from copernicus.hybrid import evidence_texts, score_hybrid
from copernicus.llm import FakeBackend
from copernicus.llm.base import GenerationRequest
from copernicus.llm.fake import STAGE_ONE_MARKER, TEMPLATE_MARKER_PATTERN
from copernicus.rubric.model import Rubric

Factory = Callable[[], FakeBackend]
SETTINGS = CodingSettings(runs=1, temperature=0.3)
METHODS = ("two_stage", "one_stage")
TEXT_PREFIX = "Text to analyze:\n"
# Texts that the earlier version sent with their whitespace collapsed, in text-id order.
SPECIAL = {
    "break": "First line.\nSecond line of the answer.",
    "nbsp": "A no-break space sits in this answer.",
    "spaces": "Two  spaces sit between these words.",
    "tab": "A tab\tsits between these words.",
}
# Texts that the earlier version sent as this version does.
PLAIN = {
    "ends": "  Whitespace sits at the ends of this answer only.  \n",
    "plain": "A plain answer with single spaces only.",
}
COLLAPSED_PHRASES = (
    "every run of whitespace collapsed to one space",
    "an earlier version of the library",
    "the prompt of such a text has changed",
    "code these texts into a new store",
    "(copernicus.data.table.collapse_whitespace) to continue the old coding",
)
CHANGED_TEXTS = "the texts changed since the store was written"
TEXTS_DIFFER = "the stores hold different texts under the same text ids"
EVIDENCE_COLUMNS = [
    "attribute_id",
    "marker_id",
    "quote_raw",
    "quote_norm",
    "verified",
    "match_score",
    "char_start",
    "char_end",
    "duplicate_of",
    "used_in_stage2",
]


def table_of(texts: dict[str, str]) -> TextTable:
    """Return a table of the texts, all in one group."""
    frame = pd.DataFrame({"text_id": list(texts), "text": list(texts.values()), "group_id": "g"})
    return TextTable.from_frame(frame)


def collapsed(texts: dict[str, str]) -> dict[str, str]:
    """Return the texts as the earlier version sent and hashed them."""
    return {text_id: " ".join(text.split()) for text_id, text in texts.items()}


def assert_names_exactly(message: str, refused: dict[str, str], kept: dict[str, str]) -> None:
    """Check that the message names every refused id and no kept one."""
    assert all(repr(text_id) in message for text_id in refused), message
    assert not any(repr(text_id) in message for text_id in kept), message
    assert all(phrase in message for phrase in COLLAPSED_PHRASES), message


@pytest.fixture
def earlier_store(toy_rubric: Rubric, cue_backend: Factory, tmp_path: Path) -> Path:
    """Return the store that the earlier version wrote for the special and the plain texts.

    That version sent and hashed every text with its whitespace collapsed, so coding the
    collapsed texts writes its store: the same prompts, hashes and passages.
    """
    store = tmp_path / "earlier.sqlite"
    everything = collapsed({**SPECIAL, **PLAIN})
    code_texts(table_of(everything), toy_rubric, cue_backend(), store, SETTINGS)
    return store


# Texts as a user may give them: whitespace between the words, at the ends, or both.
AS_GIVEN = [
    pytest.param("First line.\nSecond line.", id="line_break"),
    pytest.param("A tab\tsits between these words.", id="tab"),
    pytest.param("Two  spaces sit between these words.", id="spaces"),
    pytest.param("A no-break space sits in this answer.", id="no_break_space"),
    pytest.param("First line.\r\nSecond line.\n\nA new paragraph.", id="line_breaks"),
    pytest.param(" \tA tab,\ttwo  spaces, a no-break space\n and a break. \n", id="mixed"),
    pytest.param("  Padding sits at both ends of this answer only.\t\n", id="padding"),
]


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("given", AS_GIVEN)
def test_both_coders_send_the_text_as_given_and_store_the_hash_of_what_they_sent(
    toy_rubric: Rubric, cue_backend: Factory, tmp_path: Path, method: str, given: str
) -> None:
    backend = cue_backend()
    store = tmp_path / "run.sqlite"
    code_texts(
        table_of({"a": given}), toy_rubric, backend, store, replace(SETTINGS, runs=2, method=method)
    )
    sent = [
        call.user.removeprefix(TEXT_PREFIX)
        for call in backend.calls
        if call.user.startswith(TEXT_PREFIX)
    ]
    assert len(sent) == 2 * len(toy_rubric.attributes)
    assert set(sent) == {given.strip()}
    digests = {hashlib.sha256(text.encode("utf-8")).hexdigest() for text in sent}
    assert set(export_scores(store)["text_hash"]) == digests


def test_a_resume_refuses_exactly_the_texts_whose_whitespace_an_earlier_version_collapsed(
    toy_rubric: Rubric, cue_backend: Factory, earlier_store: Path
) -> None:
    backend = cue_backend()
    with pytest.raises(StoreError) as refused:
        code_texts(table_of({**SPECIAL, **PLAIN}), toy_rubric, backend, earlier_store, SETTINGS)
    message = str(refused.value)
    assert f"4 text ids of the store {earlier_store} were coded" in message
    assert_names_exactly(message, SPECIAL, PLAIN)
    assert CHANGED_TEXTS not in message and backend.calls == []
    plain = cue_backend()
    resumed = code_texts(table_of(PLAIN), toy_rubric, plain, earlier_store, SETTINGS)
    assert (resumed.resumed, resumed.n_attribute_calls, plain.calls) == (2, 0, [])
    old_coding = cue_backend()
    everything = table_of(collapsed({**SPECIAL, **PLAIN}))
    resumed = code_texts(everything, toy_rubric, old_coding, earlier_store, SETTINGS)
    assert (resumed.resumed, resumed.n_attribute_calls, old_coding.calls) == (6, 0, [])


def test_a_resume_names_the_changed_and_the_collapsed_texts_apart(
    toy_rubric: Rubric, cue_backend: Factory, earlier_store: Path
) -> None:
    revised = {**SPECIAL, **PLAIN, "plain": "Another answer under the same id, never coded."}
    with pytest.raises(StoreError) as refused:
        code_texts(table_of(revised), toy_rubric, cue_backend(), earlier_store, SETTINGS)
    message = str(refused.value)
    changed, _, whitespace = message.partition(f"; 4 text ids of the store {earlier_store}")
    assert CHANGED_TEXTS in changed and "1 changed text id: 'plain')" in changed
    assert_names_exactly(whitespace, SPECIAL, PLAIN)


def test_a_merge_refuses_exactly_the_texts_whose_whitespace_an_earlier_version_collapsed(
    toy_rubric: Rubric, cue_backend: Factory, earlier_store: Path, tmp_path: Path
) -> None:
    later = tmp_path / "later.sqlite"
    code_texts(table_of({**SPECIAL, **PLAIN}), toy_rubric, cue_backend(), later, SETTINGS)
    target = tmp_path / "merged.sqlite"
    with pytest.raises(StoreError) as refused:
        merge_stores([earlier_store, later], target)
    message = str(refused.value)
    assert TEXTS_DIFFER in message and "4 text ids: " in message
    assert_names_exactly(message, SPECIAL, PLAIN)
    assert not target.exists()
    plain = tmp_path / "plain.sqlite"
    code_texts(table_of(PLAIN), toy_rubric, cue_backend(), plain, replace(SETTINGS, runs=2))
    counts = merge_stores([earlier_store, plain], target)
    assert (counts["results"], counts["conflicting_cells"]) == (8, 0)


@pytest.mark.parametrize("granularity", ["construct", "attribute"])
def test_the_hybrid_coder_refuses_exactly_the_texts_whose_whitespace_an_earlier_version_collapsed(
    toy_rubric: Rubric, earlier_store: Path, granularity: str
) -> None:
    table = table_of({**SPECIAL, **PLAIN})
    with pytest.raises(ConfigurationError) as refused:
        evidence_texts(earlier_store, toy_rubric, table, granularity=granularity)  # type: ignore[call-overload]
    message = str(refused.value)
    assert f"4 of 6 texts were coded in run 1 of the store {earlier_store}" in message
    assert_names_exactly(message, SPECIAL, PLAIN)
    encoder = StubEncoder()
    refs = unit_references(toy_rubric, "stub", STUB_DIMENSION)
    with pytest.raises(ConfigurationError, match="continue the old coding"):
        score_hybrid(earlier_store, table, toy_rubric, encoder, refs, granularity=granularity)
    assert encoder.calls == []
    for texts in (PLAIN, collapsed({**SPECIAL, **PLAIN})):
        rebuilt = evidence_texts(
            earlier_store, toy_rubric, table_of(texts), granularity="construct"
        )
        assert rebuilt.ids == list(texts)


@pytest.mark.parametrize("first", [0, 1], ids=["number_first", "string_first"])
def test_the_hybrid_coder_sorts_each_row_on_its_own_when_two_ids_read_alike(
    toy_rubric: Rubric, cue_backend: Factory, tmp_path: Path, first: int
) -> None:
    store = tmp_path / "run.sqlite"
    plain = table_of({"1": "First line. Second line of the answer."})
    code_texts(plain, toy_rubric, cue_backend(), store, SETTINGS)
    rows = [(1, "First line.\nSecond line of the answer."), ("1", "Another text under that id.")]
    ordered = [rows[first], rows[1 - first]]
    frame = pd.DataFrame(
        {
            "text_id": [row[0] for row in ordered],
            "text": [row[1] for row in ordered],
            "group_id": "g",
        }
    )
    with pytest.raises(ConfigurationError) as refused:
        evidence_texts(store, toy_rubric, TextTable.from_frame(frame), granularity="construct")
    message = str(refused.value)
    assert "1 of 2 texts differ from the texts that run 1" in message
    assert "1 of 2 texts was coded in run 1" in message


def one_passage(passage: str) -> Callable[[GenerationRequest], str]:
    """Answer each extraction with one passage for its first marker, each scoring with 3."""

    def respond(request: GenerationRequest) -> str:
        if STAGE_ONE_MARKER not in request.system:
            return json.dumps({"result": 3, "explanation": "one passage"})
        names = TEMPLATE_MARKER_PATTERN.findall(request.system)
        answer = {name: {"evaluable": False, "quotes": []} for name in names}
        answer[names[0]] = {"evaluable": True, "quotes": [passage]}
        return json.dumps(answer)

    return respond


@pytest.mark.parametrize("passage", ["line. Second", "line.\nSecond"])
def test_a_passage_across_a_line_break_is_found_where_an_earlier_version_found_it(
    toy_rubric: Rubric, tmp_path: Path, passage: str
) -> None:
    stores = {}
    for name, text in (("kept", "First line.\nSecond line."), ("old", "First line. Second line.")):
        stores[name] = tmp_path / f"{name}.sqlite"
        backend = FakeBackend(one_passage(passage))
        code_texts(table_of({"a": text}), toy_rubric, backend, stores[name], SETTINGS)
    evidence = export_evidence(stores["kept"])
    n_attributes = len(toy_rubric.attributes)
    assert evidence["quote_norm"].tolist() == ["line. Second"] * n_attributes
    assert evidence["verified"].tolist() == [1] * n_attributes
    assert evidence["match_score"].tolist() == [100.0] * n_attributes
    # The offsets the earlier version stored: those of the passage in "first line. second line."
    assert evidence["char_start"].tolist() == [6] * n_attributes
    assert evidence["char_end"].tolist() == [18] * n_attributes
    old = export_evidence(stores["old"])
    pd.testing.assert_frame_equal(evidence[EVIDENCE_COLUMNS], old[EVIDENCE_COLUMNS])


@pytest.mark.parametrize("method", METHODS)
def test_the_length_rule_counts_the_characters_of_the_text_the_coder_sends(
    toy_rubric: Rubric, cue_backend: Factory, tmp_path: Path, method: str
) -> None:
    texts = {"long": "Line one.\n\n\n\nLine two.", "short": "Line one.\nLine two."}
    table = table_of(texts)
    backend = cue_backend()
    store = tmp_path / "run.sqlite"
    code_texts(table, toy_rubric, backend, store, replace(SETTINGS, method=method))
    statuses = export_scores(store).groupby("text_id")["status"].agg(set).to_dict()
    assert "too_short" not in statuses["long"] and statuses["short"] == {"too_short"}
    sent = {call.user for call in backend.calls if call.user.startswith(TEXT_PREFIX)}
    assert sent == {TEXT_PREFIX + texts["long"]}
    assert [len(table.text_of(text_id)) for text_id in texts] == [22, 19]
    assert table.frame["n_chars"].tolist() == [19, 19]
