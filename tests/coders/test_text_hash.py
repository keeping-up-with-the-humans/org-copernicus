"""The text hash binds a stored coding to the text it was made on."""

from __future__ import annotations

import hashlib
import sqlite3
from contextlib import closing
from pathlib import Path

import pandas as pd
import pytest

from copernicus.assets import load_toy_cues
from copernicus.coders import CodingSettings, code_texts, export_scores, hash_text
from copernicus.coders.text_hash import COLLAPSED_FORM, collapsed_hash
from copernicus.data import TextTable
from copernicus.errors import StoreError
from copernicus.llm import FakeBackend, cue_responder
from copernicus.rubric.model import Rubric

HEX_DIGITS = set("0123456789abcdef")
PLAIN_TEXT = "Thank you for the draft today."
# The digest that every version of the library stores for PLAIN_TEXT.
PLAIN_DIGEST = "2c9fb8b90d1de593e50dc1753fd910ccfb60172e896611d0106b5615fe78f930"


@pytest.mark.parametrize("text", ["Thank you for the draft.", "Grüße, señor: naïve café ✓", ""])
def test_the_text_hash_is_the_sha256_hex_digest_of_the_utf8_text(text: str) -> None:
    digest = hash_text(text)
    assert digest == hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert len(digest) == 64 and set(digest) <= HEX_DIGITS


def test_the_text_hash_follows_every_character_of_the_text() -> None:
    assert hash_text("The draft is clear.") != hash_text("The draft is clear!")
    assert hash_text("a b") != hash_text("a  b")
    assert hash_text("Draft") != hash_text("draft")


def test_whitespace_between_words_changes_the_hash_and_whitespace_at_the_ends_does_not() -> None:
    """The documented rule: the hash is taken of the text as given, with its ends stripped."""
    texts = {
        "one": "a b",
        "ends": "  a b \n",
        "wide": "a   b",
        "tab": "a\tb",
        "break": "a\nb",
        "split": "ab",
    }
    frame = pd.DataFrame({"text_id": list(texts), "text": list(texts.values()), "group_id": "g"})
    table = TextTable.from_frame(frame)
    hashes = {text_id: hash_text(table.text_of(text_id)) for text_id in texts}
    assert hashes["ends"] == hashes["one"] == hash_text("a b")
    assert len({hashes[text_id] for text_id in ("one", "wide", "tab", "break", "split")}) == 5


def test_every_stored_row_carries_the_hash_of_the_text_the_coder_received(
    toy_rubric: Rubric, tmp_path: Path
) -> None:
    frame = pd.DataFrame(
        {
            "text_id": ["a", "b"],
            "text": ["  Thank you   for the draft today. ", "The summary\nmisses the method."],
            "group_id": ["g1", "g2"],
        }
    )
    table = TextTable.from_frame(frame)
    store = tmp_path / "run.sqlite"
    backend = FakeBackend(cue_responder(toy_rubric, load_toy_cues()))
    code_texts(table, toy_rubric, backend, store, CodingSettings(runs=2, temperature=0.3))
    scores = export_scores(store)
    expected = {
        "a": hash_text("Thank you   for the draft today."),
        "b": hash_text("The summary\nmisses the method."),
    }
    assert scores["text_hash"].tolist() == [expected[text_id] for text_id in scores["text_id"]]
    assert {hash_text(table.text_of(text_id)) for text_id in "ab"} == set(expected.values())


@pytest.mark.parametrize("given", [PLAIN_TEXT, f"  {PLAIN_TEXT} \n"])
def test_a_text_without_whitespace_between_words_keeps_the_hash_of_the_earlier_version(
    toy_rubric: Rubric, tmp_path: Path, given: str
) -> None:
    """The earlier version hashed the collapsed text, which for this text is the text itself."""
    assert hashlib.sha256(PLAIN_TEXT.encode("utf-8")).hexdigest() == PLAIN_DIGEST
    frame = pd.DataFrame({"text_id": ["a"], "text": [given], "group_id": ["g"]})
    store = tmp_path / "run.sqlite"
    backend = FakeBackend(cue_responder(toy_rubric, load_toy_cues()))
    code_texts(TextTable.from_frame(frame), toy_rubric, backend, store, CodingSettings(runs=1))
    assert set(export_scores(store)["text_hash"]) == {PLAIN_DIGEST}


def test_an_id_counts_as_coded_collapsed_only_when_every_stored_cell_holds_the_collapsed_hash(
    toy_rubric: Rubric, tmp_path: Path
) -> None:
    """Only a store changed by hand holds two hashes under one id; this one is changed so."""
    given = "First line.\nSecond line."
    frame = pd.DataFrame({"text_id": ["a"], "text": [given], "group_id": ["g"]})
    table = TextTable.from_frame(frame)
    store = tmp_path / "run.sqlite"
    settings = CodingSettings(runs=2, temperature=0.3)
    backend = FakeBackend(cue_responder(toy_rubric, load_toy_cues()))
    code_texts(table, toy_rubric, backend, store, settings)
    refusals = []
    for second in (collapsed_hash(given), hash_text("Another text.")):
        with closing(sqlite3.connect(store)) as connection, connection:
            hashes = {1: collapsed_hash(given), 2: second}
            for run, digest in hashes.items():
                connection.execute(
                    'UPDATE "results" SET "text_hash" = ? WHERE "run_number" = ?', (digest, run)
                )
        resumed = FakeBackend(cue_responder(toy_rubric, load_toy_cues()))
        with pytest.raises(StoreError) as refused:
            code_texts(table, toy_rubric, resumed, store, settings)
        assert resumed.calls == []
        refusals.append(str(refused.value))
    every_cell, one_cell = refusals
    assert f"1 text id of the store {store} was coded {COLLAPSED_FORM}: 'a'" in every_cell
    assert "changed text id" not in every_cell
    assert "1 changed text id: 'a'" in one_cell and COLLAPSED_FORM not in one_cell
