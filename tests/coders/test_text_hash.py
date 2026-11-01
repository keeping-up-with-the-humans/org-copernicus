"""The text hash binds a stored coding to the text it was made on."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest

from copernicus.assets import load_toy_cues
from copernicus.coders import CodingSettings, code_texts, export_scores, hash_text
from copernicus.data import TextTable
from copernicus.llm import FakeBackend, cue_responder
from copernicus.rubric.model import Rubric

HEX_DIGITS = set("0123456789abcdef")


@pytest.mark.parametrize("text", ["Thank you for the draft.", "Grüße, señor: naïve café ✓", ""])
def test_the_text_hash_is_the_sha256_hex_digest_of_the_utf8_text(text: str) -> None:
    digest = hash_text(text)
    assert digest == hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert len(digest) == 64 and set(digest) <= HEX_DIGITS


def test_the_text_hash_follows_every_character_of_the_text() -> None:
    assert hash_text("The draft is clear.") != hash_text("The draft is clear!")
    assert hash_text("a b") != hash_text("a  b")
    assert hash_text("Draft") != hash_text("draft")


def test_only_a_space_between_letters_changes_the_hash_of_a_table_text() -> None:
    """The documented rule: whitespace runs collapse to one space before hashing."""
    texts = {"one": "a b", "wide": "a   b", "tab": "a\tb", "ends": "  a b \n", "split": "ab"}
    frame = pd.DataFrame({"text_id": list(texts), "text": list(texts.values()), "group_id": "g"})
    table = TextTable.from_frame(frame)
    hashes = {text_id: hash_text(table.text_of(text_id)) for text_id in texts}
    assert len({hashes[text_id] for text_id in ("one", "wide", "tab", "ends")}) == 1
    assert hashes["split"] != hashes["one"]


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
        "a": hash_text("Thank you for the draft today."),
        "b": hash_text("The summary misses the method."),
    }
    assert scores["text_hash"].tolist() == [expected[text_id] for text_id in scores["text_id"]]
    assert {hash_text(table.text_of(text_id)) for text_id in "ab"} == set(expected.values())
