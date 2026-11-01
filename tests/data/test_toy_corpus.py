"""Tests for the example corpus that ships with the package."""

from __future__ import annotations

import re

import pandas as pd
from scipy.stats import spearmanr

from copernicus.data import TextTable

BANDS = {
    "low": (range(1, 7), {1, 2}),
    "mixed": (range(7, 13), {2, 3, 4}),
    "high": (range(13, 19), {4, 5}),
}
SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


def test_shape_and_identifiers(toy_corpus: TextTable) -> None:
    frame = toy_corpus.frame
    assert frame.text_id.tolist() == [f"t{i:03d}" for i in range(1, 55)]
    assert frame.groupby("group_id").size().eq(3).all() and frame.group_id.nunique() == 18
    assert not toy_corpus.group_defaulted


def test_gold_bands_and_within_writer_variation(toy_corpus: TextTable) -> None:
    frame = toy_corpus.frame
    gold_columns = [c for c in frame.columns if c.startswith("gold_")]
    assert gold_columns == [
        "gold_constructiveness",
        "gold_specificity",
        "gold_tone",
        "gold_reasoning",
    ]
    assert frame[gold_columns].apply(lambda column: column.between(1, 5)).all().all()
    for writers, allowed in BANDS.values():
        for number in writers:
            gold = frame.loc[frame.group_id == f"w{number:02d}", "gold_constructiveness"]
            assert set(gold) <= allowed and gold.nunique() > 1


def test_texts_are_three_to_six_sentences_and_lengths_are_exact(toy_corpus: TextTable) -> None:
    frame = toy_corpus.frame
    counts = frame.text.map(lambda text: len(SENTENCE_BOUNDARY.split(text)))
    assert counts.between(3, 6).all()
    assert frame.length_words.tolist() == frame.n_words.tolist()


def test_length_is_a_weak_proxy_of_the_gold_score(toy_corpus: TextTable) -> None:
    frame: pd.DataFrame = toy_corpus.frame
    assert spearmanr(frame.gold_constructiveness, frame.length_words).correlation < 0.8
