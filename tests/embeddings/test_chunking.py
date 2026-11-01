"""Tests for the splitting of texts into token windows."""

from __future__ import annotations

import pytest
from embeddings_support import WordTokenizer

from copernicus.embeddings import split_into_token_chunks
from copernicus.errors import ConfigurationError

TEN_WORDS = "w0 w1 w2 w3 w4 w5 w6 w7 w8 w9"


def test_split_into_token_chunks_with_fake_tokenizer() -> None:
    chunks = split_into_token_chunks(TEN_WORDS, WordTokenizer(), max_tokens=6, reserved=2)
    assert chunks == ["w0 w1 w2 w3", "w4 w5 w6 w7", "w8 w9"]
    assert " ".join(chunks) == TEN_WORDS
    assert split_into_token_chunks("short text", WordTokenizer(), max_tokens=6, reserved=2) == [
        "short text"
    ]


def test_a_text_exactly_filling_the_window_stays_whole() -> None:
    assert split_into_token_chunks("a b c d", WordTokenizer(), 6, 2) == ["a b c d"]
    assert len(split_into_token_chunks("a b c d e", WordTokenizer(), 6, 2)) == 2


def test_an_empty_text_is_one_empty_window() -> None:
    assert split_into_token_chunks("", WordTokenizer(), 6, 2) == [""]


@pytest.mark.parametrize(("max_tokens", "reserved"), [(2, 2), (1, 2), (5, 5)])
def test_a_window_without_room_for_text_is_refused(max_tokens: int, reserved: int) -> None:
    with pytest.raises(ConfigurationError, match="no room"):
        split_into_token_chunks("a b c", WordTokenizer(), max_tokens, reserved)
