"""What `Encoder.embed` does with the texts it is given, checked without torch or transformers.

Each test replaces the encoder's `_load`, which imports torch and transformers, by a runtime that
holds a word tokenizer and no model, and its `_pool_fragments`, which runs the model, by a
stand-in that gives every chunk one vector. The rest of `Encoder.embed` runs as it is, so these
tests run where neither torch nor transformers is installed, as in the continuous integration,
and `tests/test_ci_embed_markers.py` lets them build an encoder without the embed marker. The
tests of the chunking, the pooling and the checkpoints on a fake model need torch and carry the
embed marker.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from embeddings_support import TOY_VERSIONS, WordTokenizer, documented_texts_hash

from copernicus.embeddings import Encoder, EncoderSpec
from copernicus.embeddings.encoder import EncoderRuntime

SPEC = EncoderSpec("plain", "org/plain", "mean", 64)
# The first text holds a line break, a tab and a run of spaces between its words, and the texts
# are not in sorted order, so neither a collapsed text nor the digest of the sorted texts passes.
TEXTS = ["z first\nsecond\tpart  end", "a last"]
IDS = ["z", "a"]


class RecordingTokenizer(WordTokenizer):
    """The word tokenizer, recording every text that the encoder hands it to split."""

    def __init__(self) -> None:
        super().__init__()
        self.seen: list[str] = []

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        self.seen.append(text)
        return super().encode(text, add_special_tokens)


def runtime_with(tokenizer: WordTokenizer) -> EncoderRuntime:
    """A runtime that holds the tokenizer alone: no torch, no model, no pooling, no device."""
    return EncoderRuntime(
        torch=None,
        tokenizer=tokenizer,
        model=None,
        pool=None,  # type: ignore[arg-type]
        device=None,
        versions=dict(TOY_VERSIONS),
    )


def one_vector_per_chunk(
    encoder: Encoder,
    runtime: EncoderRuntime,
    fragments: list[str],
    labels: list[str],
    checkpoint: Path | None,
    checkpoint_every: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Stand in for the model: the same raw vector and one token for every chunk."""
    return np.ones((len(fragments), 2)), np.ones(len(fragments))


def test_the_tokenizer_receives_each_text_as_given(monkeypatch: pytest.MonkeyPatch) -> None:
    tokenizer = RecordingTokenizer()
    monkeypatch.setattr(Encoder, "_load", lambda self: runtime_with(tokenizer))
    monkeypatch.setattr(Encoder, "_pool_fragments", one_vector_per_chunk)
    Encoder(SPEC).embed(TEXTS, IDS)
    assert all(part in TEXTS[0] for part in ("\n", "\t", "  "))
    assert tokenizer.seen == TEXTS


def test_the_set_records_the_digest_of_the_texts_in_the_order_given(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Encoder, "_load", lambda self: runtime_with(WordTokenizer()))
    monkeypatch.setattr(Encoder, "_pool_fragments", one_vector_per_chunk)
    result = Encoder(SPEC).embed(TEXTS, IDS)
    assert sorted(TEXTS) != TEXTS
    assert result.sidecar["texts_hash"] == documented_texts_hash(TEXTS)
