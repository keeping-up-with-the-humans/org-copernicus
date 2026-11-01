"""Tests that run the MiniLM encoder; skipped when its weights are not cached."""

from __future__ import annotations

import numpy as np
import pytest
from embeddings_support import MINILM_AVAILABLE, SKIP_REASON

from copernicus.data import TextTable
from copernicus.embeddings import Encoder

pytestmark = [
    pytest.mark.embed,
    pytest.mark.skipif(not MINILM_AVAILABLE, reason=SKIP_REASON),
]


def test_minilm_embeds_unit_vectors_and_chunks_long_text() -> None:
    enc = Encoder("minilm", device="cpu", batch_size=4)
    texts = [
        "The cat sat on the mat.",
        "A cat was sitting on a mat.",
        "Quarterly revenue fell sharply.",
        " ".join(["word"] * 900),
    ]
    es = enc.embed(texts, ids=[1, 2, 3, 4])
    assert es.vectors.shape == (4, 384) and np.allclose(
        np.linalg.norm(es.vectors, axis=1), 1.0, atol=1e-4
    )
    sim = es.vectors @ es.vectors.T
    assert sim[0, 1] > sim[0, 2]
    assert (es.chunk_owner == 3).sum() >= 3 and es.chunk_weight[es.chunk_owner == 3].sum() >= 900
    assert (
        es.encoder == "minilm" and es.sidecar["hf_id"] == "sentence-transformers/all-MiniLM-L6-v2"
    )
    assert es.sidecar["revision"] == "unpinned"
    resolved = es.sidecar["resolved_revision"]
    assert len(resolved) == 40 and all(character in "0123456789abcdef" for character in resolved)
    assert es.sidecar["torch_version"] and es.sidecar["transformers_version"]


def test_minilm_first_chunk_of_a_short_text_is_its_text_vector() -> None:
    es = Encoder("minilm", device="cpu").embed(["The cat sat on the mat.", "Revenue fell."], [1, 2])
    assert np.allclose(es.first_chunk_vectors(), es.vectors, atol=1e-5)


def test_minilm_embeds_the_first_three_texts_of_the_toy_corpus(toy_corpus: TextTable) -> None:
    table = TextTable.from_frame(toy_corpus.frame.head(3))
    es = Encoder("minilm", device="cpu").embed_table(table)
    assert es.ids.tolist() == table.ids() and es.vectors.shape == (3, 384)
    assert es.metadata is not None and es.metadata.text_id.tolist() == table.ids()
