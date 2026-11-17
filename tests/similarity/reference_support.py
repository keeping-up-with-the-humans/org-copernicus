"""Builders of the reference embedding tests: a hand-made reference set and a stub encoder."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd

from copernicus.embeddings import EmbeddingSet, EncoderSpec
from copernicus.references import GeneratorSettings, ReferenceSet, build_cells
from copernicus.references.reference_set import FRAME_COLUMNS
from copernicus.rubric.model import Rubric
from copernicus.similarity import ReferenceEmbeddings, embed_references

TEXTS_PER_CELL = 2
STUB_VERSIONS = {
    "resolved_revision": "abc123def",
    "torch_version": "9.9.9",
    "transformers_version": "8.8.8",
}


def toy_reference_set(rubric: Rubric, writers: tuple[str, ...] = ("writer-a",)) -> ReferenceSet:
    """Build a reference set by hand: two texts per cell and writer, nothing screened out."""
    rows: list[dict[str, Any]] = []
    for writer in writers:
        for cell in build_cells(rubric):
            for number in range(TEXTS_PER_CELL):
                rows.append(
                    {
                        "rubric_id": rubric.rubric_id,
                        "target": cell.target,
                        "polarity": cell.polarity,
                        "level": cell.level,
                        "style": "neutral",
                        "writer_model": writer,
                        "text": (
                            f"A {cell.polarity} note about {cell.target} at the {cell.level} "
                            f"level, number {number}, from {writer}."
                        ),
                        "call_index": number,
                        "attempt": 0,
                        "seed": 42 + number,
                        "prompt_tokens": 100.0,
                        "completion_tokens": 50.0,
                        "temperature_applied": 0.9,
                        "top_p_applied": None,
                        "prompt_hash": "hash",
                        "prompt_version": "v1",
                        "screened_out": False,
                        "screen_reason": "",
                    }
                )
    frame = pd.DataFrame(rows, columns=list(FRAME_COLUMNS))
    return ReferenceSet.for_rubric(frame, rubric, GeneratorSettings())


class StubEncoder:
    """An encoder stand-in: gives each text the given number of chunks of random vectors."""

    def __init__(self, chunks_per_text: list[int], dimension: int = 5) -> None:
        self.spec = EncoderSpec("stub", "org/stub", "mean", 8, revision="abc123")
        self.chunks_per_text = chunks_per_text
        self.dimension = dimension
        self.seen: tuple[list[str], list[Any]] | None = None

    def embed(self, texts: list[str], ids: list[Any]) -> EmbeddingSet:
        self.seen = (list(texts), list(ids))
        generator = np.random.default_rng(11)
        owner = np.repeat(np.arange(len(texts)), self.chunks_per_text).astype(np.int64)
        chunks = generator.normal(size=(len(owner), self.dimension))
        chunks /= np.linalg.norm(chunks, axis=1, keepdims=True)
        sums = np.zeros((len(texts), self.dimension))
        np.add.at(sums, owner, chunks)
        sums /= np.linalg.norm(sums, axis=1, keepdims=True)
        return EmbeddingSet(
            ids=np.asarray([str(text_id) for text_id in ids]),
            vectors=sums.astype(np.float32),
            chunk_vectors=chunks.astype(np.float32),
            chunk_owner=owner,
            chunk_weight=np.ones(len(owner)),
            encoder=self.spec.name,
            metadata=None,
            sidecar={**self.spec.identity(), **STUB_VERSIONS},
        )


def stub_embeddings(
    rubric: Rubric, writers: tuple[str, ...] = ("writer-a",)
) -> ReferenceEmbeddings:
    """Embed the hand-built reference set with the stub encoder, one chunk per text."""
    refset = toy_reference_set(rubric, writers)
    encoder: Any = StubEncoder([1] * len(refset.kept()))
    return embed_references(refset, encoder)


def drop_rows(refs: ReferenceEmbeddings, mask: np.ndarray) -> ReferenceEmbeddings:
    """Return the reference embeddings without the rows where ``mask`` is true."""
    keep = ~mask
    return replace(
        refs,
        vectors=refs.vectors[keep],
        target=refs.target[keep],
        polarity=refs.polarity[keep],
        level=refs.level[keep],
        style=refs.style[keep],
        writer_model=refs.writer_model[keep],
    )
