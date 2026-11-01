"""Numpy-only builders of embedding sets and reference embeddings for the similarity tests."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import numpy as np

from copernicus.embeddings import EmbeddingSet
from copernicus.rubric.model import Rubric
from copernicus.similarity import ReferenceEmbeddings

DIMENSION = 8
ENCODER = "test-encoder"
SIDECAR = {
    "hf_id": "test/encoder",
    "revision": "unpinned",
    "pooling": "mean",
    "max_tokens": 128,
    "dtype": "float32",
    "trust_remote_code": False,
    "padding_side": "right",
    "special_tokens_reserved": 2,
}


def unit(rows: np.ndarray) -> np.ndarray:
    """Return the rows scaled to unit length, as float32."""
    rows = np.atleast_2d(np.asarray(rows, dtype=np.float64))
    return (rows / np.linalg.norm(rows, axis=1, keepdims=True)).astype(np.float32)


def make_texts(
    text_vectors: np.ndarray,
    chunk_vectors: Sequence[np.ndarray] | None = None,
    encoder: str = ENCODER,
) -> EmbeddingSet:
    """Build an embedding set; ``chunk_vectors`` lists the chunk rows of each text.

    Without ``chunk_vectors`` every text has one chunk equal to its vector.
    """
    vectors = unit(text_vectors)
    chunks = [vectors[[row]] for row in range(len(vectors))]
    if chunk_vectors is not None:
        chunks = [unit(rows) for rows in chunk_vectors]
    owner = np.concatenate([np.full(len(rows), row) for row, rows in enumerate(chunks)])
    return EmbeddingSet(
        ids=np.asarray([f"t{row}" for row in range(len(vectors))], dtype=str),
        vectors=vectors,
        chunk_vectors=np.vstack(chunks),
        chunk_owner=owner.astype(np.int64),
        chunk_weight=np.ones(len(owner), dtype=np.int64),
        encoder=encoder,
        metadata=None,
        sidecar=dict(SIDECAR),
    )


def make_references(
    rubric: Rubric,
    writers: Sequence[str] = ("writer-a", "writer-b"),
    per_pole: int = 3,
    style: str | Sequence[str] = "neutral",
    encoder: str = ENCODER,
    seed: int = 0,
) -> ReferenceEmbeddings:
    """Build reference embeddings: positive poles near +e1, negative poles near -e1.

    Every target of the rubric gets ``per_pole`` references per polarity and writer;
    ``style`` is one name or one per writer. The sidecar binds the references to the id and
    the content hash of ``rubric``, as ``for_rubric`` binds a set written by hand.
    """
    rng = np.random.default_rng(seed)
    targets = ["construct", *(attribute.attribute_id for attribute in rubric.attributes)]
    styles = [style] * len(writers) if isinstance(style, str) else list(style)
    rows: list[np.ndarray] = []
    labels: list[tuple[str, str, str, str]] = []
    for target in targets:
        for polarity, sign in (("positive", 1.0), ("negative", -1.0)):
            for writer, writer_style in zip(writers, styles, strict=True):
                for _ in range(per_pole):
                    noise = rng.normal(scale=0.3, size=DIMENSION)
                    noise[0] = sign * 1.0 + rng.normal(scale=0.1)
                    rows.append(noise)
                    labels.append((target, polarity, writer, writer_style))
    columns = list(zip(*labels, strict=True))
    return ReferenceEmbeddings(
        vectors=unit(np.vstack(rows)),
        target=np.asarray(columns[0], dtype=str),
        polarity=np.asarray(columns[1], dtype=str),
        level=np.asarray(["top"] * len(rows), dtype=str),
        style=np.asarray(columns[3], dtype=str),
        writer_model=np.asarray(columns[2], dtype=str),
        rubric_id=rubric.rubric_id,
        encoder=encoder,
        sidecar={
            **SIDECAR,
            "rubric_id": rubric.rubric_id,
            "rubric_hash": rubric.content_hash(),
            "bound_with": "for_rubric",
            "n_references": len(rows),
        },
    )


def texts_along_first_axis(count: int = 5, seed: int = 1) -> np.ndarray:
    """Return ``count`` text vectors whose first coordinate rises from text to text."""
    rng = np.random.default_rng(seed)
    rows = rng.normal(scale=0.01, size=(count, DIMENSION))
    rows[:, 0] = np.linspace(-1.0, 1.0, count)
    rows[:, 1] = 0.5
    return rows


def without_pole(refs: ReferenceEmbeddings, target: str, polarity: str) -> ReferenceEmbeddings:
    """Return the references with all rows of one pole removed."""
    keep = ~((refs.target == target) & (refs.polarity == polarity))
    return replace(
        refs,
        vectors=refs.vectors[keep],
        target=refs.target[keep],
        polarity=refs.polarity[keep],
        level=refs.level[keep],
        style=refs.style[keep],
        writer_model=refs.writer_model[keep],
    )
