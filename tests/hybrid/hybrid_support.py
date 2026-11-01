"""Helpers of the hybrid coder tests: a model-free encoder and hand-built references."""

from __future__ import annotations

import zlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from copernicus.embeddings import EmbeddingSet, EncoderSpec
from copernicus.rubric.model import Rubric
from copernicus.similarity import ReferenceEmbeddings

STUB_DIMENSION = 16
STUB_NAME = "stub"
STUB_HF_ID = "test/stub"
STUB_WINDOW = 128
STUB_SIDECAR = EncoderSpec(STUB_NAME, STUB_HF_ID, "mean", STUB_WINDOW).identity()
EVIDENCE_COLUMNS = (
    "text_id",
    "run_number",
    "model_name",
    "temperature",
    "attribute_id",
    "marker_id",
    "quote_index",
    "quote_raw",
    "quote_norm",
    "verified",
    "match_score",
    "char_start",
    "char_end",
    "duplicate_of",
    "used_in_stage2",
)


class StubEncoder:
    """An encoder without a model: the vector of a text is drawn from the hash of the text.

    It records every call, so tests can see which texts were embedded.
    """

    def __init__(
        self, name: str = STUB_NAME, dimension: int = STUB_DIMENSION, hf_id: str = STUB_HF_ID
    ) -> None:
        self.spec = EncoderSpec(name, hf_id, "mean", STUB_WINDOW)
        self.dimension = dimension
        self.calls: list[list[str]] = []
        self.options: list[dict[str, Any]] = []

    def embed(
        self,
        texts: Sequence[str],
        ids: Sequence[Any],
        *,
        checkpoint: Path | None = None,
        checkpoint_every: int = 2048,
    ) -> EmbeddingSet:
        """Return one chunk per text, equal to the unit vector drawn for the text."""
        self.calls.append(list(texts))
        self.options.append({"checkpoint": checkpoint, "checkpoint_every": checkpoint_every})
        rows = np.vstack(
            [
                np.random.default_rng(zlib.crc32(text.encode("utf-8"))).normal(size=self.dimension)
                for text in texts
            ]
        )
        vectors = (rows / np.linalg.norm(rows, axis=1, keepdims=True)).astype(np.float32)
        return EmbeddingSet(
            ids=np.asarray([str(text_id) for text_id in ids]),
            vectors=vectors,
            chunk_vectors=vectors.copy(),
            chunk_owner=np.arange(len(texts), dtype=np.int64),
            chunk_weight=np.ones(len(texts), dtype=np.int64),
            encoder=self.spec.name,
            metadata=None,
            sidecar={**self.spec.identity(), "resolved_revision": "stub-commit"},
        )


def unit_references(
    rubric: Rubric,
    encoder: str,
    dimension: int,
    per_pole: int = 3,
    seed: int = 5,
    identity: Mapping[str, Any] = STUB_SIDECAR,
) -> ReferenceEmbeddings:
    """Build reference embeddings of random unit vectors: all poles of all targets, two writers.

    ``identity`` gives the encoder identity keys of the sidecar; the stub's by default. The
    sidecar binds the references to the id and the content hash of ``rubric``, as
    ``for_rubric`` binds a set written by hand.
    """
    rng = np.random.default_rng(seed)
    targets = ["construct", *(attribute.attribute_id for attribute in rubric.attributes)]
    labels = [
        (target, polarity, writer)
        for target in targets
        for polarity in ("positive", "negative")
        for writer in ("writer-a", "writer-b")
        for _ in range(per_pole)
    ]
    rows = rng.normal(size=(len(labels), dimension))
    rows /= np.linalg.norm(rows, axis=1, keepdims=True)
    columns = list(zip(*labels, strict=True))
    return ReferenceEmbeddings(
        vectors=rows.astype(np.float32),
        target=np.asarray(columns[0], dtype=str),
        polarity=np.asarray(columns[1], dtype=str),
        level=np.asarray(["top"] * len(labels), dtype=str),
        style=np.asarray(["neutral"] * len(labels), dtype=str),
        writer_model=np.asarray(columns[2], dtype=str),
        rubric_id=rubric.rubric_id,
        encoder=encoder,
        sidecar={
            **identity,
            "rubric_id": rubric.rubric_id,
            "rubric_hash": rubric.content_hash(),
            "bound_with": "for_rubric",
            "n_references": len(labels),
        },
    )


def evidence_frame(rows: Sequence[dict[str, Any]]) -> pd.DataFrame:
    """Build an evidence table from partial rows; missing fields get plain defaults."""
    defaults: dict[str, Any] = {
        "run_number": 1,
        "model_name": "fake",
        "temperature": 0.0,
        "quote_index": 0,
        "verified": 1,
        "match_score": 100.0,
        "char_start": 0,
        "char_end": 1,
        "duplicate_of": pd.NA,
        "used_in_stage2": 1,
    }
    full = []
    for row in rows:
        merged = {**defaults, **row}
        merged.setdefault("quote_raw", merged["quote_norm"])
        full.append(merged)
    frame = pd.DataFrame(full, columns=list(EVIDENCE_COLUMNS))
    frame["char_start"] = frame["char_start"].astype("Int64")
    frame["char_end"] = frame["char_end"].astype("Int64")
    frame["duplicate_of"] = frame["duplicate_of"].astype("Int64")
    return frame
