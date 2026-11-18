"""The embeddings of the texts of one attribute, taken from the one pass over distinct texts."""

from __future__ import annotations

import logging
from dataclasses import replace

import numpy as np

from copernicus._digests import without_texts_hash
from copernicus.embeddings import EmbeddingSet

logger = logging.getLogger(__name__)

__all__ = ["gather"]


def gather(unique: EmbeddingSet, rows: np.ndarray, ids: list[str]) -> EmbeddingSet:
    """Return the embeddings of the rows of ``unique``, repeated as often as asked, under new ids.

    The sidecar leaves out the ``texts_hash`` of ``unique``, which names other texts.

    Args:
        unique: Embeddings of distinct texts.
        rows: For each wanted text, the row of its string in ``unique``.
        ids: The ids of the wanted texts; unique.
    """
    by_owner = np.argsort(unique.chunk_owner, kind="stable")
    counts = np.bincount(unique.chunk_owner, minlength=len(unique))
    starts = np.cumsum(counts) - counts
    chunk_rows = np.concatenate([by_owner[starts[row] : starts[row] + counts[row]] for row in rows])
    owner = np.repeat(np.arange(len(rows), dtype=np.int64), counts[rows])
    return replace(
        unique,
        ids=np.asarray(ids),
        vectors=unique.vectors[rows],
        chunk_vectors=unique.chunk_vectors[chunk_rows],
        chunk_owner=owner,
        chunk_weight=unique.chunk_weight[chunk_rows],
        metadata=None,
        sidecar={
            **without_texts_hash(unique.sidecar),
            "n_texts": len(rows),
            "n_chunks": len(chunk_rows),
        },
    )
