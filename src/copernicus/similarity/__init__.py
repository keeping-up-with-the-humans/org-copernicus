"""Similarity coding: reference embeddings and the scoring of texts against their poles."""

from copernicus.similarity.grid import PAPER_GRID, PAPER_KEYS, run_grid
from copernicus.similarity.poles import per_reference_similarity, pool_writers
from copernicus.similarity.provenance import scoring_provenance
from copernicus.similarity.reduce import REDUCTIONS, reduce_matrix
from copernicus.similarity.references import ReferenceEmbeddings, embed_references
from copernicus.similarity.scorer import (
    rollup_scores,
    score_similarity,
    score_similarity_many,
)
from copernicus.similarity.settings import (
    LONG_TEXT,
    POLARITIES,
    TARGET_LEVELS,
    WRITER_POOLINGS,
    SimilaritySettings,
)

__all__ = [
    "LONG_TEXT",
    "PAPER_GRID",
    "PAPER_KEYS",
    "POLARITIES",
    "REDUCTIONS",
    "TARGET_LEVELS",
    "WRITER_POOLINGS",
    "ReferenceEmbeddings",
    "SimilaritySettings",
    "embed_references",
    "per_reference_similarity",
    "pool_writers",
    "reduce_matrix",
    "rollup_scores",
    "run_grid",
    "score_similarity",
    "score_similarity_many",
    "scoring_provenance",
]
