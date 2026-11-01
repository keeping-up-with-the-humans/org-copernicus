"""Encoders and embedding sets: texts as chunked, pooled, unit-length vectors.

Importing this package does not import torch or transformers; they are imported when an
`Encoder` loads its model, and are installed by the extra ``embed`` of the
distribution ``org-copernicus``.
"""

from copernicus.embeddings.chunking import split_into_token_chunks
from copernicus.embeddings.encoder import Encoder
from copernicus.embeddings.registry import (
    DEFAULT_ENCODER,
    ENCODERS,
    EncoderSpec,
    resolve_encoder,
)
from copernicus.embeddings.sets import EmbeddingSet

__all__ = [
    "DEFAULT_ENCODER",
    "ENCODERS",
    "EmbeddingSet",
    "Encoder",
    "EncoderSpec",
    "resolve_encoder",
    "split_into_token_chunks",
]
