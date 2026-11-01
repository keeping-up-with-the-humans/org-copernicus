"""Torch-free helpers of the embedding tests."""

from __future__ import annotations

import importlib.util
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
import pandas as pd
import pytest

from copernicus.embeddings import EmbeddingSet

MINILM_DIRECTORY = "models--sentence-transformers--all-MiniLM-L6-v2"
MINILM_FILES = ("config.json", "tokenizer.json", "model.safetensors")
TORCH_INSTALLED = importlib.util.find_spec("torch") is not None
TORCH_SKIP_REASON = (
    "torch is not installed; install the embed extra: "
    'pip install "org-copernicus[embed]", or pip install ".[embed]" from the package\'s root folder'
)
WEIGHTS_SKIP_REASON = "MiniLM weights not cached; set COPERNICUS_TEST_DOWNLOAD=1 to fetch"
TestFunction = TypeVar("TestFunction", bound=Callable[..., Any])
TOY_IDENTITY = {
    "hf_id": "org/toy",
    "revision": "unpinned",
    "pooling": "mean",
    "max_tokens": 6,
    "dtype": None,
    "trust_remote_code": False,
    "padding_side": "right",
    "special_tokens_reserved": 2,
}
TOY_VERSIONS = {
    "resolved_revision": "0123abc",
    "torch_version": "9.9.9",
    "transformers_version": "8.8.8",
}


def hub_cache() -> Path:
    """Return the folder of the Hugging Face hub cache."""
    if "HF_HUB_CACHE" in os.environ:
        return Path(os.environ["HF_HUB_CACHE"])
    home = (
        Path(os.environ["HF_HOME"])
        if "HF_HOME" in os.environ
        else Path.home() / ".cache" / "huggingface"
    )
    return home / "hub"


def minilm_cached() -> bool:
    """Tell whether one cached snapshot of MiniLM holds its config, tokenizer and weights."""
    snapshots = hub_cache() / MINILM_DIRECTORY / "snapshots"
    if not snapshots.is_dir():
        return False
    return any(
        all((snapshot / name).is_file() for name in MINILM_FILES)
        for snapshot in snapshots.iterdir()
    )


MINILM_AVAILABLE = TORCH_INSTALLED and (
    minilm_cached() or bool(os.environ.get("COPERNICUS_TEST_DOWNLOAD"))
)
# Why the encoder cannot run here: torch comes first, since the weights are of no use without it.
SKIP_REASON = TORCH_SKIP_REASON if not TORCH_INSTALLED else WEIGHTS_SKIP_REASON


def needs_minilm(test: TestFunction) -> TestFunction:
    """Mark a test as one that runs the MiniLM encoder, and skip it when the weights are absent.

    The ``embed`` marker keeps the test out of the CI selection, which installs no torch; the
    skip covers a local run where the weights are not cached.
    """
    skipped = pytest.mark.skipif(not MINILM_AVAILABLE, reason=SKIP_REASON)(test)
    return pytest.mark.embed(skipped)


class WordTokenizer:
    """A tokenizer with one token per whitespace-separated word.

    ``__call__`` wraps every text in the ids 1 and 2 like a sentence encoder does, and
    pads on the right with id 0; words get the ids 3, 4, ... in order of first sight.
    """

    def __init__(self) -> None:
        self.vocabulary: dict[str, int] = {}
        self.words: list[str] = []
        self.padding_side = "right"

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        assert add_special_tokens is False
        for word in text.split():
            if word not in self.vocabulary:
                self.vocabulary[word] = len(self.vocabulary) + 3
                self.words.append(word)
        return [self.vocabulary[word] for word in text.split()]

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        assert skip_special_tokens is True
        return " ".join(self.words[token - 3] for token in ids)


def small_set(
    n_texts: int = 3,
    chunks_per_text: tuple[int, ...] = (1, 2, 1),
    *,
    encoder: str = "toy",
    with_metadata: bool = True,
) -> EmbeddingSet:
    """Build a consistent embedding set by hand: 4-dimensional, deterministic."""
    assert len(chunks_per_text) == n_texts
    generator = np.random.default_rng(7)
    owner = np.repeat(np.arange(n_texts), chunks_per_text).astype(np.int64)
    chunk_vectors = generator.normal(size=(len(owner), 4)).astype(np.float32)
    chunk_vectors /= np.linalg.norm(chunk_vectors, axis=1, keepdims=True)
    vectors = generator.normal(size=(n_texts, 4)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    metadata: Any = (
        pd.DataFrame(
            {"text_id": [f"t{i}" for i in range(n_texts)], "group_id": list("abc")[:n_texts]}
        )
        if with_metadata
        else None
    )
    return EmbeddingSet(
        ids=np.asarray([f"t{i}" for i in range(n_texts)]),
        vectors=vectors,
        chunk_vectors=chunk_vectors,
        chunk_owner=owner,
        chunk_weight=np.arange(1, len(owner) + 1, dtype=np.float64),
        encoder=encoder,
        metadata=metadata,
        sidecar={**TOY_IDENTITY, **TOY_VERSIONS},
    )
