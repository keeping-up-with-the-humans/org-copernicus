"""Helpers of the command-line tests of the reference, embedding, and scoring commands."""

from __future__ import annotations

import os
from pathlib import Path

from similarity_support import make_references, make_texts, texts_along_first_axis

from copernicus.rubric.loader import load_rubric, toy_rubric_path

if not os.environ.get("COPERNICUS_TEST_DOWNLOAD"):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

TOY_CALLS = 24
TOY_TEXTS_PER_CELL = 10
TEST_ENCODER = "test-encoder"


def references_args(out: Path, *extra: str) -> list[str]:
    """Build the arguments of ``references generate`` on the toy rubric with the fake backend."""
    return [
        *("references", "generate", "--rubric", str(toy_rubric_path())),
        *("--out", str(out), "--model", "writer-a", "--backend", "fake", *extra),
    ]


def save_hand_built_sets(directory: Path, *, style: str = "neutral") -> tuple[Path, Path]:
    """Save numpy-only text embeddings and reference embeddings; return their stems."""
    rubric = load_rubric(toy_rubric_path())
    texts_stem = directory / "texts_embeddings"
    references_stem = directory / "references_embeddings"
    make_texts(texts_along_first_axis(4)).save(texts_stem)
    make_references(rubric, style=style).save(references_stem)
    return texts_stem, references_stem
