"""Tests that texts and references are scored together only when one encoder made both."""

from __future__ import annotations

from dataclasses import replace

import pytest
from similarity_support import (
    ENCODER,
    SIDECAR,
    make_references,
    make_texts,
    texts_along_first_axis,
)

from copernicus.embeddings import EmbeddingSet
from copernicus.errors import ConfigurationError
from copernicus.rubric.model import Rubric
from copernicus.similarity import (
    ReferenceEmbeddings,
    SimilaritySettings,
    run_grid,
    score_similarity,
)

CHANGED = {
    "hf_id": "other/model",
    "revision": "0123abcd",
    "pooling": "cls",
    "max_tokens": 64,
    "dtype": "bfloat16",
    "trust_remote_code": True,
    "padding_side": "left",
    "special_tokens_reserved": 0,
}


@pytest.mark.parametrize("key", sorted(CHANGED))
def test_texts_and_references_that_differ_in_an_identity_key_are_refused(
    toy_rubric: Rubric, key: str
) -> None:
    texts = make_texts(texts_along_first_axis(3))
    other = replace(texts, sidecar={**SIDECAR, key: CHANGED[key]})
    with pytest.raises(ConfigurationError, match=f"different {key}|{key} \\("):
        score_similarity(other, make_references(toy_rubric), toy_rubric, SimilaritySettings())


def with_versions(
    toy_rubric: Rubric, texts_versions: dict[str, str], refs_versions: dict[str, str]
) -> tuple[EmbeddingSet, ReferenceEmbeddings]:
    """Return texts and references whose sidecars carry the given version keys."""
    texts = make_texts(texts_along_first_axis(3))
    refs = make_references(toy_rubric)
    return (
        replace(texts, sidecar={**SIDECAR, **texts_versions}),
        replace(refs, sidecar={**refs.sidecar, **refs_versions}),
    )


def test_versions_of_torch_and_transformers_do_not_block_scoring(toy_rubric: Rubric) -> None:
    texts, refs = with_versions(
        toy_rubric,
        {"torch_version": "1.0", "transformers_version": "4.0", "resolved_revision": "a"},
        {"torch_version": "2.0", "transformers_version": "5.0", "resolved_revision": "a"},
    )
    assert len(score_similarity(texts, refs, toy_rubric, SimilaritySettings())) > 0


def test_texts_and_references_loaded_from_different_commits_are_refused_naming_both(
    toy_rubric: Rubric,
) -> None:
    texts, refs = with_versions(toy_rubric, {"resolved_revision": "a"}, {"resolved_revision": "b"})
    with pytest.raises(ConfigurationError, match="different commits .*'a' and 'b'"):
        score_similarity(texts, refs, toy_rubric, SimilaritySettings())


@pytest.mark.parametrize("unknown", ["texts", "references"])
def test_a_commit_recorded_as_unknown_on_one_side_does_not_block_scoring(
    toy_rubric: Rubric, unknown: str
) -> None:
    commits = {"texts": "a", "references": "b"} | {unknown: "unknown"}
    texts, refs = with_versions(
        toy_rubric,
        {"resolved_revision": commits["texts"]},
        {"resolved_revision": commits["references"]},
    )
    assert len(score_similarity(texts, refs, toy_rubric, SimilaritySettings())) > 0


def test_a_sidecar_without_the_identity_cannot_be_scored(toy_rubric: Rubric) -> None:
    texts = make_texts(texts_along_first_axis(3))
    bare = replace(texts, sidecar={"hf_id": SIDECAR["hf_id"]})
    with pytest.raises(ConfigurationError, match="lacks"):
        score_similarity(bare, make_references(toy_rubric), toy_rubric, SimilaritySettings())


def test_run_grid_compares_the_identity_of_each_key(toy_rubric: Rubric) -> None:
    texts = make_texts(texts_along_first_axis(3))
    other = replace(texts, sidecar={**SIDECAR, "pooling": "cls"})
    inputs = {(ENCODER, "neutral"): (other, make_references(toy_rubric))}
    with pytest.raises(ConfigurationError, match="pooling"):
        run_grid(inputs, toy_rubric, [SimilaritySettings()])
