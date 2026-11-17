"""Tests that the pole scorer compares the two loads it receives before it scores anything."""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any

import numpy as np
import pytest
from similarity_support import VERSIONS, make_references, make_texts, texts_along_first_axis

from copernicus.embeddings import EmbeddingSet
from copernicus.errors import ConfigurationError
from copernicus.rubric.model import Rubric
from copernicus.similarity import (
    ReferenceEmbeddings,
    SimilaritySettings,
    run_grid,
    score_similarity,
)
from copernicus.similarity.poles import PoleScorer

LOGGER = "copernicus.similarity.poles"
EXACT = SimilaritySettings(reduction="mean", writer_pooling="pooled", long_text="pooled_embedding")
CHANGED = {"hf_id": "other/model", "pooling": "cls", "trust_remote_code": True, "dtype": None}
UNKNOWN = {"torch_version": "unknown", "transformers_version": "unknown"}


def pair(
    rubric: Rubric, references_changes: dict[str, Any], *, bare_references: bool = False
) -> tuple[EmbeddingSet, ReferenceEmbeddings]:
    """Return texts and references of one load, with some keys of the references changed."""
    texts, refs = make_texts(texts_along_first_axis(5)), make_references(rubric)
    sidecar = {**refs.sidecar, **references_changes}
    if bare_references:
        sidecar = {key: value for key, value in sidecar.items() if key not in UNKNOWN}
    return texts, replace(refs, sidecar=sidecar)


@pytest.mark.parametrize("key", sorted(CHANGED))
def test_a_pole_scorer_refuses_references_that_differ_in_an_identity_key(
    toy_rubric: Rubric, key: str
) -> None:
    texts, refs = pair(toy_rubric, {key: CHANGED[key]})
    with pytest.raises(ConfigurationError) as refused:
        PoleScorer(texts, refs, toy_rubric.rubric_id)
    message = str(refused.value)
    assert f"{key} (" in message
    assert message.endswith("embed both sides with one encoder specification")


def test_a_pole_scorer_refuses_references_of_another_major_version(toy_rubric: Rubric) -> None:
    texts, refs = pair(toy_rubric, {"transformers_version": "5.12.1"})
    with pytest.raises(ConfigurationError) as refused:
        PoleScorer(texts, refs, toy_rubric.rubric_id)
    assert "transformers 4.57.6 and 5.12.1, two major versions" in str(refused.value)


@pytest.mark.parametrize(
    "changes", [{"transformers_version": "4.57.3"}, {"torch_version": "2.4.1"}], ids=str
)
def test_a_pole_scorer_scores_a_smaller_difference_with_one_warning_and_keeps_the_comparison(
    toy_rubric: Rubric, caplog: pytest.LogCaptureFixture, changes: dict[str, str]
) -> None:
    texts, refs = pair(toy_rubric, changes)
    with caplog.at_level(logging.WARNING):
        scorer = PoleScorer(texts, refs, toy_rubric.rubric_id)
        score, count = scorer.polarity_score("construct", "net", EXACT)
    (record,) = caplog.records
    assert (record.name, record.levelno) == (LOGGER, logging.WARNING)
    assert scorer.load_comparison["warnings"] == [record.getMessage()]
    assert scorer.load_comparison["texts"] == VERSIONS
    assert scorer.load_comparison["references"] == {**VERSIONS, **changes}
    one_load = PoleScorer(*pair(toy_rubric, {}), toy_rubric.rubric_id)
    assert one_load.load_comparison["warnings"] == []
    expected, expected_count = one_load.polarity_score("construct", "net", EXACT)
    np.testing.assert_array_equal(score, expected)
    assert count == expected_count


def test_a_pole_scorer_scores_a_side_without_versions_with_one_warning_and_records_unknown(
    toy_rubric: Rubric, caplog: pytest.LogCaptureFixture
) -> None:
    texts, refs = pair(toy_rubric, {}, bare_references=True)
    with caplog.at_level(logging.WARNING):
        scorer = PoleScorer(texts, refs, toy_rubric.rubric_id)
        scorer.polarity_score("construct", "net", EXACT)
    (record,) = caplog.records
    assert record.name == LOGGER and "could not be compared" in record.getMessage()
    assert scorer.load_comparison["references"] == UNKNOWN
    assert scorer.load_comparison["warnings"] == [record.getMessage()]


def test_a_scoring_call_and_a_grid_key_log_each_warning_of_the_comparison_once(
    toy_rubric: Rubric, caplog: pytest.LogCaptureFixture
) -> None:
    texts, refs = pair(toy_rubric, {"transformers_version": "4.57.3"})
    with caplog.at_level(logging.WARNING):
        score_similarity(texts, refs, toy_rubric, SimilaritySettings())
    assert [record.name for record in caplog.records] == [LOGGER]
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        run_grid({("test-encoder", "neutral"): (texts, refs)}, toy_rubric, [EXACT, EXACT])
    assert [record.name for record in caplog.records] == [LOGGER]
