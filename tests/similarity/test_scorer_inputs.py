"""Tests of the inputs the similarity coder refuses or warns about."""

from __future__ import annotations

import logging

import numpy as np
import pytest
from similarity_support import make_references, make_texts, texts_along_first_axis

from copernicus.errors import ConfigurationError
from copernicus.rubric.model import Rubric
from copernicus.similarity import ReferenceEmbeddings, SimilaritySettings, score_similarity

LOGGER = "copernicus.similarity.scorer"
LABELS = ("target", "polarity", "level", "style", "writer_model")


def without_writer_in_pole(
    refs: ReferenceEmbeddings, writer: str, target: str, polarity: str
) -> ReferenceEmbeddings:
    """Drop the references of one writer from one pole."""
    keep = ~((refs.writer_model == writer) & (refs.target == target) & (refs.polarity == polarity))
    return ReferenceEmbeddings(
        vectors=refs.vectors[keep],
        rubric_id=refs.rubric_id,
        encoder=refs.encoder,
        sidecar=dict(refs.sidecar),
        **{name: getattr(refs, name)[keep] for name in LABELS},
    )


def test_a_bare_string_of_targets_is_refused_instead_of_split_into_letters(
    toy_rubric: Rubric,
) -> None:
    texts = make_texts(texts_along_first_axis(3))
    with pytest.raises(ConfigurationError, match=r"'tone'.*list"):
        score_similarity(
            texts, make_references(toy_rubric), toy_rubric, SimilaritySettings(), targets="tone"
        )


def test_poles_written_by_different_writers_are_scored_with_a_warning(
    toy_rubric: Rubric, caplog: pytest.LogCaptureFixture
) -> None:
    refs = without_writer_in_pole(
        make_references(toy_rubric, writers=("writer-a", "writer-b")),
        "writer-b",
        "construct",
        "negative",
    )
    texts = make_texts(texts_along_first_axis(4))
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        frame = score_similarity(texts, refs, toy_rubric, SimilaritySettings(reduction="mean"))
    warnings = [record.getMessage() for record in caplog.records]
    assert len(warnings) == 1
    assert "target construct" in warnings[0]
    assert "['writer-a', 'writer-b']" in warnings[0] and "['writer-a']" in warnings[0]
    assert np.isfinite(frame["score"]).all()


def test_poles_written_by_the_same_writers_raise_no_warning(
    toy_rubric: Rubric, caplog: pytest.LogCaptureFixture
) -> None:
    texts = make_texts(texts_along_first_axis(4))
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        score_similarity(texts, make_references(toy_rubric), toy_rubric, SimilaritySettings())
    assert caplog.records == []


def test_the_warning_covers_only_the_targets_that_are_scored(
    toy_rubric: Rubric, caplog: pytest.LogCaptureFixture
) -> None:
    refs = without_writer_in_pole(
        make_references(toy_rubric, writers=("writer-a", "writer-b")),
        "writer-a",
        "tone",
        "positive",
    )
    texts = make_texts(texts_along_first_axis(4))
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        score_similarity(texts, refs, toy_rubric, SimilaritySettings(), targets=["reasoning"])
    assert caplog.records == []
