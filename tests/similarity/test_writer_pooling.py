"""Tests of the writer pooling: per-writer reductions pooled, or one reduction over a pole."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from similarity_support import make_references, make_texts, texts_along_first_axis

from copernicus.errors import ConfigurationError
from copernicus.rubric.model import Rubric
from copernicus.similarity import (
    REDUCTIONS,
    WRITER_POOLINGS,
    ReferenceEmbeddings,
    SimilaritySettings,
    per_reference_similarity,
    pool_writers,
    reduce_matrix,
    score_similarity,
)

LABELS = ("target", "polarity", "level", "style", "writer_model")


def combined(*parts: ReferenceEmbeddings) -> ReferenceEmbeddings:
    """Stack reference embeddings of the same rubric and encoder into one set."""
    first = parts[0]
    labels = {name: np.concatenate([getattr(part, name) for part in parts]) for name in LABELS}
    vectors = np.vstack([part.vectors for part in parts])
    return ReferenceEmbeddings(
        vectors=vectors,
        rubric_id=first.rubric_id,
        encoder=first.encoder,
        sidecar={**first.sidecar, "n_references": len(vectors)},
        **labels,
    )


def unequal_references(rubric: Rubric) -> ReferenceEmbeddings:
    """Two writers with two and five references per pole."""
    return combined(
        make_references(rubric, writers=("a",), per_pole=2, seed=1),
        make_references(rubric, writers=("b",), per_pole=5, seed=2),
    )


def pole_score(
    frame_settings: SimilaritySettings, refs: ReferenceEmbeddings, rubric: Rubric
) -> tuple[np.ndarray, pd.DataFrame]:
    """Score six texts and return the scores of the positive pole of ``tone``."""
    texts = make_texts(texts_along_first_axis(6))
    frame = score_similarity(texts, refs, rubric, frame_settings)
    return frame.loc[frame["target_id"] == "tone", "score"].to_numpy(), frame


def test_pooled_is_an_allowed_writer_pooling_with_its_own_label() -> None:
    assert WRITER_POOLINGS == ("mean", "max", "pooled")
    settings = SimilaritySettings(writer_pooling="pooled")
    assert settings.label() == "net-pca-pooled-max_chunk-construct"


@pytest.mark.parametrize("reduction", REDUCTIONS)
def test_with_one_writer_pooled_equals_the_per_writer_path(
    toy_rubric: Rubric, reduction: str
) -> None:
    refs = make_references(toy_rubric, writers=("a",), per_pole=4)
    scores = {
        pooling: pole_score(
            SimilaritySettings(polarity="positive", reduction=reduction, writer_pooling=pooling),
            refs,
            toy_rubric,
        )[0]
        for pooling in WRITER_POOLINGS
    }
    np.testing.assert_allclose(scores["pooled"], scores["mean"], atol=1e-12)
    np.testing.assert_allclose(scores["pooled"], scores["max"], atol=1e-12)


def test_with_unequal_counts_per_writer_pooling_differs_from_one_reduction_over_the_pole(
    toy_rubric: Rubric,
) -> None:
    refs = unequal_references(toy_rubric)
    texts = make_texts(texts_along_first_axis(6))
    rows = refs.pole_indices("tone", "positive")
    cosines = per_reference_similarity(texts, refs.vectors[rows], "max_chunk")
    writer = refs.writer_model[rows]
    by_writer = {name: cosines[:, writer == name].mean(axis=1) for name in ("a", "b")}
    settings = {
        pooling: SimilaritySettings(polarity="positive", reduction="mean", writer_pooling=pooling)
        for pooling in WRITER_POOLINGS
    }
    mean, _ = pole_score(settings["mean"], refs, toy_rubric)
    maximum, _ = pole_score(settings["max"], refs, toy_rubric)
    pooled, frame = pole_score(settings["pooled"], refs, toy_rubric)
    np.testing.assert_allclose(mean, (by_writer["a"] + by_writer["b"]) / 2, atol=1e-12)
    np.testing.assert_allclose(maximum, np.maximum(by_writer["a"], by_writer["b"]), atol=1e-12)
    np.testing.assert_allclose(pooled, cosines.mean(axis=1), atol=1e-12)
    assert not np.allclose(pooled, mean, atol=1e-6)
    assert (frame["n_references"] == 7).all()


@pytest.mark.parametrize("reduction", REDUCTIONS)
@pytest.mark.parametrize("long_text", ["max_chunk", "pooled_embedding"])
def test_pooled_equals_a_direct_reduction_over_the_concatenated_pole(
    toy_rubric: Rubric, reduction: str, long_text: str
) -> None:
    refs = unequal_references(toy_rubric)
    texts = make_texts(texts_along_first_axis(6))
    settings = SimilaritySettings(reduction=reduction, writer_pooling="pooled", long_text=long_text)
    frame = score_similarity(texts, refs, toy_rubric, settings)
    for target in ("construct", "tone"):
        poles = [
            reduce_matrix(
                per_reference_similarity(
                    texts, refs.vectors[refs.pole_indices(target, polarity)], long_text
                ),
                reduction,
                "direct",
            )
            for polarity in ("positive", "negative")
        ]
        np.testing.assert_allclose(
            frame.loc[frame["target_id"] == target, "score"], poles[0] - poles[1], atol=1e-12
        )
    assert (frame["n_references"] == 14).all() and set(frame["writer_pooling"]) == {"pooled"}


def test_pool_writers_refuses_the_pooled_mode() -> None:
    with pytest.raises(ConfigurationError, match="pooled"):
        pool_writers([np.zeros(2)], "pooled")


@pytest.mark.parametrize(("pooling", "named"), [("mean", "writer a"), ("pooled", "all writers")])
def test_a_failed_reduction_names_the_pole_and_its_writers_but_no_other_setting(
    toy_rubric: Rubric, pooling: str, named: str
) -> None:
    texts = make_texts(texts_along_first_axis(1))
    refs = make_references(toy_rubric, writers=("a", "b"))
    settings = SimilaritySettings(writer_pooling=pooling, target_level="attribute")
    with pytest.raises(ConfigurationError) as error:
        score_similarity(texts, refs, toy_rubric, settings)
    message = str(error.value)
    assert named in message and ":pca:max_chunk:" in message
    assert "net" not in message and "-attribute" not in message
