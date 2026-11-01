"""Tests of the similarity coder: per-reference cosines, writer pooling, and scoring."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr
from similarity_support import (
    ENCODER,
    make_references,
    make_texts,
    texts_along_first_axis,
    unit,
    without_pole,
)

from copernicus.errors import ConfigurationError
from copernicus.rubric.model import Rubric
from copernicus.similarity import ReferenceEmbeddings
from copernicus.similarity.poles import PoleScorer, per_reference_similarity, pool_writers
from copernicus.similarity.scorer import score_similarity, score_similarity_many
from copernicus.similarity.settings import SimilaritySettings

COLUMNS = [
    "text_id",
    "target_level",
    "target_id",
    "polarity",
    "reduction",
    "writer_pooling",
    "long_text",
    "style",
    "encoder",
    "score",
    "n_references",
]


def two_texts_with_chunks():
    """Text 0 has one chunk; text 1 has three chunks in a three-dimensional space."""
    chunks = [
        np.array([[1.0, 0.0, 0.0]]),
        np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
    ]
    return make_texts(np.array([[1.0, 0.0, 0.0], [1.0, 2.0, 1.0]]), chunks)


REFERENCES = unit(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))


def test_max_chunk_takes_the_best_chunk_per_reference() -> None:
    sims = per_reference_similarity(two_texts_with_chunks(), REFERENCES, "max_chunk")
    root = np.sqrt(0.5)
    np.testing.assert_allclose(sims, [[1.0, 0.0], [root, 1.0]], atol=1e-6)


def test_mean_chunk_averages_the_chunk_cosines() -> None:
    sims = per_reference_similarity(two_texts_with_chunks(), REFERENCES, "mean_chunk")
    root = np.sqrt(0.5)
    np.testing.assert_allclose(sims, [[1.0, 0.0], [root / 3, (root + 1) / 3]], atol=1e-6)


def test_first_window_takes_the_first_chunk() -> None:
    sims = per_reference_similarity(two_texts_with_chunks(), REFERENCES, "first_window")
    root = np.sqrt(0.5)
    np.testing.assert_allclose(sims, [[1.0, 0.0], [root, root]], atol=1e-6)


def test_pooled_embedding_uses_the_text_vector() -> None:
    texts = two_texts_with_chunks()
    sims = per_reference_similarity(texts, REFERENCES, "pooled_embedding")
    np.testing.assert_allclose(sims, texts.vectors @ REFERENCES.T, atol=1e-6)
    assert sims.shape == (2, 2)
    assert sims.dtype == np.float64


def test_per_reference_similarity_refuses_other_dimensions_and_modes() -> None:
    texts = two_texts_with_chunks()
    with pytest.raises(ConfigurationError, match="dimension"):
        per_reference_similarity(texts, np.ones((2, 5), dtype=np.float32), "max_chunk")
    with pytest.raises(ConfigurationError, match="long_text.*'bogus'"):
        per_reference_similarity(texts, REFERENCES, "bogus")


def test_pool_writers_mean_and_max() -> None:
    first, second = np.array([0.1, 0.9]), np.array([0.3, 0.5])
    np.testing.assert_allclose(pool_writers([first, second], "mean"), [0.2, 0.7])
    np.testing.assert_allclose(pool_writers([first, second], "max"), [0.3, 0.9])
    np.testing.assert_allclose(pool_writers([first], "max"), first)


def test_pool_writers_refuses_nothing_and_unknown_modes() -> None:
    with pytest.raises(ConfigurationError, match="no writer"):
        pool_writers([], "mean")
    with pytest.raises(ConfigurationError, match="writer_pooling.*'median'"):
        pool_writers([np.zeros(2)], "median")


def test_output_columns_and_row_counts(toy_rubric: Rubric) -> None:
    texts = make_texts(texts_along_first_axis(5))
    refs = make_references(toy_rubric, style="register")
    frame = score_similarity(texts, refs, toy_rubric, SimilaritySettings())
    assert list(frame.columns) == COLUMNS
    # construct plus three attributes, five texts each, one polarity row set
    assert len(frame) == 4 * 5
    assert set(frame["target_id"]) == {"construct", "specificity", "tone", "reasoning"}
    assert set(frame["style"]) == {"register"}
    assert set(frame["encoder"]) == {ENCODER}
    assert set(frame["target_level"]) == {"construct"}
    assert set(frame["polarity"]) == {"net"}
    assert (frame["reduction"] == "pca").all()
    assert (frame["writer_pooling"] == "mean").all()
    assert (frame["long_text"] == "max_chunk").all()
    assert list(frame.loc[frame["target_id"] == "tone", "text_id"]) == [f"t{i}" for i in range(5)]
    assert np.isfinite(frame["score"]).all()


@pytest.mark.parametrize("reduction", ["mean", "max", "pca"])
def test_net_scores_are_ordered_like_the_first_coordinate(
    toy_rubric: Rubric, reduction: str
) -> None:
    raw = texts_along_first_axis(7)
    texts = make_texts(raw)
    refs = make_references(toy_rubric)
    frame = score_similarity(texts, refs, toy_rubric, SimilaritySettings(reduction=reduction))
    for target in ("construct", "tone"):
        scores = frame.loc[frame["target_id"] == target, "score"].to_numpy()
        assert spearmanr(scores, raw[:, 0]).statistic > 0.99


def test_n_references_is_the_pole_size(toy_rubric: Rubric) -> None:
    texts = make_texts(texts_along_first_axis(4))
    refs = make_references(toy_rubric, writers=("a", "b"), per_pole=3)
    by_polarity = {
        mode: score_similarity(texts, refs, toy_rubric, SimilaritySettings(polarity=mode))
        for mode in ("net", "positive", "negative")
    }
    assert (by_polarity["net"]["n_references"] == 12).all()
    assert (by_polarity["positive"]["n_references"] == 6).all()
    assert (by_polarity["negative"]["n_references"] == 6).all()
    for mode, frame in by_polarity.items():
        assert (frame["polarity"] == mode).all()


def test_net_is_positive_minus_negative(toy_rubric: Rubric) -> None:
    texts = make_texts(texts_along_first_axis(4))
    refs = make_references(toy_rubric)
    settings = SimilaritySettings(reduction="mean")
    scores = {
        mode: score_similarity(texts, refs, toy_rubric, replace(settings, polarity=mode))[
            "score"
        ].to_numpy()
        for mode in ("net", "positive", "negative")
    }
    np.testing.assert_allclose(scores["net"], scores["positive"] - scores["negative"])


def test_writers_are_reduced_separately_then_pooled(toy_rubric: Rubric) -> None:
    texts = make_texts(texts_along_first_axis(4))
    refs = make_references(toy_rubric, writers=("a", "b"))
    combined = {
        pooling: score_similarity(
            texts,
            refs,
            toy_rubric,
            SimilaritySettings(polarity="positive", reduction="mean", writer_pooling=pooling),
        )
        for pooling in ("mean", "max")
    }
    alone = {
        writer: score_similarity(
            texts,
            refs.subset_writers([writer]),
            toy_rubric,
            SimilaritySettings(polarity="positive", reduction="mean"),
        )["score"].to_numpy()
        for writer in ("a", "b")
    }
    np.testing.assert_allclose(combined["mean"]["score"], (alone["a"] + alone["b"]) / 2, atol=1e-12)
    np.testing.assert_allclose(
        combined["max"]["score"], np.maximum(alone["a"], alone["b"]), atol=1e-12
    )


def test_attribute_level_adds_the_roll_up(toy_rubric: Rubric) -> None:
    texts = make_texts(texts_along_first_axis(5))
    refs = make_references(toy_rubric)
    for reduction in ("mean", "max", "pca"):
        settings = SimilaritySettings(reduction=reduction, target_level="attribute")
        frame = score_similarity(texts, refs, toy_rubric, settings)
        assert set(frame["target_level"]) == {"attribute"}
        assert set(frame["target_id"]) == {
            "specificity",
            "tone",
            "reasoning",
            f"{toy_rubric.rubric_id}__rollup",
        }
        rollup = frame[frame["target_id"] == f"{toy_rubric.rubric_id}__rollup"]
        assert len(rollup) == 5
        assert (rollup["n_references"] == 3).all()
        nets = np.column_stack(
            [
                frame.loc[frame["target_id"] == attribute.attribute_id, "score"].to_numpy()
                for attribute in toy_rubric.attributes
            ]
        )
        expected = {
            "mean": nets.mean(axis=1),
            "max": nets.max(axis=1),
        }
        if reduction in expected:
            np.testing.assert_allclose(rollup["score"], expected[reduction])
        else:
            assert np.corrcoef(rollup["score"], nets.mean(axis=1))[0, 1] > 0


def test_construct_level_scores_the_construct_directly_and_keeps_the_attributes(
    toy_rubric: Rubric,
) -> None:
    texts = make_texts(texts_along_first_axis(4))
    refs = make_references(toy_rubric)
    frame = score_similarity(texts, refs, toy_rubric, SimilaritySettings())
    assert "construct" in set(frame["target_id"])
    assert not frame["target_id"].str.endswith("__rollup").any()
    attribute = score_similarity(
        texts, refs, toy_rubric, SimilaritySettings(target_level="attribute")
    )
    assert "construct" not in set(attribute["target_id"])


def test_roll_up_of_other_polarities_rolls_up_that_polarity(toy_rubric: Rubric) -> None:
    texts = make_texts(texts_along_first_axis(4))
    refs = make_references(toy_rubric)
    settings = SimilaritySettings(polarity="positive", reduction="mean", target_level="attribute")
    frame = score_similarity(texts, refs, toy_rubric, settings)
    rollup = frame[frame["target_id"].str.endswith("__rollup")]
    assert (rollup["polarity"] == "positive").all()
    assert (rollup["n_references"] == 3).all()
    parts = np.column_stack(
        [
            frame.loc[frame["target_id"] == attribute.attribute_id, "score"].to_numpy()
            for attribute in toy_rubric.attributes
        ]
    )
    np.testing.assert_allclose(rollup["score"], parts.mean(axis=1))


def test_long_text_modes_differ_for_chunked_texts(toy_rubric: Rubric) -> None:
    rng = np.random.default_rng(3)
    chunks = [rng.normal(size=(n, 8)) + np.eye(8)[0] * s for n, s in ((3, 2.0), (1, -2.0), (2, 0))]
    texts = make_texts(np.vstack([rows.mean(axis=0) for rows in chunks]), chunks)
    refs = make_references(toy_rubric)
    frames = {
        mode: score_similarity(texts, refs, toy_rubric, SimilaritySettings(long_text=mode))
        for mode in ("max_chunk", "mean_chunk", "pooled_embedding", "first_window")
    }
    assert len({tuple(np.round(f["score"], 8)) for f in frames.values()}) == 4
    for mode, frame in frames.items():
        assert (frame["long_text"] == mode).all()


def test_restricting_targets_scores_only_those_and_skips_other_completeness(
    toy_rubric: Rubric,
) -> None:
    texts = make_texts(texts_along_first_axis(4))
    refs = without_pole(
        without_pole(make_references(toy_rubric), "tone", "negative"), "construct", "positive"
    )
    frame = score_similarity(texts, refs, toy_rubric, SimilaritySettings(), targets=["reasoning"])
    assert set(frame["target_id"]) == {"reasoning"}
    assert len(frame) == 4
    full = score_similarity(texts, make_references(toy_rubric), toy_rubric, SimilaritySettings())
    np.testing.assert_allclose(
        frame["score"], full.loc[full["target_id"] == "reasoning", "score"], atol=1e-12
    )
    with pytest.raises(ConfigurationError, match="negative pole of target tone"):
        score_similarity(texts, refs, toy_rubric, SimilaritySettings(), targets=["tone"])


def test_restricted_attribute_level_has_no_roll_up(toy_rubric: Rubric) -> None:
    texts = make_texts(texts_along_first_axis(4))
    refs = make_references(toy_rubric)
    frame = score_similarity(
        texts,
        refs,
        toy_rubric,
        SimilaritySettings(target_level="attribute"),
        targets=["tone", "specificity"],
    )
    assert list(dict.fromkeys(frame["target_id"])) == ["specificity", "tone"]


@pytest.mark.parametrize("targets", [["nonsense"], ["construct"], []])
def test_bad_targets_raise(toy_rubric: Rubric, targets: list[str]) -> None:
    texts = make_texts(texts_along_first_axis(3))
    with pytest.raises(ConfigurationError, match="target"):
        score_similarity(
            texts, make_references(toy_rubric), toy_rubric, SimilaritySettings(), targets=targets
        )


def test_encoder_mismatch_raises(toy_rubric: Rubric) -> None:
    texts = make_texts(texts_along_first_axis(3), encoder="other-encoder")
    with pytest.raises(ConfigurationError, match="encoder"):
        score_similarity(texts, make_references(toy_rubric), toy_rubric, SimilaritySettings())


def test_dimension_mismatch_raises(toy_rubric: Rubric) -> None:
    texts = make_texts(np.random.default_rng(0).normal(size=(3, 5)))
    with pytest.raises(ConfigurationError, match="dimension"):
        score_similarity(texts, make_references(toy_rubric), toy_rubric, SimilaritySettings())


def test_other_rubric_raises(toy_rubric: Rubric) -> None:
    refs = make_references(toy_rubric)
    other = ReferenceEmbeddings(**{**refs.__dict__, "rubric_id": "another_rubric"})
    with pytest.raises(ConfigurationError, match="another_rubric"):
        score_similarity(
            make_texts(texts_along_first_axis(3)), other, toy_rubric, SimilaritySettings()
        )


def test_missing_pole_raises_naming_it(toy_rubric: Rubric) -> None:
    refs = without_pole(make_references(toy_rubric), "tone", "negative")
    with pytest.raises(ConfigurationError, match="negative pole of target tone"):
        score_similarity(
            make_texts(texts_along_first_axis(3)), refs, toy_rubric, SimilaritySettings()
        )


def test_pole_scorer_refuses_an_empty_pole(toy_rubric: Rubric) -> None:
    refs = without_pole(make_references(toy_rubric), "tone", "negative")
    scorer = PoleScorer(make_texts(texts_along_first_axis(3)), refs, toy_rubric.rubric_id)
    with pytest.raises(ConfigurationError, match="negative pole of target tone"):
        scorer.pole_score("tone", "negative", SimilaritySettings())


def test_references_mixing_styles_raise(toy_rubric: Rubric) -> None:
    refs = make_references(toy_rubric, writers=("a", "b"), style=["neutral", "register"])
    with pytest.raises(ConfigurationError, match="style"):
        score_similarity(
            make_texts(texts_along_first_axis(3)), refs, toy_rubric, SimilaritySettings()
        )


def test_pca_over_a_single_text_raises_with_the_label(toy_rubric: Rubric) -> None:
    texts = make_texts(texts_along_first_axis(1))
    with pytest.raises(ConfigurationError, match="construct"):
        score_similarity(texts, make_references(toy_rubric), toy_rubric, SimilaritySettings())


def test_one_reference_poles_return_that_cosine(toy_rubric: Rubric) -> None:
    raw = texts_along_first_axis(4)
    texts = make_texts(raw)
    refs = make_references(toy_rubric, writers=("a",), per_pole=1)
    frame = score_similarity(texts, refs, toy_rubric, SimilaritySettings(polarity="positive"))
    mask = (refs.target == "tone") & (refs.polarity == "positive")
    expected = texts.vectors.astype(np.float64) @ refs.vectors[mask][0].astype(np.float64)
    np.testing.assert_allclose(frame.loc[frame["target_id"] == "tone", "score"], expected)


def test_scoring_does_not_change_its_inputs(toy_rubric: Rubric) -> None:
    texts = make_texts(texts_along_first_axis(4))
    refs = make_references(toy_rubric)
    before = (texts.vectors.copy(), refs.vectors.copy())
    score_similarity(texts, refs, toy_rubric, SimilaritySettings())
    np.testing.assert_array_equal(texts.vectors, before[0])
    np.testing.assert_array_equal(refs.vectors, before[1])


def test_many_concatenates_in_order_and_matches_single_calls(toy_rubric: Rubric) -> None:
    texts = make_texts(texts_along_first_axis(4))
    refs = make_references(toy_rubric)
    settings = [
        SimilaritySettings(polarity="positive", reduction="mean"),
        SimilaritySettings(reduction="max", target_level="attribute"),
    ]
    many = score_similarity_many(texts, refs, toy_rubric, settings)
    single = pd.concat(
        [score_similarity(texts, refs, toy_rubric, each) for each in settings], ignore_index=True
    )
    pd.testing.assert_frame_equal(many, single)
    with pytest.raises(ConfigurationError, match="settings"):
        score_similarity_many(texts, refs, toy_rubric, [])
