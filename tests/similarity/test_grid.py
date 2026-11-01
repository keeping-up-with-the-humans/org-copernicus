"""Tests of the paper's configuration grid and its runner."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from similarity_support import make_references, make_texts, texts_along_first_axis

from copernicus.embeddings import ENCODERS
from copernicus.errors import ConfigurationError
from copernicus.rubric.model import Rubric
from copernicus.similarity.grid import PAPER_GRID, PAPER_KEYS, run_grid
from copernicus.similarity.settings import SimilaritySettings

AXES = ("polarity", "reduction", "writer_pooling", "long_text")


def test_paper_grid_has_every_combination_at_both_target_levels() -> None:
    assert len(PAPER_GRID) == 144
    assert len(set(PAPER_GRID)) == 144
    combinations = {tuple(getattr(each, axis) for axis in AXES) for each in PAPER_GRID}
    assert len(combinations) == 72
    for level in ("construct", "attribute"):
        assert sum(each.target_level == level for each in PAPER_GRID) == 72
    assert (
        len({tuple(getattr(each, axis) for axis in (*AXES, "target_level")) for each in PAPER_GRID})
        == 144
    )
    assert {each.writer_pooling for each in PAPER_GRID} == {"mean", "max"}


def test_run_grid_over_one_key_yields_every_combination(toy_rubric: Rubric) -> None:
    texts = make_texts(texts_along_first_axis(4))
    refs = make_references(toy_rubric, style="register")
    frame = run_grid({("test-encoder", "register"): (texts, refs)}, toy_rubric)
    keys = frame[[*AXES, "target_level"]].drop_duplicates()
    assert len(keys) == 144
    assert set(frame["encoder"]) == {"test-encoder"}
    assert set(frame["style"]) == {"register"}
    assert frame["score"].notna().all()


def test_run_grid_with_two_keys_labels_each(toy_rubric: Rubric) -> None:
    inputs = {
        ("test-encoder", "neutral"): (
            make_texts(texts_along_first_axis(3)),
            make_references(toy_rubric, style="neutral"),
        ),
        ("test-encoder", "register"): (
            make_texts(texts_along_first_axis(3)),
            make_references(toy_rubric, style="register", seed=5),
        ),
    }
    settings = [SimilaritySettings(reduction="mean"), SimilaritySettings(reduction="max")]
    frame = run_grid(inputs, toy_rubric, settings)
    assert set(frame["style"]) == {"neutral", "register"}
    assert len(frame) == 2 * 2 * 4 * 3


def test_run_grid_refuses_keys_that_disagree_with_their_inputs(toy_rubric: Rubric) -> None:
    texts = make_texts(texts_along_first_axis(3))
    neutral = make_references(toy_rubric, style="neutral")
    with pytest.raises(ConfigurationError, match="style"):
        run_grid({("test-encoder", "register"): (texts, neutral)}, toy_rubric)
    with pytest.raises(ConfigurationError, match="encoder"):
        run_grid({("other", "neutral"): (texts, neutral)}, toy_rubric)
    other_texts = make_texts(texts_along_first_axis(3), encoder="other")
    with pytest.raises(ConfigurationError, match="encoder"):
        run_grid({("test-encoder", "neutral"): (other_texts, neutral)}, toy_rubric)
    with pytest.raises(ConfigurationError, match="no inputs"):
        run_grid({}, toy_rubric)


def test_run_grid_refuses_keys_whose_embeddings_cover_other_texts(toy_rubric: Rubric) -> None:
    neutral = make_texts(texts_along_first_axis(12))
    register = neutral.subset([f"t{row}" for row in range(11)])
    inputs = {
        ("test-encoder", "neutral"): (neutral, make_references(toy_rubric, style="neutral")),
        ("test-encoder", "register"): (register, make_references(toy_rubric, style="register")),
    }
    with pytest.raises(ConfigurationError, match="'t11'") as error:
        run_grid(inputs, toy_rubric, [SimilaritySettings()])
    assert "register" in str(error.value)
    other_ids = replace(neutral, ids=np.asarray([f"x{row}" for row in range(12)]))
    inputs[("test-encoder", "register")] = (other_ids, inputs[("test-encoder", "register")][1])
    with pytest.raises(ConfigurationError) as error:
        run_grid(inputs, toy_rubric, [SimilaritySettings()])
    message = str(error.value)
    assert "24 ids" in message and "and 14 more" in message
    assert "'t7'" in message and "'t8'" not in message and "'x0'" not in message


def test_paper_keys_are_four_encoders_by_two_styles() -> None:
    assert len(PAPER_KEYS) == 8
    assert {encoder for encoder, _ in PAPER_KEYS} == set(ENCODERS)
    assert {style for _, style in PAPER_KEYS} == {"neutral", "register"}
    assert len(PAPER_KEYS) * len(PAPER_GRID) // 2 == 576


def test_run_grid_names_a_required_key_that_has_no_input(toy_rubric: Rubric) -> None:
    texts = make_texts(texts_along_first_axis(3))
    refs = make_references(toy_rubric, style="neutral")
    inputs = {("test-encoder", "neutral"): (texts, refs)}
    required = [("test-encoder", "neutral"), ("test-encoder", "register")]
    with pytest.raises(ConfigurationError, match=r"test-encoder.*register"):
        run_grid(inputs, toy_rubric, [SimilaritySettings()], require_keys=required)
    frame = run_grid(inputs, toy_rubric, [SimilaritySettings()], require_keys=required[:1])
    assert set(frame["style"]) == {"neutral"}
