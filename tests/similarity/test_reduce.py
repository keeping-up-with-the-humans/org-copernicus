"""Tests of the reduction of a text-by-reference matrix to one score per text."""

from __future__ import annotations

import numpy as np
import pytest

from copernicus.errors import ConfigurationError
from copernicus.similarity.reduce import REDUCTIONS, reduce_matrix

MATRIX = np.array(
    [
        [0.1, 0.5, 0.3],
        [0.4, 0.2, 0.9],
        [0.7, 0.6, 0.8],
        [0.2, 0.1, 0.0],
        [0.9, 0.8, 0.5],
    ]
)


def test_reductions_are_named() -> None:
    assert REDUCTIONS == ("mean", "max", "pca")


def test_mean_and_max_match_hand_values() -> None:
    np.testing.assert_allclose(reduce_matrix(MATRIX, "mean", "m"), MATRIX.mean(axis=1))
    np.testing.assert_allclose(reduce_matrix(MATRIX, "max", "m"), [0.5, 0.9, 0.8, 0.2, 0.9])


def test_pca_correlates_positively_with_the_row_mean() -> None:
    scores = reduce_matrix(MATRIX, "pca", "m")
    assert scores.shape == (5,)
    assert np.corrcoef(scores, MATRIX.mean(axis=1))[0, 1] > 0


def test_pca_sign_is_anchored_whatever_the_default_sign_would_be() -> None:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    flips = 0
    for seed in range(30):
        matrix = np.random.default_rng(seed).uniform(size=(6, 4))
        raw = PCA(n_components=1).fit_transform(StandardScaler().fit_transform(matrix))[:, 0]
        anchored = reduce_matrix(matrix, "pca", "m")
        assert np.corrcoef(anchored, matrix.mean(axis=1))[0, 1] > 0
        np.testing.assert_allclose(np.abs(anchored), np.abs(raw))
        flips += int(np.corrcoef(raw, matrix.mean(axis=1))[0, 1] < 0)
    assert flips > 0, "no matrix exercised the sign flip"


@pytest.mark.parametrize("method", REDUCTIONS)
def test_a_single_column_is_returned_as_it_is(method: str) -> None:
    column = MATRIX[:, [1]]
    np.testing.assert_array_equal(reduce_matrix(column, method, "m"), column[:, 0])


def test_one_row_is_fine_for_mean_and_max() -> None:
    row = MATRIX[[0]]
    np.testing.assert_allclose(reduce_matrix(row, "mean", "m"), [0.3])
    np.testing.assert_allclose(reduce_matrix(row, "max", "m"), [0.5])


def test_input_is_not_modified() -> None:
    before = MATRIX.copy()
    reduce_matrix(MATRIX, "pca", "m")
    np.testing.assert_array_equal(MATRIX, before)


@pytest.mark.parametrize(
    ("matrix", "method"),
    [
        (MATRIX[0], "mean"),
        (np.empty((4, 0)), "mean"),
        (np.where(MATRIX > 0.5, np.nan, MATRIX), "mean"),
        (np.where(MATRIX > 0.5, np.inf, MATRIX), "max"),
        (MATRIX, "median"),
        (MATRIX[:, [0]], "median"),
        (MATRIX[[0]], "pca"),
        (np.full((4, 3), 0.5), "pca"),
    ],
    ids=[
        "one-dimensional",
        "no-columns",
        "nan",
        "infinite",
        "unknown-method",
        "unknown-method-one-column",
        "pca-one-row",
        "pca-zero-variance",
    ],
)
def test_bad_inputs_raise_with_the_label(matrix: np.ndarray, method: str) -> None:
    with pytest.raises(ConfigurationError, match="construct:net-pca"):
        reduce_matrix(matrix, method, "construct:net-pca")


def test_partly_constant_columns_still_reduce() -> None:
    matrix = np.column_stack([np.full(5, 0.5), MATRIX[:, 0], MATRIX[:, 1]])
    scores = reduce_matrix(matrix, "pca", "m")
    assert np.all(np.isfinite(scores))
