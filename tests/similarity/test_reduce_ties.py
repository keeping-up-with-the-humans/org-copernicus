"""Tests of the pca reduction on rows with equal bytes: every such row gets one score."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hybrid_support import STUB_DIMENSION, StubEncoder, unit_references

from copernicus._versions import read_version
from copernicus.assets import load_toy_cues
from copernicus.coders import CodingSettings, code_texts
from copernicus.data import TextTable
from copernicus.hybrid import evidence_texts, score_hybrid
from copernicus.llm import FakeBackend, cue_responder
from copernicus.rubric.identifiers import ROLLUP_SUFFIX
from copernicus.rubric.model import Rubric
from copernicus.similarity import (
    WRITER_POOLINGS,
    ReferenceEmbeddings,
    SimilaritySettings,
    rollup_scores,
    score_similarity,
)
from copernicus.similarity.reduce import COVARIANCE_SOLVER_FROM, _pca_solver, reduce_matrix
from copernicus.validity.metrics import association

GROUPS = 12
GROUP_SIZE = 3
# One shape of each exact route that _pca_solver names: the full SVD for fewer than ten rows per
# column, covariance_eigh for at most 1,000 columns and at least ten rows per column.
ROUTES = [(120, 30, "full"), (1500, 100, "covariance_eigh")]
SHAPES = [(rows, columns) for rows, columns, _ in ROUTES]
# Positions of the toy corpus whose texts are made one string, and the column of values that a
# rank correlation compares the scores with; the two texts hold different values in it.
SIMILARITY_PAIR = (1, 2)
HYBRID_PAIR = (0, 29)
HYBRID_TEXTS = 30
VALUES = "gold_constructiveness"
# A product of a float32 entry with 1/4 or -1/4 keeps the entry's 24 significant bits, so sixteen
# such products and every partial sum of them fit the 53 bits of a float64 when the entries of a
# text vector span at most this many binary orders of magnitude.
EXACT_SPREAD = 24


def matrix_with_groups(rows: int, columns: int, seed: int) -> tuple[np.ndarray, list[np.ndarray]]:
    """Return cosine-like values in which each group of rows repeats the bytes of its first row.

    The groups are returned as sorted arrays of row indices, the first row first.
    """
    generator = np.random.default_rng(seed)
    matrix = 0.3 + 0.05 * generator.normal(size=(rows, columns))
    chosen = generator.permutation(rows)[: GROUPS * GROUP_SIZE]
    members = np.sort(chosen.reshape(GROUPS, GROUP_SIZE), axis=1)
    for group in members:
        matrix[group[1:]] = matrix[group[0]]
    return matrix, list(members)


def plain_matrix(rows: int, columns: int, seed: int) -> np.ndarray:
    """Return cosine-like values whose rows all differ."""
    matrix = 0.3 + 0.05 * np.random.default_rng(seed).normal(size=(rows, columns))
    assert len({row.tobytes() for row in matrix}) == rows
    return matrix


def decomposition(matrix: np.ndarray) -> np.ndarray:
    """Return the scores as the exact decomposition gives them, signed as the reduction signs them.

    This is the reduction's computation without the step that gives rows with equal bytes one
    score: the route of `_pca_solver` on the standardised columns, flipped when the scores
    correlate negatively with the row means.
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    solver = _pca_solver(matrix.shape[0], matrix.shape[1], "m")
    standardised = StandardScaler().fit_transform(matrix)
    scores = PCA(n_components=1, svd_solver=solver).fit_transform(standardised)[:, 0]
    row_means = matrix.mean(axis=1)
    if np.std(scores) > 0 and np.std(row_means) > 0 and np.corrcoef(scores, row_means)[0, 1] < 0:
        scores = -scores
    return scores


def bits(values: np.ndarray) -> np.ndarray:
    """Return the bits of float64 values as unsigned integers, so that a comparison is bitwise."""
    return np.ascontiguousarray(values, dtype=np.float64).view(np.uint64)


def installed_route(route: str) -> str:
    """Return the route of a shape under the installed scikit-learn: the full SVD before 1.5."""
    import sklearn

    version = read_version(sklearn.__version__)
    has_covariance = version is not None and version.release[:2] >= COVARIANCE_SOLVER_FROM
    return route if has_covariance else "full"


@pytest.mark.parametrize(("rows", "columns", "route"), ROUTES)
def test_every_row_of_a_group_gets_the_score_of_its_first_row(
    rows: int, columns: int, route: str
) -> None:
    assert _pca_solver(rows, columns, "m") == installed_route(route)
    matrix, groups = matrix_with_groups(rows, columns, seed=rows + columns)
    scores = reduce_matrix(matrix, "pca", "m")
    split = [group.tolist() for group in groups if len(set(bits(scores[group]).tolist())) != 1]
    assert split == [], f"groups of equal rows with more than one score: {split}"


@pytest.mark.parametrize(("rows", "columns"), SHAPES)
def test_first_rows_and_unshared_rows_keep_the_scores_of_the_decomposition(
    rows: int, columns: int
) -> None:
    # The pinned scores come from the decomposition called as the reduction calls it.
    matrix, groups = matrix_with_groups(rows, columns, seed=rows + columns)
    later = np.concatenate([group[1:] for group in groups])
    kept = np.setdiff1d(np.arange(rows), later)
    scores = reduce_matrix(matrix, "pca", "m")
    np.testing.assert_array_equal(bits(scores[kept]), bits(decomposition(matrix)[kept]))


@pytest.mark.parametrize(("rows", "columns"), SHAPES)
@pytest.mark.parametrize("seed", [1, 2])
def test_a_matrix_without_equal_rows_keeps_every_score_of_the_decomposition(
    rows: int, columns: int, seed: int
) -> None:
    matrix = plain_matrix(rows, columns, seed)
    np.testing.assert_array_equal(
        bits(reduce_matrix(matrix, "pca", "m")), bits(decomposition(matrix))
    )


def assert_kept_apart(matrix: np.ndarray, pairs: list[tuple[int, int]]) -> None:
    """Assert that both rows of every pair keep the scores of the decomposition.

    The decomposition must give some pair two scores; otherwise a reduction that grouped the
    pairs could not be told from one that keeps them apart.
    """
    expected = decomposition(matrix)
    assert any(np.unique(bits(expected[[a, b]])).size == 2 for a, b in pairs)
    rows = [row for pair in pairs for row in pair]
    np.testing.assert_array_equal(
        bits(reduce_matrix(matrix, "pca", "m")[rows]), bits(expected[rows])
    )


def test_rows_that_differ_in_one_bit_keep_their_own_scores() -> None:
    matrix = plain_matrix(120, 30, seed=7)
    pairs = [(row, row + 60) for row in range(10, 20)]
    for (first, second), column in zip(pairs, (0, 0, 3, 7, 11, 15, 19, 23, 27, 29), strict=True):
        matrix[second] = matrix[first]
        matrix[second, column] = np.nextafter(matrix[first, column], np.inf)
    assert_kept_apart(matrix, pairs)


def test_rows_equal_only_after_standardisation_keep_their_own_scores() -> None:
    from sklearn.preprocessing import StandardScaler

    matrix = plain_matrix(120, 30, seed=8)
    # Values of 1e6 in a column make a one-bit change of its small values vanish when the column
    # is standardised, so the rows differ only as they are passed in.
    matrix[:6, 0] = 1e6
    pairs = [(row, row + 60) for row in range(10, 20)]
    for first, second in pairs:
        matrix[second] = matrix[first]
        matrix[second, 0] = np.nextafter(matrix[first, 0], np.inf)
    standardised = StandardScaler().fit_transform(matrix)
    assert all(standardised[a].tobytes() == standardised[b].tobytes() for a, b in pairs)
    assert_kept_apart(matrix, pairs)


def test_zero_and_negative_zero_are_different_bytes() -> None:
    matrix = plain_matrix(120, 30, seed=9)
    pairs = [(row, row + 60) for row in range(10, 20)]
    for (first, second), column in zip(pairs, (0, 0, 3, 7, 11, 15, 19, 23, 27, 29), strict=True):
        matrix[first, column] = 0.0
        matrix[second] = matrix[first]
        matrix[second, column] = -0.0
    assert_kept_apart(matrix, pairs)


def near_copies(dtype: type[np.floating]) -> np.ndarray:
    """Return rows that repeat the bytes of earlier rows or differ from them in one bit."""
    zero, quarter, one = dtype(0.0), dtype(0.25), dtype(1.0)
    return np.array(
        [
            [quarter, zero, one],
            [quarter, zero, one],  # the bytes of row 0
            [quarter, -zero, one],  # negative zero
            [np.nextafter(quarter, one), zero, one],  # one bit in the first entry
            [quarter, zero, np.nextafter(one, dtype(2.0))],  # one bit in the last entry
            [quarter, -zero, one],  # the bytes of row 2
            [quarter, zero, one],  # the bytes of row 0
        ],
        dtype=dtype,
    )


FIRST_OF_NEAR_COPIES = [0, 0, 2, 3, 4, 2, 0]


@pytest.mark.parametrize(
    "matrix",
    [near_copies(np.float64), near_copies(np.float32), near_copies(np.float64).astype(complex)],
    ids=["float64", "float32", "complex128 by its single bytes"],
)
def test_rows_are_grouped_by_their_bytes_under_the_first_of_them(matrix: np.ndarray) -> None:
    from copernicus.similarity.reduce import _first_equal_row

    assert _first_equal_row(matrix).tolist() == FIRST_OF_NEAR_COPIES
    assert _first_equal_row(np.asfortranarray(matrix)).tolist() == FIRST_OF_NEAR_COPIES
    assert _first_equal_row(matrix[::-1]).tolist() == [0, 1, 2, 3, 1, 0, 0]


def test_a_shared_digest_costs_a_comparison_never_a_wrong_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import copernicus.similarity.reduce as reduction

    def one_digest(rows: np.ndarray) -> np.ndarray:
        return np.zeros(rows.shape[0], dtype=np.uint64)

    monkeypatch.setattr(reduction, "_row_digest", one_digest)
    assert reduction._first_equal_row(near_copies(np.float64)).tolist() == FIRST_OF_NEAR_COPIES


@pytest.mark.parametrize("comparison_bytes", [1, 4_096, 8 * 1024 * 1024])
def test_the_size_of_the_comparison_blocks_does_not_change_the_groups(
    comparison_bytes: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    import copernicus.similarity.reduce as reduction

    matrix, _ = matrix_with_groups(120, 30, seed=5)
    seen: dict[bytes, int] = {}
    expected = [seen.setdefault(row.tobytes(), index) for index, row in enumerate(matrix)]
    monkeypatch.setattr(reduction, "COMPARISON_BYTES", comparison_bytes)
    assert reduction._first_equal_row(matrix).tolist() == expected
    monkeypatch.setattr(
        reduction, "_row_digest", lambda rows: np.zeros(rows.shape[0], dtype=np.uint64)
    )
    assert reduction._first_equal_row(matrix).tolist() == expected


def split_pairs(
    frame: pd.DataFrame, pair: tuple[str, str], values: pd.Series
) -> dict[str, tuple[float, float, float, float]]:
    """Return the targets on which the two texts of a pair get two scores, or on which the rank
    correlation of the scores with ``values`` differs from the one computed after the two scores
    are set equal by hand: the two scores and the two correlations of each such target."""
    found: dict[str, tuple[float, float, float, float]] = {}
    for target, block in frame.groupby("target_id", sort=False):
        scores = block.set_index("text_id")["score"].astype(float)
        tied = scores.copy()
        tied[pair[1]] = tied[pair[0]]
        gold = values.reindex(scores.index).to_numpy(dtype=float)
        rho = association(scores.to_numpy(), gold).spearman
        rho_tied = association(tied.to_numpy(), gold).spearman
        two = bits(scores[list(pair)].to_numpy())
        if two[0] != two[1] or rho != rho_tied:
            found[str(target)] = (scores[pair[0]], scores[pair[1]], rho, rho_tied)
    return found


@pytest.mark.parametrize("pooling", WRITER_POOLINGS)
@pytest.mark.parametrize("level", ["construct", "attribute"])
def test_two_texts_of_one_string_get_one_similarity_score(
    level: str, pooling: str, toy_rubric: Rubric, toy_corpus: TextTable
) -> None:
    texts = toy_corpus.frame["text"].tolist()
    ids = [str(text_id) for text_id in toy_corpus.ids()]
    first, second = SIMILARITY_PAIR
    texts[second] = texts[first]
    embedded = StubEncoder().embed(texts, ids)
    refs = unit_references(toy_rubric, "stub", STUB_DIMENSION)
    settings = SimilaritySettings(reduction="pca", writer_pooling=pooling, target_level=level)
    frame = score_similarity(embedded, refs, toy_rubric, settings)
    values = pd.Series(toy_corpus.frame[VALUES].to_numpy(dtype=float), index=ids)
    pair = (ids[first], ids[second])
    assert values[pair[0]] != values[pair[1]]
    assert split_pairs(frame, pair, values) == {}


@pytest.mark.parametrize("pooling", WRITER_POOLINGS)
def test_the_roll_up_is_the_reduction_of_the_attribute_scores_as_returned(
    pooling: str, toy_rubric: Rubric, toy_corpus: TextTable
) -> None:
    texts = toy_corpus.frame["text"].tolist()
    ids = [str(text_id) for text_id in toy_corpus.ids()]
    first, second = SIMILARITY_PAIR
    texts[second] = texts[first]
    embedded = StubEncoder().embed(texts, ids)
    refs = unit_references(toy_rubric, "stub", STUB_DIMENSION)
    settings = SimilaritySettings(reduction="pca", writer_pooling=pooling, target_level="attribute")
    frame = score_similarity(embedded, refs, toy_rubric, settings)
    by_target = {
        str(target): block.set_index("text_id")["score"].reindex(ids).to_numpy(dtype=float)
        for target, block in frame.groupby("target_id", sort=False)
    }
    rollup = by_target.pop(f"{toy_rubric.rubric_id}{ROLLUP_SUFFIX}")
    assert rollup.tobytes() == rollup_scores(by_target, toy_rubric, "pca").tobytes()
    for scores in (*by_target.values(), rollup):
        assert scores[first].tobytes() == scores[second].tobytes()


def quarter_references(rubric: Rubric) -> ReferenceEmbeddings:
    """Return the stub references with every entry set to 1/4 or -1/4 by its sign (length 1)."""
    references = unit_references(rubric, "stub", STUB_DIMENSION)
    quarters = np.where(references.vectors < 0, -0.25, 0.25).astype(np.float32)
    return replace(references, vectors=quarters)


def cosines_are_exact(vectors: np.ndarray) -> bool:
    """Tell whether every cosine of the text vectors to quarter references is exact in float64.

    An exact cosine has the same bytes whatever the order in which the linear algebra library
    sums its products, so equal vectors get equal rows of the reduction matrix on every platform.
    """
    nonzero = vectors != 0
    _, exponents = np.frexp(vectors.astype(np.float64))
    largest = np.where(nonzero, exponents, np.iinfo(np.int32).min).max(axis=1)
    smallest = np.where(nonzero, exponents, np.iinfo(np.int32).max).min(axis=1)
    return bool(np.all(largest - smallest <= EXACT_SPREAD))


def test_two_texts_of_one_string_get_one_hybrid_score(
    toy_rubric: Rubric, toy_corpus: TextTable, tmp_path: Path
) -> None:
    """The coder gives the two texts one score where their rows of the reduction matrix are equal.

    Equal vectors alone need not give equal rows: the matrix product of the linear algebra library
    can round the cosines of two equal rows differently, depending on where they sit in the matrix
    and on the platform. The references are therefore quarter references, whose cosines with the
    stub's vectors are exact, so the two rows are equal on every platform.
    """
    frame = toy_corpus.subset(toy_corpus.ids()[:HYBRID_TEXTS]).frame.reset_index(drop=True)
    first, second = HYBRID_PAIR
    frame.loc[second, "text"] = frame.loc[first, "text"]
    table = TextTable.from_frame(frame)
    store = tmp_path / "pair.sqlite"
    backend = FakeBackend(cue_responder(toy_rubric, load_toy_cues()))
    code_texts(table, toy_rubric, backend, store, CodingSettings(runs=1))
    evidence = evidence_texts(store, toy_rubric, table, granularity="construct", min_words=10)
    assert evidence.texts[first] == evidence.texts[second]
    ids = [str(text_id) for text_id in table.ids()]
    values = pd.Series(table.frame[VALUES].to_numpy(dtype=float), index=ids)
    pair = (ids[first], ids[second])
    assert values[pair[0]] != values[pair[1]]
    refs = quarter_references(toy_rubric)
    for granularity in ("construct", "attribute"):
        encoder = StubEncoder()
        result = score_hybrid(
            store, table, toy_rubric, encoder, refs, min_words=10, granularity=granularity
        )
        embedded = [
            StubEncoder().embed(texts, list(range(len(texts)))).vectors for texts in encoder.calls
        ]
        assert embedded and all(cosines_are_exact(vectors) for vectors in embedded), granularity
        assert split_pairs(result.frame, pair, values) == {}, granularity
