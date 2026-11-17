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


def pole_like(rows: int, columns: int, seed: int) -> np.ndarray:
    """Cosine-like values whose columns share two factors of about equal strength, plus noise."""
    generator = np.random.default_rng(seed)
    first, second = generator.normal(size=(rows, 1)), generator.normal(size=(rows, 1))
    load_first = generator.uniform(0.6, 1.0, size=(1, columns))
    signs = np.where(np.arange(columns) % 2 == 0, 1.0, -1.0)
    load_second = generator.uniform(0.6, 1.0, size=(1, columns)) * signs
    noise = 0.01 * generator.normal(size=(rows, columns))
    return 0.3 + 0.02 * first * load_first + 0.02 * second * load_second + noise


def signed_component(matrix: np.ndarray, **options: object) -> np.ndarray:
    """The first principal component of the standardised columns, signed as the library signs it.

    Without ``options`` this is the reduction of version 0.1.0, which let scikit-learn choose.
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    scores = PCA(n_components=1, **options).fit_transform(StandardScaler().fit_transform(matrix))
    scores = scores[:, 0]
    row_means = matrix.mean(axis=1)
    if np.std(scores) > 0 and np.std(row_means) > 0 and np.corrcoef(scores, row_means)[0, 1] < 0:
        scores = -scores
    return scores


def automatic_route(matrix: np.ndarray) -> str:
    """The solver that the installed scikit-learn's automatic choice takes for one component."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    return str(PCA(n_components=1).fit(StandardScaler().fit_transform(matrix))._fit_svd_solver)


def test_two_reductions_of_a_wide_matrix_are_bitwise_equal_and_exact() -> None:
    # 534 rows and 120 columns: more than 500 rows and fewer than ten rows per column, where the
    # automatic choice draws a randomized SVD from numpy's global random stream.
    matrix = pole_like(534, 120, seed=20261114)
    assert automatic_route(matrix) == "randomized"
    once = reduce_matrix(matrix, "pca", "m")
    twice = reduce_matrix(matrix, "pca", "m")
    np.testing.assert_array_equal(once, twice)
    exact = signed_component(matrix, svd_solver="full")
    np.testing.assert_allclose(once, exact, rtol=0, atol=1e-10)


@pytest.mark.parametrize(("rows", "columns"), [(2000, 30), (188, 30), (300, 30), (5000, 5)])
def test_where_the_automatic_choice_is_exact_the_result_is_bitwise_unchanged(
    rows: int, columns: int
) -> None:
    matrix = pole_like(rows, columns, seed=rows + columns)
    if automatic_route(matrix) == "randomized":
        pytest.skip("this scikit-learn takes the randomized SVD here; version 0.1.0 was not exact")
    np.testing.assert_array_equal(reduce_matrix(matrix, "pca", "m"), signed_component(matrix))


@pytest.mark.parametrize(
    ("rows", "columns"),
    [(2000, 30), (300, 30), (299, 30), (188, 30), (188, 120), (534, 120), (600, 600), (5000, 5)],
)
def test_the_solver_is_the_exact_route_of_the_automatic_choice(rows: int, columns: int) -> None:
    from copernicus.similarity.reduce import _pca_solver

    route = automatic_route(pole_like(rows, columns, seed=1))
    assert _pca_solver(rows, columns, "m") == ("full" if route == "randomized" else route)


@pytest.mark.parametrize("version", ["1.5.0", "1.9.1", "1.10.dev0", "2.0rc1"])
def test_the_covariance_route_is_taken_where_scikit_learn_has_it(
    version: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sklearn

    from copernicus.similarity.reduce import _pca_solver

    monkeypatch.setattr(sklearn, "__version__", version)
    tall = [(2000, 30), (300, 30), (10_000, 1_000), (5000, 5)]
    other = [(299, 30), (188, 30), (534, 120), (600, 600), (11_000, 1_001)]
    assert [_pca_solver(*shape, "m") for shape in tall] == ["covariance_eigh"] * len(tall)
    assert [_pca_solver(*shape, "m") for shape in other] == ["full"] * len(other)


@pytest.mark.parametrize("version", ["1.3.0", "1.4.2"])
def test_without_the_covariance_route_every_matrix_takes_the_full_svd(
    version: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # scikit-learn 1.3 and 1.4 have no covariance_eigh; their automatic choice takes the full
    # SVD up to 500 rows and columns and the randomized SVD beyond.
    import sklearn

    from copernicus.similarity.reduce import _pca_solver

    monkeypatch.setattr(sklearn, "__version__", version)
    shapes = [(2000, 30), (300, 30), (188, 30), (534, 120), (600, 600)]
    assert {_pca_solver(*shape, "m") for shape in shapes} == {"full"}


def test_an_unreadable_scikit_learn_version_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    import sklearn

    monkeypatch.setattr(sklearn, "__version__", "unknown")
    with pytest.raises(
        ConfigurationError, match="construct:net-pca: cannot read the scikit"
    ) as refused:
        reduce_matrix(MATRIX, "pca", "construct:net-pca")
    assert "install a released scikit-learn of version 1.3 or later" in str(refused.value)


@pytest.mark.parametrize(
    ("version", "route"),
    [("v1.5.0", "covariance_eigh"), (" 1.6.1 ", "covariance_eigh"), ("1.4.2.post1", "full")],
)
def test_the_scikit_learn_version_is_read_whole_by_the_package_parser(
    version: str, route: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sklearn

    from copernicus.similarity.reduce import _pca_solver

    monkeypatch.setattr(sklearn, "__version__", version)
    assert _pca_solver(2000, 30, "m") == route


@pytest.mark.parametrize("version", ["1.5abc", "1.5.0 rc1"])
def test_a_scikit_learn_version_that_the_package_parser_cannot_read_is_refused(
    version: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sklearn

    monkeypatch.setattr(sklearn, "__version__", version)
    with pytest.raises(
        ConfigurationError, match=f"cannot read the scikit-learn version {version!r}"
    ):
        reduce_matrix(MATRIX, "pca", "construct:net-pca")


def test_on_a_flat_spectrum_the_reduction_is_the_exact_component() -> None:
    # Normal noise has a flat spectrum, so a randomized SVD, even one with a fixed seed, lands
    # visibly off the first component; the reduction must equal the full SVD's component.
    matrix = np.random.default_rng(112).normal(size=(534, 120))
    assert automatic_route(matrix) == "randomized"
    exact = signed_component(matrix, svd_solver="full")
    seeded = signed_component(matrix, random_state=0)
    assert np.max(np.abs(seeded - exact)) > 0.1
    np.testing.assert_allclose(reduce_matrix(matrix, "pca", "m"), exact, rtol=0, atol=1e-10)


@pytest.mark.parametrize(
    ("rows", "columns", "solver"),
    [(534, 120, "full"), (600, 600, "full"), (2000, 30, "covariance_eigh"), (188, 30, "full")],
)
def test_the_exact_solver_is_the_one_handed_to_the_decomposition(
    rows: int, columns: int, solver: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from sklearn.decomposition import PCA

    from copernicus.similarity.reduce import _pca_solver

    handed: list[tuple[str, str]] = []
    fit_transform = PCA.fit_transform

    def recording(self: PCA, *args: object, **kwargs: object) -> np.ndarray:
        scores = fit_transform(self, *args, **kwargs)
        handed.append((self.svd_solver, self._fit_svd_solver))
        return scores

    monkeypatch.setattr(PCA, "fit_transform", recording)
    reduce_matrix(np.random.default_rng(rows).normal(size=(rows, columns)), "pca", "m")
    if solver == "covariance_eigh" and _pca_solver(rows, columns, "m") == "full":
        solver = "full"  # a scikit-learn before 1.5, which has no covariance_eigh
    assert handed == [(solver, solver)], handed
