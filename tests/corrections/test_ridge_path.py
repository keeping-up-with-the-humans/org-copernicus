"""Ridge predictions along a penalty grid and the choice of a penalty on grouped deletions."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import Ridge

from copernicus.baseline.ridge import ALPHAS
from copernicus.corrections.families._ridge_path import (
    GroupDeletion,
    PenaltyChoice,
    choose_penalty,
    ridge_coefficients,
    ridge_path_predictions,
    squared_error,
)
from copernicus.errors import ConfigurationError

pytestmark = pytest.mark.filterwarnings("error")


def _design(
    n_rows: int, n_columns: int, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Uncentred columns of different scales, a target with an offset, and new rows."""
    rng = np.random.default_rng(seed)
    scales = rng.uniform(0.5, 3.0, size=n_columns)
    x_fit = rng.normal(size=(n_rows, n_columns)) * scales + 2.0
    y_fit = x_fit @ rng.normal(size=n_columns) + rng.normal(size=n_rows) + 4.0
    x_pred = rng.normal(size=(7, n_columns)) * scales + 2.0
    return x_fit, y_fit, x_pred


@pytest.mark.parametrize("shape", [(40, 6), (12, 30)])
def test_the_path_equals_ridge_with_an_unpenalised_intercept_at_every_penalty(
    shape: tuple[int, int],
) -> None:
    x_fit, y_fit, x_pred = _design(*shape)
    path = ridge_path_predictions(x_fit, y_fit, x_pred, ALPHAS)
    assert path.shape == (ALPHAS.size, x_pred.shape[0])
    for index, alpha in enumerate(ALPHAS):
        oracle = Ridge(alpha=float(alpha), fit_intercept=True, solver="svd").fit(x_fit, y_fit)
        np.testing.assert_allclose(path[index], oracle.predict(x_pred), rtol=0.0, atol=1e-8)


def test_a_constant_design_predicts_the_target_mean() -> None:
    x_fit = np.full((5, 2), 3.0)
    y_fit = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    path = ridge_path_predictions(x_fit, y_fit, np.array([[0.0, 9.0]]), np.array([0.1, 10.0]))
    np.testing.assert_allclose(path, np.full((2, 1), 3.0))


@pytest.mark.parametrize("alpha", [0.01, 3.0, 1e4])
def test_coefficients_at_one_penalty_match_ridge_and_the_path(alpha: float) -> None:
    x_fit, y_fit, x_pred = _design(40, 6, seed=2)
    fit = ridge_coefficients(x_fit, y_fit, alpha)
    oracle = Ridge(alpha=alpha, fit_intercept=True, solver="svd").fit(x_fit, y_fit)
    assert fit.alpha == alpha
    np.testing.assert_allclose(fit.weights, oracle.coef_, rtol=0.0, atol=1e-8)
    assert fit.intercept == pytest.approx(float(oracle.intercept_), abs=1e-8)
    path = ridge_path_predictions(x_fit, y_fit, x_pred, np.array([alpha]))
    np.testing.assert_allclose(fit.predict(x_pred), path[0], rtol=0.0, atol=1e-10)


@pytest.mark.parametrize(
    ("x_fit", "y_fit", "x_pred", "match"),
    [
        (np.ones(4), np.ones(4), np.ones((2, 1)), "two-dimensional"),
        (np.ones((4, 2)), np.ones(3), np.ones((2, 2)), "4 fitting rows and 3 targets"),
        (np.ones((4, 2)), np.ones(4), np.ones((2, 3)), "2 fitting columns and 3 prediction"),
        (np.ones((0, 2)), np.ones(0), np.ones((2, 2)), "empty"),
        (np.ones((4, 2)), np.ones(4), np.ones((0, 2)), "empty"),
        (np.array([[1.0, np.nan], [2.0, 3.0]]), np.ones(2), np.ones((1, 2)), "non-finite"),
        (np.ones((2, 2)), np.array([1.0, np.inf]), np.ones((1, 2)), "non-finite"),
    ],
)
def test_the_path_refuses_inconsistent_inputs(
    x_fit: np.ndarray, y_fit: np.ndarray, x_pred: np.ndarray, match: str
) -> None:
    with pytest.raises(ConfigurationError, match=match):
        ridge_path_predictions(x_fit, y_fit, x_pred, ALPHAS)


@pytest.mark.parametrize(
    "alphas", [np.array([]), np.array([[1.0]]), np.array([1.0, -0.5]), np.array([np.inf])]
)
def test_the_path_refuses_a_grid_that_is_empty_nested_negative_or_infinite(
    alphas: np.ndarray,
) -> None:
    x_fit, y_fit, x_pred = _design(10, 2)
    with pytest.raises(ConfigurationError, match="penalty grid"):
        ridge_path_predictions(x_fit, y_fit, x_pred, alphas)


@pytest.mark.parametrize("alpha", [-1.0, float("nan")])
def test_coefficients_refuse_a_negative_or_undefined_penalty(alpha: float) -> None:
    x_fit, y_fit, _ = _design(10, 2)
    with pytest.raises(ConfigurationError, match="penalty"):
        ridge_coefficients(x_fit, y_fit, alpha)


def _deletions(held_sizes: tuple[int, ...]) -> list[GroupDeletion]:
    """Deletions over consecutive blocks of rows with the given held-out sizes."""
    n_rows = sum(held_sizes)
    bounds = np.cumsum((0, *held_sizes))
    rows = np.arange(n_rows)
    return [
        GroupDeletion(retained=np.setdiff1d(rows, rows[start:end]), held=rows[start:end])
        for start, end in zip(bounds[:-1], bounds[1:], strict=True)
    ]


def _choose(curves: np.ndarray, held_sizes: tuple[int, ...], alphas: np.ndarray) -> PenaltyChoice:
    """The choice on deletions whose summed errors are the given curves, in deletion order."""
    deletions = _deletions(held_sizes)
    errors = {id(deletion): curve for deletion, curve in zip(deletions, curves, strict=True)}
    seen: list[GroupDeletion] = []

    def error_fn(deletion: GroupDeletion) -> np.ndarray:
        seen.append(deletion)
        return errors[id(deletion)]

    choice = choose_penalty(deletions, error_fn, alphas, "probe_policy", 9)
    assert seen == deletions
    return choice


def test_the_choice_minimises_the_error_pooled_over_every_held_out_row() -> None:
    alphas = np.array([0.1, 1.0, 10.0, 100.0])
    summed = np.array([[3.0, 1.0, 2.0, 5.0], [2.0, 1.0, 4.0, 0.5], [6.0, 2.0, 3.0, 9.0]])
    choice = _choose(summed, (1, 2, 3), alphas)
    np.testing.assert_allclose(choice.mse, summed.sum(axis=0) / 6.0)
    assert (choice.alpha, choice.alpha_index) == (1.0, 1)
    assert not (choice.at_lower_edge or choice.at_upper_edge or choice.at_grid_edge)
    np.testing.assert_allclose(choice.per_deletion_mse, summed / np.array([[1.0], [2.0], [3.0]]))
    assert choice.deletion_edge_share(upper=True) == pytest.approx(1.0 / 3.0)
    assert choice.deletion_edge_share(upper=False) == 0.0
    assert (choice.policy, choice.n_deletions, choice.n_groups) == ("probe_policy", 3, 9)
    np.testing.assert_array_equal(choice.alphas, alphas)


def test_the_edge_flags_name_the_end_of_the_grid_the_choice_sits_on() -> None:
    alphas = np.array([0.1, 1.0, 10.0])
    lower = _choose(np.array([[1.0, 2.0, 3.0], [1.0, 1.5, 0.5]]), (2, 2), alphas)
    assert (lower.alpha, lower.at_lower_edge, lower.at_upper_edge, lower.at_grid_edge) == (
        0.1,
        True,
        False,
        True,
    )
    assert lower.deletion_edge_share(upper=False) == 0.5
    assert lower.deletion_edge_share(upper=True) == 0.5
    upper = _choose(np.array([[4.0, 3.0, 1.0]]), (3,), alphas)
    assert (upper.alpha, upper.at_lower_edge, upper.at_upper_edge, upper.at_grid_edge) == (
        10.0,
        False,
        True,
        True,
    )


def test_a_tie_goes_to_the_first_penalty_of_the_grid() -> None:
    choice = _choose(np.array([[2.0, 2.0, 3.0]]), (2,), np.array([0.1, 1.0, 10.0]))
    assert choice.alpha == 0.1


def test_the_choice_refuses_missing_misshapen_empty_and_non_finite_errors() -> None:
    alphas = np.array([0.1, 1.0])
    with pytest.raises(ConfigurationError, match="no inner deletions"):
        choose_penalty([], lambda deletion: np.zeros(2), alphas, "probe", 2)
    deletions = _deletions((2, 2))
    with pytest.raises(ConfigurationError, match="shape"):
        choose_penalty(deletions, lambda deletion: np.zeros(3), alphas, "probe", 2)
    with pytest.raises(ConfigurationError, match="non-finite"):
        choose_penalty(deletions, lambda deletion: np.array([1.0, np.nan]), alphas, "probe", 2)
    empty = [*deletions, GroupDeletion(retained=np.arange(4), held=np.array([], dtype=int))]
    with pytest.raises(ConfigurationError, match="deletion 2 holds no row"):
        choose_penalty(empty, lambda deletion: np.zeros(2), alphas, "probe", 2)


def test_squared_error_sums_the_squared_residuals_of_each_penalty() -> None:
    predictions = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    np.testing.assert_allclose(squared_error(predictions, np.array([1.0, 0.0, 1.0])), [8.0, 2.0])
    with pytest.raises(ConfigurationError, match="rows"):
        squared_error(predictions, np.array([1.0, 0.0]))
