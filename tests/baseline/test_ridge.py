"""Ridge design: exact leave-one-group-out error, plus-score variant, scaler."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import Ridge

from copernicus.baseline.ridge import (
    ALPHAS,
    RidgeDesign,
    RidgeFit,
    apply_scaler,
    fit_scaler,
    logo_ridge_fit,
    predict,
)
from copernicus.errors import ConfigurationError, EstimatorError


def explicit_logo_mse(
    features: np.ndarray, y: np.ndarray, groups: np.ndarray, alpha: float
) -> float:
    """Reference leave-one-group-out error by one explicit refit per group."""
    squared: list[float] = []
    for group in np.unique(groups):
        test = groups == group
        model = Ridge(alpha=alpha, fit_intercept=True, solver="svd").fit(features[~test], y[~test])
        squared.extend(((y[test] - model.predict(features[test])) ** 2).tolist())
    return float(np.mean(squared))


def make_data(n: int, p: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(n, p))
    beta = rng.normal(size=p)
    y = features @ beta + 0.5 * rng.normal(size=n) + 1.0
    s = y + rng.normal(size=n)
    return features, y, s


def test_alpha_grid_is_the_fixed_logspace():
    assert ALPHAS.shape == (41,)
    assert ALPHAS[0] == pytest.approx(1e-2) and ALPHAS[-1] == pytest.approx(1e8)
    assert np.allclose(np.diff(np.log10(ALPHAS)), 0.25)


@pytest.mark.parametrize("alpha", [1.0, 100.0])
def test_logo_error_equals_explicit_refits(alpha):
    features, y, _ = make_data(60, 8)
    groups = np.repeat(np.arange(12), 5)
    fit = RidgeDesign(features, y, groups).fit(alphas=np.array([alpha]))
    assert fit.logo_mse[0] == pytest.approx(explicit_logo_mse(features, y, groups, alpha), abs=1e-8)
    assert fit.n_groups == 12


def test_logo_error_with_singletons_and_unequal_groups():
    features, y, _ = make_data(30, 6, seed=3)
    groups = np.concatenate([np.arange(10), np.repeat(np.arange(10, 15), [2, 3, 5, 4, 6])])
    assert groups.shape[0] == 30
    alphas = np.array([0.5, 20.0])
    fit = RidgeDesign(features, y, groups).fit(alphas=alphas)
    for index, alpha in enumerate(alphas):
        assert fit.logo_mse[index] == pytest.approx(
            explicit_logo_mse(features, y, groups, float(alpha)), abs=1e-8
        )


def test_logo_error_for_more_columns_than_rows():
    features, y, _ = make_data(30, 50, seed=5)
    groups = np.repeat(np.arange(10), 3)
    fit = RidgeDesign(features, y, groups).fit(alphas=np.array([3.0]))
    assert fit.logo_mse[0] == pytest.approx(explicit_logo_mse(features, y, groups, 3.0), abs=1e-8)


@pytest.mark.parametrize("shape", [(60, 8), (30, 50)])
def test_plus_score_variant_equals_refit_on_augmented_design(shape):
    n, p = shape
    features, y, s = make_data(n, p, seed=7)
    groups = np.repeat(np.arange(n // 5), 5)
    alpha = 10.0
    fit = RidgeDesign(features, y, groups).fit(alphas=np.array([alpha]), extra=s)
    augmented = np.column_stack([features, s])
    reference = Ridge(alpha=alpha, fit_intercept=True, solver="svd").fit(augmented, y)
    assert fit.coef.shape == (p + 1,)
    assert fit.coef[-1] == pytest.approx(reference.coef_[-1], abs=1e-8)
    assert np.allclose(fit.coef, reference.coef_, atol=1e-8)
    assert fit.intercept == pytest.approx(reference.intercept_, abs=1e-8)
    assert np.allclose(predict(fit, augmented), reference.predict(augmented), atol=1e-8)
    assert fit.logo_mse[0] == pytest.approx(
        explicit_logo_mse(augmented, y, groups, alpha), abs=1e-8
    )


def test_fit_selects_the_minimising_alpha_and_predicts():
    features, y, _ = make_data(60, 8)
    groups = np.repeat(np.arange(12), 5)
    fit = RidgeDesign(features, y, groups).fit()
    assert fit.alpha == pytest.approx(float(ALPHAS[int(np.argmin(fit.logo_mse))]))
    assert fit.alpha_index == int(np.argmin(fit.logo_mse))
    reference = Ridge(alpha=fit.alpha, solver="svd").fit(features, y)
    assert np.allclose(predict(fit, features), reference.predict(features), atol=1e-8)


def test_logo_ridge_fit_equals_design_fit():
    features, y, s = make_data(60, 8)
    groups = np.repeat(np.arange(12), 5)
    direct = logo_ridge_fit(features, y, groups, extra=s)
    via_design = RidgeDesign(features, y, groups).fit(extra=s)
    assert direct.alpha == via_design.alpha
    assert np.allclose(direct.coef, via_design.coef)


def test_at_grid_edge_flags_either_end():
    alphas = np.array([1.0, 10.0, 100.0])
    coef = np.zeros(2)
    low = RidgeFit(1.0, coef, 0.0, alphas, np.array([0.1, 0.2, 0.3]), 4)
    middle = RidgeFit(10.0, coef, 0.0, alphas, np.array([0.3, 0.1, 0.2]), 4)
    high = RidgeFit(100.0, coef, 0.0, alphas, np.array([0.3, 0.2, 0.1]), 4)
    assert low.at_grid_edge and high.at_grid_edge and not middle.at_grid_edge
    assert middle.alpha_index == 1


def test_fit_scaler_weights_for_stacked_blocks():
    rng = np.random.default_rng(0)
    blocks = [rng.normal(size=(20, 4)), rng.normal(size=(20, 16))]
    scaler = fit_scaler(blocks)
    assert scaler.weights == pytest.approx((np.sqrt(10 / 4), np.sqrt(10 / 16)))
    design = apply_scaler(blocks, scaler)
    assert design.shape == (20, 20)
    assert np.allclose(design.mean(axis=0), 0.0, atol=1e-12)
    assert np.allclose(design[:, :4].std(axis=0), np.sqrt(10 / 4))
    assert np.allclose(design[:, 4:].std(axis=0), np.sqrt(10 / 16))


def test_single_block_keeps_weight_one_and_constant_columns_get_sd_one():
    block = np.column_stack([np.arange(6, dtype=float), np.full(6, 3.0)])
    scaler = fit_scaler([block])
    assert scaler.weights == (1.0,)
    assert scaler.sds[0][1] == 1.0
    design = apply_scaler([block], scaler)
    assert np.all(design[:, 1] == 0.0)
    assert design[:, 0].std() == pytest.approx(1.0)


def test_scaler_is_fitted_on_training_rows_and_applied_to_new_rows():
    rng = np.random.default_rng(1)
    train = rng.normal(loc=5.0, scale=2.0, size=(50, 3))
    test = rng.normal(loc=5.0, scale=2.0, size=(7, 3))
    scaler = fit_scaler([train])
    scaled = apply_scaler([test], scaler)
    assert np.allclose(scaled, (test - train.mean(axis=0)) / train.std(axis=0))


def test_scaler_errors_are_configuration_errors():
    with pytest.raises(ConfigurationError, match="no feature blocks"):
        fit_scaler([])
    with pytest.raises(ConfigurationError, match="2-D"):
        fit_scaler([np.arange(5.0)])
    scaler = fit_scaler([np.eye(4)])
    with pytest.raises(ConfigurationError, match="block count"):
        apply_scaler([np.eye(4), np.eye(4)], scaler)


def test_scaler_refuses_blocks_with_different_row_counts():
    with pytest.raises(ConfigurationError, match=r"row counts.*20.*15|20.*15.*row counts"):
        fit_scaler([np.ones((20, 3)), np.ones((15, 3))])


def test_scaler_refuses_non_finite_values_naming_the_block():
    clean = np.arange(12.0).reshape(6, 2)
    broken = clean.copy()
    broken[2, 1] = np.nan
    with pytest.raises(ConfigurationError, match=r"block 1.*non-finite"):
        fit_scaler([clean, broken])
    infinite = clean.copy()
    infinite[0, 0] = np.inf
    with pytest.raises(ConfigurationError, match=r"block 0.*non-finite"):
        fit_scaler([infinite])


def test_all_constant_training_design_is_a_degenerate_design_error():
    groups = np.repeat(np.arange(4), 5)
    y = np.arange(20.0)
    with pytest.raises(EstimatorError, match="degenerate training design"):
        RidgeDesign(np.ones((20, 3)), y, groups)
    with pytest.raises(EstimatorError, match="degenerate training design"):
        RidgeDesign(np.ones((2, 3)), y[:2], np.array([0, 1]))


def test_design_errors_are_estimator_errors():
    features, y, s = make_data(20, 3)
    groups = np.repeat(np.arange(4), 5)
    with pytest.raises(EstimatorError, match="length mismatch"):
        RidgeDesign(features, y[:-1], groups)
    with pytest.raises(EstimatorError, match="length mismatch"):
        RidgeDesign(features, y, groups[:-1])
    with pytest.raises(EstimatorError, match="at least two groups"):
        RidgeDesign(features, y, np.zeros(20, dtype=int))
    with pytest.raises(EstimatorError, match="extra column length"):
        RidgeDesign(features, y, groups).fit(extra=s[:-1])


def test_non_finite_logo_error_is_an_estimator_error():
    features, y, _ = make_data(20, 3)
    groups = np.repeat(np.arange(4), 5)
    y_bad = y.copy()
    y_bad[0] = np.nan
    with pytest.raises(EstimatorError, match="non-finite"):
        RidgeDesign(features, y_bad, groups).fit()
