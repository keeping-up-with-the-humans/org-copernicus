"""The rules that the docstrings and methods pages of the corrections state, each run on a case.

A definition, a regularisation, an extrapolation rule or a rounding guard keeps its value; what
these tests hold fixed is that the sentence which describes it is true of the code.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pytest
from corrections_support import make_correction_data
from estimates_support import classical_data

from copernicus.corrections import CorrectionData, _simex, get_correction, oriv
from copernicus.corrections.estimates_oriv import _first_stage_f
from copernicus.corrections.families import _factor, _ridge_path, calibration
from copernicus.corrections.families._draws import _pmm_draw

pytestmark = pytest.mark.filterwarnings("error")

LAMBDAS = np.asarray(_simex.LAMBDAS)


def _with(data: CorrectionData, **columns: object) -> CorrectionData:
    """The data with some columns of its frame replaced."""
    return CorrectionData(frame=data.frame.assign(**columns), score_columns=data.score_columns)


def _fit(data: CorrectionData, family: str, variant: str, **settings: object) -> Any:
    correction = get_correction(family, variant, **settings)
    return correction.fit(data, data.labelled_ids, rng=np.random.default_rng(0))


def test_an_improper_factor_solution_returns_a_loading_above_one() -> None:
    # A correlation matrix that columns can have (its eigenvalues are positive) and whose
    # one-factor loadings would need a square of 0.8 * 0.8 / 0.3, above 2.
    correlation = np.array([[1.0, 0.8, 0.8], [0.8, 1.0, 0.3], [0.8, 0.3, 1.0]])
    assert float(np.linalg.eigvalsh(correlation).min()) > 0.0
    fit = _factor.paf_fit(correlation)
    assert fit.converged
    assert float(np.max(np.abs(fit.loadings))) == pytest.approx(1.05722118, abs=1e-8)


def test_a_curve_is_constant_up_to_a_billionth_of_its_size() -> None:
    within = np.array([1.0, 1.0 + 5e-10, 1.0, 1.0, 1.0])
    flat = _simex.rational_extrapolate(LAMBDAS, within)
    assert (flat.status, flat.value) == (_simex.CONSTANT, float(within.mean()))
    assert np.isnan(flat.pole)
    beyond = np.array([1.0, 1.0 + 1e-8, 1.0, 1.0, 1.0])
    assert _simex.rational_extrapolate(LAMBDAS, beyond).status != _simex.CONSTANT


@pytest.mark.parametrize(("intercept", "slope"), [(2.0, -0.3), (5.0, 1e-3)])
def test_the_linear_limit_is_the_rational_curve_near_the_straight_line(
    intercept: float, slope: float
) -> None:
    result = _simex.rational_extrapolate(LAMBDAS, intercept + slope * LAMBDAS)
    assert result.status == _simex.LINEAR_LIMIT
    below_cap = 10.0 ** (np.log10(_simex.POLE_MAX) - 9.0 / (_simex.POLE_GRID_SIZE - 1))
    assert below_cap * (1.0 - 1e-12) <= result.pole <= _simex.POLE_MAX
    # The rational fit, not a straight line: with the five multiples and the cap of the
    # library it lies about 3.5 slopes over the pole away from the line's value.
    in_slopes_over_the_pole = abs(result.value - (intercept - slope)) * result.pole / abs(slope)
    assert 3.4 < in_slopes_over_the_pole < 3.6


def test_the_linear_limit_of_another_cap_is_far_from_the_straight_line() -> None:
    result = _simex.rational_extrapolate(LAMBDAS, 2.0 - 0.3 * LAMBDAS, pole_max=1.01)
    assert (result.status, result.pole) == (_simex.LINEAR_LIMIT, 1.01)
    assert result.value > 80.0


def test_the_first_stage_statistic_is_infinite_from_a_squared_correlation_of_one() -> None:
    assert _first_stage_f(1.0, 100) == float("inf")
    assert _first_stage_f(-1.0, 100) == float("inf")
    assert _first_stage_f(0.5, 100) == pytest.approx(98 * 0.25 / 0.75)
    # Two proxies on one line: the correlation is 1 up to rounding, so the statistic is
    # infinite or, when rounding leaves the squared correlation just below 1, very large.
    data = classical_data(4, n_texts=400, n_labelled=100)
    on_a_line = _with(data, score_b=2.0 * data.frame["score_a"] + 1.0)
    details = oriv(on_a_line, "outcome", score_a="score_a", score_b="score_b", n_boot=60).details
    assert details["instrument_correlation"] == pytest.approx(1.0, abs=1e-12)
    assert details["first_stage"]["f_statistic"] > 1e12


def test_the_error_variance_of_the_tweedie_proxy_follows_the_correlation() -> None:
    data = make_correction_data(label_all=True)
    fitted = _fit(data, "classical", "tweedie")
    score = data.frame[fitted.provenance["score_column"]].to_numpy()
    rho = float(np.corrcoef(score, data.gold)[0, 1])
    expected = float(np.var(data.gold)) * (1.0 / rho**2 - 1.0)
    assert fitted.provenance["sigma2_delta"] == pytest.approx(expected, rel=1e-10)
    on_a_line = _with(data, score_good=2.0 * data.frame["gold"] + 1.0)
    exact = _fit(on_a_line, "classical", "tweedie")
    assert 0.0 <= exact.provenance["sigma2_delta"] < 1e-12


def _beyond_the_ends(data: CorrectionData, column: str) -> CorrectionData:
    """The data with the first text far below and the second far above every score."""
    scores = data.frame[column].to_numpy().copy()
    scores[0], scores[1] = scores.min() - 50.0, scores.max() + 50.0
    return _with(data, **{column: scores})


def test_a_score_outside_the_training_range_gets_the_tweedie_value_at_the_nearer_end() -> None:
    data = make_correction_data(label_all=True)
    fitted = _fit(data, "classical", "tweedie")
    inside = fitted.predict(data, data.ids)
    moved = _beyond_the_ends(data, fitted.provenance["score_column"])
    outside = fitted.predict(moved, moved.ids[:2])
    assert (outside.iloc[0], outside.iloc[1]) == (inside.min(), inside.max())


def test_a_score_beyond_the_training_scores_gets_the_quantile_value_of_the_nearer_end() -> None:
    data = make_correction_data(label_all=True)
    gold = np.sort(data.gold)
    fitted = _fit(data, "marginal_matching", "quantile")
    moved = _beyond_the_ends(data, "score_good")
    outside = fitted.predict(moved, moved.ids[:2])
    assert (outside.iloc[0], outside.iloc[1]) == (gold[0], gold[-1])
    # Rounded scores: several training texts share the score at each end.
    rounded = _with(data, score_good=np.round(data.frame["score_good"], 0))
    scores = rounded.frame["score_good"].to_numpy()
    at_low, at_high = int((scores == scores.min()).sum()), int((scores == scores.max()).sum())
    assert at_low > 1 and at_high > 1
    tied = _fit(rounded, "marginal_matching", "quantile")
    moved = _beyond_the_ends(rounded, "score_good")
    outside = tied.predict(moved, moved.ids[:2])
    assert outside.iloc[0] == pytest.approx(np.median(gold[:at_low]), abs=1e-12)
    assert outside.iloc[1] == pytest.approx(np.median(gold[-at_high:]), abs=1e-12)


def test_platt_fits_the_penalised_likelihood_at_its_stated_strength() -> None:
    data = make_correction_data(gold_kind="binary", label_all=True)
    noise = np.random.default_rng(1).normal(scale=0.01, size=data.n)
    separated = _with(data, score_good=2.0 * data.frame["gold"] + noise)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fitted = _fit(separated, "regression_calibration", "platt")
    slope = fitted.provenance["slope"]
    intercept = fitted.provenance["intercept"]
    score, positive = separated.frame["score_good"].to_numpy(), separated.gold == 1.0
    excess = 1.0 / (1.0 + np.exp(-(slope * score + intercept))) - positive
    # The classes are separated, so the unpenalised likelihood has no finite maximum; the
    # fit is where the gradient of 0.5 * slope^2 + C * (sum of the log losses) vanishes.
    assert np.isfinite(slope) and slope > 0.0
    assert abs(slope + calibration.PLATT_C * float(excess @ score)) < 0.05
    assert abs(float(excess.sum())) < 0.05
    assert abs(float(excess @ score)) > 1.0


def test_a_ridge_path_with_no_direction_left_predicts_the_mean_of_the_target() -> None:
    # Whole numbers centre to exact zeros, so no singular value passes the cutoff.
    design = np.ones((6, 3)) * np.array([1.0, 2.0, 3.0])
    target = np.arange(6.0)
    penalties = np.array([0.0, 1.0, 100.0])
    new_rows = np.array([[5.0, 6.0, 7.0], [1.0, 2.0, 3.0]])
    predictions = _ridge_path.ridge_path_predictions(design, target, new_rows, penalties)
    assert np.array_equal(predictions, np.full((3, 2), target.mean()))


def test_with_fewer_donors_than_asked_for_every_donor_can_be_drawn() -> None:
    # Three donors where five are asked for: the uniform number picks among all three.
    donor_means, donor_gold = np.array([1.0, 2.0, 3.0]), np.array([10.0, 20.0, 30.0])
    uniform = np.array([0.0, 0.34, 0.67, 0.99])
    drawn = _pmm_draw(np.zeros(4), donor_means, donor_gold, 5, uniform)
    assert drawn.tolist() == [10.0, 20.0, 30.0, 30.0]


def test_with_fewer_training_rows_than_donors_every_training_row_is_a_donor() -> None:
    data = make_correction_data(label_all=True)
    correction = get_correction("multiple_imputation", "pmm", m_draws=40, k_donors=5)
    few = data.labelled_ids[:4]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted = correction.fit(data, few, rng=np.random.default_rng(0))
    draws = fitted.draws(data, data.ids[:30])
    donors = data.frame.set_index("text_id").loc[few, "gold"].to_numpy()
    assert np.isin(draws, donors).all()
    # Every text draws each of the four training rows in some of its 40 draws.
    for text in range(draws.shape[1]):
        assert set(draws[:, text].tolist()) == set(donors.tolist())


@pytest.mark.parametrize("variant", ["ridge", "lasso", "enet"])
def test_an_exactly_constant_feature_of_the_stacker_gets_the_coefficient_zero(
    variant: str,
) -> None:
    data = make_correction_data(label_all=True)
    frame = data.frame.assign(score_flat=0.1)
    flat = CorrectionData(frame=frame, score_columns=("score_good", "score_poor", "score_flat"))
    fitted = _fit(flat, "supervised_stacking", variant)
    position = fitted.columns.index("score_flat")
    assert fitted.sd[position] == 1.0
    assert fitted.coefficients[position] == 0.0
