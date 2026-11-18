"""Moment subtraction and SIMEX for the construct as predictor, and the SIMEX extrapolant."""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest
from estimates_support import (
    CLASSICAL_SLOPE,
    SPARSE_BOOT,
    SPARSE_SEED,
    classical_data,
    replay,
    sparse_covariation_data,
)

from copernicus.corrections import (
    CorrectionData,
    Estimate,
    estimates_classical,
    moment_subtraction,
    simex,
)
from copernicus.corrections._simex import LAMBDAS, POLE_MAX, Extrapolation, rational_extrapolate
from copernicus.errors import ConfigurationError, EstimatorError, SettingsError

pytestmark = pytest.mark.filterwarnings("error")

GRID = np.asarray(LAMBDAS)
CLASSICAL: dict[str, Callable[..., Estimate]] = {
    "moment_subtraction": lambda data, **options: moment_subtraction(
        data, "outcome", score_column="score_a", **options
    ),
    "simex": lambda data, **options: simex(data, "outcome", score_column="score_a", **options),
}


def test_the_extrapolant_recovers_the_exact_classical_curve() -> None:
    assert LAMBDAS == (0.0, 0.5, 1.0, 1.5, 2.0)
    for beta in (1.0, 0.026, 0.003, 1e-4):
        for ratio in (0.25, 3.0, 9.0, 100.0, 1000.0):
            fit = rational_extrapolate(GRID, beta / (1.0 + ratio * (1.0 + GRID)))
            assert fit.status == "ok", (beta, ratio, fit)
            assert abs(fit.value / beta - 1.0) < 1e-4, (beta, ratio, fit)
    reference = rational_extrapolate(GRID, 1.0 / (1.0 + 3.0 * (1.0 + GRID))).value
    scaled = rational_extrapolate(GRID, 1e-4 / (1.0 + 3.0 * (1.0 + GRID))).value
    assert abs(scaled / reference - 1e-4) < 1e-9


def test_the_extrapolant_reports_its_undefined_and_limiting_cases() -> None:
    inside = rational_extrapolate(GRID, 1.0 / (0.8 + GRID))
    assert inside.status == "pole_le_1" and not np.isfinite(inside.value)
    assert inside.pole <= 1.0
    for constant in (0.37, 1e-7, 0.0):
        flat = rational_extrapolate(GRID, np.full(GRID.size, constant))
        assert flat.status == "constant" and abs(flat.value - constant) <= 1e-12
    line = rational_extrapolate(GRID, 0.9 - 0.1 * GRID)
    assert line.status == "linear_limit" and abs(line.value - 1.0) < 1e-6
    wider = rational_extrapolate(GRID, 0.9 - 0.1 * GRID, pole_max=POLE_MAX * 100)
    assert abs(wider.value - line.value) < 1e-6
    broken = rational_extrapolate(GRID, np.array([1.0, np.nan, 0.5, 0.4, 0.3]))
    assert broken.status == "no_fit" and not np.isfinite(broken.value)


def test_moment_subtraction_recovers_the_slope_under_classical_error() -> None:
    data = classical_data(0)
    estimate = moment_subtraction(data, "outcome")
    assert abs(estimate.point - CLASSICAL_SLOPE) < 0.1
    assert estimate.details["naive_slope"] < 0.9
    assert 0.0 < estimate.se < 0.2
    assert estimate.ci_low < estimate.point < estimate.ci_high
    assert estimate.method == "moment_subtraction"
    assert (estimate.n_labelled, estimate.n_total) == (400, 2000)
    assert estimate.details["score_column"] == "score_a"
    assert estimate.details["n_boot"] == 200
    assert estimate.details["n_boot_undefined"] == 0
    assert estimate.details["n_boot_defined"] == 200
    assert estimate.details["corrected_variance"] == pytest.approx(
        estimate.details["proxy_variance"] - estimate.details["sigma2"]
    )
    again = moment_subtraction(data, "outcome")
    assert (again.point, again.se) == (estimate.point, estimate.se)
    assert moment_subtraction(data, "outcome", seed=7).se != estimate.se


def test_the_moment_point_follows_the_reverse_calibration_formula() -> None:
    data = classical_data(1, n_texts=500, n_labelled=100)
    lab = data.labelled_mask
    score = data.frame["score_b"].to_numpy()
    outcome = data.frame["outcome"].to_numpy()
    gold = data.gold[lab]
    slope = np.cov(score[lab], gold, ddof=1)[0, 1] / np.var(gold, ddof=1)
    proxy = (score - (score[lab].mean() - slope * gold.mean())) / slope
    sigma2 = np.var(proxy[lab] - gold, ddof=1)
    expected = np.cov(proxy, outcome, ddof=1)[0, 1] / (np.var(proxy, ddof=1) - sigma2)
    estimate = moment_subtraction(data, "outcome", score_column="score_b")
    assert estimate.point == pytest.approx(expected, abs=1e-12)
    assert estimate.details["sigma2"] == pytest.approx(sigma2, abs=1e-12)


def test_heavy_error_leaves_the_corrected_variance_non_positive() -> None:
    data = classical_data(0, n_texts=300, n_labelled=20, error_a=4.0)
    with pytest.raises(EstimatorError, match="corrected variance not positive"):
        moment_subtraction(data, "outcome")
    with pytest.raises(EstimatorError, match="pole_le_1"):
        simex(data, "outcome")


def test_undefined_bootstrap_resamples_are_counted_and_too_many_refuse() -> None:
    data = classical_data(1, n_texts=300, n_labelled=20, error_a=2.0)
    estimate = moment_subtraction(data, "outcome")
    assert 0 < estimate.details["n_boot_undefined"] <= 150
    assert estimate.details["n_boot_defined"] == 200 - estimate.details["n_boot_undefined"]
    assert np.isfinite(estimate.se) and estimate.se > 0.0
    with pytest.raises(EstimatorError, match="bootstrap resamples"):
        moment_subtraction(data, "outcome", n_boot=60)


def test_the_reverse_calibration_needs_a_line_on_the_labelled_rows() -> None:
    data = classical_data(2, n_texts=200, n_labelled=40)
    flat_gold = data.frame.assign(gold=np.where(data.labelled_mask, 1.0, np.nan))
    flat = CorrectionData(frame=flat_gold, score_columns=data.score_columns)
    with pytest.raises(EstimatorError, match="gold"):
        moment_subtraction(flat, "outcome")
    flat_score = data.frame.assign(score_a=np.where(data.labelled_mask, 2.0, 1.0))
    constant = CorrectionData(frame=flat_score, score_columns=data.score_columns)
    with pytest.raises(EstimatorError, match="does not covary"):
        moment_subtraction(constant, "outcome")
    few = data.frame.assign(gold=np.r_[0.5, 1.5, np.full(198, np.nan)])
    with pytest.raises(EstimatorError, match="at least 3 labelled"):
        simex(CorrectionData(frame=few, score_columns=data.score_columns), "outcome")


def test_the_classical_settings_and_columns_are_checked() -> None:
    data = classical_data(2, n_texts=200, n_labelled=40)
    with pytest.raises(SettingsError, match="n_boot"):
        moment_subtraction(data, "outcome", n_boot=49)
    with pytest.raises(SettingsError, match="n_boot"):
        simex(data, "outcome", n_boot=0)
    with pytest.raises(SettingsError, match="reps"):
        simex(data, "outcome", reps=0)
    with pytest.raises(SettingsError, match="boot_reps"):
        simex(data, "outcome", boot_reps=0)
    with pytest.raises(SettingsError, match="seed"):
        moment_subtraction(data, "outcome", seed=-1)
    with pytest.raises(ConfigurationError, match="score column 'nope'"):
        moment_subtraction(data, "outcome", score_column="nope")
    with pytest.raises(ConfigurationError, match="'nope' is not a column"):
        simex(data, "nope")


def test_simex_recovers_the_slope_under_classical_error() -> None:
    data = classical_data(0)
    estimate = simex(data, "outcome")
    assert estimate.details["status"] == "ok"
    assert abs(estimate.point - CLASSICAL_SLOPE) < 0.1
    assert estimate.details["pole"] > 1.0
    assert estimate.details["lambdas"] == LAMBDAS
    curve = estimate.details["curve"]
    assert isinstance(curve, tuple) and len(curve) == len(LAMBDAS)
    assert curve[0] == pytest.approx(estimate.details["naive_slope"])
    assert all(later < earlier for earlier, later in zip(curve, curve[1:], strict=False))
    assert 0.0 < estimate.se < 0.2
    assert estimate.method == "simex"
    assert estimate.details["n_boot"] == 100 and estimate.details["boot_reps"] == 10
    assert estimate.details["n_boot_defined"] == 100 - estimate.details["n_boot_undefined"]
    repeat = simex(data, "outcome")
    assert (repeat.point, repeat.se) == (estimate.point, estimate.se)


def test_simex_without_measurement_error_keeps_the_flat_curve() -> None:
    data = classical_data(3, n_texts=300, n_labelled=60, error_a=0.0)
    estimate = simex(data, "outcome", n_boot=50, boot_reps=2)
    assert estimate.details["status"] == "constant"
    assert estimate.point == pytest.approx(estimate.details["naive_slope"])


def test_simex_refuses_a_curve_that_cannot_be_fitted(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(lambdas: np.ndarray, values: np.ndarray) -> Extrapolation:
        return Extrapolation(value=float("nan"), pole=float("nan"), status="no_fit")

    monkeypatch.setattr(estimates_classical, "rational_extrapolate", broken)
    with pytest.raises(EstimatorError, match="no_fit") as refusal:
        simex(classical_data(2, n_texts=200, n_labelled=40), "outcome")
    assert "non-finite" in str(refusal.value) and "pole" not in str(refusal.value)


def test_the_outcome_must_be_numeric_and_complete() -> None:
    data = classical_data(2, n_texts=200, n_labelled=40)
    frame = data.frame.assign(label=pd.Series(["x"] * 200))
    with pytest.raises(ConfigurationError, match="numeric"):
        moment_subtraction(CorrectionData(frame=frame, score_columns=data.score_columns), "label")
    with pytest.raises(ConfigurationError, match="finite"):
        moment_subtraction(data, "gold")


@pytest.mark.parametrize("name", sorted(CLASSICAL))
def test_the_default_still_refuses_fewer_than_50_defined_resamples(name: str) -> None:
    data = sparse_covariation_data(0)
    _, values = replay(name, data, seed=SPARSE_SEED, n_boot=SPARSE_BOOT)
    with pytest.raises(EstimatorError) as refusal:
        CLASSICAL[name](data, n_boot=SPARSE_BOOT, seed=SPARSE_SEED)
    assert str(refusal.value) == (
        f"only {len(values)} of {SPARSE_BOOT} bootstrap resamples have a defined estimate; the "
        "standard error needs at least 50"
    )


@pytest.mark.parametrize("name", sorted(CLASSICAL))
def test_a_floor_of_10_takes_the_deviation_of_exactly_the_defined_resamples(name: str) -> None:
    data = sparse_covariation_data(0)
    _, values = replay(name, data, seed=SPARSE_SEED, n_boot=SPARSE_BOOT)
    assert 10 <= len(values) < SPARSE_BOOT / 2
    estimate = CLASSICAL[name](data, n_boot=SPARSE_BOOT, seed=SPARSE_SEED, min_defined=10)
    assert estimate.se == pytest.approx(float(np.std(values, ddof=1)), rel=1e-12)
    assert estimate.ci_low == pytest.approx(estimate.point - 1.96 * estimate.se, rel=1e-12)
    details = estimate.details
    assert (details["n_boot"], details["n_boot_defined"], details["n_boot_undefined"]) == (
        SPARSE_BOOT,
        len(values),
        SPARSE_BOOT - len(values),
    )
    assert (details["min_defined"], details["below_floor"], details["se_status"]) == (
        10,
        "refuse",
        "ok",
    )


@pytest.mark.parametrize("name", sorted(CLASSICAL))
def test_below_the_floor_the_point_is_kept_without_an_interval(
    name: str, caplog: pytest.LogCaptureFixture
) -> None:
    data = sparse_covariation_data(0)
    point, values = replay(name, data, seed=SPARSE_SEED, n_boot=SPARSE_BOOT)
    with caplog.at_level(logging.WARNING, logger="copernicus"):
        kept = CLASSICAL[name](data, n_boot=SPARSE_BOOT, seed=SPARSE_SEED, below_floor="point")
    [warning] = [record for record in caplog.records if record.levelno >= logging.WARNING]
    assert warning.levelno == logging.WARNING
    assert warning.name == "copernicus.corrections.estimates_classical"
    message = warning.getMessage()
    assert message.startswith(f"{name}: only {len(values)} of {SPARSE_BOOT} bootstrap resamples")
    assert "fewer than the 50" in message
    reached = CLASSICAL[name](data, n_boot=SPARSE_BOOT, seed=SPARSE_SEED, min_defined=10)
    assert kept.point == reached.point == pytest.approx(point, rel=1e-12)
    assert np.isnan(kept.se) and np.isnan(kept.ci_low) and np.isnan(kept.ci_high)
    assert (kept.n_labelled, kept.n_total, kept.method) == (20, 300, name)
    details = kept.details
    assert (details["n_boot"], details["n_boot_defined"], details["n_boot_undefined"]) == (
        SPARSE_BOOT,
        len(values),
        SPARSE_BOOT - len(values),
    )
    assert (details["min_defined"], details["below_floor"], details["se_status"]) == (
        50,
        "point",
        "too_few_defined",
    )


@pytest.mark.parametrize("name", sorted(CLASSICAL))
def test_the_floor_and_the_rule_below_it_are_checked(name: str) -> None:
    data = classical_data(2, n_texts=200, n_labelled=40)
    call = CLASSICAL[name]
    for floor in (1, 0, True, 10.0, "10"):
        with pytest.raises(SettingsError, match="min_defined must be an integer of at least 2"):
            call(data, n_boot=60, min_defined=floor)
    with pytest.raises(SettingsError, match="n_boot must be an integer of at least 61; got 60"):
        call(data, n_boot=60, min_defined=61)
    with pytest.raises(SettingsError, match="n_boot must be an integer of at least 10; got 9"):
        call(data, n_boot=9, min_defined=10)
    for rule in ("pont", "Point", None, 1):
        with pytest.raises(SettingsError, match="below_floor must be 'refuse' or 'point'"):
            call(data, below_floor=rule)
    accepted = call(data, n_boot=10, min_defined=10)
    assert (accepted.details["n_boot"], accepted.details["min_defined"]) == (10, 10)
    assert accepted.details["se_status"] == "ok" and np.isfinite(accepted.se)


def test_every_classical_estimate_records_its_floor_and_whether_it_has_a_standard_error() -> None:
    data = classical_data(2, n_texts=200, n_labelled=40)
    for estimate in (moment_subtraction(data, "outcome"), simex(data, "outcome", n_boot=50)):
        assert list(estimate.details)[-4:] == [
            "min_defined",
            "below_floor",
            "se_status",
            "se_exactly_zero",
        ]
        assert (
            estimate.details["min_defined"],
            estimate.details["below_floor"],
            estimate.details["se_status"],
        ) == (50, "refuse", "ok")
