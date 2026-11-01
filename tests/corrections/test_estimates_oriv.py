"""Obviously related instrumental variables: two measurements, each the other's instrument."""

from __future__ import annotations

import numpy as np
import pytest
from estimates_support import CLASSICAL_SLOPE, classical_data

from copernicus.corrections import CorrectionData, Estimate, oriv
from copernicus.corrections.estimates_oriv import WEAK_INSTRUMENT_FLOOR, stacked_2sls
from copernicus.errors import ConfigurationError, EstimatorError, SettingsError

pytestmark = pytest.mark.filterwarnings("error")


def two_stage_oracle(y: np.ndarray, wa: np.ndarray, wb: np.ndarray) -> float:
    """The staged equation (7): two first-stage lines, then OLS on the stacked fitted values."""
    n = y.size

    def fitted(x: np.ndarray, z: np.ndarray) -> np.ndarray:
        slope = np.cov(x, z, ddof=1)[0, 1] / np.var(z, ddof=1)
        return x.mean() - slope * z.mean() + slope * z

    stacked = np.concatenate([fitted(wa, wb), fitted(wb, wa)])
    zeros, ones = np.zeros(n), np.ones(n)
    design = np.column_stack([np.r_[ones, zeros], np.r_[zeros, ones], stacked])
    beta, *_ = np.linalg.lstsq(design, np.r_[y, y], rcond=None)
    return float(beta[2])


def test_the_stacked_2sls_equals_the_staged_equation() -> None:
    rng = np.random.default_rng(7)
    n = 2000
    construct = rng.normal(size=n)
    y = 0.9 * construct + rng.normal(size=n) * 0.8
    wa = construct + rng.normal(size=n) * 0.5
    wb = construct + rng.normal(size=n) * 1.5
    fit = stacked_2sls(y, wa, wb)
    assert abs(fit.slope - two_stage_oracle(y, wa, wb)) < 1e-8
    assert fit.se > 0.0
    assert fit.first_stage_a_on_b == pytest.approx(np.cov(wa, wb)[0, 1] / np.var(wb, ddof=1))
    assert fit.first_stage_b_on_a == pytest.approx(np.cov(wa, wb)[0, 1] / np.var(wa, ddof=1))


def test_the_clustered_standard_error_follows_the_sandwich() -> None:
    rng = np.random.default_rng(11)
    n = 400
    construct = rng.normal(size=n)
    y = construct + rng.normal(size=n)
    wa = construct + rng.normal(size=n) * 0.6
    wb = construct + rng.normal(size=n) * 0.9
    zeros, ones = np.zeros(n), np.ones(n)
    x = np.column_stack([np.r_[ones, zeros], np.r_[zeros, ones], np.r_[wa, wb]])
    z = np.column_stack([x[:, 0], x[:, 1], np.r_[wb, zeros], np.r_[zeros, wa]])
    projected = z @ np.linalg.solve(z.T @ z, z.T @ x)
    bread = np.linalg.inv(projected.T @ x)
    beta = bread @ projected.T @ np.r_[y, y]
    residual = np.r_[y, y] - x @ beta
    scores = projected[:n] * residual[:n, None] + projected[n:] * residual[n:, None]
    covariance = bread @ scores.T @ scores @ bread.T
    fit = stacked_2sls(y, wa, wb)
    assert fit.slope == pytest.approx(beta[2], abs=1e-12)
    assert fit.se == pytest.approx(np.sqrt(covariance[2, 2]), abs=1e-12)


def test_a_constant_measurement_leaves_the_system_singular() -> None:
    y = np.linspace(0.0, 1.0, 50)
    with pytest.raises(EstimatorError, match="singular"):
        stacked_2sls(y, np.linspace(1.0, 2.0, 50), np.ones(50))


def test_oriv_recovers_the_slope_from_two_independent_measurements() -> None:
    data = classical_data(0)
    estimate = oriv(data, "outcome", score_a="score_a", score_b="score_b")
    assert abs(estimate.point - CLASSICAL_SLOPE) < 0.1
    assert 0.0 < estimate.se < 0.2
    assert estimate.method == "oriv"
    assert (estimate.n_labelled, estimate.n_total) == (400, 2000)
    details = estimate.details
    assert details["instrument_correlation"] > WEAK_INSTRUMENT_FLOOR
    assert abs(details["shared_error_correlation"]) < 0.15
    assert set(details["first_stage"]) == {"a_on_b", "b_on_a", "f_statistic"}
    assert details["first_stage"]["f_statistic"] > 10.0
    assert (details["score_a"], details["score_b"]) == ("score_a", "score_b")
    assert (details["n_boot"], details["n_boot_undefined"], details["n_boot_defined"]) == (
        200,
        0,
        200,
    )
    swapped = oriv(data, "outcome", score_a="score_b", score_b="score_a")
    assert swapped.point == pytest.approx(estimate.point, abs=1e-10)


def test_the_sandwich_is_reported_beside_the_bootstrap_standard_error() -> None:
    data = classical_data(5, n_texts=400, n_labelled=40)
    lab, gold = data.labelled_mask, data.gold
    proxies = []
    for column in ("score_a", "score_b"):
        score = data.frame[column].to_numpy()
        slope = np.cov(score[lab], gold[lab], ddof=1)[0, 1] / np.var(gold[lab], ddof=1)
        proxies.append((score - (score[lab].mean() - slope * gold[lab].mean())) / slope)
    fit = stacked_2sls(data.frame["outcome"].to_numpy(), proxies[0], proxies[1])
    estimate = oriv(data, "outcome", score_a="score_a", score_b="score_b", n_boot=100, seed=3)
    assert estimate.point == pytest.approx(fit.slope, abs=1e-12)
    assert estimate.details["sandwich_se"] == pytest.approx(fit.se, abs=1e-12)
    assert estimate.se != pytest.approx(fit.se, rel=1e-3)
    again = oriv(data, "outcome", score_a="score_a", score_b="score_b", n_boot=100, seed=3)
    assert again.se == estimate.se
    with pytest.raises(SettingsError, match="n_boot"):
        oriv(data, "outcome", score_a="score_a", score_b="score_b", n_boot=49)
    with pytest.raises(SettingsError, match="seed"):
        oriv(data, "outcome", score_a="score_a", score_b="score_b", seed=-1)


def test_the_bootstrap_interval_covers_where_the_sandwich_is_too_narrow() -> None:
    """60 small samples (40 of 400 texts labelled): the calibration lines' error matters."""
    boot, sandwich = [], []
    for seed in range(60):
        data = classical_data(seed, n_texts=400, n_labelled=40)
        estimate = oriv(data, "outcome", score_a="score_a", score_b="score_b", n_boot=100)
        boot.append(estimate.ci_low <= CLASSICAL_SLOPE <= estimate.ci_high)
        half_width = 1.96 * estimate.details["sandwich_se"]
        sandwich.append(abs(estimate.point - CLASSICAL_SLOPE) <= half_width)
    assert float(np.mean(boot)) >= 0.88
    assert float(np.mean(sandwich)) <= 0.85


def test_a_shared_error_shows_in_the_diagnostic() -> None:
    data = classical_data(4, error_a=0.4, error_b=0.4, shared=0.6)
    estimate = oriv(data, "outcome", score_a="score_a", score_b="score_b")
    assert estimate.details["shared_error_correlation"] > 0.3


def test_a_weak_pair_is_refused_at_the_floor() -> None:
    assert WEAK_INSTRUMENT_FLOOR == 0.20
    data = classical_data(0, error_a=3.0, error_b=3.0)
    with pytest.raises(EstimatorError, match="0.20"):
        oriv(data, "outcome", score_a="score_a", score_b="score_b")


def test_the_weak_instrument_floor_is_a_parameter() -> None:
    data = classical_data(1, n_texts=600, n_labelled=120)

    def run(floor: object) -> Estimate:
        return oriv(
            data,
            "outcome",
            score_a="score_a",
            score_b="score_b",
            n_boot=50,
            weak_instrument_floor=floor,  # type: ignore[arg-type]
        )

    correlation = run(WEAK_INSTRUMENT_FLOOR).details["instrument_correlation"]
    assert 0.5 < correlation < 0.7  # type: ignore[operator]
    with pytest.raises(EstimatorError, match=r"below the floor of 0\.75"):
        run(0.75)
    assert run(0.4).details["weak_instrument_floor"] == 0.4
    for value in (0.0, 1.0, -0.1, 1.5, float("nan"), "0.2", True):
        with pytest.raises(SettingsError, match="weak_instrument_floor"):
            run(value)


def test_oriv_checks_its_measurements() -> None:
    data = classical_data(2, n_texts=200, n_labelled=40)
    with pytest.raises(SettingsError, match="differ"):
        oriv(data, "outcome", score_a="score_a", score_b="score_a")
    with pytest.raises(ConfigurationError, match="score column 'nope'"):
        oriv(data, "outcome", score_a="score_a", score_b="nope")
    flat_gold = data.frame.assign(gold=np.where(data.labelled_mask, 1.0, np.nan))
    with pytest.raises(EstimatorError, match="gold"):
        oriv(
            CorrectionData(frame=flat_gold, score_columns=data.score_columns),
            "outcome",
            score_a="score_a",
            score_b="score_b",
        )
