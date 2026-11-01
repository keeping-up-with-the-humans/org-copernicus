"""The four score errors and the replicate noise."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from copernicus.errors import ConfigurationError, EstimatorError
from copernicus.validity.anatomy import (
    ScoreErrors,
    constant_score_errors,
    covariate_slope,
    replicate_noise,
    score_errors,
)


def test_score_errors_recover_planted_values():
    rng = np.random.default_rng(1)
    n = 2000
    gold = rng.uniform(0, 1, n)
    length = rng.normal(size=n)
    e2 = score_errors(gold + 0.1, gold, length)
    assert e2.covariate_linked_error == pytest.approx(0.0, abs=0.02)
    assert e2.calibration_shortfall == pytest.approx(0.0, abs=0.02)
    # compression plus noise: the slope of the score on gold falls well below one
    noisy = 0.5 * gold + 0.25 + rng.normal(scale=0.2, size=n)
    e3 = score_errors(noisy, gold, length)
    assert e3.calibration_shortfall < -0.3
    assert e3.n == n


def test_pure_affine_compression_has_no_shortfall_after_observed_minmax():
    gold = np.linspace(0, 1, 101)
    e = score_errors(0.5 * gold + 0.25, gold, np.sin(np.arange(101.0)))
    assert e.calibration_shortfall == pytest.approx(0.0, abs=1e-12)
    assert e.level_displacement == pytest.approx(0.0, abs=1e-12)


def test_planted_covariate_slope_and_level_on_the_unit_scale():
    rng = np.random.default_rng(5)
    n = 4000
    gold = np.concatenate([[0.0, 1.0], rng.uniform(0, 1, n - 2)])
    cov = rng.normal(size=n)
    # a score that equals gold plus a covariate term, then min-max scaled by the function
    score = gold + 0.1 * cov
    e = score_errors(score, gold, cov, covariate_scale="raw")
    s01 = (score - score.min()) / np.ptp(score)
    diff = s01 - gold
    expected = np.cov(diff, cov, ddof=1)[0, 1] / np.var(cov, ddof=1)
    assert e.covariate_linked_error == pytest.approx(expected, rel=1e-9)
    assert e.level_displacement == pytest.approx(diff.mean(), rel=1e-9)
    assert e.mae_01 == pytest.approx(np.abs(diff).mean(), rel=1e-9)
    assert e.signed_gap_01 == pytest.approx(diff.mean(), rel=1e-9)
    assert e.calibration_shortfall == pytest.approx(
        np.cov(s01, gold, ddof=1)[0, 1] / np.var(gold, ddof=1) - 1, rel=1e-9
    )


def test_covariate_scale_options_and_noise():
    rng = np.random.default_rng(2)
    n = 500
    gold = rng.uniform(0, 1, n)
    cov = rng.normal(scale=2.0, size=n)
    s = gold + 0.05 * cov
    std = score_errors(s, gold, cov)
    raw = score_errors(s, gold, cov, covariate_scale="raw")
    assert std.covariate_scale == "standardized" and raw.covariate_scale == "raw"
    assert std.covariate_linked_error == pytest.approx(
        raw.covariate_linked_error * cov.std(ddof=0), rel=0.05
    )
    assert std.covariate_linked_error == pytest.approx(
        raw.covariate_linked_error * cov.std(ddof=1), rel=1e-9
    )
    frame = pd.DataFrame(
        {"text_id": [1, 1, 2, 2], "run_number": [1, 2, 1, 2], "score": [3, 5, 2, 2]}
    )
    assert replicate_noise(frame) == pytest.approx(np.mean([np.std([3, 5], ddof=1), 0.0]))


def test_standardized_equals_covariance_with_the_z_scored_covariate():
    rng = np.random.default_rng(7)
    gold = rng.uniform(size=300)
    cov = rng.normal(size=300)
    score = gold + 0.2 * cov + rng.normal(scale=0.1, size=300)
    e = score_errors(score, gold, cov)
    s01 = (score - score.min()) / np.ptp(score)
    g01 = (gold - gold.min()) / np.ptp(gold)
    z = (cov - cov.mean()) / cov.std(ddof=1)
    assert e.covariate_linked_error == pytest.approx(np.cov(s01 - g01, z, ddof=1)[0, 1])


def test_zero_variance_covariate_gives_nan_without_error():
    gold = np.linspace(0, 1, 20)
    e = score_errors(gold**2, gold, np.full(20, 3.7))
    assert np.isnan(e.covariate_linked_error)
    assert np.isfinite(e.level_displacement) and np.isfinite(e.calibration_shortfall)


def test_unknown_covariate_scale_is_refused():
    gold = np.linspace(0, 1, 10)
    with pytest.raises(ConfigurationError, match="covariate_scale"):
        score_errors(gold, gold, gold, covariate_scale="percent")  # type: ignore[arg-type]


def test_length_mismatch_is_refused():
    with pytest.raises(ConfigurationError, match="length"):
        score_errors(np.arange(5.0), np.arange(5.0), np.arange(4.0))


def test_zero_range_score_or_gold_raises_estimator_error():
    gold = np.linspace(0, 1, 10)
    with pytest.raises(EstimatorError):
        score_errors(np.ones(10), gold, gold)
    with pytest.raises(EstimatorError):
        score_errors(gold, np.ones(10), gold)


def test_rows_with_a_non_finite_value_are_dropped_and_counted():
    gold = np.linspace(0, 1, 12)
    score = gold.copy()
    score[0] = np.nan
    cov = np.arange(12.0)
    cov[1] = np.inf
    e = score_errors(score, gold, cov)
    assert e.n == 10
    assert isinstance(e, ScoreErrors)


def test_replicate_noise_ignores_single_run_texts():
    frame = pd.DataFrame(
        {
            "text_id": ["a", "a", "b", "c", "c", "c"],
            "run_number": [1, 2, 1, 1, 2, 3],
            "score": [1.0, 3.0, 9.0, 2.0, 2.0, 5.0],
        }
    )
    expected = np.mean([np.std([1, 3], ddof=1), np.std([2, 2, 5], ddof=1)])
    assert replicate_noise(frame) == pytest.approx(expected)


def test_replicate_noise_is_nan_when_no_text_has_two_runs():
    frame = pd.DataFrame({"text_id": [1, 2, 3], "run_number": [1, 1, 1], "score": [1.0, 2.0, 3.0]})
    assert np.isnan(replicate_noise(frame))


def test_replicate_noise_honours_column_names_and_names_a_missing_one():
    frame = pd.DataFrame({"id": [1, 1], "rep": [1, 2], "value": [0.0, 2.0]})
    assert replicate_noise(frame, text_id="id", run="rep", score="value") == pytest.approx(
        np.sqrt(2.0)
    )
    with pytest.raises(ConfigurationError, match="rep"):
        replicate_noise(frame.drop(columns="rep"), text_id="id", run="rep", score="value")
    with pytest.raises(ConfigurationError, match="score"):
        replicate_noise(frame, text_id="id", run="rep")


def test_near_constant_covariate_gives_nan_slope_without_a_warning():
    rng = np.random.default_rng(3)
    gold = rng.uniform(size=20)
    score = gold + rng.normal(scale=0.1, size=20)
    covariate = np.full(20, 7.0)
    covariate[-1] += 1e-14
    errors = score_errors(score, gold, covariate)
    assert np.isnan(errors.covariate_linked_error)


def test_replicate_noise_on_the_observed_scale_is_the_sd_of_min_max_scaled_scores():
    frame = pd.DataFrame(
        {
            "text_id": ["a", "a", "b", "b", "c"],
            "run_number": [1, 2, 1, 2, 1],
            "score": [1.0, 3.0, 2.0, 5.0, 9.0],
        }
    )
    # the range runs over every (text, run) row, the single-run text included: 1 to 9
    expected = np.mean([np.std([0.0, 0.25], ddof=1), np.std([0.125, 0.5], ddof=1)])
    assert replicate_noise(frame, scale="observed") == pytest.approx(expected, abs=1e-15)
    assert replicate_noise(frame) == pytest.approx(8.0 * expected, abs=1e-15)


def test_replicate_noise_on_the_observed_scale_leaves_missing_runs_out():
    frame = pd.DataFrame(
        {
            "text_id": ["a", "a", "b", "b"],
            "run_number": [1, 2, 1, 2],
            "score": pd.array([1, 3, 5, pd.NA], dtype="Int64"),
        }
    )
    assert replicate_noise(frame, scale="observed") == pytest.approx(np.std([0.0, 0.5], ddof=1))


def test_constant_scores_have_no_noise_on_the_observed_scale():
    frame = pd.DataFrame({"text_id": [1, 1, 2, 2], "run_number": [1, 2, 1, 2], "score": 3.0})
    assert replicate_noise(frame, scale="observed") == 0.0
    single = frame[frame["run_number"] == 1]
    assert np.isnan(replicate_noise(single, scale="observed"))


def test_replicate_noise_refuses_an_unknown_scale():
    frame = pd.DataFrame({"text_id": [1, 1], "run_number": [1, 2], "score": [0.0, 1.0]})
    with pytest.raises(ConfigurationError, match="scale"):
        replicate_noise(frame, scale="percent")  # type: ignore[arg-type]


def test_a_constant_score_keeps_the_shortfall_and_the_covariate_linked_error():
    rng = np.random.default_rng(4)
    covariate = rng.normal(size=200)
    gold = rng.uniform(1.0, 5.0, size=200) + 0.5 * covariate
    raw = constant_score_errors(gold, covariate, covariate_scale="raw")
    standardized = constant_score_errors(gold, covariate)
    gold01 = (gold - gold.min()) / np.ptp(gold)
    z = (covariate - covariate.mean()) / covariate.std(ddof=1)
    slope = np.cov(gold01, covariate, ddof=1)[0, 1] / np.var(covariate, ddof=1)
    assert raw.calibration_shortfall == -1.0 and standardized.calibration_shortfall == -1.0
    assert raw.covariate_linked_error == pytest.approx(-slope, rel=1e-12)
    assert standardized.covariate_linked_error == pytest.approx(
        -np.cov(gold01, z, ddof=1)[0, 1], rel=1e-12
    )
    assert (raw.covariate_scale, standardized.covariate_scale) == ("raw", "standardized")
    for errors in (raw, standardized):
        assert np.isnan(errors.level_displacement)
        assert np.isnan(errors.mae_01) and np.isnan(errors.signed_gap_01)
        assert errors.n == 200


def test_constant_score_errors_refuse_a_constant_gold_and_an_unknown_scale():
    with pytest.raises(EstimatorError):
        constant_score_errors(np.full(10, 2.0), np.arange(10.0))
    with pytest.raises(ConfigurationError, match="covariate_scale"):
        constant_score_errors(np.arange(10.0), np.arange(10.0), covariate_scale="log")  # type: ignore[arg-type]


def test_covariate_slope_per_unit_and_per_standard_deviation():
    rng = np.random.default_rng(8)
    covariate = rng.normal(loc=3.0, scale=2.0, size=500)
    diff = 0.25 * covariate + rng.normal(scale=0.1, size=500)
    raw = covariate_slope(diff, covariate, "raw")
    expected = np.cov(diff, covariate, ddof=1)[0, 1] / np.var(covariate, ddof=1)
    assert raw == pytest.approx(expected, rel=1e-12)
    standardized = covariate_slope(diff, covariate)
    assert standardized == pytest.approx(raw * covariate.std(ddof=1), rel=1e-12)


def test_covariate_slope_is_the_covariate_error_of_score_errors():
    rng = np.random.default_rng(9)
    gold = rng.uniform(size=300)
    covariate = rng.normal(size=300)
    score = gold + 0.2 * covariate
    errors = score_errors(score, gold, covariate)
    diff = (score - score.min()) / np.ptp(score) - (gold - gold.min()) / np.ptp(gold)
    assert covariate_slope(diff, covariate) == pytest.approx(errors.covariate_linked_error)


def test_covariate_slope_uses_finite_pairs_and_is_nan_for_a_constant_covariate():
    diff = np.array([0.1, np.nan, 0.3, 0.5, 0.2])
    covariate = np.array([1.0, 2.0, 3.0, np.inf, 5.0])
    kept = np.array([0, 2, 4])
    assert covariate_slope(diff, covariate, "raw") == pytest.approx(
        covariate_slope(diff[kept], covariate[kept], "raw")
    )
    assert np.isnan(covariate_slope(np.array([0.1, 0.2, 0.4]), np.full(3, 2.0)))


def test_covariate_slope_refuses_an_unknown_scale_and_unequal_lengths():
    with pytest.raises(ConfigurationError, match="covariate_scale"):
        covariate_slope(np.ones(3), np.arange(3.0), "per_unit")
    with pytest.raises(ConfigurationError, match="equal length"):
        covariate_slope(np.ones(3), np.arange(4.0))
