"""Unit-mean fold metrics and their weighted aggregation over folds."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import pearsonr, spearmanr

from copernicus.baseline.metrics import (
    METRIC_KEYS,
    STATUS_KEYS,
    aggregate_units,
    classes_are_separable,
    fold_metrics,
    weighted_over_folds,
)
from copernicus.errors import ConfigurationError, EstimatorError

BINARY = (0.0, 1.0)


def test_keys():
    assert METRIC_KEYS == (
        "spearman",
        "spearman_abs",
        "pearson",
        "mae_01",
        "mae_native",
        "auc",
        "somers_d",
    )
    assert STATUS_KEYS == (
        "ok",
        "constant_gold",
        "constant_pred",
        "single_class",
        "too_few_units",
        "nonbinary_unit_gold",
    )


def test_aggregate_units_averages_within_units_in_sorted_order():
    unit_id = np.array(["b", "a", "b", "c", "a"])
    pred = np.array([1.0, 2.0, 3.0, 4.0, 6.0])
    gold = np.array([0.0, 1.0, 1.0, 0.0, 1.0])
    units, pred_u, gold_u = aggregate_units(pred, gold, unit_id)
    assert list(units) == ["a", "b", "c"]
    assert np.allclose(pred_u, [4.0, 2.0, 4.0])
    assert np.allclose(gold_u, [1.0, 0.5, 0.0])


def test_aggregate_units_refuses_a_non_finite_prediction():
    with pytest.raises(EstimatorError, match="non-finite prediction"):
        aggregate_units(np.array([1.0, np.nan]), np.array([0.0, 1.0]), np.array(["a", "b"]))


def test_continuous_gold_metrics_match_scipy():
    rng = np.random.default_rng(0)
    gold = rng.normal(size=30)
    pred = gold + rng.normal(size=30)
    out = fold_metrics(pred, gold, None, gold)
    assert out["status"] == "ok" and out["n_units"] == 30.0
    assert out["spearman"] == pytest.approx(spearmanr(pred, gold).statistic)
    assert out["spearman_abs"] == pytest.approx(abs(spearmanr(pred, gold).statistic))
    assert out["pearson"] == pytest.approx(pearsonr(pred, gold).statistic)
    assert out["mae_native"] == pytest.approx(np.mean(np.abs(pred - gold)))
    assert np.isnan(out["auc"]) and np.isnan(out["somers_d"])


def test_mae_01_scales_with_the_training_gold_range():
    pred = np.array([0.0, 1.0, 3.0])
    gold = np.array([1.0, 1.0, 2.0])
    out = fold_metrics(pred, gold, None, np.array([0.0, 4.0]))
    assert out["mae_01"] == pytest.approx(np.mean(np.abs(pred - gold)) / 4.0)


def test_mae_01_is_nan_under_a_zero_training_gold_range():
    pred = np.array([0.0, 1.0, 3.0])
    gold = np.array([1.0, 1.0, 2.0])
    out = fold_metrics(pred, gold, None, np.array([2.0, 2.0, 2.0]))
    assert np.isnan(out["mae_01"])
    assert out["mae_native"] == pytest.approx(np.mean(np.abs(pred - gold)))


def test_binary_gold_gives_somers_d_of_two_auc_minus_one():
    gold = np.array([0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0])
    pred = np.array([0.1, 0.4, 0.35, 0.8, 0.8, 0.2, 0.9, 0.4])
    out = fold_metrics(pred, gold, BINARY, gold)
    assert out["status"] == "ok"
    assert out["somers_d"] == pytest.approx(2.0 * out["auc"] - 1.0)
    positives = pred[gold == 1.0]
    negatives = pred[gold == 0.0]
    wins = (positives[:, None] > negatives[None, :]).sum()
    ties = (positives[:, None] == negatives[None, :]).sum()
    assert out["auc"] == pytest.approx((wins + 0.5 * ties) / (positives.size * negatives.size))


def test_constant_prediction_on_binary_gold_keeps_auc_one_half():
    gold = np.array([0.0, 1.0, 1.0, 0.0, 1.0])
    pred = np.full(5, 0.3)
    out = fold_metrics(pred, gold, BINARY, gold)
    assert out["status"] == "constant_pred"
    assert out["auc"] == pytest.approx(0.5)
    assert out["somers_d"] == pytest.approx(0.0)
    assert np.isnan(out["spearman"]) and np.isnan(out["pearson"])
    assert out["mae_native"] == pytest.approx(np.mean(np.abs(pred - gold)))


def test_single_class_fold_leaves_auc_undefined_but_keeps_errors():
    gold = np.ones(5)
    pred = np.array([0.1, 0.5, 0.2, 0.9, 0.4])
    out = fold_metrics(pred, gold, BINARY, np.array([0.0, 1.0]))
    assert out["status"] == "single_class"
    assert np.isnan(out["auc"]) and np.isnan(out["somers_d"])
    assert np.isnan(out["spearman"])
    assert np.isfinite(out["mae_01"]) and np.isfinite(out["mae_native"])


def test_too_few_units_status_overrides_and_skips_correlations():
    out = fold_metrics(np.array([0.2, 0.9]), np.array([0.0, 1.0]), BINARY, np.array([0.0, 1.0]))
    assert out["status"] == "too_few_units"
    assert out["auc"] == pytest.approx(1.0)
    assert np.isnan(out["spearman"]) and np.isnan(out["pearson"])


def test_nonbinary_unit_gold_under_binary_flag():
    gold = np.array([0.0, 0.5, 1.0, 1.0])
    pred = np.array([0.1, 0.4, 0.6, 0.9])
    out = fold_metrics(pred, gold, BINARY, np.array([0.0, 1.0]))
    assert out["status"] == "nonbinary_unit_gold"
    assert np.isnan(out["auc"]) and np.isnan(out["somers_d"])
    assert np.isfinite(out["spearman"])


@pytest.mark.parametrize("classes", [(1.0, 2.0), (-1.0, 1.0), (2.0, 1.0)])
def test_any_two_class_values_give_the_auc_of_the_larger_value(classes):
    gold = np.array([0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0])
    pred = np.array([0.1, 0.4, 0.35, 0.8, 0.8, 0.2, 0.9, 0.4])
    low, high = sorted(classes)
    recoded = np.where(gold == 1.0, high, low)
    reference = fold_metrics(pred, gold, BINARY, gold)
    out = fold_metrics(pred, recoded, classes, recoded)
    assert out["status"] == "ok"
    assert out["auc"] == pytest.approx(reference["auc"])
    assert out["somers_d"] == pytest.approx(reference["somers_d"])


def test_unit_means_of_a_class_value_match_it_up_to_their_rounding():
    unit_id = np.repeat(np.array(["a", "b", "c", "d"]), 3)
    gold = np.repeat([0.1, 0.7, 0.1, 0.7], 3)
    pred = np.repeat([0.2, 0.9, 0.4, 0.6], 3)
    _, pred_u, gold_u = aggregate_units(pred, gold, unit_id)
    assert not np.isin(gold_u, (0.1, 0.7)).all()
    out = fold_metrics(pred_u, gold_u, (0.1, 0.7), gold)
    assert out["status"] == "ok" and out["auc"] == pytest.approx(1.0)


def test_unit_gold_outside_the_two_class_values_is_nonbinary():
    gold = np.array([1.0, 1.5, 2.0, 2.0])
    pred = np.array([0.1, 0.4, 0.6, 0.9])
    out = fold_metrics(pred, gold, (1.0, 2.0), np.array([1.0, 2.0]))
    assert out["status"] == "nonbinary_unit_gold"
    assert np.isnan(out["auc"])


@pytest.mark.parametrize("classes", [(1.0, 1.0), (0.0, 1.0, 2.0), (0.0, float("nan"))])
def test_class_values_must_be_two_distinct_finite_numbers(classes):
    gold = np.array([0.0, 1.0, 1.0])
    with pytest.raises(ConfigurationError, match="two distinct"):
        fold_metrics(np.array([0.1, 0.2, 0.3]), gold, classes, gold)


def test_constant_gold_status():
    out = fold_metrics(np.array([0.1, 0.5, 0.2]), np.full(3, 2.0), None, np.array([0.0, 4.0]))
    assert out["status"] == "constant_gold"
    assert np.isnan(out["spearman"]) and np.isfinite(out["mae_01"])


def test_numerically_constant_prediction_is_constant():
    pred = 0.5 + np.array([0.0, 1e-16, 0.0, 1e-16])
    gold = np.array([0.0, 1.0, 0.0, 1.0])
    out = fold_metrics(pred, gold, None, gold)
    assert out["status"] == "constant_pred" and np.isnan(out["spearman"])


def fold(n_units: float, value: float, status: str = "ok") -> dict[str, float | str]:
    entry: dict[str, float | str] = {key: value for key in METRIC_KEYS}
    entry["n_units"] = n_units
    entry["status"] = status
    return entry


def test_weighted_over_folds_weights_by_unit_count():
    out = weighted_over_folds([fold(10, 0.2), fold(30, 0.6)])
    assert out["spearman"] == pytest.approx(0.75 * 0.6 + 0.25 * 0.2)
    assert out["spearman_n_folds"] == 2.0
    assert out["spearman_unit_weight"] == 40.0
    assert out["n_test_units_total"] == 40.0


def test_weighted_over_folds_counts_only_finite_folds():
    folds = [fold(10, 0.2), fold(30, np.nan, "single_class"), fold(20, 0.8, "constant_pred")]
    out = weighted_over_folds(folds)
    assert out["auc"] == pytest.approx((10 * 0.2 + 20 * 0.8) / 30)
    assert out["auc_n_folds"] == 2.0
    assert out["auc_unit_weight"] == 30.0
    assert out["n_test_units_total"] == 60.0
    assert out["n_folds_ok"] == 1.0
    assert out["n_folds_single_class"] == 1.0
    assert out["n_folds_constant_pred"] == 1.0
    assert out["n_folds_too_few_units"] == 0.0


def test_weighted_over_folds_with_no_finite_fold_is_nan_with_zero_counts():
    out = weighted_over_folds([fold(10, np.nan), fold(5, np.nan)])
    assert np.isnan(out["spearman"])
    assert out["spearman_n_folds"] == 0.0 and out["spearman_unit_weight"] == 0.0


def test_weighted_over_folds_emits_every_output_key():
    out = weighted_over_folds([fold(10, 0.2)])
    expected = {"n_test_units_total"}
    for key in METRIC_KEYS:
        expected |= {key, f"{key}_n_folds", f"{key}_unit_weight"}
    expected |= {f"n_folds_{status}" for status in STATUS_KEYS}
    assert set(out) == expected


def test_class_values_that_the_tolerance_cannot_separate_are_refused():
    gold = np.array([0.3, 0.1 + 0.2, 0.3])
    with pytest.raises(ConfigurationError, match="cannot be told apart"):
        fold_metrics(np.array([0.1, 0.2, 0.3]), gold, (0.3, 0.1 + 0.2), gold)


@pytest.mark.parametrize(
    ("first", "second", "separable"),
    [
        (1.0, 1.0 + 1e-12, False),
        (1.0, 1.0 + 3e-12, True),
        (1.0 + 1e-12, 1.0, False),
        (1, 1.0 + 3e-12, True),
        (0, 1, True),
        (1, 1.0, False),
        (0.3, 0.1 + 0.2, False),
        (1e6, 1e6 + 1e-7, False),
        (1e6, 1e6 + 1e-5, True),
    ],
)
def test_classes_are_separable_at_the_tolerance_boundary(first, second, separable):
    assert classes_are_separable(first, second) is separable
