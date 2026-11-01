"""Association metrics, finite pairing and min-max scaling."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.metrics import roc_auc_score

from copernicus.errors import ConfigurationError, EstimatorError
from copernicus.validity.metrics import (
    Association,
    association,
    auc_from_ranks,
    finite_pairs,
    is_binary,
    is_constant,
    minmax01,
)


def test_association_matches_scipy_and_flags_binary():
    rng = np.random.default_rng(0)
    g = rng.normal(size=50)
    s = g + rng.normal(scale=0.5, size=50)
    a = association(s, g)
    assert a.status == "ok" and a.n == 50
    assert np.isclose(a.spearman, spearmanr(s, g).statistic)
    assert a.rank_validity == abs(a.spearman)
    assert np.isclose(a.pearson, pearsonr(s, g).statistic)
    assert np.isclose(a.kendall, kendalltau(s, g).statistic)
    assert np.isnan(a.auc) and np.isnan(a.somers_d)
    b = association(s, (g > 0).astype(float))
    assert np.isclose(b.somers_d, 2 * b.auc - 1)


def test_association_statuses():
    assert association(np.ones(10), np.arange(10.0)).status == "constant_score"
    assert association(np.arange(3.0), np.ones(3)).status == "constant_gold"
    assert association(np.arange(2.0), np.arange(2.0)).status == "too_few"


def test_status_order_prefers_too_few_then_score_then_gold():
    assert association(np.ones(2), np.ones(2)).status == "too_few"
    assert association(np.ones(5), np.ones(5)).status == "constant_score"


def test_non_ok_status_has_nan_metrics_and_counts_finite_pairs():
    score = np.array([1.0, np.nan, 1.0, 1.0, 1.0])
    gold = np.array([0.0, 1.0, 2.0, np.nan, 3.0])
    a = association(score, gold)
    assert a.status == "constant_score" and a.n == 3
    for name in ("spearman", "rank_validity", "pearson", "kendall", "auc", "somers_d"):
        assert np.isnan(getattr(a, name))


def test_too_few_counts_pairs_after_dropping_non_finite():
    a = association(np.array([1.0, 2.0, 3.0, np.nan]), np.array([1.0, 2.0, np.nan, np.nan]))
    assert a.status == "too_few" and a.n == 2


def test_auc_equals_sklearn_with_ties():
    rng = np.random.default_rng(3)
    gold = rng.integers(0, 2, size=200).astype(float)
    score = np.round(gold * 0.8 + rng.normal(size=200), 0)
    a = association(score, gold)
    assert len(np.unique(score)) < 20
    assert a.auc == pytest.approx(roc_auc_score(gold, score), abs=1e-12)
    assert a.somers_d == pytest.approx(2 * a.auc - 1)


def test_auc_positive_class_is_the_larger_gold_value():
    score = np.array([0.1, 0.2, 0.8, 0.9])
    flipped = association(score, np.array([5.0, 5.0, 2.0, 2.0]))
    assert flipped.auc == pytest.approx(0.0)
    straight = association(score, np.array([2.0, 2.0, 5.0, 5.0]))
    assert straight.auc == pytest.approx(1.0)


def test_three_valued_gold_has_no_auc():
    a = association(np.arange(6.0), np.array([0.0, 0, 1, 1, 2, 2]))
    assert a.status == "ok" and np.isnan(a.auc) and np.isnan(a.somers_d)


def test_is_binary():
    assert is_binary(np.array([0.0, 1.0, 1.0, 0.0]))
    assert is_binary(np.array([3.0, 7.0, np.nan]))
    assert not is_binary(np.array([1.0, 1.0]))
    assert not is_binary(np.array([0.0, 1.0, 2.0]))
    assert not is_binary(np.array([np.nan, np.nan]))


def test_association_is_frozen():
    a = association(np.arange(5.0), np.arange(5.0))
    assert isinstance(a, Association)
    with pytest.raises(AttributeError):
        a.n = 3  # type: ignore[misc]


def test_minmax01_scales_to_unit_interval():
    out = minmax01(np.array([2.0, 4.0, 6.0]))
    assert out.tolist() == [0.0, 0.5, 1.0]


def test_minmax01_zero_range_raises():
    with pytest.raises(EstimatorError):
        minmax01(np.array([3.0, 3.0, 3.0]))


def test_minmax01_keeps_non_finite_entries_out_of_the_range():
    out = minmax01(np.array([0.0, np.nan, 10.0]))
    assert out[0] == 0.0 and out[2] == 1.0 and np.isnan(out[1])


def test_finite_pairs_keeps_rows_where_every_array_is_finite():
    s, g, c = finite_pairs(
        np.array([1.0, np.nan, 3.0, 4.0]),
        np.array([1, 2, np.inf, 4]),
        [0.5, 0.6, 0.7, 0.8],
    )
    assert s.tolist() == [1.0, 4.0] and g.tolist() == [1.0, 4.0] and c.tolist() == [0.5, 0.8]
    assert s.dtype == float and g.dtype == float


def test_finite_pairs_rejects_length_mismatch():
    with pytest.raises(ConfigurationError, match="length"):
        finite_pairs(np.arange(3.0), np.arange(4.0))
    with pytest.raises(ConfigurationError, match="length"):
        finite_pairs(np.arange(3.0), np.arange(3.0), np.arange(2.0))


def test_near_constant_score_is_constant_without_a_scipy_warning():
    score = np.array([1.0, 1.0, 1.0, 1.0 + 1e-15])
    result = association(score, np.arange(4.0))
    assert result.status == "constant_score"
    assert np.isnan(result.spearman) and result.n == 4


def test_near_constant_gold_is_constant_gold():
    gold = np.array([5.0, 5.0, 5.0, 5.0 + 1e-14])
    assert association(np.arange(4.0), gold).status == "constant_gold"


def test_is_constant_uses_a_relative_tolerance():
    assert is_constant(np.ones(5))
    assert is_constant(np.array([1.0, 1.0 + 1e-15]))
    assert is_constant(np.array([1e9, 1e9 + 1e-4]))
    assert not is_constant(np.array([1.0, 1.0 + 1e-6]))
    assert not is_constant(np.array([0.0, 1e-9]))


def test_minmax01_refuses_a_near_constant_input():
    with pytest.raises(EstimatorError):
        minmax01(np.array([2.0, 2.0, 2.0 + 1e-15]))


def test_finite_pairs_checks_rank_before_length():
    with pytest.raises(ConfigurationError):
        finite_pairs(np.float64(1), np.float64(2))


def test_auc_from_ranks_equals_sklearn_on_tie_heavy_data():
    rng = np.random.default_rng(11)
    score = rng.integers(0, 4, size=80).astype(float)
    positive = rng.random(80) < 0.4
    assert auc_from_ranks(score, positive) == pytest.approx(
        roc_auc_score(positive.astype(int), score), abs=1e-12
    )
