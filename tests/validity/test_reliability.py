"""Cronbach's alpha, the reliability ceiling and the share of the ceiling."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from copernicus.errors import ConfigurationError, EstimatorError
from copernicus.validity.reliability import cronbach_alpha, reliability_ceiling, share_of_ceiling

# Item variances 5/3, 4/3, 5/3 (sum 14/3); item sums 5, 7, 11, 13 have variance 40/3.
# alpha = (3/2) * (1 - 14/40) = 0.975.
HAND_ITEMS = np.array([[1.0, 2.0, 2.0], [2.0, 2.0, 3.0], [3.0, 4.0, 4.0], [4.0, 4.0, 5.0]])


def test_alpha_of_a_known_matrix_equals_the_hand_value():
    assert cronbach_alpha(HAND_ITEMS) == pytest.approx(0.975, abs=1e-12)


def test_alpha_accepts_a_dataframe_with_the_same_result():
    frame = pd.DataFrame(HAND_ITEMS, columns=["a", "b", "c"])
    assert cronbach_alpha(frame) == pytest.approx(0.975, abs=1e-12)


@pytest.mark.parametrize("n_items", [2, 3, 5])
def test_alpha_of_perfectly_parallel_items_is_one(n_items):
    column = np.random.default_rng(1).normal(size=30)
    items = np.tile(column[:, None], (1, n_items))
    assert cronbach_alpha(items) == pytest.approx(1.0, abs=1e-12)


def test_alpha_of_unrelated_items_is_near_zero_or_below():
    items = np.random.default_rng(2).normal(size=(2000, 4))
    assert cronbach_alpha(items) < 0.1


def test_rows_with_a_missing_item_are_dropped_and_counted_in_a_warning(caplog):
    items = np.vstack([HAND_ITEMS, [[np.nan, 1.0, 2.0], [3.0, np.inf, 1.0]]])
    with caplog.at_level(logging.WARNING, logger="copernicus.validity.reliability"):
        value = cronbach_alpha(items)
    assert value == pytest.approx(0.975, abs=1e-12)
    messages = [record.getMessage() for record in caplog.records]
    assert messages == ["cronbach_alpha: dropped 2 of 6 rows with a missing or non-finite item"]


def test_nullable_dtype_items_with_pd_na_are_dropped_like_nan_rows(caplog):
    items = pd.DataFrame(HAND_ITEMS, columns=["a", "b", "c"]).astype("Float64")
    items.loc[len(items)] = [None, 1.0, 2.0]
    assert items["a"].isna().sum() == 1
    with caplog.at_level(logging.WARNING, logger="copernicus.validity.reliability"):
        value = cronbach_alpha(items)
    assert value == pytest.approx(0.975, abs=1e-12)
    assert [record.getMessage() for record in caplog.records] == [
        "cronbach_alpha: dropped 1 of 5 rows with a missing or non-finite item"
    ]


def test_no_warning_is_logged_when_no_row_is_dropped(caplog):
    with caplog.at_level(logging.WARNING, logger="copernicus.validity.reliability"):
        cronbach_alpha(HAND_ITEMS)
    assert not caplog.records


def test_fewer_than_two_items_is_refused():
    with pytest.raises(ConfigurationError, match="two"):
        cronbach_alpha(np.arange(5.0).reshape(5, 1))


def test_one_dimensional_input_is_refused():
    with pytest.raises(ConfigurationError):
        cronbach_alpha(np.arange(5.0))


def test_non_numeric_items_are_refused():
    with pytest.raises(ConfigurationError, match="numeric"):
        cronbach_alpha(pd.DataFrame({"a": ["x", "y", "z"], "b": ["u", "v", "w"]}))


def test_fewer_than_two_complete_rows_raises_estimator_error():
    items = np.array([[1.0, 2.0], [np.nan, 3.0], [2.0, np.nan]])
    with pytest.raises(EstimatorError, match="complete"):
        cronbach_alpha(items)


def test_zero_variance_of_the_item_sum_raises_estimator_error():
    items = np.array([[1.0, 2.0], [2.0, 1.0], [3.0, 0.0], [0.0, 3.0]])
    with pytest.raises(EstimatorError, match="variance"):
        cronbach_alpha(items)


def test_reliability_ceiling_is_the_square_root():
    assert reliability_ceiling(0.81) == pytest.approx(0.9)
    assert reliability_ceiling(1.0) == 1.0


@pytest.mark.parametrize("bad", [0.0, -0.2, 1.2, float("nan"), float("inf")])
def test_reliability_outside_zero_one_is_refused(bad):
    with pytest.raises(ConfigurationError, match="reliability"):
        reliability_ceiling(bad)


def test_share_of_ceiling_divides_rank_validity_by_the_ceiling():
    assert share_of_ceiling(0.45, 0.9) == pytest.approx(0.5)


def test_share_of_ceiling_of_a_nan_rank_validity_is_nan():
    assert np.isnan(share_of_ceiling(float("nan"), 0.9))


@pytest.mark.parametrize("bad", [0.0, -0.5, float("nan"), float("inf")])
def test_share_of_ceiling_refuses_an_unusable_ceiling(bad):
    with pytest.raises(ConfigurationError, match="ceiling"):
        share_of_ceiling(0.4, bad)
