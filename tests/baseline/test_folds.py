"""Grouped folds, whole-group draws and size feasibility."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from copernicus.baseline.folds import (
    FLOOR_UNITS,
    SIZES,
    DrawReceipt,
    draw_groups,
    grouped_kfold,
    rng_for,
    size_skip_reason,
)
from copernicus.errors import ConfigurationError, EstimatorError, SettingsError


def make_ids(n_groups: int = 20, rows_per_group: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Group id per row and a unit id per row; two rows of a group share a unit."""
    group_id = np.repeat([f"g{i:02d}" for i in range(n_groups)], rows_per_group)
    unit_id = np.array(
        [f"{g}:{j // 2}" for g in group_id[::rows_per_group] for j in range(rows_per_group)]
    )
    return group_id, unit_id


def test_sizes_and_floor_constants():
    assert SIZES == (50, 100, 150, 200, 250, 500, 1000, "full")
    assert FLOOR_UNITS == 50


def test_rng_for_is_reproducible_and_keyed_by_parts():
    first = rng_for("run", 42, "fold", 1).random(4)
    again = rng_for("run", 42, "fold", 1).random(4)
    other = rng_for("run", 43, "fold", 1).random(4)
    assert np.array_equal(first, again)
    assert not np.array_equal(first, other)


def test_rng_for_seed_is_the_md5_prefix_of_the_joined_parts():
    digest = hashlib.md5(b"a::1::b").hexdigest()
    reference = np.random.default_rng(int(digest[:16], 16)).random(3)
    assert np.array_equal(rng_for("a", 1, "b").random(3), reference)


def test_grouped_kfold_partitions_rows_and_never_splits_a_group():
    group_id, _ = make_ids(23, 4)
    folds = grouped_kfold(group_id, 5, rng_for("folds"))
    assert len(folds) == 5
    test_all = np.concatenate([test for _, test in folds])
    assert np.array_equal(np.sort(test_all), np.arange(group_id.shape[0]))
    for train, test in folds:
        assert np.intersect1d(train, test).size == 0
        assert train.size + test.size == group_id.shape[0]
        assert np.intersect1d(group_id[train], group_id[test]).size == 0


def test_grouped_kfold_is_deterministic_for_a_key():
    group_id, _ = make_ids(12, 2)
    first = grouped_kfold(group_id, 4, rng_for("x"))
    second = grouped_kfold(group_id, 4, rng_for("x"))
    for (a_train, a_test), (b_train, b_test) in zip(first, second, strict=True):
        assert np.array_equal(a_train, b_train) and np.array_equal(a_test, b_test)


def test_grouped_kfold_refuses_k_below_two():
    group_id, _ = make_ids(6, 2)
    with pytest.raises(ConfigurationError, match="k=1"):
        grouped_kfold(group_id, 1, rng_for("x"))


def test_grouped_kfold_names_the_counts_when_groups_are_fewer_than_folds():
    group_id, _ = make_ids(3, 4)
    with pytest.raises(ConfigurationError, match=r"got 3 groups for k=5 in 12 rows"):
        grouped_kfold(group_id, 5, rng_for("x"))
    one_group = np.repeat("only", 10)
    with pytest.raises(ConfigurationError, match=r"got 1 groups for k=5 in 10 rows"):
        grouped_kfold(one_group, 5, rng_for("x"))


class _DuplicatingRng:
    """A stand-in whose permutation repeats an index, so the folds cannot partition."""

    def permutation(self, n: int) -> np.ndarray:
        order = np.arange(n)
        order[-1] = 0
        return order


def test_grouped_kfold_partition_invariant_is_an_estimator_error():
    group_id, _ = make_ids(5, 2)
    with pytest.raises(EstimatorError, match="partition"):
        grouped_kfold(group_id, 5, _DuplicatingRng())  # type: ignore[arg-type]


def test_draw_groups_reaches_the_quota_with_whole_groups():
    group_id, unit_id = make_ids(40, 3)
    train_pos = np.arange(group_id.shape[0])
    drawn, receipt = draw_groups(train_pos, group_id, unit_id, 10, rng_for("draw"))
    assert receipt.actual_units >= 10
    assert receipt.requested_units == 10
    assert receipt.size == 10
    assert np.array_equal(drawn, np.sort(drawn))
    chosen = np.unique(group_id[drawn])
    for group in chosen:
        assert np.isin(np.flatnonzero(group_id == group), drawn).all()
    assert receipt.actual_groups == chosen.shape[0]
    assert receipt.actual_rows == drawn.shape[0]
    assert receipt.pool_groups == 40 and receipt.pool_rows == 120 and receipt.pool_units == 80
    assert receipt.overshoot == receipt.actual_units - 10
    assert 0 <= receipt.overshoot < 2


def test_draw_groups_stops_as_soon_as_the_quota_is_met():
    group_id, unit_id = make_ids(40, 3)
    drawn, receipt = draw_groups(np.arange(120), group_id, unit_id, 10, rng_for("draw"))
    last_group_units = 2
    assert receipt.actual_units - last_group_units < 10 <= receipt.actual_units
    assert drawn.shape[0] % 3 == 0


def test_draw_groups_full_returns_the_sorted_pool():
    group_id, unit_id = make_ids(10, 3)
    train_pos = np.array([29, 3, 4, 5, 17, 0, 1, 2])
    drawn, receipt = draw_groups(train_pos, group_id, unit_id, "full", rng_for("draw"))
    assert np.array_equal(drawn, np.sort(train_pos))
    assert receipt.requested_units == 0
    assert receipt.size == "full"
    assert receipt.overshoot == 0
    assert receipt.actual_units == receipt.pool_units


def test_draw_groups_restricts_to_the_training_positions():
    group_id, unit_id = make_ids(10, 3)
    train_pos = np.arange(15, 30)
    drawn, _ = draw_groups(train_pos, group_id, unit_id, 4, rng_for("draw"))
    assert np.isin(drawn, train_pos).all()


def test_draw_groups_uses_the_whole_pool_when_the_quota_is_just_reachable():
    group_id, unit_id = make_ids(5, 2)
    drawn, receipt = draw_groups(np.arange(10), group_id, unit_id, 5, rng_for("draw"))
    assert receipt.actual_units == 5 and drawn.shape[0] == 10


class _ShortRng:
    """A stand-in whose permutation drops groups, so the draw stops below the quota."""

    def permutation(self, n: int) -> np.ndarray:
        return np.array([0])


def test_draw_groups_stopping_below_quota_with_groups_left_is_an_estimator_error():
    group_id, unit_id = make_ids(10, 3)
    with pytest.raises(EstimatorError, match="below quota"):
        draw_groups(np.arange(30), group_id, unit_id, 12, _ShortRng())  # type: ignore[arg-type]


def test_draw_receipt_overshoot():
    receipt = DrawReceipt("full", 0, 30, 90, 20, 30, 90, 20)
    assert receipt.overshoot == 0
    receipt = DrawReceipt(50, 50, 53, 160, 40, 400, 1200, 300)
    assert receipt.overshoot == 3


def test_size_skip_reason_on_both_sides_of_the_half_rule():
    assert size_skip_reason(50, 100) == ""
    assert size_skip_reason(50, 101) == ""
    assert size_skip_reason(51, 101) == "quota_exceeds_half_units(quota=51,units=101)"
    assert size_skip_reason(1000, 1500) == "quota_exceeds_half_units(quota=1000,units=1500)"
    assert size_skip_reason("full", 3) == ""


@pytest.mark.parametrize("size", [0, -5, "n50", "Full", 2.5, True, None])
def test_size_must_be_a_positive_int_or_full(size):
    with pytest.raises(SettingsError):
        size_skip_reason(size, 100)
    group_id, unit_id = make_ids(5, 2)
    with pytest.raises(SettingsError):
        draw_groups(np.arange(10), group_id, unit_id, size, rng_for("x"))
