"""Grouped folds, the holdout split, inner splits, and cross-fitted predictions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from corrections_support import make_correction_data

from copernicus.corrections import (
    MIN_TRAIN,
    CorrectionData,
    cross_fit_predictions,
    get_correction,
    grouped_folds,
    holdout_split,
    inner_k,
    inner_splits,
)
from copernicus.corrections.families import ScoreCorrection
from copernicus.errors import (
    ConfigurationError,
    EstimatorError,
    NotApplicableError,
    SettingsError,
)

pytestmark = pytest.mark.filterwarnings("error")


class FixedOrder:
    """A stand-in generator whose permutation is the identity or its reverse."""

    def __init__(self, reverse: bool = False) -> None:
        self.reverse = reverse

    def permutation(self, n: int) -> np.ndarray:
        order = np.arange(n)
        return order[::-1] if self.reverse else order


def _groups(sizes: list[int]) -> np.ndarray:
    return np.repeat([f"g{index:03d}" for index in range(len(sizes))], sizes)


def test_grouped_folds_partition_the_rows_and_keep_groups_whole() -> None:
    groups = _groups([4, 3, 5, 2, 6, 4, 3])
    folds = grouped_folds(groups, 3, np.random.default_rng(0))
    assert len(folds) == 3
    tested = np.concatenate([test for _, test in folds])
    assert sorted(tested.tolist()) == list(range(groups.size))
    for train, test in folds:
        assert not set(groups[train]) & set(groups[test])
        assert train.size + test.size == groups.size


def test_grouped_folds_refuse_more_folds_than_groups() -> None:
    with pytest.raises(ConfigurationError, match="at least k groups"):
        grouped_folds(_groups([3, 3]), 5, np.random.default_rng(0))


def test_holdout_split_respects_whole_groups_and_the_band() -> None:
    sizes = [int(size) for size in np.random.default_rng(1).integers(4, 12, size=30)]
    groups = _groups(sizes)
    for seed in range(40):
        train, test = holdout_split(groups, np.random.default_rng(seed))
        assert not set(groups[train]) & set(groups[test])
        assert sorted(np.concatenate([train, test]).tolist()) == list(range(groups.size))
        assert 0.75 <= train.size / groups.size <= 0.85


def test_holdout_split_takes_the_prefix_nearest_the_share() -> None:
    groups = _groups([10] * 10)
    train, test = holdout_split(groups, np.random.default_rng(3))
    assert (train.size, test.size) == (80, 20)
    assert np.unique(groups[test]).size == 2


def test_holdout_split_breaks_a_tie_toward_the_smaller_training_side() -> None:
    groups = _groups([76, 4, 10, 10])
    train, test = holdout_split(groups, FixedOrder(), share=0.2, band=(0.7, 0.9))
    assert (train.size, test.size) == (80, 20)
    equal_distance = _groups([1, 2, 1])
    train, _ = holdout_split(equal_distance, FixedOrder(), share=0.5, band=(0.0, 1.0))
    assert train.size == 1


def test_holdout_split_refuses_a_share_below_the_band() -> None:
    groups = _groups([98, 1, 1])
    with pytest.raises(EstimatorError, match=r"0\.02.*0\.75.*0\.85.*3 groups.*100 rows"):
        holdout_split(groups, FixedOrder(reverse=True))


def test_holdout_split_refuses_a_share_above_the_band() -> None:
    groups = _groups([98, 1, 1])
    with pytest.raises(EstimatorError, match=r"0\.98.*0\.75.*0\.85.*3 groups.*100 rows"):
        holdout_split(groups, FixedOrder())


def test_holdout_split_refuses_fewer_than_two_groups() -> None:
    with pytest.raises(ConfigurationError, match="two groups"):
        holdout_split(_groups([12]), np.random.default_rng(0))


def test_holdout_split_refuses_bad_arguments() -> None:
    groups = _groups([5] * 10)
    with pytest.raises(SettingsError, match="share"):
        holdout_split(groups, np.random.default_rng(0), share=0.0)
    with pytest.raises(SettingsError, match="band"):
        holdout_split(groups, np.random.default_rng(0), band=(0.9, 0.8))


def test_inner_k_is_five_at_most() -> None:
    assert [inner_k(n) for n in (2, 3, 5, 6, 40)] == [2, 3, 5, 5, 5]
    with pytest.raises(ConfigurationError, match="at least one group"):
        inner_k(0)


def test_inner_splits_partition_and_refuse_a_single_group() -> None:
    groups = _groups([5, 5, 5])
    folds = inner_splits(groups, inner_k(3), np.random.default_rng(0))
    assert len(folds) == 3
    for train, test in folds:
        assert not set(groups[train]) & set(groups[test])
    with pytest.raises(ConfigurationError, match="two groups"):
        inner_splits(_groups([9]), 1, np.random.default_rng(0))


def test_cross_fit_covers_every_id_exactly_once_in_data_order() -> None:
    data = make_correction_data()
    predictions = cross_fit_predictions(get_correction("regression_calibration", "linear"), data)
    assert isinstance(predictions, pd.Series)
    assert predictions.index.tolist() == data.ids.tolist()
    assert predictions.notna().all()
    assert predictions.dtype == float


def test_cross_fit_never_lets_a_fit_see_the_gold_of_the_text_it_predicts() -> None:
    data = make_correction_data()
    seen: list[tuple[set[str], set[str]]] = []

    class Recorder:
        family = "recorder"
        variant = "mean"
        target = "A"

        def applies(self, data: CorrectionData) -> tuple[bool, str]:
            return True, ""

        def fit(
            self, data: CorrectionData, train_ids: np.ndarray, *, rng: np.random.Generator
        ) -> object:
            fold_train = set(train_ids.tolist())
            mean = float(np.nanmean(data.gold[np.isin(data.ids, train_ids)]))

            class Constant:
                provenance = {"family": "recorder"}

                def predict(self, data: CorrectionData, ids: np.ndarray) -> pd.Series:
                    seen.append((fold_train, set(ids.tolist())))
                    return pd.Series(np.full(ids.size, mean), index=ids)

            return Constant()

    cross_fit_predictions(Recorder(), data)  # type: ignore[arg-type]
    assert len(seen) == 5
    labelled = set(data.labelled_ids.tolist())
    for train, predicted in seen:
        assert train <= labelled
        groups_of = dict(zip(data.ids.tolist(), data.groups.tolist(), strict=True))
        assert not {groups_of[i] for i in train} & {groups_of[i] for i in predicted}
        assert not train & predicted
    assert sorted(set().union(*[predicted for _, predicted in seen])) == sorted(data.ids.tolist())


def test_cross_fit_is_deterministic_in_the_seed() -> None:
    data = make_correction_data()
    correction = get_correction("regression_calibration", "isotonic")
    first = cross_fit_predictions(correction, data, seed=5)
    again = cross_fit_predictions(correction, data, seed=5)
    other = cross_fit_predictions(correction, data, seed=6)
    pd.testing.assert_series_equal(first, again)
    assert not np.allclose(first, other)


def test_cross_fit_raises_below_the_minimum_of_labelled_training_rows() -> None:
    data = make_correction_data(n_groups=10, group_size=3, label_all=True)
    frame = data.frame.copy()
    frame.loc[np.arange(len(frame)) >= MIN_TRAIN - 1, "gold"] = np.nan
    sparse = CorrectionData(frame=frame, score_columns=data.score_columns)
    assert sparse.n_labelled == MIN_TRAIN - 1
    with pytest.raises(
        EstimatorError, match=rf"fold 0 has \d+ labelled training rows.*{MIN_TRAIN}"
    ):
        cross_fit_predictions(get_correction("regression_calibration", "linear"), sparse, k=2)


def test_cross_fit_refuses_a_correction_that_does_not_apply() -> None:
    data = make_correction_data(gold_kind="ordinal")
    with pytest.raises(NotApplicableError, match="binary"):
        cross_fit_predictions(get_correction("regression_calibration", "platt"), data)


def test_cross_fit_checks_applies_on_the_training_rows_of_every_fold() -> None:
    binary = make_correction_data(n_groups=40, group_size=3, gold_kind="binary", label_all=True)
    counts = [int((binary.gold == value).sum()) for value in binary.gold_classes]
    assert counts == [60, 60]
    correction = get_correction("misclassification_matrix", "posterior", per_class_floor=55)
    assert correction.applies(binary) == (True, "")
    with pytest.raises(NotApplicableError, match=r"training rows of fold 0: class .* at least 55"):
        cross_fit_predictions(correction, binary, k=5, seed=1)
    lenient = get_correction("misclassification_matrix", "posterior", per_class_floor=40)
    assert cross_fit_predictions(lenient, binary, k=5, seed=1).notna().all()


def test_each_fold_is_fitted_on_data_that_carry_only_its_training_gold() -> None:
    data = make_correction_data()
    checked: list[int] = []
    fitted_on: list[tuple[set[str], set[str]]] = []

    class Recorder:
        family = "recorder"
        variant = "mean"
        target = "A"

        def applies(self, data: CorrectionData) -> tuple[bool, str]:
            checked.append(data.n_labelled)
            return True, ""

        def fit(
            self, data: CorrectionData, train_ids: np.ndarray, *, rng: np.random.Generator
        ) -> object:
            fitted_on.append((set(data.labelled_ids.tolist()), set(train_ids.tolist())))

            class Constant:
                provenance = {"family": "recorder"}

                def predict(self, data: CorrectionData, ids: np.ndarray) -> pd.Series:
                    assert not set(data.labelled_ids.tolist()) & set(ids.tolist())
                    return pd.Series(np.zeros(ids.size), index=ids)

            return Constant()

    cross_fit_predictions(Recorder(), data)  # type: ignore[arg-type]
    assert checked[0] == data.n_labelled and len(checked) == 6
    assert all(count < data.n_labelled for count in checked[1:])
    assert all(labelled == train for labelled, train in fitted_on) and len(fitted_on) == 5


def test_cross_fit_refuses_more_folds_than_groups() -> None:
    data = make_correction_data(n_groups=3)
    with pytest.raises(ConfigurationError, match="at least k groups"):
        cross_fit_predictions(get_correction("regression_calibration", "linear"), data, k=5)


def test_protocol_is_satisfied_by_a_registered_correction() -> None:
    correction: ScoreCorrection = get_correction("regression_calibration", "linear")
    assert (correction.family, correction.variant, correction.target) == (
        "regression_calibration",
        "linear",
        "A",
    )
