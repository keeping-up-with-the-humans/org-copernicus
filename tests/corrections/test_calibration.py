"""Regression calibration and marginal matching."""

from __future__ import annotations

from types import MappingProxyType

import numpy as np
import pandas as pd
import pytest
from corrections_support import make_correction_data, planted_data
from scipy.stats import spearmanr

from copernicus.corrections import CorrectionData, cross_fit_predictions, get_correction
from copernicus.corrections.families import (
    LinePredictor,
    MarginalMatching,
    RegressionCalibration,
)
from copernicus.errors import ConfigurationError, EstimatorError, NotApplicableError, SettingsError

pytestmark = pytest.mark.filterwarnings("error")

RNG = np.random.default_rng(0)


def _fit(family: str, variant: str, data: CorrectionData, train_ids: np.ndarray | None = None):
    ids = data.labelled_ids if train_ids is None else train_ids
    return get_correction(family, variant).fit(data, ids, rng=np.random.default_rng(0))


def test_line_predictor_applies_its_line() -> None:
    line = LinePredictor(intercept=1.0, slope=2.0, column="score_good")
    assert line.predict(np.array([0.0, 1.5])).tolist() == [1.0, 4.0]
    with pytest.raises(AttributeError):
        line.slope = 3.0  # type: ignore[misc]


def test_linear_recovers_a_planted_line_on_noiseless_data() -> None:
    data = planted_data(lambda score, covariate: 1.0 + 2.0 * score)
    fitted = _fit("regression_calibration", "linear", data)
    assert fitted.provenance["score_column"] == "score_good"
    assert fitted.provenance["intercept"] == pytest.approx(1.0, abs=1e-6)
    assert fitted.provenance["slope"] == pytest.approx(2.0, abs=1e-6)
    predicted = fitted.predict(data, data.ids)
    assert predicted.index.tolist() == data.ids.tolist()
    np.testing.assert_allclose(
        predicted.to_numpy(), 1.0 + 2.0 * data.frame["score_good"], atol=1e-6
    )


def test_covariate_augmented_recovers_planted_coefficients() -> None:
    data = planted_data(lambda score, covariate: 1.0 + 2.0 * score - 0.5 * covariate)
    fitted = _fit("regression_calibration", "covariate_augmented", data)
    assert fitted.provenance["intercept"] == pytest.approx(1.0, abs=1e-6)
    assert fitted.provenance["slope"] == pytest.approx(2.0, abs=1e-6)
    assert fitted.provenance["covariate_coefficient"] == pytest.approx(-0.5, abs=1e-6)
    expected = 1.0 + 2.0 * data.frame["score_good"] - 0.5 * data.covariate
    np.testing.assert_allclose(fitted.predict(data, data.ids).to_numpy(), expected, atol=1e-6)


def test_covariate_augmented_refuses_a_constant_covariate() -> None:
    data = planted_data(lambda score, covariate: 1.0 + 2.0 * score)
    frame = data.frame.copy()
    frame["log_chars"] = 5.0
    constant = CorrectionData(frame=frame, score_columns=data.score_columns)
    with pytest.raises(EstimatorError, match="rank"):
        _fit("regression_calibration", "covariate_augmented", constant)


def test_the_score_column_is_the_best_one_on_the_training_rows() -> None:
    data = make_correction_data()
    for variant in ("linear", "covariate_augmented", "isotonic"):
        assert _fit("regression_calibration", variant, data).provenance["score_column"] == (
            "score_good"
        )
    swapped = CorrectionData(frame=data.frame, score_columns=("score_poor", "score_good"))
    assert _fit("regression_calibration", "linear", swapped).provenance["score_column"] == (
        "score_good"
    )


def test_a_constant_score_column_is_skipped_and_all_constant_is_refused() -> None:
    data = make_correction_data()
    frame = data.frame.copy()
    frame["score_good"] = 2.0
    one_good = CorrectionData(frame=frame, score_columns=("score_good", "score_poor"))
    assert _fit("regression_calibration", "linear", one_good).provenance["score_column"] == (
        "score_poor"
    )
    frame["score_poor"] = 3.0
    constant = CorrectionData(frame=frame, score_columns=("score_good", "score_poor"))
    for family, variant in (
        ("regression_calibration", "linear"),
        ("regression_calibration", "isotonic"),
        ("marginal_matching", "mean_sd"),
        ("marginal_matching", "quantile"),
    ):
        with pytest.raises(EstimatorError, match="no usable score column"):
            _fit(family, variant, constant)


def test_fit_uses_only_the_training_ids() -> None:
    data = planted_data(lambda score, covariate: 1.0 + 2.0 * score)
    frame = data.frame.copy()
    frame.loc[frame.index[:20], "gold"] = 100.0
    corrupted = CorrectionData(frame=frame, score_columns=data.score_columns)
    clean_ids = corrupted.ids[20:]
    fitted = _fit("regression_calibration", "linear", corrupted, clean_ids)
    assert fitted.provenance["slope"] == pytest.approx(2.0, abs=1e-6)


def test_fit_refuses_unlabelled_or_unknown_training_ids() -> None:
    data = make_correction_data()
    unlabelled = data.ids[~data.labelled_mask][:3]
    correction = get_correction("regression_calibration", "linear")
    with pytest.raises(ConfigurationError, match="no gold value"):
        correction.fit(data, np.concatenate([data.labelled_ids, unlabelled]), rng=RNG)
    with pytest.raises(ConfigurationError, match="not in the data"):
        correction.fit(data, np.array(["nope", *data.labelled_ids]), rng=RNG)


def test_fit_refuses_repeated_training_ids() -> None:
    data = make_correction_data()
    ids = data.labelled_ids
    with pytest.raises(ConfigurationError, match=r"repeat.*" + ids[0]):
        get_correction("regression_calibration", "linear").fit(
            data, np.concatenate([ids, ids[:2]]), rng=RNG
        )


def test_predict_refuses_unknown_ids() -> None:
    data = make_correction_data()
    fitted = _fit("regression_calibration", "linear", data)
    with pytest.raises(ConfigurationError, match="not in the data"):
        fitted.predict(data, np.array(["missing"]))


@pytest.mark.parametrize(
    ("family", "variant", "gold_kind"),
    [
        ("regression_calibration", "linear", "continuous"),
        ("regression_calibration", "covariate_augmented", "continuous"),
        ("regression_calibration", "platt", "binary"),
        ("regression_calibration", "isotonic", "continuous"),
        ("marginal_matching", "mean_sd", "continuous"),
        ("marginal_matching", "quantile", "continuous"),
    ],
)
def test_predict_names_the_score_column_the_data_lack(
    family: str, variant: str, gold_kind: str
) -> None:
    data = make_correction_data(gold_kind=gold_kind, n_groups=40)
    fitted = _fit(family, variant, data)
    assert fitted.provenance["score_column"] == "score_good"
    other = CorrectionData(
        frame=data.frame.drop(columns=["score_good"]), score_columns=("score_poor",)
    )
    with pytest.raises(ConfigurationError, match="score_good"):
        fitted.predict(other, other.ids[:3])


def test_provenance_is_read_only_and_names_the_family() -> None:
    fitted = _fit("regression_calibration", "linear", make_correction_data())
    assert isinstance(fitted.provenance, MappingProxyType)
    assert fitted.provenance["family"] == "regression_calibration"
    assert fitted.provenance["variant"] == "linear"
    with pytest.raises(TypeError):
        fitted.provenance["slope"] = 0.0  # type: ignore[index]


def test_platt_applies_only_to_binary_gold() -> None:
    correction = get_correction("regression_calibration", "platt")
    assert correction.applies(make_correction_data(gold_kind="binary")) == (True, "")
    applies, reason = correction.applies(make_correction_data(gold_kind="ordinal"))
    assert applies is False
    assert "binary" in reason and "5" in reason
    applies, reason = correction.applies(make_correction_data())
    assert applies is False and "binary" in reason
    for variant in ("linear", "covariate_augmented", "isotonic"):
        assert get_correction("regression_calibration", variant).applies(
            make_correction_data(gold_kind="ordinal")
        ) == (True, "")


def test_platt_predicts_the_probability_of_the_larger_gold_class() -> None:
    data = make_correction_data(gold_kind="binary", n_groups=40)
    fitted = _fit("regression_calibration", "platt", data)
    predicted = fitted.predict(data, data.ids)
    assert ((predicted > 0) & (predicted < 1)).all()
    assert fitted.provenance["positive_class"] == 1.0
    assert fitted.provenance["slope"] > 0
    high = data.frame["score_good"] > data.frame["score_good"].median()
    assert predicted[high.to_numpy()].mean() > predicted[~high.to_numpy()].mean()


def test_platt_predicts_on_the_gold_scale_for_gold_coded_one_and_two() -> None:
    zero_one = make_correction_data(gold_kind="binary", n_groups=40)
    frame = zero_one.frame.copy()
    frame["gold"] = frame["gold"] + 1.0
    coded = CorrectionData(frame=frame, score_columns=zero_one.score_columns)
    plain = _fit("regression_calibration", "platt", zero_one).predict(zero_one, zero_one.ids)
    shifted_fit = _fit("regression_calibration", "platt", coded)
    shifted = shifted_fit.predict(coded, coded.ids)
    np.testing.assert_allclose(shifted.to_numpy(), plain.to_numpy() + 1.0, atol=1e-9)
    assert ((shifted > 1.0) & (shifted < 2.0)).all()
    assert shifted_fit.provenance["negative_class"] == 1.0
    assert shifted_fit.provenance["positive_class"] == 2.0


def test_platt_refuses_training_gold_that_is_not_two_classes_as_applies_does() -> None:
    data = make_correction_data(gold_kind="binary")
    one_class = data.labelled_ids[data.gold[data.labelled_mask] == 1.0]
    refusal = "regression_calibration/platt does not apply to the training rows: "
    with pytest.raises(NotApplicableError) as refused:
        _fit("regression_calibration", "platt", data, one_class)
    assert str(refused.value) == (
        f"{refusal}platt calibration needs binary gold; the training gold values take 1 "
        "distinct values"
    )
    ordinal = make_correction_data(gold_kind="ordinal")
    _, reason = get_correction("regression_calibration", "platt").applies(ordinal)
    with pytest.raises(NotApplicableError) as refused:
        _fit("regression_calibration", "platt", ordinal)
    assert str(refused.value) == refusal + reason.replace("labelled", "training")


def test_isotonic_is_non_decreasing_and_clips_outside_the_training_range() -> None:
    data = make_correction_data(n_groups=30)
    fitted = _fit("regression_calibration", "isotonic", data)
    scores = data.frame["score_good"]
    probe_frame = pd.DataFrame(
        {
            "text_id": ["a", "b", "c"],
            "group_id": ["x", "x", "x"],
            "score_good": [scores.min() - 5.0, scores.median(), scores.max() + 5.0],
            "score_poor": 0.0,
            "gold": np.nan,
            "log_chars": 0.0,
        }
    )
    probe = CorrectionData(frame=probe_frame, score_columns=("score_good", "score_poor"))
    low, middle, high = fitted.predict(probe, probe.ids).to_numpy()
    train_gold = data.gold[data.labelled_mask]
    assert low <= middle <= high
    assert low >= train_gold.min() - 1e-12
    assert high <= train_gold.max() + 1e-12
    fitted_values = fitted.predict(data, data.ids).to_numpy()
    order = np.argsort(scores.to_numpy(), kind="stable")
    assert (np.diff(fitted_values[order]) >= -1e-12).all()


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    return float(spearmanr(a, b).statistic)


def _held_out_case(variant: str) -> tuple[CorrectionData, np.ndarray]:
    """Data whose training gold is noise-free (or binary) and whose held-out gold is noisy."""
    held_out = np.arange(5, 120, 10)
    rng = np.random.default_rng(21)

    def gold_of(score: np.ndarray, covariate: np.ndarray) -> np.ndarray:
        if variant == "platt":
            return (score + rng.normal(scale=0.8, size=score.size) > 2.0).astype(float)
        gold = score**3
        gold[held_out] += rng.normal(scale=2.0, size=held_out.size)
        return gold

    return planted_data(gold_of), held_out


@pytest.mark.parametrize("variant", ["linear", "isotonic", "platt"])
def test_monotone_calibrators_leave_the_held_out_rank_validity_unchanged(variant: str) -> None:
    data, held_out = _held_out_case(variant)
    score = data.frame["score_good"].to_numpy()
    train = np.setdiff1d(np.arange(data.n), held_out)
    assert {int(np.argmin(score)), int(np.argmax(score))} <= set(train.tolist())
    fitted = _fit("regression_calibration", variant, data, data.ids[train])
    predicted = fitted.predict(data, data.ids[held_out]).to_numpy()
    gold = data.gold[held_out]
    assert _spearman(predicted, gold) == pytest.approx(_spearman(score[held_out], gold), abs=1e-12)
    assert abs(_spearman(score[held_out], gold)) < 1.0


def test_covariate_augmented_can_move_the_rank_validity() -> None:
    data = planted_data(lambda score, covariate: score + 3.0 * covariate)
    fitted = _fit("regression_calibration", "covariate_augmented", data)
    predicted = fitted.predict(data, data.ids).to_numpy()
    assert _spearman(predicted, data.gold) > _spearman(data.frame["score_good"], data.gold) + 0.1


def test_mean_sd_matches_the_training_mean_and_population_sd() -> None:
    data = make_correction_data(n_groups=30)
    fitted = _fit("marginal_matching", "mean_sd", data)
    labelled = data.labelled_mask
    column = fitted.provenance["score_column"]
    train_score = data.frame.loc[labelled, column].to_numpy()
    train_gold = data.gold[labelled]
    predicted = fitted.predict(data, data.labelled_ids).to_numpy()
    assert predicted.mean() == pytest.approx(train_gold.mean(), abs=1e-9)
    assert predicted.std() == pytest.approx(train_gold.std(), abs=1e-9)
    assert fitted.provenance["slope"] == pytest.approx(train_gold.std() / train_score.std())
    assert fitted.provenance["comparability_only"] is True


def test_mean_sd_refuses_a_constant_training_score() -> None:
    from copernicus.corrections.families.marginal import fit_mean_sd_line

    with pytest.raises(EstimatorError, match="constant"):
        fit_mean_sd_line(np.full(10, 2.0), np.arange(10.0), "score_good")


def test_quantile_maps_the_training_scores_onto_the_sorted_gold() -> None:
    data = make_correction_data(n_groups=30)
    fitted = _fit("marginal_matching", "quantile", data)
    column = fitted.provenance["score_column"]
    labelled = data.labelled_mask
    train_score = data.frame.loc[labelled, column].to_numpy()
    mapped = fitted.predict(data, data.labelled_ids).to_numpy()
    np.testing.assert_allclose(np.sort(mapped), np.sort(data.gold[labelled]), atol=1e-12)
    assert _spearman(mapped, train_score) == pytest.approx(1.0)
    assert fitted.provenance["comparability_only"] is True


def _discrete_data() -> CorrectionData:
    rng = np.random.default_rng(5)
    score = rng.integers(1, 6, size=400).astype(float)
    frame = pd.DataFrame(
        {
            "text_id": [f"t{index:04d}" for index in range(400)],
            "group_id": [f"g{index // 5:03d}" for index in range(400)],
            "score_good": score,
            "score_poor": rng.normal(size=400),
            "gold": score + rng.normal(scale=0.3, size=400),
            "log_chars": rng.normal(size=400),
        }
    )
    return CorrectionData(frame=frame, score_columns=("score_good", "score_poor"))


def test_quantile_is_unbiased_for_tied_discrete_scores() -> None:
    data = _discrete_data()
    fitted = _fit("marginal_matching", "quantile", data)
    assert fitted.provenance["score_column"] == "score_good"
    mapped = fitted.predict(data, data.ids).to_numpy()
    assert mapped.mean() == pytest.approx(data.gold.mean(), abs=0.05)
    assert mapped.std() == pytest.approx(data.gold.std(), abs=0.1)
    by_score = pd.Series(mapped).groupby(data.frame["score_good"].to_numpy()).first()
    assert by_score.is_monotonic_increasing
    assert by_score.loc[5.0] < data.gold.max()


def test_quantile_clips_new_scores_to_the_end_quantiles_and_interpolates_between() -> None:
    data = _discrete_data()
    fitted = _fit("marginal_matching", "quantile", data)
    probe_frame = pd.DataFrame(
        {
            "text_id": list("abcde"),
            "group_id": "x",
            "score_good": [-10.0, 1.0, 2.5, 5.0, 99.0],
            "score_poor": 0.0,
            "gold": np.nan,
            "log_chars": 0.0,
        }
    )
    probe = CorrectionData(frame=probe_frame, score_columns=("score_good", "score_poor"))
    low, one, between, five, high = fitted.predict(probe, probe.ids).to_numpy()
    assert low == one and high == five
    two = fitted.predict(data, data.ids[data.frame["score_good"].to_numpy() == 2.0][:1])
    three = fitted.predict(data, data.ids[data.frame["score_good"].to_numpy() == 3.0][:1])
    assert two.iloc[0] < between < three.iloc[0]


def test_both_marginal_variants_say_they_match_marginals_only() -> None:
    assert "comparability" in (MarginalMatching.__doc__ or "").lower()
    assert "conditional mean" in (MarginalMatching.__doc__ or "").lower()


def test_variant_names_are_validated() -> None:
    with pytest.raises(SettingsError, match="unknown variant 'cubic'.*linear"):
        RegressionCalibration("cubic")
    with pytest.raises(SettingsError, match="unknown variant 'cubic'.*mean_sd"):
        MarginalMatching("cubic")


def test_cross_fitted_calibration_improves_the_level_of_a_displaced_score() -> None:
    data = make_correction_data(n_groups=30, label_all=True)
    corrected = cross_fit_predictions(get_correction("regression_calibration", "linear"), data)
    raw = data.frame["score_good"].to_numpy()
    assert np.abs(corrected.to_numpy() - data.gold).mean() < np.abs(raw - data.gold).mean()
