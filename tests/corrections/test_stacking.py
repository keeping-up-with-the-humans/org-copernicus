"""Supervised stacking: a penalised linear learner over every score column."""

from __future__ import annotations

from types import MappingProxyType

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr
from sklearn.linear_model import ElasticNetCV, LassoCV

from copernicus.baseline.folds import rng_for
from copernicus.baseline.ridge import ALPHAS, RidgeDesign
from copernicus.baseline.ridge import predict as ridge_predict
from copernicus.corrections import (
    CorrectionData,
    best_score_column,
    cross_fit_predictions,
    get_correction,
    inner_k,
    inner_splits,
)
from copernicus.corrections.families import SupervisedStacking
from copernicus.corrections.families._factor import standardize_apply, standardize_fit
from copernicus.corrections.families.stacking import StackingFitted, _alpha_count_keyword
from copernicus.errors import ConfigurationError, EstimatorError, NotApplicableError, SettingsError

# Warnings are errors here, so a ConvergenceWarning of LassoCV or ElasticNetCV would fail a
# test. None is reachable on these data: the four standardised features are few and only
# moderately correlated (the two score columns about 0.55), so coordinate descent meets the
# duality-gap tolerance 1e-3 within at most a few hundred passes at every penalty of the
# 12-point path, far below max_iter = 1500.
pytestmark = pytest.mark.filterwarnings("error")

VARIANTS = ("ridge", "lasso", "enet")
N_GROUPS = 60
GROUP_SIZE = 5
TRAIN_GROUPS = 40
SEED_BOUND = 2**31 - 1


def _data(seed: int = 3) -> CorrectionData:
    """Two score columns that each carry the signal with independent noise; gold is the signal."""
    rng = np.random.default_rng(seed)
    n = N_GROUPS * GROUP_SIZE
    signal = rng.normal(size=n)
    frame = pd.DataFrame(
        {
            "text_id": [f"t{index:04d}" for index in range(n)],
            "group_id": [f"g{index // GROUP_SIZE:03d}" for index in range(n)],
            "score_a": signal + rng.normal(scale=0.9, size=n) + 3.0,
            "score_b": 2.0 * signal + rng.normal(scale=1.8, size=n) - 1.0,
            "gold": signal + rng.normal(scale=0.3, size=n),
            "log_chars": rng.normal(loc=5.0, scale=0.4, size=n),
        }
    )
    return CorrectionData(frame=frame, score_columns=("score_a", "score_b"))


TRAIN = np.arange(TRAIN_GROUPS * GROUP_SIZE)
HELD = np.arange(TRAIN_GROUPS * GROUP_SIZE, N_GROUPS * GROUP_SIZE)


def _fit(variant: str, data: CorrectionData, seed: int = 0) -> StackingFitted:
    correction = get_correction("supervised_stacking", variant)
    return correction.fit(data, data.ids[TRAIN], rng=np.random.default_rng(seed))


def _spread(scores: np.ndarray) -> np.ndarray:
    """The row-wise population SD of the scores standardised on the training rows."""
    centre, scale = standardize_fit(scores[TRAIN])
    return standardize_apply(scores, centre, scale).std(axis=1)


def _designs(data: CorrectionData) -> tuple[np.ndarray, np.ndarray]:
    """Scores, covariate and their spread of the training and held-out rows.

    Both are standardised with the mean and SD of the training rows.
    """
    scores = data.frame[["score_a", "score_b"]].to_numpy()
    features = np.column_stack([scores, data.covariate, _spread(scores)])
    mean, sd = standardize_fit(features[TRAIN])
    return standardize_apply(features[TRAIN], mean, sd), standardize_apply(features[HELD], mean, sd)


def _spearman(first: np.ndarray, second: np.ndarray) -> float:
    return float(spearmanr(first, second).statistic)


def test_ridge_stacking_beats_the_best_single_column_on_held_out_groups() -> None:
    data = _data()
    fitted = _fit("ridge", data)
    column = best_score_column(data.scores.iloc[TRAIN].reset_index(drop=True), data.gold[TRAIN])
    gold = data.gold[HELD]
    stacked = _spearman(fitted.predict(data, data.ids[HELD]).to_numpy(), gold)
    raw = _spearman(data.frame[column].to_numpy()[HELD], gold)
    assert stacked > raw + 0.05


def test_the_ridge_penalty_is_the_exact_grouped_leave_one_group_out_choice() -> None:
    data = _data()
    fitted = _fit("ridge", data)
    training, held_out = _designs(data)
    reference = RidgeDesign(training, data.gold[TRAIN], data.groups[TRAIN]).fit(ALPHAS)
    np.testing.assert_allclose(
        fitted.predict(data, data.ids[HELD]).to_numpy(),
        ridge_predict(reference, held_out),
        rtol=0.0,
        atol=1e-12,
    )
    provenance = fitted.provenance
    assert provenance["penalty_alpha"] == reference.alpha
    np.testing.assert_allclose(provenance["coefficients"], reference.coef, rtol=0.0, atol=1e-12)
    assert provenance["intercept"] == pytest.approx(reference.intercept, abs=1e-12)
    assert provenance["alpha_selection"] == "grouped_logo"
    assert provenance["k_inner_folds"] == TRAIN_GROUPS
    assert provenance["n_inner_groups"] == TRAIN_GROUPS
    assert provenance["l1_ratio"] == 0.0


@pytest.mark.parametrize("variant", ["lasso", "enet"])
def test_lasso_and_enet_choose_their_penalty_on_the_recorded_grouped_inner_splits(
    variant: str,
) -> None:
    data = _data()
    fitted = _fit(variant, data)
    provenance = fitted.provenance
    seed = provenance["inner_seed"]
    groups = data.groups[TRAIN]
    splits = inner_splits(
        groups, inner_k(TRAIN_GROUPS), rng_for(seed, "stacking", variant, "inner_splits")
    )
    for inner_train, inner_test in splits:
        assert np.intersect1d(groups[inner_train], groups[inner_test]).size == 0
    settings = {"cv": splits, "random_state": seed, "max_iter": 1500, "tol": 1e-3}
    count = {_alpha_count_keyword(): 12}
    training, held_out = _designs(data)
    model = (
        LassoCV(**settings, **count)
        if variant == "lasso"
        else ElasticNetCV(l1_ratio=[0.5, 0.9], **settings, **count)
    ).fit(training, data.gold[TRAIN])
    np.testing.assert_allclose(
        fitted.predict(data, data.ids[HELD]).to_numpy(),
        model.predict(held_out),
        rtol=0.0,
        atol=1e-12,
    )
    assert provenance["penalty_alpha"] == pytest.approx(float(model.alpha_), rel=1e-12)
    np.testing.assert_allclose(provenance["coefficients"], model.coef_, rtol=0.0, atol=1e-12)
    expected_ratio = 1.0 if variant == "lasso" else float(model.l1_ratio_)
    assert provenance["l1_ratio"] == expected_ratio
    assert provenance["l1_ratio"] in (0.5, 0.9, 1.0)
    assert provenance["alpha_selection"] == "grouped_kfold"
    assert provenance["k_inner_folds"] == 5
    assert provenance["n_inner_groups"] == TRAIN_GROUPS


@pytest.mark.parametrize("variant", VARIANTS)
def test_provenance_records_the_features_and_is_read_only(variant: str) -> None:
    fitted = _fit(variant, _data())
    provenance = fitted.provenance
    assert isinstance(provenance, MappingProxyType)
    assert (provenance["family"], provenance["variant"]) == ("supervised_stacking", variant)
    assert provenance["score_column"] == "all"
    assert provenance["n_features"] == 4
    assert provenance["feature_names"] == ["score_a", "score_b", "log_chars", "score_spread"]
    assert len(provenance["coefficients"]) == 4
    assert provenance["penalty_alpha"] > 0.0
    with pytest.raises(TypeError):
        provenance["penalty_alpha"] = 1.0  # type: ignore[index]


def test_the_inner_seed_is_drawn_from_the_fit_generator_and_makes_the_fit_repeatable() -> None:
    data = _data()
    first, second = _fit("enet", data, seed=5), _fit("enet", data, seed=5)
    assert first.provenance["inner_seed"] == int(np.random.default_rng(5).integers(SEED_BOUND))
    np.testing.assert_array_equal(
        first.predict(data, data.ids).to_numpy(), second.predict(data, data.ids).to_numpy()
    )
    assert _fit("enet", data, seed=6).provenance["inner_seed"] != first.provenance["inner_seed"]


@pytest.mark.parametrize("variant", VARIANTS)
def test_a_texts_prediction_does_not_depend_on_the_other_ids_passed(variant: str) -> None:
    data = _data()
    fitted = _fit(variant, data)
    full = fitted.predict(data, data.ids).to_numpy()
    for picked in ([5, 250, 17], [299], list(range(299, -1, -1))):
        part = fitted.predict(data, data.ids[picked])
        assert part.index.tolist() == data.ids[picked].tolist()
        np.testing.assert_allclose(part.to_numpy(), full[picked], rtol=0.0, atol=1e-12)


@pytest.mark.parametrize("variant", VARIANTS)
def test_the_fit_keeps_its_numbers_and_predicts_no_text_as_an_empty_series(variant: str) -> None:
    data = _data()
    fitted = _fit(variant, data)
    assert fitted.alpha == fitted.provenance["penalty_alpha"]
    assert fitted.intercept == fitted.provenance["intercept"]
    np.testing.assert_array_equal(fitted.coefficients, fitted.provenance["coefficients"])
    for no_ids in (np.array([], dtype=str), []):
        empty = fitted.predict(data, np.asarray(no_ids))
        assert empty.empty and empty.dtype == float and empty.index.tolist() == []


@pytest.mark.parametrize("variant", VARIANTS)
def test_cross_fitting_corrects_every_text_of_partly_labelled_data(variant: str) -> None:
    data = _data()
    frame = data.frame.copy()
    frame.loc[frame.index % 3 == 0, "gold"] = np.nan
    partial = CorrectionData(frame=frame, score_columns=data.score_columns)
    corrected = cross_fit_predictions(get_correction("supervised_stacking", variant), partial)
    assert corrected.index.tolist() == partial.ids.tolist()
    assert np.isfinite(corrected.to_numpy()).all()
    assert _spearman(corrected.to_numpy(), data.gold) > _spearman(
        data.frame["score_a"].to_numpy(), data.gold
    )


def test_a_constant_score_column_is_tolerated_and_gets_no_ridge_weight() -> None:
    data = _data()
    frame = data.frame.copy()
    frame["score_flat"] = 2.0
    flat = CorrectionData(frame=frame, score_columns=("score_a", "score_b", "score_flat"))
    provenance = _fit("ridge", flat).provenance
    assert provenance["n_features"] == 5
    assert provenance["coefficients"][2] == 0.0


def test_the_spread_feature_is_the_sd_of_the_training_standardised_scores() -> None:
    data = _data()
    fitted = _fit("ridge", data)
    scores = data.frame[["score_a", "score_b"]].to_numpy()
    standardised = (scores - scores[TRAIN].mean(axis=0)) / scores[TRAIN].std(axis=0)
    spread = np.abs(standardised[TRAIN, 0] - standardised[TRAIN, 1]) / 2.0
    assert fitted.mean[3] == pytest.approx(spread.mean())
    assert fitted.sd[3] == pytest.approx(spread.std())


@pytest.mark.parametrize("variant", VARIANTS)
def test_rescaling_a_score_column_leaves_the_predictions_unchanged(variant: str) -> None:
    data = _data()
    rescaled = CorrectionData(
        frame=data.frame.assign(score_b=100.0 * data.frame["score_b"] + 500.0),
        score_columns=data.score_columns,
    )
    original = _fit(variant, data).predict(data, data.ids[HELD]).to_numpy()
    changed = _fit(variant, rescaled).predict(rescaled, rescaled.ids[HELD]).to_numpy()
    np.testing.assert_allclose(changed, original, rtol=0.0, atol=1e-9)


def test_applies_needs_two_score_columns_and_two_labelled_groups() -> None:
    data = _data()
    assert SupervisedStacking("ridge").applies(data) == (True, "")
    single = CorrectionData(frame=data.frame, score_columns=("score_a",))
    applies, reason = SupervisedStacking("lasso").applies(single)
    assert not applies
    assert "two score columns" in reason and "1" in reason
    frame = data.frame.copy()
    frame.loc[frame["group_id"] != "g000", "gold"] = np.nan
    one_group = CorrectionData(frame=frame, score_columns=data.score_columns)
    applies, reason = SupervisedStacking("enet").applies(one_group)
    assert not applies
    assert "two groups" in reason and "1" in reason


def test_fit_on_training_rows_of_one_group_carries_the_frame_of_every_refusal() -> None:
    data = _data()
    one_group = data.ids[data.groups == "g001"]
    with pytest.raises(NotApplicableError) as refused:
        SupervisedStacking("ridge").fit(data, one_group, rng=np.random.default_rng(0))
    message = str(refused.value)
    assert message.startswith("supervised_stacking/ridge does not apply: ")
    assert "two groups among the training rows; they hold 1" in message


def test_fit_refuses_one_score_column_one_group_and_constant_features_or_gold() -> None:
    data = _data()
    rng = np.random.default_rng(0)
    single = CorrectionData(frame=data.frame, score_columns=("score_a",))
    _, reason = SupervisedStacking("ridge").applies(single)
    with pytest.raises(NotApplicableError) as refused:
        SupervisedStacking("ridge").fit(single, single.ids[TRAIN], rng=rng)
    assert str(refused.value) == f"supervised_stacking/ridge does not apply: {reason}"
    one_group = data.ids[data.groups == "g001"]
    for variant in VARIANTS:
        with pytest.raises(NotApplicableError, match="two groups among the training rows"):
            SupervisedStacking(variant).fit(data, one_group, rng=rng)
    frame = data.frame.copy()
    frame["gold"] = 1.5
    with pytest.raises(EstimatorError, match="gold values are constant"):
        SupervisedStacking("ridge").fit(
            CorrectionData(frame=frame, score_columns=data.score_columns), data.ids[TRAIN], rng=rng
        )
    frame = data.frame.copy()
    frame[["score_a", "score_b", "log_chars"]] = 1.0
    with pytest.raises(EstimatorError, match="every feature is constant"):
        SupervisedStacking("lasso").fit(
            CorrectionData(frame=frame, score_columns=data.score_columns), data.ids[TRAIN], rng=rng
        )


def test_fit_refuses_unlabelled_unknown_and_repeated_training_ids() -> None:
    data = _data()
    frame = data.frame.copy()
    frame.loc[frame.index[:3], "gold"] = np.nan
    partial = CorrectionData(frame=frame, score_columns=data.score_columns)
    correction, rng = SupervisedStacking("ridge"), np.random.default_rng(0)
    with pytest.raises(ConfigurationError, match="no gold value"):
        correction.fit(partial, partial.ids[TRAIN], rng=rng)
    with pytest.raises(ConfigurationError, match="not in the data"):
        correction.fit(data, np.array(["nope", *data.ids[TRAIN]]), rng=rng)
    with pytest.raises(ConfigurationError, match="repeat"):
        correction.fit(data, np.concatenate([data.ids[TRAIN], data.ids[:2]]), rng=rng)


def test_an_unknown_variant_is_a_settings_error() -> None:
    with pytest.raises(SettingsError, match="unknown variant 'cubic'.*ridge"):
        SupervisedStacking("cubic")


def test_predict_names_a_score_column_the_data_lack() -> None:
    data = _data()
    fitted = _fit("lasso", data)
    other = CorrectionData(frame=data.frame.drop(columns=["score_b"]), score_columns=("score_a",))
    with pytest.raises(ConfigurationError, match="score_b"):
        fitted.predict(other, other.ids[:3])
    with pytest.raises(ConfigurationError, match="not in the data"):
        fitted.predict(data, np.array(["missing"]))


@pytest.mark.parametrize(
    ("version", "keyword"),
    [("1.3.2", "n_alphas"), ("1.6.1", "n_alphas"), ("1.7.0", "alphas"), ("1.10.dev0", "alphas")],
)
def test_the_number_of_penalties_is_passed_under_the_keyword_of_the_installed_version(
    version: str, keyword: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sklearn

    monkeypatch.setattr(sklearn, "__version__", version)
    assert _alpha_count_keyword() == keyword


def test_an_unreadable_scikit_learn_version_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    import sklearn

    monkeypatch.setattr(sklearn, "__version__", "unknown")
    with pytest.raises(ConfigurationError, match="scikit-learn version"):
        _alpha_count_keyword()
