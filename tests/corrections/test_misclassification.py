"""The misclassification-matrix correction: applicability, posterior means, class shares."""

from __future__ import annotations

from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd
import pytest

from copernicus.corrections import CorrectionData, get_correction
from copernicus.corrections.families import MisclassificationMatrix
from copernicus.errors import ConfigurationError, NotApplicableError, SettingsError

pytestmark = pytest.mark.filterwarnings("error")

N_TEXTS = 3000
N_LABELLED = 1000
PREVALENCE = 0.3


def _data(
    classes: np.ndarray,
    shares: np.ndarray,
    *,
    separation: float,
    seed: int = 5,
    n: int = N_TEXTS,
    n_labelled: int = N_LABELLED,
) -> CorrectionData:
    """Gold classes drawn with the given shares; the score shifts by ``separation`` per class."""
    rng = np.random.default_rng(seed)
    gold = rng.choice(classes, size=n, p=shares)
    score = separation * np.searchsorted(classes, gold) + rng.normal(size=n)
    observed = gold.copy()
    observed[n_labelled:] = np.nan
    frame = pd.DataFrame(
        {
            "text_id": [f"t{index:05d}" for index in range(n)],
            "group_id": [f"g{index // 10:04d}" for index in range(n)],
            "score_good": score,
            "score_poor": rng.normal(size=n),
            "gold": observed,
            "log_chars": rng.normal(size=n),
        }
    )
    return CorrectionData(frame=frame, score_columns=("score_good", "score_poor"))


def _binary(**kwargs: Any) -> CorrectionData:
    return _data(np.array([0.0, 1.0]), np.array([1 - PREVALENCE, PREVALENCE]), **kwargs)


def _two_classes(score: np.ndarray, n: int) -> CorrectionData:
    """Fifty labelled texts per class, in class order, with the given score, among ``n`` texts."""
    gold = np.repeat([0.0, 1.0], 50)
    frame = pd.DataFrame(
        {
            "text_id": [f"t{index:04d}" for index in range(n)],
            "group_id": [f"g{index:04d}" for index in range(n)],
            "score": np.r_[score, np.ones(n - score.size)],
            "gold": np.r_[gold, np.full(n - gold.size, np.nan)],
            "log_chars": np.linspace(1.0, 2.0, n),
        }
    )
    return CorrectionData(frame=frame, score_columns=("score",))


def _fit(data: CorrectionData, **params: object) -> Any:
    correction = get_correction("misclassification_matrix", "posterior", **params)
    return correction.fit(data, data.labelled_ids, rng=np.random.default_rng(0))


def test_defaults_follow_the_registry_wording() -> None:
    correction = MisclassificationMatrix()
    assert (correction.youden_floor, correction.per_class_floor, correction.max_classes) == (
        0.3,
        30,
        5,
    )
    assert (correction.family, correction.variant, correction.target) == (
        "misclassification_matrix",
        "posterior",
        "A",
    )


@pytest.mark.parametrize(
    "params",
    [
        {"youden_floor": -0.1},
        {"youden_floor": 1.5},
        {"youden_floor": float("nan")},
        {"youden_floor": "0.3"},
        {"youden_floor": True},
        {"per_class_floor": 0},
        {"per_class_floor": -1},
        {"per_class_floor": 2.5},
        {"per_class_floor": "30"},
        {"max_classes": 1},
        {"max_classes": 3.0},
        {"max_classes": "5"},
        {"max_classes": True},
    ],
)
def test_settings_outside_their_range_are_refused(params: dict[str, object]) -> None:
    with pytest.raises(SettingsError, match=next(iter(params))):
        MisclassificationMatrix(**params)  # type: ignore[arg-type]


def test_the_edges_of_the_ranges_are_accepted() -> None:
    edges = MisclassificationMatrix(youden_floor=1, per_class_floor=1, max_classes=2)
    assert (edges.youden_floor, edges.per_class_floor, edges.max_classes) == (1.0, 1, 2)
    assert MisclassificationMatrix(youden_floor=0.0).youden_floor == 0.0
    typed = MisclassificationMatrix(per_class_floor=np.int64(12), max_classes=np.int64(3))  # type: ignore[arg-type]
    assert type(typed.per_class_floor) is int and type(typed.max_classes) is int


def test_an_unknown_variant_is_refused_by_the_factory() -> None:
    from copernicus.corrections.families.misclassification import make_misclassification

    with pytest.raises(SettingsError, match="posterior"):
        make_misclassification("cubic")


def test_applies_on_a_well_separated_binary_gold() -> None:
    assert MisclassificationMatrix().applies(_binary(separation=2.0)) == (True, "")


def test_applies_refuses_a_single_class_and_names_the_count() -> None:
    frame = _binary(separation=2.0).frame.copy()
    frame["gold"] = np.where(frame["gold"].isna(), np.nan, 1.0)
    applies, reason = MisclassificationMatrix().applies(
        CorrectionData(frame=frame, score_columns=("score_good", "score_poor"))
    )
    assert not applies
    assert "1 distinct" in reason


def test_applies_refuses_more_classes_than_the_maximum_and_names_the_count() -> None:
    classes = np.arange(6.0)
    data = _data(classes, np.full(6, 1 / 6), separation=1.5)
    applies, reason = MisclassificationMatrix().applies(data)
    assert not applies
    assert "6 distinct" in reason
    assert "5" in reason
    assert MisclassificationMatrix(max_classes=6).applies(data)[0]


def test_applies_refuses_a_thin_class_and_names_class_and_count() -> None:
    data = _data(
        np.array([0.0, 1.0, 2.0]),
        np.array([0.495, 0.495, 0.01]),
        separation=2.0,
        n=1500,
        n_labelled=1500,
    )
    count = int((data.gold == 2.0).sum())
    assert count < 30
    applies, reason = MisclassificationMatrix().applies(data)
    assert not applies
    assert f"class 2 has {count} labelled rows" in reason
    assert MisclassificationMatrix(per_class_floor=count).applies(data)[0]


def test_applies_refuses_a_low_youden_j_and_names_it() -> None:
    data = _binary(separation=0.2)
    applies, reason = MisclassificationMatrix().applies(data)
    assert not applies
    assert "informedness" in reason
    assert "0.3" in reason
    assert MisclassificationMatrix(youden_floor=0.0).applies(data)[0]


def test_applies_reports_when_no_score_column_is_usable() -> None:
    frame = _binary(separation=2.0).frame.copy()
    frame["score_good"] = 1.0
    frame["score_poor"] = 2.0
    applies, reason = MisclassificationMatrix().applies(
        CorrectionData(frame=frame, score_columns=("score_good", "score_poor"))
    )
    assert not applies
    assert "no usable score column" in reason


def test_a_planted_confusion_matrix_recovers_the_prevalence_within_three_points() -> None:
    data = _binary(separation=2.0)
    fitted = _fit(data)
    shares = fitted.class_shares(data, data.ids)
    truth = float((data.gold[:N_LABELLED] == 1.0).mean())
    assert truth == pytest.approx(PREVALENCE, abs=0.04)
    assert abs(shares["rogan_gladen"] - PREVALENCE) < 0.03
    assert set(shares["raw"]) == {0.0, 1.0}
    assert sum(shares["raw"].values()) == pytest.approx(1.0)
    assert 0.0 < shares["specificity"] < 1.0
    assert 0.0 < shares["sensitivity"] < 1.0


def test_rogan_gladen_corrects_a_biased_raw_share() -> None:
    train = _binary(separation=1.5, seed=8)
    fitted = _fit(train)
    drifted = _data(
        np.array([0.0, 1.0]), np.array([0.6, 0.4]), separation=1.5, seed=9, n_labelled=0
    )
    shares = fitted.class_shares(drifted, drifted.ids)
    raw_larger = shares["raw"][1.0]
    assert abs(shares["rogan_gladen"] - 0.4) < abs(raw_larger - 0.4)
    assert abs(shares["rogan_gladen"] - 0.4) < 0.05


def test_class_shares_keys_for_binary_and_three_classes() -> None:
    binary = _binary(separation=2.0)
    shares = _fit(binary).class_shares(binary, binary.ids)
    assert set(shares) == {"raw", "rogan_gladen", "sensitivity", "specificity"}
    three = _data(np.array([0.0, 1.0, 2.0]), np.array([0.3, 0.4, 0.3]), separation=2.0)
    shares = _fit(three).class_shares(three, three.ids)
    assert set(shares) == {"raw"}
    assert set(shares["raw"]) == {0.0, 1.0, 2.0}
    assert sum(shares["raw"].values()) == pytest.approx(1.0)


def test_rogan_gladen_is_nan_with_a_note_when_sensitivity_equals_the_false_positive_rate() -> None:
    # Two of four positives and two of four negatives fall above the median score, so the
    # smoothed rates are sensitivity 0.5 and specificity 0.5.
    score = np.array([2.0, 3.0, 6.0, 7.0, 1.0, 4.0, 5.0, 8.0])
    gold = np.array([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    data = CorrectionData(
        frame=pd.DataFrame(
            {
                "text_id": [f"t{index}" for index in range(8)],
                "group_id": [f"g{index // 4}" for index in range(8)],
                "score": score,
                "gold": gold,
                "log_chars": np.arange(8.0),
            }
        ),
        score_columns=("score",),
    )
    fitted = _fit(data, per_class_floor=4, youden_floor=0.0)
    shares = fitted.class_shares(data, data.ids)
    assert shares["sensitivity"] == 0.5
    assert shares["specificity"] == 0.5
    assert np.isnan(shares["rogan_gladen"])
    assert "sensitivity" in shares["rogan_gladen_note"]
    assert shares["raw"] == {0.0: 0.5, 1.0: 0.5}


def test_predict_returns_the_posterior_mean_of_the_discretised_score() -> None:
    data = _binary(separation=2.0)
    fitted = _fit(data)
    predicted = fitted.predict(data, data.ids)
    assert predicted.index.tolist() == data.ids.tolist()
    assert set(np.round(predicted.unique(), 12)) == set(np.round(fitted.posterior_means, 12))
    low, high = sorted(fitted.posterior_means)
    assert 0.0 < low < high < 1.0
    order = np.argsort(data.frame["score_good"].to_numpy(), kind="stable")
    assert np.all(np.diff(predicted.to_numpy()[order]) >= 0.0)


def test_posterior_means_follow_bayes_with_dirichlet_smoothing() -> None:
    data = _binary(separation=2.0)
    fitted = _fit(data)
    confusion = np.array(fitted.provenance["confusion_matrix"])
    priors = np.array(fitted.priors)
    for index in range(2):
        joint = confusion[index] * priors
        assert fitted.posterior_means[index] == pytest.approx((joint / joint.sum())[1])
    np.testing.assert_allclose(confusion.sum(axis=0), 1.0)


def test_provenance_records_the_classification() -> None:
    data = _binary(separation=2.0)
    provenance = _fit(data).provenance
    assert isinstance(provenance, MappingProxyType)
    assert provenance["family"] == "misclassification_matrix"
    assert provenance["variant"] == "posterior"
    assert provenance["score_column"] == "score_good"
    assert provenance["n_classes"] == 2
    assert provenance["classes"] == [0.0, 1.0]
    assert 0.3 < provenance["youden_j"] <= 1.0
    matrix = provenance["confusion_matrix"]
    assert isinstance(matrix, list) and isinstance(matrix[0], list)
    assert len(matrix) == 2 and len(matrix[0]) == 2


def test_fit_refuses_one_class_too_many_classes_and_an_unusable_score_on_its_rows() -> None:
    data = _binary(separation=2.0)
    one_class = data.labelled_ids[data.gold[:N_LABELLED] == 1.0]
    with pytest.raises(NotApplicableError, match="training rows: the labelled gold takes 1 dist"):
        get_correction("misclassification_matrix", "posterior").fit(
            data, one_class, rng=np.random.default_rng(0)
        )
    many = _data(np.arange(6.0), np.full(6, 1 / 6), separation=1.5)
    with pytest.raises(NotApplicableError, match="takes 6 distinct values; .* between 2 and 5"):
        _fit(many)
    flat = data.frame.assign(score_good=1.0, score_poor=2.0)
    with pytest.raises(NotApplicableError, match="no informedness .*: no usable score column"):
        _fit(CorrectionData(frame=flat, score_columns=data.score_columns))


def test_fit_refuses_training_rows_below_the_informedness_floor() -> None:
    # Youden's J of the cut score is (24 + 26) / 50 - 1 = 0.04 on the 100 labelled texts.
    score = np.r_[np.zeros(26), np.ones(24), np.zeros(24), np.ones(26)]
    data = _two_classes(score, 1000)
    correction = MisclassificationMatrix()
    applies, reason = correction.applies(data)
    assert not applies and reason.endswith("is 0.040, below the floor 0.3")
    floor = r"^misclassification_matrix/posterior does not apply to the training rows: the "
    with pytest.raises(
        NotApplicableError, match=floor + r"informedness .* 0\.040, below the floor 0\.3$"
    ):
        correction.fit(data, data.labelled_ids, rng=np.random.default_rng(0))
    fitted = _fit(data, youden_floor=0.0)
    assert fitted.provenance["youden_j"] == pytest.approx(0.04)


def test_fit_judges_its_training_rows_after_applies_passed_on_the_whole_sample() -> None:
    data = _two_classes(np.repeat([0.0, 1.0], 50) + np.linspace(-0.1, 0.1, 100), 100)
    correction = MisclassificationMatrix()
    assert correction.applies(data) == (True, "")
    train = np.r_[data.ids[:3], data.ids[50:53]]
    thin = r"training rows: class 0 has 3 labelled rows; at least 30 are needed$"
    with pytest.raises(NotApplicableError, match=thin):
        correction.fit(data, train, rng=np.random.default_rng(0))
    fitted = get_correction("misclassification_matrix", "posterior", per_class_floor=3).fit(
        data, train, rng=np.random.default_rng(0)
    )
    assert (fitted.provenance["youden_j"], fitted.provenance["n_classes"]) == (1.0, 2)


def test_class_shares_refuses_an_empty_id_list() -> None:
    data = _binary(separation=2.0)
    with pytest.raises(ConfigurationError, match="at least one"):
        _fit(data).class_shares(data, np.array([], dtype=str))


def test_gold_coded_one_and_two_keys_the_shares_by_those_values() -> None:
    data = _binary(separation=2.0)
    frame = data.frame.assign(gold=data.frame["gold"] + 1.0)
    shifted = CorrectionData(frame=frame, score_columns=data.score_columns)
    assert shifted.gold_classes.tolist() == [1.0, 2.0]
    assert MisclassificationMatrix().applies(shifted) == (True, "")
    fitted = _fit(shifted)
    shares = fitted.class_shares(shifted, shifted.ids)
    assert set(shares["raw"]) == {1.0, 2.0}
    assert abs(shares["rogan_gladen"] - PREVALENCE) < 0.03
    assert fitted.provenance["classes"] == [1.0, 2.0]
    assert set(np.round(fitted.predict(shifted, shifted.ids).unique(), 12)) == set(
        np.round(fitted.posterior_means, 12)
    )
    assert 1.0 < min(fitted.posterior_means) < max(fitted.posterior_means) < 2.0


def test_the_rogan_gladen_share_is_clipped_to_the_unit_interval() -> None:
    data = _binary(separation=2.0)
    fitted = _fit(data)
    high = CorrectionData(
        frame=data.frame.assign(score_good=100.0 + data.frame["score_good"]),
        score_columns=data.score_columns,
    )
    low = CorrectionData(
        frame=data.frame.assign(score_good=-100.0 + data.frame["score_good"]),
        score_columns=data.score_columns,
    )
    above = fitted.class_shares(high, high.ids)
    below = fitted.class_shares(low, low.ids)
    assert above["raw"][1.0] == 1.0
    assert above["rogan_gladen"] == 1.0
    assert below["raw"][1.0] == 0.0
    assert below["rogan_gladen"] == 0.0


def test_predict_names_the_score_column_the_data_lack() -> None:
    data = _binary(separation=2.0)
    fitted = _fit(data)
    other = CorrectionData(
        frame=data.frame.drop(columns=["score_good"]), score_columns=("score_poor",)
    )
    with pytest.raises(ConfigurationError, match="score_good"):
        fitted.predict(other, other.ids[:3])
