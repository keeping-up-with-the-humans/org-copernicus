"""Prediction-powered means and group differences, the estimate container, and the split."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest
from estimates_support import (
    TRUE_DIFFERENCE,
    TRUE_MEAN,
    labels_by_srs,
    ppi_frame,
    with_labels,
)

from copernicus.corrections import (
    CorrectionData,
    Estimate,
    group_difference_ppi,
    long_short_split,
    mean_ppi,
)
from copernicus.corrections.estimates import CI_LEVEL, Z, make_estimate, naive_se, plugin_value
from copernicus.errors import ConfigurationError, EstimatorError

pytestmark = pytest.mark.filterwarnings("error")

N_TEXTS = 40
MONTE_CARLO_SEED = 5


def _data(
    labelled_every: int = 2, *, pi: float | None = 0.5, seed: int = 3
) -> tuple[CorrectionData, pd.DataFrame]:
    """Forty texts with gold on every ``labelled_every``-th row, and the full frame."""
    rng = np.random.default_rng(seed)
    frame = ppi_frame(rng, N_TEXTS)
    labelled = np.arange(N_TEXTS) % labelled_every == 0
    return with_labels(frame, labelled, score_columns=("score",), pi=pi), frame


def test_the_estimate_is_frozen_with_a_read_only_details_mapping() -> None:
    estimate = make_estimate(
        2.0, 0.5, n_labelled=10, n_total=40, method="probe", details={"key": 1.0}
    )
    assert (CI_LEVEL, Z) == (0.95, 1.96)
    assert isinstance(estimate, Estimate)
    assert estimate.ci_low == pytest.approx(2.0 - 1.96 * 0.5)
    assert estimate.ci_high == pytest.approx(2.0 + 1.96 * 0.5)
    with pytest.raises(FrozenInstanceError):
        estimate.point = 3.0  # type: ignore[misc]
    with pytest.raises(TypeError):
        estimate.details["key"] = 2.0  # type: ignore[index]


def test_on_a_full_draw_the_ppi_estimates_equal_the_gold_estimates_exactly() -> None:
    frame = ppi_frame(np.random.default_rng(1), N_TEXTS)
    data = with_labels(frame, np.ones(N_TEXTS, dtype=bool), score_columns=("score",), pi=1.0)
    gold = frame["gold"].to_numpy()
    split = long_short_split(data)
    assert mean_ppi(data).point == float(np.mean(gold))
    expected = float(np.mean(gold[split]) - np.mean(gold[~split]))
    assert group_difference_ppi(data, split).point == expected
    assert mean_ppi(data, power_tuning=True).point == float(np.mean(gold))


def test_the_mean_follows_the_ppi_formula_with_sample_standard_errors() -> None:
    data, frame = _data()
    labelled = data.labelled_mask
    score, gold = frame["score"].to_numpy(), frame["gold"].to_numpy()
    estimate = mean_ppi(data)
    rectifier = score[labelled] - gold[labelled]
    expected = score.mean() - rectifier.mean()
    se = np.sqrt(np.var(rectifier, ddof=1) / labelled.sum() + np.var(score, ddof=1) / N_TEXTS)
    assert estimate.point == pytest.approx(expected, abs=1e-12)
    assert estimate.se == pytest.approx(se, abs=1e-12)
    assert (estimate.n_labelled, estimate.n_total, estimate.method) == (20, N_TEXTS, "ppi_mean")
    assert estimate.details["plugin"] == pytest.approx(score.mean())
    assert estimate.details["rectifier"] == pytest.approx(rectifier.mean())
    assert estimate.details["lambda_star"] == 1.0
    assert estimate.details["prediction"] == "score"


def test_the_difference_follows_the_ppi_formula_per_group() -> None:
    data, frame = _data()
    labelled, split = data.labelled_mask, long_short_split(data)
    score, gold = frame["score"].to_numpy(), frame["gold"].to_numpy()
    rectifier = score - gold

    def gap(values: np.ndarray, mask: np.ndarray) -> float:
        return float(values[mask].mean() - values[~mask].mean())

    def gap_se(values: np.ndarray, mask: np.ndarray) -> float:
        a, b = values[mask], values[~mask]
        return float(np.sqrt(np.var(a, ddof=1) / a.size + np.var(b, ddof=1) / b.size))

    estimate = group_difference_ppi(data, split)
    expected = gap(score, split) - gap(rectifier[labelled], split[labelled])
    se = np.hypot(gap_se(rectifier[labelled], split[labelled]), gap_se(score, split))
    assert estimate.point == pytest.approx(expected, abs=1e-12)
    assert estimate.se == pytest.approx(se, abs=1e-12)
    assert estimate.method == "ppi_difference"
    assert plugin_value(score, split) == pytest.approx(gap(score, split))
    assert naive_se(score, split) == pytest.approx(gap_se(score, split))


def test_the_group_can_be_a_boolean_series_indexed_by_id() -> None:
    data, _ = _data()
    split = long_short_split(data)
    by_id = pd.Series(split, index=data.ids)[::-1]
    assert group_difference_ppi(data, by_id).point == group_difference_ppi(data, split).point


def test_power_tuning_uses_population_moments_and_the_tuned_formula() -> None:
    data, frame = _data()
    labelled = data.labelled_mask
    score, gold = frame["score"].to_numpy(), frame["gold"].to_numpy()
    f_lab, y_lab = score[labelled], gold[labelled]
    lam = float(np.mean((y_lab - y_lab.mean()) * (f_lab - f_lab.mean())) / np.var(f_lab))
    estimate = mean_ppi(data, power_tuning=True)
    se = np.sqrt(
        np.var(lam * f_lab - y_lab, ddof=1) / f_lab.size + lam**2 * np.var(score, ddof=1) / N_TEXTS
    )
    assert estimate.details["lambda_star"] == pytest.approx(lam, abs=1e-12)
    assert estimate.point == pytest.approx(y_lab.mean() + lam * (score.mean() - f_lab.mean()))
    assert estimate.se == pytest.approx(se, abs=1e-12)
    assert estimate.method == "ppi_mean_tuned"


def test_power_tuning_reduces_to_the_plain_estimate_when_lambda_star_is_one() -> None:
    rng = np.random.default_rng(8)
    frame = ppi_frame(rng, N_TEXTS)
    labelled = np.arange(N_TEXTS) % 2 == 0
    score = frame["score"].to_numpy()
    noise = rng.normal(size=N_TEXTS)
    design = np.column_stack([np.ones(labelled.sum()), score[labelled]])
    beta, *_ = np.linalg.lstsq(design, noise[labelled], rcond=None)
    gold = score.copy()
    gold[labelled] = score[labelled] + noise[labelled] - design @ beta
    frame["gold"] = gold
    data = with_labels(frame, labelled, score_columns=("score",), pi=0.5)
    split = long_short_split(data)
    for plain, tuned in (
        (mean_ppi(data), mean_ppi(data, power_tuning=True)),
        (group_difference_ppi(data, split), group_difference_ppi(data, split, power_tuning=True)),
    ):
        assert tuned.details["lambda_star"] == pytest.approx(1.0, abs=1e-12)
        assert tuned.point == pytest.approx(plain.point, abs=1e-12)
        assert tuned.se == pytest.approx(plain.se, abs=1e-12)


def test_a_prediction_series_is_matched_by_id_and_replaces_the_score() -> None:
    data, frame = _data()
    prediction = pd.Series(frame["gold"].to_numpy() + 0.25, index=data.ids)
    shuffled = prediction.sample(frac=1.0, random_state=1).rename("calibrated")
    estimate = mean_ppi(data, prediction=shuffled)
    assert estimate.point == pytest.approx(mean_ppi(data, prediction=prediction).point)
    assert estimate.details["prediction"] == "calibrated"


def test_a_prediction_must_cover_every_text_with_a_finite_unique_value() -> None:
    data, _ = _data()
    prediction = pd.Series(np.ones(N_TEXTS), index=data.ids)
    with pytest.raises(ConfigurationError, match="lacks text ids"):
        mean_ppi(data, prediction=prediction.iloc[1:])
    with pytest.raises(ConfigurationError, match="finite"):
        mean_ppi(data, prediction=prediction.where(prediction.index != data.ids[3]))
    with pytest.raises(ConfigurationError, match="repeats"):
        mean_ppi(data, prediction=pd.concat([prediction, prediction.iloc[:1]]))
    with pytest.raises(ConfigurationError, match="Series"):
        mean_ppi(data, prediction=np.ones(N_TEXTS))  # type: ignore[arg-type]


def test_a_labelled_sample_without_pi_is_refused_with_a_pointer_to_the_draw() -> None:
    data, _ = _data(pi=None)
    split = long_short_split(data)
    for call in (lambda: mean_ppi(data), lambda: group_difference_ppi(data, split)):
        with pytest.raises(EstimatorError, match=r"\bpi\b.*draw_human_sample"):
            call()


def test_pi_must_be_finite_and_equal_on_the_labelled_rows() -> None:
    data, _ = _data()
    frame = data.frame.copy()
    frame.loc[0, "pi"] = np.nan
    with pytest.raises(EstimatorError, match=r"\bpi\b.*labelled"):
        mean_ppi(CorrectionData(frame=frame, score_columns=("score",)))
    frame.loc[0, "pi"] = 0.25
    with pytest.raises(EstimatorError, match="same on every labelled row"):
        mean_ppi(CorrectionData(frame=frame, score_columns=("score",)))
    frame["pi"] = 0.5
    frame.loc[1, "pi"] = np.nan
    assert np.isfinite(mean_ppi(CorrectionData(frame=frame, score_columns=("score",))).point)


def test_too_few_labelled_rows_are_refused() -> None:
    data, _ = _data(labelled_every=N_TEXTS)
    with pytest.raises(EstimatorError, match="at least 2 labelled"):
        mean_ppi(data)
    data, _ = _data(labelled_every=2)
    lonely = np.zeros(N_TEXTS, dtype=bool)
    lonely[0] = True
    with pytest.raises(EstimatorError, match="at least 2 labelled"):
        group_difference_ppi(data, lonely)


def test_a_group_must_be_boolean_and_match_the_data() -> None:
    data, _ = _data()
    split = long_short_split(data)
    with pytest.raises(ConfigurationError, match="boolean"):
        group_difference_ppi(data, split.astype(int))
    with pytest.raises(ConfigurationError, match="boolean"):
        group_difference_ppi(data, split[:-1])
    by_id = pd.Series(split, index=data.ids)
    with pytest.raises(ConfigurationError, match="lacks text ids"):
        group_difference_ppi(data, by_id.iloc[1:])
    with pytest.raises(ConfigurationError, match="repeats"):
        group_difference_ppi(data, pd.concat([by_id, by_id.iloc[:1]]))
    with pytest.raises(ConfigurationError, match="boolean"):
        group_difference_ppi(data, by_id.astype(float))


def test_the_long_short_split_is_strictly_above_the_median_covariate() -> None:
    data, frame = _data()
    covariate = frame["log_chars"].to_numpy()
    split = long_short_split(data)
    assert split.dtype == bool
    np.testing.assert_array_equal(split, covariate > np.median(covariate))
    tied = frame.assign(log_chars=np.r_[np.zeros(21), np.ones(19)])
    tied_data = CorrectionData(frame=tied, score_columns=("score",))
    assert long_short_split(tied_data).sum() == 19


def test_ppi_covers_the_truth_in_300_draws_while_the_raw_difference_is_biased() -> None:
    """300 replications of 2,000 texts with a draw of 100 (a simple random sample).

    The texts are generated afresh for every draw, so the truth is the parameter of the
    generating model (mean 1, difference 0) and the interval's machine term is real variance.
    """
    rng = np.random.default_rng(MONTE_CARLO_SEED)
    estimates: dict[str, list[Estimate]] = {"mean": [], "difference": []}
    raw_differences = []
    for _ in range(300):
        frame = ppi_frame(rng, 2000)
        data = with_labels(
            frame, labels_by_srs(rng, 2000, 100), score_columns=("score",), pi=100 / 2000
        )
        split = long_short_split(data)
        estimates["mean"].append(mean_ppi(data))
        estimates["difference"].append(group_difference_ppi(data, split))
        raw_differences.append(plugin_value(frame["score"].to_numpy(), split))
    for estimand, truth in (("mean", TRUE_MEAN), ("difference", TRUE_DIFFERENCE)):
        points = np.array([estimate.point for estimate in estimates[estimand]])
        covered = [e.ci_low <= truth <= e.ci_high for e in estimates[estimand]]
        assert abs(points.mean() - truth) < 0.01, estimand
        assert 0.92 <= float(np.mean(covered)) <= 0.98, estimand
    assert float(np.mean(raw_differences)) - TRUE_DIFFERENCE > 0.4
