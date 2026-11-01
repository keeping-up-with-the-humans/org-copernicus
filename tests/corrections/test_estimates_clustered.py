"""Cluster-robust standard errors of the prediction-powered estimators under whole-group draws."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from estimates_support import TRUE_MEAN, clustered_frame, ppi_frame, with_labels

from copernicus.corrections import (
    CorrectionData,
    attach_draw,
    draw_groups,
    group_difference_ppi,
    mean_ppi,
    ols_outcome_ppi,
    ols_predictor_ppi,
)
from copernicus.errors import ConfigurationError, EstimatorError, SettingsError

pytestmark = pytest.mark.filterwarnings("error")

UNEQUAL_SIZES = (1, 2, 3, 6, 9, 12, 2, 5, 4, 7, 1, 8) * 5
MONTE_CARLO_SEED = 17
SLOPE = 0.9


def _drawn(frame: pd.DataFrame, quota: int, seed: int) -> CorrectionData:
    """The frame's texts with the gold values of a whole-group draw of ``quota`` texts."""
    base = CorrectionData(frame=frame.assign(gold=np.nan), score_columns=("score",))
    draw = draw_groups(base.ids, base.groups, base.ids, quota, seed=seed)
    return attach_draw(base, pd.Series(frame["gold"].to_numpy(), index=base.ids), draw)


def _cluster_variance(contributions: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """``G / (G - 1)`` times the sum of the squared group sums of the centred rows, over ``n^2``."""
    rows = contributions.reshape(contributions.shape[0], -1)
    centred = rows - rows.mean(axis=0)
    labels = np.unique(groups)
    sums = np.array([centred[groups == label].sum(axis=0) for label in labels])
    return labels.size / (labels.size - 1) * (sums.T @ sums) / rows.shape[0] ** 2


def _difference_contributions(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """``n / n_a (v - mean_a)`` in the group, ``-n / n_b (v - mean_b)`` outside it."""
    n = values.size
    contributions = np.empty(n)
    contributions[mask] = (values[mask] - values[mask].mean()) * n / mask.sum()
    contributions[~mask] = -(values[~mask] - values[~mask].mean()) * n / (~mask).sum()
    return contributions


def test_with_one_text_per_group_the_clustered_mean_se_is_the_independent_one() -> None:
    frame = ppi_frame(np.random.default_rng(3), 40)
    data = with_labels(frame, np.arange(40) % 2 == 0, score_columns=("score",), pi=0.5)
    independent = mean_ppi(data, cluster=False)
    clustered = mean_ppi(data, cluster=True)
    assert clustered.point == independent.point
    assert clustered.se == pytest.approx(independent.se, abs=1e-14)
    assert (clustered.details["n_clusters"], clustered.details["n_clusters_labelled"]) == (40, 20)
    auto = mean_ppi(data)
    assert auto.details["cluster"] is None and auto.details["clustered"] is False
    assert auto.se == independent.se and "n_clusters" not in auto.details


def test_the_default_clusters_a_draw_whose_groups_hold_several_labelled_texts() -> None:
    data = _drawn(clustered_frame(np.random.default_rng(1), UNEQUAL_SIZES), 60, seed=2)
    auto, forced = mean_ppi(data), mean_ppi(data, cluster=True)
    assert auto.details["clustered"] is True and auto.se == forced.se
    assert auto.details["n_clusters"] == len(UNEQUAL_SIZES)
    drawn_groups = np.unique(data.groups[data.labelled_mask]).size
    assert auto.details["n_clusters_labelled"] == drawn_groups
    independent = mean_ppi(data, cluster=False)
    assert independent.details["clustered"] is False and independent.point == auto.point
    flag = data.frame["flag"].to_numpy(dtype=bool)
    assert group_difference_ppi(data, flag).details["clustered"] is True


@pytest.mark.parametrize("power_tuning", [False, True])
def test_the_clustered_errors_sum_the_centred_contributions_within_groups(
    power_tuning: bool,
) -> None:
    data = _drawn(clustered_frame(np.random.default_rng(4), UNEQUAL_SIZES), 90, seed=6)
    lab, groups = data.labelled_mask, data.groups
    f, y = data.frame["score"].to_numpy(), data.gold
    flag = data.frame["flag"].to_numpy(dtype=bool)
    mean = mean_ppi(data, cluster=True, power_tuning=power_tuning)
    lam = float(mean.details["lambda_star"])  # type: ignore[arg-type]
    rectifier = lam * f[lab] - y[lab]
    variance = _cluster_variance(rectifier, groups[lab]) + lam**2 * _cluster_variance(f, groups)
    assert mean.se == pytest.approx(float(np.sqrt(variance[0, 0])), abs=1e-12)
    difference = group_difference_ppi(data, flag, cluster=True, power_tuning=power_tuning)
    variance = _cluster_variance(
        _difference_contributions(rectifier, flag[lab]), groups[lab]
    ) + lam**2 * _cluster_variance(_difference_contributions(f, flag), groups)
    assert difference.se == pytest.approx(float(np.sqrt(variance[0, 0])), abs=1e-12)
    assert difference.details["n_clusters_labelled"] == np.unique(groups[lab]).size


def test_an_intercept_only_regression_gives_the_clustered_mean() -> None:
    frame = clustered_frame(np.random.default_rng(7), UNEQUAL_SIZES, shift=2.0)
    data = _drawn(frame, 90, seed=8)
    mean = mean_ppi(data, cluster=True)
    intercept = ols_outcome_ppi(data, "1", coefficient=0, cluster=True)
    assert intercept.point == pytest.approx(mean.point, abs=1e-12)
    assert intercept.se == pytest.approx(mean.se, abs=1e-12)
    default = ols_outcome_ppi(data, "1", coefficient=0)
    assert default.details["cluster"] is None and default.details["clustered"] is True
    assert default.se == pytest.approx(mean_ppi(data).se, abs=1e-12)


def test_clustering_needs_two_groups_among_the_labelled_texts() -> None:
    frame = _with_outcome(clustered_frame(np.random.default_rng(2), (6, 4, 5)), seed=3)
    data = _drawn(frame, 2, seed=1)
    assert np.unique(data.groups[data.labelled_mask]).size == 1
    with pytest.raises(EstimatorError, match="two groups"):
        mean_ppi(data)
    with pytest.raises(EstimatorError, match="two groups"):
        ols_predictor_ppi(data, "outcome", cluster=True)
    assert np.isfinite(mean_ppi(data, cluster=False).se)


def _one_group_side(data: CorrectionData) -> tuple[np.ndarray, str]:
    """A grouping whose labelled texts where it is True all lie in one drawn group.

    The side also holds three groups that were not drawn, like a group-level attribute held by
    several groups of which the draw took one.
    """
    drawn, counts = np.unique(data.groups[data.labelled_mask], return_counts=True)
    lonely = str(drawn[np.argmax(counts >= 2)])
    undrawn = np.setdiff1d(np.unique(data.groups), drawn)[:3]
    return np.isin(data.groups, [lonely, *undrawn]), lonely


def test_a_clustered_difference_needs_two_labelled_groups_on_each_side() -> None:
    data = _drawn(clustered_frame(np.random.default_rng(1), UNEQUAL_SIZES), 60, seed=2)
    mask, lonely = _one_group_side(data)
    assert np.unique(data.groups[mask]).size == 4 and (mask & data.labelled_mask).sum() >= 2
    for cluster in (None, True):
        with pytest.raises(EstimatorError, match=f"where the grouping is True .*'{lonely}'") as no:
            group_difference_ppi(data, mask, cluster=cluster)
        assert "draw more groups" in str(no.value) and "cluster=False" in str(no.value)
        assert "ignores the correlation of the texts within a group" in str(no.value)
    with pytest.raises(EstimatorError, match=f"where the grouping is False .*'{lonely}'"):
        group_difference_ppi(data, ~mask)
    independent = group_difference_ppi(data, mask, cluster=False)
    assert independent.details["clustered"] is False and np.isfinite(independent.se)
    assert mean_ppi(data).details["clustered"] is True


def test_the_cluster_option_must_be_a_boolean_or_none() -> None:
    frame = ppi_frame(np.random.default_rng(3), 40)
    data = with_labels(frame, np.arange(40) % 2 == 0, score_columns=("score",), pi=0.5)
    for value in ("yes", 1, 0.0):
        with pytest.raises(SettingsError, match="cluster"):
            mean_ppi(data, cluster=value)  # type: ignore[arg-type]
        with pytest.raises(SettingsError, match="cluster"):
            ols_outcome_ppi(data, "1", coefficient=0, cluster=value)  # type: ignore[arg-type]


def test_the_cluster_option_takes_numpy_booleans_as_booleans() -> None:
    data = _drawn(clustered_frame(np.random.default_rng(1), UNEQUAL_SIZES), 60, seed=2)
    for value in (np.True_, np.False_):
        plain = bool(value)
        estimate = mean_ppi(data, cluster=value)  # type: ignore[arg-type]
        assert estimate.se == mean_ppi(data, cluster=plain).se
        assert type(estimate.details["cluster"]) is bool and estimate.details["cluster"] is plain
        outcome = ols_outcome_ppi(data, "1", coefficient=0, cluster=value)  # type: ignore[arg-type]
        assert outcome.details["clustered"] is plain and outcome.details["cluster"] is plain


@pytest.mark.slow
def test_clustered_intervals_cover_under_group_correlated_machine_error() -> None:
    """400 whole-group draws of 200 of 2,000 texts in 200 groups of ten, texts made afresh.

    The machine error has a part shared by the texts of a group (SD 0.6) and a part of each text
    (SD 0.3), so the labelled texts of a drawn group err together. The truth is the parameter of
    the generating model: mean 1, and difference 0 between the even and the odd groups. Twenty
    groups are drawn: with 1,000 draws the clustered intervals cover 0.96 (mean) and 0.95
    (difference), the independent ones 0.59 and 0.57. With ten drawn groups the clustered
    interval of the mean covers 0.92, with five 0.88 (few group sums carry the variance).
    """
    rng = np.random.default_rng(MONTE_CARLO_SEED)
    covered: dict[tuple[str, bool], list[bool]] = {
        (estimand, cluster): [] for estimand in ("mean", "difference") for cluster in (True, False)
    }
    errors: dict[bool, list[float]] = {True: [], False: []}
    for draw in range(400):
        data = _drawn(clustered_frame(rng, [10] * 200), 200, seed=draw)
        flag = data.frame["flag"].to_numpy(dtype=bool)
        for cluster in (True, False):
            mean = mean_ppi(data, cluster=cluster)
            difference = group_difference_ppi(data, flag, cluster=cluster)
            covered[("mean", cluster)].append(mean.ci_low <= TRUE_MEAN <= mean.ci_high)
            covered[("difference", cluster)].append(difference.ci_low <= 0.0 <= difference.ci_high)
            errors[cluster].append(mean.se)
    coverage = {key: float(np.mean(values)) for key, values in covered.items()}
    assert 0.92 <= coverage[("mean", True)] <= 0.98, coverage
    assert 0.92 <= coverage[("difference", True)] <= 0.98, coverage
    assert coverage[("mean", False)] < 0.75 and coverage[("difference", False)] < 0.75, coverage
    assert float(np.mean(errors[True])) > 2.0 * float(np.mean(errors[False]))


def _with_outcome(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    """The frame with an outcome ``0.5 + 0.9 V + 0.4 W + 0.5 e`` of the gold value and covariate."""
    rng = np.random.default_rng(seed)
    outcome = 0.5 + SLOPE * frame["gold"].to_numpy() + 0.4 * frame["log_chars"].to_numpy()
    return frame.assign(outcome=outcome + 0.5 * rng.normal(size=len(frame)))


def _predictor_data(seed: int) -> CorrectionData:
    """A whole-group draw of 90 texts with an outcome on the construct and a covariate."""
    frame = clustered_frame(np.random.default_rng(seed), UNEQUAL_SIZES)
    return _drawn(_with_outcome(frame, seed + 100), 90, seed=seed)


def _clustered_slope(data: CorrectionData, base: np.ndarray) -> tuple[float, float]:
    """The predictor slope and its clustered sandwich error, written out by hand."""
    lab, groups = data.labelled_mask, data.groups
    s, y, gold = data.frame["score"].to_numpy(), data.frame["outcome"].to_numpy(), data.gold

    def design(values: np.ndarray, rows: np.ndarray) -> np.ndarray:
        return np.column_stack([np.ones(rows.sum()), values, data.covariate[rows]])

    def moments(x: np.ndarray, out: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return x.T @ x / x.shape[0], x.T @ out / x.shape[0]

    x_base, x_lab, x_gold = design(s[base], base), design(s[lab], lab), design(gold[lab], lab)
    m_b, c_b = moments(x_base, y[base])
    m_s, c_s = moments(x_lab, y[lab])
    m_g, c_g = moments(x_gold, y[lab])
    moment = m_b - m_s + m_g
    beta = np.linalg.solve(moment, c_b - c_s + c_g)
    psi_base = x_base * (y[base] - x_base @ beta)[:, None]
    psi_lab = x_lab * (y[lab] - x_lab @ beta)[:, None] - x_gold * (y[lab] - x_gold @ beta)[:, None]
    variance = _cluster_variance(psi_base, groups[base]) + _cluster_variance(psi_lab, groups[lab])
    inverse = np.linalg.inv(moment)
    return float(beta[1]), float(np.sqrt((inverse @ variance @ inverse)[1, 1]))


@pytest.mark.parametrize("variant", ["ppi", "ludwig"])
def test_the_clustered_predictor_error_follows_the_hand_written_sandwich(variant: str) -> None:
    data = _predictor_data(5)
    base = np.ones(data.n, dtype=bool) if variant == "ppi" else ~data.labelled_mask
    slope, se = _clustered_slope(data, base)
    estimate = ols_predictor_ppi(data, "outcome", "log_chars", variant=variant)  # type: ignore[arg-type]
    assert estimate.details["clustered"] is True
    assert estimate.point == pytest.approx(slope, abs=1e-12)
    assert estimate.se == pytest.approx(se, abs=1e-12)
    assert estimate.details["n_clusters"] == np.unique(data.groups[base]).size
    independent = ols_predictor_ppi(data, "outcome", "log_chars", variant=variant, cluster=False)  # type: ignore[arg-type]
    assert independent.point == estimate.point and independent.details["clustered"] is False


def test_the_predictor_covariates_are_a_formula_or_a_matrix() -> None:
    data = _predictor_data(9)
    by_formula = ols_predictor_ppi(data, "outcome", "log_chars")
    matrix = data.frame[["log_chars"]].to_numpy()
    by_matrix = ols_predictor_ppi(data, "outcome", X=matrix)
    assert by_matrix.point == pytest.approx(by_formula.point, abs=1e-12)
    assert by_matrix.se == pytest.approx(by_formula.se, abs=1e-12)
    assert by_matrix.details["columns"] == ("Intercept", "construct", "x0")
    with pytest.raises(SettingsError, match="at most one of covariates and X"):
        ols_predictor_ppi(data, "outcome", "log_chars", X=matrix)
    with pytest.raises(ConfigurationError, match=r"'x0' are constant"):
        ols_predictor_ppi(data, "outcome", X=np.column_stack([np.ones(data.n), matrix]))
    with pytest.raises(ConfigurationError, match="rows"):
        ols_predictor_ppi(data, "outcome", X=matrix[1:])
