"""Prediction-powered OLS coefficients: the construct as outcome and as predictor."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from estimates_support import TRUE_SLOPE, labels_by_srs, slope_frame, with_labels

from copernicus.corrections import (
    CorrectionData,
    mean_ppi,
    ols_outcome_ppi,
    ols_predictor_ppi,
)
from copernicus.errors import ConfigurationError, EstimatorError, SettingsError

pytestmark = pytest.mark.filterwarnings("error")

N_TEXTS = 60
MONTE_CARLO_SEED = 3


def _outcome_frame(seed: int = 2, *, group_size: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    covariate = rng.normal(size=N_TEXTS)
    kind = np.where(np.arange(N_TEXTS) % 2 == 0, "a", "b")
    gold = 1.0 + 0.6 * covariate + 0.4 * (kind == "b") + rng.normal(scale=0.5, size=N_TEXTS)
    return pd.DataFrame(
        {
            "text_id": [f"t{index:03d}" for index in range(N_TEXTS)],
            "group_id": [f"g{index // group_size:03d}" for index in range(N_TEXTS)],
            "score": gold + 0.3 + 0.2 * covariate + rng.normal(scale=0.3, size=N_TEXTS),
            "gold": gold,
            "log_chars": covariate,
            "kind": kind,
        }
    )


def _outcome_data(
    *, labelled_every: int = 3, pi: float | None = 1 / 3, **kwargs: int
) -> tuple[CorrectionData, pd.DataFrame]:
    frame = _outcome_frame(**kwargs)
    labelled = np.arange(N_TEXTS) % labelled_every == 0
    return with_labels(frame, labelled, score_columns=("score",), pi=pi), frame


def _covariance_of_mean(psi: np.ndarray, groups: np.ndarray | None) -> np.ndarray:
    """Cov(psi) / n, or the cluster version: rows centred, Cov(cluster sums) * G / n^2."""
    if groups is None:
        return np.atleast_2d(np.cov(psi, rowvar=False, ddof=1)) / psi.shape[0]
    labels, inverse = np.unique(groups, return_inverse=True)
    sums = np.zeros((labels.size, psi.shape[1]))
    np.add.at(sums, inverse, psi - psi.mean(axis=0))
    covariance = np.atleast_2d(np.cov(sums, rowvar=False, ddof=1))
    return covariance * labels.size / psi.shape[0] ** 2


def _outcome_oracle(
    design: np.ndarray, s: np.ndarray, gold: np.ndarray, lab: np.ndarray, groups: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray]:
    """Rectified normal equations with the construct as outcome, written out by hand."""
    n_all, n_lab = s.size, int(lab.sum())
    moment = design.T @ design / n_all
    target = (
        design.T @ s / n_all - design[lab].T @ s[lab] / n_lab + design[lab].T @ gold[lab] / n_lab
    )
    beta = np.linalg.solve(moment, target)
    psi_s = design * (s - design @ beta)[:, None]
    psi_d = design[lab] * (s[lab] - gold[lab])[:, None]
    variance = _covariance_of_mean(psi_s, groups) + _covariance_of_mean(
        psi_d, None if groups is None else groups[lab]
    )
    inverse = np.linalg.inv(moment)
    return beta, np.sqrt(np.diag(inverse @ variance @ inverse))


def _slope_oracle(
    s: np.ndarray, y: np.ndarray, lab: np.ndarray, g_lab: np.ndarray, *, primary: bool
) -> tuple[float, float]:
    """The rectified normal equations written out by hand: every text as the base, or the
    unlabelled texts (the Ludwig variant)."""

    def moments(v: np.ndarray, out: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = np.column_stack([np.ones(v.size), v])
        return x.T @ x / v.size, x.T @ out / v.size

    def psi(v: np.ndarray, out: np.ndarray, beta: np.ndarray) -> np.ndarray:
        x = np.column_stack([np.ones(v.size), v])
        return x * (out - x @ beta)[:, None]

    base = ~lab if primary else np.ones(s.size, dtype=bool)
    m_b, c_b = moments(s[base], y[base])
    m_ls, c_ls = moments(s[lab], y[lab])
    m_lg, c_lg = moments(g_lab, y[lab])
    h = m_b - m_ls + m_lg
    beta = np.linalg.solve(h, c_b - c_ls + c_lg)
    h_inv = np.linalg.inv(h)
    variance = _covariance_of_mean(psi(s[base], y[base], beta), None) + _covariance_of_mean(
        psi(s[lab], y[lab], beta) - psi(g_lab, y[lab], beta), None
    )
    return float(beta[1]), float(np.sqrt((h_inv @ variance @ h_inv)[1, 1]))


def test_on_a_full_draw_the_outcome_coefficients_equal_ols_on_gold() -> None:
    frame = _outcome_frame()
    data = with_labels(frame, np.ones(N_TEXTS, dtype=bool), score_columns=("score",), pi=1.0)
    estimate = ols_outcome_ppi(data, "1 + log_chars + kind")
    design = np.column_stack(
        [np.ones(N_TEXTS), frame["log_chars"], (frame["kind"] == "b").astype(float)]
    )
    beta, *_ = np.linalg.lstsq(design, frame["gold"].to_numpy(), rcond=None)
    assert estimate.point == pytest.approx(beta[1], abs=1e-10)
    coefficients = estimate.details["coefficients"]
    assert list(coefficients) == ["Intercept", "log_chars", "kind[T.b]"]
    np.testing.assert_allclose(list(coefficients.values()), beta, atol=1e-10)
    by_matrix = ols_outcome_ppi(data, X=design, coefficient=2)
    assert by_matrix.point == pytest.approx(beta[2], abs=1e-10)
    assert by_matrix.details["columns"] == ("x0", "x1", "x2")


@pytest.mark.parametrize("cluster", [False, True])
def test_the_outcome_estimate_follows_the_rectified_equations(cluster: bool) -> None:
    data, frame = _outcome_data()
    design = np.column_stack(
        [np.ones(N_TEXTS), frame["log_chars"], (frame["kind"] == "b").astype(float)]
    )
    beta, se = _outcome_oracle(
        design,
        frame["score"].to_numpy(),
        frame["gold"].to_numpy(),
        data.labelled_mask,
        data.groups if cluster else None,
    )
    estimate = ols_outcome_ppi(data, "1 + log_chars + kind", cluster=cluster)
    assert estimate.point == pytest.approx(beta[1], abs=1e-12)
    assert estimate.se == pytest.approx(se[1], abs=1e-12)
    assert list(estimate.details["standard_errors"].values()) == pytest.approx(se, abs=1e-12)
    assert estimate.method == "ppi_ols_outcome"
    assert estimate.details["cluster"] is cluster


def test_clustering_changes_the_standard_error_but_not_the_point() -> None:
    data, _ = _outcome_data()
    plain = ols_outcome_ppi(data, "1 + log_chars")
    clustered = ols_outcome_ppi(data, "1 + log_chars", cluster=True)
    assert clustered.point == plain.point
    assert clustered.se != pytest.approx(plain.se, rel=1e-6)
    assert clustered.details["n_clusters"] == N_TEXTS // 3
    singletons, _ = _outcome_data(group_size=1)
    assert ols_outcome_ppi(singletons, "1 + log_chars", cluster=True).se == pytest.approx(
        ols_outcome_ppi(singletons, "1 + log_chars").se, abs=1e-14
    )


def test_an_intercept_only_outcome_model_is_the_ppi_mean() -> None:
    data, _ = _outcome_data()
    intercept = ols_outcome_ppi(data, "1", coefficient=0)
    mean = mean_ppi(data)
    assert intercept.point == pytest.approx(mean.point, abs=1e-12)
    assert intercept.se == pytest.approx(mean.se, abs=1e-12)


def test_a_coefficient_can_be_named_and_a_prediction_supplied() -> None:
    data, frame = _outcome_data()
    by_name = ols_outcome_ppi(data, "1 + log_chars + kind", coefficient="kind[T.b]")
    assert by_name.point == ols_outcome_ppi(data, "1 + log_chars + kind", coefficient=2).point
    assert by_name.details["coefficient"] == "kind[T.b]"
    prediction = pd.Series(frame["gold"].to_numpy(), index=data.ids, name="perfect")
    exact = ols_outcome_ppi(data, "1 + log_chars", prediction=prediction)
    design = np.column_stack([np.ones(N_TEXTS), frame["log_chars"]])
    beta, *_ = np.linalg.lstsq(design, frame["gold"].to_numpy(), rcond=None)
    assert exact.point == pytest.approx(beta[1], abs=1e-10)
    assert exact.details["prediction"] == "perfect"


def test_the_outcome_design_is_refused_when_inconsistent() -> None:
    data, _ = _outcome_data()
    with pytest.raises(SettingsError, match="exactly one of formula and X"):
        ols_outcome_ppi(data)
    with pytest.raises(SettingsError, match="exactly one of formula and X"):
        ols_outcome_ppi(data, "1 + log_chars", X=np.ones((N_TEXTS, 1)))
    with pytest.raises(SettingsError, match="right-hand side"):
        ols_outcome_ppi(data, "gold ~ log_chars")
    with pytest.raises(ConfigurationError, match="'nope'"):
        ols_outcome_ppi(data, "1 + nope")
    with pytest.raises(ConfigurationError, match="missing values"):
        ols_outcome_ppi(data, "1 + gold")
    with pytest.raises(ConfigurationError, match="rows"):
        ols_outcome_ppi(data, X=np.ones((N_TEXTS - 1, 2)))
    with pytest.raises(ConfigurationError, match="finite"):
        ols_outcome_ppi(data, X=np.full((N_TEXTS, 2), np.inf))
    with pytest.raises(ConfigurationError, match="numeric"):
        ols_outcome_ppi(data, X=np.full((N_TEXTS, 2), "a"))
    with pytest.raises(ConfigurationError, match="'size'.*columns"):
        ols_outcome_ppi(data, "1 + log_chars", coefficient="size")
    with pytest.raises(ConfigurationError, match="position 5"):
        ols_outcome_ppi(data, "1 + log_chars", coefficient=5)


def test_a_stateful_transform_of_a_missing_or_incomplete_column_is_refused() -> None:
    data, _ = _outcome_data()
    with pytest.raises(ConfigurationError, match=r"scale\(missing_col\)") as refusal:
        ols_outcome_ppi(data, "1 + scale(missing_col)")
    assert refusal.value.__cause__ is not None
    with pytest.raises(ConfigurationError, match=r"center\(gold\).*missing value"):
        ols_outcome_ppi(data, "1 + center(gold)")
    assert np.isfinite(ols_outcome_ppi(data, "1 + scale(log_chars)").point)


def test_a_singular_design_or_thin_clusters_leave_the_estimate_undefined() -> None:
    data, frame = _outcome_data()
    collinear = np.column_stack([np.ones(N_TEXTS), frame["log_chars"], frame["log_chars"]])
    with pytest.raises(EstimatorError, match="rectified moment matrix not positive definite"):
        ols_outcome_ppi(data, X=collinear)
    one_group = data.frame.assign(group_id="g")
    lonely = CorrectionData(frame=one_group, score_columns=("score",))
    with pytest.raises(EstimatorError, match="two groups"):
        ols_outcome_ppi(lonely, "1 + log_chars", cluster=True)
    unlabelled, _ = _outcome_data(pi=None)
    with pytest.raises(EstimatorError, match=r"\bpi\b.*draw_human_sample"):
        ols_outcome_ppi(unlabelled, "1 + log_chars")


def _slope_data(
    seed: int = 4, n_texts: int = 300, n_labelled: int = 60
) -> tuple[CorrectionData, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    frame = slope_frame(rng, n_texts)
    labelled = labels_by_srs(rng, n_texts, n_labelled)
    data = with_labels(frame, labelled, score_columns=("score",), pi=n_labelled / n_texts)
    return data, frame


@pytest.mark.parametrize(("variant", "primary"), [("ppi", False), ("ludwig", True)])
def test_the_predictor_estimates_match_the_hand_written_equations(
    variant: str, primary: bool
) -> None:
    data, frame = _slope_data()
    lab = data.labelled_mask
    slope, se = _slope_oracle(
        frame["score"].to_numpy(),
        frame["outcome"].to_numpy(),
        lab,
        frame["gold"].to_numpy()[lab],
        primary=primary,
    )
    estimate = ols_predictor_ppi(data, "outcome", variant=variant)  # type: ignore[arg-type]
    assert estimate.point == pytest.approx(slope, abs=1e-12)
    assert estimate.se == pytest.approx(se, abs=1e-12)
    assert estimate.method == f"{variant}_ols_predictor"
    assert estimate.details["columns"] == ("Intercept", "construct")


def test_on_full_gold_the_predictor_estimate_equals_ols_on_gold() -> None:
    frame = slope_frame(np.random.default_rng(6), 200)
    data = with_labels(frame, np.ones(200, dtype=bool), score_columns=("score",), pi=1.0)
    estimate = ols_predictor_ppi(data, "outcome", "log_chars")
    design = np.column_stack([np.ones(200), frame["gold"], frame["log_chars"]])
    beta, *_ = np.linalg.lstsq(design, frame["outcome"].to_numpy(), rcond=None)
    assert estimate.point == pytest.approx(beta[1], abs=1e-10)
    assert estimate.details["columns"] == ("Intercept", "construct", "log_chars")
    with pytest.raises(EstimatorError, match="unlabelled"):
        ols_predictor_ppi(data, "outcome", variant="ludwig")


def test_the_predictor_inputs_are_refused_when_inconsistent() -> None:
    data, _ = _slope_data()
    with pytest.raises(SettingsError, match="variant"):
        ols_predictor_ppi(data, "outcome", variant="naive")  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError, match="'missing' is not a column"):
        ols_predictor_ppi(data, "missing")
    with pytest.raises(ConfigurationError, match="numeric"):
        ols_predictor_ppi(data, "text_id")
    with pytest.raises(ConfigurationError, match="finite"):
        ols_predictor_ppi(data, "gold")
    with pytest.raises(SettingsError, match="intercept"):
        ols_predictor_ppi(data, "outcome", "0 + log_chars")
    with pytest.raises(ConfigurationError, match="position 3"):
        ols_predictor_ppi(data, "outcome", "log_chars", coefficient=3)


def test_a_predictor_matrix_that_is_not_positive_definite_is_refused() -> None:
    n = 50
    score = np.r_[np.array([10.0, -10.0, 10.0, -10.0]), np.random.default_rng(1).normal(size=n - 4)]
    score[4:] *= 0.1
    frame = pd.DataFrame(
        {
            "text_id": [f"t{index:02d}" for index in range(n)],
            "group_id": [f"t{index:02d}" for index in range(n)],
            "score": score,
            "gold": np.r_[np.array([0.1, -0.1, 0.0, 0.05]), np.zeros(n - 4)],
            "outcome": np.linspace(0.0, 1.0, n),
            "log_chars": np.zeros(n),
        }
    )
    data = with_labels(frame, np.arange(n) < 4, score_columns=("score",), pi=4 / n)
    with pytest.raises(EstimatorError, match="rectified moment matrix not positive definite"):
        ols_predictor_ppi(data, "outcome")


def test_the_predictor_ppi_covers_a_planted_slope_and_the_ludwig_variant_agrees() -> None:
    """300 replications of 4,000 texts with a simple random draw of 200 human-coded texts."""
    rng = np.random.default_rng(MONTE_CARLO_SEED)
    ppi_points, ludwig_points, covered = [], [], []
    for _ in range(300):
        frame = slope_frame(rng, 4000)
        data = with_labels(
            frame, labels_by_srs(rng, 4000, 200), score_columns=("score",), pi=200 / 4000
        )
        ppi = ols_predictor_ppi(data, "outcome", "log_chars")
        ludwig = ols_predictor_ppi(data, "outcome", "log_chars", variant="ludwig")
        ppi_points.append(ppi.point)
        ludwig_points.append(ludwig.point)
        covered.append(ppi.ci_low <= TRUE_SLOPE <= ppi.ci_high)
    assert 0.92 <= float(np.mean(covered)) <= 0.98
    assert abs(float(np.mean(ppi_points)) - TRUE_SLOPE) < 0.02
    assert abs(float(np.mean(ludwig_points)) - TRUE_SLOPE) < 0.02
