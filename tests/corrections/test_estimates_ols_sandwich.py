"""The sandwich standard errors of the regression estimators, on nearly collinear designs too."""

from __future__ import annotations

import logging
import math
from fractions import Fraction
from typing import Literal

import numpy as np
import pandas as pd
import pytest
from estimates_support import slope_frame, with_labels

from copernicus.corrections import (
    CorrectionData,
    attach_draw,
    draw_groups,
    ols_outcome_ppi,
    ols_predictor_ppi,
)
from copernicus.corrections.estimates_oriv import stacked_2sls

pytestmark = pytest.mark.filterwarnings("error")

# A regressor that is nearly constant, and a response whose pairs cancel at each of its values,
# so the slope is 0 and its standard error is about 5.48; the moment matrix passes the check of
# positive definiteness with a condition number of about 1.5e14.
NEAR_X = np.array([0.1, 0.1, 0.10000010000000001, 0.10000010000000001, 0.0999999, 0.0999999])
PAIRS_Y = np.array([1.0, -1.0, 1e-6, -1e-6, 1e-6, -1e-6])
DIGITS_HALF_LOST = 1.0 / math.sqrt(float(np.finfo(float).eps))
TERMS = list[tuple[np.ndarray, np.ndarray | None]]
# The parts of the warning above the threshold: what it states, and the two causes it names.
NEARLY_SINGULAR = "the moment matrix scaled to a unit diagonal is nearly singular"
NEARLY_DEPENDENT = "a regressor that is nearly a combination of the others"
SUBTRACTION = (
    "the subtraction of the labelled texts' moments leaving little of the construct's variance"
)


def _exact_slope_variance(x: np.ndarray, y: np.ndarray) -> float:
    """The sandwich variance of the slope of ``y`` on ``[1, x]``, in exact rational arithmetic.

    The inputs are taken as the binary floats they are; the moment matrix, the least-squares
    line, each text's contribution transformed by the inverse moment matrix and the variance of
    their mean (divisor ``n - 1``, independent texts) are then exact, so the only rounding is the
    conversion of the result to a float.
    """
    n = len(x)
    xs, ys = [Fraction(float(value)) for value in x], [Fraction(float(value)) for value in y]
    mean_x, mean_xx = sum(xs) / n, sum(value * value for value in xs) / n
    mean_y, mean_xy = sum(ys) / n, sum(a * b for a, b in zip(xs, ys, strict=True)) / n
    determinant = mean_xx - mean_x * mean_x
    first, second = -mean_x / determinant, 1 / determinant
    intercept = (mean_xx * mean_y - mean_x * mean_xy) / determinant
    slope = first * mean_y + second * mean_xy
    transformed = [
        (first + second * a) * (b - intercept - slope * a) for a, b in zip(xs, ys, strict=True)
    ]
    centre = sum(transformed) / n
    return float(sum((value - centre) ** 2 for value in transformed) / (n * (n - 1)))


def _six_texts(score: np.ndarray, gold: np.ndarray, **extra: np.ndarray) -> CorrectionData:
    """Six independent texts, every one human-coded with the inclusion probability 1."""
    ids = [f"t{index}" for index in range(6)]
    frame = pd.DataFrame(
        {"text_id": ids, "group_id": ids, "score": score, "gold": gold, "log_chars": 0.0}
    )
    return CorrectionData(frame=frame.assign(pi=1.0, **extra), score_columns=("score",))


def test_a_nearly_collinear_design_gets_its_exact_standard_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    exact = math.sqrt(_exact_slope_variance(NEAR_X, PAIRS_Y))
    assert exact == pytest.approx(math.sqrt(30.0), rel=1e-6)
    design = np.column_stack([np.ones(6), NEAR_X])
    with caplog.at_level(logging.WARNING, logger="copernicus.corrections.estimates_ols"):
        outcome = ols_outcome_ppi(_six_texts(PAIRS_Y, PAIRS_Y), X=design, cluster=False)
        predictor = ols_predictor_ppi(
            _six_texts(NEAR_X, NEAR_X, outcome=PAIRS_Y), "outcome", cluster=False
        )
    for estimate in (outcome, predictor):
        assert estimate.se == pytest.approx(exact, rel=0.01)
        assert estimate.ci_low < 0.0 < estimate.ci_high
        assert estimate.details["condition_number"] > DIGITS_HALF_LOST  # type: ignore[operator]
    warnings = [record.getMessage() for record in caplog.records]
    assert len(warnings) == 2
    for method, warning in zip(("ppi_ols_outcome", "ppi_ols_predictor"), warnings, strict=True):
        assert warning.startswith(f"{method}: {NEARLY_SINGULAR}")
        assert "so the solve can lose more than half of its digits" in warning
    # Only the corrected regression subtracts the labelled texts' moments in its matrix.
    assert warnings[0].endswith(f"check for {NEARLY_DEPENDENT}")
    assert warnings[1].endswith(f"check for {NEARLY_DEPENDENT}, and for {SUBTRACTION}")


def test_a_perfect_fit_on_the_nearly_collinear_design_rightly_has_the_standard_error_zero() -> None:
    """A response that is zero on every text leaves every transformed contribution zero."""
    zeros = np.zeros(6)
    design = np.column_stack([np.ones(6), NEAR_X])
    outcome = ols_outcome_ppi(_six_texts(zeros, zeros), X=design, cluster=False)
    predictor = ols_predictor_ppi(
        _six_texts(NEAR_X, NEAR_X, outcome=zeros), "outcome", cluster=False
    )
    for estimate in (outcome, predictor):
        assert (estimate.point, estimate.se, estimate.ci_low, estimate.ci_high) == (0.0,) * 4
        errors = list(estimate.details["standard_errors"].values())  # type: ignore[union-attr]
        assert errors == [0.0, 0.0]
        assert estimate.details["condition_number"] > DIGITS_HALF_LOST  # type: ignore[operator]


def _exact_stacked_slope_variance(xs: list[Fraction], ys: list[Fraction]) -> Fraction:
    """The text-clustered sandwich variance of the stacked slope with one measurement twice.

    Valid when the least-squares line of ``y`` on ``x`` is flat (``sum(y) = sum(x y) = 0``).
    With ``x`` as both measurements, the stacked regressor ``(x; x)`` is the sum of the two
    instrument columns, so the first stage reproduces the design and the stacked estimator is
    least squares of ``(y; y)`` on the two block intercepts and ``(x; x)``: every coefficient
    is 0 and the residuals are ``y``. The score row ``y_i (1, 1, 2 x_i)`` of a text, solved
    against the system's matrix, has the slope component ``(x_i - mean(x)) y_i / S`` with
    ``S = sum((x - mean(x))^2)``; the sandwich clustered by text is the sum of their squares,
    without centring and without a factor ``n / (n - 1)``.
    """
    mean_x = sum(xs) / len(xs)
    spread = sum((value - mean_x) ** 2 for value in xs)
    return sum(((a - mean_x) * b / spread) ** 2 for a, b in zip(xs, ys, strict=True))


def test_the_stacked_system_gets_its_exact_standard_error_on_the_same_design() -> None:
    xs = [Fraction(float(value)) for value in NEAR_X]
    ys = [Fraction(float(value)) for value in PAIRS_Y]
    assert sum(ys) == 0 and sum(a * b for a, b in zip(xs, ys, strict=True)) == 0
    variance = _exact_stacked_slope_variance(xs, ys)
    exact = math.sqrt(float(variance))
    assert exact == pytest.approx(5.0, rel=1e-9)
    # The least-squares sandwich of the same slope differs by the factor n / (n - 1) alone.
    assert _exact_slope_variance(NEAR_X, PAIRS_Y) == pytest.approx(float(variance * 6 / 5))
    fit = stacked_2sls(PAIRS_Y, NEAR_X, NEAR_X)
    assert fit.se == pytest.approx(exact, rel=0.01)
    assert abs(fit.slope) < 0.01 * exact


def test_the_warning_names_the_subtraction_when_every_design_is_well_conditioned(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The scores and the gold values span their designs well; their moments nearly cancel."""
    gold_size = math.sqrt(48.5 + 1e-10)
    score, gold = np.array([-10.0, 10.0, 1.0, 3.0]), np.array([-gold_size, gold_size])
    score_design = np.column_stack([np.ones(4), score])
    gold_design = np.column_stack([np.ones(2), gold])
    assert max(np.linalg.cond(score_design), np.linalg.cond(gold_design)) < 10.0
    rectified = (
        gold_design.T @ gold_design / 2
        + score_design.T @ score_design / 4
        - score_design[:2].T @ score_design[:2] / 2
    )
    assert rectified == pytest.approx(np.ones((2, 2)), abs=1e-9)
    ids = [f"t{index}" for index in range(4)]
    frame = pd.DataFrame(
        {"text_id": ids, "group_id": ids, "score": score, "gold": [*gold, np.nan, np.nan]}
    )
    data = CorrectionData(
        frame=frame.assign(log_chars=0.0, pi=0.5, outcome=[1.0, -1.0, 2.0, -2.0]),
        score_columns=("score",),
    )
    with caplog.at_level(logging.WARNING, logger="copernicus.corrections.estimates_ols"):
        estimate = ols_predictor_ppi(data, "outcome", cluster=False)
    assert estimate.details["condition_number"] == pytest.approx(4e10, rel=0.01)
    [warning] = [record.getMessage() for record in caplog.records]
    assert warning.startswith(f"ppi_ols_predictor: {NEARLY_SINGULAR}")
    assert warning.endswith(f"check for {NEARLY_DEPENDENT}, and for {SUBTRACTION}")


def test_the_condition_number_does_not_depend_on_the_units_of_a_regressor(
    caplog: pytest.LogCaptureFixture,
) -> None:
    frame = slope_frame(np.random.default_rng(4), 200)
    data = with_labels(frame, np.arange(200) % 4 == 0, score_columns=("score",), pi=0.25)
    rescaled = CorrectionData(
        frame=data.frame.assign(log_chars=1e6 * data.frame["log_chars"]), score_columns=("score",)
    )
    with caplog.at_level(logging.WARNING, logger="copernicus.corrections.estimates_ols"):
        first = ols_outcome_ppi(data, "1 + log_chars")
        second = ols_outcome_ppi(rescaled, "1 + log_chars")
        third = ols_predictor_ppi(data, "outcome", "1 + log_chars")
    assert first.details["condition_number"] == pytest.approx(
        second.details["condition_number"], rel=1e-9
    )
    assert 1.0 <= third.details["condition_number"] < 10.0  # type: ignore[operator]
    assert not caplog.records


def _expanded_sandwich(moment: np.ndarray, terms: TERMS) -> np.ndarray:
    """``H^-1 [sum over the terms of the covariance of the mean of psi] H^-1``, multiplied out."""
    variance = np.zeros_like(moment)
    for rows, groups in terms:
        if groups is None:
            variance += np.atleast_2d(np.cov(rows, rowvar=False, ddof=1)) / rows.shape[0]
            continue
        labels, inverse = np.unique(groups, return_inverse=True)
        sums = np.zeros((labels.size, rows.shape[1]))
        np.add.at(sums, inverse, rows - rows.mean(axis=0))
        covariance = np.atleast_2d(np.cov(sums, rowvar=False, ddof=1))
        variance += covariance * labels.size / rows.shape[0] ** 2
    inverse_moment = np.linalg.inv(moment)
    return np.asarray(inverse_moment @ variance @ inverse_moment)


def _grouped_draw(seed: int) -> tuple[CorrectionData, pd.DataFrame]:
    """240 texts in 60 groups of four and a whole-group draw of 80 texts."""
    frame = slope_frame(np.random.default_rng(seed), 240)
    frame = frame.assign(group_id=[f"g{index // 4:03d}" for index in range(240)])
    base = CorrectionData(frame=frame.assign(gold=np.nan), score_columns=("score",))
    draw = draw_groups(base.ids, base.groups, base.ids, 80, seed=seed)
    gold = pd.Series(frame["gold"].to_numpy(), index=base.ids)
    return attach_draw(base, gold, draw), frame


def _rows(design: np.ndarray, response: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """The estimating function ``x (v - x'beta)`` of every row."""
    return design * (response - design @ beta)[:, None]


@pytest.mark.parametrize("cluster", [False, True])
@pytest.mark.parametrize("formula", ["1", "1 + log_chars"])
def test_on_a_well_conditioned_design_the_outcome_errors_equal_the_expanded_sandwich(
    cluster: bool, formula: str
) -> None:
    data, frame = _grouped_draw(9)
    lab = data.labelled_mask
    columns = [np.ones(data.n)] + ([frame["log_chars"].to_numpy()] if formula != "1" else [])
    design = np.column_stack(columns)
    s, gold = frame["score"].to_numpy(), frame["gold"].to_numpy()
    moment = design.T @ design / data.n
    target = design.T @ s / data.n + design[lab].T @ (gold[lab] - s[lab]) / int(lab.sum())
    beta = np.linalg.solve(moment, target)
    groups = data.groups if cluster else None
    terms: TERMS = [
        (_rows(design, s, beta), groups),
        (design[lab] * (s[lab] - gold[lab])[:, None], None if groups is None else groups[lab]),
    ]
    expected = np.sqrt(np.diag(_expanded_sandwich(moment, terms)))
    position = 0 if formula == "1" else 1
    estimate = ols_outcome_ppi(data, formula, coefficient=position, cluster=cluster)
    errors = list(estimate.details["standard_errors"].values())  # type: ignore[union-attr]
    assert errors == pytest.approx(expected.tolist(), rel=1e-10, abs=0.0)
    assert estimate.point == pytest.approx(float(beta[position]), rel=1e-10, abs=0.0)


@pytest.mark.parametrize("cluster", [False, True])
@pytest.mark.parametrize("covariates", [None, "1 + log_chars"])
@pytest.mark.parametrize("variant", ["ppi", "ludwig"])
def test_on_a_well_conditioned_design_the_predictor_errors_equal_the_expanded_sandwich(
    cluster: bool, covariates: str | None, variant: Literal["ppi", "ludwig"]
) -> None:
    data, frame = _grouped_draw(9)
    lab = data.labelled_mask
    base = np.ones(data.n, dtype=bool) if variant == "ppi" else ~lab
    columns = [np.ones(data.n), frame["score"].to_numpy()]
    if covariates is not None:
        columns.append(frame["log_chars"].to_numpy())
    design = np.column_stack(columns)
    gold_design = design[lab].copy()
    gold_design[:, 1] = frame["gold"].to_numpy()[lab]
    y = frame["outcome"].to_numpy()
    blocks = [(design[base], y[base]), (design[lab], y[lab]), (gold_design, y[lab])]
    (m_b, c_b), (m_s, c_s), (m_g, c_g) = (
        (rows.T @ rows / rows.shape[0], rows.T @ out / rows.shape[0]) for rows, out in blocks
    )
    moment = m_g + m_b - m_s
    beta = np.linalg.solve(moment, c_g + c_b - c_s)
    groups = data.groups if cluster else None
    rectifier = _rows(design[lab], y[lab], beta) - _rows(gold_design, y[lab], beta)
    terms: TERMS = [
        (_rows(design[base], y[base], beta), None if groups is None else groups[base]),
        (rectifier, None if groups is None else groups[lab]),
    ]
    expected = np.sqrt(np.diag(_expanded_sandwich(moment, terms)))
    estimate = ols_predictor_ppi(data, "outcome", covariates, variant=variant, cluster=cluster)
    errors = list(estimate.details["standard_errors"].values())  # type: ignore[union-attr]
    assert errors == pytest.approx(expected.tolist(), rel=1e-10, abs=0.0)
    coefficients = list(estimate.details["coefficients"].values())  # type: ignore[union-attr]
    assert coefficients == pytest.approx(beta.tolist(), rel=1e-10, abs=0.0)
