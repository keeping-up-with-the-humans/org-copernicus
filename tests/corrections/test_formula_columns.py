"""Formulas of the OLS estimators: what is checked before a formula is evaluated, and what after.

Before it evaluates a formula, ``formulaic`` reports the columns the formula reads, and the
report is not complete: version 1.2.2 names no column inside a stateful transform such as
``scale()``. The library refuses a reported name that the data lack and, after the evaluation,
a missing value that reached the design: a formula is judged by its design. The tests run under
the installed report and under reports that the test puts in its place; in every case the
design is built by the installed library.
"""

from __future__ import annotations

import formulaic
import numpy as np
import pandas as pd
import pytest
from formula_support import N_TEXTS, fix_report, formula_data, formula_frame
from formulaic.errors import FactorEvaluationError

from copernicus.corrections import ols_outcome_ppi, ols_predictor_ppi
from copernicus.errors import ConfigurationError

pytestmark = pytest.mark.filterwarnings("error")

REPORTS = ("installed", "empty")
# Formulas over complete columns. Several hold the name of an incomplete column of the data
# without reading it: as a keyword, an attribute, a method, a word of a text, a word of a
# comment, the parameter of a lambda and the variable of a comprehension.
COMPLETE = (
    "1 + log_chars",
    "1 + scale(log_chars)",
    "1 + `log chars`",
    "1 + scale(`log chars`)",
    "1 + np.log(log_chars**2 + 1)",
    "1 + center(log_chars + len(log_chars))",
    "1 + bs(log_chars, df=3)",
    "1 + center(log_chars - log_chars.mean())",
    "1 + center(log_chars.values)",
    '1 + I(log_chars + ("`gap`" == "a"))',
    "1 + I(log_chars # gap\n)",
    "1 + center((lambda gap: gap)(log_chars))",
    "1 + center([gap for gap in log_chars])",
)
# Formulas that leave a missing value in their design.
UNFILLED = ("1 + gap", "1 + center(gap)", "1 + `gap col`", "1 + log_chars:gap", '1 + Q("gap")')
UNEVALUABLE = ("1 + nope", "1 + scale(missing_col)", "1 + scale(log_chars) + center(missing_col)")
# The column ``gap`` written in full-width letters, which Python reads as the same name.
FULL_WIDTH = "1 + center(\uff47\uff41\uff50.fillna(0))"
# Names of the library's own code, which a formula does not see.
OWN_NAMES = (
    "1 + center(frame.gap.fillna(0))",
    "1 + center(pd.Series(log_chars))",
    "1 + center(role)",
)


@pytest.mark.parametrize("report", REPORTS)
@pytest.mark.parametrize("formula", COMPLETE)
def test_a_formula_over_complete_columns_is_evaluated_under_every_report(
    monkeypatch: pytest.MonkeyPatch, report: str, formula: str
) -> None:
    fix_report(monkeypatch, report)
    assert np.isfinite(ols_outcome_ppi(formula_data(), formula).point)


@pytest.mark.parametrize("report", REPORTS)
def test_a_transform_and_a_quoted_name_give_the_design_built_by_hand(
    monkeypatch: pytest.MonkeyPatch, report: str
) -> None:
    fix_report(monkeypatch, report)
    data = formula_data()
    covariate = data.frame["log_chars"].to_numpy(dtype=float)
    scaled = (covariate - covariate.mean()) / covariate.std(ddof=1)
    by_hand = ols_outcome_ppi(data, X=np.column_stack([np.ones(N_TEXTS), scaled]), coefficient=1)
    for formula in ("1 + scale(log_chars)", "1 + scale(`log chars`)"):
        estimate = ols_outcome_ppi(data, formula, coefficient=1)
        assert estimate.point == pytest.approx(by_hand.point, rel=1e-12)
        assert estimate.se == pytest.approx(by_hand.se, rel=1e-12)
    unscaled = ols_outcome_ppi(data, "1 + log_chars", coefficient=1)
    assert abs(unscaled.point - by_hand.point) > 1e-3


@pytest.mark.parametrize("report", REPORTS)
@pytest.mark.parametrize("formula", UNFILLED)
def test_a_formula_that_leaves_a_missing_value_in_its_design_is_refused_under_every_report(
    monkeypatch: pytest.MonkeyPatch, report: str, formula: str
) -> None:
    fix_report(monkeypatch, report)
    with pytest.raises(ConfigurationError, match="cannot be evaluated on the data") as refusal:
        ols_outcome_ppi(formula_data(), formula)
    assert isinstance(refusal.value.__cause__, ValueError)
    assert str(refusal.value.__cause__) in str(refusal.value)


def test_a_reported_name_that_the_data_lack_is_named_before_any_evaluation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fix_report(monkeypatch, {"missing_col"})
    lacking = r"scale\(missing_col\).*names column\(s\) 'missing_col' that the data lack"
    with pytest.raises(ConfigurationError, match=lacking) as refusal:
        ols_outcome_ppi(formula_data(), "1 + scale(missing_col)")
    assert refusal.value.__cause__ is None


@pytest.mark.parametrize(
    "report", [*REPORTS, {"gap"}], ids=["installed", "empty", "the column reported"]
)
def test_a_formula_that_fills_a_missing_value_itself_is_evaluated_under_every_report(
    monkeypatch: pytest.MonkeyPatch, report: str | set[str]
) -> None:
    """A formula is judged by its design, whether or not the report names the filled column."""
    fix_report(monkeypatch, report)
    data = formula_data()
    filled = data.frame["gap"].fillna(0.0).to_numpy(dtype=float)
    design = np.column_stack([np.ones(N_TEXTS), filled - filled.mean()])
    by_hand = ols_outcome_ppi(data, X=design, coefficient=1)
    for formula in ("1 + center(gap.fillna(0))", FULL_WIDTH, "1 + gap.fillna(0)"):
        estimate = ols_outcome_ppi(data, formula, coefficient=1)
        assert estimate.point == pytest.approx(by_hand.point, rel=1e-12)
        assert estimate.se == pytest.approx(by_hand.se, rel=1e-12)


@pytest.mark.skipif(
    formulaic.__version__ != "1.2.2", reason="the report described is that of formulaic 1.2.2"
)
def test_the_report_of_version_1_2_2_names_a_builtin_outside_a_transform_and_none_inside() -> None:
    data = formula_data()
    with pytest.raises(ConfigurationError, match=r"names column\(s\) 'abs' that the data lack"):
        ols_outcome_ppi(data, "1 + abs(log_chars)")
    assert np.isfinite(ols_outcome_ppi(data, "1 + center(abs(log_chars))").point)


def test_a_formula_that_turns_a_missing_value_into_a_value_is_evaluated_as_written() -> None:
    """A comparison is false for NaN in a column of floats, so its design holds no missing value."""
    data = formula_data()
    assert int(data.frame["gap"].isna().sum()) == 3
    above = (data.frame["gap"] > 0).to_numpy(dtype=float)
    by_hand = ols_outcome_ppi(data, X=np.column_stack([np.ones(N_TEXTS), above]), coefficient=1)
    estimate = ols_outcome_ppi(data, "1 + I(gap > 0)", coefficient=1)
    assert estimate.point == pytest.approx(by_hand.point, rel=1e-12)
    assert estimate.se == pytest.approx(by_hand.se, rel=1e-12)


@pytest.mark.parametrize("estimator", ["outcome", "predictor"])
def test_the_design_of_a_formula_is_asked_for_once_for_each_estimate(estimator: str) -> None:
    """With random draws in a formula, an estimate rests on the design of one request.

    ``formulaic`` evaluates a factor more than once for one design, so the comparison is with
    the designs it gives under the same seed, one request for each estimate, and with the
    random draw that follows the last of them.
    """
    data = formula_data()
    term = "center(log_chars + np.random.normal(size=len(log_chars)))"
    state = np.random.get_state()
    try:
        np.random.seed(8234)
        if estimator == "outcome":
            estimates = [ols_outcome_ppi(data, f"1 + {term}", coefficient=1) for _ in range(3)]
        else:
            estimates = [ols_predictor_ppi(data, "y", covariates=term) for _ in range(3)]
        next_draw = np.random.normal()
        np.random.seed(8234)
        for estimate in estimates:
            design = np.asarray(formulaic.model_matrix(f"1 + {term}", data.frame, context={}))
            if estimator == "outcome":
                by_hand = ols_outcome_ppi(data, X=design, coefficient=1)
            else:
                by_hand = ols_predictor_ppi(data, "y", X=design[:, 1:])
            assert estimate.point == pytest.approx(by_hand.point, rel=1e-12)
            assert estimate.se == pytest.approx(by_hand.se, rel=1e-12)
        assert np.random.normal() == next_draw
        assert estimates[0].point != estimates[1].point
    finally:
        np.random.set_state(state)


def test_a_missing_value_that_the_formula_produces_is_refused_with_the_library_s_words() -> None:
    with pytest.raises(ConfigurationError, match="cannot be evaluated on the data") as refusal:
        ols_outcome_ppi(formula_data(), "1 + lag(log_chars)")
    assert isinstance(refusal.value.__cause__, ValueError)
    assert str(refusal.value.__cause__) in str(refusal.value)


@pytest.mark.parametrize("formula", UNEVALUABLE)
def test_a_missing_column_that_is_not_reported_fails_when_it_is_read(
    monkeypatch: pytest.MonkeyPatch, formula: str
) -> None:
    fix_report(monkeypatch, "empty")
    with pytest.raises(ConfigurationError) as refusal:
        ols_outcome_ppi(formula_data(), formula)
    cause = refusal.value.__cause__
    assert isinstance(cause, FactorEvaluationError)
    message = str(refusal.value)
    assert f"formula {formula!r} cannot be evaluated on the data" in message
    assert str(cause) in message
    assert formula.rsplit(" + ", 1)[1] in str(cause)


@pytest.mark.parametrize("report", REPORTS)
@pytest.mark.parametrize("formula", OWN_NAMES)
def test_a_formula_sees_no_variable_of_the_library(
    monkeypatch: pytest.MonkeyPatch, report: str, formula: str
) -> None:
    fix_report(monkeypatch, report)
    with pytest.raises(ConfigurationError, match="cannot be evaluated on the data") as refusal:
        ols_outcome_ppi(formula_data(), formula)
    assert isinstance(refusal.value.__cause__, FactorEvaluationError)
    assert "NameError" in str(refusal.value)


@pytest.mark.parametrize("report", REPORTS)
def test_the_covariates_of_a_predictor_regression_are_checked_in_the_same_way(
    monkeypatch: pytest.MonkeyPatch, report: str
) -> None:
    fix_report(monkeypatch, report)
    data = formula_data()
    assert np.isfinite(ols_predictor_ppi(data, "y", covariates="scale(`log chars`)").point)
    with pytest.raises(ConfigurationError, match="covariates 'center.gap.' cannot be evaluated"):
        ols_predictor_ppi(data, "y", covariates="center(gap)")
    with pytest.raises(ConfigurationError, match="covariates .* cannot be evaluated"):
        ols_predictor_ppi(data, "y", covariates="center(frame.gap.fillna(0))")


@pytest.mark.parametrize("report", REPORTS)
def test_columns_with_unusual_names_do_not_disturb_a_formula_that_does_not_read_them(
    monkeypatch: pytest.MonkeyPatch, report: str
) -> None:
    fix_report(monkeypatch, report)
    frame = formula_frame()
    plain = ols_outcome_ppi(formula_data(frame), "1 + log_chars")
    twice = frame[["log_chars", "log_chars"]].set_axis(["extra", "extra"], axis=1)
    numbered = pd.concat([frame.assign(**{"in": frame["gap"]}), twice], axis=1)
    numbered[10] = frame["log_chars"]
    data = formula_data(numbered)
    assert ols_outcome_ppi(data, "1 + log_chars").point == plain.point
    assert np.isfinite(ols_outcome_ppi(data, "1 + log_chars %in% y").point)
    with pytest.raises(ConfigurationError, match="cannot be evaluated on the data"):
        ols_outcome_ppi(data, "1 + center(`10`)")


@pytest.mark.parametrize("report", REPORTS)
def test_a_column_named_like_a_function_hides_the_function(
    monkeypatch: pytest.MonkeyPatch, report: str
) -> None:
    fix_report(monkeypatch, report)
    frame = formula_frame()
    data = formula_data(frame.assign(center=frame["log_chars"]))
    with pytest.raises(ConfigurationError, match="cannot be evaluated on the data") as refusal:
        ols_outcome_ppi(data, "1 + center(log_chars)")
    assert isinstance(refusal.value.__cause__, FactorEvaluationError)
    assert "not callable" in str(refusal.value)
