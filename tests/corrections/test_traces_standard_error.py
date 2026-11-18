"""Every estimate records a standard error of exactly 0, whose interval has zero width.

``make_estimate`` adds ``se_exactly_zero`` to the details of every estimator and logs one
warning when the standard error is exactly 0; the point, the standard error and the interval
stay as they were. The three bootstrapped estimators keep ``se_status`` beside it.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest
from estimates_support import (
    SPARSE_BOOT,
    SPARSE_SEED,
    classical_data,
    ppi_frame,
    slope_frame,
    sparse_covariation_data,
    with_labels,
)

from copernicus.corrections import (
    CorrectionData,
    Estimate,
    group_difference_ppi,
    mean_ppi,
    moment_subtraction,
    ols_outcome_ppi,
    ols_predictor_ppi,
    oriv,
    simex,
)
from copernicus.corrections.estimates import make_estimate

pytestmark = pytest.mark.filterwarnings("error")

ESTIMATES = "copernicus.corrections.estimates"
N_TEXTS = 12
COVARIATE = np.array([0.0, 1.0, 2.0, 3.0] * 3)
LABELLED = np.arange(N_TEXTS) % 2 == 0


def _texts(score: np.ndarray, gold: np.ndarray, **extra: np.ndarray) -> CorrectionData:
    """Twelve texts, each its own group; every second one is labelled, with pi = 0.5."""
    ids = [f"t{index:02d}" for index in range(N_TEXTS)]
    frame = pd.DataFrame(
        {
            "text_id": ids,
            "group_id": ids,
            "score": score,
            "log_chars": COVARIATE,
            "gold": np.where(LABELLED, gold, np.nan),
            "pi": 0.5,
            **extra,
        }
    )
    return CorrectionData(frame=frame, score_columns=("score",))


# The score 2 on every text and the gold value 3 on every labelled one: every contribution to the
# standard error of a mean or a difference is the same, 0. A constant outcome gives every
# bootstrap resample of the classical estimators the slope 0. These zeros come from sums and
# differences of equal numbers, which every platform computes exactly. A score and a gold value
# on the line 1 + 2 * covariate, with the outcome 3 * line - 1, leave every transformed
# contribution of the regressions 0 where the least-squares fit leaves no residual; a numerical
# library can also leave residuals of the size of rounding, and the standard error is then tiny
# and not 0.
FLAT = _texts(np.full(N_TEXTS, 2.0), np.full(N_TEXTS, 3.0))
LINE = 1.0 + 2.0 * COVARIATE
ON_THE_LINE = _texts(LINE, LINE, outcome=3.0 * LINE - 1.0)
CLASSICAL = classical_data(2, n_texts=200, n_labelled=40)
CONSTANT_OUTCOME = CorrectionData(
    frame=CLASSICAL.frame.assign(outcome=1.5), score_columns=CLASSICAL.score_columns
)
ZERO_CASES: dict[str, tuple[Callable[[], Estimate], float]] = {
    "ppi_mean": (lambda: mean_ppi(FLAT), 3.0),
    "ppi_difference": (lambda: group_difference_ppi(FLAT, COVARIATE >= 1.5), 0.0),
    "ppi_ols_outcome": (lambda: ols_outcome_ppi(ON_THE_LINE, formula="1 + log_chars"), 2.0),
    "ppi_ols_predictor": (lambda: ols_predictor_ppi(ON_THE_LINE, "outcome"), 3.0),
    "moment_subtraction": (lambda: moment_subtraction(CONSTANT_OUTCOME, "outcome"), 0.0),
    "simex": (lambda: simex(CONSTANT_OUTCOME, "outcome", n_boot=50), 0.0),
}
ZERO_THROUGH_A_FIT = ("ppi_ols_outcome", "ppi_ols_predictor")
ROUNDING = 1e-12


def _warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        record.getMessage()
        for record in caplog.records
        if record.name == ESTIMATES and record.levelno == logging.WARNING
    ]


def test_a_standard_error_of_zero_is_recorded_with_a_zero_width_interval(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        zero = make_estimate(2.5, 0.0, n_labelled=4, n_total=8, method="probe", details={"a": 1})
    assert zero.ci_low == zero.ci_high == zero.point == 2.5 and zero.se == 0.0
    assert list(zero.details) == ["a", "se_exactly_zero"]
    assert zero.details["se_exactly_zero"] is True
    (message,) = _warnings(caplog)
    assert message.startswith(
        "probe: the standard error is exactly 0, so the 95 percent interval has zero width "
        "(ci_low = ci_high = point)"
    )
    assert "se_exactly_zero" in message


@pytest.mark.parametrize("se", [0.5, 5e-324, float("nan")], ids=["positive", "tiny", "missing"])
def test_a_standard_error_other_than_zero_is_recorded_as_such(
    se: float, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING):
        estimate = make_estimate(2.5, se, n_labelled=4, n_total=8, method="probe", details={})
    assert estimate.details["se_exactly_zero"] is False
    assert _warnings(caplog) == []
    if math.isnan(se):
        assert math.isnan(estimate.ci_low) and math.isnan(estimate.ci_high)
    else:
        assert (estimate.ci_low, estimate.ci_high) == (2.5 - 1.96 * se, 2.5 + 1.96 * se)


@pytest.mark.parametrize("method", list(ZERO_CASES))
def test_every_estimator_records_its_standard_error_of_zero(
    method: str, caplog: pytest.LogCaptureFixture
) -> None:
    call, point = ZERO_CASES[method]
    with caplog.at_level(logging.WARNING):
        estimate = call()
    assert estimate.method == method
    found = (estimate.point, estimate.se, estimate.ci_low, estimate.ci_high)
    zero = estimate.se == 0.0
    if method in ZERO_THROUGH_A_FIT:
        # Zero or of the size of rounding, by the platform: the record follows the number.
        assert estimate.point == pytest.approx(point, abs=ROUNDING)
        assert 0.0 <= estimate.se < ROUNDING
    else:
        assert found == (point, 0.0, point, point)
    assert estimate.details["se_exactly_zero"] is zero
    messages = _warnings(caplog)
    assert len(messages) == int(zero)
    if zero:
        assert estimate.ci_low == estimate.ci_high == estimate.point
        assert messages[0].startswith(f"{method}: the standard error is exactly 0")


def _ordinary() -> list[Estimate]:
    """One estimate of every estimator on data whose standard error is positive."""
    mean_data = with_labels(
        ppi_frame(np.random.default_rng(3), 40),
        np.arange(40) % 2 == 0,
        score_columns=("score",),
        pi=0.5,
    )
    slope_data = with_labels(
        slope_frame(np.random.default_rng(4), 200),
        np.arange(200) % 4 == 0,
        score_columns=("score",),
        pi=0.25,
    )
    return [
        mean_ppi(mean_data),
        group_difference_ppi(mean_data, mean_data.covariate >= 0.0),
        ols_outcome_ppi(slope_data, "1 + log_chars"),
        ols_predictor_ppi(slope_data, "outcome"),
        moment_subtraction(CLASSICAL, "outcome"),
        simex(CLASSICAL, "outcome", n_boot=50),
        oriv(CLASSICAL, "outcome", score_a="score_a", score_b="score_b"),
    ]


def test_an_ordinary_standard_error_is_recorded_as_not_zero(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        estimates = _ordinary()
    assert [estimate.method for estimate in estimates] == [
        "ppi_mean",
        "ppi_difference",
        "ppi_ols_outcome",
        "ppi_ols_predictor",
        "moment_subtraction",
        "simex",
        "oriv",
    ]
    for estimate in estimates:
        assert estimate.se > 0.0 and estimate.details["se_exactly_zero"] is False
        assert list(estimate.details)[-1] == "se_exactly_zero"
    assert _warnings(caplog) == []


def test_the_bootstrapped_estimators_keep_their_status_beside_the_record() -> None:
    sparse = sparse_covariation_data(SPARSE_SEED)
    kept = moment_subtraction(
        sparse, "outcome", n_boot=SPARSE_BOOT, seed=SPARSE_SEED, below_floor="point"
    )
    assert math.isnan(kept.se) and kept.details["se_status"] == "too_few_defined"
    assert kept.details["se_exactly_zero"] is False
    zero = ZERO_CASES["moment_subtraction"][0]()
    assert zero.details["se_status"] == "ok" and zero.details["se_exactly_zero"] is True
    for estimate in (kept, zero):
        assert list(estimate.details)[-4:] == [
            "min_defined",
            "below_floor",
            "se_status",
            "se_exactly_zero",
        ]
