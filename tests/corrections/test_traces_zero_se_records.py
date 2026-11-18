"""A standard error of exactly 0 is recorded wherever the library returns one.

Beside the details of an estimate (``make_estimate``), the repeated study counts the draws of
every row whose standard error is exactly 0, for each of its four estimators; ``rubin_pool``
and the stacked two-stage fit say so on their results; and the details of ``oriv`` say so for
its sandwich. Every value stays as it was.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest
from estimates_support import classical_data
from traces_support import study_corpus

from copernicus.corrections import CorrectionData, oriv, repeated_study, rubin_pool
from copernicus.corrections.estimates_oriv import stacked_2sls

pytestmark = pytest.mark.filterwarnings("error")

STUDY, IMPUTATION = "copernicus.corrections.study", "copernicus.corrections.families.imputation"
ZERO = "a standard error is exactly 0 (se_exactly_zero)"


def _warnings(caplog: pytest.LogCaptureFixture, name: str) -> list[str]:
    return [
        r.getMessage() for r in caplog.records if r.name == name and r.levelno == logging.WARNING
    ]


def _flat_corpus() -> CorrectionData:
    """Forty texts, each its own group, with the score 2 and the human value 3 on every text."""
    ids = [f"t{index:02d}" for index in range(40)]
    frame = pd.DataFrame(
        {
            "text_id": ids,
            "group_id": ids,
            "score": 2.0,
            "gold": 3.0,
            "truth": 3.0,
            "log_chars": np.random.default_rng(0).normal(size=40),
        }
    )
    return CorrectionData(frame=frame, score_columns=("score",))


def test_the_study_counts_the_zero_standard_errors_of_the_plugged_in_score(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # The raw score is the same on every text, so its naive standard error is 0 on every draw.
    with caplog.at_level(logging.WARNING):
        table = repeated_study(
            _flat_corpus(), truth_gold="truth", sizes=(10,), draws=3, estimators=("raw",)
        )
    assert table["se_mean"].tolist() == [0.0, 0.0]
    assert table["n_se_exactly_zero"].tolist() == [3, 3]
    assert str(table["n_se_exactly_zero"].dtype) == "int64"
    (summary,) = _warnings(caplog, STUDY)
    assert f", 6 say that {ZERO}; the column n_se_exactly_zero holds" in summary


def test_a_study_whose_standard_errors_are_positive_counts_none(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        table = repeated_study(
            study_corpus(covariate_on_score=False), truth_gold="truth", sizes=(20,), draws=4
        )
    assert (table["n_se_exactly_zero"] == 0).all() and (table["se_mean"] > 0.0).all()
    assert _warnings(caplog, STUDY) == []


def test_a_pooled_standard_error_of_zero_is_recorded_and_warned(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        pooled = rubin_pool([1.0, 1.0], [0.0, 0.0])
    assert (pooled.estimate, pooled.se, pooled.se_exactly_zero) == (1.0, 0.0, True)
    (message,) = _warnings(caplog, IMPUTATION)
    assert message.startswith("rubin_pool: the standard error is exactly 0")
    assert "the computed within-draw and between-draw variances are both 0" in message
    assert message.endswith("the pooled estimate records se_exactly_zero")


def test_a_pooled_standard_error_above_zero_is_recorded_as_such(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        pooled = rubin_pool([1.0, 2.0], [0.1, 0.1])
        # Equal estimates of 0.1: rounding leaves a between-draw variance that is not 0.
        rounded = rubin_pool([0.1, 0.1, 0.1], [0.0, 0.0, 0.0])
    assert pooled.se > 0.0 and pooled.se_exactly_zero is False
    assert 0.0 < rounded.se < 1e-15 and rounded.se_exactly_zero is False
    assert _warnings(caplog, IMPUTATION) == []


def test_the_stacked_fit_records_a_standard_error_of_zero() -> None:
    # An outcome of zeros leaves no residual, so every score row of the sandwich is 0.
    zero = stacked_2sls(np.zeros(4), np.array([1.0, 2.0, 3.0, 4.0]), np.array([2.0, 4.0, 5.0, 8.0]))
    assert (zero.slope, zero.se, zero.se_exactly_zero) == (0.0, 0.0, True)
    rng = np.random.default_rng(7)
    construct = rng.normal(size=200)
    fit = stacked_2sls(
        construct + rng.normal(size=200),
        construct + 0.5 * rng.normal(size=200),
        construct + 0.7 * rng.normal(size=200),
    )
    assert fit.se > 0.0 and fit.se_exactly_zero is False


def test_oriv_records_a_sandwich_of_zero_beside_its_own_record() -> None:
    data = classical_data(2, n_texts=200, n_labelled=40)
    ordinary = oriv(data, "outcome", score_a="score_a", score_b="score_b", n_boot=60).details
    assert ordinary["sandwich_se"] > 0.0 and ordinary["sandwich_se_exactly_zero"] is False
    flat = CorrectionData(frame=data.frame.assign(outcome=0.0), score_columns=data.score_columns)
    details = oriv(flat, "outcome", score_a="score_a", score_b="score_b", n_boot=60).details
    assert (details["sandwich_se"], details["sandwich_se_exactly_zero"]) == (0.0, True)
    assert list(details)[-1] == "se_exactly_zero" and details["se_exactly_zero"] is True
