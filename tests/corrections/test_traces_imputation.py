"""Multiple imputation records the bootstrap fits of rank below 3 or without residual freedom.

A resample whose design ``[1, score, covariate]`` has a numerical rank below 3, as
``numpy.linalg.lstsq`` with ``rcond=None`` returns it, keeps the solution of smallest norm that
``lstsq`` returns; in these data that is a resample of at most two distinct texts, whose design
has many least-squares solutions. For a resample of no more rows than the three coefficients
the residual variance is the
residual sum of squares divided by 1. Both values stay; the provenance counts the resamples and
each case logs one warning per fit. The expected values are replayed from the documented
bootstrap and compared exactly.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest
from corrections_support import planted_data
from estimates_support import RELATIVE_TOLERANCE
from traces_support import imputation_data, imputation_design, replayed_resamples

from copernicus.corrections import CorrectionData, get_correction
from copernicus.corrections.families.imputation import ImputationFitted

pytestmark = pytest.mark.filterwarnings("error")

IMPUTATION = "copernicus.corrections.families.imputation"
M_DRAWS, SEED = 20, 0
# Above the residual variance that rounding leaves of a fit through every training row.
ROUNDING_OF_AN_EXACT_FIT = 1e-20
VARIANTS = ("model_draws", "pmm")


def _fit(data: CorrectionData, variant: str) -> ImputationFitted:
    correction = get_correction("multiple_imputation", variant, m_draws=M_DRAWS)
    fitted = correction.fit(data, data.ids, rng=np.random.default_rng(SEED))
    assert isinstance(fitted, ImputationFitted)
    return fitted


def _warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        record.getMessage()
        for record in caplog.records
        if record.name == IMPUTATION and record.levelno == logging.WARNING
    ]


def _replayed(data: CorrectionData) -> tuple[list[int], np.ndarray, np.ndarray]:
    """The rank, the minimum-norm coefficients and the variance of every replayed resample."""
    design, gold = imputation_design(data), data.gold
    ranks, coefficients, variances = [], [], []
    for sample in replayed_resamples(data, M_DRAWS, SEED):
        beta, _, rank, _ = np.linalg.lstsq(design[sample], gold[sample], rcond=None)
        residual = gold[sample] - design[sample] @ beta
        ranks.append(int(rank))
        coefficients.append(beta)
        variances.append(float(residual @ residual) / max(data.n - 3, 1))
    return ranks, np.vstack(coefficients), np.array(variances)


@pytest.mark.parametrize("variant", VARIANTS)
def test_rank_deficient_resamples_are_counted_and_warned_about_once(
    variant: str, caplog: pytest.LogCaptureFixture
) -> None:
    # Four training texts: a resample that draws at most two of them has a design of rank 2.
    data = imputation_data(4)
    ranks, coefficients, variances = _replayed(data)
    deficient = sum(rank < 3 for rank in ranks)
    assert 0 < deficient < M_DRAWS
    with caplog.at_level(logging.WARNING):
        fitted = _fit(data, variant)
    assert fitted.provenance["n_rank_deficient_resamples"] == deficient
    assert fitted.provenance["n_resamples_without_residual_df"] == 0
    (message,) = _warnings(caplog)
    assert message.startswith(
        f"multiple_imputation/{variant}: {deficient} of the {M_DRAWS} bootstrap resamples of the "
        "4 training rows have a design of intercept, score 'score' and covariate of numerical rank "
        "below 3, as numpy.linalg.lstsq with rcond=None returns it"
    )
    assert "n_rank_deficient_resamples" in message
    # The values stay: the coefficients of smallest norm, the variance on n - 3 = 1 degree.
    assert np.array_equal(fitted.models.coefficients, coefficients)
    assert np.array_equal(fitted.models.residual_variance, variances)
    assert fitted.provenance["within_variance"] == float(variances.mean())
    assert float(variances.mean()) == pytest.approx(
        0.0087096556240434, rel=RELATIVE_TOLERANCE, abs=0.0
    )


@pytest.mark.parametrize("variant", VARIANTS)
def test_three_training_rows_put_every_resample_on_the_divisor_of_one(
    variant: str, caplog: pytest.LogCaptureFixture
) -> None:
    # Three training texts for three coefficients: every resample has as many rows as columns.
    data = imputation_data(3)
    ranks, coefficients, variances = _replayed(data)
    deficient = sum(rank < 3 for rank in ranks)
    assert 0 < deficient < M_DRAWS
    with caplog.at_level(logging.WARNING):
        fitted = _fit(data, variant)
    assert fitted.provenance["n_resamples_without_residual_df"] == M_DRAWS
    assert fitted.provenance["n_rank_deficient_resamples"] == deficient
    rank_message, df_message = _warnings(caplog)
    assert rank_message.startswith(f"multiple_imputation/{variant}: {deficient} of the 20 ")
    assert df_message.startswith(
        f"multiple_imputation/{variant}: 20 of the 20 bootstrap fits have no more rows than "
        "coefficients (3 rows for 3 coefficients)"
    )
    assert "divided by 1" in df_message and "n_resamples_without_residual_df" in df_message
    # The value stays: the residual sum of squares divided by 1.
    assert np.array_equal(fitted.models.residual_variance, variances)
    assert np.array_equal(fitted.models.coefficients, coefficients)
    # An exact fit: what is left is rounding, whose digits differ between platforms.
    assert fitted.provenance["within_variance"] == float(variances.mean())
    assert 0.0 <= float(variances.mean()) < ROUNDING_OF_AN_EXACT_FIT


@pytest.mark.parametrize("variant", VARIANTS)
def test_an_ordinary_fit_counts_nothing_and_logs_nothing(
    variant: str, caplog: pytest.LogCaptureFixture
) -> None:
    noise = np.random.default_rng(3).normal(scale=0.5, size=400)
    data = planted_data(
        lambda score, covariate: 1.0 + 2.0 * score + noise, n_groups=40, group_size=10
    )
    with caplog.at_level(logging.WARNING):
        fitted = _fit(data, variant)
    assert fitted.provenance["n_rank_deficient_resamples"] == 0
    assert fitted.provenance["n_resamples_without_residual_df"] == 0
    assert _warnings(caplog) == []
