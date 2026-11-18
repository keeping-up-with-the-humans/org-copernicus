"""Latent pooling records an orientation, a direction or an iteration that the data leave open.

The sign of the factor is set by its correlation with the row mean of the standardised scores,
which a constant row mean does not fix; two equal largest eigenvalues in a round of the
principal-axis iteration leave the eigenvector taken there one of many; and the iteration can
reach its cap of rounds. Every value stays as it was; the provenance says which case occurred,
and each case logs one warning per fit. The assertions pin the results for fixed test inputs.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest
from traces_support import (
    close,
    columns_data,
    mirror_columns,
    orthogonal_columns,
    strong_columns,
    weak_columns,
)

from copernicus.corrections import get_correction
from copernicus.corrections.families._factor import (
    PAF_MAX_ITER,
    paf_fit,
    standardize_apply,
    standardize_fit,
)
from copernicus.corrections.families.latent import LatentFitted

pytestmark = pytest.mark.filterwarnings("error")

LATENT = "copernicus.corrections.families.latent"
RECORDS = ("orientation_identified", "direction_unique", "factor_converged", "n_factor_rounds")


def _fit(columns: dict[str, np.ndarray], gold: np.ndarray) -> LatentFitted:
    data = columns_data(columns, gold)
    fitted = get_correction("latent_pooling", "factor_anchor").fit(
        data, data.ids, rng=np.random.default_rng(0)
    )
    assert isinstance(fitted, LatentFitted)
    return fitted


def _warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        record.getMessage()
        for record in caplog.records
        if record.name == LATENT and record.levelno == logging.WARNING
    ]


def _records(fitted: LatentFitted) -> tuple[object, ...]:
    return tuple(fitted.provenance[key] for key in RECORDS)


def test_a_constant_row_mean_leaves_the_orientation_open(caplog: pytest.LogCaptureFixture) -> None:
    # Two columns that mirror each other standardise to z and -z, whose row mean is 0.
    with caplog.at_level(logging.WARNING):
        fitted = _fit(*mirror_columns())
    assert _records(fitted) == (False, True, True, 1)
    (message,) = _warnings(caplog)
    assert message.startswith(
        "latent pooling: the row mean of the standardised score columns is constant on the "
        "training rows (is_constant), so the sign of the factor is not set by a correlation"
    )
    assert "orientation_identified is False" in message
    # The sign of the factor is the one that the decomposition returned; the loadings and the
    # slope of the anchor line carry it, so both are compared up to that sign.
    sign = float(np.sign(fitted.loadings[0]))
    assert (sign * fitted.loadings).tolist() == close([0.9999999999999998, -0.9999999999999998])
    assert (sign * fitted.slope, fitted.intercept) == close(
        (0.4842955162450939, -0.040985261366970414)
    )


def test_two_equal_largest_eigenvalues_leave_the_direction_one_of_many(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Two exactly orthogonal columns: the reduced matrix of the first round is 0.001 times I.
    # Which eigenvector the decomposition returns for it is its own choice and differs between
    # numerical libraries, and with it the rounds that follow and the slope of the anchor line.
    # The expectations are therefore computed here, by the same calls on the same matrix; that
    # the first round meets a tie holds whatever the vector.
    columns, gold = orthogonal_columns()
    with caplog.at_level(logging.WARNING):
        fitted = _fit(columns, gold)
    matrix = np.column_stack(list(columns.values()))
    standardised = standardize_apply(matrix, *standardize_fit(matrix))
    fit = paf_fit(standardised.T @ standardised / standardised.shape[0])
    assert not fit.direction_unique and fit.tied_rounds[0] == 1
    assert _records(fitted)[1:] == (False, fit.converged, fit.rounds)
    (message,) = (m for m in _warnings(caplog) if "direction_unique is False" in m)
    assert message.startswith(
        f"latent pooling: in {len(fit.tied_rounds)} of the {fit.rounds} rounds of the "
        "principal-axis iteration the two largest eigenvalues of the reduced correlation matrix "
        "are equal up to a relative 1e-12"
    )
    assert np.array_equal(np.abs(fitted.loadings), np.abs(fit.loadings))
    # The factor score has the mean 0 whatever its direction, so the anchor line passes through
    # the mean of the gold values.
    assert np.isfinite(fitted.slope)
    assert fitted.intercept == pytest.approx(float(np.mean(gold)), abs=1e-12)


def test_an_iteration_at_its_cap_of_rounds_is_recorded_as_not_converged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        fitted = _fit(*weak_columns())
    assert _records(fitted) == (True, True, False, PAF_MAX_ITER)
    (message,) = _warnings(caplog)
    assert message.startswith(
        "latent pooling: the principal-axis iteration reached its cap of 50 rounds without the "
        "communalities changing by less than 1e-06 in a round"
    )
    assert "factor_converged is False" in message
    expected = [-0.08630243816839006, 0.21195287326394224, 0.5083879947545962]
    assert fitted.loadings.tolist() == close(expected)
    assert (fitted.slope, fitted.intercept) == close((0.7838863388933244, 0.03260294024969081))


def test_an_ordinary_fit_records_nothing_open_and_logs_nothing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        fitted = _fit(*strong_columns())
    orientation, direction, converged, rounds = _records(fitted)
    assert (orientation, direction, converged) == (True, True, True)
    assert isinstance(rounds, int) and 1 <= rounds < PAF_MAX_ITER
    assert _warnings(caplog) == []
    expected = [0.8558433201816431, 0.8010830000701479, 0.7118760904146417]
    assert fitted.loadings.tolist() == close(expected)
    assert (fitted.slope, fitted.intercept) == close((0.44547803451559354, 0.012728882877384301))
