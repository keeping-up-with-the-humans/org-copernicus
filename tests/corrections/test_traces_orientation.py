"""An orientation set by a correlation of at most 1e-12 is recorded as not identified.

The sign of a factor or of a first principal component is set by its correlation with the mean
of the standardised columns. For two columns that are both scaled to the variance 1 and
correlate negatively that correlation is zero in exact arithmetic: the component is their
difference and the mean is half their sum. The computed number is then of the size of rounding
errors, and its sign changes from one data set to the next. The library treats a computed
correlation of at most ``1e-12`` in absolute value as too small to identify the sign: the
value stays, the record ``orientation_identified`` is false, and one warning says so. The
threshold is a convention: a correlation below it can be genuine, and rounding can leave one
above it.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest
from traces_support import columns_data, members_of

from copernicus.corrections import CorrectionEnsemble, get_correction
from copernicus.corrections.families import _factor

pytestmark = pytest.mark.filterwarnings("error")

LATENT = "copernicus.corrections.families.latent"
ENSEMBLE = "copernicus.corrections.families.ensemble"


def _opposed(seed: int) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Two columns that correlate at about -0.5, and gold values."""
    rng = np.random.default_rng(seed)
    first = rng.normal(size=60)
    return {"a": first, "b": -0.6 * first + rng.normal(size=60)}, first + rng.normal(size=60)


def _warnings(caplog: pytest.LogCaptureFixture, name: str) -> list[str]:
    return [
        r.getMessage() for r in caplog.records if r.name == name and r.levelno == logging.WARNING
    ]


def test_a_correlation_of_zero_up_to_rounding_does_not_identify_the_orientation() -> None:
    first = np.array([1.0, -1.0, 1.0, -1.0])
    second = np.array([1.0, 1.0, -1.0, -1.0])
    assert _factor.orientation_identified(first, second) is False
    assert _factor.relation_sign(first, second) == 1.0
    scores = np.arange(8.0)
    assert _factor.orientation_identified(scores, scores + 1e-6 * second.repeat(2)) is True
    assert _factor.ORIENTATION_TOLERANCE == 1e-12
    nearly = np.array([1.0, -1.0, 1.0, -1.0 + 1e-9])
    correlation = float(np.corrcoef(nearly, second)[0, 1])
    assert 1e-12 < abs(correlation) < 1e-6
    assert _factor.orientation_identified(nearly, second) is True


def test_the_threshold_is_a_convention_on_the_computed_correlation() -> None:
    # A correlation that is genuine and lies below the threshold: it counts as not identifying.
    u, v = np.array([1.0, -1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0, -1.0])
    slightly = v + 2.0**-43 * u
    genuine = _factor.orientation_correlation(slightly, u)
    assert genuine == pytest.approx(2.0**-43, rel=1e-6)
    assert _factor.orientation_identified(slightly, u) is False
    assert _factor.relation_sign(slightly, u) == 1.0
    assert _factor.relation_sign(-slightly, u) == -1.0
    # Three columns that correlate negatively in pairs: their component does correlate with
    # their mean, so the rule of the two columns does not carry over.
    rng = np.random.default_rng(2)
    covariance = np.array([[1.0, -0.2, -0.4], [-0.2, 1.0, -0.1], [-0.4, -0.1, 1.0]])
    columns = rng.normal(size=(4000, 3)) @ np.linalg.cholesky(covariance).T
    standardised = _factor.standardize_apply(columns, *_factor.standardize_fit(columns))
    centred = standardised - standardised.mean(axis=0)
    component = centred @ np.linalg.svd(centred, full_matrices=False)[2][0]
    assert abs(_factor.orientation_correlation(component, standardised.mean(axis=1))) > 0.05


def test_values_treated_as_constant_are_not_asked_for_their_correlation() -> None:
    # Four values 1e-13 apart vary, and their mirror image correlates with them at -1; their
    # range lies within the tolerance of is_constant, so no correlation is consulted and the
    # sign stays 1: the mirror image is returned as it is.
    values = 1e-13 * np.arange(4.0)
    assert np.corrcoef(-values, values)[0, 1] == pytest.approx(-1.0)
    assert np.isnan(_factor.orientation_correlation(-values, values))
    assert not _factor.orientation_identified(-values, values)
    assert _factor.relation_sign(-values, values) == 1.0
    assert np.array_equal(_factor.sign_align(-values, values), -values)


@pytest.mark.parametrize("variant", ["pc1", "factor"])
def test_two_opposed_members_leave_the_sign_of_the_combined_column_open(
    variant: str, caplog: pytest.LogCaptureFixture
) -> None:
    for seed in range(6):
        columns, _ = _opposed(seed)
        members, ids = members_of(columns)
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            combined = CorrectionEnsemble(variant).combine(members, ids)
        assert combined.provenance["orientation_identified"] is False
        (message,) = _warnings(caplog, ENSEMBLE)
        assert message.startswith(
            f"correction ensemble ({variant}): the correlation of the combined column with the "
            "mean of the standardised members is "
        )
        assert "at most 1e-12 in absolute value" in message
        assert "treats as too small to identify the sign" in message
        assert "follows the computed correlation" in message
        assert message.endswith("(orientation_identified is False)")
        # The sign follows the computed correlation, whatever rounding made of it: the column
        # is what the same calls return here, negated when their correlation with the mean of
        # the standardised members is below 0.
        matrix = np.column_stack([columns["a"], columns["b"]])
        standardised = _factor.standardize_apply(matrix, *_factor.standardize_fit(matrix))
        if variant == "pc1":
            centred = standardised - standardised.mean(axis=0)
            raw = centred @ np.linalg.svd(centred, full_matrices=False)[2][0]
        else:
            correlation = standardised.T @ standardised / standardised.shape[0]
            loadings = _factor.paf_fit(correlation).loadings
            raw = _factor.regression_factor_score(standardised, correlation, loadings)
        computed = float(np.corrcoef(raw, standardised.mean(axis=1))[0, 1])
        assert abs(computed) <= 1e-12
        assert np.array_equal(combined.column.to_numpy(), -raw if computed < 0.0 else raw)


def test_two_opposed_columns_leave_the_sign_of_the_factor_open(
    caplog: pytest.LogCaptureFixture,
) -> None:
    data = columns_data(*_opposed(0))
    with caplog.at_level(logging.WARNING):
        fitted = get_correction("latent_pooling", "factor_anchor").fit(
            data, data.ids, rng=np.random.default_rng(0)
        )
    assert fitted.provenance["orientation_identified"] is False
    (message,) = _warnings(caplog, LATENT)
    assert message.startswith(
        "latent pooling: the correlation of the factor with the row mean of the standardised "
        "score columns is "
    )
    assert "the anchor line's slope takes up that sign" in message
