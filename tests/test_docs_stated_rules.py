"""The numbers that the methods pages state for the library's rules are those of the code.

A floor, a cap, a penalty or a tolerance that a page names is a constant of the code. Each case
holds the sentence of the page that states the number, the constant, and the number: the page
must hold the sentence, and the constant must have the value, so that neither can change alone.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from copernicus.baseline import ridge as baseline_ridge
from copernicus.coders import results
from copernicus.corrections import _simex, crossfit
from copernicus.corrections.families import (
    _factor,
    _ridge_path,
    calibration,
    imputation,
    misclassification,
)
from copernicus.validity import metrics

METHODS = Path(__file__).resolve().parents[1] / "docs" / "methods"
LATENT = "corrections/latent-pooling.md"
CLASSICAL = "corrections/classical-corrections.md"
EMBEDDING = "corrections/embedding-augmented-calibration.md"
STACKING = "corrections/supervised-stacking.md"
CONSTANT_RULE = "a range at most `1e-12` times the larger of 1 and its largest absolute value"
STATED = [
    (LATENT, "they are kept between 0.001 and 1", _factor.COMMUNALITY_FLOOR, 1e-3),
    (LATENT, "kept between `1e-6` and 1, are the", _factor.LOADING_COMMUNALITY_FLOOR, 1e-6),
    (LATENT, "change by less than `1e-6` or after", _factor.PAF_TOLERANCE, 1e-6),
    (LATENT, "or after 50 rounds", _factor.PAF_MAX_ITER, 50),
    (LATENT, "equal up to a relative `1e-12`", _factor.TIE_RELATIVE_TOLERANCE, 1e-12),
    (LATENT, CONSTANT_RULE, metrics.CONSTANT_RELATIVE_TOLERANCE, 1e-12),
    (
        "corrections/correction-ensemble.md",
        "solved with `1e-3` added to the diagonal",
        _factor.REGRESSION_RIDGE,
        1e-3,
    ),
    (CLASSICAL, "of 0, 0.5, 1, 1.5 and 2,", _simex.LAMBDAS, (0.0, 0.5, 1.0, 1.5, 2.0)),
    (CLASSICAL, "a log grid of 181 values", _simex.POLE_GRID_SIZE, 181),
    (CLASSICAL, "values from 0.001 to 10^6", _simex.POLE_GRID_LOW_EXPONENT, -3.0),
    (CLASSICAL, "values from 0.001 to 10^6", _simex.POLE_MAX, 1e6),
    (CLASSICAL, "at most `1e-9` times its largest absolute", _simex.CONSTANT_TOLERANCE, 1e-9),
    (
        "corrections/regression-calibration.md",
        "default L2 penalty (`C = 1`) and up to 1000 iterations",
        (calibration.PLATT_C, calibration.PLATT_MAX_ITER),
        (1.0, 1000),
    ),
    (
        EMBEDDING,
        "at or below `1e-12` times the larger of the largest singular value and `1e-300`",
        (_ridge_path.SINGULAR_RELATIVE_TOLERANCE, _ridge_path.SINGULAR_FLOOR),
        (1e-12, 1e-300),
    ),
    (EMBEDDING, "`min(5, groups)` grouped folds", crossfit.INNER_FOLDS, 5),
    (STACKING, "`min(5, groups)` grouped inner splits", crossfit.INNER_FOLDS, 5),
    (STACKING, CONSTANT_RULE, metrics.CONSTANT_RELATIVE_TOLERANCE, 1e-12),
    (
        "corrections/multiple-imputation.md",
        "one of its `k_donors` (5) nearest training rows by fitted mean, chosen at random, so "
        "every draw lies on the observed gold values; with fewer training rows than `k_donors`, "
        "every training row is a donor",
        imputation.MultipleImputation("pmm").k_donors,
        5,
    ),
    (
        "nlp-ml-baseline.md",
        "at or below `1e-10` times the larger of the largest eigenvalue and `1e-300` are dropped",
        baseline_ridge.EIGENVALUE_RELATIVE_TOLERANCE,
        1e-10,
    ),
    (
        "corrections/misclassification-matrix.md",
        "minus one is at most `1e-12` in absolute value",
        misclassification.SINGULAR_TOLERANCE,
        1e-12,
    ),
    (
        "validity-panel.md",
        "at most `1e-12` times the larger of 1 and its largest absolute value",
        metrics.CONSTANT_RELATIVE_TOLERANCE,
        1e-12,
    ),
    (
        "validity-panel.md",
        "whose absolute value is above `1e150`, under either unit",
        results.MAX_ABSOLUTE_VALUE,
        1e150,
    ),
]


@pytest.mark.parametrize(("page", "sentence", "constant", "value"), STATED)
def test_a_number_that_a_page_states_is_the_constant_of_the_code(
    page: str, sentence: str, constant: object, value: object
) -> None:
    text = (METHODS / page).read_text(encoding="utf-8")
    assert sentence in text, f"{page} no longer holds: {sentence}"
    assert constant == value
