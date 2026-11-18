"""The ``factor`` and ``pc1`` combiners record an orientation or a direction the data leave open.

Their sign is set by the correlation with the mean of the standardised members, which a constant
mean does not fix; two equal largest eigenvalues of a round of the principal-axis iteration, or
two equal largest singular values of the members, leave the direction one of many; and the
iteration of ``factor`` can reach its cap of rounds. Every value stays as it was; the provenance
says which case occurred, and each case logs one warning per call. The assertions pin the
results for fixed test inputs.
"""

from __future__ import annotations

import logging
from types import MappingProxyType

import numpy as np
import pandas as pd
import pytest
from traces_support import (
    close,
    members_of,
    mirror_columns,
    orthogonal_columns,
    strong_columns,
    weak_columns,
)

from copernicus.corrections.families._factor import (
    PAF_MAX_ITER,
    paf_fit,
    standardize_apply,
    standardize_fit,
)
from copernicus.corrections.families.ensemble import CorrectionEnsemble, EnsembleResult

pytestmark = pytest.mark.filterwarnings("error")

ENSEMBLE = "copernicus.corrections.families.ensemble"
FACTOR_RECORDS = (
    "orientation_identified",
    "direction_unique",
    "factor_converged",
    "n_factor_rounds",
)
PC1_RECORDS = ("orientation_identified", "direction_unique")


def _combine(variant: str, columns: dict[str, np.ndarray]) -> EnsembleResult:
    members, ids = members_of(columns)
    return CorrectionEnsemble(variant).combine(members, ids)


def _warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        record.getMessage()
        for record in caplog.records
        if record.name == ENSEMBLE and record.levelno == logging.WARNING
    ]


def _records(result: EnsembleResult, keys: tuple[str, ...]) -> tuple[object, ...]:
    return tuple(result.provenance[key] for key in keys)


def _signed(result: EnsembleResult, key: str) -> list[float]:
    mapping = result.provenance[key]
    assert isinstance(mapping, MappingProxyType)
    return [float(value) for value in mapping.values()]


def test_a_constant_mean_of_the_members_leaves_the_orientation_open(
    caplog: pytest.LogCaptureFixture,
) -> None:
    columns, _ = mirror_columns()
    with caplog.at_level(logging.WARNING):
        factor = _combine("factor", columns)
        pc1 = _combine("pc1", columns)
    assert _records(factor, FACTOR_RECORDS) == (False, True, True, 1)
    assert _records(pc1, PC1_RECORDS) == (False, True)
    factor_message, pc1_message = _warnings(caplog)
    for variant, message in (("factor", factor_message), ("pc1", pc1_message)):
        assert message.startswith(
            f"correction ensemble ({variant}): the mean of the standardised members is constant "
            "(is_constant), so the sign of the combined column is not set by a correlation"
        )
        assert "orientation_identified is False" in message
    # The sign of each column is the one that its decomposition returned, and the loadings or
    # the direction carry it, so the values are compared up to that sign.
    for result, key, values, first in (
        (factor, "loadings", [0.9999999999999998, -0.9999999999999998], 0.3684300873444709),
        (pc1, "direction", [0.7071067811865476, -0.7071067811865475], 0.5212993457220091),
    ):
        sign = float(np.sign(_signed(result, key)[0]))
        assert [sign * value for value in _signed(result, key)] == close(values)
        assert sign * float(result.column.iloc[0]) == close(first)


def test_two_equal_largest_eigenvalues_or_singular_values_leave_the_direction_one_of_many(
    caplog: pytest.LogCaptureFixture,
) -> None:
    columns, _ = orthogonal_columns()
    with caplog.at_level(logging.WARNING):
        factor = _combine("factor", columns)
        pc1 = _combine("pc1", columns)
    # Which vector a decomposition returns for two equal eigenvalues or singular values is
    # its own choice and differs between numerical libraries, and with it the rounds that
    # follow. The expectations are therefore computed here, by the same calls on the same
    # matrices; that the first round meets a tie holds whatever the vector.
    matrix = np.column_stack(list(columns.values()))
    standardised = standardize_apply(matrix, *standardize_fit(matrix))
    fit = paf_fit(standardised.T @ standardised / standardised.shape[0])
    assert not fit.direction_unique and fit.tied_rounds[0] == 1
    assert _records(factor, FACTOR_RECORDS)[1:] == (False, fit.converged, fit.rounds)
    assert _records(pc1, PC1_RECORDS)[1] is False
    factor_message, pc1_message = (
        message for message in _warnings(caplog) if "direction_unique is False" in message
    )
    assert factor_message.startswith(
        f"correction ensemble (factor): in {len(fit.tied_rounds)} of the {fit.rounds} rounds of "
        "the principal-axis iteration the two largest eigenvalues of the reduced correlation "
        "matrix are equal up to a relative 1e-12"
    )
    assert pc1_message.startswith(
        "correction ensemble (pc1): the two largest singular values of the centred standardised "
        "members are equal up to a relative 1e-12"
    )
    assert np.array_equal(np.abs(_signed(factor, "loadings")), np.abs(fit.loadings))
    centred = standardised - standardised.mean(axis=0)
    direction = np.array(_signed(pc1, "direction"))
    returned = np.linalg.svd(centred, full_matrices=False)[2][0]
    assert np.array_equal(np.abs(direction), np.abs(returned))
    assert float(np.linalg.norm(direction)) == close(1.0)
    assert pc1.column.to_numpy() == close(centred @ direction)


def test_a_factor_iteration_at_its_cap_of_rounds_is_recorded_as_not_converged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    columns, _ = weak_columns()
    with caplog.at_level(logging.WARNING):
        factor = _combine("factor", columns)
        pc1 = _combine("pc1", columns)
    assert _records(factor, FACTOR_RECORDS) == (True, True, False, PAF_MAX_ITER)
    assert _records(pc1, PC1_RECORDS) == (True, True)
    (message,) = _warnings(caplog)
    assert message.startswith(
        "correction ensemble (factor): the principal-axis iteration reached its cap of 50 rounds"
    )
    assert "factor_converged is False" in message
    expected = [-0.08630243816839006, 0.21195287326394224, 0.5083879947545962]
    assert _signed(factor, "loadings") == close(expected)
    assert float(factor.column.iloc[0]) == close(0.1645707066651441)
    assert float(pc1.column.iloc[0]) == close(0.7147122831609234)


def test_ordinary_members_record_nothing_open_and_log_nothing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    columns, _ = strong_columns()
    with caplog.at_level(logging.WARNING):
        factor = _combine("factor", columns)
        pc1 = _combine("pc1", columns)
        mean = _combine("mean", columns)
    orientation, direction, converged, rounds = _records(factor, FACTOR_RECORDS)
    assert (orientation, direction, converged) == (True, True, True)
    assert isinstance(rounds, int) and 1 <= rounds < PAF_MAX_ITER
    assert _records(pc1, PC1_RECORDS) == (True, True)
    assert _warnings(caplog) == []
    assert float(factor.column.iloc[0]) == close(-1.0427107933705495)
    assert float(pc1.column.iloc[0]) == close(-1.7049280038646548)
    assert set(mean.provenance) == {
        "family",
        "variant",
        "member_families",
        "n_members",
        "dropped_members",
    }


def test_the_mode_combiner_keeps_its_provenance() -> None:
    ids = np.array(["a", "b", "c", "d"])
    members = {
        "m1": pd.Series([0.1, 0.9, 0.4, 2.2], index=ids),
        "m2": pd.Series([0.2, 1.2, 1.1, 1.9], index=ids),
    }
    result = CorrectionEnsemble("mode").combine(
        members, ids, gold_classes=np.array([0.0, 1.0, 2.0])
    )
    assert set(result.provenance) == {
        "family",
        "variant",
        "member_families",
        "n_members",
        "dropped_members",
    }
