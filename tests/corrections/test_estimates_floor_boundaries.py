"""The floor of the bootstrap of moment subtraction, SIMEX and ORIV at its boundaries.

On ``sparse_covariation_data`` most resamples have no reverse calibration line. With 60
resamples and seed 2 each estimator has 24 defined resamples; with two resamples the seeds of
``TWO_RESAMPLES`` give none, one or both of them defined. The tests assert these counts, so a
changed fixture fails here instead of passing for another reason.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np
import pytest
from estimates_support import SPARSE_BOOT, SPARSE_SEED, replay, sparse_covariation_data

from copernicus.corrections import Estimate, moment_subtraction, oriv, simex
from copernicus.errors import EstimatorError

pytestmark = pytest.mark.filterwarnings("error")

ESTIMATORS: dict[str, Callable[..., Estimate]] = {
    "moment_subtraction": lambda data, **options: moment_subtraction(
        data, "outcome", score_column="score_a", **options
    ),
    "simex": lambda data, **options: simex(data, "outcome", score_column="score_a", **options),
    "oriv": lambda data, **options: oriv(
        data, "outcome", score_a="score_a", score_b="score_b", **options
    ),
}
LOGGERS = {
    "moment_subtraction": "copernicus.corrections.estimates_classical",
    "simex": "copernicus.corrections.estimates_classical",
    "oriv": "copernicus.corrections.estimates_oriv",
}
DEFINED_OF_60 = 24
# (estimator, defined resamples of two, seed): found by running each estimator on the fixture.
TWO_RESAMPLES = [
    ("moment_subtraction", 0, 2),
    ("moment_subtraction", 1, 1),
    ("moment_subtraction", 2, 0),
    ("simex", 0, 2),
    ("simex", 1, 0),
    ("simex", 2, 4),
    ("oriv", 0, 2),
    ("oriv", 1, 1),
    ("oriv", 2, 0),
]


def warned(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    """The records at WARNING and above that the package logged."""
    return [record for record in caplog.records if record.levelno >= logging.WARNING]


def counts(estimate: Estimate) -> tuple[object, object, object]:
    """The resamples drawn, defined and undefined that the details record."""
    details = estimate.details
    return details["n_boot"], details["n_boot_defined"], details["n_boot_undefined"]


def floor_record(estimate: Estimate) -> tuple[object, object, object]:
    """The floor, the rule below it and the status that the details record."""
    details = estimate.details
    return details["min_defined"], details["below_floor"], details["se_status"]


def assert_interval(estimate: Estimate) -> None:
    """Both ends of the interval lie 1.96 standard errors from the point."""
    assert estimate.ci_low == pytest.approx(estimate.point - 1.96 * estimate.se, rel=1e-12)
    assert estimate.ci_high == pytest.approx(estimate.point + 1.96 * estimate.se, rel=1e-12)


def kept_message(name: str, defined: int, n_boot: int, floor: int) -> str:
    """The warning of an estimate that keeps its point below the floor."""
    return (
        f"{name}: only {defined} of {n_boot} bootstrap resamples have a defined estimate, fewer "
        f"than the {floor} that a standard error needs (min_defined), so the estimate keeps its "
        "point and has no standard error and no interval (below_floor='point')"
    )


@pytest.mark.parametrize("rule", ["refuse", "point"])
@pytest.mark.parametrize("name", sorted(ESTIMATORS))
def test_a_floor_equal_to_the_defined_count_takes_exactly_those_resamples(
    name: str, rule: str, caplog: pytest.LogCaptureFixture
) -> None:
    data = sparse_covariation_data(0)
    point, values = replay(name, data, seed=SPARSE_SEED, n_boot=SPARSE_BOOT)
    assert len(values) == DEFINED_OF_60
    with caplog.at_level(logging.WARNING, logger="copernicus"):
        estimate = ESTIMATORS[name](
            data, n_boot=SPARSE_BOOT, seed=SPARSE_SEED, min_defined=DEFINED_OF_60, below_floor=rule
        )
    assert not warned(caplog)
    assert estimate.point == pytest.approx(point, rel=1e-12)
    assert estimate.se == pytest.approx(float(np.std(values, ddof=1)), rel=1e-12)
    assert_interval(estimate)
    assert counts(estimate) == (SPARSE_BOOT, DEFINED_OF_60, SPARSE_BOOT - DEFINED_OF_60)
    assert floor_record(estimate) == (DEFINED_OF_60, rule, "ok")


@pytest.mark.parametrize("name", sorted(ESTIMATORS))
def test_a_floor_one_above_the_defined_count_keeps_the_point_or_refuses(
    name: str, caplog: pytest.LogCaptureFixture
) -> None:
    data = sparse_covariation_data(0)
    point, values = replay(name, data, seed=SPARSE_SEED, n_boot=SPARSE_BOOT)
    assert len(values) == DEFINED_OF_60
    floor = DEFINED_OF_60 + 1
    with caplog.at_level(logging.WARNING, logger="copernicus"):
        kept = ESTIMATORS[name](
            data, n_boot=SPARSE_BOOT, seed=SPARSE_SEED, min_defined=floor, below_floor="point"
        )
    [warning] = warned(caplog)
    assert (warning.levelno, warning.name) == (logging.WARNING, LOGGERS[name])
    assert warning.getMessage() == kept_message(name, DEFINED_OF_60, SPARSE_BOOT, floor)
    assert kept.point == pytest.approx(point, rel=1e-12)
    assert np.isnan(kept.se) and np.isnan(kept.ci_low) and np.isnan(kept.ci_high)
    assert counts(kept) == (SPARSE_BOOT, DEFINED_OF_60, SPARSE_BOOT - DEFINED_OF_60)
    assert floor_record(kept) == (floor, "point", "too_few_defined")
    with pytest.raises(EstimatorError) as refusal:
        ESTIMATORS[name](data, n_boot=SPARSE_BOOT, seed=SPARSE_SEED, min_defined=floor)
    assert str(refusal.value) == (
        f"only {DEFINED_OF_60} of {SPARSE_BOOT} bootstrap resamples have a defined estimate; the "
        f"standard error needs at least {floor}"
    )


@pytest.mark.parametrize(
    ("name", "defined", "seed"), [case for case in TWO_RESAMPLES if case[1] < 2]
)
def test_fewer_than_two_of_two_defined_resamples_keep_the_point_without_an_interval(
    name: str, defined: int, seed: int, caplog: pytest.LogCaptureFixture
) -> None:
    data = sparse_covariation_data(0)
    point, values = replay(name, data, seed=seed, n_boot=2)
    assert len(values) == defined
    with caplog.at_level(logging.WARNING, logger="copernicus"):
        kept = ESTIMATORS[name](data, n_boot=2, min_defined=2, below_floor="point", seed=seed)
    [warning] = warned(caplog)
    assert (warning.levelno, warning.name) == (logging.WARNING, LOGGERS[name])
    assert warning.getMessage() == kept_message(name, defined, 2, 2)
    assert kept.point == pytest.approx(point, rel=1e-12)
    assert np.isnan(kept.se) and np.isnan(kept.ci_low) and np.isnan(kept.ci_high)
    assert counts(kept) == (2, defined, 2 - defined)
    assert floor_record(kept) == (2, "point", "too_few_defined")


@pytest.mark.parametrize(
    ("name", "defined", "seed"), [case for case in TWO_RESAMPLES if case[1] == 2]
)
def test_two_of_two_defined_resamples_give_their_deviation_with_divisor_n_minus_1(
    name: str, defined: int, seed: int, caplog: pytest.LogCaptureFixture
) -> None:
    data = sparse_covariation_data(0)
    point, values = replay(name, data, seed=seed, n_boot=2)
    assert len(values) == defined
    with caplog.at_level(logging.WARNING, logger="copernicus"):
        estimate = ESTIMATORS[name](data, n_boot=2, min_defined=2, below_floor="point", seed=seed)
    assert not warned(caplog)
    assert estimate.point == pytest.approx(point, rel=1e-12)
    assert estimate.se == pytest.approx(abs(values[0] - values[1]) / np.sqrt(2.0), rel=1e-12)
    assert_interval(estimate)
    assert counts(estimate) == (2, 2, 0)
    assert floor_record(estimate) == (2, "point", "ok")
