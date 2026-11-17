"""The reverse calibration refuses a score that covaries with gold only up to rounding."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from copernicus.corrections import CorrectionData, Estimate, moment_subtraction, oriv, simex
from copernicus.corrections._proxy import reverse_calibrate, sample_covariance
from copernicus.errors import EstimatorError

pytestmark = pytest.mark.filterwarnings("error")

FLAT = "does not covary"
# Multiples of the score and of the gold values: the rule must not depend on their units.
SCALES = ((1.0, 1.0), (1e3, 1.0), (1e6, 1.0), (1e-3, 1.0), (1.0, 1e3), (1.0, 1e-3))


def _rounding_rows() -> tuple[np.ndarray, np.ndarray]:
    """Five labelled texts whose gold values are symmetric around the middle score.

    Their covariance is zero in exact arithmetic; the first evenly spaced scores whose
    floating-point covariance is not exactly zero leave one of the size of the rounding.
    """
    gold = np.array([0.7, 0.3, 0.1, 0.3, 0.7])
    for step in np.arange(0.01, 0.5, 0.01):
        scores = 0.1 + step * np.arange(5)
        if sample_covariance(scores, gold) != 0.0:
            return scores, gold
    raise AssertionError("no spacing leaves a covariance of the size of the rounding")


def _residual_of_gold(rng: np.random.Generator, gold: np.ndarray) -> np.ndarray:
    """Normal values with the least-squares line on ``1`` and ``gold`` removed."""
    design = np.column_stack([np.ones(gold.size), gold])
    values = rng.normal(size=gold.size)
    return values - design @ np.linalg.lstsq(design, values, rcond=None)[0]


def _tilted_rows(rise: float, *, n: int = 20, seed: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """A score whose line on gold rises over the range of gold by ``rise`` times its own range."""
    rng = np.random.default_rng(seed)
    gold = rng.normal(size=n)
    other = _residual_of_gold(rng, gold)
    tilt = rise * np.ptp(other) / np.ptp(gold)
    return other + tilt * (gold - gold.mean()), gold


def _correlated_rows(correlation: float, *, n: int = 40, seed: int = 4) -> tuple[np.ndarray, ...]:
    """A score on the unit interval whose sample correlation with gold is ``correlation``."""
    rng = np.random.default_rng(seed)
    gold = rng.normal(size=n)
    other = _residual_of_gold(rng, gold)
    standard_gold = (gold - gold.mean()) / gold.std()
    score = correlation * standard_gold + np.sqrt(1.0 - correlation**2) * other / other.std()
    return 0.5 + 0.2 * score, gold


def _data(score_labelled: np.ndarray, gold_labelled: np.ndarray) -> CorrectionData:
    """The labelled rows first, then 200 unlabelled texts; ``second`` tracks the gold values."""
    rng = np.random.default_rng(0)
    n_labelled = score_labelled.size
    n = n_labelled + 200
    ids = [f"t{index:04d}" for index in range(n)]
    second = rng.normal(size=n)
    second[:n_labelled] = gold_labelled + 0.1 * rng.normal(size=n_labelled)
    frame = pd.DataFrame(
        {
            "text_id": ids,
            "group_id": ids,
            "score": np.concatenate([score_labelled, rng.uniform(size=n - n_labelled)]),
            "second": second,
            "gold": np.concatenate([gold_labelled, np.full(n - n_labelled, np.nan)]),
            "outcome": rng.normal(size=n),
            "log_chars": rng.normal(size=n),
        }
    )
    return CorrectionData(frame=frame, score_columns=("score", "second"))


ESTIMATORS: dict[str, Callable[[CorrectionData], Estimate]] = {
    "moment_subtraction": lambda data: moment_subtraction(data, "outcome", score_column="score"),
    "simex": lambda data: simex(data, "outcome", score_column="score"),
    "oriv_first": lambda data: oriv(data, "outcome", score_a="score", score_b="second"),
    "oriv_second": lambda data: oriv(data, "outcome", score_a="second", score_b="score"),
}


@pytest.mark.parametrize("estimator", sorted(ESTIMATORS))
def test_a_covariance_of_rounding_size_is_refused_by_every_classical_estimator(
    estimator: str,
) -> None:
    score, gold = _rounding_rows()
    covariance = sample_covariance(score, gold)
    assert covariance != 0.0 and abs(covariance) < 1e-17
    with pytest.raises(EstimatorError, match=FLAT):
        ESTIMATORS[estimator](_data(score, gold))


def test_a_covariance_of_rounding_size_is_refused_in_any_units() -> None:
    score, gold = _rounding_rows()
    labelled = np.ones(score.size, dtype=bool)
    for score_factor, gold_factor in SCALES:
        with pytest.raises(EstimatorError, match=FLAT):
            reverse_calibrate(score * score_factor, labelled, gold * gold_factor)


def test_the_line_lies_at_a_rise_of_1e_12_of_the_score_range_in_any_units() -> None:
    """A line whose rise over the gold range is 0.5e-12 of the score's range is flat; 2e-12 is not.

    The rise of the line is its slope times the range of the gold values; the score's own range
    is its unit, so multiplying the score or the gold values by any factor leaves the decision.
    """
    for rise, refused in ((0.5e-12, True), (2e-12, False)):
        score, gold = _tilted_rows(rise)
        labelled = np.ones(score.size, dtype=bool)
        for score_factor, gold_factor in SCALES:
            scaled_score, scaled_gold = score * score_factor, gold * gold_factor
            if refused:
                with pytest.raises(EstimatorError, match=FLAT):
                    reverse_calibrate(scaled_score, labelled, scaled_gold)
                continue
            line = reverse_calibrate(scaled_score, labelled, scaled_gold)
            measured = abs(line.slope) * np.ptp(scaled_gold) / np.ptp(scaled_score)
            assert measured == pytest.approx(rise, rel=1e-3), (score_factor, gold_factor)


def test_a_small_real_covariance_is_accepted_in_any_units() -> None:
    score, gold = _correlated_rows(0.05)
    assert np.corrcoef(score, gold)[0, 1] == pytest.approx(0.05, rel=1e-9)
    labelled = np.ones(score.size, dtype=bool)
    for score_factor, gold_factor in SCALES:
        scaled_score, scaled_gold = score * score_factor, gold * gold_factor
        line = reverse_calibrate(scaled_score, labelled, scaled_gold)
        expected = np.cov(scaled_score, scaled_gold)[0, 1] / np.var(scaled_gold, ddof=1)
        assert line.slope == pytest.approx(expected, rel=1e-12)
        assert np.isfinite(line.proxy).all()
