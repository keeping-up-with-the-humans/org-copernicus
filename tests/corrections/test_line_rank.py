"""A calibration line is refused when its design loses rank in numbers although the score varies.

``numpy.linalg.lstsq`` with ``rcond=None`` judges the two columns ``[1, score]`` as they are. It
finds a numerical rank of 1 when the score's values are very large beside 1 (a score of
standard deviation 1e16 around 0) or vary very little beside their mean (a mean of 1e8 with a
standard deviation of 1), and then returns the coefficients of smallest norm, which need not be
the least-squares line. The fit refuses such a design. A design of rank 2 is fitted as before;
the rank does not certify that the line is accurate.
"""

from __future__ import annotations

import numpy as np
import pytest
from corrections_support import make_correction_data

from copernicus.corrections import CorrectionData, get_correction, pseudo_values
from copernicus.corrections.families.calibration import fit_ols_line
from copernicus.errors import EstimatorError

pytestmark = pytest.mark.filterwarnings("error")

REFUSAL = "has the numerical rank 1 of 2 on the training rows although the score varies"


def _line_data(offset: float) -> tuple[np.ndarray, np.ndarray]:
    """Scores of standard deviation about 1 around an offset, and gold on a line of slope 2."""
    rng = np.random.default_rng(0)
    varying = rng.normal(size=200)
    return offset + varying, 2.0 * varying + 1.0 + 0.5 * rng.normal(size=200)


def test_a_score_that_varies_too_little_beside_its_size_has_no_line() -> None:
    score, gold = _line_data(1e8)
    with pytest.raises(EstimatorError, match=REFUSAL) as refused:
        fit_ols_line(score, gold, "huge")
    message = str(refused.value)
    assert message.startswith("score column 'huge': the design of intercept and score ")
    assert "centre the score and rescale it to a spread near 1" in message
    with pytest.raises(EstimatorError, match=REFUSAL):
        fit_ols_line(1e16 * np.array([1.0, 2.0, 3.0, 4.0]), np.array([0.0, 1.0, 2.0, 3.0]), "s")


def test_a_score_of_very_large_values_around_zero_is_refused_as_well() -> None:
    # The mean is 0 and the spread is large: the rank is lost to the size of the values beside
    # the column of ones. For a gold with an intercept of 1 the coefficients of smallest norm
    # miss every gold value by 1; the refusal names both causes.
    score = 1e16 * np.array([-2.0, -1.0, 1.0, 2.0])
    gold = np.array([-3.0, -1.0, 3.0, 5.0])
    smallest_norm = np.linalg.lstsq(np.column_stack([np.ones(4), score]), gold, rcond=None)[0]
    missed = smallest_norm[0] + smallest_norm[1] * score - gold
    assert np.allclose(missed, -1.0, atol=1e-9)
    with pytest.raises(EstimatorError, match=REFUSAL) as refused:
        fit_ols_line(score, gold, "s")
    message = str(refused.value)
    assert "very large beside 1" in message and "vary very little beside their mean" in message
    assert "need not be the least-squares line" in message


@pytest.mark.parametrize("offset", [0.0, 1e2, 1e6])
def test_a_score_whose_design_keeps_its_rank_gets_the_least_squares_line(offset: float) -> None:
    score, gold = _line_data(offset)
    line = fit_ols_line(score, gold, "s")
    centred = score - score.mean()
    slope = float(centred @ (gold - gold.mean()) / (centred @ centred))
    assert line.slope == pytest.approx(slope, rel=1e-9)
    assert line.intercept == pytest.approx(float(gold.mean() - slope * score.mean()), rel=1e-9)


def test_the_refusal_reaches_the_calibration_and_the_calibrated_pseudo_values() -> None:
    data = make_correction_data()
    frame = data.frame.assign(
        score_good=1e9 + data.frame["score_good"],
        score_poor=1e9 + data.frame["score_poor"],
        pi=2.0 / 3.0,
    )
    huge = CorrectionData(frame=frame, score_columns=data.score_columns)
    with pytest.raises(EstimatorError, match=REFUSAL):
        get_correction("regression_calibration", "linear").fit(
            huge, huge.labelled_ids, rng=np.random.default_rng(0)
        )
    with pytest.raises(EstimatorError, match=REFUSAL):
        pseudo_values(huge, rectifier="calibrated")
