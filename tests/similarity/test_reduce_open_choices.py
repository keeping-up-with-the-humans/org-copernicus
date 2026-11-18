"""The pca reduction warns where it treats its sign or its direction as open.

The sign of the first principal component follows its correlation with the row means; the
direction is unique when the largest singular value of the centred standardised matrix stands
alone. The reduction treats the sign as open when the computed standard deviation of the row
means is 0; otherwise, when that correlation is at most ``1e-12`` in absolute value, and when
the range of the row means is at most ``1e-12`` times the larger of 1 and their largest
absolute value. It treats the direction as open when the two largest singular values, which it
computes for this check from the centred matrix in 64-bit floats, differ by at most a relative
``1e-12``. These thresholds are conventions: a value below one can be genuine. The scores stay
as they are computed, and the warnings say which condition was met.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from copernicus.errors import ConfigurationError
from copernicus.similarity.reduce import _pca_solver, reduce_matrix

pytestmark = pytest.mark.filterwarnings("error")

REDUCE = "copernicus.similarity.reduce"


def _warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        r.getMessage() for r in caplog.records if r.name == REDUCE and r.levelno == logging.WARNING
    ]


def _reduce(matrix: np.ndarray, caplog: pytest.LogCaptureFixture) -> tuple[np.ndarray, list[str]]:
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        scores = reduce_matrix(matrix, "pca", "a pole")
    return scores, _warnings(caplog)


def _standardised(matrix: np.ndarray) -> np.ndarray:
    return (matrix - matrix.mean(axis=0)) / matrix.std(axis=0)


def _decomposed(matrix: np.ndarray) -> np.ndarray:
    """The scores of the decomposition before the reduction sets their sign, by its own calls."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    solver = _pca_solver(matrix.shape[0], matrix.shape[1], "a pole")
    standardised = StandardScaler().fit_transform(matrix)
    scores: np.ndarray = PCA(n_components=1, svd_solver=solver).fit_transform(standardised)[:, 0]
    return scores


def test_an_ordinary_matrix_logs_nothing(caplog: pytest.LogCaptureFixture) -> None:
    matrix = np.random.default_rng(3).normal(size=(40, 3))
    _, warnings = _reduce(matrix, caplog)
    assert warnings == []
    for method in ("mean", "max"):
        with caplog.at_level(logging.WARNING):
            reduce_matrix(matrix, method, "a pole")
    assert _warnings(caplog) == []


def test_constant_row_means_leave_the_sign_to_the_decomposition(
    caplog: pytest.LogCaptureFixture,
) -> None:
    column = np.random.default_rng(3).normal(size=40)
    _, warnings = _reduce(np.column_stack([column, -column]), caplog)
    (message,) = warnings
    assert message.startswith(
        "reduction of a pole: the row means of the matrix have a computed standard deviation of 0"
    )
    assert "the sign of the pca scores" in message and "the decomposition gave" in message


def test_scores_that_do_not_correlate_with_the_row_means_get_the_sign_of_rounding(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Two standardised columns that correlate negatively: the component is their difference,
    # which does not correlate with their mean in exact arithmetic. Which sign rounding gives
    # the computed correlation differs between cases and between numerical libraries; the
    # scores follow it: they are what the decomposition returns here, negated when their
    # correlation with the row means is below 0.
    for seed in range(8):
        rng = np.random.default_rng(seed)
        first = rng.normal(size=60)
        matrix = _standardised(np.column_stack([first, -0.6 * first + rng.normal(size=60)]))
        scores, warnings = _reduce(matrix, caplog)
        (message,) = warnings
        assert message.startswith(
            "reduction of a pole: the pca scores correlate with the row means of the matrix at "
        )
        assert "at most 1e-12 in absolute value" in message
        assert "treats as too small to identify their sign" in message
        assert "follows the computed correlation" in message
        raw = _decomposed(matrix)
        computed = float(np.corrcoef(raw, matrix.mean(axis=1))[0, 1])
        assert abs(computed) <= 1e-12
        assert np.array_equal(scores, -raw if computed < 0.0 else raw)


def test_two_equal_largest_singular_values_leave_the_direction_open(
    caplog: pytest.LogCaptureFixture,
) -> None:
    first = np.array([1.0, -1.0, 1.0, -1.0] * 10)
    second = np.array([1.0, 1.0, -1.0, -1.0] * 10)
    _, warnings = _reduce(np.column_stack([first, second]), caplog)
    tied = [message for message in warnings if "two largest singular values" in message]
    (message,) = tied
    assert message.startswith("reduction of a pole: the two largest singular values")
    assert "at most a relative 1e-12" in message and "treats as a tie" in message


@pytest.mark.parametrize("repeats", [1, 5], ids=["full_svd", "covariance"])
def test_the_tie_is_judged_on_the_matrix_that_the_decomposition_centres(
    repeats: int, caplog: pytest.LogCaptureFixture
) -> None:
    # Columns that differ from 0.1 in their last bit, and one constant column: standardised, the
    # matrix has columns whose means are not zero, and only after centring are its two largest
    # singular values equal.
    step = np.spacing(0.1)
    first = np.tile([1.0, -1.0, 1.0, -1.0], 2 * repeats)
    second = np.tile([1.0, 1.0, -1.0, -1.0], 2 * repeats)
    matrix = np.column_stack([0.1 + step * first, 0.1 + step * second, np.full(first.size, 0.1)])
    _, warnings = _reduce(matrix, caplog)
    assert len([message for message in warnings if "treats as a tie" in message]) == 1


def test_row_means_within_the_tolerance_are_treated_as_constant(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Two columns on one line at a scale of 1e-13: the scores correlate at 1 with the row
    # means, whose range lies below 1e-12. The convention flags it although nothing is open.
    column = 2.0**-44 * np.arange(-5.0, 5.0)
    matrix = np.column_stack([column, 2.0 * column])
    scores, warnings = _reduce(matrix, caplog)
    assert np.corrcoef(scores, matrix.mean(axis=1))[0, 1] == pytest.approx(1.0)
    (message,) = warnings
    assert message.startswith("reduction of a pole: the row means of the matrix have a range of ")
    assert "treats as constant" in message and "follows their computed correlation" in message


U = np.array([1.0, -1.0, 1.0, -1.0])
V = np.array([1.0, 1.0, -1.0, -1.0])


def _ties(messages: list[str]) -> list[str]:
    return [message for message in messages if "treats as a tie" in message]


@pytest.mark.parametrize(("repeats", "angle"), [(1, 0.7), (3, 0.3), (5, 0.4), (10, 0.1)])
def test_an_exact_tie_in_32_bit_floats_is_found(
    repeats: int, angle: float, caplog: pytest.LogCaptureFixture
) -> None:
    # Two columns of the pattern (a, b, -b, -a) and (-b, a, -a, b): orthogonal and of one
    # length whatever the rounding of a and b, so the two singular values are equal. In
    # 32-bit floats the sums of their squares can differ in the last bits, about 7e-8 for the
    # last three cases, which is far above the tolerance.
    cosine, sine = np.float32(np.cos(angle)), np.float32(np.sin(angle))
    first, second = np.tile(U, repeats), np.tile(V, repeats)
    matrix = np.column_stack([cosine * first + sine * second, cosine * second - sine * first])
    _, warnings = _reduce(matrix.astype(np.float32), caplog)
    assert len(_ties(warnings)) == 1


def test_a_clear_gap_between_singular_values_whose_squares_underflow_is_no_tie(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Entries that differ from 2e-147 by a few of its last bits: the scaler finds the columns
    # constant and leaves them unscaled, so the centred entries lie near 1e-162 and their
    # squares underflow. The two singular values differ by more than a quarter of the larger.
    offsets = np.array([[-5.0, 10.0], [-11.0, -8.0], [10.0, 8.0], [4.0, -8.0]])
    matrix = 2e-147 + np.spacing(2e-147) * offsets
    singular = np.linalg.svd(matrix - matrix.mean(axis=0), compute_uv=False)
    assert 0.25 < (singular[0] - singular[1]) / singular[0] < 0.3
    _, warnings = _reduce(matrix, caplog)
    assert _ties(warnings) == []


@pytest.mark.parametrize(("share", "tied"), [(2.0**-42, True), (2.0**-38, False)])
def test_unequal_singular_values_are_a_tie_within_the_tolerance_only(
    share: float, tied: bool, caplog: pytest.LogCaptureFixture
) -> None:
    # The second column leans on the first by a share of 2**-42 or 2**-38, which sets the two
    # singular values apart by a relative 2.3e-13 or 3.6e-12: on either side of 1e-12.
    first = np.tile(U, 20)
    matrix = np.column_stack([first, np.tile(V, 20) + share * first])
    _, warnings = _reduce(matrix, caplog)
    assert len(_ties(warnings)) == int(tied)


def test_a_matrix_of_32_bit_floats_is_judged_on_its_values_read_as_64_bit_floats(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # One entry of the second column lies four of its last bits above 1: the singular values
    # of the stored matrix differ by a relative 3e-9, far outside the tolerance, although a
    # decomposition in 32-bit floats, which rounds at about 1e-7, returns them as equal.
    first = np.tile(U, 50).astype(np.float32)
    second = np.tile(V, 50).astype(np.float32)
    second[0] = np.float32(1.0) + 4 * np.spacing(np.float32(1.0))
    matrix = np.column_stack([first, second])
    assert matrix.dtype == np.float32
    _, warnings = _reduce(matrix, caplog)
    assert _ties(warnings) == []


def test_a_failure_of_the_check_of_the_singular_values_stops_the_reduction_by_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The scores come from scikit-learn's own decomposition; the check of the tie is a second
    # one, by numpy. The test puts a failure in its place.
    def no_convergence(*args: object, **kwargs: object) -> None:
        raise np.linalg.LinAlgError("SVD did not converge")

    matrix = np.random.default_rng(3).normal(size=(40, 3))
    monkeypatch.setattr(np.linalg, "svd", no_convergence)
    with pytest.raises(ConfigurationError) as refused:
        reduce_matrix(matrix, "pca", "a pole")
    assert str(refused.value) == (
        "reduction of a pole: the check of the two largest singular values failed, so the pca "
        "scores are not returned: SVD did not converge"
    )
