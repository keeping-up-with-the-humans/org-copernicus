"""What the bound of ``text_count_variation`` covers and what it does not, in exact fractions.

The bound holds for a mean over the drawn texts of a quantity fixed before the draw, with its
bias and its standard deviation taken over every possible draw of ``m`` of the ``G`` groups.
The first case shows that it does not hold for a summary over the accepted draws only; the
second that the variation of all texts does not bound a difference, while the bound with each
side's number of texts per group does; the third that the sides' bound is on each side's mean of
the gap between the human value and the prediction, not on its mean of the human values.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest
from text_count_support import group_ids

from copernicus.corrections import (
    CorrectionData,
    group_difference_ppi,
    groups_for_quota,
    mean_ppi,
    text_count_variation,
)
from copernicus.errors import EstimatorError


def exact_square(sizes: Sequence[int], m: int) -> Fraction:
    """``c^2`` from its definition in exact arithmetic, over all ``G`` groups given."""
    n_groups = len(sizes)
    mean = Fraction(sum(sizes), n_groups)
    variance = sum(((Fraction(size) - mean) ** 2 for size in sizes), Fraction(0)) / (n_groups - 1)
    return variance / mean**2 * (1 - Fraction(m, n_groups)) / m


def moments(values: Sequence[Fraction]) -> tuple[Fraction, Fraction]:
    """The mean and the variance, with the divisor that counts the values, of equally likely
    values."""
    mean = sum(values, Fraction(0)) / len(values)
    return mean, sum(((value - mean) ** 2 for value in values), Fraction(0)) / len(values)


def coded(
    groups: np.ndarray,
    drawn: Sequence[str],
    gold: np.ndarray,
    pi: float,
    score: np.ndarray | None = None,
) -> CorrectionData:
    """The data of one whole-group draw: gold on the texts of the drawn groups; score 0 if none."""
    labelled = np.isin(groups, list(drawn))
    frame = pd.DataFrame(
        {
            "text_id": [f"t{index:03d}" for index in range(groups.size)],
            "group_id": groups,
            "score": np.zeros(groups.size) if score is None else score,
            "gold": np.where(labelled, gold, np.nan),
            "log_chars": np.arange(groups.size, dtype=float),
            "pi": pi,
        }
    )
    return CorrectionData(frame=frame, score_columns=("score",))


def test_the_bound_does_not_cover_a_summary_over_the_accepted_draws_only() -> None:
    # One group of one text, whose gold value is 1, and eleven groups of three texts whose gold
    # values are 0; the score is 0. A quota of 2 texts draws one group. mean_ppi refuses the draw
    # of the single text and estimates 0 on every other draw.
    sizes = (1, *([3] * 11))
    groups = group_ids(sizes)
    assert groups_for_quota(2, groups, np.arange(groups.size)) == 1
    gold = (groups == "g000").astype(float)
    names = list(dict.fromkeys(groups.tolist()))
    with pytest.raises(EstimatorError, match="needs at least 2 labelled rows; there are 1"):
        mean_ppi(coded(groups, ["g000"], gold, 1 / 12), cluster=False)
    points = [
        mean_ppi(coded(groups, [name], gold, 1 / 12), cluster=False).point for name in names[1:]
    ]
    assert points == [0.0] * 11
    truth = Fraction(1, 34)
    square = exact_square(sizes, 1)
    assert square == Fraction(11, 289) and text_count_variation(groups, 1) > 0.1
    # Over the eleven accepted draws the bias is -1/34 at a standard deviation of 0, so it is not
    # at most c times that standard deviation.
    accepted_mean, accepted_variance = moments([Fraction(0)] * 11)
    assert (accepted_mean - truth, accepted_variance) == (Fraction(-1, 34), Fraction(0))
    # Over every draw, the refused one counted with the mean of its one text, the bound holds,
    # here with equality.
    every_mean, every_variance = moments([Fraction(1), *([Fraction(0)] * 11)])
    assert (every_mean - truth) ** 2 == square * every_variance


def test_the_variation_of_all_texts_does_not_bound_a_difference_and_its_sides_do() -> None:
    # Groups of 8, 4 and 4 texts with 1, 2 and 3 of them on side A and 7, 2 and 1 on side B. The
    # gold value is 1 on the side-A text of the first group and 0 elsewhere, the score 0. Under
    # the default clustering every draw of two of the groups is defined.
    sizes, side_a = (8, 4, 4), (1, 2, 3)
    side_b = tuple(size - count for size, count in zip(sizes, side_a, strict=True))
    groups = group_ids(sizes)
    split = np.concatenate(
        [np.arange(size) < count for size, count in zip(sizes, side_a, strict=True)]
    )
    gold = (split & (groups == "g000")).astype(float)
    draws = list(itertools.combinations(("g000", "g001", "g002"), 2))
    points = [
        group_difference_ppi(coded(groups, drawn, gold, 2 / 3), split).point for drawn in draws
    ]
    on_a = [Fraction(1, 3), Fraction(1, 4), Fraction(0)]
    assert points == pytest.approx([float(value) for value in on_a], abs=1e-15)
    bias, variance = moments(on_a)
    bias -= Fraction(1, 6)
    assert (bias, variance) == (Fraction(1, 36), Fraction(13, 648))
    square = exact_square(sizes, 2)
    assert square == Fraction(1, 32) and text_count_variation(groups, 2) > 0.1
    assert bias**2 / variance == Fraction(1, 26) > square
    # Each side's c counts that side's texts in all three groups. The mean of side B is 0 on
    # every draw, so the bound c_A * sd(R_A) + c_B * sd(R_B) is c_A * sd(R_A) here.
    on_b_mean, on_b_variance = moments([Fraction(0)] * 3)
    assert (on_b_mean, on_b_variance) == (0, 0) and exact_square(side_b, 2) == Fraction(31, 200)
    assert bias**2 <= exact_square(side_a, 2) * variance
    bound = math.sqrt(exact_square(side_a, 2) * variance) + math.sqrt(
        exact_square(side_b, 2) * on_b_variance
    )
    assert abs(float(bias)) <= bound


def test_the_bound_of_a_side_is_on_the_gap_between_the_gold_value_and_the_prediction() -> None:
    # The corpus and the split of the test above, with a gold value of 0 on every text and a score
    # of 1 on the side-A text of the first group. Each side's mean of the gold values is 0 on
    # every draw, so the standard deviations of those means are 0 and bound nothing. Side A's mean
    # of the gap between the gold value and the score is -1/3, -1/4 and 0 on the three draws, and
    # the difference adds the mean score of all side-A texts, 1/6, to it.
    sizes, side_a = (8, 4, 4), (1, 2, 3)
    groups = group_ids(sizes)
    split = np.concatenate(
        [np.arange(size) < count for size, count in zip(sizes, side_a, strict=True)]
    )
    score = (split & (groups == "g000")).astype(float)
    gold = np.zeros(groups.size)
    draws = list(itertools.combinations(("g000", "g001", "g002"), 2))
    points = [
        group_difference_ppi(coded(groups, drawn, gold, 2 / 3, score), split).point
        for drawn in draws
    ]
    gaps = [Fraction(-1, 3), Fraction(-1, 4), Fraction(0)]
    found = [gap + Fraction(1, 6) for gap in gaps]
    assert points == pytest.approx([float(value) for value in found], abs=1e-15)
    bias, _ = moments(found)
    _, gap_variance = moments(gaps)
    assert (bias, gap_variance) == (Fraction(-1, 36), Fraction(13, 648))
    assert bias**2 <= exact_square(side_a, 2) * gap_variance
