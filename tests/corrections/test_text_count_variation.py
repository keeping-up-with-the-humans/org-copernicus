"""The variation of the number of texts over whole-group draws, and the bias that it bounds."""

from __future__ import annotations

import itertools
import logging
import math
from collections.abc import Sequence
from fractions import Fraction

import numpy as np
import pytest
from text_count_support import AT_THRESHOLD, JUST_ABOVE, WORKED, group_ids, written_out

import copernicus.corrections
from copernicus.corrections import (
    TEXT_COUNT_VARIATION_THRESHOLD,
    group_sizes,
    groups_for_quota,
    text_count_variation,
)
from copernicus.errors import ConfigurationError, SettingsError

# Three corpora of at most twelve groups of unequal size, the first with a dominant group.
CORPORA = {
    "dominant": (9, 1, 1, 1, 1, 1, 1, 1, 1, 1),
    "spread": (1, 2, 3, 4, 5, 6, 7, 8),
    "two_large": (6, 6, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
}


def made_up_values(n_texts: int) -> list[Fraction]:
    """A value for every text that varies within and between groups: ``(37 i mod 23 - 11) / 3``."""
    return [Fraction((37 * index) % 23 - 11, 3) for index in range(n_texts)]


def exact_square(sizes: Sequence[int], m: int) -> Fraction:
    """``c^2`` from its definition in exact arithmetic: ``sd_n^2 / mean_n^2 * (1 - m / G) / m``."""
    n_groups = len(sizes)
    mean = Fraction(sum(sizes), n_groups)
    variance = sum(((Fraction(size) - mean) ** 2 for size in sizes), Fraction(0)) / (n_groups - 1)
    return variance / mean**2 * (1 - Fraction(m, n_groups)) / m


def over_every_draw(
    sizes: Sequence[int], values: Sequence[Fraction], m: int
) -> tuple[Fraction, Fraction]:
    """The exact bias and variance of the mean over the drawn texts, over every draw of ``m``.

    Every draw of ``m`` of the groups is equally likely, so the bias is the mean of the draws'
    means minus the mean of all values, and the variance is taken with the divisor that counts
    the draws.
    """
    starts = list(itertools.accumulate(sizes, initial=0))
    totals = [sum(values[starts[g] : starts[g + 1]], Fraction(0)) for g in range(len(sizes))]
    corpus = sum(totals, Fraction(0)) / sum(sizes)
    means = [
        sum((totals[g] for g in drawn), Fraction(0)) / sum(sizes[g] for g in drawn)
        for drawn in itertools.combinations(range(len(sizes)), m)
    ]
    average = sum(means, Fraction(0)) / len(means)
    variance = sum(((mean - average) ** 2 for mean in means), Fraction(0)) / len(means)
    return average - corpus, variance


def test_equal_groups_single_texts_the_full_draw_and_one_group_vary_by_zero() -> None:
    for m in range(1, 7):
        assert text_count_variation(group_ids([4] * 6), m) == 0.0
    assert text_count_variation(np.arange(10), 3) == 0.0
    assert text_count_variation(group_ids([5, 1, 2, 9]), 4) == 0.0
    assert text_count_variation(np.array(["only", "only", "only"]), 1) == 0.0


@pytest.mark.parametrize(
    ("sizes", "quota", "m"),
    [(WORKED, 300, 151), (WORKED, 24, 13), ((5, 1, 2, 9), 5, 2), ((2, 3, 3, 7, 1), 10, 4)],
)
def test_the_variation_is_the_formula_written_out(sizes: Sequence[int], quota: int, m: int) -> None:
    groups = group_ids(sizes)
    assert groups_for_quota(quota, groups, np.arange(groups.size)) == m
    assert text_count_variation(groups, m) == pytest.approx(written_out(sizes, m), rel=1e-12)


def test_the_worked_case_varies_by_about_0_286_and_1_89() -> None:
    groups = group_ids(WORKED)
    assert round(text_count_variation(groups, 151), 3) == 0.286
    assert round(text_count_variation(groups, 13), 2) == 1.89


def test_at_the_threshold_the_variation_is_exactly_a_tenth() -> None:
    # The square of the variation is 1 / 100 in the arithmetic of the counts, and the value is
    # computed from the counts with one division and one square root, both correctly rounded.
    groups = group_ids(AT_THRESHOLD)
    assert text_count_variation(groups, 3) == TEXT_COUNT_VARIATION_THRESHOLD == 0.1
    assert text_count_variation(groups, 2) > TEXT_COUNT_VARIATION_THRESHOLD


@pytest.mark.parametrize("name", sorted(CORPORA))
def test_the_variation_is_that_of_the_number_of_drawn_texts_over_every_draw(name: str) -> None:
    sizes = CORPORA[name]
    groups = group_ids(sizes)
    for m in range(1, len(sizes) + 1):
        counts = np.array(
            [
                sum(sizes[g] for g in drawn)
                for drawn in itertools.combinations(range(len(sizes)), m)
            ],
            dtype=float,
        )
        expected = counts.std() / counts.mean()
        assert text_count_variation(groups, m) == pytest.approx(expected, rel=1e-12, abs=1e-15)


@pytest.mark.parametrize("name", sorted(CORPORA))
def test_the_bias_of_a_mean_over_the_drawn_texts_is_at_most_c_times_its_deviation(
    name: str,
) -> None:
    sizes = CORPORA[name]
    groups = group_ids(sizes)
    values = made_up_values(sum(sizes))
    for m in range(1, len(sizes)):
        bias, variance = over_every_draw(sizes, values, m)
        c = text_count_variation(groups, m)
        assert c > 0.0
        assert abs(float(bias)) <= c * math.sqrt(float(variance)) * (1 + 1e-12), (name, m)
        assert bias**2 <= exact_square(sizes, m) * variance, (name, m)


def test_the_bound_is_reached_when_the_quantity_lies_in_the_dominant_group() -> None:
    # The quantity is 1 on the texts of the dominant group and 0 elsewhere. A draw holds the
    # dominant group or not, so the mean over its texts is a line in the number of its texts,
    # and the Cauchy-Schwarz inequality holds with equality.
    sizes = CORPORA["dominant"]
    values = [Fraction(1)] * sizes[0] + [Fraction(0)] * (sum(sizes) - sizes[0])
    for m in (2, 3, 5, 9):
        bias, variance = over_every_draw(sizes, values, m)
        c = text_count_variation(group_ids(sizes), m)
        assert abs(float(bias)) > 0.5 * c * math.sqrt(float(variance)), m
        assert abs(float(bias)) == pytest.approx(c * math.sqrt(float(variance)), rel=1e-12), m
        assert bias**2 == exact_square(sizes, m) * variance, m


def test_group_ids_are_compared_as_strings_as_the_draw_compares_them() -> None:
    mixed = np.array([1, "1", 2, 2, 2], dtype=object)
    assert text_count_variation(mixed, 1) == text_count_variation(group_ids([2, 3]), 1)


def test_the_variation_refuses_a_number_of_groups_outside_one_to_the_groups() -> None:
    groups = group_ids((2, 3))
    for m in (0, -1, 3):
        with pytest.raises(
            SettingsError, match=r"^m must be an integer between 1 and the 2 groups"
        ):
            text_count_variation(groups, m)
    for wrong in (1.5, True, "2"):
        with pytest.raises(SettingsError, match=r"^m must be an integer"):
            text_count_variation(groups, wrong)  # type: ignore[arg-type]
    assert text_count_variation(groups, np.int64(1)) == pytest.approx(written_out((2, 3), 1))


def test_the_variation_refuses_group_ids_that_hold_no_text_or_more_than_one_dimension() -> None:
    with pytest.raises(ConfigurationError, match="one-dimensional"):
        text_count_variation(np.array([["a", "b"], ["a", "c"]]), 1)
    with pytest.raises(ConfigurationError, match="holds no text"):
        text_count_variation(np.array([], dtype=str), 1)


def test_the_threshold_is_the_exact_tenth_that_the_constant_names() -> None:
    assert group_sizes._THRESHOLD == Fraction(1, 10)
    assert float(group_sizes._THRESHOLD) == TEXT_COUNT_VARIATION_THRESHOLD


def test_the_threshold_is_decided_in_exact_arithmetic_on_the_counts() -> None:
    # The counts of two billion texts lie above the threshold by 3 parts in 5 * 10^18, which the
    # rounded value of c does not show; the counts of AT_THRESHOLD lie on it.
    assert exact_square(JUST_ABOVE, 3) - Fraction(1, 100) == Fraction(3, 5 * 10**18)
    assert group_sizes._variation(list(JUST_ABOVE), 3) == TEXT_COUNT_VARIATION_THRESHOLD
    assert group_sizes._exceeds_threshold(list(JUST_ABOVE), 3)
    assert exact_square(AT_THRESHOLD, 3) == Fraction(1, 100)
    assert not group_sizes._exceeds_threshold(list(AT_THRESHOLD), 3)
    assert group_sizes._exceeds_threshold(list(AT_THRESHOLD), 2)
    assert not group_sizes._exceeds_threshold([4] * 6, 2)
    assert not group_sizes._exceeds_threshold(list(JUST_ABOVE), 6)


def test_a_variation_just_above_the_threshold_on_the_counts_is_called_so(
    caplog: pytest.LogCaptureFixture,
) -> None:
    log = logging.getLogger("text_count_threshold_test")
    with caplog.at_level(logging.WARNING, logger=log.name):
        group_sizes._warn_for_counts(list(JUST_ABOVE), 3, log=log, caller="draw", size=None)
    (record,) = caplog.records
    message = record.getMessage()
    shown = "is just above 0.1, and the largest group holds 25 percent of the 2000000000 texts"
    assert shown in message and "is 0.1, above 0.1" not in message


def test_the_threshold_is_a_tenth_and_the_package_exports_both_names() -> None:
    assert TEXT_COUNT_VARIATION_THRESHOLD == 0.1
    exported = set(copernicus.corrections.__all__)
    assert {"TEXT_COUNT_VARIATION_THRESHOLD", "text_count_variation"} <= exported
    assert copernicus.corrections.text_count_variation is group_sizes.text_count_variation
    assert group_sizes.TEXT_COUNT_VARIATION_THRESHOLD == TEXT_COUNT_VARIATION_THRESHOLD
