"""The draw, the repeated study and the battery's diagnostic warn when the drawn texts vary much.

The expected values come from the definition of the variation written out with numpy
(``written_out``), not from the library.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest
from estimates_support import clustered_frame
from text_count_support import (
    AT_THRESHOLD,
    BELOW_THRESHOLD,
    DOMINANT_BATTERY,
    DOMINANT_SMALL,
    EVEN_BATTERY,
    SHOWN_AS_THRESHOLD,
    WORKED,
    battery_data,
    group_ids,
    table_of,
    written_out,
)

from copernicus.corrections import (
    BatterySettings,
    CorrectionData,
    draw_human_sample,
    repeated_study,
    run_battery,
)

SAMPLING = "copernicus.corrections.sampling"
STUDY = "copernicus.corrections.study"
BATTERY = "copernicus.corrections.battery"
BATTERY_SETTINGS = {
    "families": [("regression_calibration", "linear")],
    "sizes": (8, 12, 50, 0.1),
    "schemes": ("kfold2", "holdout"),
    "seeds": (1, 2),
}
# Every warning of the variation names it with these words.
VARIATION = "coefficient of variation"
# The sentences that every warning of the variation holds, and the advice of each caller.
BIAS = "A mean over the drawn texts can then be biased."
BOUND = (
    "For the point of mean_ppi without power tuning and with a prediction fixed before the "
    "draw, such as a score, that bias is at most this coefficient times the point's standard "
    "deviation, both taken over all possible draws of "
)
BOUND_CONDITIONS = (
    "provided mean_ppi refuses none of them (it refuses a draw with fewer than two labelled "
    "texts and, when it clusters, one whose labelled texts lie in one group)."
)
NOT_BOUNDED = (
    "The coefficient does not bound the bias of a difference (group_difference_ppi), whose two "
    "sides each vary on their own, nor that of a power-tuned estimate or of one whose prediction "
    "is fitted on the drawn texts, such as a cross-fitted one, and the intervals of all these "
    "estimates can cover less often than 95 percent."
)
DRAW_ADVICE = (
    "Check the design with repeated_study, or estimate the mean with "
    "pseudo_values(data).mean_rectified, which, with its default identity rectifier, the "
    "recorded pi and every text of the drawn groups coded, is exactly unbiased over draws"
)
STUDY_ADVICE = (
    "Read the bias and the coverage of the rows of this size and the reasons of their undefined "
    "draws: a row's bias is taken over its defined draws of this run only, which the bound does "
    "not cover, and the bound says nothing about coverage"
)
DIAGNOSTIC_ADVICE = (
    "Of the two means that the diagnostic records for this size, mean_rectified, the mean of the "
    "identity pseudo-values over all texts, is exactly unbiased over draws, and "
    "mean_rectified_tuned, a mean over the drawn texts with a weight tuned on them, is a ratio "
    "whose bias the coefficient does not bound"
)


def variation_warnings(caplog: pytest.LogCaptureFixture, module: str) -> list[str]:
    """The messages of the warnings of the variation that the given module logged."""
    return [
        record.getMessage()
        for record in caplog.records
        if record.name == module
        and record.levelno == logging.WARNING
        and VARIATION in record.getMessage()
    ]


def test_a_draw_above_the_threshold_logs_one_warning_that_names_the_four_numbers(
    caplog: pytest.LogCaptureFixture,
) -> None:
    table = table_of(WORKED)
    with caplog.at_level(logging.WARNING, logger=SAMPLING):
        draw = draw_human_sample(table, n=24, seed=1)
    assert (draw.m, draw.n_groups) == (13, 201)
    assert len(caplog.records) == 1
    (message,) = variation_warnings(caplog, SAMPLING)
    assert f"{written_out(WORKED, 13):.3g}" == "1.89"
    assert "the number of texts in a draw of 13 of the 201 groups" in message
    assert "is 1.89, above 0.1" in message
    assert "the largest group holds 50 percent of the 400 texts" in message
    for sentence in (BIAS, BOUND, BOUND_CONDITIONS, NOT_BOUNDED, DRAW_ADVICE):
        assert sentence in message
    assert "of 13 of the 201 groups, provided mean_ppi refuses none of them" in message
    assert "ratios whose bias can reach" not in message


def test_a_quota_above_half_the_units_keeps_its_own_warning_beside_that_of_the_variation(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger=SAMPLING):
        assert draw_human_sample(table_of(WORKED), n=300, seed=1).m == 151
    assert len(caplog.records) == 2
    assert "exceeds half" in caplog.records[0].getMessage()
    (message,) = variation_warnings(caplog, SAMPLING)
    assert f"{written_out(WORKED, 151):.3g}" == "0.286"
    assert "in a draw of 151 of the 201 groups" in message and "is 0.286, above 0.1" in message


def test_a_draw_at_or_below_the_threshold_logs_no_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    at, below = table_of(AT_THRESHOLD), table_of(BELOW_THRESHOLD)
    assert 0.0 < written_out(BELOW_THRESHOLD, 9) < 0.04
    with caplog.at_level(logging.WARNING, logger=SAMPLING):
        assert draw_human_sample(at, n=10, seed=4).m == 3
        assert draw_human_sample(at, n=7, seed=5).m == 3
        assert draw_human_sample(below, n=30, seed=2).m == 9
    assert not caplog.records
    with caplog.at_level(logging.WARNING, logger=SAMPLING):
        assert draw_human_sample(at, n=6, seed=4).m == 2
    assert len(caplog.records) == 1
    (message,) = variation_warnings(caplog, SAMPLING)
    assert "in a draw of 2 of the 6 groups varies with the groups drawn" in message
    assert f"is {written_out(AT_THRESHOLD, 2):.3g}, above 0.1" in message


def test_the_full_draw_of_an_uneven_corpus_logs_no_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    table = table_of(WORKED)
    with caplog.at_level(logging.WARNING, logger=SAMPLING):
        draws = (draw_human_sample(table, n=400), draw_human_sample(table, share=1.0))
    for draw in draws:
        assert (draw.m, draw.pi, draw.ids.size) == (201, 1.0, 400)
    assert not caplog.records


def test_with_a_unit_column_the_variation_counts_the_texts(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Eleven groups, each one unit: counted by units the groups are equal, counted by texts the
    # first holds ten texts and every other one.
    sizes = (10, *([1] * 10))
    table = table_of(sizes, units=list(group_ids(sizes)))
    with caplog.at_level(logging.WARNING, logger=SAMPLING):
        draw = draw_human_sample(table, n=3, unit_id="unit", seed=0)
    assert (draw.m, draw.n_units, draw.n_groups) == (3, 11, 11)
    assert len(caplog.records) == 1
    (message,) = variation_warnings(caplog, SAMPLING)
    assert f"is {written_out(sizes, 3):.3g}, above 0.1" in message
    assert "the largest group holds 50 percent of the 20 texts" in message


@pytest.mark.parametrize("draws", [1, 7])
def test_the_study_logs_one_warning_for_each_size_above_the_threshold(
    draws: int, caplog: pytest.LogCaptureFixture
) -> None:
    # The draws of 4 texts take 3 of the 21 groups and those of 38 texts 20 of them, both above
    # the threshold; those of 40 texts take every group.
    frame = clustered_frame(np.random.default_rng(3), DOMINANT_SMALL)
    data = CorrectionData(frame=frame, score_columns=("score",))
    with caplog.at_level(logging.WARNING, logger=STUDY):
        table = repeated_study(
            data, truth_gold="gold", sizes=(4, 38, 40), draws=draws, estimators=("raw",)
        )
    assert (table["n_draws"] == draws).all() and (table["status"] == "ok").all()
    assert len(caplog.records) == 2
    small, large = variation_warnings(caplog, STUDY)
    assert small.startswith("repeated study, size 4: the number of texts in a draw of 3 of the 21")
    assert f"is {written_out(DOMINANT_SMALL, 3):.3g}, above 0.1" in small
    assert large.startswith("repeated study, size 38: the number of texts in a draw of 20 of the")
    assert f"is {written_out(DOMINANT_SMALL, 20):.3g}, above 0.1" in large
    for message in (small, large):
        assert "the largest group holds 50 percent of the 40 texts" in message
        for sentence in (BIAS, BOUND, BOUND_CONDITIONS, NOT_BOUNDED, STUDY_ADVICE):
            assert sentence in message
        assert "Check the design with repeated_study" not in message


def test_a_variation_shown_as_the_threshold_is_called_just_above_it(
    caplog: pytest.LogCaptureFixture,
) -> None:
    value = written_out(SHOWN_AS_THRESHOLD, 4)
    assert 0.1 < value < 0.1005 and f"{value:.3g}" == "0.1"
    with caplog.at_level(logging.WARNING, logger=SAMPLING):
        assert draw_human_sample(table_of(SHOWN_AS_THRESHOLD), n=8, seed=3).m == 4
    (message,) = variation_warnings(caplog, SAMPLING)
    assert "its standard deviation divided by its mean, is just above 0.1, and the" in message
    assert "is 0.1, above 0.1" not in message


def test_the_battery_warns_once_for_each_diagnostic_size_whose_draws_vary_much(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Two seeds draw the diagnostic of 8 and of 12 texts twice each, whatever the two schemes; the
    # size 50 exceeds half of the 80 texts and the share 0.1 has no diagnostic, so neither is drawn.
    settings = BatterySettings(**BATTERY_SETTINGS)  # type: ignore[arg-type]
    with caplog.at_level(logging.WARNING):
        result = run_battery(battery_data(DOMINANT_BATTERY, 11), settings)
    assert int(result.results["diagnostic"].sum()) == 4
    warned = [record.getMessage() for record in caplog.records if VARIATION in record.getMessage()]
    assert warned == variation_warnings(caplog, BATTERY) and len(warned) == 2
    small, large = warned
    assert small.startswith("battery diagnostic, size 8: the number of texts in a draw of 3 of the")
    assert f"is {written_out(DOMINANT_BATTERY, 3):.3g}, above 0.1" in small
    assert large.startswith("battery diagnostic, size 12: the number of texts in a draw of 4 of")
    assert f"is {written_out(DOMINANT_BATTERY, 4):.3g}, above 0.1" in large
    for message in warned:
        for sentence in (BIAS, BOUND, BOUND_CONDITIONS, NOT_BOUNDED, DIAGNOSTIC_ADVICE):
            assert sentence in message


def test_the_battery_does_not_warn_for_an_even_corpus_or_without_its_diagnostic(
    caplog: pytest.LogCaptureFixture,
) -> None:
    dominant = battery_data(DOMINANT_BATTERY, 11)
    missing = np.where(np.arange(dominant.n) % 9 == 0, np.nan, dominant.gold)
    partial = CorrectionData(frame=dominant.frame.assign(gold=missing), score_columns=("score",))
    settings = BatterySettings(**BATTERY_SETTINGS)  # type: ignore[arg-type]
    with caplog.at_level(logging.WARNING):
        even = run_battery(battery_data(EVEN_BATTERY, 11), settings)
        without = run_battery(partial, settings)
    assert int(even.results["diagnostic"].sum()) == 4 and not without.results["diagnostic"].any()
    assert not [record for record in caplog.records if VARIATION in record.getMessage()]
