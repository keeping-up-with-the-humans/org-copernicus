"""Means of finite values that overflow are reported by the functions the validity panel calls.

A score near the largest float is finite, but the sum of two is not. The panel refuses a score,
a gold value or a covariate above the library's limit (1e150) before it averages anything, so
these inputs stop it with that refusal (the first test). The functions it calls after its checks
keep their own handling of an average that overflows, for a caller who gives them such values,
and the other tests call them directly: under ``unit="text"`` `compute_target` drops and counts a
text whose finite run scores have an average that is not finite, with one warning that names it,
and under ``unit="group"`` it refuses such a text, and the group unit refuses three means that are
not finite although their values are: a group's score over its texts, a group's score in one run
for the noise, and a group's covariate over its texts. With warnings turned into errors the
message is what is raised, not numpy's warning of the overflow.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
import pytest
from units_support import GOLD, N_BOOT, SIX_TEXTS, TARGET, TARGETS, Entry, build, computed

from copernicus.errors import ConfigurationError
from copernicus.validity.panel import PanelSettings, validity_panel
from copernicus.validity.panel_units import group_runs, reduce_to_groups

HUGE = 1e308
OVERFLOWING: dict[str, list[Entry]] = {
    "group-mean": [("t1", "G1", [HUGE], 3.0), ("t2", "G1", [HUGE], 3.0), *SIX_TEXTS[2:]],
    "run-mean": [
        ("t1", "G1", [HUGE, -HUGE], 3.0),
        ("t2", "G1", [HUGE, -HUGE], 3.0),
        *SIX_TEXTS[2:],
    ],
    "text-mean": [("t1", "G1", [HUGE, HUGE], 3.0), *SIX_TEXTS[1:]],
    "both-directions": [("t1", "G1", [HUGE] * 150 + [-HUGE] * 150, 3.0), *SIX_TEXTS[1:]],
}


def _refusal(entries: list[Entry]) -> str:
    """The message with which `compute_target` refuses the entries under the group unit."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ConfigurationError) as refused:
            computed(entries, "group")
    return str(refused.value)


@pytest.mark.parametrize("unit", ["text", "group"])
@pytest.mark.parametrize("name", sorted(OVERFLOWING))
def test_the_panel_refuses_these_scores_before_it_averages_them(name: str, unit: str) -> None:
    scores, gold, table = build(OVERFLOWING[name])
    settings = PanelSettings(n_boot=N_BOOT, unit=unit)  # type: ignore[arg-type]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ConfigurationError) as refused:
            validity_panel(scores, gold, table, targets=TARGETS, settings=settings)
    message = str(refused.value)
    assert message.startswith(f"target {TARGET!r}: text(s) 't1'")
    assert "have a score above 1e+150 in absolute value (the largest is 1e+308)" in message


def test_a_group_whose_mean_score_overflows_is_refused_by_name() -> None:
    # t1 and t2 of G1 each score 1e308 in their one run: both run averages are finite, their
    # mean is not.
    message = _refusal(OVERFLOWING["group-mean"])
    assert message.startswith(f"target {TARGET!r}: ")
    assert "the mean score of group(s) 'G1' is not finite although the score of every" in message
    assert "divide every score by one constant" in message


def test_a_group_whose_mean_in_one_run_overflows_is_refused_by_name_and_run() -> None:
    # Both texts of G1 score 1e308 in run 1 and -1e308 in run 2: each text's run average is 0,
    # so is the group's score, and only the group's mean within a run is not finite.
    message = _refusal(OVERFLOWING["run-mean"])
    assert message.startswith(f"target {TARGET!r}: ")
    assert (
        "the mean score of group(s) 'G1' in run 1, 'G1' in run 2 is not finite although the "
        "score of every" in message
    )


def _aligned(covariate: list[float]) -> tuple[pd.DataFrame, pd.Series]:
    ids = ["t1", "t2", "u1"]
    aligned = pd.DataFrame(
        {"score": [1.0, 2.0, 3.0], "gold": [3.0, 3.0, 2.0], "covariate": covariate}, index=ids
    )
    return aligned, pd.Series(["G1", "G1", "G2"], index=ids)


def test_a_group_whose_mean_covariate_overflows_is_refused_by_name() -> None:
    aligned, groups = _aligned([HUGE, HUGE, 1.0])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ConfigurationError) as refused:
            reduce_to_groups(TARGET, GOLD, aligned, groups)
    message = str(refused.value)
    assert "the mean covariate of group(s) 'G1' is not finite although the covariate of" in message
    assert "divide every covariate by one constant" in message


def test_large_values_whose_means_are_finite_are_reduced_as_before() -> None:
    aligned, groups = _aligned([1e300, 3e300, 1.0])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        reduced = reduce_to_groups(TARGET, GOLD, aligned, groups)
    assert reduced["covariate"].tolist() == [2e300, 1.0]
    runs = pd.DataFrame(
        {
            "text_id": ["t1", "t2", "t1", "t2"],
            "run_number": [1, 1, 2, 2],
            "score": [1e300, 3e300, 1.0, 3.0],
        }
    )
    means = group_runs(runs, groups, pd.Index(["t1", "t2"]))
    assert means["score"].tolist() == [2e300, 2.0]
    assert np.isfinite(means["score"]).all()


def test_the_refusal_of_a_run_average_names_the_overflow_among_its_causes() -> None:
    message = _refusal(OVERFLOWING["text-mean"])
    assert (
        "run-averaged score is not finite (no run with a score, an infinite value in a run, or "
        "finite run scores whose average overflows)" in message
    )
    assert (
        "unit 'text' drops and counts a text without a run with a score or with finite run "
        "scores whose average overflows, and refuses an infinite score too" in message
    )


def test_run_scores_that_overflow_in_both_directions_meet_the_panels_own_rules() -> None:
    # 150 runs of 1e308 and then 150 of -1e308: sums of the first overflow upwards and sums of
    # the last downwards, so the computed average of t1 is infinite or, where the two meet,
    # NaN, with numpy's warning of an invalid value beside that of the overflow.
    message = _refusal(OVERFLOWING["both-directions"])
    assert "'t1' of group 'G1'" in message
    assert "run-averaged score is not finite" in message
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        by_text = computed(OVERFLOWING["both-directions"], "text").row
    assert (by_text["status"], by_text["n_used"], by_text["n_dropped_missing"]) == ("ok", 5, 1)


OVERFLOW_NOTE = "are finite but their average is not"


def test_the_text_unit_warns_once_of_a_text_it_drops_for_an_average_that_overflows(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Under the text unit t1 is dropped and counted as a text without a score. numpy's warning
    # of the overflow is not raised, so the function's own warning is what names the text.
    with caplog.at_level(logging.WARNING, logger="copernicus.validity"), warnings.catch_warnings():
        warnings.simplefilter("error")
        row = computed(OVERFLOWING["text-mean"], "text").row
    assert (row["n_used"], row["n_dropped_missing"]) == (5, 1)
    notes = [record.getMessage() for record in caplog.records if OVERFLOW_NOTE in record.message]
    assert len(notes) == 1
    assert notes[0].startswith(f"target {TARGET!r}: the run scores of text(s) 't1' {OVERFLOW_NOTE}")
    assert "drops and counts" in notes[0]


def test_the_text_unit_does_not_warn_of_a_text_without_a_score_or_without_a_gold_value(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # t1 has a run without a score and one with; u1 has no run with a score; v1's average
    # overflows but v1 has no gold value, so the drop of v1 is the drop of a text without one.
    entries = [
        ("t1", "G1", [float("nan"), 3.0], 3.0),
        SIX_TEXTS[1],
        ("u1", "G2", [float("nan")], 2.0),
        ("v1", "G3", [HUGE, HUGE], None),
        *SIX_TEXTS[4:],
        ("y1", "G6", [2.0], 2.5),
    ]
    with caplog.at_level(logging.WARNING, logger="copernicus.validity"), warnings.catch_warnings():
        warnings.simplefilter("error")
        row = computed(entries, "text").row
    assert (row["n_used"], row["n_dropped_missing"]) == (5, 2)
    assert not [record for record in caplog.records if OVERFLOW_NOTE in record.getMessage()]


def test_a_run_that_a_text_of_the_group_lacks_is_left_out_whatever_its_mean() -> None:
    # t3 has no score in run 1, so run 1 is no run of G1's noise and its mean, which would
    # overflow, is never looked at; run 2 has all three texts.
    runs = pd.DataFrame(
        {
            "text_id": ["t1", "t2", "t1", "t2", "t3"],
            "run_number": [1, 1, 2, 2, 2],
            "score": [HUGE, HUGE, 1.0, 2.0, 6.0],
        }
    )
    groups = pd.Series(["G1", "G1", "G1"], index=["t1", "t2", "t3"])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        means = group_runs(runs, groups, pd.Index(["t1", "t2", "t3"]))
    assert means.to_dict("list") == {"text_id": ["G1"], "run_number": [2], "score": [3.0]}
