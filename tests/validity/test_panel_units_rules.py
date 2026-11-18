"""The validity panel scored once per group: which texts and groups take part, and refusals."""

from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd
import pytest
from units_support import (
    GOLD,
    N_BOOT,
    SIX_TEXTS,
    TARGET,
    TARGETS,
    Entry,
    build,
    group_settings,
    refusal,
)

from copernicus.data import TextTable
from copernicus.errors import ConfigurationError
from copernicus.validity.panel import PanelSettings, validity_panel
from copernicus.validity.panel_units import reduce_to_groups


def with_group_ids(table: TextTable, group_ids: list[object]) -> TextTable:
    """The table with other raw group ids, one per text in table order."""
    frame = table.frame.assign(group_id=pd.Series(group_ids, dtype=object))
    return TextTable.from_frame(frame)


@pytest.mark.parametrize(
    ("group_ids", "named", "remedy"),
    [
        ([1, "1", "B", "C", "D", "E"], ("1 (int)", "'1' (str)"), "rename"),
        ([1, 1.0, "B", "C", "D", "E"], ("1 (int)", "1.0 (float)"), "write"),
    ],
)
def test_group_ids_that_merge_or_split_as_text_are_refused(group_ids, named, remedy):
    scores, gold, table = build(SIX_TEXTS)
    table = with_group_ids(table, group_ids)
    assert table.frame["group_id"].map(type).tolist()[:2] == [type(v) for v in group_ids[:2]]
    with pytest.raises(ConfigurationError) as refused:
        validity_panel(scores, gold, table, targets=TARGETS, settings=group_settings())
    message = str(refused.value)
    assert named[0] in message and named[1] in message
    assert remedy in message
    by_text = validity_panel(
        scores, gold, table, targets=TARGETS, settings=PanelSettings(n_boot=N_BOOT)
    ).frame.iloc[0]
    assert (by_text["unit"], by_text["n_used"]) == ("text", 6)


def unscored(entries: list[Entry], text: str, gold: float) -> list[Entry]:
    """The entries with no score row for one text, which keeps a gold row with ``gold``."""
    return [
        (name, group, [], gold) if name == text else (name, group, runs, value)
        for name, group, runs, value in entries
    ]


def test_a_text_with_a_gold_row_but_no_score_row_stays_out_of_its_group():
    # t2 has a gold row but was never scored for the target, so G1's score is t1's alone,
    # mean(1, 3) = 2. The scores rank G3 1, G1 2, G2 3.25, G4 6, G5 8 and the gold values G3, G2,
    # G1, G4, G5: the rank differences are 0, -1, 1, 0, 0 and rho = 1 - 6 * 2 / (5 * 24) = 0.9.
    scores, gold, table = build(unscored(SIX_TEXTS, "t2", 3.0))
    panel = validity_panel(scores, gold, table, targets=TARGETS, settings=group_settings())
    row = panel.frame.iloc[0]
    assert (row["n_used"], row["n_texts"], row["n_dropped_missing"]) == (5, 5, 0)
    assert row["rank_validity"] == pytest.approx(0.9, abs=1e-12)
    assert (panel.n_used, panel.n_dropped_missing) == (5, 0)


def test_a_text_scored_only_for_another_target_stays_out_of_this_targets_group():
    scores, gold, table = build(SIX_TEXTS)
    both = pd.concat([scores, scores.assign(attribute_id="beta")], ignore_index=True)
    both = both[(both["attribute_id"] != TARGET) | (both["text_id"] != "t2")]
    targets = {TARGET: TARGETS[TARGET], "beta": TARGETS[TARGET]}
    panel = validity_panel(both, gold, table, targets=targets, settings=group_settings())
    rows = panel.frame.set_index("target")
    assert (rows.loc[TARGET, "n_used"], rows.loc[TARGET, "n_texts"]) == (5, 5)
    assert rows.loc[TARGET, "rank_validity"] == pytest.approx(0.9, abs=1e-12)
    assert (rows.loc["beta", "n_used"], rows.loc["beta", "n_texts"]) == (5, 6)
    assert rows.loc["beta", "rank_validity"] == pytest.approx(1.0, abs=1e-12)
    assert panel.n_used == 6


def test_a_gold_conflict_on_an_unscored_text_is_refused():
    scores, gold, table = build(unscored(SIX_TEXTS, "t2", 9.0))
    with pytest.raises(ConfigurationError) as refused:
        validity_panel(scores, gold, table, targets=TARGETS, settings=group_settings())
    message = str(refused.value)
    assert f"target {TARGET!r}" in message
    assert f"gold column {TARGETS[TARGET]!r}" in message
    assert "'G1' (3.0 on 't1', 9.0 on 't2')" in message


def test_two_gold_values_in_a_group_without_a_scored_text_do_not_stop_the_panel():
    # The gold values are compared within the groups that take part, those that hold a text
    # scored for the target with a finite gold value. No text of G1 is scored here, so its values
    # 3.0 and 9.0 are not compared; scoring t1 makes them stop the panel (the test above).
    scores, gold, table = build(unscored(unscored(SIX_TEXTS, "t1", 3.0), "t2", 9.0))
    panel = validity_panel(scores, gold, table, targets=TARGETS, settings=group_settings())
    row = panel.frame.iloc[0]
    assert (row["status"], row["n_used"], row["n_texts"], row["n_dropped_missing"]) == (
        "ok",
        4,
        4,
        0,
    )


def mixed_runs(middle: float) -> list[Entry]:
    """G1's text t1 scores 2, ``middle`` and 4 in its runs, its text t2 6, 6.25 and 6.5.

    The other groups' single run scores equal their gold values, and the scores and the gold
    values both span 0 to 10, so only G1 departs from its gold value (5).
    """
    return [
        ("t1", "G1", [2.0, middle, 4.0], 5.0),
        ("t2", "G1", [6.0, 6.25, 6.5], 5.0),
        ("u1", "G2", [0.0], 0.0),
        ("v1", "G3", [10.0], 10.0),
        ("w1", "G4", [2.0], 2.0),
        ("x1", "G5", [7.0], 7.0),
        ("y1", "G6", [9.0], 9.0),
    ]


@pytest.mark.parametrize("infinite", [math.inf, -math.inf])
@pytest.mark.parametrize("key", [TARGET, "construct", "construct:construct"])
def test_an_infinite_run_score_is_refused_beside_finite_runs(key, infinite):
    # An infinite score is a broken value, not a missing one: both units refuse t1 from its raw
    # scores, before the runs are averaged (version 0.1.0 let the text unit drop and count it).
    scores, gold, table = build(mixed_runs(infinite))
    for settings in (group_settings(), PanelSettings(n_boot=N_BOOT)):
        with pytest.raises(ConfigurationError) as refused:
            validity_panel(scores, gold, table, targets={key: GOLD}, settings=settings)
        message = str(refused.value)
        assert f"target {key!r}" in message
        assert "text(s) 't1' of group 'G1' carry an infinite score value" in message


@pytest.mark.parametrize("unit", ["text", "group"])
@pytest.mark.parametrize("key", [TARGET, "construct", "construct:construct"])
def test_finite_run_scores_whose_average_would_overflow_are_refused_by_the_limit(key, unit):
    # t1's run scores of 1e308 are finite, but their sum is larger than the largest float, so its
    # run average would be infinite. Version 0.1.0 dropped and counted t1 under the text unit.
    # The scores are above the library's limit, so the panel refuses them under both units before
    # it averages the runs; the overflow tests reach the group unit's refusal of such an average,
    # and the text unit's drop, through compute_target.
    scores, gold, table = build([("t1", "G1", [1e308, 1e308], 3.0), *SIX_TEXTS[1:]])
    settings = PanelSettings(n_boot=N_BOOT, unit=unit)  # type: ignore[arg-type]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ConfigurationError) as refused:
            validity_panel(scores, gold, table, targets={key: GOLD}, settings=settings)
    assert str(refused.value).startswith(f"target {key!r}: text(s) 't1' have a score above 1e+150")


@pytest.mark.parametrize("key", [TARGET, "construct", "construct:construct"])
def test_a_missing_run_score_is_averaged_over_the_texts_other_runs(key):
    # t1's missing run 2 leaves mean(2, 4) = 3, so G1's score is (3 + mean(6, 6.25, 6.5)) / 2 =
    # (3 + 6.25) / 2 = 4.625 against its gold value 5. On the 0-1 scale (both span 0 to 10) G1
    # departs by (4.625 - 5) / 10 and every other group by 0, so the level displacement over the
    # six groups is -0.0375 / 6 and the mean absolute error 0.0375 / 6. G1's noise has its runs
    # 1 and 3 only, with the means 4 and 5.25, or .4 and .525 on the scale: .125 / sqrt(2).
    scores, gold, table = build(mixed_runs(math.nan))
    panel = validity_panel(scores, gold, table, targets={key: GOLD}, settings=group_settings())
    row = panel.frame.iloc[0]
    assert (row["status"], row["n_used"], row["n_texts"], row["n_dropped_missing"]) == (
        "ok",
        6,
        7,
        0,
    )
    assert row["level_displacement"] == pytest.approx(-0.0375 / 6, abs=1e-12)
    assert row["mae_01"] == pytest.approx(0.0375 / 6, abs=1e-12)
    assert row["noise"] == pytest.approx(0.125 / math.sqrt(2), abs=1e-12)


@pytest.mark.parametrize("scored", [True, False])
def test_gold_values_one_float_step_apart_are_two_values(scored):
    # 1.0 and the next 64-bit float above it are two values: the check compares 64-bit floats
    # exactly, with no tolerance, also when the second value sits on a text never scored.
    step = float(np.nextafter(1.0, 2.0))
    entries = [(t, g, runs, 1.0 if t == "t1" else value) for t, g, runs, value in SIX_TEXTS]
    entries = [(t, g, runs, step if t == "t2" else value) for t, g, runs, value in entries]
    if not scored:
        entries = unscored(entries, "t2", step)
    message = refusal(entries)
    assert f"target {TARGET!r}" in message
    assert f"gold column {GOLD!r}" in message
    assert f"'G1' (1.0 on 't1', {step!r} on 't2')" in message


def test_integer_gold_values_that_are_one_64_bit_float_are_one_value():
    # 2**53 and 2**53 + 1 differ as integers but are the same 64-bit float, the form in which the
    # gold values are compared, so G1 has one gold value.
    scores, gold, table = build(SIX_TEXTS)
    integers = np.array([2**53, 2**53 + 1, 2, 1, 4, 5], dtype="int64")
    gold = gold.assign(**{GOLD: integers})
    panel = validity_panel(scores, gold, table, targets=TARGETS, settings=group_settings())
    row = panel.frame.iloc[0]
    assert (row["status"], row["n_used"], row["n_texts"]) == ("ok", 5, 6)


def test_reduce_to_groups_compares_integer_gold_as_64_bit_floats():
    aligned = pd.DataFrame(
        {
            "score": [1.0, 3.0],
            "gold": np.array([2**53, 2**53 + 1], dtype="int64"),
            "covariate": [1.0, 2.0],
        },
        index=pd.Index(["a", "b"], name="text_id"),
    )
    groups = pd.Series(["A", "A"], index=["a", "b"])
    reduced = reduce_to_groups(TARGET, GOLD, aligned, groups)
    assert reduced.loc["A"].tolist() == [2.0, float(2**53), 1.5, 2]


@pytest.mark.parametrize("infinite", [math.inf, -math.inf])
def test_reduce_to_groups_refuses_a_run_average_that_an_infinite_run_score_makes_infinite(
    infinite: float,
) -> None:
    # Called directly, the reduction receives the run averages of its caller, so an infinite run
    # score reaches it as an infinite average; the panel refuses such a score before it averages.
    aligned = pd.DataFrame(
        {
            "score": [float(np.mean([2.0, infinite, 4.0])), 3.0],
            "gold": [5.0, 5.0],
            "covariate": [1.0, 2.0],
        },
        index=pd.Index(["a", "b"], name="text_id"),
    )
    groups = pd.Series(["A", "A"], index=["a", "b"])
    with pytest.raises(ConfigurationError) as refused:
        reduce_to_groups(TARGET, GOLD, aligned, groups)
    message = str(refused.value)
    assert "'a' of group 'A'" in message
    assert "run-averaged score is not finite (no run with a score, an infinite value in" in message
