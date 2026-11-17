"""The validity panel scored once per group: the unit setting, its counts and its refusals."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from units_support import (
    GOLD,
    N_BOOT,
    NINE_TEXTS,
    TARGET,
    TARGETS,
    Entry,
    build,
    group_settings,
    panel_row,
    refusal,
)

from copernicus.data import TextTable
from copernicus.errors import ConfigurationError, SettingsError
from copernicus.validity.panel import PanelSettings, validity_panel
from copernicus.validity.panel_units import group_runs, reduce_to_groups


def test_group_unit_matches_the_hand_computed_rank_validity():
    # Group means: E 1, A 2, C 3, D 4.5, B 5; gold ranks: E 1, A 2, D 3, C 4, B 5. The rank
    # differences are 0, 0, 1, -1, 0, so rho = 1 - 6 * 2 / (5 * 24) = 0.9.
    row = panel_row(NINE_TEXTS, "group")
    assert row["unit"] == "group"
    assert row["n_used"] == 5
    assert row["n"] == 5
    assert row["n_texts"] == 9
    assert row["n_dropped_missing"] == 0
    assert row["rank_validity"] == pytest.approx(0.9, abs=1e-12)
    by_text = panel_row(NINE_TEXTS, "text")
    assert by_text["unit"] == "text"
    assert by_text["n_used"] == 9
    assert by_text["n_texts"] == 9
    assert abs(by_text["rank_validity"] - 0.9) > 0.01
    # Per text, with average ranks for the ties, the ranks of score and gold both have the sum
    # of squared deviations 57 and the cross-product 32.75.
    assert by_text["rank_validity"] == pytest.approx(32.75 / 57, abs=1e-12)


def test_group_unit_averages_runs_per_text_first():
    # Group G1: t1 has runs 1 and 2 (scores 1 and 3), t2 has run 1 only (score 5). The runs are
    # averaged per text first, so G1 enters with (mean(1, 3) + 5) / 2 = 3.5; the pooled mean of
    # its three run scores would be 3.0. The other groups score G3 1, G2 3.25, G4 6 and G5 8,
    # and the gold values rank G3, G2, G1, G4, G5. With 3.5 the scores rank the groups the same
    # way (rho = 1.0); with 3.0, G1 and G2 swap places (rho = 1 - 6 * 2 / (5 * 24) = 0.9).
    entries: list[Entry] = [
        ("t1", "G1", [1.0, 3.0], 3.0),
        ("t2", "G1", [5.0], 3.0),
        ("u1", "G2", [3.25], 2.0),
        ("v1", "G3", [1.0], 1.0),
        ("w1", "G4", [6.0], 4.0),
        ("x1", "G5", [8.0], 5.0),
    ]
    row = panel_row(entries, "group")
    assert (row["n_used"], row["n_texts"], row["n_runs"]) == (5, 6, 2)
    assert row["rank_validity"] == pytest.approx(1.0, abs=1e-12)
    assert row["rank_validity"] != pytest.approx(0.9, abs=1e-3)


def test_single_text_groups_equal_the_text_unit():
    rng = np.random.default_rng(3)
    gold = rng.integers(1, 8, size=12).astype(float)
    entries: list[Entry] = [
        (f"t{index:02d}", f"g{index:02d}", list(gold[index] + rng.normal(size=2)), gold[index])
        for index in range(12)
    ]
    by_text = panel_row(entries, "text")
    by_group = panel_row(entries, "group")
    assert (by_text["unit"], by_group["unit"]) == ("text", "group")
    assert by_text["status"] == "ok" and np.isfinite(by_text["noise"])
    pd.testing.assert_series_equal(by_group.drop("unit"), by_text.drop("unit"))


def test_group_with_two_gold_values_is_refused():
    entries: list[Entry] = [
        ("a1", "p1", [1.0], 1.0),
        ("a2", "p1", [2.0], 1.0),
        ("b1", "p2", [3.0], 1.0),
        ("b2", "p2", [4.0], 2.0),
        ("c1", "p3", [5.0], 3.0),
    ]
    message = refusal(entries)
    assert f"target {TARGET!r}" in message
    assert f"gold column {GOLD!r}" in message
    assert "'p2'" in message
    assert "'p1'" not in message and "'p3'" not in message


def test_missing_score_in_a_group_is_refused():
    entries: list[Entry] = [
        ("a1", "p1", [1.0], 1.0),
        ("a2", "p1", [2.0], 1.0),
        ("b1", "p2", [math.nan], 2.0),
        ("b2", "p2", [3.0], 2.0),
        ("c1", "p3", [4.0], 3.0),
        ("c2", "p3", [5.0], 3.0),
        ("d1", "p4", [6.0], 4.0),
    ]
    message = refusal(entries)
    assert f"target {TARGET!r}" in message
    assert "'b1' of group 'p2'" in message
    assert "run-averaged score is not finite (no run with a score" in message
    assert panel_row(entries, "text")["n_dropped_missing"] == 1


def test_missing_covariate_in_a_group_is_refused():
    scores, gold, table = build(NINE_TEXTS)
    sized = gold.assign(size=np.arange(1.0, 10.0))
    sized.loc[sized["text_id"] == "d2", "size"] = np.nan
    with pytest.raises(ConfigurationError) as refused:
        validity_panel(
            scores, sized, table, targets=TARGETS, settings=group_settings(covariate="size")
        )
    message = str(refused.value)
    assert f"target {TARGET!r}" in message
    assert "'d2' of group 'D'" in message
    assert "covariate" in message
    by_text = PanelSettings(n_boot=N_BOOT, covariate="size")
    frame = validity_panel(scores, sized, table, targets=TARGETS, settings=by_text).frame
    assert frame.iloc[0]["n_dropped_missing"] == 1


def test_the_gold_values_of_a_group_are_checked_before_its_scores():
    entries: list[Entry] = [
        ("a1", "p1", [1.0], 1.0),
        ("a2", "p1", [math.nan], 2.0),
        ("b1", "p2", [3.0], 3.0),
        ("c1", "p3", [4.0], 4.0),
    ]
    message = refusal(entries)
    assert f"gold column {GOLD!r}" in message
    assert "'p1' (1.0 on 'a1', 2.0 on 'a2')" in message
    assert "carry different values" in message and "score is not finite" not in message


def test_text_without_gold_row_stays_out_of_its_group():
    # Group A: a1 (score 2, gold 3) and a2 (score 10, no gold row). With A's score equal to a1's,
    # the scores rank B, A, C, D, E and the gold values B, C, A, D, E: the rank differences are
    # 0, -1, 1, 0, 0 and rho = 1 - 6 * 2 / (5 * 24) = 0.9. Had a2 entered, A would score 6 and
    # rank last: rho = 1 - 6 * 6 / (5 * 24) = 0.7.
    entries: list[Entry] = [
        ("a1", "A", [2.0], 3.0),
        ("a2", "A", [10.0], None),
        ("b1", "B", [1.0], 1.0),
        ("c1", "C", [3.0], 2.0),
        ("d1", "D", [4.0], 4.0),
        ("e1", "E", [5.0], 5.0),
    ]
    scores, gold, table = build(entries)
    panel = validity_panel(scores, gold, table, targets=TARGETS, settings=group_settings())
    row = panel.frame.iloc[0]
    assert row["n_dropped_missing"] == 1
    assert (row["n_used"], row["n_texts"]) == (5, 5)
    assert row["rank_validity"] == pytest.approx(0.9, abs=1e-12)
    assert (panel.n_used, panel.n_dropped_missing) == (5, 1)


def test_group_unit_needs_real_groups():
    scores, gold, table = build(NINE_TEXTS)
    sized = gold.assign(size=np.arange(1.0, 10.0))
    with pytest.raises(ConfigurationError, match="needs a group column"):
        validity_panel(
            scores, sized, None, targets=TARGETS, settings=group_settings(covariate="size")
        )
    ungrouped = TextTable.from_frame(table.frame[["text_id", "text"]])
    assert ungrouped.group_defaulted
    with pytest.raises(ConfigurationError, match="needs a group column"):
        validity_panel(scores, gold, ungrouped, targets=TARGETS, settings=group_settings())


def test_unknown_unit_is_refused():
    with pytest.raises(SettingsError, match="unit"):
        PanelSettings(unit="person")  # type: ignore[arg-type]


def test_group_unit_noise_uses_group_run_means():
    # Five groups of two texts with two runs each. The (group, run) means are A 2 and 2,
    # B 4.5 and 5, C 1.5 and 2, D 3.5 and 4, E 2.5 and 3. Scaled to 0-1 over all of them (1.5 to
    # 5), each of B, C, D and E moves by 0.5 / 3.5 = 1/7 between its runs and A does not move,
    # so the noise is (4 * (1/7) / sqrt(2)) / 5 = 4 / (35 * sqrt(2)). Per text the run scores
    # span 1 to 5 and move by 0.5, 0.5, 0.25, 0, 0, 0.25, 0.25, 0, 0.25, 0 on the 0-1 scale, so
    # the text-unit noise is 2 / (10 * sqrt(2)) = 0.2 / sqrt(2).
    entries: list[Entry] = [
        ("a1", "A", [1.0, 3.0], 2.0),
        ("a2", "A", [3.0, 1.0], 2.0),
        ("b1", "B", [4.0, 5.0], 5.0),
        ("b2", "B", [5.0, 5.0], 5.0),
        ("c1", "C", [2.0, 2.0], 1.0),
        ("c2", "C", [1.0, 2.0], 1.0),
        ("d1", "D", [3.0, 4.0], 4.0),
        ("d2", "D", [4.0, 4.0], 4.0),
        ("e1", "E", [2.0, 3.0], 3.0),
        ("e2", "E", [3.0, 3.0], 3.0),
    ]
    by_group = panel_row(entries, "group")
    by_text = panel_row(entries, "text")
    assert by_group["n_runs"] == 2
    assert by_group["noise"] == pytest.approx(4 / (35 * math.sqrt(2)), abs=1e-12)
    assert by_text["noise"] == pytest.approx(0.2 / math.sqrt(2), abs=1e-12)
    assert by_group["noise"] != pytest.approx(by_text["noise"], abs=1e-3)


@pytest.mark.parametrize("second_run_of_p2", [[], [math.nan]])
def test_group_noise_leaves_out_runs_that_not_every_text_of_the_group_has(second_run_of_p2):
    # No text's score changes between runs. P's texts share run 1 only (p2 has no run 2, or a
    # run 2 without a score), so P has one complete run and no spread; Q, R and S keep their
    # scores. The group noise is 0, as the text noise is; a mean over p1 alone in run 2 would
    # move P from 3 to 1 and report that as noise.
    entries: list[Entry] = [
        ("p1", "P", [1.0, 1.0], 1.0),
        ("p2", "P", [5.0, *second_run_of_p2], 1.0),
        ("q1", "Q", [2.0, 2.0], 2.0),
        ("r1", "R", [3.0, 3.0], 3.0),
        ("s1", "S", [4.0, 4.0], 4.0),
    ]
    by_group = panel_row(entries, "group")
    by_text = panel_row(entries, "text")
    assert by_group["noise"] == 0.0
    assert by_text["noise"] == 0.0


def test_group_noise_uses_only_the_runs_every_text_of_the_group_has():
    # A: a1 has runs 1 to 3, a2 runs 1 and 2, so A's complete runs are 1 and 2, with means 2 and
    # 4. B (one text) keeps 4, 4 and 6. C: c2's run 2 has no score, so C has one complete run
    # (mean 1.5), which counts toward the range but gives no spread. D keeps 5 and 6. Over all
    # kept (group, run) means, 1.5 to 6, the 0-1 values are A 1/9 and 5/9, B 5/9, 5/9 and 1,
    # D 7/9 and 1; their standard deviations are (4/9) / sqrt(2), 4 * sqrt(3) / 27 and
    # (2/9) / sqrt(2), and the noise is their mean: sqrt(2) / 9 + 4 * sqrt(3) / 81.
    entries: list[Entry] = [
        ("a1", "A", [1.0, 3.0, 2.0], 2.0),
        ("a2", "A", [3.0, 5.0], 2.0),
        ("b1", "B", [4.0, 4.0, 6.0], 3.0),
        ("c1", "C", [1.0, 1.0], 1.0),
        ("c2", "C", [2.0, math.nan], 1.0),
        ("d1", "D", [5.0, 6.0], 4.0),
    ]
    row = panel_row(entries, "group")
    assert row["n_runs"] == 3
    expected = math.sqrt(2) / 9 + 4 * math.sqrt(3) / 81
    assert row["noise"] == pytest.approx(expected, abs=1e-12)


def test_report_names_the_unit():
    scores, gold, table = build(NINE_TEXTS)
    panel = validity_panel(scores, gold, table, targets=TARGETS, settings=group_settings())
    lines = panel.report().splitlines()
    assert "one score per group" in lines[0] and "9 texts entered" in lines[0]
    header = next(line for line in lines if line.startswith("target "))
    assert header.split()[:3] == ["target", "groups", "texts"]
    (line,) = [line for line in lines if line.startswith(f"{TARGET} ")]
    assert line.split()[:3] == [TARGET, "5", "9"]


def test_the_covariate_line_of_the_group_unit_says_that_each_group_averages_its_covariate() -> None:
    scores, gold, table = build(NINE_TEXTS)
    lines = {
        unit: validity_panel(
            scores, gold, table, targets=TARGETS, settings=PanelSettings(n_boot=N_BOOT, unit=unit)
        )
        .report()
        .splitlines()[1]
        for unit in ("text", "group")
    }
    assert lines["text"].startswith("Intervals: ") and "averaged" not in lines["text"]
    assert lines["group"] == lines["text"].removesuffix(".") + ", averaged over each group's texts."


@pytest.mark.parametrize("unit", ["text", "group"])
def test_every_count_of_the_panel_is_a_nullable_integer_column(unit: str) -> None:
    # The second target scores every text alike, so its status is not ok and its n_boot missing.
    scores, gold, table = build(NINE_TEXTS)
    constant = scores.assign(attribute_id="beta", score=2.0)
    frame = validity_panel(
        pd.concat([scores, constant], ignore_index=True),
        gold,
        table,
        targets={TARGET: GOLD, "beta": GOLD},
        settings=PanelSettings(n_boot=N_BOOT, unit=unit),
    ).frame.set_index("target")
    counts = ("n_used", "n_texts", "n_dropped_missing", "n_runs", "n", "n_boot", "n_undefined_max")
    assert {name: str(frame[name].dtype) for name in counts} == dict.fromkeys(counts, "Int64")
    assert frame.loc["beta", "status"] != "ok" and pd.isna(frame.loc["beta", "n_boot"])
    assert frame["n_texts"].tolist() == [9, 9]


def test_reduce_to_groups_and_group_runs_return_one_row_per_group_and_per_group_and_run():
    aligned = pd.DataFrame(
        {"score": [1.0, 3.0, 5.0], "gold": [2.0, 2.0, 7.0], "covariate": [0.5, 1.5, 4.0]},
        index=pd.Index(["t1", "t2", "t3"], name="text_id"),
    )
    groups = pd.Series(["g1", "g1", "g2"], index=["t1", "t2", "t3"])
    reduced = reduce_to_groups(TARGET, GOLD, aligned, groups)
    assert list(reduced.index) == ["g1", "g2"]
    assert list(reduced.columns) == ["score", "gold", "covariate", "n_texts"]
    assert reduced.loc["g1"].tolist() == [2.0, 2.0, 1.0, 2]
    assert reduced.loc["g2"].tolist() == [5.0, 7.0, 4.0, 1]
    runs = pd.DataFrame(
        {
            "text_id": ["t1", "t1", "t2", "t2", "t3"],
            "run_number": [1, 2, 1, 2, 1],
            "score": [0.0, 2.0, 4.0, 4.0, 9.0],
        }
    )
    means = group_runs(runs, groups, pd.Index(["t1", "t2"]))
    assert list(means.columns) == ["text_id", "run_number", "score"]
    assert means.to_numpy().tolist() == [["g1", 1, 2.0], ["g1", 2, 3.0]]
