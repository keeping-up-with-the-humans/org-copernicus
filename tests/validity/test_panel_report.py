"""The plain-text report of the validity panel."""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest
from validity_support import make_world

from copernicus.data import TextTable
from copernicus.validity.panel import PanelSettings, validity_panel
from copernicus.validity.report import render_report

TARGETS = {"construct": "gold_construct", "alpha": "gold_alpha", "beta": "gold_beta"}
RANK_NOTE = (
    "Rank validity is an absolute value; its interval is bounded below by 0 and does not test "
    "zero validity."
)
ONE_GROUP_PER_TEXT = "groups: one per text (no group column)"


@pytest.fixture(scope="module")
def world():
    return make_world()


@pytest.fixture(scope="module")
def panel(world):
    return validity_panel(
        world.scores,
        world.gold,
        world.table,
        targets=TARGETS,
        settings=PanelSettings(n_boot=50, reliability=0.81),
    )


def test_the_report_is_what_the_panel_method_returns(panel):
    assert panel.report() == render_report(panel)


def test_the_header_gives_the_texts_that_entered_and_the_texts_dropped(panel):
    first_line = render_report(panel).splitlines()[0]
    assert "40 texts entered" in first_line
    assert "0 dropped" in first_line


def test_each_target_row_shows_the_rank_validity_with_its_interval_to_two_decimals(panel):
    lines = render_report(panel).splitlines()
    row = panel.frame.set_index("target").loc["alpha"]
    expected = (
        f"{row['rank_validity']:.2f} "
        f"[{row['rank_validity_low']:.2f}, {row['rank_validity_high']:.2f}]"
    )
    (line,) = [line for line in lines if line.startswith("alpha ")]
    assert expected in line


def test_the_ceiling_share_is_shown_when_a_ceiling_exists(panel):
    row = panel.frame.set_index("target").loc["alpha"]
    assert f"{row['share_of_ceiling']:.2f} of 0.90" in render_report(panel)


def test_a_panel_without_a_ceiling_shows_not_available_in_the_ceiling_column(world):
    plain = validity_panel(
        world.scores, world.gold, world.table, targets=TARGETS, settings=PanelSettings(n_boot=20)
    )
    (line,) = [line for line in render_report(plain).splitlines() if line.startswith("alpha ")]
    assert line.endswith("n/a")


def test_a_panel_with_every_target_ok_says_so(panel):
    assert "Every target was scored." in render_report(panel)


def test_a_target_that_is_not_ok_is_named_with_its_status(world):
    flat = world.scores.copy()
    flat.loc[flat["attribute_id"] == "beta", "score"] = 3
    panel = validity_panel(
        flat, world.gold, world.table, targets=TARGETS, settings=PanelSettings(n_boot=20)
    )
    text = render_report(panel)
    assert "beta (constant_score)" in text
    assert "Every target was scored." not in text
    (line,) = [line for line in text.splitlines() if line.startswith("beta ")]
    assert "n/a" in line and "-1.00" in line
    assert "calibration shortfall of -1" in " ".join(text.split())


def test_a_constant_score_against_a_constant_gold_does_not_claim_two_errors(world):
    flat = world.scores.copy()
    flat.loc[flat["attribute_id"].isin(["alpha", "beta"]), "score"] = 3
    gold = world.gold.assign(flat_gold=2.0)
    panel = validity_panel(
        flat,
        gold,
        world.table,
        targets={"alpha": "gold_alpha", "beta": "flat_gold"},
        settings=PanelSettings(n_boot=20),
    )
    assert list(panel.frame["status"]) == ["constant_score", "constant_score"]
    text = " ".join(render_report(panel).split())
    assert "For alpha, a constant score still has two errors" in text
    assert "calibration shortfall of -1" in text
    assert "beta, whose gold is constant too, has neither error" in text


def test_a_constant_score_against_a_constant_gold_alone_has_no_two_error_sentence(world):
    flat = world.scores.copy()
    flat.loc[flat["attribute_id"] == "beta", "score"] = 3
    gold = world.gold.assign(flat_gold=2.0)
    panel = validity_panel(
        flat, gold, world.table, targets={"beta": "flat_gold"}, settings=PanelSettings(n_boot=20)
    )
    text = " ".join(render_report(panel).split())
    assert "still has two errors" not in text and "calibration shortfall of -1" not in text
    assert "beta, whose gold is constant too, has neither error" in text


def test_the_noise_clause_leaves_out_a_target_that_lost_every_text(world):
    gold = world.gold.assign(gold_alpha=np.nan)
    flat = world.scores.copy()
    flat.loc[flat["attribute_id"] == "beta", "score"] = 3
    lost = validity_panel(
        world.scores,
        gold,
        world.table,
        targets={"alpha": "gold_alpha"},
        settings=PanelSettings(n_boot=20),
    )
    only = " ".join(render_report(lost).split())
    assert "alpha (too_few)" in only
    assert "noise is still reported" not in only
    assert "No text of alpha kept a score, a gold value and a covariate" in only
    mixed = validity_panel(
        flat,
        gold,
        world.table,
        targets={"alpha": "gold_alpha", "beta": "gold_beta"},
        settings=PanelSettings(n_boot=20),
    )
    text = " ".join(render_report(mixed).split())
    assert "the noise is still reported for beta." in text
    assert "No text of alpha kept" in text


def test_the_docstring_of_render_report_qualifies_the_constant_score_claim():
    doc = " ".join((render_report.__doc__ or "").split())
    assert "unless the gold is constant too" in doc and "noise" in doc


def test_the_report_names_the_targets_whose_score_was_negated(world, panel):
    reversed_beta = world.gold.assign(gold_beta=6 - world.gold["gold_beta"])
    flipped = validity_panel(
        world.scores,
        reversed_beta,
        world.table,
        targets=TARGETS,
        settings=PanelSettings(n_boot=20, direction="auto"),
    )
    assert list(flipped.frame["direction_applied"]) == ["positive", "positive", "negative"]
    flat = " ".join(render_report(flipped).split())
    assert "Score negated before every metric (direction negative): beta." in flat
    assert "negated" not in render_report(panel)


def test_the_note_under_the_table_says_the_interval_does_not_test_zero_validity(panel):
    lines = render_report(panel).splitlines()
    last_row = max(index for index, line in enumerate(lines) if line.startswith("beta "))
    assert lines[last_row + 1] == RANK_NOTE
    assert lines.count(RANK_NOTE) == 1


def test_a_score_that_falls_as_the_gold_rises_is_warned_under_the_default_direction(world, panel):
    reversed_beta = world.gold.assign(gold_beta=6 - world.gold["gold_beta"])
    reversed_panel = validity_panel(
        world.scores, reversed_beta, world.table, targets=TARGETS, settings=PanelSettings(n_boot=20)
    )
    spearman = reversed_panel.frame.set_index("target").loc["beta", "spearman"]
    assert spearman < 0
    warnings = [
        line for line in render_report(reversed_panel).splitlines() if line.startswith("Warning:")
    ]
    assert warnings == [
        f"Warning: the score falls as the gold rises for beta (signed Spearman {spearman:.2f}); "
        "rank validity does not show the sign, so check the direction of the score "
        "(direction negative or auto)."
    ]
    assert "Warning:" not in render_report(panel)
    flipped = validity_panel(
        world.scores,
        reversed_beta,
        world.table,
        targets=TARGETS,
        settings=PanelSettings(n_boot=20, direction="auto"),
    )
    assert "Warning:" not in render_report(flipped)


def test_the_report_says_one_group_per_text_when_the_table_names_no_group(world, panel):
    ungrouped = TextTable.from_frame(world.table.frame[["text_id", "text"]])
    assert ungrouped.group_defaulted
    plain = validity_panel(
        world.scores, world.gold, ungrouped, targets=TARGETS, settings=PanelSettings(n_boot=20)
    )
    assert plain.groups_defaulted and not panel.groups_defaulted
    basis = render_report(plain).splitlines()[1]
    assert basis.startswith("Intervals: 95 percent") and ONE_GROUP_PER_TEXT in basis
    assert ONE_GROUP_PER_TEXT not in render_report(panel)
    gold = world.gold.assign(length=np.arange(len(world.gold), dtype=float))
    tableless = validity_panel(
        world.scores,
        gold,
        None,
        targets={"alpha": "gold_alpha"},
        settings=PanelSettings(n_boot=20, covariate="length"),
    )
    assert tableless.groups_defaulted and ONE_GROUP_PER_TEXT in render_report(tableless)


def test_the_table_is_fixed_width(panel):
    lines = [line for line in render_report(panel).splitlines() if line.strip()]
    table = lines[2:6]
    separator = table[1]
    assert set(separator) <= {"-", " "}
    assert all(len(line) <= len(separator) for line in table)


def test_importing_the_validity_package_does_not_load_scikit_learn():
    code = "import sys, copernicus.validity; sys.exit('sklearn' in sys.modules)"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
