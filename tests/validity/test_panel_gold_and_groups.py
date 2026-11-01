"""Gold encodings and the clustering of the validity panel's bootstrap."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from validity_support import GROUP_SIZE, N_TEXTS, make_table, make_world

from copernicus.data import TextTable
from copernicus.validity.panel import PanelSettings, validity_panel


@pytest.fixture(scope="module")
def world():
    return make_world()


def test_a_boolean_gold_column_is_numeric_and_gives_auc(world):
    gold = world.gold.assign(high=world.gold["gold_alpha"] >= 3)
    panel = validity_panel(
        world.scores,
        gold,
        world.table,
        targets={"alpha": "high"},
        settings=PanelSettings(n_boot=30),
    )
    row = panel.frame.iloc[0]
    assert 0.5 < row["auc"] <= 1.0
    assert row["somers_d"] == pytest.approx(2 * row["auc"] - 1)


def test_a_nullable_integer_gold_with_missing_values_is_dropped_and_counted(world):
    nullable = pd.array(world.gold["gold_alpha"], dtype="Int64")
    nullable[[0, 1, 2]] = pd.NA
    gold = world.gold.assign(gold_alpha=nullable)
    panel = validity_panel(
        world.scores,
        gold,
        world.table,
        targets={"alpha": "gold_alpha"},
        settings=PanelSettings(n_boot=30),
    )
    assert panel.frame.iloc[0]["n_dropped_missing"] == 3
    assert panel.frame.iloc[0]["n"] == N_TEXTS - 3


def test_a_text_with_a_missing_covariate_is_dropped_and_counted(world):
    gold = world.gold.assign(size=np.arange(N_TEXTS, dtype=float) % 7)
    gold.loc[3, "size"] = np.nan
    panel = validity_panel(
        world.scores,
        gold,
        None,
        targets={"alpha": "gold_alpha"},
        settings=PanelSettings(n_boot=30, covariate="size"),
    )
    assert panel.frame.iloc[0]["n_dropped_missing"] == 1
    assert panel.n_used == N_TEXTS - 1


def _clustered_world():
    """Texts of one group share their gold and score up to a small jitter."""
    rng = np.random.default_rng(11)
    effects = rng.normal(size=N_TEXTS // GROUP_SIZE)
    gold = np.repeat(effects, GROUP_SIZE) + rng.normal(scale=0.05, size=N_TEXTS)
    score = gold + rng.normal(scale=0.5, size=N_TEXTS)
    ids = make_table().ids()
    scores = pd.DataFrame(
        {"text_id": ids, "run_number": 1, "attribute_id": "alpha", "score": score}
    )
    return scores, pd.DataFrame({"text_id": ids, "gold_alpha": gold})


def test_the_bootstrap_resamples_the_groups_of_the_table():
    scores, gold = _clustered_world()
    grouped = make_table()
    ungrouped = TextTable.from_frame(
        grouped.frame[["text_id", "text"]].assign(group_id=grouped.frame["text_id"])
    )
    settings = PanelSettings(n_boot=400)
    by_group = validity_panel(
        scores, gold, grouped, targets={"alpha": "gold_alpha"}, settings=settings
    )
    by_text = validity_panel(
        scores, gold, ungrouped, targets={"alpha": "gold_alpha"}, settings=settings
    )

    def width(panel):
        row = panel.frame.iloc[0]
        return row["pearson_high"] - row["pearson_low"]

    assert by_group.frame.iloc[0]["pearson"] == pytest.approx(by_text.frame.iloc[0]["pearson"])
    assert width(by_group) > width(by_text)


def test_a_target_whose_gold_is_missing_everywhere_keeps_a_too_few_row(world):
    gold = world.gold.assign(gold_alpha=np.nan)
    panel = validity_panel(
        world.scores,
        gold,
        world.table,
        targets={"alpha": "gold_alpha", "beta": "gold_beta"},
        settings=PanelSettings(n_boot=30, direction="auto"),
    )
    row = panel.frame.set_index("target").loc["alpha"]
    assert (row["status"], row["n"], row["n_used"], row["n_dropped_missing"]) == (
        "too_few",
        0,
        0,
        N_TEXTS,
    )
    assert row["direction_applied"] == "positive"
    assert panel.n_used == N_TEXTS
    assert panel.n_dropped_missing == N_TEXTS
