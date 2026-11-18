"""The validity panel on hand-built frames: where the covariate comes from and what it must be."""

from __future__ import annotations

import numpy as np
import pytest
from validity_support import N_TEXTS, World, make_table, make_world

from copernicus.errors import ConfigurationError
from copernicus.validity.panel import PanelSettings, validity_panel

FAST = PanelSettings(n_boot=30)
TARGETS = {"construct": "gold_construct", "alpha": "gold_alpha", "beta": "gold_beta"}


@pytest.fixture(scope="module")
def world() -> World:
    return make_world()


def test_without_a_table_the_covariate_comes_from_the_gold_and_groups_are_the_texts(world):
    gold = world.gold.assign(size=np.arange(N_TEXTS) % 7)
    panel = validity_panel(
        world.scores,
        gold,
        None,
        targets={"alpha": "gold_alpha"},
        settings=PanelSettings(n_boot=30, covariate="size"),
    )
    assert panel.frame.iloc[0]["covariate"] == "size"
    assert np.isfinite(panel.frame.iloc[0]["covariate_linked_error"])


def test_without_a_table_a_covariate_is_required(world):
    with pytest.raises(ConfigurationError, match="table or a covariate"):
        validity_panel(world.scores, world.gold, None, targets=TARGETS, settings=FAST)


def test_a_named_covariate_absent_everywhere_is_refused(world):
    with pytest.raises(ConfigurationError, match="nowhere"):
        validity_panel(
            world.scores,
            world.gold,
            world.table,
            targets=TARGETS,
            settings=PanelSettings(n_boot=30, covariate="nowhere"),
        )


def test_a_named_covariate_is_looked_up_in_the_table_frame(world):
    panel = validity_panel(
        world.scores,
        world.gold,
        world.table,
        targets=TARGETS,
        settings=PanelSettings(n_boot=30, covariate="n_words", covariate_scale="raw"),
    )
    assert (panel.frame["covariate"] == "n_words").all()
    assert (panel.frame["covariate_scale"] == "raw").all()


def test_a_non_numeric_covariate_is_refused(world):
    gold = world.gold.assign(size=["big"] * N_TEXTS)
    with pytest.raises(ConfigurationError, match="size.*numeric"):
        validity_panel(
            world.scores,
            gold,
            None,
            targets=TARGETS,
            settings=PanelSettings(n_boot=30, covariate="size"),
        )


def test_a_scored_text_absent_from_the_table_is_refused_with_its_id(world):
    smaller = make_table(N_TEXTS - 3)
    with pytest.raises(ConfigurationError, match=r"s037.*s038.*s039"):
        validity_panel(world.scores, world.gold, smaller, targets=TARGETS, settings=FAST)


def test_a_constant_covariate_gives_a_nan_error_and_no_interval_but_no_failure(world):
    gold = world.gold.assign(flat=1.0)
    panel = validity_panel(
        world.scores,
        gold,
        None,
        targets={"alpha": "gold_alpha"},
        settings=PanelSettings(n_boot=30, covariate="flat"),
    )
    row = panel.frame.iloc[0]
    assert np.isnan(row["covariate_linked_error"])
    assert np.isnan(row["covariate_linked_error_low"])
    assert np.isfinite(row["rank_validity_low"])
