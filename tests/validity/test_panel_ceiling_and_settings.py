"""The validity panel: ceilings, non-ok rows, bootstrap failures and settings."""

from __future__ import annotations

import dataclasses
from types import MappingProxyType

import numpy as np
import pandas as pd
import pytest
from validity_support import N_TEXTS, make_world

from copernicus.errors import ConfigurationError, EstimatorError, SettingsError
from copernicus.validity import cronbach_alpha
from copernicus.validity.panel import PanelSettings, validity_panel

FAST = PanelSettings(n_boot=30)
TARGETS = {"construct": "gold_construct", "alpha": "gold_alpha", "beta": "gold_beta"}


@pytest.fixture(scope="module")
def world():
    return make_world()


def test_reliability_and_items_together_are_refused_as_settings():
    items = pd.DataFrame(np.random.default_rng(0).normal(size=(30, 3)))
    with pytest.raises(SettingsError, match="reliability.*items|items.*reliability"):
        PanelSettings(n_boot=30, reliability=0.8, items=items)
    with pytest.raises(SettingsError, match="reliability.*items|items.*reliability"):
        PanelSettings(reliability={"gold_alpha": 0.8}, items=items)


def test_a_reliability_mapping_sets_the_ceiling_of_the_rows_of_its_gold_columns(world):
    reliability = {"gold_alpha": 0.81, "gold_beta": 0.64}
    panel = validity_panel(
        world.scores,
        world.gold,
        world.table,
        targets=TARGETS,
        settings=PanelSettings(n_boot=30, reliability=reliability),
    ).frame.set_index("target")
    assert panel.loc["alpha", "ceiling"] == pytest.approx(0.9)
    assert panel.loc["beta", "ceiling"] == pytest.approx(0.8)
    assert panel.loc["alpha", "share_of_ceiling"] == pytest.approx(
        panel.loc["alpha", "rank_validity"] / 0.9
    )
    assert np.isnan(panel.loc["construct", "ceiling"])
    assert np.isnan(panel.loc["construct", "share_of_ceiling"])


def test_a_reliability_mapping_naming_no_gold_column_of_the_targets_is_refused(world):
    with pytest.raises(ConfigurationError, match=r"gold_gamma.*gold_alpha"):
        validity_panel(
            world.scores,
            world.gold,
            world.table,
            targets=TARGETS,
            settings=PanelSettings(n_boot=30, reliability={"gold_gamma": 0.8}),
        )


@pytest.mark.parametrize(
    "reliability",
    [{}, {"gold_alpha": 1.5}, {"gold_alpha": 0.0}, {"gold_alpha": float("nan")}, {"": 0.8}],
)
def test_a_reliability_mapping_outside_its_range_is_refused(reliability):
    with pytest.raises(SettingsError, match="reliability"):
        PanelSettings(reliability=reliability)


def test_a_reliability_mapping_is_copied_and_the_settings_still_compare_and_hash():
    given = {"gold_alpha": 0.81}
    settings = PanelSettings(reliability=given)
    given["gold_alpha"] = 0.5
    assert settings.reliability == {"gold_alpha": 0.81}
    assert settings == PanelSettings(reliability={"gold_alpha": 0.81})
    assert settings != PanelSettings(reliability={"gold_alpha": 0.64})
    assert hash(settings) == hash(PanelSettings(reliability={"gold_alpha": 0.81}))


def test_a_reliability_mapping_is_read_only_inside_the_frozen_settings():
    settings = PanelSettings(reliability={"gold_alpha": 0.81})
    assert isinstance(settings.reliability, MappingProxyType)
    with pytest.raises(TypeError):
        settings.reliability["gold_alpha"] = 0.5  # type: ignore[index]
    with pytest.raises(TypeError):
        settings.reliability["gold_beta"] = 0.5  # type: ignore[index]
    with pytest.raises(dataclasses.FrozenInstanceError):
        settings.reliability = 0.5  # type: ignore[misc]
    assert dataclasses.replace(settings, n_boot=5).reliability == {"gold_alpha": 0.81}


def test_items_give_the_ceiling_on_the_construct_row_only(world):
    base = np.random.default_rng(3).normal(size=60)
    items = pd.DataFrame(
        {
            name: base + np.random.default_rng(4 + i).normal(size=60) * 0.5
            for i, name in enumerate("abc")
        }
    )
    alpha = cronbach_alpha(items)
    panel = validity_panel(
        world.scores,
        world.gold,
        world.table,
        targets=TARGETS,
        settings=PanelSettings(n_boot=30, items=items),
    ).frame.set_index("target")
    assert panel.loc["construct", "ceiling"] == pytest.approx(np.sqrt(alpha))
    assert panel.loc["construct", "share_of_ceiling"] == pytest.approx(
        panel.loc["construct", "rank_validity"] / np.sqrt(alpha)
    )
    assert panel.loc[["alpha", "beta"], "ceiling"].isna().all()
    assert panel.loc[["alpha", "beta"], "share_of_ceiling"].isna().all()


def test_items_without_a_construct_target_are_refused(world):
    items = pd.DataFrame(np.random.default_rng(0).normal(size=(30, 3)))
    with pytest.raises(ConfigurationError, match="construct"):
        validity_panel(
            world.scores,
            world.gold,
            world.table,
            targets={"alpha": "gold_alpha"},
            settings=PanelSettings(n_boot=30, items=items),
        )


def test_items_with_a_non_positive_alpha_are_refused(world):
    items = pd.DataFrame(np.random.default_rng(5).normal(size=(200, 3)))
    items.iloc[:, 1] = -items.iloc[:, 0]
    with pytest.raises(ConfigurationError, match="alpha"):
        validity_panel(
            world.scores,
            world.gold,
            world.table,
            targets=TARGETS,
            settings=PanelSettings(n_boot=30, items=items),
        )


def test_a_constant_score_keeps_its_row_with_two_errors_and_no_intervals(world):
    flat = world.scores.copy()
    flat.loc[flat["attribute_id"] == "beta", "score"] = 3
    panel = validity_panel(
        flat, world.gold, world.table, targets=TARGETS, settings=FAST
    ).frame.set_index("target")
    row = panel.loc["beta"]
    assert row["status"] == "constant_score"
    assert np.isnan(row[["rank_validity", "spearman", "pearson", "kendall"]].astype(float)).all()
    for column in (
        "mae_01",
        "level_displacement",
        "rank_validity_low",
        "rank_validity_high",
        "mae_01_low",
    ):
        assert np.isnan(row[column])
    assert row["calibration_shortfall"] == -1.0
    assert np.isfinite(row["covariate_linked_error"])
    assert row["noise"] == 0.0
    assert row["n_used"] == N_TEXTS
    assert pd.isna(row["n_undefined_max"])
    assert panel.loc["alpha", "status"] == "ok"


def test_the_direction_is_positive_under_auto_when_the_association_is_not_ok(world):
    flat = world.scores.copy()
    flat.loc[flat["attribute_id"] == "beta", "score"] = 3
    panel = validity_panel(
        flat,
        world.gold,
        world.table,
        targets={"beta": "gold_beta"},
        settings=PanelSettings(n_boot=30, direction="auto"),
    )
    assert panel.frame.iloc[0]["direction_applied"] == "positive"


def test_a_bootstrap_failure_carries_the_target_name(world):
    gold = world.gold.assign(rare=0.0)
    gold.loc[0, "rare"] = 1.0
    with pytest.raises(EstimatorError, match=r"^target alpha: bootstrap metric .*undefined"):
        validity_panel(
            world.scores,
            gold,
            None,
            targets={"alpha": "rare"},
            settings=PanelSettings(n_boot=200, covariate="gold_beta"),
        )


def test_single_run_scores_have_no_noise():
    world = make_world(n_runs=1)
    panel = validity_panel(world.scores, world.gold, world.table, targets=TARGETS, settings=FAST)
    assert (panel.frame["n_runs"] == 1).all()
    assert panel.frame["noise"].isna().all()


def test_an_empty_target_mapping_is_refused(world):
    with pytest.raises(SettingsError, match="target"):
        validity_panel(world.scores, world.gold, world.table, targets={}, settings=FAST)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_boot": 0},
        {"covariate_scale": "log"},
        {"direction": "up"},
        {"reliability": 0.0},
        {"reliability": 1.5},
        {"reliability": float("nan")},
        {"seed": -1},
        {"seed": 2.5},
        {"seed": True},
    ],
)
def test_settings_outside_their_range_are_refused(kwargs):
    with pytest.raises(SettingsError):
        PanelSettings(**kwargs)


def test_a_numpy_integer_seed_is_accepted():
    assert PanelSettings(seed=np.int64(7)).seed == 7  # type: ignore[arg-type]


def test_settings_are_frozen_and_compare_without_the_items_frame():
    first = PanelSettings(items=pd.DataFrame({"a": [1.0, 2.0], "b": [2.0, 1.0]}))
    second = PanelSettings(items=pd.DataFrame({"a": [3.0, 4.0], "b": [4.0, 3.0]}))
    assert first == second
    assert "items" not in repr(first)
    with pytest.raises(dataclasses.FrozenInstanceError):
        first.n_boot = 5  # type: ignore[misc]
    assert PanelSettings() == PanelSettings(
        covariate=None, covariate_scale="standardized", n_boot=1000, seed=42, direction="positive"
    )


def test_a_bootstrap_that_cannot_run_names_the_target(world):
    values = [1.0, 2.0, 3.0, 4.0] + [np.nan] * (N_TEXTS - 4)
    gold = world.gold.assign(only_one_group=values)
    with pytest.raises(ConfigurationError, match=r"^target alpha: .*two clusters") as raised:
        validity_panel(
            world.scores,
            gold,
            world.table,
            targets={"alpha": "only_one_group"},
            settings=PanelSettings(n_boot=30),
        )
    assert isinstance(raised.value.__cause__, ConfigurationError)


def test_rows_without_a_bootstrap_report_no_replicate_count(world):
    flat = world.scores.copy()
    flat.loc[flat["attribute_id"] == "beta", "score"] = 3
    frame = validity_panel(
        flat, world.gold, world.table, targets=TARGETS, settings=PanelSettings(n_boot=30)
    ).frame.set_index("target")
    assert frame.loc["alpha", "n_boot"] == 30
    assert pd.isna(frame.loc["beta", "n_boot"])
    assert pd.isna(frame.loc["beta", "n_undefined_max"])
