"""The panel's six intervals, pinned to the values computed before the vector bootstrap path."""

from __future__ import annotations

import pytest
from validity_support import make_world

import copernicus.validity.panel_targets as panel_targets
from copernicus.validity.panel import PanelSettings, validity_panel

PINNED = {
    "standardized": {
        "rank_validity_low": 0.9083130856290029,
        "rank_validity_high": 0.972714320283449,
        "pearson_low": 0.9139779636551213,
        "pearson_high": 0.9725525968595629,
        "mae_01_low": 0.053125,
        "mae_01_high": 0.11875,
        "level_displacement_low": -0.025,
        "level_displacement_high": 0.06890625000000004,
        "calibration_shortfall_low": -0.16443471038001164,
        "calibration_shortfall_high": 0.0034793195934503395,
        "covariate_linked_error_low": -0.042705440230677995,
        "covariate_linked_error_high": 0.0230242159007148,
    },
    "raw": {
        "rank_validity_low": 0.9083130856290029,
        "rank_validity_high": 0.972714320283449,
        "pearson_low": 0.9139779636551213,
        "pearson_high": 0.9725525968595629,
        "mae_01_low": 0.053125,
        "mae_01_high": 0.11875,
        "level_displacement_low": -0.025,
        "level_displacement_high": 0.06890625000000004,
        "calibration_shortfall_low": -0.16443471038001164,
        "calibration_shortfall_high": 0.0034793195934503395,
        "covariate_linked_error_low": -0.09696193227977824,
        "covariate_linked_error_high": 0.04806623330697365,
    },
}
N_BOOT = 200


@pytest.fixture(scope="module")
def world():
    return make_world()


def alpha_panel(world, scale: str):
    return validity_panel(
        world.scores,
        world.gold,
        world.table,
        targets={"alpha": "gold_alpha"},
        settings=PanelSettings(n_boot=N_BOOT, seed=7, covariate_scale=scale),  # type: ignore[arg-type]
    )


@pytest.mark.parametrize("scale", ["standardized", "raw"])
def test_the_six_intervals_equal_the_values_of_the_per_metric_bootstrap(world, scale):
    row = alpha_panel(world, scale).frame.iloc[0]
    for column, expected in PINNED[scale].items():
        assert row[column] == pytest.approx(expected, rel=1e-12, abs=1e-12), column
    assert row["n_undefined_max"] == 0


def test_the_score_errors_run_once_per_replicate(world, monkeypatch):
    calls = []
    original = panel_targets.score_errors

    def counting(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(panel_targets, "score_errors", counting)
    alpha_panel(world, "standardized")
    point_estimates = 2  # one call per covariate scale
    assert len(calls) == point_estimates + N_BOOT
