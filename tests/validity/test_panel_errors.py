"""The error anatomy of the validity panel: the noise scale and a constant score."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from validity_support import World, make_world

from copernicus.validity import score_errors
from copernicus.validity.panel import PanelSettings, validity_panel

FAST = PanelSettings(n_boot=30)


@pytest.fixture(scope="module")
def world() -> World:
    return make_world()


def noise_01(scores: pd.DataFrame, attribute: str, used: set[str]) -> float:
    """Mean over texts of sd(score01): the run scores of the used texts, min-max scaled
    with the range over all their (text, run) rows, sd with ddof=1 per text."""
    runs = scores[(scores["attribute_id"] == attribute) & scores["text_id"].isin(used)]
    values = runs["score"].astype(float)
    scaled = (values - values.min()) / (values.max() - values.min())
    return float(runs.assign(score01=scaled).groupby("text_id")["score01"].std(ddof=1).mean())


def test_the_noise_is_the_sd_of_the_min_max_scaled_run_scores(world: World):
    panel = validity_panel(
        world.scores, world.gold, world.table, targets={"alpha": "gold_alpha"}, settings=FAST
    )
    noise = panel.frame.iloc[0]["noise"]
    assert noise == pytest.approx(noise_01(world.scores, "alpha", set(world.gold["text_id"])))
    alpha = world.scores[world.scores["attribute_id"] == "alpha"]
    raw = alpha.assign(score=alpha["score"].astype(float)).groupby("text_id")["score"].std(ddof=1)
    assert noise == pytest.approx(raw.mean() / 4.0, abs=1e-12)  # the run scores span 1 to 5


def test_the_noise_uses_only_the_texts_that_entered_the_target(world: World):
    alpha = world.scores[world.scores["attribute_id"] == "alpha"]
    tops = set(alpha.loc[alpha["score"].astype(float) == 5, "text_id"])
    gold = world.gold.assign(gold_alpha=world.gold["gold_alpha"].astype(float))
    gold.loc[gold["text_id"].isin(tops), "gold_alpha"] = np.nan
    panel = validity_panel(
        world.scores, gold, world.table, targets={"alpha": "gold_alpha"}, settings=FAST
    )
    row = panel.frame.iloc[0]
    used = set(world.gold["text_id"]) - tops
    assert row["n_used"] == len(used)
    assert row["noise"] == pytest.approx(noise_01(world.scores, "alpha", used), abs=1e-12)
    every_text = noise_01(world.scores, "alpha", set(world.gold["text_id"]))
    assert row["noise"] != pytest.approx(every_text, abs=1e-3)


def flat_beta(world: World) -> pd.DataFrame:
    flat = world.scores.copy()
    flat.loc[flat["attribute_id"] == "beta", "score"] = 3
    return flat


def log_chars(world: World) -> np.ndarray:
    n_chars = world.table.frame.set_index("text_id").loc[world.gold["text_id"], "n_chars"]
    return np.log(n_chars.to_numpy(dtype=float))


@pytest.mark.parametrize("scale", ["standardized", "raw"])
def test_a_constant_score_reports_a_shortfall_of_minus_one_and_its_covariate_linked_error(
    world: World, scale: str
):
    settings = PanelSettings(n_boot=30, covariate_scale=scale)  # type: ignore[arg-type]
    panel = validity_panel(
        flat_beta(world), world.gold, world.table, targets={"beta": "gold_beta"}, settings=settings
    )
    row = panel.frame.iloc[0]
    assert row["status"] == "constant_score"
    assert row["calibration_shortfall"] == -1.0
    covariate = log_chars(world)
    gold = world.gold["gold_beta"].to_numpy(dtype=float)
    gold01 = (gold - gold.min()) / np.ptp(gold)
    slope = np.cov(gold01, covariate, ddof=1)[0, 1] / np.var(covariate, ddof=1)
    expected = -slope * (covariate.std(ddof=1) if scale == "standardized" else 1.0)
    assert row["covariate_linked_error"] == pytest.approx(expected, rel=1e-12)
    assert row["covariate_linked_error_raw"] == pytest.approx(-slope, rel=1e-12)
    assert row["covariate_linked_error_standardized"] == pytest.approx(
        -slope * covariate.std(ddof=1), rel=1e-12
    )
    for column in (
        "level_displacement",
        "mae_01",
        "signed_gap_01",
        "rank_validity",
        "calibration_shortfall_low",
        "covariate_linked_error_low",
        "covariate_linked_error_high",
    ):
        assert np.isnan(row[column]), column
    assert pd.isna(row["n_boot"]) and row["noise"] == 0.0


def test_a_constant_score_against_a_constant_gold_keeps_every_error_missing(world: World):
    gold = world.gold.assign(flat_gold=2.0)
    panel = validity_panel(
        flat_beta(world), gold, world.table, targets={"beta": "flat_gold"}, settings=FAST
    )
    row = panel.frame.iloc[0]
    assert row["status"] == "constant_score"
    assert np.isnan(row["calibration_shortfall"]) and np.isnan(row["covariate_linked_error"])
    assert np.isnan(row["covariate_linked_error_raw"])
    assert np.isnan(row["covariate_linked_error_standardized"])


def run_averaged(world: World, attribute: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The run-averaged score of an attribute, its gold value and log characters, by text."""
    runs = world.scores[world.scores["attribute_id"] == attribute]
    per_text = runs.assign(score=runs["score"].astype(float)).groupby("text_id")["score"].mean()
    ids = list(per_text.index)
    gold = world.gold.set_index("text_id").loc[ids, f"gold_{attribute}"].to_numpy(dtype=float)
    n_chars = world.table.frame.set_index("text_id").loc[ids, "n_chars"].to_numpy(dtype=float)
    return per_text.to_numpy(), gold, np.log(n_chars)


@pytest.mark.parametrize("scale", ["standardized", "raw"])
def test_both_covariate_scales_are_reported_beside_the_headline(world: World, scale: str):
    settings = PanelSettings(n_boot=30, covariate_scale=scale)  # type: ignore[arg-type]
    panel = validity_panel(
        world.scores, world.gold, world.table, targets={"alpha": "gold_alpha"}, settings=settings
    )
    row = panel.frame.iloc[0]
    score, gold, covariate = run_averaged(world, "alpha")
    raw = score_errors(score, gold, covariate, covariate_scale="raw")
    standardized = score_errors(score, gold, covariate, covariate_scale="standardized")
    assert row["covariate_linked_error_raw"] == pytest.approx(raw.covariate_linked_error, abs=1e-12)
    assert row["covariate_linked_error_standardized"] == pytest.approx(
        standardized.covariate_linked_error, abs=1e-12
    )
    assert row["covariate_linked_error"] == row[f"covariate_linked_error_{scale}"]
    assert row["covariate_linked_error_raw"] != pytest.approx(
        row["covariate_linked_error_standardized"]
    )
