"""What the baseline result records: fold penalties, settings, resolved inputs, fold keys."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from baseline_support import GOLD, World, run

from copernicus.baseline.folds import grouped_kfold, rng_for
from copernicus.baseline.ridge import ALPHAS
from copernicus.baseline.runner import BaselineResult, BaselineSettings

POINT = ["seed", "encoder_set", "variant"]
SETTINGS = BaselineSettings(sizes=(50, "full"), encoder_sets=("alpha", "stacked"))
ONE_FIT = BaselineSettings(sizes=("full",), seeds=(42,), encoder_sets=("alpha",))


@pytest.fixture(scope="module")
def world() -> World:
    return World()


@pytest.fixture(scope="module")
def baseline(world: World) -> BaselineResult:
    return run(world, SETTINGS, coder_score=world.score)


def test_each_fold_carries_one_penalty_and_one_edge_flag(baseline: BaselineResult):
    oof = baseline.oof_predictions
    per_fold = oof.groupby([*POINT, "fold"])[["alpha", "at_grid_edge"]].nunique()
    assert (per_fold == 1).all().all()
    assert oof["at_grid_edge"].dtype == bool
    on_grid = np.isclose(oof["alpha"].to_numpy()[:, None], ALPHAS[None, :], rtol=1e-12).any(axis=1)
    assert on_grid.all()


def test_the_fold_penalties_summarise_to_the_results_row(baseline: BaselineResult):
    folds = baseline.oof_predictions.drop_duplicates([*POINT, "fold"])
    summary = folds.groupby(POINT).agg(
        alpha_median=("alpha", "median"), n_edge_folds=("at_grid_edge", "sum")
    )
    rows = baseline.results
    full = rows[rows["size"] == "full"].set_index(POINT)[["alpha_median", "n_edge_folds"]]
    joined = summary.join(full.add_suffix("_of_results"))
    assert len(joined) == len(full) == 3 * 2 * 2
    assert np.allclose(joined["alpha_median"], joined["alpha_median_of_results"], rtol=1e-12)
    assert (joined["n_edge_folds"] == joined["n_edge_folds_of_results"]).all()


def test_the_result_carries_its_settings_and_the_resolved_inputs(baseline: BaselineResult):
    assert baseline.settings == SETTINGS
    assert (baseline.gold_column, baseline.unit_id, baseline.binary) == (GOLD, "text_id", False)


def test_the_default_unit_and_an_inferred_binary_gold_are_recorded(world: World):
    by_group = [int(group[1:]) % 2 for group in world.groups]
    gold = pd.DataFrame({"text_id": world.ids, GOLD: by_group})
    settings = BaselineSettings(sizes=("full",), seeds=(42,), floor_units=10)
    result = run(world, settings, gold=gold, unit_id=None)
    assert (result.unit_id, result.binary) == ("group_id", True)


def fold_of_each_text(result: BaselineResult) -> dict[str, int]:
    oof = result.oof_predictions
    return dict(zip(oof["text_id"], oof["fold"], strict=True))


def test_the_folds_follow_the_documented_generator_key(world: World):
    folds = grouped_kfold(np.asarray(world.groups), 5, rng_for(42, GOLD, "folds"))
    ids = np.asarray(world.ids)
    expected = {text: index for index, (_, test) in enumerate(folds) for text in ids[test]}
    assert fold_of_each_text(run(world, ONE_FIT)) == expected
    renamed = world.frame.rename(columns={GOLD: "renamed"})
    other = run(world, ONE_FIT, gold=renamed, gold_column="renamed")
    assert fold_of_each_text(other) != expected
