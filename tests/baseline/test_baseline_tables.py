"""What the baseline result records: fold penalties, settings, resolved inputs, fold keys."""

from __future__ import annotations

import dataclasses

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


# --- seeds of two integer types in one run ----------------------------------------------


@pytest.mark.parametrize(
    "seeds",
    [(42, 2**63), (2**63, 42), (-1, 2**63), (42, 2**64 + 5)],
    ids=["signed_then_unsigned", "unsigned_then_signed", "negative_and_unsigned", "beyond_64_bits"],
)
def test_seeds_of_two_integer_types_keep_their_values_in_every_table(
    world: World, seeds: tuple[int, int]
) -> None:
    # The out-of-fold predictions of each seed arrive as their own frames; joined as they
    # were, a signed and an unsigned seed column became floats, and the run failed.
    settings = BaselineSettings(sizes=("full",), seeds=seeds, encoder_sets=("alpha",))
    both = run(world, settings)
    for table in (both.results, both.oof_predictions):
        assert [int(seed) for seed in table["seed"].drop_duplicates()] == list(seeds)
        assert all(float(seed).is_integer() for seed in table["seed"])
    for seed in seeds:
        alone = run(world, dataclasses.replace(settings, seeds=(seed,)))
        mine = both.oof_predictions[[int(value) == seed for value in both.oof_predictions["seed"]]]
        assert mine["prediction"].tolist() == alone.oof_predictions["prediction"].tolist()
        assert mine["text_id"].tolist() == alone.oof_predictions["text_id"].tolist()


def test_ordinary_seeds_keep_the_integer_type_of_the_out_of_fold_table(world: World) -> None:
    settings = BaselineSettings(sizes=("full",), seeds=(0, 1), encoder_sets=("alpha",))
    assert str(run(world, settings).oof_predictions["seed"].dtype) == "int64"


@pytest.mark.parametrize(
    ("seeds", "kind"),
    [
        ((np.int32(42), np.int32(43)), "int32"),
        ((np.uint32(42), np.uint32(43)), "uint32"),
        ((np.int8(-1), np.uint8(255)), "int16"),
    ],
    ids=["int32", "uint32", "int8_and_uint8"],
)
def test_seed_columns_that_join_exactly_keep_the_type_they_join_to(
    world: World, seeds: tuple[np.integer, np.integer], kind: str
) -> None:
    # Seed columns of one integer type, or of two that a wider integer type holds, join
    # without loss as they are: the out-of-fold table keeps the type of that join.
    settings = BaselineSettings(sizes=("full",), seeds=seeds, encoder_sets=("alpha",))
    column = run(world, settings).oof_predictions["seed"]
    assert str(column.dtype) == kind
    assert [int(seed) for seed in column.drop_duplicates()] == [int(seed) for seed in seeds]
