"""A baseline run given its own splits again: floor skips, empty tables, large seeds, ids."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from baseline_support import GOLD, World, make_set, run
from splits_support import (
    assert_same_splits,
    assert_same_tables,
    draw_table,
    fold_labels,
    fold_table,
    keyed_draws,
    keyed_folds,
)

from copernicus.baseline import BaselineSettings, BaselineSplits, read_splits, write_splits
from copernicus.baseline.folds import grouped_kfold, rng_for
from copernicus.data.table import TextTable
from copernicus.errors import ConfigurationError

FLOOR = BaselineSettings(
    sizes=(50, 60, "full"), seeds=(42, 43), floor_units=56, encoder_sets=("alpha",)
)


@pytest.fixture(scope="module")
def world() -> World:
    return World()


def through_the_file(splits: BaselineSplits, folder: Path) -> BaselineSplits:
    path = folder / "splits.json"
    write_splits(splits, path)
    return read_splits(path)


# --- draws that a floor skip consulted --------------------------------------------------


def test_a_run_with_fitted_and_floor_skipped_sizes_replays_bit_for_bit(
    world: World, tmp_path: Path
) -> None:
    first = run(world, FLOOR, coder_score=world.score)
    assert list(first.skipped["reason"]) == ["below_floor(56):min_draw_units=52"] * 2
    assert set(first.results["size"]) == {"60", "full"}
    consulted = keyed_draws(world, keyed_folds(world, (42, 43)), (50, 60, "full"))
    assert first.splits.draws is not None
    expected = BaselineSplits(folds=first.splits.folds, draws=draw_table(world, consulted))
    assert_same_splits(first.splits, expected)
    options = {"coder_score": world.score}
    assert_same_tables(first, run(world, FLOOR, splits=first.splits, **options))
    from_file = through_the_file(first.splits, tmp_path)
    assert_same_tables(first, run(world, FLOOR, splits=from_file, **options))
    renamed = world.frame.rename(columns={GOLD: "renamed"})
    other = run(world, FLOOR, splits=first.splits, gold=renamed, gold_column="renamed", **options)
    assert_same_tables(first, other)


def test_a_run_whose_every_count_size_is_skipped_for_the_floor_replays(
    world: World, tmp_path: Path
) -> None:
    settings = BaselineSettings(
        sizes=(50, 100), seeds=(42,), floor_units=110, encoder_sets=("alpha",)
    )
    first = run(world, settings)
    assert first.results.empty
    assert list(first.skipped["reason"]) == [
        "below_floor(110):min_draw_units=52",
        "below_floor(110):min_draw_units=100",
    ]
    assert first.splits.draws is not None and set(first.splits.draws["size"]) == {"50", "100"}
    assert_same_tables(first, run(world, settings, splits=first.splits))
    assert_same_tables(first, run(world, settings, splits=through_the_file(first.splits, tmp_path)))


def test_without_a_draw_a_count_size_is_skipped_when_its_pools_lie_below_the_floor(
    world: World,
) -> None:
    settings = BaselineSettings(
        sizes=(50, "full"), seeds=(42,), floor_units=200, encoder_sets=("alpha",)
    )
    folds = keyed_folds(world, (42,))
    no_draws = draw_table(world, keyed_draws(world, folds, ("full",))).iloc[:0]
    splits = BaselineSplits(folds=fold_table(world, folds), draws=no_draws)
    result = run(world, settings, splits=splits)
    assert list(result.skipped["reason"]) == [
        "below_floor(200):min_pool_units=192",
        "below_floor(200):min_draw_units=192",
    ]
    assert result.splits.draws is not None and set(result.splits.draws["size"]) == {"full"}
    assert_same_tables(result, run(world, settings, splits=result.splits))
    drawn = run(world, settings)
    assert list(drawn.skipped["reason"]) == [
        "below_floor(200):min_draw_units=52",
        "below_floor(200):min_draw_units=192",
    ]
    lower = dataclasses.replace(settings, floor_units=150)
    with pytest.raises(ConfigurationError, match="seed 42 size 50 fold 0: the splits hold no"):
        run(world, lower, splits=splits)


# --- empty tables -----------------------------------------------------------------------


def test_an_all_skipped_run_replays_through_the_file_with_its_empty_draws(
    world: World, tmp_path: Path
) -> None:
    settings = BaselineSettings(sizes=(500,), seeds=(42,), encoder_sets=("alpha",))
    first = run(world, settings)
    assert first.splits.draws is not None and first.splits.draws.empty
    again = through_the_file(first.splits, tmp_path)
    assert again.draws is not None and again.draws.empty
    assert_same_splits(again, first.splits)
    assert_same_tables(first, run(world, settings, splits=again))


# --- seeds at and beyond the ends of 64 bits --------------------------------------------


@pytest.mark.parametrize("seed", [2**63 - 1, -(2**63), 2**63, -(2**63) - 1, 2**64 + 5])
def test_a_seed_at_or_beyond_the_ends_of_64_bits_runs_and_replays(
    world: World, tmp_path: Path, seed: int
) -> None:
    settings = BaselineSettings(sizes=(50, "full"), seeds=(seed,), encoder_sets=("alpha",))
    result = run(world, settings)
    folds = grouped_kfold(np.asarray(world.groups), 5, rng_for(seed, GOLD, "folds"))
    assert result.splits.folds["seed"].tolist() == [seed] * len(world.ids)
    assert result.splits.folds["fold"].tolist() == fold_labels(folds, len(world.ids)).tolist()
    assert result.splits.draws is not None and set(result.splits.draws["seed"]) == {seed}
    assert_same_tables(result, run(world, settings, splits=result.splits))
    again = through_the_file(result.splits, tmp_path)
    content = json.loads((tmp_path / "splits.json").read_text(encoding="utf-8"))
    assert set(content["splits"]["folds"]["seed"]) == {seed}
    assert_same_splits(again, result.splits)
    assert_same_tables(result, run(world, settings, splits=again))


# --- text ids given as unsigned 64-bit integers -----------------------------------------


def unsigned_world(world: World) -> dict[str, object]:
    """The world's inputs with the text ids 2**63 to 2**63 + 239, as unsigned integers."""
    ids = np.arange(len(world.ids), dtype=np.uint64) + np.uint64(2**63)
    frame = world.table.frame[["text", "group_id"]].assign(text_id=ids)
    vectors = {name: np.asarray(item.vectors) for name, item in world.embeddings.items()}
    return {
        "embeddings": {name: make_set(ids, block, name) for name, block in vectors.items()},
        "gold": pd.DataFrame({"text_id": ids, GOLD: world.gold}),
        "table": TextTable.from_frame(frame),
    }


def as_unsigned(table: pd.DataFrame) -> pd.DataFrame:
    ids = np.asarray([int(text) for text in table["text_id"]], dtype=np.uint64)
    return table.assign(text_id=ids)


def test_unsigned_text_ids_take_their_decimal_strings_in_folds_draws_and_the_file(
    world: World, tmp_path: Path
) -> None:
    settings = BaselineSettings(sizes=(50, "full"), seeds=(42,), encoder_sets=("alpha",))
    inputs = unsigned_world(world)
    first = run(world, settings, **inputs)
    assert first.splits.folds["text_id"].iloc[0] == str(2**63)
    assert first.splits.draws is not None
    given = BaselineSplits(
        folds=as_unsigned(first.splits.folds), draws=as_unsigned(first.splits.draws)
    )
    assert_same_splits(given, first.splits)
    assert_same_tables(first, run(world, settings, splits=given, **inputs))
    assert_same_splits(through_the_file(given, tmp_path), first.splits)
