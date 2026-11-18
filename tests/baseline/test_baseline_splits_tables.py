"""The splits a baseline run used, and the tables that supplied splits give."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from baseline_support import GOLD, World, run
from splits_support import (
    assert_same_splits,
    assert_same_tables,
    draw_table,
    fold_labels,
    fold_table,
    folds_from_labels,
    keyed_draws,
    keyed_folds,
    whole_groups_until,
)

from copernicus.baseline import (
    BaselineResult,
    BaselineSettings,
    BaselineSplits,
    read_splits,
    write_splits,
)
from copernicus.baseline.metrics import aggregate_units, fold_metrics, weighted_over_folds
from copernicus.baseline.ridge import RidgeDesign, apply_scaler, fit_scaler, predict

SEEDS = (42, 43)
SIZES: tuple[int | str, ...] = (50, "full")
SETTINGS = BaselineSettings(sizes=SIZES, seeds=SEEDS, encoder_sets=("alpha", "stacked"))
ONE_SEED = BaselineSettings(sizes=SIZES, seeds=(42,), encoder_sets=("alpha",))


@pytest.fixture(scope="module")
def world() -> World:
    return World()


@pytest.fixture(scope="module")
def keyed(world: World) -> BaselineResult:
    return run(world, SETTINGS, coder_score=world.score)


def test_a_run_records_the_folds_and_draws_of_the_keyed_generators(
    world: World, keyed: BaselineResult
) -> None:
    folds = keyed_folds(world, SEEDS)
    expected = BaselineSplits(
        folds=fold_table(world, folds), draws=draw_table(world, keyed_draws(world, folds, SIZES))
    )
    assert_same_splits(keyed.splits, expected)


def test_the_recorded_splits_give_the_run_again_bit_for_bit(
    world: World, keyed: BaselineResult
) -> None:
    again = run(world, SETTINGS, coder_score=world.score, splits=keyed.splits)
    assert_same_tables(keyed, again)
    assert_same_splits(again.splits, keyed.splits)


def fold_of_each_text(result: BaselineResult, seed: int) -> pd.Series:
    oof = result.oof_predictions
    part = oof[(oof["seed"] == seed) & (oof["encoder_set"] == "alpha")]
    part = part[part["variant"] == "embeddings_only"]
    return part.set_index("text_id")["fold"].sort_index()


def test_under_another_name_of_the_gold_column_the_numbers_follow_the_splits(
    world: World, keyed: BaselineResult
) -> None:
    renamed = world.frame.rename(columns={GOLD: "renamed"})
    options = {"gold": renamed, "gold_column": "renamed", "coder_score": world.score}
    other = run(world, SETTINGS, splits=keyed.splits, **options)
    assert_same_tables(keyed, other)
    assert other.gold_column == "renamed"
    drawn_again = run(world, SETTINGS, **options)
    assert not fold_of_each_text(drawn_again, 42).equals(fold_of_each_text(keyed, 42))


def test_the_splits_file_gives_the_run_again_bit_for_bit(
    world: World, keyed: BaselineResult, tmp_path: Path
) -> None:
    path = tmp_path / "splits.json"
    write_splits(keyed.splits, path)
    again = run(world, SETTINGS, coder_score=world.score, splits=read_splits(path))
    assert_same_tables(keyed, again)


def test_folds_alone_take_their_draws_from_the_keyed_generators(world: World) -> None:
    folds = keyed_folds(world, SEEDS, gold_column="another key")
    settings = BaselineSettings(sizes=SIZES, seeds=SEEDS, encoder_sets=("alpha",))
    result = run(world, settings, splits=BaselineSplits(folds=fold_table(world, folds)))
    expected = BaselineSplits(
        folds=fold_table(world, folds), draws=draw_table(world, keyed_draws(world, folds, SIZES))
    )
    assert_same_splits(result.splits, expected)
    oof = result.oof_predictions
    for seed in SEEDS:
        part = oof[(oof["seed"] == seed) & (oof["encoder_set"] == "alpha")].set_index("text_id")
        labels = fold_labels(folds[seed], len(world.ids))
        assert (part.loc[world.ids, "fold"].to_numpy() == labels).all()


def test_relabelled_folds_move_the_fold_column_and_keep_every_prediction(
    world: World, keyed: BaselineResult
) -> None:
    folds = keyed.splits.folds.assign(fold=(keyed.splits.folds["fold"] + 1) % 5)
    assert keyed.splits.draws is not None
    draws = keyed.splits.draws.assign(fold=(keyed.splits.draws["fold"] + 1) % 5)
    moved = run(
        world, SETTINGS, coder_score=world.score, splits=BaselineSplits(folds=folds, draws=draws)
    )
    keys = ["encoder_set", "variant", "seed", "text_id"]
    mine = keyed.oof_predictions.set_index(keys).sort_index()
    other = moved.oof_predictions.set_index(keys).sort_index()
    assert np.array_equal(mine["prediction"].to_numpy(), other["prediction"].to_numpy())
    assert np.array_equal(mine["alpha"].to_numpy(), other["alpha"].to_numpy())
    assert ((mine["fold"] + 1) % 5 == other["fold"]).all()
    for column in ("spearman", "pearson", "mae_01", "draw_units_mean", "alpha_median"):
        assert np.allclose(keyed.results[column], moved.results[column], rtol=1e-12, atol=0)


def hand_fit(world: World, train: np.ndarray, test: np.ndarray) -> np.ndarray:
    """The ridge of encoder alpha fitted on ``train`` and applied to ``test``, step by step."""
    vectors = np.asarray(world.embeddings["alpha"].vectors, dtype=np.float64)
    scaler = fit_scaler([vectors[train]])
    design = RidgeDesign(
        apply_scaler([vectors[train]], scaler), world.gold[train], np.asarray(world.groups)[train]
    )
    fit = design.fit(np.asarray(ONE_SEED.alphas))
    return predict(fit, apply_scaler([vectors[test]], scaler))


def test_a_text_is_predicted_by_the_model_of_the_fold_the_splits_put_it_in(world: World) -> None:
    settings = BaselineSettings(sizes=("full",), seeds=(42,), encoder_sets=("alpha",))
    labels = fold_labels(keyed_folds(world, (42,))[42], len(world.ids))
    moved = labels.copy()
    moved[:4] = (labels[0] + 2) % 5  # group g0, the texts t0 to t3, goes to another fold
    splits = BaselineSplits(folds=fold_table(world, {42: folds_from_labels(moved, 5)}))
    oof = run(world, settings, splits=splits).oof_predictions.set_index("text_id")
    assert oof.loc["t0", "fold"] == moved[0]
    train, test = folds_from_labels(moved, 5)[int(moved[0])]
    by_hand = hand_fit(world, train, test)
    assert oof.loc["t0", "prediction"] == by_hand[int(np.flatnonzero(test == 0)[0])]
    keyed_oof = run(world, settings).oof_predictions.set_index("text_id")
    assert keyed_oof.loc["t0", "prediction"] != oof.loc["t0", "prediction"]


def test_a_draw_of_other_groups_gives_its_receipts_and_the_metrics_of_a_fit_by_hand(
    world: World,
) -> None:
    settings = BaselineSettings(sizes=(50,), seeds=(42,), encoder_sets=("alpha",))
    folds = keyed_folds(world, (42,))[42]
    order = sorted(set(world.groups), reverse=True)
    drawn = [
        whole_groups_until(world, train, 56 if index == 0 else 50, order)
        for index, (train, _) in enumerate(folds)
    ]
    splits = BaselineSplits(
        folds=fold_table(world, {42: folds}), draws=draw_table(world, {(42, "50"): drawn})
    )
    row = run(world, settings, splits=splits).results.iloc[0]
    assert (row["draw_units_min"], row["draw_units_max"], row["draw_overshoot_max"]) == (52, 56, 6)
    assert row["draw_groups_mean"] == (14 + 4 * 13) / 5 and row["pool_units_mean"] == 192
    per_fold = []
    for (_, test), draw in zip(folds, drawn, strict=True):
        prediction = hand_fit(world, draw, test)
        ids = np.asarray(world.ids)
        _, unit_pred, unit_gold = aggregate_units(prediction, world.gold[test], ids[test])
        per_fold.append(fold_metrics(unit_pred, unit_gold, None, world.gold[draw]))
    expected = weighted_over_folds(per_fold)
    for key in ("spearman", "spearman_abs", "pearson", "mae_01", "mae_native"):
        assert row[key] == expected[key], key
    keyed_row = run(world, settings).results.iloc[0]
    assert keyed_row["spearman"] != row["spearman"]


# --- skips ----------------------------------------------------------------------------


def test_a_size_above_half_the_units_is_skipped_without_a_supplied_draw(world: World) -> None:
    settings = BaselineSettings(sizes=(50, 200, "full"), seeds=(42,), encoder_sets=("alpha",))
    folds = keyed_folds(world, (42,))
    splits = BaselineSplits(
        folds=fold_table(world, folds), draws=draw_table(world, keyed_draws(world, folds, SIZES))
    )
    result = run(world, settings, splits=splits)
    assert list(result.skipped["reason"]) == ["quota_exceeds_half_units(quota=200,units=240)"]
    assert set(result.results["size"]) == {"50", "full"}
    assert result.splits.draws is not None and set(result.splits.draws["size"]) == {"50", "full"}


def test_a_pool_below_the_quota_is_skipped_without_a_supplied_draw() -> None:
    skewed = World(group_sizes=[120] + [8] * 10)
    settings = BaselineSettings(sizes=(100, "full"), seeds=(42,), encoder_sets=("alpha",))
    folds = keyed_folds(skewed, (42,))
    splits = BaselineSplits(
        folds=fold_table(skewed, folds),
        draws=draw_table(skewed, keyed_draws(skewed, folds, ("full",))),
    )
    result = run(skewed, settings, splits=splits)
    assert list(result.skipped["size"]) == ["100"]
    assert str(result.skipped["reason"].iloc[0]).startswith("pool_below_quota:min_pool_units=")
    assert set(result.results["size"]) == {"full"}


def test_a_supplied_draw_below_the_floor_is_a_recorded_skip(world: World) -> None:
    settings = BaselineSettings(sizes=(50,), seeds=(42,), floor_units=53, encoder_sets=("alpha",))
    folds = keyed_folds(world, (42,))[42]
    order = sorted(set(world.groups))
    drawn = [whole_groups_until(world, train, 56, order) for train, _ in folds]
    drawn[3] = whole_groups_until(world, folds[3][0], 50, order)
    splits = BaselineSplits(
        folds=fold_table(world, {42: folds}), draws=draw_table(world, {(42, "50"): drawn})
    )
    result = run(world, settings, splits=splits)
    assert result.results.empty
    assert list(result.skipped["reason"]) == ["below_floor(53):min_draw_units=52"]
    assert_same_splits(result.splits, splits)
    assert_same_tables(result, run(world, settings, splits=result.splits))


def test_a_full_draw_may_be_left_out_and_is_recorded_as_the_whole_pool(world: World) -> None:
    folds = keyed_folds(world, (42,))
    splits = BaselineSplits(
        folds=fold_table(world, folds), draws=draw_table(world, keyed_draws(world, folds, (50,)))
    )
    result = run(world, ONE_SEED, splits=splits)
    assert set(result.results["size"]) == {"50", "full"}
    expected = draw_table(world, keyed_draws(world, folds, SIZES))
    assert result.splits.draws is not None
    pd.testing.assert_frame_equal(
        result.splits.draws, BaselineSplits(folds=splits.folds, draws=expected).draws
    )


def test_supplied_draws_of_a_skipped_size_are_not_used_and_a_warning_says_so(
    world: World, caplog: pytest.LogCaptureFixture
) -> None:
    settings = BaselineSettings(sizes=(50, 200, "full"), seeds=(42,), encoder_sets=("alpha",))
    folds = keyed_folds(world, (42,))
    draws = keyed_draws(world, folds, SIZES)
    draws[(42, "200")] = draws[(42, "full")]
    splits = BaselineSplits(folds=fold_table(world, folds), draws=draw_table(world, draws))
    with caplog.at_level(logging.WARNING, logger="copernicus.baseline"):
        result = run(world, settings, splits=splits)
    warnings = [
        record
        for record in caplog.records
        if record.levelno == logging.WARNING and record.name.startswith("copernicus.baseline")
    ]
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    assert message.startswith("seed 42 size 200: the supplied draws are not used")
    assert "quota_exceeds_half_units(quota=200,units=240)" in message
    assert list(result.skipped["size"]) == ["200"]
