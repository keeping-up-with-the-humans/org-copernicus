"""NLP-ML baseline runner: loop, alignment, refusals, curves."""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import pandas as pd
import pytest
from baseline_support import DIM, GOLD, N_GROUPS, PER_GROUP, World, make_set, run
from pandas.api.types import is_string_dtype

from copernicus.baseline.metrics import METRIC_KEYS
from copernicus.baseline.runner import (
    BaselineResult,
    BaselineSettings,
    curves,
    run_ridge_baseline,
)
from copernicus.data.table import TextTable
from copernicus.errors import EstimatorError, SettingsError


@pytest.fixture(scope="module")
def world() -> World:
    return World()


@pytest.fixture(scope="module")
def baseline(world: World) -> BaselineResult:
    settings = BaselineSettings(sizes=(50, "full"), encoder_sets=("alpha", "stacked"))
    return run(world, settings, coder_score=world.score)


def metric(result: BaselineResult, encoder_set: str, variant: str, size: Any) -> float:
    rows = result.results
    mask = (
        (rows["encoder_set"] == encoder_set)
        & (rows["variant"] == variant)
        & (rows["size"].astype(str) == str(size))
    )
    return float(rows.loc[mask, "spearman_abs"].mean())


# --- settings -------------------------------------------------------------------------


def test_settings_defaults_match_the_documented_grid():
    settings = BaselineSettings()
    assert settings.k_folds == 5
    assert settings.seeds == (42, 43, 44)
    assert settings.sizes == (50, 100, 150, 200, 250, 500, 1000, "full")
    assert settings.floor_units == 50
    assert len(settings.alphas) == 41 and settings.alphas[0] == pytest.approx(0.01)
    assert settings.encoder_sets == ("single",) and settings.binary is None
    with pytest.raises(dataclasses.FrozenInstanceError):
        settings.k_folds = 3  # type: ignore[misc]


def test_settings_accept_numpy_integers():
    settings = BaselineSettings(
        k_folds=np.int64(3),  # type: ignore[arg-type]
        seeds=(np.int64(1), np.int32(2)),
        sizes=(np.int64(50), "full"),
        floor_units=np.int64(10),  # type: ignore[arg-type]
    )
    assert settings.k_folds == 3


@pytest.mark.parametrize(
    "overrides",
    [
        {"k_folds": 1},
        {"seeds": ()},
        {"seeds": (42, 42)},
        {"seeds": (4.5,)},
        {"sizes": (50, 50)},
        {"sizes": ()},
        {"sizes": (50, 0)},
        {"sizes": (50, "everything")},
        {"sizes": (True,)},
        {"floor_units": 0},
        {"alphas": ()},
        {"alphas": (1.0, -1.0)},
        {"alphas": (1.0, float("nan"))},
        {"encoder_sets": ()},
        {"binary": "yes"},
    ],
)
def test_settings_refuse_values_outside_their_range(overrides: dict[str, Any]):
    with pytest.raises(SettingsError):
        BaselineSettings(**overrides)


# --- the loop -------------------------------------------------------------------------


def test_full_size_recovers_the_planted_signal_and_beats_the_small_draw():
    noisy = World(noise=2.5)
    settings = BaselineSettings(sizes=(50, "full"), encoder_sets=("alpha",))
    result = run(noisy, settings)
    full = metric(result, "alpha", "embeddings_only", "full")
    small = metric(result, "alpha", "embeddings_only", 50)
    assert full >= 0.8
    assert full > small + 0.03


def test_results_layout(baseline: BaselineResult):
    rows = baseline.results
    leading = ["encoder_set", "variant", "seed", "size", "size_units", "n_rows", "n_units"]
    assert list(rows.columns[:9]) == [*leading, "n_groups", "n_features"]
    for key in METRIC_KEYS:
        assert key in rows.columns
    for name in (
        "alpha_median",
        "n_edge_folds",
        "n_folds",
        "draw_units_min",
        "draw_units_mean",
        "draw_units_max",
        "draw_rows_mean",
        "draw_groups_mean",
        "draw_overshoot_max",
        "pool_units_mean",
    ):
        assert name in rows.columns
    assert is_string_dtype(rows["size"]) and set(rows["size"]) == {"50", "full"}
    assert rows["size_units"].dtype == "Int64"
    assert set(rows.loc[rows["size"] == "50", "size_units"]) == {50}
    assert rows.loc[rows["size"] == "full", "size_units"].isna().all()
    assert len(rows) == 2 * 2 * 3 * 2
    assert (rows["n_folds"] == 5).all()
    assert (rows["n_rows"] == 240).all() and (rows["n_units"] == 240).all()
    assert (rows["n_groups"] == 60).all()
    full = rows[rows["size"] == "full"]
    assert (full["draw_units_mean"] == full["pool_units_mean"]).all()
    small = rows[rows["size"] == "50"]
    assert (small["draw_units_min"] >= 50).all()
    assert (small["draw_overshoot_max"] >= 0).all()
    assert (rows["alpha_median"] > 0).all()


def test_stacked_set_concatenates_both_blocks(baseline: BaselineResult):
    rows = baseline.results
    assert set(rows["encoder_set"]) == {"alpha", "stacked"}
    features = rows.groupby(["encoder_set", "variant"])["n_features"].first()
    assert features[("alpha", "embeddings_only")] == DIM
    assert features[("alpha", "embeddings_plus_score")] == DIM + 1
    assert features[("stacked", "embeddings_only")] == 2 * DIM
    assert features[("stacked", "embeddings_plus_score")] == 2 * DIM + 1


def test_plus_score_variant_is_not_worse_than_embeddings_only(baseline: BaselineResult):
    plus = metric(baseline, "alpha", "embeddings_plus_score", "full")
    only = metric(baseline, "alpha", "embeddings_only", "full")
    assert plus >= only - 0.01


def test_variants_are_absent_without_a_coder_score(world: World):
    result = run(world)
    assert set(result.results["variant"]) == {"embeddings_only"}


def test_single_expands_to_one_set_per_key_named_by_the_key(world: World):
    result = run(world, BaselineSettings(sizes=("full",), seeds=(42,)))
    assert list(pd.unique(result.results["encoder_set"])) == ["alpha", "beta"]
    informative = metric(result, "alpha", "embeddings_only", "full")
    noise_only = metric(result, "beta", "embeddings_only", "full")
    assert informative > 0.8 > noise_only


def test_quota_above_half_the_units_is_recorded_as_skipped(world: World):
    settings = BaselineSettings(sizes=(50, 200, "full"), seeds=(42, 43), encoder_sets=("alpha",))
    result = run(world, settings)
    skipped = result.skipped
    assert list(skipped.columns) == ["seed", "size", "size_units", "reason"]
    assert sorted(skipped["seed"]) == [42, 43]
    assert set(skipped["size"]) == {"200"} and set(skipped["size_units"]) == {200}
    assert skipped["size_units"].dtype == "Int64"
    assert set(skipped["reason"]) == {"quota_exceeds_half_units(quota=200,units=240)"}
    assert "200" not in set(result.results["size"])


def test_a_draw_below_the_floor_is_a_recorded_skip_not_a_smaller_run(world: World):
    settings = BaselineSettings(sizes=(50,), seeds=(42,), floor_units=60, encoder_sets=("alpha",))
    result = run(world, settings)
    assert result.results.empty
    assert len(result.skipped) == 1
    reason = str(result.skipped["reason"].iloc[0])
    assert reason.startswith("below_floor(60):min_draw_units=")
    low = int(reason.rsplit("=", 1)[1])
    assert 50 <= low < 60


def test_a_pool_smaller_than_the_quota_is_a_recorded_skip_not_a_smaller_run():
    skewed = World(group_sizes=[120] + [8] * 10)
    settings = BaselineSettings(
        sizes=(100, "full"), seeds=(42,), floor_units=50, encoder_sets=("alpha",)
    )
    result = run(skewed, settings)
    assert list(result.skipped["size"]) == ["100"]
    reason = str(result.skipped["reason"].iloc[0])
    assert reason.startswith("pool_below_quota:min_pool_units=")
    assert int(reason.rsplit("=", 1)[1]) < 100
    assert set(result.results["size"]) == {"full"}


def test_default_unit_is_the_group(world: World):
    settings = BaselineSettings(
        sizes=(10, "full"), seeds=(42,), floor_units=1, encoder_sets=("alpha",)
    )
    result = run_ridge_baseline(
        world.embeddings, world.frame, world.table, settings, gold_column=GOLD
    )
    assert set(result.results["size"]) == {"10", "full"}
    assert (result.results["n_units"] == N_GROUPS).all()
    assert (result.results["n_rows"] == 240).all()
    full = result.results[result.results["size"] == "full"]
    assert (full["pool_units_mean"] == 48).all()


def test_full_pool_below_the_floor_is_a_recorded_skip(world: World):
    settings = BaselineSettings(sizes=("full",), seeds=(42,), encoder_sets=("alpha",))
    result = run_ridge_baseline(
        world.embeddings, world.frame, world.table, settings, gold_column=GOLD
    )
    assert result.results.empty
    assert list(result.skipped["reason"]) == ["below_floor(50):min_draw_units=48"]


def test_unit_id_can_name_another_table_column(world: World):
    frame = world.table.frame.assign(pair=[f"p{i // 2}" for i in range(240)])
    table = TextTable.from_frame(frame)
    settings = BaselineSettings(sizes=("full",), seeds=(42,), encoder_sets=("alpha",))
    result = run_ridge_baseline(
        world.embeddings, world.frame, table, settings, gold_column=GOLD, unit_id="pair"
    )
    assert (result.results["n_units"] == 120).all()
    assert (result.results["draw_units_mean"] == result.results["pool_units_mean"]).all()


def test_oof_predictions_cover_every_text_once_per_seed_set_and_variant(
    world: World, baseline: BaselineResult
):
    oof = baseline.oof_predictions
    assert list(oof.columns) == [
        "text_id",
        "seed",
        "size",
        "size_units",
        "encoder_set",
        "variant",
        "fold",
        "prediction",
        "gold",
        "alpha",
        "at_grid_edge",
    ]
    assert set(oof["size"]) == {"full"} and oof["size_units"].isna().all()
    assert oof["size_units"].dtype == "Int64"
    counts = oof.groupby(["seed", "encoder_set", "variant"])["text_id"].agg(["size", "nunique"])
    assert len(counts) == 3 * 2 * 2
    assert (counts["size"] == 240).all() and (counts["nunique"] == 240).all()
    assert set(oof["text_id"]) == set(world.ids)
    assert np.isfinite(oof["prediction"]).all()
    one = oof[(oof["seed"] == 42) & (oof["encoder_set"] == "alpha")].drop_duplicates("text_id")
    expected = pd.Series(world.gold, index=world.ids)
    assert np.allclose(one["gold"].to_numpy(), expected[one["text_id"]].to_numpy())


def test_groups_never_straddle_a_fold_in_the_oof_predictions(
    world: World, baseline: BaselineResult
):
    oof = baseline.oof_predictions
    group_of = pd.Series(world.groups, index=world.ids)
    one = oof[(oof["seed"] == 42) & (oof["encoder_set"] == "alpha")].drop_duplicates("text_id")
    folds_per_group = one.groupby(group_of[one["text_id"]].to_numpy())["fold"].nunique()
    assert (folds_per_group == 1).all()


def test_reproducible(world: World):
    settings = BaselineSettings(sizes=(50, "full"), seeds=(42, 43), encoder_sets=("alpha",))
    first = run(world, settings, coder_score=world.score)
    second = run(world, settings, coder_score=world.score)
    pd.testing.assert_frame_equal(first.results, second.results)
    pd.testing.assert_frame_equal(first.oof_predictions, second.oof_predictions)


def test_binary_gold_reports_auc_when_inferred(world: World):
    binary_frame = pd.DataFrame(
        {"text_id": world.ids, GOLD: (world.gold > np.median(world.gold)).astype(int)}
    )
    settings = BaselineSettings(sizes=("full",), seeds=(42,), encoder_sets=("alpha",))
    result = run(world, settings, gold=binary_frame)
    assert (result.results["auc"] > 0.8).all()
    continuous = run(world, settings)
    assert continuous.results["auc"].isna().all()
    forced = run(world, dataclasses.replace(settings, binary=False), gold=binary_frame)
    assert forced.results["auc"].isna().all()


def test_boolean_gold_counts_as_numeric(world: World):
    frame = pd.DataFrame({"text_id": world.ids, GOLD: world.gold > np.median(world.gold)})
    settings = BaselineSettings(sizes=("full",), seeds=(42,), encoder_sets=("alpha",))
    assert run(world, settings, gold=frame).results["auc"].notna().all()


def test_constant_coder_score_is_an_estimator_error(world: World):
    settings = BaselineSettings(sizes=("full",), seeds=(42,), encoder_sets=("alpha",))
    constant = pd.Series(0.5, index=world.ids)
    with pytest.raises(EstimatorError, match="constant coder score"):
        run(world, settings, coder_score=constant)


def test_gold_rows_define_the_analysis_set(world: World):
    subset = world.frame.iloc[:200]
    settings = BaselineSettings(sizes=("full",), seeds=(42,), encoder_sets=("alpha",))
    result = run(world, settings, gold=subset)
    assert (result.results["n_rows"] == 200).all()
    assert set(result.oof_predictions["text_id"]) == set(world.ids[:200])


def test_ids_are_compared_as_strings():
    numeric = World()
    numeric_ids = [str(i) for i in range(240)]
    table = TextTable.from_frame(
        pd.DataFrame(
            {
                "text_id": list(range(240)),
                "text": [f"text number {i}" for i in range(240)],
                "group_id": [i // PER_GROUP for i in range(240)],
            }
        )
    )
    embeddings = {
        "alpha": make_set(numeric_ids, numeric.embeddings["alpha"].vectors.astype(float), "alpha")
    }
    gold = pd.DataFrame({"text_id": list(range(240)), GOLD: numeric.gold})
    score = pd.Series(numeric.gold + 0.1, index=list(range(240)))
    settings = BaselineSettings(sizes=("full",), seeds=(42,), encoder_sets=("alpha",))
    result = run_ridge_baseline(
        embeddings, gold, table, settings, gold_column=GOLD, unit_id="text_id", coder_score=score
    )
    assert set(result.oof_predictions["text_id"]) == set(numeric_ids)


# --- curves ---------------------------------------------------------------------------


def test_curves_average_over_seeds_in_settings_order(baseline: BaselineResult):
    table = curves(baseline)
    assert list(table.columns) == [
        "encoder_set",
        "variant",
        "size",
        "size_units",
        "rank_validity",
        "rank_validity_sd",
        "spearman",
        "n_seeds",
    ]
    assert list(zip(table["encoder_set"], table["variant"], table["size"], strict=True)) == [
        (name, variant, size)
        for name in ("alpha", "stacked")
        for variant in ("embeddings_only", "embeddings_plus_score")
        for size in ("50", "full")
    ]
    assert table["size_units"].dtype == "Int64"
    assert table["size_units"].iloc[0] == 50 and pd.isna(table["size_units"].iloc[1])
    assert (table["n_seeds"] == 3).all()
    first = table.iloc[0]
    rows = baseline.results
    own = rows[
        (rows["encoder_set"] == "alpha")
        & (rows["variant"] == "embeddings_only")
        & (rows["size"] == "50")
    ]
    assert first["rank_validity"] == pytest.approx(own["spearman_abs"].mean())
    assert first["rank_validity_sd"] == pytest.approx(own["spearman_abs"].std(ddof=1))
    assert first["spearman"] == pytest.approx(own["spearman"].mean())


def test_curves_omit_skipped_sizes_and_give_nan_sd_for_one_seed(world: World):
    settings = BaselineSettings(sizes=(50, 200, "full"), seeds=(42,), encoder_sets=("alpha",))
    table = curves(run(world, settings))
    assert list(table["size"]) == ["50", "full"]
    assert (table["n_seeds"] == 1).all()
    assert table["rank_validity_sd"].isna().all()
