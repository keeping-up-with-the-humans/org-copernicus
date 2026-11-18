"""The baseline runner refuses supplied splits that would break a guarantee of the baseline."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from baseline_support import World, run
from splits_support import (
    draw_table,
    fold_table,
    keyed_draws,
    keyed_folds,
    whole_groups_until,
)

from copernicus.baseline import BaselineSettings, BaselineSplits, write_splits
from copernicus.errors import ConfigurationError

SEEDS = (42, 43)
SIZES: tuple[int | str, ...] = (50, "full")
SETTINGS = BaselineSettings(sizes=SIZES, seeds=SEEDS, encoder_sets=("alpha",))


@pytest.fixture(scope="module")
def world() -> World:
    return World()


@pytest.fixture(scope="module")
def tables(world: World) -> tuple[pd.DataFrame, pd.DataFrame]:
    folds = keyed_folds(world, SEEDS)
    return fold_table(world, folds), draw_table(world, keyed_draws(world, folds, SIZES))


def refusal(
    world: World,
    folds: pd.DataFrame,
    draws: pd.DataFrame | None = None,
    settings: BaselineSettings = SETTINGS,
    **kwargs: object,
) -> str:
    with pytest.raises(ConfigurationError) as error:
        run(world, settings, splits=BaselineSplits(folds=folds, draws=draws), **kwargs)
    return str(error.value)


def relabelled(folds: pd.DataFrame, seed: int, texts: list[str], fold: int) -> pd.DataFrame:
    chosen = (folds["seed"] == seed) & folds["text_id"].isin(texts)
    return folds.assign(fold=np.where(chosen, fold, folds["fold"]))


# --- the folds ------------------------------------------------------------------------


def test_seeds_that_are_not_the_seeds_of_the_settings_are_named(
    world: World, tables: tuple[pd.DataFrame, pd.DataFrame]
) -> None:
    folds, _ = tables
    message = refusal(world, folds[folds["seed"] == 42])
    assert "seeds of the splits' folds (42) are not the seeds of the settings (42, 43)" in message
    assert "missing: 43" in message
    extra = pd.concat([folds, folds[folds["seed"] == 43].assign(seed=99)], ignore_index=True)
    assert "not asked for: 99" in refusal(world, extra)


@pytest.mark.parametrize("label", [5, -1])
def test_a_fold_label_outside_the_folds_is_named_with_its_seed_and_texts(
    world: World, tables: tuple[pd.DataFrame, pd.DataFrame], label: int
) -> None:
    folds, _ = tables
    message = refusal(world, relabelled(folds, 43, ["t0", "t1"], label))
    assert message.startswith("seed 43: fold labels must lie from 0 to 4 (k_folds 5)")
    assert f"t0 (fold {label}), t1 (fold {label})" in message


def test_a_fold_without_texts_is_named_with_its_seed(
    world: World, tables: tuple[pd.DataFrame, pd.DataFrame]
) -> None:
    folds, _ = tables
    emptied = folds.assign(
        fold=np.where((folds["seed"] == 42) & (folds["fold"] == 3), 2, folds["fold"])
    )
    assert refusal(world, emptied).startswith("seed 42: fold 3 holds no text")


def test_analysis_texts_missing_from_the_folds_are_named_ten_at_most(
    world: World, tables: tuple[pd.DataFrame, pd.DataFrame]
) -> None:
    folds, _ = tables
    dropped = [f"t{index}" for index in range(12)]
    message = refusal(world, folds[~((folds["seed"] == 43) & folds["text_id"].isin(dropped))])
    shown = ", ".join(dropped[:10])
    assert message.startswith(f"seed 43: the folds lack the analysis texts {shown}, ... (2 more)")


def test_an_analysis_text_held_twice_is_named(
    world: World, tables: tuple[pd.DataFrame, pd.DataFrame]
) -> None:
    folds, _ = tables
    row = folds[(folds["seed"] == 42) & (folds["text_id"] == "t5")]
    twice = pd.concat([folds, row.assign(fold=(row["fold"] + 1) % 5)], ignore_index=True)
    message = refusal(world, twice)
    assert message.startswith("seed 42: the folds hold the texts t5 in more than one row")


def test_texts_that_are_not_analysis_texts_are_named(
    world: World, tables: tuple[pd.DataFrame, pd.DataFrame]
) -> None:
    folds, _ = tables
    message = refusal(world, folds, gold=world.frame.iloc[:200])
    assert message.startswith("seed 42: the folds hold texts that are not analysis texts")
    assert "t200, t201" in message and "(30 more)" in message


def test_a_group_in_two_folds_is_named_with_its_seed_and_folds(
    world: World, tables: tuple[pd.DataFrame, pd.DataFrame]
) -> None:
    folds, _ = tables
    seed_42 = folds[folds["seed"] == 42].set_index("text_id")["fold"]
    other = (int(seed_42["t1"]) + 1) % 5
    message = refusal(world, relabelled(folds, 42, ["t1"], other))
    folds_named = ", ".join(str(label) for label in sorted({int(seed_42["t0"]), other}))
    assert message.startswith(f"seed 42: the groups g0 (folds {folds_named}) have texts in more")
    assert "a group never straddles train and test" in message


# --- the draws ------------------------------------------------------------------------


def fold_pool(world: World, folds: pd.DataFrame, seed: int, fold: int) -> np.ndarray:
    labels = folds[folds["seed"] == seed]["fold"].to_numpy()
    return np.flatnonzero(labels != fold)


def with_draw(
    world: World, draws: pd.DataFrame, seed: int, size: str, fold: int, drawn: np.ndarray
) -> pd.DataFrame:
    point = (draws["seed"] == seed) & (draws["size"] == size) & (draws["fold"] == fold)
    replaced = draw_table(world, {(seed, size): [np.array([], dtype=np.int64)] * fold + [drawn]})
    return pd.concat([draws[~point], replaced], ignore_index=True)


def test_a_missing_draw_of_a_fitted_size_is_named_with_seed_size_and_fold(
    world: World, tables: tuple[pd.DataFrame, pd.DataFrame]
) -> None:
    folds, draws = tables
    point = (draws["seed"] == 43) & (draws["size"] == "50") & (draws["fold"] == 2)
    message = refusal(world, folds, draws[~point])
    assert message.startswith("seed 43 size 50 fold 2: the splits hold no draw for this fold")


def test_a_draw_that_reaches_outside_the_training_pool_is_named(
    world: World, tables: tuple[pd.DataFrame, pd.DataFrame]
) -> None:
    folds, draws = tables
    labels = folds[folds["seed"] == 42]["fold"].to_numpy()
    test_group = np.flatnonzero(
        np.asarray(world.groups) == world.groups[int(np.argmax(labels == 0))]
    )
    drawn = draws[(draws["seed"] == 42) & (draws["size"] == "50") & (draws["fold"] == 0)]
    positions = np.sort(
        np.concatenate([pd.Index(world.ids).get_indexer(drawn["text_id"]), test_group])
    )
    message = refusal(world, folds, with_draw(world, draws, 42, "50", 0, positions))
    inside = ", ".join(np.asarray(world.ids)[test_group])
    assert message.startswith("seed 42 size 50 fold 0: the draw holds texts outside the fold's")
    assert inside in message
    unknown = pd.concat(
        [draws, pd.DataFrame({"seed": [42], "size": ["50"], "fold": [1], "text_id": ["zz"]})],
        ignore_index=True,
    )
    message = refusal(world, folds, unknown)
    assert message.startswith("seed 42 size 50 fold 1: the draw holds texts that are not analysis")
    assert message.endswith(": zz")


def test_a_draw_that_holds_part_of_a_group_is_named_with_the_group(
    world: World, tables: tuple[pd.DataFrame, pd.DataFrame]
) -> None:
    folds, draws = tables
    point = draws[(draws["seed"] == 42) & (draws["size"] == "50") & (draws["fold"] == 1)]
    first = str(point["text_id"].iloc[0])
    group = world.groups[world.ids.index(first)]
    message = refusal(world, folds, draws[~(draws.index == point.index[0])])
    assert message.startswith("seed 42 size 50 fold 1: a draw takes whole groups")
    assert f"part of the groups {group} (3 of 4 texts)" in message


def test_a_draw_with_fewer_units_than_its_size_is_refused_while_the_pool_holds_more(
    world: World, tables: tuple[pd.DataFrame, pd.DataFrame]
) -> None:
    folds, draws = tables
    pool = fold_pool(world, folds, 43, 4)
    drawn = whole_groups_until(world, pool, 48, sorted(set(world.groups)))
    message = refusal(world, folds, with_draw(world, draws, 43, "50", 4, drawn))
    texts = np.asarray(world.ids)[drawn].tolist()
    assert message == (
        "seed 43 size 50 fold 4: the draw holds 48 units, fewer than the size names (50), "
        "although the fold's training pool holds 192; no size is fitted on fewer units than it "
        f"names; the draw holds the texts {', '.join(texts[:10])}, ... (38 more)"
    )


def test_units_count_in_the_analysis_unit(world: World) -> None:
    settings = BaselineSettings(sizes=(10,), seeds=(42,), floor_units=1, encoder_sets=("alpha",))
    folds = keyed_folds(world, (42,))
    groups = np.asarray(world.groups)
    units = keyed_draws(world, folds, (10,), units=groups)
    pool = folds[42][0][0]
    nine_groups = sorted(set(groups[pool]))[:9]
    units[(42, "10")][0] = np.sort(pool[np.isin(groups[pool], nine_groups)])
    message = refusal(
        world, fold_table(world, folds), draw_table(world, units), settings, unit_id=None
    )
    assert message.startswith("seed 42 size 10 fold 0: the draw holds 9 units, fewer than")


def test_a_full_draw_that_is_not_the_whole_pool_is_named_with_the_texts_it_lacks(
    world: World, tables: tuple[pd.DataFrame, pd.DataFrame]
) -> None:
    folds, draws = tables
    pool = fold_pool(world, folds, 42, 0)
    groups = np.asarray(world.groups)
    lacking = pool[groups[pool] == groups[pool[0]]]
    drawn = np.setdiff1d(pool, lacking)
    message = refusal(world, folds, with_draw(world, draws, 42, "full", 0, drawn))
    assert message.startswith("seed 42 size full fold 0: a draw of size full must be the whole")
    assert ", ".join(np.asarray(world.ids)[lacking]) in message


@pytest.mark.parametrize(
    ("column", "value", "counts"),
    [
        ("seed", 99, "seeds that the settings do not ask for: 99; the settings ask for 42, 43"),
        (
            "size",
            "100",
            "sizes that the settings do not ask for: 100; the settings ask for 50, full",
        ),
        ("fold", 7, "fold labels outside 0 to 4: 7"),
    ],
)
def test_draws_that_the_settings_do_not_ask_for_are_named_with_their_first_point_and_texts(
    world: World,
    tables: tuple[pd.DataFrame, pd.DataFrame],
    column: str,
    value: object,
    counts: str,
) -> None:
    folds, draws = tables
    first = draws[(draws["seed"] == 42) & (draws["size"] == "50") & (draws["fold"] == 0)]
    extra = pd.concat([draws, first.assign(**{column: value})], ignore_index=True)
    point = {"seed": 42, "size": "50", "fold": 0, column: value}
    texts = first["text_id"].tolist()
    assert refusal(world, folds, extra) == (
        f"the splits' draws hold {counts}; the first is seed {point['seed']} size "
        f"{point['size']} fold {point['fold']}, with the texts {', '.join(texts[:10])}, ... "
        f"({len(texts) - 10} more)"
    )


def test_the_folds_are_refused_before_any_draw_is_checked(
    world: World, tables: tuple[pd.DataFrame, pd.DataFrame]
) -> None:
    folds, draws = tables
    broken = relabelled(folds, 43, ["t0"], 9)
    assert refusal(world, broken, draws.iloc[:0]).startswith("seed 43: fold labels")


# --- splits whose tables changed after construction -----------------------------------


def test_fold_labels_turned_into_floats_after_construction_are_refused_by_run_and_writer(
    world: World, tables: tuple[pd.DataFrame, pd.DataFrame], tmp_path: Path
) -> None:
    folds, draws = tables
    changed = BaselineSplits(folds=folds, draws=draws)
    changed.folds["fold"] = changed.folds["fold"].astype(float) + 0.5
    pattern = "the folds table's column fold must hold integers; got dtype float64"
    with pytest.raises(ConfigurationError, match=pattern):
        run(world, SETTINGS, splits=changed)
    with pytest.raises(ConfigurationError, match=pattern):
        write_splits(changed, tmp_path / "splits.json")
    assert not (tmp_path / "splits.json").exists()


def test_a_duplicate_row_put_into_the_draws_is_refused_before_any_fit(
    world: World, tables: tuple[pd.DataFrame, pd.DataFrame], caplog: pytest.LogCaptureFixture
) -> None:
    folds, draws = tables
    changed = BaselineSplits(folds=folds, draws=draws)
    assert changed.draws is not None
    changed.draws.loc[len(changed.draws)] = changed.draws.iloc[0]
    row = changed.draws.iloc[0]
    shown = rf"\({row['seed']}, {row['size']}, {row['fold']}, {row['text_id']}\)"
    with caplog.at_level(logging.INFO, logger="copernicus.baseline"):
        with pytest.raises(ConfigurationError, match=rf"draws table holds duplicate rows: {shown}"):
            run(world, SETTINGS, splits=changed)
    assert not [record for record in caplog.records if "fitted on" in record.getMessage()]


def test_splits_that_are_not_baseline_splits_are_refused(world: World) -> None:
    with pytest.raises(ConfigurationError, match="splits must be a BaselineSplits; got dict"):
        run(world, SETTINGS, splits={"folds": None})


def test_the_writer_refuses_splits_that_are_not_baseline_splits(tmp_path: Path) -> None:
    path = tmp_path / "splits.json"
    with pytest.raises(ConfigurationError, match="splits must be a BaselineSplits; got dict"):
        write_splits({"folds": None}, path)
    assert not path.exists()
