"""Baseline runner: refusals of inconsistent inputs and encoder-set names."""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest
from baseline_support import GOLD, World, make_set, run

from copernicus.baseline.runner import BaselineSettings
from copernicus.data.table import TextTable
from copernicus.embeddings.sets import EmbeddingSet
from copernicus.errors import ConfigurationError


@pytest.fixture(scope="module")
def world() -> World:
    return World()


def settings_one() -> BaselineSettings:
    return BaselineSettings(sizes=("full",), seeds=(42,), encoder_sets=("alpha",))


def test_fewer_groups_than_folds_raises():
    small = World(n_groups=4)
    with pytest.raises(ConfigurationError, match="groups"):
        run(small, settings_one())


def test_stacked_with_one_key_is_refused(world: World):
    settings = dataclasses.replace(settings_one(), encoder_sets=("stacked",))
    with pytest.raises(ConfigurationError, match="at least two"):
        run(world, settings, embeddings={"alpha": world.embeddings["alpha"]})


def test_unknown_encoder_set_name_lists_the_keys(world: World):
    settings = dataclasses.replace(settings_one(), encoder_sets=("gamma",))
    with pytest.raises(ConfigurationError, match=r"gamma.*alpha.*beta"):
        run(world, settings)


def test_repeated_encoder_set_is_refused(world: World):
    settings = dataclasses.replace(settings_one(), encoder_sets=("single", "alpha"))
    with pytest.raises(ConfigurationError, match="more than once"):
        run(world, settings)


def test_gold_needs_its_columns(world: World):
    with pytest.raises(ConfigurationError, match="nope"):
        run(world, settings_one(), gold_column="nope")
    with pytest.raises(ConfigurationError, match="text_id"):
        run(world, settings_one(), gold=world.frame.drop(columns="text_id"))


def test_string_gold_is_refused(world: World):
    frame = world.frame.assign(**{GOLD: ["low", "high"] * 120})
    with pytest.raises(ConfigurationError, match="numeric"):
        run(world, settings_one(), gold=frame)


def test_missing_gold_names_the_text_ids(world: World):
    values = world.gold.copy()
    values[[3, 7]] = np.nan
    values[11] = np.inf
    frame = world.frame.assign(**{GOLD: values})
    with pytest.raises(ConfigurationError, match=r"t3.*t7.*t11"):
        run(world, settings_one(), gold=frame)


def test_duplicated_gold_ids_are_refused(world: World):
    frame = pd.concat([world.frame, world.frame.iloc[:2]], ignore_index=True)
    with pytest.raises(ConfigurationError, match="duplicate"):
        run(world, settings_one(), gold=frame)


def test_gold_text_missing_from_the_table_is_named(world: World):
    table = TextTable.from_frame(world.table.frame.iloc[2:].reset_index(drop=True))
    with pytest.raises(ConfigurationError, match=r"table.*t0.*t1|t0.*t1.*table"):
        run(world, settings_one(), table=table)


def test_gold_text_missing_from_an_encoder_is_named(world: World):
    keep = slice(1, None)
    partial = EmbeddingSet(
        ids=world.embeddings["alpha"].ids[keep],
        vectors=world.embeddings["alpha"].vectors[keep],
        chunk_vectors=world.embeddings["alpha"].vectors[keep],
        chunk_owner=np.arange(239, dtype=np.int64),
        chunk_weight=np.ones(239, dtype=np.int64),
        encoder="alpha",
        metadata=None,
        sidecar={},
    )
    with pytest.raises(ConfigurationError, match=r"alpha.*t0"):
        run(world, settings_one(), embeddings={"alpha": partial, "beta": world.embeddings["beta"]})


def test_unused_encoder_sets_do_not_need_the_gold_texts(world: World):
    partial = make_set(world.ids[:10], world.embeddings["beta"].vectors[:10], "beta")
    result = run(
        world, settings_one(), embeddings={"alpha": world.embeddings["alpha"], "beta": partial}
    )
    assert not result.results.empty


def test_unknown_unit_column_is_refused(world: World):
    with pytest.raises(ConfigurationError, match="unit_id"):
        run(world, settings_one(), unit_id="no_such_column")


def test_coder_score_must_cover_every_gold_text_with_finite_values(world: World):
    with pytest.raises(ConfigurationError, match=r"coder score.*t0"):
        run(world, settings_one(), coder_score=world.score.iloc[1:])
    broken = world.score.copy()
    broken.iloc[5] = np.nan
    with pytest.raises(ConfigurationError, match=r"coder score.*t5"):
        run(world, settings_one(), coder_score=broken)


def two_valued(world: World, low: float, high: float) -> pd.DataFrame:
    """Text-level gold coded ``low`` below the median and ``high`` above it."""
    above = world.gold > np.median(world.gold)
    return pd.DataFrame({"text_id": world.ids, GOLD: np.where(above, high, low)})


@pytest.mark.parametrize(("low", "high"), [(1, 2), (-1, 1)])
def test_a_two_valued_gold_gives_the_auc_of_its_zero_one_coding(world: World, low, high):
    coded = run(world, settings_one(), gold=two_valued(world, low, high)).results.iloc[0]
    reference = run(world, settings_one(), gold=two_valued(world, 0, 1)).results.iloc[0]
    assert coded["n_folds_ok"] == 5 and coded["n_folds_nonbinary_unit_gold"] == 0
    assert coded["auc"] > 0.8
    assert coded["auc"] == pytest.approx(reference["auc"], abs=1e-9)
    assert coded["somers_d"] == pytest.approx(reference["somers_d"], abs=1e-9)


def test_a_binary_gold_that_varies_within_a_unit_is_refused_with_the_unit_column(world: World):
    gold = two_valued(world, 0, 1)
    with pytest.raises(ConfigurationError, match=r"'group_id'.*: g\d+"):
        run(world, settings_one(), gold=gold, unit_id=None)
    forced = dataclasses.replace(settings_one(), binary=True)
    with pytest.raises(ConfigurationError, match="group_id"):
        run(world, forced, gold=gold, unit_id=None)


def test_a_gold_that_varies_within_a_unit_is_accepted_when_binary_is_off(world: World):
    settings = dataclasses.replace(settings_one(), binary=False, floor_units=10)
    result = run(world, settings, gold=two_valued(world, 0, 1), unit_id=None)
    assert len(result.results) == 1 and result.results["auc"].isna().all()


def test_binary_set_on_a_gold_with_more_than_two_values_is_refused(world: World):
    with pytest.raises(ConfigurationError, match=r"binary.*240 distinct"):
        run(world, dataclasses.replace(settings_one(), binary=True))


def test_a_unit_that_spans_two_groups_is_refused_with_its_name(world: World):
    units = list(world.ids)
    units[3] = units[4] = "shared"  # t3 lies in g0 and t4 in g1
    table = TextTable.from_frame(world.table.frame.assign(unit=units))
    with pytest.raises(ConfigurationError, match=r"unit column 'unit'.*: shared$"):
        run(world, settings_one(), table=table, unit_id="unit")


def test_a_non_numeric_coder_score_is_refused_with_its_column(world: World):
    words = pd.Series(["high"] * len(world.ids), index=world.ids, name="rating")
    with pytest.raises(ConfigurationError, match=r"coder score column 'rating'.*numeric"):
        run(world, settings_one(), coder_score=words)


def test_a_missing_value_in_a_nullable_coder_score_names_the_text(world: World):
    values = pd.array(world.score.to_numpy(), dtype="Float64")
    values[7] = pd.NA
    with pytest.raises(ConfigurationError, match=r"coder score.*t7"):
        run(world, settings_one(), coder_score=pd.Series(values, index=world.ids))


def noisy_two_valued(world: World) -> pd.DataFrame:
    """Gold of two values that differ by float noise only: 0.3 and 0.1 + 0.2."""
    above = world.gold > np.median(world.gold)
    return pd.DataFrame({"text_id": world.ids, GOLD: np.where(above, 0.1 + 0.2, 0.3)})


@pytest.mark.parametrize("binary", [None, True])
def test_two_class_values_that_float_noise_alone_separates_are_refused(world: World, binary):
    settings = dataclasses.replace(settings_one(), binary=binary)
    with pytest.raises(ConfigurationError, match=r"gold column 'gold'.*cannot be told apart"):
        run(world, settings, gold=noisy_two_valued(world))


def test_the_within_unit_refusal_names_the_command_line_options(world: World):
    gold = two_valued(world, 0, 1)
    with pytest.raises(ConfigurationError) as error:
        run(world, settings_one(), gold=gold, unit_id=None)
    message = str(error.value)
    assert "--no-binary" in message and "--unit-id text_id" in message
    assert "set binary to False" in message and "unit_id" in message
