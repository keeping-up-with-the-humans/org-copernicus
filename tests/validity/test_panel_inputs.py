"""The validity panel on hand-built frames: input shapes, refusals and edge cases."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from validity_support import N_TEXTS, World, make_table, make_world, similarity_frame

from copernicus.errors import ConfigurationError
from copernicus.validity.panel import PanelSettings, validity_panel
from copernicus.validity.panel_inputs import resolve_target_scores

FAST = PanelSettings(n_boot=30)
TARGETS = {"construct": "gold_construct", "alpha": "gold_alpha", "beta": "gold_beta"}
EXPECTED_COLUMNS = [
    "target",
    "gold_column",
    "n_used",
    "n_dropped_missing",
    "n_runs",
    "status",
    "direction_applied",
    "n",
    "spearman",
    "rank_validity",
    "rank_validity_low",
    "rank_validity_high",
    "pearson",
    "pearson_low",
    "pearson_high",
    "kendall",
    "auc",
    "somers_d",
    "mae_01",
    "mae_01_low",
    "mae_01_high",
    "signed_gap_01",
    "level_displacement",
    "level_displacement_low",
    "level_displacement_high",
    "calibration_shortfall",
    "calibration_shortfall_low",
    "calibration_shortfall_high",
    "covariate_linked_error",
    "covariate_linked_error_low",
    "covariate_linked_error_high",
    "covariate_linked_error_raw",
    "covariate_linked_error_standardized",
    "covariate",
    "covariate_scale",
    "noise",
    "ceiling",
    "share_of_ceiling",
    "n_boot",
    "n_undefined_max",
]


@pytest.fixture(scope="module")
def world() -> World:
    return make_world()


def test_the_output_columns_come_in_the_documented_order(world):
    panel = validity_panel(world.scores, world.gold, world.table, targets=TARGETS, settings=FAST)
    assert list(panel.frame.columns) == EXPECTED_COLUMNS


def test_a_run_averaged_score_enters_once_per_text(world):
    panel = validity_panel(world.scores, world.gold, world.table, targets=TARGETS, settings=FAST)
    row = panel.frame.set_index("target").loc["alpha"]
    assert (row["n_runs"], row["n_used"], row["n"]) == (2, N_TEXTS, N_TEXTS)
    assert row["rank_validity"] > 0.5


def test_the_construct_is_the_mean_over_the_attributes(world):
    by_hand = (
        world.scores.assign(score=world.scores["score"].astype(float))
        .groupby(["text_id", "run_number"], as_index=False)["score"]
        .mean()
    )
    averaged = by_hand.assign(attribute_id="mean_score")
    direct = validity_panel(
        world.scores,
        world.gold,
        world.table,
        targets={"construct": "gold_construct"},
        settings=FAST,
    )
    reference = validity_panel(
        averaged,
        world.gold,
        world.table,
        targets={"mean_score": "gold_construct"},
        settings=FAST,
    )
    for column in ("rank_validity", "mae_01", "calibration_shortfall", "noise"):
        assert direct.frame.iloc[0][column] == pytest.approx(reference.frame.iloc[0][column])


def test_a_similarity_frame_with_target_levels_is_accepted(world):
    frame = similarity_frame(world)
    targets = {
        "construct": "gold_construct",
        "alpha__rollup": "gold_alpha",
        "attribute:beta__rollup": "gold_beta",
    }
    panel = validity_panel(frame, world.gold, world.table, targets=targets, settings=FAST)
    assert list(panel.frame["target"]) == list(targets)
    assert (panel.frame["n_runs"] == 1).all()
    assert panel.frame["noise"].isna().all()
    assert (panel.frame["rank_validity"] > 0.5).all()


def test_a_similarity_frame_without_target_levels_is_keyed_on_the_target_id(world):
    frame = similarity_frame(world).drop(columns="target_level")
    panel = validity_panel(
        frame, world.gold, world.table, targets={"construct": "gold_construct"}, settings=FAST
    )
    assert panel.frame.iloc[0]["n"] == N_TEXTS


def test_a_plain_key_that_exists_at_two_levels_is_refused(world):
    frame = similarity_frame(world)
    custom = frame[frame["target_id"] == "construct"].assign(target_id="custom")
    both = pd.concat([custom, custom.assign(target_level="attribute")])
    with pytest.raises(ConfigurationError, match="custom.*level"):
        validity_panel(
            pd.concat([frame, both]),
            world.gold,
            world.table,
            targets={"custom": "gold_construct"},
            settings=FAST,
        )


def test_two_scores_per_text_and_target_ask_for_one_setting(world):
    frame = similarity_frame(world)
    both = pd.concat([frame, frame.assign(encoder="second-encoder")])
    with pytest.raises(ConfigurationError, match="one setting"):
        validity_panel(both, world.gold, world.table, targets=TARGETS, settings=FAST)


def test_per_run_rows_without_a_run_number_point_to_the_missing_column(world):
    frame = similarity_frame(world)
    stacked = pd.concat([frame, frame.assign(score=frame["score"] + 0.1)])
    targets = {"construct": "gold_construct"}
    with pytest.raises(ConfigurationError, match=r"no run_number column.*one setting"):
        validity_panel(stacked, world.gold, world.table, targets=targets, settings=FAST)
    numbered = stacked.assign(run_number=1)
    with pytest.raises(ConfigurationError, match="one setting") as raised:
        validity_panel(numbered, world.gold, world.table, targets=targets, settings=FAST)
    assert "run_number" not in str(raised.value)


def test_scored_ids_without_any_gold_row_are_refused_naming_the_id_formats(world):
    scores = world.scores.assign(text_id=world.scores["text_id"].str[1:])
    gold = world.gold.assign(
        text_id=world.gold["text_id"].str[1:].astype(int), size=np.arange(N_TEXTS) % 7
    )
    pattern = r"none of the 40 scored texts.*zero padding.*000, 001, 002.*0, 1, 2"
    with pytest.raises(ConfigurationError, match=pattern):
        validity_panel(
            scores,
            gold,
            None,
            targets={"alpha": "gold_alpha"},
            settings=PanelSettings(n_boot=30, covariate="size"),
        )


def test_duplicated_attribute_scores_are_refused(world):
    doubled = pd.concat([world.scores, world.scores.iloc[:1]])
    with pytest.raises(ConfigurationError, match="duplicate"):
        validity_panel(doubled, world.gold, world.table, targets=TARGETS, settings=FAST)


def test_a_target_that_matches_nothing_is_named_with_the_available_keys(world):
    with pytest.raises(ConfigurationError, match=r"'gamma'.*alpha.*beta"):
        validity_panel(
            world.scores, world.gold, world.table, targets={"gamma": "gold_alpha"}, settings=FAST
        )


def test_missing_score_columns_are_named(world):
    with pytest.raises(ConfigurationError, match="attribute_id"):
        validity_panel(
            world.scores.drop(columns="attribute_id"),
            world.gold,
            world.table,
            targets=TARGETS,
            settings=FAST,
        )


def test_a_missing_gold_column_is_named(world):
    with pytest.raises(ConfigurationError, match="gold_missing"):
        validity_panel(
            world.scores,
            world.gold,
            world.table,
            targets={"alpha": "gold_missing"},
            settings=FAST,
        )


def test_a_gold_frame_without_text_ids_is_refused(world):
    with pytest.raises(ConfigurationError, match="text_id"):
        validity_panel(
            world.scores,
            world.gold.drop(columns="text_id"),
            world.table,
            targets=TARGETS,
            settings=FAST,
        )


def test_duplicated_gold_ids_are_refused(world):
    with pytest.raises(ConfigurationError, match="duplicate"):
        validity_panel(
            world.scores,
            pd.concat([world.gold, world.gold.iloc[:1]]),
            world.table,
            targets=TARGETS,
            settings=FAST,
        )


@pytest.mark.parametrize("scores_are_strings", [True, False])
def test_text_ids_are_compared_as_strings_on_both_sides(world, scores_are_strings):
    numbers = world.scores["text_id"].str[1:].astype(int)
    scores = world.scores.assign(text_id=numbers.astype(str) if scores_are_strings else numbers)
    gold_numbers = world.gold["text_id"].str[1:].astype(int)
    gold = world.gold.assign(
        text_id=gold_numbers if scores_are_strings else gold_numbers.astype(str),
        size=np.arange(N_TEXTS) % 7,
    )
    panel = validity_panel(
        scores,
        gold,
        None,
        targets={"alpha": "gold_alpha"},
        settings=PanelSettings(n_boot=30, covariate="size"),
    )
    assert panel.frame.iloc[0]["n"] == N_TEXTS
    assert panel.n_dropped_missing == 0


def test_without_a_table_the_covariate_comes_from_the_gold_and_groups_are_the_texts(world):
    gold = world.gold.assign(size=np.arange(N_TEXTS) % 7)
    panel = validity_panel(
        world.scores,
        gold,
        None,
        targets={"alpha": "gold_alpha"},
        settings=PanelSettings(n_boot=30, covariate="size"),
    )
    assert panel.frame.iloc[0]["covariate"] == "size"
    assert np.isfinite(panel.frame.iloc[0]["covariate_linked_error"])


def test_without_a_table_a_covariate_is_required(world):
    with pytest.raises(ConfigurationError, match="table or a covariate"):
        validity_panel(world.scores, world.gold, None, targets=TARGETS, settings=FAST)


def test_a_named_covariate_absent_everywhere_is_refused(world):
    with pytest.raises(ConfigurationError, match="nowhere"):
        validity_panel(
            world.scores,
            world.gold,
            world.table,
            targets=TARGETS,
            settings=PanelSettings(n_boot=30, covariate="nowhere"),
        )


def test_a_named_covariate_is_looked_up_in_the_table_frame(world):
    panel = validity_panel(
        world.scores,
        world.gold,
        world.table,
        targets=TARGETS,
        settings=PanelSettings(n_boot=30, covariate="n_words", covariate_scale="raw"),
    )
    assert (panel.frame["covariate"] == "n_words").all()
    assert (panel.frame["covariate_scale"] == "raw").all()


def test_a_non_numeric_covariate_is_refused(world):
    gold = world.gold.assign(size=["big"] * N_TEXTS)
    with pytest.raises(ConfigurationError, match="size.*numeric"):
        validity_panel(
            world.scores,
            gold,
            None,
            targets=TARGETS,
            settings=PanelSettings(n_boot=30, covariate="size"),
        )


def test_a_scored_text_absent_from_the_table_is_refused_with_its_id(world):
    smaller = make_table(N_TEXTS - 3)
    with pytest.raises(ConfigurationError, match=r"s037.*s038.*s039"):
        validity_panel(world.scores, world.gold, smaller, targets=TARGETS, settings=FAST)


def test_a_constant_covariate_gives_a_nan_error_and_no_interval_but_no_failure(world):
    gold = world.gold.assign(flat=1.0)
    panel = validity_panel(
        world.scores,
        gold,
        None,
        targets={"alpha": "gold_alpha"},
        settings=PanelSettings(n_boot=30, covariate="flat"),
    )
    row = panel.frame.iloc[0]
    assert np.isnan(row["covariate_linked_error"])
    assert np.isnan(row["covariate_linked_error_low"])
    assert np.isfinite(row["rank_validity_low"])


@pytest.mark.parametrize(
    ("attribute_id", "message"),
    [
        ("construct", "id 'construct' is the name of the whole-construct target"),
        ("Construct", "id 'Construct' is the name of the whole-construct target"),
        ("x__rollup", "id 'x__rollup' ends in '__rollup', the suffix of the roll-up target"),
    ],
)
def test_attribute_ids_that_name_a_generated_target_are_refused(
    attribute_id: str, message: str
) -> None:
    """An attribute literally named ``construct`` would resolve to the mean of the attributes."""
    scores = pd.DataFrame(
        {
            "text_id": ["a", "a", "a"],
            "run_number": [1, 1, 1],
            "attribute_id": [attribute_id, "second", "third"],
            "score": [1.0, 5.0, 5.0],
        }
    )
    with pytest.raises(ConfigurationError) as refused:
        resolve_target_scores(scores, ["construct"])
    assert message in str(refused.value) and "rename" in str(refused.value)
    plain = scores.assign(attribute_id=["first", "second", "third"])
    assert resolve_target_scores(plain, ["construct"])["construct"].score.iloc[0] == 11 / 3


def test_a_target_frame_never_turns_a_target_name_into_the_aggregate() -> None:
    """A target frame selects the rows labelled with a key: a clash is refused, not averaged."""
    rows = {
        "text_id": ["a", "a", "a"],
        "target_level": ["construct"] * 3,
        "target_id": ["construct", "construct", "second"],
        "score": [2.0, 1.0, 5.0],
    }
    with pytest.raises(ConfigurationError, match="more than one score per text and run"):
        resolve_target_scores(pd.DataFrame(rows), ["construct"])
    by_attribute = pd.DataFrame({**rows, "target_level": ["attribute"] * 3}).iloc[1:]
    with pytest.raises(ConfigurationError, match="matches no scores"):
        resolve_target_scores(by_attribute, ["construct"])
