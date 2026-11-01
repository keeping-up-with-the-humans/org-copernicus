"""The human-sample draw and its attachment to correction data."""

from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd
import pytest

from copernicus.corrections.data import CorrectionData
from copernicus.corrections.sampling import (
    SampleDraw,
    attach_draw,
    draw_groups,
    draw_human_sample,
    groups_for_quota,
)
from copernicus.data import TextTable
from copernicus.errors import ConfigurationError, SettingsError


def toy_data(toy_corpus: TextTable) -> CorrectionData:
    rng = np.random.default_rng(5)
    frame = toy_corpus.frame
    scores = pd.DataFrame(
        {
            "text_id": frame["text_id"],
            "construct_score": frame["gold_constructiveness"].to_numpy(float)
            + rng.normal(scale=0.4, size=len(frame)),
        }
    )
    empty_gold = pd.DataFrame({"text_id": frame["text_id"], "gold": np.nan})
    return CorrectionData.from_scores(
        scores, toy_corpus, empty_gold, gold_column="gold", score_columns=("construct_score",)
    )


def coded_values(toy_corpus: TextTable) -> pd.Series:
    frame = toy_corpus.frame
    return pd.Series(frame["gold_constructiveness"].to_numpy(float), index=frame["text_id"])


def test_groups_for_quota_matches_the_integer_ceiling() -> None:
    group_id = np.repeat(np.arange(60), 3)
    unit_id = np.arange(180)
    assert groups_for_quota(50, group_id, unit_id) == 17
    assert groups_for_quota(1, group_id, unit_id) == 1
    assert groups_for_quota(180, group_id, unit_id) == 60
    assert groups_for_quota(10_000, group_id, unit_id) == 60


@pytest.mark.parametrize(
    ("quota", "n_groups", "n_units"),
    [(50, 60, 180), (7, 13, 40), (1, 5, 5), (23, 18, 54), (100, 9, 31), (99, 4, 1000)],
)
def test_groups_for_quota_equals_the_integer_formula(
    quota: int, n_groups: int, n_units: int
) -> None:
    group_id = np.arange(n_units) % n_groups
    unit_id = np.arange(n_units)
    expected = min(n_groups, max(1, -(-quota * n_groups // n_units)))
    assert groups_for_quota(quota, group_id, unit_id) == expected


def test_groups_for_quota_refuses_bad_input() -> None:
    group_id = np.repeat(np.arange(4), 3)
    unit_id = np.arange(12)
    with pytest.raises(SettingsError, match="quota"):
        groups_for_quota(0, group_id, unit_id)
    with pytest.raises(ConfigurationError, match="empty"):
        groups_for_quota(5, np.array([]), np.array([]))
    with pytest.raises(ConfigurationError):
        groups_for_quota(5, group_id, unit_id[:5])


def test_draw_with_n_on_the_toy_corpus(toy_corpus: TextTable) -> None:
    draw = draw_human_sample(toy_corpus, n=24)
    assert isinstance(draw, SampleDraw)
    assert draw.m == 8
    assert draw.n_groups == 18
    assert draw.n_units == 54
    assert draw.quota == 24
    assert draw.seed == 42
    assert draw.pi == 8 / 18
    assert len(draw.groups) == 8
    assert draw.groups.tolist() == sorted(draw.groups.tolist())


def test_drawn_ids_are_whole_groups_in_table_order(toy_corpus: TextTable) -> None:
    draw = draw_human_sample(toy_corpus, n=24, seed=7)
    frame = toy_corpus.frame
    expected = frame.loc[frame["group_id"].isin(draw.groups), "text_id"].tolist()
    assert draw.ids.tolist() == expected
    assert len(draw.ids) == 24
    for group in draw.groups:
        members = frame.loc[frame["group_id"] == group, "text_id"]
        assert set(members) <= set(draw.ids)


def test_share_draw_on_the_toy_corpus(toy_corpus: TextTable) -> None:
    draw = draw_human_sample(toy_corpus, share=0.5)
    assert draw.quota == 27
    assert draw.m == 9
    assert draw.pi == 0.5
    assert len(draw.ids) == 27


def test_draw_is_reproducible_and_seed_dependent(toy_corpus: TextTable) -> None:
    first = draw_human_sample(toy_corpus, n=20, seed=1)
    again = draw_human_sample(toy_corpus, n=20, seed=1)
    other = draw_human_sample(toy_corpus, n=20, seed=2)
    assert first.ids.tolist() == again.ids.tolist()
    assert first.ids.tolist() != other.ids.tolist()


def test_draw_matches_the_documented_rule(toy_corpus: TextTable) -> None:
    draw = draw_human_sample(toy_corpus, n=21, seed=9)
    groups = np.array(sorted(toy_corpus.frame["group_id"].unique()))
    expected = np.sort(np.random.default_rng(9).choice(groups, size=draw.m, replace=False))
    assert draw.groups.tolist() == expected.tolist()
    assert draw.m == math.ceil(21 * 18 / 54)


def test_draw_with_another_unit_column(toy_corpus: TextTable) -> None:
    table = TextTable.from_frame(
        toy_corpus.frame.assign(thread=lambda f: f["group_id"].str.replace("w", "x")),
        group_id="group_id",
    )
    draw = draw_human_sample(table, n=9, unit_id="thread")
    assert draw.n_units == 18
    assert draw.quota == 9
    assert draw.m == 9
    assert draw.pi == 0.5


def test_draw_refuses_bad_arguments(toy_corpus: TextTable) -> None:
    with pytest.raises(SettingsError, match="exactly one"):
        draw_human_sample(toy_corpus, n=24, share=0.5)
    with pytest.raises(SettingsError, match="exactly one"):
        draw_human_sample(toy_corpus)
    with pytest.raises(SettingsError, match="integer"):
        draw_human_sample(toy_corpus, n=2.5)  # type: ignore[arg-type]
    with pytest.raises(SettingsError, match="share"):
        draw_human_sample(toy_corpus, share=0.0)
    with pytest.raises(SettingsError, match="share"):
        draw_human_sample(toy_corpus, share=1.2)


def test_a_quota_above_half_the_units_warns(
    toy_corpus: TextTable, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger="copernicus.corrections.sampling"):
        draw = draw_human_sample(toy_corpus, n=30)
    assert (draw.m, draw.pi) == (10, 10 / 18)
    assert any("exceeds half" in record.getMessage() for record in caplog.records)
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="copernicus.corrections.sampling"):
        assert draw_human_sample(toy_corpus, n=40).m == 14
    assert any("battery refuses" in record.getMessage() for record in caplog.records)


def test_a_quota_up_to_half_the_units_does_not_warn(
    toy_corpus: TextTable, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger="copernicus.corrections.sampling"):
        assert draw_human_sample(toy_corpus, n=27).quota == 27
    assert not caplog.records


def test_full_draw_takes_every_group_with_probability_one(
    toy_corpus: TextTable, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger="copernicus.corrections.sampling"):
        draws = (
            draw_human_sample(toy_corpus, n=54),
            draw_human_sample(toy_corpus, share=1.0),
        )
    for draw in draws:
        assert draw.m == 18
        assert draw.pi == 1.0
        assert draw.quota == 54
        assert draw.ids.tolist() == toy_corpus.frame["text_id"].tolist()
        assert draw.groups.tolist() == sorted(toy_corpus.frame["group_id"].unique())
    assert not caplog.records


def test_the_draw_logs_one_group_per_text_when_the_table_names_no_group(
    toy_corpus: TextTable, caplog: pytest.LogCaptureFixture
) -> None:
    ungrouped = TextTable.from_frame(toy_corpus.frame[["text_id", "text"]])
    with caplog.at_level(logging.INFO, logger="copernicus.corrections.sampling"):
        draw_human_sample(ungrouped, n=10, seed=1)
    (line,) = [r.getMessage() for r in caplog.records if r.getMessage().startswith("drew")]
    assert line == "drew 10 of 54 groups (10 texts); groups: one per text (no group column)"
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="copernicus.corrections.sampling"):
        draw_human_sample(toy_corpus, n=10, seed=1)
    (line,) = [r.getMessage() for r in caplog.records if r.getMessage().startswith("drew")]
    assert line == "drew 4 of 18 groups (12 texts)"


def test_quota_outside_one_to_the_unit_count_is_refused(toy_corpus: TextTable) -> None:
    with pytest.raises(ConfigurationError, match=r"55.*54"):
        draw_human_sample(toy_corpus, n=55)
    with pytest.raises(ConfigurationError, match=r"0 units.*54"):
        draw_human_sample(toy_corpus, n=0)
    with pytest.raises(ConfigurationError, match=r"0 units.*54"):
        draw_human_sample(toy_corpus, share=1e-12)
    assert draw_human_sample(toy_corpus, share=0.0001).quota == 1


def test_draw_refuses_a_bad_seed(toy_corpus: TextTable) -> None:
    for seed in (-1, 1.5, True):
        with pytest.raises(SettingsError, match="seed"):
            draw_human_sample(toy_corpus, n=10, seed=seed)  # type: ignore[arg-type]
    assert draw_human_sample(toy_corpus, n=10, seed=np.int64(5)).seed == 5


def test_draw_with_units_shared_across_groups(toy_corpus: TextTable) -> None:
    thread = [f"u{i // 4}" for i in range(54)]
    table = TextTable.from_frame(toy_corpus.frame.assign(thread=thread), group_id="group_id")
    draw = draw_human_sample(table, n=7, unit_id="thread")
    assert draw.n_units == 14
    assert draw.n_groups == 18
    assert draw.m == math.ceil(7 * 18 / 14)
    assert draw.pi == draw.m / 18


def test_draw_refuses_an_unknown_unit_column(toy_corpus: TextTable) -> None:
    with pytest.raises(ConfigurationError, match="nonexistent"):
        draw_human_sample(toy_corpus, n=10, unit_id="nonexistent")


def test_attach_draw_sets_gold_on_drawn_ids_and_pi_on_every_row(toy_corpus: TextTable) -> None:
    data = toy_data(toy_corpus)
    assert data.n_labelled == 0
    draw = draw_human_sample(toy_corpus, n=24, seed=3)
    labelled = attach_draw(data, coded_values(toy_corpus), draw)
    assert labelled.labelled_ids.tolist() == draw.ids.tolist()
    assert labelled.pi is not None
    assert np.all(labelled.pi == draw.pi)
    drawn = np.isin(labelled.ids, draw.ids)
    assert np.isnan(labelled.gold[~drawn]).all()
    expected = toy_corpus.frame.set_index("text_id").loc[draw.ids, "gold_constructiveness"]
    np.testing.assert_array_equal(labelled.gold[drawn], expected.to_numpy())
    assert data.n_labelled == 0
    assert data.pi is None


def test_attach_draw_ignores_values_outside_the_draw(toy_corpus: TextTable) -> None:
    data = toy_data(toy_corpus)
    draw = draw_human_sample(toy_corpus, n=12, seed=3)
    values = coded_values(toy_corpus)
    labelled = attach_draw(data, values, draw)
    assert labelled.n_labelled == 12


def test_attach_draw_refuses_missing_coded_values(toy_corpus: TextTable) -> None:
    data = toy_data(toy_corpus)
    draw = draw_human_sample(toy_corpus, n=24, seed=3)
    values = coded_values(toy_corpus).drop(draw.ids[0])
    with pytest.raises(ConfigurationError, match=str(draw.ids[0])):
        attach_draw(data, values, draw)
    values = coded_values(toy_corpus)
    values.loc[draw.ids[1]] = np.nan
    with pytest.raises(ConfigurationError, match=str(draw.ids[1])):
        attach_draw(data, values, draw)


def test_attach_draw_refuses_drawn_ids_absent_from_the_data(toy_corpus: TextTable) -> None:
    data = toy_data(toy_corpus)
    draw = draw_human_sample(toy_corpus, n=24, seed=3)
    smaller = SampleDraw(
        ids=np.append(draw.ids, "elsewhere"),
        groups=draw.groups,
        pi=draw.pi,
        m=draw.m,
        n_groups=draw.n_groups,
        n_units=draw.n_units,
        quota=draw.quota,
        seed=draw.seed,
        population_hash=draw.population_hash,
    )
    values = pd.concat([coded_values(toy_corpus), pd.Series({"elsewhere": 3.0})])
    with pytest.raises(ConfigurationError, match="drawn texts not in the data: 'elsewhere'"):
        attach_draw(data, values, smaller)


def test_attach_draw_refuses_another_table_and_a_draw_that_miscounts_its_own(
    toy_corpus: TextTable,
) -> None:
    data = toy_data(toy_corpus)
    draw = draw_human_sample(toy_corpus, n=24, seed=3)
    values = coded_values(toy_corpus)
    renamed = TextTable.from_frame(
        toy_corpus.frame.assign(group_id=lambda f: "x" + f["group_id"]), group_id="group_id"
    )
    with pytest.raises(ConfigurationError, match=r"different table than the data \(different"):
        attach_draw(toy_data(renamed), values, draw)
    left_out = SampleDraw(
        ids=draw.ids[:-1],
        groups=draw.groups,
        pi=draw.pi,
        m=draw.m,
        n_groups=draw.n_groups,
        n_units=draw.n_units,
        quota=draw.quota,
        seed=draw.seed,
        population_hash=draw.population_hash,
    )
    missing = rf"does not describe the table.*missing from the drawn ids: '{draw.ids[-1]}'$"
    with pytest.raises(ConfigurationError, match=missing):
        attach_draw(data, values, left_out)
    wrong_count = SampleDraw(
        ids=draw.ids,
        groups=draw.groups,
        pi=draw.pi,
        m=draw.m,
        n_groups=40,
        n_units=draw.n_units,
        quota=draw.quota,
        seed=draw.seed,
        population_hash=draw.population_hash,
    )
    miscounted = r"records m = 8, n_groups = 40 and .* the data give m = 8, n_groups = 18 and"
    with pytest.raises(ConfigurationError, match=miscounted):
        attach_draw(data, values, wrong_count)


def test_attach_draw_refuses_non_numeric_values(toy_corpus: TextTable) -> None:
    data = toy_data(toy_corpus)
    draw = draw_human_sample(toy_corpus, n=24, seed=3)
    words = pd.Series("high", index=coded_values(toy_corpus).index)
    with pytest.raises(ConfigurationError, match="numeric"):
        attach_draw(data, words, draw)


def test_attach_draw_refuses_a_gold_series_that_repeats_an_id(toy_corpus: TextTable) -> None:
    data = toy_data(toy_corpus)
    draw = draw_human_sample(toy_corpus, n=24, seed=3)
    values = coded_values(toy_corpus)
    with pytest.raises(ConfigurationError, match=f"repeats text ids '{values.index[0]}'$"):
        attach_draw(data, pd.concat([values, values.iloc[:1]]), draw)


def test_the_group_draw_helper_is_the_draw_of_draw_human_sample(toy_corpus: TextTable) -> None:
    frame = toy_corpus.frame
    ids, groups = frame["text_id"].to_numpy(), frame["group_id"].to_numpy()
    for seed in (0, 7, 2**40):
        helper = draw_groups(ids, groups, ids, 20, seed=seed)
        table_draw = draw_human_sample(toy_corpus, n=20, seed=seed)
        assert helper.ids.tolist() == table_draw.ids.tolist()
        assert (helper.groups.tolist(), helper.pi, helper.seed) == (
            table_draw.groups.tolist(),
            table_draw.pi,
            seed,
        )
    with pytest.raises(ConfigurationError, match="equal length"):
        draw_groups(ids, groups[:-1], ids, 20, seed=0)
    with pytest.raises(ConfigurationError, match=r"55.*54"):
        draw_groups(ids, groups, ids, 55, seed=0)
    with pytest.raises(SettingsError, match="seed"):
        draw_groups(ids, groups, ids, 20, seed=-1)
