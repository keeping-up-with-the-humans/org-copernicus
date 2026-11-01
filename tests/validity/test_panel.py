"""The validity panel on the toy corpus coded with the cue backend."""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from copernicus.assets import load_toy_cues, toy_corpus_path
from copernicus.coders import CodingSettings, code_texts
from copernicus.coders.results import export_construct_scores, export_scores
from copernicus.data import TextTable
from copernicus.errors import ConfigurationError
from copernicus.llm import FakeBackend, cue_responder
from copernicus.rubric.loader import load_rubric, toy_rubric_path
from copernicus.validity import association, score_errors
from copernicus.validity.panel import PanelSettings, ValidityPanel, validity_panel

TARGETS = {
    "construct": "gold_constructiveness",
    "specificity": "gold_specificity",
    "tone": "gold_tone",
    "reasoning": "gold_reasoning",
}
N_BOOT = 200
INTERVALS = (
    "rank_validity",
    "pearson",
    "mae_01",
    "level_displacement",
    "calibration_shortfall",
    "covariate_linked_error",
)


@pytest.fixture(scope="module")
def store(tmp_path_factory: pytest.TempPathFactory):
    rubric = load_rubric(toy_rubric_path())
    table = TextTable.read(toy_corpus_path(), group_id="group_id")
    path = tmp_path_factory.mktemp("panel") / "toy.sqlite"
    backend = FakeBackend(cue_responder(rubric, load_toy_cues()))
    code_texts(table, rubric, backend, path, CodingSettings())
    return path


@pytest.fixture(scope="module")
def scores(store) -> pd.DataFrame:
    return export_scores(store)


@pytest.fixture(scope="module")
def table() -> TextTable:
    return TextTable.read(toy_corpus_path(), group_id="group_id")


@pytest.fixture(scope="module")
def gold() -> pd.DataFrame:
    corpus = pd.read_csv(toy_corpus_path())
    return corpus[["text_id", *[c for c in corpus.columns if c.startswith("gold_")]]]


@pytest.fixture(scope="module")
def panel(scores, gold, table) -> ValidityPanel:
    return validity_panel(
        scores, gold, table, targets=TARGETS, settings=PanelSettings(n_boot=N_BOOT)
    )


def test_the_panel_has_one_row_per_target_in_the_order_of_the_mapping(panel):
    assert list(panel.frame["target"]) == list(TARGETS)
    assert list(panel.frame["gold_column"]) == list(TARGETS.values())


def test_the_construct_rank_validity_is_substantial(panel):
    row = panel.frame.set_index("target").loc["construct"]
    assert row["rank_validity"] >= 0.5
    assert row["status"] == "ok"
    assert row["direction_applied"] == "positive"


def test_the_construct_rank_validity_matches_the_run_averaged_construct_score(store, panel, gold):
    construct = export_construct_scores(store)
    per_text = construct.groupby("text_id")["construct_score"].mean()
    aligned = gold.set_index("text_id")["gold_constructiveness"].loc[per_text.index]
    expected = abs(spearmanr(per_text.to_numpy(), aligned.to_numpy()).statistic)
    row = panel.frame.set_index("target").loc["construct"]
    assert row["rank_validity"] == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("name", INTERVALS)
def test_the_intervals_bracket_the_point_estimates(panel, name):
    frame = panel.frame
    assert (frame[f"{name}_low"] <= frame[name]).all()
    assert (frame[name] <= frame[f"{name}_high"]).all()


def test_every_row_reports_two_runs_and_a_finite_noise(panel):
    assert (panel.frame["n_runs"] == 2).all()
    assert np.isfinite(panel.frame["noise"]).all()


def test_the_default_covariate_is_log_characters_on_the_standardized_scale(panel):
    assert (panel.frame["covariate"] == "log_n_chars").all()
    assert (panel.frame["covariate_scale"] == "standardized").all()


def test_counts_cover_all_texts_when_nothing_is_missing(panel, gold):
    assert panel.n_used == len(gold)
    assert panel.n_dropped_missing == 0
    assert (panel.frame["n_used"] == len(gold)).all()
    assert (panel.frame["n"] == len(gold)).all()
    assert (panel.frame["n_dropped_missing"] == 0).all()
    assert (panel.frame["n_boot"] == N_BOOT).all()
    assert (panel.frame["n_undefined_max"] >= 0).all()


def test_without_a_reliability_the_ceiling_and_the_share_are_nan(panel):
    assert panel.frame["ceiling"].isna().all()
    assert panel.frame["share_of_ceiling"].isna().all()


def test_a_reliability_sets_the_ceiling_on_every_row(scores, gold, table):
    result = validity_panel(
        scores, gold, table, targets=TARGETS, settings=PanelSettings(n_boot=20, reliability=0.81)
    )
    assert result.frame["ceiling"].tolist() == pytest.approx([0.9] * 4)
    expected = result.frame["rank_validity"] / 0.9
    assert result.frame["share_of_ceiling"].to_numpy() == pytest.approx(expected.to_numpy())


def test_string_gold_is_refused_and_the_column_is_named(scores, gold, table):
    text_gold = gold.assign(gold_tone=np.where(gold["gold_tone"] > 2, "high", "low"))
    with pytest.raises(ConfigurationError, match=r"gold_tone.*numeric"):
        validity_panel(scores, text_gold, table, targets=TARGETS, settings=PanelSettings(n_boot=20))


def test_missing_gold_for_five_texts_is_dropped_and_counted(scores, gold, table):
    holes = gold.copy()
    holes["gold_tone"] = holes["gold_tone"].astype(float)
    holes.loc[holes.index[:5], "gold_tone"] = np.nan
    result = validity_panel(
        scores, holes, table, targets=TARGETS, settings=PanelSettings(n_boot=20)
    ).frame.set_index("target")
    assert result.loc["tone", "n_dropped_missing"] == 5
    assert result.loc["tone", "n_used"] == len(gold) - 5
    assert result.loc["tone", "n"] == len(gold) - 5
    assert result.loc["specificity", "n_dropped_missing"] == 0


def test_gold_rows_that_are_absent_are_dropped_like_missing_values(scores, gold, table):
    result = validity_panel(
        scores, gold.iloc[5:], table, targets=TARGETS, settings=PanelSettings(n_boot=20)
    )
    assert (result.frame["n_dropped_missing"] == 5).all()
    assert result.n_dropped_missing == 5
    assert result.n_used == len(gold) - 5


def test_the_panel_counts_are_distinct_texts_over_all_targets(scores, gold, table):
    holes = gold.copy()
    holes["gold_tone"] = holes["gold_tone"].astype(float)
    holes["gold_reasoning"] = holes["gold_reasoning"].astype(float)
    holes.loc[holes.index[:3], "gold_tone"] = np.nan
    holes.loc[holes.index[2:6], "gold_reasoning"] = np.nan
    result = validity_panel(
        scores, holes, table, targets=TARGETS, settings=PanelSettings(n_boot=20)
    )
    assert result.n_dropped_missing == 6
    assert result.n_used == len(gold)


def test_a_text_without_any_attribute_score_is_dropped_from_the_construct(scores, gold, table):
    silent = scores.copy()
    silent.loc[silent["text_id"] == "t001", "score"] = pd.NA
    result = validity_panel(
        silent, gold, table, targets=TARGETS, settings=PanelSettings(n_boot=20)
    ).frame.set_index("target")
    assert result.loc["construct", "n_dropped_missing"] == 1
    assert result.loc["tone", "n_dropped_missing"] == 1


def test_a_negated_gold_is_flipped_under_auto_and_records_the_direction(scores, gold, table):
    negated = gold.assign(gold_constructiveness=6 - gold["gold_constructiveness"])
    settings = PanelSettings(n_boot=20, direction="auto")
    flipped = validity_panel(
        scores, negated, table, targets={"construct": "gold_constructiveness"}, settings=settings
    )
    plain = validity_panel(
        scores, gold, table, targets={"construct": "gold_constructiveness"}, settings=settings
    )
    row = flipped.frame.iloc[0]
    assert row["direction_applied"] == "negative"
    assert row["spearman"] > 0
    assert row["rank_validity"] == pytest.approx(plain.frame.iloc[0]["rank_validity"])
    assert plain.frame.iloc[0]["direction_applied"] == "positive"


def test_a_fixed_direction_is_applied_as_given(scores, gold, table):
    targets = {"construct": "gold_constructiveness"}
    positive = validity_panel(
        scores, gold, table, targets=targets, settings=PanelSettings(n_boot=20)
    )
    negative = validity_panel(
        scores,
        gold,
        table,
        targets=targets,
        settings=PanelSettings(n_boot=20, direction="negative"),
    )
    assert negative.frame.iloc[0]["direction_applied"] == "negative"
    assert negative.frame.iloc[0]["spearman"] == pytest.approx(-positive.frame.iloc[0]["spearman"])
    assert negative.frame.iloc[0]["rank_validity"] == pytest.approx(
        positive.frame.iloc[0]["rank_validity"]
    )


def test_the_report_names_the_construct_and_its_rank_validity(panel):
    text = panel.report()
    rank_validity = panel.frame.set_index("target").loc["construct", "rank_validity"]
    assert "construct" in text
    assert f"{rank_validity:.2f}" in text


def test_the_panel_is_reproducible(scores, gold, table):
    settings = PanelSettings(n_boot=50)
    first = validity_panel(scores, gold, table, targets=TARGETS, settings=settings)
    second = validity_panel(scores, gold, table, targets=TARGETS, settings=settings)
    pd.testing.assert_frame_equal(first.frame, second.frame)


def test_the_panel_is_frozen(panel):
    with pytest.raises(dataclasses.FrozenInstanceError):
        panel.n_used = 3  # type: ignore[misc]


def _run_averaged(scores: pd.DataFrame, gold: pd.DataFrame, table: TextTable):
    """Independently align the run-averaged tone score, its gold value and log characters."""
    tone = scores[scores["attribute_id"] == "tone"]
    per_text = tone.assign(score=tone["score"].astype(float)).groupby("text_id")["score"].mean()
    ids = list(per_text.index)
    gold_values = gold.set_index("text_id")["gold_tone"].loc[ids].to_numpy(dtype=float)
    n_chars = table.frame.set_index("text_id")["n_chars"].loc[ids].to_numpy(dtype=float)
    return per_text.to_numpy(), gold_values, np.log(n_chars)


@pytest.mark.parametrize("direction", ["positive", "negative"])
def test_the_point_estimates_equal_the_metrics_on_independently_aligned_arrays(
    scores, gold, table, direction
):
    averaged, gold_values, covariate = _run_averaged(scores, gold, table)
    signed = -averaged if direction == "negative" else averaged
    panel = validity_panel(
        scores,
        gold,
        table,
        targets={"tone": "gold_tone"},
        settings=PanelSettings(n_boot=20, direction=direction),
    )
    row = panel.frame.iloc[0]
    found = association(signed, gold_values)
    errors = score_errors(signed, gold_values, covariate)
    assert row["rank_validity"] == pytest.approx(found.rank_validity, abs=1e-12)
    assert row["spearman"] == pytest.approx(found.spearman, abs=1e-12)
    assert row["pearson"] == pytest.approx(found.pearson, abs=1e-12)
    assert row["mae_01"] == pytest.approx(errors.mae_01, abs=1e-12)
    assert row["level_displacement"] == pytest.approx(errors.level_displacement, abs=1e-12)
    assert row["calibration_shortfall"] == pytest.approx(errors.calibration_shortfall, abs=1e-12)
    assert row["covariate_linked_error"] == pytest.approx(errors.covariate_linked_error, abs=1e-12)


def test_a_negated_score_changes_the_level_and_the_shortfall_as_computed_on_the_negated_arrays(
    scores, gold, table
):
    averaged, gold_values, covariate = _run_averaged(scores, gold, table)
    positive = score_errors(averaged, gold_values, covariate)
    negative = score_errors(-averaged, gold_values, covariate)
    assert negative.level_displacement != pytest.approx(positive.level_displacement)
    assert negative.calibration_shortfall != pytest.approx(positive.calibration_shortfall)
    panel = validity_panel(
        scores,
        gold,
        table,
        targets={"tone": "gold_tone"},
        settings=PanelSettings(n_boot=20, direction="negative"),
    )
    row = panel.frame.iloc[0]
    assert row["level_displacement"] == pytest.approx(negative.level_displacement, abs=1e-12)
    assert row["calibration_shortfall"] == pytest.approx(negative.calibration_shortfall, abs=1e-12)
