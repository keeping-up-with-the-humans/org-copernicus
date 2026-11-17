"""The validity panel scored once per group equals a panel of one text per group."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from units_support import GOLD, NINE_TEXTS, TARGET, TARGETS, build, group_settings

from copernicus.data import TextTable
from copernicus.validity.panel import PanelSettings, validity_panel

GROUP_SIZES = (1, 2, 3, 1, 4, 2, 3, 1)
SEED = 11
N_BOOT = 200
EXACT_COLUMNS = ("status", "direction_applied", "n", "n_used", "n_boot", "n_undefined_max")
FLOAT_COLUMNS = (
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
    "ceiling",
    "share_of_ceiling",
)


def grouped_world() -> tuple[pd.DataFrame, pd.DataFrame, TextTable]:
    """Groups of unequal size, two runs per text, a covariate that varies within each group."""
    rng = np.random.default_rng(SEED)
    texts, scores, gold = [], [], []
    for number, size in enumerate(GROUP_SIZES, start=1):
        group, value = f"g{number:02d}", float(rng.integers(1, 8))
        for member in range(size):
            text = f"{group}_{member}"
            texts.append(
                {"text_id": text, "text": "word " * (3 + number + member), "group_id": group}
            )
            gold.append({"text_id": text, GOLD: value, "size": number + rng.normal()})
            scores += [
                {
                    "text_id": text,
                    "run_number": run,
                    "attribute_id": TARGET,
                    "score": value + rng.normal(),
                }
                for run in (1, 2)
            ]
    return pd.DataFrame(scores), pd.DataFrame(gold), TextTable.from_frame(pd.DataFrame(texts))


def one_text_per_group(
    scores: pd.DataFrame, gold: pd.DataFrame, table: TextTable
) -> tuple[pd.DataFrame, pd.DataFrame, TextTable]:
    """One text per group, named after it, with the group's mean score, mean covariate and gold.

    The group's score is the plain mean of its texts' run means, computed here without the
    library; the text and its group carry the group's id, so the rows and the clusters of the
    bootstrap come in the order of the groups.
    """
    group_of = dict(zip(table.frame["text_id"], table.frame["group_id"], strict=True))
    run_means = scores.groupby("text_id")["score"].apply(lambda runs: float(np.mean(runs)))
    rows = []
    for group in sorted(set(group_of.values())):
        members = sorted(text for text, owner in group_of.items() if owner == group)
        own = gold.set_index("text_id").loc[members]
        rows.append(
            {
                "text_id": group,
                "score": float(np.mean([run_means[text] for text in members])),
                GOLD: float(own[GOLD].iloc[0]),
                "size": float(np.mean(own["size"].to_numpy())),
            }
        )
    single = pd.DataFrame(rows)
    single_scores = single[["text_id", "score"]].assign(run_number=1, attribute_id=TARGET)
    single_table = TextTable.from_frame(
        single.assign(text="one text", group_id=single["text_id"])[["text_id", "text", "group_id"]]
    )
    return single_scores, single[["text_id", GOLD, "size"]], single_table


@pytest.mark.parametrize("direction", ["positive", "negative"])
def test_group_rows_equal_the_text_rows_of_a_table_with_one_text_per_group(direction):
    scores, gold, table = grouped_world()
    settings = PanelSettings(
        n_boot=N_BOOT, seed=SEED, covariate="size", reliability=0.81, direction=direction
    )
    grouped = replace(settings, unit="group")
    by_group = validity_panel(scores, gold, table, targets=TARGETS, settings=grouped).frame.iloc[0]
    single = one_text_per_group(scores, gold, table)
    by_text = validity_panel(*single, targets=TARGETS, settings=settings).frame.iloc[0]
    assert (by_group["n_texts"], by_text["n_texts"]) == (sum(GROUP_SIZES), len(GROUP_SIZES))
    assert by_group["status"] == "ok" and np.isfinite(by_group["covariate_linked_error_low"])
    for column in EXACT_COLUMNS:
        assert by_group[column] == by_text[column], column
    for column in FLOAT_COLUMNS:
        assert by_group[column] == pytest.approx(by_text[column], abs=1e-12, nan_ok=True), column


def test_group_score_errors_equal_hand_computed_values():
    # Group scores A 2, B 5, C 3, D 4.5, E 1 against gold 10, 30, 25, 20, 5 and group covariate
    # means 1 to 5 (the texts' own values differ within A, C and D). On the 0-1 scale the scores
    # are .25, 1, .5, .875, 0 and the gold values .2, 1, .8, .6, 0, so the differences are .05,
    # 0, -.3, .275, 0: the mean absolute error is .625 / 5 = .125 and the level .025 / 5 = .005.
    # Their cross-product about the means is .61 and the gold's squared deviations .688, so
    # the shortfall is .61 / .688 - 1. The slope of the differences on the covariate is .175 / 10
    # per unit, times the covariate's standard deviation sqrt(2.5) per standard deviation.
    # Kendall's tau-b has one discordant pair (C, D) of ten, and Pearson is 61 / sqrt(11.2 * 430).
    covariate = {"a1": 0.5, "a2": 1.5, "b1": 2.0, "c1": 2.0, "c2": 3.0, "c3": 4.0}
    covariate |= {"d1": 3.0, "d2": 5.0, "e1": 5.0}
    scores, gold, table = build(NINE_TEXTS)
    gold = gold.assign(size=gold["text_id"].map(covariate))
    row = validity_panel(
        scores, gold, table, targets=TARGETS, settings=group_settings(covariate="size")
    ).frame.iloc[0]
    expected = {
        "spearman": 0.9,
        "kendall": 0.8,
        "pearson": 61 / math.sqrt(11.2 * 430),
        "mae_01": 0.125,
        "signed_gap_01": 0.005,
        "level_displacement": 0.005,
        "calibration_shortfall": 0.61 / 0.688 - 1,
        "covariate_linked_error_raw": 0.0175,
        "covariate_linked_error_standardized": 0.0175 * math.sqrt(2.5),
        "covariate_linked_error": 0.0175 * math.sqrt(2.5),
    }
    for column, value in expected.items():
        assert row[column] == pytest.approx(value, abs=1e-12), column
