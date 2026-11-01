"""The correction battery: folds, draws, fits, metrics, skip rows and the summary."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from corrections_support import BATTERY_SCORES, battery_data, battery_frame

from copernicus.corrections import CorrectionData
from copernicus.corrections.battery import (
    RAW_COLUMNS,
    RESULT_COLUMNS,
    SKIPPED_COLUMNS,
    SUMMARY_COLUMNS,
    BatteryResult,
    BatterySettings,
    run_battery,
    summarize_battery,
)
from copernicus.errors import ConfigurationError
from copernicus.validity import association

MAIN_FAMILIES = [
    ("regression_calibration", "linear"),
    ("supervised_stacking", "ridge"),
    ("correction_ensemble", "mean"),
]
SMALL = {"sizes": (100,), "schemes": ("kfold5",), "seeds": (42,)}


@pytest.fixture(scope="module")
def main_result() -> BatteryResult:
    """The brief's battery: three entries, size 100, five grouped folds, one seed."""
    return run_battery(battery_data(), BatterySettings(families=MAIN_FAMILIES, **SMALL))


def fitted_rows(result: BatteryResult) -> pd.DataFrame:
    return result.results[~result.results["diagnostic"]].reset_index(drop=True)


def test_one_row_per_entry_with_a_monotone_line_and_a_gaining_stacker(
    main_result: BatteryResult,
) -> None:
    rows = fitted_rows(main_result)
    assert list(zip(rows["family"], rows["variant"], strict=True)) == [
        (family, variant) for family, variant in MAIN_FAMILIES
    ]
    linear, ridge, ensemble = (rows.iloc[index] for index in range(3))
    assert abs(linear["gain_rank_validity"]) <= 0.01
    assert ridge["gain_rank_validity"] > 0.0
    assert np.isfinite(ensemble["rank_validity_corrected"])
    assert (rows["n_folds"] == 5).all()
    assert (rows["n_labelled"] == 100).all()
    assert (
        rows[["seed", "scheme", "size"]].astype(str).agg("/".join, axis=1) == "42/kfold5/100"
    ).all()
    assert main_result.skipped.empty


def test_gains_are_the_differences_of_the_fold_weighted_metrics(
    main_result: BatteryResult,
) -> None:
    rows = fitted_rows(main_result)
    for row in rows.iloc[:2].itertuples(index=False):
        assert row.gain_rank_validity == pytest.approx(
            row.rank_validity_corrected - row.rank_validity_raw, abs=1e-12
        )
        assert row.gain_mae == pytest.approx(row.mae_raw - row.mae_corrected, abs=1e-12)
        assert row.gain_covariate_error == pytest.approx(
            abs(row.covariate_error_raw) - abs(row.covariate_error_corrected), abs=1e-12
        )
    raw = main_result.raw
    weights = raw["n_test"].to_numpy(dtype=float)
    weighted = float(np.sum(weights * raw["rank_validity_raw"]) / weights.sum())
    assert rows.loc[0, "rank_validity_raw"] == pytest.approx(weighted, abs=1e-12)


def test_the_ensemble_is_rank_only(main_result: BatteryResult) -> None:
    ensemble = fitted_rows(main_result).iloc[2]
    assert np.isfinite(ensemble["gain_rank_validity"])
    assert np.isnan(ensemble["mae_corrected"]) and np.isnan(ensemble["gain_mae"])
    assert np.isnan(ensemble["covariate_error_corrected"])
    assert "NaN" not in ensemble["provenance"]
    folds = json.loads(ensemble["provenance"])
    assert len(folds) == 5 and folds[0]["mae_corrected"] is None
    assert set(folds[0]["fit"]["member_families"]) == {
        "regression_calibration:linear",
        "supervised_stacking:ridge",
    }


def test_the_tables_have_the_documented_columns(main_result: BatteryResult) -> None:
    assert list(main_result.results.columns) == list(RESULT_COLUMNS)
    assert list(main_result.skipped.columns) == list(SKIPPED_COLUMNS)
    assert list(main_result.raw.columns) == list(RAW_COLUMNS)
    assert list(RESULT_COLUMNS) == [
        *("family", "variant", "seed", "scheme", "size", "n_labelled", "n_folds", "n_folds_used"),
        *("rank_validity_raw", "rank_validity_corrected", "spearman_raw", "spearman_corrected"),
        *("gain_rank_validity", "mae_raw", "mae_corrected", "gain_mae"),
        *("mae_native_raw", "mae_native_corrected", "gain_mae_native"),
        *("covariate_error_raw", "covariate_error_corrected", "gain_covariate_error"),
        *("raw_columns", "diagnostic", "provenance"),
    ]
    assert list(SUMMARY_COLUMNS) == [
        *("family", "variant"),
        *("gain_rank_validity", "gain_rank_validity_se", "n_defined_gain_rank_validity"),
        *("gain_mae", "gain_mae_se", "n_defined_gain_mae"),
        *("gain_mae_native", "gain_mae_native_se", "n_defined_gain_mae_native"),
        *("gain_covariate_error", "gain_covariate_error_se", "n_defined_gain_covariate_error"),
        *("n_rows", "n_seeds", "n_skipped"),
    ]
    assert (main_result.results["n_folds_used"] == main_result.results["n_folds"]).all()
    assert list(SKIPPED_COLUMNS) == [
        "family",
        "variant",
        "seed",
        "scheme",
        "size",
        "fold",
        "reason",
    ]
    assert list(RAW_COLUMNS) == [
        *("seed", "scheme", "size", "fold", "raw_column", "n_test"),
        *("rank_validity_raw", "spearman_raw", "mae_raw", "mae_native_raw", "covariate_error_raw"),
    ]
    assert main_result.results["size"].map(type).eq(str).all()
    assert main_result.results["diagnostic"].dtype == bool


def test_the_raw_table_holds_one_row_per_fold_with_the_chosen_column(
    main_result: BatteryResult,
) -> None:
    raw = main_result.raw
    assert list(raw["fold"]) == [0, 1, 2, 3, 4]
    assert set(raw["raw_column"]) <= set(BATTERY_SCORES)
    assert raw["n_test"].sum() == 600
    linear = fitted_rows(main_result).iloc[0]
    assert linear["raw_columns"] == ";".join(raw["raw_column"])
    data = battery_data()
    fold = json.loads(linear["provenance"])[0]
    assert fold["fit"]["score_column"] == fold["raw_column"]
    assert fold["n_labelled"] == 100 and fold["pi"] == pytest.approx(25 / 120)
    assert fold["quota"] == 100 and fold["pool_units"] == 480
    assert raw.loc[0, "rank_validity_raw"] == pytest.approx(fold["rank_validity_raw"])
    assert data.n == 600


def test_the_pseudo_value_diagnostic_row_uses_every_text(main_result: BatteryResult) -> None:
    diagnostic = main_result.results[main_result.results["diagnostic"]]
    assert len(diagnostic) == 1
    row = diagnostic.iloc[0]
    assert (row["family"], row["variant"], row["scheme"], row["size"]) == (
        "pseudo_values",
        "identity",
        "whole_data",
        "100",
    )
    data = battery_data()
    raw = association(data.frame["score_a"].to_numpy(), data.gold).rank_validity
    assert row["rank_validity_raw"] == pytest.approx(raw)
    assert row["raw_columns"] == "score_a"
    assert np.isfinite(row["rank_validity_corrected"])
    assert row["gain_rank_validity"] == pytest.approx(
        row["rank_validity_corrected"] - row["rank_validity_raw"]
    )
    for column in ("mae_raw", "mae_corrected", "gain_mae", "gain_covariate_error"):
        assert np.isnan(row[column])
    assert row["n_labelled"] == 100 and row["n_folds"] == 1


def test_partial_gold_skips_the_diagnostic_once_per_seed_and_size() -> None:
    data = battery_data(battery_frame(label_all=False))
    settings = BatterySettings(
        families=[("regression_calibration", "linear")],
        sizes=(60, 0.1),
        schemes=("kfold5",),
        seeds=(42, 43),
    )
    result = run_battery(data, settings)
    assert not result.results["diagnostic"].any()
    skipped = result.skipped[result.skipped["family"] == "pseudo_values"]
    assert list(zip(skipped["seed"], skipped["size"], strict=True)) == [(42, "60"), (43, "60")]
    assert skipped["reason"].str.startswith("diagnostic_needs_full_gold").all()
    assert (skipped["scheme"] == "whole_data").all() and (skipped["fold"] == -1).all()
    pools = [json.loads(text)[0]["pool_units"] for text in result.results["provenance"]]
    assert all(pool < 480 for pool in pools)


def test_a_size_above_half_the_pool_is_skipped_for_every_fold() -> None:
    settings = BatterySettings(
        families=[("regression_calibration", "linear")],
        sizes=(400, 100),
        schemes=("kfold5",),
        seeds=(42,),
    )
    result = run_battery(battery_data(), settings)
    rows = fitted_rows(result)
    assert list(rows["size"]) == ["100"]
    over = result.skipped[result.skipped["size"] == "400"]
    linear = over[over["family"] == "regression_calibration"]
    assert list(linear["fold"]) == [0, 1, 2, 3, 4]
    assert linear["reason"].str.startswith("quota_exceeds_half_units").all()
    diagnostic = over[over["family"] == "pseudo_values"]
    assert diagnostic["reason"].str.startswith("quota_exceeds_half_units").all()
    assert set(result.raw["size"]) == {"100"}


def test_a_share_size_draws_its_quota_from_the_pool() -> None:
    settings = BatterySettings(
        families=[("regression_calibration", "linear")],
        sizes=(0.25, 0.001),
        schemes=("kfold5",),
        seeds=(42,),
    )
    result = run_battery(battery_data(), settings)
    row = fitted_rows(result).iloc[0]
    assert row["size"] == "0.25"
    assert [fold["quota"] for fold in json.loads(row["provenance"])] == [120] * 5
    small = result.skipped[result.skipped["size"] == "0.001"]
    assert len(small) == 5 and small["reason"].str.startswith("quota_below_2").all()


def test_a_constant_score_column_does_not_crash_the_battery() -> None:
    frame = battery_frame()
    frame["score_flat"] = 3.0
    data = CorrectionData(frame=frame, score_columns=(*BATTERY_SCORES, "score_flat"))
    result = run_battery(data, BatterySettings(families=MAIN_FAMILIES, **SMALL))
    assert len(fitted_rows(result)) == 3
    assert "score_flat" not in set(result.raw["raw_column"])


def test_a_single_constant_score_column_skips_every_fold_with_the_reason() -> None:
    frame = battery_frame()
    frame["score_flat"] = 3.0
    data = CorrectionData(frame=frame, score_columns=("score_flat",))
    settings = BatterySettings(families=[("regression_calibration", "linear")], **SMALL)
    result = run_battery(data, settings)
    assert fitted_rows(result).empty and result.raw.empty
    linear = result.skipped[result.skipped["family"] == "regression_calibration"]
    assert len(linear) == 5
    assert linear["reason"].str.startswith("EstimatorError: no usable score column").all()


def test_holdout_outside_the_band_is_recorded_for_every_entry_and_size() -> None:
    frame = battery_frame(n_groups=2, group_size=60)
    settings = BatterySettings(
        families=[("regression_calibration", "linear"), ("regression_calibration", "isotonic")],
        sizes=(10, 20),
        schemes=("holdout",),
        seeds=(42,),
    )
    result = run_battery(battery_data(frame), settings)
    band = result.skipped[result.skipped["scheme"] == "holdout"]
    assert len(band) == 4 and (band["fold"] == -1).all()
    assert band["reason"].str.startswith("holdout_share_outside_band").all()
    assert fitted_rows(result).empty
    assert result.results["diagnostic"].all() and len(result.results) == 2


def test_a_held_out_fold_without_gold_is_skipped_for_every_entry() -> None:
    frame = battery_frame(n_groups=10, group_size=12)
    frame.loc[~frame["group_id"].isin(["g000", "g001"]), "gold"] = np.nan
    settings = BatterySettings(
        families=[("regression_calibration", "linear"), ("regression_calibration", "isotonic")],
        sizes=(4,),
        schemes=("kfold5",),
        seeds=(42,),
    )
    result = run_battery(battery_data(frame), settings)
    empty = result.skipped[result.skipped["reason"].str.startswith("test_fold_too_few_pairs: ")]
    folds = sorted(set(empty["fold"]))
    assert len(folds) >= 3
    for fold in folds:
        assert set(empty.loc[empty["fold"] == fold, "variant"]) == {"linear", "isotonic"}
    assert set(result.raw["fold"]).isdisjoint(folds)


def test_fewer_groups_than_folds_is_refused_before_any_fit() -> None:
    frame = battery_frame(n_groups=4, group_size=30)
    settings = BatterySettings(
        families=[("regression_calibration", "linear")], sizes=(20,), schemes=("kfold5",)
    )
    with pytest.raises(ConfigurationError, match="kfold5 needs at least 5 groups"):
        run_battery(battery_data(frame), settings)


def test_data_without_gold_are_refused() -> None:
    frame = battery_frame()
    frame["gold"] = np.nan
    settings = BatterySettings(families=[("regression_calibration", "linear")], **SMALL)
    with pytest.raises(ConfigurationError, match="no text of the data has a gold value"):
        run_battery(battery_data(frame), settings)


def test_the_battery_is_reproducible() -> None:
    settings = BatterySettings(
        families=[("supervised_stacking", "ridge")], sizes=(80,), schemes=("holdout",), seeds=(7,)
    )
    first = run_battery(battery_data(), settings)
    second = run_battery(battery_data(), settings)
    pd.testing.assert_frame_equal(first.results, second.results)
    pd.testing.assert_frame_equal(first.raw, second.raw)


def test_summary_has_one_row_per_entry_with_seed_clustered_errors(
    main_result: BatteryResult,
) -> None:
    summary = summarize_battery(main_result)
    assert list(summary.columns) == list(SUMMARY_COLUMNS)
    assert list(zip(summary["family"], summary["variant"], strict=True)) == MAIN_FAMILIES
    assert (summary["n_rows"] == 1).all() and (summary["n_seeds"] == 1).all()
    assert summary["gain_rank_validity_se"].isna().all()
    rows = fitted_rows(main_result)
    assert summary["gain_rank_validity"].tolist() == pytest.approx(
        rows["gain_rank_validity"].tolist()
    )
    assert (summary["n_skipped"] == 0).all()


def test_summary_errors_come_from_the_seed_means() -> None:
    settings = BatterySettings(
        families=[("regression_calibration", "linear"), ("supervised_stacking", "ridge")],
        sizes=(60, 400),
        schemes=("holdout",),
        seeds=(42, 43, 44),
    )
    result = run_battery(battery_data(), settings)
    summary = summarize_battery(result).set_index("variant")
    rows = fitted_rows(result)
    ridge = rows[rows["variant"] == "ridge"]
    seed_means = ridge.groupby("seed")["gain_rank_validity"].mean()
    assert summary.loc["ridge", "gain_rank_validity"] == pytest.approx(
        ridge["gain_rank_validity"].mean()
    )
    assert summary.loc["ridge", "gain_rank_validity_se"] == pytest.approx(
        seed_means.std(ddof=1) / np.sqrt(3)
    )
    assert summary.loc["ridge", "n_seeds"] == 3
    skipped = result.skipped
    assert summary.loc["ridge", "n_skipped"] == int(
        ((skipped["family"] == "supervised_stacking") & (skipped["variant"] == "ridge")).sum()
    )
    assert summary.loc["ridge", "n_skipped"] == 3


def test_the_summary_runs_over_the_rows_where_a_gain_is_defined() -> None:
    settings = BatterySettings(
        families=[("regression_calibration", "linear"), ("supervised_stacking", "ridge")],
        sizes=(100,),
        schemes=("kfold5",),
        seeds=(42, 43),
    )
    gains = {
        "gain_rank_validity": [0.1, np.nan, 0.3, 0.5],
        "gain_mae": [np.nan, np.nan, np.nan, np.nan],
        "gain_mae_native": [1.0, 2.0, 3.0, 4.0],
        "gain_covariate_error": [0.2, 0.4, np.nan, 0.8],
    }
    results = pd.DataFrame(
        {
            "family": ["regression_calibration"] * 4,
            "variant": ["linear"] * 4,
            "seed": [42, 42, 43, 43],
            "diagnostic": [False] * 4,
            **gains,
        }
    )
    skipped = pd.DataFrame(columns=list(SKIPPED_COLUMNS))
    summary = summarize_battery(
        BatteryResult(results=results, skipped=skipped, raw=pd.DataFrame(), settings=settings)
    ).set_index("variant")
    linear = summary.loc["linear"]
    assert linear["n_defined_gain_rank_validity"] == 3
    assert linear["gain_rank_validity"] == pytest.approx((0.1 + 0.4) / 2)
    assert linear["gain_rank_validity_se"] == pytest.approx(abs(0.4 - 0.1) / 2)
    assert linear["n_defined_gain_mae"] == 0 and np.isnan(linear["gain_mae"])
    assert np.isnan(linear["gain_mae_se"])
    assert linear["n_defined_gain_mae_native"] == 4
    assert linear["gain_mae_native"] == pytest.approx(2.5)
    assert linear["n_defined_gain_covariate_error"] == 3
    assert linear["gain_covariate_error"] == pytest.approx((0.3 + 0.8) / 2)
    assert linear["n_rows"] == 4 and linear["n_seeds"] == 2
    ridge = summary.loc["ridge"]
    assert ridge["n_rows"] == 0 and ridge["n_defined_gain_rank_validity"] == 0
    assert np.isnan(ridge["gain_rank_validity"])
