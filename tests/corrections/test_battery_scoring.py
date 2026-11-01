"""How the battery scores folds: undefined metrics, native errors, signs, leakage, recomputation."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from battery_support import (
    LINEAR,
    RIDGE,
    SMALL,
    HeldOutConstant,
    altered_linear,
    binary_data,
    patch_linear,
)
from corrections_support import BATTERY_SCORES, battery_data, battery_frame, make_embedding_set

from copernicus.baseline.folds import rng_for
from copernicus.corrections import (
    CorrectionData,
    best_score_column,
    get_correction,
    grouped_folds,
    holdout_split,
    scale_test,
)
from copernicus.corrections.battery import (
    RESULT_COLUMNS,
    SKIPPED_COLUMNS,
    BatteryResult,
    BatterySettings,
    run_battery,
    summarize_battery,
)
from copernicus.corrections.battery_rows import keyed_seed
from copernicus.corrections.sampling import draw_groups
from copernicus.validity import association, covariate_slope


def fitted_rows(result: BatteryResult) -> pd.DataFrame:
    return result.results[~result.results["diagnostic"]].reset_index(drop=True)


def test_a_fold_without_a_defined_rank_validity_is_left_out_of_its_means(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_linear(monkeypatch, lambda inner: HeldOutConstant(inner, failing_call=3))
    result = run_battery(binary_data(), BatterySettings(families=[LINEAR, RIDGE], **SMALL))
    linear = fitted_rows(result).iloc[0]
    assert (linear["n_folds"], linear["n_folds_used"]) == (5, 4)
    folds = json.loads(linear["provenance"])
    dropped = [fold for fold in folds if fold["dropped_from"]]
    assert [fold["fold"] for fold in dropped] == [2]
    assert dropped[0]["dropped_from"] == ["rank_validity", "spearman"]
    assert (dropped[0]["status"], dropped[0]["status_raw"]) == ("constant_score", "ok")
    assert dropped[0]["rank_validity_corrected"] is None
    used = [fold for fold in folds if not fold["dropped_from"]]
    weights = np.array([fold["n_test"] for fold in used], dtype=float)
    for side in ("raw", "corrected"):
        values = np.array([fold[f"rank_validity_{side}"] for fold in used])
        assert linear[f"rank_validity_{side}"] == pytest.approx(
            float(np.sum(weights * values) / weights.sum()), abs=1e-12
        )
    all_weights = np.array([fold["n_test"] for fold in folds], dtype=float)
    maes = np.array([fold["mae_corrected"] for fold in folds])
    assert linear["mae_corrected"] == pytest.approx(
        float(np.sum(all_weights * maes) / all_weights.sum()), abs=1e-12
    )
    summary = summarize_battery(result)
    assert np.isfinite(summary[["gain_rank_validity", "gain_mae", "gain_mae_native"]]).all().all()


def test_a_fold_whose_held_out_gold_is_constant_is_skipped_for_every_entry() -> None:
    frame = binary_data().frame.copy()
    folds = grouped_folds(frame["group_id"].to_numpy(), 5, rng_for(42, "kfold5", "partition"))
    frame.loc[folds[1][1], "gold"] = 0.0
    data = CorrectionData(frame=frame, score_columns=BATTERY_SCORES)
    result = run_battery(data, BatterySettings(families=[LINEAR, RIDGE], **SMALL))
    constant = result.skipped[result.skipped["reason"].str.startswith("test_fold_constant_gold: ")]
    assert set(zip(constant["variant"], constant["fold"], strict=True)) == {
        ("linear", 1),
        ("ridge", 1),
    }
    assert list(result.raw["fold"]) == [0, 2, 3, 4]
    assert (fitted_rows(result)[["n_folds", "n_folds_used"]] == 4).all().all()


def test_sparse_binary_gold_gives_finite_rows_and_summaries() -> None:
    families = [
        ("regression_calibration", "isotonic"),
        ("multiple_imputation", "pmm"),
        ("marginal_matching", "quantile"),
    ]
    settings = BatterySettings(families=families, sizes=(50,), schemes=("kfold10",), seeds=(42,))
    result = run_battery(binary_data(share=0.05), settings)
    rows = fitted_rows(result)
    assert len(rows) == 3 and np.isfinite(rows["gain_rank_validity"]).all()
    assert (rows["n_folds_used"] < rows["n_folds"]).any()
    assert np.isfinite(summarize_battery(result)["gain_rank_validity"]).all()


def test_an_affine_calibration_moves_only_the_native_error() -> None:
    data = CorrectionData(frame=battery_frame(), score_columns=("score_b",))
    result = run_battery(data, BatterySettings(families=[LINEAR], **SMALL))
    linear = fitted_rows(result).iloc[0]
    assert abs(linear["gain_mae"]) < 1e-12 and abs(linear["gain_covariate_error"]) < 1e-12
    assert linear["mae_native_raw"] > 5.0 and linear["gain_mae_native"] > 5.0
    assert (result.raw["mae_native_raw"] > 5.0).all()


def test_a_reversing_correction_shows_in_the_signed_spearman(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    altered_linear(monkeypatch, lambda predicted: -predicted)
    result = run_battery(battery_data(), BatterySettings(families=[LINEAR], **SMALL))
    linear = fitted_rows(result).iloc[0]
    assert linear["rank_validity_corrected"] == pytest.approx(linear["rank_validity_raw"])
    assert linear["spearman_raw"] > 0.5
    assert linear["spearman_corrected"] == pytest.approx(-linear["spearman_raw"])


def test_the_summary_averages_the_seed_means() -> None:
    settings = BatterySettings(
        families=[LINEAR], sizes=(50, 100), schemes=("kfold5",), seeds=(1, 2)
    )
    rows = [
        {**dict.fromkeys(RESULT_COLUMNS, np.nan), "family": LINEAR[0], "variant": LINEAR[1]}
        | {"seed": seed, "diagnostic": False, "gain_rank_validity": gain}
        for seed, gain in ((1, 0.1), (1, 0.3), (2, 0.5))
    ]
    result = BatteryResult(
        results=pd.DataFrame(rows, columns=list(RESULT_COLUMNS)),
        skipped=pd.DataFrame(columns=list(SKIPPED_COLUMNS)),
        raw=pd.DataFrame(),
        settings=settings,
    )
    summary = summarize_battery(result).iloc[0]
    assert summary["gain_rank_validity"] == pytest.approx(0.35)
    assert summary["gain_rank_validity_se"] == pytest.approx(
        np.std([0.2, 0.5], ddof=1) / np.sqrt(2)
    )
    assert (summary["n_rows"], summary["n_seeds"], summary["n_skipped"]) == (3, 2, 0)


def test_gold_outside_the_drawn_texts_never_reaches_a_fit() -> None:
    data = battery_data()
    settings = BatterySettings(families=[LINEAR, RIDGE], **SMALL)
    base = run_battery(data, settings)
    first = json.loads(fitted_rows(base).loc[0, "provenance"])[0]
    train_pos, test_pos = grouped_folds(data.groups, 5, rng_for(42, "kfold5", "partition"))[0]
    pool = data.ids[train_pos]
    drawn = set(draw_groups(pool, data.groups[train_pos], pool, 100, seed=first["draw_seed"]).ids)
    undrawn = next(text for text in pool if text not in drawn)
    frame = data.frame.copy()
    frame.loc[frame["text_id"].isin([undrawn, data.ids[test_pos][0]]), "gold"] = 1e6
    moved = run_battery(CorrectionData(frame=frame, score_columns=BATTERY_SCORES), settings)
    for index in range(2):
        before = json.loads(fitted_rows(base).loc[index, "provenance"])[0]
        after = json.loads(fitted_rows(moved).loc[index, "provenance"])[0]
        for key in ("draw_seed", "quota", "pool_units", "raw_column", "fit"):
            assert after[key] == before[key]
        assert after["mae_corrected"] != before["mae_corrected"]
        assert after["rank_validity_corrected"] != before["rank_validity_corrected"]


def test_one_fold_recomputed_by_hand_equals_the_raw_and_results_rows() -> None:
    data = battery_data()
    settings = BatterySettings(families=[LINEAR], sizes=(80,), schemes=("holdout",), seeds=(42,))
    result = run_battery(data, settings)
    ids, groups, gold = data.ids, data.groups, data.gold
    train_pos, test_pos = holdout_split(groups, rng_for(42, "holdout", "partition"))
    seed = keyed_seed(42, "holdout", "80", 0, "draw")
    assert seed == int(rng_for(42, "holdout", "80", 0, "draw").integers(2**31 - 1))
    draw = draw_groups(ids[train_pos], groups[train_pos], ids[train_pos], 80, seed=seed)
    drawn = train_pos[np.isin(ids[train_pos], draw.ids)]
    only_drawn = np.where(np.isin(np.arange(data.n), drawn), gold, np.nan)
    labelled = CorrectionData(
        frame=data.frame.assign(gold=only_drawn, pi=draw.pi), score_columns=BATTERY_SCORES
    )
    rng = rng_for(42, "holdout", "80", 0, LINEAR[0])
    fitted = get_correction(*LINEAR).fit(labelled, ids[drawn], rng=rng)
    corrected_train = fitted.predict(labelled, ids[drawn]).to_numpy()
    corrected_test = fitted.predict(labelled, ids[test_pos]).to_numpy()
    raw_column = best_score_column(data.scores.iloc[drawn].reset_index(drop=True), gold[drawn])
    raw_values = data.frame[raw_column].to_numpy()

    def scores(train: np.ndarray, test: np.ndarray) -> list[float]:
        found = association(test, gold[test_pos])
        pred01, gold01 = scale_test(train, gold[drawn], test, gold[test_pos])
        difference = pred01 - gold01
        return [
            found.rank_validity,
            found.spearman,
            float(np.mean(np.abs(difference))),
            float(np.mean(np.abs(test - gold[test_pos]))),
            covariate_slope(difference, data.covariate[test_pos]),
        ]

    names = ("rank_validity", "spearman", "mae", "mae_native", "covariate_error")
    raw = dict(zip(names, scores(raw_values[drawn], raw_values[test_pos]), strict=True))
    corrected = dict(zip(names, scores(corrected_train, corrected_test), strict=True))
    raw_row, row = result.raw.iloc[0], fitted_rows(result).iloc[0]
    assert raw_row["raw_column"] == raw_column and row["raw_columns"] == raw_column
    assert raw_row["n_test"] == test_pos.size
    for name in names:
        assert raw_row[f"{name}_raw"] == pytest.approx(raw[name], abs=1e-12)
        assert row[f"{name}_raw"] == pytest.approx(raw[name], abs=1e-12)
        assert row[f"{name}_corrected"] == pytest.approx(corrected[name], abs=1e-12)
    assert row["gain_mae_native"] == pytest.approx(
        raw["mae_native"] - corrected["mae_native"], abs=1e-12
    )
    assert json.loads(row["provenance"])[0]["draw_seed"] == seed


def test_binary_gold_lets_platt_the_matrix_and_the_mode_ensemble_apply() -> None:
    families = [
        ("regression_calibration", "platt"),
        ("misclassification_matrix", "posterior"),
        LINEAR,
        ("correction_ensemble", "mode"),
    ]
    result = run_battery(binary_data(), BatterySettings(families=families, **SMALL))
    rows = fitted_rows(result)
    assert list(rows["variant"]) == ["platt", "posterior", "linear", "mode"]
    assert result.skipped.empty
    assert (rows["n_folds_used"] == 5).all() and np.isfinite(rows["gain_rank_validity"]).all()
    mode = json.loads(rows.loc[3, "provenance"])[0]["fit"]
    assert len(mode["member_families"]) == 3


def _small_world(names: tuple[str, ...]) -> CorrectionData:
    """120 texts in 30 groups with one informative and two noise scores and embedding sets.

    The coordinate descent of the lasso and elastic-net stackers converges on this
    near-orthogonal design, so no ConvergenceWarning stops a run under -W error (three
    collinear scores do not converge on 24 drawn texts).
    """
    frame = battery_frame(n_groups=30, group_size=4, seed=5)
    rng = np.random.default_rng(105)
    n = len(frame)
    frame["score_b"], frame["score_c"] = rng.normal(size=n), rng.normal(size=n)
    gold = frame["gold"].to_numpy()
    sets = {
        name: make_embedding_set(
            frame["text_id"].tolist(),
            gold[:, None] * rng.normal(size=4) + rng.normal(size=(n, 4)),
            name,
        )
        for name in names
    }
    return CorrectionData(frame=frame, score_columns=BATTERY_SCORES, embeddings=sets or None)


@pytest.mark.slow
def test_all_runs_the_embedding_family_on_the_one_set_of_the_data() -> None:
    settings = BatterySettings(sizes=(30,), schemes=("kfold5",), seeds=(42,))
    result = run_battery(_small_world(("e1",)), settings)
    labels = [e.label for e in result.settings.entries if e.family == "embedding_augmented"]
    assert labels == [
        "embeddings_only(encoder=e1)",
        "embeddings_plus_score(encoder=e1)",
        "score_anchored_residual(encoder=e1)",
    ]
    scored = fitted_rows(result)
    embedding = scored[scored["family"] == "embedding_augmented"]
    assert list(embedding["variant"]) == labels and (embedding["n_folds"] == 5).all()
    assert result.skipped[result.skipped["family"] == "embedding_augmented"].empty
    assert len(summarize_battery(result)) == len(result.settings.entries) == 22


@pytest.mark.slow
def test_every_family_runs_under_all_on_a_small_world() -> None:
    data = _small_world(("e1", "e2"))
    settings = BatterySettings(sizes=(30,), schemes=("kfold5",), seeds=(42,))
    result = run_battery(data, settings)
    scored = {f"{row.family}:{row.variant}" for row in fitted_rows(result).itertuples()}
    skipped = result.skipped.assign(key=result.skipped["family"] + ":" + result.skipped["variant"])
    keys = [entry.key for entry in settings.entries]
    assert len(keys) == 22 and set(keys) == scored | set(skipped["key"])
    reasons = skipped.groupby("key")["reason"].first()
    assert reasons["regression_calibration:platt"].startswith("not_applicable: ")
    assert reasons["misclassification_matrix:posterior"].startswith("not_applicable: ")
    assert reasons["correction_ensemble:mode"].startswith("NotApplicableError: ")
    assert len(scored) == 19 and (fitted_rows(result)["n_folds"] == 5).all()
    assert len(summarize_battery(result)) == 22
