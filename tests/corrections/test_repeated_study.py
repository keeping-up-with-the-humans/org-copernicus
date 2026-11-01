"""The repeated-study check: bias and coverage of each estimator over many draws."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest
from estimates_support import grouped_frame

import copernicus.corrections
import copernicus.corrections.study
from copernicus.baseline.folds import rng_for
from copernicus.corrections import (
    CorrectionData,
    SampleDraw,
    attach_draw,
    draw_groups,
    groups_for_quota,
    long_short_split,
    mean_ppi,
    population_hash_of,
    repeated_study,
)
from copernicus.corrections.families._fitted import SEED_BOUND
from copernicus.errors import ConfigurationError, EstimatorError, SettingsError

pytestmark = pytest.mark.filterwarnings("error")

COLUMNS = [
    "estimator",
    "size",
    "estimand",
    "truth",
    "estimate_mean",
    "bias",
    "coverage",
    "se_mean",
    "n_draws",
    "n_undefined",
    "status",
]
ESTIMATORS = ("raw", "calibrated", "ppi_identity", "ppi_crossfit")
ESTIMANDS = ("mean", "long_short_difference")


def _full_data(seed: int = 0, *, n_groups: int = 60, group_size: int = 2) -> CorrectionData:
    frame = grouped_frame(np.random.default_rng(seed), n_groups=n_groups, group_size=group_size)
    return CorrectionData(frame=frame, score_columns=("score", "noise"))


def test_the_study_has_one_row_per_estimator_size_and_estimand() -> None:
    data = _full_data()
    table = repeated_study(data, truth_gold="truth", sizes=(20, 40), draws=40, seed=3)
    assert list(table.columns) == COLUMNS
    expected = [(e, s, k) for e in ESTIMATORS for s in (20, 40) for k in ESTIMANDS]
    assert list(zip(table["estimator"], table["size"], table["estimand"], strict=True)) == expected
    truth = data.frame["truth"].to_numpy()
    split = long_short_split(data)
    truths = {
        "mean": truth.mean(),
        "long_short_difference": truth[split].mean() - truth[~split].mean(),
    }
    for row in table.itertuples():
        assert row.truth == pytest.approx(truths[row.estimand], abs=1e-12)
        assert row.bias == pytest.approx(row.estimate_mean - row.truth, abs=1e-12)
        assert 0.0 <= row.coverage <= 1.0
        assert row.se_mean > 0.0
        assert (row.n_draws, row.n_undefined, row.status) == (40, 0, "ok")
    score = data.frame["score"].to_numpy()
    raw = table[table["estimator"] == "raw"].set_index(["size", "estimand"])
    assert raw.loc[(20, "mean"), "estimate_mean"] == pytest.approx(score.mean(), abs=1e-12)
    assert raw.loc[(40, "mean"), "bias"] == pytest.approx(raw.loc[(20, "mean"), "bias"])
    assert raw.loc[(20, "mean"), "coverage"] in (0.0, 1.0)
    design_based = table[table["estimator"].isin(["ppi_identity", "ppi_crossfit"])]
    assert (design_based["bias"].abs() < 0.15).all()
    calibrated = table[table["estimator"] == "calibrated"].set_index(["size", "estimand"])
    assert (calibrated.xs("mean", level="estimand")["bias"].abs() < 0.15).all()
    assert (raw["bias"].abs() > 0.3).all()


def test_the_draws_follow_the_documented_scheme() -> None:
    data = _full_data(n_groups=30)
    size, seed, draws = 12, 4, 6
    truth = data.frame["truth"]
    points, errors = [], []
    for index in range(draws):
        draw_seed = int(
            rng_for(seed, "repeated_study", size, index).integers(SEED_BOUND, size=2)[0]
        )
        m = groups_for_quota(size, data.groups, data.ids)
        rng = np.random.default_rng(draw_seed)
        drawn = np.sort(rng.choice(np.unique(data.groups), size=m, replace=False))
        in_draw = np.isin(data.groups, drawn)
        sample = SampleDraw(
            ids=data.ids[in_draw],
            groups=drawn,
            pi=m / 30,
            m=m,
            n_groups=30,
            n_units=data.n,
            quota=size,
            seed=draw_seed,
            population_hash=population_hash_of(data.ids, data.groups),
        )
        helper = draw_groups(data.ids, data.groups, data.ids, size, seed=draw_seed)
        assert helper.groups.tolist() == drawn.tolist() and helper.seed == draw_seed
        estimate = mean_ppi(attach_draw(data, pd.Series(truth.to_numpy(), index=data.ids), sample))
        points.append(estimate.point)
        errors.append(estimate.se)
    table = repeated_study(
        data,
        truth_gold="truth",
        sizes=(size,),
        draws=draws,
        seed=seed,
        estimators=("ppi_identity",),
    )
    row = table[table["estimand"] == "mean"].iloc[0]
    true_mean = float(truth.mean())
    lows = np.array(points) - 1.96 * np.array(errors)
    highs = np.array(points) + 1.96 * np.array(errors)
    assert row["estimate_mean"] == pytest.approx(np.mean(points), abs=1e-12)
    assert row["se_mean"] == pytest.approx(np.mean(errors), abs=1e-12)
    assert row["coverage"] == pytest.approx(np.mean((lows <= true_mean) & (true_mean <= highs)))
    shifted = data.frame.assign(score=data.frame["truth"] + 1.5 * float(truth.std()) / data.n**0.5)
    raw = repeated_study(
        CorrectionData(frame=shifted, score_columns=("score",)),
        truth_gold="truth",
        sizes=(size,),
        draws=2,
        estimators=("raw",),
    )
    assert raw.loc[raw["estimand"] == "mean", "coverage"].tolist() == [1.0]


def test_the_full_size_reproduces_the_truth_exactly_with_ppi() -> None:
    data = _full_data(n_groups=20)
    table = repeated_study(
        data, truth_gold="truth", sizes=(40,), draws=2, estimators=("ppi_identity",)
    )
    assert table["bias"].abs().max() < 1e-12
    assert (table["coverage"] == 1.0).all()


def test_the_study_is_reproducible_for_a_seed() -> None:
    data = _full_data()
    first = repeated_study(data, truth_gold="truth", sizes=(20,), draws=5, seed=9)
    again = repeated_study(data, truth_gold="truth", sizes=(20,), draws=5, seed=9)
    other = repeated_study(data, truth_gold="truth", sizes=(20,), draws=5, seed=10)
    pd.testing.assert_frame_equal(first, again)
    assert not first["estimate_mean"].equals(other["estimate_mean"])


def test_rare_undefined_draws_are_counted() -> None:
    frame = grouped_frame(np.random.default_rng(1), n_groups=50, group_size=2)
    frame.loc[:19, "score"] = 3.0
    data = CorrectionData(frame=frame, score_columns=("score", "noise"))
    table = repeated_study(
        data, truth_gold="truth", sizes=(4,), draws=100, seed=2, estimators=("raw", "calibrated")
    )
    counts = table.set_index(["estimator", "estimand"])["n_undefined"]
    assert counts.loc[("raw", "mean")] == 0
    assert 1 <= counts.loc[("calibrated", "mean")] <= 10
    assert counts.loc[("calibrated", "mean")] == counts.loc[("calibrated", "long_short_difference")]
    assert (table["n_draws"] == 100).all() and (table["status"] == "ok").all()


def test_a_row_undefined_on_more_than_a_tenth_of_draws_is_reported_and_the_rest_kept(
    caplog: pytest.LogCaptureFixture,
) -> None:
    data = _full_data(n_groups=60, group_size=2)
    with caplog.at_level(logging.WARNING, logger="copernicus.corrections.study"):
        table = repeated_study(
            data, truth_gold="truth", sizes=(4, 60), draws=20, estimators=("raw", "ppi_crossfit")
        )
    rows = table.set_index(["estimator", "size", "estimand"])
    summaries = ["estimate_mean", "bias", "coverage", "se_mean"]
    for estimand in ESTIMANDS:
        small = rows.loc[("ppi_crossfit", 4, estimand)]
        assert small["status"] == "undefined_share_exceeded" and small["n_undefined"] > 2
        assert small[summaries].isna().all() and small["truth"] == small["truth"]
        assert rows.loc[("ppi_crossfit", 60, estimand), "status"] == "ok"
        assert np.isfinite(rows.loc[("ppi_crossfit", 60, estimand), summaries].astype(float)).all()
    assert (table.loc[table["estimator"] == "raw", "status"] == "ok").all()
    warned = [record.getMessage() for record in caplog.records]
    assert len(warned) == 2 and all("ppi_crossfit" in text and "4" in text for text in warned)


def test_the_study_stops_only_when_every_row_is_undefined() -> None:
    data = _full_data(n_groups=4, group_size=10)
    table = repeated_study(
        data, truth_gold="truth", sizes=(10,), draws=5, estimators=("raw", "ppi_crossfit")
    )
    crossfit = table[table["estimator"] == "ppi_crossfit"]
    assert (crossfit["status"] == "undefined_share_exceeded").all()
    assert (crossfit["n_undefined"] == 5).all()
    with pytest.raises(EstimatorError, match=r"every row.*inner_groups_below_k"):
        repeated_study(data, truth_gold="truth", sizes=(10,), draws=5, estimators=("ppi_crossfit",))


def test_the_cluster_option_reaches_the_prediction_powered_estimates() -> None:
    lonely = _full_data(n_groups=8, group_size=10)
    with pytest.raises(EstimatorError, match=r"every row.*two groups.*cluster=False"):
        repeated_study(
            lonely, truth_gold="truth", sizes=(5,), draws=20, estimators=("ppi_identity",)
        )
    independent = repeated_study(
        lonely,
        truth_gold="truth",
        sizes=(5,),
        draws=20,
        estimators=("ppi_identity",),
        cluster=False,
    )
    mean_row = independent.set_index("estimand").loc["mean"]
    assert (mean_row["status"], mean_row["n_undefined"]) == ("ok", 0)
    data = _full_data()
    options = {"truth_gold": "truth", "sizes": (20,), "draws": 10, "estimators": ("ppi_identity",)}
    automatic = repeated_study(data, **options)  # type: ignore[arg-type]
    pd.testing.assert_frame_equal(automatic, repeated_study(data, **options, cluster=True))  # type: ignore[arg-type]
    pd.testing.assert_frame_equal(automatic, repeated_study(data, **options, cluster=np.True_))  # type: ignore[arg-type]
    plain = repeated_study(data, **options, cluster=False)  # type: ignore[arg-type]
    assert plain["estimate_mean"].tolist() == automatic["estimate_mean"].tolist()
    assert (plain["se_mean"] != automatic["se_mean"]).all()
    with pytest.raises(SettingsError, match="cluster"):
        repeated_study(data, **options, cluster=1)  # type: ignore[arg-type]


def test_sizes_draws_seed_and_estimators_are_checked() -> None:
    data = _full_data()
    for sizes in ((1,), (121,), (), ("10",), (True,), (10, 10)):
        with pytest.raises(SettingsError, match="size"):
            repeated_study(data, truth_gold="truth", sizes=sizes, draws=2)  # type: ignore[arg-type]
    with pytest.raises(SettingsError, match="draws"):
        repeated_study(data, truth_gold="truth", sizes=(10,), draws=0)
    with pytest.raises(SettingsError, match="seed"):
        repeated_study(data, truth_gold="truth", sizes=(10,), draws=2, seed=-1)
    for estimators in (("raw", "bogus"), ("raw", "raw"), (), "raw"):
        with pytest.raises(SettingsError, match="estimators"):
            repeated_study(data, truth_gold="truth", sizes=(10,), draws=2, estimators=estimators)


def test_the_truth_column_must_be_complete_and_numeric() -> None:
    data = _full_data()
    with pytest.raises(ConfigurationError, match="'nope' is not a column"):
        repeated_study(data, truth_gold="nope", sizes=(10,), draws=2)
    holes = data.frame.assign(truth=np.r_[np.nan, data.frame["truth"].to_numpy()[1:]])
    with pytest.raises(ConfigurationError, match="every text"):
        repeated_study(
            CorrectionData(frame=holes, score_columns=data.score_columns),
            truth_gold="truth",
            sizes=(10,),
            draws=2,
        )
    words = data.frame.assign(truth="high")
    with pytest.raises(ConfigurationError, match="numeric"):
        repeated_study(
            CorrectionData(frame=words, score_columns=data.score_columns),
            truth_gold="truth",
            sizes=(10,),
            draws=2,
        )


def test_a_covariate_without_two_halves_is_refused() -> None:
    data = _full_data()
    flat = CorrectionData(frame=data.frame.assign(log_chars=1.0), score_columns=data.score_columns)
    with pytest.raises(EstimatorError, match="each side"):
        repeated_study(flat, truth_gold="truth", sizes=(10,), draws=2)


def test_the_module_and_the_function_have_distinct_names() -> None:
    assert copernicus.corrections.study.repeated_study is repeated_study
    assert copernicus.corrections.repeated_study is repeated_study
    assert callable(copernicus.corrections.repeated_study)
    assert copernicus.corrections.study.RESULT_COLUMNS == tuple(COLUMNS)
