"""Design-based pseudo-values: the rectified column, the tuned mean, and the cross-fit."""

from __future__ import annotations

import itertools
import logging
from collections.abc import Sequence

import numpy as np
import pandas as pd
import pytest

from copernicus.corrections import (
    REGISTRY,
    CorrectionData,
    SampleDraw,
    attach_draw,
    list_corrections,
    population_hash_of,
)
from copernicus.corrections.families._crossfit_ghat import crossfit_ghat
from copernicus.corrections.families.pseudo import PseudoValues, pseudo_values
from copernicus.errors import ConfigurationError, EstimatorError, SettingsError

pytestmark = pytest.mark.filterwarnings("error")

TOLERANCE = 1e-12


def _pool(units_per_group: Sequence[int], seed: int = 0) -> tuple[CorrectionData, pd.Series]:
    """Texts in groups with a biased score and a planted gold; no gold on the data yet."""
    rng = np.random.default_rng(seed)
    groups = [f"g{index}" for index, size in enumerate(units_per_group) for _ in range(size)]
    n = len(groups)
    gold = 1.0 + 0.37 * np.arange(n) + rng.normal(scale=0.5, size=n)
    frame = pd.DataFrame(
        {
            "text_id": [f"t{index:03d}" for index in range(n)],
            "group_id": groups,
            "score": 2.0 + 0.5 * gold + rng.normal(scale=0.4, size=n),
            "noise": rng.normal(size=n),
            "gold": np.nan,
            "log_chars": rng.normal(size=n),
        }
    )
    data = CorrectionData(frame=frame, score_columns=("score", "noise"))
    return data, pd.Series(gold, index=frame["text_id"])


def _draw(data: CorrectionData, drawn: Sequence[str]) -> SampleDraw:
    """A hand-built draw of whole groups with the exact inclusion probability m / G."""
    all_groups = np.unique(data.groups)
    in_draw = np.isin(data.groups, list(drawn))
    return SampleDraw(
        ids=data.ids[in_draw],
        groups=np.array(sorted(drawn)),
        pi=len(drawn) / all_groups.size,
        m=len(drawn),
        n_groups=int(all_groups.size),
        n_units=data.n,
        quota=int(in_draw.sum()),
        seed=0,
        population_hash=population_hash_of(data.ids, data.groups),
    )


def _drawn_data(
    data: CorrectionData, gold: pd.Series, drawn: Sequence[str]
) -> tuple[CorrectionData, SampleDraw]:
    draw = _draw(data, drawn)
    return attach_draw(data, gold, draw), draw


@pytest.mark.parametrize("units_per_group", [(2,) * 6, (1, 1, 3, 2, 4, 3)])
def test_every_sample_mean_equals_the_row_mean(units_per_group: tuple[int, ...]) -> None:
    data, gold = _pool(units_per_group)
    groups = sorted(set(data.groups.tolist()))
    for m in range(1, len(groups) + 1):
        means = []
        for drawn in itertools.combinations(groups, m):
            drawn_data, _ = _drawn_data(data, gold, drawn)
            means.append(pseudo_values(drawn_data, score_column="score").mean_rectified)
        assert abs(float(np.mean(means)) - float(gold.mean())) <= TOLERANCE


def test_the_column_is_the_score_off_the_sample_and_the_rectified_score_on_it() -> None:
    data, gold = _pool((2,) * 6)
    drawn_data, draw = _drawn_data(data, gold, ["g1", "g4"])
    result = pseudo_values(drawn_data, score_column="score")
    score = drawn_data.frame["score"].to_numpy()
    labelled = drawn_data.labelled_mask
    expected = np.where(labelled, score + (gold.to_numpy() - score) / draw.pi, score)
    assert isinstance(result, PseudoValues)
    assert result.column.index.tolist() == drawn_data.ids.tolist()
    np.testing.assert_allclose(result.column.to_numpy(), expected, atol=TOLERANCE)
    np.testing.assert_allclose(result.ghat.to_numpy(), score, atol=TOLERANCE)
    assert result.mean_plugin == pytest.approx(float(score.mean()))
    assert result.mean_rectified == pytest.approx(float(expected.mean()))
    assert (result.rectifier, result.score_column, result.n_labelled) == ("identity", "score", 4)


def test_the_default_score_is_the_first_column_and_never_chosen_on_gold() -> None:
    data, gold = _pool((2,) * 6)
    drawn_data, _ = _drawn_data(data, gold, ["g0", "g3"])
    assert pseudo_values(drawn_data).score_column == "score"
    reordered = CorrectionData(frame=drawn_data.frame, score_columns=("noise", "score"))
    assert pseudo_values(reordered).score_column == "noise"


def test_lambda_star_uses_population_moments_and_the_tuned_mean_follows() -> None:
    data, gold = _pool((2,) * 6)
    drawn_data, _ = _drawn_data(data, gold, ["g0", "g2", "g5"])
    result = pseudo_values(drawn_data, score_column="score")
    mask = drawn_data.labelled_mask
    f_lab, y_lab = drawn_data.frame["score"].to_numpy()[mask], drawn_data.gold[mask]
    covariance = float(np.mean((y_lab - y_lab.mean()) * (f_lab - f_lab.mean())))
    assert result.lambda_star == pytest.approx(covariance / float(np.var(f_lab)), abs=1e-12)
    tuned = y_lab.mean() + result.lambda_star * (drawn_data.frame["score"].mean() - f_lab.mean())
    assert result.mean_rectified_tuned == pytest.approx(float(tuned), abs=1e-12)


def test_lambda_star_is_one_when_the_score_is_constant_on_the_labelled_rows() -> None:
    data, gold = _pool((2,) * 6)
    frame = data.frame.assign(flat=3.0)
    flat = CorrectionData(frame=frame, score_columns=("flat",))
    drawn_data, _ = _drawn_data(flat, gold, ["g0", "g2"])
    result = pseudo_values(drawn_data)
    assert result.lambda_star == 1.0
    labelled_mean = float(drawn_data.gold[drawn_data.labelled_mask].mean())
    assert result.mean_rectified_tuned == pytest.approx(labelled_mean)


def test_the_tuned_mean_is_undefined_when_pi_differs_across_the_labelled_rows(
    caplog: pytest.LogCaptureFixture,
) -> None:
    data, gold = _pool((2,) * 6)
    frame = data.frame.copy()
    labelled = np.arange(len(frame)) < 6
    frame["gold"] = np.where(labelled, gold.to_numpy(), np.nan)
    frame["pi"] = np.where(np.arange(len(frame)) < 2, 0.25, 0.5)
    unequal = CorrectionData(frame=frame, score_columns=data.score_columns)
    with caplog.at_level(logging.INFO, logger="copernicus.corrections.families.pseudo"):
        result = pseudo_values(unequal, score_column="score")
    assert np.isnan(result.mean_rectified_tuned)
    assert np.isfinite(result.mean_rectified) and np.isfinite(result.lambda_star)
    assert any("pi" in record.getMessage() for record in caplog.records)
    equal = CorrectionData(frame=frame.assign(pi=0.5), score_columns=data.score_columns)
    assert np.isfinite(pseudo_values(equal, score_column="score").mean_rectified_tuned)


def test_a_full_draw_returns_the_gold_mean() -> None:
    data, gold = _pool((2,) * 6)
    drawn_data, _ = _drawn_data(data, gold, sorted(set(data.groups.tolist())))
    result = pseudo_values(drawn_data, score_column="score")
    assert result.mean_rectified == pytest.approx(float(gold.mean()), abs=1e-12)
    np.testing.assert_allclose(result.column.to_numpy(), gold.to_numpy(), atol=1e-12)


def test_calibrated_leaves_the_mean_unbiased_in_a_monte_carlo() -> None:
    rng = np.random.default_rng(5)
    data, gold = _pool((5,) * 40, seed=3)
    groups = sorted(set(data.groups.tolist()))
    estimates = []
    for _ in range(200):
        drawn = rng.choice(groups, size=10, replace=False).tolist()
        drawn_data, _ = _drawn_data(data, gold, drawn)
        estimates.append(pseudo_values(drawn_data, rectifier="calibrated").mean_rectified)
    spread = float(np.std(estimates, ddof=1)) / np.sqrt(len(estimates))
    assert abs(float(np.mean(estimates)) - float(gold.mean())) < 3.0 * spread


def test_calibrated_fits_the_line_on_the_labelled_rows_with_the_best_column() -> None:
    data, gold = _pool((5,) * 12, seed=2)
    drawn_data, _ = _drawn_data(data, gold, ["g1", "g4", "g7", "g9"])
    result = pseudo_values(drawn_data, rectifier="calibrated")
    assert result.score_column == "score"
    mask = drawn_data.labelled_mask
    slope, intercept = np.polyfit(
        drawn_data.frame["score"].to_numpy()[mask], drawn_data.gold[mask], 1
    )
    expected = intercept + slope * drawn_data.frame["score"].to_numpy()
    np.testing.assert_allclose(result.ghat.to_numpy(), expected, atol=1e-9)
    named = pseudo_values(drawn_data, rectifier="calibrated", score_column="noise")
    assert named.score_column == "noise"


def test_crossfit_covers_every_row_and_is_reproducible() -> None:
    data, gold = _pool((5,) * 20, seed=4)
    drawn_data, _ = _drawn_data(data, gold, [f"g{i}" for i in range(0, 20, 2)])
    first = pseudo_values(drawn_data, rectifier="crossfit", k=4, seed=9)
    again = pseudo_values(drawn_data, rectifier="crossfit", k=4, seed=9)
    other = pseudo_values(drawn_data, rectifier="crossfit", k=4, seed=10)
    assert np.isfinite(first.ghat.to_numpy()).all()
    assert np.isfinite(first.column.to_numpy()).all()
    assert first.ghat.index.tolist() == drawn_data.ids.tolist()
    pd.testing.assert_series_equal(first.column, again.column)
    assert not np.allclose(first.ghat.to_numpy(), other.ghat.to_numpy())
    assert first.rectifier == "crossfit"
    assert set(first.score_column.split("|")) <= {"score", "noise"}


def test_crossfit_output_is_pinned_for_a_fixed_seed() -> None:
    data, gold = _pool((5,) * 20, seed=4)
    drawn_data, _ = _drawn_data(data, gold, [f"g{i}" for i in range(0, 20, 2)])
    ghat, columns = crossfit_ghat(drawn_data, k=4, seed=9)
    assert columns == ("score", "score", "score", "score")
    positions = [0, 17, 41, 58, 73, 99]
    expected = [0.11507, 6.586115, 17.219283, 22.457458, 29.86259, 37.686253]
    np.testing.assert_allclose(ghat[positions], expected, atol=1e-6)


def test_crossfit_predictions_never_use_the_gold_of_their_own_group() -> None:
    data, gold = _pool((5,) * 20, seed=4)
    names = [f"g{i}" for i in range(0, 20, 2)]
    drawn_data, _ = _drawn_data(data, gold, names)
    base = pseudo_values(drawn_data, rectifier="crossfit", k=4, seed=9)
    frame = drawn_data.frame.copy()
    target = (frame["group_id"] == "g6").to_numpy()
    frame.loc[target, "gold"] = frame.loc[target, "gold"] + 1000.0
    poisoned = pseudo_values(
        CorrectionData(frame=frame, score_columns=drawn_data.score_columns),
        rectifier="crossfit",
        k=4,
        seed=9,
    )
    np.testing.assert_allclose(poisoned.ghat.to_numpy()[target], base.ghat.to_numpy()[target])
    assert not np.allclose(poisoned.ghat.to_numpy()[~target], base.ghat.to_numpy()[~target])


def _with_pi(frame: pd.DataFrame, pi: float = 0.5) -> pd.DataFrame:
    return frame.assign(pi=pi)


def test_crossfit_refuses_fewer_groups_than_folds() -> None:
    data, gold = _pool((4,) * 3)
    drawn_data, _ = _drawn_data(data, gold, ["g0", "g1"])
    with pytest.raises(EstimatorError, match="inner_groups_below_k"):
        pseudo_values(drawn_data, rectifier="crossfit", k=5)


def test_crossfit_refuses_a_fold_with_fewer_than_four_labelled_rows() -> None:
    data, gold = _pool((2,) * 6)
    frame = data.frame.copy()
    frame.loc[[0, 4], "gold"] = gold.to_numpy()[[0, 4]]
    thin = CorrectionData(frame=_with_pi(frame), score_columns=data.score_columns)
    with pytest.raises(EstimatorError, match="fold_too_thin"):
        pseudo_values(thin, rectifier="crossfit", k=2)


def test_crossfit_refuses_a_selection_half_that_cannot_be_formed() -> None:
    data, gold = _pool((6, 6))
    one_group_per_fold = CorrectionData(
        frame=_with_pi(data.frame.assign(gold=gold.to_numpy()), 1.0),
        score_columns=data.score_columns,
    )
    with pytest.raises(EstimatorError, match="nested_selection_half_degenerate"):
        pseudo_values(one_group_per_fold, rectifier="crossfit", k=2)


def test_crossfit_refuses_a_selection_half_without_a_usable_column() -> None:
    data, gold = _pool((3,) * 6)
    flat = data.frame.assign(score=1.0, noise=2.0, gold=gold.to_numpy(), pi=1.0)
    with pytest.raises(EstimatorError, match="nested_selection_half_degenerate"):
        pseudo_values(
            CorrectionData(frame=flat, score_columns=("score", "noise")), rectifier="crossfit", k=3
        )


def test_inclusion_probabilities_are_required_on_labelled_rows_only() -> None:
    data, gold = _pool((2,) * 6)
    unweighted = data.frame.copy()
    unweighted["gold"] = np.where(np.arange(len(unweighted)) < 4, gold.to_numpy(), np.nan)
    without = CorrectionData(frame=unweighted, score_columns=data.score_columns)
    with pytest.raises(EstimatorError, match=r"\bpi\b.*draw_human_sample"):
        pseudo_values(without)
    on_labelled_only = unweighted.assign(pi=np.where(np.arange(len(unweighted)) < 4, 0.5, np.nan))
    full_pi = unweighted.assign(pi=0.5)
    partial = pseudo_values(
        CorrectionData(frame=on_labelled_only, score_columns=data.score_columns)
    )
    complete = pseudo_values(CorrectionData(frame=full_pi, score_columns=data.score_columns))
    pd.testing.assert_series_equal(partial.column, complete.column)
    missing_on_labelled = unweighted.assign(
        pi=np.where(np.arange(len(unweighted)) == 2, np.nan, 0.5)
    )
    with pytest.raises(EstimatorError, match=r"\bpi\b.*labelled"):
        pseudo_values(CorrectionData(frame=missing_on_labelled, score_columns=data.score_columns))


def test_settings_and_data_are_checked() -> None:
    data, gold = _pool((2,) * 6)
    drawn_data, _ = _drawn_data(data, gold, ["g0", "g1"])
    with pytest.raises(SettingsError, match="rectifier"):
        pseudo_values(drawn_data, rectifier="bogus")  # type: ignore[arg-type]
    with pytest.raises(SettingsError, match="k"):
        pseudo_values(drawn_data, rectifier="crossfit", k=1)
    with pytest.raises(ConfigurationError, match="missing"):
        pseudo_values(drawn_data, score_column="missing")
    nobody = CorrectionData(frame=_with_pi(data.frame), score_columns=data.score_columns)
    with pytest.raises(EstimatorError, match="labelled"):
        pseudo_values(nobody)


def test_the_family_is_registered_at_the_estimate_level() -> None:
    spec = REGISTRY["pseudo_values"]
    assert spec.variants == ("identity", "calibrated", "crossfit")
    assert (spec.target, spec.level, spec.errors_addressed, spec.factory) == (
        "C",
        "estimate",
        (),
        None,
    )
    assert spec.estimate_errors == (
        "estimate off and confidence-interval misses, for means and for slopes on any "
        "covariate with the construct as outcome"
    )
    assert "pseudo_values" in set(list_corrections()["family"])
