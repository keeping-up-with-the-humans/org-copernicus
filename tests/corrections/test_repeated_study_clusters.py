"""The cross-fitted estimator of the repeated study follows the clustering rule of ppi_identity."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from estimates_support import RELATIVE_TOLERANCE, grouped_frame, same_study_row

from copernicus.baseline.folds import rng_for
from copernicus.corrections import (
    CorrectionData,
    attach_draw,
    draw_from_data,
    long_short_split,
    mean_ppi,
    pseudo_values,
    repeated_study,
)
from copernicus.corrections.families._fitted import SEED_BOUND

pytestmark = pytest.mark.filterwarnings("error")

PINNED_TRUTH = {"mean": 2.0729049730097255, "long_short_difference": 0.19074931557854313}
# repeated_study of version 0.1.0 on the corpus of one text per group below, with cluster=None
# and cluster=False (the same frame): (estimate_mean, coverage, se_mean, n_undefined). The points,
# the standard errors and the truths are compared to twelve significant digits, because platforms
# differ in the last digits of floating-point calculations, linear algebra among them.
PINNED = {
    ("raw", "mean"): (2.5561930676384628, 0.0, 0.12876111093453332, 0),
    ("raw", "long_short_difference"): (0.6123866649159453, 1.0, 0.24971985479084158, 0),
    ("calibrated", "mean"): (2.0654512321419616, 1.0, 0.11170281487110972, 0),
    ("calibrated", "long_short_difference"): (0.5312575649912018, 1.0, 0.21663692171407478, 0),
    ("ppi_identity", "mean"): (2.0706649774795367, 1.0, 0.143865280884911, 0),
    ("ppi_identity", "long_short_difference"): (0.1969444249960016, 1.0, 0.2705159496036559, 0),
    ("ppi_crossfit", "mean"): (2.070839900258573, 1.0, 0.12226535163969908, 0),
    ("ppi_crossfit", "long_short_difference"): (0.19094661335349145, 1.0, 0.2450206265295255, 0),
}
# With cluster=True, version 0.1.0 clustered ppi_identity only; its difference row read:
PINNED_IDENTITY_CLUSTERED_DIFFERENCE_SE = 0.2683261079358293


def _shared_error_data(seed: int = 7, *, n_groups: int = 300, size: int = 4) -> CorrectionData:
    """Groups of texts whose machine error is shared within the group.

    The gold value is a trait of the group plus noise of the text; the score is 0.7 times the
    gold value plus an error common to the texts of the group and an error of each text. The
    covariate does not depend on the group.
    """
    rng = np.random.default_rng(seed)
    group = np.repeat(np.arange(n_groups), size)
    gold = rng.normal(size=n_groups)[group] + rng.normal(scale=0.5, size=group.size)
    shared = rng.normal(scale=0.8, size=n_groups)[group]
    score = 0.7 * gold + shared + rng.normal(scale=0.3, size=group.size)
    ids = [f"t{index:05d}" for index in range(group.size)]
    frame = pd.DataFrame(
        {
            "text_id": ids,
            "group_id": [f"g{value:04d}" for value in group],
            "score": score,
            "gold": np.nan,
            "truth": gold,
            "log_chars": rng.normal(6.0, 0.5, size=group.size),
        }
    )
    return CorrectionData(frame=frame, score_columns=("score",))


def _single_text_data() -> CorrectionData:
    """Eighty texts, each its own group, with no text at the median of the covariate."""
    frame = grouped_frame(np.random.default_rng(11), n_groups=80, group_size=1)
    return CorrectionData(frame=frame, score_columns=("score", "noise"))


def _clustered_se(values: np.ndarray, split: np.ndarray | None, groups: np.ndarray) -> float:
    """The clustered standard error of a mean or a difference, written out from its formula.

    ``sqrt(G / (G - 1) * sum_g S_g^2) / n``, with ``S_g`` the sum over the texts of group ``g``
    of the centred contributions: ``v - mean(v)`` for a mean, ``n / n_a (v - mean_a)`` and
    ``-n / n_b (v - mean_b)`` on the two sides of a difference.
    """
    n = values.size
    if split is None:
        psi = values - values.mean()
    else:
        long_part = n / split.sum() * (values - values[split].mean())
        short_part = -n / (~split).sum() * (values - values[~split].mean())
        psi = np.where(split, long_part, short_part)
    labels, inverse = np.unique(groups, return_inverse=True)
    sums = np.bincount(inverse, weights=psi - psi.mean())
    return float(np.sqrt(labels.size / (labels.size - 1) * np.sum(sums**2)) / n)


def _naive_se(values: np.ndarray, split: np.ndarray | None) -> float:
    """``sd / sqrt(n)``, or ``sqrt(var_a / n_a + var_b / n_b)``, with sample moments."""
    parts = [values] if split is None else [values[split], values[~split]]
    return float(np.sqrt(sum(np.var(part, ddof=1) / part.size for part in parts)))


def test_the_crossfit_interval_covers_like_the_identity_interval_when_groups_share_an_error() -> (
    None
):
    """300 draws of 200 texts (50 of the 300 groups of four) from one corpus.

    The intervals cover in about 98 percent of the draws, so the Monte-Carlo standard error of a
    coverage over 300 draws is about 0.008, and that of the difference of two coverages on the
    same draws at most 0.011 (less, as the two are positively correlated): the tolerance of 0.03
    is more than two and a half of them, while the 0.21 by which an interval that treats the
    texts of a drawn group as independent falls short is about nineteen.
    """
    data = _shared_error_data()
    table = repeated_study(
        data,
        truth_gold="truth",
        sizes=(200,),
        draws=300,
        seed=3,
        estimators=("ppi_identity", "ppi_crossfit"),
    )
    rows = table.set_index(["estimator", "estimand"])
    assert (table["status"] == "ok").all() and (table["n_undefined"] == 0).all()
    identity = rows.loc[("ppi_identity", "mean"), "coverage"]
    crossfit = rows.loc[("ppi_crossfit", "mean"), "coverage"]
    assert identity >= 0.95, identity
    assert abs(crossfit - identity) <= 0.03, (crossfit, identity)


def test_the_crossfit_standard_error_follows_the_cluster_option() -> None:
    data = CorrectionData(
        frame=grouped_frame(np.random.default_rng(4), n_groups=60, group_size=3),
        score_columns=("score", "noise"),
    )
    size, seed, draws = 60, 8, 6
    truth = pd.Series(data.frame["truth"].to_numpy(), index=data.ids)
    split = long_short_split(data)
    # Per draw, by the documented scheme: the clustered and the independent standard errors of
    # the pseudo-value mean and difference, and the independent one of ppi_identity's mean.
    expected: dict[tuple[str, str], list[float]] = {
        (kind, estimand): [] for kind in ("clustered", "naive") for estimand in ("mean", "diff")
    }
    identity_naive: list[float] = []
    for index in range(draws):
        draw_seed, crossfit_seed = (
            int(value)
            for value in rng_for(seed, "repeated_study", size, index).integers(SEED_BOUND, size=2)
        )
        attached = attach_draw(data, truth, draw_from_data(data, size, seed=draw_seed))
        column = pseudo_values(attached, rectifier="crossfit", seed=crossfit_seed).column
        values = column.to_numpy(dtype=float)
        expected[("clustered", "mean")].append(_clustered_se(values, None, data.groups))
        expected[("clustered", "diff")].append(_clustered_se(values, split, data.groups))
        expected[("naive", "mean")].append(_naive_se(values, None))
        expected[("naive", "diff")].append(_naive_se(values, split))
        identity_naive.append(mean_ppi(attached, cluster=False).se)
    tables = {
        option: repeated_study(
            data,
            truth_gold="truth",
            sizes=(size,),
            draws=draws,
            seed=seed,
            estimators=("ppi_identity", "ppi_crossfit"),
            cluster=option,
        ).set_index(["estimator", "estimand"])
        for option in (None, True, False)
    }
    for option, kind in ((None, "clustered"), (True, "clustered"), (False, "naive")):
        crossfit = tables[option].loc["ppi_crossfit"]
        for estimand, short in (("mean", "mean"), ("long_short_difference", "diff")):
            assert crossfit.loc[estimand, "se_mean"] == pytest.approx(
                np.mean(expected[(kind, short)]), rel=1e-12
            ), (option, estimand)
    unclustered = tables[False]
    assert unclustered.loc[("ppi_identity", "mean"), "se_mean"] == pytest.approx(
        np.mean(identity_naive), rel=1e-12
    )
    for option in (None, True):
        assert tables[option]["estimate_mean"].tolist() == unclustered["estimate_mean"].tolist()
        assert (tables[option]["n_undefined"] == 0).all()


def _long_groups_data() -> CorrectionData:
    """Three groups of twenty long texts and sixty short texts, each its own group."""
    rng = np.random.default_rng(2)
    group = np.repeat(np.arange(63), [20, 20, 20] + [1] * 60)
    long_group = group < 3
    gold = 1.0 + rng.normal(size=group.size) + 0.5 * long_group
    shared = rng.normal(scale=0.5, size=63)[group]
    frame = pd.DataFrame(
        {
            "text_id": [f"t{index:04d}" for index in range(group.size)],
            "group_id": [f"g{value:03d}" for value in group],
            "score": gold + 0.3 + shared + 0.3 * rng.normal(size=group.size),
            "gold": np.nan,
            "truth": gold,
            "log_chars": np.where(long_group, 7.0, 5.0) + 0.1 * rng.normal(size=group.size),
        }
    )
    return CorrectionData(frame=frame, score_columns=("score",))


def test_a_side_whose_labelled_texts_lie_in_one_group_leaves_the_crossfit_difference_defined(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The first draw of 40 texts with seed 0 holds one of the three long groups.

    ``ppi_identity`` refuses the clustered difference: the contributions of its labelled term on
    that side cancel within the one group. The pseudo-value column covers every text, so the
    side's contributions do not cancel there, and ``ppi_crossfit`` keeps a clustered difference.
    """
    data = _long_groups_data()
    size, seed = 40, 0
    draw_seed, crossfit_seed = (
        int(value)
        for value in rng_for(seed, "repeated_study", size, 0).integers(SEED_BOUND, size=2)
    )
    sample = draw_from_data(data, size, seed=draw_seed)
    assert [group for group in sample.groups if group in ("g000", "g001", "g002")] == ["g002"]
    with caplog.at_level("WARNING", logger="copernicus.corrections.study"):
        table = repeated_study(
            data,
            truth_gold="truth",
            sizes=(size,),
            draws=1,
            seed=seed,
            estimators=("ppi_identity", "ppi_crossfit"),
        ).set_index(["estimator", "estimand"])
    assert table.loc[("ppi_identity", "long_short_difference"), "n_undefined"] == 1
    assert "two groups on each side" in caplog.records[0].getMessage()
    assert (table.loc["ppi_crossfit", "n_undefined"] == 0).all()
    truth = pd.Series(data.frame["truth"].to_numpy(), index=data.ids)
    attached = attach_draw(data, truth, sample)
    column = pseudo_values(attached, rectifier="crossfit", seed=crossfit_seed).column
    expected = _clustered_se(column.to_numpy(dtype=float), long_short_split(data), data.groups)
    assert table.loc[("ppi_crossfit", "long_short_difference"), "se_mean"] == pytest.approx(
        expected, rel=1e-12
    )


def _drawn_groups(data: CorrectionData, size: int, seed: int, index: int) -> set[str]:
    """The groups of the study's draw of ``size`` texts with ``index``, by the documented scheme."""
    draw_seed = int(rng_for(seed, "repeated_study", size, index).integers(SEED_BOUND, size=2)[0])
    return {str(group) for group in draw_from_data(data, size, seed=draw_seed).groups}


def _long_side_data(long_groups: int) -> CorrectionData:
    """Forty long texts in ``long_groups`` groups that share an error of 3, forty short texts alone.

    The covariate separates the two sides, so the long side of the corpus holds exactly the
    texts of the long groups.
    """
    rng = np.random.default_rng(81)
    long = np.r_[np.ones(40, dtype=bool), np.zeros(40, dtype=bool)]
    gold = rng.normal(size=80)
    long_ids = [f"long{index % long_groups}" for index in range(40)]
    frame = pd.DataFrame(
        {
            "text_id": [f"t{index:03d}" for index in range(80)],
            "group_id": long_ids + [f"short{index:03d}" for index in range(40)],
            "score": gold + np.where(long, 3.0, 0.0) + rng.normal(scale=0.1, size=80),
            "gold": np.nan,
            "truth": gold,
            "log_chars": np.where(long, 7.0, 5.0) + rng.normal(scale=0.01, size=80),
        }
    )
    return CorrectionData(frame=frame, score_columns=("score",))


@pytest.mark.parametrize("cluster", [None, True])
def test_a_clustered_difference_whose_side_lies_in_one_group_is_refused(
    cluster: bool | None, caplog: pytest.LogCaptureFixture
) -> None:
    # Every long text lies in one group, so the contributions of the long side, centred at the
    # side's mean, cancel within it: a clustered standard error would leave out the whole side.
    # A draw that holds the group clusters and is refused for it; a draw that does not holds no
    # labelled long text and is refused by the floor of two labelled texts on each side, as
    # ppi_identity refuses it, under either setting. The last draw holds the group, so the
    # warning gives the reason of the side.
    data = _long_side_data(long_groups=1)
    size, seed, draws = 40, 0, 18
    holds = ["long0" in _drawn_groups(data, size, seed, index) for index in range(draws)]
    assert holds[-1] and not all(holds)
    with caplog.at_level("WARNING", logger="copernicus.corrections.study"):
        table = repeated_study(
            data,
            truth_gold="truth",
            sizes=(size,),
            draws=draws,
            seed=seed,
            estimators=("ppi_identity", "ppi_crossfit"),
            cluster=cluster,
        ).set_index(["estimator", "estimand"])
    difference = table.loc[("ppi_crossfit", "long_short_difference")]
    assert difference["status"] == "undefined_share_exceeded"
    assert np.isnan(difference["coverage"]) and np.isnan(difference["se_mean"])
    assert difference["n_undefined"] == draws
    assert table.loc[("ppi_crossfit", "mean"), "status"] == "ok"
    reasons = [
        record.getMessage() for record in caplog.records if "'ppi_crossfit'" in record.getMessage()
    ]
    assert len(reasons) == 1
    assert "the long side (the texts at or above the covariate's median)" in reasons[0]
    assert "'long0'" in reasons[0] and "cluster=False" in reasons[0]


@pytest.mark.parametrize("cluster", [None, True])
def test_a_clustered_difference_with_two_groups_on_each_side_is_estimated(
    cluster: bool | None,
) -> None:
    # A draw of 77 of the 80 texts takes 41 of the 42 groups, so every draw holds a long group
    # and twenty labelled long texts: the floor refuses no draw, and the side check, at its
    # boundary of two groups on the long side of the corpus, refuses none either.
    data = _long_side_data(long_groups=2)
    size, seed, draws = 77, 0, 20
    for index in range(draws):
        assert {"long0", "long1"} & _drawn_groups(data, size, seed, index)
    table = repeated_study(
        data,
        truth_gold="truth",
        sizes=(size,),
        draws=draws,
        seed=seed,
        estimators=("ppi_crossfit",),
        cluster=cluster,
    ).set_index("estimand")
    assert (table["status"] == "ok").all() and (table["n_undefined"] == 0).all()
    assert np.isfinite(table.loc["long_short_difference", "se_mean"])


def _pinned_rows(table: pd.DataFrame) -> dict[tuple[str, str], tuple[float, float, float, int]]:
    """The pinned columns of a study frame, keyed by estimator and estimand."""
    return {
        (row.estimator, row.estimand): (
            row.estimate_mean,
            row.coverage,
            row.se_mean,
            row.n_undefined,
        )
        for row in table.itertuples()
    }


def test_with_one_text_per_group_the_study_returns_what_version_0_1_0_returned() -> None:
    data = _single_text_data()
    covariate = data.covariate
    assert not (covariate == np.median(covariate)).any()
    settings = {"truth_gold": "truth", "sizes": (30,), "draws": 30, "seed": 5}
    for option in (None, False):
        table = repeated_study(data, **settings, cluster=option)  # type: ignore[arg-type]
        rows = _pinned_rows(table)
        assert rows.keys() == PINNED.keys(), option
        for key, pinned in PINNED.items():
            assert same_study_row(rows[key], pinned), (option, key, rows[key])
        truths = dict(zip(table["estimand"], table["truth"], strict=True))
        assert truths == pytest.approx(PINNED_TRUTH, rel=RELATIVE_TOLERANCE, abs=0.0)
        assert (table["status"] == "ok").all() and (table["n_draws"] == 30).all()
    forced = _pinned_rows(repeated_study(data, **settings, cluster=True))  # type: ignore[arg-type]
    clustered_identity = (*PINNED[("ppi_identity", "long_short_difference")][:2],)
    for key, pinned in PINNED.items():
        if key == ("ppi_identity", "long_short_difference"):
            expected = (*clustered_identity, PINNED_IDENTITY_CLUSTERED_DIFFERENCE_SE, 0)
            assert same_study_row(forced[key], expected), (key, forced[key])
        elif key[0] != "ppi_crossfit":
            assert same_study_row(forced[key], pinned), (key, forced[key])
    crossfit_mean = ("ppi_crossfit", "mean")
    assert same_study_row(forced[crossfit_mean], PINNED[crossfit_mean]), forced[crossfit_mean]
    # The clustered variance of a difference over 80 texts split 40 and 40 is the independent
    # one times (n - 2) / (n - 1), whatever the values, so the forced choice moves it by that.
    point, coverage, se, undefined = PINNED[("ppi_crossfit", "long_short_difference")]
    moved = (point, coverage, se * np.sqrt(78 / 79), undefined)
    found = forced[("ppi_crossfit", "long_short_difference")]
    assert same_study_row(found, moved), found
