"""The cross-fitted estimator of the repeated study takes the identity estimator's floors."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from estimates_support import RELATIVE_TOLERANCE, grouped_frame, same_study_row

from copernicus.baseline.folds import rng_for
from copernicus.corrections import (
    CorrectionData,
    SampleDraw,
    draw_from_data,
    long_short_split,
    repeated_study,
)
from copernicus.corrections.families._fitted import SEED_BOUND

pytestmark = pytest.mark.filterwarnings("error")

# repeated_study before the floors on the corpus of ``_two_sided_data`` (sizes 60, 12 draws, seed
# 8), where every draw holds at least 22 labelled texts on each side: (estimate_mean, coverage,
# se_mean, n_undefined), for the default clustering and for cluster=False. The points, the standard
# errors and the truths are compared to twelve significant digits, because platforms differ in the
# last digits of floating-point calculations, linear algebra among them.
PINNED = {
    None: {
        ("raw", "mean"): (2.5941156615188246, 0.0, 0.08233151721865017, 0),
        ("raw", "long_short_difference"): (0.48393995046519844, 0.0, 0.16109165674298526, 0),
        ("calibrated", "mean"): (2.127472910360097, 1.0, 0.06941730760382879, 0),
        ("calibrated", "long_short_difference"): (0.4080309647884695, 0.0, 0.13582343027689375, 0),
        ("ppi_identity", "mean"): (2.1252111432490213, 1.0, 0.10074404376102582, 0),
        ("ppi_identity", "long_short_difference"): (
            0.06853464564276852,
            1.0,
            0.17325556874152062,
            0,
        ),
        ("ppi_crossfit", "mean"): (2.1235214220083334, 1.0, 0.07777838367028811, 0),
        ("ppi_crossfit", "long_short_difference"): (
            0.05865140359158072,
            1.0,
            0.16143015850583273,
            0,
        ),
    },
    False: {
        ("raw", "mean"): (2.5941156615188246, 0.0, 0.08233151721865017, 0),
        ("raw", "long_short_difference"): (0.48393995046519844, 0.0, 0.16109165674298526, 0),
        ("calibrated", "mean"): (2.127472910360097, 1.0, 0.06941730760382879, 0),
        ("calibrated", "long_short_difference"): (0.4080309647884695, 0.0, 0.13582343027689375, 0),
        ("ppi_identity", "mean"): (2.1252111432490213, 1.0, 0.09629240326584199, 0),
        ("ppi_identity", "long_short_difference"): (
            0.06853464564276852,
            1.0,
            0.18222670862354265,
            0,
        ),
        ("ppi_crossfit", "mean"): (2.1235214220083334, 1.0, 0.07939178526941766, 0),
        ("ppi_crossfit", "long_short_difference"): (
            0.05865140359158072,
            1.0,
            0.15907180206792784,
            0,
        ),
    },
}
PINNED_TRUTH = {"mean": 2.1204549271207505, "long_short_difference": 0.031397523417867834}


def _draw_of(data: CorrectionData, size: int, seed: int, index: int) -> SampleDraw:
    """The study's draw of ``size`` texts with ``index``, by the documented scheme."""
    draw_seed = int(rng_for(seed, "repeated_study", size, index).integers(SEED_BOUND, size=2)[0])
    return draw_from_data(data, size, seed=draw_seed)


def _two_sided_data() -> CorrectionData:
    """Sixty groups of three texts; the covariate of every text is its own, so both sides mix."""
    frame = grouped_frame(np.random.default_rng(4), n_groups=60, group_size=3)
    return CorrectionData(frame=frame, score_columns=("score", "noise"))


def _rare_side_data() -> CorrectionData:
    """200 long texts in one group with a shared error of 3, and 200 short texts, each alone.

    A draw of 24 texts takes 13 of the 201 groups, so it holds the long group, and with it every
    labelled long text, in about one draw of 15; the other draws hold none.
    """
    rng = np.random.default_rng(5)
    long = np.r_[np.ones(200, dtype=bool), np.zeros(200, dtype=bool)]
    gold = rng.normal(size=400)
    frame = pd.DataFrame(
        {
            "text_id": [f"t{index:03d}" for index in range(400)],
            "group_id": ["long"] * 200 + [f"short{index:03d}" for index in range(200)],
            "score": gold + np.where(long, 3.0, 0.0) + rng.normal(scale=0.1, size=400),
            "gold": np.nan,
            "truth": gold,
            "log_chars": np.where(long, 7.0, 5.0) + rng.normal(scale=0.01, size=400),
        }
    )
    return CorrectionData(frame=frame, score_columns=("score",))


def _few_short_texts_data() -> CorrectionData:
    """Eighty texts, each its own group; four lie below the covariate's median, the rest at it.

    The other 76 texts tie at the median and form the long side, the four the short side, so a
    draw of 20 of the 80 texts holds none, one or more of the four.
    """
    rng = np.random.default_rng(9)
    gold = rng.normal(size=80)
    frame = pd.DataFrame(
        {
            "text_id": [f"t{index:03d}" for index in range(80)],
            "group_id": [f"g{index:03d}" for index in range(80)],
            "score": gold + 0.5 + rng.normal(scale=0.3, size=80),
            "gold": np.nan,
            "truth": gold,
            "log_chars": np.where(np.arange(80) < 4, 5.0, 7.0),
        }
    )
    return CorrectionData(frame=frame, score_columns=("score",))


@pytest.mark.parametrize("cluster", [None, False, True])
def test_one_labelled_text_on_a_side_is_too_few_for_both_estimators(cluster: bool | None) -> None:
    # Every text is its own group, so only the floor refuses a draw here: both estimators refuse
    # exactly the draws that hold fewer than two of the four short texts, one as well as none.
    data = _few_short_texts_data()
    short = ~long_short_split(data)
    assert int(short.sum()) == 4
    size, seed, draws = 20, 0, 50
    held = [
        int((np.isin(data.ids, np.asarray(_draw_of(data, size, seed, index).ids)) & short).sum())
        for index in range(draws)
    ]
    assert held.count(0) > 0 and held.count(1) > 0 and max(held) >= 2
    table = repeated_study(
        data,
        truth_gold="truth",
        sizes=(size,),
        draws=draws,
        seed=seed,
        estimators=("ppi_identity", "ppi_crossfit"),
        cluster=cluster,
    ).set_index(["estimator", "estimand"])
    for estimator in ("ppi_identity", "ppi_crossfit"):
        found = table.loc[(estimator, "long_short_difference"), "n_undefined"]
        assert found == sum(count < 2 for count in held), estimator
        assert table.loc[(estimator, "mean"), "n_undefined"] == 0


@pytest.mark.parametrize("cluster", [None, False])
def test_a_crossfit_difference_without_two_labelled_texts_on_a_side_is_refused(
    cluster: bool | None, caplog: pytest.LogCaptureFixture
) -> None:
    # A draw without the long group holds no labelled long text, so the pseudo-values of the
    # long side are its cross-fitted predictions alone. Before the floor such draws were
    # estimated and the row reported status ok: under the default with a bias of about 2.3, and
    # with cluster=False with intervals that missed the truth in 49 of the 50 draws. Now both
    # estimators refuse the same draws for the same reason: every draw under the default (a
    # draw that holds the long group clusters, and its long side lies in one group), and the 48
    # draws without the long group with cluster=False.
    data = _rare_side_data()
    size, seed, draws = 24, 0, 50
    holds = ["long" in _draw_of(data, size, seed, index).groups.tolist() for index in range(draws)]
    assert sum(holds) == 2 and not holds[-1]
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
    expected = draws if cluster is None else draws - sum(holds)
    for estimator in ("ppi_identity", "ppi_crossfit"):
        difference = table.loc[(estimator, "long_short_difference")]
        assert difference["status"] == "undefined_share_exceeded"
        assert np.isnan(difference["bias"]) and np.isnan(difference["coverage"])
        assert difference["n_undefined"] == expected, estimator
        assert table.loc[(estimator, "mean"), "status"] == "ok"
    reasons = {
        name: [text for text in caplog.messages if f"estimator '{name}'" in text]
        for name in ("ppi_identity", "ppi_crossfit")
    }
    assert [len(found) for found in reasons.values()] == [1, 1]
    floor = (
        "difference needs at least 2 labelled rows on each side; there are 0 on one side: "
        "label more texts on that side, or use a grouping that puts at least 2 labelled texts "
        "on each side"
    )
    assert reasons["ppi_identity"][0].endswith(f"last reason: the prediction-powered {floor}")
    assert reasons["ppi_crossfit"][0].endswith(
        f"last reason: the cross-fitted prediction-powered {floor}"
    )


def test_where_every_draw_has_two_labelled_texts_on_each_side_nothing_changes() -> None:
    data = _two_sided_data()
    split = long_short_split(data)
    size, seed, draws = 60, 8, 12
    for index in range(draws):
        drawn = np.isin(data.ids, np.asarray(_draw_of(data, size, seed, index).ids))
        assert min(int((drawn & split).sum()), int((drawn & ~split).sum())) >= 22
    for cluster, pinned in PINNED.items():
        table = repeated_study(
            data, truth_gold="truth", sizes=(size,), draws=draws, seed=seed, cluster=cluster
        )
        found = {
            (row.estimator, row.estimand): (
                row.estimate_mean,
                row.coverage,
                row.se_mean,
                row.n_undefined,
            )
            for row in table.itertuples()
        }
        assert found.keys() == pinned.keys(), cluster
        for key, row in pinned.items():
            assert same_study_row(found[key], row), (cluster, key, found[key])
        truths = dict(zip(table["estimand"], table["truth"], strict=True))
        assert truths == pytest.approx(PINNED_TRUTH, rel=RELATIVE_TOLERANCE, abs=0.0)
        assert (table["status"] == "ok").all()
