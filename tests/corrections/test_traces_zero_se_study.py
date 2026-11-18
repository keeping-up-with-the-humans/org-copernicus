"""The study's count of zero standard errors agrees, row by row, with its draws replayed one by one.

Four small corpora reach a standard error of exactly 0: a score that is the same on every text
(the plugged-in score and ``ppi_identity``), human values of 0 on every text (the calibrated
line is then exactly 0), sizes at which some draws are undefined, and balanced groups on which
a cross-fitted difference can be exactly 0. The first three zeros come from sums of equal
numbers and from a line through zeros, which every platform computes exactly; whether the
cross-fit's least squares leaves an exact 0 depends on the numerical library. For every row the
column ``n_se_exactly_zero`` must hold the draws that the row's own estimator gives a zero on,
whatever the platform computes, and the warning of the size must count exactly what the column
holds.
"""

from __future__ import annotations

import dataclasses
import logging
import re

import numpy as np
import pandas as pd
import pytest

from copernicus.corrections import CorrectionData, repeated_study
from copernicus.corrections._study_draws import draw_estimate, study_draw
from copernicus.corrections.estimates import long_short_split, plugin_value
from copernicus.corrections.study import ESTIMANDS, ESTIMATORS
from copernicus.errors import EstimatorError

pytestmark = pytest.mark.filterwarnings("error")

STUDY = "copernicus.corrections.study"
COUNTED = re.compile(r"(\d+) says? that a standard error is exactly 0")


def _corpus(kind: str) -> CorrectionData:
    rng = np.random.default_rng(914)
    n = 80
    ids = [f"t{index}" for index in range(n)]
    flat, balanced = kind == "flat", kind == "balanced"
    score = np.full(n, 2.0) if flat else np.tile([1.0, -1.0, 1.0, -1.0], 20)
    truth = np.full(n, 3.0) if flat else (score if balanced else np.zeros(n))
    frame = pd.DataFrame(
        {
            "text_id": ids,
            "group_id": [f"g{index // 4:03d}" for index in range(n)] if balanced else ids,
            "score": score,
            "gold": truth,
            "truth": truth,
            "log_chars": np.tile([1.0, 1.0, -1.0, -1.0], 20) if balanced else rng.normal(size=n),
        }
    )
    return CorrectionData(frame=frame, score_columns=("score",))


@pytest.mark.parametrize(
    ("kind", "size", "draws", "seed"),
    [
        ("flat", 10, 11, 42),
        ("zero_gold", 2, 11, 42),
        ("zero_gold", 4, 11, 42),
        ("balanced", 40, 7, 5),
    ],
)
def test_every_row_counts_the_zero_standard_errors_of_its_own_draws(
    kind: str, size: int, draws: int, seed: int, caplog: pytest.LogCaptureFixture
) -> None:
    data = _corpus(kind)
    split = long_short_split(data)
    truth = pd.Series(data.frame["truth"].to_numpy(), index=data.ids)
    expected: dict[tuple[str, str], tuple[int, int]] = {}
    for estimator in ESTIMATORS:
        for estimand in ESTIMANDS:
            zero = undefined = 0
            for index in range(draws):
                draw = study_draw(data, truth, size, seed, split, index)
                try:
                    _, se = draw_estimate(estimator, estimand, draw, None)
                except EstimatorError:
                    undefined += 1
                else:
                    zero += int(se == 0.0)
            expected[(estimator, estimand)] = (zero, undefined)
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        table = repeated_study(data, truth_gold="truth", sizes=(size,), draws=draws, seed=seed)
    found = {
        (row.estimator, row.estimand): (row.n_se_exactly_zero, row.n_undefined)
        for row in table.itertuples()
    }
    assert found == expected
    # Each of the four corpora gives a zero on a draw of some row, and a row without zeros.
    zeros = [zero for zero, _ in expected.values()]
    assert max(zeros) > 0 and min(zeros) == 0
    summaries = [r.getMessage() for r in caplog.records if r.name == STUDY]
    counted = sum(int(match[1]) for text in summaries for match in COUNTED.finditer(text))
    assert counted == int(table["n_se_exactly_zero"].sum())


def test_the_corpora_reach_a_zero_through_every_estimator_that_needs_no_least_squares() -> None:
    reached: set[str] = set()
    for kind, size, draws, seed in (
        ("flat", 10, 11, 42),
        ("zero_gold", 4, 11, 42),
        ("balanced", 40, 7, 5),
    ):
        table = repeated_study(
            _corpus(kind), truth_gold="truth", sizes=(size,), draws=draws, seed=seed
        )
        reached |= set(table.loc[table["n_se_exactly_zero"] > 0, "estimator"])
    assert {"raw", "ppi_identity", "calibrated"} <= reached <= set(ESTIMATORS)


def test_the_point_of_a_plugged_in_estimate_is_computed_before_its_standard_error() -> None:
    # A side without texts: the refusal of the point comes first, as it did before the count.
    data = _corpus("zero_gold")
    truth = pd.Series(data.frame["truth"].to_numpy(), index=data.ids)
    draw = study_draw(data, truth, 10, 42, long_short_split(data), 0)
    one_sided = dataclasses.replace(draw, split=np.zeros(draw.data.n, dtype=bool))
    with pytest.raises(EstimatorError) as plugged:
        plugin_value(one_sided.raw, one_sided.split)
    with pytest.raises(EstimatorError) as estimated:
        draw_estimate("raw", "long_short_difference", one_sided, None)
    assert str(estimated.value) == str(plugged.value)
