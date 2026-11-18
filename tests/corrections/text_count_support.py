"""Corpora of unequal groups for the tests of the variation of the number of drawn texts.

A corpus is given by the number of texts of each group, and the texts of a group lie next to
each other. ``written_out`` computes the variation from its definition with numpy, so that a
test does not take the value it checks from the library.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from copernicus.corrections import CorrectionData
from copernicus.data import TextTable

# The corpus of the worked case: one group of 200 texts and 200 groups of one text each.
WORKED = (200, *([1] * 200))
# Six groups of 3, 3, 3, 3, 3 and 5 texts. Over the draws of three of them the square of the
# variation is (6 * 70 - 20^2) * (6 - 3) / ((6 - 1) * 20^2 * 3) = 1 / 100 in the arithmetic of
# the counts, so the variation is the threshold 0.1; over draws of two it is about 0.141.
AT_THRESHOLD = (3, 3, 3, 3, 3, 5)
# Twenty groups, ten of three texts and ten of four: over draws of nine the variation is
# about 0.036.
BELOW_THRESHOLD = (*([3] * 10), *([4] * 10))
# One group of twenty texts and twenty groups of one text.
DOMINANT_SMALL = (20, *([1] * 20))
# Seven groups whose draws of four (a quota of 8 of the 18 texts) vary by about 0.10015, which
# three significant digits show as 0.1.
SHOWN_AS_THRESHOLD = (1, 2, 3, 3, 3, 3, 3)
# Counts of two billion texts whose draws of three vary by a square of 1 / 100 + 3 / (5 * 10^18):
# above the threshold in exact arithmetic, while the square root rounds to the float 0.1.
JUST_ABOVE = (300_000_001, 299_999_999, 300_000_000, 300_000_000, 300_000_000, 500_000_000)
# One group of forty texts and twenty groups of two, and twenty groups of four: a corpus with a
# dominant group and an even one of the same 80 texts, for the battery.
DOMINANT_BATTERY = (40, *([2] * 20))
EVEN_BATTERY = (4,) * 20


def group_ids(sizes: Sequence[int]) -> np.ndarray:
    """One group id per text: ``g000`` for every text of the first group, and so on."""
    return np.repeat([f"g{index:03d}" for index in range(len(sizes))], sizes)


def written_out(sizes: Sequence[int], m: int) -> float:
    """``(sd_n / mean_n) * sqrt((1 - m / G) / m)``, with the divisor ``G - 1`` in ``sd_n``."""
    counts = np.asarray(sizes, dtype=float)
    spread = np.std(counts, ddof=1) / counts.mean()
    return float(spread * np.sqrt((1.0 - m / counts.size) / m))


def texts_frame(sizes: Sequence[int]) -> pd.DataFrame:
    """The texts of a corpus: ``text_id``, a ``text`` and the ``group_id`` of every text."""
    groups = group_ids(sizes)
    return pd.DataFrame(
        {
            "text_id": [f"t{index:04d}" for index in range(groups.size)],
            "text": "a short text",
            "group_id": groups,
        }
    )


def battery_data(sizes: Sequence[int], seed: int) -> CorrectionData:
    """Correction data of a corpus with a gold value on every text, for the battery.

    The score is a standard normal value ``x`` of each text, and the gold value is ``x^2`` plus
    a fifth of the group's number plus a little noise, so the score ranks the gold values in
    part.
    """
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(len(sizes)), sizes)
    score = rng.normal(size=groups.size)
    frame = texts_frame(sizes).drop(columns="text")
    frame["score"] = score
    frame["gold"] = score**2 + groups / 5 + 0.1 * rng.normal(size=groups.size)
    frame["log_chars"] = rng.normal(size=groups.size)
    return CorrectionData(frame=frame, score_columns=("score",))


def table_of(sizes: Sequence[int], *, units: Sequence[str] | None = None) -> TextTable:
    """A text table of a corpus, with the column ``unit`` when ``units`` are given."""
    frame = texts_frame(sizes)
    if units is not None:
        frame["unit"] = list(units)
    return TextTable.from_frame(frame, group_id="group_id")
