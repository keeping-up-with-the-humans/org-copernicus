"""Synthetic data for the tests of the estimate-level estimators.

Every builder draws from the generator it is given in a fixed order, so a test that seeds the
generator gets the same texts and the same labelled sample on every run.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from copernicus.corrections.data import CorrectionData

TRUE_MEAN = 1.0
TRUE_DIFFERENCE = 0.0
TRUE_SLOPE = 0.9
CLASSICAL_SLOPE = 1.0


def text_ids(n: int) -> list[str]:
    """Text ids ``t00000``, ``t00001``, ..."""
    return [f"t{index:05d}" for index in range(n)]


def labels_by_srs(rng: np.random.Generator, n_texts: int, n_labelled: int) -> np.ndarray:
    """A simple random sample of texts as a boolean mask."""
    labelled = np.zeros(n_texts, dtype=bool)
    labelled[rng.choice(n_texts, size=n_labelled, replace=False)] = True
    return labelled


def with_labels(
    frame: pd.DataFrame,
    labelled: np.ndarray,
    *,
    score_columns: Sequence[str],
    pi: float | None,
) -> CorrectionData:
    """Correction data that keep the gold value on the labelled rows only.

    Args:
        frame: Texts with a complete ``gold`` column.
        labelled: Boolean mask of the human-coded texts.
        score_columns: The score columns.
        pi: The inclusion probability written on every row, or ``None`` for no ``pi`` column.

    Returns:
        The correction data.
    """
    coded = frame.copy()
    coded["gold"] = np.where(labelled, coded["gold"].to_numpy(dtype=float), np.nan)
    if pi is not None:
        coded["pi"] = pi
    return CorrectionData(frame=coded, score_columns=tuple(score_columns))


def ppi_frame(rng: np.random.Generator, n_texts: int) -> pd.DataFrame:
    """Texts whose score is the gold value shifted by 0.5 plus an error that grows with length.

    The gold value is ``1 + 0.5 * N(0, 1)`` and independent of the covariate, so the true mean is
    1 and the true long-minus-short difference is 0, while the raw score's difference is about
    ``0.3 * 1.6``. Every text is its own group, so a group draw is a simple random sample.
    """
    covariate = rng.normal(size=n_texts)
    gold = TRUE_MEAN + 0.5 * rng.normal(size=n_texts)
    score = gold + 0.5 + 0.3 * covariate + rng.normal(scale=0.1, size=n_texts)
    ids = text_ids(n_texts)
    return pd.DataFrame(
        {"text_id": ids, "group_id": ids, "score": score, "gold": gold, "log_chars": covariate}
    )


def slope_frame(rng: np.random.Generator, n_texts: int) -> pd.DataFrame:
    """Texts for the construct as predictor: ``y = 0.5 + 0.9 V + 0.4 W + e``.

    ``V`` is the construct (the gold value), ``W`` the covariate ``log_chars`` and the score
    ``0.2 + 0.8 V + 0.3 W + 0.4 u`` measures ``V`` with an error that depends on ``W``.
    """
    covariate = rng.normal(size=n_texts)
    construct = rng.normal(size=n_texts)
    outcome = 0.5 + TRUE_SLOPE * construct + 0.4 * covariate + 0.8 * rng.normal(size=n_texts)
    score = 0.2 + 0.8 * construct + 0.3 * covariate + 0.4 * rng.normal(size=n_texts)
    ids = text_ids(n_texts)
    return pd.DataFrame(
        {
            "text_id": ids,
            "group_id": ids,
            "score": score,
            "gold": construct,
            "outcome": outcome,
            "log_chars": covariate,
        }
    )


def classical_data(
    seed: int,
    *,
    n_texts: int = 2000,
    n_labelled: int = 400,
    error_a: float = 0.4,
    error_b: float = 0.6,
    shared: float = 0.0,
) -> CorrectionData:
    """Two measurements of the construct with classical errors and an outcome with slope 1.

    ``score_a = 0.5 + 0.8 V + error_a * u_a`` and ``score_b = -0.2 + 0.6 V + error_b * u_b``,
    plus ``shared`` times a common error term in both when ``shared`` is not zero; the outcome
    is ``V + 0.5 e``. The labelled texts are a simple random sample with ``pi`` on every row.
    """
    rng = np.random.default_rng(seed)
    construct = rng.normal(size=n_texts)
    outcome = CLASSICAL_SLOPE * construct + 0.5 * rng.normal(size=n_texts)
    score_a = 0.5 + 0.8 * construct + error_a * rng.normal(size=n_texts)
    score_b = -0.2 + 0.6 * construct + error_b * rng.normal(size=n_texts)
    labelled = labels_by_srs(rng, n_texts, n_labelled)
    if shared:
        common = rng.normal(size=n_texts)
        score_a = score_a + shared * common
        score_b = score_b + shared * common
    ids = text_ids(n_texts)
    frame = pd.DataFrame(
        {
            "text_id": ids,
            "group_id": ids,
            "score_a": score_a,
            "score_b": score_b,
            "gold": construct,
            "outcome": outcome,
            "log_chars": rng.normal(size=n_texts),
        }
    )
    return with_labels(
        frame, labelled, score_columns=("score_a", "score_b"), pi=n_labelled / n_texts
    )


def grouped_frame(
    rng: np.random.Generator, *, n_groups: int, group_size: int, level_shift: float = 0.5
) -> pd.DataFrame:
    """Texts in groups with a complete gold column, a biased score and a noise column."""
    n = n_groups * group_size
    covariate = rng.normal(size=n)
    gold = 2.0 + rng.normal(size=n)
    ids = text_ids(n)
    return pd.DataFrame(
        {
            "text_id": ids,
            "group_id": [f"g{index // group_size:04d}" for index in range(n)],
            "score": gold + level_shift + 0.3 * covariate + rng.normal(scale=0.3, size=n),
            "noise": rng.normal(size=n),
            "gold": gold,
            "truth": gold,
            "log_chars": covariate,
        }
    )


def clustered_frame(
    rng: np.random.Generator,
    sizes: Sequence[int],
    *,
    group_error: float = 0.6,
    text_error: float = 0.3,
    shift: float = 0.5,
) -> pd.DataFrame:
    """Texts in groups of the given sizes whose machine error is shared within a group.

    The gold value is ``1 + N(0, 1)`` for every text, so the true mean is 1 and the true difference
    between two sets of groups fixed in advance is 0. The score adds ``shift``, an error
    ``N(0, group_error^2)`` common to the texts of a group, and an error ``N(0, text_error^2)``
    of each text. ``flag`` marks the groups with an even number.
    """
    counts = np.asarray(sizes, dtype=int)
    groups = np.repeat(np.arange(counts.size), counts)
    n = groups.size
    gold = TRUE_MEAN + rng.normal(size=n)
    shared = rng.normal(scale=group_error, size=counts.size)[groups]
    ids = text_ids(n)
    return pd.DataFrame(
        {
            "text_id": ids,
            "group_id": [f"g{group:04d}" for group in groups],
            "score": gold + shift + shared + text_error * rng.normal(size=n),
            "gold": gold,
            "flag": groups % 2 == 0,
            "log_chars": rng.normal(size=n),
        }
    )
