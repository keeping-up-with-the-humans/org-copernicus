"""Seeded data that reach, or do not reach, a value that stands in for an unidentified quantity.

Each builder says how its data reach the case. The builders use only the public data container
and the documented generator of the cross-fit's inner folds, so the same data can be built at
any commit of the library.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from estimates_support import RELATIVE_TOLERANCE

from copernicus.baseline.folds import rng_for
from copernicus.corrections import CorrectionData, inner_splits

CROSSFIT_K, CROSSFIT_SEED = 2, 9
WEAK_SEED = 0
CONSTANT_COVARIATE = 0.5


def close(pinned: Any) -> Any:
    """A value pinned with all its digits, to be met to twelve significant digits.

    Platforms differ in the last digits of floating-point calculations. A pinned 0 has to be
    met exactly.
    """
    return pytest.approx(pinned, rel=RELATIVE_TOLERANCE, abs=0.0)


def text_frame(columns: dict[str, np.ndarray], gold: np.ndarray, group_size: int) -> pd.DataFrame:
    """Texts with the given score columns, gold on every text and a covariate that varies."""
    n = gold.size
    return pd.DataFrame(
        {
            "text_id": [f"t{index:03d}" for index in range(n)],
            "group_id": [f"g{index // group_size:03d}" for index in range(n)],
            **columns,
            "gold": gold,
            "log_chars": np.linspace(0.0, 1.0, n),
        }
    )


def crossfit_frame(seed: int = 4) -> pd.DataFrame:
    """Eight groups of four texts; six of them carry gold with the inclusion probability 0.75."""
    rng = np.random.default_rng(seed)
    n = 32
    groups = np.array([f"g{index // 4:02d}" for index in range(n)])
    gold = 1.0 + 0.37 * np.arange(n) + rng.normal(scale=0.5, size=n)
    labelled = ~np.isin(groups, ["g02", "g05"])
    return pd.DataFrame(
        {
            "text_id": [f"t{index:03d}" for index in range(n)],
            "group_id": groups,
            "score": 2.0 + 0.5 * gold + rng.normal(scale=0.4, size=n),
            "noise": rng.normal(size=n),
            "gold": np.where(labelled, gold, np.nan),
            "log_chars": rng.normal(size=n),
            "pi": 0.75,
        }
    )


def crossfit_data(frame: pd.DataFrame) -> CorrectionData:
    """The correction data of a cross-fit frame, with its two score columns."""
    return CorrectionData(frame=frame, score_columns=("score", "noise"))


def inner_fold_rows(frame: pd.DataFrame) -> list[np.ndarray]:
    """The rows of each inner fold of ``crossfit_ghat`` at ``CROSSFIT_K`` and ``CROSSFIT_SEED``.

    The folds come from ``rng_for(seed, "pseudo", "crossfit", k)``, the generator that the
    cross-fit documents, before it draws anything else.
    """
    groups = crossfit_data(frame).groups
    rng = rng_for(CROSSFIT_SEED, "pseudo", "crossfit", CROSSFIT_K)
    return [test for _, test in inner_splits(groups, CROSSFIT_K, rng)]


def rank_deficient_crossfit_frame() -> pd.DataFrame:
    """A cross-fit frame whose covariate is constant on the texts of the second inner fold.

    The first fold is fitted on the labelled texts of the second, where the design
    ``[1, score, covariate]`` then has rank 2; the second fold is fitted on the labelled texts of
    the first, where the covariate varies and the design has rank 3.
    """
    frame = crossfit_frame()
    second = np.zeros(len(frame), dtype=bool)
    second[inner_fold_rows(frame)[1]] = True
    return frame.assign(log_chars=np.where(second, CONSTANT_COVARIATE, frame["log_chars"]))


def study_corpus(*, covariate_on_score: bool, seed: int = 3) -> CorrectionData:
    """Sixty groups of two texts, all with gold (``truth``), and one score column.

    With ``covariate_on_score`` the covariate is the line ``0.5 * score + 1``, so on the labelled
    training rows of every inner fold the design ``[1, score, covariate]`` has rank 2; otherwise
    the covariate is drawn on its own.
    """
    rng = np.random.default_rng(seed)
    n = 120
    gold = 2.0 + rng.normal(size=n)
    score = gold + 0.5 + rng.normal(scale=0.3, size=n)
    covariate = 0.5 * score + 1.0 if covariate_on_score else rng.normal(size=n)
    frame = pd.DataFrame(
        {
            "text_id": [f"t{index:03d}" for index in range(n)],
            "group_id": [f"g{index // 2:03d}" for index in range(n)],
            "score": score,
            "gold": gold,
            "truth": gold,
            "log_chars": covariate,
        }
    )
    return CorrectionData(frame=frame, score_columns=("score",))


def imputation_data(n: int, seed: int = 0) -> CorrectionData:
    """``n`` labelled texts, each its own group, with one score column in general position."""
    rng = np.random.default_rng(seed)
    score = rng.uniform(0.0, 4.0, size=n)
    covariate = rng.normal(size=n)
    gold = 1.0 + 2.0 * score + 0.5 * covariate + rng.normal(scale=0.3, size=n)
    frame = pd.DataFrame(
        {
            "text_id": [f"t{index:03d}" for index in range(n)],
            "group_id": [f"t{index:03d}" for index in range(n)],
            "score": score,
            "gold": gold,
            "log_chars": covariate,
        }
    )
    return CorrectionData(frame=frame, score_columns=("score",))


def replayed_resamples(data: CorrectionData, m_draws: int, seed: int) -> list[np.ndarray]:
    """The bootstrap resamples of a fit of ``multiple_imputation`` on every text of the data.

    The fit draws ``rng.integers(0, n, size=n)`` once per draw from ``default_rng(seed)``, in the
    order of the training ids, before anything else.
    """
    rng = np.random.default_rng(seed)
    return [rng.integers(0, data.n, size=data.n) for _ in range(m_draws)]


def imputation_design(data: CorrectionData) -> np.ndarray:
    """The design ``[1, score, covariate]`` of every text, in data order."""
    score = data.frame["score"].to_numpy(dtype=float)
    return np.column_stack([np.ones(data.n), score, data.covariate])


def mirror_columns(seed: int = 1) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Two columns that mirror each other, so the row mean of their standardised values is 0."""
    rng = np.random.default_rng(seed)
    signal = rng.normal(size=40)
    return {"a": signal, "b": -signal}, signal + rng.normal(scale=0.3, size=40)


def orthogonal_columns(seed: int = 1) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Two centred columns of equal length that are exactly orthogonal over their 40 rows.

    Their correlation matrix is the identity, so the start of the principal-axis iteration has
    two equal eigenvalues, and so do the singular values of the two standardised columns.
    """
    first = np.array([1.0, -1.0, 1.0, -1.0] * 10)
    second = np.array([1.0, 1.0, -1.0, -1.0] * 10)
    noise = np.random.default_rng(seed).normal(scale=0.1, size=40)
    return {"a": first, "b": second}, first + 0.5 * second + noise


def weak_columns(seed: int = WEAK_SEED) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Three columns that share a weak signal: 50 rounds of the iteration do not converge."""
    rng = np.random.default_rng(seed)
    signal = rng.normal(size=60)
    columns = {f"c{index}": 0.2 * signal + rng.normal(size=60) for index in range(3)}
    return columns, signal + rng.normal(scale=0.5, size=60)


def strong_columns(seed: int = 4) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Three noisy copies of one strong signal, on which the iteration converges."""
    rng = np.random.default_rng(seed)
    signal = rng.normal(size=600)
    columns = {
        f"copy_{index}": 2.0 * signal + 3.0 + rng.normal(scale=scale * 2.0, size=600)
        for index, scale in enumerate((0.6, 0.8, 1.0))
    }
    return columns, signal + rng.normal(scale=0.2, size=600)


def columns_data(columns: dict[str, np.ndarray], gold: np.ndarray) -> CorrectionData:
    """Correction data whose score columns are the given columns, every text labelled."""
    return CorrectionData(frame=text_frame(columns, gold, 2), score_columns=tuple(columns))


def members_of(columns: dict[str, np.ndarray]) -> tuple[dict[str, pd.Series], np.ndarray]:
    """The columns as ensemble members indexed by text id, with the ids."""
    n = next(iter(columns.values())).size
    ids = np.array([f"t{index:03d}" for index in range(n)])
    return {name: pd.Series(values, index=ids) for name, values in columns.items()}, ids
