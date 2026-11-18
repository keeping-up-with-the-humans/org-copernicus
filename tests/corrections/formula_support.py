"""Synthetic data for the tests of the formulas of the OLS estimators.

The frame holds complete covariates and incomplete columns under several names, so a test can
tell a formula that reads an incomplete column from one that only holds its name. The report of
``formulaic`` on the columns a formula reads can be replaced by one that a test chooses.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from estimates_support import with_labels

from copernicus.corrections import CorrectionData

N_TEXTS = 60


def formula_frame() -> pd.DataFrame:
    """Sixty texts with complete covariates and incomplete columns of several names.

    ``gap`` and ``gap col`` are covariates with three missing values; ``df``, ``mean`` and
    ``values`` hold the same values under the names of a keyword argument, a method and an
    attribute that formulas use.
    """
    rng = np.random.default_rng(5)
    covariate = rng.normal(size=N_TEXTS)
    gold = 1.0 + 0.6 * covariate + rng.normal(scale=0.5, size=N_TEXTS)
    gap = rng.normal(size=N_TEXTS)
    gap[[4, 17, 40]] = np.nan
    return pd.DataFrame(
        {
            "text_id": [f"t{index:03d}" for index in range(N_TEXTS)],
            "group_id": [f"g{index // 3:03d}" for index in range(N_TEXTS)],
            "score": gold + 0.3 + rng.normal(scale=0.3, size=N_TEXTS),
            "gold": gold,
            "y": 0.5 * gold + rng.normal(scale=0.4, size=N_TEXTS),
            "log_chars": covariate,
            "log chars": covariate,
            "gap": gap,
            "gap col": gap,
            "df": gap,
            "mean": gap,
            "values": gap,
        }
    )


def formula_data(frame: pd.DataFrame | None = None) -> CorrectionData:
    """The frame with the gold value kept on every third text only."""
    labelled = np.arange(N_TEXTS) % 3 == 0
    texts = formula_frame() if frame is None else frame
    return with_labels(texts, labelled, score_columns=("score",), pi=1 / 3)


def fix_report(monkeypatch: pytest.MonkeyPatch, report: str | set[str]) -> None:
    """Make the formula library report the given columns or nothing, or leave it as installed."""
    if report == "installed":
        return
    columns = set() if report == "empty" else report
    monkeypatch.setattr(
        "formulaic.Formula", lambda formula: SimpleNamespace(required_variables=columns)
    )
