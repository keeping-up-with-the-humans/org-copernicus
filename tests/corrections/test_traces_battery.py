"""The battery's provenance carries the records of the fitted families and of the ensemble.

Each scored fold's ``fit`` holds the records of latent pooling, multiple imputation and the
``factor`` and ``pc1`` combiners. Their warnings are summed up in the battery's own warnings,
one for each entry and size, and the battery's warning about the variation of its diagnostic
draws stays as it was.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pytest
from text_count_support import DOMINANT_BATTERY, battery_data

from copernicus.corrections import BatterySettings, CorrectionData, run_battery

pytestmark = pytest.mark.filterwarnings("error")

FAMILIES = "copernicus.corrections.families."
RECORDS = {
    ("latent_pooling", "factor_anchor"): {
        "orientation_identified": bool,
        "direction_unique": bool,
        "factor_converged": bool,
        "n_factor_rounds": int,
    },
    ("multiple_imputation", "model_draws"): {
        "n_rank_deficient_resamples": int,
        "n_resamples_without_residual_df": int,
    },
    ("correction_ensemble", "factor"): {
        "orientation_identified": bool,
        "direction_unique": bool,
        "factor_converged": bool,
        "n_factor_rounds": int,
    },
    ("correction_ensemble", "pc1"): {"orientation_identified": bool, "direction_unique": bool},
}
SETTINGS = {"sizes": (8, 12, 50, 0.1), "schemes": ("kfold2", "holdout"), "seeds": (1, 2)}


def _data() -> CorrectionData:
    """The battery corpus with a dominant group, and a second, noisier score column."""
    data = battery_data(DOMINANT_BATTERY, 11)
    noise = np.random.default_rng(12).normal(scale=0.5, size=data.n)
    frame = data.frame.assign(second=data.frame["score"].to_numpy() + noise)
    return CorrectionData(frame=frame, score_columns=("score", "second"))


def _variation(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if "coefficient of variation" in r.getMessage()]


def test_every_scored_fold_carries_the_records_and_the_diagnostic_warning_stays(
    caplog: pytest.LogCaptureFixture,
) -> None:
    families = [("regression_calibration", "linear"), *RECORDS]
    with caplog.at_level(logging.WARNING):
        baseline = run_battery(
            _data(),
            BatterySettings(families=[("regression_calibration", "linear")], **SETTINGS),  # type: ignore[arg-type]
        )
    alone = _variation(caplog)
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        result = run_battery(_data(), BatterySettings(families=families, **SETTINGS))  # type: ignore[arg-type]
    assert _variation(caplog) == alone and len(alone) == 2
    assert int(result.results["diagnostic"].sum()) == int(baseline.results["diagnostic"].sum())
    others = [r for r in caplog.records if "coefficient of variation" not in r.getMessage()]
    assert not [record for record in others if record.name.startswith(FAMILIES)]
    assert all(record.getMessage().startswith("battery, entry ") for record in others)
    checked = 0
    for row in result.results.itertuples(index=False):
        expected = RECORDS.get((row.family, row.variant))
        if expected is None:
            continue
        for fold in json.loads(row.provenance):
            for key, kind in expected.items():
                assert isinstance(fold["fit"][key], kind), (row.family, row.variant, key)
            checked += 1
    fitted = set(zip(result.results["family"], result.results["variant"], strict=True))
    assert checked > 0 and set(RECORDS) <= fitted
