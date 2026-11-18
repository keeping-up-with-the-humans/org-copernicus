"""The battery and the repeated study sum up the warnings of the cases their inner calls meet.

A fit, a combination, a cross-fit or an estimate called on its own logs one warning for each
case it meets. Inside ``run_battery`` and ``repeated_study`` these warnings are counted instead
and logged as one summary per entry and size of the battery, or per size of the study, with the
count of each case; every record stays, no logger's level changes and no other warning is held
back. The expected counts come from the records of the folds, which hold every case here.
"""

from __future__ import annotations

import json
import logging
from collections import Counter

import numpy as np
import pandas as pd
import pytest
from corrections_support import BATTERY_SCORES, battery_data, battery_frame
from traces_support import columns_data, study_corpus, weak_columns

from copernicus.corrections import (
    BatterySettings,
    CorrectionData,
    get_correction,
    repeated_study,
    run_battery,
)
from copernicus.corrections.battery_settings import size_label

pytestmark = pytest.mark.filterwarnings("error")

INNER = {
    "copernicus.corrections.families._crossfit_ghat",
    "copernicus.corrections.families.imputation",
    "copernicus.corrections.families.latent",
    "copernicus.corrections.families.ensemble",
    "copernicus.corrections.estimates",
}
BATTERY, STUDY = "copernicus.corrections.battery", "copernicus.corrections.study"
PHRASES = {
    "rank_deficient_folds": "a cross-fitted prediction has folds whose design has a numerical "
    "rank below 3 (n_rank_deficient_folds)",
    "rank_deficient_resamples": "a fit has bootstrap resamples whose design has a numerical rank "
    "below 3 (n_rank_deficient_resamples)",
    "no_residual_df": "a fit has bootstrap fits with no more rows than coefficients "
    "(n_resamples_without_residual_df)",
    "orientation_open": "the sign of a result is the one that its decomposition gave it, because "
    "the column that would set it or its reference is constant by is_constant "
    "(orientation_identified)",
    "direction_tied": "the direction of a result counts as not unique (direction_unique)",
    "not_converged": "the principal-axis iteration reached its cap of rounds without meeting "
    "its tolerance (factor_converged)",
    "se_zero": "a standard error is exactly 0 (se_exactly_zero)",
}
FAMILIES = [
    ("regression_calibration", "linear"),
    ("latent_pooling", "factor_anchor"),
    ("multiple_imputation", "model_draws"),
    ("multiple_imputation", "pmm"),
    ("correction_ensemble", "mean"),
    ("correction_ensemble", "factor"),
    ("correction_ensemble", "pc1"),
]
SETTINGS = BatterySettings(
    families=FAMILIES,  # type: ignore[arg-type]
    sizes=(3, 8, 30, 0.1),
    schemes=("kfold2", "holdout"),
    seeds=(1, 2),
)


def _frame(label: str) -> pd.DataFrame:
    """The corpora of the three small batteries: ordinary scores, weak scores, single texts."""
    if label == "single":
        return battery_frame(n_groups=80, group_size=1)
    ordinary = battery_frame(n_groups=60, group_size=2)
    if label == "ordinary":
        return ordinary
    rng = np.random.default_rng(8)
    noisy = {
        name: 0.15 * ordinary["gold"] + rng.normal(size=len(ordinary)) for name in BATTERY_SCORES
    }
    return ordinary.assign(**noisy)


def _cases(fit: dict[str, object]) -> Counter[str]:
    """The cases that the records of one fitted fold show, each counted once."""
    found: Counter[str] = Counter()
    found["rank_deficient_resamples"] += int(fit.get("n_rank_deficient_resamples", 0) > 0)  # type: ignore[operator]
    found["no_residual_df"] += int(fit.get("n_resamples_without_residual_df", 0) > 0)  # type: ignore[operator]
    found["orientation_open"] += int(fit.get("orientation_identified") is False)
    found["direction_tied"] += int(fit.get("direction_unique") is False)
    found["not_converged"] += int(fit.get("factor_converged") is False)
    return +found


def _expected(result: object) -> dict[tuple[str, str, str], Counter[str]]:
    """The cases of each entry and size from the folds' records, in the order of the settings."""
    found: dict[tuple[str, str, str], Counter[str]] = {}
    for row in result.results.itertuples(index=False):  # type: ignore[attr-defined]
        if not row.diagnostic:
            for fold in json.loads(row.provenance):
                key = (row.family, row.variant, row.size)
                found[key] = found.get(key, Counter()) + _cases(fold["fit"])
    order = [
        (family, variant, size_label(size))
        for family, variant in FAMILIES
        for size in SETTINGS.sizes
    ]
    return {key: found[key] for key in order if found.get(key)}


def _warnings(caplog: pytest.LogCaptureFixture, name: str) -> list[str]:
    return [
        r.getMessage() for r in caplog.records if r.name == name and r.levelno == logging.WARNING
    ]


@pytest.mark.parametrize(("label", "inner"), [("ordinary", 22), ("weak", 40), ("single", 34)])
def test_the_battery_logs_one_summary_per_entry_and_size(
    label: str, inner: int, caplog: pytest.LogCaptureFixture
) -> None:
    levels = {name: logging.getLogger(name).level for name in INNER}
    with caplog.at_level(logging.WARNING):
        result = run_battery(battery_data(_frame(label)), SETTINGS)
    assert [record for record in caplog.records if record.name in INNER] == []
    assert {name: logging.getLogger(name).level for name in INNER} == levels
    summaries = [text for text in _warnings(caplog, BATTERY) if text.startswith("battery, entry ")]
    expected = _expected(result)
    # The fits of these runs, called on their own, would log 22, 40 and 34 warnings.
    assert sum(sum(counts.values()) for counts in expected.values()) == inner
    assert len(summaries) == len(expected)
    for text, ((family, variant, size), counts) in zip(summaries, expected.items(), strict=True):
        assert text.startswith(
            f"battery, entry {family}/{variant}, size {size}: of the warnings that its fits log "
            "one by one when called on their own, "
        )
        for case, phrase in PHRASES.items():
            verb = "says" if counts[case] == 1 else "say"
            assert (f"{counts[case]} {verb} that {phrase}" in text) is (case in counts), (
                text,
                case,
            )
        assert text.endswith("; the provenance of each scored fold holds the records of its fit")


def test_the_study_logs_one_summary_per_size_with_its_cross_fitted_draws_counted_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        table = repeated_study(
            study_corpus(covariate_on_score=True), truth_gold="truth", sizes=(20, 40), draws=6
        )
    assert [record for record in caplog.records if record.name in INNER] == []
    # Six draws of each size, each draw cross-fitted once for both of its estimands.
    assert (table.loc[table["estimator"] == "ppi_crossfit", "n_rank_deficient_folds"] == 30).all()
    summaries = _warnings(caplog, STUDY)
    assert summaries == [
        f"repeated study, size {size}: of the warnings that its inner calls log one by one when "
        f"called on their own, 6 say that {PHRASES['rank_deficient_folds']}; the column "
        "n_rank_deficient_folds holds the folds of each row"
        for size in (20, 40)
    ]


def test_the_study_counts_the_standard_errors_of_exactly_zero(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # A score of 2 and a human value of 3 on every text: every estimate of ppi_identity has the
    # standard error 0, the mean and the difference of each of the three draws.
    ids = [f"t{index:02d}" for index in range(40)]
    frame = pd.DataFrame(
        {
            "text_id": ids,
            "group_id": ids,
            "score": 2.0,
            "gold": 3.0,
            "truth": 3.0,
            "log_chars": np.random.default_rng(0).normal(size=40),
        }
    )
    with caplog.at_level(logging.WARNING):
        table = repeated_study(
            CorrectionData(frame=frame, score_columns=("score",)),
            truth_gold="truth",
            sizes=(10,),
            draws=3,
            estimators=("ppi_identity",),
        )
    assert [record for record in caplog.records if record.name in INNER] == []
    assert _warnings(caplog, STUDY) == [
        "repeated study, size 10: of the warnings that its inner calls log one by one when called "
        f"on their own, 6 say that {PHRASES['se_zero']}; the column n_se_exactly_zero holds the "
        "draws of each row"
    ]
    assert table["n_se_exactly_zero"].tolist() == [3, 3]
    assert table["se_mean"].tolist() == [0.0, 0.0]


def test_a_call_on_its_own_still_warns_once_after_a_battery(
    caplog: pytest.LogCaptureFixture,
) -> None:
    run_battery(battery_data(_frame("weak")), SETTINGS)
    data = columns_data(*weak_columns())
    with caplog.at_level(logging.WARNING):
        fitted = get_correction("latent_pooling", "factor_anchor").fit(
            data, data.ids, rng=np.random.default_rng(0)
        )
    assert fitted.provenance["factor_converged"] is False
    (message,) = _warnings(caplog, "copernicus.corrections.families.latent")
    assert message.startswith("latent pooling: the principal-axis iteration reached its cap of 50")


def test_only_the_warnings_of_the_cases_are_counted_instead_of_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from copernicus.corrections import _open_cases

    log = logging.getLogger("copernicus.corrections.families.latent")
    with caplog.at_level(logging.WARNING):
        with _open_cases.gathered() as counts:
            _open_cases.warn_not_converged(log, "probe", 50, 1e-6)
            log.warning("another warning")
        _open_cases.warn_not_converged(log, "probe", 50, 1e-6)
    assert counts == Counter({"not_converged": 1})
    messages = [record.getMessage() for record in caplog.records]
    assert messages[0] == "another warning" and len(messages) == 2
    assert messages[1].startswith("probe: the principal-axis iteration reached its cap of 50")
