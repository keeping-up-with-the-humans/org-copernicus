"""Battery settings, the entries they expand to, and how fitting failures are recorded."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd
import pytest
from battery_support import LINEAR, RIDGE, SMALL, altered_linear, failing_linear
from corrections_support import battery_data, battery_frame, make_embedding_set

from copernicus.corrections import REGISTRY, CorrectionData
from copernicus.corrections.battery import BatterySettings, run_battery, summarize_battery
from copernicus.corrections.battery_settings import BatteryEntry
from copernicus.errors import EstimatorError, SettingsError


def test_the_defaults_are_the_documented_grid() -> None:
    settings = BatterySettings()
    assert settings.families == "all"
    assert settings.sizes == (50, 100, 250, 500, 1000, 0.01, 0.05, 0.10, 0.25)
    assert settings.schemes == ("kfold5", "kfold10", "holdout")
    assert settings.seeds == (42, 43, 44)
    assert settings.covariate_scale == "standardized"
    assert json.loads(json.dumps(asdict(settings), default=str))["sizes"][0] == 50


def test_all_expands_to_every_per_text_variant_and_then_the_ensemble() -> None:
    entries = BatterySettings().entries
    per_text = [
        (spec.family, variant)
        for spec in REGISTRY.values()
        if spec.level == "score" and spec.factory is not None
        for variant in spec.variants
    ]
    ensemble = [
        ("correction_ensemble", variant) for variant in REGISTRY["correction_ensemble"].variants
    ]
    assert [(entry.family, entry.variant) for entry in entries] == per_text + ensemble
    families = {entry.family for entry in entries}
    assert families.isdisjoint(
        {"pseudo_values", "blind_transport", "ludwig_moment", "moment_subtraction", "simex"}
    )
    assert "instrumented_coding" not in families
    assert all(entry.params == () for entry in entries)


def test_all_resolves_the_embedding_entries_against_the_sets_of_the_data(
    caplog: pytest.LogCaptureFixture,
) -> None:
    settings = BatterySettings(**SMALL)
    stacked = [entry for entry in settings.entries if entry.family == "embedding_augmented"]
    assert len(stacked) == 3 and all(entry.params == () for entry in stacked)
    assert settings.for_embedding_sets(("e1", "e2")) is settings
    single = settings.for_embedding_sets(("e1",))
    labels = [entry.label for entry in single.entries if entry.family == "embedding_augmented"]
    assert labels == [
        "embeddings_only(encoder=e1)",
        "embeddings_plus_score(encoder=e1)",
        "score_anchored_residual(encoder=e1)",
    ]
    assert [entry.family for entry in single.entries] == [e.family for e in settings.entries]
    assert (single.sizes, single.schemes, single.seeds) == (
        settings.sizes,
        settings.schemes,
        settings.seeds,
    )
    with caplog.at_level(logging.INFO, logger="copernicus.corrections.battery_settings"):
        none = settings.for_embedding_sets(())
    assert [entry for entry in none.entries if entry.family == "embedding_augmented"] == []
    assert len(none.entries) == len(settings.entries) - 3
    notes = [record for record in caplog.records if "embedding" in record.getMessage()]
    assert len(notes) == 1 and notes[0].levelno == logging.INFO
    listed = BatterySettings(families=[LINEAR, RIDGE], **SMALL)
    assert listed.for_embedding_sets(()) is listed


def test_an_entry_label_names_its_parameters() -> None:
    entry = BatteryEntry("embedding_augmented", "embeddings_only", (("encoder", "minilm"),))
    assert entry.label == "embeddings_only(encoder=minilm)"
    assert entry.key == "embedding_augmented:embeddings_only(encoder=minilm)"
    assert entry.kwargs == {"encoder": "minilm"}
    assert BatteryEntry(*LINEAR).label == "linear"


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"families": []}, "at least one"),
        ({"families": "some"}, "'all' or a sequence"),
        ({"families": [("nonesuch", "x")]}, "unknown family 'nonesuch'"),
        ({"families": [("regression_calibration", "cubic")]}, "unknown variant 'cubic'"),
        ({"families": [("pseudo_values", "identity")]}, "estimate level"),
        ({"families": [("simex", "simex")]}, "estimate level"),
        ({"families": [("blind_transport", "transport")]}, "not part of the battery"),
        ({"families": ["regression_calibration"]}, "(family, variant)"),
        ({"families": [("regression_calibration", "linear", "x")]}, "mapping"),
        ({"families": [("multiple_imputation", "pmm", {"draws": 3})]}, "'draws'"),
        ({"families": [("multiple_imputation", "pmm", {"m_draws": 0})]}, "m_draws"),
        ({"families": [("multiple_imputation", "pmm", {"m_draws": 2.5})]}, "m_draws"),
        ({"families": [("multiple_imputation", "pmm", {"k_donors": 2.5})]}, "k_donors"),
        (
            {"families": [("misclassification_matrix", "posterior", {"youden_floor": "x"})]},
            "youden",
        ),
        ({"families": [("multiple_imputation", "pmm", {"m_draws": [3]})]}, "hashable"),
        ({"families": [("correction_ensemble", "mean", {"k": 1}), LINEAR, RIDGE]}, "no parameters"),
        ({"families": [LINEAR, LINEAR]}, "twice"),
        ({"families": [("correction_ensemble", "mean"), LINEAR]}, "two per-text entries"),
        ({"sizes": ()}, "at least one size"),
        ({"sizes": (1,)}, "at least 2"),
        ({"sizes": (0.6,)}, "(0, 0.5]"),
        ({"sizes": (0.0,)}, "(0, 0.5]"),
        ({"sizes": (float("nan"),)}, "(0, 0.5]"),
        ({"sizes": (True,)}, "integer"),
        ({"sizes": ("50",)}, "integer"),
        ({"sizes": (50, 50)}, "repeat"),
        ({"sizes": "50"}, "sequence"),
        ({"schemes": ()}, "at least one scheme"),
        ({"schemes": ("kfold1",)}, "kfold<k>"),
        ({"schemes": ("kfold",)}, "kfold<k>"),
        ({"schemes": ("kfold05",)}, "kfold<k>"),
        ({"schemes": ("loo",)}, "kfold<k>"),
        ({"schemes": ("holdout", "holdout")}, "repeat"),
        ({"seeds": ()}, "at least one seed"),
        ({"seeds": (42, 42)}, "repeat"),
        ({"seeds": (1.5,)}, "integer"),
        ({"seeds": (True,)}, "integer"),
        ({"covariate_scale": "per_unit"}, "covariate_scale"),
    ],
)
def test_settings_outside_their_range_are_refused(options: dict[str, Any], message: str) -> None:
    with pytest.raises(SettingsError) as raised:
        BatterySettings(**options)
    assert message in str(raised.value)


def test_parameters_reach_the_family() -> None:
    entry = ("multiple_imputation", "model_draws", {"m_draws": 3})
    settings = BatterySettings(families=[entry, LINEAR], **SMALL)
    result = run_battery(battery_data(), settings)
    rows = result.results[~result.results["diagnostic"]]
    assert list(rows["variant"]) == ["model_draws(m_draws=3)", "linear"]
    folds = json.loads(rows.iloc[0]["provenance"])
    assert {fold["fit"]["m_draws"] for fold in folds} == {3}


def test_the_variants_of_one_family_share_their_inner_draws() -> None:
    frame = battery_frame()
    rng = np.random.default_rng(2)
    vectors = frame["gold"].to_numpy()[:, None] * rng.normal(size=8) + rng.normal(size=(600, 8))
    data = CorrectionData(
        frame=frame,
        score_columns=("score_a", "score_b", "score_c"),
        embeddings={"e1": make_embedding_set(frame["text_id"].tolist(), vectors, "e1")},
    )
    on_e1 = {"encoder": "e1"}
    families = [
        ("embedding_augmented", "embeddings_only", on_e1),
        ("embedding_augmented", "score_anchored_residual", on_e1),
        RIDGE,
    ]
    result = run_battery(data, BatterySettings(families=families, **SMALL))
    rows = result.results[~result.results["diagnostic"]]
    assert list(rows["variant"]) == [
        "embeddings_only(encoder=e1)",
        "score_anchored_residual(encoder=e1)",
        "ridge",
    ]
    seeds = [
        [fold["fit"]["inner_seed"] for fold in json.loads(text)] for text in rows["provenance"]
    ]
    assert seeds[0] == seeds[1]
    assert len(set(seeds[0])) == 5
    assert not set(seeds[0]) & set(seeds[2])
    encoders = {fold["fit"]["encoder"] for fold in json.loads(rows.iloc[0]["provenance"])}
    assert encoders == {"e1"}


def test_the_mode_ensemble_on_continuous_gold_is_a_skip_row() -> None:
    settings = BatterySettings(families=[LINEAR, RIDGE, ("correction_ensemble", "mode")], **SMALL)
    result = run_battery(battery_data(), settings)
    mode = result.skipped[result.skipped["variant"] == "mode"]
    assert len(mode) == 5
    assert mode["reason"].str.startswith("NotApplicableError: the mode combiner").all()
    assert len(result.results[~result.results["diagnostic"]]) == 2


def test_an_entry_that_does_not_apply_is_skipped_with_its_reason() -> None:
    settings = BatterySettings(families=[("regression_calibration", "platt"), LINEAR], **SMALL)
    result = run_battery(battery_data(), settings)
    platt = result.skipped[result.skipped["variant"] == "platt"]
    assert len(platt) == 5
    assert platt["reason"].str.startswith("not_applicable: platt calibration needs binary").all()
    summary = summarize_battery(result).set_index("variant")
    assert summary.loc["platt", "n_rows"] == 0 and summary.loc["platt", "n_skipped"] == 5
    assert np.isnan(summary.loc["platt", "gain_rank_validity"])
    assert summary.loc["linear", "n_rows"] == 1


def test_an_estimator_error_on_one_fold_is_a_skip_row_and_the_rest_survive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failing_linear(monkeypatch, EstimatorError("planted failure"))
    result = run_battery(battery_data(), BatterySettings(families=[LINEAR, RIDGE], **SMALL))
    assert list(result.skipped["fold"]) == [2]
    assert result.skipped.loc[0, "reason"] == "EstimatorError: planted failure"
    rows = result.results[~result.results["diagnostic"]].set_index("variant")
    assert rows.loc["linear", "n_folds"] == 4 and rows.loc["ridge", "n_folds"] == 5
    folds = [fold["fold"] for fold in json.loads(rows.loc["linear", "provenance"])]
    assert folds == [0, 1, 3, 4]


def test_any_other_exception_stops_the_battery(monkeypatch: pytest.MonkeyPatch) -> None:
    failing_linear(monkeypatch, ValueError("a bug"))
    with pytest.raises(ValueError, match="a bug"):
        run_battery(battery_data(), BatterySettings(families=[LINEAR, RIDGE], **SMALL))


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        (lambda predicted: pd.Series(1.5, index=predicted.index), "constant_training_range"),
        (
            lambda predicted: predicted.where(predicted.index != predicted.index[0]),
            "non_finite_predictions",
        ),
    ],
)
def test_predictions_without_a_scale_are_skip_rows(
    monkeypatch: pytest.MonkeyPatch, change: Callable[[pd.Series], pd.Series], reason: str
) -> None:
    altered_linear(monkeypatch, change)
    result = run_battery(battery_data(), BatterySettings(families=[LINEAR, RIDGE], **SMALL))
    assert list(result.skipped["fold"]) == [0, 1, 2, 3, 4]
    assert result.skipped["reason"].str.startswith(f"{reason}: ").all()
    rows = result.results[~result.results["diagnostic"]]
    assert list(rows["variant"]) == ["ridge"]


def test_predictions_indexed_by_other_ids_stop_the_battery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    altered_linear(monkeypatch, lambda predicted: predicted.iloc[::-1])
    with pytest.raises(EstimatorError, match="not indexed by the ids asked for"):
        run_battery(battery_data(), BatterySettings(families=[LINEAR, RIDGE], **SMALL))
