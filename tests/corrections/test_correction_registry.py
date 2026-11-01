"""The registry of corrections: lookup, refusals, and the listing."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import FrozenInstanceError
from types import MappingProxyType

import pandas as pd
import pytest

from copernicus.corrections import (
    REGISTRY,
    FamilySpec,
    get_correction,
    list_corrections,
    register,
)
from copernicus.errors import ConfigurationError, SettingsError

REGISTERED = {
    "regression_calibration": ("linear", "covariate_augmented", "platt", "isotonic"),
    "marginal_matching": ("mean_sd", "quantile"),
}
FAMILY_ERRORS = {
    "regression_calibration": (
        "level_displacement",
        "calibration_shortfall",
        "covariate_linked_error",
    ),
    "marginal_matching": ("level_displacement", "calibration_shortfall"),
}
LEVEL_AND_SPREAD = ("level_displacement", "calibration_shortfall")
VARIANT_ERRORS = {
    ("regression_calibration", "linear"): LEVEL_AND_SPREAD,
    (
        "regression_calibration",
        "covariate_augmented",
    ): ("level_displacement", "calibration_shortfall", "covariate_linked_error"),
    ("regression_calibration", "platt"): LEVEL_AND_SPREAD,
    ("regression_calibration", "isotonic"): LEVEL_AND_SPREAD,
    ("marginal_matching", "mean_sd"): LEVEL_AND_SPREAD,
    ("marginal_matching", "quantile"): LEVEL_AND_SPREAD,
    ("classical", "fuller_shrinkage"): ("calibration_shortfall",),
    ("classical", "tweedie"): ("calibration_shortfall", "noise"),
}


def test_registry_holds_the_registered_families_read_only() -> None:
    assert isinstance(REGISTRY, MappingProxyType)
    for family, variants in REGISTERED.items():
        spec = REGISTRY[family]
        assert spec.family == family
        assert spec.variants == variants
        assert (spec.target, spec.level) == ("A", "score")
        assert spec.errors_addressed == FAMILY_ERRORS[family]
        assert spec.estimate_errors == "none"
        assert spec.factory is not None
    with pytest.raises(FrozenInstanceError):
        REGISTRY["regression_calibration"].target = "B"  # type: ignore[misc]


def test_list_corrections_has_one_row_per_variant() -> None:
    table = list_corrections()
    assert isinstance(table, pd.DataFrame)
    assert list(table.columns) == [
        "family",
        "variant",
        "target",
        "level",
        "errors_addressed",
        "estimate_errors",
    ]
    mine = table[table["family"].isin(REGISTERED)]
    expected = [(f, v) for f, variants in REGISTERED.items() for v in variants]
    assert list(zip(mine["family"], mine["variant"], strict=True)) == expected
    assert set(mine["level"]) == {"score"}
    assert set(mine["target"]) == {"A"}
    assert set(mine["estimate_errors"]) == {"none"}
    for (family, variant), words in VARIANT_ERRORS.items():
        row = table[(table["family"] == family) & (table["variant"] == variant)]
        assert row["errors_addressed"].tolist() == [", ".join(words)]


def test_variant_errors_are_validated_and_fall_back_to_the_family_tuple() -> None:
    def build(variant_errors: dict[str, tuple[str, ...]] | None) -> FamilySpec:
        return FamilySpec(
            family="probe",
            variants=("a", "b"),
            target="A",
            level="score",
            errors_addressed=("level_displacement", "noise"),
            estimate_errors="none",
            factory=lambda variant: None,  # type: ignore[arg-type, return-value]
            variant_errors=variant_errors,
        )

    spec = build({"a": ("noise",)})
    assert spec.errors_for("a") == ("noise",)
    assert spec.errors_for("b") == ("level_displacement", "noise")
    assert build(None).errors_for("a") == ("level_displacement", "noise")
    with pytest.raises(SettingsError, match=r"'c'.*not variants"):
        build({"c": ("noise",)})
    with pytest.raises(SettingsError, match="unknown error word"):
        build({"a": ("bias",)})


def test_get_correction_builds_the_named_variant() -> None:
    correction = get_correction("marginal_matching", "quantile")
    assert (correction.family, correction.variant, correction.target) == (
        "marginal_matching",
        "quantile",
        "A",
    )


def test_unknown_family_lists_the_registered_families() -> None:
    with pytest.raises(SettingsError, match=r"unknown family 'nope'.*regression_calibration"):
        get_correction("nope", "linear")
    with pytest.raises(SettingsError, match="marginal_matching"):
        get_correction("nope", "linear")


def test_unknown_variant_lists_the_variants_of_the_family() -> None:
    with pytest.raises(SettingsError, match=r"unknown variant 'cubic'.*covariate_augmented"):
        get_correction("regression_calibration", "cubic")


def test_the_pmm_variant_carries_its_note_in_the_registry() -> None:
    spec = REGISTRY["multiple_imputation"]
    note = spec.note_for("pmm")
    assert "predictive mean matching" in note and "removed" in note and "battery" in note
    assert spec.note_for("model_draws") == ""
    assert REGISTRY["regression_calibration"].note_for("linear") == ""
    with pytest.raises(SettingsError, match=r"'c'.*not variants"):
        _spec(variants=("a", "b"), variant_notes={"c": "a note"})
    with pytest.raises(SettingsError, match="non-empty text"):
        _spec(variants=("a",), variant_notes={"a": ""})


def test_a_spec_without_a_factory_cannot_be_built() -> None:
    spec = FamilySpec(
        family="estimate_only_probe",
        variants=("one",),
        target="C",
        level="estimate",
        errors_addressed=("noise",),
        estimate_errors="probe estimate error",
        factory=None,
    )
    register(spec)
    try:
        assert "estimate_only_probe" in REGISTRY
        with pytest.raises(SettingsError, match="estimate level"):
            get_correction("estimate_only_probe", "one")
        with pytest.raises(ConfigurationError, match="already registered"):
            register(spec)
    finally:
        from copernicus.corrections.families import _registry

        _registry._REGISTRY.pop("estimate_only_probe")


def _spec(**changes: object) -> FamilySpec:
    fields: dict[str, object] = {
        "family": "probe",
        "variants": ("v",),
        "target": "A",
        "level": "score",
        "errors_addressed": ("noise",),
        "estimate_errors": "none",
        "factory": lambda variant: None,
    }
    return FamilySpec(**{**fields, **changes})  # type: ignore[arg-type]


def test_family_spec_refuses_inconsistent_entries() -> None:
    assert _spec().family == "probe"
    with pytest.raises(SettingsError, match="level"):
        _spec(level="other")
    with pytest.raises(SettingsError, match="non-empty and unique"):
        _spec(variants=())
    with pytest.raises(SettingsError, match="non-empty and unique"):
        _spec(variants=("v", "w", "v"))
    assert _spec(factory=None).factory is None
    with pytest.raises(SettingsError, match="callable factory"):
        _spec(factory="not callable")
    with pytest.raises(SettingsError, match="factory=None"):
        _spec(level="estimate")
    assert _spec(level="estimate", factory=None, errors_addressed=()).level == "estimate"
    with pytest.raises(SettingsError, match="unknown error word"):
        _spec(errors_addressed=("level displacement",))
    with pytest.raises(SettingsError, match="name"):
        _spec(family="")
    with pytest.raises(SettingsError, match="estimate_errors"):
        _spec(estimate_errors="")


def test_importing_corrections_loads_no_scikit_learn_even_with_families_registered() -> None:
    code = (
        "import sys, copernicus.corrections as c; "
        "assert 'regression_calibration' in c.REGISTRY; "
        "bad = [m for m in ('sklearn', 'torch') if m in sys.modules]; "
        "sys.exit(len(bad))"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_the_second_group_of_families_is_registered_with_the_wording_of_the_table() -> None:
    expected = {
        "multiple_imputation": (
            ("model_draws", "pmm"),
            "B",
            ("level_displacement", "calibration_shortfall", "noise"),
            "confidence-interval coverage via Rubin's rules",
        ),
        "misclassification_matrix": (
            ("posterior",),
            "A, C",
            ("level_displacement",),
            "bias of class shares",
        ),
        "latent_pooling": (("factor_anchor",), "A", ("noise", "calibration_shortfall"), "none"),
        "classical": (
            ("fuller_shrinkage", "tweedie"),
            "A",
            ("calibration_shortfall", "noise"),
            "none",
        ),
    }
    for family, (variants, target, words, estimate_errors) in expected.items():
        spec = REGISTRY[family]
        assert (spec.variants, spec.target, spec.level) == (variants, target, "score")
        assert spec.errors_addressed == words
        assert spec.estimate_errors == estimate_errors
        assert spec.factory is not None
    listed = list_corrections()["family"].unique().tolist()
    assert set(listed) >= {"regression_calibration", "marginal_matching", *expected}
    assert len(listed) >= 6


def test_stacking_and_embedding_augmented_calibration_complete_eight_families() -> None:
    words = ("calibration_shortfall", "level_displacement", "covariate_linked_error")
    expected = {
        "supervised_stacking": (("ridge", "lasso", "enet"), "A", "none"),
        "embedding_augmented": (
            ("embeddings_only", "embeddings_plus_score", "score_anchored_residual"),
            "A, then C",
            "with pseudo-values on top: bias and coverage, narrower confidence intervals",
        ),
    }
    for family, (variants, target, estimate_errors) in expected.items():
        spec = REGISTRY[family]
        assert (spec.variants, spec.target, spec.level) == (variants, target, "score")
        assert spec.errors_addressed == words
        assert spec.estimate_errors == estimate_errors
        assert spec.factory is not None
    assert set(list_corrections()["family"]) >= {
        "regression_calibration",
        "marginal_matching",
        "multiple_imputation",
        "misclassification_matrix",
        "latent_pooling",
        "classical",
        "supervised_stacking",
        "embedding_augmented",
    }


ELEVEN_FAMILIES = {
    "regression_calibration",
    "marginal_matching",
    "multiple_imputation",
    "misclassification_matrix",
    "latent_pooling",
    "classical",
    "supervised_stacking",
    "embedding_augmented",
    "pseudo_values",
    "correction_ensemble",
    "blind_transport",
}


def test_pseudo_values_ensemble_and_transport_complete_eleven_families() -> None:
    assert set(list_corrections()["family"]) >= ELEVEN_FAMILIES


def test_the_estimate_level_estimators_complete_fifteen_families() -> None:
    coefficient = "attenuation of the construct's coefficient as predictor"
    expected = {
        "ludwig_moment": (
            ("mean", "difference", "ols_outcome", "ols_predictor"),
            "C / D",
            "bias and coverage of the construct's coefficient as predictor (D) or of means "
            "and differences (C)",
        ),
        "moment_subtraction": (("moment_subtraction",), "D", coefficient),
        "simex": (("simex",), "D", coefficient),
        "instrumented_coding": (
            ("oriv",),
            "D",
            "attenuation of the association from two machine measurements; the scale step uses "
            "the labelled rows",
        ),
    }
    for family, (variants, target, estimate_errors) in expected.items():
        spec = REGISTRY[family]
        assert (spec.variants, spec.target, spec.level) == (variants, target, "estimate")
        assert spec.errors_addressed == ()
        assert spec.estimate_errors == estimate_errors
        assert spec.factory is None
        with pytest.raises(SettingsError, match="estimate level"):
            get_correction(family, variants[0])
    families = list_corrections()["family"].unique().tolist()
    assert len(families) == 15
    assert set(families) == ELEVEN_FAMILIES | set(expected)


def test_a_score_level_function_has_no_correction_to_build() -> None:
    for family, variant in (("correction_ensemble", "mean"), ("blind_transport", "transport")):
        with pytest.raises(SettingsError, match="called as a function"):
            get_correction(family, variant)
