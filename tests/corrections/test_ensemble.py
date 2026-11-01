"""The correction ensemble: four combiners over standardised member columns."""

from __future__ import annotations

import time
from types import MappingProxyType

import numpy as np
import pandas as pd
import pytest

from copernicus.corrections import REGISTRY, list_corrections
from copernicus.corrections.families._factor import (
    paf_one_factor,
    regression_factor_score,
    relation_sign,
    standardize_apply,
    standardize_fit,
)
from copernicus.corrections.families.ensemble import CorrectionEnsemble, EnsembleResult
from copernicus.errors import ConfigurationError, EstimatorError, NotApplicableError, SettingsError

pytestmark = pytest.mark.filterwarnings("error")

N_ROWS = 300
IDS = np.array([f"t{index:03d}" for index in range(N_ROWS)])


def _members(seed: int = 1) -> dict[str, pd.Series]:
    """Three noisy copies of one signal on different scales."""
    rng = np.random.default_rng(seed)
    signal = rng.normal(size=N_ROWS)
    return {
        "first": pd.Series(2.0 * signal + rng.normal(scale=0.5, size=N_ROWS) + 4.0, index=IDS),
        "second": pd.Series(0.3 * signal + rng.normal(scale=0.1, size=N_ROWS) - 1.0, index=IDS),
        "third": pd.Series(5.0 * signal + rng.normal(scale=2.0, size=N_ROWS), index=IDS),
    }


def _standardised(members: dict[str, pd.Series]) -> np.ndarray:
    matrix = np.column_stack([series.to_numpy() for series in members.values()])
    return (matrix - matrix.mean(axis=0)) / matrix.std(axis=0)


def test_mean_is_the_average_of_the_standardised_members() -> None:
    members = _members()
    two = {name: members[name] for name in ("first", "third")}
    result = CorrectionEnsemble("mean").combine(two, IDS)
    assert isinstance(result, EnsembleResult)
    expected = _standardised(two).mean(axis=1)
    np.testing.assert_allclose(result.column.to_numpy(), expected, atol=1e-12)
    assert result.column.index.tolist() == IDS.tolist()
    provenance = result.provenance
    assert provenance["variant"] == "mean"
    assert provenance["n_members"] == 2
    assert tuple(provenance["member_families"]) == ("first", "third")  # type: ignore[arg-type]
    assert tuple(provenance["dropped_members"]) == ()  # type: ignore[arg-type]


def test_members_are_matched_to_the_ids_by_label_not_by_position() -> None:
    members = _members()
    shuffled = {name: series.sample(frac=1.0, random_state=3) for name, series in members.items()}
    ensemble = CorrectionEnsemble("mean")
    pd.testing.assert_series_equal(
        ensemble.combine(members, IDS).column, ensemble.combine(shuffled, IDS).column
    )


def test_a_member_missing_an_id_is_refused_naming_it() -> None:
    members = _members()
    members["second"] = members["second"].drop("t007")
    with pytest.raises(ConfigurationError, match=r"second.*t007"):
        CorrectionEnsemble("mean").combine(members, IDS)


def test_a_member_missing_many_ids_is_refused_naming_the_first_ten() -> None:
    members = _members()
    members["third"] = members["third"].iloc[:250]
    with pytest.raises(ConfigurationError) as refusal:
        CorrectionEnsemble("mean").combine(members, IDS)
    message = str(refusal.value)
    assert "'third'" in message and "'t250'" in message and "'t259'" in message
    assert "'t260'" not in message and "(40 more)" in message


@pytest.mark.slow
def test_members_over_twenty_thousand_ids_combine_in_linear_time() -> None:
    n_ids = 20_000
    ids = np.array([f"text{index:06d}" for index in range(n_ids)])
    rng = np.random.default_rng(4)
    signal = rng.normal(size=n_ids)
    members = {
        "first": pd.Series(signal + rng.normal(size=n_ids), index=ids),
        "second": pd.Series(2.0 * signal + rng.normal(size=n_ids), index=ids),
    }
    shuffled = {name: series.sample(frac=1.0, random_state=5) for name, series in members.items()}
    started = time.perf_counter()
    result = CorrectionEnsemble("mean").combine(shuffled, ids)
    elapsed = time.perf_counter() - started
    assert elapsed < 2.0, f"combining two members over {n_ids} ids took {elapsed:.1f} s"
    expected = _standardised(members).mean(axis=1)
    np.testing.assert_allclose(result.column.to_numpy(), expected, atol=1e-12)
    assert result.column.index.tolist() == ids.tolist()


def test_constant_and_incomplete_members_are_dropped_and_named() -> None:
    members = _members()
    members["flat"] = pd.Series(np.full(N_ROWS, 2.5), index=IDS)
    broken = members["third"].copy()
    broken.iloc[5] = np.nan
    members["third"] = broken
    result = CorrectionEnsemble("pc1").combine(members, IDS)
    assert tuple(result.provenance["dropped_members"]) == ("third", "flat")  # type: ignore[arg-type]
    assert tuple(result.provenance["member_families"]) == ("first", "second")  # type: ignore[arg-type]
    assert result.provenance["n_members"] == 2


def test_fewer_than_two_usable_members_is_refused() -> None:
    members = _members()
    only_one = {"first": members["first"], "flat": pd.Series(np.zeros(N_ROWS), index=IDS)}
    with pytest.raises(EstimatorError, match=r"two.*flat"):
        CorrectionEnsemble("mean").combine(only_one, IDS)
    with pytest.raises(EstimatorError, match="two"):
        CorrectionEnsemble("factor").combine({"first": members["first"]}, IDS)


def test_mode_rounds_to_the_classes_and_takes_the_row_mode() -> None:
    ids = np.array(["a", "b", "c", "d"])
    members = {
        "m1": pd.Series([0.1, 0.9, 0.4, 2.2], index=ids),
        "m2": pd.Series([0.2, 1.2, 1.1, 1.9], index=ids),
        "m3": pd.Series([0.9, 0.8, 1.0, 0.4], index=ids),
    }
    classes = np.array([0.0, 1.0, 2.0])
    result = CorrectionEnsemble("mode").combine(members, ids, gold_classes=classes)
    # a: 0,0,1 -> 0   b: 1,1,1 -> 1   c: 0,1,1 -> 1   d: 2,2,0 -> 2
    assert result.column.to_numpy().tolist() == [0.0, 1.0, 1.0, 2.0]
    assert result.provenance["variant"] == "mode"


def test_mode_breaks_a_tie_towards_the_smaller_class() -> None:
    ids = np.array(["a", "b"])
    members = {
        "m1": pd.Series([0.1, 0.0], index=ids),
        "m2": pd.Series([0.9, 1.0], index=ids),
    }
    result = CorrectionEnsemble("mode").combine(members, ids, gold_classes=np.array([0.0, 1.0]))
    assert result.column.to_numpy().tolist() == [0.0, 0.0]


def test_mode_needs_gold_classes() -> None:
    with pytest.raises(NotApplicableError, match="classes"):
        CorrectionEnsemble("mode").combine(_members(), IDS)


def test_mode_refuses_malformed_gold_classes() -> None:
    mode = CorrectionEnsemble("mode")
    with pytest.raises(ConfigurationError, match=r"finite.*nan"):
        mode.combine(_members(), IDS, gold_classes=np.array([0.0, 1.0, 2.0, np.nan]))
    with pytest.raises(ConfigurationError, match=r"finite.*inf"):
        mode.combine(_members(), IDS, gold_classes=np.array([0.0, np.inf]))
    with pytest.raises(ConfigurationError, match="non-empty"):
        mode.combine(_members(), IDS, gold_classes=np.array([]))
    with pytest.raises(ConfigurationError, match="one-dimensional"):
        mode.combine(_members(), IDS, gold_classes=np.array([[0.0, 1.0], [2.0, 3.0]]))


def test_pc1_of_two_members_is_their_scaled_sum() -> None:
    members = _members()
    two = {name: members[name] for name in ("first", "second")}
    result = CorrectionEnsemble("pc1").combine(two, IDS)
    standardised = _standardised(two)
    np.testing.assert_allclose(
        result.column.to_numpy(), standardised.sum(axis=1) / np.sqrt(2.0), atol=1e-9
    )
    direction = np.asarray(list(result.provenance["direction"].values()))  # type: ignore[attr-defined]
    np.testing.assert_allclose(np.abs(direction), np.full(2, np.sqrt(0.5)), atol=1e-9)


@pytest.mark.parametrize("variant", ["pc1", "factor"])
@pytest.mark.parametrize("seed", range(6))
def test_pc1_and_factor_are_signed_to_agree_with_the_member_mean(variant: str, seed: int) -> None:
    members = _members(seed)
    result = CorrectionEnsemble(variant).combine(members, IDS)
    reference = _standardised(members).mean(axis=1)
    assert np.corrcoef(result.column.to_numpy(), reference)[0, 1] > 0.9
    negated = {name: -series for name, series in members.items()}
    flipped = CorrectionEnsemble(variant).combine(negated, IDS)
    assert np.corrcoef(flipped.column.to_numpy(), -reference)[0, 1] > 0.9


def test_factor_is_the_regression_factor_score_of_the_one_factor_solution() -> None:
    members = _members()
    result = CorrectionEnsemble("factor").combine(members, IDS)
    matrix = np.column_stack([series.to_numpy() for series in members.values()])
    mean, sd = standardize_fit(matrix)
    standardised = standardize_apply(matrix, mean, sd)
    correlation = np.corrcoef(standardised, rowvar=False)
    loadings = paf_one_factor(correlation)
    score = regression_factor_score(standardised, correlation, loadings)
    expected = score if np.corrcoef(score, standardised.mean(axis=1))[0, 1] > 0 else -score
    np.testing.assert_allclose(result.column.to_numpy(), expected, atol=1e-9)
    stored = np.asarray(list(result.provenance["loadings"].values()))  # type: ignore[union-attr]
    np.testing.assert_allclose(np.abs(stored), np.abs(loadings), atol=1e-9)


def test_an_unknown_variant_is_refused() -> None:
    with pytest.raises(SettingsError, match="variant"):
        CorrectionEnsemble("median")


def test_repeated_ids_are_refused() -> None:
    members = _members()
    with pytest.raises(ConfigurationError, match="repeat"):
        CorrectionEnsemble("mean").combine(members, np.concatenate([IDS[:-1], IDS[:1]]))


def test_the_family_is_registered_as_a_rank_only_score_combiner() -> None:
    spec = REGISTRY["correction_ensemble"]
    assert spec.variants == ("mean", "mode", "factor", "pc1")
    assert (spec.target, spec.level, spec.estimate_errors, spec.factory) == (
        "A",
        "score",
        "none",
        None,
    )
    assert spec.errors_addressed == ("noise", "calibration_shortfall")
    assert (CorrectionEnsemble.__doc__ or "").lower().count("rank") >= 1
    assert "correction_ensemble" in set(list_corrections()["family"])


def test_nested_provenance_mappings_are_read_only() -> None:
    for variant, key in (("factor", "loadings"), ("pc1", "direction")):
        nested = CorrectionEnsemble(variant).combine(_members(), IDS).provenance[key]
        assert isinstance(nested, MappingProxyType)
        with pytest.raises(TypeError):
            nested["first"] = 0.0  # type: ignore[index]


def test_relation_sign_is_the_sign_that_orients_towards_the_reference() -> None:
    reference = np.arange(10.0)
    assert relation_sign(reference, reference) == 1.0
    assert relation_sign(-reference, reference) == -1.0
    assert relation_sign(np.ones(10), reference) == 1.0
    assert relation_sign(reference, np.ones(10)) == 1.0
