"""Latent pooling over several score columns and the one-factor helpers behind it."""

from __future__ import annotations

from types import MappingProxyType

import numpy as np
import pandas as pd
import pytest
from scipy.stats import pearsonr

from copernicus.corrections import CorrectionData, get_correction
from copernicus.corrections.families import LatentPooling
from copernicus.corrections.families._factor import (
    paf_one_factor,
    regression_factor_score,
    sign_align,
    standardize_apply,
    standardize_fit,
    weighted_factor_score,
)
from copernicus.errors import ConfigurationError, EstimatorError, NotApplicableError

pytestmark = pytest.mark.filterwarnings("error")

N_TEXTS = 600
NOISE = (0.6, 0.8, 1.0)


def _copies(seed: int = 4, noise: tuple[float, ...] = NOISE) -> CorrectionData:
    """Noisy copies of one signal; gold is the signal plus a little noise."""
    rng = np.random.default_rng(seed)
    signal = rng.normal(size=N_TEXTS)
    columns = {
        f"copy_{index}": 2.0 * signal + 3.0 + rng.normal(scale=scale * 2.0, size=N_TEXTS)
        for index, scale in enumerate(noise)
    }
    frame = pd.DataFrame(
        {
            "text_id": [f"t{index:04d}" for index in range(N_TEXTS)],
            "group_id": [f"g{index // 6:03d}" for index in range(N_TEXTS)],
            **columns,
            "gold": signal + rng.normal(scale=0.2, size=N_TEXTS),
            "log_chars": rng.normal(size=N_TEXTS),
        }
    )
    return CorrectionData(frame=frame, score_columns=tuple(columns))


def _fit(data: CorrectionData):  # noqa: ANN202
    return get_correction("latent_pooling", "factor_anchor").fit(
        data, data.labelled_ids, rng=np.random.default_rng(0)
    )


def test_the_correction_has_one_variant_and_no_parameters() -> None:
    correction = LatentPooling()
    assert (correction.family, correction.variant, correction.target) == (
        "latent_pooling",
        "factor_anchor",
        "A",
    )
    assert correction.applies(_copies()) == (True, "")


def test_applies_needs_two_score_columns() -> None:
    frame = _copies().frame
    single = CorrectionData(frame=frame, score_columns=("copy_0",))
    applies, reason = LatentPooling().applies(single)
    assert not applies
    assert "two score columns" in reason
    assert "1" in reason


def test_fit_makes_the_column_check_of_applies() -> None:
    single = CorrectionData(frame=_copies().frame, score_columns=("copy_0",))
    _, reason = LatentPooling().applies(single)
    with pytest.raises(NotApplicableError) as refused:
        _fit(single)
    assert str(refused.value) == f"latent_pooling/factor_anchor does not apply: {reason}"


def test_the_pooled_score_correlates_more_with_gold_than_any_single_copy() -> None:
    data = _copies()
    fitted = _fit(data)
    pooled = fitted.predict(data, data.ids)
    gold = data.gold
    pooled_r = pearsonr(pooled.to_numpy(), gold).statistic
    for column in data.score_columns:
        single_r = pearsonr(data.frame[column].to_numpy(), gold).statistic
        assert pooled_r > single_r
    assert pooled.index.tolist() == data.ids.tolist()


def test_the_pooled_score_is_on_the_gold_scale_and_does_not_depend_on_the_column_signs() -> None:
    data = _copies()
    fitted = _fit(data)
    pooled = fitted.predict(data, data.ids).to_numpy()
    assert pooled.mean() == pytest.approx(data.gold.mean(), abs=1e-9)
    assert fitted.provenance["anchor_slope"] > 0.0
    reversed_frame = data.frame.copy()
    for column in data.score_columns:
        reversed_frame[column] = -reversed_frame[column]
    flipped = _fit(CorrectionData(frame=reversed_frame, score_columns=data.score_columns))
    assert flipped.provenance["anchor_slope"] < 0.0
    np.testing.assert_allclose(
        flipped.predict(CorrectionData(reversed_frame, data.score_columns), data.ids).to_numpy(),
        pooled,
        atol=1e-8,
    )


def test_provenance_records_loadings_anchor_and_indicators() -> None:
    provenance = _fit(_copies()).provenance
    assert isinstance(provenance, MappingProxyType)
    assert provenance["family"] == "latent_pooling"
    assert provenance["variant"] == "factor_anchor"
    assert provenance["score_column"] == "all"
    assert provenance["n_indicators"] == 3
    assert len(provenance["loadings"]) == 3
    assert {"anchor_slope", "anchor_intercept"} <= set(provenance)
    loadings = np.array(provenance["loadings"])
    assert loadings[0] > loadings[2] > 0.0


def test_a_constant_column_is_tolerated_and_adds_nothing() -> None:
    data = _copies()
    frame = data.frame.copy()
    frame["flat"] = 4.0
    with_flat = CorrectionData(frame=frame, score_columns=(*data.score_columns, "flat"))
    fitted = _fit(with_flat)
    assert fitted.provenance["n_indicators"] == 4
    assert fitted.provenance["loadings"][3] == pytest.approx(0.0, abs=1e-6)


def test_a_constant_factor_score_is_an_estimator_error() -> None:
    frame = _copies().frame.copy()
    for column in ("copy_0", "copy_1", "copy_2"):
        frame[column] = 1.0
    with pytest.raises(EstimatorError, match="constant"):
        _fit(CorrectionData(frame=frame, score_columns=("copy_0", "copy_1", "copy_2")))


def test_fit_uses_the_training_standardisation_for_new_rows() -> None:
    data = _copies()
    fitted = get_correction("latent_pooling", "factor_anchor").fit(
        data, data.labelled_ids[:300], rng=np.random.default_rng(0)
    )
    held_out = data.ids[300:]
    expected = fitted.predict(data, data.ids).to_numpy()[300:]
    np.testing.assert_allclose(fitted.predict(data, held_out).to_numpy(), expected)


def test_paf_recovers_the_loadings_of_a_planted_one_factor_matrix() -> None:
    planted = np.array([0.9, 0.7, 0.5, 0.8])
    correlation = np.outer(planted, planted)
    np.fill_diagonal(correlation, 1.0)
    loadings = paf_one_factor(correlation)
    np.testing.assert_allclose(np.abs(loadings), planted, atol=1e-4)


def test_paf_handles_a_singular_matrix() -> None:
    correlation = np.array([[1.0, 1.0, 0.5], [1.0, 1.0, 0.5], [0.5, 0.5, 1.0]])
    loadings = paf_one_factor(correlation)
    assert np.all(np.isfinite(loadings))
    assert loadings.shape == (3,)


def test_paf_refuses_a_matrix_that_is_not_square() -> None:
    with pytest.raises(ConfigurationError, match="square"):
        paf_one_factor(np.ones((2, 3)))


def test_sign_align_flips_only_a_negative_relation() -> None:
    reference = np.array([1.0, 2.0, 3.0, 4.0])
    np.testing.assert_array_equal(sign_align(-reference, reference), reference)
    np.testing.assert_array_equal(sign_align(reference, reference), reference)
    constant = np.full(4, 2.0)
    np.testing.assert_array_equal(sign_align(constant, reference), constant)
    np.testing.assert_array_equal(sign_align(reference, constant), reference)


def test_standardize_uses_population_sd_and_unit_sd_for_constants() -> None:
    matrix = np.column_stack([[1.0, 2.0, 3.0, 4.0], [5.0, 5.0, 5.0, 5.0]])
    mean, sd = standardize_fit(matrix)
    np.testing.assert_allclose(mean, [2.5, 5.0])
    np.testing.assert_allclose(sd, [np.std([1.0, 2.0, 3.0, 4.0]), 1.0])
    standardised = standardize_apply(matrix, mean, sd)
    assert standardised[:, 0].std() == pytest.approx(1.0)
    assert np.all(standardised[:, 1] == 0.0)


def test_weighted_and_regression_factor_scores() -> None:
    standardised = np.array([[1.0, -1.0], [0.0, 2.0], [-1.0, 0.5]])
    loadings = np.array([0.8, 0.4])
    np.testing.assert_allclose(
        weighted_factor_score(standardised, loadings), standardised @ loadings
    )
    correlation = np.array([[1.0, 0.32], [0.32, 1.0]])
    weights = np.linalg.solve(correlation + 1e-3 * np.eye(2), loadings)
    np.testing.assert_allclose(
        regression_factor_score(standardised, correlation, loadings), standardised @ weights
    )


def test_fit_refuses_unlabelled_unknown_and_repeated_training_ids() -> None:
    data = _copies()
    frame = data.frame.copy()
    frame.loc[frame.index[:10], "gold"] = np.nan
    partial = CorrectionData(frame=frame, score_columns=data.score_columns)
    correction = get_correction("latent_pooling", "factor_anchor")
    rng = np.random.default_rng(0)
    with pytest.raises(ConfigurationError, match="no gold value"):
        correction.fit(partial, partial.ids, rng=rng)
    with pytest.raises(ConfigurationError, match="not in the data"):
        correction.fit(partial, np.array(["nope", *partial.labelled_ids]), rng=rng)
    with pytest.raises(ConfigurationError, match="repeat"):
        correction.fit(
            partial, np.concatenate([partial.labelled_ids, partial.labelled_ids[:2]]), rng=rng
        )


def test_predict_names_a_pooled_column_the_data_lack() -> None:
    data = _copies()
    fitted = _fit(data)
    other = CorrectionData(
        frame=data.frame.drop(columns=["copy_1"]), score_columns=("copy_0", "copy_2")
    )
    with pytest.raises(ConfigurationError, match="copy_1"):
        fitted.predict(other, other.ids[:5])
