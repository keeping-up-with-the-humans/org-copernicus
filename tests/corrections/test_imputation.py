"""Multiple imputation (bootstrap-then-impute, predictive mean matching) and Rubin's rules."""

from __future__ import annotations

from types import MappingProxyType
from typing import Any

import numpy as np
import pytest
from corrections_support import make_correction_data, planted_data

from copernicus.corrections import CorrectionData, get_correction
from copernicus.corrections.families import MultipleImputation
from copernicus.corrections.families.imputation import PooledEstimate, rubin_pool
from copernicus.errors import ConfigurationError, EstimatorError, SettingsError

pytestmark = pytest.mark.filterwarnings("error")

NOISE_SD = 0.5


def _noisy_line_data(seed: int = 3) -> CorrectionData:
    """Gold is 1 + 2 * score plus N(0, 0.5^2) noise, with every text labelled."""
    noise = np.random.default_rng(seed).normal(scale=NOISE_SD, size=400)
    return planted_data(
        lambda score, covariate: 1.0 + 2.0 * score + noise, n_groups=40, group_size=10
    )


def _fit(variant: str, data: CorrectionData, seed: int = 0, **params: object) -> Any:
    correction = get_correction("multiple_imputation", variant, **params)
    return correction.fit(data, data.labelled_ids, rng=np.random.default_rng(seed))


def test_defaults_follow_the_registry_wording() -> None:
    correction = MultipleImputation("model_draws")
    assert (correction.m_draws, correction.k_donors) == (20, 5)
    assert (correction.family, correction.variant, correction.target) == (
        "multiple_imputation",
        "model_draws",
        "B",
    )
    assert correction.applies(make_correction_data()) == (True, "")


@pytest.mark.parametrize(
    ("variant", "params", "named"),
    [
        ("model_draws", {"m_draws": 0}, "m_draws"),
        ("model_draws", {"m_draws": -3}, "m_draws"),
        ("model_draws", {"m_draws": 2.5}, "m_draws"),
        ("model_draws", {"m_draws": 20.0}, "m_draws"),
        ("model_draws", {"m_draws": "20"}, "m_draws"),
        ("model_draws", {"m_draws": True}, "m_draws"),
        ("pmm", {"k_donors": 0}, "k_donors"),
        ("pmm", {"k_donors": 2.5}, "k_donors"),
        ("pmm", {"k_donors": "5"}, "k_donors"),
        ("pmm", {"k_donors": False}, "k_donors"),
        ("cubic", {}, "variant"),
    ],
)
def test_settings_outside_their_range_are_refused(
    variant: str, params: dict[str, object], named: str
) -> None:
    with pytest.raises(SettingsError, match=named):
        MultipleImputation(variant, **params)  # type: ignore[arg-type]


def test_the_pmm_variant_ships_with_its_note() -> None:
    from copernicus.corrections.families import imputation

    for text in (imputation.__doc__ or "", MultipleImputation.__doc__ or ""):
        assert "predictive mean matching" in text and "removed" in text


def test_integer_parameters_of_numpy_type_are_taken_as_integers() -> None:
    correction = MultipleImputation("pmm", m_draws=np.int64(4), k_donors=np.int32(2))  # type: ignore[arg-type]
    assert (correction.m_draws, correction.k_donors) == (4, 2)
    assert type(correction.m_draws) is int and type(correction.k_donors) is int


def test_model_draws_mean_recovers_the_planted_line() -> None:
    data = _noisy_line_data()
    fitted = _fit("model_draws", data, m_draws=60)
    predicted = fitted.predict(data, data.ids)
    expected = 1.0 + 2.0 * data.frame["score_good"].to_numpy()
    assert predicted.index.tolist() == data.ids.tolist()
    assert np.abs(predicted.to_numpy() - expected).mean() < 0.15
    assert abs(float(np.mean(predicted.to_numpy() - expected))) < 0.1
    assert fitted.provenance["between_variance"] > 0.0
    assert fitted.provenance["within_variance"] == pytest.approx(NOISE_SD**2, rel=0.25)


def test_draws_have_shape_m_by_n_and_predict_is_their_mean() -> None:
    data = _noisy_line_data()
    for variant in ("model_draws", "pmm"):
        fitted = _fit(variant, data, m_draws=7)
        ids = data.ids[:25]
        draws = fitted.draws(data, ids)
        assert draws.shape == (7, 25)
        np.testing.assert_allclose(fitted.predict(data, ids).to_numpy(), draws.mean(axis=0))


def test_draws_and_predictions_are_deterministic_for_a_fitted_object() -> None:
    data = _noisy_line_data()
    for variant in ("model_draws", "pmm"):
        fitted = _fit(variant, data, m_draws=5)
        np.testing.assert_array_equal(fitted.draws(data, data.ids), fitted.draws(data, data.ids))
        np.testing.assert_array_equal(
            fitted.predict(data, data.ids).to_numpy(), fitted.predict(data, data.ids).to_numpy()
        )
        other = _fit(variant, data, seed=1, m_draws=5)
        assert not np.array_equal(fitted.draws(data, data.ids), other.draws(data, data.ids))


def test_a_texts_draws_do_not_depend_on_the_other_ids_passed() -> None:
    data = _noisy_line_data()
    ids = data.ids
    for variant in ("model_draws", "pmm"):
        fitted = _fit(variant, data, m_draws=6)
        wide = fitted.draws(data, ids[:20])
        same = {"rtol": 0.0, "atol": 1e-12}
        np.testing.assert_allclose(fitted.draws(data, ids[:10]), wide[:, :10], **same)
        np.testing.assert_allclose(fitted.draws(data, ids[7:8]), wide[:, 7:8], **same)
        np.testing.assert_allclose(fitted.draws(data, ids[[5, 2, 5]]), wide[:, [5, 2, 5]], **same)
        np.testing.assert_allclose(
            fitted.predict(data, ids[:10]).to_numpy(),
            fitted.predict(data, ids[:20]).to_numpy()[:10],
            **same,
        )
        np.testing.assert_allclose(
            fitted.draws(data, ids[::-1]), fitted.draws(data, ids)[:, ::-1], **same
        )


def test_draws_of_different_texts_are_not_copies_of_one_stream() -> None:
    data = _noisy_line_data()
    fitted = _fit("model_draws", data, m_draws=8)
    noise = fitted.draws(data, data.ids[:30]) - (
        fitted.models.coefficients
        @ np.vstack([np.ones(30), data.frame["score_good"].to_numpy()[:30], data.covariate[:30]])
    )
    assert np.corrcoef(noise[0], noise[1])[0, 1] != pytest.approx(1.0)
    assert len(np.unique(np.round(noise, 10))) == noise.size


def test_model_draws_differ_across_draws() -> None:
    data = _noisy_line_data()
    draws = _fit("model_draws", data, m_draws=4).draws(data, data.ids)
    assert np.all(draws.std(axis=0) > 0.0)


def test_pmm_draws_take_only_observed_training_gold_values() -> None:
    data = _noisy_line_data()
    train_ids = data.labelled_ids[:200]
    fitted = get_correction("multiple_imputation", "pmm", m_draws=6, k_donors=3).fit(
        data, train_ids, rng=np.random.default_rng(2)
    )
    observed = set(data.gold[np.isin(data.ids, train_ids)].tolist())
    draws = fitted.draws(data, data.ids)
    assert set(np.unique(draws).tolist()) <= observed
    assert fitted.provenance["k_donors"] == 3
    assert fitted.provenance["variant"] == "pmm"


def test_pmm_with_fewer_training_rows_than_donors_still_draws_observed_values() -> None:
    data = _noisy_line_data()
    train_ids = data.labelled_ids[:5]
    fitted = get_correction("multiple_imputation", "pmm", m_draws=3, k_donors=9).fit(
        data, train_ids, rng=np.random.default_rng(0)
    )
    observed = set(data.gold[np.isin(data.ids, train_ids)].tolist())
    assert set(np.unique(fitted.draws(data, data.ids)).tolist()) <= observed


def test_provenance_reports_the_variances_the_column_and_the_inference_note() -> None:
    data = _noisy_line_data()
    fitted = _fit("model_draws", data, m_draws=12)
    provenance = fitted.provenance
    assert isinstance(provenance, MappingProxyType)
    assert provenance["family"] == "multiple_imputation"
    assert provenance["variant"] == "model_draws"
    assert provenance["score_column"] == "score_good"
    assert provenance["m_draws"] == 12
    assert provenance["within_variance"] > 0.0
    assert "150" in str(provenance["inference_note"])
    train_draws = fitted.draws(data, data.labelled_ids)
    assert provenance["between_variance"] == pytest.approx(
        float(np.mean(train_draws.var(axis=0, ddof=1)))
    )


def test_between_variance_of_a_single_draw_is_undefined() -> None:
    fitted = _fit("model_draws", _noisy_line_data(), m_draws=1)
    assert np.isnan(fitted.provenance["between_variance"])
    assert fitted.draws(_noisy_line_data(), np.array(["t0001"])).shape == (1, 1)


def test_fit_refuses_a_design_without_a_covariate_or_score_spread() -> None:
    data = _noisy_line_data()
    frame = data.frame.copy()
    frame["log_chars"] = 5.0
    with pytest.raises(EstimatorError, match="rank"):
        _fit("model_draws", CorrectionData(frame=frame, score_columns=data.score_columns))
    frame = data.frame.copy()
    frame["score_good"] = 2.0
    frame["score_poor"] = 3.0
    with pytest.raises(EstimatorError, match="no usable score column"):
        _fit("pmm", CorrectionData(frame=frame, score_columns=data.score_columns))


def test_draws_refuse_data_without_the_fitted_score_column() -> None:
    data = _noisy_line_data()
    fitted = _fit("model_draws", data)
    other = CorrectionData(
        frame=data.frame.drop(columns=["score_good"]), score_columns=("score_poor",)
    )
    with pytest.raises(ConfigurationError, match="score_good"):
        fitted.draws(other, other.ids[:3])


def test_fit_and_draws_refuse_unlabelled_and_unknown_ids() -> None:
    data = make_correction_data()
    correction = get_correction("multiple_imputation", "model_draws")
    with pytest.raises(ConfigurationError, match="no gold value"):
        correction.fit(data, data.ids, rng=np.random.default_rng(0))
    fitted = correction.fit(data, data.labelled_ids, rng=np.random.default_rng(0))
    with pytest.raises(ConfigurationError, match="not in the data"):
        fitted.draws(data, np.array(["missing"]))


def test_rubin_pool_of_equal_estimates_returns_them_with_the_within_variance() -> None:
    pooled = rubin_pool([2.5, 2.5, 2.5, 2.5], [0.04, 0.04, 0.04, 0.04])
    assert isinstance(pooled, PooledEstimate)
    assert pooled.estimate == 2.5
    assert pooled.variance == pytest.approx(0.04)
    assert pooled.between == 0.0
    assert pooled.within == pytest.approx(0.04)
    assert pooled.se == pytest.approx(0.2)
    assert pooled.df == np.inf
    assert pooled.m == 4


def test_rubin_pool_combines_within_and_between_variance() -> None:
    estimates = [1.0, 2.0, 3.0]
    variances = [0.5, 0.25, 0.75]
    pooled = rubin_pool(estimates, variances)
    within, between, m = 0.5, 1.0, 3
    total = within + (1 + 1 / m) * between
    assert pooled.estimate == pytest.approx(2.0)
    assert pooled.variance == pytest.approx(total)
    assert pooled.se == pytest.approx(np.sqrt(total))
    ratio = within / ((1 + 1 / m) * between)
    assert pooled.df == pytest.approx((m - 1) * (1 + ratio) ** 2)
    assert (pooled.within, pooled.between) == (pytest.approx(within), pytest.approx(between))


def test_rubin_pool_with_zero_within_variance_has_m_minus_one_degrees_of_freedom() -> None:
    assert rubin_pool([1.0, 2.0, 4.0], [0.0, 0.0, 0.0]).df == pytest.approx(2.0)


@pytest.mark.parametrize(
    ("estimates", "variances"),
    [
        ([1.0], [0.1]),
        ([], []),
        ([1.0, 2.0], [0.1]),
        ([1.0, np.nan], [0.1, 0.1]),
        ([1.0, 2.0], [0.1, -0.1]),
        ([1.0, 2.0], [0.1, np.inf]),
    ],
)
def test_rubin_pool_refuses_inconsistent_input(
    estimates: list[float], variances: list[float]
) -> None:
    with pytest.raises(ConfigurationError):
        rubin_pool(estimates, variances)
