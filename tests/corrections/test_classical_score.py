"""Classical score corrections: Fuller shrinkage and Tweedie smoothing."""

from __future__ import annotations

from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from copernicus.corrections import CorrectionData, get_correction
from copernicus.corrections.families import ClassicalScore
from copernicus.errors import EstimatorError, SettingsError

pytestmark = pytest.mark.filterwarnings("error")

N_TEXTS = 500


def _gaussian_data(seed: int = 6) -> CorrectionData:
    """Gold is standard normal; the score is a shifted, scaled, noisy copy."""
    rng = np.random.default_rng(seed)
    gold = rng.normal(size=N_TEXTS)
    frame = pd.DataFrame(
        {
            "text_id": [f"t{index:04d}" for index in range(N_TEXTS)],
            "group_id": [f"g{index // 5:03d}" for index in range(N_TEXTS)],
            "score_good": 2.0 + 1.5 * gold + rng.normal(scale=0.8, size=N_TEXTS),
            "score_poor": rng.normal(size=N_TEXTS),
            "gold": gold,
            "log_chars": rng.normal(size=N_TEXTS),
        }
    )
    return CorrectionData(frame=frame, score_columns=("score_good", "score_poor"))


def _fit(variant: str, data: CorrectionData) -> Any:
    return get_correction("classical", variant).fit(
        data, data.labelled_ids, rng=np.random.default_rng(0)
    )


def test_the_correction_names_its_family_target_and_variants() -> None:
    for variant in ("fuller_shrinkage", "tweedie"):
        correction = ClassicalScore(variant)
        assert (correction.family, correction.variant, correction.target) == (
            "classical",
            variant,
            "A",
        )
        assert correction.applies(_gaussian_data()) == (True, "")


def test_an_unknown_variant_is_refused() -> None:
    with pytest.raises(SettingsError, match="fuller_shrinkage"):
        ClassicalScore("blup")


def test_fuller_kappa_is_the_population_covariance_over_the_population_variance() -> None:
    data = _gaussian_data()
    fitted = _fit("fuller_shrinkage", data)
    score = data.frame["score_good"].to_numpy()
    gold = data.gold
    kappa = float(np.mean((gold - gold.mean()) * (score - score.mean())) / np.var(score))
    assert fitted.provenance["kappa"] == pytest.approx(kappa, rel=1e-12)
    assert fitted.provenance["score_column"] == "score_good"
    predicted = fitted.predict(data, data.ids)
    np.testing.assert_allclose(
        predicted.to_numpy(), gold.mean() + kappa * (score - score.mean()), atol=1e-12
    )
    assert predicted.index.tolist() == data.ids.tolist()


def test_fuller_kappa_is_not_the_sample_covariance_ratio() -> None:
    data = _gaussian_data()
    kappa = _fit("fuller_shrinkage", data).provenance["kappa"]
    score = data.frame["score_good"].to_numpy()
    mixed = float(np.cov(data.gold, score)[0, 1] / np.var(score))
    assert kappa != pytest.approx(mixed, rel=1e-6)
    assert kappa == pytest.approx(mixed * (N_TEXTS - 1) / N_TEXTS, rel=1e-12)


def test_fuller_prediction_matches_the_gold_mean_on_the_training_rows() -> None:
    data = _gaussian_data()
    predicted = _fit("fuller_shrinkage", data).predict(data, data.ids)
    assert predicted.mean() == pytest.approx(data.gold.mean(), abs=1e-12)


def test_fuller_provenance_is_read_only() -> None:
    provenance = _fit("fuller_shrinkage", _gaussian_data()).provenance
    assert isinstance(provenance, MappingProxyType)
    assert provenance["family"] == "classical"
    assert provenance["variant"] == "fuller_shrinkage"


def test_tweedie_reduces_to_the_linear_line_on_gaussian_data() -> None:
    data = _gaussian_data()
    fitted = _fit("tweedie", data)
    tweedie = fitted.predict(data, data.ids).to_numpy()
    line = _fit("fuller_shrinkage", data).predict(data, data.ids).to_numpy()
    assert spearmanr(tweedie, line).statistic > 0.99
    assert np.abs(tweedie - line).mean() < 0.15
    assert np.all(np.diff(tweedie[np.argsort(data.frame["score_good"].to_numpy())]) >= 0.0)


def test_tweedie_provenance_records_the_error_variance_and_bandwidth() -> None:
    fitted = _fit("tweedie", _gaussian_data())
    provenance = fitted.provenance
    assert provenance["family"] == "classical"
    assert provenance["variant"] == "tweedie"
    assert provenance["score_column"] == "score_good"
    assert provenance["sigma2_delta"] >= 0.0
    assert provenance["kde_bandwidth"] > 0.0
    assert provenance["kde_bandwidth"] == pytest.approx(
        1.06 * np.std(_proxy(_gaussian_data())) * N_TEXTS ** (-1 / 5),
        rel=1e-9,
    )


def _proxy(data: CorrectionData) -> np.ndarray:
    score = data.frame["score_good"].to_numpy()
    gold = data.gold
    beta = np.mean((score - score.mean()) * (gold - gold.mean())) / np.var(gold)
    return gold.mean() + (score - score.mean()) / beta


def test_tweedie_refuses_a_score_with_zero_covariance_with_gold() -> None:
    x = np.tile([-2.0, -1.0, 0.0, 1.0, 2.0], 8)
    frame = pd.DataFrame(
        {
            "text_id": [f"t{index}" for index in range(x.size)],
            "group_id": [f"g{index // 4}" for index in range(x.size)],
            "score": x,
            "gold": x**2,
            "log_chars": np.arange(float(x.size)),
        }
    )
    data = CorrectionData(frame=frame, score_columns=("score",))
    with pytest.raises(EstimatorError, match="covariance"):
        _fit("tweedie", data)
    assert _fit("fuller_shrinkage", data).provenance["kappa"] == 0.0


def test_tweedie_refuses_a_negatively_associated_score_and_says_to_reverse_it() -> None:
    rng = np.random.default_rng(2)
    gold = rng.normal(size=N_TEXTS)
    frame = pd.DataFrame(
        {
            "text_id": [f"t{index:04d}" for index in range(N_TEXTS)],
            "group_id": [f"g{index // 5:03d}" for index in range(N_TEXTS)],
            "score": -gold + rng.normal(scale=0.5, size=N_TEXTS),
            "gold": gold,
            "log_chars": rng.normal(size=N_TEXTS),
        }
    )
    reversed_data = CorrectionData(frame=frame, score_columns=("score",))
    with pytest.raises(EstimatorError, match=r"negative covariance.*reverse"):
        _fit("tweedie", reversed_data)
    kappa = _fit("fuller_shrinkage", reversed_data).provenance["kappa"]
    assert kappa < 0.0
    flipped = CorrectionData(frame=frame.assign(score=-frame["score"]), score_columns=("score",))
    predicted = _fit("tweedie", flipped).predict(flipped, flipped.ids)
    assert predicted.std() > 0.5


@pytest.mark.parametrize("variant", ["fuller_shrinkage", "tweedie"])
def test_a_constant_score_is_an_estimator_error(variant: str) -> None:
    frame = _gaussian_data().frame.copy()
    frame["score_good"] = 2.0
    frame["score_poor"] = 3.0
    with pytest.raises(EstimatorError, match="no usable score column"):
        _fit(variant, CorrectionData(frame=frame, score_columns=("score_good", "score_poor")))
