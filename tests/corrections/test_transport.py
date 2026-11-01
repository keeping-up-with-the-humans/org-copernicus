"""Blind transport: the column and the line are chosen and fitted on other corpora only."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from copernicus.corrections import REGISTRY, CorrectionData, list_corrections
from copernicus.corrections.families.transport import TransportResult, blind_transport
from copernicus.errors import ConfigurationError, EstimatorError

pytestmark = pytest.mark.filterwarnings("error")


def _corpus(
    seed: int,
    n: int = 120,
    *,
    columns: tuple[str, ...] = ("alpha", "beta"),
    gold_scale: float = 1.0,
    gold_shift: float = 0.0,
    with_gold: bool = True,
    prefix: str = "t",
) -> CorrectionData:
    """A corpus whose ``alpha`` column tracks the latent value and ``beta`` is noise."""
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=n)
    values = {
        "alpha": 3.0 * latent + rng.normal(scale=0.6, size=n) + 10.0 * seed,
        "beta": rng.normal(size=n),
        "gamma": rng.normal(size=n) - latent * 0.1,
    }
    gold = gold_scale * latent + gold_shift + rng.normal(scale=0.2, size=n)
    frame = pd.DataFrame(
        {
            "text_id": [f"{prefix}{seed}_{index:03d}" for index in range(n)],
            "group_id": [f"g{seed}_{index // 4:03d}" for index in range(n)],
            **{name: values[name] for name in columns},
            "gold": gold if with_gold else np.nan,
            "log_chars": rng.normal(size=n),
        }
    )
    return CorrectionData(frame=frame, score_columns=columns)


def _z(values: np.ndarray) -> np.ndarray:
    return (values - values.mean()) / values.std()


def test_the_line_is_fitted_on_the_pooled_sources_and_applied_to_the_target() -> None:
    sources = [
        _corpus(1, gold_scale=1.0),
        _corpus(2, gold_scale=5.0, gold_shift=40.0),
        _corpus(3, gold_scale=0.2),
    ]
    target = _corpus(9, with_gold=False)
    result = blind_transport(sources, target)
    assert isinstance(result, TransportResult)
    assert result.score_column == "alpha"
    assert result.n_sources == 3
    assert result.n_shared_columns == 2
    pooled_x = np.concatenate([_z(source.frame["alpha"].to_numpy()) for source in sources])
    pooled_y = np.concatenate([_z(source.gold) for source in sources])
    slope, intercept = np.polyfit(pooled_x, pooled_y, 1)
    assert result.slope == pytest.approx(slope, abs=1e-10)
    assert result.intercept == pytest.approx(intercept, abs=1e-10)
    assert result.selection_rho == pytest.approx(float(spearmanr(pooled_x, pooled_y).statistic))
    expected = intercept + slope * _z(target.frame["alpha"].to_numpy())
    np.testing.assert_allclose(result.column.to_numpy(), expected, atol=1e-10)
    assert result.column.index.tolist() == target.ids.tolist()
    assert result.slope > 0.5


def test_the_target_gold_is_never_read() -> None:
    sources = [_corpus(1), _corpus(2)]
    blind = blind_transport(sources, _corpus(9, with_gold=False))
    with_gold = blind_transport(sources, _corpus(9, with_gold=True, gold_scale=-7.0))
    pd.testing.assert_series_equal(blind.column, with_gold.column)
    assert (blind.score_column, blind.slope) == (with_gold.score_column, with_gold.slope)


def test_only_the_shared_columns_are_candidates() -> None:
    sources = [
        _corpus(1, columns=("alpha", "beta", "gamma")),
        _corpus(2, columns=("alpha", "beta")),
    ]
    target = _corpus(9, columns=("gamma", "beta", "alpha"), with_gold=False)
    result = blind_transport(sources, target)
    assert result.n_shared_columns == 2
    only_target = _corpus(9, columns=("gamma",), with_gold=False)
    with pytest.raises(EstimatorError, match="share no score column"):
        blind_transport(sources, only_target)


def test_a_column_constant_in_a_source_is_not_a_candidate() -> None:
    first, second = _corpus(1), _corpus(2)
    frame = second.frame.assign(alpha=4.0)
    second = CorrectionData(frame=frame, score_columns=("alpha", "beta"))
    result = blind_transport([first, second], _corpus(9, with_gold=False))
    assert result.score_column == "beta"
    assert result.n_shared_columns == 2


def test_no_usable_column_is_refused() -> None:
    first, second = _corpus(1), _corpus(2)
    flat = CorrectionData(
        frame=second.frame.assign(alpha=4.0, beta=1.0), score_columns=("alpha", "beta")
    )
    with pytest.raises(EstimatorError, match="usable"):
        blind_transport([first, flat], _corpus(9, with_gold=False))


def test_a_constant_target_column_is_refused() -> None:
    target = _corpus(9, with_gold=False)
    flat = CorrectionData(frame=target.frame.assign(alpha=2.0), score_columns=target.score_columns)
    with pytest.raises(EstimatorError, match="alpha"):
        blind_transport([_corpus(1), _corpus(2)], flat)


def test_a_source_without_usable_gold_is_refused() -> None:
    target = _corpus(9, with_gold=False)
    with pytest.raises(ConfigurationError, match="gold"):
        blind_transport([_corpus(1), _corpus(2, with_gold=False)], target)
    constant = _corpus(2)
    flat_gold = CorrectionData(
        frame=constant.frame.assign(gold=1.5), score_columns=constant.score_columns
    )
    with pytest.raises(ConfigurationError, match="constant"):
        blind_transport([_corpus(1), flat_gold], target)


def test_a_source_list_must_not_be_empty() -> None:
    with pytest.raises(ConfigurationError, match="source"):
        blind_transport([], _corpus(9, with_gold=False))


def test_only_labelled_rows_of_a_source_enter_the_pooled_line() -> None:
    source = _corpus(1)
    partial_frame = source.frame.copy()
    partial_frame.loc[partial_frame.index[::2], "gold"] = np.nan
    partial = CorrectionData(frame=partial_frame, score_columns=source.score_columns)
    target = _corpus(9, with_gold=False)
    result = blind_transport([partial, _corpus(2)], target)
    mask = partial.labelled_mask
    first_x = _z(partial.frame["alpha"].to_numpy()[mask])
    first_y = _z(partial.gold[mask])
    other = _corpus(2)
    pooled_x = np.concatenate([first_x, _z(other.frame["alpha"].to_numpy())])
    pooled_y = np.concatenate([first_y, _z(other.gold)])
    slope, intercept = np.polyfit(pooled_x, pooled_y, 1)
    assert result.slope == pytest.approx(slope, abs=1e-10)
    assert result.intercept == pytest.approx(intercept, abs=1e-10)


def test_the_family_is_registered_as_a_score_level_function() -> None:
    spec = REGISTRY["blind_transport"]
    assert (spec.variants, spec.target, spec.level, spec.estimate_errors, spec.factory) == (
        ("transport",),
        "A",
        "score",
        "none",
        None,
    )
    assert spec.errors_addressed == ("level_displacement", "calibration_shortfall")
    assert "blind_transport" in set(list_corrections()["family"])
