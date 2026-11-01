"""Embedding-augmented calibration: what it learns, how it chooses its penalty, provenance."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from types import MappingProxyType

import numpy as np
import pytest
from corrections_support import embedding_data, make_embedding_set
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

from copernicus.baseline.folds import rng_for
from copernicus.baseline.ridge import ALPHAS, apply_scaler, fit_scaler
from copernicus.corrections import (
    CorrectionData,
    best_score_column,
    cross_fit_predictions,
    get_correction,
    inner_k,
    inner_splits,
)
from copernicus.corrections.families import _embedding_search as search
from copernicus.corrections.families._ridge_path import Deletion, PenaltyChoice
from copernicus.corrections.families.embedding_augmented import EmbeddingFitted
from copernicus.errors import EstimatorError

pytestmark = pytest.mark.filterwarnings("error")

VARIANTS = ("embeddings_only", "embeddings_plus_score", "score_anchored_residual")
GROUP_SIZE = 5
SEED_BOUND = 2**31 - 1


def _split(n_groups: int, train_groups: int) -> tuple[np.ndarray, np.ndarray]:
    """Rows of the first ``train_groups`` groups and rows of the others."""
    rows = np.arange(n_groups * GROUP_SIZE)
    return rows[: train_groups * GROUP_SIZE], rows[train_groups * GROUP_SIZE :]


TRAIN, HELD = _split(60, 45)


def _signal_world(seed: int = 0) -> CorrectionData:
    """Two columns of ``alpha`` and one of ``beta`` carry the latent value; gold is that value."""
    rng = np.random.default_rng(seed)
    n = 60 * GROUP_SIZE
    latent = rng.normal(size=n)
    alpha = rng.normal(size=(n, 6))
    alpha[:, :2] = latent[:, None] + rng.normal(scale=0.4, size=(n, 2))
    beta = rng.normal(size=(n, 4))
    beta[:, 0] = latent + rng.normal(scale=0.8, size=n)
    gold = latent + rng.normal(scale=0.3, size=n)
    score = 0.5 * latent + rng.normal(scale=0.6, size=n) + 3.0
    return embedding_data(gold, score, {"alpha": alpha, "beta": beta}, seed=seed + 1)


def _fit(
    variant: str, data: CorrectionData, train: np.ndarray, encoder: str = "alpha", seed: int = 0
) -> EmbeddingFitted:
    correction = get_correction("embedding_augmented", variant, encoder=encoder)
    return correction.fit(data, data.ids[train], rng=np.random.default_rng(seed))


def test_embeddings_only_ranks_held_out_texts_well_when_the_embeddings_carry_the_signal() -> None:
    data = _signal_world()
    predicted = _fit("embeddings_only", data, TRAIN).predict(data, data.ids[HELD])
    assert float(spearmanr(predicted.to_numpy(), data.gold[HELD]).statistic) > 0.8


def test_the_anchored_residual_stays_on_the_calibration_line_when_the_embeddings_are_noise() -> (
    None
):
    rng = np.random.default_rng(1)
    n = 100 * GROUP_SIZE
    latent = rng.normal(size=n)
    gold = latent + rng.normal(scale=0.3, size=n)
    score = latent + rng.normal(scale=0.5, size=n) + 3.0
    data = embedding_data(gold, score, {"alpha": rng.normal(size=(n, 4))}, seed=2)
    train, held = _split(100, 80)
    anchored = _fit("score_anchored_residual", data, train).predict(data, data.ids[held])
    line = get_correction("regression_calibration", "linear").fit(
        data, data.ids[train], rng=np.random.default_rng(0)
    )
    calibrated = line.predict(data, data.ids[held])
    assert np.abs(anchored.to_numpy() - calibrated.to_numpy()).mean() < 0.05


def test_embeddings_plus_score_beats_embeddings_only_when_the_score_adds_signal() -> None:
    rng = np.random.default_rng(2)
    n = 60 * GROUP_SIZE
    from_text, from_score = rng.normal(size=n), rng.normal(size=n)
    alpha = rng.normal(size=(n, 6))
    alpha[:, 0] = from_text + rng.normal(scale=0.3, size=n)
    gold = from_text + from_score + rng.normal(scale=0.3, size=n)
    score = from_score + rng.normal(scale=0.3, size=n) + 3.0
    data = embedding_data(gold, score, {"alpha": alpha}, seed=3)
    error = {
        variant: np.abs(
            _fit(variant, data, TRAIN).predict(data, data.ids[HELD]).to_numpy() - gold[HELD]
        ).mean()
        for variant in ("embeddings_only", "embeddings_plus_score")
    }
    assert error["embeddings_plus_score"] <= error["embeddings_only"]
    assert error["embeddings_plus_score"] < 0.6 * error["embeddings_only"]


def _five_group_world(seed: int = 11) -> CorrectionData:
    """Six groups of nine texts with two score columns of equal quality; the last group is spare."""
    rng = np.random.default_rng(seed)
    n = 54
    latent = rng.normal(size=6)[np.repeat(np.arange(6), 9)] + rng.normal(scale=0.6, size=n)
    alpha = np.column_stack(
        [latent + rng.normal(scale=0.6, size=n) for _ in range(2)] + [rng.normal(size=(n, 3))]
    )
    beta = np.column_stack([latent + rng.normal(scale=0.9, size=n), rng.normal(size=(n, 2))])
    return embedding_data(
        latent + rng.normal(scale=0.6, size=n),
        latent + rng.normal(scale=0.6, size=n),
        {"alpha": alpha, "beta": beta},
        group_size=9,
        seed=seed + 1,
        second_score=latent + rng.normal(scale=0.6, size=n),
    )


def _explicit_alpha(
    data: CorrectionData,
    train: np.ndarray,
    variant: str,
    names: list[str],
    deletions: list[tuple[np.ndarray, np.ndarray]] | None = None,
) -> tuple[float, set[str]]:
    """The penalty of one explicit refit of the whole pipeline per deletion and penalty.

    ``deletions`` are ``(retained, held)`` positions into the training rows; by default one
    per training group. The appended score is standardised with the mean and population SD
    of all training rows.
    """
    assert data.embeddings is not None
    blocks = [data.embeddings[name].vectors[train].astype(np.float64) for name in names]
    design = apply_scaler(blocks, fit_scaler(blocks))
    gold, groups = data.gold[train], data.groups[train]
    scores = data.scores.iloc[train].reset_index(drop=True)
    if deletions is None:
        deletions = [
            (np.flatnonzero(groups != group), np.flatnonzero(groups == group))
            for group in np.unique(groups)
        ]
    total = np.zeros(ALPHAS.size)
    columns: set[str] = set()
    for kept, held in deletions:
        column = best_score_column(scores.iloc[kept].reset_index(drop=True), gold[kept])
        columns.add(column)
        score = scores[column].to_numpy()
        features, target, offset = design, gold, np.zeros(held.size)
        if variant == "embeddings_plus_score":
            features = np.column_stack([design, (score - score.mean()) / score.std()])
        elif variant == "score_anchored_residual":
            slope, intercept = np.polyfit(score[kept], gold[kept], 1)
            target = gold - (intercept + slope * score)
            offset = intercept + slope * score[held]
        for index, alpha in enumerate(ALPHAS):
            model = Ridge(alpha=float(alpha), fit_intercept=True, solver="svd")
            model.fit(features[kept], target[kept])
            total[index] += np.sum((gold[held] - offset - model.predict(features[held])) ** 2)
    return float(ALPHAS[int(np.argmin(total))]), columns


@pytest.mark.parametrize(
    ("variant", "encoder"),
    [
        ("embeddings_only", "alpha"),
        ("embeddings_plus_score", "alpha"),
        ("score_anchored_residual", "alpha"),
        ("embeddings_only", "stacked"),
        ("score_anchored_residual", "stacked"),
    ],
)
def test_the_penalty_equals_an_explicit_refit_of_the_pipeline_in_every_deletion(
    variant: str, encoder: str
) -> None:
    """Five training groups: the min(5, groups) deletions leave out one group each."""
    data = _five_group_world()
    train = np.arange(45)
    names = ["alpha"] if encoder == "alpha" else ["alpha", "beta"]
    expected, columns = _explicit_alpha(data, train, variant, names)
    assert columns == {"score_good", "score_poor"}, "the deletions must select different columns"
    fitted = _fit(variant, data, train, encoder=encoder)
    assert fitted.provenance["penalty_alpha"] == expected
    assert fitted.provenance["k_inner_folds"] == 5


@pytest.mark.parametrize("variant", VARIANTS)
def test_the_deletions_follow_the_recorded_seed_under_the_family_key(
    variant: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fourteen training groups: five deletions of two or three groups each."""
    seen: list[list[tuple[np.ndarray, np.ndarray]]] = []
    real = search.choose_penalty

    def spy(
        deletions: Sequence[Deletion],
        error_fn: Callable[[Deletion], np.ndarray],
        alphas: np.ndarray,
        policy: str,
        n_groups: int,
    ) -> PenaltyChoice:
        seen.append([(deletion.retained, deletion.held) for deletion in deletions])
        return real(deletions, error_fn, alphas, policy, n_groups)

    monkeypatch.setattr(search, "choose_penalty", spy)
    data = _signal_world()
    train = np.arange(14 * GROUP_SIZE)
    groups = data.groups[train]
    fitted = _fit(variant, data, train)
    seed = fitted.provenance["inner_seed"]
    expected = inner_splits(groups, inner_k(14), rng_for(seed, "embedding_augmented", "deletions"))
    assert len(seen) == 1 and len(seen[0]) == len(expected) == 5
    for (retained, held), (expected_retained, expected_held) in zip(seen[0], expected, strict=True):
        np.testing.assert_array_equal(retained, expected_retained)
        np.testing.assert_array_equal(held, expected_held)
    assert sorted(int(np.unique(groups[held]).size) for _, held in seen[0]) == [2, 3, 3, 3, 3]
    oracle, _ = _explicit_alpha(data, train, variant, ["alpha"], deletions=expected)
    assert fitted.provenance["penalty_alpha"] == oracle


@pytest.mark.parametrize("factor", [0.1, 40.0])
@pytest.mark.parametrize("variant", VARIANTS)
def test_rescaling_the_scores_changes_neither_the_penalty_nor_the_predictions(
    variant: str, factor: float
) -> None:
    data = _signal_world()
    frame = data.frame.copy()
    frame[["score_good", "score_poor"]] *= factor
    rescaled = CorrectionData(
        frame=frame, score_columns=data.score_columns, embeddings=data.embeddings
    )
    original, scaled = _fit(variant, data, TRAIN), _fit(variant, rescaled, TRAIN)
    assert scaled.provenance["penalty_alpha"] == original.provenance["penalty_alpha"]
    np.testing.assert_allclose(
        scaled.predict(rescaled, rescaled.ids).to_numpy(),
        original.predict(data, data.ids).to_numpy(),
        rtol=0.0,
        atol=1e-10,
    )
    if variant == "embeddings_plus_score":
        assert scaled.provenance["score_sd"] == pytest.approx(
            factor * original.provenance["score_sd"], rel=1e-12
        )
        assert scaled.provenance["score_coefficient"] == pytest.approx(
            original.provenance["score_coefficient"], rel=1e-9
        )


def test_each_deletion_selects_its_score_column_on_its_retained_rows_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sizes: list[int] = []
    real = search.best_score_column

    def spy(scores: object, gold: np.ndarray) -> str:
        sizes.append(int(np.asarray(gold).size))
        return real(scores, gold)  # type: ignore[arg-type]

    monkeypatch.setattr(search, "best_score_column", spy)
    data = _five_group_world()
    _fit("score_anchored_residual", data, np.arange(45))
    assert sorted(sizes) == [36, 36, 36, 36, 36, 45]
    sizes.clear()
    _fit("embeddings_only", data, np.arange(45))
    assert sizes == [], "embeddings_only reads no score column"


@pytest.mark.parametrize(
    ("variant", "policy", "n_features", "column"),
    [
        ("embeddings_only", "grouped_deletions", 6, "none"),
        ("embeddings_plus_score", "grouped_deletions_reselected", 7, "score_good"),
        ("score_anchored_residual", "grouped_deletions_reselected", 6, "score_good"),
    ],
)
def test_provenance_names_the_policy_the_grid_edges_and_the_encoder(
    variant: str, policy: str, n_features: int, column: str
) -> None:
    provenance = _fit(variant, _signal_world(), TRAIN).provenance
    assert isinstance(provenance, MappingProxyType)
    assert (provenance["family"], provenance["variant"]) == ("embedding_augmented", variant)
    assert (provenance["encoder"], provenance["encoders"]) == ("alpha", ["alpha"])
    assert provenance["penalty_selection"] == policy
    assert (provenance["k_inner_folds"], provenance["n_inner_groups"]) == (5, 45)
    assert provenance["penalty_alpha"] in ALPHAS.tolist()
    assert provenance["penalty_at_grid_edge"] == (
        provenance["penalty_at_lower_edge"] or provenance["penalty_at_upper_edge"]
    )
    for key in ("penalty_deletion_share_lower", "penalty_deletion_share_upper"):
        assert 0.0 <= provenance[key] <= 1.0
        assert provenance[key] * 5 == round(provenance[key] * 5)
    assert (provenance["n_features"], provenance["score_column"]) == (n_features, column)
    assert provenance["inner_seed"] == int(np.random.default_rng(0).integers(SEED_BOUND))


@pytest.mark.parametrize("variant", VARIANTS)
def test_a_texts_prediction_does_not_depend_on_the_other_ids_passed(variant: str) -> None:
    data = _signal_world()
    fitted = _fit(variant, data, TRAIN, encoder="stacked")
    full = fitted.predict(data, data.ids).to_numpy()
    for picked in ([5, 250, 17], [299], list(range(299, -1, -1))):
        part = fitted.predict(data, data.ids[picked])
        assert part.index.tolist() == data.ids[picked].tolist()
        np.testing.assert_allclose(part.to_numpy(), full[picked], rtol=0.0, atol=1e-12)


def test_the_fit_is_repeatable_for_a_generator_in_the_same_state() -> None:
    data = _signal_world()
    first = _fit("embeddings_plus_score", data, TRAIN, seed=4)
    second = _fit("embeddings_plus_score", data, TRAIN, seed=4)
    np.testing.assert_array_equal(
        first.predict(data, data.ids).to_numpy(), second.predict(data, data.ids).to_numpy()
    )
    assert first.provenance["inner_seed"] == int(np.random.default_rng(4).integers(SEED_BOUND))


def test_stacked_concatenates_every_set_in_mapping_order_with_equal_total_variance() -> None:
    fitted = _fit("embeddings_only", _signal_world(), TRAIN, encoder="stacked")
    assert fitted.encoders == ("alpha", "beta")
    assert fitted.scaler.weights == pytest.approx((np.sqrt(5.0 / 6.0), np.sqrt(5.0 / 4.0)))
    assert fitted.provenance["n_features"] == 10
    assert fitted.provenance["encoder"] == "stacked"
    assert fitted.provenance["encoders"] == ["alpha", "beta"]


@pytest.mark.parametrize("variant", VARIANTS)
def test_cross_fitting_corrects_every_text_of_partly_labelled_data(variant: str) -> None:
    data = _signal_world()
    frame = data.frame.copy()
    frame.loc[frame.index % 3 == 0, "gold"] = np.nan
    partial = CorrectionData(
        frame=frame, score_columns=data.score_columns, embeddings=data.embeddings
    )
    corrected = cross_fit_predictions(get_correction("embedding_augmented", variant), partial)
    assert corrected.index.tolist() == partial.ids.tolist()
    assert np.isfinite(corrected.to_numpy()).all()
    calibrated = cross_fit_predictions(get_correction("regression_calibration", "linear"), partial)
    gain = (
        spearmanr(corrected.to_numpy(), data.gold).statistic
        - spearmanr(calibrated.to_numpy(), data.gold).statistic
    )
    assert gain > 0.1, "the embeddings carry signal the score lacks"


def test_embeddings_only_needs_no_usable_score_column() -> None:
    data = _signal_world()
    frame = data.frame.copy()
    frame[["score_good", "score_poor"]] = 1.0
    flat = CorrectionData(frame=frame, score_columns=data.score_columns, embeddings=data.embeddings)
    predicted = _fit("embeddings_only", flat, TRAIN).predict(flat, flat.ids[HELD])
    assert np.isfinite(predicted.to_numpy()).all()
    with pytest.raises(EstimatorError, match="no usable score column"):
        _fit("embeddings_plus_score", flat, TRAIN)


def test_ids_are_matched_as_strings_and_the_vectors_follow_the_data_order() -> None:
    data = _signal_world()
    assert data.embeddings is not None
    frame = data.frame.copy()
    frame["text_id"] = np.arange(data.n)
    vectors = data.embeddings["alpha"].vectors.astype(np.float64)
    order = np.random.default_rng(9).permutation(data.n)
    extra = np.random.default_rng(10).normal(size=(5, vectors.shape[1]))
    shuffled = make_embedding_set(
        np.concatenate([order, np.arange(data.n, data.n + 5)]),
        np.vstack([vectors[order], extra]),
        "alpha",
    )
    renamed = CorrectionData(
        frame=frame, score_columns=data.score_columns, embeddings={"alpha": shuffled}
    )
    for variant in VARIANTS:
        expected = _fit(variant, data, TRAIN).predict(data, data.ids)
        got = _fit(variant, renamed, TRAIN).predict(renamed, renamed.ids)
        assert got.index.tolist() == [str(index) for index in range(data.n)]
        np.testing.assert_allclose(got.to_numpy(), expected.to_numpy(), rtol=0.0, atol=1e-10)
