"""Embedding-augmented calibration: encoders, text ids, applicability and refusals."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pytest
from corrections_support import embedding_data, make_embedding_set

from copernicus.corrections import CorrectionData, cross_fit_predictions, get_correction
from copernicus.corrections.families import EmbeddingAugmented
from copernicus.corrections.families.embedding_augmented import EmbeddingFitted
from copernicus.embeddings.sets import EmbeddingSet
from copernicus.errors import ConfigurationError, NotApplicableError, SettingsError

pytestmark = pytest.mark.filterwarnings("error")

VARIANTS = ("embeddings_only", "embeddings_plus_score", "score_anchored_residual")
N_TEXTS = 80
TRAIN = np.arange(60)
UNLABELLED_FROM = 70


def _world(seed: int = 5) -> CorrectionData:
    """Twenty groups of four texts with a three-column and a two-column embedding set."""
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=N_TEXTS)
    alpha = rng.normal(size=(N_TEXTS, 3))
    alpha[:, 0] = latent + rng.normal(scale=0.5, size=N_TEXTS)
    beta = rng.normal(size=(N_TEXTS, 2))
    gold = latent + rng.normal(scale=0.3, size=N_TEXTS)
    score = latent + rng.normal(scale=0.5, size=N_TEXTS)
    return embedding_data(gold, score, {"alpha": alpha, "beta": beta}, group_size=4, seed=seed + 1)


def _with_sets(data: CorrectionData, sets: Mapping[str, EmbeddingSet] | None) -> CorrectionData:
    return CorrectionData(frame=data.frame, score_columns=data.score_columns, embeddings=sets)


def _fit(
    data: CorrectionData, variant: str = "embeddings_plus_score", encoder: str = "alpha"
) -> EmbeddingFitted:
    return EmbeddingAugmented(variant, encoder).fit(
        data, data.ids[TRAIN], rng=np.random.default_rng(0)
    )


def test_get_correction_builds_the_variant_for_the_named_encoder() -> None:
    correction = get_correction("embedding_augmented", "score_anchored_residual", encoder="beta")
    assert isinstance(correction, EmbeddingAugmented)
    assert (correction.family, correction.variant, correction.target) == (
        "embedding_augmented",
        "score_anchored_residual",
        "A",
    )
    assert correction.encoder == "beta"
    assert EmbeddingAugmented("embeddings_only").encoder == "stacked"


@pytest.mark.parametrize(
    ("variant", "encoder", "match"),
    [
        ("cubic", "alpha", "unknown variant 'cubic'.*embeddings_only"),
        ("embeddings_only", "", "encoder"),
        ("embeddings_only", "   ", "encoder"),
    ],
)
def test_the_constructor_refuses_an_unknown_variant_and_an_empty_encoder(
    variant: str, encoder: str, match: str
) -> None:
    with pytest.raises(SettingsError, match=match):
        EmbeddingAugmented(variant, encoder)


def test_applies_needs_the_embeddings_the_encoder_names_and_two_labelled_groups() -> None:
    data = _world()
    assert data.embeddings is not None
    for encoder in ("alpha", "beta", "stacked"):
        assert EmbeddingAugmented("embeddings_only", encoder).applies(data) == (True, "")
    applies, reason = EmbeddingAugmented("embeddings_only", "alpha").applies(_with_sets(data, None))
    assert not applies and "embeddings" in reason
    applies, reason = EmbeddingAugmented("embeddings_only", "gamma").applies(data)
    assert not applies and "'gamma'" in reason and "'alpha'" in reason and "'beta'" in reason
    single = _with_sets(data, {"alpha": data.embeddings["alpha"]})
    applies, reason = EmbeddingAugmented("embeddings_only", "stacked").applies(single)
    assert not applies and "two embedding sets" in reason and "1" in reason
    frame = data.frame.copy()
    frame.loc[frame["group_id"] != "g000", "gold"] = np.nan
    one_group = CorrectionData(
        frame=frame, score_columns=data.score_columns, embeddings=data.embeddings
    )
    applies, reason = EmbeddingAugmented("embeddings_only", "alpha").applies(one_group)
    assert not applies and "two groups" in reason


def test_fit_refuses_data_without_the_embeddings_it_needs_as_applies_does() -> None:
    data = _world()
    assert data.embeddings is not None
    cases = [
        (_with_sets(data, None), "alpha"),
        (data, "gamma"),
        (_with_sets(data, {"beta": data.embeddings["beta"]}), "stacked"),
    ]
    for lacking, encoder in cases:
        _, reason = EmbeddingAugmented("embeddings_plus_score", encoder).applies(lacking)
        with pytest.raises(NotApplicableError) as refused:
            _fit(lacking, encoder=encoder)
        refusal = f"embedding_augmented/embeddings_plus_score does not apply: {reason}"
        assert str(refused.value) == refusal


def test_fit_on_training_rows_of_one_group_carries_the_frame_of_every_refusal() -> None:
    data = _world()
    with pytest.raises(NotApplicableError) as refused:
        EmbeddingAugmented("embeddings_only", "alpha").fit(
            data, data.ids[data.groups == "g003"], rng=np.random.default_rng(0)
        )
    message = str(refused.value)
    assert message.startswith("embedding_augmented/embeddings_only does not apply: ")
    assert "two groups among the training rows; they hold 1" in message


def test_fit_refuses_training_rows_in_one_group_and_bad_training_ids() -> None:
    data = _world()
    rng = np.random.default_rng(0)
    with pytest.raises(NotApplicableError, match="two groups among the training rows"):
        EmbeddingAugmented("embeddings_only", "alpha").fit(
            data, data.ids[data.groups == "g003"], rng=rng
        )
    correction = EmbeddingAugmented("embeddings_only", "alpha")
    with pytest.raises(ConfigurationError, match="not in the data"):
        correction.fit(data, np.array(["nope", *data.ids[TRAIN]]), rng=rng)
    with pytest.raises(ConfigurationError, match="repeat"):
        correction.fit(data, np.concatenate([data.ids[TRAIN], data.ids[:2]]), rng=rng)


def test_a_set_that_misses_texts_is_refused_with_the_first_ten_missing_ids() -> None:
    data = _world()
    assert data.embeddings is not None
    missing = data.ids[[3, 10, 17, 22, 31, 40, 47, 55, 61, 66, 72, 79]]
    kept = [text_id for text_id in data.ids if text_id not in set(missing)]
    partial = _with_sets(
        data, {"alpha": data.embeddings["alpha"].subset(kept), "beta": data.embeddings["beta"]}
    )
    for encoder in ("alpha", "stacked"):
        with pytest.raises(ConfigurationError) as caught:
            _fit(partial, encoder=encoder)
        message = str(caught.value)
        assert "'alpha'" in message
        for text_id in missing[:10]:
            assert repr(str(text_id)) in message
        for text_id in missing[10:]:
            assert repr(str(text_id)) not in message
        assert "(2 more)" in message


def test_predict_names_the_ids_sets_columns_and_dimensions_the_prediction_data_lack() -> None:
    data = _world()
    assert data.embeddings is not None
    fitted = _fit(data, encoder="stacked")
    absent = data.ids[[70, 75]]
    kept = [text_id for text_id in data.ids if text_id not in set(absent)]
    short = _with_sets(
        data, {"alpha": data.embeddings["alpha"].subset(kept), "beta": data.embeddings["beta"]}
    )
    with pytest.raises(ConfigurationError, match=f"'alpha'.*'{absent[0]}', '{absent[1]}'"):
        fitted.predict(short, data.ids[60:])
    assert fitted.predict(short, data.ids[:60]).index.tolist() == data.ids[:60].tolist()
    with pytest.raises(ConfigurationError, match="'beta'"):
        fitted.predict(_with_sets(data, {"alpha": data.embeddings["alpha"]}), data.ids[:5])
    with pytest.raises(ConfigurationError, match="embeddings"):
        fitted.predict(_with_sets(data, None), data.ids[:5])
    wider = make_embedding_set(data.ids, np.ones((N_TEXTS, 5)), "beta")
    with pytest.raises(ConfigurationError, match=r"'beta'.*5.*2"):
        fitted.predict(
            _with_sets(data, {"alpha": data.embeddings["alpha"], "beta": wider}), data.ids[:5]
        )
    no_column = CorrectionData(
        frame=data.frame.drop(columns=["score_good"]),
        score_columns=("score_poor",),
        embeddings=data.embeddings,
    )
    with pytest.raises(ConfigurationError, match="score_good"):
        fitted.predict(no_column, data.ids[:5])
    with pytest.raises(ConfigurationError, match="not in the data"):
        fitted.predict(data, np.array(["missing"]))


def _with_non_finite_vectors(data: CorrectionData) -> CorrectionData:
    """The last ten texts unlabelled; two of them get a NaN and an infinite ``alpha`` component."""
    assert data.embeddings is not None
    frame = data.frame.copy()
    frame.loc[frame.index >= UNLABELLED_FROM, "gold"] = np.nan
    vectors = data.embeddings["alpha"].vectors.astype(np.float64)
    vectors[75, 1] = np.nan
    vectors[77, 0] = np.inf
    sets = {
        "alpha": make_embedding_set(data.ids, vectors, "alpha"),
        "beta": data.embeddings["beta"],
    }
    return CorrectionData(frame=frame, score_columns=data.score_columns, embeddings=sets)


@pytest.mark.parametrize("encoder", ["alpha", "stacked"])
@pytest.mark.parametrize("variant", VARIANTS)
def test_a_non_finite_vector_of_an_unlabelled_text_is_refused_by_fit_and_predict(
    variant: str, encoder: str
) -> None:
    data = _world()
    broken = _with_non_finite_vectors(data)
    correction = EmbeddingAugmented(variant, encoder)
    with pytest.raises(ConfigurationError, match=r"'alpha'.*non-finite.*'t0075', 't0077'"):
        correction.fit(broken, broken.labelled_ids, rng=np.random.default_rng(0))
    fitted = correction.fit(data, data.ids[TRAIN], rng=np.random.default_rng(0))
    with pytest.raises(ConfigurationError, match=r"'alpha'.*non-finite.*'t0075', 't0077'"):
        fitted.predict(broken, broken.ids[UNLABELLED_FROM:])
    finite = fitted.predict(broken, broken.ids[:UNLABELLED_FROM]).to_numpy()
    assert np.isfinite(finite).all()


@pytest.mark.parametrize("variant", VARIANTS)
def test_predicting_no_text_gives_an_empty_series(variant: str) -> None:
    data = _world()
    empty = _fit(data, variant=variant, encoder="stacked").predict(data, np.array([], dtype=str))
    assert empty.empty and empty.dtype == float and empty.index.tolist() == []


def test_cross_fitting_stops_at_its_first_fit_on_a_non_finite_vector() -> None:
    broken = _with_non_finite_vectors(_world())
    with pytest.raises(ConfigurationError, match="'t0075'"):
        cross_fit_predictions(EmbeddingAugmented("embeddings_only", "alpha"), broken)


def test_a_fitted_embeddings_only_model_reads_no_score_column() -> None:
    data = _world()
    fitted = _fit(data, variant="embeddings_only")
    no_scores = CorrectionData(
        frame=data.frame.drop(columns=["score_good"]),
        score_columns=("score_poor",),
        embeddings=data.embeddings,
    )
    np.testing.assert_allclose(
        fitted.predict(no_scores, data.ids).to_numpy(),
        fitted.predict(data, data.ids).to_numpy(),
        rtol=0.0,
        atol=1e-12,
    )
