"""CorrectionData.from_scores, best_score_column and the train-internal scaling helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from corrections_support import N_ROWS, make_embedding_set, make_frame

from copernicus.corrections.data import (
    CorrectionData,
    best_score_column,
    minmax_apply,
    minmax_fit,
    scale_test,
)
from copernicus.data import TextTable
from copernicus.errors import ConfigurationError, EstimatorError


def toy_scores(table: TextTable) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    truth = table.frame["gold_constructiveness"].to_numpy(float)
    return pd.DataFrame(
        {
            "text_id": table.frame["text_id"],
            "construct_score": truth + rng.normal(scale=0.3, size=len(truth)),
            "cues": truth + rng.normal(scale=0.5, size=len(truth)),
            "balance": rng.normal(size=len(truth)),
            "tone": truth + rng.normal(scale=0.8, size=len(truth)),
            "n_runs": 3,
        }
    )


def test_best_score_column_picks_planted_best_and_skips_constant() -> None:
    frame = make_frame()
    frame["flat"] = 3.0
    gold = frame["gold"].to_numpy()
    columns = frame[["flat", "poor", "good"]]
    assert best_score_column(columns, gold) == "good"
    assert best_score_column(columns[["flat", "poor"]], gold) == "poor"


def test_best_score_column_is_signed() -> None:
    frame = make_frame()
    reversed_score = -frame["good"]
    columns = pd.DataFrame({"reversed": reversed_score, "poor": frame["poor"]})
    gold = np.linspace(1.0, 5.0, N_ROWS)
    assert best_score_column(columns, gold) == "poor"


def test_best_score_column_uses_finite_rows_only() -> None:
    gold = np.array([1.0, 2.0, 3.0, 4.0, np.nan, np.nan])
    columns = pd.DataFrame(
        {"a": [1.0, 2.0, 3.0, 4.0, 9.0, -9.0], "b": [1.0, 3.0, 2.0, 4.0, 0.0, 0.0]}
    )
    assert best_score_column(columns, gold) == "a"


def test_best_score_column_raises_when_nothing_is_usable() -> None:
    gold = np.array([1.0, 2.0, 3.0, 4.0])
    with pytest.raises(EstimatorError, match="no usable score column"):
        best_score_column(pd.DataFrame({"flat": [2.0] * 4}), gold)
    with pytest.raises(EstimatorError, match="no usable score column"):
        best_score_column(
            pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]}), np.array([1.0, 2.0, np.nan, np.nan])
        )
    with pytest.raises(EstimatorError, match="no usable score column"):
        best_score_column(pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]}), np.full(4, 2.0))


def test_best_score_column_refuses_length_mismatch() -> None:
    with pytest.raises(ConfigurationError):
        best_score_column(pd.DataFrame({"a": [1.0, 2.0, 3.0]}), np.array([1.0, 2.0]))


def test_minmax_fit_ignores_non_finite_values() -> None:
    assert minmax_fit(np.array([3.0, np.nan, -1.0, 7.0, np.inf])) == (-1.0, 7.0)
    with pytest.raises(EstimatorError):
        minmax_fit(np.array([np.nan, np.nan]))


def test_minmax_apply_scales_and_refuses_zero_range() -> None:
    scaled = minmax_apply(np.array([0.0, 5.0, 10.0, 15.0]), 0.0, 10.0)
    assert scaled.tolist() == [0.0, 0.5, 1.0, 1.5]
    with pytest.raises(EstimatorError, match="range"):
        minmax_apply(np.array([1.0, 2.0]), 4.0, 4.0)


def test_minmax_apply_refuses_a_reversed_range() -> None:
    with pytest.raises(EstimatorError, match="range"):
        minmax_apply(np.array([1.0, 2.0]), 10.0, 0.0)


@pytest.mark.parametrize("name", ["text_id", "group_id", "pi"])
def test_from_scores_refuses_reserved_names(toy_corpus: TextTable, name: str) -> None:
    scores = toy_scores(toy_corpus).assign(group_id="x", pi=0.5)
    gold = toy_corpus.frame[["text_id", "gold_constructiveness"]]
    with pytest.raises(ConfigurationError, match=name):
        CorrectionData.from_scores(
            scores, toy_corpus, gold, gold_column="gold_constructiveness", score_columns=[name]
        )
    with pytest.raises(ConfigurationError, match=name):
        CorrectionData.from_scores(
            scores,
            toy_corpus,
            gold.rename(columns={"gold_constructiveness": name}),
            gold_column=name,
            score_columns=["cues"],
        )
    with pytest.raises(ConfigurationError, match=name):
        CorrectionData.from_scores(
            scores,
            toy_corpus,
            gold,
            gold_column="gold_constructiveness",
            score_columns=["cues"],
            covariate=name,
        )


def test_from_scores_refuses_a_repeated_score_column(toy_corpus: TextTable) -> None:
    with pytest.raises(ConfigurationError, match="cues"):
        CorrectionData.from_scores(
            toy_scores(toy_corpus),
            toy_corpus,
            toy_corpus.frame[["text_id", "gold_constructiveness"]],
            gold_column="gold_constructiveness",
            score_columns=["cues", "cues"],
        )


def test_scale_test_uses_train_range_for_both_columns() -> None:
    train_pred = np.array([0.0, 10.0, 5.0])
    train_gold = np.array([1.0, 5.0, 3.0])
    test_pred = np.array([5.0, 20.0, -10.0])
    test_gold = np.array([3.0, 9.0, 0.0])
    pred, gold = scale_test(train_pred, train_gold, test_pred, test_gold)
    assert pred.tolist() == [0.5, 2.0, -1.0]
    assert gold.tolist() == [0.5, 2.0, -0.25]


def test_scale_test_refuses_a_constant_training_gold() -> None:
    with pytest.raises(EstimatorError, match="range"):
        scale_test(np.array([0.0, 1.0]), np.array([2.0, 2.0]), np.array([0.5]), np.array([2.0]))
    with pytest.raises(EstimatorError, match="range"):
        scale_test(np.array([3.0, 3.0]), np.array([1.0, 2.0]), np.array([0.5]), np.array([2.0]))


def test_from_scores_on_the_toy_corpus(toy_corpus: TextTable) -> None:
    scores = toy_scores(toy_corpus)
    subset = toy_corpus.frame.iloc[:20][["text_id", "gold_constructiveness"]]
    data = CorrectionData.from_scores(
        scores,
        toy_corpus,
        subset,
        gold_column="gold_constructiveness",
        score_columns=("construct_score", "cues", "balance", "tone"),
    )
    assert data.n == 54
    assert data.gold_column == "gold_constructiveness"
    assert data.covariate_column == "log_chars"
    assert data.n_labelled == 20
    assert data.labelled_ids.tolist() == subset["text_id"].tolist()
    np.testing.assert_allclose(data.covariate, np.log(toy_corpus.frame["n_chars"].to_numpy(float)))
    assert data.groups.tolist() == toy_corpus.frame["group_id"].tolist()
    assert data.score_columns == ("construct_score", "cues", "balance", "tone")


def test_from_scores_treats_nan_gold_and_absent_rows_as_unlabelled(toy_corpus: TextTable) -> None:
    scores = toy_scores(toy_corpus)
    gold = toy_corpus.frame[["text_id", "gold_constructiveness"]].copy()
    gold.loc[0:9, "gold_constructiveness"] = np.nan
    gold = gold.iloc[:30]
    data = CorrectionData.from_scores(
        scores, toy_corpus, gold, gold_column="gold_constructiveness", score_columns=("cues",)
    )
    assert data.n_labelled == 20


def test_from_scores_matches_ids_as_strings_and_keeps_the_score_order(
    toy_corpus: TextTable,
) -> None:
    scores = toy_scores(toy_corpus).iloc[::-1].reset_index(drop=True)
    data = CorrectionData.from_scores(
        scores,
        toy_corpus,
        toy_corpus.frame[["text_id", "gold_constructiveness"]],
        gold_column="gold_constructiveness",
        score_columns=("cues",),
    )
    assert data.ids.tolist()[0] == "t054"
    assert data.n_labelled == 54


def test_from_scores_supports_another_covariate_from_the_table(toy_corpus: TextTable) -> None:
    data = CorrectionData.from_scores(
        toy_scores(toy_corpus),
        toy_corpus,
        toy_corpus.frame[["text_id", "gold_constructiveness"]],
        gold_column="gold_constructiveness",
        score_columns=("cues",),
        covariate="n_words",
    )
    assert data.covariate_column == "n_words"
    np.testing.assert_array_equal(data.covariate, toy_corpus.frame["n_words"].to_numpy(float))


def test_from_scores_refuses_scored_texts_missing_from_the_table(toy_corpus: TextTable) -> None:
    scores = toy_scores(toy_corpus)
    scores.loc[0, "text_id"] = "unknown"
    with pytest.raises(ConfigurationError, match="unknown"):
        CorrectionData.from_scores(
            scores,
            toy_corpus,
            toy_corpus.frame[["text_id", "gold_constructiveness"]],
            gold_column="gold_constructiveness",
            score_columns=("cues",),
        )


def test_from_scores_refuses_unknown_columns(toy_corpus: TextTable) -> None:
    scores = toy_scores(toy_corpus)
    gold = toy_corpus.frame[["text_id", "gold_constructiveness"]]
    with pytest.raises(ConfigurationError, match="absent_score"):
        CorrectionData.from_scores(
            scores,
            toy_corpus,
            gold,
            gold_column="gold_constructiveness",
            score_columns=("absent_score",),
        )
    with pytest.raises(ConfigurationError, match="absent_gold"):
        CorrectionData.from_scores(
            scores, toy_corpus, gold, gold_column="absent_gold", score_columns=("cues",)
        )
    with pytest.raises(ConfigurationError, match="nonexistent"):
        CorrectionData.from_scores(
            scores,
            toy_corpus,
            gold,
            gold_column="gold_constructiveness",
            score_columns=("cues",),
            covariate="nonexistent",
        )


def test_from_scores_refuses_duplicate_gold_ids(toy_corpus: TextTable) -> None:
    gold = toy_corpus.frame[["text_id", "gold_constructiveness"]]
    gold = pd.concat([gold, gold.iloc[:1]])
    with pytest.raises(ConfigurationError, match="t001"):
        CorrectionData.from_scores(
            toy_scores(toy_corpus),
            toy_corpus,
            gold,
            gold_column="gold_constructiveness",
            score_columns=("cues",),
        )


def toy_gold(toy_corpus: TextTable) -> pd.DataFrame:
    """Gold on the first 20 toy texts, with their inclusion probability and a rating."""
    gold = toy_corpus.frame.iloc[:20][["text_id", "gold_constructiveness"]].copy()
    return gold.assign(inclusion=0.4, rating=np.arange(20.0))


def test_from_scores_carries_embeddings_pi_and_extra_columns(toy_corpus: TextTable) -> None:
    ids = toy_corpus.frame["text_id"].tolist()
    vectors = make_embedding_set(ids, np.random.default_rng(1).normal(size=(54, 4)), "enc")
    data = CorrectionData.from_scores(
        toy_scores(toy_corpus),
        toy_corpus,
        toy_gold(toy_corpus),
        gold_column="gold_constructiveness",
        score_columns=("cues",),
        embeddings={"enc": vectors},
        pi_column="inclusion",
        extra_columns=("n_words", "rating"),
    )
    assert data.embeddings is not None and data.embeddings["enc"] is vectors
    assert data.pi is not None
    np.testing.assert_array_equal(data.pi[:20], np.full(20, 0.4))
    assert np.isnan(data.pi[20:]).all()
    np.testing.assert_array_equal(data.frame["n_words"], toy_corpus.frame["n_words"])
    np.testing.assert_array_equal(data.frame["rating"].to_numpy()[:20], np.arange(20.0))
    assert data.frame["rating"].iloc[20:].isna().all()


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"pi_column": "absent"}, "'absent'"),
        ({"pi_column": "gold_constructiveness"}, "pi_column"),
        ({"pi_column": "rating"}, "(0, 1]"),
        ({"extra_columns": ("absent",)}, "'absent'"),
        ({"extra_columns": ("cues",)}, "'cues'"),
        ({"extra_columns": ("group_id",)}, "'group_id'"),
        ({"extra_columns": ("n_words", "n_words")}, "'n_words'"),
        ({"extra_columns": "n_words"}, "sequence of column names"),
    ],
)
def test_from_scores_refuses_unusable_pi_and_extra_columns(
    toy_corpus: TextTable, options: dict[str, object], message: str
) -> None:
    with pytest.raises(ConfigurationError) as raised:
        CorrectionData.from_scores(
            toy_scores(toy_corpus),
            toy_corpus,
            toy_gold(toy_corpus),
            gold_column="gold_constructiveness",
            score_columns=("cues",),
            **options,  # type: ignore[arg-type]
        )
    assert message in str(raised.value)
