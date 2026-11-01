"""Tests of the hybrid coder: evidence texts embedded and scored against the reference poles."""

from __future__ import annotations

import inspect
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from embeddings_support import needs_minilm
from hybrid_support import STUB_DIMENSION, STUB_SIDECAR, StubEncoder, unit_references

from copernicus.data import TextTable
from copernicus.embeddings import ENCODERS, Encoder
from copernicus.errors import ConfigurationError, SettingsError
from copernicus.hybrid import HYBRID_DEFAULTS, HybridResult, evidence_texts, score_hybrid
from copernicus.rubric.model import Rubric
from copernicus.similarity import (
    SimilaritySettings,
    reduce_matrix,
    rollup_scores,
    score_similarity,
)

COLUMNS = [
    "text_id",
    "target_level",
    "target_id",
    "polarity",
    "reduction",
    "writer_pooling",
    "long_text",
    "style",
    "encoder",
    "score",
    "n_references",
    "method",
    "fallback",
]
MEAN = SimilaritySettings(reduction="mean")


def attribute_ids(rubric: Rubric) -> list[str]:
    """Return the attribute ids in rubric order."""
    return [attribute.attribute_id for attribute in rubric.attributes]


def test_the_hybrid_defaults_are_pinned_and_are_the_default_of_score_hybrid() -> None:
    assert HYBRID_DEFAULTS == SimilaritySettings(
        polarity="net",
        reduction="pca",
        writer_pooling="pooled",
        long_text="pooled_embedding",
        target_level="construct",
    )
    assert inspect.signature(score_hybrid).parameters["settings"].default is HYBRID_DEFAULTS
    assert SimilaritySettings() != HYBRID_DEFAULTS


def test_the_defaults_score_text_vectors_against_each_whole_pole(
    coded_store: Path, rubric: Rubric, table: TextTable
) -> None:
    encoder = StubEncoder()
    refs = unit_references(rubric, "stub", STUB_DIMENSION)
    frame = score_hybrid(coded_store, table, rubric, encoder, refs, min_words=10).frame
    assert set(frame.writer_pooling) == {"pooled"} and set(frame.long_text) == {"pooled_embedding"}
    assert set(frame.reduction) == {"pca"} and set(frame.polarity) == {"net"}
    texts = evidence_texts(coded_store, rubric, table, granularity="construct", min_words=10)
    vectors = encoder.embed(texts.texts, texts.ids).vectors.astype(np.float64)
    for target in ["construct", *attribute_ids(rubric)]:
        poles = [
            reduce_matrix(
                vectors @ refs.vectors[refs.pole_indices(target, polarity)].astype(np.float64).T,
                "pca",
                target,
            )
            for polarity in ("positive", "negative")
        ]
        rows = frame[frame.target_id == target]
        assert rows.text_id.tolist() == texts.ids
        np.testing.assert_allclose(rows.score, poles[0] - poles[1], atol=1e-10)


def test_construct_granularity_scores_construct_and_attributes_per_text(
    coded_store: Path, rubric: Rubric, table: TextTable
) -> None:
    encoder = StubEncoder()
    refs = unit_references(rubric, "stub", STUB_DIMENSION)
    result = score_hybrid(coded_store, table, rubric, encoder, refs, MEAN, min_words=10)
    assert isinstance(result, HybridResult)
    frame = result.frame
    assert list(frame.columns) == COLUMNS
    targets = ["construct", *attribute_ids(rubric)]
    assert len(frame) == len(table) * len(targets)
    assert set(frame.polarity) == {"net"} and set(frame.method) == {"hybrid_construct"}
    assert sorted(frame.target_id.unique()) == sorted(targets)
    assert not frame.duplicated(["text_id", "target_id"]).any()
    texts = evidence_texts(coded_store, rubric, table, granularity="construct", min_words=10)
    assert encoder.calls == [texts.texts] and result.summary == dict(texts.summary)
    expected = dict(zip(texts.ids, texts.fallback, strict=True))
    assert (frame.fallback == frame.text_id.map(expected)).all()
    assert {"", "no_evidence"} <= set(frame.fallback)
    direct = score_similarity(encoder.embed(texts.texts, texts.ids), refs, rubric, MEAN)
    assert np.allclose(frame.score, direct.score)


def test_attribute_granularity_scores_each_attribute_against_its_own_poles_and_rolls_up(
    coded_store: Path, rubric: Rubric, table: TextTable
) -> None:
    encoder = StubEncoder()
    refs = unit_references(rubric, "stub", STUB_DIMENSION)
    result = score_hybrid(
        coded_store, table, rubric, encoder, refs, MEAN, granularity="attribute", min_words=10
    )
    frame = result.frame
    names = attribute_ids(rubric)
    rollup_id = f"{rubric.rubric_id}__rollup"
    assert list(frame.columns) == COLUMNS and set(frame.method) == {"hybrid_attribute"}
    assert list(frame.target_id.unique()) == [*names, rollup_id]
    assert len(frame) == len(table) * (len(names) + 1)
    rollup = frame[frame.target_id == rollup_id]
    assert (rollup.n_references == len(names)).all() and (rollup.target_level == "attribute").all()
    assert (rollup.polarity == "net").all()
    assert set(frame.target_level) == {"attribute"}
    by_attribute = evidence_texts(coded_store, rubric, table, granularity="attribute", min_words=10)
    assert isinstance(by_attribute, dict)
    scores = {}
    for name in names:
        mine = by_attribute[name]
        direct = score_similarity(
            encoder.embed(mine.texts, mine.ids), refs, rubric, MEAN, targets=[name]
        )
        rows = frame[frame.target_id == name]
        assert rows.text_id.tolist() == mine.ids and np.allclose(rows.score, direct.score)
        assert (rows.n_references == 2 * 2 * 3).all()
        assert (rows.fallback.to_numpy() == np.asarray(mine.fallback)).all()
        scores[name] = direct.score.to_numpy()
    assert np.allclose(rollup.score, rollup_scores(scores, rubric, "mean"))
    for row, text_id in enumerate(by_attribute[names[0]].ids):
        distinct = sorted(
            {each.fallback[row] for each in by_attribute.values()} - {""},
            key=[f for each in by_attribute.values() for f in each.fallback].index,
        )
        assert rollup.fallback.iloc[row] == ",".join(distinct)
        assert rollup.text_id.iloc[row] == text_id


def test_attribute_granularity_embeds_each_distinct_text_once(
    coded_store: Path, rubric: Rubric, table: TextTable
) -> None:
    encoder = StubEncoder()
    refs = unit_references(rubric, "stub", STUB_DIMENSION)
    result = score_hybrid(
        coded_store, table, rubric, encoder, refs, MEAN, granularity="attribute", min_words=10
    )
    by_attribute = evidence_texts(coded_store, rubric, table, granularity="attribute", min_words=10)
    assert isinstance(by_attribute, dict)
    everything = [text for each in by_attribute.values() for text in each.texts]
    assert len(encoder.calls) == 1
    assert sorted(encoder.calls[0]) == sorted(set(everything))
    assert len(encoder.calls[0]) < len(everything)
    summary = result.summary
    assert summary["n_texts"] == len(table) and summary["n_attributes"] == len(by_attribute)
    assert summary["n_distinct_texts_embedded"] == len(encoder.calls[0])
    assert set(summary["by_attribute"]) == set(by_attribute)
    assert summary["by_attribute"]["tone"] == dict(by_attribute["tone"].summary)
    assert summary["n_passages_embedded"] == sum(
        each.summary["n_passages_embedded"] for each in by_attribute.values()
    )


def test_the_roll_up_follows_the_requested_polarity_and_default_reduction(
    coded_store: Path, rubric: Rubric, table: TextTable
) -> None:
    refs = unit_references(rubric, "stub", STUB_DIMENSION)
    settings = SimilaritySettings(polarity="positive")
    frame = score_hybrid(
        coded_store, table, rubric, StubEncoder(), refs, settings, granularity="attribute"
    ).frame
    rollup = frame[frame.target_id.str.endswith("__rollup")]
    assert (rollup.polarity == "positive").all() and (rollup.reduction == "pca").all()
    assert set(frame.polarity) == {"positive"} and rollup.score.notna().all()
    short = table.frame["n_words"].to_numpy() < 40
    assert (rollup.fallback.to_numpy()[short] == "short").all()


def test_an_encoder_other_than_the_one_of_the_references_is_refused(
    coded_store: Path, rubric: Rubric, table: TextTable
) -> None:
    refs = unit_references(rubric, "stub", STUB_DIMENSION)
    with pytest.raises(ConfigurationError, match="other.*stub|stub.*other"):
        score_hybrid(coded_store, table, rubric, StubEncoder("other"), refs, MEAN)


@pytest.mark.parametrize("granularity", ["construct", "attribute"])
def test_the_result_carries_the_provenance_of_the_run(
    coded_store: Path, rubric: Rubric, table: TextTable, granularity: str
) -> None:
    encoder = StubEncoder()
    refs = unit_references(rubric, "stub", STUB_DIMENSION)
    result = score_hybrid(
        coded_store, table, rubric, encoder, refs, MEAN, granularity=granularity, min_words=10
    )
    provenance = result.provenance
    assert provenance["coder"] == "hybrid" and provenance["settings"] == asdict(MEAN)
    assert (provenance["granularity"], provenance["run_number"], provenance["min_words"]) == (
        granularity,
        1,
        10,
    )
    assert provenance["rubric_id"] == rubric.rubric_id
    assert provenance["rubric_hash"] == rubric.content_hash()
    assert provenance["texts"] == {
        "encoder": "stub",
        **STUB_SIDECAR,
        "resolved_revision": "stub-commit",
        "n_texts": len(table),
    }
    references = provenance["references"]
    assert (references["encoder"], references["hf_id"]) == ("stub", "test/stub")
    assert references["rubric_hash"] == rubric.content_hash()
    assert references["writers"] == ["writer-a", "writer-b"]
    assert references["n_references"] == len(refs)
    assert provenance["store"]["method"] == "two_stage"
    assert provenance["store"]["rubric_hash"] == rubric.content_hash()
    assert provenance["n_rows"] == len(result.frame)
    json.dumps(provenance)


@pytest.mark.parametrize("granularity", ["construct", "attribute"])
def test_the_checkpoint_options_reach_the_one_embedding_pass(
    coded_store: Path, rubric: Rubric, table: TextTable, tmp_path: Path, granularity: str
) -> None:
    encoder = StubEncoder()
    refs = unit_references(rubric, "stub", STUB_DIMENSION)
    score_hybrid(
        coded_store,
        table,
        rubric,
        encoder,
        refs,
        MEAN,
        granularity=granularity,
        checkpoint=tmp_path / "state",
        checkpoint_every=7,
    )
    assert encoder.options == [{"checkpoint": tmp_path / "state", "checkpoint_every": 7}]


def test_an_encoder_of_another_identity_is_refused_before_anything_is_embedded(
    coded_store: Path, rubric: Rubric, table: TextTable
) -> None:
    encoder = StubEncoder(hf_id="test/another-model")
    refs = unit_references(rubric, "stub", STUB_DIMENSION)
    with pytest.raises(ConfigurationError, match="hf_id"):
        score_hybrid(coded_store, table, rubric, encoder, refs)
    assert encoder.calls == []


def test_references_of_another_commit_are_refused_once_the_loaded_commit_is_known(
    coded_store: Path, rubric: Rubric, table: TextTable
) -> None:
    encoder = StubEncoder()
    refs = unit_references(rubric, "stub", STUB_DIMENSION)
    refs = replace(refs, sidecar={**refs.sidecar, "resolved_revision": "other-commit"})
    with pytest.raises(ConfigurationError, match="'stub-commit' and 'other-commit'"):
        score_hybrid(coded_store, table, rubric, encoder, refs, MEAN)
    assert len(encoder.calls) == 1


def test_a_granularity_outside_the_two_is_refused(
    coded_store: Path, rubric: Rubric, table: TextTable
) -> None:
    refs = unit_references(rubric, "stub", STUB_DIMENSION)
    with pytest.raises(SettingsError, match="granularity"):
        score_hybrid(coded_store, table, rubric, StubEncoder(), refs, MEAN, granularity="marker")


def test_the_rubric_of_the_store_is_checked_before_anything_is_embedded(
    coded_store: Path, rubric: Rubric, table: TextTable
) -> None:
    other = replace(rubric, concept="another construct")
    encoder = StubEncoder()
    refs = unit_references(other, "stub", STUB_DIMENSION)
    with pytest.raises(ConfigurationError, match=rubric.content_hash()):
        score_hybrid(coded_store, table, other, encoder, refs, MEAN)
    assert encoder.calls == []


@pytest.mark.parametrize("granularity", ["construct", "attribute"])
def test_a_text_that_changed_since_the_coding_is_refused_before_anything_is_embedded(
    coded_store: Path, rubric: Rubric, table: TextTable, granularity: str
) -> None:
    frame = table.frame.copy()
    changed_id = str(frame.loc[4, "text_id"])
    frame.loc[4, "text"] = "A different text now stands under this id of the table."
    encoder = StubEncoder()
    refs = unit_references(rubric, "stub", STUB_DIMENSION)
    with pytest.raises(ConfigurationError, match=f"1 of {len(table)} texts differ") as refused:
        score_hybrid(
            coded_store, TextTable.from_frame(frame), rubric, encoder, refs, MEAN,
            granularity=granularity,
        )  # fmt: skip
    assert repr(changed_id) in str(refused.value)
    assert encoder.calls == []


@needs_minilm
def test_minilm_scores_one_net_row_per_text_and_target(
    coded_store: Path, rubric: Rubric, table: TextTable
) -> None:
    encoder = Encoder("minilm", device="cpu")
    refs = unit_references(rubric, "minilm", 384, identity=ENCODERS["minilm"].identity())
    frame = score_hybrid(coded_store, table, rubric, encoder, refs, MEAN, min_words=10).frame
    targets = ["construct", *attribute_ids(rubric)]
    assert len(frame) == len(table) * len(targets) and set(frame.polarity) == {"net"}
    assert (frame.groupby(["text_id", "target_id"]).size() == 1).all()
    assert set(frame.encoder) == {"minilm"} and frame.score.between(-2, 2).all()
    assert set(frame.method) == {"hybrid_construct"} and "fallback" in frame.columns


@needs_minilm
def test_minilm_attribute_granularity_adds_the_roll_up_of_three_attributes(
    coded_store: Path, rubric: Rubric, table: TextTable
) -> None:
    encoder = Encoder("minilm", device="cpu")
    refs = unit_references(rubric, "minilm", 384, identity=ENCODERS["minilm"].identity())
    frame = score_hybrid(
        coded_store, table, rubric, encoder, refs, MEAN, granularity="attribute", min_words=10
    ).frame
    names = attribute_ids(rubric)
    rollup = frame[frame.target_id == f"{rubric.rubric_id}__rollup"]
    assert len(rollup) == len(table) and (rollup.n_references == 3).all()
    assert set(frame.target_id) == {*names, f"{rubric.rubric_id}__rollup"}
    assert len(frame) == len(table) * 4 and set(frame.method) == {"hybrid_attribute"}
    assert isinstance(frame, pd.DataFrame) and frame.score.notna().all()
