"""Tests that texts are scored against references only when one load of the encoder made both.

Another major version of transformers is refused. A smaller difference of transformers, any
difference of torch and a side that records no version are scored with a warning, which the
provenance of the scoring records under ``load_comparison`` with the versions of both sides.
"""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from similarity_support import (
    ENCODER,
    SIDECAR,
    VERSIONS,
    make_references,
    make_texts,
    texts_along_first_axis,
)

import copernicus.similarity.grid as grid
from copernicus._digests import arrays_hash
from copernicus.embeddings import EmbeddingSet
from copernicus.errors import ConfigurationError
from copernicus.rubric.model import Rubric
from copernicus.similarity import (
    ReferenceEmbeddings,
    SimilaritySettings,
    run_grid,
    score_similarity,
    scoring_provenance,
)

LOGGER = "copernicus.similarity.poles"
EXACT = SimilaritySettings(reduction="mean", writer_pooling="pooled", long_text="pooled_embedding")
PACKAGE_KEYS = ("torch_version", "transformers_version")
UNKNOWN = dict.fromkeys(PACKAGE_KEYS, "unknown")
ONE_ENVIRONMENT = "embed both sides in one environment"
COMMIT = "0123456789abcdef0123456789abcdef01234567"
TEXT_ARRAYS = ("ids", "vectors", "chunk_vectors", "chunk_owner", "chunk_weight")
REFERENCE_ARRAYS = ("vectors", "target", "polarity", "level", "style", "writer_model")


def one_load(rubric: Rubric) -> tuple[EmbeddingSet, ReferenceEmbeddings]:
    """Return texts and references of one load: the builders' identity and versions."""
    return make_texts(texts_along_first_axis(5)), make_references(rubric)


def with_versions(
    rubric: Rubric, texts_versions: dict[str, Any], references_versions: dict[str, Any]
) -> tuple[EmbeddingSet, ReferenceEmbeddings]:
    """Return texts and references of one load with some of their version keys replaced."""
    texts, refs = one_load(rubric)
    return (
        replace(texts, sidecar={**texts.sidecar, **texts_versions}),
        replace(refs, sidecar={**refs.sidecar, **references_versions}),
    )


def without_versions(sidecar: Any) -> dict[str, Any]:
    """Return a sidecar without the torch and transformers versions, as one built by hand."""
    return {key: value for key, value in sidecar.items() if key not in PACKAGE_KEYS}


def recorded(texts: EmbeddingSet, refs: ReferenceEmbeddings, rubric: Rubric) -> dict[str, Any]:
    """Return the comparison of the two loads that the provenance of a scoring records."""
    comparison: dict[str, Any] = scoring_provenance(texts, refs, rubric, [EXACT], 0)[
        "load_comparison"
    ]
    return comparison


def one_load_scores(rubric: Rubric) -> pd.DataFrame:
    """Return the scores of texts and references of one load."""
    texts, refs = one_load(rubric)
    return score_similarity(texts, refs, rubric, EXACT)


def test_one_load_scores_as_an_independent_computation_without_a_warning(
    toy_rubric: Rubric, caplog: pytest.LogCaptureFixture
) -> None:
    texts, refs = one_load(toy_rubric)
    with caplog.at_level(logging.WARNING):
        frame = score_similarity(texts, refs, toy_rubric, EXACT)
    assert caplog.records == []
    vectors = texts.vectors.astype(np.float64)
    for target in ["construct", *(attribute.attribute_id for attribute in toy_rubric.attributes)]:
        poles = []
        for polarity in ("positive", "negative"):
            rows = np.flatnonzero((refs.target == target) & (refs.polarity == polarity))
            poles.append((vectors @ refs.vectors[rows].astype(np.float64).T).mean(axis=1))
        scores = frame.loc[frame["target_id"] == target, "score"].to_numpy()
        np.testing.assert_array_equal(scores, poles[0] - poles[1])
    assert recorded(texts, refs, toy_rubric) == {
        "texts": VERSIONS,
        "references": VERSIONS,
        "warnings": [],
    }


@pytest.mark.parametrize(
    ("key", "texts_version", "references_version"),
    [
        ("transformers_version", "4.57.3", "4.57.6"),
        ("transformers_version", "5.12.1", "5.0.0"),
        ("torch_version", "2.5.0", "2.4.1+cu121"),
    ],
)
def test_a_difference_below_the_major_version_or_of_torch_is_scored_with_a_warning_and_recorded(
    toy_rubric: Rubric,
    caplog: pytest.LogCaptureFixture,
    key: str,
    texts_version: str,
    references_version: str,
) -> None:
    texts, refs = with_versions(toy_rubric, {key: texts_version}, {key: references_version})
    with caplog.at_level(logging.WARNING):
        frame = score_similarity(texts, refs, toy_rubric, EXACT)
    (record,) = caplog.records
    assert (record.name, record.levelno) == (LOGGER, logging.WARNING)
    message = record.getMessage()
    package = key.removesuffix("_version")
    assert f"{package} {texts_version} and {references_version}" in message
    assert ONE_ENVIRONMENT in message
    pd.testing.assert_frame_equal(frame, one_load_scores(toy_rubric), check_exact=True)
    comparison = recorded(texts, refs, toy_rubric)
    assert comparison["texts"] == {**VERSIONS, key: texts_version}
    assert comparison["references"] == {**VERSIONS, key: references_version}
    assert comparison["warnings"] == [message]


@pytest.mark.parametrize("bare", ["texts", "references", "both"])
def test_a_side_that_records_no_version_is_scored_with_a_warning_and_recorded_as_unknown(
    toy_rubric: Rubric, caplog: pytest.LogCaptureFixture, bare: str
) -> None:
    texts, refs = one_load(toy_rubric)
    if bare != "references":
        texts = replace(texts, sidecar=without_versions(texts.sidecar))
    if bare != "texts":
        refs = replace(refs, sidecar=without_versions(refs.sidecar))
    with caplog.at_level(logging.WARNING):
        frame = score_similarity(texts, refs, toy_rubric, EXACT)
    messages = [record.getMessage() for record in caplog.records]
    assert len(messages) == (2 if bare == "both" else 1)
    assert all("could not be compared" in message for message in messages)
    pd.testing.assert_frame_equal(frame, one_load_scores(toy_rubric), check_exact=True)
    comparison = recorded(texts, refs, toy_rubric)
    assert comparison["texts"] == (VERSIONS if bare == "references" else UNKNOWN)
    assert comparison["references"] == (VERSIONS if bare == "texts" else UNKNOWN)
    assert comparison["warnings"] == messages


@pytest.mark.parametrize("value", [None, "unknown"])
def test_a_version_recorded_as_null_or_unknown_counts_as_not_recorded(
    toy_rubric: Rubric, caplog: pytest.LogCaptureFixture, value: str | None
) -> None:
    texts, refs = with_versions(toy_rubric, {}, {"transformers_version": value})
    with caplog.at_level(logging.WARNING):
        score_similarity(texts, refs, toy_rubric, EXACT)
    (record,) = caplog.records
    assert "transformers" in record.getMessage()
    assert "could not be compared" in record.getMessage()
    expected = {**VERSIONS, "transformers_version": "unknown"}
    assert recorded(texts, refs, toy_rubric)["references"] == expected


def grid_inputs(
    rubric: Rubric, register_versions: dict[str, str]
) -> dict[tuple[str, str], tuple[EmbeddingSet, ReferenceEmbeddings]]:
    """Two keys of one encoder; the references of the register key carry the given versions."""
    register = make_references(rubric, style="register", seed=5)
    register = replace(register, sidecar={**register.sidecar, **register_versions})
    return {
        (ENCODER, "neutral"): (make_texts(texts_along_first_axis(4)), make_references(rubric)),
        (ENCODER, "register"): (make_texts(texts_along_first_axis(4)), register),
    }


def test_run_grid_refuses_two_major_versions_before_any_key_is_scored(
    toy_rubric: Rubric, monkeypatch: pytest.MonkeyPatch
) -> None:
    scored: list[object] = []
    monkeypatch.setattr(grid, "score_similarity_many", lambda *args, **kwargs: scored.append(args))
    inputs = grid_inputs(toy_rubric, {"transformers_version": "5.12.1"})
    with pytest.raises(ConfigurationError) as refused:
        run_grid(inputs, toy_rubric, [EXACT])
    message = str(refused.value)
    assert message.startswith(f"key {(ENCODER, 'register')}: ")
    assert "transformers 4.57.6 and 5.12.1" in message and message.endswith(ONE_ENVIRONMENT)
    assert scored == []


def test_run_grid_scores_a_key_of_another_release_with_a_warning(
    toy_rubric: Rubric, caplog: pytest.LogCaptureFixture
) -> None:
    inputs = grid_inputs(toy_rubric, {"transformers_version": "4.57.3"})
    with caplog.at_level(logging.WARNING):
        frame = run_grid(inputs, toy_rubric, [EXACT])
    (record,) = caplog.records
    assert "transformers 4.57.6 and 4.57.3" in record.getMessage()
    assert set(frame["style"]) == {"neutral", "register"}


def save_in_the_form_of_version_0_1_0(
    texts: EmbeddingSet, refs: ReferenceEmbeddings, folder: Path, versions: dict[str, str]
) -> tuple[Path, Path]:
    """Write texts and references with the keys that version 0.1.0 wrote, and no others.

    Version 0.1.0 wrote the encoder name, the identity keys, the version keys that the sidecar
    held (all three for a set its encoder made, none for a set built by hand), the counts, the
    library version, the creation time and the digest of the arrays; for references also the
    rubric id, its content hash and the binding. It wrote no ``text_form`` and no
    ``texts_hash``.
    """
    created = datetime.fromtimestamp(0, timezone.utc).isoformat()
    text_arrays = {name: getattr(texts, name) for name in TEXT_ARRAYS}
    reference_arrays = {name: getattr(refs, name) for name in REFERENCE_ARRAYS}
    descriptions = (
        {
            "encoder": texts.encoder,
            **SIDECAR,
            **versions,
            "n_texts": len(texts),
            "n_chunks": len(texts.chunk_owner),
            "library_version": "0.1.0",
            "created_at": created,
            "arrays_hash": arrays_hash(text_arrays),
        },
        {
            "encoder": refs.encoder,
            **SIDECAR,
            **versions,
            "rubric_id": refs.rubric_id,
            "rubric_hash": refs.rubric_hash,
            "bound_with": refs.bound_with,
            "n_references": len(refs),
            "library_version": "0.1.0",
            "created_at": created,
            "arrays_hash": arrays_hash(reference_arrays),
        },
    )
    stems = (folder / "texts", folder / "references")
    for stem, arrays, description in zip(
        stems, (text_arrays, reference_arrays), descriptions, strict=True
    ):
        np.savez(stem.with_name(stem.name + ".npz"), **arrays)
        text = json.dumps(description, indent=2) + "\n"
        stem.with_name(stem.name + ".json").write_text(text, encoding="utf-8")
    return stems


def test_a_pair_that_the_encoder_of_version_0_1_0_saved_scores_as_one_load(
    toy_rubric: Rubric, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    texts, refs = one_load(toy_rubric)
    versions = {"resolved_revision": COMMIT, **VERSIONS}
    texts_stem, refs_stem = save_in_the_form_of_version_0_1_0(texts, refs, tmp_path, versions)
    loaded_texts = EmbeddingSet.load(texts_stem)
    loaded_refs = ReferenceEmbeddings.load(refs_stem, expect_encoder=ENCODER)
    assert loaded_texts.sidecar["library_version"] == "0.1.0"
    assert "text_form" not in loaded_texts.sidecar
    with caplog.at_level(logging.WARNING):
        frame = score_similarity(loaded_texts, loaded_refs, toy_rubric, EXACT)
    assert caplog.records == []
    pd.testing.assert_frame_equal(frame, one_load_scores(toy_rubric), check_exact=True)
    assert recorded(loaded_texts, loaded_refs, toy_rubric) == {
        "texts": VERSIONS,
        "references": VERSIONS,
        "warnings": [],
    }


def test_a_pair_built_by_hand_and_saved_by_version_0_1_0_scores_with_the_warning(
    toy_rubric: Rubric, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    texts, refs = one_load(toy_rubric)
    texts_stem, refs_stem = save_in_the_form_of_version_0_1_0(texts, refs, tmp_path, {})
    loaded_texts = EmbeddingSet.load(texts_stem)
    loaded_refs = ReferenceEmbeddings.load(refs_stem)
    with caplog.at_level(logging.WARNING):
        frame = score_similarity(loaded_texts, loaded_refs, toy_rubric, EXACT)
    messages = [record.getMessage() for record in caplog.records]
    assert len(messages) == 2 and all("could not be compared" in m for m in messages)
    pd.testing.assert_frame_equal(frame, one_load_scores(toy_rubric), check_exact=True)
    assert recorded(loaded_texts, loaded_refs, toy_rubric) == {
        "texts": UNKNOWN,
        "references": UNKNOWN,
        "warnings": messages,
    }
