"""Tests that the hybrid coder scores its evidence texts only against references of one load."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from types import MethodType
from typing import Any

import pandas as pd
import pytest
from hybrid_support import STUB_DIMENSION, StubEncoder, unit_references
from similarity_support import VERSIONS

from copernicus.data import TextTable
from copernicus.embeddings import EmbeddingSet, Encoder
from copernicus.embeddings.encoder import EncoderRuntime
from copernicus.errors import ConfigurationError
from copernicus.hybrid import score_hybrid
from copernicus.rubric.model import Rubric
from copernicus.similarity import ReferenceEmbeddings, SimilaritySettings

MEAN = SimilaritySettings(reduction="mean")
UNKNOWN = {"torch_version": "unknown", "transformers_version": "unknown"}
GRANULARITIES = ("construct", "attribute")


class VersionedStubEncoder(StubEncoder):
    """The stub encoder, recording the torch and transformers versions of one environment."""

    def __init__(self, versions: dict[str, str]) -> None:
        super().__init__()
        self.versions = versions

    def embed(
        self,
        texts: Sequence[str],
        ids: Sequence[Any],
        *,
        checkpoint: Path | None = None,
        checkpoint_every: int = 2048,
    ) -> EmbeddingSet:
        """Return the stub's vectors, with the versions in the sidecar."""
        embedded = super().embed(
            texts, ids, checkpoint=checkpoint, checkpoint_every=checkpoint_every
        )
        return replace(embedded, sidecar={**embedded.sidecar, **self.versions})


def references(rubric: Rubric, versions: dict[str, str]) -> ReferenceEmbeddings:
    """Return the stub's reference embeddings with the versions of `VERSIONS`, some replaced."""
    refs = unit_references(rubric, "stub", STUB_DIMENSION)
    return replace(refs, sidecar={**refs.sidecar, **VERSIONS, **versions})


def unused_pool(hidden: Any, mask: Any) -> Any:
    """Stand in for the pooling of a loaded runtime; the stand-in pass never pools."""
    raise AssertionError("the stand-in runtime was asked to pool a batch")


def library_encoder(versions: dict[str, str]) -> tuple[Encoder, list[list[str]]]:
    """Return the library's encoder with a stand-in runtime already loaded, and its pass's calls.

    The runtime records the versions of one environment, as a loaded runtime records those it
    runs under. The pass is the stub's: it records each call and returns the stub's vectors with
    the runtime's record in the sidecar, as the library's own pass records it. No model is
    loaded, and torch is not needed.
    """
    stub = StubEncoder()
    encoder = Encoder(stub.spec)
    encoder._runtime = EncoderRuntime(
        torch=None,
        tokenizer=None,
        model=None,
        pool=unused_pool,
        device=None,
        versions={"resolved_revision": "stub-commit", **versions},
    )
    calls: list[list[str]] = []

    def embed(
        self: Encoder, texts: Sequence[str], ids: Sequence[Any], **options: Any
    ) -> EmbeddingSet:
        calls.append(list(texts))
        embedded = stub.embed(texts, ids, **options)
        assert self._runtime is not None
        return replace(embedded, sidecar={**embedded.sidecar, **self._runtime.versions})

    encoder.embed = MethodType(embed, encoder)  # type: ignore[method-assign]
    return encoder, calls


@pytest.mark.parametrize("granularity", GRANULARITIES)
def test_the_provenance_records_the_comparison_of_the_two_loads(
    coded_store: Path, rubric: Rubric, table: TextTable, granularity: str
) -> None:
    encoder = VersionedStubEncoder(VERSIONS)
    refs = references(rubric, VERSIONS)
    result = score_hybrid(
        coded_store, table, rubric, encoder, refs, MEAN, granularity=granularity, min_words=10
    )
    assert result.provenance["load_comparison"] == {
        "texts": VERSIONS,
        "references": VERSIONS,
        "warnings": [],
    }


@pytest.mark.parametrize("granularity", GRANULARITIES)
def test_references_of_another_major_version_are_refused_once_the_texts_are_embedded(
    coded_store: Path, rubric: Rubric, table: TextTable, granularity: str
) -> None:
    encoder = VersionedStubEncoder(VERSIONS)
    refs = references(rubric, {"transformers_version": "5.12.1"})
    with pytest.raises(ConfigurationError) as refused:
        score_hybrid(coded_store, table, rubric, encoder, refs, MEAN, granularity=granularity)
    message = str(refused.value)
    assert "transformers 4.57.6 and 5.12.1" in message
    assert message.endswith("embed both sides in one environment")
    assert len(encoder.calls) == 1


def test_a_difference_of_torch_is_scored_and_recorded_in_the_provenance(
    coded_store: Path, rubric: Rubric, table: TextTable, caplog: pytest.LogCaptureFixture
) -> None:
    encoder = VersionedStubEncoder(VERSIONS)
    refs = references(rubric, {"torch_version": "2.4.1"})
    with caplog.at_level(logging.WARNING):
        result = score_hybrid(coded_store, table, rubric, encoder, refs, MEAN, min_words=10)
    comparison = result.provenance["load_comparison"]
    assert comparison["texts"] == VERSIONS
    assert comparison["references"] == {**VERSIONS, "torch_version": "2.4.1"}
    (warning,) = comparison["warnings"]
    assert "torch 2.5.0 and 2.4.1" in warning
    assert [record.getMessage() for record in caplog.records] == [warning]


@pytest.mark.parametrize("granularity", GRANULARITIES)
def test_the_library_encoder_is_refused_before_its_pass_for_references_of_another_major_version(
    coded_store: Path, rubric: Rubric, table: TextTable, granularity: str
) -> None:
    encoder, calls = library_encoder(VERSIONS)
    refs = references(rubric, {"transformers_version": "5.12.1"})
    with pytest.raises(ConfigurationError) as refused:
        score_hybrid(coded_store, table, rubric, encoder, refs, MEAN, granularity=granularity)
    assert "transformers 4.57.6 and 5.12.1, two major versions" in str(refused.value)
    assert calls == []


def test_the_library_encoder_is_refused_before_its_pass_for_references_of_another_commit(
    coded_store: Path, rubric: Rubric, table: TextTable
) -> None:
    encoder, calls = library_encoder(VERSIONS)
    refs = references(rubric, {"resolved_revision": "other-commit"})
    with pytest.raises(ConfigurationError, match="'stub-commit' and 'other-commit'"):
        score_hybrid(coded_store, table, rubric, encoder, refs, MEAN)
    assert calls == []


def test_the_store_and_the_table_are_checked_before_the_library_encoder_is_loaded(
    coded_store: Path, rubric: Rubric, table: TextTable
) -> None:
    encoder, calls = library_encoder(VERSIONS)
    runtime = encoder._runtime
    assert runtime is not None
    loads: list[str] = []

    def counted_load() -> EncoderRuntime:
        loads.append(encoder.spec.name)
        return runtime

    encoder._load = counted_load  # type: ignore[method-assign]
    frame = table.frame.copy()
    frame.loc[4, "text"] = "A different text now stands under this id of the table."
    with pytest.raises(ConfigurationError, match="texts differ"):
        score_hybrid(
            coded_store, TextTable.from_frame(frame), rubric, encoder, references(rubric, {}), MEAN
        )
    assert loads == [] and calls == []
    score_hybrid(coded_store, table, rubric, encoder, references(rubric, {}), MEAN, min_words=10)
    assert loads == ["stub"] and len(calls) == 1


@pytest.mark.parametrize("granularity", GRANULARITIES)
def test_one_load_of_the_library_encoder_scores_as_an_encoder_object_of_ones_own(
    coded_store: Path,
    rubric: Rubric,
    table: TextTable,
    caplog: pytest.LogCaptureFixture,
    granularity: str,
) -> None:
    encoder, calls = library_encoder(VERSIONS)
    refs = references(rubric, {})
    with caplog.at_level(logging.WARNING):
        library = score_hybrid(
            coded_store, table, rubric, encoder, refs, MEAN, granularity=granularity, min_words=10
        )
    assert caplog.records == [] and len(calls) == 1
    own = score_hybrid(
        coded_store,
        table,
        rubric,
        VersionedStubEncoder(VERSIONS),
        refs,
        MEAN,
        granularity=granularity,
        min_words=10,
    )
    pd.testing.assert_frame_equal(library.frame, own.frame, check_exact=True)
    assert library.provenance["load_comparison"] == own.provenance["load_comparison"]
    assert library.provenance["texts"] == own.provenance["texts"]


def test_an_encoder_and_references_that_record_no_version_are_recorded_as_unknown(
    coded_store: Path, rubric: Rubric, table: TextTable
) -> None:
    refs = unit_references(rubric, "stub", STUB_DIMENSION)
    result = score_hybrid(coded_store, table, rubric, StubEncoder(), refs, MEAN, min_words=10)
    comparison = result.provenance["load_comparison"]
    assert (comparison["texts"], comparison["references"]) == (UNKNOWN, UNKNOWN)
    assert len(comparison["warnings"]) == 2
    assert all("could not be compared" in warning for warning in comparison["warnings"])
