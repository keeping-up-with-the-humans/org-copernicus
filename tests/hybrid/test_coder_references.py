"""Tests of the hybrid coder's check of its reference embeddings, made before anything is embedded.

The references must have been written for the content of the rubric of the run, and must hold
the poles that the granularity scores: every target at construct granularity, the attributes at
attribute granularity.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from hybrid_support import STUB_DIMENSION, StubEncoder, unit_references

from copernicus.assets import load_toy_cues
from copernicus.coders import CodingSettings, code_texts
from copernicus.data import TextTable
from copernicus.errors import ConfigurationError
from copernicus.hybrid import score_hybrid
from copernicus.llm import FakeBackend, cue_responder
from copernicus.rubric.model import Rubric
from copernicus.similarity import ReferenceEmbeddings, SimilaritySettings

MEAN = SimilaritySettings(reduction="mean")
N_TEXTS = 4
LABELS = ("target", "polarity", "level", "style", "writer_model")


def without_target(refs: ReferenceEmbeddings, target: str) -> ReferenceEmbeddings:
    """Return the references without the rows of one target."""
    keep = refs.target != target
    labels = {name: getattr(refs, name)[keep] for name in LABELS}
    return replace(refs, vectors=refs.vectors[keep], **labels)


@pytest.mark.parametrize("granularity", ["construct", "attribute"])
def test_references_of_the_rubric_before_its_rewrite_are_refused_before_anything_is_embedded(
    tmp_path: Path,
    table: TextTable,
    toy_rubric: Rubric,
    rewritten_toy_rubric: Rubric,
    granularity: str,
) -> None:
    rewritten = rewritten_toy_rubric
    small = table.subset(table.ids()[:N_TEXTS])
    store = tmp_path / "rewritten.sqlite"
    backend = FakeBackend(cue_responder(rewritten, load_toy_cues()))
    code_texts(small, rewritten, backend, store, CodingSettings(runs=1))
    refs = unit_references(toy_rubric, "stub", STUB_DIMENSION)
    encoder = StubEncoder()
    with pytest.raises(ConfigurationError) as refused:
        score_hybrid(store, small, rewritten, encoder, refs, MEAN, granularity=granularity)
    assert str(refused.value) == (
        f"the reference embeddings were generated for rubric {toy_rubric.rubric_id} with hash "
        f"{toy_rubric.content_hash()}, but the rubric now has hash {rewritten.content_hash()}: "
        "the references describe the earlier definition; regenerate them (or rebuild a "
        "hand-written set with ReferenceSet.for_rubric)"
    )
    assert encoder.calls == []
    bound = unit_references(rewritten, "stub", STUB_DIMENSION)
    result = score_hybrid(store, small, rewritten, encoder, bound, MEAN, granularity=granularity)
    assert result.provenance["references"]["rubric_hash"] == rewritten.content_hash()
    assert result.provenance["rubric_hash"] == rewritten.content_hash()


def test_a_missing_construct_pole_is_refused_before_anything_is_embedded_at_construct_level(
    coded_store: Path, rubric: Rubric, table: TextTable
) -> None:
    refs = without_target(unit_references(rubric, "stub", STUB_DIMENSION), "construct")
    encoder = StubEncoder()
    with pytest.raises(ConfigurationError, match="lack the positive pole of target construct"):
        score_hybrid(coded_store, table, rubric, encoder, refs, MEAN)
    assert encoder.calls == []
    result = score_hybrid(coded_store, table, rubric, encoder, refs, MEAN, granularity="attribute")
    assert "construct" not in set(result.frame.target_id) and len(encoder.calls) == 1
    assert np.isfinite(result.frame.score).all()
