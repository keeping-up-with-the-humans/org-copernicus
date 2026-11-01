"""Tests of the binding of reference texts and their embeddings to the content of their rubric.

A reference set, its embeddings and the files of both record the content hash of the rubric the
texts were written for. Every scoring path refuses a rubric of other content, also one rewritten
under the same ids, and the provenance of a scoring run records the hash of the references.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hybrid_support import StubEncoder
from references_support import clean_responder, writer

import copernicus.similarity.grid as grid
from copernicus import __version__
from copernicus.embeddings import EmbeddingSet
from copernicus.errors import ConfigurationError
from copernicus.references import GeneratorSettings, ReferenceSet, build_cells, generate_references
from copernicus.references.reference_set import FRAME_COLUMNS
from copernicus.rubric.model import Rubric
from copernicus.similarity import (
    ReferenceEmbeddings,
    SimilaritySettings,
    embed_references,
    run_grid,
    score_similarity,
    score_similarity_many,
    scoring_provenance,
)

MEAN = SimilaritySettings(reduction="mean")
ROLLED_UP = SimilaritySettings(reduction="mean", target_level="attribute")
TEXTS = ["I wrote the first document.", "I wrote a second document."]
Scorer = Callable[[EmbeddingSet, ReferenceEmbeddings, Rubric], pd.DataFrame]
SCORERS: dict[str, Scorer] = {
    "construct level": lambda texts, refs, rubric: score_similarity(texts, refs, rubric, MEAN),
    "attribute level": lambda texts, refs, rubric: score_similarity(texts, refs, rubric, ROLLED_UP),
    "restricted targets": lambda texts, refs, rubric: score_similarity(
        texts, refs, rubric, MEAN, targets=["tone"]
    ),
    "several settings": lambda texts, refs, rubric: score_similarity_many(
        texts, refs, rubric, [MEAN, ROLLED_UP]
    ),
    "grid": lambda texts, refs, rubric: run_grid(
        {("stub", "neutral"): (texts, refs)}, rubric, [MEAN]
    ),
}


def refusal(written_for: Rubric, given: Rubric) -> str:
    """Return the message that refuses references written for one rubric with another."""
    return (
        f"the reference embeddings were generated for rubric {written_for.rubric_id} with hash "
        f"{written_for.content_hash()}, but the rubric now has hash {given.content_hash()}: the "
        "references describe the earlier definition; regenerate them (or rebuild a hand-written "
        "set with ReferenceSet.for_rubric)"
    )


def hand_written_frame(rubric: Rubric) -> pd.DataFrame:
    """Write one text by hand for every cell of the rubric, in the columns of a reference set."""
    rows = [
        {
            "rubric_id": rubric.rubric_id,
            "target": cell.target,
            "polarity": cell.polarity,
            "level": cell.level,
            "style": "neutral",
            "writer_model": "hand",
            "text": f"A {cell.polarity} note by hand on {cell.target}, at the {cell.level} level.",
            "call_index": 0,
            "attempt": 0,
            "seed": 0,
            "prompt_tokens": None,
            "completion_tokens": None,
            "temperature_applied": None,
            "top_p_applied": None,
            "prompt_hash": "hand",
            "prompt_version": "hand",
            "screened_out": False,
            "screen_reason": "",
        }
        for cell in build_cells(rubric)
    ]
    return pd.DataFrame(rows, columns=list(FRAME_COLUMNS))


@pytest.fixture
def through_the_files(
    toy_rubric: Rubric, tmp_path: Path
) -> tuple[EmbeddingSet, ReferenceSet, ReferenceEmbeddings]:
    """Generate references, save and load them, embed them, save and load the embeddings.

    Returns:
        Two embedded texts, the loaded reference set, and the loaded reference embeddings.
    """
    settings = GeneratorSettings(texts_per_call=1, max_retries=1)
    generated = generate_references(toy_rubric, [writer(clean_responder(1))], settings)
    loaded = ReferenceSet.load(generated.save(tmp_path / "references.csv"))
    encoder = StubEncoder()
    embed_references(loaded, encoder).save(tmp_path / "vectors")
    texts = encoder.embed(TEXTS, ["a", "b"])
    return texts, loaded, ReferenceEmbeddings.load(tmp_path / "vectors")


def test_the_hash_of_the_generating_rubric_survives_every_file(
    toy_rubric: Rubric, through_the_files: tuple[EmbeddingSet, ReferenceSet, ReferenceEmbeddings]
) -> None:
    _, loaded, refs = through_the_files
    assert loaded.rubric_hash == toy_rubric.content_hash()
    assert refs.rubric_hash == toy_rubric.content_hash()
    assert refs.sidecar["rubric_hash"] == toy_rubric.content_hash()
    assert loaded.bound_with == refs.bound_with == refs.sidecar["bound_with"] == "generated"


@pytest.mark.parametrize("path", sorted(SCORERS))
def test_a_rubric_rewritten_under_the_same_ids_is_refused_on_every_scoring_path(
    toy_rubric: Rubric,
    rewritten_toy_rubric: Rubric,
    through_the_files: tuple[EmbeddingSet, ReferenceSet, ReferenceEmbeddings],
    path: str,
) -> None:
    texts, _, refs = through_the_files
    assert rewritten_toy_rubric.rubric_id == toy_rubric.rubric_id
    assert rewritten_toy_rubric.content_hash() != toy_rubric.content_hash()
    assert not SCORERS[path](texts, refs, toy_rubric).empty
    with pytest.raises(ConfigurationError) as refused:
        SCORERS[path](texts, refs, rewritten_toy_rubric)
    named = "key ('stub', 'neutral'): " if path == "grid" else ""
    assert str(refused.value) == named + refusal(toy_rubric, rewritten_toy_rubric)


def test_the_grid_checks_the_references_of_every_key_before_it_scores_one(
    toy_rubric: Rubric,
    rewritten_toy_rubric: Rubric,
    through_the_files: tuple[EmbeddingSet, ReferenceSet, ReferenceEmbeddings],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    texts, _, refs = through_the_files
    other_content = replace(
        refs,
        style=np.full(len(refs), "register"),
        sidecar={**refs.sidecar, "rubric_hash": rewritten_toy_rubric.content_hash()},
    )
    scored: list[object] = []
    monkeypatch.setattr(grid, "score_similarity_many", lambda *args, **kwargs: scored.append(args))
    inputs = {("stub", "neutral"): (texts, refs), ("stub", "register"): (texts, other_content)}
    with pytest.raises(ConfigurationError) as refused:
        run_grid(inputs, toy_rubric, [MEAN])
    key = ("stub", "register")
    assert str(refused.value) == f"key {key}: {refusal(rewritten_toy_rubric, toy_rubric)}"
    keep = ~((refs.target == "construct") & (refs.polarity == "negative"))
    lacking = replace(
        refs,
        vectors=refs.vectors[keep],
        target=refs.target[keep],
        polarity=refs.polarity[keep],
        level=refs.level[keep],
        writer_model=refs.writer_model[keep],
        style=np.full(int(keep.sum()), "register"),
    )
    inputs = {("stub", "neutral"): (texts, refs), key: (texts, lacking)}
    with pytest.raises(ConfigurationError) as missing:
        run_grid(inputs, toy_rubric, [MEAN])
    assert str(missing.value).startswith(f"key {key}: reference embeddings for rubric ")
    assert str(missing.value).endswith("lack the negative pole of target construct")
    assert scored == []


def test_the_provenance_of_a_scoring_run_records_the_hash_of_the_references(
    toy_rubric: Rubric, through_the_files: tuple[EmbeddingSet, ReferenceSet, ReferenceEmbeddings]
) -> None:
    texts, _, refs = through_the_files
    frame = score_similarity(texts, refs, toy_rubric, MEAN)
    provenance = scoring_provenance(texts, refs, toy_rubric, [MEAN], len(frame))
    assert provenance["references"]["rubric_id"] == toy_rubric.rubric_id
    assert provenance["references"]["rubric_hash"] == toy_rubric.content_hash()
    assert provenance["references"]["bound_with"] == "generated"
    assert provenance["rubric_hash"] == toy_rubric.content_hash()


def test_a_hand_written_set_bound_with_for_rubric_scores_with_that_content_only(
    toy_rubric: Rubric, rewritten_toy_rubric: Rubric
) -> None:
    frame = hand_written_frame(toy_rubric)
    bound = ReferenceSet.for_rubric(frame, toy_rubric, GeneratorSettings())
    assert (bound.rubric_id, bound.rubric_hash) == (toy_rubric.rubric_id, toy_rubric.content_hash())
    assert (bound.shortfalls, dict(bound.writers), dict(bound.usage)) == ((), {}, {})
    encoder = StubEncoder()
    texts = encoder.embed(TEXTS, ["a", "b"])
    refs = embed_references(bound, encoder)
    scored = score_similarity(texts, refs, toy_rubric, MEAN)
    assert len(scored) == len(TEXTS) * 4 and refs.bound_with == "for_rubric"
    provenance = scoring_provenance(texts, refs, toy_rubric, [MEAN], len(scored))
    assert provenance["references"]["bound_with"] == "for_rubric"
    with pytest.raises(ConfigurationError) as refused:
        score_similarity(texts, refs, rewritten_toy_rubric, MEAN)
    assert str(refused.value) == refusal(toy_rubric, rewritten_toy_rubric)
    rebound = embed_references(
        ReferenceSet.for_rubric(frame, rewritten_toy_rubric, GeneratorSettings()), encoder
    )
    assert len(score_similarity(texts, rebound, rewritten_toy_rubric, MEAN)) == len(TEXTS) * 4


def test_the_binding_reaches_the_embedding_files_and_an_earlier_one_is_embedded_again(
    through_the_files: tuple[EmbeddingSet, ReferenceSet, ReferenceEmbeddings], tmp_path: Path
) -> None:
    _, _, refs = through_the_files
    _, sidecar_path = refs.save(tmp_path / "poles")
    description = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert description["bound_with"] == "generated"
    assert refs.subset_writers(refs.writers()).bound_with == "generated"
    del description["bound_with"]
    sidecar_path.write_text(json.dumps(description), encoding="utf-8")
    with pytest.raises(ConfigurationError) as refused:
        ReferenceEmbeddings.load(tmp_path / "poles")
    assert str(refused.value) == (
        f"reference embeddings {tmp_path / 'poles'}: poles.json has no bound_with: it was "
        f"written by an earlier version of copernicus than this one ({__version__}); embed the "
        "reference set again (embed_references, or copernicus embed references), which makes "
        "no writer call"
    )
    sidecar_path.write_text(json.dumps({**description, "bound_with": "guessed"}), encoding="utf-8")
    with pytest.raises(ConfigurationError, match=r"poles\.json: bound_with must be one of"):
        ReferenceEmbeddings.load(tmp_path / "poles")


@pytest.mark.parametrize("value", ["guessed", "", None])
def test_embeddings_need_a_binding_in_their_sidecar(
    through_the_files: tuple[EmbeddingSet, ReferenceSet, ReferenceEmbeddings], value: object
) -> None:
    _, _, refs = through_the_files
    bare = {key: entry for key, entry in refs.sidecar.items() if key != "bound_with"}
    with pytest.raises(ConfigurationError, match="lacks bound_with"):
        replace(refs, sidecar=bare)
    with pytest.raises(ConfigurationError, match="bound_with must be one of"):
        replace(refs, sidecar={**bare, "bound_with": value})
