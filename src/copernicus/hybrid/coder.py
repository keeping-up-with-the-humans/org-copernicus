"""The hybrid coder: evidence texts embedded and scored against the reference poles."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from copernicus._io import check_same_encoder
from copernicus.coders import export_provenance
from copernicus.data import TextTable
from copernicus.embeddings import EmbeddingSet, Encoder
from copernicus.embeddings.encoder import CHECKPOINT_EVERY
from copernicus.errors import ConfigurationError, SettingsError
from copernicus.hybrid.texts import GRANULARITIES, HybridTexts, evidence_texts
from copernicus.rubric.identifiers import ROLLUP_SUFFIX
from copernicus.rubric.model import Rubric
from copernicus.similarity import (
    ReferenceEmbeddings,
    SimilaritySettings,
    rollup_scores,
    score_similarity,
)
from copernicus.similarity.provenance import (
    embedding_provenance,
    library_stamp,
    references_provenance,
    rubric_provenance,
)
from copernicus.similarity.scorer import OUTPUT_COLUMNS

logger = logging.getLogger(__name__)

__all__ = ["HYBRID_DEFAULTS", "HybridResult", "score_hybrid"]

METHOD_CONSTRUCT = "hybrid_construct"
METHOD_ATTRIBUTE = "hybrid_attribute"
HYBRID_CODER = "hybrid"

HYBRID_DEFAULTS = SimilaritySettings(
    polarity="net",
    reduction="pca",
    writer_pooling="pooled",
    long_text="pooled_embedding",
    target_level="construct",
)
"""The default settings of the hybrid coder, which reproduce the paper's hybrid coder: the
vector of each whole evidence text, and each pole reduced once over the references of all
writers. The similarity coder's own defaults, ``SimilaritySettings()``, reproduce the paper's
similarity configuration instead."""


@dataclass(frozen=True, eq=False)
class HybridResult:
    """The scores of the hybrid coder and what went into them.

    Attributes:
        frame: The long table of `copernicus.similarity.score_similarity` plus
            ``method`` (``"hybrid_construct"`` or ``"hybrid_attribute"``) and
            ``fallback`` (``""``, ``"short"``, ``"no_evidence"`` of the text, or at the
            roll-up the distinct non-empty fallbacks of its attributes joined by commas).
        summary: At construct granularity the summary of the evidence texts (``n_texts``,
            ``n_short``, ``n_no_evidence``, ``n_evidence``, ``mean_words_evidence``,
            ``n_passages_embedded``). At attribute granularity ``n_texts``,
            ``n_attributes``, ``n_short`` (the same for every attribute),
            ``n_no_evidence_cells`` and ``n_evidence_cells`` (summed over attributes),
            ``n_passages_embedded``, ``n_distinct_texts_embedded`` (strings that went
            through the encoder), and ``by_attribute`` (the summary of each attribute).
        provenance: What the run used: ``coder`` (``"hybrid"``), ``settings``,
            ``granularity``, ``run_number``, ``min_words``, ``texts`` (the encoder of the
            evidence texts with its identity and version keys, and ``n_texts``),
            ``references`` (with the rubric hash the references were written for),
            ``rubric_id``, ``rubric_hash``, ``store`` (its path and run information),
            ``n_rows``, ``library_version``, and ``created_at``.
    """

    frame: pd.DataFrame
    summary: dict[str, Any]
    provenance: dict[str, Any]


def _gather(unique: EmbeddingSet, rows: np.ndarray, ids: list[str]) -> EmbeddingSet:
    """Return the embeddings of the rows of ``unique``, repeated as often as asked, under new ids.

    Args:
        unique: Embeddings of distinct texts.
        rows: For each wanted text, the row of its string in ``unique``.
        ids: The ids of the wanted texts; unique.
    """
    by_owner = np.argsort(unique.chunk_owner, kind="stable")
    counts = np.bincount(unique.chunk_owner, minlength=len(unique))
    starts = np.cumsum(counts) - counts
    chunk_rows = np.concatenate([by_owner[starts[row] : starts[row] + counts[row]] for row in rows])
    owner = np.repeat(np.arange(len(rows), dtype=np.int64), counts[rows])
    return replace(
        unique,
        ids=np.asarray(ids),
        vectors=unique.vectors[rows],
        chunk_vectors=unique.chunk_vectors[chunk_rows],
        chunk_owner=owner,
        chunk_weight=unique.chunk_weight[chunk_rows],
        metadata=None,
        sidecar={**unique.sidecar, "n_texts": len(rows), "n_chunks": len(chunk_rows)},
    )


Scored = tuple[pd.DataFrame, dict[str, Any], Mapping[str, Any]]


@dataclass(frozen=True)
class _Checkpoint:
    """Where and how often the one embedding pass of a hybrid run saves its progress."""

    path: Path | None
    every: int


def _score_construct(
    texts: HybridTexts,
    encoder: Encoder,
    refs: ReferenceEmbeddings,
    rubric: Rubric,
    settings: SimilaritySettings,
    checkpoint: _Checkpoint,
) -> Scored:
    """Embed the joined evidence once and score it with the similarity coder.

    Returns:
        The scores, the summary of the evidence texts, and the sidecar of their embeddings.
    """
    embedded = encoder.embed(
        texts.texts, texts.ids, checkpoint=checkpoint.path, checkpoint_every=checkpoint.every
    )
    frame = score_similarity(embedded, refs, rubric, settings)
    frame["method"] = METHOD_CONSTRUCT
    frame["fallback"] = frame["text_id"].map(dict(zip(texts.ids, texts.fallback, strict=True)))
    return frame, dict(texts.summary), embedded.sidecar


def _rollup_frame(
    attribute_frames: list[pd.DataFrame],
    names: list[str],
    rubric: Rubric,
    settings: SimilaritySettings,
    fallbacks: list[list[str]],
) -> pd.DataFrame:
    """Roll the attribute scores up into one construct row per text.

    Args:
        attribute_frames: The scores of each attribute, in rubric order, over the same texts.
        names: The attribute ids, in rubric order.
        rubric: The rubric the references were written for.
        settings: Supplies the reduction of the roll-up and the labels of its rows.
        fallbacks: For each attribute, the fallback of each text.
    """
    first = attribute_frames[0]
    for frame in attribute_frames:
        if not frame["text_id"].equals(first["text_id"]):
            raise ConfigurationError("the attribute scores do not cover the texts in one order")
    scores = {
        name: frame["score"].to_numpy() for name, frame in zip(names, attribute_frames, strict=True)
    }
    rolled = rollup_scores(scores, rubric, settings.reduction)
    rollup = pd.DataFrame(
        {
            "text_id": first["text_id"].to_numpy(),
            "target_level": "attribute",
            "target_id": f"{rubric.rubric_id}{ROLLUP_SUFFIX}",
            "polarity": settings.polarity,
            "reduction": settings.reduction,
            "writer_pooling": settings.writer_pooling,
            "long_text": settings.long_text,
            "style": first["style"].iloc[0],
            "encoder": first["encoder"].iloc[0],
            "score": rolled,
            "n_references": len(names),
        },
        columns=list(OUTPUT_COLUMNS),
    )
    rollup["fallback"] = [
        ",".join(dict.fromkeys(label for label in labels if label != ""))
        for labels in zip(*fallbacks, strict=True)
    ]
    return rollup


def _score_attribute(
    by_attribute: dict[str, HybridTexts],
    encoder: Encoder,
    refs: ReferenceEmbeddings,
    rubric: Rubric,
    settings: SimilaritySettings,
    checkpoint: _Checkpoint,
) -> Scored:
    """Embed every distinct attribute text once, score each attribute on its own poles, roll up.

    Returns:
        The scores, the summary of the evidence texts, and the sidecar of their embeddings.
    """
    names = list(by_attribute)
    attribute_settings = replace(settings, target_level="attribute")
    distinct: dict[str, int] = {}
    for texts in by_attribute.values():
        for text in texts.texts:
            distinct.setdefault(text, len(distinct))
    unique = encoder.embed(
        list(distinct),
        [str(row) for row in range(len(distinct))],
        checkpoint=checkpoint.path,
        checkpoint_every=checkpoint.every,
    )
    frames: list[pd.DataFrame] = []
    for name, texts in by_attribute.items():
        rows = np.asarray([distinct[text] for text in texts.texts], dtype=np.int64)
        embedded = _gather(unique, rows, texts.ids)
        frame = score_similarity(embedded, refs, rubric, attribute_settings, targets=[name])
        frame["fallback"] = frame["text_id"].map(dict(zip(texts.ids, texts.fallback, strict=True)))
        frames.append(frame)
    rollup = _rollup_frame(
        frames, names, rubric, settings, [texts.fallback for texts in by_attribute.values()]
    )
    frame = pd.concat([*frames, rollup], ignore_index=True)
    frame["method"] = METHOD_ATTRIBUTE
    summaries = {name: dict(texts.summary) for name, texts in by_attribute.items()}
    # The short fallback depends only on the text, so every attribute counts the same texts.
    short_counts = {each["n_short"] for each in summaries.values()}
    if len(short_counts) != 1:
        raise ConfigurationError(
            f"the attributes count different numbers of short texts: {summaries}"
        )
    n_short = short_counts.pop()
    summary: dict[str, Any] = {
        "n_texts": summaries[names[0]]["n_texts"],
        "n_attributes": len(names),
        "n_short": n_short,
        "n_no_evidence_cells": sum(each["n_no_evidence"] for each in summaries.values()),
        "n_evidence_cells": sum(each["n_evidence"] for each in summaries.values()),
        "n_passages_embedded": sum(each["n_passages_embedded"] for each in summaries.values()),
        "n_distinct_texts_embedded": len(distinct),
        "by_attribute": summaries,
    }
    return frame[[*OUTPUT_COLUMNS, "method", "fallback"]], summary, unique.sidecar


def score_hybrid(
    store_path: Path,
    table: TextTable,
    rubric: Rubric,
    encoder: Encoder,
    refs: ReferenceEmbeddings,
    settings: SimilaritySettings = HYBRID_DEFAULTS,
    *,
    granularity: str = "construct",
    run_number: int = 1,
    min_words: int = 40,
    checkpoint: Path | None = None,
    checkpoint_every: int = CHECKPOINT_EVERY,
) -> HybridResult:
    """Score the texts through the verified passages of a two-stage run.

    The default settings, `HYBRID_DEFAULTS`, reproduce the paper's hybrid coder: the
    vector of each whole evidence text, scored against each pole reduced once over the
    references of all writers (net polarity, first principal component, construct level).
    The similarity coder's own defaults reproduce the paper's similarity configuration;
    pass them, or any other settings, to score the evidence texts that way.

    At ``"construct"`` granularity the evidence text of each text is embedded once and
    scored like any text by `copernicus.similarity.score_similarity`, so
    ``settings.target_level`` decides whether the construct poles or the attribute poles
    with their roll-up are used. At ``"attribute"`` granularity the evidence text of each
    attribute is embedded (a string shared by several attributes, such as a raw text, goes
    through the encoder once), each attribute is scored only against its own poles, and the
    attribute scores are rolled up into ``<rubric_id>__rollup`` with the reduction of
    ``settings``, for the polarity of ``settings``. Every row of the result then has
    ``target_level`` ``"attribute"`` whatever ``settings.target_level`` says.

    Args:
        store_path: Run store of a two-stage run.
        table: The texts, as they were coded: the text hash of each must equal the one
            stored with its results in ``run_number``, which is checked before anything is
            embedded.
        rubric: The rubric of the run and of the references: its content hash must equal
            both the one stored with the run and the ``rubric_hash`` of ``refs``.
        encoder: The encoder; its name and the identity of its specification must equal
            those of the encoder of ``refs``.
        refs: Reference embeddings made with the same encoder.
        settings: The similarity configuration; `HYBRID_DEFAULTS` by default.
        granularity: ``"construct"`` or ``"attribute"``.
        run_number: The run whose passages are read.
        min_words: Texts with fewer words are embedded as they are.
        checkpoint: Checkpoint path (without suffix) of the run's one embedding pass, which
            makes a killed run resumable; see `copernicus.embeddings.Encoder.embed`.
        checkpoint_every: Chunks between two checkpoint writes.

    Returns:
        The scores, the summary of the evidence texts, and the provenance of the run.

    Raises:
        SettingsError: If the granularity, ``run_number`` or ``min_words`` is out of range.
        ConfigurationError: If the encoder differs from the one of the references by name
            or by an identity key, including the requested revision (checked before
            anything is embedded), or was loaded from another commit than the references
            (known, and checked, only once the texts are embedded), the references were
            written for another rubric or for another content of it, or lack a pole the
            granularity scores (every target at ``"construct"``, every attribute at
            ``"attribute"``; checked before the store is read), the store does not fit the
            rubric and the table, a text of the table differs from the text the store coded
            under its id, or a text with at least ``min_words`` words has an attribute whose
            extraction failed in the run (see `copernicus.hybrid.evidence_texts`; both
            checked before anything is embedded), or scoring fails.
        EvidenceContractError: If a verified passage is malformed.
    """
    if granularity not in GRANULARITIES:
        raise SettingsError(f"granularity is {granularity!r}; use one of {GRANULARITIES}")
    if encoder.spec.name != refs.encoder:
        raise ConfigurationError(
            f"the encoder is {encoder.spec.name!r}, the reference embeddings were made with "
            f"{refs.encoder!r}"
        )
    # The specification records no loaded commit, so this compares the requested revision;
    # the scorer compares the loaded commits once the evidence texts are embedded.
    check_same_encoder(encoder.spec.identity(), refs.sidecar, "encoder", "reference embeddings")
    # Attribute granularity scores each attribute on its own poles; it needs no construct pole.
    attribute_ids = [attribute.attribute_id for attribute in rubric.attributes]
    refs.assert_complete(rubric, None if granularity == "construct" else attribute_ids)
    saving = _Checkpoint(checkpoint, checkpoint_every)
    if granularity == "construct":
        texts = evidence_texts(
            store_path,
            rubric,
            table,
            granularity="construct",
            run_number=run_number,
            min_words=min_words,
        )
        frame, summary, sidecar = _score_construct(texts, encoder, refs, rubric, settings, saving)
    else:
        by_attribute = evidence_texts(
            store_path,
            rubric,
            table,
            granularity="attribute",
            run_number=run_number,
            min_words=min_words,
        )
        frame, summary, sidecar = _score_attribute(
            by_attribute, encoder, refs, rubric, settings, saving
        )
    provenance = {
        "coder": HYBRID_CODER,
        "settings": asdict(settings),
        "granularity": granularity,
        "run_number": run_number,
        "min_words": min_words,
        "texts": {
            **embedding_provenance(sidecar, encoder.spec.name, "evidence texts"),
            "n_texts": len(table),
        },
        "references": references_provenance(refs),
        **rubric_provenance(rubric),
        "store": {"path": str(store_path), **export_provenance(store_path)},
        "n_rows": len(frame),
        **library_stamp(),
    }
    return HybridResult(frame=frame, summary=summary, provenance=provenance)
