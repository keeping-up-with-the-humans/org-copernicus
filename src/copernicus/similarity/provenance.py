"""Provenance of a scoring run: its inputs, their encoders, the rubric, and the settings."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

from copernicus._io import encoder_identity, encoder_versions
from copernicus._version import __version__
from copernicus.embeddings import EmbeddingSet
from copernicus.rubric.model import Rubric
from copernicus.similarity.references import ReferenceEmbeddings
from copernicus.similarity.settings import SimilaritySettings

logger = logging.getLogger(__name__)

__all__ = [
    "embedding_provenance",
    "library_stamp",
    "references_provenance",
    "rubric_provenance",
    "scoring_provenance",
]

SIMILARITY_CODER = "similarity"


def embedding_provenance(sidecar: Mapping[str, Any], encoder: str, owner: str) -> dict[str, Any]:
    """Describe the encoder behind a set of vectors.

    Args:
        sidecar: The description of the embedding set or the reference embeddings.
        encoder: Name of the encoder.
        owner: What the description belongs to, named when an identity key is missing.

    Returns:
        ``encoder``, the identity keys, and the version keys the sidecar holds.

    Raises:
        ConfigurationError: If the sidecar lacks an identity key.
    """
    return {"encoder": encoder, **encoder_identity(sidecar, owner), **encoder_versions(sidecar)}


def references_provenance(refs: ReferenceEmbeddings) -> dict[str, Any]:
    """Describe reference embeddings: their encoder, rubric, size, writers, and styles.

    Args:
        refs: The reference embeddings.

    Returns:
        The encoder description of `embedding_provenance`, ``rubric_id``, ``rubric_hash``
        (the content hash of the rubric the references were written for), ``bound_with``
        (``"generated"`` when the generator wrote them for that rubric, ``"for_rubric"`` when
        they were bound to it by an explicit act), ``n_references``, ``writers``,
        ``styles``, and ``prompt_versions`` when known.
    """
    description = embedding_provenance(refs.sidecar, refs.encoder, "reference embeddings")
    description.update(
        rubric_id=refs.rubric_id,
        rubric_hash=refs.rubric_hash,
        bound_with=refs.bound_with,
        n_references=len(refs),
        writers=refs.writers(),
        styles=sorted(set(refs.style.tolist())),
    )
    if "prompt_versions" in refs.sidecar:
        description["prompt_versions"] = list(refs.sidecar["prompt_versions"])
    return description


def rubric_provenance(rubric: Rubric) -> dict[str, str]:
    """Return the id and the content hash of a rubric."""
    return {"rubric_id": rubric.rubric_id, "rubric_hash": rubric.content_hash()}


def library_stamp() -> dict[str, str]:
    """Return the library version and the current time in UTC."""
    return {
        "library_version": __version__,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def scoring_provenance(
    texts: EmbeddingSet,
    refs: ReferenceEmbeddings,
    rubric: Rubric,
    settings: Sequence[SimilaritySettings],
    n_rows: int,
) -> dict[str, Any]:
    """Describe a run of the similarity coder, for the file written beside its scores.

    Args:
        texts: The embedded texts that were scored.
        refs: The reference embeddings they were scored against.
        rubric: The rubric of the references.
        settings: The settings scored, in order.
        n_rows: Rows of the score table.

    Returns:
        ``coder``, ``settings`` (one mapping per setting), ``texts`` (the encoder
        description and ``n_texts``), ``references`` (see `references_provenance`, with the
        rubric hash the references were written for), ``rubric_id`` and ``rubric_hash`` of
        ``rubric``, ``n_rows``, ``library_version``, and ``created_at``. Scoring refuses
        references whose rubric hash differs from the rubric's, so after a scoring run the
        two hashes are equal.
    """
    return {
        "coder": SIMILARITY_CODER,
        "settings": [asdict(each) for each in settings],
        "texts": {
            **embedding_provenance(texts.sidecar, texts.encoder, "embedding set"),
            "n_texts": len(texts),
        },
        "references": references_provenance(refs),
        **rubric_provenance(rubric),
        "n_rows": n_rows,
        **library_stamp(),
    }
