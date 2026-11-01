"""The similarity coder: cosines of texts to the poles of reference embeddings."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from copernicus._io import check_same_encoder
from copernicus.embeddings import EmbeddingSet
from copernicus.errors import ConfigurationError
from copernicus.rubric.identifiers import ROLLUP_SUFFIX, TARGET_CONSTRUCT
from copernicus.rubric.model import Rubric
from copernicus.similarity.poles import PoleScorer
from copernicus.similarity.reduce import reduce_matrix
from copernicus.similarity.references import ReferenceEmbeddings
from copernicus.similarity.settings import SimilaritySettings

logger = logging.getLogger(__name__)

__all__ = [
    "OUTPUT_COLUMNS",
    "ROLLUP_SUFFIX",
    "rollup_scores",
    "score_similarity",
    "score_similarity_many",
]

POSITIVE = "positive"
NEGATIVE = "negative"
OUTPUT_COLUMNS = (
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
)


def rollup_scores(
    attribute_scores: Mapping[str, np.ndarray], rubric: Rubric, reduction: str
) -> np.ndarray:
    """Reduce the scores of the attributes, in rubric order, to one construct score.

    Args:
        attribute_scores: Score vector of shape ``[N]`` per attribute id.
        rubric: The rubric whose attributes are rolled up.
        reduction: ``"mean"``, ``"max"``, or ``"pca"``.

    Returns:
        Array of shape ``[N]``.

    Raises:
        ConfigurationError: If an attribute of the rubric has no scores.
    """
    attribute_ids = [attribute.attribute_id for attribute in rubric.attributes]
    missing = [name for name in attribute_ids if name not in attribute_scores]
    if missing:
        raise ConfigurationError(f"roll-up for rubric {rubric.rubric_id} lacks scores of {missing}")
    matrix = np.column_stack([attribute_scores[name] for name in attribute_ids])
    return reduce_matrix(matrix, reduction, f"{rubric.rubric_id}:rollup")


def _single_style(refs: ReferenceEmbeddings) -> str:
    """Return the one style of the references, or raise if they mix styles."""
    styles = sorted(set(refs.style.tolist()))
    if len(styles) != 1:
        raise ConfigurationError(f"reference embeddings must have one style, they have {styles}")
    return styles[0]


def _check_inputs(
    texts: EmbeddingSet, refs: ReferenceEmbeddings, rubric: Rubric, targets: Sequence[str] | None
) -> list[str]:
    """Check that texts, references, and rubric belong together; return the attributes to score.

    Raises:
        ConfigurationError: On a different encoder name, encoder identity (any identity
            key of the sidecars, or the loaded commits when both sidecars record one),
            dimension, rubric, or rubric content (the references record another rubric
            hash), targets given as one string, an unknown or empty target list, or a
            missing pole of a target that will be scored.
    """
    if isinstance(targets, str):
        raise ConfigurationError(
            f"targets is the string {targets!r}; give a list of attribute ids, for example "
            f"[{targets!r}]"
        )
    if texts.encoder != refs.encoder:
        raise ConfigurationError(
            f"texts were embedded with encoder {texts.encoder}, references with {refs.encoder}"
        )
    check_same_encoder(texts.sidecar, refs.sidecar, "texts", "reference embeddings")
    if texts.vectors.shape[1] != refs.vectors.shape[1]:
        raise ConfigurationError(
            f"texts have dimension {texts.vectors.shape[1]}, references {refs.vectors.shape[1]}"
        )
    attribute_ids = [attribute.attribute_id for attribute in rubric.attributes]
    if targets is None:
        refs.assert_complete(rubric)
        return attribute_ids
    if len(targets) == 0:
        raise ConfigurationError("targets must name at least one attribute, or be None")
    unknown = [name for name in targets if name not in attribute_ids]
    if unknown:
        raise ConfigurationError(
            f"unknown targets {unknown} for rubric {rubric.rubric_id}; use {attribute_ids}"
        )
    wanted = [name for name in attribute_ids if name in targets]
    refs.assert_complete(rubric, wanted)
    return wanted


def _warn_about_writer_asymmetry(refs: ReferenceEmbeddings, targets: Sequence[str]) -> None:
    """Warn about each target whose two poles were written by different writer models.

    Such a target is still scored: its net score then compares poles written by different
    writers, which mixes a difference between writers into the difference between poles.
    """
    for target in targets:
        writers = [
            sorted(set(refs.writer_model[(refs.target == target) & (refs.polarity == pole)]))
            for pole in (POSITIVE, NEGATIVE)
        ]
        if writers[0] != writers[1]:
            logger.warning(
                "target %s: the positive pole was written by %s, the negative pole by %s; "
                "the scores compare poles of different writers",
                target,
                [str(name) for name in writers[0]],
                [str(name) for name in writers[1]],
            )


def _score_with(
    scorer: PoleScorer,
    texts: EmbeddingSet,
    rubric: Rubric,
    settings: SimilaritySettings,
    attribute_ids: list[str],
    style: str,
    encoder: str,
    restricted: bool,
) -> pd.DataFrame:
    """Build the long-format scores of one setting from a pole scorer."""
    blocks: list[tuple[str, np.ndarray, int]] = []
    scored = [] if restricted or settings.target_level == "attribute" else [TARGET_CONSTRUCT]
    for target in [*scored, *attribute_ids]:
        score, n_references = scorer.polarity_score(target, settings.polarity, settings)
        blocks.append((target, score, n_references))
    if settings.target_level == "attribute" and not restricted:
        rolled = rollup_scores(
            {target: score for target, score, _ in blocks}, rubric, settings.reduction
        )
        blocks.append((f"{rubric.rubric_id}{ROLLUP_SUFFIX}", rolled, len(attribute_ids)))
    n_texts = len(texts)
    return pd.DataFrame(
        {
            "text_id": np.tile(texts.ids.astype(str), len(blocks)),
            "target_level": settings.target_level,
            "target_id": np.repeat([name for name, _, _ in blocks], n_texts),
            "polarity": settings.polarity,
            "reduction": settings.reduction,
            "writer_pooling": settings.writer_pooling,
            "long_text": settings.long_text,
            "style": style,
            "encoder": encoder,
            "score": np.concatenate([score for _, score, _ in blocks]),
            "n_references": np.repeat([count for _, _, count in blocks], n_texts),
        },
        columns=list(OUTPUT_COLUMNS),
    )


def score_similarity_many(
    texts: EmbeddingSet,
    refs: ReferenceEmbeddings,
    rubric: Rubric,
    settings_list: Sequence[SimilaritySettings],
    *,
    targets: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Score the texts under several settings and stack the results in the order given.

    The cosines of a pole are computed anew for every reduction, writer pooling, and
    long-text mode (once per writer under ``mean`` and ``max`` pooling), and the vectors are
    cast to float64 on every computation; the paper's grid computes the cosines of each
    pole 24 times (3 reductions x 2 writer poolings x 4 long-text modes, shared by the
    polarities and target levels). One computation costs about C x K x D
    multiply-adds and C x K x 8 bytes, for C chunks (texts under ``pooled_embedding`` and
    ``first_window``), K references in the pole, and dimension D: for 150,000 chunks,
    150 references, and 768 dimensions, some 17 billion multiply-adds and 180 MB.

    Args:
        texts: The embedded texts.
        refs: Reference embeddings made with the same encoder, of one style.
        rubric: The rubric the references were written for; its content hash must equal
            the ``rubric_hash`` of the references.
        settings_list: The settings to score under; at least one.
        targets: Attribute ids to score; ``None`` scores the construct and every
            attribute. With ids, only their rows are emitted: no construct row and no
            roll-up, and only their poles must be complete.

    Returns:
        Table with the columns ``text_id, target_level, target_id, polarity, reduction,
        writer_pooling, long_text, style, encoder, score, n_references``.

    Raises:
        ConfigurationError: If the inputs do not belong together (the references of
            another rubric, or of another content of this one, among them), a pole is
            missing, a reduction fails, or no settings are given.
    """
    if len(settings_list) == 0:
        raise ConfigurationError("score_similarity_many needs at least one settings object")
    attribute_ids = _check_inputs(texts, refs, rubric, targets)
    scored = attribute_ids if targets is not None else [TARGET_CONSTRUCT, *attribute_ids]
    _warn_about_writer_asymmetry(refs, scored)
    style = _single_style(refs)
    scorer = PoleScorer(texts, refs, rubric.rubric_id)
    frames = [
        _score_with(
            scorer, texts, rubric, settings, attribute_ids, style, refs.encoder, targets is not None
        )
        for settings in settings_list
    ]
    return pd.concat(frames, ignore_index=True)


def score_similarity(
    texts: EmbeddingSet,
    refs: ReferenceEmbeddings,
    rubric: Rubric,
    settings: SimilaritySettings,
    *,
    targets: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Score the texts against the poles of the reference embeddings.

    For each target and pole, the cosines of the texts to the references of each writer
    model are reduced (``settings.reduction``) and the writer models are pooled
    (``settings.writer_pooling`` ``mean`` or ``max``), or the cosines to all references of
    the pole are reduced at once (``pooled``). The polarity ``net`` is the positive pole
    minus the negative pole. A target whose two poles were written by different writer
    models is scored with a warning. At target level ``construct`` the construct is scored
    directly, next to the attributes; at target level ``attribute`` the attributes are
    scored and rolled up into the row ``<rubric_id>__rollup`` with the same reduction (for
    ``positive`` and ``negative`` the roll-up takes the scores of that polarity).

    Args:
        texts: The embedded texts.
        refs: Reference embeddings made with the same encoder, of one style.
        rubric: The rubric the references were written for; its content hash must equal
            the ``rubric_hash`` of the references.
        settings: The configuration to score under.
        targets: Attribute ids to score; ``None`` scores the construct and every
            attribute. With ids, only their rows are emitted: no construct row and no
            roll-up, and only their poles must be complete.

    Returns:
        Table with one row per text, target, and requested polarity, and the columns
        ``text_id, target_level, target_id, polarity, reduction, writer_pooling, long_text,
        style, encoder, score, n_references``; ``n_references`` is the size of the pole (both
        poles for ``net``, the number of attributes for the roll-up).

    Raises:
        ConfigurationError: If texts and references come from different encoders (by name
            or by any identity key of their sidecars: model id, revision, pooling, window,
            dtype, remote-code switch, padding side, reserved special tokens), from
            different commits of the model (when both sidecars record one other than
            ``"unknown"``; a commit changes even with the documentation alone, so re-embed
            or pin), or have different dimensions, the references are for another rubric
            or for another content of it (their rubric hash differs from the rubric's,
            as after an edit under the same ids) or mix styles, a pole is missing, a
            target is unknown, or a reduction fails.
    """
    return score_similarity_many(texts, refs, rubric, [settings], targets=targets)
