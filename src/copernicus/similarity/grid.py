"""The paper's grid of similarity settings and its runner."""

from __future__ import annotations

import logging
from collections.abc import Collection, Mapping, Sequence
from itertools import product

import pandas as pd

from copernicus.embeddings import ENCODERS, EmbeddingSet
from copernicus.errors import ConfigurationError
from copernicus.rubric.model import Rubric
from copernicus.similarity.reduce import REDUCTIONS
from copernicus.similarity.references import ReferenceEmbeddings
from copernicus.similarity.scorer import score_similarity_many
from copernicus.similarity.settings import (
    LONG_TEXT,
    POLARITIES,
    TARGET_LEVELS,
    SimilaritySettings,
)

logger = logging.getLogger(__name__)

__all__ = ["PAPER_GRID", "PAPER_KEYS", "PAPER_WRITER_POOLINGS", "run_grid"]

MAX_IDS_IN_MESSAGE = 10
PAPER_WRITER_POOLINGS = ("mean", "max")
"""The writer poolings of the paper's grid; ``pooled`` is a setting of its own, outside it."""

PAPER_GRID: tuple[SimilaritySettings, ...] = tuple(
    SimilaritySettings(polarity, reduction, writer_pooling, long_text, target_level)
    for target_level, polarity, reduction, long_text, writer_pooling in product(
        TARGET_LEVELS, POLARITIES, REDUCTIONS, LONG_TEXT, PAPER_WRITER_POOLINGS
    )
)
"""The paper's settings: 3 polarities x 3 reductions x 4 long-text modes x 2 writer poolings
(``mean`` and ``max``; 72 settings) at each of the 2 target levels, 144 in all.

Run over the 8 keys of 4 encoders and 2 reference styles, this gives the paper's 576 cells
(8 x 72), each scored at both target levels.
"""

PAPER_KEYS: tuple[tuple[str, str], ...] = tuple(
    (encoder, style) for encoder in ENCODERS for style in ("neutral", "register")
)
"""The 8 ``(encoder, style)`` keys of the paper: the 4 registered encoders by the 2 reference
styles. Pass them as ``require_keys`` to `run_grid` to refuse a partial preset."""


def _check_key(
    key: tuple[str, str], texts: EmbeddingSet, refs: ReferenceEmbeddings, rubric: Rubric
) -> None:
    """Check that a key's encoder and style match its inputs and its references the rubric.

    Every refusal starts with the key, so a stale input of a sweep is named.
    """
    encoder, style = key
    for owner, name in ((texts.encoder, "embedding set"), (refs.encoder, "reference embeddings")):
        if owner != encoder:
            raise ConfigurationError(
                f"key {key}: the {name} was made with encoder {owner}, not {encoder}"
            )
    styles = sorted(set(refs.style.tolist()))
    if styles != [style]:
        raise ConfigurationError(f"key {key}: the reference embeddings have style {styles}")
    refs.assert_complete(rubric, label=f"key {key}")


def _check_same_texts(
    inputs: Mapping[tuple[str, str], tuple[EmbeddingSet, ReferenceEmbeddings]],
) -> None:
    """Check that the embedding sets of all keys hold the same text ids.

    Raises:
        ConfigurationError: Naming the first key whose ids differ from those of the first
            key, the number of ids that are in only one of the two, and the first ten of
            them in sorted order.
    """
    (first_key, (first_texts, _)), *others = inputs.items()
    expected = set(first_texts.ids.tolist())
    for key, (texts, _) in others:
        differing = sorted(expected.symmetric_difference(texts.ids.tolist()))
        if differing:
            shown = ", ".join(repr(text_id) for text_id in differing[:MAX_IDS_IN_MESSAGE])
            rest = len(differing) - MAX_IDS_IN_MESSAGE
            more = f" and {rest} more" if rest > 0 else ""
            raise ConfigurationError(
                f"the embedding sets of key {key} and key {first_key} cover different texts; "
                f"{len(differing)} ids are in only one of them: {shown}{more}"
            )


def run_grid(
    inputs: Mapping[tuple[str, str], tuple[EmbeddingSet, ReferenceEmbeddings]],
    rubric: Rubric,
    settings: Sequence[SimilaritySettings] = PAPER_GRID,
    *,
    require_keys: Collection[tuple[str, str]] | None = None,
) -> pd.DataFrame:
    """Score every key of ``inputs`` under every setting.

    Args:
        inputs: For each ``(encoder, style)`` key, the embedded texts and the reference
            embeddings made with that encoder and written in that style. A per-encoder
            sweep is one key. The embedding sets of all keys must hold the same texts.
        rubric: The rubric the references were written for; its content hash must equal
            the ``rubric_hash`` of the reference embeddings of every key.
        settings: The settings to run; the paper's grid by default.
        require_keys: Keys that must all have an input, for example `PAPER_KEYS` for
            the full preset; ``None`` accepts whatever keys are given.

    Returns:
        The scores of all keys and settings in one table; its ``encoder`` and ``style``
        columns identify the key.

    Raises:
        ConfigurationError: If there is no input, a required key has none, the encoder or
            style of a key disagrees with its embedding set or reference embeddings, the
            reference embeddings of a key belong to another rubric or to another content
            of it, or lack a pole (checked for every key before any key is scored; the
            message starts with the key), the embedding sets of the keys hold different
            text ids, or scoring fails.
    """
    if len(inputs) == 0:
        raise ConfigurationError("run_grid has no inputs")
    if require_keys is not None:
        absent = [key for key in require_keys if key not in inputs]
        if absent:
            raise ConfigurationError(f"run_grid lacks inputs for the required keys {absent}")
    for key, (texts, refs) in inputs.items():
        _check_key(key, texts, refs, rubric)
    _check_same_texts(inputs)
    frames = []
    for key, (texts, refs) in inputs.items():
        logger.info("scoring %d settings for encoder %s, style %s", len(settings), *key)
        frames.append(score_similarity_many(texts, refs, rubric, settings))
    return pd.concat(frames, ignore_index=True)
