"""Pole scores: cosines of texts to the references of a pole, reduced and pooled over writers."""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np

from copernicus.embeddings import EmbeddingSet
from copernicus.errors import ConfigurationError
from copernicus.similarity.reduce import reduce_matrix
from copernicus.similarity.references import ReferenceEmbeddings
from copernicus.similarity.settings import LONG_TEXT, SimilaritySettings

logger = logging.getLogger(__name__)

__all__ = [
    "PER_WRITER_POOLINGS",
    "POOLED",
    "PoleScorer",
    "per_reference_similarity",
    "pool_writers",
]

POSITIVE = "positive"
NEGATIVE = "negative"
POOLED = "pooled"
PER_WRITER_POOLINGS = ("mean", "max")


def per_reference_similarity(
    texts: EmbeddingSet, ref_vectors: np.ndarray, long_text: str
) -> np.ndarray:
    """Return the cosine of every text to every reference.

    Args:
        texts: The embedded texts.
        ref_vectors: Unit-length reference vectors, shape ``[K, D]``.
        long_text: ``"pooled_embedding"`` uses the vector of the whole text;
            ``"first_window"`` the first chunk; ``"mean_chunk"`` and ``"max_chunk"``
            compute the cosine of every chunk and take, per text and reference, the mean
            or the maximum over its chunks.

    Returns:
        Array of shape ``[N, K]`` in float64.

    Raises:
        ConfigurationError: If the dimensions differ, the mode is unknown, or a text has
            no finite chunk cosine.
    """
    if long_text not in LONG_TEXT:
        raise ConfigurationError(f"long_text is {long_text!r}; use one of {LONG_TEXT}")
    if texts.vectors.shape[1] != ref_vectors.shape[1]:
        raise ConfigurationError(
            f"texts have dimension {texts.vectors.shape[1]}, references {ref_vectors.shape[1]}"
        )
    references = ref_vectors.astype(np.float64).T
    if long_text == "pooled_embedding":
        return texts.vectors.astype(np.float64) @ references
    if long_text == "first_window":
        return texts.first_chunk_vectors().astype(np.float64) @ references
    chunk_cosines = texts.chunk_vectors.astype(np.float64) @ references
    n_texts, n_references = len(texts), references.shape[1]
    if long_text == "mean_chunk":
        total = np.zeros((n_texts, n_references))
        np.add.at(total, texts.chunk_owner, chunk_cosines)
        return total / np.bincount(texts.chunk_owner, minlength=n_texts)[:, None]
    best = np.full((n_texts, n_references), -np.inf)
    np.maximum.at(best, texts.chunk_owner, chunk_cosines)
    if not np.all(np.isfinite(best)):
        raise ConfigurationError("max_chunk produced a non-finite cosine; a text has no chunk")
    return best


def pool_writers(per_writer: Sequence[np.ndarray], mode: str) -> np.ndarray:
    """Pool the pole scores of the writer models, one score vector per writer.

    Args:
        per_writer: Score vectors of shape ``[N]``, one per writer model.
        mode: ``"mean"`` or ``"max"`` over the writer models. The writer pooling
            ``"pooled"`` has no per-writer scores to pool: it reduces a whole pole at once.

    Returns:
        Array of shape ``[N]``.

    Raises:
        ConfigurationError: If there is no writer or the mode is not ``"mean"`` or ``"max"``.
    """
    if mode not in PER_WRITER_POOLINGS:
        raise ConfigurationError(
            f"writer_pooling is {mode!r}; per-writer scores are pooled with one of "
            f"{PER_WRITER_POOLINGS}, while {POOLED!r} reduces each pole over all writers at once"
        )
    if len(per_writer) == 0:
        raise ConfigurationError("no writer model has scores to pool")
    stacked = np.column_stack(per_writer)
    return stacked.mean(axis=1) if mode == "mean" else stacked.max(axis=1)


class PoleScorer:
    """Scores one set of texts against the poles of one set of reference embeddings.

    The pole scores are kept per target, pole, reduction, writer pooling, and long-text
    mode, so scoring many settings that share them (the polarities, the target levels)
    repeats no work.
    """

    def __init__(self, texts: EmbeddingSet, refs: ReferenceEmbeddings, rubric_id: str) -> None:
        """Remember the texts and references; ``rubric_id`` names them in messages."""
        self._texts = texts
        self._refs = refs
        self._rubric_id = rubric_id
        self._poles: dict[tuple[str, str, str, str, str], tuple[np.ndarray, int]] = {}

    def _writer_cosines(self, rows: np.ndarray, writer: str, long_text: str) -> np.ndarray:
        """Return the cosines to the references of one writer among the rows of a pole."""
        mine = rows[self._refs.writer_model[rows] == writer]
        return per_reference_similarity(self._texts, self._refs.vectors[mine], long_text)

    def pole_score(
        self, target: str, polarity: str, settings: SimilaritySettings
    ) -> tuple[np.ndarray, int]:
        """Score the texts against one pole.

        Under the writer pooling ``"mean"`` or ``"max"`` the cosines to the references of
        each writer model present in the pole are reduced on their own and the per-writer
        scores are pooled. Under ``"pooled"`` the cosines to all references of the pole
        are reduced at once.

        Args:
            target: ``"construct"`` or an attribute id.
            polarity: ``"positive"`` or ``"negative"``.
            settings: Supplies the reduction, writer pooling, and long-text handling.

        Returns:
            The score per text, shape ``[N]``, and the number of references in the pole.

        Raises:
            ConfigurationError: If the pole has no reference, or a reduction fails; the
                message names the pole, the reduction, the long-text mode, and the writer
                (or all writers).
        """
        key = (target, polarity, settings.reduction, settings.writer_pooling, settings.long_text)
        if key not in self._poles:
            rows = self._refs.pole_indices(target, polarity)
            if rows.size == 0:
                raise ConfigurationError(
                    f"reference embeddings lack the {polarity} pole of target {target} "
                    f"for rubric {self._rubric_id}"
                )
            pole = (
                f"{self._rubric_id}:{target}:{polarity}:{settings.reduction}:{settings.long_text}"
            )
            if settings.writer_pooling == POOLED:
                cosines = per_reference_similarity(
                    self._texts, self._refs.vectors[rows], settings.long_text
                )
                score = reduce_matrix(cosines, settings.reduction, f"{pole}:all writers")
            else:
                per_writer = [
                    reduce_matrix(
                        self._writer_cosines(rows, writer, settings.long_text),
                        settings.reduction,
                        f"{pole}:writer {writer}",
                    )
                    for writer in sorted(set(self._refs.writer_model[rows].tolist()))
                ]
                score = pool_writers(per_writer, settings.writer_pooling)
            self._poles[key] = (score, int(rows.size))
        return self._poles[key]

    def polarity_score(
        self, target: str, polarity: str, settings: SimilaritySettings
    ) -> tuple[np.ndarray, int]:
        """Score the texts for one polarity mode: ``net``, ``positive``, or ``negative``.

        Returns:
            The score per text and the number of references behind it; the net score is
            the positive pole minus the negative pole and counts both poles.
        """
        if polarity != "net":
            return self.pole_score(target, polarity, settings)
        positive, n_positive = self.pole_score(target, POSITIVE, settings)
        negative, n_negative = self.pole_score(target, NEGATIVE, settings)
        return positive - negative, n_positive + n_negative
