"""The embedding vectors of the texts of a correction, matched by text id.

The embedding sets of a ``CorrectionData`` are matched to its texts by id, both compared as
strings, so the order of a set does not matter and a set may hold further texts. A set that
lacks a text an operation needs, or whose vector of such a text holds a value that is not
finite, is refused with the first ten offending ids: a fit checks every text of the data, so
cross-fitting stops before its first fold, and a prediction checks the texts it predicts.
The encoder name ``"stacked"`` stands for every set of the mapping, in mapping order; any
other name must be a key of the mapping.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import pandas as pd

from copernicus.corrections._messages import list_first_ten
from copernicus.corrections.data import CorrectionData
from copernicus.errors import ConfigurationError, NotApplicableError

if TYPE_CHECKING:
    from copernicus.embeddings.sets import EmbeddingSet

logger = logging.getLogger(__name__)

__all__ = [
    "STACKED",
    "encoder_names",
    "encoder_problem",
    "gather_vectors",
    "training_vectors",
]

STACKED = "stacked"
MIN_STACKED_SETS = 2
NO_EMBEDDINGS = "embedding-augmented calibration needs embeddings; the data carry none"


def encoder_problem(data: CorrectionData, encoder: str) -> str:
    """Why the data cannot supply the embedding sets an encoder name stands for.

    Args:
        data: The correction data.
        encoder: The name of an embedding set, or ``"stacked"`` for every set.

    Returns:
        ``""`` when the data carry the sets, otherwise the reason: no embeddings (``None`` or
        an empty mapping), no set of that name (the message lists the sets), or fewer than
        two sets to stack.
    """
    if not data.embeddings:
        return NO_EMBEDDINGS
    names = list(data.embeddings)
    if encoder == STACKED:
        if len(names) < MIN_STACKED_SETS:
            return (
                "the stacked encoder set needs at least two embedding sets; the data carry "
                f"{len(names)}: {list_first_ten(names)}"
            )
        return ""
    if encoder not in data.embeddings:
        return f"the data carry no embedding set {encoder!r}; their sets: {list_first_ten(names)}"
    return ""


def encoder_names(data: CorrectionData, encoder: str, label: str) -> tuple[str, ...]:
    """The embedding sets an encoder name stands for, in the order of the design.

    Args:
        data: The correction data.
        encoder: The name of an embedding set, or ``"stacked"`` for every set.
        label: ``<family>/<variant>`` of the fit, named in the refusal.

    Returns:
        The set names: the one named set, or every set in mapping order.

    Raises:
        NotApplicableError: If the data carry no embeddings, no set of that name, or fewer
            than two sets to stack; the message gives the reason of `encoder_problem`, the
            one ``applies`` gives.
    """
    problem = encoder_problem(data, encoder)
    if problem:
        raise NotApplicableError(f"{label} does not apply: {problem}")
    return tuple(_sets(data)) if encoder == STACKED else (encoder,)


def _sets(data: CorrectionData) -> Mapping[str, EmbeddingSet]:
    """The data's embedding sets, refused when there are none."""
    if not data.embeddings:
        raise ConfigurationError(NO_EMBEDDINGS)
    return data.embeddings


def _embedding_set(data: CorrectionData, name: str) -> EmbeddingSet:
    """One embedding set of the data, refused when the data do not carry it."""
    sets = _sets(data)
    if name not in sets:
        raise ConfigurationError(
            f"the data carry no embedding set {name!r}, which the correction reads; their "
            f"sets: {list_first_ten(list(sets))}"
        )
    return sets[name]


def _set_positions(embedding_set: EmbeddingSet, name: str, ids: np.ndarray) -> np.ndarray:
    """Positions of the texts in an embedding set, refused when a text is missing."""
    wanted = np.asarray(ids).astype(str)
    positions = pd.Index(embedding_set.ids).get_indexer(wanted)
    missing = wanted[positions < 0]
    if missing.size:
        raise ConfigurationError(
            f"embedding set {name!r} lacks {missing.size} of the texts: "
            f"{list_first_ten(missing.tolist())}"
        )
    return positions


def _refuse_non_finite(name: str, finite_rows: npt.NDArray[np.bool_], ids: np.ndarray) -> None:
    """Refuse texts whose vector holds a NaN or an infinite value, naming the first ten."""
    offending = np.asarray(ids)[~finite_rows]
    if offending.size:
        raise ConfigurationError(
            f"embedding set {name!r} holds non-finite values for {offending.size} of the "
            f"texts: {list_first_ten(offending.tolist())}"
        )


def training_vectors(
    data: CorrectionData, names: Sequence[str], positions: np.ndarray
) -> list[np.ndarray]:
    """The vectors of the training rows from each set, once every text of the data is usable.

    Every text of the data, not only the training rows, must be in each set with a finite
    vector, so that a fit refuses data on which some prediction would not be defined.

    Args:
        data: The correction data.
        names: The embedding sets, in the order of the design.
        positions: Row positions of the training rows in the data's frame.

    Returns:
        One float64 block per set, rows in the order of ``positions``.

    Raises:
        ConfigurationError: If a set is absent, its vectors are not one row per text, it does
            not cover every text of the data, or it holds a non-finite value in the vector of
            one of them (the message names the set and the first ten offending ids).
    """
    blocks = []
    for name in names:
        embedding_set = _embedding_set(data, name)
        if embedding_set.vectors.ndim != 2:
            raise ConfigurationError(
                f"embedding set {name!r} has vectors of shape {embedding_set.vectors.shape}; "
                "a set holds one vector per text"
            )
        lookup = _set_positions(embedding_set, name, data.ids)
        finite_rows = np.isfinite(embedding_set.vectors).all(axis=1)
        _refuse_non_finite(name, finite_rows[lookup], data.ids)
        blocks.append(np.asarray(embedding_set.vectors[lookup[positions]], dtype=np.float64))
    return blocks


def gather_vectors(
    data: CorrectionData, names: Sequence[str], dims: Sequence[int], ids: np.ndarray
) -> list[np.ndarray]:
    """The vectors of the named texts from each set, checked against the fitted dimensions.

    Args:
        data: The correction data holding the embedding sets.
        names: The embedding sets of the fit, in the order of the design.
        dims: The dimension of each set at fit time.
        ids: Text ids of the rows.

    Returns:
        One float64 block per set, rows in the order of ``ids``.

    Raises:
        ConfigurationError: If the data carry no embeddings or not one of the sets, a set
            lacks one of the texts or holds a non-finite value in the vector of one of them
            (the message names the first ten), or a set's vectors have another dimension
            than at fit time.
    """
    wanted = np.asarray(ids).astype(str)
    blocks = []
    for name, dim in zip(names, dims, strict=True):
        embedding_set = _embedding_set(data, name)
        vectors = np.asarray(
            embedding_set.vectors[_set_positions(embedding_set, name, wanted)], dtype=np.float64
        )
        if vectors.ndim != 2 or vectors.shape[1] != dim:
            raise ConfigurationError(
                f"embedding set {name!r} has vectors of shape {embedding_set.vectors.shape}; "
                f"the correction was fitted on dimension {dim}"
            )
        _refuse_non_finite(name, np.asarray(np.isfinite(vectors).all(axis=1), dtype=bool), wanted)
        blocks.append(vectors)
    return blocks
