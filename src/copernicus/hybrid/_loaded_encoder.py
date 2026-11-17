"""The hybrid coder's check before its embedding pass, with the library's loaded encoder."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from copernicus._load_check import compare_loads
from copernicus.embeddings import Encoder
from copernicus.embeddings._loading import check_embedding_request
from copernicus.similarity import ReferenceEmbeddings

logger = logging.getLogger(__name__)

__all__ = ["check_before_the_pass"]


def check_before_the_pass(
    encoder: Any,
    refs: ReferenceEmbeddings,
    texts: Sequence[str],
    ids: Sequence[Any],
    checkpoint_every: int,
) -> None:
    """Refuse an invalid request and references of another load before the embedding pass.

    The library's encoder is recognised by its class: an instance of
    `copernicus.embeddings.Encoder`, a subclass included. For it, the request of the pass is
    checked first with the checks that its embedding call makes before it loads
    (`copernicus.embeddings._loading.check_embedding_request`: no text, ids and texts in
    different numbers, ids that repeat, an empty text, ``checkpoint_every`` below 1), so a
    request that is wrong whatever the model is refused before any load, with the message the
    embedding call gives. The encoder is then loaded once, with the check of its
    ``transformers_below`` and the context of a failed load; an encoder that is already loaded
    returns its runtime. Its loaded runtime records the commit and the torch and transformers
    versions that its embedding pass writes into the sidecar of the set, so comparing its
    identity and that record with the references (`copernicus._load_check.compare_loads`)
    refuses a certain mismatch before any text is tokenised. The warnings of a smaller
    difference are not logged here: the scoring that follows the pass logs them, once. Any
    other encoder object is left to its own pass, which receives the request as given, and to
    the comparison of the embedding set that the pass returns; the versions installed in the
    process are not its record.

    Args:
        encoder: The encoder of the hybrid run: the library's `Encoder`, or an object of
            one's own with ``spec`` and ``embed``.
        refs: The reference embeddings of the run.
        texts: The texts the pass will embed.
        ids: Their ids, as the pass receives them.
        checkpoint_every: The checkpoint cadence the pass receives.

    Raises:
        ConfigurationError: If the request of the library's encoder holds no text, ids and
            texts in different numbers, ids that repeat or an empty text; if that encoder
            cannot be loaded; or if its identity and loaded runtime and the references record
            two loads (another identity key or loaded commit, transformers versions of two
            major versions or two epochs, or a version that cannot be read).
        SettingsError: If the request of the library's encoder has a ``checkpoint_every``
            below 1.
    """
    if not isinstance(encoder, Encoder):
        return
    check_embedding_request(texts, ids, checkpoint_every)
    runtime = encoder._load()
    compare_loads({**encoder.spec.identity(), **runtime.versions}, refs.sidecar)
