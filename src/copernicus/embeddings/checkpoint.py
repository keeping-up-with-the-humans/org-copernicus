"""A checkpoint of the finished fragments of an embedding pass, so a killed run resumes."""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from copernicus._io import write_atomically
from copernicus.embeddings.registry import EncoderSpec
from copernicus.errors import ConfigurationError

logger = logging.getLogger(__name__)

__all__ = ["checkpoint_file", "fragments_fingerprint", "read_checkpoint", "write_checkpoint"]


def checkpoint_file(checkpoint: Path) -> Path:
    """Return the file ``<checkpoint>.npz`` that holds the finished fragments."""
    return checkpoint.with_name(checkpoint.name + ".npz")


def fragments_fingerprint(
    spec: EncoderSpec, fragments: Sequence[str], *, resolved_revision: str
) -> str:
    """Hash the encoder settings, the loaded commit, and the fragment texts of a checkpoint.

    Args:
        spec: The encoder specification; its model id, revision, pooling, window, dtype,
            padding side, and remote-code switch enter the digest.
        fragments: The texts of all fragments, in order.
        resolved_revision: The commit the weights were loaded from (``"unknown"`` when the
            model names none), so a pass resumed after the default branch of the model
            repository moved is refused instead of mixing vectors of two commits.

    Returns:
        A hexadecimal digest; a checkpoint is only resumed under the same digest.
    """
    digest = hashlib.sha256()
    settings = (
        spec.hf_id,
        spec.revision,
        resolved_revision,
        spec.pooling,
        spec.max_tokens,
        spec.dtype,
        spec.padding_side,
        spec.trust_remote_code,
        len(fragments),
    )
    digest.update("|".join(str(value) for value in settings).encode("utf-8"))
    for fragment in fragments:
        digest.update(b"\x00")
        digest.update(fragment.encode("utf-8"))
    return digest.hexdigest()


def write_checkpoint(
    checkpoint: Path,
    fingerprint: str,
    finished: np.ndarray,
    raw: np.ndarray,
    weight: np.ndarray,
) -> None:
    """Write the finished fragments, replacing any earlier checkpoint atomically.

    A failed write leaves the earlier checkpoint as it was and no partial file behind.

    Args:
        checkpoint: Checkpoint path without the ``.npz`` suffix.
        fingerprint: Digest of the encoder settings, the loaded commit, and the fragments.
        finished: Indices of the finished fragments.
        raw: Raw pooled vectors of the finished fragments, in the order of ``finished``.
        weight: Token counts of the finished fragments, in the order of ``finished``.
    """
    target = checkpoint_file(checkpoint)
    write_atomically(
        target,
        lambda handle: np.savez(
            handle,
            fingerprint=np.asarray(fingerprint),
            finished=finished,
            raw=raw,
            weight=weight,
        ),
    )
    logger.info("checkpointed %d finished fragments to %s", len(finished), target)


def read_checkpoint(
    checkpoint: Path, fingerprint: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Read the finished fragments of an earlier run.

    Args:
        checkpoint: Checkpoint path without the ``.npz`` suffix.
        fingerprint: Digest of the current encoder settings, loaded commit, and fragments.

    Returns:
        The finished indices, raw vectors and token counts, or ``None`` when there is no
        checkpoint file.

    Raises:
        ConfigurationError: If the file was made for other texts, another encoder, or
            another commit of the encoder's model.
    """
    target = checkpoint_file(checkpoint)
    if not target.is_file():
        return None
    with np.load(target, allow_pickle=False) as arrays:
        if str(arrays["fingerprint"]) != fingerprint:
            raise ConfigurationError(
                f"checkpoint {target} belongs to other texts or another encoder (other settings, "
                "or another commit of the model); delete it or choose another path"
            )
        return arrays["finished"], arrays["raw"], arrays["weight"]
