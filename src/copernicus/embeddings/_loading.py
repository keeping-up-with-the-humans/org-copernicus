"""What an encoder checks before it loads its tokenizer and model, and says when a load fails."""

from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any

from copernicus._versions import is_below, read_version, release_text
from copernicus.embeddings.registry import (
    TRANSFORMERS_LIMITS,
    TRANSFORMERS_MINIMUM,
    EncoderSpec,
)
from copernicus.errors import ConfigurationError, SettingsError

logger = logging.getLogger(__name__)

__all__ = [
    "check_embedding_request",
    "check_transformers_line",
    "loading",
    "revision_text",
    "transformers_remedy",
]

MAX_IDS_IN_MESSAGE = 10


def check_embedding_request(
    texts: Sequence[str], ids: Sequence[Any], checkpoint_every: int
) -> None:
    """Refuse a request to embed that is wrong whatever the model, before anything is loaded.

    `copernicus.embeddings.Encoder.embed` makes these checks before it loads its tokenizer and
    model, and the hybrid coder makes them before it loads the library's encoder to compare
    its runtime with the references, so a request that is wrong whatever the model is refused
    with the same message on both paths, and no model is loaded for it.

    Args:
        texts: The texts to embed.
        ids: One id per text.
        checkpoint_every: Chunks between two checkpoint writes.

    Raises:
        ConfigurationError: If there are no texts, ids and texts differ in number, ids repeat
            (compared as strings), or a text is empty or whitespace only (the first ten such
            ids are named); in this order.
        SettingsError: If ``checkpoint_every`` is below 1, after the checks of the texts.
    """
    if len(texts) == 0:
        raise ConfigurationError("there are no texts to embed")
    if len(ids) != len(texts):
        raise ConfigurationError(f"{len(ids)} ids for {len(texts)} texts")
    if len(set(map(str, ids))) != len(ids):
        raise ConfigurationError("ids must be unique")
    blank = [str(text_id) for text_id, text in zip(ids, texts, strict=True) if not text.strip()]
    if blank:
        shown = ", ".join(repr(text_id) for text_id in blank[:MAX_IDS_IN_MESSAGE])
        rest = len(blank) - MAX_IDS_IN_MESSAGE
        raise ConfigurationError(
            f"{len(blank)} texts are empty or whitespace only and cannot be embedded: "
            f"{shown}" + (f" and {rest} more" if rest > 0 else "")
        )
    if checkpoint_every < 1:
        raise SettingsError(f"checkpoint_every must be at least 1, got {checkpoint_every}")


def transformers_remedy(below: str) -> str:
    """Return the command that installs a transformers release of the extra below a bound.

    Args:
        below: The ``transformers_below`` of a specification, such as ``"5"``.

    Returns:
        For ``"5"``: ``pip install "transformers>=4.56,<5"``.
    """
    return f'pip install "transformers>={release_text(TRANSFORMERS_MINIMUM)},<{below}"'


def revision_text(spec: EncoderSpec) -> str:
    """Name the revision a specification loads, for a message: its pin or the default branch."""
    if spec.revision is None:
        return "the default branch, no revision pinned"
    return f"revision {spec.revision}"


def _reason(spec: EncoderSpec, below: str) -> str:
    """Say why the model code of the specification needs a transformers version below ``below``."""
    known = TRANSFORMERS_LIMITS.get((spec.hf_id, spec.revision))
    if known is not None:
        return known
    return (
        "its specification says, through transformers_below, that its own model code does not "
        f"load with transformers {below} or later"
    )


def check_transformers_line(spec: EncoderSpec, installed: str) -> None:
    """Refuse to load an encoder whose model code needs an earlier line of transformers.

    The check runs before anything is downloaded or loaded, so that the user learns the
    remedy before a long download and instead of an error inside the model's own code.

    Args:
        spec: The encoder specification; nothing is checked when its ``transformers_below``
            is ``None``.
        installed: The version of the installed transformers, ``transformers.__version__``.

    Raises:
        ConfigurationError: If the installed version, read whole as the package index reads
            it (`copernicus._versions.read_version`), is not below ``transformers_below`` (a
            pre-release of a line counts as that line, and a version of another epoch is not
            below it) or cannot be read as a version; the message names the encoder, its model
            id and revision, the installed version, the reason and the command that installs a
            release below the bound.
    """
    below = spec.transformers_below
    if below is None:
        return
    bound = read_version(below)
    version = read_version(installed)
    if bound is not None and version is not None and is_below(version, bound):
        return
    loaded = f"encoder {spec.name!r} ({spec.hf_id}, {revision_text(spec)})"
    remedy = f"install a release below {below} before loading it: {transformers_remedy(below)}"
    if version is None:
        raise ConfigurationError(
            f"{loaded} needs a transformers version below {below}, because "
            f"{_reason(spec, below)}, and the installed version {installed!r} cannot be read "
            f"as a version; {remedy}"
        )
    raise ConfigurationError(
        f"{loaded} cannot be loaded with transformers {installed}: {_reason(spec, below)}; {remedy}"
    )


@contextmanager
def loading(
    part: str, spec: EncoderSpec, transformers_version: str, torch_version: str
) -> Iterator[None]:
    """Say what was being loaded when the loader of a tokenizer or a model fails.

    The failure is not swallowed: the load still fails, with the library's error raised from
    the original one, which stays in the traceback. An interruption, such as
    ``KeyboardInterrupt``, is not an ``Exception`` and passes as it is.

    Args:
        part: What the block loads, ``"tokenizer"`` or ``"model"``.
        spec: The specification of the encoder being loaded.
        transformers_version: The version of the installed transformers.
        torch_version: The version of the installed torch.

    Yields:
        Nothing; the loader runs inside the block.

    Raises:
        ConfigurationError: If the loader raises an exception, inside transformers or inside
            the model repository's own code; the message names the encoder, its model id and
            revision, whether the repository's own code was allowed, the installed
            transformers and torch versions, and the type and text of the original error.
    """
    try:
        yield
    except Exception as error:
        allowed = "allowed" if spec.trust_remote_code else "not allowed"
        raise ConfigurationError(
            f"encoder {spec.name!r} could not load its {part} ({spec.hf_id}, "
            f"{revision_text(spec)}, the repository's own code {allowed}) with transformers "
            f"{transformers_version} and torch {torch_version}: {type(error).__name__}: {error}; "
            "check that the model and the revision exist, and are in the cache when the hub is "
            "offline, and that these versions of transformers and torch suit the model's code"
        ) from error
