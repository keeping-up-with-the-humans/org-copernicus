"""The registry of encoders and the specification that describes one encoder."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal

from copernicus._commits import COMMIT_HASH_LENGTH, is_commit_hash
from copernicus.errors import SettingsError

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_ENCODER",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_POOLING",
    "ENCODERS",
    "POOLING_NAMES",
    "PADDING_SIDES",
    "EncoderSpec",
    "resolve_encoder",
]

POOLING_NAMES = ("mean", "cls", "last_token")
PADDING_SIDES = ("left", "right")
DEFAULT_POOLING = "mean"
DEFAULT_MAX_TOKENS = 512
DEFAULT_SPECIAL_TOKENS_RESERVED = 2
DEFAULT_ENCODER = "mpnet"
"""The name of the encoder that the commands and ``Encoder`` use when none is given."""
UNPINNED = "unpinned"


@dataclass(frozen=True)
class EncoderSpec:
    """Everything that determines the vectors an encoder produces.

    Attributes:
        name: Short name; the registry key, or the Hugging Face id for an ad-hoc encoder.
        hf_id: Hugging Face model id (or a local path) the weights are loaded from.
        pooling: How token vectors become one vector: ``"mean"`` over the attended
            tokens, ``"cls"`` for the first token, ``"last_token"`` for the last
            attended token.
        max_tokens: Window of the encoder in tokens; longer texts are split into
            consecutive windows.
        dtype: Name of the torch dtype to load the weights in, for example
            ``"bfloat16"``; ``None`` keeps the model's default.
        trust_remote_code: Whether the model's own code may be executed on loading.
        padding_side: Side on which batches are padded; both sides are handled by
            every pooling function.
        special_tokens_reserved: Tokens the tokenizer adds around every window; they
            are subtracted from ``max_tokens`` when a text is split.
        revision: Commit, branch, or tag of the model repository to load, for an encoder
            that runs no remote code; ``None`` loads the default branch, whose commit is
            recorded in the sidecar of every embedding set as ``resolved_revision``. Texts
            and references are scored together only when this setting agrees and, where
            both sidecars record a commit, the commits agree. A repository gets a new commit
            even when only its documentation changes, so sets embedded from the default
            branch before and after such a commit must be re-embedded, or embedded with one
            pinned revision. Pinning a commit matters most for an encoder that runs remote
            code, so a specification that allows remote code needs the full 40-character
            commit hash here: a branch, a tag or a shortened hash would let other code run
            once the repository changes. The pin covers only the code in the model's own
            repository: an ``auto_map`` that names a class in a second repository
            (``repo--module.Class``) loads that repository's default branch, so review such a
            configuration before allowing remote code. The registry pins ``gte_qwen2_7b`` to
            the commit ``a8d08b36ada9cacfe34c4d6f80957772a025daf2``.
    """

    name: str
    hf_id: str
    pooling: Literal["mean", "cls", "last_token"]
    max_tokens: int
    dtype: str | None = None
    trust_remote_code: bool = False
    padding_side: Literal["left", "right"] = "right"
    special_tokens_reserved: int = DEFAULT_SPECIAL_TOKENS_RESERVED
    revision: str | None = None

    def __post_init__(self) -> None:
        """Refuse a specification that cannot describe an encoder.

        Raises:
            SettingsError: If the pooling or padding side is unknown, the window leaves
                no room for text after the reserved special tokens, or the model
                repository's own code may run (``trust_remote_code``) but the revision is
                not a full 40-character commit hash (none, a blank value, a branch, a tag
                or a shortened hash).
        """
        if self.pooling not in POOLING_NAMES:
            raise SettingsError(
                f"unknown pooling {self.pooling!r} for encoder {self.name!r}; "
                f"choose one of {', '.join(POOLING_NAMES)}"
            )
        if self.padding_side not in PADDING_SIDES:
            raise SettingsError(
                f"unknown padding side {self.padding_side!r} for encoder {self.name!r}; "
                f"choose one of {', '.join(PADDING_SIDES)}"
            )
        if self.max_tokens - self.special_tokens_reserved <= 0:
            raise SettingsError(
                f"encoder {self.name!r}: max_tokens {self.max_tokens} leaves no room for text "
                f"after {self.special_tokens_reserved} reserved special tokens"
            )
        if self.trust_remote_code and not is_commit_hash(self.revision):
            raise SettingsError(
                f"encoder {self.name!r} allows the model repository's own code to run "
                "(trust_remote_code) but pins no commit: set revision to the "
                f"{COMMIT_HASH_LENGTH}-character commit hash whose code you have reviewed"
            )

    def identity(self) -> dict[str, Any]:
        """Return the settings that decide the vectors, as the sidecars record them.

        Returns:
            ``hf_id``, ``revision`` (``"unpinned"`` when none is set), ``pooling``,
            ``max_tokens``, ``dtype``, ``trust_remote_code``, ``padding_side``, and
            ``special_tokens_reserved``; texts and references are scored together only when
            all of them agree.
        """
        return {
            "hf_id": self.hf_id,
            "revision": self.revision if self.revision is not None else UNPINNED,
            "pooling": self.pooling,
            "max_tokens": self.max_tokens,
            "dtype": self.dtype,
            "trust_remote_code": self.trust_remote_code,
            "padding_side": self.padding_side,
            "special_tokens_reserved": self.special_tokens_reserved,
        }


ENCODERS: Mapping[str, EncoderSpec] = MappingProxyType(
    {
        "minilm": EncoderSpec("minilm", "sentence-transformers/all-MiniLM-L6-v2", "mean", 256),
        "mpnet": EncoderSpec("mpnet", "sentence-transformers/all-mpnet-base-v2", "mean", 384),
        "bge_m3": EncoderSpec("bge_m3", "BAAI/bge-m3", "cls", 8192),
        "gte_qwen2_7b": EncoderSpec(
            "gte_qwen2_7b",
            "Alibaba-NLP/gte-Qwen2-7B-instruct",
            "last_token",
            8192,
            dtype="bfloat16",
            trust_remote_code=True,
            revision="a8d08b36ada9cacfe34c4d6f80957772a025daf2",
        ),
    }
)
"""The registered encoders by name, as read-only specifications.

The names are ``minilm``, ``mpnet``, ``bge_m3`` and ``gte_qwen2_7b``. Only
``gte_qwen2_7b`` runs the model repository's own code, and it is pinned to one commit.
"""


def _check_fixed(spec: EncoderSpec, pooling: str | None, max_tokens: int | None) -> EncoderSpec:
    """Return a registered or given specification, refusing options that contradict it."""
    contradicting = (pooling is not None and pooling != spec.pooling) or (
        max_tokens is not None and max_tokens != spec.max_tokens
    )
    if contradicting:
        raise SettingsError(
            f"encoder {spec.name!r} has {spec.pooling} pooling and a window of "
            f"{spec.max_tokens} tokens; pooling and max_tokens choose them only for a Hugging "
            "Face id that is not in the registry (build an EncoderSpec to change them)"
        )
    return spec


def resolve_encoder(
    name_or_spec: str | EncoderSpec,
    *,
    pooling: str | None = None,
    max_tokens: int | None = None,
) -> EncoderSpec:
    """Return the specification of an encoder.

    Args:
        name_or_spec: A registry name, a specification (returned unchanged), or any
            other string, which is taken as a Hugging Face model id.
        pooling: Pooling of an encoder that is not in the registry; ``mean`` when
            ``None``. For a registered or given specification it may only repeat its
            pooling.
        max_tokens: Window of an encoder that is not in the registry; 512 when ``None``.
            For a registered or given specification it may only repeat its window.

    Returns:
        The registered or given specification, or a new one named by the string, with
        the model id equal to the string and the given (or default) pooling and window.
        An ad-hoc id that gets a default pooling or window is logged as a warning, because
        the right values depend on the model.

    Raises:
        SettingsError: If ``pooling`` or ``max_tokens`` contradicts a registered or given
            specification, or the pooling of an ad-hoc encoder is not one of ``mean``,
            ``cls``, ``last_token``, or its window is too small.
    """
    if isinstance(name_or_spec, EncoderSpec):
        return _check_fixed(name_or_spec, pooling, max_tokens)
    if name_or_spec in ENCODERS:
        return _check_fixed(ENCODERS[name_or_spec], pooling, max_tokens)
    chosen_pooling = pooling if pooling is not None else DEFAULT_POOLING
    chosen_window = max_tokens if max_tokens is not None else DEFAULT_MAX_TOKENS
    if pooling is None or max_tokens is None:
        logger.warning(
            "encoder %r is not in the registry; it is loaded as a Hugging Face id with %s "
            "pooling and a window of %d tokens; check both against the model card and set "
            "pooling and max_tokens",
            name_or_spec,
            chosen_pooling,
            chosen_window,
        )
    return EncoderSpec(
        name_or_spec,
        name_or_spec,
        chosen_pooling,  # type: ignore[arg-type]
        chosen_window,
    )
