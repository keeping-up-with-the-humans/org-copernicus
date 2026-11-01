"""Backend protocol and the request and result types every LLM coder uses."""

from __future__ import annotations

import dataclasses
import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

logger = logging.getLogger(__name__)

__all__ = [
    "TOP_P",
    "Backend",
    "BackendInfo",
    "GenerationRequest",
    "GenerationResult",
    "ModelCapabilities",
    "capabilities_for_model",
    "capability_provenance",
    "provenance_text",
]

TOP_P = 0.9
"""The nucleus-sampling threshold that a backend sends with a request when the model accepts it."""
RESTRICTED_MODEL_PREFIXES = ("gpt-5", "o1", "o3", "o4")
MAX_TOKENS_PARAMETER = "max_tokens"
MAX_COMPLETION_TOKENS_PARAMETER = "max_completion_tokens"
NONE_TEXT = "none"


@dataclass(frozen=True)
class GenerationRequest:
    """One generation call to a backend.

    Attributes:
        system: System prompt.
        user: User message.
        max_new_tokens: Upper bound on the number of generated tokens.
        temperature: Sampling temperature; ``None`` requests greedy decoding.
        seed: Sampling seed; ``None`` leaves the backend unseeded.
    """

    system: str
    user: str
    max_new_tokens: int
    temperature: float | None
    seed: int | None


@dataclass(frozen=True)
class GenerationResult:
    """The outcome of one generation call.

    Attributes:
        text: Generated text.
        prompt_tokens: Number of prompt tokens, if the backend reports it.
        completion_tokens: Number of generated tokens, if the backend reports it.
        temperature_applied: Temperature the backend actually used, if any.
        top_p_applied: Nucleus-sampling threshold the backend actually used, if any.
        seed_applied: Seed the backend actually used, if any.
    """

    text: str
    prompt_tokens: int | None
    completion_tokens: int | None
    temperature_applied: float | None
    top_p_applied: float | None
    seed_applied: int | None


@dataclass(frozen=True)
class BackendInfo:
    """Description of a backend instance.

    The runner stores ``model_name`` in the identity of a run; the settings that decide
    what the backend generates are stored from `Backend.provenance`.

    Attributes:
        model_name: Name of the model served.
        backend: Kind of backend, for example ``"fake"``.
        quantization: Quantization scheme of the weights, if known.
        context_window: Context window in tokens, if known.
    """

    model_name: str
    backend: str
    quantization: str | None
    context_window: int | None


@dataclass(frozen=True)
class ModelCapabilities:
    """Which sampling parameters a model accepts.

    Attributes:
        supports_temperature: Whether the model accepts a temperature.
        supports_top_p: Whether the model accepts a nucleus-sampling threshold.
        supports_seed: Whether the model accepts a seed.
        max_tokens_parameter: Name of the request field that bounds the output length.
        reasoning_effort: Reasoning effort to request, if the model takes one.
    """

    supports_temperature: bool = True
    supports_top_p: bool = True
    supports_seed: bool = True
    max_tokens_parameter: str = MAX_TOKENS_PARAMETER
    reasoning_effort: str | None = None


class Backend(Protocol):
    """An LLM backend that answers generation requests."""

    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Answer one request.

        Args:
            request: The generation request.

        Returns:
            The generated text with the sampling parameters actually applied.
        """
        ...

    def generate_batch(self, requests: Sequence[GenerationRequest]) -> list[GenerationResult]:
        """Answer several requests, returning results in request order.

        Args:
            requests: The generation requests.

        Returns:
            One result per request, in the order of ``requests``.
        """
        ...

    def info(self) -> BackendInfo:
        """Describe the backend.

        Returns:
            Model name, backend kind, quantization, and context window.
        """
        ...

    def provenance(self) -> Mapping[str, str]:
        """Describe the settings that decide what the backend generates.

        The runner stores every entry, under its key prefixed with ``backend_``, in the
        identity of a run, so a store is resumed or merged only with a backend that
        generates alike.

        Returns:
            Text values by setting name; credentials never appear.
        """
        ...


def provenance_text(value: object) -> str:
    """Render one provenance value as text.

    Args:
        value: ``None``, a mapping, or a scalar.

    Returns:
        ``"none"`` for ``None``, canonical JSON (sorted keys, no spaces) for a mapping,
        and ``str(value)`` otherwise.
    """
    if value is None:
        return NONE_TEXT
    if isinstance(value, Mapping):
        return json.dumps(dict(value), sort_keys=True, separators=(",", ":"))
    return str(value)


def capability_provenance(capabilities: ModelCapabilities) -> dict[str, str]:
    """Describe the capability flags of a model for run provenance.

    Args:
        capabilities: The capabilities a backend works with.

    Returns:
        One entry per field of `ModelCapabilities`, rendered with
        `provenance_text`.
    """
    return {
        field.name: provenance_text(getattr(capabilities, field.name))
        for field in dataclasses.fields(capabilities)
    }


def capabilities_for_model(model: str) -> ModelCapabilities:
    """Look up which sampling parameters a model accepts.

    Reasoning models (names starting with ``gpt-5``, ``o1``, ``o3``, ``o4``) reject
    temperature and nucleus sampling and bound the output with
    ``max_completion_tokens``; every other model accepts all parameters.

    Args:
        model: Model name; matched case-insensitively.

    Returns:
        The capabilities of the model.
    """
    if model.lower().startswith(RESTRICTED_MODEL_PREFIXES):
        return ModelCapabilities(
            supports_temperature=False,
            supports_top_p=False,
            supports_seed=True,
            max_tokens_parameter=MAX_COMPLETION_TOKENS_PARAMETER,
        )
    return ModelCapabilities()
