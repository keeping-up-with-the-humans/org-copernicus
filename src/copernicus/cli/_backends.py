"""Construction of the LLM backend selected on the command line."""

from __future__ import annotations

import dataclasses
import hashlib
import inspect
import json
import logging
from collections.abc import Callable, Mapping
from enum import Enum
from pathlib import Path
from typing import Any

from copernicus.assets import load_toy_cues
from copernicus.errors import BackendError, SettingsError
from copernicus.llm import (
    Backend,
    FakeBackend,
    GenerationRequest,
    ModelCapabilities,
    OpenAICompatibleBackend,
    Responder,
    ScriptedResponder,
    VLLMBackend,
    capabilities_for_model,
    cue_responder,
)
from copernicus.llm.openai_compatible import API_KEY_VARIABLE
from copernicus.rubric import Rubric, load_rubric, toy_rubric_path

logger = logging.getLogger(__name__)

__all__ = ["BackendKind", "build_backend", "placeholder_reference_responder"]

# Options that have their own flag, or that must never come from the command line.
REFUSED_OPTIONS = ("model", "base_url", "api_key", "client", "timeout")
CAPABILITIES_OPTION = "capabilities"
PLACEHOLDER_DIGEST_LENGTH = 8
PLACEHOLDER_POLES = (("HIGH end", "high"), ("LOW end", "low"))
PLACEHOLDER_STRENGTHS = (("STRONGLY", "strong"), ("MODERATELY", "moderate"), ("FAINTLY", "faint"))


class BackendKind(str, Enum):
    """The backends selectable on the command line."""

    OPENAI = "openai"
    VLLM = "vllm"
    FAKE = "fake"


def build_backend(
    kind: BackendKind,
    *,
    model: str | None,
    base_url: str | None,
    timeout: float | None,
    options: Mapping[str, Any],
    rubric: Rubric,
    fake_script: Path | None,
    fake_responder: Responder | None = None,
) -> Backend:
    """Build the backend a command-line run codes with.

    The OpenAI-compatible backend reads its key from the environment; no key is ever
    taken from or printed on the command line.

    Args:
        kind: The backend to build.
        model: Model name; required for ``"openai"`` and ``"vllm"``, ignored for ``"fake"``.
        base_url: Endpoint of an OpenAI-compatible server; allowed for ``"openai"`` only.
        timeout: Per-request timeout in seconds; allowed for ``"openai"`` only, and the
            backend's default when ``None``.
        options: Further keyword arguments of the backend's constructor, decoded from
            JSON, for example ``chat_template_kwargs`` or ``max_model_len`` for vLLM and
            ``extra_body`` for an HTTP endpoint. For ``"openai"``, ``capabilities`` is an
            object of :class:`~copernicus.llm.base.ModelCapabilities` fields applied over
            the capabilities looked up for the model.
        rubric: The rubric of the run; selects the cue table of the fake backend.
        fake_script: JSON file holding a list of answers for the fake backend; required
            when the rubric is not the toy rubric, and allowed for ``"fake"`` only.
        fake_responder: Responder of the fake backend that replaces the cue table and the
            script, and names the backend after ``model``; ``None`` keeps them.

    Returns:
        The backend.

    Raises:
        BackendError: If the options do not fit the backend kind, the script is invalid,
            no API key is set, or vLLM is not installed.
        SettingsError: If an option has its own flag or is the API key, the backend's
            constructor takes no such argument, or a value is out of range.
    """
    if base_url is not None and kind is not BackendKind.OPENAI:
        raise BackendError("--base-url applies to --backend openai only")
    if timeout is not None and kind is not BackendKind.OPENAI:
        raise BackendError("--timeout applies to --backend openai only")
    if fake_script is not None and kind is not BackendKind.FAKE:
        raise BackendError("--fake-script applies to --backend fake only")
    refused = sorted(set(options) & set(REFUSED_OPTIONS))
    if refused:
        raise SettingsError(
            f"--backend-option cannot set {', '.join(refused)}; use --model, --base-url, "
            f"or --timeout, and the {API_KEY_VARIABLE} environment variable for the key"
        )
    logger.info("building the %s backend with the options %s", kind.value, sorted(options))
    if kind is BackendKind.FAKE and fake_responder is not None:
        named = {"model_name": model} if model is not None else {}
        return _construct(FakeBackend, fake_responder, **{**named, **options})
    if kind is BackendKind.FAKE:
        return _construct(FakeBackend, _fake_responder(rubric, fake_script), **options)
    if model is None:
        raise BackendError(f"--model is required for --backend {kind.value}")
    if kind is BackendKind.VLLM:
        return _construct(VLLMBackend, model, **options)
    arguments = dict(options)
    if CAPABILITIES_OPTION in arguments:
        arguments[CAPABILITIES_OPTION] = _capabilities(model, arguments[CAPABILITIES_OPTION])
    if timeout is not None:
        arguments["timeout"] = timeout
    return _construct(OpenAICompatibleBackend, model, base_url=base_url, **arguments)


def _construct(factory: Callable[..., Backend], *args: Any, **kwargs: Any) -> Backend:
    """Build a backend once its constructor is known to take the given arguments.

    Raises:
        SettingsError: If the constructor takes no such argument.
    """
    try:
        inspect.signature(factory).bind(*args, **kwargs)
    except TypeError as error:
        raise SettingsError(f"--backend-option: {error}") from error
    return factory(*args, **kwargs)


def _capabilities(model: str, value: Any) -> ModelCapabilities:
    """Apply a JSON object of capability fields over the capabilities of the model.

    Raises:
        SettingsError: If the value is not an object of capability fields.
    """
    if not isinstance(value, dict):
        raise SettingsError("--backend-option capabilities must be a JSON object of fields")
    try:
        return dataclasses.replace(capabilities_for_model(model), **value)
    except TypeError as error:
        raise SettingsError(f"--backend-option capabilities: {error}") from error


def _fake_responder(rubric: Rubric, fake_script: Path | None) -> Responder:
    """Return the fake backend's responder: cue-based for the toy rubric unless scripted."""
    if fake_script is not None:
        return ScriptedResponder(_read_script(fake_script))
    if rubric.content_hash() != load_rubric(toy_rubric_path()).content_hash():
        raise BackendError("--backend fake needs --fake-script unless the rubric is the toy rubric")
    return cue_responder(rubric, load_toy_cues())


def _read_script(path: Path) -> list[str]:
    """Read a JSON list of answer strings."""
    try:
        answers = json.loads(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError as error:
        raise SettingsError(f"--fake-script {path}: not UTF-8 text") from error
    except json.JSONDecodeError as error:
        raise BackendError(f"--fake-script {path} is not valid JSON: {error}") from error
    if not isinstance(answers, list) or not all(isinstance(a, str) for a in answers):
        raise BackendError(f"--fake-script {path} must hold a JSON list of strings")
    return answers


def placeholder_reference_responder(texts_per_call: int) -> Responder:
    """Return a responder that answers a reference-text request with placeholder sentences.

    For tests and dry runs of the pipeline only: the texts say nothing about the construct.
    Each answer is a JSON array of ``texts_per_call`` objects with a ``text``; the pole and
    the intensity are read off the request, and a digest of the request keeps the texts of
    different targets apart.

    Args:
        texts_per_call: Number of objects in each answer.

    Returns:
        The responder.
    """

    def respond(request: GenerationRequest) -> str:
        pole = next((name for marker, name in PLACEHOLDER_POLES if marker in request.user), "mid")
        strength = next(
            (name for marker, name in PLACEHOLDER_STRENGTHS if marker in request.user), "mid"
        )
        digest = hashlib.sha256(request.user.encode("utf-8")).hexdigest()[
            :PLACEHOLDER_DIGEST_LENGTH
        ]
        texts = [
            {"text": f"Placeholder sentence {number} for the {pole} pole, {strength}, id {digest}."}
            for number in range(1, texts_per_call + 1)
        ]
        return json.dumps(texts)

    return respond
