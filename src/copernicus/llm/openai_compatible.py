"""Backend for OpenAI and OpenAI-compatible chat-completion servers."""

from __future__ import annotations

import hashlib
import logging
import os
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from urllib.parse import SplitResult, urlsplit

import openai

from copernicus.errors import BackendError, SettingsError
from copernicus.llm.base import (
    NONE_TEXT,
    TOP_P,
    BackendInfo,
    GenerationRequest,
    GenerationResult,
    ModelCapabilities,
    capabilities_for_model,
    capability_provenance,
    provenance_text,
)

logger = logging.getLogger(__name__)

__all__ = ["OpenAICompatibleBackend", "endpoint_of", "endpoint_path_digest"]

API_KEY_VARIABLE = "OPENAI_API_KEY"
BACKEND_KIND = "openai_compatible"
DEFAULT_BASE_URL = "https://api.openai.com/v1"
BACKOFF_BASE_SECONDS = 0.5
TOO_MANY_REQUESTS_STATUS = 429
SERVER_ERROR_STATUS = 500
PATH_DIGEST_LENGTH = 16
_SLEEP = time.sleep


def _split(url: str) -> SplitResult:
    """Split an endpoint URL into its parts.

    Raises:
        BackendError: If the URL cannot be parsed; the message does not repeat the URL.
    """
    try:
        return urlsplit(url)
    except ValueError as error:
        raise BackendError("invalid base_url: the URL cannot be parsed") from error


def endpoint_path_digest(url: str) -> str:
    """Fingerprint the path of an endpoint URL, which tells deployments on one host apart.

    The path is hashed as `urllib.parse.urlsplit` returns it without its trailing slashes,
    with no decoding and no change of case. The OpenAI client ends every base URL with a
    slash before it appends the route of a request, so ``/v1`` and ``/v1/`` reach the same
    deployment and give one digest, while ``/v1`` and ``/v2`` give two. User name, password,
    query, and fragment are not part of the path, and the path itself is kept only as a
    digest, so neither a credential of the URL nor a token in its path reaches the
    provenance.

    Args:
        url: Absolute URL of an endpoint.

    Returns:
        The first 16 hexadecimal characters of the SHA-256 of the path without trailing
        slashes, encoded as UTF-8, or ``"none"`` when nothing is left (an empty path or
        ``/``).

    Raises:
        BackendError: If the URL cannot be parsed; the message does not repeat the URL.
    """
    path = _split(url).path.rstrip("/")
    if not path:
        return NONE_TEXT
    return hashlib.sha256(path.encode("utf-8")).hexdigest()[:PATH_DIGEST_LENGTH]


def endpoint_of(url: str) -> str:
    """Reduce an endpoint URL to its scheme, host, and port.

    User name, password, path, query, and fragment are dropped, so the result can be
    stored without leaking credentials; `endpoint_path_digest` records the path as a
    digest.

    Args:
        url: Absolute URL of an endpoint.

    Returns:
        ``scheme://host`` with ``:port`` when the URL names one; an IPv6 host is
        bracketed.

    Raises:
        BackendError: If the URL cannot be parsed, has no scheme or host, or has an
            invalid port; the message does not repeat the URL.
    """
    parts = _split(url)
    if not parts.scheme or not parts.hostname:
        raise BackendError("base_url must be an absolute URL with a scheme and a host")
    try:
        port = parts.port
    except ValueError as error:
        raise BackendError("base_url names an invalid port") from error
    host = f"[{parts.hostname}]" if ":" in parts.hostname else parts.hostname
    return f"{parts.scheme}://{host}" + (f":{port}" if port is not None else "")


def _endpoint_identity(url: str) -> tuple[str, str]:
    """Return `endpoint_of` and `endpoint_path_digest` of an endpoint URL."""
    return endpoint_of(url), endpoint_path_digest(url)


class OpenAICompatibleBackend:
    """A backend that calls a chat-completions endpoint over HTTP.

    Sampling parameters are sent only when the model's capabilities allow them, and the
    result reports which parameters were actually sent. Transport failures are retried
    with exponential backoff; the SDK's own retries are disabled so that
    ``transport_retries`` is the only retry count.
    """

    def __init__(
        self,
        model: str,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 120.0,
        transport_retries: int = 3,
        concurrency: int = 4,
        capabilities: ModelCapabilities | None = None,
        extra_body: Mapping[str, Any] | None = None,
        client: Any | None = None,
    ) -> None:
        """Initialise the backend.

        Args:
            model: Model name sent with every request.
            base_url: Endpoint of an OpenAI-compatible server; ``None`` uses the OpenAI API.
            api_key: API key; defaults to the ``OPENAI_API_KEY`` environment variable.
            timeout: Per-request timeout in seconds; above 0.
            transport_retries: Number of attempts per request for transport failures;
                at least 1.
            concurrency: Number of parallel requests in ``generate_batch``; at least 1.
            capabilities: Sampling parameters the model accepts; looked up from the model
                name when ``None``.
            extra_body: Extra fields merged into every request body.
            client: A ready client exposing ``chat.completions.create``; built from the
                other arguments when ``None``. An injected client is described in the
                provenance by ``base_url``, or by the OpenAI API when that is ``None``.

        Raises:
            SettingsError: If the timeout, the transport retries, or the concurrency is
                out of range.
            BackendError: If no client is given and no API key is available, or the
                endpoint URL has no usable host.
        """
        checks = (
            ("timeout", timeout, timeout > 0, "greater than 0"),
            ("transport_retries", transport_retries, transport_retries >= 1, "at least 1"),
            ("concurrency", concurrency, concurrency >= 1, "at least 1"),
        )
        for field, value, valid, expectation in checks:
            if not valid:
                raise SettingsError(f"{field} must be {expectation}, got {value!r}")
        self._model = model
        self._capabilities = capabilities or capabilities_for_model(model)
        self._transport_retries = transport_retries
        self._concurrency = concurrency
        self._extra_body = dict(extra_body) if extra_body else {}
        if client is not None:
            self._client = client
            self._endpoint, self._path_digest = _endpoint_identity(
                base_url if base_url is not None else DEFAULT_BASE_URL
            )
            return
        given = _endpoint_identity(base_url) if base_url is not None else None
        key = api_key if api_key is not None else os.environ.get(API_KEY_VARIABLE)
        if not key:
            raise BackendError(f"no API key: pass api_key or set {API_KEY_VARIABLE}")
        self._client = openai.OpenAI(base_url=base_url, api_key=key, timeout=timeout, max_retries=0)
        # Without a base_url the SDK picks the endpoint (its environment variable or default).
        self._endpoint, self._path_digest = given or _endpoint_identity(str(self._client.base_url))

    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Send one request, retrying transport failures.

        Args:
            request: The generation request.

        Returns:
            The generated text, token usage, and the sampling parameters actually sent. A
            response without choices, or without content, counts as an empty answer, which
            the coders treat as an unparseable output.

        Raises:
            BackendError: On a client error, or when the transport retries are exhausted.
        """
        kwargs, temperature, top_p, seed = self._build_call(request)
        response = self._call_with_retries(kwargs)
        usage = response.usage
        if not response.choices:
            logger.warning("the endpoint returned no choices; the answer counts as empty")
        return GenerationResult(
            text=(response.choices[0].message.content or "") if response.choices else "",
            prompt_tokens=usage.prompt_tokens if usage is not None else None,
            completion_tokens=usage.completion_tokens if usage is not None else None,
            temperature_applied=temperature,
            top_p_applied=top_p,
            seed_applied=seed,
        )

    def generate_batch(self, requests: Sequence[GenerationRequest]) -> list[GenerationResult]:
        """Send the requests in parallel, returning results in request order.

        Args:
            requests: The generation requests.

        Returns:
            One result per request, in the order of ``requests``.

        Raises:
            BackendError: If any request fails; raised after all workers have finished.
        """
        with ThreadPoolExecutor(max_workers=self._concurrency) as pool:
            futures = [pool.submit(self.generate, request) for request in requests]
            return [future.result() for future in futures]

    def info(self) -> BackendInfo:
        """Describe the backend.

        Returns:
            Backend info with the model name, kind ``"openai_compatible"``, and no
            quantization or window.
        """
        return BackendInfo(self._model, BACKEND_KIND, None, None)

    def provenance(self) -> dict[str, str]:
        """Describe the settings that decide what the endpoint generates.

        Returns:
            Kind ``"openai_compatible"``, the endpoint (scheme, host, and port only), the
            digest of the endpoint's path (``endpoint_path_digest``, see
            `endpoint_path_digest`), which tells two deployments on one host apart, the
            model, the capability flags, the nucleus-sampling threshold, and the extra
            request body as canonical JSON.
        """
        return {
            "kind": BACKEND_KIND,
            "endpoint": self._endpoint,
            "endpoint_path_digest": self._path_digest,
            "model": self._model,
            **capability_provenance(self._capabilities),
            "top_p": provenance_text(TOP_P),
            "extra_body": provenance_text(self._extra_body),
        }

    def _build_call(
        self, request: GenerationRequest
    ) -> tuple[dict[str, Any], float | None, float | None, int | None]:
        """Build the call arguments and the sampling parameters they carry."""
        caps = self._capabilities
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": request.system},
                {"role": "user", "content": request.user},
            ],
            caps.max_tokens_parameter: request.max_new_tokens,
        }
        temperature = request.temperature if caps.supports_temperature else None
        top_p = TOP_P if temperature and caps.supports_top_p else None
        seed = request.seed if caps.supports_seed else None
        if temperature is not None:
            kwargs["temperature"] = temperature
        if top_p is not None:
            kwargs["top_p"] = top_p
        if seed is not None:
            kwargs["seed"] = seed
        if caps.reasoning_effort is not None:
            kwargs["reasoning_effort"] = caps.reasoning_effort
        if self._extra_body:
            kwargs["extra_body"] = self._extra_body
        return kwargs, temperature, top_p, seed

    def _call_with_retries(self, kwargs: dict[str, Any]) -> Any:
        """Call the endpoint, retrying transport failures with exponential backoff."""
        last_error: Exception | None = None
        for attempt in range(self._transport_retries):
            try:
                return self._client.chat.completions.create(**kwargs)
            except openai.APIStatusError as error:
                if not _is_transient(error):
                    raise BackendError(str(error), attempts=1) from error
                last_error = error
            except openai.APIConnectionError as error:
                last_error = error
            logger.warning(
                "transport failure on attempt %d of %d: %s",
                attempt + 1,
                self._transport_retries,
                type(last_error).__name__,
            )
            if attempt + 1 < self._transport_retries:
                _SLEEP(BACKOFF_BASE_SECONDS * 2**attempt)
        raise BackendError(str(last_error), attempts=self._transport_retries) from last_error


def _is_transient(error: openai.APIStatusError) -> bool:
    """Return whether a status error is worth retrying (rate limit 429 or a server error)."""
    status = error.status_code
    return status == TOO_MANY_REQUESTS_STATUS or status >= SERVER_ERROR_STATUS
