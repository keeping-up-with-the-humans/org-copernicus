"""Stub OpenAI clients and SDK errors for the backend tests."""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import httpx2
import openai

STUB_URL = "http://x"


def stub_response(content: str | None = '{"ok": true}', with_usage: bool = True) -> SimpleNamespace:
    """Build an object shaped like the SDK's chat completion response."""
    usage = SimpleNamespace(prompt_tokens=12, completion_tokens=3) if with_usage else None
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))], usage=usage
    )


class StubClient:
    """A client whose ``chat.completions.create`` records its calls.

    Attributes:
        calls: Keyword arguments of every call, in order.
        last_kwargs: Keyword arguments of the latest call.
    """

    def __init__(
        self,
        failures: list[Exception] | None = None,
        respond: Callable[[dict[str, Any]], SimpleNamespace] | None = None,
    ) -> None:
        self.calls: list[dict[str, Any]] = []
        self.last_kwargs: dict[str, Any] = {}
        self._failures = list(failures or [])
        self._respond = respond or (lambda kwargs: stub_response())
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        self.last_kwargs = kwargs
        if self._failures:
            raise self._failures.pop(0)
        return self._respond(kwargs)


def connection_error() -> openai.APIConnectionError:
    """Build a connection error with the SDK's own class."""
    return openai.APIConnectionError(request=httpx2.Request("POST", STUB_URL))


def status_error(status_code: int) -> openai.APIStatusError:
    """Build an HTTP status error with the SDK's own base class."""
    request = httpx2.Request("POST", STUB_URL)
    response = httpx2.Response(status_code, request=request)
    return openai.APIStatusError(f"status {status_code}", response=response, body=None)


def rate_limit_error() -> openai.RateLimitError:
    """Build the error the SDK raises for an HTTP 429 answer."""
    response = httpx2.Response(429, request=httpx2.Request("POST", STUB_URL))
    return openai.RateLimitError("rate limited", response=response, body=None)
