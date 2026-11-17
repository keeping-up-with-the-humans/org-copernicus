"""Stub OpenAI clients and SDK errors for the backend tests."""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any, get_type_hints

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


def sdk_class(constructor: Callable[..., object], parameter: str) -> Any:
    """Return the class that the SDK names for a parameter of one of its error constructors.

    The errors take a request or a response of the HTTP package that the SDK depends on. Taking
    the class from the SDK builds them without an import of that package, which this library does
    not declare, and with the classes of the HTTP package that the installed SDK uses.
    """
    return get_type_hints(constructor)[parameter]


def http_request() -> Any:
    """Build a request of the class that the SDK's errors keep."""
    return sdk_class(openai.APIConnectionError.__init__, "request")("POST", STUB_URL)


def http_response(status_code: int) -> Any:
    """Build a response with a status code, of the class that the SDK's status errors read."""
    response_class = sdk_class(openai.APIStatusError.__init__, "response")
    return response_class(status_code, request=http_request())


def connection_error() -> openai.APIConnectionError:
    """Build a connection error with the SDK's own class."""
    return openai.APIConnectionError(request=http_request())


def status_error(status_code: int) -> openai.APIStatusError:
    """Build an HTTP status error with the SDK's own base class."""
    response = http_response(status_code)
    return openai.APIStatusError(f"status {status_code}", response=response, body=None)


def rate_limit_error() -> openai.RateLimitError:
    """Build the error the SDK raises for an HTTP 429 answer."""
    return openai.RateLimitError("rate limited", response=http_response(429), body=None)
