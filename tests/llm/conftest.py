"""Fixtures for the backend tests; the stub client and SDK errors live in ``openai_support``."""

from __future__ import annotations

import pytest
from openai_support import StubClient, connection_error


@pytest.fixture()
def stub_client() -> StubClient:
    return StubClient()


@pytest.fixture()
def stub_client_failing() -> StubClient:
    """A client that raises the connection error on every call."""
    return StubClient(failures=[connection_error() for _ in range(10)])


@pytest.fixture()
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Replace the backoff sleeper with a recorder."""
    delays: list[float] = []
    monkeypatch.setattr("copernicus.llm.openai_compatible._SLEEP", delays.append)
    return delays
