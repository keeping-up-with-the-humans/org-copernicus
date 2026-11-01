"""Helpers of the reference-text tests: responders of plain texts, and writers that fail once.

The responders write plain, unquoting texts. A failing writer replaces the payload writer or the
sidecar writer of a module and fails once, as a full disk would.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from copernicus.llm.base import GenerationRequest
from copernicus.llm.fake import FakeBackend

Responder = Callable[[GenerationRequest], str]

HIGH_MARKER = "HIGH end of this quality"
LOW_MARKER = "active LOW end"


def texts_json(count: int, tag: object, *, extra_keys: bool = False) -> str:
    """Return a JSON array of ``count`` distinct objects with a ``text`` field.

    Args:
        count: Number of objects.
        tag: Value that makes the texts of this answer differ from other answers.
        extra_keys: Whether each object carries an additional ``id`` key.

    Returns:
        The array as JSON text.
    """
    items: list[dict[str, Any]] = []
    for index in range(count):
        item: dict[str, Any] = {"text": f"Note {tag}-{index}: I went for a long walk today."}
        if extra_keys:
            item["id"] = index
        items.append(item)
    return json.dumps(items)


def clean_responder(count: int = 10) -> Responder:
    """Return a responder that answers every request with ``count`` distinct clean texts."""

    def respond(request: GenerationRequest) -> str:
        return texts_json(count, request.seed)

    return respond


def seed_responder(by_seed: dict[int, str], count: int = 10) -> Responder:
    """Return a responder that answers chosen seeds with fixed text and the rest cleanly.

    Args:
        by_seed: Raw answer per request seed.
        count: Number of texts of a clean answer.

    Returns:
        The responder.
    """

    def respond(request: GenerationRequest) -> str:
        assert request.seed is not None
        if request.seed in by_seed:
            return by_seed[request.seed]
        return texts_json(count, request.seed)

    return respond


def writer(responder: Responder, name: str = "writer-a") -> FakeBackend:
    """Return a fake writer model with the given name."""
    return FakeBackend(responder, model_name=name)


class FailingWrite:
    """Wraps a file writer so that its ``at``-th call fails once, as a full disk would."""

    def __init__(self, write: Callable[[Path, Any], None], at: int) -> None:
        self.write = write
        self.at = at
        self.count = 0

    def __call__(self, target: Path, content: Any) -> None:
        self.count += 1
        if self.count == self.at:
            raise OSError("disk full")
        self.write(target, content)


def fail_write(monkeypatch: pytest.MonkeyPatch, module: ModuleType, name: str, at: int = 1) -> None:
    """Make the file writer ``name`` of ``module`` fail once, at its ``at``-th call.

    Args:
        monkeypatch: The fixture that restores the writer after the test.
        module: The module that writes the files, such as ``copernicus.references.files``.
        name: ``"write_atomically"`` for the payload, ``"write_json"`` for the sidecar.
        at: The call that fails; the calls before and after it write as usual.
    """
    monkeypatch.setattr(module, name, FailingWrite(getattr(module, name), at))
