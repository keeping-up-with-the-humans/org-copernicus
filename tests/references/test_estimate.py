"""Tests of the token estimate printed before a paid generation run."""

from __future__ import annotations

import logging
import sys
from types import SimpleNamespace

import pytest
from references_support import clean_responder, writer

import copernicus.references
from copernicus.errors import ConfigurationError
from copernicus.references import (
    GeneratorSettings,
    TokenEstimate,
    build_cells,
    estimate_tokens,
    plan_calls,
    system_prompt,
    token_counter,
    user_prompt,
)
from copernicus.rubric.model import Rubric

SETTINGS = GeneratorSettings(texts_per_call=8, calls_per_cell=2)


def prompts(rubric: Rubric, settings: GeneratorSettings) -> list[tuple[str, str]]:
    """Return the system and user prompt of every cell."""
    system = system_prompt(settings.style, settings.texts_per_call, None)
    return [
        (system, user_prompt(cell, rubric, settings.texts_per_call)) for cell in build_cells(rubric)
    ]


def fake_tiktoken(load: object) -> SimpleNamespace:
    """A stand-in for the tiktoken module whose encodings count words."""
    return SimpleNamespace(get_encoding=load)


def word_encoding(name: str) -> SimpleNamespace:
    """An encoding that has one token per word; only ``o200k_base`` is asked for."""
    assert name == "o200k_base"
    return SimpleNamespace(encode=lambda text: text.split())


def test_without_tiktoken_the_estimate_counts_a_quarter_of_the_characters(
    toy_rubric: Rubric,
) -> None:
    estimate = estimate_tokens(toy_rubric, SETTINGS, n_writers=3)
    per_round = sum(
        len(system) // 4 + len(user) // 4 for system, user in prompts(toy_rubric, SETTINGS)
    )
    writers = [writer(clean_responder(), name) for name in ("a", "b", "c")]
    assert estimate == TokenEstimate(
        prompt_tokens=per_round * 3 * 2,
        completion_tokens=plan_calls(toy_rubric, writers, SETTINGS) * 8 * 120,
        calls=plan_calls(toy_rubric, writers, SETTINGS),
        counter="characters / 4",
    )


def test_with_tiktoken_the_estimate_counts_the_tokens_of_its_encoding(
    toy_rubric: Rubric, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "tiktoken", fake_tiktoken(word_encoding))
    estimate = estimate_tokens(toy_rubric, SETTINGS)
    words = sum(
        len(system.split()) + len(user.split()) for system, user in prompts(toy_rubric, SETTINGS)
    )
    assert (estimate.prompt_tokens, estimate.counter) == (words * 2, "tiktoken o200k_base")


def test_an_encoding_that_cannot_be_loaded_falls_back_with_a_warning(
    toy_rubric: Rubric, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def offline(name: str) -> object:
        raise OSError("no network")

    monkeypatch.setitem(sys.modules, "tiktoken", fake_tiktoken(offline))
    with caplog.at_level(logging.WARNING, logger="copernicus.references.estimate"):
        estimate = estimate_tokens(toy_rubric, SETTINGS)
    assert estimate.counter == "characters / 4"
    (record,) = caplog.records
    assert "o200k_base" in record.getMessage() and "no network" in record.getMessage()


def test_the_cost_uses_prices_per_million_tokens() -> None:
    estimate = TokenEstimate(
        prompt_tokens=2_000_000, completion_tokens=500_000, calls=3, counter="x"
    )
    assert estimate.cost(price_input=2.5, price_output=10.0) == pytest.approx(10.0)


def test_an_estimate_needs_a_writer(toy_rubric: Rubric) -> None:
    with pytest.raises(ConfigurationError, match="writer"):
        estimate_tokens(toy_rubric, SETTINGS, n_writers=0)


def test_the_token_counter_is_public_and_returns_the_counter_with_its_name() -> None:
    count, name = token_counter()
    assert name == "characters / 4"
    assert count("abcdefgh") == 2
    assert "token_counter" in copernicus.references.__all__


def test_the_public_counter_uses_the_encoding_when_it_loads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "tiktoken", fake_tiktoken(word_encoding))
    count, name = token_counter()
    assert (count("one two three"), name) == (3, "tiktoken o200k_base")


def test_the_counter_counts_characters_without_touching_tiktoken_when_asked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(name: str) -> object:
        raise AssertionError("tiktoken must not be loaded")

    monkeypatch.setitem(sys.modules, "tiktoken", fake_tiktoken(forbidden))
    count, name = token_counter(use_tiktoken=False)
    assert (count("abcdefgh"), name) == (2, "characters / 4")
