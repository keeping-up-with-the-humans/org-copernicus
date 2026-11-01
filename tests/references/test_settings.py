"""Tests for the settings of reference-text generation."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from copernicus.errors import SettingsError
from copernicus.references import GeneratorSettings


def test_the_defaults_are_the_documented_design() -> None:
    settings = GeneratorSettings()
    assert settings.style == "neutral"
    assert settings.register_description is None
    assert settings.texts_per_call == 10
    assert settings.calls_per_cell == 1
    assert settings.temperature == 0.9
    assert settings.max_new_tokens == 16384
    assert settings.seed == 42
    assert settings.max_retries == 3
    assert settings.concurrency == 1


def test_a_register_style_with_a_description_is_valid() -> None:
    settings = GeneratorSettings(style="register", register_description="short posts")
    assert settings.register_description == "short posts"


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"style": "formal"}, "style"),
        ({"style": "register"}, "register_description"),
        ({"style": "register", "register_description": ""}, "register_description"),
        ({"style": "register", "register_description": "  "}, "register_description"),
        ({"register_description": "short posts"}, "register_description"),
        ({"texts_per_call": 0}, "texts_per_call"),
        ({"calls_per_cell": 0}, "calls_per_cell"),
        ({"temperature": -0.1}, "temperature"),
        ({"temperature": 2.1}, "temperature"),
        ({"max_new_tokens": 0}, "max_new_tokens"),
        ({"max_retries": 0}, "max_retries"),
        ({"texts_per_call": True}, "texts_per_call.*True"),
        ({"calls_per_cell": 1.5}, "calls_per_cell.*1.5"),
        ({"max_new_tokens": "100"}, "max_new_tokens"),
        ({"max_retries": None}, "max_retries"),
        ({"seed": 4.0}, "seed"),
        ({"temperature": "hot"}, "temperature.*hot"),
        ({"temperature": True}, "temperature.*True"),
        ({"temperature": None}, "temperature"),
        ({"concurrency": 0}, "concurrency"),
        ({"concurrency": 2.0}, "concurrency"),
        ({"concurrency": True}, "concurrency"),
    ],
)
def test_invalid_settings_are_refused(changes: dict[str, object], message: str) -> None:
    with pytest.raises(SettingsError, match=message):
        GeneratorSettings(**changes)  # type: ignore[arg-type]


def test_the_boundary_values_are_valid() -> None:
    GeneratorSettings(temperature=0.0, texts_per_call=1, calls_per_cell=1, max_retries=1)
    GeneratorSettings(temperature=2.0, max_new_tokens=1)
    GeneratorSettings(temperature=1)


def test_settings_are_immutable() -> None:
    with pytest.raises(FrozenInstanceError):
        GeneratorSettings().seed = 1  # type: ignore[misc]
