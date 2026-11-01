"""Tests of the settings of the similarity coder."""

from __future__ import annotations

import dataclasses

import pytest

from copernicus.errors import SettingsError
from copernicus.similarity.settings import (
    LONG_TEXT,
    POLARITIES,
    TARGET_LEVELS,
    WRITER_POOLINGS,
    SimilaritySettings,
)


def test_axes_are_named() -> None:
    assert POLARITIES == ("net", "positive", "negative")
    assert WRITER_POOLINGS == ("mean", "max", "pooled")
    assert LONG_TEXT == ("max_chunk", "mean_chunk", "pooled_embedding", "first_window")
    assert TARGET_LEVELS == ("construct", "attribute")


def test_defaults_are_the_documented_ones() -> None:
    settings = SimilaritySettings()
    assert (
        settings.polarity,
        settings.reduction,
        settings.writer_pooling,
        settings.long_text,
        settings.target_level,
    ) == ("net", "pca", "mean", "max_chunk", "construct")


def test_label_joins_the_five_axes() -> None:
    assert SimilaritySettings().label() == "net-pca-mean-max_chunk-construct"
    assert (
        SimilaritySettings("positive", "max", "max", "first_window", "attribute").label()
        == "positive-max-max-first_window-attribute"
    )


@pytest.mark.parametrize(
    "field", ["polarity", "reduction", "writer_pooling", "long_text", "target_level"]
)
def test_every_field_is_validated_by_name_and_value(field: str) -> None:
    with pytest.raises(SettingsError, match=f"{field}.*'bogus'"):
        SimilaritySettings(**{field: "bogus"})


def test_settings_are_frozen_and_hashable() -> None:
    settings = SimilaritySettings()
    with pytest.raises(dataclasses.FrozenInstanceError):
        settings.polarity = "positive"  # type: ignore[misc]
    assert len({settings, SimilaritySettings()}) == 1
