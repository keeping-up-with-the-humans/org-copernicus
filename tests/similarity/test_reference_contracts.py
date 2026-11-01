"""Tests of the array and file contracts of reference embeddings."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from similarity_support import make_references

from copernicus.errors import ConfigurationError
from copernicus.rubric.model import Rubric
from copernicus.similarity import ReferenceEmbeddings

LABELS = ("target", "polarity", "level", "style", "writer_model")


def test_object_and_list_labels_become_string_arrays_and_survive_a_round_trip(
    toy_rubric: Rubric, tmp_path: Path
) -> None:
    refs = make_references(toy_rubric)
    as_objects = replace(
        refs, **{name: np.asarray(getattr(refs, name).tolist(), dtype=object) for name in LABELS}
    )
    as_lists = replace(refs, **{name: getattr(refs, name).tolist() for name in LABELS})
    for each in (as_objects, as_lists):
        for name in LABELS:
            assert getattr(each, name).dtype.kind == "U", name
            assert getattr(each, name).tolist() == getattr(refs, name).tolist(), name
    as_objects.save(tmp_path / "poles")
    loaded = ReferenceEmbeddings.load(tmp_path / "poles")
    assert loaded.writer_model.tolist() == refs.writer_model.tolist()


def test_integer_writer_names_become_strings(toy_rubric: Rubric) -> None:
    refs = make_references(toy_rubric, writers=("a",))
    numbered = replace(refs, writer_model=np.zeros(len(refs), dtype=np.int64))
    assert numbered.writers() == ["0"]


def test_a_two_dimensional_label_array_is_refused(toy_rubric: Rubric) -> None:
    refs = make_references(toy_rubric)
    with pytest.raises(ConfigurationError, match="level must be one-dimensional"):
        replace(refs, level=refs.level.reshape(-1, 1))


def test_a_label_that_is_neither_a_string_nor_an_integer_is_refused(toy_rubric: Rubric) -> None:
    refs = make_references(toy_rubric)
    styles = np.asarray([None, *refs.style.tolist()[1:]], dtype=object)
    with pytest.raises(ConfigurationError, match="style must hold strings or integers"):
        replace(refs, style=styles)


@pytest.mark.parametrize("content", ["{not json", "[1, 2]"])
def test_a_malformed_sidecar_is_a_configuration_error_naming_the_file(
    toy_rubric: Rubric, tmp_path: Path, content: str
) -> None:
    make_references(toy_rubric).save(tmp_path / "poles")
    (tmp_path / "poles.json").write_text(content, encoding="utf-8")
    with pytest.raises(ConfigurationError, match=r"poles\.json"):
        ReferenceEmbeddings.load(tmp_path / "poles")
