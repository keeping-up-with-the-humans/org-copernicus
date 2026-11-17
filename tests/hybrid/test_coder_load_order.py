"""Tests that the hybrid coder refuses a request that needs no model before it loads the encoder."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from hybrid_support import (
    LOADER_FAILURE,
    STUB_DIMENSION,
    StubEncoder,
    encoder_with_failing_loader,
    unit_references,
)

from copernicus.data import TextTable
from copernicus.errors import ConfigurationError, CopernicusError, SettingsError
from copernicus.hybrid import score_hybrid
from copernicus.rubric.model import Rubric
from copernicus.similarity import SimilaritySettings

MEAN = SimilaritySettings(reduction="mean")
GRANULARITIES = ("construct", "attribute")
PROBLEMS = ("checkpoint_every", "empty_table", "changed_text")
CHANGED_TEXT = "A changed text now stands under the first id of the table."


def request(
    problem: str, store: Path, table: TextTable
) -> tuple[TextTable, dict[str, Any], type[CopernicusError], str]:
    """Return the table, the options, the error class and the message of one invalid request.

    Each request is refused whatever the model: a checkpoint cadence below 1, a table without
    a text, or a text that differs from the text the store coded under its id.
    """
    if problem == "checkpoint_every":
        message = "checkpoint_every must be at least 1, got 0"
        return table, {"checkpoint_every": 0}, SettingsError, message
    if problem == "empty_table":
        return table.subset([]), {}, ConfigurationError, "there are no texts to embed"
    frame = table.frame.copy()
    frame.loc[0, "text"] = CHANGED_TEXT
    message = (
        f"1 of {len(table)} texts differ from the texts that run 1 of the store {store} was coded "
        f"on: {table.ids()[0]!r}; a store binds every text id to the text it was coded on; pass "
        "the texts the store was coded on, or code the changed texts into a new store"
    )
    return TextTable.from_frame(frame), {}, ConfigurationError, message


@pytest.mark.parametrize("granularity", GRANULARITIES)
@pytest.mark.parametrize("problem", PROBLEMS)
def test_a_request_that_needs_no_model_is_refused_before_the_library_encoder_is_loaded(
    coded_store: Path, rubric: Rubric, table: TextTable, granularity: str, problem: str
) -> None:
    encoder, loads = encoder_with_failing_loader()
    given, options, error, message = request(problem, coded_store, table)
    refs = unit_references(rubric, "stub", STUB_DIMENSION)
    with pytest.raises(CopernicusError) as refused:
        score_hybrid(
            coded_store,
            given,
            rubric,
            encoder,
            refs,
            MEAN,
            granularity=granularity,
            min_words=10,
            **options,
        )
    assert (type(refused.value), str(refused.value)) == (error, message)
    assert loads == []


@pytest.mark.parametrize("granularity", GRANULARITIES)
def test_a_request_that_passes_those_checks_reaches_the_loader_once(
    coded_store: Path, rubric: Rubric, table: TextTable, granularity: str
) -> None:
    encoder, loads = encoder_with_failing_loader()
    refs = unit_references(rubric, "stub", STUB_DIMENSION)
    with pytest.raises(ConfigurationError) as refused:
        score_hybrid(
            coded_store, table, rubric, encoder, refs, MEAN, granularity=granularity, min_words=10
        )
    assert str(refused.value) == LOADER_FAILURE
    assert loads == ["stub"]


@pytest.mark.parametrize("granularity", GRANULARITIES)
def test_an_encoder_object_of_ones_own_receives_the_request_as_given(
    coded_store: Path, rubric: Rubric, table: TextTable, granularity: str
) -> None:
    encoder = StubEncoder()
    refs = unit_references(rubric, "stub", STUB_DIMENSION)
    result = score_hybrid(
        coded_store,
        table,
        rubric,
        encoder,
        refs,
        MEAN,
        granularity=granularity,
        min_words=10,
        checkpoint_every=0,
    )
    assert encoder.options == [{"checkpoint": None, "checkpoint_every": 0}]
    assert len(result.frame) > 0
