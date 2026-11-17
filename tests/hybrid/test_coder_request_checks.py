"""Tests that the hybrid coder's check before the pass makes the embedding call's own checks."""

from __future__ import annotations

from typing import Any

import pytest
from hybrid_support import STUB_DIMENSION, StubEncoder, encoder_with_failing_loader, unit_references

from copernicus.errors import ConfigurationError, CopernicusError, SettingsError
from copernicus.hybrid._loaded_encoder import check_before_the_pass
from copernicus.rubric.model import Rubric

# Requests that the embedding call refuses before it loads, with the error class and message.
# The last three hold two problems each, so their messages show which check comes first.
REQUESTS: list[tuple[list[str], list[Any], int, type[CopernicusError], str]] = [
    ([], [], 2048, ConfigurationError, "there are no texts to embed"),
    (["a", "b"], ["x", "y", "z"], 2048, ConfigurationError, "3 ids for 2 texts"),
    (["a", "b"], [1, "1"], 2048, ConfigurationError, "ids must be unique"),
    (
        ["a b", "", " \n\t "],
        ["x", "y", "z"],
        2048,
        ConfigurationError,
        "2 texts are empty or whitespace only and cannot be embedded: 'y', 'z'",
    ),
    (["a"], ["x"], 0, SettingsError, "checkpoint_every must be at least 1, got 0"),
    ([], [], 0, ConfigurationError, "there are no texts to embed"),
    (["a", "b"], ["x", "x"], 0, ConfigurationError, "ids must be unique"),
    (
        ["", "b"],
        ["x", "y"],
        -1,
        ConfigurationError,
        "1 texts are empty or whitespace only and cannot be embedded: 'x'",
    ),
]


@pytest.mark.parametrize(("texts", "ids", "every", "error", "message"), REQUESTS)
def test_the_check_before_the_pass_refuses_as_the_embedding_call_does_before_any_load(
    rubric: Rubric,
    texts: list[str],
    ids: list[Any],
    every: int,
    error: type[CopernicusError],
    message: str,
) -> None:
    refs = unit_references(rubric, "stub", STUB_DIMENSION)
    called, call_loads = encoder_with_failing_loader()
    with pytest.raises(CopernicusError) as by_the_call:
        called.embed(texts, ids, checkpoint_every=every)
    checked, check_loads = encoder_with_failing_loader()
    with pytest.raises(CopernicusError) as by_the_check:
        check_before_the_pass(checked, refs, texts, ids, every)
    assert (type(by_the_call.value), str(by_the_call.value), call_loads) == (error, message, [])
    assert (type(by_the_check.value), str(by_the_check.value), check_loads) == (error, message, [])


def test_the_check_before_the_pass_leaves_an_encoder_object_of_ones_own_to_its_pass(
    rubric: Rubric,
) -> None:
    encoder = StubEncoder()
    check_before_the_pass(encoder, unit_references(rubric, "stub", STUB_DIMENSION), [], [], 0)
    assert encoder.calls == []
