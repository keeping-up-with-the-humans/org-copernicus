"""The rule by which texts and reference embeddings count as one load of one encoder.

`copernicus._load_check` imports neither torch nor transformers, so these tests of
`compare_loads` on the two descriptions run where neither is installed, as in the continuous
integration. The tests of the functions that call it are in ``tests/similarity`` and
``tests/hybrid``.
"""

from __future__ import annotations

from typing import Any

import pytest

from copernicus._load_check import compare_loads
from copernicus.embeddings import EncoderSpec
from copernicus.errors import ConfigurationError

MODEL = "org/mine"
IDENTITY = EncoderSpec("mine", MODEL, "mean", 64).identity()
ONE_ENVIRONMENT = "embed both sides in one environment"


def side(**versions: Any) -> dict[str, Any]:
    """Return the description of one side: the encoder's identity, a commit, and ``versions``."""
    return {**IDENTITY, "resolved_revision": "f00dfeed", **versions}


def test_one_load_is_compared_without_a_warning() -> None:
    texts = side(transformers_version="4.57.6", torch_version="2.5.1")
    recorded = {"transformers_version": "4.57.6", "torch_version": "2.5.1"}
    assert compare_loads(texts, dict(texts)) == {
        "texts": recorded,
        "references": recorded,
        "warnings": [],
    }


@pytest.mark.parametrize(
    ("first", "second", "lines"),
    [
        ("4.57.6", "5.12.1", "two major versions"),
        ("1!4.57.6", "1!5.12.1", "two major versions"),
        ("1!4.57.6", "4.57.6", "versions of two epochs"),
        ("5.12.1", "1!5.12.1", "versions of two epochs"),
    ],
)
def test_transformers_versions_of_two_lines_are_refused(
    first: str, second: str, lines: str
) -> None:
    with pytest.raises(ConfigurationError) as refused:
        compare_loads(side(transformers_version=first), side(transformers_version=second))
    message = str(refused.value)
    assert f"embedded under transformers {first} and {second}, {lines}," in message
    assert message.endswith(ONE_ENVIRONMENT)


@pytest.mark.parametrize(
    ("texts", "references", "owner", "value"),
    [
        ("4.57.3oops", "4.57.6", "texts", "4.57.3oops"),
        ("4.57.6", "4..57", "reference embeddings", "4..57"),
        ("4.57.3oops", "4.57.3oops", "texts", "4.57.3oops"),
        ("4.garbage", None, "texts", "4.garbage"),
        (None, 457, "reference embeddings", 457),
    ],
)
def test_a_recorded_version_that_cannot_be_read_is_refused_whatever_the_other_side(
    texts: Any, references: Any, owner: str, value: Any
) -> None:
    with pytest.raises(ConfigurationError) as refused:
        compare_loads(side(transformers_version=texts), side(transformers_version=references))
    message = str(refused.value)
    assert f"the {owner} of {MODEL} record transformers_version {value!r}" in message
    assert "which is not a version that can be read" in message


def test_a_release_of_one_line_and_another_torch_are_compared_with_a_warning_each() -> None:
    result = compare_loads(
        side(transformers_version="4.57.3", torch_version="2.5.0"),
        side(transformers_version="v4.57.6", torch_version="2.5.1"),
    )
    assert result["texts"] == {"transformers_version": "4.57.3", "torch_version": "2.5.0"}
    assert result["references"] == {"transformers_version": "v4.57.6", "torch_version": "2.5.1"}
    (transformers, torch) = result["warnings"]
    assert "transformers 4.57.3 and v4.57.6, two releases of one major version" in transformers
    assert "torch 2.5.0 and 2.5.1;" in torch


def test_two_spellings_of_one_version_are_one_version() -> None:
    result = compare_loads(
        side(transformers_version="v4.57.6", torch_version=" 2.5.1 "),
        side(transformers_version="4.57.6", torch_version="2.5.1"),
    )
    assert result["warnings"] == []


def test_a_side_without_a_version_is_compared_with_a_warning_and_recorded_as_unknown() -> None:
    result = compare_loads(side(), side(transformers_version="unknown", torch_version=None))
    unknown = {"transformers_version": "unknown", "torch_version": "unknown"}
    assert result["texts"] == unknown and result["references"] == unknown
    texts, references = result["warnings"]
    assert texts.startswith(f"the texts of {MODEL} record no version of transformers or torch")
    assert references.startswith(f"the reference embeddings of {MODEL} record no version")
