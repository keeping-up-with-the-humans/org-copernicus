"""Tests that the comparison of two loads reads every recorded version as a package version.

A recorded transformers or torch version is read as the ``packaging`` library reads a package
version: an epoch, a leading ``v``, surrounding spaces and the labels of a pre-release,
post-release, development release or local build are parts of it. Text that the parser cannot
read is refused on either side, whatever the other side records, before anything is compared.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any

import pytest
from similarity_support import VERSIONS, make_references, make_texts, texts_along_first_axis

from copernicus.embeddings import EmbeddingSet
from copernicus.errors import ConfigurationError
from copernicus.rubric.model import Rubric
from copernicus.similarity import (
    ReferenceEmbeddings,
    SimilaritySettings,
    score_similarity,
    scoring_provenance,
)

EXACT = SimilaritySettings(reduction="mean", writer_pooling="pooled", long_text="pooled_embedding")
ONE_ENVIRONMENT = "embed both sides in one environment"
OWNERS = {"texts": "texts", "references": "reference embeddings"}


def with_versions(
    rubric: Rubric, texts_versions: dict[str, Any], references_versions: dict[str, Any]
) -> tuple[EmbeddingSet, ReferenceEmbeddings]:
    """Return texts and references of one load with some of their version keys replaced."""
    texts, refs = make_texts(texts_along_first_axis(5)), make_references(rubric)
    return (
        replace(texts, sidecar={**texts.sidecar, **texts_versions}),
        replace(refs, sidecar={**refs.sidecar, **references_versions}),
    )


@pytest.mark.parametrize(
    ("key", "texts_value", "references_value", "refused"),
    [
        ("transformers_version", "4.57.6oops", "4.57.6", "texts"),
        ("transformers_version", "4.57.6", "4..57", "references"),
        ("transformers_version", "4.garbage", "4.57.6", "texts"),
        ("transformers_version", "fake-transformers", "fake-transformers", "texts"),
        ("transformers_version", None, "fake-transformers", "references"),
        ("transformers_version", "fake-transformers", "unknown", "texts"),
        ("torch_version", "fake-torch", "fake-torch", "texts"),
        ("torch_version", "2.5.0", "2.5.0 cu121", "references"),
    ],
)
def test_a_recorded_version_that_the_parser_cannot_read_is_refused_whatever_the_other_side(
    toy_rubric: Rubric, key: str, texts_value: Any, references_value: Any, refused: str
) -> None:
    texts, refs = with_versions(toy_rubric, {key: texts_value}, {key: references_value})
    with pytest.raises(ConfigurationError) as refusal:
        score_similarity(texts, refs, toy_rubric, EXACT)
    value = texts_value if refused == "texts" else references_value
    owner = OWNERS[refused]
    message = str(refusal.value)
    assert f"the {owner} of test/encoder record {key} {value!r}" in message
    assert "which is not a version that can be read" in message
    assert f"embed the {owner} again with copernicus" in message


@pytest.mark.parametrize("value", [4.57, 5, ["4.57.6"], "", "  "])
def test_a_recorded_version_that_is_not_text_is_refused(toy_rubric: Rubric, value: Any) -> None:
    texts, refs = with_versions(toy_rubric, {"transformers_version": value}, {})
    refusal = r"the texts of test/encoder record transformers_version .* is not a version"
    with pytest.raises(ConfigurationError, match=refusal):
        score_similarity(texts, refs, toy_rubric, EXACT)


@pytest.mark.parametrize(
    ("texts_version", "references_version", "lines"),
    [
        ("1!4.57.6", "1!5.12.1", "two major versions"),
        ("4.57.6", "v5.0.0", "two major versions"),
        ("1!4.57.6", "4.57.6", "versions of two epochs"),
        ("1!5.12.1", "5.12.1", "versions of two epochs"),
    ],
)
def test_recorded_versions_of_two_lines_are_refused(
    toy_rubric: Rubric, texts_version: str, references_version: str, lines: str
) -> None:
    texts, refs = with_versions(
        toy_rubric,
        {"transformers_version": texts_version},
        {"transformers_version": references_version},
    )
    with pytest.raises(ConfigurationError) as refusal:
        score_similarity(texts, refs, toy_rubric, EXACT)
    message = str(refusal.value)
    assert f"transformers {texts_version} and {references_version}, {lines}" in message
    assert message.endswith(ONE_ENVIRONMENT)


@pytest.mark.parametrize(
    ("texts_version", "references_version", "warned"),
    [
        ("0!5.12.1", "5.12.1", False),
        ("v4.57.6", "4.57.6", False),
        (" 4.57.6 ", "4.57.6", False),
        ("4.57.6+local.1", "4.57.6+local.2", True),
        ("5.0.0rc1", "5.12.1", True),
        ("4.57.6.post1", "4.57.6", True),
    ],
)
def test_recorded_versions_of_one_line_are_compared_as_the_parser_reads_them(
    toy_rubric: Rubric,
    caplog: pytest.LogCaptureFixture,
    texts_version: str,
    references_version: str,
    warned: bool,
) -> None:
    texts, refs = with_versions(
        toy_rubric,
        {"transformers_version": texts_version},
        {"transformers_version": references_version},
    )
    with caplog.at_level(logging.WARNING):
        frame = score_similarity(texts, refs, toy_rubric, EXACT)
    messages = [record.getMessage() for record in caplog.records]
    assert len(messages) == int(warned)
    if warned:
        assert f"transformers {texts_version} and {references_version}, two releases" in messages[0]
    comparison = scoring_provenance(texts, refs, toy_rubric, [EXACT], len(frame))["load_comparison"]
    assert comparison["texts"] == {**VERSIONS, "transformers_version": texts_version}
    assert comparison["references"] == {**VERSIONS, "transformers_version": references_version}
    assert comparison["warnings"] == messages
