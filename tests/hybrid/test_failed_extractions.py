"""Tests of texts whose extraction failed: the hybrid coder refuses them before it embeds.

A failed extraction leaves its attribute without passages, so the evidence text would lack them
and a text left without any passage would pass for one without evidence. A short text keeps its
raw text whatever its passages, and a failed scoring leaves the passages in place: such texts
stay scoreable, with the fallbacks and counts of a run without failures.
"""

from __future__ import annotations

import ast
from collections.abc import Sequence
from pathlib import Path

import pytest
from hybrid_support import STUB_DIMENSION, StubEncoder, unit_references

import copernicus.coders.two_stage_types as two_stage_types
import copernicus.hybrid.texts as texts_module
from copernicus.assets import load_toy_cues
from copernicus.coders import CodingSettings, code_texts, export_scores
from copernicus.coders.store_cells import FAILURE_STATUSES
from copernicus.data import TextTable
from copernicus.errors import ConfigurationError
from copernicus.hybrid import evidence_texts, score_hybrid
from copernicus.llm import FakeBackend, cue_responder
from copernicus.llm.base import GenerationRequest
from copernicus.llm.fake import SCORING_MARKER, STAGE_ONE_MARKER
from copernicus.rubric.model import Rubric
from copernicus.similarity import SimilaritySettings

MEAN = SimilaritySettings(reduction="mean")
ACCEPT_FAILURES = CodingSettings(runs=1, max_failure_share=1.0)
MIN_WORDS = 5
N_TEXTS = 4
FAILING = 1


def extraction_fails_for(rubric: Rubric, texts: Sequence[str]) -> FakeBackend:
    """Return a cue backend that answers the extraction requests of the given texts with nothing."""
    cue = cue_responder(rubric, load_toy_cues())

    def respond(request: GenerationRequest) -> str:
        if STAGE_ONE_MARKER in request.system and any(text in request.user for text in texts):
            return ""
        return cue(request)

    return FakeBackend(respond)


def scoring_fails(rubric: Rubric) -> FakeBackend:
    """Return a cue backend that answers every scoring request with nothing."""
    cue = cue_responder(rubric, load_toy_cues())

    def respond(request: GenerationRequest) -> str:
        return "" if SCORING_MARKER in request.system else cue(request)

    return FakeBackend(respond)


@pytest.fixture
def failed_store(tmp_path: Path, rubric: Rubric, table: TextTable) -> tuple[Path, TextTable, str]:
    """Code four texts, the extraction of the second failing for every attribute.

    Returns:
        The store, the four texts, and the id of the text whose extraction failed.
    """
    small = table.subset(table.ids()[:N_TEXTS])
    failing = small.ids()[FAILING]
    store = tmp_path / "failed.sqlite"
    backend = extraction_fails_for(rubric, [small.text_of(failing)])
    code_texts(small, rubric, backend, store, ACCEPT_FAILURES)
    return store, small, str(failing)


@pytest.mark.parametrize("granularity", ["construct", "attribute"])
def test_a_text_whose_extraction_failed_is_refused_before_anything_is_embedded(
    failed_store: tuple[Path, TextTable, str], rubric: Rubric, granularity: str
) -> None:
    store, small, failing = failed_store
    statuses = export_scores(store).groupby("text_id")["status"].agg(set)
    assert statuses[failing] == {"extraction_failed"}
    others = [str(text_id) for text_id in small.ids() if str(text_id) != failing]
    assert all("extraction_failed" not in statuses[text_id] for text_id in others)
    encoder = StubEncoder()
    refs = unit_references(rubric, "stub", STUB_DIMENSION)
    with pytest.raises(ConfigurationError) as refused:
        score_hybrid(
            store, small, rubric, encoder, refs, MEAN, granularity=granularity, min_words=MIN_WORDS
        )
    message = " ".join(str(refused.value).split())
    assert (
        f"1 of {N_TEXTS} texts have an attribute whose extraction failed in run 1 of the store "
        f"{store}: {failing!r};"
    ) in message
    assert "recode them with code_texts(..., CodingSettings(retry_failed=True))" in message
    assert message.endswith(
        "or score a table without them; export_scores(store) lists every one: its rows of run "
        "1 with status 'extraction_failed'"
    )
    assert encoder.calls == []
    with pytest.raises(ConfigurationError, match="extraction failed"):
        evidence_texts(store, rubric, small, granularity=granularity, min_words=MIN_WORDS)


def test_a_table_without_the_text_or_a_recoded_store_is_scored(
    failed_store: tuple[Path, TextTable, str], rubric: Rubric
) -> None:
    store, small, failing = failed_store
    refs = unit_references(rubric, "stub", STUB_DIMENSION)
    rest = small.subset([text_id for text_id in small.ids() if str(text_id) != failing])
    scored = score_hybrid(store, rest, rubric, StubEncoder(), refs, MEAN, min_words=MIN_WORDS)
    assert failing not in set(scored.frame.text_id) and len(scored.frame) == (N_TEXTS - 1) * 4
    cue = FakeBackend(cue_responder(rubric, load_toy_cues()))
    code_texts(small, rubric, cue, store, CodingSettings(runs=1, retry_failed=True))
    recoded = score_hybrid(store, small, rubric, StubEncoder(), refs, MEAN, min_words=MIN_WORDS)
    assert failing in set(recoded.frame.text_id) and len(recoded.frame) == N_TEXTS * 4


def test_a_short_text_whose_extraction_failed_keeps_its_short_fallback(
    failed_store: tuple[Path, TextTable, str], rubric: Rubric
) -> None:
    store, small, failing = failed_store
    words = dict(zip(small.frame.text_id.astype(str), small.frame.n_words, strict=True))
    short = evidence_texts(
        store, rubric, small, granularity="attribute", min_words=int(words[failing]) + 1
    )
    for texts in short.values():
        assert texts.fallback[texts.ids.index(failing)] == "short"
    with pytest.raises(ConfigurationError, match="extraction failed"):
        evidence_texts(store, rubric, small, granularity="attribute", min_words=int(words[failing]))


def test_the_message_names_the_first_ten_texts_in_table_order(
    tmp_path: Path, rubric: Rubric, table: TextTable
) -> None:
    small = table.subset(table.ids()[:14])
    failing = small.ids()[2:]
    store = tmp_path / "many.sqlite"
    backend = extraction_fails_for(rubric, [small.text_of(text_id) for text_id in failing])
    code_texts(small, rubric, backend, store, ACCEPT_FAILURES)
    with pytest.raises(ConfigurationError) as refused:
        evidence_texts(store, rubric, small, granularity="construct", min_words=MIN_WORDS)
    message = str(refused.value)
    assert "12 of 14 texts" in message and "and 2 more" in message
    named = [str(text_id) for text_id in failing]
    assert all(repr(text_id) in message for text_id in named[:10])
    assert not any(repr(text_id) in message for text_id in named[10:])


def test_failed_scorings_keep_the_passages_and_the_counts_of_a_run_without_failures(
    tmp_path: Path, rubric: Rubric, table: TextTable
) -> None:
    small = table.subset(table.ids()[:6])
    normal, failed = tmp_path / "normal.sqlite", tmp_path / "scoring_failed.sqlite"
    cue = FakeBackend(cue_responder(rubric, load_toy_cues()))
    code_texts(small, rubric, cue, normal, CodingSettings(runs=1))
    code_texts(small, rubric, scoring_fails(rubric), failed, ACCEPT_FAILURES)
    statuses = set(export_scores(failed)["status"])
    assert "scoring_failed" in statuses and "extraction_failed" not in statuses
    for granularity in ("construct", "attribute"):
        expected = evidence_texts(normal, rubric, small, granularity=granularity, min_words=10)
        assert evidence_texts(failed, rubric, small, granularity=granularity, min_words=10) == (
            expected
        )
    refs = unit_references(rubric, "stub", STUB_DIMENSION)
    result = score_hybrid(failed, small, rubric, StubEncoder(), refs, MEAN, min_words=10)
    baseline = score_hybrid(normal, small, rubric, StubEncoder(), refs, MEAN, min_words=10)
    assert result.summary == baseline.summary and result.summary["n_evidence"] > 0
    assert result.frame.fallback.tolist() == baseline.frame.fallback.tolist()


def test_the_failure_status_is_the_coders_own() -> None:
    """The hybrid check takes the status name from the coders, which write it."""
    assert texts_module.__file__ is not None
    tree = ast.parse(Path(texts_module.__file__).read_text(encoding="utf-8"))
    imported = {
        alias.name
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module == two_stage_types.__name__
        for alias in node.names
    }
    assert "EXTRACTION_FAILED" in imported
    assert texts_module.EXTRACTION_FAILED == two_stage_types.EXTRACTION_FAILED
    assert two_stage_types.EXTRACTION_FAILED in FAILURE_STATUSES
    literals = {
        node.value.value
        for node in tree.body
        if isinstance(node, ast.Assign | ast.AnnAssign) and isinstance(node.value, ast.Constant)
    }
    assert "extraction_failed" not in literals
