"""Tests for the runner under concurrency and under a failing backend."""

from __future__ import annotations

import itertools
import sqlite3
import threading
import time
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest

from copernicus.assets import load_toy_cues
from copernicus.coders import CodingSettings, code_texts, export_evidence, export_scores
from copernicus.data import TextTable
from copernicus.errors import BackendError, RunFailureCeilingError
from copernicus.llm import FakeBackend, GenerationRequest, GenerationResult, cue_responder
from copernicus.rubric.model import Rubric

Factory = Callable[[], FakeBackend]
CUES = load_toy_cues()
FAILING_MESSAGE = "connection lost while coding t005"
FIRST_ATTRIBUTE = "specificity"


class QuotaError(BackendError):
    """A backend error subclass, as a custom backend might define one."""


class SlowEarlyBackend(FakeBackend):
    """Answers like the cue backend, but the first requests are the slowest.

    Under threads the first submitted jobs then finish last, so the completion order
    differs from the submission order.
    """

    def __init__(self, rubric: Rubric) -> None:
        super().__init__(cue_responder(rubric, CUES))
        self._counter = itertools.count()
        self._lock = threading.Lock()

    def generate(self, request: GenerationRequest) -> GenerationResult:
        with self._lock:
            position = next(self._counter)
        time.sleep(max(0.0, (24 - position) * 0.001))
        return super().generate(request)


class FailingOnText(FakeBackend):
    """Raises a backend error for every request that carries the given text."""

    def __init__(self, rubric: Rubric, failing_text: str) -> None:
        super().__init__(cue_responder(rubric, CUES))
        self._failing_text = failing_text

    def generate(self, request: GenerationRequest) -> GenerationResult:
        if self._failing_text in request.user:
            raise BackendError(FAILING_MESSAGE)
        return super().generate(request)


def stored_shuffles(path: Path) -> int:
    with sqlite3.connect(path) as connection:
        return int(connection.execute("SELECT COUNT(*) FROM shuffles").fetchone()[0])


def test_concurrent_equals_sequential_when_completion_order_differs(
    toy_rubric: Rubric, toy_corpus: TextTable, cue_backend: Factory, tmp_path: Path
) -> None:
    sequential, concurrent = tmp_path / "a.sqlite", tmp_path / "b.sqlite"
    code_texts(
        toy_corpus, toy_rubric, cue_backend(), sequential,
        CodingSettings(runs=2, concurrency=1, save_every=7),
    )  # fmt: skip
    code_texts(
        toy_corpus, toy_rubric, SlowEarlyBackend(toy_rubric), concurrent,
        CodingSettings(runs=2, concurrency=4, save_every=7),
    )  # fmt: skip
    pd.testing.assert_frame_equal(export_scores(sequential), export_scores(concurrent))
    pd.testing.assert_frame_equal(export_evidence(sequential), export_evidence(concurrent))
    assert stored_shuffles(sequential) == stored_shuffles(concurrent) > 0


@pytest.mark.parametrize("concurrency", [1, 4])
def test_backend_failure_flushes_finished_cells_then_reraises(
    concurrency: int, toy_rubric: Rubric, toy_corpus: TextTable, cue_backend: Factory,
    tmp_path: Path,
) -> None:  # fmt: skip
    store = tmp_path / "r.sqlite"
    settings = CodingSettings(runs=1, save_every=10, concurrency=concurrency)
    backend = FailingOnText(toy_rubric, toy_corpus.text_of("t005"))
    with pytest.raises(BackendError) as error:
        code_texts(toy_corpus, toy_rubric, backend, store, settings)
    assert str(error.value) == f"text 't005', attribute '{FIRST_ATTRIBUTE}': {FAILING_MESSAGE}"
    partial = export_scores(store)
    assert sorted(set(partial.text_id)) == ["t001", "t002", "t003", "t004"]
    assert len(partial) == 12 and stored_shuffles(store) == 12

    resumed = code_texts(toy_corpus, toy_rubric, cue_backend(), store, settings)
    assert resumed.resumed == 4 and resumed.n_texts == 50

    reference = tmp_path / "reference.sqlite"
    code_texts(toy_corpus, toy_rubric, cue_backend(), reference, settings)
    pd.testing.assert_frame_equal(export_scores(store), export_scores(reference))
    pd.testing.assert_frame_equal(export_evidence(store), export_evidence(reference))
    assert stored_shuffles(store) == stored_shuffles(reference)


def test_concurrent_flush_stores_whole_cells_that_match_an_uninterrupted_run(
    toy_rubric: Rubric, toy_corpus: TextTable, cue_backend: Factory, tmp_path: Path
) -> None:
    store, reference = tmp_path / "r.sqlite", tmp_path / "ref.sqlite"
    settings = CodingSettings(runs=1, save_every=10, concurrency=4)
    backend = FailingOnText(toy_rubric, toy_corpus.text_of("t005"))
    with pytest.raises(BackendError, match="connection lost"):
        code_texts(toy_corpus, toy_rubric, backend, store, settings)
    code_texts(toy_corpus, toy_rubric, cue_backend(), reference, settings)
    partial, full = export_scores(store), export_scores(reference)
    assert len(partial) % 3 == 0
    merged = partial.merge(full, how="left", indicator=True, on=list(full.columns))
    assert (merged["_merge"] == "both").all()


def test_failure_in_the_first_cell_writes_nothing(
    toy_rubric: Rubric, toy_corpus: TextTable, tmp_path: Path
) -> None:
    store = tmp_path / "r.sqlite"
    backend = FailingOnText(toy_rubric, toy_corpus.text_of("t001"))
    with pytest.raises(BackendError):
        code_texts(toy_corpus, toy_rubric, backend, store, CodingSettings(runs=1))
    assert export_scores(store).empty


@pytest.mark.parametrize("error_type", [BackendError, QuotaError])
def test_a_backend_error_names_the_text_and_the_attribute(
    error_type: type[BackendError], toy_rubric: Rubric, toy_corpus: TextTable, tmp_path: Path
) -> None:
    def refuse(request: GenerationRequest) -> str:
        raise error_type("prompt exceeds the context window", attempts=3, last_output="partial")

    with pytest.raises(BackendError) as error:
        code_texts(
            toy_corpus.subset(["t003"]), toy_rubric, FakeBackend(refuse), tmp_path / "r.sqlite",
            CodingSettings(runs=1),
        )  # fmt: skip
    assert type(error.value) is error_type
    assert str(error.value) == (
        f"text 't003', attribute '{FIRST_ATTRIBUTE}': prompt exceeds the context window"
    )
    assert (error.value.attempts, error.value.last_output) == (3, "partial")
    assert isinstance(error.value.__cause__, error_type)


def test_the_ceiling_stops_the_run_after_the_first_batch_past_thirty_calls(
    toy_rubric: Rubric, toy_corpus: TextTable, tmp_path: Path
) -> None:
    store = tmp_path / "r.sqlite"
    backend = FakeBackend(lambda request: "")
    with pytest.raises(RunFailureCeilingError) as error:
        code_texts(toy_corpus, toy_rubric, backend, store, CodingSettings(runs=1, save_every=2))
    summary = error.value.summary
    assert (summary.n_texts, summary.n_attribute_calls, summary.store_path) == (10, 30, store)
    assert summary.status_counts["extraction_failed"] == 30
    assert sum(summary.stored_status_counts.values()) == 0
    assert len(backend.calls) == 30 * 3
    assert "30 of 30 attribute calls in this invocation failed" in str(error.value)
    stored = export_scores(store)
    assert sorted(set(stored.text_id)) == [f"t{number:03d}" for number in range(1, 11)]
    assert len(stored) == 30


def test_the_ceiling_waits_for_one_full_batch(
    toy_rubric: Rubric, toy_corpus: TextTable, tmp_path: Path
) -> None:
    store = tmp_path / "r.sqlite"
    backend = FakeBackend(lambda request: "")
    with pytest.raises(RunFailureCeilingError) as error:
        code_texts(toy_corpus, toy_rubric, backend, store, CodingSettings(runs=1, save_every=20))
    assert (error.value.summary.n_texts, error.value.summary.n_attribute_calls) == (20, 60)
    assert len(export_scores(store)) == 60
