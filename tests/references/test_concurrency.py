"""Tests of the concurrency of reference generation and of writers built one at a time."""

from __future__ import annotations

import gc
import sys
import types
import weakref
from collections.abc import Iterator, Sequence
from pathlib import Path

import pandas as pd
import pytest
from references_support import clean_responder, seed_responder, writer

from copernicus.errors import ConfigurationError
from copernicus.llm.base import Backend, GenerationRequest, GenerationResult
from copernicus.llm.fake import FakeBackend
from copernicus.references import (
    DeferredWriter,
    GeneratorSettings,
    estimate_tokens,
    generate_references,
    plan_calls,
)
from copernicus.rubric.model import Rubric

CELL_COUNT = 24


class BatchRecorder(FakeBackend):
    """A fake writer that records the size of every batch it is sent."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self.batches: list[int] = []

    def generate_batch(self, requests: Sequence[GenerationRequest]) -> list[GenerationResult]:
        self.batches.append(len(requests))
        return super().generate_batch(requests)


class CyclicWriter(FakeBackend):
    """A fake writer that refers to itself, like an engine only the cyclic collector frees."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self.cycle = self


@pytest.fixture
def automatic_collection_paused() -> Iterator[None]:
    """Pause the automatic garbage collection, so only an explicit collection frees a cycle."""
    was_enabled = gc.isenabled()
    gc.disable()
    yield
    if was_enabled:
        gc.enable()


def test_first_attempts_go_through_generate_batch_in_chunks_and_retries_one_by_one(
    toy_rubric: Rubric,
) -> None:
    backend = BatchRecorder(seed_responder({42: "not json"}), model_name="writer-a")
    batched = generate_references(toy_rubric, [backend], GeneratorSettings(concurrency=5))
    assert backend.batches == [5, 5, 5, 5, 4]
    assert len(backend.calls) == CELL_COUNT + 1
    assert backend.calls[5].seed == 100042
    single = generate_references(
        toy_rubric, [writer(seed_responder({42: "not json"}))], GeneratorSettings()
    )
    pd.testing.assert_frame_equal(batched.frame, single.frame)


def test_a_concurrency_of_one_sends_single_requests(toy_rubric: Rubric) -> None:
    backend = BatchRecorder(clean_responder(), model_name="writer-a")
    generate_references(toy_rubric, [backend], GeneratorSettings())
    assert backend.batches == [] and len(backend.calls) == CELL_COUNT


def test_deferred_writers_are_built_one_after_the_other(toy_rubric: Rubric) -> None:
    built: list[weakref.ref[Backend]] = []

    def build(name: str) -> Backend:
        assert all(reference() is None for reference in built), "an earlier writer is alive"
        backend = writer(clean_responder(), name)
        built.append(weakref.ref(backend))
        return backend

    writers = [
        DeferredWriter("model-x", lambda: build("model-x")),
        DeferredWriter("model-y", lambda: build("model-y")),
    ]
    result = generate_references(toy_rubric, writers, GeneratorSettings())
    assert len(built) == 2 and all(reference() is None for reference in built)
    assert result.frame.groupby("writer_model").size().to_dict() == {
        "model-x": CELL_COUNT * 10,
        "model-y": CELL_COUNT * 10,
    }
    assert set(result.writers) == {"model-x", "model-y"}


@pytest.mark.usefixtures("automatic_collection_paused")
def test_a_dropped_deferred_writer_is_collected_before_the_next_one_is_built(
    toy_rubric: Rubric, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "torch", None)  # importing torch now fails
    collector_runs: list[int] = []
    collect = gc.collect

    def counting_collect(generation: int = 2) -> int:
        collector_runs.append(generation)
        return collect(generation)

    monkeypatch.setattr(gc, "collect", counting_collect)
    built: list[weakref.ref[Backend]] = []

    def build(name: str) -> Backend:
        assert all(reference() is None for reference in built), "an earlier writer is alive"
        backend = CyclicWriter(clean_responder(), model_name=name)
        built.append(weakref.ref(backend))
        return backend

    writers = [DeferredWriter(name, lambda name=name: build(name)) for name in ("x", "y")]
    generate_references(toy_rubric, writers, GeneratorSettings())
    assert len(built) == 2 and all(reference() is None for reference in built)
    assert len(collector_runs) == 2


@pytest.mark.parametrize("cuda", [True, False])
def test_the_cuda_cache_is_emptied_after_each_deferred_writer_when_torch_is_loaded(
    toy_rubric: Rubric, monkeypatch: pytest.MonkeyPatch, cuda: bool
) -> None:
    emptied: list[bool] = []
    fake_cuda = types.SimpleNamespace(
        is_available=lambda: cuda, empty_cache=lambda: emptied.append(True)
    )
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(cuda=fake_cuda))
    deferred = [
        DeferredWriter(name, lambda name=name: writer(clean_responder(), name))
        for name in ("x", "y")
    ]
    generate_references(
        toy_rubric, [*deferred, writer(clean_responder(), "z")], GeneratorSettings()
    )
    assert emptied == ([True, True] if cuda else [])


def test_a_deferred_writer_whose_calls_are_all_finished_is_not_built(
    toy_rubric: Rubric, tmp_path: Path
) -> None:
    path = tmp_path / "progress.parquet"
    generate_references(toy_rubric, [writer(clean_responder())], GeneratorSettings(), resume=path)

    def never() -> Backend:
        raise AssertionError("a finished writer was built")

    again = generate_references(
        toy_rubric, [DeferredWriter("writer-a", never)], GeneratorSettings(), resume=path
    )
    assert len(again.frame) == CELL_COUNT * 10


def test_a_deferred_writer_must_build_the_model_it_names(toy_rubric: Rubric) -> None:
    deferred = DeferredWriter("model-x", lambda: writer(clean_responder(), "model-z"))
    with pytest.raises(ConfigurationError, match="model-x.*model-z"):
        generate_references(toy_rubric, [deferred], GeneratorSettings())


def test_deferred_writers_count_in_the_plan_and_the_estimate(toy_rubric: Rubric) -> None:
    deferred = [DeferredWriter(name, lambda: writer(clean_responder())) for name in ("a", "b")]
    assert plan_calls(toy_rubric, deferred, GeneratorSettings()) == CELL_COUNT * 2
    assert estimate_tokens(toy_rubric, GeneratorSettings(), n_writers=2).calls == CELL_COUNT * 2
    with pytest.raises(ConfigurationError, match="unique"):
        generate_references(
            toy_rubric, [deferred[0], writer(clean_responder(), "a")], GeneratorSettings()
        )
