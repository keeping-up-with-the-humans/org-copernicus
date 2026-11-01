"""The reference-text generator: texts for every pole of every target, through any backend."""

from __future__ import annotations

import gc
import hashlib
import logging
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from copernicus.errors import ConfigurationError
from copernicus.llm.base import Backend
from copernicus.references.answers import CallOutcome, attempt_request, finish_call
from copernicus.references.binding import BOUND_GENERATED
from copernicus.references.cells import Cell, build_cells
from copernicus.references.progress import GenerationProgress
from copernicus.references.prompts import PROMPT_VERSION, system_prompt, user_prompt
from copernicus.references.reference_set import ReferenceSet, Shortfall
from copernicus.references.screen import leakage_terms, screen_report, screen_texts
from copernicus.references.settings import GeneratorSettings
from copernicus.rubric.model import Rubric

logger = logging.getLogger(__name__)

__all__ = [
    "DeferredWriter",
    "GeneratorSettings",
    "ReferenceSet",
    "Writer",
    "generate_references",
    "plan_calls",
]

CALL_SEED_STEP = 1000
PROMPT_HASH_LENGTH = 16
NO_WRITER_MESSAGE = "reference texts need at least one writer model"


@dataclass(frozen=True)
class DeferredWriter:
    """A writer model that is built when its turn comes and dropped when its calls are done.

    An in-process engine such as vLLM holds the memory of its GPU; writers built one after
    the other can share a GPU within one run. When the calls of a writer are done, its
    backend is dropped, the garbage collector runs (an engine held in a reference cycle is
    freed only then), and, when torch is loaded, its CUDA cache is emptied, so the next
    writer finds the memory free. A writer whose calls a resume file already holds is not
    built at all.

    Attributes:
        model_name: Name of the writer model; it labels the rows and must equal the
            ``info().model_name`` of the backend that ``build`` returns.
        build: Builds the backend.
    """

    model_name: str
    build: Callable[[], Backend]


Writer = Backend | DeferredWriter


def plan_calls(rubric: Rubric, writers: Sequence[Writer], settings: GeneratorSettings) -> int:
    """Count the generation calls of a run, before it starts.

    Args:
        rubric: The rubric whose cells get texts.
        writers: The writer models.
        settings: The generation settings.

    Returns:
        Cells times writers times ``calls_per_cell``; retries come on top.

    Raises:
        ConfigurationError: If there is no writer.
    """
    if not writers:
        raise ConfigurationError(NO_WRITER_MESSAGE)
    return len(build_cells(rubric)) * len(writers) * settings.calls_per_cell


def _release_memory() -> None:
    """Free what a dropped deferred writer held before the next writer is built.

    The garbage collector frees an engine that sits in a reference cycle, and the CUDA
    cache of torch is emptied when torch is loaded and a GPU is available; torch is never
    imported here.
    """
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _prompt_hash(system: str, user: str) -> str:
    """Fingerprint a pair of prompts: the first 16 hex characters of the SHA-256."""
    return hashlib.sha256((system + "\n" + user).encode("utf-8")).hexdigest()[:PROMPT_HASH_LENGTH]


def _check_writers(writers: Sequence[Writer]) -> list[str]:
    """Return the model names of the writers, refusing none or repeated names."""
    if not writers:
        raise ConfigurationError(NO_WRITER_MESSAGE)
    names = [
        writer.model_name if isinstance(writer, DeferredWriter) else writer.info().model_name
        for writer in writers
    ]
    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        raise ConfigurationError(f"writer model names must be unique; repeated: {repeated}")
    return names


def _build(writer: Writer, name: str) -> Backend:
    """Return the backend of a writer, building a deferred one.

    Raises:
        ConfigurationError: If a deferred writer builds a backend of another model name.
    """
    if not isinstance(writer, DeferredWriter):
        return writer
    backend = writer.build()
    built = backend.info().model_name
    if built != name:
        raise ConfigurationError(f"the deferred writer {name!r} built a backend of model {built!r}")
    return backend


@dataclass(frozen=True)
class _Run:
    """What every call of a run shares: the rubric, settings, cells, and their prompts."""

    rubric: Rubric
    settings: GeneratorSettings
    cells: list[Cell]
    system: str
    users: list[str]
    hashes: list[str]

    def seed(self, cell_index: int, call_index: int) -> int:
        """Return the seed of the first attempt of a call."""
        return self.settings.seed + CALL_SEED_STEP * call_index + cell_index

    def rows(
        self, name: str, cell_index: int, call_index: int, outcome: CallOutcome
    ) -> list[dict[str, Any]]:
        """Return one table row per kept text of a call."""
        cell = self.cells[cell_index]
        return [
            {
                "rubric_id": self.rubric.rubric_id,
                "target": cell.target,
                "polarity": cell.polarity,
                "level": cell.level,
                "style": self.settings.style,
                "writer_model": name,
                "text": text,
                "call_index": call_index,
                "attempt": outcome.attempt,
                "seed": outcome.seed,
                "prompt_tokens": outcome.result.prompt_tokens,
                "completion_tokens": outcome.result.completion_tokens,
                "temperature_applied": outcome.result.temperature_applied,
                "top_p_applied": outcome.result.top_p_applied,
                "prompt_hash": self.hashes[cell_index],
                "prompt_version": PROMPT_VERSION,
            }
            for text in outcome.texts
        ]


def _run_writer(
    backend: Backend,
    name: str,
    pending: list[tuple[int, int]],
    run: _Run,
    progress: GenerationProgress,
) -> None:
    """Make the pending calls of one writer, recording each as soon as it is finished.

    The first attempts go to the writer ``concurrency`` at a time, through
    ``generate_batch`` when there are several; the retries of each call follow one at a
    time. A failing request raises after every call finished before it was recorded.
    """
    calls_per_cell = run.settings.calls_per_cell
    for start in range(0, len(pending), run.settings.concurrency):
        chunk = pending[start : start + run.settings.concurrency]
        requests = [
            attempt_request(
                run.system, run.users[cell_index], run.settings, run.seed(cell_index, call), 0
            )
            for cell_index, call in chunk
        ]
        results = (
            backend.generate_batch(requests)
            if len(requests) > 1
            else [backend.generate(requests[0])]
        )
        for (cell_index, call_index), result in zip(chunk, results, strict=True):
            cell = run.cells[cell_index]
            label = (
                f"writer {name}, target {cell.target}, {cell.level} {cell.polarity}, "
                f"call {call_index + 1}/{calls_per_cell}"
            )
            seed = run.seed(cell_index, call_index)
            user = run.users[cell_index]
            outcome = finish_call(backend, run.system, user, seed, result, run.settings, label)
            rows = run.rows(name, cell_index, call_index, outcome)
            progress.add_call(name, cell_index, call_index, outcome, rows)


def _shortfalls(frame: pd.DataFrame, names: list[str], run: _Run) -> tuple[Shortfall, ...]:
    """Find the cells of each writer that kept fewer texts than requested over their calls."""
    expected = run.settings.texts_per_call * run.settings.calls_per_cell
    counts = frame.groupby(["writer_model", "target", "polarity", "level"]).size()
    found = {key: int(count) for key, count in counts.items()}
    shortfalls: list[Shortfall] = []
    for name in names:
        for cell in run.cells:
            key = (name, cell.target, cell.polarity, cell.level)
            received = found[key] if key in found else 0
            if received < expected:
                logger.warning(
                    "writer %s, target %s, %s %s: %d of %d texts",
                    name,
                    cell.target,
                    cell.level,
                    cell.polarity,
                    received,
                    expected,
                )
                shortfalls.append(
                    Shortfall(name, cell.target, cell.polarity, cell.level, expected, received)
                )
    return tuple(shortfalls)


def generate_references(
    rubric: Rubric,
    writers: Sequence[Writer],
    settings: GeneratorSettings,
    *,
    screen: bool = True,
    resume: Path | None = None,
) -> ReferenceSet:
    """Write reference texts for every pole of every target of a rubric.

    For each writer, cell, and call the writer is asked for ``texts_per_call`` texts. An
    answer is accepted when it is a JSON array that holds at least that many objects with a
    non-empty ``text`` (a surplus is cut to the first ones; of an array that the output cut
    off, the complete objects count); otherwise the call is repeated, up to ``max_retries``
    attempts, and the usable texts of the last attempt are kept. A call's seed is
    ``seed + 1000 * call_index + cell_index``; attempt ``k`` adds ``100000 * k``. The first
    attempts go to a writer ``concurrency`` at a time; retries go one at a time.

    With ``resume``, every finished call is written to that file at once, so a run stopped
    by a failing backend or by the user keeps every call it recorded, and a rerun with the
    same file makes only the calls the file lacks. The table is written before its
    description, which lists the call: a stop between the two writes costs that one call,
    which the rerun makes again, and a stop between the two writes of the first call
    leaves a table that the rerun refuses until it is deleted. With ``concurrency`` above
    1 the first attempts of a batch are answered together, so a failing request of the
    batch, or a failing retry of one of its calls, discards the answers of the batch that
    were not yet recorded: up to ``concurrency - 1`` first-attempt answers, paid for but
    missing from the file and from ``usage``. A concurrency of 1 loses none. The file is
    kept after the run: a rerun with a complete file makes no call. The result does not
    depend on whether, or how often, a run was resumed.

    Args:
        rubric: The rubric whose construct and attributes get reference texts.
        writers: The writer models, as backends or as `DeferredWriter`, which are
            built one at a time; their model names label the rows and must differ.
        settings: The generation settings.
        screen: Whether to flag texts that quote the vocabulary of the rubric; flagged rows
            stay in the table with ``screened_out`` set.
        resume: A ``.parquet`` or ``.csv`` file for the finished calls, in an existing
            folder; ``None`` keeps them in memory only.

    Returns:
        The reference set, bound to the content hash of ``rubric`` (``rubric_hash``, with
        ``bound_with`` ``"generated"``), with the cells that fell short recorded in
        ``shortfalls``, the provenance of every writer in ``writers``, and the calls,
        attempts, and tokens of every writer in ``usage``.

    Raises:
        ConfigurationError: If there is no writer, two writers share a model name, a
            deferred writer builds another model, or the resume file cannot be written or
            belongs to another rubric, other settings, other writers, or other prompts.
        BackendError: If a backend fails to answer a request; the calls recorded before
            are in the resume file, and with ``concurrency`` above 1 the answers of the
            failing batch that were not yet recorded, up to ``concurrency - 1``, are lost.
    """
    names = _check_writers(writers)
    cells = build_cells(rubric)
    system = system_prompt(settings.style, settings.texts_per_call, settings.register_description)
    users = [user_prompt(cell, rubric, settings.texts_per_call) for cell in cells]
    run = _Run(rubric, settings, cells, system, users, [_prompt_hash(system, u) for u in users])
    progress = GenerationProgress.open(resume, rubric, settings, names, cells, run.hashes)
    logger.info(
        "generating reference texts: %d calls (%d cells x %d writers x %d calls per cell), "
        "%d of them finished before",
        plan_calls(rubric, writers, settings),
        len(cells),
        len(writers),
        settings.calls_per_cell,
        progress.finished,
    )
    for writer, name in zip(writers, names, strict=True):
        pending = progress.pending(name)
        if not pending:
            logger.info("writer %s: every call is finished", name)
            continue
        backend = _build(writer, name)
        progress.record_writer(name, backend.provenance())
        _run_writer(backend, name, pending, run, progress)
        del backend
        if isinstance(writer, DeferredWriter):
            _release_memory()
    frame = progress.frame()
    shortfalls = _shortfalls(frame, names, run)
    if screen:
        frame = screen_texts(frame, leakage_terms(rubric))
        report = screen_report(frame)
        logger.info(
            "leakage screen: %d of %d texts screened out %s",
            report["n_screened_out"],
            report["n_texts"],
            report["by_term"],
        )
    else:
        frame["screened_out"] = False
        frame["screen_reason"] = ""
    return ReferenceSet(
        frame,
        rubric.rubric_id,
        rubric.content_hash(),
        settings,
        shortfalls,
        progress.writers(),
        progress.usage(),
        BOUND_GENERATED,
    )
