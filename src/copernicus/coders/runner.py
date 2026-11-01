"""The runner that codes a text table with a rubric through a backend into a run store.

The runner is resumable (cells already in the store are kept, as long as each text id still
holds the text its cells were coded on; cells with a failed attribute are coded again on
request), shardable (``text_range`` picks a slice of the texts), and optionally concurrent.
The seeds and prompts of a cell do not depend on the order of execution, so a concurrent run
stores the rows a sequential run stores. ``concurrency > 1`` needs a backend whose
``generate`` may be called from several threads: the OpenAI-compatible backend is
thread-safe and gains throughput from it. The vLLM backend serialises its calls with a lock,
so its throughput comes from ``concurrency=1``; the runner does not fan out batches over
``generate_batch``. A ``ScriptedResponder`` is not deterministic under threads.

If the backend fails with a ``BackendError``, the cells finished before the failing one
are written before the error propagates, so a resumed run starts after them; the error
names the text and the attribute that failed. The share of failed attribute calls of the
invocation is checked after every batch, so a broken endpoint stops the run early instead of
being called for the whole corpus, however many good results the store already holds. At
the end the share of failed attribute results in the requested scope, the stored results
the invocation keeps included, is checked, so a resumed run over failed cells does not end
as a success.
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import get_args

from copernicus.coders.rotation import WordingPool, load_wording_pool, select, text_key
from copernicus.coders.runner_jobs import (
    AttributeEvaluator,
    CellPlan,
    assemble_row,
    iter_cell_results,
    plan_cell,
)
from copernicus.coders.runner_scope import Scope, plan_scope
from copernicus.coders.runner_settings import CodingSettings, CodingSummary, resolve_temperature
from copernicus.coders.store import RunIdentity, RunStore, identity_settings
from copernicus.coders.store_cells import FAILURE_STATUSES
from copernicus.coders.text_hash import hash_text
from copernicus.coders.two_stage_types import AttributeResult, Status
from copernicus.data.table import TextTable
from copernicus.errors import BackendError, RunFailureCeilingError, StoreError, TextTableError
from copernicus.llm.base import Backend
from copernicus.rubric.model import Rubric

logger = logging.getLogger(__name__)

__all__ = ["CodingSettings", "CodingSummary", "code_texts", "resolve_temperature"]

# Fewest attribute calls before the failure share is judged during a run.
EARLY_CEILING_MIN_CALLS = 30
CEILING_ADVICE = (
    "set retry_failed=True to recode them, raise max_failure_share to accept them, or code "
    "into a new store"
)
TEMPERATURE_MISMATCH = "mismatch on temperature"
AUTO_TEMPERATURE_RULE = (
    "temperature 'auto' resolves to 0.0 for one run and 0.3 otherwise; "
    "pass an explicit temperature to extend a store"
)
BACKEND_KEY_PREFIX = "backend_"


def _backend_settings(backend: Backend) -> dict[str, str]:
    """Return the backend's provenance as identity settings, each key prefixed ``backend_``.

    Raises:
        TypeError: If a key or value of the provenance is not a string.
    """
    provenance = backend.provenance()
    wrong = sorted(
        repr(key)
        for key, value in provenance.items()
        if not isinstance(key, str) or not isinstance(value, str)
    )
    if wrong:
        raise TypeError(f"backend provenance must map strings to strings; not so for {wrong}")
    return {f"{BACKEND_KEY_PREFIX}{key}": value for key, value in provenance.items()}


def _identity(
    rubric: Rubric,
    backend: Backend,
    settings: CodingSettings,
    pool: WordingPool,
    temperature: float,
) -> RunIdentity:
    """Build the identity of the store from the settings that fix what the results mean."""
    config = settings.two_stage if settings.method == "two_stage" else settings.one_stage
    return RunIdentity(
        method=settings.method,
        model_name=backend.info().model_name,
        temperature=temperature,
        seed=settings.seed,
        rubric_hash=rubric.content_hash(),
        settings={
            **identity_settings(config, pool),
            "shuffle_markers": str(settings.shuffle_markers),
            **_backend_settings(backend),
        },
    )


def _open_store(
    store_path: Path, rubric: Rubric, identity: RunIdentity, settings: CodingSettings
) -> RunStore:
    """Open the store; explain a temperature mismatch that comes from ``"auto"``."""
    try:
        return RunStore(store_path, rubric, identity)
    except StoreError as error:
        if settings.temperature == "auto" and TEMPERATURE_MISMATCH in str(error):
            raise StoreError(f"{error}; {AUTO_TEMPERATURE_RULE}") from error
        raise


def _ordered_texts(
    table: TextTable, text_range: tuple[int, int] | None
) -> list[tuple[str, object]]:
    """Return ``(text id as string, original id)`` in string order, cut to the range."""
    pairs = sorted(((str(raw), raw) for raw in table.ids()), key=lambda pair: pair[0])
    if len({text_id for text_id, _ in pairs}) != len(pairs):
        raise TextTableError("text ids must stay distinct when converted to strings")
    if text_range is None:
        return pairs
    return pairs[text_range[0] : text_range[1]]


class _Totals:
    """Running counts of the attribute results written by one invocation of a scope."""

    def __init__(self, scope: Scope) -> None:
        self.scope = scope
        self.statuses: Counter[str] = Counter()
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.retried = 0

    def add(self, plan: CellPlan, results: Sequence[AttributeResult]) -> None:
        """Fold the results of one written cell into the totals."""
        self.statuses.update(result.status for result in results)
        self.prompt_tokens += sum(result.prompt_tokens for result in results)
        self.completion_tokens += sum(result.completion_tokens for result in results)
        if (plan.text_id, plan.run_number) in self.scope.retry:
            self.retried += 1

    def calls(self) -> int:
        """Return the number of attribute results counted so far."""
        return sum(self.statuses.values())

    def summary(self, n_texts: int, n_runs: int, started: float, store_path: Path) -> CodingSummary:
        """Describe what the invocation did so far."""
        return CodingSummary(
            n_texts=n_texts,
            n_runs=n_runs,
            n_attribute_calls=self.calls(),
            status_counts={status: self.statuses[status] for status in get_args(Status)},
            stored_status_counts=dict(self.scope.stored_status_counts),
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            seconds=time.perf_counter() - started,
            store_path=store_path,
            resumed=self.scope.resumed,
            retried=self.retried,
        )


def _write_batch(
    store: RunStore, finished: Sequence[tuple[CellPlan, list[AttributeResult]]], totals: _Totals
) -> None:
    """Write the finished cells and their shuffles in one transaction and count them.

    A cell that is coded again replaces its stored row and evidence in that transaction.
    """
    rows = [assemble_row(plan, results) for plan, results in finished]
    shuffles = [
        (plan.assignment.shuffle_seed, context.attribute.attribute_id, context.marker_order)
        for plan, _ in finished
        for context in plan.contexts
    ]
    retried = any((plan.text_id, plan.run_number) in totals.scope.retry for plan, _ in finished)
    store.insert(rows, shuffles, replace_failed=retried)
    for plan, results in finished:
        totals.add(plan, results)


def _code_batch(
    store: RunStore,
    plans: Sequence[CellPlan],
    evaluator: AttributeEvaluator,
    concurrency: int,
    totals: _Totals,
) -> None:
    """Code one batch of cells and write it; on a backend failure write what finished first."""
    finished: list[tuple[CellPlan, list[AttributeResult]]] = []
    try:
        for cell in iter_cell_results(plans, evaluator, concurrency):
            finished.append(cell)
    except BackendError:
        if finished:
            _write_batch(store, finished, totals)
        raise
    _write_batch(store, finished, totals)


def code_texts(
    table: TextTable,
    rubric: Rubric,
    backend: Backend,
    store_path: str | Path,
    settings: CodingSettings,
) -> CodingSummary:
    """Code a text table with a rubric through a backend into a run store.

    The texts are taken in text-id order (ids compared as strings) and cut to
    ``settings.text_range``; with the run numbers 1 to ``settings.runs`` they form the
    requested scope. A cell of the scope that the store does not hold is coded; a stored
    cell is kept, unless it holds a failed attribute and ``settings.retry_failed`` is set,
    in which case it is coded again and replaces the stored one. Before any backend call,
    the text hash of every text of the range is compared with the stored cells of its id,
    so a store never keeps results of another text under an id. Rows are written every
    ``settings.save_every`` texts and at the end, so an interrupted run loses at most the
    current batch and resumes from the store. If the backend raises a
    `copernicus.errors.BackendError`, the cells of the current batch that finished
    before the failing one are written and the error is re-raised with the text id and
    the attribute in front of its message. After every batch, once at least
    ``EARLY_CEILING_MIN_CALLS`` (30) attribute calls and one full batch have been made, the
    share of extraction and scoring failures among this invocation's attribute calls is
    compared with ``settings.max_failure_share``; at the end, the share among the attribute
    results of the scope (this invocation's and the stored ones it keeps) is compared.

    Args:
        table: The texts to code.
        rubric: The rubric to code them with.
        backend: The LLM backend; with ``settings.concurrency > 1`` it must be thread-safe.
        store_path: Location of the run store, a ``str`` or a ``Path``; created if absent. An
            existing store must have been made with the same method, model, temperature,
            seed, rubric, coder config, wording, and marker shuffling, and with a backend of
            the same provenance (see `copernicus.llm.base.Backend.provenance`).
        settings: Settings of the run.

    Returns:
        A summary of this invocation.

    Raises:
        TypeError: If the backend's provenance holds a key or value that is not a string.
        TextTableError: If two text ids of the table are equal as strings.
        PromptContractError: If the wording selection of the settings leaves a dimension
            without a variant or names an unknown variant.
        StoreError: If the existing store belongs to another run identity (the message
            explains the rule when ``temperature="auto"`` caused the mismatch), was made by
            an earlier version of the library or changed by hand, or holds a cell of a text of
            the range that was coded on another text (the first ten ids are named; nothing is
            coded).
        BackendError: If the backend fails; the finished cells are already written, and
            the type, ``attempts``, and ``last_output`` of the backend's error are kept.
        RunFailureCeilingError: If the share of extraction and scoring failures exceeds
            ``settings.max_failure_share``: among this invocation's attribute calls at a
            check after a batch, or among the attribute results of the scope at the end,
            also when the invocation codes nothing. The rows coded so far are written and
            the error's ``summary`` describes them and the stored results of the scope.
    """
    started = time.perf_counter()
    path = Path(store_path)
    pool = select(load_wording_pool(), settings.wording_enable, settings.wording_disable)
    temperature = resolve_temperature(settings.temperature, settings.runs)
    evaluator = AttributeEvaluator(
        backend, settings.method, settings.two_stage, settings.one_stage, temperature
    )
    texts = _ordered_texts(table, settings.text_range)
    hashes = {text_id: hash_text(table.text_of(raw_id)) for text_id, raw_id in texts}
    identity = _identity(rubric, backend, settings, pool, temperature)
    early_ceiling_calls = max(
        EARLY_CEILING_MIN_CALLS, settings.save_every * settings.runs * len(rubric.attributes)
    )
    with _open_store(path, rubric, identity, settings) as store:
        scope = plan_scope(
            texts, hashes, settings.runs, store.completed_cells(), settings.retry_failed, path
        )
        totals = _Totals(scope)
        store.record_wording(pool)
        for start in range(0, len(scope.pending), settings.save_every):
            batch = scope.pending[start : start + settings.save_every]
            plans = [
                plan_cell(
                    rubric,
                    pool,
                    text_id=text_id,
                    text=table.text_of(raw_id),
                    text_key=text_key(raw_id),
                    run_number=run,
                    run_seed=settings.seed,
                    shuffle_markers=settings.shuffle_markers,
                )
                for text_id, raw_id, runs in batch
                for run in runs
            ]
            _code_batch(store, plans, evaluator, settings.concurrency, totals)
            n_coded = start + len(batch)
            logger.info("coded %d of %d texts", n_coded, len(scope.pending))
            if totals.calls() >= early_ceiling_calls:
                _enforce_early_ceiling(
                    totals.summary(n_coded, settings.runs, started, path),
                    settings.max_failure_share,
                )
    summary = totals.summary(len(scope.pending), settings.runs, started, path)
    _enforce_ceiling(summary, settings.max_failure_share)
    return summary


def _enforce_early_ceiling(summary: CodingSummary, max_failure_share: float) -> None:
    """Raise, carrying the summary, if too many of this invocation's attribute calls failed.

    The checks during a run judge the invocation's own calls alone, so the stored results
    of the scope cannot hide a backend that fails now: it stops after the first judged batch,
    however many good results the store holds.
    """
    failed = sum(summary.status_counts[status] for status in FAILURE_STATUSES)
    calls = summary.n_attribute_calls
    if calls and failed / calls > max_failure_share:
        raise RunFailureCeilingError(
            f"{failed} of {calls} attribute calls in this invocation failed (share "
            f"{failed / calls:.3f} above the ceiling {max_failure_share}), so the run stopped "
            f"early; check the backend, then {CEILING_ADVICE}; the results coded so far are "
            f"stored in {summary.store_path}",
            summary,
        )


def _enforce_ceiling(summary: CodingSummary, max_failure_share: float) -> None:
    """Raise, carrying the summary, if too many attribute results of the scope failed.

    This is the check at the end of a run. The results this invocation coded and the stored
    results of the scope it kept count alike, in the numerator and in the denominator.
    """
    stored_failed = sum(summary.stored_status_counts[status] for status in FAILURE_STATUSES)
    failed = stored_failed + sum(summary.status_counts[status] for status in FAILURE_STATUSES)
    total = summary.n_attribute_calls + sum(summary.stored_status_counts.values())
    if total and failed / total > max_failure_share:
        raise RunFailureCeilingError(
            f"{failed} of {total} attribute results in the requested scope failed, "
            f"{stored_failed} of them stored before this invocation (share "
            f"{failed / total:.3f} above the ceiling {max_failure_share}); {CEILING_ADVICE}; "
            f"the results coded so far are stored in {summary.store_path}",
            summary,
        )
