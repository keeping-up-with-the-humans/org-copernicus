"""Cells, jobs, and rows of the coding runner.

A cell is one text in one run. Planning a cell fixes everything that does not depend on
the order of execution: the wording, the marker order of each attribute, and the seed. A
job codes one attribute of one cell, so the jobs of a batch can run in any order and still
give the rows a sequential run would give.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Literal

from copernicus.coders.one_stage import OneStageConfig, evaluate_attribute_one_stage
from copernicus.coders.render import PromptContext
from copernicus.coders.rotation import (
    Assignment,
    WordingPool,
    assign,
    decode_seed,
    marker_order,
)
from copernicus.coders.store_types import ResultRow
from copernicus.coders.text_hash import hash_text
from copernicus.coders.two_stage import evaluate_attribute
from copernicus.coders.two_stage_types import AttributeResult, TwoStageConfig
from copernicus.errors import BackendError
from copernicus.llm.base import Backend
from copernicus.rubric.model import Rubric

logger = logging.getLogger(__name__)

__all__ = ["AttributeEvaluator", "CellPlan", "assemble_row", "iter_cell_results", "plan_cell"]


@dataclass(frozen=True)
class CellPlan:
    """Everything needed to code one text in one run.

    Attributes:
        text_id: Identifier of the text, as a string.
        text: The normalised text.
        text_key: Integer key of the text used for seeding.
        run_number: Number of the run, starting at 1.
        assignment: Wording and shuffle seed drawn for this text and run.
        seed: Sampling seed of the run for every attribute call.
        contexts: One prompt context per attribute, in rubric order.
    """

    text_id: str
    text: str
    text_key: int
    run_number: int
    assignment: Assignment
    seed: int
    contexts: tuple[PromptContext, ...]


def plan_cell(
    rubric: Rubric,
    pool: WordingPool,
    *,
    text_id: str,
    text: str,
    text_key: int,
    run_number: int,
    run_seed: int,
    shuffle_markers: bool,
) -> CellPlan:
    """Draw the wording and marker orders of one text in one run.

    Args:
        rubric: The rubric to code.
        pool: The wording pool the rotation draws from.
        text_id: Identifier of the text, as a string.
        text: The normalised text.
        text_key: Integer key of the text used for seeding.
        run_number: Number of the run, starting at 1.
        run_seed: Seed of the run.
        shuffle_markers: Whether the markers of each attribute are shuffled.

    Returns:
        The plan of the cell.
    """
    assignment = assign(pool, text_key, run_number, run_seed)
    contexts = tuple(
        PromptContext(
            rubric,
            attribute,
            marker_order(
                rubric.marker_names(attribute.attribute_id),
                shuffle_markers,
                assignment.shuffle_seed,
            ),
            pool.get("role", assignment.role_id),
            pool.get("task", assignment.task_id),
            pool.get("level", assignment.level_id),
        )
        for attribute in rubric.attributes
    )
    return CellPlan(
        text_id=text_id,
        text=text,
        text_key=text_key,
        run_number=run_number,
        assignment=assignment,
        seed=decode_seed(run_seed, run_number),
        contexts=contexts,
    )


class AttributeEvaluator:
    """Codes one attribute of one cell with the coder chosen for the run.

    The evaluator holds no state that changes between calls; it is as thread-safe as
    its backend.
    """

    def __init__(
        self,
        backend: Backend,
        method: Literal["two_stage", "one_stage"],
        two_stage: TwoStageConfig,
        one_stage: OneStageConfig,
        temperature: float,
    ) -> None:
        """Initialise the evaluator.

        Args:
            backend: The LLM backend.
            method: ``"two_stage"`` or ``"one_stage"``.
            two_stage: Config of the two-stage LLM coder.
            one_stage: Config of the one-stage LLM coder.
            temperature: Resolved sampling temperature, used for every call.
        """
        self._backend = backend
        self._method = method
        self._two_stage = two_stage
        self._one_stage = one_stage
        self._temperature = temperature

    def __call__(self, plan: CellPlan, context: PromptContext) -> AttributeResult:
        """Code the attribute of the context in the cell.

        Args:
            plan: The cell.
            context: The prompt context of one attribute of the cell.

        Returns:
            The attribute result.

        Raises:
            BackendError: If the backend fails; the error is raised again with the text id
                and the attribute in front of its message, with the same type (a subclass
                must accept the constructor arguments of ``BackendError``), ``attempts``,
                and ``last_output``, and the original as its cause.
        """
        try:
            return self._evaluate(plan, context)
        except BackendError as error:
            located = f"text {plan.text_id!r}, attribute {context.attribute.attribute_id!r}"
            raise type(error)(
                f"{located}: {error}", attempts=error.attempts, last_output=error.last_output
            ) from error

    def _evaluate(self, plan: CellPlan, context: PromptContext) -> AttributeResult:
        """Run the coder of the run's method on one attribute of the cell."""
        if self._method == "two_stage":
            return evaluate_attribute(
                self._backend,
                plan.text,
                context,
                self._two_stage,
                temperature_stage1=self._temperature,
                temperature_stage2=self._temperature,
                seed=plan.seed,
            )
        return evaluate_attribute_one_stage(
            self._backend,
            plan.text,
            context,
            self._one_stage,
            temperature=self._temperature,
            seed=plan.seed,
        )


def assemble_row(plan: CellPlan, results: Sequence[AttributeResult]) -> ResultRow:
    """Combine the attribute results of one cell into a store row.

    Args:
        plan: The cell.
        results: One result per attribute, in the order of ``plan.contexts``.

    Returns:
        The row; its text hash is that of ``plan.text``, the text the coder received, and
        its seconds and token counts are the sums over the attributes.
    """
    return ResultRow(
        text_id=plan.text_id,
        text_key=plan.text_key,
        text_hash=hash_text(plan.text),
        run_number=plan.run_number,
        assignment=plan.assignment,
        attributes={result.attribute_id: result for result in results},
        seconds=sum(result.seconds for result in results),
        prompt_tokens=sum(result.prompt_tokens for result in results),
        completion_tokens=sum(result.completion_tokens for result in results),
    )


def iter_cell_results(
    plans: Sequence[CellPlan], evaluator: AttributeEvaluator, concurrency: int
) -> Iterator[tuple[CellPlan, list[AttributeResult]]]:
    """Code the plans and yield each finished cell in the order of ``plans``.

    With ``concurrency == 1`` the attributes are coded one after the other. Otherwise
    every attribute of every plan is submitted to a thread pool at once and the futures
    are read in submission order, so the cells come out in the order of a sequential
    run whatever the order of completion. A cell is yielded only after all its
    attributes finished.

    If an evaluation raises, the exception propagates from the iterator after every
    cell before the failing one has been yielded; in the concurrent case the jobs not
    yet started are cancelled and the running ones are awaited before it propagates.

    Args:
        plans: The cells to code.
        evaluator: Codes one attribute of one cell.
        concurrency: Number of evaluations in flight.

    Yields:
        ``(plan, results)`` with one result per attribute in the order of ``plan.contexts``.
    """
    if concurrency == 1:
        for plan in plans:
            yield plan, [evaluator(plan, context) for context in plan.contexts]
        return
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures: list[list[Future[AttributeResult]]] = [
            [executor.submit(evaluator, plan, context) for context in plan.contexts]
            for plan in plans
        ]
        try:
            for plan, plan_futures in zip(plans, futures, strict=True):
                yield plan, [future.result() for future in plan_futures]
        finally:
            for plan_futures in futures:
                for future in plan_futures:
                    future.cancel()
