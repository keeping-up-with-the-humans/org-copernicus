"""Shared fixtures and helpers for the tests of the two-stage LLM coder."""

import json
from collections.abc import Sequence

from copernicus.coders import (
    AttributeResult,
    PromptContext,
    TwoStageConfig,
    evaluate_attribute,
    load_wording_pool,
    select,
)
from copernicus.coders.rotation import WordingPool
from copernicus.llm.base import Backend, BackendInfo, GenerationRequest, GenerationResult
from copernicus.rubric.model import Rubric

DOC = (
    "The team met on Monday. We decided to paint the garden shed with two colours first. "
    "Everyone agreed."
)
PASSAGE = "We decided to paint the garden shed"
PINNED = ["role_baseline", "task_baseline", "level_independent"]
OK_SCORE = json.dumps({"result": 4, "explanation": "clear evidence"})


def make_ctx(rubric: Rubric, attr_index: int = 0, pool: WordingPool | None = None) -> PromptContext:
    chosen = pool if pool is not None else select(load_wording_pool(), enable=PINNED)
    attribute = rubric.attributes[attr_index]
    order = [marker.name for marker in attribute.markers]
    return PromptContext(
        rubric,
        attribute,
        order,
        chosen.get("role", "role_baseline"),
        chosen.get("task", "task_baseline"),
        chosen.get("level", "level_independent"),
    )


def names_of(rubric: Rubric, attr_index: int = 0) -> list[str]:
    return [marker.name for marker in rubric.attributes[attr_index].markers]


def run(
    backend: Backend,
    rubric: Rubric,
    config: TwoStageConfig | None = None,
    seed: int = 1,
) -> AttributeResult:
    return evaluate_attribute(
        backend,
        DOC,
        make_ctx(rubric),
        config if config is not None else TwoStageConfig(),
        temperature_stage1=0.3,
        temperature_stage2=0.3,
        seed=seed,
    )


def evidence_json(names: list[str], quote: str = PASSAGE) -> str:
    return json.dumps({n: {"evaluable": True, "quotes": [quote]} for n in names})


def no_evidence_json(names: list[str]) -> str:
    return json.dumps({n: {"evaluable": False, "quotes": []} for n in names})


class CountingBackend:
    """A backend that answers in order and reports fixed token counts (or none)."""

    def __init__(
        self,
        answers: Sequence[str],
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
    ) -> None:
        self._answers = list(answers)
        self._prompt_tokens = prompt_tokens
        self._completion_tokens = completion_tokens
        self.calls: list[GenerationRequest] = []

    def generate(self, request: GenerationRequest) -> GenerationResult:
        self.calls.append(request)
        return GenerationResult(
            self._answers.pop(0), self._prompt_tokens, self._completion_tokens, None, None, None
        )

    def generate_batch(self, requests: Sequence[GenerationRequest]) -> list[GenerationResult]:
        return [self.generate(request) for request in requests]

    def info(self) -> BackendInfo:
        return BackendInfo("counting", "counting", None, None)

    def provenance(self) -> dict[str, str]:
        return {"kind": "counting", "model": "counting"}
