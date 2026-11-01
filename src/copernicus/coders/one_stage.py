"""The one-stage LLM coder for one attribute of one text.

A single prompt asks the model to rate the attribute directly from the whole
text; there is no extraction pass and no passage verification. The coder is the
comparison baseline of the two-stage coder and shares its result type, its
integer rules for the score, and its retry conventions: up to
``config.max_retries`` attempts with seed ``seed + attempt`` and an unchanged
temperature.

The result always carries a status. A failure yields a missing score, never a
score of 0. A transport failure of the backend is not a model-level status and
propagates to the caller as a `copernicus.errors.BackendError`.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from copernicus.coders.render import (
    PromptContext,
    render_one_stage_prompt,
    user_message_for_text,
)
from copernicus.coders.stage_runner import SECONDS_DECIMALS, Rejected, StageOutcome, run_stage
from copernicus.coders.two_stage import validate_scoring_output
from copernicus.coders.two_stage_types import AttributeResult, Status
from copernicus.errors import SettingsError
from copernicus.llm.base import Backend, GenerationRequest

logger = logging.getLogger(__name__)

__all__ = ["OneStageConfig", "evaluate_attribute_one_stage", "validate_one_stage_output"]

RESULT_WRAPPER_KEY = "dimension_result"
STAGE_SCORING = "scoring"
EXPLANATION_TOO_SHORT = "too_short"


@dataclass(frozen=True)
class OneStageConfig:
    """Settings of the one-stage LLM coder.

    Attributes:
        max_retries: Number of attempts (not the retries after a first attempt)
            before the coder gives up; at least 1.
        max_new_tokens: Generation budget of the scoring call.
        min_chars: Texts shorter than this (after stripping) are not coded.
    """

    max_retries: int = 3
    max_new_tokens: int = 2048
    min_chars: int = 20

    def __post_init__(self) -> None:
        """Check the settings.

        Raises:
            SettingsError: If a setting is outside its documented range; the message
                names the field and its value.
        """
        checks = (
            ("max_retries", self.max_retries >= 1, "at least 1"),
            ("max_new_tokens", self.max_new_tokens >= 1, "at least 1"),
            ("min_chars", self.min_chars >= 0, "at least 0"),
        )
        for field, valid, expectation in checks:
            if not valid:
                raise SettingsError(f"{field} must be {expectation}, got {getattr(self, field)!r}")


def validate_one_stage_output(parsed: Any) -> tuple[bool, str, int | None, str]:
    """Check the parsed output of the one-stage LLM coder.

    The output must be an object whose ``dimension_result`` is an object that
    holds an integer ``result`` from 1 to 5 under the same rules as the scoring
    call of the two-stage coder (see
    `copernicus.coders.two_stage.validate_scoring_output`). The
    ``explanation`` inside ``dimension_result`` is optional.

    Args:
        parsed: The parsed JSON value.

    Returns:
        Whether the output is valid, the reason if not, the score, and the explanation.
    """
    if not isinstance(parsed, Mapping):
        return False, f"top-level type {type(parsed).__name__} is not a JSON object", None, ""
    inner = parsed[RESULT_WRAPPER_KEY] if RESULT_WRAPPER_KEY in parsed else None
    if not isinstance(inner, Mapping):
        return False, f"missing or non-object {RESULT_WRAPPER_KEY}", None, ""
    return validate_scoring_output(dict(inner))


def _check_one_stage(parsed: Any) -> tuple[int, str] | Rejected:
    """Validate the one-stage output and pair the score with its explanation."""
    valid, reason, score, explanation = validate_one_stage_output(parsed)
    if valid and score is not None:
        return score, explanation
    return Rejected(reason)


def evaluate_attribute_one_stage(
    backend: Backend,
    text: str,
    ctx: PromptContext,
    config: OneStageConfig,
    *,
    temperature: float | None,
    seed: int,
) -> AttributeResult:
    """Code one attribute of one text with the one-stage LLM coder.

    The text is used as given (the caller normalises whitespace). A text
    shorter than ``config.min_chars`` after stripping is not sent to the model.
    Otherwise the scoring call makes up to ``config.max_retries`` attempts with
    seed ``seed + attempt``; the temperature never changes between attempts.

    The result has no evidence: ``evidence`` is empty and ``evaluable`` and
    ``justification`` are empty mappings. There is no extraction stage, so
    ``stage1_attempt`` and ``stage1_temperature_applied`` are always ``None``;
    ``stage2_attempt`` is the zero-based attempt of the scoring call that
    succeeded. ``temperature_applied`` and ``top_p_applied`` are what the backend
    applied in the last call.

    Args:
        backend: The LLM backend.
        text: The text to code.
        ctx: Prompt context: rubric, attribute, marker order, and wording.
        config: Retry, budget, and length settings.
        temperature: Temperature of the scoring call; ``None`` is greedy.
        seed: Base seed, already combined with the run number.

    Returns:
        The result with status ``ok``, ``too_short``, or ``scoring_failed``.

    Raises:
        BackendError: If the backend itself fails to answer a request.
    """
    started = time.perf_counter()
    attribute_id = ctx.attribute.attribute_id

    def finish(
        status: Status,
        score: int | None,
        explanation: str,
        outcome: StageOutcome[Any] | None = None,
    ) -> AttributeResult:
        return AttributeResult(
            attribute_id=attribute_id,
            score=score,
            status=status,
            explanation=explanation,
            stage1_attempt=None,
            stage2_attempt=None if outcome is None else outcome.attempt,
            evidence=(),
            evaluable={},
            justification={},
            seconds=round(time.perf_counter() - started, SECONDS_DECIMALS),
            prompt_tokens=0 if outcome is None else outcome.prompt_tokens,
            completion_tokens=0 if outcome is None else outcome.completion_tokens,
            temperature_applied=None if outcome is None else outcome.temperature_applied,
            top_p_applied=None if outcome is None else outcome.top_p_applied,
        )

    if len(text.strip()) < config.min_chars:
        return finish("too_short", None, EXPLANATION_TOO_SHORT)

    system = render_one_stage_prompt(ctx)
    user = user_message_for_text(text)

    def make_request(attempt: int) -> GenerationRequest:
        return GenerationRequest(system, user, config.max_new_tokens, temperature, seed + attempt)

    outcome = run_stage(
        backend, attribute_id, STAGE_SCORING, config.max_retries, make_request, _check_one_stage
    )
    scored = outcome.result
    if isinstance(scored, Rejected):
        return finish("scoring_failed", None, f"scoring_failed: {scored.reason}", outcome)
    score, explanation = scored
    return finish("ok", score, explanation, outcome)
