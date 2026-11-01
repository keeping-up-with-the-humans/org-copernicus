"""The retry loop that the LLM coders share.

A stage sends one request per attempt, parses the model's output as JSON and
hands the parsed value to a check. The first output that passes the check ends
the stage; if every attempt fails, the last rejection is reported. Every failed
attempt is logged with the reason and the head and tail of the raw output.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from copernicus.errors import BackendError
from copernicus.llm.base import Backend, GenerationRequest
from copernicus.llm.json_extract import extract_json

logger = logging.getLogger(__name__)

__all__ = ["Rejected", "StageOutcome", "run_stage"]

T = TypeVar("T")

EXCERPT_HEAD = 300
EXCERPT_TAIL = 200
REASON_JSON_PARSE_FAILURE = "json_parse_failure"
DIRECT_STRATEGY = "direct"
SECONDS_DECIMALS = 3


@dataclass(frozen=True)
class Rejected:
    """A stage output that failed parsing or validation.

    Attributes:
        reason: Why the output was rejected.
    """

    reason: str


@dataclass(frozen=True)
class StageOutcome(Generic[T]):
    """What one stage produced, with the tokens it used and the sampling it got.

    Attributes:
        result: The checked value, or the rejection of the last attempt.
        attempt: Zero-based attempt that was accepted; ``None`` if none was.
        prompt_tokens: Prompt tokens over the stage's calls; a missing count adds 0.
        completion_tokens: Generated tokens over the stage's calls; a missing count adds 0.
        temperature_applied: Temperature the backend applied in the stage's last call, which
            for an accepted stage is the accepted attempt; ``None`` if it applied none or no
            call was made.
        top_p_applied: Nucleus-sampling threshold the backend applied in the stage's last
            call; ``None`` if it applied none or no call was made.
    """

    result: T | Rejected
    attempt: int | None
    prompt_tokens: int
    completion_tokens: int
    temperature_applied: float | None
    top_p_applied: float | None


def _excerpt(text: str, head: int = EXCERPT_HEAD, tail: int = EXCERPT_TAIL) -> str:
    """Show the head and tail of a raw model output for the log, newlines flattened."""
    flat = " ".join(text.split())
    if len(flat) <= head + tail:
        return repr(flat)
    return repr(flat[:head]) + f" ... [{len(flat) - head - tail} chars] ... " + repr(flat[-tail:])


def _parse_json(raw: str) -> tuple[Any, str] | None:
    """Parse model output as JSON.

    Returns:
        The parsed value and the name of the strategy that parsed it, or ``None``
        when no JSON value could be extracted.
    """
    try:
        return extract_json(raw)
    except BackendError:
        return None


def run_stage(
    backend: Backend,
    attribute_id: str,
    stage: str,
    max_attempts: int,
    make_request: Callable[[int], GenerationRequest],
    check: Callable[[Any], T | Rejected],
) -> StageOutcome[T]:
    """Send a stage's request until its output passes the check or the attempts run out.

    Args:
        backend: The LLM backend.
        attribute_id: Attribute that is being coded, for the log.
        stage: Stage name for the log.
        max_attempts: Number of attempts.
        make_request: Builds the request of a zero-based attempt.
        check: Maps parsed JSON to the checked value or to a rejection.

    Returns:
        The outcome: the checked value and attempt of the first accepted output,
        otherwise the last rejection, plus the tokens used and the sampling the backend
        applied in the last call.

    Raises:
        BackendError: If the backend itself fails to answer a request.
    """
    rejected = Rejected(REASON_JSON_PARSE_FAILURE)
    prompt_tokens = completion_tokens = 0
    applied: tuple[float | None, float | None] = (None, None)
    for attempt in range(max_attempts):
        label = f"attribute {attribute_id}, {stage} stage, attempt {attempt + 1}/{max_attempts}"
        request = make_request(attempt)
        generated = backend.generate(request)
        prompt_tokens += generated.prompt_tokens or 0
        completion_tokens += generated.completion_tokens or 0
        applied = (generated.temperature_applied, generated.top_p_applied)
        parsed = _parse_json(generated.text)
        if parsed is None:
            rejected = Rejected(REASON_JSON_PARSE_FAILURE)
        else:
            value, strategy = parsed
            if strategy != DIRECT_STRATEGY:
                logger.info("%s: JSON read with strategy %s", label, strategy)
            checked = check(value)
            if not isinstance(checked, Rejected):
                return StageOutcome(checked, attempt, prompt_tokens, completion_tokens, *applied)
            rejected = checked
        logger.info(
            "%s failed: %s (seed=%s, temperature=%s); raw output head/tail: %s",
            label,
            rejected.reason,
            request.seed,
            request.temperature,
            _excerpt(generated.text),
        )
    return StageOutcome(rejected, None, prompt_tokens, completion_tokens, *applied)
