"""The answers of a writer model: the texts of an answer, and the attempts of one call."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from copernicus.errors import BackendError
from copernicus.llm.base import Backend, GenerationRequest, GenerationResult
from copernicus.llm.json_extract import extract_json
from copernicus.references.settings import GeneratorSettings

logger = logging.getLogger(__name__)

__all__ = ["RETRY_SEED_STEP", "CallOutcome", "attempt_request", "finish_call", "read_answer"]

RETRY_SEED_STEP = 100000
EXCERPT_HEAD = 300
EXCERPT_TAIL = 200
REASON_JSON_PARSE_FAILURE = "json_parse_failure"
REASON_NOT_AN_ARRAY = "not_a_json_array"
TRUNCATED_STRATEGY = "truncated_array"
TEXT_KEY = "text"


@dataclass(frozen=True)
class CallOutcome:
    """What one call of a writer produced.

    Attributes:
        texts: The kept texts: those of the accepted answer, or the usable texts of the
            last attempt when no answer was accepted.
        attempt: Zero-based attempt whose texts were kept.
        seed: Request seed of that attempt.
        result: The backend's result of that attempt.
        attempts: Attempts made.
        prompt_tokens: Prompt tokens over all attempts, as the backend reported them; an
            attempt without a count adds 0.
        completion_tokens: Completion tokens over all attempts, counted the same way.
    """

    texts: tuple[str, ...]
    attempt: int
    seed: int
    result: GenerationResult
    attempts: int
    prompt_tokens: int
    completion_tokens: int


def attempt_request(
    system: str, user: str, settings: GeneratorSettings, seed: int, attempt: int
) -> GenerationRequest:
    """Build the request of one attempt; attempt ``k`` adds ``100000 * k`` to the seed."""
    return GenerationRequest(
        system=system,
        user=user,
        max_new_tokens=settings.max_new_tokens,
        temperature=settings.temperature,
        seed=seed + RETRY_SEED_STEP * attempt,
    )


def _excerpt(text: str) -> str:
    """Show the head and tail of a raw model output for the log, newlines flattened."""
    flat = " ".join(text.split())
    if len(flat) <= EXCERPT_HEAD + EXCERPT_TAIL:
        return repr(flat)
    hidden = len(flat) - EXCERPT_HEAD - EXCERPT_TAIL
    return repr(flat[:EXCERPT_HEAD]) + f" ... [{hidden} chars] ... " + repr(flat[-EXCERPT_TAIL:])


def read_answer(raw: str, n: int, label: str) -> tuple[list[str], str | None]:
    """Read the texts out of a writer's answer.

    The answer is read as a JSON array, also behind an unmatched ``[`` of a prose preamble
    and inside an object that wraps it (``{"texts": [...]}``); of an array that the output
    cut off, the complete objects count. An answer is accepted when it holds at least ``n``
    usable texts (objects with a non-empty string ``text``); a surplus is cut to the first
    ``n``.

    Args:
        raw: The raw answer.
        n: Number of texts the answer must hold.
        label: Description of the call for the log.

    Returns:
        The usable texts, in order and at most ``n`` when accepted, and the reason the
        answer is not accepted, or ``None`` when it is.
    """
    try:
        value, strategy = extract_json(raw, expect="array")
    except BackendError:
        return [], REASON_JSON_PARSE_FAILURE
    if not isinstance(value, list):
        return [], REASON_NOT_AN_ARRAY
    usable = [
        item[TEXT_KEY]
        for item in value
        if isinstance(item, dict)
        and TEXT_KEY in item
        and isinstance(item[TEXT_KEY], str)
        and item[TEXT_KEY].strip()
    ]
    if len(usable) >= n:
        if len(usable) > n:
            logger.info("%s: kept the first %d of %d usable texts", label, n, len(usable))
        return usable[:n], None
    cut_off = " from an array the output cut off" if strategy == TRUNCATED_STRATEGY else ""
    return (
        usable,
        f"expected {n} objects with a text, got {len(value)}{cut_off} of which "
        f"{len(usable)} usable",
    )


def finish_call(
    backend: Backend,
    system: str,
    user: str,
    seed: int,
    first: GenerationResult,
    settings: GeneratorSettings,
    label: str,
) -> CallOutcome:
    """Read the answer to the first attempt of a call and retry until one is accepted.

    The retries are sent one at a time, built by `attempt_request`.

    Args:
        backend: The writer model.
        system: System prompt of the call.
        user: User prompt of the call.
        seed: Seed of the call's first attempt.
        first: The backend's answer to the first attempt.
        settings: The generation settings.
        label: Description of the call for the log.

    Returns:
        The texts of the accepted answer, or, when no attempt is accepted, the usable texts
        of the last attempt, at most ``texts_per_call`` of them; with the number, seed, and
        result of that attempt and the tokens of all attempts.

    Raises:
        BackendError: If the backend fails to answer a retry.
    """
    n = settings.texts_per_call
    result = first
    texts: list[str] = []
    prompt_tokens = completion_tokens = attempt = 0
    for attempt in range(settings.max_retries):
        if attempt > 0:
            result = backend.generate(attempt_request(system, user, settings, seed, attempt))
        prompt_tokens += result.prompt_tokens or 0
        completion_tokens += result.completion_tokens or 0
        texts, reason = read_answer(result.text, n, label)
        if reason is None:
            break
        logger.info(
            "%s, attempt %d/%d failed: %s (seed=%s, temperature=%s); raw output head/tail: %s",
            label,
            attempt + 1,
            settings.max_retries,
            reason,
            seed + RETRY_SEED_STEP * attempt,
            settings.temperature,
            _excerpt(result.text),
        )
    return CallOutcome(
        texts=tuple(texts[:n]),
        attempt=attempt,
        seed=seed + RETRY_SEED_STEP * attempt,
        result=result,
        attempts=attempt + 1,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )
