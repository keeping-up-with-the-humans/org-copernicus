"""The two-stage LLM coder for one attribute of one text.

The extraction stage asks the model for verbatim passages per marker. The
passages are verified against the text and deduplicated (see
`copernicus.coders.evidence`), and the scoring stage rates the attribute
from the surviving passages alone. Both stages retry on unusable output.

The result always carries a status. A failure of either stage yields a missing
score, never a score of 0: an attribute score is an integer from 1 to 5 or
``None``. A transport failure of the backend is not a model-level status and
propagates to the caller as a `copernicus.errors.BackendError`.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Mapping
from typing import Any

from copernicus.coders.evidence import (
    ProcessedExtraction,
    process_extraction,
    validate_extraction_block,
)
from copernicus.coders.render import (
    PromptContext,
    render_extraction_prompt,
    render_scoring_prompt,
    scoring_rubric_text,
    scoring_user_message,
    user_message_for_text,
)
from copernicus.coders.stage_runner import (
    SECONDS_DECIMALS,
    Rejected,
    StageOutcome,
    run_stage,
)
from copernicus.coders.two_stage_types import (
    MAX_SCORE,
    MIN_SCORE,
    AttributeResult,
    Status,
    TwoStageConfig,
)
from copernicus.llm.base import Backend, GenerationRequest

logger = logging.getLogger(__name__)

__all__ = [
    "AttributeResult",
    "TwoStageConfig",
    "evaluate_attribute",
    "stage1_attempt_temperature",
    "validate_scoring_output",
]

SCORE_STRING_PATTERN = re.compile(r"\s*[1-5]\s*")
EXTRACTION_WRAPPER_KEY = "marker_analysis"
TEMPERATURE_DECIMALS = 4
EXPLANATION_TOO_SHORT = "too_short"
EXPLANATION_NO_MARKERS = "no_markers_evaluable"
STAGE_EXTRACTION = "extraction"
STAGE_SCORING = "scoring"


def stage1_attempt_temperature(
    base: float | None, attempt: int, step: float, cap: float
) -> float | None:
    """Decoding temperature of one extraction attempt.

    Args:
        base: Design temperature of the extraction stage; ``None`` is greedy.
        attempt: Zero-based attempt number; 0 is the first attempt.
        step: Temperature added per retry.
        cap: Highest temperature a retry may use.

    Returns:
        ``base`` unchanged for the first attempt; for a retry, ``base + attempt * step``
        capped at ``cap`` and rounded (a ``None`` base escalates from 0.0).
    """
    if attempt <= 0:
        return base
    return round(min(cap, (base or 0.0) + attempt * step), TEMPERATURE_DECIMALS)


def validate_scoring_output(parsed: Any) -> tuple[bool, str, int | None, str]:
    """Check the parsed output of the scoring stage.

    The output must be an object with an integer ``result`` from 1 to 5. An
    integral float or a digit string is accepted; a boolean is not. The
    ``explanation`` is optional but must be a string when present.

    Args:
        parsed: The parsed JSON value.

    Returns:
        Whether the output is valid, the reason if not, the score, and the explanation.
    """
    if not isinstance(parsed, dict):
        return False, f"top-level type {type(parsed).__name__} is not a JSON object", None, ""
    if "result" not in parsed:
        return False, "missing 'result'", None, ""
    raw = parsed["result"]
    if isinstance(raw, bool):
        return False, "'result' is a boolean", None, ""
    if isinstance(raw, int):
        value = raw
    elif isinstance(raw, float) and raw.is_integer():
        value = int(raw)
    elif isinstance(raw, str) and SCORE_STRING_PATTERN.fullmatch(raw):
        value = int(raw.strip())
    else:
        return False, f"'result' is not an integer: {raw!r}", None, ""
    if not MIN_SCORE <= value <= MAX_SCORE:
        return False, f"'result' out of range {MIN_SCORE}..{MAX_SCORE}: {value}", None, ""
    explanation = parsed["explanation"] if "explanation" in parsed else ""
    if explanation is None:
        explanation = ""
    if not isinstance(explanation, str):
        return False, "'explanation' is not a string", None, ""
    return True, "", value, explanation


def _check_extraction(ctx: PromptContext, parsed: Any) -> Mapping[str, Any] | Rejected:
    """Unwrap the marker block of the extraction output and validate it."""
    wrapped = isinstance(parsed, Mapping) and EXTRACTION_WRAPPER_KEY in parsed
    block = parsed[EXTRACTION_WRAPPER_KEY] if wrapped else parsed
    valid, reason = validate_extraction_block(block, ctx.attribute)
    return block if valid else Rejected(reason)


def _check_scoring(parsed: Any) -> tuple[int, str] | Rejected:
    """Validate the scoring output and pair the score with its explanation."""
    valid, reason, score, explanation = validate_scoring_output(parsed)
    if valid and score is not None:
        return score, explanation
    return Rejected(reason)


def evaluate_attribute(
    backend: Backend,
    text: str,
    ctx: PromptContext,
    config: TwoStageConfig,
    *,
    temperature_stage1: float | None,
    temperature_stage2: float | None,
    seed: int,
) -> AttributeResult:
    """Code one attribute of one text with the two-stage LLM coder.

    The text is used as given (the caller normalises whitespace). A text
    shorter than ``config.min_chars`` after stripping is not sent to the model.
    The extraction stage makes up to ``config.max_retries`` attempts with seed
    ``seed + attempt`` and a temperature that rises by
    ``config.retry_temperature_step`` per further attempt, capped at
    ``config.retry_temperature_cap``. If no marker keeps a verified passage the
    scoring stage is skipped. The scoring stage makes up to the same number of
    attempts with seed ``seed + attempt`` and an unchanged temperature. The result
    records the temperature the backend applied in the accepted extraction attempt
    (``stage1_temperature_applied``) and in the last scoring call
    (``temperature_applied``).

    Args:
        backend: The LLM backend.
        text: The text to code.
        ctx: Prompt context: rubric, attribute, marker order, and wording.
        config: Retry, budget, and verification settings.
        temperature_stage1: Design temperature of the extraction stage.
        temperature_stage2: Temperature of the scoring stage.
        seed: Base seed, already combined with the run number.

    Returns:
        The result; its ``status`` says whether and how a score was produced.

    Raises:
        BackendError: If the backend itself fails to answer a request.
    """
    started = time.perf_counter()
    attribute_id = ctx.attribute.attribute_id

    def finish(
        status: Status,
        score: int | None,
        explanation: str,
        extraction: StageOutcome[Any] | None = None,
        processed: ProcessedExtraction | None = None,
        scoring: StageOutcome[Any] | None = None,
    ) -> AttributeResult:
        stages = [stage for stage in (extraction, scoring) if stage is not None]
        # A stage that accepted an answer ended with that attempt, so its last call is it.
        stage1_temperature = (
            extraction.temperature_applied
            if extraction is not None and extraction.attempt is not None
            else None
        )
        return AttributeResult(
            attribute_id=attribute_id,
            score=score,
            status=status,
            explanation=explanation,
            stage1_attempt=None if extraction is None else extraction.attempt,
            stage2_attempt=None if scoring is None else scoring.attempt,
            evidence=() if processed is None else processed.evidence,
            evaluable={} if processed is None else processed.evaluable,
            justification={} if processed is None else processed.justification,
            seconds=round(time.perf_counter() - started, SECONDS_DECIMALS),
            prompt_tokens=sum(stage.prompt_tokens for stage in stages),
            completion_tokens=sum(stage.completion_tokens for stage in stages),
            stage1_temperature_applied=stage1_temperature,
            temperature_applied=None if scoring is None else scoring.temperature_applied,
            top_p_applied=None if scoring is None else scoring.top_p_applied,
        )

    if len(text.strip()) < config.min_chars:
        return finish("too_short", None, EXPLANATION_TOO_SHORT)

    extraction_system = render_extraction_prompt(ctx)
    extraction_user = user_message_for_text(text)

    def extraction_request(attempt: int) -> GenerationRequest:
        temperature = stage1_attempt_temperature(
            temperature_stage1, attempt, config.retry_temperature_step, config.retry_temperature_cap
        )
        return GenerationRequest(
            extraction_system,
            extraction_user,
            config.stage1_max_new_tokens,
            temperature,
            seed + attempt,
        )

    extraction = run_stage(
        backend,
        attribute_id,
        STAGE_EXTRACTION,
        config.max_retries,
        extraction_request,
        lambda parsed: _check_extraction(ctx, parsed),
    )
    block = extraction.result
    if isinstance(block, Rejected):
        return finish("extraction_failed", None, f"extraction_failed: {block.reason}", extraction)

    processed = process_extraction(
        block,
        ctx.attribute,
        text,
        ctx.marker_order,
        threshold=config.verify_threshold,
        min_fuzzy_words=config.min_fuzzy_words,
    )
    if not processed.markers_to_rate:
        return finish(
            "no_evidence", config.no_evidence_score, EXPLANATION_NO_MARKERS, extraction, processed
        )

    scoring_system = render_scoring_prompt(ctx, processed.markers_to_rate)
    scoring_user = scoring_user_message(
        scoring_rubric_text(ctx.attribute, processed.markers_to_rate),
        processed.passages,
        mode=config.stage2_evidence,
        by_marker=processed.by_marker,
        markers_to_rate=processed.markers_to_rate,
    )

    def scoring_request(attempt: int) -> GenerationRequest:
        return GenerationRequest(
            scoring_system,
            scoring_user,
            config.stage2_max_new_tokens,
            temperature_stage2,
            seed + attempt,
        )

    scoring = run_stage(
        backend, attribute_id, STAGE_SCORING, config.max_retries, scoring_request, _check_scoring
    )
    scored = scoring.result
    if isinstance(scored, Rejected):
        failure = f"scoring_failed: {scored.reason}"
        return finish("scoring_failed", None, failure, extraction, processed, scoring)
    score, explanation = scored
    return finish("ok", score, explanation, extraction, processed, scoring)
