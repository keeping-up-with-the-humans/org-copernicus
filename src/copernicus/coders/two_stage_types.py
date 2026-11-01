"""Settings and result types of the two-stage LLM coder."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from copernicus.coders.evidence import EvidenceRow
from copernicus.errors import SettingsError

logger = logging.getLogger(__name__)

__all__ = [
    "EXTRACTION_FAILED",
    "MAX_SCORE",
    "MIN_SCORE",
    "SCORING_FAILED",
    "AttributeResult",
    "Status",
    "TwoStageConfig",
]

Status = Literal["ok", "no_evidence", "too_short", "extraction_failed", "scoring_failed"]
# The two failure statuses by name, for the modules that read them back from a store.
EXTRACTION_FAILED: Status = "extraction_failed"
SCORING_FAILED: Status = "scoring_failed"

MIN_SCORE = 1
MAX_SCORE = 5
MAX_THRESHOLD = 100.0
EVIDENCE_MODES = ("unlabelled", "labelled")
SCORELESS_STATUSES = ("too_short", "extraction_failed", "scoring_failed")


def _is_score(value: object) -> bool:
    """Return whether the value is an integer (not a bool) from MIN_SCORE to MAX_SCORE."""
    if isinstance(value, bool) or not isinstance(value, int):
        return False
    return MIN_SCORE <= value <= MAX_SCORE


@dataclass(frozen=True)
class TwoStageConfig:
    """Settings of the two-stage LLM coder.

    Attributes:
        max_retries: Number of attempts per stage (not the retries after a first
            attempt) before the stage counts as failed; at least 1.
        stage1_max_new_tokens: Generation budget of the extraction stage.
        stage2_max_new_tokens: Generation budget of the scoring stage.
        verify_threshold: Alignment score a passage needs to count as verified.
        min_fuzzy_words: Shortest passage, in words, that may match approximately.
        min_chars: Texts shorter than this (after stripping) are not coded.
        no_evidence_score: Score recorded when no marker kept a verified
            passage; ``None`` leaves the score missing.
        stage2_evidence: ``"unlabelled"`` lists the passages once;
            ``"labelled"`` groups them under their markers.
        retry_temperature_step: Temperature added per extraction retry.
        retry_temperature_cap: Highest temperature an extraction retry may use.
    """

    max_retries: int = 3
    stage1_max_new_tokens: int = 8192
    stage2_max_new_tokens: int = 2048
    verify_threshold: float = 88.0
    min_fuzzy_words: int = 3
    min_chars: int = 20
    no_evidence_score: int | None = 1
    stage2_evidence: Literal["unlabelled", "labelled"] = "unlabelled"
    retry_temperature_step: float = 0.3
    retry_temperature_cap: float = 1.0

    def __post_init__(self) -> None:
        """Check the settings.

        Raises:
            SettingsError: If a setting is outside its documented range; the message
                names the field and its value.
        """
        score_range = f"None or an integer in {MIN_SCORE}..{MAX_SCORE}"
        checks = (
            ("max_retries", self.max_retries >= 1, "at least 1"),
            ("stage1_max_new_tokens", self.stage1_max_new_tokens >= 1, "at least 1"),
            ("stage2_max_new_tokens", self.stage2_max_new_tokens >= 1, "at least 1"),
            ("min_chars", self.min_chars >= 0, "at least 0"),
            ("min_fuzzy_words", self.min_fuzzy_words >= 1, "at least 1"),
            ("verify_threshold", 0 <= self.verify_threshold <= MAX_THRESHOLD, "in 0..100"),
            ("retry_temperature_step", self.retry_temperature_step >= 0, "at least 0"),
            ("retry_temperature_cap", self.retry_temperature_cap >= 0, "at least 0"),
            ("stage2_evidence", self.stage2_evidence in EVIDENCE_MODES, f"one of {EVIDENCE_MODES}"),
            (
                "no_evidence_score",
                self.no_evidence_score is None or _is_score(self.no_evidence_score),
                score_range,
            ),
        )
        for field, valid, expectation in checks:
            if not valid:
                raise SettingsError(f"{field} must be {expectation}, got {getattr(self, field)!r}")


@dataclass(frozen=True)
class AttributeResult:
    """The outcome of coding one attribute of one text.

    Attributes:
        attribute_id: Attribute that was coded.
        score: Integer from 1 to 5, or ``None`` when no score was produced.
        status: ``ok`` (scored), ``no_evidence`` (no verified passage; the score
            is the configured ``no_evidence_score``), ``too_short``,
            ``extraction_failed``, or ``scoring_failed``.
        explanation: The model's explanation when scored, otherwise the reason
            for the status.
        stage1_attempt: Zero-based attempt of the extraction stage that
            succeeded; ``None`` if none did or the stage did not run.
        stage2_attempt: Zero-based attempt of the scoring stage that succeeded;
            ``None`` if none did or the stage did not run.
        evidence: Every passage the extraction stage returned, verified or not.
        evaluable: Per marker id, whether the marker kept a verified passage.
        justification: Per marker id, the kept passages joined by newlines.
        seconds: Wall-clock time of the whole evaluation.
        prompt_tokens: Prompt tokens summed over every backend call; a call
            that reports no count adds 0.
        completion_tokens: Generated tokens summed over every backend call; a
            call that reports no count adds 0.
        stage1_temperature_applied: Temperature the backend applied in the extraction
            attempt that was accepted (attempt ``stage1_attempt``), whose requested
            temperature a retry raises by ``retry_temperature_step`` up to
            ``retry_temperature_cap``; ``None`` when no attempt was accepted (the stage
            failed or did not run, and the one-stage coder has no such stage) or the
            backend sent no temperature.
        temperature_applied: Temperature the backend applied in the last scoring
            call (the one call of the one-stage coder); ``None`` when no scoring
            call was made or the backend sent no temperature.
        top_p_applied: Nucleus-sampling threshold the backend applied in that
            call; ``None`` when no scoring call was made or none was sent.
    """

    attribute_id: str
    score: int | None
    status: Status
    explanation: str
    stage1_attempt: int | None
    stage2_attempt: int | None
    evidence: tuple[EvidenceRow, ...]
    evaluable: Mapping[str, bool]
    justification: Mapping[str, str]
    seconds: float
    prompt_tokens: int
    completion_tokens: int
    stage1_temperature_applied: float | None = None
    temperature_applied: float | None = None
    top_p_applied: float | None = None

    def __post_init__(self) -> None:
        """Check the score invariants.

        Raises:
            ValueError: If the score is not ``None`` or an integer (not a bool)
                from 1 to 5, if the status is ``ok`` without a score, or if the
                status is ``too_short``, ``extraction_failed``, or
                ``scoring_failed`` with a score.
        """
        if self.score is not None and not _is_score(self.score):
            raise ValueError(
                f"score must be None or an integer in {MIN_SCORE}..{MAX_SCORE}, got {self.score!r}"
            )
        if self.status == "ok" and self.score is None:
            raise ValueError("status 'ok' requires a score")
        if self.status in SCORELESS_STATUSES and self.score is not None:
            raise ValueError(f"status {self.status!r} requires score None, got {self.score!r}")
