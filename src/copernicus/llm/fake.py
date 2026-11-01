"""Deterministic fake backend for tests and examples."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable, Mapping, Sequence

from copernicus.assets import validate_toy_cues
from copernicus.errors import BackendError
from copernicus.llm.base import (
    TOP_P,
    BackendInfo,
    GenerationRequest,
    GenerationResult,
    ModelCapabilities,
    capability_provenance,
    provenance_text,
)
from copernicus.rubric import Attribute, Rubric

logger = logging.getLogger(__name__)

__all__ = ["FakeBackend", "Responder", "ScriptedResponder", "cue_responder"]

Responder = Callable[[GenerationRequest], str]
"""A function that maps a generation request to the text of the answer."""

BACKEND_KIND = "fake"
STAGE_ONE_MARKER = "EXTRACTION ONLY"
SCORING_MARKER = "You are scoring the attribute"
ONE_STAGE_MARKER = "single integrated evaluator"
ONE_STAGE_ATTRIBUTE_PHRASE = "evaluating the attribute:"
TEXT_PREFIX = "Text to analyze:\n"
PASSAGE_LINE_PREFIX = '- "'
MIN_LEVEL = 1
MAX_LEVEL = 5
TEMPLATE_MARKER_PATTERN = re.compile(r'^\s*"([^"]+)": \{', re.MULTILINE)
SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


class FakeBackend:
    """A backend that answers every request with a responder function.

    Attributes:
        calls: Every request received, in order.
    """

    def __init__(self, responder: Responder, model_name: str = "fake") -> None:
        """Initialise the backend.

        Args:
            responder: Function mapping a request to the answer text.
            model_name: Model name reported by ``info``.
        """
        self._responder = responder
        self._model_name = model_name
        self.calls: list[GenerationRequest] = []

    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Record the request and answer it with the responder.

        Args:
            request: The generation request.

        Returns:
            The responder's text with word-count token numbers and the requested
            temperature and seed echoed as applied.

        Raises:
            BackendError: If the responder cannot answer.
        """
        self.calls.append(request)
        text = self._responder(request)
        return GenerationResult(
            text=text,
            prompt_tokens=len(request.system.split()) + len(request.user.split()),
            completion_tokens=len(text.split()),
            temperature_applied=request.temperature,
            top_p_applied=TOP_P if request.temperature else None,
            seed_applied=request.seed,
        )

    def generate_batch(self, requests: Sequence[GenerationRequest]) -> list[GenerationResult]:
        """Answer the requests one after the other, in order.

        Args:
            requests: The generation requests.

        Returns:
            One result per request.
        """
        return [self.generate(request) for request in requests]

    def info(self) -> BackendInfo:
        """Describe the fake backend.

        Returns:
            Backend info with the model name, kind ``"fake"``, and no quantization or window.
        """
        return BackendInfo(self._model_name, BACKEND_KIND, None, None)

    def provenance(self) -> dict[str, str]:
        """Describe the fake backend for the identity of a run.

        Returns:
            Kind ``"fake"``, the model name, the capability flags (the fake applies every
            sampling parameter it is given), and the nucleus-sampling threshold it reports.
        """
        return {
            "kind": BACKEND_KIND,
            "model": self._model_name,
            **capability_provenance(ModelCapabilities()),
            "top_p": provenance_text(TOP_P),
        }


class ScriptedResponder:
    """A responder that returns a fixed list of answers in order."""

    def __init__(self, responses: Sequence[str]) -> None:
        """Initialise the responder.

        Args:
            responses: The answers to return, in call order.
        """
        self._responses = tuple(responses)
        self._next = 0

    def __call__(self, request: GenerationRequest) -> str:
        """Return the next scripted answer.

        Args:
            request: The generation request (ignored).

        Returns:
            The next answer.

        Raises:
            BackendError: If all answers have been used.
        """
        if self._next >= len(self._responses):
            raise BackendError(
                f"scripted responses exhausted after {self._next}", attempts=self._next
            )
        answer = self._responses[self._next]
        self._next += 1
        return answer


def cue_responder(
    rubric: Rubric,
    cues: Mapping[str, Sequence[str]],
    one_stage_level: Callable[[int], int] | None = None,
) -> Responder:
    """Build a deterministic responder that codes texts by cue words.

    Three kinds of prompt are recognised by their system prompt. Extraction prompts
    (``EXTRACTION ONLY``) are answered, for each marker named in the output template, with
    the verbatim sentences of the text that contain a cue word of that marker. Scoring
    prompts (``You are scoring the attribute``) are answered with a level derived from the
    number of passages. One-stage prompts (``single integrated evaluator``) are answered with
    a level derived from the number of distinct cue words of the attribute found in the text;
    the attribute is the one named after ``evaluating the attribute:`` in the prompt.
    Cue words are matched case-insensitively as substrings; the sentences of a text end at
    ``.``, ``?``, or ``!`` followed by whitespace and keep their terminal punctuation.

    Args:
        rubric: Rubric whose markers the cues belong to and whose attributes the one-stage
            prompts name.
        cues: Cue words per marker, keyed by marker id or by marker name; every marker of
            the rubric needs a non-empty list.
        one_stage_level: Maps the number of distinct cue words to a level; defaults to
            ``min(5, 1 + n // 2)``.

    Returns:
        A responder for use with ``FakeBackend``.

    Raises:
        ValueError: If a cue key is neither a marker id nor a marker name of the rubric,
            a marker has no cues or an empty list, or one marker name is given two
            different cue lists.
    """
    cues_by_name = _cues_by_marker_name(rubric, cues)

    def respond(request: GenerationRequest) -> str:
        if STAGE_ONE_MARKER in request.system:
            return json.dumps(_extraction_answer(request, cues_by_name))
        if SCORING_MARKER in request.system:
            return json.dumps(_scoring_answer(request.user))
        if ONE_STAGE_MARKER in request.system:
            attribute = _attribute_in_prompt(rubric, request.system)
            return json.dumps(
                _one_stage_answer(request.user, attribute, cues_by_name, one_stage_level)
            )
        raise BackendError("cue responder cannot classify the prompt")

    return respond


def _cues_by_marker_name(
    rubric: Rubric, cues: Mapping[str, Sequence[str]]
) -> dict[str, tuple[str, ...]]:
    """Normalise a cue table keyed by marker id or marker name to one keyed by name."""
    markers = [marker for attribute in rubric.attributes for marker in attribute.markers]
    name_of_key = {marker.marker_id: marker.name for marker in markers}
    name_of_key.update({marker.name: marker.name for marker in markers})
    validated = validate_toy_cues(cues, name_of_key)
    by_name: dict[str, tuple[str, ...]] = {}
    for key, words in validated.items():
        name = name_of_key[key]
        if name in by_name and by_name[name] != tuple(words):
            raise ValueError(f"marker {name!r} is given two different cue lists")
        by_name[name] = tuple(words)
    missing = [marker.marker_id for marker in markers if marker.name not in by_name]
    if missing:
        raise ValueError(f"no cue words for marker(s) {missing!r}")
    return by_name


def _text_of(user: str) -> str:
    """Return the text after the ``Text to analyze:`` prefix of a user message."""
    _, separator, text = user.partition(TEXT_PREFIX)
    if not separator:
        raise BackendError("cue responder found no text to analyze", last_output=user[:500])
    return text


def _casefolded_hits(text: str, words: Sequence[str]) -> bool:
    """Return whether any cue word occurs in the text, ignoring case."""
    folded = text.casefold()
    return any(word.casefold() in folded for word in words)


def _extraction_answer(
    request: GenerationRequest, cues: Mapping[str, Sequence[str]]
) -> dict[str, dict[str, object]]:
    """Build the extraction answer for every marker named in the output template."""
    sentences = SENTENCE_BOUNDARY.split(_text_of(request.user).strip())
    answer: dict[str, dict[str, object]] = {}
    for name in TEMPLATE_MARKER_PATTERN.findall(request.system):
        quotes = [s for s in sentences if _casefolded_hits(s, cues[name])]
        answer[name] = {"quotes": quotes, "evaluable": len(quotes) > 0}
    return answer


def _scoring_answer(user: str) -> dict[str, object]:
    """Build the scoring answer from the number of passages in the user message."""
    n_passages = sum(1 for line in user.splitlines() if line.startswith(PASSAGE_LINE_PREFIX))
    return {
        "result": min(MAX_LEVEL, max(MIN_LEVEL, 1 + n_passages // 2)),
        "explanation": f"{n_passages} passages",
    }


def _attribute_in_prompt(rubric: Rubric, system: str) -> Attribute:
    """Return the attribute named after ``evaluating the attribute:`` in a one-stage prompt."""
    _, separator, rest = system.partition(ONE_STAGE_ATTRIBUTE_PHRASE)
    named = rest.partition("\n")[0].strip()
    if not separator or named not in {a.name for a in rubric.attributes}:
        raise BackendError(
            f"cue responder found no attribute of the rubric after {ONE_STAGE_ATTRIBUTE_PHRASE!r}",
            last_output=system[:500],
        )
    return next(a for a in rubric.attributes if a.name == named)


def _one_stage_answer(
    user: str,
    attribute: Attribute,
    cues: Mapping[str, Sequence[str]],
    one_stage_level: Callable[[int], int] | None,
) -> dict[str, object]:
    """Build the one-stage answer from the distinct cue words of the attribute's markers."""
    folded = _text_of(user).casefold()
    found = {
        word.casefold()
        for marker in attribute.markers
        for word in cues[marker.name]
        if word.casefold() in folded
    }
    n_cues = len(found)
    level = one_stage_level(n_cues) if one_stage_level else min(MAX_LEVEL, 1 + n_cues // 2)
    return {"dimension_result": {"result": level, "explanation": f"{n_cues} cue words"}}
