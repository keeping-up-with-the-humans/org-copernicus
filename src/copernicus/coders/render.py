"""Render the extraction, scoring, and one-stage prompts of the LLM coders.

A prompt is built from a rubric attribute, an order of its markers, and the
three chosen wordings (role, task, level). The marker order moves the
markers, their level descriptions, and the output template together, so the
order in which the model reads the markers is one consistent order. The
glossary of definitions is never reordered.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

import jinja2

from copernicus.assets import asset_path
from copernicus.coders.rotation import Wording
from copernicus.coders.sections import (
    LEVEL_NUMBERS,
    definitions_section,
    extraction_output_template,
    level_descriptions_section,
    lookup_markers,
    marker_lines,
    markers_section,
    ordered_markers,
    subrubrics_section,
)
from copernicus.errors import PromptContractError
from copernicus.rubric.model import Attribute, Rubric

logger = logging.getLogger(__name__)

__all__ = [
    "PromptContext",
    "definitions_section",
    "extraction_output_template",
    "level_descriptions_section",
    "markers_section",
    "render_extraction_prompt",
    "render_one_stage_prompt",
    "render_scoring_prompt",
    "scoring_rubric_text",
    "scoring_user_message",
    "subrubrics_section",
    "template_hashes",
    "user_message_for_text",
]

PROMPT_DIRECTORY = "coders/prompts"
TEMPLATE_FILES: Mapping[str, str] = {
    "extraction": "extraction.jinja2",
    "scoring": "scoring.jinja2",
    "one_stage": "one_stage.jinja2",
}
HASH_LENGTH = 16
CONCEPT_PLACEHOLDER = "{concept_name}"
EVIDENCE_MODES = ("unlabelled", "labelled")
UNLABELLED_HEADER = (
    "The extracted evidence (verbatim passages from the text, in the order they occur):"
)
LABELLED_HEADER = "The extracted evidence (verbatim passages from the text, grouped by Marker):"


@dataclass(frozen=True)
class PromptContext:
    """Everything the three renderers need to build the prompts of one attribute.

    Attributes:
        rubric: The rubric that holds the attribute and the glossary.
        attribute: The attribute that is coded.
        marker_order: Names of the attribute's markers in the order the prompt
            lists them; a permutation of the attribute's marker names.
        role: Chosen role wording.
        task: Chosen task wording; ``{concept_name}`` stands for the concept.
        level: Chosen wording of the level protocol. The extraction prompt has
            no level protocol and ignores it.
    """

    rubric: Rubric
    attribute: Attribute
    marker_order: Sequence[str]
    role: Wording
    task: Wording
    level: Wording


@lru_cache(maxsize=1)
def _environment() -> jinja2.Environment:
    """Return the Jinja environment over the shipped prompt templates.

    The single newline that ends each template file is dropped, as Jinja does by
    default, so a prompt ends with its last line of text.

    Returns:
        A cached environment in which an undefined variable raises an error.
    """
    return jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(asset_path(PROMPT_DIRECTORY))),
        autoescape=False,
        keep_trailing_newline=False,
        undefined=jinja2.StrictUndefined,
    )


def template_hashes() -> dict[str, str]:
    """Hash the three prompt templates.

    Returns:
        For ``extraction``, ``scoring``, and ``one_stage`` the first 16 hex
        characters of the SHA-256 of the template file's bytes.
    """
    hashes: dict[str, str] = {}
    for key, file_name in TEMPLATE_FILES.items():
        digest = hashlib.sha256(asset_path(f"{PROMPT_DIRECTORY}/{file_name}").read_bytes())
        hashes[key] = digest.hexdigest()[:HASH_LENGTH]
    return hashes


def _task_framing(ctx: PromptContext, kind: str) -> str:
    """Return the task wording of one prompt kind with the concept filled in.

    Args:
        ctx: Prompt context.
        kind: ``extraction``, ``scoring``, or ``one_stage``.

    Returns:
        The task text with every ``{concept_name}`` replaced by the concept.
    """
    return ctx.task.texts[kind].replace(CONCEPT_PLACEHOLDER, ctx.rubric.concept)


def _common_variables(ctx: PromptContext, kind: str) -> dict[str, str]:
    """Collect the template variables shared by all three prompts.

    Args:
        ctx: Prompt context.
        kind: ``extraction``, ``scoring``, or ``one_stage``.

    Returns:
        Role, task, attribute, and glossary variables.
    """
    return {
        "role_description": ctx.role.texts[kind],
        "task_framing": _task_framing(ctx, kind),
        "attribute_name": ctx.attribute.name,
        "attribute_description": ctx.attribute.description,
        "definitions_section": definitions_section(ctx.rubric),
    }


def _render(kind: str, variables: Mapping[str, str]) -> str:
    """Render one template.

    Args:
        kind: ``extraction``, ``scoring``, or ``one_stage``.
        variables: Template variables.

    Returns:
        The prompt text.
    """
    return _environment().get_template(TEMPLATE_FILES[kind]).render(**variables)


def render_extraction_prompt(ctx: PromptContext) -> str:
    """Render the extraction prompt of the two-stage LLM coder.

    Args:
        ctx: Prompt context.

    Returns:
        The system prompt that asks for verbatim passages per marker.

    Raises:
        PromptContractError: If the marker order is not a permutation of the markers.
    """
    order = list(ctx.marker_order)
    variables = _common_variables(ctx, "extraction")
    variables["markers_section"] = markers_section(ctx.attribute, order)
    variables["level_descriptions_section"] = level_descriptions_section(ctx.attribute, order)
    variables["output_json_template_extract_only"] = extraction_output_template(order)
    return _render("extraction", variables)


def render_scoring_prompt(ctx: PromptContext, markers_to_rate: Sequence[str]) -> str:
    """Render the scoring prompt of the two-stage LLM coder.

    Args:
        ctx: Prompt context.
        markers_to_rate: Names of the markers that have evidence; the prompt lists
            them in the context's marker order.

    Returns:
        The system prompt that asks for one level from the extracted evidence.

    Raises:
        PromptContractError: If ``markers_to_rate`` is empty or names an unknown
            marker, or the marker order is not a permutation of the markers.
    """
    if not markers_to_rate:
        raise PromptContractError(
            f"no markers to rate for attribute {ctx.attribute.attribute_id!r}"
        )
    lookup_markers(ctx.attribute, markers_to_rate)
    ordered = ordered_markers(ctx.attribute, list(ctx.marker_order))
    wanted = set(markers_to_rate)
    variables = _common_variables(ctx, "scoring")
    variables["markers_section"] = marker_lines(m for m in ordered if m.name in wanted)
    variables["level_protocol"] = ctx.level.texts["scoring"]
    return _render("scoring", variables)


def render_one_stage_prompt(ctx: PromptContext) -> str:
    """Render the prompt of the one-stage LLM coder.

    Args:
        ctx: Prompt context.

    Returns:
        The system prompt that asks for one level from the whole text.

    Raises:
        PromptContractError: If the marker order is not a permutation of the markers.
    """
    order = list(ctx.marker_order)
    variables = _common_variables(ctx, "one_stage")
    variables["markers_section"] = markers_section(ctx.attribute, order)
    variables["subrubrics_section"] = subrubrics_section(ctx.attribute, order)
    variables["level_protocol"] = ctx.level.texts["one_stage"]
    return _render("one_stage", variables)


def scoring_rubric_text(attribute: Attribute, markers_to_rate: Sequence[str]) -> str:
    """Build the scoring rubric from the markers that have evidence.

    Args:
        attribute: Attribute that owns the markers.
        markers_to_rate: Names of the markers to include.

    Returns:
        Five lines ``Level N: ...``; each joins the stripped level descriptions
        of the markers to rate with single spaces, in the attribute's marker order.

    Raises:
        PromptContractError: If ``markers_to_rate`` is empty or names an unknown
            marker, or the joined text of a level is empty.
    """
    if not markers_to_rate:
        raise PromptContractError(f"no markers to rate for attribute {attribute.attribute_id!r}")
    lookup_markers(attribute, markers_to_rate)
    wanted = set(markers_to_rate)
    selected = [marker for marker in attribute.markers if marker.name in wanted]
    lines: list[str] = []
    for number in LEVEL_NUMBERS:
        text = " ".join(marker.level(number).description.strip() for marker in selected).strip()
        if not text:
            raise PromptContractError(
                f"empty scoring rubric text at level {number} for attribute "
                f"{attribute.attribute_id!r}"
            )
        lines.append(f"Level {number}: {text}")
    return "\n".join(lines)


def _passage_lines(passages: Iterable[str]) -> list[str]:
    """Format passages as quoted list items.

    Args:
        passages: Verbatim passages.

    Returns:
        One ``- "passage"`` line per passage.
    """
    return [f'- "{passage}"' for passage in passages]


def scoring_user_message(
    rubric_text: str,
    passages: Sequence[str],
    *,
    mode: Literal["unlabelled", "labelled"] = "unlabelled",
    by_marker: Mapping[str, Sequence[str]] | None = None,
    markers_to_rate: Sequence[str] | None = None,
) -> str:
    """Build the user message of the scoring prompt: the rubric and the evidence.

    Args:
        rubric_text: Scoring rubric from `scoring_rubric_text`.
        passages: Verbatim passages in the order they occur in the text.
        mode: ``unlabelled`` lists the passages once; ``labelled`` groups them
            under the markers they were returned for.
        by_marker: Passages per marker name; required in ``labelled`` mode.
        markers_to_rate: Marker names to list, in order; required in ``labelled`` mode.

    Returns:
        The message text.

    Raises:
        PromptContractError: If the mode is unknown, ``passages`` is empty, the
            labelled inputs are missing, or a marker to rate has no entry in
            ``by_marker``.
    """
    if mode not in EVIDENCE_MODES:
        raise PromptContractError(f"unknown evidence mode {mode!r}; known: {EVIDENCE_MODES}")
    if not passages:
        raise PromptContractError("the scoring message needs at least one passage")
    lines = [f"The scoring rubric:\n{rubric_text}", ""]
    if mode == "unlabelled":
        lines.append(UNLABELLED_HEADER)
        lines.extend(_passage_lines(passages))
        return "\n".join(lines)
    if by_marker is None or markers_to_rate is None:
        raise PromptContractError("labelled evidence needs by_marker and markers_to_rate")
    lines.append(LABELLED_HEADER)
    for name in markers_to_rate:
        if name not in by_marker:
            raise PromptContractError(f"no passages given for marker {name!r}")
        lines.append(f"Marker: {name}")
        lines.extend(_passage_lines(by_marker[name]))
    return "\n".join(lines)


def user_message_for_text(text: str) -> str:
    """Build the user message that hands a text to the extraction or one-stage prompt.

    Args:
        text: The text to code.

    Returns:
        The message text.
    """
    return f"Text to analyze:\n{text}"
