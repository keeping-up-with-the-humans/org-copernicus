"""The prompts that ask a writer model for reference texts of one cell."""

from __future__ import annotations

import logging

from copernicus.errors import ConfigurationError
from copernicus.references.cells import TARGET_CONSTRUCT, Cell
from copernicus.references.settings import STYLES, Style
from copernicus.rubric.model import Rubric

logger = logging.getLogger(__name__)

__all__ = ["PROMPT_VERSION", "system_prompt", "user_prompt"]

PROMPT_VERSION = "references-v1"
"""The version label of the prompts that write reference texts.

Every generated text carries it in its ``prompt_version`` column.
"""
POSITIVE = "positive"
SENTENCE_ENDINGS = (".", "!", "?")


def _neutral_system_prompt(n: int) -> str:
    """Return the system prompt of the neutral style, which leaves the medium open."""
    return f"""\
You write short, realistic, first-person texts of the kind an ordinary person might \
actually write or say in everyday life — a message to a friend, a diary line, a quick \
note, an offhand remark, a review, a comment, a journal entry. Each text is a few \
sentences (roughly 1-5 sentences). The writer is a normal person going about their \
life; they are NOT aware of any rubric, theory, scale, or measurement, and they are \
not trying to demonstrate any trait or quality. Whatever quality the task asks for \
must emerge naturally from what the person says and how they say it — never by naming \
it or describing themselves in those terms.

STYLE REQUIREMENTS:
- Write in the FIRST PERSON, as a real, specific person with their own situation.
- Keep each text short and natural: roughly 1-5 sentences.
- Vary the everyday context across the {n} texts (home, work, relationships, hobbies, \
errands, plans, small frustrations and pleasures, etc.). No two texts should share a \
scenario.
- Use plain, ordinary language. No academic, clinical, or theoretical terminology.
- Do NOT name, label, or describe the target quality or its near-synonyms. It must be \
shown through concrete content, not stated.
- Keep every text grounded in ordinary, everyday personal life (home, friends, family, \
work, errands, hobbies, small plans and feelings). Avoid any specialized, institutional, \
political, financial, or online-platform setting; write as a private individual in \
day-to-day life, not as a member of any organization or online community.
- Each text is self-contained and clearly distinct from the others.

OUTPUT FORMAT:
Return ONLY a JSON array of exactly {n} objects, each with a single "text" field:
[{{"text": "..."}}, {{"text": "..."}}, ...]
Return EXACTLY {n} entries and no other text."""


def _register_system_prompt(n: int, register_description: str) -> str:
    """Return the system prompt of the register style, which names the format of the texts."""
    return f"""\
You write short, realistic, first-person texts in a SPECIFIC FORMAT, of the kind an \
ordinary person might actually produce. {register_description}

The writer is a normal person going about their life; they are NOT aware of any rubric, \
theory, scale, or measurement, and they are not trying to demonstrate any trait or \
quality. Whatever quality the task asks for must emerge naturally from what the person \
says and how they say it -- never by naming it or describing themselves in those terms.

STYLE REQUIREMENTS:
- Write in the FIRST PERSON, as a real, specific person with their own situation.
- Honor the format described above for every one of the {n} texts.
- Vary the everyday context across the {n} texts (home, work, relationships, hobbies, \
errands, plans, small frustrations and pleasures, etc.). No two texts should share a \
scenario.
- Use plain, ordinary language. No academic, clinical, or theoretical terminology.
- Do NOT name, label, or describe the target quality or its near-synonyms. It must be \
shown through concrete content, not stated.
- Keep every text grounded in ordinary, everyday personal life. Avoid any specialized, \
institutional, political, financial, or online-platform setting; write as a private \
individual in day-to-day life, not as a member of any organization or community.
- Each text is self-contained and clearly distinct from the others.

OUTPUT FORMAT:
Return ONLY a JSON array of exactly {n} objects, each with a single "text" field:
[{{"text": "..."}}, {{"text": "..."}}, ...]
Return EXACTLY {n} entries and no other text."""


def system_prompt(style: Style, n: int, register_description: str | None) -> str:
    """Build the system prompt of a generation call.

    Args:
        style: ``"neutral"`` leaves the medium of the texts open; ``"register"`` names it.
        n: Number of texts the writer is asked for.
        register_description: One or two imperative instruction sentences that say what
            format the texts take and how they sound, for example "Write each text as a
            short piece of written feedback on a colleague's draft report, in plain,
            slightly informal wording."; inserted unchanged after "...might actually
            produce." and before "Honor the format described above". Required for
            ``"register"`` and ignored for ``"neutral"``. A description that does not end
            with ``.``, ``!`` or ``?`` is used as given and logged as a warning.

    Returns:
        The system prompt.

    Raises:
        ConfigurationError: If the style is unknown or ``"register"`` has no description.
    """
    if style == "neutral":
        return _neutral_system_prompt(n)
    if style == "register":
        if register_description is None or not register_description.strip():
            raise ConfigurationError("style 'register' needs a non-empty register_description")
        if register_description.rstrip()[-1] not in SENTENCE_ENDINGS:
            logger.warning(
                "register_description %r does not end with a full stop; it is inserted "
                "unchanged after 'might actually produce.' and should read as an instruction "
                "sentence",
                register_description,
            )
        return _register_system_prompt(n, register_description)
    raise ConfigurationError(f"unknown style {style!r}; use one of {STYLES}")


def _scope_block(cell: Cell, rubric: Rubric) -> str:
    """Name the target quality and say what it means, for the construct or one attribute."""
    if cell.target == TARGET_CONSTRUCT:
        return (
            f'TARGET QUALITY: "{rubric.concept}"\n\n'
            f"WHAT THIS QUALITY MEANS (background for you, the writer; never quote it):\n"
            f"{rubric.concept_definition}"
        )
    attribute = rubric.attribute(cell.target)
    return (
        f'TARGET QUALITY: "{rubric.concept}" — specifically the facet '
        f'"{attribute.name}"\n\n'
        f"WHAT THIS FACET MEANS (background for you; never quote it):\n"
        f"{attribute.description}"
    )


def user_prompt(cell: Cell, rubric: Rubric, n: int) -> str:
    """Build the user prompt of one cell.

    Args:
        cell: The cell the texts are written for.
        rubric: The rubric that supplies the concept, the definitions, and the facet.
        n: Number of texts the writer is asked for.

    Returns:
        The user prompt.

    Raises:
        KeyError: If the cell names an attribute that the rubric does not have.
    """
    scope = _scope_block(cell, rubric)
    intensity = cell.intensity
    if cell.polarity == POSITIVE:
        pole_header = (
            f"You are writing texts that {intensity} exemplify the HIGH end of this "
            f"quality. The HIGH end looks like this in concrete terms:"
        )
    else:
        pole_header = (
            f"You are writing texts that {intensity} exemplify the active LOW end of "
            f"this quality — not the mere absence of it, but the opposite end as a "
            f"real, present state. The LOW end looks like this in concrete terms:"
        )
    return f"""\
{scope}

{pole_header}
{cell.pole_guidance}

Now write {n} short, first-person, everyday texts (about 1-5 sentences each) by \
different ordinary people, each of which {intensity} shows the end of the quality \
described above through natural, concrete content. Do not name or describe the \
quality; let it come through in what each person says. Vary the everyday situations \
widely across the {n} texts."""
