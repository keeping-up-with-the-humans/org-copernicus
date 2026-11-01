"""Syntactic repairs of JSON text from LLM output that leave every string value intact."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["fix_common_json_errors", "parse_block", "repair_string_escapes"]

VALID_ESCAPES = frozenset('"\\/bfnrtu')
STRING_CLOSERS = frozenset(",]}:")
WHITESPACE = " \t\r\n"
TRAILING_COMMA = re.compile(r",\s*([}\]])")
PYTHON_LITERALS = (
    (re.compile(r"\bTrue\b"), "true"),
    (re.compile(r"\bFalse\b"), "false"),
    (re.compile(r"\bNone\b"), "null"),
)


def repair_string_escapes(text: str) -> tuple[str, bool]:
    r"""Escape stray quotes and fix invalid backslash escapes inside JSON strings.

    Inside a string value a backslash before an apostrophe is dropped (``\'`` is not a JSON
    escape and the passage had a plain apostrophe), a backslash before any other non-escape
    character is doubled (a literal backslash), and a double quote that is not followed by a
    structural character (``, ] } :`` after optional whitespace) is treated as content and
    escaped. Structural quotes are left alone.

    Args:
        text: Candidate JSON text.

    Returns:
        The repaired text and whether anything was rewritten.
    """
    out: list[str] = []
    in_string = False
    changed = False
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if not in_string:
            if c == '"':
                in_string = True
            out.append(c)
            i += 1
            continue
        if c == "\\":
            following = text[i + 1] if i + 1 < n else ""
            if following == "'":
                out.append("'")
                changed = True
                i += 2
            elif following in VALID_ESCAPES:
                out.append(c + following)
                i += 2
            else:
                out.append("\\\\")
                changed = True
                i += 1
            continue
        if c == '"':
            j = i + 1
            while j < n and text[j] in WHITESPACE:
                j += 1
            if j >= n or text[j] in STRING_CLOSERS:
                in_string = False
                out.append(c)
            else:
                out.append('\\"')
                changed = True
            i += 1
            continue
        out.append(c)
        i += 1
    return "".join(out), changed


def parse_block(candidate: str) -> tuple[Any, str] | None:
    """Parse a balanced block, trying the common fixes if the plain parse fails.

    Args:
        candidate: Text of one balanced ``{...}`` or ``[...]`` block.

    Returns:
        The parsed value with ``"balanced"``, or with ``"common_fixes"`` when it parses
        only after `fix_common_json_errors`; ``None`` when it does not parse.
    """
    try:
        return json.loads(candidate), "balanced"
    except ValueError:
        pass
    fixed = fix_common_json_errors(candidate)
    if not fixed:
        return None
    try:
        return json.loads(fixed), "common_fixes"
    except ValueError:
        return None


def fix_common_json_errors(text: str) -> str | None:
    """Fix common JSON formatting issues from LLM output, outside string values only.

    Converts single quotes to double quotes when the text holds no double quote. Then,
    outside JSON strings only, removes a comma before ``}`` or ``]`` and converts the
    Python literals ``True``, ``False``, ``None``; a verbatim passage such as "None of the
    figures" keeps its wording.

    Args:
        text: Candidate JSON text.

    Returns:
        The cleaned text, or ``None`` if ``text`` is empty.
    """
    if not text:
        return None

    cleaned = text.strip()
    if '"' not in cleaned and "'" in cleaned:
        cleaned = cleaned.replace("'", '"')
    return "".join(
        part if inside else _fix_outside_strings(part) for part, inside in _string_spans(cleaned)
    )


def _string_spans(text: str) -> list[tuple[str, bool]]:
    """Split JSON text into runs outside and inside string values.

    A string run includes its quotes. Inside a string a backslash escapes the next
    character, so an escaped quote does not end the string; an unterminated string runs
    to the end of the text.

    Args:
        text: Candidate JSON text.

    Returns:
        ``(run, inside_string)`` pairs that concatenate to ``text``.
    """
    spans: list[tuple[str, bool]] = []
    in_string = False
    start = i = 0
    while i < len(text):
        c = text[i]
        if in_string and c == "\\":
            i += 2
            continue
        if c == '"':
            end = i + 1 if in_string else i
            spans.append((text[start:end], in_string))
            start = end
            in_string = not in_string
        i += 1
    spans.append((text[start:], in_string))
    return spans


def _fix_outside_strings(text: str) -> str:
    """Remove trailing commas and convert Python literals in text that holds no string."""
    fixed = TRAILING_COMMA.sub(r"\1", text)
    for pattern, replacement in PYTHON_LITERALS:
        fixed = pattern.sub(replacement, fixed)
    return fixed
