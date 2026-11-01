"""Robust JSON extraction from LLM output.

Local models wrap JSON in code fences, add commentary, leave trailing commas, use single
quotes, or copy verbatim passages whose quotes and escapes break the syntax. The strategies
below are tried in order of increasing invasiveness.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Literal

from copernicus.errors import BackendError, SettingsError
from copernicus.llm._json_spans import (
    FENCE_PATTERNS,
    complete_objects,
    extract_balanced_braces,
    list_after,
    matching_close,
    object_span,
    strip_fences,
    truncated_array_start,
    unwrap_list_of_objects,
    wrapping_object,
)
from copernicus.llm.json_repair import fix_common_json_errors, parse_block, repair_string_escapes

logger = logging.getLogger(__name__)

__all__ = ["extract_json"]

EXCERPT_LENGTH = 500


Expect = Literal["any", "object", "array"]
EXPECTATIONS = ("any", "object", "array")


def extract_json(text: str, *, expect: Expect = "any") -> tuple[Any, str]:
    """Extract a JSON value from LLM output with several fallback strategies.

    The strategies are tried in this order: ``direct`` parse, ``fence`` (code fence),
    ``balanced`` (first balanced ``{...}`` block, then first balanced ``[...]`` block),
    ``string_escapes`` (repair of stray quotes and invalid escapes inside strings), and
    ``common_fixes`` (single quotes, and trailing commas and Python literals outside string
    values). A balanced block that parses only after the common fixes reports
    ``common_fixes``. No strategy changes the text of a string value.

    ``expect`` changes the ``balanced`` strategy only. ``"object"`` and ``"any"`` try the
    first ``{...}`` block, then the first ``[...]`` block. ``"array"`` tries the ``[...]``
    blocks one after the other, in order of appearance, and returns the first that parses as
    a list, so neither a bracketed word in a preamble nor the first object of a bare array is
    read instead of the array. When no block parses, the order continues:

    1. ``truncated_array``: the complete top-level objects of an array that the output cut
       off (the first unclosed ``[`` that holds any), returned as a list.
    2. ``array_after_open_bracket``: the first complete bracket block that parses as a list
       and follows an unmatched ``[`` of a prose preamble (``"Here you go [10 texts:\n[...]"``).
       The prose between the two brackets must hold no brace, so a list nested inside an
       object that the output cut off is never returned in its place.
    3. The first ``{...}`` block.

    Under ``"array"`` a parsed object (direct, fenced, or a brace block) that holds exactly
    one list of objects among its values and no string value of its own, as in
    ``{"texts": [{...}, ...]}``, is replaced by that list with the strategy ``wrapped_array``;
    any other object, a record such as ``{"text": "outer", "items": [...]}`` included, is
    returned as it is. The same holds for the one object of a ``truncated_array`` salvage
    (``'Here [x:\n{"texts": [{"text": "a"}]}'``), provided no further object starts after it:
    a record that a cut-off second object follows stays a record of the salvaged list.

    Args:
        text: Raw model output.
        expect: The kind of JSON value the caller wants: ``"any"``, ``"object"``, or
            ``"array"``.

    Returns:
        The parsed value and the name of the strategy that parsed it.

    Raises:
        BackendError: If no strategy yields a JSON value; ``last_output`` holds the first
            500 characters of ``text``.
        SettingsError: If ``expect`` is not one of the three kinds.
    """
    if expect not in EXPECTATIONS:
        raise SettingsError(f"expect must be one of {EXPECTATIONS}, got {expect!r}")
    stripped = text.strip() if isinstance(text, str) else ""

    found = _try_direct(stripped)
    if found is None:
        found = _try_fence(stripped)
    if found is None:
        found = _try_balanced(stripped, expect)
    if found is None:
        found = _try_string_escapes(stripped)
    if found is None:
        found = _try_common_fixes(stripped)
    if found is None:
        excerpt = text[:EXCERPT_LENGTH] if isinstance(text, str) else ""
        raise BackendError("no JSON object found", attempts=0, last_output=excerpt)
    if expect == "array":
        wrapper = wrapping_object(found, stripped)
        unwrapped = unwrap_list_of_objects(wrapper) if wrapper is not None else None
        if unwrapped is not None:
            return unwrapped, "wrapped_array"
    return found


def _try_direct(text: str) -> tuple[Any, str] | None:
    """Parse the whole text as JSON."""
    try:
        return json.loads(text), "direct"
    except ValueError:
        return None


def _try_fence(text: str) -> tuple[Any, str] | None:
    """Parse the content of the first code fence that holds valid JSON."""
    for pattern in FENCE_PATTERNS:
        match = re.search(pattern, text, re.DOTALL)
        if match is None:
            continue
        try:
            return json.loads(match.group(1).strip()), "fence"
        except ValueError:
            continue
    return None


def _try_balanced(text: str, expect: Expect) -> tuple[Any, str] | None:
    """Parse balanced blocks of the text, in the order the expectation asks for.

    Without an array expectation the first brace block comes before the first bracket
    block. With one, every bracket block is tried in turn, then the complete objects of a
    cut-off array, then the complete array behind an unmatched bracket, then the first brace
    block.
    """
    if expect != "array":
        found = extract_balanced_braces(text, "{", "}")
        return found if found is not None else extract_balanced_braces(text, "[", "]")
    found = _try_array_spans(text)
    if found is None:
        found = _try_truncated_array(text)
    if found is None:
        found = _try_array_after_open_bracket(text)
    return found if found is not None else extract_balanced_braces(text, "{", "}")


def _try_array_spans(text: str) -> tuple[Any, str] | None:
    """Parse the bracket blocks one after the other and return the first that is a list.

    A block that does not parse is skipped as a whole, so a list nested inside it is never
    returned in its place; an unclosed bracket ends the search.
    """
    start = text.find("[")
    while start >= 0:
        end = matching_close(text, start, "[", "]")
        if end is None:
            return None
        found = parse_block(text[start : end + 1])
        if found is not None and isinstance(found[0], list):
            return found
        start = text.find("[", end + 1)
    return None


def _try_truncated_array(text: str) -> tuple[Any, str] | None:
    """Return the complete top-level objects of an array that the output cut off.

    The unclosed brackets are tried in order of appearance, the outermost (longest) first;
    the objects of the first one that holds any complete object are returned.
    """
    start = truncated_array_start(text)
    if start is None:
        return None
    return complete_objects(text, start)[0], "truncated_array"


def _try_array_after_open_bracket(text: str) -> tuple[Any, str] | None:
    """Return the first complete list that follows an unmatched ``[`` of a prose preamble.

    For each unclosed ``[``, in order of appearance, the later ``[`` blocks are tried while
    the text between the unclosed bracket and the block holds no brace; the first block that
    parses as a list is returned with the strategy ``array_after_open_bracket``.
    """
    start = text.find("[")
    while start >= 0:
        if matching_close(text, start, "[", "]") is None:
            found = list_after(text, start)
            if found is not None:
                return found, "array_after_open_bracket"
        start = text.find("[", start + 1)
    return None


def _try_string_escapes(text: str) -> tuple[Any, str] | None:
    """Repair stray quotes and invalid escapes inside strings, then parse.

    Verbatim passages copied from short messages or long documents carry double quotes and
    backslash-escaped apostrophes that make an otherwise complete object unparseable; the
    repair is purely syntactic and leaves the content intact.
    """
    repaired, changed = repair_string_escapes(object_span(strip_fences(text)))
    if not changed:
        return None
    fixed = fix_common_json_errors(repaired)
    candidates = [repaired] if fixed is None else [repaired, fixed]
    for candidate in candidates:
        try:
            return json.loads(candidate), "string_escapes"
        except ValueError:
            continue
    return None


def _try_common_fixes(text: str) -> tuple[Any, str] | None:
    """Apply the common fixes to the whole text, then parse."""
    cleaned = fix_common_json_errors(text)
    if not cleaned:
        return None
    try:
        return json.loads(cleaned), "common_fixes"
    except ValueError:
        return None
