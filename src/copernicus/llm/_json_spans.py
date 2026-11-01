"""The span and brace helpers behind ``copernicus.llm.json_extract``.

They locate the text of a JSON value inside model output without trying to parse it
themselves: the code-fence patterns, the delimiter that closes a block (outside strings), the
complete objects of an array that was cut off, and the object that may wrap an answer. The
strategies that parse what these helpers find live in ``json_extract``.
"""

from __future__ import annotations

import re
from typing import Any

from copernicus.llm.json_repair import parse_block

__all__ = [
    "FENCE_PATTERNS",
    "after_string",
    "complete_objects",
    "extract_balanced_braces",
    "has_brace",
    "list_after",
    "matching_close",
    "object_span",
    "object_starts_after",
    "strip_fences",
    "truncated_array_start",
    "unwrap_list_of_objects",
    "wrapping_object",
]

FENCE_PATTERNS = (
    r"```json\s*\n?(.*?)\n?\s*```",
    r"```\s*\n?(.*?)\n?\s*```",
    r"`([\{\[].*?[\}\]])`",
)


def wrapping_object(found: tuple[Any, str], text: str) -> dict[str, Any] | None:
    """Return the object that may wrap the answer: a parsed object or a salvage's only object.

    The only object of a ``truncated_array`` salvage counts only when no further object
    starts after it: a record that a second, cut-off object follows is a record of the
    array, not a wrapper.
    """
    value, strategy = found
    if isinstance(value, dict):
        return value
    if strategy == "truncated_array" and len(value) == 1 and isinstance(value[0], dict):
        start = truncated_array_start(text)
        if start is not None and not object_starts_after(text, complete_objects(text, start)[1]):
            return value[0]
    return None


def truncated_array_start(text: str) -> int | None:
    """Return the index of the outermost unclosed ``[`` that holds a complete object."""
    start = text.find("[")
    while start >= 0:
        if matching_close(text, start, "[", "]") is None and complete_objects(text, start)[0]:
            return start
        start = text.find("[", start + 1)
    return None


def object_starts_after(text: str, index: int) -> bool:
    """Tell whether a ``{`` outside a string follows ``index``."""
    i = index
    while i < len(text):
        if text[i] == '"':
            i = after_string(text, i)
        elif text[i] == "{":
            return True
        else:
            i += 1
    return False


def list_after(text: str, open_index: int) -> list[Any] | None:
    """Return the first complete list behind ``open_index`` that no brace separates from it."""
    start = text.find("[", open_index + 1)
    while start >= 0 and not has_brace(text[open_index + 1 : start]):
        end = matching_close(text, start, "[", "]")
        if end is not None:
            found = parse_block(text[start : end + 1])
            if found is not None and isinstance(found[0], list):
                return found[0]
        start = text.find("[", start + 1)
    return None


def has_brace(segment: str) -> bool:
    """Tell whether the segment holds a curly brace."""
    return "{" in segment or "}" in segment


def unwrap_list_of_objects(value: dict[str, Any]) -> list[dict[str, Any]] | None:
    """Return the one non-empty list of objects among the values of an object, if exactly one.

    An object with a string value of its own is a record, not a wrapper, and is not unwrapped.
    """
    if any(isinstance(item, str) for item in value.values()):
        return None
    lists = [
        item
        for item in value.values()
        if isinstance(item, list) and item and all(isinstance(entry, dict) for entry in item)
    ]
    return lists[0] if len(lists) == 1 else None


def complete_objects(text: str, start: int) -> tuple[list[dict[str, Any]], int]:
    """Collect the objects directly inside the array opened at ``start`` that are complete.

    Nested arrays and strings between the objects are stepped over; the collection stops
    at the first object or array that never closes.

    Returns:
        The complete objects and the index just after the last of them (``start + 1`` when
        there is none).
    """
    objects: list[dict[str, Any]] = []
    resume = start + 1
    i = start + 1
    while i < len(text):
        c = text[i]
        if c == '"':
            i = after_string(text, i)
        elif c in "{[":
            end = matching_close(text, i, c, "}" if c == "{" else "]")
            if end is None:
                break
            found = parse_block(text[i : end + 1]) if c == "{" else None
            if found is not None and isinstance(found[0], dict):
                objects.append(found[0])
                resume = end + 1
            i = end + 1
        elif c == "]":
            break
        else:
            i += 1
    return objects, resume


def after_string(text: str, start: int) -> int:
    """Return the index after the string that opens at ``start``, or the end of the text."""
    i = start + 1
    while i < len(text):
        if text[i] == "\\":
            i += 2
        elif text[i] == '"':
            return i + 1
        else:
            i += 1
    return len(text)


def strip_fences(text: str) -> str:
    """Return the content of the first code fence, or the text itself."""
    for pattern in FENCE_PATTERNS[:2]:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
    return text


def object_span(text: str) -> str:
    """Return the text from its first ``{`` to its last ``}`` (the text itself if absent)."""
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return text
    return text[start : end + 1]


def extract_balanced_braces(text: str, open_char: str, close_char: str) -> tuple[Any, str] | None:
    """Parse the first balanced block delimited by ``open_char`` and ``close_char``.

    Args:
        text: Text that may contain the block.
        open_char: Opening delimiter.
        close_char: Closing delimiter.

    Returns:
        The parsed value with strategy ``balanced``, or ``common_fixes`` if the block parses
        only after the common fixes; ``None`` if the block is missing or does not parse.
    """
    start = text.find(open_char)
    if start < 0:
        return None
    end = matching_close(text, start, open_char, close_char)
    return None if end is None else parse_block(text[start : end + 1])


def matching_close(text: str, start: int, open_char: str, close_char: str) -> int | None:
    """Return the index of the delimiter that closes the one at ``start``.

    Only the two delimiters are counted, outside double-quoted strings; a backslash escapes
    the next character.

    Returns:
        The index of the closing delimiter, or ``None`` if the block never closes.
    """
    depth = 0
    in_string = False
    escape_next = False
    for i in range(start, len(text)):
        c = text[i]
        if escape_next:
            escape_next = False
        elif c == "\\":
            escape_next = True
        elif c == '"':
            in_string = not in_string
        elif in_string:
            continue
        elif c == open_char:
            depth += 1
        elif c == close_char:
            depth -= 1
            if depth == 0:
                return i
    return None
