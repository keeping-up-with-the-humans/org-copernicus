"""The steps of ``strip_html`` that take more than one substitution of a pattern.

They are the text of an image, the scan that replaces comments and tags in one pass from left to
right, and the svg and math spans that a scan selects. Each step moves forward through the text:
none reads the rest of the text again from every comment opener, or once per level of nesting,
which would make the time grow with the square of the length.
"""

from __future__ import annotations

import re

from copernicus.data._html_syntax import (
    ATTRIBUTE,
    COMMENT_CLOSER,
    COMMENT_OPENER,
    FOREIGN_ELEMENT_TAG,
    FOREIGN_TAG,
)

__all__ = [
    "image_alt_text",
    "outermost_foreign_elements",
    "remove_foreign_elements",
    "replace_comments_and_tags",
]


def image_alt_text(match: re.Match[str]) -> str:
    """Return the value of the last double-quoted ``alt`` attribute of an image tag, or ``""``.

    Args:
        match: A match of the image tag, with its attributes in the group ``attributes``.

    Returns:
        The alt text, or an empty string when the tag has no double-quoted ``alt`` attribute.
    """
    alt_text = ""
    for attribute in ATTRIBUTE.finditer(match.group("attributes")):
        value = attribute.group("value")
        if attribute.group("name") == "alt" and value is not None and value.startswith('"'):
            alt_text = value[1:-1]
    return alt_text


def replace_comments_and_tags(text: str, tag_pattern: re.Pattern[str], begin: int, end: int) -> str:
    """Return ``text[begin:end]`` with every comment and every tag of ``tag_pattern`` as a space.

    The span is read once, from left to right, as an HTML reader reads it: where a comment or a
    tag starts, all of it is replaced, so a ``<!--`` inside a quoted attribute value belongs to its
    tag and a tag that starts inside a comment belongs to the comment. A comment runs from
    ``<!--`` to the first ``-->`` after it; an opener that no ``-->`` follows is no comment and
    stays text, and then no later opener is one either. Every search starts where the previous
    search of its kind left off, so no search goes back over the span. Do not match comments with
    a pattern such as ``<!--.*?-->``: from every opener that no ``-->`` follows it reads to the end
    of the text again, which takes time that grows with the square of the length.

    Args:
        text: The text.
        tag_pattern: A compiled pattern of a tag; it must not match a comment.
        begin: Where the span begins.
        end: Where the span ends; nothing from here on is read.

    Returns:
        The span with every comment and every tag replaced by a space.
    """
    pieces: list[str] = []
    cursor = begin
    tag = tag_pattern.search(text, cursor, end)
    opener = text.find(COMMENT_OPENER, cursor, end)
    closer = -1
    while True:
        if tag is not None and tag.start() < cursor:
            tag = tag_pattern.search(text, cursor, end)
        if opener != -1 and opener < cursor:
            opener = text.find(COMMENT_OPENER, cursor, end)
        if opener != -1 and closer < opener + len(COMMENT_OPENER):
            closer = text.find(COMMENT_CLOSER, opener + len(COMMENT_OPENER), end)
            if closer == -1:
                opener = -1
        if opener != -1 and (tag is None or opener < tag.start()):
            pieces.append(text[cursor:opener])
            cursor = closer + len(COMMENT_CLOSER)
        elif tag is not None:
            pieces.append(text[cursor : tag.start()])
            cursor = tag.end()
        else:
            break
        pieces.append(" ")
    pieces.append(text[cursor:end])
    return "".join(pieces)


def outermost_foreign_elements(text: str) -> list[tuple[int, int, int, int]]:
    """Return the outermost complete svg and math elements of the text, in the order of the text.

    The tags of ``FOREIGN_ELEMENT_TAG`` are read once, from left to right. An accepted start tag
    opens an element. An end tag closes the innermost open element if their names agree, without
    regard to case, and the closed element is recorded; otherwise it stays content of that
    element, or, with no element open, is left to the last rule. A ``<svg`` or ``<math`` followed
    by ASCII white space, ``/`` or ``>`` that begins no accepted start tag, such as a self-closing
    ``<svg/>``, clears the elements that are open at that place: none of them is recorded, while
    an element recorded before stays recorded and one that opens after it can still be. This
    scan does not see comments, so such a tag inside a comment counts as well. An element that is
    never closed is not recorded. When an element is recorded, the elements recorded inside it
    are dropped, so only the outermost ones are returned. Do not reduce nested elements one level
    per pass over the text: that reads the whole text once per level, which takes time that grows
    with the square of the length.

    Args:
        text: The text.

    Returns:
        For each element, where its start tag begins, where its content begins and ends, and
        where its end tag ends.
    """
    open_elements: list[tuple[str, int, int]] = []
    elements: list[tuple[int, int, int, int]] = []
    for tag in FOREIGN_ELEMENT_TAG.finditer(text):
        if tag.group("start_name") is not None:
            open_elements.append((tag.group("start_name").lower(), tag.start(), tag.end()))
        elif tag.group("other_start_name") is not None:
            open_elements.clear()
        elif open_elements and open_elements[-1][0] == tag.group("end_name").lower():
            _, element_start, content_start = open_elements.pop()
            while elements and elements[-1][0] > element_start:
                elements.pop()
            elements.append((element_start, content_start, tag.start(), tag.end()))
    return elements


def remove_foreign_elements(text: str) -> str:
    """Clean each span that ``outermost_foreign_elements`` selects, and leave the rest of the text.

    A selected span's own start and end tags become a space each; inside it every comment and
    every tag that ``FOREIGN_TAG`` matches becomes a space, the tags of the svg and math elements
    nested in it included, and the text stays. An outer element that is never completed, or that
    a reset clears, is not selected, while a complete element nested in it is selected on its own.
    What this step leaves, outside the spans and inside them, passes to the last rule.

    Args:
        text: The text.

    Returns:
        The text with the tags and comments of the selected spans replaced by spaces.
    """
    pieces: list[str] = []
    cursor = 0
    for element_start, content_start, content_end, element_end in outermost_foreign_elements(text):
        pieces.append(text[cursor:element_start])
        pieces.append(" ")
        pieces.append(replace_comments_and_tags(text, FOREIGN_TAG, content_start, content_end))
        pieces.append(" ")
        cursor = element_end
    pieces.append(text[cursor:])
    return "".join(pieces)
