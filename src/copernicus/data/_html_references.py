"""The character references that ``strip_html`` decodes.

The decision which references to decode lives in one function, ``decode_character_references``,
so that a rule that decodes more references, or fewer, changes this module only.

A decimal reference closed by ``;`` that reaches the decoding step and holds more digits than the
interpreter converts to an integer (``sys.get_int_max_str_digits()``, 4,300 by default where
Python has the limit) makes ``html.unescape`` raise ``ValueError``, as in the paper's analysis
code; a reference that an earlier rule discarded with a comment or a tag never reaches it, and
one kept as the ``alt`` text of an image does. The error is not caught here, so the cleaning of
a text stops when such a reference reaches it.
"""

from __future__ import annotations

import re
from html import unescape
from html.entities import html5

__all__ = ["decode_character_references"]

# A character reference closed by ";": a decimal or hexadecimal number, or a name.
_CLOSED_REFERENCE = re.compile(r"&(?:#[0-9]+|#[xX][0-9A-Fa-f]+|[A-Za-z][A-Za-z0-9]*);")


def decode_character_references(text: str) -> str:
    """Decode the character references of a text that ``;`` closes, and no other ``&``.

    A reference is decoded when ``;`` closes it and it is a decimal number (``&#39;``), a
    hexadecimal number (``&#x27;``) or a name that the HTML standard defines (``&amp;``,
    ``&hellip;``). Every other ``&`` stays as written: a name without ``;`` and a name that the
    standard does not define (``&notaname;``). The standard itself decodes some legacy names
    without ``;`` in running text, which in plain text corrupts a word such as ``Kind&regards``
    and the ``&copy=2`` of a link; this function never does.

    Args:
        text: The text, after its markup is removed.

    Returns:
        The text with recognised references replaced by their decoded text.

    Raises:
        ValueError: If the text holds a decimal reference closed by ``;`` with more digits than
            the interpreter converts to an integer (``sys.get_int_max_str_digits()``, 4,300 by
            default where Python has the limit); ``html.unescape`` raises it.
    """
    return _CLOSED_REFERENCE.sub(_character, text)


def _character(match: re.Match[str]) -> str:
    """Return the character that a reference closed by ``;`` stands for.

    Args:
        match: A match of the reference.

    Returns:
        The character, or the reference itself when its name is not defined by the standard.

    Raises:
        ValueError: If a decimal reference holds more digits than the interpreter converts to an
            integer.
    """
    reference = match.group(0)
    if reference.startswith("&#"):
        return unescape(reference)
    return html5.get(reference[1:], reference)
