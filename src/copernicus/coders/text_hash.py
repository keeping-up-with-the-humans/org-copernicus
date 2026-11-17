"""The text hash: the fingerprint that binds a stored coding to the text it was made on.

An earlier version of the library collapsed every run of whitespace in a text to one space
before it sent the text to a model and hashed it. A store of that version therefore holds,
for a text with a line break, a tab, a run of spaces or other whitespace between its words, the
hash of the collapsed text, which this version tells apart from a text that changed.
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections.abc import Mapping, Sequence, Set

from copernicus.data.table import collapse_whitespace

logger = logging.getLogger(__name__)

__all__ = [
    "COLLAPSED_CODING",
    "COLLAPSED_FORM",
    "TEXT_HASH_PATTERN",
    "collapsed_hash",
    "hash_text",
    "split_collapsed",
]

TEXT_HASH_PATTERN = re.compile(r"[0-9a-f]{64}")
"""The form of a text hash: 64 lowercase hexadecimal characters."""
COLLAPSED_FORM = (
    "on the text with every run of whitespace collapsed to one space, as an earlier version "
    "of the library sent every text"
)
"""How a refusal says that texts were coded with their whitespace collapsed."""
COLLAPSED_CODING = (
    "this version of the library sends each text as given, with its line breaks, tabs and "
    "runs of spaces, so the prompt of such a text has changed: code these texts into a new "
    "store, or pass each of them with its whitespace collapsed "
    "(copernicus.data.table.collapse_whitespace) to continue the old coding"
)
"""Why texts coded with their whitespace collapsed are refused, and what to do."""


def hash_text(text: str) -> str:
    """Return the text hash of a text.

    The run store writes the hash of the text with every result row, so that resuming,
    merging, and the hybrid coder can tell whether a text id still holds the text it was
    coded on. The runner hashes the text exactly as the coder sends it,
    ``TextTable.text_of(text_id)``: the text as given, with only its ends stripped. Every
    character counts, whitespace between words included, so ``a b``, ``a  b``, ``a\\tb`` and
    ``a\\nb`` hash differently, as do ``ab`` and ``a b``; whitespace at the ends of a text is
    stripped by the table and so changes no hash.

    Args:
        text: The text as the coder sends it.

    Returns:
        The SHA-256 digest of the UTF-8 encoding of ``text``, as 64 lowercase hexadecimal
        characters.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def collapsed_hash(text: str) -> str:
    """Return the text hash of a text with every run of whitespace collapsed to one space.

    It is the hash that a store of an earlier version of the library holds for the text,
    because that version sent and hashed every text in this form.

    Args:
        text: The text as the coder sends it.

    Returns:
        ``hash_text(collapse_whitespace(text))``.
    """
    return hash_text(collapse_whitespace(text))


def split_collapsed(
    changed: Sequence[tuple[str, str]], stored: Mapping[str, Set[str]]
) -> tuple[list[str], list[str]]:
    """Split the texts whose stored hashes are not the hash of the text offered.

    A text whose id holds, in every stored cell, the `collapsed_hash` of the text was coded on
    it with its whitespace collapsed: by an earlier version of the library, or before the text
    gained the whitespace it holds now. Every other text was coded on another text. Each text is
    judged on its own, so two rows whose ids are equal as strings can fall on either side.

    Args:
        changed: ``(text id, collapsed_hash of the text offered)`` of each text whose stored
            hashes differ from the hash of the text offered.
        stored: The text hashes stored under each id of ``changed``.

    Returns:
        The ids of the texts coded on the collapsed text, and the ids of the texts coded on
        another text, each in the order of ``changed``.
    """
    on_collapsed: list[str] = []
    others: list[str] = []
    for text_id, collapsed in changed:
        (on_collapsed if stored[text_id] == {collapsed} else others).append(text_id)
    return on_collapsed, others
