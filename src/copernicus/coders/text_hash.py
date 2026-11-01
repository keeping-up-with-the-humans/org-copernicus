"""The text hash: the fingerprint that binds a stored coding to the text it was made on."""

from __future__ import annotations

import hashlib
import logging
import re

logger = logging.getLogger(__name__)

__all__ = ["TEXT_HASH_PATTERN", "hash_text"]

TEXT_HASH_PATTERN = re.compile(r"[0-9a-f]{64}")
"""The form of a text hash: 64 lowercase hexadecimal characters."""


def hash_text(text: str) -> str:
    """Return the text hash of a text.

    The run store writes the hash of the text with every result row, so that resuming,
    merging, and the hybrid coder can tell whether a text id still holds the text it was
    coded on. The runner hashes the text exactly as the coder receives it,
    ``TextTable.text_of(text_id)``: the text of the table, in which every run of
    whitespace is one space and the ends are stripped. Runs of whitespace thus collapse to
    one space before hashing, so a change in the amount or the kind of whitespace between
    words, or at the ends of the text, is no change; inserting or removing a space between
    letters is one (``ab`` and ``a b`` hash differently), as is any other change of a
    character.

    Args:
        text: The text as the coder receives it.

    Returns:
        The SHA-256 digest of the UTF-8 encoding of ``text``, as 64 lowercase hexadecimal
        characters.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
