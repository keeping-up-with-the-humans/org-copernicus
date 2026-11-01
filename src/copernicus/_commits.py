"""The full commit hash that pins the code of a model repository whose own code may run."""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

__all__ = ["COMMIT_HASH_LENGTH", "is_commit_hash"]

COMMIT_HASH_LENGTH = 40
COMMIT_HASH = re.compile(rf"[0-9a-f]{{{COMMIT_HASH_LENGTH}}}")


def is_commit_hash(value: object) -> bool:
    """Tell whether a value is a full commit hash: 40 lowercase hexadecimal characters.

    A branch or a tag names whatever commit it points to when the model is loaded, and a
    shortened hash can match more than one commit, so neither pins the code that runs. The
    Hugging Face hub writes commit hashes in lowercase.

    Args:
        value: The revision as given; ``None`` and non-strings are not commit hashes.

    Returns:
        True for a string of exactly 40 lowercase hexadecimal characters.
    """
    return isinstance(value, str) and COMMIT_HASH.fullmatch(value) is not None
