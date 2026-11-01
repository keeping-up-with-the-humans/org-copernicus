"""Splitting a text into consecutive windows of tokens."""

from __future__ import annotations

import logging
from typing import Any

from copernicus.errors import ConfigurationError

logger = logging.getLogger(__name__)

__all__ = ["split_into_token_chunks"]


def split_into_token_chunks(text: str, tokenizer: Any, max_tokens: int, reserved: int) -> list[str]:
    """Split a text into non-overlapping windows that fit the encoder after re-encoding.

    The text is tokenised once, the token ids are packed into windows of at most
    ``max_tokens - reserved`` tokens (the reserve is left for the special tokens the
    tokenizer adds around every window), and each window is decoded back to a string.
    A text that fits is returned as it is, as the only window.

    Args:
        text: The text to split.
        tokenizer: A Hugging Face style tokenizer with ``encode(text, add_special_tokens=False)``
            and ``decode(ids, skip_special_tokens=True)``.
        max_tokens: Window of the encoder in tokens.
        reserved: Tokens the tokenizer adds around every window.

    Returns:
        The windows in text order; never empty.

    Raises:
        ConfigurationError: If the window leaves no room for text after the reserve.
    """
    budget = max_tokens - reserved
    if budget <= 0:
        raise ConfigurationError(
            f"max_tokens {max_tokens} leaves no room for text after {reserved} reserved tokens"
        )
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) <= budget:
        return [text]
    return [
        tokenizer.decode(token_ids[start : start + budget], skip_special_tokens=True)
        for start in range(0, len(token_ids), budget)
    ]
