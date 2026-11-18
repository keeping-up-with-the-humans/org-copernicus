"""What an embedding pass refuses: a window without an attended token, a vector without length.

The encoder makes these checks with the ids of the texts, so its refusals name texts; the
pooling functions of `copernicus.embeddings.pooling`, which see one batch alone, name rows.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import NoReturn

import numpy as np

from copernicus.errors import ConfigurationError

logger = logging.getLogger(__name__)

__all__ = [
    "MAX_IDS_IN_MESSAGE",
    "NORM_FLOOR",
    "listed",
    "refuse_windows_without_token",
    "unit_rows",
]

MAX_IDS_IN_MESSAGE = 10
NORM_FLOOR = 1e-12
"""The shortest length that the encoder scales to unit length; it refuses a shorter vector."""
LACKING = {
    "mean": "so mean pooling has no token vector to average there",
    "cls": "so cls pooling has no first token to take there",
    "last_token": "so last-token pooling has no last token to take there",
}
"""What each pooling lacks in a window without an attended token, for the refusal."""
WITHOUT_TOKEN_REMEDY = (
    "leave these texts out, or change them so that the tokenizer gives every window a token"
)
WITHOUT_LENGTH_REMEDY = "leave these texts out or embed them with another encoder"


def listed(items: Sequence[str]) -> str:
    """Join the first ten items with commas, followed by the number of the others."""
    rest = len(items) - MAX_IDS_IN_MESSAGE
    return ", ".join(items[:MAX_IDS_IN_MESSAGE]) + (f" and {rest} more" if rest > 0 else "")


def _named(labels: Sequence[str]) -> list[str]:
    """Quote each text id once, in the order given."""
    return [repr(label) for label in dict.fromkeys(labels)]


def refuse_windows_without_token(pooling: str, labels: Sequence[str]) -> NoReturn:
    """Stop a pass whose tokenizer turned windows into no token, naming their texts.

    Args:
        pooling: ``"mean"``, ``"cls"`` or ``"last_token"``, the pooling that finds nothing
            to read.
        labels: The text id of each such window, in text order; a text with several such
            windows is named once.

    Raises:
        ConfigurationError: Always; the message counts the texts and names the first ten.
    """
    named = _named(labels)
    raise ConfigurationError(
        f"{len(named)} texts have a window that the tokenizer turned into no token, "
        f"{LACKING[pooling]}: {listed(named)}; {WITHOUT_TOKEN_REMEDY}"
    )


def unit_rows(matrix: np.ndarray, labels: Sequence[str], *, windows: bool) -> np.ndarray:
    """Divide every row by its length, refusing a row shorter than `NORM_FLOOR`.

    A row of length 1e-12 or more is divided by its length. A shorter row is refused: a zero
    row has no direction to keep, and the encoder treats every row shorter than 1e-12 alike.
    A row whose computed length is not finite is refused as well: it holds a value that is
    NaN or infinite, or the sum of its squares overflows.

    Args:
        matrix: One vector per row.
        labels: The text id of each row; a text named by several rows is named once.
        windows: Whether the rows are the pooled vectors of windows rather than the vectors
            of texts, which the message says.

    Returns:
        The rows of unit length, in their order.

    Raises:
        ConfigurationError: If a row is shorter than 1e-12 or its length is not finite; the
            message counts the texts of such rows and names the first ten.
    """
    with np.errstate(over="ignore", invalid="ignore"):
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    endless = np.flatnonzero(~np.isfinite(norms[:, 0]))
    if endless.size:
        named = _named([labels[index] for index in endless.tolist()])
        what = "a window whose pooled vector has a length" if windows else "a vector of a length"
        raise ConfigurationError(
            f"{len(named)} texts have {what} that is not finite (it holds a value that is NaN "
            "or infinite, or the sum of its squares overflows), which the encoder cannot scale "
            f"to unit length: {listed(named)}; {WITHOUT_LENGTH_REMEDY}"
        )
    short = np.flatnonzero(norms[:, 0] < NORM_FLOOR)
    if short.size:
        named = _named([labels[index] for index in short.tolist()])
        what = "a window whose pooled vector has a length" if windows else "a vector of length"
        raise ConfigurationError(
            f"{len(named)} texts have {what} below {NORM_FLOOR:g}, which the encoder does not "
            f"scale to unit length: {listed(named)}; {WITHOUT_LENGTH_REMEDY}"
        )
    return matrix / norms
