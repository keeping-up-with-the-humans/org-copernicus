"""Message helpers shared by the correction modules."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

__all__ = ["list_first_ten"]

MAX_IDS_IN_MESSAGE = 10


def list_first_ten(values: Sequence[Any]) -> str:
    """Render up to ten values for an error message, with the count of the rest.

    Args:
        values: The offending ids or names.

    Returns:
        The first ten values in ``repr`` form, followed by ``... (N more)`` when there
        are more.
    """
    shown = ", ".join(repr(str(value)) for value in values[:MAX_IDS_IN_MESSAGE])
    rest = len(values) - MAX_IDS_IN_MESSAGE
    return f"{shown}, ... ({rest} more)" if rest > 0 else shown
