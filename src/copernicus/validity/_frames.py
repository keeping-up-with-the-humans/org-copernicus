"""Helpers shared by the modules that read the frames given to the validity panel."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence

import pandas as pd

from copernicus.errors import ConfigurationError

logger = logging.getLogger(__name__)

__all__ = ["MAX_LISTED", "listed", "require_columns"]

MAX_LISTED = 10


def listed(values: Iterable[object]) -> str:
    """Render up to ``MAX_LISTED`` values, saying how many were left out."""
    items = [str(value) for value in values]
    shown = ", ".join(items[:MAX_LISTED])
    rest = len(items) - MAX_LISTED
    return f"{shown}, ... ({rest} more)" if rest > 0 else shown


def require_columns(frame: pd.DataFrame, columns: Sequence[str], what: str) -> None:
    """Refuse a frame that lacks one of ``columns``.

    Raises:
        ConfigurationError: Naming the absent columns and the frame's columns.
    """
    absent = [name for name in columns if name not in frame.columns]
    if absent:
        raise ConfigurationError(f"{what} needs column(s) {absent}; it has {list(frame.columns)}")
