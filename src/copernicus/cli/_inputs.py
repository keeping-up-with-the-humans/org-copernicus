"""Option types and readers shared by the ``validity`` and ``baseline`` commands."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer

from copernicus.data import read_frame
from copernicus.errors import SettingsError

logger = logging.getLogger(__name__)

__all__ = ["TEXT_ID", "GoldOption", "parse_pairs", "read_gold"]

TEXT_ID = "text_id"
GoldOption = Annotated[
    Path,
    typer.Option(
        "--gold",
        exists=True,
        dir_okay=False,
        help="CSV or parquet file with a text_id column and the gold columns.",
    ),
]


def parse_pairs(values: Sequence[str], option: str, form: str) -> list[tuple[str, str]]:
    """Parse repeated ``LEFT=RIGHT`` option values into pairs.

    Args:
        values: The values as given on the command line.
        option: The option's name, for messages.
        form: The expected form, for example ``KEY=COLUMN``, for messages.

    Returns:
        The pairs in the order given.

    Raises:
        SettingsError: If a value has no ``=``, one side is empty, or the left side is
            given twice.
    """
    pairs: list[tuple[str, str]] = []
    for value in values:
        left, separator, right = value.partition("=")
        if not separator or not left.strip() or not right.strip():
            raise SettingsError(f"{option} expects {form}, got {value!r}")
        if any(left.strip() == earlier for earlier, _ in pairs):
            raise SettingsError(f"{option} names {left.strip()!r} twice")
        pairs.append((left.strip(), right.strip()))
    return pairs


def read_gold(path: Path) -> pd.DataFrame:
    """Read a gold file, keeping the text ids as strings.

    Args:
        path: CSV, parquet, or JSONL file with a ``text_id`` column.

    Returns:
        The gold table.
    """
    frame = read_frame(path, protected_columns=(TEXT_ID,))
    logger.info("read %d gold rows from %s", len(frame), path)
    return frame
