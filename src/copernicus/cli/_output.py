"""Output files and summary rows shared by the commands that write tables.

Every file is written through a partial file beside it and moved into place in one step, so a
write that fails keeps the old file and leaves no partial one.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from copernicus._io import write_atomically, write_json
from copernicus.errors import SettingsError

logger = logging.getLogger(__name__)

__all__ = [
    "check_stem_output",
    "check_table_output",
    "format_rows",
    "write_description",
    "write_report",
    "write_table",
]

SUFFIX_CSV = ".csv"
SUFFIX_PARQUET = ".parquet"
LABEL_WIDTH = 18
LABEL_GAP = 1


def check_table_output(out: Path) -> None:
    """Refuse an output file that cannot be written as a table, before any work is done.

    Args:
        out: The output file; its suffix must be ``.csv`` or ``.parquet`` and its folder
            must exist.

    Raises:
        SettingsError: If the suffix is another one or the folder does not exist.
    """
    if out.suffix not in (SUFFIX_CSV, SUFFIX_PARQUET):
        raise SettingsError(f"output suffix must be .csv or .parquet, got {out.suffix!r}")
    if not out.parent.is_dir():
        raise SettingsError(f"folder of the output file does not exist: {out.parent}")


def check_stem_output(stem: Path) -> None:
    """Refuse an output stem whose folder does not exist, before any work is done.

    Args:
        stem: Path without suffix; the files ``<stem>.npz`` and ``<stem>.json`` follow.

    Raises:
        SettingsError: If the folder of the stem does not exist.
    """
    if not stem.parent.is_dir():
        raise SettingsError(f"folder of the output files does not exist: {stem.parent}")


def write_table(frame: pd.DataFrame, out: Path) -> None:
    """Write a table to CSV or parquet, as the suffix says, atomically.

    Args:
        frame: The table.
        out: The output file, checked by :func:`check_table_output`.

    Raises:
        OSError: If the file cannot be written; an earlier file of that name is kept.
    """
    if out.suffix == SUFFIX_CSV:
        write_atomically(out, lambda handle: frame.to_csv(handle, index=False))
    else:
        write_atomically(out, lambda handle: frame.to_parquet(handle, index=False))
    logger.info("wrote %d rows to %s", len(frame), out)


def write_report(path: Path, text: str) -> None:
    """Write a plain-text report as UTF-8, atomically.

    Args:
        path: The report file; its folder must exist.
        text: The report, with its final newline.

    Raises:
        OSError: If the file cannot be written; an earlier file of that name is kept.
    """
    write_atomically(path, lambda handle: handle.write(text.encode("utf-8")))
    logger.info("wrote the report to %s", path)


def write_description(out: Path, content: Mapping[str, Any]) -> Path:
    """Write the description of an output table to ``<out>.json`` beside it.

    Args:
        out: The output table.
        content: JSON values describing how the table was made.

    Returns:
        The path of the description.
    """
    path = out.with_name(out.name + ".json")
    write_json(path, content)
    logger.info("wrote the description of %s to %s", out, path)
    return path


def format_rows(rows: Iterable[tuple[str, str]]) -> str:
    """Format ``(label, value)`` pairs as aligned ``label  value`` lines.

    The label column is `LABEL_WIDTH` characters wide, or wider when a label needs it: at
    least one space always parts the longest label from its value, so no label runs
    into its value and the values of one summary start in one column.

    Args:
        rows: The pairs, one per line.

    Returns:
        The lines joined by newlines.
    """
    pairs = list(rows)
    width = max([LABEL_WIDTH, *(len(label) + LABEL_GAP for label, _ in pairs)])
    return "\n".join(f"{label:<{width}}{value}" for label, value in pairs)
