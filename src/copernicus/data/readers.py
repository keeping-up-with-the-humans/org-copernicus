"""Read tabular text files into data frames, dispatching on the file suffix."""

from __future__ import annotations

import json
import logging
from collections.abc import Collection, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from copernicus.errors import TextTableError

logger = logging.getLogger(__name__)

__all__ = ["PANDAS_DEFAULT_NA_STRINGS", "READER_SUFFIXES", "read_frame"]

READER_SUFFIXES = (".csv", ".parquet", ".jsonl")
PANDAS_DEFAULT_NA_STRINGS = (
    "",
    "#N/A",
    "#N/A N/A",
    "#NA",
    "-1.#IND",
    "-1.#QNAN",
    "-NaN",
    "-nan",
    "1.#IND",
    "1.#QNAN",
    "<NA>",
    "N/A",
    "NA",
    "NULL",
    "NaN",
    "None",
    "n/a",
    "nan",
    "null",
)
EMPTY_FIELD_ONLY = ("",)


def read_frame(path: Path, *, protected_columns: Sequence[str] = ()) -> pd.DataFrame:
    """Read a CSV, parquet, or JSON-lines file into a data frame.

    In CSV and JSON-lines files the protected columns (the id, text, and group columns)
    are read as text: an id ``007`` stays ``"007"``, a text ``12345`` stays a text, and a
    JSON number in such a column becomes its decimal text. In a CSV file only an empty
    field counts as missing in a protected column, so a text, id, or group with the
    literal content ``NA`` or ``null`` is not lost; every other column keeps pandas' usual
    missing-value strings (``NA``, ``N/A``, ``null``, ``nan``, and so on), so a numeric
    column that holds them stays numeric. A JSON-lines file keeps every other value as
    JSON gives it: numbers stay numbers, strings stay strings, and no column is converted
    to dates. Parquet columns keep their stored types.

    Args:
        path: File to read; the suffix (``.csv``, ``.parquet``, ``.jsonl``) selects
            the reader.
        protected_columns: Columns read as text in CSV and JSON-lines files; names the
            file does not have are ignored.

    Returns:
        The file contents as a data frame.

    Raises:
        TextTableError: If the suffix is not one of the supported suffixes, or the file
            cannot be parsed (malformed CSV, JSON, or parquet, or text that is not UTF-8);
            the message names the file.
    """
    suffix = path.suffix.lower()
    if suffix not in READER_SUFFIXES:
        raise TextTableError(
            f"cannot read {path.name!r}: unsupported file suffix {suffix!r}; "
            f"supported suffixes are {', '.join(READER_SUFFIXES)}"
        )
    try:
        if suffix == ".csv":
            return _read_csv(path, protected_columns)
        if suffix == ".parquet":
            return pd.read_parquet(path)
        return _read_jsonl(path, protected_columns)
    except ValueError as error:
        # The parsers report malformed content, undecodable text included, as ValueError.
        raise TextTableError(f"cannot read {path.name!r}: {error}") from error


def _read_csv(path: Path, protected_columns: Collection[str]) -> pd.DataFrame:
    """Read a CSV file with the protected columns as text that only an empty field misses."""
    header = pd.read_csv(path, nrows=0).columns
    na_values = {
        column: list(EMPTY_FIELD_ONLY if column in protected_columns else PANDAS_DEFAULT_NA_STRINGS)
        for column in header
    }
    text_columns = {column: str for column in header if column in protected_columns}
    return pd.read_csv(path, keep_default_na=False, na_values=na_values, dtype=text_columns)


def _read_jsonl(path: Path, protected_columns: Collection[str]) -> pd.DataFrame:
    """Read a JSON-lines file with the protected values as text and no other conversion.

    The standard JSON parser reads the lines, so no date or number inference applies on
    any pandas version. Blank lines are skipped.

    Raises:
        TextTableError: If a line is not valid JSON or not a JSON object; the message
            names the file and the line number.
    """
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise TextTableError(
                    f"cannot read {path.name!r}: line {number} is not valid JSON ({error})"
                ) from error
            if not isinstance(record, dict):
                raise TextTableError(
                    f"cannot read {path.name!r}: line {number} is not a JSON object"
                )
            records.append(
                {
                    key: _number_as_text(value) if key in protected_columns else value
                    for key, value in record.items()
                }
            )
    return pd.DataFrame.from_records(records)


def _number_as_text(value: Any) -> Any:
    """Return a JSON number as its decimal text and every other value unchanged.

    A JSON ``null`` stays missing, and a boolean, array, or object is left for the table
    validation to report.
    """
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    return value
