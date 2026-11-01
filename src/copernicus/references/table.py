"""The table of a reference set: its columns, their dtypes, and its canonical form.

Every reference set holds its table in one canonical form, whatever its caller passed, and its
files write and hash that same form (`copernicus.references.files`), so a table saved to CSV or
to parquet reads back with the values it held and the digest its description records.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import (
    infer_dtype,
    is_bool_dtype,
    is_float_dtype,
    is_integer_dtype,
    is_object_dtype,
    is_unsigned_integer_dtype,
)

from copernicus.errors import ConfigurationError

__all__ = [
    "FLOAT_COLUMNS",
    "FRAME_COLUMNS",
    "INTEGER_COLUMNS",
    "SCREEN_COLUMNS",
    "STRING_COLUMNS",
    "canonical_frame",
    "columns_problem",
    "rows_frame",
]

FRAME_COLUMNS = (
    "rubric_id",
    "target",
    "polarity",
    "level",
    "style",
    "writer_model",
    "text",
    "call_index",
    "attempt",
    "seed",
    "prompt_tokens",
    "completion_tokens",
    "temperature_applied",
    "top_p_applied",
    "prompt_hash",
    "prompt_version",
    "screened_out",
    "screen_reason",
)
SCREEN_COLUMNS = ("screened_out", "screen_reason")
INTEGER_COLUMNS = ("call_index", "attempt", "seed")
FLOAT_COLUMNS = ("prompt_tokens", "completion_tokens", "temperature_applied", "top_p_applied")
STRING_COLUMNS = tuple(
    name
    for name in FRAME_COLUMNS
    if name not in (*INTEGER_COLUMNS, *FLOAT_COLUMNS, SCREEN_COLUMNS[0])
)
# What a column holds instead of a missing value, named when it has one.
MISSING_REMEDIES = {
    SCREEN_COLUMNS[0]: "write True or False",
    SCREEN_COLUMNS[1]: "write an empty string '' for no screen reason",
    "text": "a text is never missing: leave out a row that has none",
}
MISSING_REMEDY = "every row needs a value in this column"
# The kinds of values, as pandas infers them, that a float column takes from an object column.
NUMBER_KINDS = ("empty", "integer", "floating", "mixed-integer-float")
INT64_MAX = int(np.iinfo(np.int64).max)


def rows_frame(rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Build the table of generated rows, without the screen columns, in their dtypes.

    Args:
        rows: One mapping per text with every column of ``FRAME_COLUMNS`` but the two
            screen columns; ``None`` stands for a value the backend did not report.

    Returns:
        The table: integer columns as int64, token counts and applied sampling as float64
        with NaN for ``None``.
    """
    frame = pd.DataFrame(list(rows), columns=list(FRAME_COLUMNS[: -len(SCREEN_COLUMNS)]))
    for name in INTEGER_COLUMNS:
        frame[name] = frame[name].astype("int64")
    for name in FLOAT_COLUMNS:
        frame[name] = frame[name].astype("float64")
    return frame


def columns_problem(columns: Sequence[Any]) -> str | None:
    """Say how the columns of a table differ from those of a reference set.

    Args:
        columns: The column names of the table, in its order.

    Returns:
        ``None`` when the table holds every column of `FRAME_COLUMNS` once and no other, in
        any order; otherwise the columns it lacks, the unexpected ones and the repeated ones.
    """
    names = list(columns)
    found = (
        ("lacks", [name for name in FRAME_COLUMNS if name not in names]),
        ("holds the unexpected columns", [name for name in names if name not in FRAME_COLUMNS]),
        ("repeats the columns", sorted({str(name) for name in names if names.count(name) > 1})),
    )
    parts = [f"{label} {listed}" for label, listed in found if listed]
    if not parts:
        return None
    return f"a reference set needs the columns {list(FRAME_COLUMNS)}; the table " + ", ".join(parts)


def canonical_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the table of a reference set in its canonical columns and dtypes.

    The canonical table has the columns `FRAME_COLUMNS`, in that order, and a fresh range
    index, as a table is read back from its file. The integer columns are int64, taken from
    any integer dtype. The float columns are float64 with NaN for a missing value, taken from
    any float or integer dtype, or from an object column of numbers (Python's or numpy's
    integers and floats) and missing values. The string columns hold strings in the plain
    string dtype that reading the file gives. ``screened_out`` is bool, taken from a bool
    dtype or an object column of booleans. Columns in another order are reordered. Every
    reference set holds this form, and `copernicus.references.files.write_reference_files`
    writes and hashes it, so the table on disk and its recorded digest agree whatever a
    caller passed; on a canonical table this changes nothing but the copy.

    Args:
        frame: One row per text with the columns `FRAME_COLUMNS`, in any order.

    Returns:
        A new frame in the canonical form.

    Raises:
        ConfigurationError: If a column of `FRAME_COLUMNS` is missing or repeated or another
            column is present (naming them); or naming the column, if an integer, string or
            flag column has a missing value (with what to write instead), or a column holds a
            value of another type: a float or a boolean in an integer column, a string or a
            boolean in a float column, a non-string in a string column, or anything but True
            and False in ``screened_out``.
    """
    problem = columns_problem(frame.columns)
    if problem is not None:
        raise ConfigurationError(problem)
    source = frame.reset_index(drop=True)
    return pd.DataFrame({name: _canonical_column(source[name], name) for name in FRAME_COLUMNS})


def _canonical_column(column: pd.Series, name: str) -> Any:
    """Return one column in its canonical dtype, refusing values it cannot take."""
    if name in INTEGER_COLUMNS:
        return _integers(column, name)
    if name in FLOAT_COLUMNS:
        return _floats(column, name)
    if name == SCREEN_COLUMNS[0]:
        return _flags(column, name)
    return _strings(column, name)


def _check_complete(column: pd.Series, name: str) -> None:
    """Refuse a column with a missing value, naming what to write instead."""
    missing = int(column.isna().sum())
    if missing:
        raise ConfigurationError(
            f"column {name!r} of a reference set has a missing value in {missing} of "
            f"{len(column)} rows; {MISSING_REMEDIES.get(name, MISSING_REMEDY)}"
        )


def _integers(column: pd.Series, name: str) -> np.ndarray:
    """Return an integer column as int64, refusing other dtypes and missing values."""
    if not is_integer_dtype(column.dtype):
        raise ConfigurationError(
            f"column {name!r} of a reference set must hold integers, in an integer dtype; "
            f"got dtype {column.dtype}"
        )
    _check_complete(column, name)
    if is_unsigned_integer_dtype(column.dtype) and len(column) and column.max() > INT64_MAX:
        raise ConfigurationError(f"column {name!r} of a reference set exceeds the int64 range")
    return column.to_numpy(dtype=np.int64)


def _floats(column: pd.Series, name: str) -> np.ndarray:
    """Return a float column as float64 with NaN for a missing value, refusing non-numbers."""
    dtype = column.dtype
    numeric = is_float_dtype(dtype) or is_integer_dtype(dtype)
    kind = infer_dtype(column, skipna=True)
    if not numeric and not (is_object_dtype(dtype) and kind in NUMBER_KINDS):
        raise ConfigurationError(
            f"column {name!r} of a reference set must hold numbers or missing values; got "
            f"dtype {dtype} with {kind} values"
        )
    return column.to_numpy(dtype=np.float64, na_value=np.nan)


def _strings(column: pd.Series, name: str) -> pd.Series:
    """Return a string column in the plain string dtype, refusing missing and other values."""
    _check_complete(column, name)
    kind = infer_dtype(column.astype(object), skipna=False)
    if kind not in ("string", "empty"):
        raise ConfigurationError(
            f"column {name!r} of a reference set must hold strings; got {kind} values"
        )
    return column.astype(str)


def _flags(column: pd.Series, name: str) -> np.ndarray:
    """Return the screen flag as bool, refusing a missing value and any but True and False."""
    _check_complete(column, name)
    dtype = column.dtype
    if not is_bool_dtype(dtype) and not (
        is_object_dtype(dtype) and infer_dtype(column, skipna=False) in ("boolean", "empty")
    ):
        raise ConfigurationError(
            f"column {name!r} of a reference set must hold booleans, True or False; got dtype "
            f"{dtype} with {infer_dtype(column, skipna=False)} values"
        )
    return column.to_numpy(dtype=bool)
