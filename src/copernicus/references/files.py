"""The files of a reference set: a parquet or CSV table and a JSON description beside it."""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from copernicus._digests import TABLE_HASH_KEY, digest_problem, table_hash
from copernicus._io import read_sidecar, write_atomically, write_json
from copernicus._version import __version__
from copernicus.errors import ConfigurationError
from copernicus.references.binding import (
    BOUND_ON_REQUEST,
    RUBRIC_HASH_KEY,
    RUBRIC_KEY,
    hash_problem,
)
from copernicus.references.table import (
    FLOAT_COLUMNS,
    FRAME_COLUMNS,
    INTEGER_COLUMNS,
    SCREEN_COLUMNS,
    STRING_COLUMNS,
    canonical_frame,
    columns_problem,
    rows_frame,
)

logger = logging.getLogger(__name__)

# The columns, rows_frame and canonical_frame are defined in copernicus.references.table and
# exported here too, beside the functions that write and read the table.
__all__ = [
    "FRAME_COLUMNS",
    "SCREEN_COLUMNS",
    "USAGE_KEYS",
    "canonical_frame",
    "check_described_rows",
    "check_suffix",
    "description_path",
    "parse_usage",
    "parse_writers",
    "read_reference_files",
    "rows_frame",
    "write_reference_files",
]

USAGE_KEYS = ("calls", "attempts", "prompt_tokens", "completion_tokens")
WRITTEN_BY_KEY = "written_by"
ROWS_KEY = "n_rows"
# Keys that every description of this version holds; a description without one of them was
# written by an earlier version, and the file has to be written again, or, for a set file
# without the rubric hash or the table hash, bound on request.
EARLIER_VERSION_LACKS = (WRITTEN_BY_KEY, RUBRIC_HASH_KEY, TABLE_HASH_KEY)
SUFFIXES = (".parquet", ".csv")
SIDECAR_SUFFIX = ".json"
# What the reader of a set file whose description does not name its table can do.
SET_REMEDY = (
    "a save stopped between its two writes leaves such a pair: save the set again, or, for a "
    "table you can vouch for, read it with pandas and bind it with ReferenceSet.for_rubric"
)


def description_path(path: Path) -> Path:
    """Return the description file ``<path>.json`` that belongs to a table."""
    return path.with_name(path.name + SIDECAR_SUFFIX)


def check_suffix(path: Path) -> None:
    """Refuse a file name that is neither parquet nor CSV.

    Raises:
        ConfigurationError: If the suffix is another one.
    """
    if path.suffix not in SUFFIXES:
        raise ConfigurationError(f"unsupported suffix {path.suffix!r} in {path}; use {SUFFIXES}")


def write_reference_files(path: Path, frame: pd.DataFrame, description: Mapping[str, Any]) -> None:
    """Write the table to ``path``, then the description to ``<path>.json``, each atomically.

    The description is written last and names the content of the table it describes: it
    gains ``written_by``, the library version, ``n_rows``, the number of rows of the table,
    and ``table_hash``, the digest of the rows over every column of `FRAME_COLUMNS`, the
    screen columns included (`copernicus._digests.table_hash`). A stop between the two
    writes leaves a table whose digest the description does not name, never a description
    of rows that were not written. Every reader refuses such a pair (`read_reference_files`),
    except the generator's resume, which keeps the rows of the calls its description lists
    after checking their digest, and makes the other calls again.

    Args:
        path: A file name ending in ``.parquet`` or ``.csv``.
        frame: The table, with the columns `FRAME_COLUMNS`; its canonical form
            (`canonical_frame`) is written and hashed, so the file and the recorded digest
            agree whatever dtypes it holds.
        description: JSON values that describe the table.

    Raises:
        ConfigurationError: If the suffix is neither ``.parquet`` nor ``.csv``, or the frame
            is not a table of a reference set (`canonical_frame`); nothing is written then.
    """
    check_suffix(path)
    table = canonical_frame(frame)
    digest = table_hash(table, FRAME_COLUMNS)
    if path.suffix == ".parquet":
        write_atomically(path, lambda handle: table.to_parquet(handle, index=False))
    else:
        write_atomically(path, lambda handle: table.to_csv(handle, index=False))
    write_json(
        description_path(path),
        {**description, WRITTEN_BY_KEY: __version__, ROWS_KEY: len(table), TABLE_HASH_KEY: digest},
    )


def read_reference_files(
    path: Path,
    required: Sequence[str],
    *,
    rows_may_exceed: bool = False,
    allow_missing_hash: bool = False,
    remedy: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read a table written by `write_reference_files` and its description.

    The table is checked against the description: the description must name the version
    that wrote it, the content hash of the rubric the texts were written for, and the digest
    of the table; the rows of the table must belong to its rubric; and the table must be the
    one the description was written for (`check_described_rows`: as many rows as it records,
    with the digest it records), so that a table of one save beside the description of
    another is refused, also when the two hold as many rows.

    Args:
        path: The table.
        required: Keys the description must hold.
        rows_may_exceed: The table may hold rows that the description does not describe (a
            resume file whose run stopped between its two writes). This function then does
            not run `check_described_rows`, since only the caller knows which rows the
            description describes; the caller runs it on the rows it keeps.
        allow_missing_hash: Accept a description without ``rubric_hash`` or without
            ``table_hash``, which a set file of an earlier version lacks, for a caller that
            binds the set itself (`copernicus.references.binding.resolve_binding`); a hash
            the description holds is checked either way, and a table whose description
            records no digest is checked on its row count alone.
        remedy: What the refusal of a file of an earlier version tells the reader to do;
            ``None`` tells them to regenerate the file.

    Returns:
        The table, with the dtypes of `rows_frame` and boolean ``screened_out``, and
        the description.

    Raises:
        ConfigurationError: If the suffix is unknown, a file is missing, the description is
            not a JSON object, lacks a required key, lacks ``written_by`` or, unless
            ``allow_missing_hash``, ``rubric_hash`` or ``table_hash`` (it was written by an
            earlier version), has a ``rubric_hash`` that is not the content hash of a
            rubric, a ``table_hash`` that is not a digest or a malformed ``n_rows``, or
            disagrees with the table in the rubric or, unless ``rows_may_exceed``, in the row
            count or the digest.
    """
    check_suffix(path)
    sidecar = description_path(path)
    for needed in (path, sidecar):
        if not needed.is_file():
            raise ConfigurationError(f"reference set {path}: {needed.name} does not exist")
    description = read_sidecar(sidecar)
    absent = [
        key for key in required if key not in description and key not in EARLIER_VERSION_LACKS
    ]
    if absent:
        raise ConfigurationError(f"{sidecar} lacks {absent}")
    for key in EARLIER_VERSION_LACKS:
        if key not in description and not (allow_missing_hash and key in BOUND_ON_REQUEST):
            raise ConfigurationError(
                f"{sidecar} has no {key}: it was written by an earlier version of "
                f"copernicus than this one ({__version__}); "
                + (f"regenerate {path.name}" if remedy is None else remedy)
            )
    if RUBRIC_HASH_KEY in description:
        problem = hash_problem(description[RUBRIC_HASH_KEY])
        if problem is not None:
            raise ConfigurationError(f"{sidecar}: {problem}")
    if TABLE_HASH_KEY in description:
        problem = digest_problem(TABLE_HASH_KEY, description[TABLE_HASH_KEY])
        if problem is not None:
            raise ConfigurationError(f"{sidecar}: {problem}")
    if ROWS_KEY not in description or not _is_count(description[ROWS_KEY]):
        raise ConfigurationError(f"{sidecar}: {ROWS_KEY} must be a non-negative integer")
    frame = _read_table(path)
    _check_rubric_of_rows(frame, description, path, sidecar)
    if not rows_may_exceed:
        check_described_rows(
            frame, description, path, sidecar, allow_missing_hash=allow_missing_hash
        )
    return frame, description


def check_described_rows(
    rows: pd.DataFrame,
    description: Mapping[str, Any],
    path: Path,
    sidecar: Path,
    *,
    which: str = "rows",
    remedy: str = SET_REMEDY,
    allow_missing_hash: bool = False,
) -> None:
    """Refuse rows that are not the rows a description was written for.

    The rows must be as many as the description records (``n_rows``) and, in their order,
    have the digest it records (``table_hash``, over the columns `FRAME_COLUMNS`; see
    `copernicus._digests.table_hash`).

    Args:
        rows: The rows the description must describe: the table of a set file, or the rows
            of the calls that the description of a resume file lists.
        description: The description; its ``n_rows`` and a ``table_hash`` it holds have been
            checked to be well formed.
        path: The table, named in the message.
        sidecar: The description file, named in the message.
        which: What the rows are, for the message.
        remedy: How such a pair comes about and what the reader can do, for the message.
        allow_missing_hash: Check rows whose description records no digest, as a set file
            of an earlier version lacks it, on their count alone, for a caller that then
            binds the set itself (`copernicus.references.binding.resolve_binding`);
            otherwise such a description is refused.

    Raises:
        ConfigurationError: Naming both files, both row counts and both digests, if the
            count or the digest differs; or if the description records no digest and
            ``allow_missing_hash`` is false.
    """
    if TABLE_HASH_KEY not in description and not allow_missing_hash:
        raise ConfigurationError(
            f"{sidecar} has no {TABLE_HASH_KEY}, so the rows of {path} cannot be checked against it"
        )
    actual = table_hash(rows, FRAME_COLUMNS)
    recorded = description.get(TABLE_HASH_KEY)
    if len(rows) == description[ROWS_KEY] and (recorded is None or recorded == actual):
        return
    described = "and no hash" if recorded is None else f"with hash {recorded}"
    raise ConfigurationError(
        f"{path} and {sidecar.name} do not belong together: the table holds {len(rows)} "
        f"{which} with hash {actual}, the description records {description[ROWS_KEY]} rows "
        f"{described}; {remedy}"
    )


def _check_rubric_of_rows(
    frame: pd.DataFrame, description: Mapping[str, Any], path: Path, sidecar: Path
) -> None:
    """Refuse a table whose rows name another rubric than its description."""
    rubric = str(description[RUBRIC_KEY])
    others = sorted(set(frame[RUBRIC_KEY].astype(str)) - {rubric})
    if others:
        raise ConfigurationError(
            f"{path} and {sidecar.name} do not belong together: the description is for "
            f"rubric {rubric}, but rows name the rubric(s) {others}"
        )


def _read_table(path: Path) -> pd.DataFrame:
    """Read the table of a reference set file with the dtypes of `rows_frame`.

    A table that lacks a column of `FRAME_COLUMNS`, repeats one or holds another is refused
    before any column is used. The float columns of a CSV file are read as text and parsed
    with Python's ``float``, which returns exactly the float whose shortest text ``to_csv``
    wrote, on every version of pandas, so the digest of the table read equals the one
    recorded when it was saved. The CSV parser of pandas reads such a text exactly from
    pandas 3.1 on, which deprecates its ``float_precision`` switch; earlier versions need
    ``float_precision="round_trip"``.

    Raises:
        ConfigurationError: Naming the file, if the columns are not those of a reference set
            (naming them), a CSV file is empty, cannot be parsed or is not UTF-8 text, a cell
            of a CSV file is outside the range of its integer column, or a cell cannot be read
            in the dtype of its column (naming the column where it is known, and the reason of
            the parser).
    """
    if path.suffix == ".parquet":
        frame = pd.read_parquet(path)
    else:
        dtypes: dict[str, Any] = {name: str for name in (*STRING_COLUMNS, *FLOAT_COLUMNS)}
        dtypes.update({name: "int64" for name in INTEGER_COLUMNS})
        dtypes[SCREEN_COLUMNS[0]] = "bool"
        try:
            frame = pd.read_csv(path, dtype=dtypes, keep_default_na=False)
        except pd.errors.EmptyDataError as error:
            raise ConfigurationError(f"{path}: the file is empty") from error
        except pd.errors.ParserError as error:
            raise ConfigurationError(
                f"{path}: the file cannot be parsed as CSV: {error}"
            ) from error
        except UnicodeDecodeError as error:
            raise ConfigurationError(f"{path}: the file is not UTF-8 text") from error
        except OverflowError as error:
            raise ConfigurationError(
                f"{path}: a cell is outside the range of its integer column: {error}"
            ) from error
        except ValueError as error:
            raise ConfigurationError(
                f"{path}: a cell cannot be read in the dtype of its column (an integer in "
                f"{', '.join(INTEGER_COLUMNS[:-1])} or {INTEGER_COLUMNS[-1]}; True or False in "
                f"{SCREEN_COLUMNS[0]}): {error}"
            ) from error
    problem = columns_problem(frame.columns)
    if problem is not None:
        raise ConfigurationError(f"{path}: {problem}")
    if path.suffix == ".parquet":
        return frame
    for name in FLOAT_COLUMNS:
        try:
            values = [math.nan if text == "" else float(text) for text in frame[name]]
        except ValueError as error:
            raise ConfigurationError(
                f"{path}: column {name!r} holds a cell that is not a number: {error}"
            ) from error
        frame[name] = np.asarray(values, dtype=np.float64)
    return frame


def parse_writers(value: Any, source: Path) -> dict[str, dict[str, str]]:
    """Check the writer provenance of a description.

    Args:
        value: The ``writers`` entry of the description.
        source: The description file, named in the error.

    Returns:
        The provenance of each writer model, by model name.

    Raises:
        ConfigurationError: If the entry is not an object that maps each writer model to an
            object of text values.
    """
    well_formed = isinstance(value, dict) and all(
        isinstance(entry, dict) and all(isinstance(text, str) for text in entry.values())
        for entry in value.values()
    )
    if not well_formed:
        raise ConfigurationError(
            f"{source}: writers must map each writer model to its provenance, an object of "
            "text values"
        )
    return {str(name): dict(entry) for name, entry in value.items()}


def _is_count(value: Any) -> bool:
    """Tell whether a value is a non-negative integer, booleans excluded."""
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def parse_usage(value: Any, source: Path) -> dict[str, dict[str, int]]:
    """Check the usage of a description: calls, attempts, and tokens per writer model.

    Args:
        value: The ``usage`` entry of the description.
        source: The description file, named in the error.

    Returns:
        The usage of each writer model, by model name, with the keys ``USAGE_KEYS``.

    Raises:
        ConfigurationError: If the entry is not an object that maps each writer model to an
            object of exactly the keys ``USAGE_KEYS`` with non-negative integer values.
    """
    well_formed = isinstance(value, dict) and all(
        isinstance(entry, dict)
        and set(entry) == set(USAGE_KEYS)
        and all(_is_count(count) for count in entry.values())
        for entry in value.values()
    )
    if not well_formed:
        raise ConfigurationError(
            f"{source}: usage must map each writer model to {list(USAGE_KEYS)}, each a "
            "non-negative integer"
        )
    return {
        str(name): {key: int(entry[key]) for key in USAGE_KEYS} for name, entry in value.items()
    }
