"""The splits file of the NLP-ML baseline: ``BaselineSplits`` written to JSON and read back.

The file follows the convention of the draw file of the human sample
(``copernicus.corrections.draw_files``): one JSON object that holds the splits block under
``splits``, beside ``analysis`` and the library version. ``write_splits`` writes it atomically
and ``read_splits`` reads it back, refusing a file whose block lacks a field or holds another.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from copernicus._io import read_sidecar, write_json
from copernicus._version import __version__
from copernicus.baseline.splits import BaselineSplits, checked_splits, listed
from copernicus.errors import ConfigurationError

logger = logging.getLogger(__name__)

__all__ = ["SPLITS_KEY", "read_splits", "write_splits"]

SPLITS_KEY = "splits"
"""The key of the splits block in a splits file."""
ANALYSIS = "NLP-ML baseline splits"
BLOCK_FIELDS = ("folds", "draws")
WRITE_HINT = "write them with write_splits or --write-splits"


def _columns_of(frame: pd.DataFrame) -> dict[str, list[Any]]:
    """A table as JSON values: each column a list."""
    return {str(column): frame[column].tolist() for column in frame.columns}


def write_splits(splits: BaselineSplits, path: str | Path) -> None:
    """Write splits to a JSON file that ``read_splits`` reads back.

    The file holds the splits block under ``splits``: ``folds`` and ``draws`` as objects of
    columns, each a list of values, and ``draws`` as ``null`` when the splits hold none; the
    library version stands beside it. The tables are checked again first (``checked_splits``),
    so that no table changed after construction is written.

    Args:
        splits: The splits, such as ``BaselineResult.splits`` of a run.
        path: The file to write, a ``str`` or a ``Path``; it is replaced atomically.

    Raises:
        ConfigurationError: If ``splits`` is not a ``BaselineSplits`` or its tables are now
            refused; nothing is written.
    """
    splits = checked_splits(splits)
    target = Path(path)
    block = {
        "folds": _columns_of(splits.folds),
        "draws": None if splits.draws is None else _columns_of(splits.draws),
    }
    write_json(target, {"analysis": ANALYSIS, SPLITS_KEY: block, "library_version": __version__})
    draw_rows = "no" if splits.draws is None else str(len(splits.draws))
    logger.info(
        "wrote splits of %d fold rows and %s draw rows to %s", len(splits.folds), draw_rows, target
    )


def _table(value: object, name: str, source: Path) -> pd.DataFrame:
    """A table of the splits block: an object of columns, each a list of the same length."""
    if not isinstance(value, Mapping):
        raise ConfigurationError(
            f"{source}: the {name} must be an object of columns, each a list of values; "
            f"got {type(value).__name__}"
        )
    for column, values in value.items():
        if not isinstance(values, list):
            raise ConfigurationError(
                f"{source}: the column {column} of the {name} must be a list; got {values!r}"
            )
    lengths = {str(column): len(values) for column, values in value.items()}
    if len(set(lengths.values())) > 1:
        shown = ", ".join(f"{column} {length}" for column, length in lengths.items())
        raise ConfigurationError(f"{source}: the columns of the {name} differ in length: {shown}")
    return pd.DataFrame({str(column): values for column, values in value.items()})


def read_splits(path: str | Path) -> BaselineSplits:
    """Read the splits of a splits file, as ``write_splits`` and ``--write-splits`` write it.

    A table without rows comes back with the types of its columns, and an empty draws table
    stays a table, distinct from ``null``.

    Args:
        path: The splits file, a ``str`` or a ``Path``.

    Returns:
        The splits, equal in both tables to the splits that were written.

    Raises:
        ConfigurationError: Naming the file, if it is not a JSON object with a splits block,
            the block lacks ``folds`` or ``draws`` or holds another field, a table is not an
            object of columns of equal length, or the tables are refused by
            ``BaselineSplits``.
    """
    source = Path(path)
    content = read_sidecar(source)
    block = content.get(SPLITS_KEY)
    if not isinstance(block, Mapping):
        raise ConfigurationError(f"{source} holds no splits: {WRITE_HINT}")
    missing = [name for name in BLOCK_FIELDS if name not in block]
    if missing:
        raise ConfigurationError(f"{source}: the splits lack {listed(missing)}; {WRITE_HINT}")
    unknown = sorted(str(name) for name in block if name not in BLOCK_FIELDS)
    if unknown:
        raise ConfigurationError(
            f"{source}: the splits hold fields that no version writes: {listed(unknown)}"
        )
    folds = _table(block["folds"], "folds", source)
    draws = None if block["draws"] is None else _table(block["draws"], "draws", source)
    try:
        splits = BaselineSplits(folds=folds, draws=draws)
    except ConfigurationError as error:
        # The checks of the tables do not know the file; a refusal of a file names it first.
        raise ConfigurationError(f"{source}: {error}") from error
    logger.info("read splits of %d fold rows from %s", len(splits.folds), source)
    return splits
