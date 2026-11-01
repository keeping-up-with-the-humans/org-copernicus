"""Tidy tables read from a run store.

Every function opens the store file read-only, so a store that a run is still
writing to can be exported, and takes the store's path as a ``str`` or a ``Path``.
Rows are ordered by ``text_id`` (as text, so ``"10"`` comes before ``"2"``) and
``run_number``; the long score table lists the attributes of a text and run in
rubric order.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from contextlib import closing
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from copernicus.coders.store_layout import check_layout
from copernicus.coders.store_schema import (
    ATTRIBUTE_FIELDS,
    EVIDENCE_COLUMNS,
    attribute_column,
    open_read_only,
    quote,
    read_run_info,
)

logger = logging.getLogger(__name__)

__all__ = [
    "construct_scores",
    "export_construct_scores",
    "export_evidence",
    "export_provenance",
    "export_scores",
    "export_text_scores",
]

SCORE_COLUMNS = (
    ("text_id", "str"),
    ("run_number", "int"),
    ("attribute_id", "str"),
    ("score", "Int64"),
    ("status", "str"),
    ("explanation", "str"),
    ("stage1_attempt", "Int64"),
    ("stage2_attempt", "Int64"),
    ("stage1_temperature_applied", "float"),
    ("role_id", "str"),
    ("task_id", "str"),
    ("level_id", "str"),
    ("model_name", "str"),
    ("temperature", "float"),
    ("seed", "int"),
    ("shuffle_seed", "int"),
    ("text_key", "int"),
    ("text_hash", "str"),
    ("library_version", "str"),
    ("temperature_applied", "float"),
    ("top_p_applied", "float"),
)
PROVENANCE_COLUMNS = (
    "role_id",
    "task_id",
    "level_id",
    "model_name",
    "temperature",
    "seed",
    "shuffle_seed",
    "text_key",
    "text_hash",
    "library_version",
    "temperature_applied",
    "top_p_applied",
)
KIND_OF_DECLARATION = {
    "INTEGER NOT NULL": "int",
    "INTEGER": "Int64",
    "REAL NOT NULL": "float",
    "TEXT NOT NULL": "str",
}


def _column(values: Sequence[Any], kind: str) -> Any:
    """Convert a list of cell values into a typed column."""
    if kind == "int":
        return np.array(values, dtype="int64")
    if kind == "float":
        return np.array(values, dtype="float64")
    if kind == "Int64":
        return pd.array(values, dtype="Int64")
    return pd.Series(values, dtype="object")


def _frame(columns: Sequence[tuple[str, str]], rows: Sequence[Sequence[Any]]) -> pd.DataFrame:
    """Build a DataFrame from rows, with one typed column per ``(name, kind)``."""
    data = {
        name: _column([row[index] for row in rows], kind)
        for index, (name, kind) in enumerate(columns)
    }
    return pd.DataFrame(data, columns=[name for name, _ in columns])


def _attribute_ids(info: dict[str, str]) -> list[str]:
    """Return the attribute ids in rubric order from the ``run_info`` of a store."""
    ids: list[str] = json.loads(info["attribute_ids"])
    return ids


def export_scores(store_path: str | Path) -> pd.DataFrame:
    """Return the long score table: one row per text, run, and attribute.

    Args:
        store_path: Path of the run store, a ``str`` or a ``Path``.

    Returns:
        Columns ``text_id``, ``run_number``, ``attribute_id``, ``score``
        (nullable ``Int64``; missing when the attribute got no score),
        ``status``, ``explanation``, ``stage1_attempt``, ``stage2_attempt``
        (nullable ``Int64``), ``stage1_temperature_applied`` (what the backend applied
        in the accepted extraction attempt of the attribute; NaN when no attempt was
        accepted, the one-stage coder made none, or the backend applied none),
        ``role_id``, ``task_id``, ``level_id``,
        ``model_name``, ``temperature``, ``seed``, ``shuffle_seed`` (seed of the
        wording draw and the marker order), ``text_key`` (integer key of the text
        used for seeding), ``text_hash`` (the text hash of the text the row was coded
        on, see `copernicus.coders.text_hash.hash_text`), ``library_version``, and
        ``temperature_applied`` and ``top_p_applied`` (what the backend applied in the
        scoring calls of the text and run; NaN when it applied none or no scoring call
        was made).

    Raises:
        StoreError: If the file does not exist, is not a run store, or was made by an earlier
            version of the library or changed by hand.
    """
    with closing(open_read_only(Path(store_path))) as connection:
        check_layout(connection, Path(store_path))
        attribute_ids = _attribute_ids(read_run_info(connection))
        attribute_columns = [
            quote(attribute_column(attribute_id, field))
            for attribute_id in attribute_ids
            for field, _ in ATTRIBUTE_FIELDS
        ]
        provenance = ", ".join(quote(name) for name in PROVENANCE_COLUMNS)
        # One statement reads a consistent snapshot even while a run commits batches.
        results = connection.execute(
            f'SELECT "text_id", "run_number", {provenance}, {", ".join(attribute_columns)} '
            'FROM "results"'
        ).fetchall()
    width = len(ATTRIBUTE_FIELDS)
    offset = 2 + len(PROVENANCE_COLUMNS)
    rows: list[tuple[Any, ...]] = []
    for result in results:
        for index, attribute_id in enumerate(attribute_ids):
            start = offset + index * width
            rows.append(
                (
                    result[0],
                    result[1],
                    attribute_id,
                    *result[start : start + width],
                    *result[2:offset],
                )
            )
    rank = {attribute_id: index for index, attribute_id in enumerate(attribute_ids)}
    rows.sort(key=lambda r: (r[0], r[1], rank[r[2]]))
    return _frame(SCORE_COLUMNS, rows)


def _numeric_scores(scores: pd.DataFrame) -> pd.DataFrame:
    """Return the long table reduced to its keys and a float score (NaN when missing)."""
    return pd.DataFrame(
        {
            "text_id": scores["text_id"],
            "run_number": scores["run_number"],
            "attribute_id": scores["attribute_id"],
            "value": scores["score"].to_numpy(dtype="float64", na_value=np.nan),
        }
    )


def construct_scores(numeric: pd.DataFrame) -> pd.DataFrame:
    """Return the construct score per text and run from a numeric long table.

    The construct score is the mean of the attribute scores that are not missing, which is
    the single definition used by the exports and by the validity panel.

    Args:
        numeric: One row per text, run and attribute with columns ``text_id``,
            ``run_number`` and ``value`` (float, NaN when the attribute got no score).

    Returns:
        Columns ``text_id``, ``run_number``, ``construct_score`` (NaN when no attribute
        score exists) and ``n_attributes_scored`` (how many scores entered the mean),
        sorted by text and run.
    """
    grouped = numeric.groupby(["text_id", "run_number"], sort=True)["value"]
    table = pd.DataFrame(
        {
            "construct_score": grouped.mean().astype("float64"),
            "n_attributes_scored": grouped.count().astype("int64"),
        }
    )
    return table.reset_index()


def export_construct_scores(store_path: str | Path) -> pd.DataFrame:
    """Return the construct score per text and run.

    Args:
        store_path: Path of the run store, a ``str`` or a ``Path``.

    Returns:
        Columns ``text_id``, ``run_number``, ``construct_score`` (mean of the
        attribute scores that are not missing; NaN when none is), and
        ``n_attributes_scored`` (how many attributes entered the mean).

    Raises:
        StoreError: If the file does not exist, is not a run store, or was made by an earlier
            version of the library or changed by hand.
    """
    return construct_scores(_numeric_scores(export_scores(store_path)))


def export_text_scores(store_path: str | Path) -> pd.DataFrame:
    """Return the scores per text, averaged over runs.

    Args:
        store_path: Path of the run store, a ``str`` or a ``Path``.

    Returns:
        Columns ``text_id``; ``construct_score`` (mean over runs of the per-run
        construct score); one column per attribute id (mean over runs of its
        numeric scores); and ``n_runs`` (runs stored for the text). Missing
        scores are left out of every mean.

    Raises:
        StoreError: If the file does not exist, is not a run store, or was made by an earlier
            version of the library or changed by hand.
    """
    path = Path(store_path)
    numeric = _numeric_scores(export_scores(path))
    with closing(open_read_only(path)) as connection:
        attribute_ids = _attribute_ids(read_run_info(connection))
    per_text = construct_scores(numeric).groupby("text_id", sort=True)
    table = pd.DataFrame(
        {
            "construct_score": per_text["construct_score"].mean().astype("float64"),
            "n_runs": per_text["run_number"].count().astype("int64"),
        }
    )
    means = numeric.groupby(["text_id", "attribute_id"], sort=True)["value"].mean().unstack()
    means = means.reindex(columns=attribute_ids).astype("float64")
    result = pd.concat([table[["construct_score"]], means, table[["n_runs"]]], axis=1)
    return result.reset_index()


def export_evidence(store_path: str | Path) -> pd.DataFrame:
    """Return every passage the extraction stage returned.

    Args:
        store_path: Path of the run store, a ``str`` or a ``Path``.

    Returns:
        One row per passage with ``text_id``, ``run_number``, ``model_name``,
        ``temperature``, and the fields of
        `copernicus.coders.evidence.EvidenceRow`, ordered by text, run,
        attribute and marker (in rubric order), and passage index.

    Raises:
        StoreError: If the file does not exist, is not a run store, or was made by an earlier
            version of the library or changed by hand.
    """
    kinds = [(name, KIND_OF_DECLARATION[declaration]) for name, declaration in EVIDENCE_COLUMNS]
    columns = ", ".join(quote(name) for name, _ in kinds)
    names = [name for name, _ in kinds]
    with closing(open_read_only(Path(store_path))) as connection:
        info = read_run_info(connection)
        check_layout(connection, Path(store_path))
        rows = connection.execute(f'SELECT {columns} FROM "evidence"').fetchall()
    attribute_rank = {a: i for i, a in enumerate(json.loads(info["attribute_ids"]))}
    marker_rank = {m: i for i, m in enumerate(json.loads(info["marker_ids"]))}
    position = {name: names.index(name) for name in names}
    rows.sort(
        key=lambda r: (
            r[position["text_id"]],
            r[position["run_number"]],
            attribute_rank[r[position["attribute_id"]]],
            marker_rank[r[position["marker_id"]]],
            r[position["quote_index"]],
        )
    )
    return _frame(kinds, rows)


def export_provenance(store_path: str | Path) -> dict[str, str]:
    """Return the ``run_info`` of a store.

    Only the ``run_info`` table is read, so the provenance of a store that an earlier
    version of the library made can be read too, while the other exports refuse it.

    Args:
        store_path: Path of the run store, a ``str`` or a ``Path``.

    Returns:
        Key to value: method, model, temperature, seed, rubric id and hash,
        attribute and marker ids, every setting, and the library version.

    Raises:
        StoreError: If the file does not exist or is not a run store.
    """
    with closing(open_read_only(Path(store_path))) as connection:
        return read_run_info(connection)
