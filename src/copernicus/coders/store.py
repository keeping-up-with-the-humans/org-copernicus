"""The resumable SQLite store of one coding run.

A `RunStore` is created for a run identity: the method, model, temperature,
seed, and rubric, every setting of the coder config, the hashes of the prompt
templates and of the wording, the verifier, the marker shuffling, and the backend's
provenance (its kind, endpoint and the digest of the endpoint's path, quantization,
dtypes, context window, chat-template and request options, and capability flags).
Opening an existing file checks every one of these and fails loudly on a difference;
only the library version may differ, which is logged. Settings a backend does not
report, such as the vLLM version or the hardware, are not checked. What the backend
applied in the scoring calls is stored per row (``temperature_applied``,
``top_p_applied``), the temperature it applied in the accepted extraction attempt per
attribute (``<attribute_id>__stage1_temperature_applied``), and the text hash of the
text the row was coded on (``text_hash``), which binds every text id of the store to its
text. Results and the evidence that belongs to them are written in one transaction, and
``completed_cells`` tells the runner which texts and runs are stored, on which text, and
with which statuses. A cell with a failed attribute can be replaced on request. Shards are
folded together with `copernicus.coders.store_merge.merge_stores`.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from collections.abc import Mapping, Sequence
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType

from copernicus.coders.rotation import DIMENSIONS, WordingPool
from copernicus.coders.store_cells import StoredCell, delete_failed_cells, read_cells
from copernicus.coders.store_layout import check_layout, count_rows
from copernicus.coders.store_merge import merge_stores
from copernicus.coders.store_rows import row_values
from copernicus.coders.store_schema import (
    EVIDENCE_COLUMNS,
    create_statements,
    is_blank,
    open_read_only,
    open_read_write,
    quote,
    read_run_info,
    result_columns,
    validate_identifiers,
)
from copernicus.coders.store_types import (
    METHODS,
    RESERVED_KEYS,
    ResultRow,
    RunIdentity,
    identity_settings,
)
from copernicus.errors import StoreError
from copernicus.rubric.model import Rubric

logger = logging.getLogger(__name__)

__all__ = [
    "ResultRow",
    "RunIdentity",
    "RunStore",
    "StoredCell",
    "identity_settings",
    "merge_stores",
]

DUPLICATE_PREFIX = "UNIQUE constraint failed: results."
EXEMPT_KEY = "library_version"
SHUFFLE_SQL = (
    'INSERT OR REPLACE INTO "shuffles" ("shuffle_seed", "attribute_id", "marker_order_json") '
    "VALUES (?, ?, ?)"
)


class RunStore:
    """A SQLite file that holds the results of one run.

    Args:
        path: Location of the file, a ``str`` or a ``Path`` (kept as a ``Path`` in
            ``path``); created if it does not exist.
        rubric: The rubric the run codes.
        identity: Identity of the run.

    Raises:
        StoreError: If the identity does not fit the rubric, if an id of the
            rubric cannot be a column name, or if an existing file was made
            for a different identity or by an earlier version of the library or changed by hand (its
            tables lack a column this version writes, such as ``text_hash``, or hold another).
    """

    def __init__(self, path: str | Path, rubric: Rubric, identity: RunIdentity) -> None:
        self._rubric = rubric
        self._identity = identity
        self._markers = {
            a.attribute_id: [m.marker_id for m in a.markers] for a in rubric.attributes
        }
        self._result_names = [name for name, _ in result_columns(rubric)]
        expected = self._expected_run_info()
        self._evidence_names = [name for name, _ in EVIDENCE_COLUMNS]
        self._closed = False
        self.path = Path(path)
        if is_blank(self.path):
            self._connection = open_read_write(self.path)
            self._create(expected)
        else:
            with closing(open_read_only(self.path)) as peek:
                stored = read_run_info(peek)
                check_layout(peek, self.path)
                _check_run_info(stored, expected)
                check_layout(peek, self.path, self._result_names)
            self._connection = open_read_write(self.path)

    def _create(self, expected: dict[str, str]) -> None:
        """Create the tables and write ``run_info`` in one transaction.

        On failure nothing is left behind and the connection is closed.
        """
        try:
            self._connection.execute("BEGIN")
            with self._connection:
                for statement in create_statements(self._rubric):
                    self._connection.execute(statement)
                self._connection.executemany(
                    'INSERT INTO "run_info" ("key", "value") VALUES (?, ?)', expected.items()
                )
        except BaseException:
            self._connection.close()
            raise
        logger.info("created run store %s for %s", self.path, self._rubric.rubric_id)

    def _expected_run_info(self) -> dict[str, str]:
        """Return the ``run_info`` entries this store must hold."""
        identity = self._identity
        if identity.method not in METHODS:
            raise StoreError(f"method must be one of {METHODS}, got {identity.method!r}")
        if identity.rubric_hash != self._rubric.content_hash():
            raise StoreError(
                f"rubric_hash {identity.rubric_hash!r} does not match the rubric "
                f"{self._rubric.rubric_id!r} ({self._rubric.content_hash()!r})"
            )
        validate_identifiers(self._rubric)
        clash = sorted(set(identity.settings) & set(RESERVED_KEYS))
        if clash:
            raise StoreError(f"settings use reserved run_info keys: {clash}")
        info = {
            "method": identity.method,
            "model_name": identity.model_name,
            "temperature": repr(float(identity.temperature)),
            "seed": str(identity.seed),
            "rubric_hash": identity.rubric_hash,
            "rubric_id": self._rubric.rubric_id,
            "attribute_ids": json.dumps(list(self._markers)),
            "marker_ids": json.dumps([m for ids in self._markers.values() for m in ids]),
        }
        info.update(identity.settings)
        return info

    def completed_cells(self) -> dict[tuple[str, int], StoredCell]:
        """Return the stored cells of this identity's model and temperature.

        Returns:
            For each ``(text_id, run_number)`` the text hash of the text it was coded on
            and the status of each attribute result, in rubric order.

        Raises:
            StoreError: If a stored status is not one the library writes.
        """
        identity = self._identity
        return read_cells(
            self._connection, list(self._markers), identity.model_name, identity.temperature
        )

    def completed_keys(self) -> set[tuple[str, int]]:
        """Return ``(text_id, run_number)`` of every cell `completed_cells` returns."""
        return set(self.completed_cells())

    def insert(
        self,
        rows: Sequence[ResultRow],
        shuffles: Sequence[tuple[int, str, Sequence[str]]] = (),
        *,
        replace_failed: bool = False,
    ) -> None:
        """Write results, their evidence, and the marker shuffles in one transaction.

        Args:
            rows: The results; each must cover every attribute of the rubric.
            shuffles: ``(shuffle_seed, attribute_id, marker_order)`` of the marker
                orders the rows were coded with; rows already present are replaced.
            replace_failed: Whether a row takes the place of the stored cell of its text
                and run when that cell holds an attribute result in ``extraction_failed``
                or ``scoring_failed``: the stored results row and its evidence are deleted
                in the transaction that writes the row. A stored cell without such a
                result is never replaced.

        Raises:
            StoreError: If a row does not fit the rubric or carries a malformed text hash,
                the attribute results of a row disagree on the sampling the backend applied,
                a shuffle names an unknown attribute, a result with the same text, run,
                model, and temperature is already stored (with ``replace_failed``: a stored
                result without a failed attribute), or another constraint fails; in every
                case nothing is written.
        """
        created_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        names = (self._result_names, self._evidence_names)
        prepared = [
            (row, *row_values(row, self._identity, self._markers, names, created_at))
            for row in rows
        ]
        shuffle_values = [self._shuffle_values(*shuffle) for shuffle in shuffles]
        results_sql = _insert_sql("results", self._result_names)
        evidence_sql = _insert_sql("evidence", self._evidence_names)
        identity = self._identity
        with self._connection:
            if replace_failed:
                # The write lock comes first, so no other writer changes a checked cell.
                self._connection.execute("BEGIN IMMEDIATE")
                keys = [(row.text_id, row.run_number) for row in rows]
                delete_failed_cells(
                    self._connection,
                    keys,
                    list(self._markers),
                    identity.model_name,
                    identity.temperature,
                )
            self._connection.executemany(SHUFFLE_SQL, shuffle_values)
            for row, result_values, evidence_values in prepared:
                try:
                    self._connection.execute(results_sql, result_values)
                    self._connection.executemany(evidence_sql, evidence_values)
                except sqlite3.IntegrityError as exc:
                    key = (
                        f"text_id={row.text_id!r}, run_number={row.run_number}, "
                        f"model_name={self._identity.model_name!r}, "
                        f"temperature={self._identity.temperature!r}"
                    )
                    if str(exc).startswith(DUPLICATE_PREFIX):
                        raise StoreError(f"result already stored for {key}: {exc}") from exc
                    raise StoreError(f"constraint failed while storing {key}: {exc}") from exc

    def record_wording(self, pool: WordingPool) -> None:
        """Store the wording texts of the pool.

        Args:
            pool: The wording pool of the run; rows already present are replaced.
        """
        rows = [
            (dimension, wording.variant_id, column_name, text)
            for dimension in DIMENSIONS
            for wording in getattr(pool, dimension)
            for column_name, text in wording.texts.items()
        ]
        with self._connection:
            self._connection.executemany(
                'INSERT OR REPLACE INTO "wording" ("dimension", "variant_id", "column_name", '
                '"text") VALUES (?, ?, ?, ?)',
                rows,
            )

    def record_shuffle(self, shuffle_seed: int, attribute_id: str, order: Sequence[str]) -> None:
        """Store the marker order that a shuffle seed produced for an attribute.

        Args:
            shuffle_seed: Seed of the shuffle.
            attribute_id: Attribute whose markers were ordered.
            order: The marker names in prompt order.

        Raises:
            StoreError: If the attribute is not in the rubric.
        """
        values = self._shuffle_values(shuffle_seed, attribute_id, order)
        with self._connection:
            self._connection.execute(SHUFFLE_SQL, values)

    def _shuffle_values(
        self, shuffle_seed: int, attribute_id: str, order: Sequence[str]
    ) -> tuple[int, str, str]:
        """Check the attribute of a shuffle and return the values of its row."""
        if attribute_id not in self._markers:
            raise StoreError(f"unknown attribute {attribute_id!r}")
        return shuffle_seed, attribute_id, json.dumps(list(order))

    def counts(self) -> dict[str, int]:
        """Count the rows of each table.

        Returns:
            Row count for ``results``, ``evidence``, ``wording``, ``shuffles``,
            and ``run_info``.
        """
        return count_rows(self._connection)

    def close(self) -> None:
        """Commit and close the connection; a second call does nothing."""
        if self._closed:
            return
        self._closed = True
        self._connection.commit()
        self._connection.close()

    def __enter__(self) -> RunStore:
        """Return the store for use in a ``with`` block."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the store when the block ends."""
        self.close()


def _insert_sql(table: str, names: Sequence[str]) -> str:
    """Return the INSERT statement of one table with a placeholder per column."""
    columns = ", ".join(quote(name) for name in names)
    return f"INSERT INTO {quote(table)} ({columns}) VALUES ({', '.join('?' * len(names))})"


def _check_run_info(stored: Mapping[str, str], expected: Mapping[str, str]) -> None:
    """Raise unless the stored ``run_info`` and the requested one carry the same keys and values.

    The library version is exempt: a mismatch is logged, not raised.
    """
    for key, requested in expected.items():
        if key == EXEMPT_KEY:
            if key in stored and stored[key] != requested:
                logger.warning(
                    "store was created with %s %s, resumed with %s", key, stored[key], requested
                )
            continue
        if key not in stored:
            raise StoreError(f"run_info missing {key}")
        if stored[key] != requested:
            raise StoreError(
                f"run_info mismatch on {key}: stored {stored[key]}, requested {requested}"
            )
    for key in stored:
        if key not in expected and key != EXEMPT_KEY:
            raise StoreError(f"run_info unexpected {key}")
