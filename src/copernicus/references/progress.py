"""The progress of a generation run: its finished calls, kept in a resume file on request."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from copernicus.errors import ConfigurationError, SettingsError
from copernicus.references.answers import CallOutcome
from copernicus.references.binding import BOUND_GENERATED
from copernicus.references.cells import Cell
from copernicus.references.files import (
    SCREEN_COLUMNS,
    USAGE_KEYS,
    check_described_rows,
    check_suffix,
    description_path,
    parse_usage,
    parse_writers,
    read_reference_files,
    rows_frame,
    write_reference_files,
)
from copernicus.references.reference_set import ReferenceSet
from copernicus.references.settings import GeneratorSettings
from copernicus.rubric.model import Rubric

logger = logging.getLogger(__name__)

__all__ = ["GenerationProgress"]

RESUME_KEYS = (
    "rubric_id",
    "settings",
    "shortfalls",
    "writers",
    "usage",
    "rubric_hash",
    "completed",
)
# Settings that change how fast a run goes but not what it writes; a run may resume with others.
EXECUTION_SETTINGS = ("concurrency",)
# What the reader of a resume file that cannot be resumed can do.
ANOTHER_RESUME_FILE = "start the run with another resume file, which regenerates every text"
# The rows of a resume file that its description describes, as a refusal names them.
LISTED_ROWS = "rows of the listed calls"
LISTED_ROWS_DIFFER = (
    "a run stopped between the two writes of a call leaves only rows of calls the description "
    f"does not list; {ANOTHER_RESUME_FILE}"
)

CallKey = tuple[str, int, int]


class GenerationProgress:
    """The finished calls of a run: their rows, the writers' provenance, and their usage.

    With a resume file, every finished call is written to it at once and a rerun skips the
    calls the file lists. The file is a reference set file (the table with the screen
    columns unset, and its JSON description, which holds the ``rubric_hash`` of the run's
    rubric like that of every reference set) whose description also holds ``completed``,
    the ``[writer model, cell index, call index]`` of every finished call.
    It is rewritten whole, through a partial file, after every call, which costs little at
    the size of a reference set. The table is written first and the description last, as
    the commit point, and the description names the content of the table it was written
    with (``table_hash``). A run stopped between the two writes leaves the rows of a call
    that the description does not list. A rerun checks that the rows of the calls the
    description lists are the rows it was written for, as many and with its digest, keeps
    them, and drops the others, whose call it makes again; a file whose listed rows differ is
    refused, and so is one of an earlier version, which records no digest. A run stopped
    between the two writes of its first call leaves a table without a description, which is
    refused; deleting the table makes that call again.
    """

    def __init__(
        self,
        path: Path | None,
        rubric: Rubric,
        settings: GeneratorSettings,
        names: Sequence[str],
        cells: Sequence[Cell],
        prompt_hashes: Sequence[str],
    ) -> None:
        """Start without finished calls; use `open` to read a resume file."""
        self._path = path
        self._rubric = rubric
        self._settings = settings
        self._names = list(names)
        self._cells = list(cells)
        self._hashes = list(prompt_hashes)
        self._rows: list[dict[str, Any]] = []
        self._completed: set[CallKey] = set()
        self._writers: dict[str, dict[str, str]] = {}
        self._usage: dict[str, dict[str, int]] = {}

    @classmethod
    def open(
        cls,
        path: Path | None,
        rubric: Rubric,
        settings: GeneratorSettings,
        names: Sequence[str],
        cells: Sequence[Cell],
        prompt_hashes: Sequence[str],
    ) -> GenerationProgress:
        """Return the progress of a run, read from its resume file when there is one.

        Args:
            path: The resume file, ``.parquet`` or ``.csv``; ``None`` keeps the progress in
                memory only. A file that does not exist yet is created by the first call.
            rubric: The rubric of the run.
            settings: The settings of the run.
            names: The writer models of the run, in order.
            cells: The cells of the rubric, in order.
            prompt_hashes: The prompt hash of every cell, in the order of ``cells``.

        Returns:
            The progress.

        Raises:
            ConfigurationError: If the path has another suffix or its folder does not
                exist, the table has no description or the description no table, or the
                file belongs to another rubric, other settings (other than the
                concurrency), other writers, or other prompts, was written by an earlier
                version (no ``written_by``, ``rubric_hash`` or ``table_hash``), holds other
                rows of the finished calls than the description was written for (another
                count or another digest), or is malformed.
        """
        progress = cls(path, rubric, settings, names, cells, prompt_hashes)
        if path is None:
            return progress
        check_suffix(path)
        if not path.parent.is_dir():
            raise ConfigurationError(f"the folder of the resume file {path} does not exist")
        if path.exists() or description_path(path).exists():
            progress._load(path)
        return progress

    @property
    def finished(self) -> int:
        """Return the number of finished calls."""
        return len(self._completed)

    def pending(self, name: str) -> list[tuple[int, int]]:
        """Return the ``(cell index, call index)`` of every unfinished call of a writer."""
        return [
            (cell_index, call_index)
            for cell_index in range(len(self._cells))
            for call_index in range(self._settings.calls_per_cell)
            if (name, cell_index, call_index) not in self._completed
        ]

    def record_writer(self, name: str, provenance: Mapping[str, str]) -> None:
        """Remember the provenance of a writer, refusing one that differs from the file's.

        Raises:
            ConfigurationError: Naming the writer and the provenance keys that differ, if
                the resume file was written by a writer of that name with another
                provenance.
        """
        current = dict(provenance)
        if name in self._writers and self._writers[name] != current:
            recorded = self._writers[name]
            differing = sorted(
                key
                for key in set(recorded) | set(current)
                if key not in recorded or key not in current or recorded[key] != current[key]
            )
            raise ConfigurationError(
                f"writer {name!r} is not the writer of the resume file {self._path}: its "
                f"provenance differs in {differing}"
            )
        self._writers[name] = current

    def add_call(
        self,
        name: str,
        cell_index: int,
        call_index: int,
        outcome: CallOutcome,
        rows: Sequence[Mapping[str, Any]],
    ) -> None:
        """Record a finished call and write the resume file, if there is one."""
        self._rows.extend(dict(row) for row in rows)
        self._completed.add((name, cell_index, call_index))
        if name not in self._usage:
            self._usage[name] = dict.fromkeys(USAGE_KEYS, 0)
        usage = self._usage[name]
        usage["calls"] += 1
        usage["attempts"] += outcome.attempts
        usage["prompt_tokens"] += outcome.prompt_tokens
        usage["completion_tokens"] += outcome.completion_tokens
        if self._path is not None:
            self._write(self._path)

    def frame(self) -> pd.DataFrame:
        """Return the rows of the finished calls, ordered by writer, cell, and call."""
        frame = rows_frame(self._rows)
        writer_rank = {name: rank for rank, name in enumerate(self._names)}
        cell_rank = {(c.target, c.polarity, c.level): rank for rank, c in enumerate(self._cells)}
        order = pd.DataFrame(
            {
                "writer": [writer_rank[name] for name in frame["writer_model"]],
                "cell": [
                    cell_rank[key]
                    for key in zip(frame["target"], frame["polarity"], frame["level"], strict=True)
                ],
                "call": frame["call_index"].to_numpy(),
            }
        )
        positions = order.sort_values(["writer", "cell", "call"], kind="stable").index
        return frame.loc[positions].reset_index(drop=True)

    def writers(self) -> dict[str, dict[str, str]]:
        """Return the provenance of every recorded writer, in the order of the run."""
        return {name: self._writers[name] for name in self._names if name in self._writers}

    def usage(self) -> dict[str, dict[str, int]]:
        """Return the usage of every writer that finished a call, in the order of the run."""
        return {name: self._usage[name] for name in self._names if name in self._usage}

    def _write(self, path: Path) -> None:
        """Write the rows and the description of the finished calls to the resume file."""
        frame = rows_frame(self._rows)
        frame[SCREEN_COLUMNS[0]] = False
        frame[SCREEN_COLUMNS[1]] = ""
        snapshot = ReferenceSet(
            frame,
            self._rubric.rubric_id,
            self._rubric.content_hash(),
            self._settings,
            (),
            self.writers(),
            self.usage(),
            BOUND_GENERATED,
        )
        rank = {name: position for position, name in enumerate(self._names)}
        completed = sorted(self._completed, key=lambda key: (rank[key[0]], key[1], key[2]))
        description = {
            **snapshot.description(),
            "completed": [list(key) for key in completed],
        }
        write_reference_files(path, frame, description)

    def _load(self, path: Path) -> None:
        """Read and check a resume file, then take over the calls its description lists.

        Rows of a call of this run that the description does not list were written by a
        run stopped between the two writes of that call; they are dropped and logged, and
        the call is made again. The rows of the listed calls must be the rows the description
        was written for, as many and with its digest (`check_described_rows`); a table whose
        listed rows differ is refused.
        """
        sidecar = description_path(path)
        if path.is_file() and not sidecar.is_file():
            raise ConfigurationError(
                f"the resume file {path} has no description {sidecar.name}; a run stopped "
                "between the two writes of its first call leaves such a table, whose call was "
                "never recorded: delete the table to make that call again, or choose another "
                "resume file"
            )
        frame, description = read_reference_files(
            path, RESUME_KEYS, rows_may_exceed=True, remedy=ANOTHER_RESUME_FILE
        )
        self._check_origin(description, sidecar)
        completed = self._parse_completed(description["completed"], sidecar)
        writers = parse_writers(description["writers"], sidecar)
        usage = parse_usage(description["usage"], sidecar)
        named = {key[0] for key in completed} | set(frame["writer_model"].astype(str))
        strangers = sorted((named | set(writers) | set(usage)) - set(self._names))
        if strangers:
            raise ConfigurationError(
                f"the resume file {path} holds the writers {strangers}, which this run does "
                f"not have; run with them or choose another resume file"
            )
        calls = self._row_calls(frame, sidecar)
        listed = [call in completed for call in calls]
        check_described_rows(
            frame.loc[listed],
            description,
            path,
            sidecar,
            which=LISTED_ROWS,
            remedy=LISTED_ROWS_DIFFER,
        )
        unlisted = sorted({call for call in calls if call not in completed})
        if unlisted:
            logger.info(
                "the resume file %s holds %d rows of calls its description does not list, %s; "
                "a run stopped between the two writes of a call leaves them, so they are "
                "dropped and those calls are made again",
                path,
                listed.count(False),
                [list(call) for call in unlisted],
            )
        self._rows = frame.loc[listed].drop(columns=list(SCREEN_COLUMNS)).to_dict("records")
        self._completed, self._writers, self._usage = completed, writers, usage
        logger.info("resuming from %s: %d calls are finished", path, len(completed))

    def _check_origin(self, description: Mapping[str, Any], sidecar: Path) -> None:
        """Refuse a resume file of another rubric or of other settings."""
        rubric, stored_hash = description["rubric_id"], description["rubric_hash"]
        if (rubric, stored_hash) != (self._rubric.rubric_id, self._rubric.content_hash()):
            raise ConfigurationError(
                f"{sidecar} was written for rubric {rubric} with hash {stored_hash}, not for "
                f"rubric {self._rubric.rubric_id} with hash {self._rubric.content_hash()}"
            )
        try:
            stored = asdict(GeneratorSettings(**description["settings"]))
        except (TypeError, SettingsError) as error:
            raise ConfigurationError(f"{sidecar}: the settings are malformed: {error}") from error
        current = asdict(self._settings)
        differing = [
            f"{name} ({stored[name]!r} in the file, {current[name]!r} now)"
            for name in current
            if name not in EXECUTION_SETTINGS and stored[name] != current[name]
        ]
        if differing:
            raise ConfigurationError(
                f"{sidecar} was written with other settings: {', '.join(differing)}; resume "
                "with the same settings or choose another resume file"
            )

    def _parse_completed(self, value: Any, sidecar: Path) -> set[CallKey]:
        """Read the finished calls, refusing entries that are not calls of this run."""
        if not isinstance(value, list):
            raise ConfigurationError(f"{sidecar}: completed must be a list of calls")
        keys: set[CallKey] = set()
        for entry in value:
            valid = (
                isinstance(entry, list)
                and len(entry) == 3
                and isinstance(entry[0], str)
                and all(isinstance(n, int) and not isinstance(n, bool) for n in entry[1:])
                and 0 <= entry[1] < len(self._cells)
                and 0 <= entry[2] < self._settings.calls_per_cell
            )
            if not valid:
                raise ConfigurationError(
                    f"{sidecar}: completed holds {entry!r}, which is not "
                    "[writer model, cell index, call index] of a call of this run"
                )
            keys.add((entry[0], entry[1], entry[2]))
        return keys

    def _row_calls(self, frame: pd.DataFrame, sidecar: Path) -> list[CallKey]:
        """Return the call of every row, refusing a row that is not of a call of this run.

        Raises:
            ConfigurationError: If a row belongs to another rubric, names an unknown cell,
                has a call index outside ``calls_per_cell``, or was written with other
                prompts.
        """
        cell_index = {(c.target, c.polarity, c.level): index for index, c in enumerate(self._cells)}
        calls: list[CallKey] = []
        for position, row in enumerate(frame.itertuples(index=False)):
            if str(row.rubric_id) != self._rubric.rubric_id:
                raise ConfigurationError(
                    f"{sidecar}: row {position} belongs to rubric {row.rubric_id}, not to "
                    f"rubric {self._rubric.rubric_id}"
                )
            key = (str(row.target), str(row.polarity), str(row.level))
            if key not in cell_index:
                raise ConfigurationError(f"{sidecar}: row {position} names the unknown cell {key}")
            call = (str(row.writer_model), cell_index[key], int(row.call_index))
            if not 0 <= call[2] < self._settings.calls_per_cell:
                raise ConfigurationError(
                    f"{sidecar}: row {position} belongs to the call {list(call)}, which is not "
                    "a call of this run"
                )
            if str(row.prompt_hash) != self._hashes[cell_index[key]]:
                raise ConfigurationError(
                    f"{sidecar}: row {position} was written with other prompts than this run "
                    f"sends (prompt hash {row.prompt_hash})"
                )
            calls.append(call)
        return calls
