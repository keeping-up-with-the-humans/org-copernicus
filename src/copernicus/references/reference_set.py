"""The reference texts of a run, with their files, summary, and balance check."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from copernicus._version import __version__
from copernicus.errors import ConfigurationError, SettingsError
from copernicus.references.binding import (
    BOUND_FOR_RUBRIC,
    binding_problem,
    resolve_binding,
)
from copernicus.references.files import (
    FRAME_COLUMNS,
    canonical_frame,
    check_suffix,
    description_path,
    parse_usage,
    parse_writers,
    read_reference_files,
    write_reference_files,
)
from copernicus.references.settings import GeneratorSettings
from copernicus.rubric.model import HASH_LENGTH, Rubric, is_content_hash

logger = logging.getLogger(__name__)

__all__ = ["FRAME_COLUMNS", "ReferenceSet", "Shortfall"]

REQUIRED_KEYS = ("rubric_id", "rubric_hash", "settings", "shortfalls", "writers", "usage")
CELL_KEY = ["writer_model", "target", "polarity", "level"]
POSITIVE = "positive"
NEGATIVE = "negative"


@dataclass(frozen=True)
class Shortfall:
    """A cell of a writer that produced fewer texts than requested.

    Attributes:
        writer_model: Name of the writer model.
        target: ``"construct"`` or an attribute id.
        polarity: ``"positive"`` or ``"negative"``.
        level: ``"top"``, ``"mid"``, or ``"low"``.
        expected: Texts requested over all calls of the cell.
        received: Texts kept over all calls of the cell.
    """

    writer_model: str
    target: str
    polarity: str
    level: str
    expected: int
    received: int

    @property
    def missing(self) -> int:
        """Return the number of texts that are missing."""
        return self.expected - self.received


def _normalised(text: str) -> str:
    """Casefold a text and collapse its whitespace, for the duplicate count."""
    return " ".join(text.split()).casefold()


@dataclass(frozen=True, eq=False)
class ReferenceSet:
    """Reference texts of one rubric, written by one or more writer models.

    A set is bound to the content of its rubric through ``rubric_hash``: the reference
    embeddings made from it carry the hash, and scoring refuses a rubric whose content hash
    differs, also one edited under the same ids. `generate_references` binds the set to the
    rubric it wrote for; build a set of texts written by other means with `for_rubric`.
    ``bound_with`` records which of the two happened, and the provenance of a scoring run
    carries it.

    Attributes:
        frame: One row per text with the columns ``FRAME_COLUMNS``: the cell
            (``rubric_id``, ``target``, ``polarity``, ``level``, ``style``), the
            ``writer_model``, the ``text``; the call that wrote it (``call_index``, the
            zero-based ``attempt`` whose texts were kept, the request ``seed`` of that
            attempt, its ``prompt_tokens`` and ``completion_tokens``, and the
            ``temperature_applied`` and ``top_p_applied`` by the backend, NaN when the
            backend does not report them); ``prompt_hash`` and ``prompt_version``; and the
            screen columns ``screened_out`` and ``screen_reason``. The set holds a new frame
            in the canonical form (`canonical_frame`) with a fresh range index, so neither
            the caller's frame object nor its index is kept.
        rubric_id: Identifier of the rubric the texts were written for.
        rubric_hash: Content hash of that rubric (`copernicus.rubric.Rubric.content_hash`):
            the definition the texts express.
        settings: The settings of the generation.
        shortfalls: The cells that got fewer texts than requested, including cells that
            got none, which have no row in ``frame``.
        writers: The provenance of each writer model (its backend's
            `copernicus.llm.Backend.provenance`), by model name; empty for a set
            built by hand.
        usage: What each writer model used, by model name: ``calls``, ``attempts`` (retries
            included), and the ``prompt_tokens`` and ``completion_tokens`` of all attempts as
            the backend reported them (an attempt without a count adds 0); empty for a set
            built by hand.
        bound_with: How the set came by its rubric hash: ``"generated"`` when
            `generate_references` wrote the texts for the rubric, ``"for_rubric"`` when
            they were bound to it by an explicit act (`for_rubric`, or `load` with
            ``bind_to``). It defaults to ``"for_rubric"``, since a caller of the
            constructor supplies the hash itself.
    """

    frame: pd.DataFrame
    rubric_id: str
    rubric_hash: str
    settings: GeneratorSettings
    shortfalls: tuple[Shortfall, ...] = ()
    writers: Mapping[str, Mapping[str, str]] = field(default_factory=dict)
    usage: Mapping[str, Mapping[str, int]] = field(default_factory=dict)
    bound_with: str = BOUND_FOR_RUBRIC

    def __post_init__(self) -> None:
        """Hold the frame in its canonical form, and check the rubric of its rows and the binding.

        Raises:
            ConfigurationError: If the frame is not a table of a reference set (its columns,
                a missing value or a value of another type; see `canonical_frame`), a row
                names another rubric than ``rubric_id``, ``rubric_hash`` is not the content
                hash of a rubric (empty, for example), or ``bound_with`` is not
                ``"generated"`` or ``"for_rubric"``.
        """
        object.__setattr__(self, "frame", canonical_frame(self.frame))
        others = sorted(set(self.frame["rubric_id"].astype(str)) - {self.rubric_id})
        if others:
            raise ConfigurationError(
                f"the reference set is for rubric {self.rubric_id}, but rows name the "
                f"rubric(s) {others}"
            )
        if not is_content_hash(self.rubric_hash):
            raise ConfigurationError(
                f"the reference set for rubric {self.rubric_id} needs rubric_hash, the content "
                f"hash of that rubric ({HASH_LENGTH} lowercase hexadecimal characters as "
                f"Rubric.content_hash returns), got {self.rubric_hash!r}; "
                "ReferenceSet.for_rubric takes it from the rubric"
            )
        problem = binding_problem(self.bound_with)
        if problem is not None:
            raise ConfigurationError(f"the reference set for rubric {self.rubric_id}: {problem}")

    @classmethod
    def for_rubric(
        cls,
        frame: pd.DataFrame,
        rubric: Rubric,
        settings: GeneratorSettings,
        *,
        shortfalls: tuple[Shortfall, ...] = (),
        writers: Mapping[str, Mapping[str, str]] | None = None,
        usage: Mapping[str, Mapping[str, int]] | None = None,
    ) -> ReferenceSet:
        """Bind reference texts written by hand, or by any other means, to a rubric.

        The set takes the id and the content hash of the rubric, so scoring accepts it only
        with a rubric of exactly this content. Binding states that the texts express the
        poles of this rubric as it reads now: after an edit of the rubric, check the texts
        and build the set again with ``for_rubric`` (regenerate a generated set instead).

        Args:
            frame: One row per text with the columns ``FRAME_COLUMNS``, in any order and any
                dtypes that `canonical_frame` takes; every row names the rubric in
                ``rubric_id``. The set holds a new frame in the canonical form with a fresh
                range index, so neither this object nor its index is kept.
            rubric: The rubric the texts were written for.
            settings: The settings recorded with the set.
            shortfalls: The cells that got fewer texts than intended.
            writers: The provenance of each writer model, by model name; ``None`` for none.
            usage: The calls, attempts and tokens of each writer model, by model name;
                ``None`` for none.

        Returns:
            The reference set, bound to ``rubric.content_hash()``, with ``bound_with``
            ``"for_rubric"``.

        Raises:
            ConfigurationError: If the frame is not a table of a reference set
                (`canonical_frame`) or a row names another rubric.
        """
        return cls(
            frame,
            rubric.rubric_id,
            rubric.content_hash(),
            settings,
            shortfalls,
            {} if writers is None else writers,
            {} if usage is None else usage,
            BOUND_FOR_RUBRIC,
        )

    def kept(self) -> pd.DataFrame:
        """Return the rows that the leakage screen did not exclude."""
        return self.frame.loc[~self.frame["screened_out"]]

    def save(self, path: str | Path) -> Path:
        """Write the table to ``path`` and the rest of the set to ``<path>.json``.

        Args:
            path: A file name ending in ``.parquet`` or ``.csv``; the parent folder must exist.

        Returns:
            The path of the table.

        Raises:
            ConfigurationError: If the suffix is neither ``.parquet`` nor ``.csv``.
        """
        target = Path(path)
        check_suffix(target)
        write_reference_files(target, self.frame, self.description())
        logger.info("wrote %d reference texts to %s", len(self.frame), target)
        return target

    def description(self) -> dict[str, Any]:
        """Return the JSON description that `save` writes beside the table.

        Returns:
            ``rubric_id``, ``rubric_hash``, ``bound_with``, ``settings``, ``shortfalls``,
            ``writers``, ``usage``, and ``library_version``; `save` adds ``written_by``,
            ``n_rows`` and ``table_hash``, the digest of the table it writes.
        """
        return {
            "rubric_id": self.rubric_id,
            "rubric_hash": self.rubric_hash,
            "bound_with": self.bound_with,
            "settings": asdict(self.settings),
            "shortfalls": [asdict(shortfall) for shortfall in self.shortfalls],
            "writers": {name: dict(values) for name, values in self.writers.items()},
            "usage": {name: dict(values) for name, values in self.usage.items()},
            "library_version": __version__,
        }

    @classmethod
    def load(cls, path: str | Path, *, bind_to: Rubric | None = None) -> ReferenceSet:
        """Read a reference set written by `save`.

        The description names the table it was written with (``table_hash``), and a table
        it does not name, as a save stopped between its two writes leaves, is refused, also
        with ``bind_to``. A file of an earlier version records no ``rubric_hash``, no
        ``bound_with`` or no ``table_hash``. It is refused unless ``bind_to`` names a
        rubric, to which the set is then bound as `for_rubric` binds one; a file that
        records its binding and lacks only ``table_hash`` keeps that binding, ``"generated"``
        for a generated set. A table without a recorded digest is then accepted as it is.
        Binding is an explicit statement that the table is the one its description was
        written for and that its texts express this rubric as it reads now. A file of this
        version is read as it is; ``bind_to`` must then name the rubric it records, and
        never overrides it.

        Args:
            path: The table written by ``save``; the description is read from
                ``<path>.json``.
            bind_to: The rubric to bind a file of an earlier version to; ``None`` refuses
                such a file.

        Returns:
            The reference set.

        Raises:
            ConfigurationError: If the suffix is unknown, a file is missing, or the
                description is not a JSON object, lacks a field (``written_by`` among them:
                a file of an earlier version must be regenerated; without ``rubric_hash``,
                ``bound_with`` or ``table_hash`` it can be bound with ``bind_to``), holds a
                rubric hash that is not a content hash, a table hash that is not a digest, a
                binding other than ``"generated"`` and ``"for_rubric"``, or settings,
                shortfalls, or writers that are malformed or out of range, does not belong to
                the table (another rubric, another row count, another digest), or names
                another rubric, or another content, than ``bind_to``; or if the table holds a
                value its set cannot take (`canonical_frame`), such as a missing text or a float
                in an integer column in a file of an earlier version; the message names the file.
        """
        source = Path(path)
        frame, description = read_reference_files(source, REQUIRED_KEYS, allow_missing_hash=True)
        sidecar = description_path(source)
        rubric_hash, bound_with = resolve_binding(
            description, source, sidecar, bind_to, rows=len(frame)
        )
        try:
            settings = GeneratorSettings(**description["settings"])
            shortfalls = tuple(Shortfall(**record) for record in description["shortfalls"])
        except (TypeError, SettingsError) as error:
            raise ConfigurationError(
                f"{sidecar}: the settings or shortfalls are malformed: {error}"
            ) from error
        writers = parse_writers(description["writers"], sidecar)
        usage = parse_usage(description["usage"], sidecar)
        rubric_id = description["rubric_id"]
        try:
            return cls(
                frame, rubric_id, rubric_hash, settings, shortfalls, writers, usage, bound_with
            )
        except ConfigurationError as error:
            raise ConfigurationError(
                f"{source}: {error}; fill or fix the column with pandas and bind the table with "
                f"ReferenceSet.for_rubric, or regenerate {source.name}"
            ) from error

    def summary(self) -> dict[str, Any]:
        """Summarise the texts, the shortfalls, the screen, the duplicates, and the balance.

        Returns:
            ``texts_per_cell``: texts per ``(target, polarity, level)`` over all writers;
            ``shortfalls``: missing texts per ``(writer_model, target, polarity, level)``;
            ``screened_out``: ``total`` and ``by_writer`` counts; ``duplicates``: texts
            that repeat an earlier text of the same writer and cell (casefolded, whitespace
            collapsed); ``balance``: ``balanced`` (positive and negative kept counts agree for
            every writer and target), ``complete`` (no pole of a writer and target is
            without kept texts), and ``problems``: the unbalanced writer-targets followed by
            the empty poles, written ``"<writer>/<target>/<polarity>: no kept texts"``;
            ``usage``: the calls, attempts, and prompt and completion tokens of each writer
            model (see `usage`).
        """
        frame = self.frame
        per_cell = frame.groupby(["target", "polarity", "level"]).size()
        writers = sorted(set(frame["writer_model"]) | {s.writer_model for s in self.shortfalls})
        screened = frame.groupby("writer_model")["screened_out"].sum()
        by_writer = {
            name: int(screened.loc[name]) if name in screened.index else 0 for name in writers
        }
        normalised = frame["text"].astype(str).map(_normalised)
        duplicates = int(
            frame.assign(normalised=normalised).duplicated(CELL_KEY + ["normalised"]).sum()
        )
        problems = self._balance_problems()
        empty_poles = self._empty_poles()
        return {
            "texts_per_cell": {key: int(count) for key, count in per_cell.items()},
            "shortfalls": {
                (s.writer_model, s.target, s.polarity, s.level): s.missing
                for s in self.shortfalls
                if s.missing > 0
            },
            "screened_out": {"total": sum(by_writer.values()), "by_writer": by_writer},
            "duplicates": duplicates,
            "balance": {
                "balanced": not problems,
                "complete": not empty_poles,
                "problems": problems + empty_poles,
            },
            "usage": {name: dict(values) for name, values in self.usage.items()},
        }

    def _kept_counts(self) -> tuple[dict[tuple[str, str, str], int], list[tuple[str, str]]]:
        """Count the kept texts per writer, target, and polarity, and list the writer-targets."""
        counts = self.kept().groupby(["writer_model", "target", "polarity"]).size()
        found = {
            (writer, target, polarity): int(n) for (writer, target, polarity), n in counts.items()
        }
        pairs = set(zip(self.frame["writer_model"], self.frame["target"], strict=True))
        pairs |= {(s.writer_model, s.target) for s in self.shortfalls}
        return found, sorted(pairs)

    def _empty_poles(self) -> list[str]:
        """Name each pole of a writer and target that has no kept text."""
        found, pairs = self._kept_counts()
        return [
            f"{writer}/{target}/{polarity}: no kept texts"
            for writer, target in pairs
            for polarity in (POSITIVE, NEGATIVE)
            if (writer, target, polarity) not in found
        ]

    def _balance_problems(self) -> list[str]:
        """Describe each writer and target whose kept positive and negative counts differ."""
        found, pairs = self._kept_counts()
        problems = []
        for writer, target in pairs:
            positive = found.get((writer, target, POSITIVE), 0)
            negative = found.get((writer, target, NEGATIVE), 0)
            if positive != negative:
                problems.append(
                    f"writer {writer!r}, target {target!r}: "
                    f"{positive} positive and {negative} negative kept texts"
                )
        return problems

    def assert_balanced(self) -> None:
        """Check that every writer has as many positive as negative kept texts per target.

        Raises:
            ConfigurationError: Naming each writer and target whose counts differ, for
                example because a cell fell short or its texts were all screened out.
        """
        problems = self._balance_problems()
        if problems:
            raise ConfigurationError("unbalanced reference texts: " + "; ".join(problems))
