"""Wording variants and the seeded rotation used by the LLM coders.

The prompts of the two-stage and one-stage LLM coders contain three pieces of
wording that can be rotated: the role description, the task framing, and the
way the level is determined. A wording pool holds the variants of each
dimension. For every text and run the rotation draws one variant per
dimension from a generator seeded by the run seed, the text, and the run
number, so that a rerun reproduces the same prompts.
"""

from __future__ import annotations

import csv
import hashlib
import logging
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

from copernicus.assets import asset_path
from copernicus.errors import PromptContractError

logger = logging.getLogger(__name__)

__all__ = [
    "Assignment",
    "Wording",
    "WordingPool",
    "assign",
    "decode_seed",
    "load_wording_pool",
    "marker_order",
    "select",
    "text_key",
]

Dimension = Literal["role", "task", "level"]

DIMENSIONS: tuple[Dimension, ...] = ("role", "task", "level")
WORDING_FILES: Mapping[str, str] = MappingProxyType(
    {
        "role": "wording_role.csv",
        "task": "wording_task.csv",
        "level": "wording_level.csv",
    }
)
TEXT_COLUMNS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "role": ("extraction", "scoring", "one_stage"),
        "task": ("extraction", "scoring", "one_stage"),
        "level": ("scoring", "one_stage"),
    }
)
DEFAULT_POOL_DIRECTORY = "coders/prompts"
ID_COLUMN = "variant_id"
ENABLED_COLUMN = "enabled"
ENABLE_ALL = "all"
ENABLED_VALUES: Mapping[str, bool] = MappingProxyType({"true": True, "false": False})
RUN_STRIDE = 1000
KEY_BYTES = 4
KEY_LIMIT = 2**31
KEY_MASK = KEY_LIMIT - 1


@dataclass(frozen=True)
class Wording:
    """One variant of one wording dimension.

    Attributes:
        variant_id: Identifier of the variant, unique across all dimensions.
        enabled: Whether the rotation may draw this variant.
        texts: Variant text per prompt kind (``extraction``, ``scoring``,
            ``one_stage``; the level dimension has no ``extraction`` text).
    """

    variant_id: str
    enabled: bool
    texts: Mapping[str, str]


@dataclass(frozen=True)
class WordingPool:
    """The variants of the three wording dimensions.

    Attributes:
        role: Role descriptions.
        task: Task framings.
        level: Level-determination procedures.
    """

    role: tuple[Wording, ...]
    task: tuple[Wording, ...]
    level: tuple[Wording, ...]

    def get(self, dimension: str, variant_id: str) -> Wording:
        """Return one variant by dimension and identifier.

        Args:
            dimension: ``"role"``, ``"task"``, or ``"level"``.
            variant_id: Identifier of the variant.

        Returns:
            The variant, whether or not it is enabled.

        Raises:
            PromptContractError: If the dimension or the identifier is unknown.
        """
        if dimension not in DIMENSIONS:
            raise PromptContractError(
                f"unknown wording dimension '{dimension}'; expected one of {list(DIMENSIONS)}"
            )
        for wording in self._variants(dimension):
            if wording.variant_id == variant_id:
                return wording
        raise PromptContractError(f"unknown {dimension} variant id '{variant_id}'")

    def _variants(self, dimension: str) -> tuple[Wording, ...]:
        """Return the variants of a dimension that is known to be valid."""
        if dimension == "role":
            return self.role
        if dimension == "task":
            return self.task
        return self.level


@dataclass(frozen=True)
class Assignment:
    """The wording drawn for one text and run.

    Attributes:
        role_id: Identifier of the role variant.
        task_id: Identifier of the task variant.
        level_id: Identifier of the level-determination variant.
        shuffle_seed: Seed for ordering the markers in the prompt.
    """

    role_id: str
    task_id: str
    level_id: str
    shuffle_seed: int


def load_wording_pool(directory: str | Path | None = None) -> WordingPool:
    """Read the wording variants of the three dimensions from CSV files.

    Each file is named ``wording_<dimension>.csv`` and has the columns
    ``variant_id``, ``enabled`` and one text column per prompt kind.

    Args:
        directory: Folder holding the three files, a ``str`` or a ``Path``;
            defaults to the wording shipped with the package.

    Returns:
        The pool with every row of every file, enabled or not.

    Raises:
        PromptContractError: If a file lacks a required column, holds an
            ``enabled`` value other than ``true``/``false``, repeats an
            identifier, or has no rows.
    """
    folder = Path(directory) if directory is not None else asset_path(DEFAULT_POOL_DIRECTORY)
    variants = {dimension: _read_dimension(folder, dimension) for dimension in DIMENSIONS}
    _check_ids_across_dimensions(variants)
    return WordingPool(role=variants["role"], task=variants["task"], level=variants["level"])


def select(
    pool: WordingPool,
    enable: Sequence[str] | Literal["all"] = "all",
    disable: Sequence[str] = (),
) -> WordingPool:
    """Narrow a pool to the variants the rotation may use.

    With ``enable="all"`` every row flagged enabled is kept. With a list of
    identifiers, each dimension that has at least one listed identifier keeps
    exactly the listed ones of that dimension (naming a row enables it even
    when its flag is false), and every other dimension keeps its enabled rows.
    ``disable`` then removes identifiers.

    Args:
        pool: The pool to narrow.
        enable: ``"all"`` or the identifiers to pin.
        disable: Identifiers to remove.

    Returns:
        A pool whose variants all carry ``enabled=True``.

    Raises:
        PromptContractError: If an identifier is unknown or a dimension ends
            empty.
    """
    if isinstance(enable, str) and enable != ENABLE_ALL:
        raise PromptContractError(
            f"enable must be '{ENABLE_ALL}' or a sequence of variant ids, got the string "
            f"'{enable}'; pass [{enable!r}] to pin a single variant"
        )
    known = {w.variant_id for dimension in DIMENSIONS for w in pool._variants(dimension)}
    pinned = set() if isinstance(enable, str) else set(enable)
    removed = set(disable)
    for variant_id in sorted(pinned | removed):
        if variant_id not in known:
            raise PromptContractError(f"unknown wording variant id '{variant_id}'")
    chosen: dict[str, tuple[Wording, ...]] = {}
    for dimension in DIMENSIONS:
        rows = pool._variants(dimension)
        pins = [w for w in rows if w.variant_id in pinned]
        kept = pins if pins else [w for w in rows if w.enabled]
        result = tuple(
            Wording(w.variant_id, True, w.texts) for w in kept if w.variant_id not in removed
        )
        if not result:
            raise PromptContractError(f"no {dimension} wording left after selection")
        chosen[dimension] = result
    return WordingPool(role=chosen["role"], task=chosen["task"], level=chosen["level"])


def text_key(text_id: Any) -> int:
    """Turn a text identifier into the non-negative integer used for seeding.

    Args:
        text_id: Identifier of the text.

    Returns:
        The identifier itself if it is an integer in ``[0, 2**31)``; otherwise
        a stable 31-bit hash of its string form.
    """
    if isinstance(text_id, int) and not isinstance(text_id, bool) and 0 <= text_id < KEY_LIMIT:
        return text_id
    digest = hashlib.blake2b(str(text_id).encode(), digest_size=KEY_BYTES).digest()
    return int.from_bytes(digest, "big") & KEY_MASK


def assign(pool: WordingPool, text_key: int, run_number: int, seed: int) -> Assignment:
    """Draw the wording for one text and run.

    The draws come from ``random.Random(seed + text_key * 1000 + run_number)``,
    one ``choice`` each for role, task, and level in that order, among the
    variants flagged enabled.

    Args:
        pool: The wording pool.
        text_key: Integer key of the text (see ``text_key``).
        run_number: Number of the run, starting at 1.
        seed: Run seed.

    Returns:
        The drawn variant identifiers and the marker-shuffle seed.

    Raises:
        PromptContractError: If a dimension has no enabled variant.
    """
    shuffle_seed = seed + text_key * RUN_STRIDE + run_number
    rng = random.Random(shuffle_seed)
    chosen: list[str] = []
    for dimension in DIMENSIONS:
        candidates = [w for w in pool._variants(dimension) if w.enabled]
        if not candidates:
            raise PromptContractError(f"no enabled {dimension} wording to draw from")
        chosen.append(rng.choice(candidates).variant_id)
    return Assignment(chosen[0], chosen[1], chosen[2], shuffle_seed)


def marker_order(names: Sequence[str], shuffle: bool, shuffle_seed: int) -> list[str]:
    """Return the marker names in the order they appear in the prompt.

    Args:
        names: Marker names in rubric order.
        shuffle: Whether to shuffle.
        shuffle_seed: Seed of the shuffle.

    Returns:
        A new list, shuffled with ``random.Random(shuffle_seed)`` when
        ``shuffle`` is true and unchanged otherwise.
    """
    ordered = list(names)
    if shuffle:
        random.Random(shuffle_seed).shuffle(ordered)
    return ordered


def decode_seed(seed: int, run_number: int) -> int:
    """Return the sampling seed of a run.

    Args:
        seed: Run seed.
        run_number: Number of the run.

    Returns:
        The seed plus the run number.
    """
    return seed + run_number


def _read_dimension(folder: Path, dimension: Dimension) -> tuple[Wording, ...]:
    """Read and validate the wording file of one dimension."""
    path = folder / WORDING_FILES[dimension]
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = (ID_COLUMN, ENABLED_COLUMN, *TEXT_COLUMNS[dimension])
        missing = [column for column in required if column not in (reader.fieldnames or [])]
        if missing:
            raise PromptContractError(f"{path}: missing column(s) {missing}")
        rows = list(reader)
    if not rows:
        raise PromptContractError(f"{path}: no wording rows")
    for index, row in enumerate(rows, start=1):
        _check_cells(path, index, row, required)
    wordings = tuple(_build_wording(path, row, TEXT_COLUMNS[dimension]) for row in rows)
    ids = [w.variant_id for w in wordings]
    if len(set(ids)) != len(ids):
        raise PromptContractError(f"{path}: duplicate variant_id")
    return wordings


def _check_cells(
    path: Path, index: int, row: Mapping[str, str | None], required: tuple[str, ...]
) -> None:
    """Reject a row whose required cell is absent (short row) or blank."""
    identifier = row[ID_COLUMN]
    label = identifier if identifier and identifier.strip() else f"#{index}"
    for column in required:
        cell = row[column]
        if cell is None or not cell.strip():
            raise PromptContractError(f"{path}: row {label} has no value in column '{column}'")


def _check_ids_across_dimensions(variants: Mapping[Dimension, tuple[Wording, ...]]) -> None:
    """Reject a variant id that occurs in more than one dimension."""
    owner: dict[str, str] = {}
    for dimension, wordings in variants.items():
        for wording in wordings:
            if wording.variant_id in owner:
                raise PromptContractError(
                    f"variant id '{wording.variant_id}' occurs in both the "
                    f"{owner[wording.variant_id]} and {dimension} wording"
                )
            owner[wording.variant_id] = dimension


def _build_wording(path: Path, row: Mapping[str, str], text_columns: tuple[str, ...]) -> Wording:
    """Build one wording from a CSV row."""
    variant_id = row[ID_COLUMN]
    flag = row[ENABLED_COLUMN].strip().lower()
    if flag not in ENABLED_VALUES:
        raise PromptContractError(
            f"{path}: variant '{variant_id}' has enabled value '{row[ENABLED_COLUMN]}' "
            "(expected true or false)"
        )
    texts = MappingProxyType({column: row[column] for column in text_columns})
    return Wording(variant_id, ENABLED_VALUES[flag], texts)
