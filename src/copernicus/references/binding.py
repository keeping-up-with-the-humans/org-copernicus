"""How reference texts and their embeddings are bound to the content of their rubric.

A reference set records the content hash of the rubric its texts express (``rubric_hash``) and
how it came by that hash (``bound_with``): ``"generated"`` when `generate_references` wrote the
texts for the rubric, ``"for_rubric"`` when they were bound to it by an explicit act, through
`ReferenceSet.for_rubric` or `ReferenceSet.load` with ``bind_to``. Its description file keeps
both, the reference embeddings carry both in their sidecar, and the provenance of a scoring run
records both. A file written before these keys existed is bound only on request, and a recorded
binding is never overridden. The same request accepts the table of a file written before its
description recorded the digest of the table (``table_hash``); a recorded digest that does not
name the table is refused before any binding.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from copernicus._digests import TABLE_HASH_KEY
from copernicus._version import __version__
from copernicus.errors import ConfigurationError
from copernicus.rubric.model import HASH_LENGTH, Rubric, is_content_hash

logger = logging.getLogger(__name__)

__all__ = [
    "BINDINGS",
    "BOUND_FOR_RUBRIC",
    "BOUND_GENERATED",
    "BOUND_ON_REQUEST",
    "BOUND_WITH_KEY",
    "RUBRIC_HASH_KEY",
    "RUBRIC_KEY",
    "binding_problem",
    "check_binding",
    "hash_problem",
    "resolve_binding",
]

RUBRIC_HASH_KEY = "rubric_hash"
BOUND_WITH_KEY = "bound_with"
BOUND_GENERATED = "generated"
BOUND_FOR_RUBRIC = "for_rubric"
BINDINGS = (BOUND_GENERATED, BOUND_FOR_RUBRIC)
# The key of the rubric id in the description of a set file; the files module reads it from here.
RUBRIC_KEY = "rubric_id"
BINDING_KEYS = (RUBRIC_HASH_KEY, BOUND_WITH_KEY)
# The keys that a set file of an earlier version lacks and an explicit binding stands in for:
# the binding, and the digest of the table, whose table the binding then accepts as it is.
BOUND_ON_REQUEST = (*BINDING_KEYS, TABLE_HASH_KEY)
MEANINGS = {
    RUBRIC_HASH_KEY: "the content hash of the rubric the references were written for "
    "(Rubric.content_hash())",
    BOUND_WITH_KEY: f"how the references came by that hash, one of {BINDINGS}",
}


def hash_problem(value: object) -> str | None:
    """Say why a value is not a rubric's content hash, or return ``None`` when it is one."""
    if is_content_hash(value):
        return None
    return (
        f"{RUBRIC_HASH_KEY} must be the content hash of a rubric, {HASH_LENGTH} lowercase "
        f"hexadecimal characters as Rubric.content_hash returns, got {value!r}"
    )


def binding_problem(value: object) -> str | None:
    """Say why a value is not one of `BINDINGS`, or return ``None`` when it is one."""
    if isinstance(value, str) and value in BINDINGS:
        return None
    return f"{BOUND_WITH_KEY} must be one of {BINDINGS}, got {value!r}"


def check_binding(values: Mapping[str, Any], owner: str) -> None:
    """Refuse a sidecar without a well-formed rubric hash and binding.

    Args:
        values: The sidecar or description of reference embeddings.
        owner: What the values belong to, named in the message.

    Raises:
        ConfigurationError: If ``rubric_hash`` or ``bound_with`` is absent, or holds a value
            that is not a content hash or not one of `BINDINGS`.
    """
    for key in BINDING_KEYS:
        if key not in values:
            raise ConfigurationError(
                f"{owner} lacks {key}, {MEANINGS[key]}; embed_references takes it from the "
                "reference set"
            )
    problem = hash_problem(values[RUBRIC_HASH_KEY])
    if problem is None:
        problem = binding_problem(values[BOUND_WITH_KEY])
    if problem is not None:
        raise ConfigurationError(f"{owner}: {problem}")


def _check_bind_to(description: Mapping[str, Any], sidecar: Path, bind_to: Rubric) -> None:
    """Refuse to bind a set file to a rubric of another id, or of another recorded content."""
    stored_id = description[RUBRIC_KEY]
    if stored_id != bind_to.rubric_id:
        raise ConfigurationError(
            f"{sidecar} is for rubric {stored_id}, but bind_to is rubric {bind_to.rubric_id}"
        )
    current = bind_to.content_hash()
    if RUBRIC_HASH_KEY in description and description[RUBRIC_HASH_KEY] != current:
        raise ConfigurationError(
            f"{sidecar} was written for rubric {stored_id} with hash "
            f"{description[RUBRIC_HASH_KEY]}, but bind_to has hash {current}: the texts describe "
            "another definition; regenerate them, or bind texts you checked against the rubric "
            "with ReferenceSet.for_rubric"
        )


def resolve_binding(
    description: Mapping[str, Any],
    table: Path,
    sidecar: Path,
    bind_to: Rubric | None,
    *,
    rows: int,
) -> tuple[str, str]:
    """Return the rubric hash and the binding of a reference set file.

    A description of this version holds the hash, the binding and the digest of the table
    (``table_hash``), and is taken as it is; ``bind_to`` must then name the rubric it
    records. A description of an earlier version lacks one of the three and is refused
    unless ``bind_to`` names a rubric. The set is then bound to that rubric as an explicit
    act (``"for_rubric"``), or keeps a binding it records, whose hash must be that rubric's;
    and a table without a recorded digest is accepted as it is by the same act, which is
    logged. A digest the description records has been checked against the table before, so
    a table that its description does not name is never bound.

    Args:
        description: The description of the file; a ``rubric_hash`` it holds has been checked
            to be a content hash, and a ``table_hash`` it holds to be the digest of the table.
        table: The table of the file, named in the message.
        sidecar: The description file, named in the message.
        bind_to: The rubric to bind a file of an earlier version to, or ``None``.
        rows: The number of rows of the table, named in the log when the table is accepted
            without a recorded digest.

    Returns:
        ``(rubric_hash, bound_with)``.

    Raises:
        ConfigurationError: If the description lacks a binding key or the table hash and
            ``bind_to`` is ``None`` (naming the first key it lacks and both remedies, and,
            without the table hash, asking the reader to vouch for the table as well), holds
            a binding that is not one of `BINDINGS`, or names another rubric, or another
            content, than ``bind_to``.
    """
    recorded = all(key in description for key in BINDING_KEYS)
    if recorded:
        problem = binding_problem(description[BOUND_WITH_KEY])
        if problem is not None:
            raise ConfigurationError(f"{sidecar}: {problem}")
    missing = [key for key in BOUND_ON_REQUEST if key not in description]
    if missing and bind_to is None:
        # Binding also accepts a table without a recorded digest, so the reader vouches for it.
        vouched = (
            f"{table.name} is the table this description was written for and "
            if TABLE_HASH_KEY in missing
            else ""
        )
        raise ConfigurationError(
            f"{sidecar} has no {missing[0]}: it was written by an earlier version of copernicus "
            f"than this one ({__version__}); if {vouched}the rubric is unchanged since its texts "
            "were written, bind the set to it with ReferenceSet.load(path, bind_to=rubric) and "
            f"save it again; otherwise regenerate {table.name}"
        )
    if bind_to is None:
        return description[RUBRIC_HASH_KEY], description[BOUND_WITH_KEY]
    _check_bind_to(description, sidecar, bind_to)
    if TABLE_HASH_KEY in missing:
        logger.info(
            "bound the reference set %s to rubric %s with hash %s and accepted its table of %d "
            "rows without a recorded table hash on request",
            table,
            bind_to.rubric_id,
            bind_to.content_hash(),
            rows,
        )
    elif missing:
        logger.info(
            "bound the reference set %s to rubric %s with hash %s on request",
            table,
            bind_to.rubric_id,
            bind_to.content_hash(),
        )
    if recorded:
        return description[RUBRIC_HASH_KEY], description[BOUND_WITH_KEY]
    return bind_to.content_hash(), BOUND_FOR_RUBRIC
