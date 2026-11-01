"""The naming rule for the attribute and marker ids of a rubric, and the generated target names.

The ids become column names of the run store (``<id>__score``, ``<id>__evaluable``) and of
the tables the store exports, and the attribute ids become target ids of the reference cells,
the similarity and hybrid scores, and the validity panel, next to the targets the library
generates: the whole construct, ``construct``, and the roll-up of the attribute scores,
``<rubric_id>__rollup``. Rubric validation and the store therefore apply the same rule, from
this module: an id is a column-safe name, it is not a name the store or the exports use for
their own columns, it is not the name of a generated target and does not end in the roll-up
suffix, and it differs from every other attribute and marker id when case is ignored (SQLite
column names ignore case). The modules that generate or resolve the targets take the target
names from here. The marker names of an attribute follow a rule of their own,
`marker_name_problems`, which rubric validation and a rubric's constructor apply alike.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from typing import TypeGuard

logger = logging.getLogger(__name__)

__all__ = [
    "EXPORT_COLUMN_NAMES",
    "IDENTIFIER_PATTERN",
    "RESERVED_NAMES",
    "RESERVED_TARGET_NAMES",
    "ROLLUP_SUFFIX",
    "TARGET_CONSTRUCT",
    "LocatedMarkers",
    "identifier_problems",
    "is_text",
    "located_identifier_problems",
    "marker_name_problems",
    "target_name_problem",
]

IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
# The target of the whole construct, scored next to the attributes.
TARGET_CONSTRUCT = "construct"
# The suffix of the roll-up target of the attribute scores, ``<rubric_id>__rollup``.
ROLLUP_SUFFIX = "__rollup"
# The generated target names an id may not take, in lower case.
RESERVED_TARGET_NAMES = frozenset({TARGET_CONSTRUCT})
# Columns the tidy exports write next to one column per attribute id.
EXPORT_COLUMN_NAMES = ("construct_score", "n_runs")
LocatedMarkers = Sequence[tuple[int, object, Sequence[tuple[int, object, object]]]]
"""Per attribute ``(index, attribute id, markers)``, a marker as ``(index, marker id, name)``."""
# The fixed columns of the results table and the export columns, in lower case.
RESERVED_NAMES = frozenset(
    {
        "text_id",
        "text_key",
        "text_hash",
        "run_number",
        "model_name",
        "temperature",
        "seed",
        "role_id",
        "task_id",
        "level_id",
        "shuffle_seed",
        "seconds",
        "prompt_tokens",
        "completion_tokens",
        "created_at",
        "library_version",
        "temperature_applied",
        "top_p_applied",
        *EXPORT_COLUMN_NAMES,
    }
)


def target_name_problem(identifier: str) -> str | None:
    """Say why an id takes the name of a target the library generates, or return ``None``.

    Args:
        identifier: The id; case is ignored.

    Returns:
        The message for an id equal to ``construct`` or ending in ``__rollup``, else ``None``.
    """
    folded = identifier.lower()
    if folded in RESERVED_TARGET_NAMES:
        return f"id {identifier!r} is the name of the whole-construct target"
    if folded.endswith(ROLLUP_SUFFIX):
        return f"id {identifier!r} ends in {ROLLUP_SUFFIX!r}, the suffix of the roll-up target"
    return None


def identifier_problems(identifiers: Sequence[str]) -> list[tuple[int, str]]:
    """Check attribute and marker ids against the naming rule.

    An id must match ``[A-Za-z_][A-Za-z0-9_]*``, must not equal a reserved column name or
    ``construct``, the name of the whole-construct target, must not end in ``__rollup``, the
    suffix of the roll-up target, and must differ from every earlier id, all when case is
    ignored.

    Args:
        identifiers: The attribute ids, then the marker ids, in rubric order.

    Returns:
        ``(position, message)`` for every id that breaks the rule, in input order; each
        id is reported once, for the first rule it breaks. Empty when all ids pass.
    """
    problems: list[tuple[int, str]] = []
    earlier: dict[str, str] = {}
    for position, identifier in enumerate(identifiers):
        if not (isinstance(identifier, str) and IDENTIFIER_PATTERN.fullmatch(identifier)):
            problems.append(
                (
                    position,
                    f"id {identifier!r} cannot be a column name; it must match "
                    f"{IDENTIFIER_PATTERN.pattern}",
                )
            )
            continue
        folded = identifier.lower()
        if folded in RESERVED_NAMES:
            problems.append((position, f"id {identifier!r} collides with a reserved column name"))
        elif (clash := target_name_problem(identifier)) is not None:
            problems.append((position, clash))
        elif folded in earlier:
            problems.append(
                (
                    position,
                    f"id {identifier!r} collides with another attribute or marker id "
                    f"({earlier[folded]!r}); ids are compared without regard to case",
                )
            )
        earlier.setdefault(folded, identifier)
    return problems


def located_identifier_problems(located: Sequence[tuple[str, str]]) -> list[str]:
    """Check located ids against the naming rule and name each problem with its location.

    The loader checks the ids of a rubric dictionary with it, and a rubric checks its own ids
    with it when it is constructed, so both report the same lines.

    Args:
        located: ``(location, id)`` pairs, the attribute ids then the marker ids in rubric
            order; a location is the JSON path of the id, such as
            ``attributes/0/markers/1/marker_id``.

    Returns:
        ``<location>: <message>`` for every id that breaks the rule, in input order; empty
        when all ids pass.
    """
    problems = identifier_problems([identifier for _, identifier in located])
    return [f"{located[position][0]}: {message}" for position, message in problems]


def marker_name_problems(attributes: LocatedMarkers) -> list[str]:
    """Check the marker names of each attribute and name each problem with its location.

    Within an attribute a marker name must be unique and must not equal the id of another
    marker of the attribute: the extraction output is keyed by marker name or id, so such a key
    could name either marker. The loader checks the markers of a rubric dictionary with it, and
    a rubric checks its own with it when it is constructed, so both report the same lines. A
    name that is not a non-empty string is skipped, because the schema check reports it.

    Args:
        attributes: ``(index, attribute id, markers)`` for each attribute, with ``(index,
            marker id, name)`` for each of its markers; an index is the position in the
            rubric's list, and an attribute whose id is not a non-empty string is named by
            its index.

    Returns:
        ``attributes/<a>/markers/<m>/name: <message>`` for every problem, attribute by
        attribute and marker by marker; empty when every name passes.
    """
    problems: list[str] = []
    for attribute_index, attribute_id, markers in attributes:
        owner = f"'{attribute_id}'" if is_text(attribute_id) else f"at index {attribute_index}"
        ids = {marker_index: marker_id for marker_index, marker_id, _ in markers}
        seen: set[str] = set()
        for marker_index, _, name in markers:
            if not is_text(name):
                continue
            location = f"attributes/{attribute_index}/markers/{marker_index}/name"
            if name in seen:
                problems.append(f"{location}: duplicate marker name '{name}' in attribute {owner}")
            seen.add(name)
            if any(other == name for index, other in ids.items() if index != marker_index):
                problems.append(
                    f"{location}: marker name '{name}' equals the id of another marker of "
                    f"attribute {owner}; an extraction key could name either marker"
                )
    return problems


def is_text(value: object) -> TypeGuard[str]:
    """Tell whether a value is a non-empty string."""
    return isinstance(value, str) and value != ""
