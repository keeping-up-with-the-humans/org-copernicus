"""Section builders for the prompt templates.

Each builder turns one part of a rubric attribute into the text block that a
template inserts. The marker order moves the markers, their level
descriptions, and the output template together; the glossary is never
reordered.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence

from copernicus.errors import PromptContractError
from copernicus.rubric.model import Attribute, Marker, Rubric

logger = logging.getLogger(__name__)

__all__ = [
    "LEVEL_NUMBERS",
    "definitions_section",
    "extraction_output_template",
    "level_descriptions_section",
    "lookup_markers",
    "marker_lines",
    "markers_section",
    "ordered_markers",
    "subrubrics_section",
]

LEVEL_NUMBERS = (1, 2, 3, 4, 5)
OUTPUT_ENTRY_TEMPLATE = (
    '        "{name}": {{\n'
    '            "evaluable": true/false (bool),\n'
    '            "quotes": ["exact verbatim passage from the text", ...] '
    "(list of strings; [] when evaluable is false)\n"
    "        }}"
)


def lookup_markers(attribute: Attribute, names: Iterable[str]) -> list[Marker]:
    """Resolve marker names to markers, in the given order.

    Args:
        attribute: Attribute that owns the markers.
        names: Marker names.

    Returns:
        The markers, one per name.

    Raises:
        PromptContractError: If a name is not a marker of the attribute.
    """
    by_name = {marker.name: marker for marker in attribute.markers}
    markers: list[Marker] = []
    for name in names:
        if name not in by_name:
            raise PromptContractError(
                f"marker {name!r} is not a marker of attribute {attribute.attribute_id!r}; "
                f"known: {list(by_name)}"
            )
        markers.append(by_name[name])
    return markers


def ordered_markers(attribute: Attribute, order: Sequence[str]) -> list[Marker]:
    """Resolve a marker order that must list every marker exactly once.

    Args:
        attribute: Attribute that owns the markers.
        order: Marker names in prompt order.

    Returns:
        The markers in prompt order.

    Raises:
        PromptContractError: If the order has an unknown name, a duplicate, or
            leaves a marker of the attribute out.
    """
    markers = lookup_markers(attribute, order)
    names = list(order)
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise PromptContractError(
            f"marker order for attribute {attribute.attribute_id!r} has duplicate names: "
            f"{duplicates}"
        )
    missing = [m.name for m in attribute.markers if m.name not in names]
    if missing:
        raise PromptContractError(
            f"marker order for attribute {attribute.attribute_id!r} omits markers: {missing}"
        )
    return markers


def marker_lines(markers: Iterable[Marker]) -> str:
    """Format markers as ``name: description`` lines.

    Args:
        markers: Markers in output order.

    Returns:
        One line per marker.
    """
    return "\n".join(f"{marker.name}: {marker.description}" for marker in markers)


def markers_section(attribute: Attribute, order: Sequence[str]) -> str:
    """Build the markers section: one ``name: description`` line per marker.

    Args:
        attribute: Attribute that owns the markers.
        order: Marker names in prompt order.

    Returns:
        The section text.

    Raises:
        PromptContractError: If ``order`` is not a permutation of the marker names.
    """
    return marker_lines(ordered_markers(attribute, order))


def definitions_section(rubric: Rubric) -> str:
    """Build the glossary section in the rubric's own order.

    Args:
        rubric: Rubric that holds the glossary.

    Returns:
        One ``term: definition`` line per glossary entry.
    """
    return "\n".join(f"{term}: {definition}" for term, definition in rubric.definitions.items())


def level_descriptions_section(attribute: Attribute, order: Sequence[str]) -> str:
    """Build the level descriptions of every marker, highest level first.

    Args:
        attribute: Attribute that owns the markers.
        order: Marker names in prompt order.

    Returns:
        Per marker a ``name:`` line followed by ``Level N [label]: description``
        lines for N from 5 down to 1.

    Raises:
        PromptContractError: If ``order`` is not a permutation of the marker names.
    """
    lines: list[str] = []
    for marker in ordered_markers(attribute, order):
        lines.append(f"{marker.name}:")
        for number in reversed(LEVEL_NUMBERS):
            level = marker.level(number)
            lines.append(f"    Level {number} [{level.label}]: {level.description}")
    return "\n".join(lines)


def subrubrics_section(attribute: Attribute, order: Sequence[str]) -> str:
    """Build the sub-rubric of every marker as a level-to-description mapping.

    The levels run from 5 down to 1, the order in which the extraction prompt lists
    the level descriptions.

    Args:
        attribute: Attribute that owns the markers.
        order: Marker names in prompt order.

    Returns:
        One ``name: {5: '...', 4: '...', ..., 1: '...'}`` line per marker, with the
        mapping written in Python dictionary notation.

    Raises:
        PromptContractError: If ``order`` is not a permutation of the marker names.
    """
    lines: list[str] = []
    for marker in ordered_markers(attribute, order):
        levels = {number: marker.level(number).description for number in reversed(LEVEL_NUMBERS)}
        lines.append(f"{marker.name}: {levels}")
    return "\n".join(lines)


def extraction_output_template(order: Sequence[str]) -> str:
    """Build the JSON skeleton that the extraction prompt asks the model to fill.

    Args:
        order: Marker names in prompt order.

    Returns:
        An indented block with one ``evaluable``/``quotes`` entry per marker,
        the entries separated by a blank line.
    """
    entries = [OUTPUT_ENTRY_TEMPLATE.format(name=name) for name in order]
    return "    {\n" + ",\n\n".join(entries) + "\n    }"
