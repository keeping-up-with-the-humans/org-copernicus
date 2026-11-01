"""The cells of reference-text generation: one per target, polarity, and intensity level."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from copernicus.rubric.identifiers import TARGET_CONSTRUCT
from copernicus.rubric.model import Attribute, Rubric

logger = logging.getLogger(__name__)

__all__ = ["ANCHOR_MAP", "LEVELS", "POLARITIES", "TARGET_CONSTRUCT", "Cell", "build_cells"]

LEVELS = ("top", "mid", "low")
POLARITIES = ("positive", "negative")

# Per (level, polarity): the rubric level whose descriptions guide the writer, and the phrase
# that says how strongly the texts should show the pole. Both poles meet at level 3, where the
# texts are only faintly on one side.
ANCHOR_MAP: dict[tuple[str, str], tuple[int, str]] = {
    ("top", "positive"): (5, "STRONGLY and unambiguously"),
    ("mid", "positive"): (4, "MODERATELY and recognizably"),
    ("low", "positive"): (3, "FAINTLY but discernibly (a mild, threshold-level)"),
    ("top", "negative"): (1, "STRONGLY and unambiguously"),
    ("mid", "negative"): (2, "MODERATELY and recognizably"),
    ("low", "negative"): (3, "FAINTLY but discernibly (a mild, threshold-level)"),
}


@dataclass(frozen=True)
class Cell:
    """One unit of reference-text generation.

    Attributes:
        target: ``"construct"`` or the id of an attribute.
        polarity: ``"positive"`` or ``"negative"``.
        level: ``"top"``, ``"mid"``, or ``"low"``.
        anchor_level: Rubric level (1 to 5) whose descriptions guide the writer.
        intensity: Phrase that says how strongly the texts show the pole.
        pole_guidance: What the pole looks like in concrete terms, from the rubric.
    """

    target: str
    polarity: str
    level: str
    anchor_level: int
    intensity: str
    pole_guidance: str


def _attribute_guidance(attribute: Attribute, anchor_level: int) -> str:
    """List the level description of each marker of an attribute, one ``- `` line per marker."""
    return "\n".join(
        f"- {marker.name}: {marker.level(anchor_level).description}" for marker in attribute.markers
    )


def _construct_guidance(rubric: Rubric, anchor_level: int) -> str:
    """List the guidance of every attribute, each under its name in square brackets."""
    return "\n".join(
        f"[{attribute.name}]\n{_attribute_guidance(attribute, anchor_level)}"
        for attribute in rubric.attributes
    )


def build_cells(rubric: Rubric) -> list[Cell]:
    """Enumerate the cells of a rubric.

    The construct comes first, then each attribute in rubric order; within a target the
    levels run top, mid, low and, within a level, the polarity runs positive then negative.
    The naming rule of `copernicus.rubric.identifiers` keeps every attribute id apart from
    the construct's target name.

    Args:
        rubric: The rubric whose construct and attributes get reference texts.

    Returns:
        Six cells for the construct and six per attribute.
    """
    cells: list[Cell] = []
    for target in [TARGET_CONSTRUCT, *(attribute.attribute_id for attribute in rubric.attributes)]:
        for level in LEVELS:
            for polarity in POLARITIES:
                anchor_level, intensity = ANCHOR_MAP[(level, polarity)]
                if target == TARGET_CONSTRUCT:
                    guidance = _construct_guidance(rubric, anchor_level)
                else:
                    guidance = _attribute_guidance(rubric.attribute(target), anchor_level)
                cells.append(Cell(target, polarity, level, anchor_level, intensity, guidance))
    return cells
