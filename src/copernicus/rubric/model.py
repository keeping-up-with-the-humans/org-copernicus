"""Immutable data model of a coding rubric."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from copernicus.errors import RubricValidationError
from copernicus.rubric.identifiers import located_identifier_problems, marker_name_problems

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_LEVEL_ORDER",
    "HASH_LENGTH",
    "Attribute",
    "Level",
    "Marker",
    "Rubric",
    "is_content_hash",
]

LEVEL_MIN = 1
LEVEL_MAX = 5
HASH_LENGTH = 16
CONTENT_HASH_PATTERN = re.compile(rf"[0-9a-f]{{{HASH_LENGTH}}}")
DEFAULT_LEVEL_ORDER: tuple[int, ...] = (5, 4, 3, 2, 1)
"""The default level order, of a marker built in code without ``level_order``: 5 down to 1."""


def is_content_hash(value: object) -> bool:
    """Tell whether a value has the form of a rubric's content hash.

    Args:
        value: The value, for example a ``rubric_hash`` read from a file.

    Returns:
        Whether it is a string of `HASH_LENGTH` lowercase hexadecimal characters, the form
        that `Rubric.content_hash` returns.
    """
    return isinstance(value, str) and CONTENT_HASH_PATTERN.fullmatch(value) is not None


@dataclass(frozen=True)
class Level:
    """One level of a marker's sub-rubric.

    Attributes:
        level: Integer level, 1 (lowest) to 5 (highest).
        label: Short name of the level.
        description: What a text at this level looks like for the marker.
    """

    level: int
    label: str
    description: str


@dataclass(frozen=True)
class Marker:
    """An observable feature of a text that evidences an attribute.

    A marker checks its level order when it is constructed, so the level order of a marker
    built in code, or changed with ``dataclasses.replace``, names each of the levels 1 to 5
    once, as that of a loaded marker does.

    Attributes:
        marker_id: Identifier, unique within the rubric.
        name: Human-readable name, unique within its attribute.
        description: What the marker captures.
        levels: The five levels, ordered 1 to 5.
        level_order: The level numbers in the order in which the rubric file lists the
            marker's levels, each of 1 to 5 once. The one-stage prompt lists the level
            descriptions in this order; a marker built in code without it lists them from 5
            down to 1 (`DEFAULT_LEVEL_ORDER`). `Rubric.content_hash` does not see the order.
    """

    marker_id: str
    name: str
    description: str
    levels: tuple[Level, ...]
    level_order: tuple[int, ...] = DEFAULT_LEVEL_ORDER

    def __post_init__(self) -> None:
        """Check the level order.

        Raises:
            RubricValidationError: If ``level_order`` is not a tuple of whole numbers that
                lists each of the levels 1 to 5 once.
        """
        order = self.level_order
        whole = isinstance(order, tuple) and all(type(number) is int for number in order)
        if not (whole and sorted(order) == list(range(LEVEL_MIN, LEVEL_MAX + 1))):
            raise RubricValidationError(
                f"marker {self.marker_id!r}: level_order must be a tuple of whole numbers that "
                f"lists each of the levels {LEVEL_MIN} to {LEVEL_MAX} once, such as "
                f"{DEFAULT_LEVEL_ORDER}; got {order!r}"
            )

    def level(self, n: int) -> Level:
        """Return the level with number ``n``.

        Args:
            n: Level number from 1 to 5.

        Returns:
            The matching level.

        Raises:
            KeyError: If ``n`` lies outside 1..5.
        """
        if not LEVEL_MIN <= n <= LEVEL_MAX:
            raise KeyError(
                f"level {n!r} is outside {LEVEL_MIN}..{LEVEL_MAX} for marker {self.marker_id!r}"
            )
        return self.levels[n - LEVEL_MIN]


@dataclass(frozen=True)
class Attribute:
    """A dimension of the concept, scored on its own.

    Attributes:
        attribute_id: Identifier, unique within the rubric.
        name: Human-readable name.
        description: What the attribute captures.
        literature_sources: Sources that motivate the attribute; empty when the rubric names
            sources for the construct as a whole only (a rubric file may then leave the key
            out).
        markers: The markers that evidence the attribute.
    """

    attribute_id: str
    name: str
    description: str
    literature_sources: tuple[str, ...]
    markers: tuple[Marker, ...]


@dataclass(frozen=True)
class Rubric:
    """A complete rubric: a concept, its attributes, markers, and glossary.

    A rubric checks its attribute and marker ids when it is constructed, against the naming
    rule that the loader and the run store apply, and the names of the markers of each
    attribute, against the loader's rule for marker names. A rubric built in code, or changed
    with ``dataclasses.replace``, therefore obeys both rules as a loaded one does.

    Attributes:
        rubric_id: Identifier of the rubric and its version.
        concept: Name of the construct that is coded.
        concept_definition: Definition of the construct.
        literature_sources: Sources that motivate the construct as a whole; the attributes
            may name sources of their own as well, or instead.
        attributes: The attributes of the construct.
        definitions: Glossary mapping a term to its meaning.
    """

    rubric_id: str
    concept: str
    concept_definition: str
    literature_sources: tuple[str, ...]
    attributes: tuple[Attribute, ...]
    definitions: Mapping[str, str]

    def __post_init__(self) -> None:
        """Check the marker names and the attribute and marker ids against the rules of rubrics.

        The marker names are checked with `copernicus.rubric.identifiers.marker_name_problems`,
        then the ids with `copernicus.rubric.identifiers.located_identifier_problems` (the rule
        of `copernicus.rubric.identifiers.identifier_problems`), as the loader checks those of a
        rubric dictionary and in its order.

        Raises:
            RubricValidationError: Holding one ``<json path>: <message>`` line per marker name
                and per id that breaks a rule, the lines the loader reports for them.
        """
        named = [
            (
                a_index,
                attribute.attribute_id,
                [
                    (m_index, marker.marker_id, marker.name)
                    for m_index, marker in enumerate(attribute.markers)
                ],
            )
            for a_index, attribute in enumerate(self.attributes)
        ]
        located = [
            (f"attributes/{a_index}/attribute_id", attribute.attribute_id)
            for a_index, attribute in enumerate(self.attributes)
        ]
        located += [
            (f"attributes/{a_index}/markers/{m_index}/marker_id", marker.marker_id)
            for a_index, attribute in enumerate(self.attributes)
            for m_index, marker in enumerate(attribute.markers)
        ]
        violations = marker_name_problems(named) + located_identifier_problems(located)
        if violations:
            raise RubricValidationError("\n".join(violations))

    def attribute(self, attribute_id: str) -> Attribute:
        """Return the attribute with the given id.

        Args:
            attribute_id: Identifier of the attribute.

        Returns:
            The matching attribute.

        Raises:
            KeyError: If the rubric has no such attribute.
        """
        for candidate in self.attributes:
            if candidate.attribute_id == attribute_id:
                return candidate
        known = [a.attribute_id for a in self.attributes]
        raise KeyError(
            f"no attribute {attribute_id!r} in rubric {self.rubric_id!r}; known: {known}"
        )

    def marker_names(self, attribute_id: str) -> list[str]:
        """Return the marker names of an attribute in rubric order.

        Args:
            attribute_id: Identifier of the attribute.

        Returns:
            Marker names, in the order of the rubric.
        """
        return [marker.name for marker in self.attribute(attribute_id).markers]

    def marker_ids(self, attribute_id: str) -> list[str]:
        """Return the marker ids of an attribute in rubric order.

        Args:
            attribute_id: Identifier of the attribute.

        Returns:
            Marker ids, in the order of the rubric.
        """
        return [marker.marker_id for marker in self.attribute(attribute_id).markers]

    def to_dict(self) -> dict[str, Any]:
        """Return the rubric in the JSON layout that the schema describes.

        Every attribute is written with its ``literature_sources``, an empty list when it has
        none of its own: a rubric file that leaves an attribute's list out and one that writes
        it empty are the same rubric, with this one layout and one content hash. Each marker's
        sub-rubric lists the levels in the marker's `Marker.level_order`, so the rubric read
        back from the dictionary lists them in the same order.

        Returns:
            A plain dictionary that validates against the rubric schema.
        """
        return {
            "rubric_id": self.rubric_id,
            "concept": self.concept,
            "concept_definition": self.concept_definition,
            "literature_sources": list(self.literature_sources),
            "attributes": [_attribute_to_dict(a) for a in self.attributes],
            "definitions": dict(self.definitions),
        }

    def content_hash(self) -> str:
        """Return a short fingerprint of the rubric content.

        The canonical JSON is `to_dict` with sorted keys, except that the glossary
        enters as the ordered list of ``[term, meaning]`` pairs: the prompts list the
        glossary in rubric order, so a reordered glossary is a different rubric. The order in
        which a marker lists its levels (`Marker.level_order`) does not enter, as the keys are
        sorted, so the hash of a rubric stays what it was before that order was kept; the
        one-stage coder, whose prompt follows it, records it in its run store instead.

        Returns:
            The first 16 hex characters of the SHA-256 of the canonical JSON.
        """
        content = {
            **self.to_dict(),
            "definitions": [[term, meaning] for term, meaning in self.definitions.items()],
        }
        canonical = json.dumps(content, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:HASH_LENGTH]


def _attribute_to_dict(attribute: Attribute) -> dict[str, Any]:
    """Return one attribute in the JSON layout of the schema."""
    return {
        "attribute_id": attribute.attribute_id,
        "name": attribute.name,
        "description": attribute.description,
        "literature_sources": list(attribute.literature_sources),
        "markers": [_marker_to_dict(m) for m in attribute.markers],
    }


def _marker_to_dict(marker: Marker) -> dict[str, Any]:
    """Return one marker in the JSON layout of the schema, its levels in its level order.

    The sub-rubric maps the text of each level's number to the level, built from
    `Marker.levels` in their order, so a later level of the same key replaces an earlier one, as
    it did before the level order was kept; only then are its keys put in the level order. A
    marker built in code may hold a level whose key the level order does not name; such a key
    follows the others, in the order of the mapping. The content of the sub-rubric, and so the
    content hash of every rubric, is therefore what it was before.
    """
    held = {
        str(level.level): {"label": level.label, "description": level.description}
        for level in marker.levels
    }
    named = [str(number) for number in marker.level_order if str(number) in held]
    unnamed = [key for key in held if key not in named]
    return {
        "marker_id": marker.marker_id,
        "name": marker.name,
        "description": marker.description,
        "sub_rubric": {key: held[key] for key in (*named, *unnamed)},
    }
