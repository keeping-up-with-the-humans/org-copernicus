"""Load, validate, and build rubrics from JSON."""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Any

from jsonschema import Draft202012Validator

from copernicus.assets import asset_path
from copernicus.errors import RubricValidationError
from copernicus.rubric.identifiers import (
    is_text,
    located_identifier_problems,
    marker_name_problems,
)
from copernicus.rubric.model import LEVEL_MAX, LEVEL_MIN, Attribute, Level, Marker, Rubric

logger = logging.getLogger(__name__)

__all__ = ["load_rubric", "rubric_from_dict", "toy_rubric_path", "validate_rubric_dict"]

SCHEMA_ASSET = "rubric/schema.json"
TOY_RUBRIC_ASSET = "assets/rubrics/constructive_feedback.json"
ROOT_PATH_LABEL = "<root>"


def toy_rubric_path() -> Path:
    """Return the path of the small example rubric shipped with the package.

    Returns:
        Path of the toy rubric JSON file inside the installed package.
    """
    return asset_path(TOY_RUBRIC_ASSET)


@lru_cache(maxsize=1)
def _schema() -> dict[str, Any]:
    """Load the rubric JSON Schema from the package."""
    return json.loads(asset_path(SCHEMA_ASSET).read_text(encoding="utf-8"))


def validate_rubric_dict(data: dict[str, Any]) -> None:
    """Check a rubric dictionary against the schema, the id rule, and the marker names.

    The attribute and marker ids follow the rule of
    `copernicus.rubric.identifiers.identifier_problems`, the one a `Rubric` applies when
    it is constructed and the run store applies, so a rubric that validates can be stored
    and no id takes the name of a target the library generates (``construct``,
    ``<rubric_id>__rollup``). Marker names
    are unique within their attribute and differ from the ids of the attribute's other
    markers.

    Args:
        data: The rubric as parsed from JSON.

    Raises:
        RubricValidationError: If any rule is violated. The message holds one
            line per violation, formatted ``<json path>: <message>``.
    """
    violations = [
        f"{_format_path(error.absolute_path)}: {error.message}"
        for error in sorted(
            Draft202012Validator(_schema()).iter_errors(data),
            key=lambda error: [str(part) for part in error.absolute_path],
        )
    ]
    if isinstance(data, dict):
        violations.extend(_marker_name_violations(data))
        violations.extend(_identifier_violations(data))
    if violations:
        raise RubricValidationError("\n".join(violations))


def rubric_from_dict(data: dict[str, Any]) -> Rubric:
    """Validate a rubric dictionary and build the immutable rubric.

    Args:
        data: The rubric as parsed from JSON.

    Returns:
        The validated rubric.

    Raises:
        RubricValidationError: If the dictionary violates the schema.
    """
    validate_rubric_dict(data)
    return Rubric(
        rubric_id=data["rubric_id"],
        concept=data["concept"],
        concept_definition=data["concept_definition"],
        literature_sources=tuple(data["literature_sources"]),
        attributes=tuple(_build_attribute(a) for a in data["attributes"]),
        definitions=MappingProxyType(dict(data["definitions"])),
    )


def load_rubric(path: str | Path) -> Rubric:
    """Read a rubric JSON file, validate it, and build the rubric.

    Args:
        path: Path of the rubric JSON file.

    Returns:
        The validated rubric.

    Raises:
        FileNotFoundError: If the file does not exist.
        RubricValidationError: If the file is not UTF-8 text, is not valid JSON, or
            violates the schema.
    """
    raw = Path(path).read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise RubricValidationError(f"{path}: not a UTF-8 JSON file ({error})") from error
    try:
        data = json.loads(text)
    except json.JSONDecodeError as error:
        raise RubricValidationError(f"{path}: not valid JSON ({error})") from error
    return rubric_from_dict(data)


def _format_path(parts: Any) -> str:
    """Join a JSON path into ``a/0/b`` style, or a root label when empty."""
    joined = "/".join(str(part) for part in parts)
    return joined if joined else ROOT_PATH_LABEL


def _build_attribute(data: dict[str, Any]) -> Attribute:
    """Build one attribute from validated data.

    ``literature_sources`` is optional for an attribute: an attribute without the key has no
    sources of its own, the empty tuple, as one that gives an empty list.
    """
    sources = tuple(data["literature_sources"]) if "literature_sources" in data else ()
    return Attribute(
        attribute_id=data["attribute_id"],
        name=data["name"],
        description=data["description"],
        literature_sources=sources,
        markers=tuple(_build_marker(m) for m in data["markers"]),
    )


def _build_marker(data: dict[str, Any]) -> Marker:
    """Build one marker from validated data, with levels ordered 1 to 5."""
    sub_rubric = data["sub_rubric"]
    levels = tuple(
        Level(
            level=number,
            label=sub_rubric[str(number)]["label"],
            description=sub_rubric[str(number)]["description"],
        )
        for number in range(LEVEL_MIN, LEVEL_MAX + 1)
    )
    return Marker(
        marker_id=data["marker_id"],
        name=data["name"],
        description=data["description"],
        levels=levels,
    )


def _marker_name_violations(data: dict[str, Any]) -> list[str]:
    """Report marker names that repeat or that equal another marker's id.

    The rule is `copernicus.rubric.identifiers.marker_name_problems`, the one a `Rubric`
    applies when it is constructed. Parts that are not well-formed are skipped because the
    schema check reports them.
    """
    attributes = [
        (
            a_index,
            attribute.get("attribute_id"),
            [
                (m_index, marker.get("marker_id"), marker.get("name"))
                for m_index, marker in _dict_items(attribute.get("markers"))
            ],
        )
        for a_index, attribute in _dict_items(data.get("attributes"))
    ]
    return marker_name_problems(attributes)


def _identifier_violations(data: dict[str, Any]) -> list[str]:
    """Report attribute and marker ids that break the naming rule of rubrics.

    The ids are checked as a rubric checks its own when it is constructed, through
    `copernicus.rubric.identifiers.located_identifier_problems`: the attribute ids, then the
    marker ids in rubric order. Ids that are not non-empty strings are skipped because the
    schema check reports them.
    """
    attributes = _dict_items(data.get("attributes"))
    located = [
        (f"attributes/{a_index}/attribute_id", attribute.get("attribute_id"))
        for a_index, attribute in attributes
    ]
    located += [
        (f"attributes/{a_index}/markers/{m_index}/marker_id", marker.get("marker_id"))
        for a_index, attribute in attributes
        for m_index, marker in _dict_items(attribute.get("markers"))
    ]
    return located_identifier_problems([(path, value) for path, value in located if is_text(value)])


def _dict_items(value: Any) -> list[tuple[int, dict[str, Any]]]:
    """Return the (index, dict) pairs of a list, ignoring anything else."""
    if not isinstance(value, list):
        return []
    return [(index, item) for index, item in enumerate(value) if isinstance(item, dict)]
