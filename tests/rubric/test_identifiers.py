"""Tests for the id rule that rubric validation and the run store share, and for marker names."""

from __future__ import annotations

import ast
import dataclasses
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from store_support import unvalidated_rubric

import copernicus.corrections.estimates_ols as estimates_ols
import copernicus.hybrid.coder as hybrid_coder
import copernicus.references.cells as cells
import copernicus.rubric.identifiers as identifiers
import copernicus.similarity.scorer as scorer
import copernicus.validity.panel_inputs as panel_inputs
from copernicus.coders.store_schema import FIXED_RESULT_COLUMNS, validate_identifiers
from copernicus.errors import RubricValidationError, StoreError
from copernicus.rubric import load_rubric, rubric_from_dict, toy_rubric_path, validate_rubric_dict
from copernicus.rubric.identifiers import (
    EXPORT_COLUMN_NAMES,
    RESERVED_NAMES,
    identifier_problems,
    marker_name_problems,
)
from copernicus.rubric.model import Attribute, Rubric

CONSTRUCT_TARGET = "is the name of the whole-construct target"
ROLLUP_TARGET = "ends in '__rollup', the suffix of the roll-up target"
# (attribute index, marker index or None for the attribute id, new id, JSON path, message)
BAD_IDS = [
    (0, None, "bad-id", "attributes/0/attribute_id", "cannot be a column name"),
    (1, 2, "1st_marker", "attributes/1/markers/2/marker_id", "cannot be a column name"),
    (2, None, "seed", "attributes/2/attribute_id", "collides with a reserved column name"),
    (0, 1, "Text_ID", "attributes/0/markers/1/marker_id", "collides with a reserved column"),
    (1, 0, "Text_Hash", "attributes/1/markers/0/marker_id", "collides with a reserved column"),
    (1, None, "SPECIFICITY", "attributes/1/attribute_id", "collides with another attribute"),
    (2, 0, "tone", "attributes/2/markers/0/marker_id", "collides with another attribute"),
    (0, None, "construct", "attributes/0/attribute_id", CONSTRUCT_TARGET),
    (2, 1, "Construct", "attributes/2/markers/1/marker_id", CONSTRUCT_TARGET),
    (1, None, "constructive_feedback_v1__rollup", "attributes/1/attribute_id", ROLLUP_TARGET),
    (0, 2, "spec__ROLLUP", "attributes/0/markers/2/marker_id", ROLLUP_TARGET),
]
# The modules that use a generated target name, with the names each takes from the rule.
TARGET_NAME_USERS: dict[ModuleType, tuple[str, ...]] = {
    cells: ("TARGET_CONSTRUCT",),
    scorer: ("ROLLUP_SUFFIX", "TARGET_CONSTRUCT"),
    panel_inputs: ("ROLLUP_SUFFIX", "TARGET_CONSTRUCT"),
    hybrid_coder: ("ROLLUP_SUFFIX",),
    estimates_ols: ("TARGET_CONSTRUCT",),
}


def toy_data() -> dict[str, Any]:
    return json.loads(toy_rubric_path().read_text(encoding="utf-8"))


def with_id(
    rubric: Rubric, attribute_index: int, marker_index: int | None, identifier: str
) -> tuple[Attribute, ...]:
    """Return the attributes of the rubric with one id replaced (attributes check no ids)."""
    attributes = list(rubric.attributes)
    attribute = attributes[attribute_index]
    if marker_index is None:
        attributes[attribute_index] = dataclasses.replace(attribute, attribute_id=identifier)
    else:
        markers = list(attribute.markers)
        markers[marker_index] = dataclasses.replace(markers[marker_index], marker_id=identifier)
        attributes[attribute_index] = dataclasses.replace(attribute, markers=tuple(markers))
    return tuple(attributes)


@pytest.mark.parametrize(("attribute", "marker", "identifier", "path", "message"), BAD_IDS)
def test_validation_the_rubric_and_the_store_refuse_the_same_ids(
    attribute: int, marker: int | None, identifier: str, path: str, message: str
) -> None:
    """The loader, the rubric's constructor and the store apply one rule with one message.

    A bad id can no longer reach the store through a rubric, so the store's own check is
    shown on a stand-in made without the constructor.
    """
    data = toy_data()
    target = data["attributes"][attribute]
    (target if marker is None else target["markers"][marker])[
        "attribute_id" if marker is None else "marker_id"
    ] = identifier
    with pytest.raises(RubricValidationError) as raised:
        validate_rubric_dict(data)
    lines = [line for line in str(raised.value).splitlines() if line.startswith(f"{path}: ")]
    assert len(lines) == 1 and lines[0].startswith(f"{path}: id '{identifier}' {message}")
    toy = load_rubric(toy_rubric_path())
    attributes = with_id(toy, attribute, marker, identifier)
    with pytest.raises(RubricValidationError) as built:
        dataclasses.replace(toy, attributes=attributes)
    assert str(built.value) == lines[0]
    with pytest.raises(StoreError, match=message):
        validate_identifiers(unvalidated_rubric(toy, attributes=attributes))


def test_the_toy_rubric_passes_the_rule() -> None:
    rubric = load_rubric(toy_rubric_path())
    ids = [a.attribute_id for a in rubric.attributes]
    ids += [m.marker_id for a in rubric.attributes for m in a.markers]
    assert identifier_problems(ids) == []
    validate_identifiers(rubric)


def test_every_problem_is_reported_once_in_input_order() -> None:
    problems = identifier_problems(["ok", "OK", "n_runs", "x y", "ok"])
    assert [position for position, _ in problems] == [1, 2, 3, 4]
    assert "('ok')" in problems[0][1] and "reserved" in problems[1][1]
    assert "column name" in problems[2][1] and "('ok')" in problems[3][1]


def test_reserved_names_cover_the_store_and_export_columns() -> None:
    fixed = {name.lower() for name, _ in FIXED_RESULT_COLUMNS}
    assert RESERVED_NAMES == fixed | set(EXPORT_COLUMN_NAMES)


def test_marker_name_equal_to_another_markers_id_is_refused() -> None:
    data = toy_data()
    markers = data["attributes"][0]["markers"]
    markers[0]["name"] = markers[1]["marker_id"]
    with pytest.raises(RubricValidationError) as raised:
        validate_rubric_dict(data)
    assert str(raised.value) == (
        "attributes/0/markers/0/name: marker name 'spec_location' equals the id of another "
        "marker of attribute 'specificity'; an extraction key could name either marker"
    )


def test_marker_name_equal_to_its_own_id_or_to_a_marker_id_elsewhere_is_accepted() -> None:
    data = toy_data()
    first, second = data["attributes"][0], data["attributes"][1]
    first["markers"][0]["name"] = first["markers"][0]["marker_id"]
    second["markers"][0]["name"] = first["markers"][1]["marker_id"]
    rubric = rubric_from_dict(data)
    assert rubric.attributes[0].markers[0].name == "spec_examples"


@pytest.mark.parametrize(
    ("identifier", "message"),
    [
        ("construct", "id 'construct' is the name of the whole-construct target"),
        (
            "constructive_feedback_v1__rollup",
            "id 'constructive_feedback_v1__rollup' ends in '__rollup', the suffix of the "
            "roll-up target",
        ),
    ],
)
def test_an_attribute_named_like_a_generated_target_is_refused_by_the_loader(
    identifier: str, message: str
) -> None:
    data = toy_data()
    data["attributes"][0]["attribute_id"] = identifier
    with pytest.raises(RubricValidationError) as raised:
        rubric_from_dict(data)
    assert str(raised.value) == f"attributes/0/attribute_id: {message}"


def test_ids_that_merely_contain_a_target_name_are_accepted() -> None:
    names = ["construct_clarity", "constructs", "my_construct", "rollup", "x_rollup", "a__roll"]
    assert identifier_problems(names) == []


def test_the_generated_target_names_have_one_source() -> None:
    assert identifiers.TARGET_CONSTRUCT == "construct"
    assert identifiers.ROLLUP_SUFFIX == "__rollup"
    assert identifiers.RESERVED_TARGET_NAMES == frozenset({identifiers.TARGET_CONSTRUCT})
    assert cells.TARGET_CONSTRUCT is identifiers.TARGET_CONSTRUCT
    assert scorer.TARGET_CONSTRUCT is identifiers.TARGET_CONSTRUCT
    assert scorer.ROLLUP_SUFFIX is identifiers.ROLLUP_SUFFIX
    assert panel_inputs.ROLLUP_SUFFIX is identifiers.ROLLUP_SUFFIX


def module_assignments(tree: ast.Module) -> tuple[set[str], set[object]]:
    """Return the names a module assigns at its top level, and the constants it assigns them.

    Plain assignments (``NAME = value``) and annotated ones (``NAME: str = value``) both count.
    """
    names: set[str] = set()
    literals: set[object] = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign):
            targets, value = [node.target], node.value
        else:
            continue
        names |= {target.id for target in targets if isinstance(target, ast.Name)}
        if isinstance(value, ast.Constant):
            literals.add(value.value)
    return names, literals


def test_the_source_check_sees_plain_and_annotated_copies() -> None:
    for source in ('TARGET_CONSTRUCT = "construct"', 'TARGET_CONSTRUCT: str = "construct"'):
        names, literals = module_assignments(ast.parse(source))
        assert names == {"TARGET_CONSTRUCT"} and literals == {"construct"}


@pytest.mark.parametrize("module", list(TARGET_NAME_USERS), ids=lambda module: module.__name__)
def test_every_user_of_a_target_name_imports_it_from_the_rule(module: ModuleType) -> None:
    """A string is the same object wherever it is interned, so the source is read instead."""
    assert module.__file__ is not None
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    imported = {
        alias.name
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module == identifiers.__name__
        for alias in node.names
    }
    assigned, literals = module_assignments(tree)
    assert set(TARGET_NAME_USERS[module]) <= imported
    assert not assigned & {"TARGET_CONSTRUCT", "ROLLUP_SUFFIX", "CONSTRUCT"}
    assert not literals & {identifiers.TARGET_CONSTRUCT, identifiers.ROLLUP_SUFFIX}


def test_a_rubric_built_in_code_obeys_the_rule_from_its_construction() -> None:
    toy = load_rubric(toy_rubric_path())
    first, *rest = toy.attributes
    renamed = (dataclasses.replace(first, attribute_id="construct"), *rest)
    with pytest.raises(RubricValidationError) as raised:
        dataclasses.replace(toy, attributes=renamed)
    assert str(raised.value) == (
        "attributes/0/attribute_id: id 'construct' is the name of the whole-construct target"
    )
    with pytest.raises(RubricValidationError, match="attributes/0/attribute_id: id 'x__rollup'"):
        Rubric(
            toy.rubric_id,
            toy.concept,
            toy.concept_definition,
            (),
            (dataclasses.replace(first, attribute_id="x__rollup"),),
            {},
        )
    with pytest.raises(RubricValidationError, match="attributes/0/attribute_id: id 5 cannot be"):
        dataclasses.replace(toy, attributes=(dataclasses.replace(first, attribute_id=5), *rest))
    edited = dataclasses.replace(toy, concept_definition="An edited definition.")
    assert edited.attributes == toy.attributes and edited.content_hash() != toy.content_hash()


def with_marker_name(
    rubric: Rubric, attribute_index: int, marker_index: int, name: str
) -> tuple[Attribute, ...]:
    """Return the attributes of the rubric with one marker renamed (attributes check nothing)."""
    attributes = list(rubric.attributes)
    attribute = attributes[attribute_index]
    markers = list(attribute.markers)
    markers[marker_index] = dataclasses.replace(markers[marker_index], name=name)
    attributes[attribute_index] = dataclasses.replace(attribute, markers=tuple(markers))
    return tuple(attributes)


@pytest.mark.parametrize(
    ("marker", "source", "field"),
    [(1, 0, "name"), (0, 1, "marker_id")],
    ids=["two markers named alike", "a name equal to the id of another marker"],
)
def test_a_rubric_built_in_code_obeys_the_marker_name_rule_with_the_loaders_message(
    marker: int, source: int, field: str
) -> None:
    data = toy_data()
    markers = data["attributes"][0]["markers"]
    markers[marker]["name"] = markers[source][field]
    with pytest.raises(RubricValidationError) as loaded:
        validate_rubric_dict(data)
    assert str(loaded.value).startswith(f"attributes/0/markers/{marker}/name: ")
    toy = load_rubric(toy_rubric_path())
    attributes = with_marker_name(toy, 0, marker, markers[source][field])
    with pytest.raises(RubricValidationError) as built:
        dataclasses.replace(toy, attributes=attributes)
    assert str(built.value) == str(loaded.value)


def test_marker_names_are_checked_before_the_ids_by_the_loader_and_the_rubric_alike() -> None:
    data = toy_data()
    data["attributes"][1]["markers"][2]["name"] = data["attributes"][1]["markers"][0]["name"]
    data["attributes"][2]["attribute_id"] = "seed"
    with pytest.raises(RubricValidationError) as loaded:
        validate_rubric_dict(data)
    toy = load_rubric(toy_rubric_path())
    renamed = with_marker_name(toy, 1, 2, toy.attributes[1].markers[0].name)
    attributes = (*renamed[:2], dataclasses.replace(renamed[2], attribute_id="seed"))
    with pytest.raises(RubricValidationError) as built:
        dataclasses.replace(toy, attributes=attributes)
    assert str(built.value) == str(loaded.value)
    assert [line.split(": ")[0] for line in str(built.value).splitlines()] == [
        "attributes/1/markers/2/name",
        "attributes/2/attribute_id",
    ]


def test_the_marker_name_rule_skips_names_that_are_no_text_and_names_an_unnamed_owner() -> None:
    unnamed = [(0, "m0", ""), (1, "m1", ""), (2, "m2", None), (3, "m3", 7)]
    assert marker_name_problems([(0, "attribute", unnamed)]) == []
    clash = [(0, "m0", "m1"), (1, "m1", "A name")]
    assert marker_name_problems([(4, None, clash)]) == [
        "attributes/4/markers/0/name: marker name 'm1' equals the id of another marker of "
        "attribute at index 4; an extraction key could name either marker"
    ]


def test_every_problem_of_a_rubric_built_in_code_is_named_on_its_own_line() -> None:
    toy = load_rubric(toy_rubric_path())
    first, second, third = toy.attributes
    markers = (dataclasses.replace(second.markers[0], marker_id="Construct"), *second.markers[1:])
    attributes = (
        dataclasses.replace(first, attribute_id="seed"),
        dataclasses.replace(second, markers=markers),
        dataclasses.replace(third, attribute_id="TONE"),
    )
    with pytest.raises(RubricValidationError) as raised:
        dataclasses.replace(toy, attributes=attributes)
    assert str(raised.value).splitlines() == [
        "attributes/0/attribute_id: id 'seed' collides with a reserved column name",
        "attributes/2/attribute_id: id 'TONE' collides with another attribute or marker id "
        "('tone'); ids are compared without regard to case",
        "attributes/1/markers/0/marker_id: id 'Construct' is the name of the whole-construct "
        "target",
    ]
