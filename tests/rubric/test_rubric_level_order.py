"""A marker keeps the order in which its rubric file lists its levels; the content hash does not.

The hashes pinned here are the ones version 0.1.0 computed for these rubrics, which kept no
level order at all. A marker built in code may hold levels that the model's types do not admit,
such as a level numbered by the text "1" next to the level 1; version 0.1.0 accepted such a
marker, and its rubric keeps the layout and hash of that version.
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from level_order_support import (
    FIVE_DOWN,
    ONE_UP,
    SCATTERED,
    built_in_code,
    clarity_data,
    toy_data,
    toy_file,
    toy_listed,
    write_rubric,
)

from copernicus.errors import RubricValidationError
from copernicus.rubric import Attribute, Level, Marker, Rubric, load_rubric, rubric_from_dict

TOY_HASH = "4bdee2efb58b0003"
CLARITY_HASH = "0e5c01dadc777a24"
CLARITY_THREE_MARKERS_HASH = "8bbf07e84351dcf0"
UNUSUAL_LEVELS_HASH = "484a12ecccec1f54"
LEVEL_NAMED_TWICE_HASH = "edd4b3a5b1df907a"


def test_a_marker_keeps_the_level_order_of_its_rubric_file(tmp_path: Path) -> None:
    path = write_rubric(tmp_path, clarity_data(ONE_UP, FIVE_DOWN, SCATTERED), "clarity")
    markers = load_rubric(path).attributes[0].markers
    assert [marker.level_order for marker in markers] == [ONE_UP, FIVE_DOWN, SCATTERED]
    for marker in markers:
        assert [level.level for level in marker.levels] == [1, 2, 3, 4, 5]
        assert [marker.level(n).level for n in ONE_UP] == [1, 2, 3, 4, 5]
    assert {marker.level_order for a in toy_file().attributes for marker in a.markers} == {ONE_UP}


def test_a_marker_built_in_code_lists_its_levels_from_five_down_to_one() -> None:
    levels = tuple(Level(n, f"Level {n}", f"Text of level {n}.") for n in ONE_UP)
    assert Marker("m1", "Marker", "A marker.", levels).level_order == FIVE_DOWN
    assert Marker("m1", "Marker", "A marker.", levels, ONE_UP).level_order == ONE_UP
    rubric = built_in_code(toy_file())
    assert {m.level_order for a in rubric.attributes for m in a.markers} == {FIVE_DOWN}


@pytest.mark.parametrize(
    "order",
    [
        (1, 2, 3, 4),
        (1, 2, 3, 4, 5, 5),
        (1, 1, 2, 3, 4),
        (0, 1, 2, 3, 4),
        (2, 3, 4, 5, 6),
        [1, 2, 3, 4, 5],
        (True, 2, 3, 4, 5),
        ("1", "2", "3", "4", "5"),
        (1.0, 2, 3, 4, 5),
    ],
)
def test_a_level_order_that_is_no_order_of_the_five_levels_is_refused(order: object) -> None:
    marker = toy_file().attributes[0].markers[0]
    with pytest.raises(RubricValidationError) as refused:
        dataclasses.replace(marker, level_order=order)
    message = str(refused.value)
    assert "marker 'spec_examples'" in message and repr(order) in message
    assert "lists each of the levels 1 to 5 once" in message


def test_the_content_hash_is_the_hash_of_0_1_0_whatever_the_level_order(tmp_path: Path) -> None:
    assert toy_file().content_hash() == TOY_HASH
    assert toy_listed(FIVE_DOWN).content_hash() == TOY_HASH
    assert toy_listed(SCATTERED).content_hash() == TOY_HASH
    assert built_in_code(toy_file()).content_hash() == TOY_HASH
    for orders in ((ONE_UP, ONE_UP), (FIVE_DOWN, FIVE_DOWN), (ONE_UP, SCATTERED)):
        path = write_rubric(tmp_path, clarity_data(*orders), "clarity")
        assert load_rubric(path).content_hash() == CLARITY_HASH
    three = clarity_data(ONE_UP, FIVE_DOWN, SCATTERED, rubric_id="clarity_mixed_v1")
    assert load_rubric(write_rubric(tmp_path, three, "three")).content_hash() == (
        CLARITY_THREE_MARKERS_HASH
    )


def test_to_dict_writes_each_sub_rubric_in_its_level_order_and_reads_back_equal() -> None:
    rubric = rubric_from_dict(clarity_data(ONE_UP, FIVE_DOWN, SCATTERED))
    written = rubric.to_dict()
    keys = [list(marker["sub_rubric"]) for marker in written["attributes"][0]["markers"]]
    assert keys == [["1", "2", "3", "4", "5"], ["5", "4", "3", "2", "1"], ["3", "1", "5", "2", "4"]]
    assert written == clarity_data(ONE_UP, FIVE_DOWN, SCATTERED)
    for each in (rubric, toy_file(), toy_listed(FIVE_DOWN), built_in_code(toy_file())):
        assert rubric_from_dict(each.to_dict()) == each
    in_code = built_in_code(toy_file()).to_dict()
    assert list(in_code["attributes"][0]["markers"][0]["sub_rubric"]) == ["5", "4", "3", "2", "1"]
    assert toy_file().to_dict() == toy_data()


def with_first_marker_levels(rubric: Rubric, levels: tuple[Level, ...]) -> Rubric:
    """Return the rubric with its first attribute cut to its first marker, holding ``levels``."""
    first = rubric.attributes[0]
    marker = dataclasses.replace(first.markers[0], levels=levels)
    attribute = dataclasses.replace(first, markers=(marker,))
    return dataclasses.replace(rubric, attributes=(attribute, *rubric.attributes[1:]))


def test_a_marker_built_in_code_with_other_levels_keeps_every_level_and_its_hash() -> None:
    toy = toy_file()
    held = toy.attributes[0].markers[0].levels
    beyond = Level(6, "Beyond", "Outside the five levels.")
    rubric = with_first_marker_levels(toy, (held[1], held[0], beyond))
    written = rubric.to_dict()["attributes"][0]["markers"][0]["sub_rubric"]
    assert list(written) == ["1", "2", "6"]
    assert written["6"] == {"label": "Beyond", "description": "Outside the five levels."}
    assert rubric.content_hash() == UNUSUAL_LEVELS_HASH


def test_a_level_held_as_text_and_as_number_keeps_the_layout_and_hash_of_0_1_0() -> None:
    levels = (
        Level("1", "string", "string first"),  # type: ignore[arg-type]
        Level(1, "integer", "integer last"),
    )
    marker = Marker("m00", "Marker 00", "Marker description 00.", levels)
    attribute = Attribute("a0", "Facet 0", "Facet description 0.", (), (marker,))
    definitions = {"part": "A part of the text."}
    rubric = Rubric("review", "Clarity", "A clear explanation.", (), (attribute,), definitions)
    written = rubric.to_dict()["attributes"][0]["markers"][0]["sub_rubric"]
    assert written == {"1": {"label": "integer", "description": "integer last"}}
    assert rubric.content_hash() == LEVEL_NAMED_TWICE_HASH


def drop_level(levels: dict[str, Any]) -> None:
    del levels["3"]


def add_level(levels: dict[str, Any]) -> None:
    levels["6"] = levels["5"]


def rename_level(levels: dict[str, Any]) -> None:
    levels["01"] = levels.pop("1")


@pytest.mark.parametrize(
    "change,message",
    [
        (drop_level, "'3' is a required property"),
        (add_level, "('6' was unexpected)"),
        (rename_level, "('01' was unexpected)"),
    ],
)
def test_the_loader_refuses_a_missing_level_and_any_other_key_as_before(
    change: Callable[[dict[str, Any]], None], message: str
) -> None:
    data = clarity_data(SCATTERED)
    change(data["attributes"][0]["markers"][0]["sub_rubric"])
    with pytest.raises(RubricValidationError, match=re.escape(message)):
        rubric_from_dict(data)
