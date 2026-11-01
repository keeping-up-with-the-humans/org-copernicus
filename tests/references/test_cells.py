"""Tests for the cells of reference-text generation."""

from __future__ import annotations

import pytest

from copernicus.references import Cell, build_cells
from copernicus.references.cells import ANCHOR_MAP, LEVELS, POLARITIES
from copernicus.rubric.model import Rubric

ORDER = [(level, polarity) for level in LEVELS for polarity in POLARITIES]


def test_the_anchor_map_has_the_documented_anchors_and_intensity_phrases() -> None:
    assert LEVELS == ("top", "mid", "low")
    assert POLARITIES == ("positive", "negative")
    assert ANCHOR_MAP == {
        ("top", "positive"): (5, "STRONGLY and unambiguously"),
        ("mid", "positive"): (4, "MODERATELY and recognizably"),
        ("low", "positive"): (3, "FAINTLY but discernibly (a mild, threshold-level)"),
        ("top", "negative"): (1, "STRONGLY and unambiguously"),
        ("mid", "negative"): (2, "MODERATELY and recognizably"),
        ("low", "negative"): (3, "FAINTLY but discernibly (a mild, threshold-level)"),
    }


def test_the_toy_rubric_has_six_cells_for_the_construct_and_six_per_attribute(
    toy_rubric: Rubric,
) -> None:
    cells = build_cells(toy_rubric)
    assert len(cells) == 6 + 6 * 3 == 24
    targets = ["construct"] + [attribute.attribute_id for attribute in toy_rubric.attributes]
    for position, target in enumerate(targets):
        block = cells[6 * position : 6 * position + 6]
        assert [cell.target for cell in block] == [target] * 6
        assert [(cell.level, cell.polarity) for cell in block] == ORDER


def test_each_cell_carries_its_anchor_level_and_intensity_phrase(toy_rubric: Rubric) -> None:
    for cell in build_cells(toy_rubric):
        anchor_level, intensity = ANCHOR_MAP[(cell.level, cell.polarity)]
        assert cell.anchor_level == anchor_level
        assert cell.intensity == intensity


def test_the_construct_guidance_has_one_block_per_attribute(toy_rubric: Rubric) -> None:
    construct_cell = build_cells(toy_rubric)[0]
    assert construct_cell.target == "construct"
    for attribute in toy_rubric.attributes:
        assert f"[{attribute.name}]\n" in construct_cell.pole_guidance
        for marker in attribute.markers:
            assert marker.name in construct_cell.pole_guidance
    blocks = construct_cell.pole_guidance.split("\n[")
    assert len(blocks) == len(toy_rubric.attributes)
    guidance_lines = [
        line for line in construct_cell.pole_guidance.splitlines() if not line.startswith("[")
    ]
    assert len(guidance_lines) == sum(len(a.markers) for a in toy_rubric.attributes)
    assert all(line.startswith("- ") for line in guidance_lines)


def test_the_attribute_guidance_lists_each_marker_with_its_level_description(
    toy_rubric: Rubric,
) -> None:
    cells = {(c.target, c.level, c.polarity): c for c in build_cells(toy_rubric)}
    attribute = toy_rubric.attributes[0]
    top_positive = cells[(attribute.attribute_id, "top", "positive")]
    expected = "\n".join(
        f"- {marker.name}: {marker.level(5).description}" for marker in attribute.markers
    )
    assert top_positive.pole_guidance == expected
    top_negative = cells[(attribute.attribute_id, "top", "negative")]
    assert attribute.markers[0].level(1).description in top_negative.pole_guidance
    low_negative = cells[(attribute.attribute_id, "low", "negative")]
    low_positive = cells[(attribute.attribute_id, "low", "positive")]
    assert low_negative.pole_guidance == low_positive.pole_guidance


def test_a_cell_is_immutable(toy_rubric: Rubric) -> None:
    cell = build_cells(toy_rubric)[0]
    assert isinstance(cell, Cell)
    with pytest.raises(AttributeError):
        cell.level = "mid"  # type: ignore[misc]


def test_the_sub_package_exports_the_public_names() -> None:
    import copernicus.references as references

    assert set(references.__all__) == {
        "PROMPT_VERSION",
        "Cell",
        "DeferredWriter",
        "GeneratorSettings",
        "ReferenceSet",
        "Shortfall",
        "TokenEstimate",
        "build_cells",
        "estimate_tokens",
        "generate_references",
        "leakage_terms",
        "plan_calls",
        "screen_report",
        "screen_texts",
        "system_prompt",
        "token_counter",
        "user_prompt",
    }
    for name in references.__all__:
        assert hasattr(references, name)
