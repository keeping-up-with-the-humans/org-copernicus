"""Golden prompts: the rendered prompts and their inputs pinned byte for byte.

The files under ``golden/`` hold the extraction, scoring, and one-stage prompts of the first
attribute of the toy rubric with the markers in rubric order and the wording ``role_baseline``,
``task_baseline``, and ``level_independent``, and ``one_stage_levels_5_to_1`` the one-stage
prompt of the same attribute when the rubric lists every marker's levels from 5 down to 1, as a
rubric built in code does by default: the one-stage prompt that version 0.1.0 printed for every
rubric. The
toy rubric's file lists the levels from 1 to 5, and the one-stage prompt lists them in the order
of the file. A change to a template, the wording, a section builder, or the Jinja settings shows
up here first. To accept an intended change, write ``render_golden(kind)`` to
``golden/<kind>.txt`` as UTF-8 without a final newline.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from level_order_support import FIVE_DOWN, SCATTERED, built_in_code, toy_data, toy_listed

from copernicus.coders import (
    PromptContext,
    load_wording_pool,
    render_extraction_prompt,
    render_one_stage_prompt,
    render_scoring_prompt,
    template_hashes,
)
from copernicus.coders.store_types import wording_hash
from copernicus.rubric import Rubric, load_rubric, toy_rubric_path

GOLDEN_DIRECTORY = Path(__file__).parent / "golden"
PROMPT_KINDS = ("extraction", "scoring", "one_stage", "one_stage_levels_5_to_1")
TWO_STAGE_KINDS = ("extraction", "scoring")
PINNED_WORDING = {"role": "role_baseline", "task": "task_baseline", "level": "level_independent"}
TEMPLATE_HASHES = {
    "extraction": "aeb5603655859c0b",
    "scoring": "2a845247cefb784c",
    "one_stage": "33ae2dfe54c81258",
}
WORDING_HASH = "f22e9149bb718907"
VARIANT_ORDER = {
    "role": ["role_baseline", "role_calibrated", "role_analytical"],
    "task": ["task_baseline", "task_calibrated", "task_analytical"],
    "level": ["level_topdown", "level_midpoint", "level_independent", "level_bottomup"],
}


def golden_context(rubric: Rubric | None = None) -> PromptContext:
    """Return the prompt context of the golden prompts, of the toy rubric unless one is given."""
    rubric = rubric if rubric is not None else load_rubric(toy_rubric_path())
    attribute = rubric.attributes[0]
    pool = load_wording_pool()
    return PromptContext(
        rubric,
        attribute,
        [marker.name for marker in attribute.markers],
        pool.get("role", PINNED_WORDING["role"]),
        pool.get("task", PINNED_WORDING["task"]),
        pool.get("level", PINNED_WORDING["level"]),
    )


def render_golden(kind: str, rubric: Rubric | None = None) -> str:
    """Render one golden prompt, of the toy rubric unless one is given; scoring rates every marker.

    ``one_stage_levels_5_to_1`` renders the one-stage prompt of the toy rubric as read from a
    file that lists every marker's levels from 5 down to 1.
    """
    if kind == "one_stage_levels_5_to_1":
        return render_one_stage_prompt(golden_context(toy_listed(FIVE_DOWN)))
    context = golden_context(rubric)
    if kind == "extraction":
        return render_extraction_prompt(context)
    if kind == "scoring":
        return render_scoring_prompt(context, list(context.marker_order))
    return render_one_stage_prompt(context)


@pytest.mark.parametrize("kind", PROMPT_KINDS)
def test_rendered_prompt_matches_its_golden_file_byte_for_byte(kind: str) -> None:
    expected = (GOLDEN_DIRECTORY / f"{kind}.txt").read_bytes()
    assert render_golden(kind).encode("utf-8") == expected


@pytest.mark.parametrize("kind", PROMPT_KINDS)
def test_prompts_end_without_a_trailing_newline(kind: str) -> None:
    assert not render_golden(kind).endswith("\n")


def listed_levels(prompt: str, name: str) -> dict[int, str]:
    """Read the sub-rubric mapping of one marker back from a one-stage prompt."""
    line = next(line for line in prompt.splitlines() if line.startswith(f"{name}: {{"))
    levels: dict[int, str] = ast.literal_eval(line.removeprefix(f"{name}: "))
    return levels


def test_one_stage_subrubrics_list_the_levels_in_the_order_of_the_rubric_file() -> None:
    prompt = render_golden("one_stage")
    written = toy_data()["attributes"][0]["markers"]
    for marker, data in zip(golden_context().attribute.markers, written, strict=True):
        levels = listed_levels(prompt, marker.name)
        assert list(levels) == [int(key) for key in data["sub_rubric"]] == [1, 2, 3, 4, 5]
        assert levels == {level.level: level.description for level in marker.levels}


def test_one_stage_subrubrics_of_a_file_listed_from_five_down_run_from_five_down() -> None:
    prompt = render_golden("one_stage_levels_5_to_1")
    for marker in golden_context().attribute.markers:
        levels = listed_levels(prompt, marker.name)
        assert list(levels) == [5, 4, 3, 2, 1]
        assert levels == {level.level: level.description for level in marker.levels}


def test_a_rubric_built_in_code_renders_the_one_stage_prompt_of_0_1_0() -> None:
    expected = (GOLDEN_DIRECTORY / "one_stage_levels_5_to_1.txt").read_bytes()
    toy = load_rubric(toy_rubric_path())
    assert render_golden("one_stage", built_in_code(toy)).encode("utf-8") == expected


@pytest.mark.parametrize("kind", TWO_STAGE_KINDS)
def test_the_two_stage_prompts_do_not_depend_on_the_order_of_the_levels(kind: str) -> None:
    expected = (GOLDEN_DIRECTORY / f"{kind}.txt").read_bytes()
    toy = load_rubric(toy_rubric_path())
    for rubric in (toy_listed(FIVE_DOWN), toy_listed(SCATTERED), built_in_code(toy)):
        assert render_golden(kind, rubric).encode("utf-8") == expected


def test_template_hashes_are_pinned() -> None:
    assert template_hashes() == TEMPLATE_HASHES


def test_wording_hash_of_the_shipped_pool_is_pinned() -> None:
    assert wording_hash(load_wording_pool()) == WORDING_HASH


def test_variant_order_of_each_dimension_is_pinned() -> None:
    pool = load_wording_pool()
    order = {name: [w.variant_id for w in getattr(pool, name)] for name in VARIANT_ORDER}
    assert order == VARIANT_ORDER
