"""Golden prompts: the rendered prompts and their inputs pinned byte for byte.

The three files under ``golden/`` hold the extraction, scoring, and one-stage prompts of the
first attribute of the toy rubric with the markers in rubric order and the wording
``role_baseline``, ``task_baseline``, and ``level_independent``. A change to a template, the
wording, a section builder, or the Jinja settings shows up here first. To accept an intended
change, write ``render_golden(kind)`` to ``golden/<kind>.txt`` as UTF-8 without a final newline.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from copernicus.coders import (
    PromptContext,
    load_wording_pool,
    render_extraction_prompt,
    render_one_stage_prompt,
    render_scoring_prompt,
    template_hashes,
)
from copernicus.coders.store_types import wording_hash
from copernicus.rubric import load_rubric, toy_rubric_path

GOLDEN_DIRECTORY = Path(__file__).parent / "golden"
PROMPT_KINDS = ("extraction", "scoring", "one_stage")
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


def golden_context() -> PromptContext:
    """Return the prompt context of the golden prompts."""
    rubric = load_rubric(toy_rubric_path())
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


def render_golden(kind: str) -> str:
    """Render one golden prompt; the scoring prompt rates every marker."""
    context = golden_context()
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


def test_one_stage_subrubrics_list_the_levels_from_five_down_to_one() -> None:
    prompt = render_golden("one_stage")
    for marker in golden_context().attribute.markers:
        line = next(line for line in prompt.splitlines() if line.startswith(f"{marker.name}: {{"))
        levels = ast.literal_eval(line.removeprefix(f"{marker.name}: "))
        assert list(levels) == [5, 4, 3, 2, 1]
        assert levels == {level.level: level.description for level in marker.levels}


def test_template_hashes_are_pinned() -> None:
    assert template_hashes() == TEMPLATE_HASHES


def test_wording_hash_of_the_shipped_pool_is_pinned() -> None:
    assert wording_hash(load_wording_pool()) == WORDING_HASH


def test_variant_order_of_each_dimension_is_pinned() -> None:
    pool = load_wording_pool()
    order = {name: [w.variant_id for w in getattr(pool, name)] for name in VARIANT_ORDER}
    assert order == VARIANT_ORDER
