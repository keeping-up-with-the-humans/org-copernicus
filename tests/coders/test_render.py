import re

import pytest

from copernicus.coders import (
    PromptContext,
    definitions_section,
    extraction_output_template,
    level_descriptions_section,
    load_wording_pool,
    markers_section,
    render_extraction_prompt,
    render_one_stage_prompt,
    render_scoring_prompt,
    scoring_rubric_text,
    scoring_user_message,
    subrubrics_section,
    template_hashes,
    user_message_for_text,
)
from copernicus.errors import PromptContractError
from copernicus.rubric.model import Rubric


def _context(rubric: Rubric, attribute_index: int = 0, order: list[str] | None = None):
    pool = load_wording_pool()
    attribute = rubric.attributes[attribute_index]
    names = order if order is not None else [m.name for m in attribute.markers]
    return PromptContext(rubric, attribute, names, pool.role[0], pool.task[0], pool.level[0])


def test_every_wording_combination_renders_for_the_toy_rubric(toy_rubric: Rubric) -> None:
    pool = load_wording_pool()
    for role in pool.role:
        for task in pool.task:
            for level in pool.level:
                for attr in toy_rubric.attributes:
                    names = [m.name for m in attr.markers]
                    ctx = PromptContext(toy_rubric, attr, names, role, task, level)
                    s1 = render_extraction_prompt(ctx)
                    s2 = render_scoring_prompt(ctx, [attr.markers[0].name])
                    s3 = render_one_stage_prompt(ctx)
                    joined = s1 + s2 + s3
                    assert "{{" not in joined and "{concept_name}" not in joined
                    assert "EXTRACTION ONLY" in s1 and "Level 5 [" in s1
                    assert "Level determination" not in s1
                    assert "You are scoring the attribute" in s2
                    assert level.texts["scoring"] in s2
                    assert "single integrated evaluator" in s3
                    assert level.texts["one_stage"] in s3


def test_role_and_task_texts_follow_the_prompt_kind(toy_rubric: Rubric) -> None:
    ctx = _context(toy_rubric)
    attr = ctx.attribute
    extraction = render_extraction_prompt(ctx)
    scoring = render_scoring_prompt(ctx, [attr.markers[0].name])
    one_stage = render_one_stage_prompt(ctx)
    task_extraction = ctx.task.texts["extraction"].replace("{concept_name}", toy_rubric.concept)
    task_scoring = ctx.task.texts["scoring"].replace("{concept_name}", toy_rubric.concept)
    task_one_stage = ctx.task.texts["one_stage"].replace("{concept_name}", toy_rubric.concept)
    assert ctx.role.texts["extraction"] in extraction and task_extraction in extraction
    assert ctx.role.texts["scoring"] in scoring and task_scoring in scoring
    assert ctx.role.texts["one_stage"] in one_stage and task_one_stage in one_stage


def test_concept_name_placeholder_is_replaced_literally(toy_rubric: Rubric) -> None:
    pool = load_wording_pool()
    task = pool.task[0]
    marked = type(task)(task.variant_id, True, {k: "About {concept_name}." for k in task.texts})
    attr = toy_rubric.attributes[0]
    ctx = PromptContext(
        toy_rubric, attr, [m.name for m in attr.markers], pool.role[0], marked, pool.level[0]
    )
    assert f"About {toy_rubric.concept}." in render_extraction_prompt(ctx)


def test_marker_order_moves_markers_levels_and_output_template_together(
    toy_rubric: Rubric,
) -> None:
    attr = toy_rubric.attributes[0]
    first, last = attr.markers[0].name, attr.markers[-1].name
    forward = [m.name for m in attr.markers]
    backward = list(reversed(forward))
    results = {}
    for label, order in (("forward", forward), ("backward", backward)):
        text = render_extraction_prompt(_context(toy_rubric, 0, order))
        markers_part = text.split("1. Markers:")[1].split("2. Marker Definitions:")[0]
        levels_part = text.split("Level Descriptions")[1].split("EXTRACTION PROCESS:")[0]
        template_part = text.split("OUTPUT FORMAT:")[1]
        results[label] = tuple(
            part.index(first) < part.index(last)
            for part in (markers_part, levels_part, template_part)
        )
    assert results["forward"] == (True, True, True)
    assert results["backward"] == (False, False, False)


def test_scoring_rubric_text_restricted_to_markers_to_rate(toy_rubric: Rubric) -> None:
    attr = toy_rubric.attributes[0]
    txt = scoring_rubric_text(attr, [attr.markers[0].name])
    assert txt.startswith("Level 1: ")
    assert attr.markers[1].level(5).description not in txt
    assert txt.splitlines()[4] == f"Level 5: {attr.markers[0].level(5).description.strip()}"


def test_scoring_rubric_text_uses_rubric_order_not_argument_order(toy_rubric: Rubric) -> None:
    attr = toy_rubric.attributes[0]
    names = [attr.markers[2].name, attr.markers[0].name]
    expected = f"{attr.markers[0].level(3).description} {attr.markers[2].level(3).description}"
    assert scoring_rubric_text(attr, names).splitlines()[2] == f"Level 3: {expected}"


def test_scoring_rubric_text_rejects_empty_and_unknown(toy_rubric: Rubric) -> None:
    attr = toy_rubric.attributes[0]
    with pytest.raises(PromptContractError):
        scoring_rubric_text(attr, [])
    with pytest.raises(PromptContractError, match="No such marker"):
        scoring_rubric_text(attr, ["No such marker"])


def test_scoring_rubric_text_rejects_an_empty_level(toy_rubric: Rubric) -> None:
    attr = toy_rubric.attributes[0]
    marker = attr.markers[0]
    blank_levels = tuple(
        type(level)(level.level, level.label, "  " if level.level == 4 else level.description)
        for level in marker.levels
    )
    blank_marker = type(marker)(marker.marker_id, marker.name, marker.description, blank_levels)
    blank_attr = type(attr)(
        attr.attribute_id, attr.name, attr.description, attr.literature_sources, (blank_marker,)
    )
    with pytest.raises(PromptContractError, match="level 4"):
        scoring_rubric_text(blank_attr, [marker.name])


def test_scoring_user_message_unlabelled_and_labelled() -> None:
    msg = scoring_user_message("Level 1: x", ["p1", "p2"])
    assert msg.count('- "') == 2 and "Marker:" not in msg
    assert msg == (
        "The scoring rubric:\nLevel 1: x\n\n"
        "The extracted evidence (verbatim passages from the text, in the order they occur):\n"
        '- "p1"\n- "p2"'
    )
    lab = scoring_user_message(
        "Level 1: x", ["p1"], mode="labelled", by_marker={"m": ["p1"]}, markers_to_rate=["m"]
    )
    assert "Marker: m" in lab
    assert lab == (
        "The scoring rubric:\nLevel 1: x\n\n"
        "The extracted evidence (verbatim passages from the text, grouped by Marker):\n"
        'Marker: m\n- "p1"'
    )


def test_scoring_user_message_contract_violations() -> None:
    with pytest.raises(PromptContractError):
        scoring_user_message("r", [])
    with pytest.raises(PromptContractError):
        scoring_user_message("r", ["p"], mode="sorted")  # type: ignore[arg-type]
    with pytest.raises(PromptContractError):
        scoring_user_message("r", ["p"], mode="labelled", markers_to_rate=["m"])
    with pytest.raises(PromptContractError):
        scoring_user_message("r", ["p"], mode="labelled", by_marker={"m": ["p"]})
    with pytest.raises(PromptContractError, match="other"):
        scoring_user_message(
            "r", ["p"], mode="labelled", by_marker={"m": ["p"]}, markers_to_rate=["other"]
        )


def test_user_message_for_text() -> None:
    assert user_message_for_text("Hello") == "Text to analyze:\nHello"


def test_section_builders_follow_the_order(toy_rubric: Rubric) -> None:
    attr = toy_rubric.attributes[0]
    order = [attr.markers[1].name, attr.markers[2].name, attr.markers[0].name]
    lines = markers_section(attr, order).splitlines()
    assert lines == [f"{m.name}: {m.description}" for m in (attr.markers[i] for i in (1, 2, 0))]
    levels = level_descriptions_section(attr, order).splitlines()
    assert len(levels) == 18
    assert levels[0] == f"{attr.markers[1].name}:"
    level_line = f"    Level 5 [{attr.markers[1].level(5).label}]: "
    assert levels[1] == level_line + attr.markers[1].level(5).description
    assert levels[5].startswith("    Level 1 [")
    assert levels[6] == f"{attr.markers[2].name}:"
    subs = subrubrics_section(attr, order).splitlines()
    marker = attr.markers[1]
    expected = {lv.level: lv.description for lv in reversed(marker.levels)}
    assert subs[0] == f"{marker.name}: {expected}"
    assert subs[0].startswith(f"{marker.name}: {{5: '")


def test_definitions_section_keeps_rubric_order(toy_rubric: Rubric) -> None:
    lines = definitions_section(toy_rubric).splitlines()
    assert lines == [f"{term}: {text}" for term, text in toy_rubric.definitions.items()]


def test_extraction_output_template_exact_text() -> None:
    expected = (
        "    {\n"
        '        "A": {\n'
        '            "evaluable": true/false (bool),\n'
        '            "quotes": ["exact verbatim passage from the text", ...] '
        "(list of strings; [] when evaluable is false)\n"
        "        },\n\n"
        '        "B": {\n'
        '            "evaluable": true/false (bool),\n'
        '            "quotes": ["exact verbatim passage from the text", ...] '
        "(list of strings; [] when evaluable is false)\n"
        "        }\n"
        "    }"
    )
    assert extraction_output_template(["A", "B"]) == expected


@pytest.mark.parametrize(
    "order",
    [["Tone", "Respectful wording"], ["x"], []],
)
def test_marker_order_must_be_a_permutation(toy_rubric: Rubric, order: list[str]) -> None:
    attr = toy_rubric.attributes[1]
    with pytest.raises(PromptContractError):
        markers_section(attr, order)
    with pytest.raises(PromptContractError):
        render_extraction_prompt(_context(toy_rubric, 1, order))


def test_unknown_marker_name_is_named_in_the_error(toy_rubric: Rubric) -> None:
    attr = toy_rubric.attributes[1]
    order = [m.name for m in attr.markers]
    order[0] = "Imaginary marker"
    with pytest.raises(PromptContractError, match="Imaginary marker"):
        level_descriptions_section(attr, order)


def test_duplicate_marker_name_is_rejected(toy_rubric: Rubric) -> None:
    attr = toy_rubric.attributes[1]
    names = [m.name for m in attr.markers]
    with pytest.raises(PromptContractError, match="duplicate"):
        subrubrics_section(attr, [names[0], names[0], names[1]])


def test_scoring_prompt_lists_only_markers_to_rate_in_context_order(toy_rubric: Rubric) -> None:
    attr = toy_rubric.attributes[0]
    names = [m.name for m in attr.markers]
    ctx = _context(toy_rubric, 0, [names[2], names[1], names[0]])
    text = render_scoring_prompt(ctx, [names[0], names[2]])
    listed = text.split("1. Markers to rate:")[1].split("2. Marker Definitions:")[0]
    assert names[1] not in listed
    assert listed.index(names[2]) < listed.index(names[0])
    with pytest.raises(PromptContractError):
        render_scoring_prompt(ctx, [])
    with pytest.raises(PromptContractError, match="Imaginary"):
        render_scoring_prompt(ctx, ["Imaginary"])


def test_one_stage_prompt_carries_the_subrubrics(toy_rubric: Rubric) -> None:
    ctx = _context(toy_rubric)
    text = render_one_stage_prompt(ctx)
    assert subrubrics_section(ctx.attribute, ctx.marker_order) in text


def test_template_hashes_have_three_keys() -> None:
    hashes = template_hashes()
    assert set(hashes) == {"extraction", "scoring", "one_stage"}
    assert all(re.fullmatch(r"[0-9a-f]{16}", value) for value in hashes.values())
    assert hashes == template_hashes()
