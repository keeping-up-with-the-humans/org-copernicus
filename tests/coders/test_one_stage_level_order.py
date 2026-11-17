"""The one-stage prompt lists each marker's levels in the order of the rubric file.

The sub-rubric section prints each marker's level descriptions as a mapping in Python's
dictionary notation, in the order in which the rubric file lists the levels; a marker built in
code lists them in its level order, by default (without ``level_order``) from 5 down to 1, as
version 0.1.0 printed them for every rubric. The lines expected here are written out by hand,
quotes and escapes included. The two-stage prompts do not
depend on the order: the extraction prompt lists the levels from 5 down to 1, the scoring rubric
from 1 to 5, for every rubric, as in version 0.1.0.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from level_order_support import (
    FIVE_DOWN,
    ONE_UP,
    SCATTERED,
    built_in_code,
    clarity_data,
    write_rubric,
)
from two_stage_support import make_ctx

from copernicus.coders import (
    level_descriptions_section,
    render_extraction_prompt,
    render_one_stage_prompt,
    render_scoring_prompt,
    scoring_rubric_text,
    subrubrics_section,
)
from copernicus.coders.level_orders import level_order_settings
from copernicus.rubric import Marker, Rubric, load_rubric

SIGNPOSTS_ONE_UP = (
    "Signposting: {1: 'No phrase announces what comes next.', "
    "2: 'One phrase announces a later part, vaguely.', "
    "3: 'Some parts are announced; others arrive unannounced.', "
    "4: 'Most parts are announced with a clear phrase.', "
    "5: 'Every part is announced with a clear phrase.'}"
)
SIGNPOSTS_FIVE_DOWN = (
    "Signposting: {5: 'Every part is announced with a clear phrase.', "
    "4: 'Most parts are announced with a clear phrase.', "
    "3: 'Some parts are announced; others arrive unannounced.', "
    "2: 'One phrase announces a later part, vaguely.', "
    "1: 'No phrase announces what comes next.'}"
)
SUMMARY_ONE_UP = (
    "Closing summary: {1: 'The text ends without a \"so what\" of any kind.', "
    "2: 'A closing line restates one point.', "
    "3: 'The reader\\'s \"next step\" is named for some points.', "
    "4: 'A closing passage restates most points in order.', "
    '5: "A closing passage restates every point and says why it matters to '
    "the reader's work.\"}"
)
SUMMARY_FIVE_DOWN = (
    'Closing summary: {5: "A closing passage restates every point and says why it matters to '
    "the reader's work.\", "
    "4: 'A closing passage restates most points in order.', "
    "3: 'The reader\\'s \"next step\" is named for some points.', "
    "2: 'A closing line restates one point.', "
    "1: 'The text ends without a \"so what\" of any kind.'}"
)
HEADINGS_SCATTERED = (
    "Headings: {3: 'Some parts are announced; others arrive unannounced.', "
    "1: 'No phrase announces what comes next.', "
    "5: 'Every part is announced with a clear phrase.', "
    "2: 'One phrase announces a later part, vaguely.', "
    "4: 'Most parts are announced with a clear phrase.'}"
)
LEVEL_DESCRIPTIONS = "\n".join(
    [
        "Signposting:",
        "    Level 5 [Clearly high]: Every part is announced with a clear phrase.",
        "    Level 4 [High]: Most parts are announced with a clear phrase.",
        "    Level 3 [Mixed]: Some parts are announced; others arrive unannounced.",
        "    Level 2 [Low]: One phrase announces a later part, vaguely.",
        "    Level 1 [Clearly low]: No phrase announces what comes next.",
        "Closing summary:",
        "    Level 5 [Clearly high]: A closing passage restates every point and says why it "
        "matters to the reader's work.",
        "    Level 4 [High]: A closing passage restates most points in order.",
        '    Level 3 [Mixed]: The reader\'s "next step" is named for some points.',
        "    Level 2 [Low]: A closing line restates one point.",
        '    Level 1 [Clearly low]: The text ends without a "so what" of any kind.',
    ]
)
SCORING_RUBRIC = "\n".join(
    [
        'Level 1: No phrase announces what comes next. The text ends without a "so what" of any '
        "kind.",
        "Level 2: One phrase announces a later part, vaguely. A closing line restates one point.",
        'Level 3: Some parts are announced; others arrive unannounced. The reader\'s "next step" '
        "is named for some points.",
        "Level 4: Most parts are announced with a clear phrase. A closing passage restates most "
        "points in order.",
        "Level 5: Every part is announced with a clear phrase. A closing passage restates every "
        "point and says why it matters to the reader's work.",
    ]
)
ORDERS_OF_TWO = ((ONE_UP, ONE_UP), (FIVE_DOWN, FIVE_DOWN), (ONE_UP, SCATTERED))


def clarity(tmp_path: Path, *orders: tuple[int, ...]) -> Rubric:
    """Load the rubric of a short report from a file that lists its markers' levels in order."""
    rubric_id = "clarity_v1" if len(orders) < 3 else "clarity_mixed_v1"
    return load_rubric(write_rubric(tmp_path, clarity_data(*orders, rubric_id=rubric_id), "c"))


def section(rubric: Rubric) -> list[str]:
    """Return the sub-rubric lines of the first attribute, markers in rubric order."""
    attribute = rubric.attributes[0]
    return subrubrics_section(attribute, [marker.name for marker in attribute.markers]).splitlines()


def test_a_rubric_file_listing_the_levels_from_one_to_five_is_printed_so(tmp_path: Path) -> None:
    assert section(clarity(tmp_path, ONE_UP, ONE_UP)) == [SIGNPOSTS_ONE_UP, SUMMARY_ONE_UP]


def test_a_rubric_file_listing_the_levels_from_five_down_gives_the_lines_of_0_1_0(
    tmp_path: Path,
) -> None:
    assert section(clarity(tmp_path, FIVE_DOWN, FIVE_DOWN)) == [
        SIGNPOSTS_FIVE_DOWN,
        SUMMARY_FIVE_DOWN,
    ]


def test_a_rubric_built_in_code_gives_the_lines_of_0_1_0(tmp_path: Path) -> None:
    rubric = built_in_code(clarity(tmp_path, ONE_UP, ONE_UP))
    assert section(rubric) == [SIGNPOSTS_FIVE_DOWN, SUMMARY_FIVE_DOWN]


def test_a_marker_built_in_code_with_a_level_order_is_printed_and_recorded_in_it(
    tmp_path: Path,
) -> None:
    rubric = built_in_code(clarity(tmp_path, ONE_UP, ONE_UP))
    signposts, summary = rubric.attributes[0].markers
    given = (
        Marker(
            signposts.marker_id, signposts.name, signposts.description, signposts.levels, ONE_UP
        ),
        Marker(summary.marker_id, summary.name, summary.description, summary.levels, FIVE_DOWN),
    )
    rubric = replace(rubric, attributes=(replace(rubric.attributes[0], markers=given),))
    assert section(rubric) == [SIGNPOSTS_ONE_UP, SUMMARY_FIVE_DOWN]
    assert level_order_settings(rubric) == {
        "level_orders": '{"signposts": [1, 2, 3, 4, 5], "summary": [5, 4, 3, 2, 1]}'
    }


def test_each_marker_is_printed_in_its_own_level_order_whatever_the_marker_order(
    tmp_path: Path,
) -> None:
    rubric = clarity(tmp_path, ONE_UP, FIVE_DOWN, SCATTERED)
    assert section(rubric) == [SIGNPOSTS_ONE_UP, SUMMARY_FIVE_DOWN, HEADINGS_SCATTERED]
    shuffled = ["Headings", "Signposting", "Closing summary"]
    lines = subrubrics_section(rubric.attributes[0], shuffled).splitlines()
    assert lines == [HEADINGS_SCATTERED, SIGNPOSTS_ONE_UP, SUMMARY_FIVE_DOWN]


def test_the_one_stage_prompts_of_two_orders_differ_in_the_sub_rubric_lines_alone(
    tmp_path: Path,
) -> None:
    one_up = render_one_stage_prompt(make_ctx(clarity(tmp_path, ONE_UP, ONE_UP)))
    five_down = render_one_stage_prompt(make_ctx(clarity(tmp_path, FIVE_DOWN, FIVE_DOWN)))
    in_code = render_one_stage_prompt(make_ctx(built_in_code(clarity(tmp_path, ONE_UP, ONE_UP))))
    assert in_code == five_down
    pairs = zip(one_up.splitlines(), five_down.splitlines(), strict=True)
    assert [(a, b) for a, b in pairs if a != b] == [
        (SIGNPOSTS_ONE_UP, SIGNPOSTS_FIVE_DOWN),
        (SUMMARY_ONE_UP, SUMMARY_FIVE_DOWN),
    ]


def test_the_two_stage_prompts_are_those_of_0_1_0_whatever_the_level_order(
    tmp_path: Path,
) -> None:
    rendered = set()
    rubrics = [clarity(tmp_path, *orders) for orders in ORDERS_OF_TWO]
    for rubric in [*rubrics, built_in_code(rubrics[0])]:
        ctx = make_ctx(rubric)
        names = list(ctx.marker_order)
        assert level_descriptions_section(ctx.attribute, names) == LEVEL_DESCRIPTIONS
        assert scoring_rubric_text(ctx.attribute, names) == SCORING_RUBRIC
        extraction = render_extraction_prompt(ctx)
        assert LEVEL_DESCRIPTIONS in extraction
        rendered.add((extraction, render_scoring_prompt(ctx, names)))
    assert len(rendered) == 1
