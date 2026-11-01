"""Tests for the prompts of reference-text generation."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from copernicus.errors import ConfigurationError
from copernicus.references import build_cells, system_prompt, user_prompt
from copernicus.references.prompts import PROMPT_VERSION
from copernicus.rubric.model import Rubric

GOLDEN_DIRECTORY = Path(__file__).parent / "golden"
DESCRIPTION = (
    "Write each text as a short piece of written feedback on a colleague's draft report, "
    "in plain, slightly informal wording."
)


def golden(name: str) -> str:
    """Read a golden prompt file."""
    return (GOLDEN_DIRECTORY / name).read_text(encoding="utf-8")


def test_the_prompt_version_is_a_library_label() -> None:
    assert PROMPT_VERSION == "references-v1"


def test_the_neutral_system_prompt_matches_its_golden_file() -> None:
    assert system_prompt("neutral", 10, None) == golden("system_neutral.txt")


def test_the_register_system_prompt_matches_its_golden_file() -> None:
    assert system_prompt("register", 10, DESCRIPTION) == golden("system_register.txt")


def test_the_user_prompt_of_the_top_positive_construct_cell_matches_its_golden_file(
    toy_rubric: Rubric,
) -> None:
    cell = build_cells(toy_rubric)[0]
    assert (cell.target, cell.level, cell.polarity) == ("construct", "top", "positive")
    assert user_prompt(cell, toy_rubric, 10) == golden("user_construct_top_positive.txt")


def test_the_number_of_texts_enters_the_prompts() -> None:
    assert "exactly 7 objects" in system_prompt("neutral", 7, None)
    assert "exactly 7 objects" in system_prompt("register", 7, DESCRIPTION)


def test_the_register_description_enters_the_register_prompt_only() -> None:
    notes = "Write each text as a short note to a neighbour about a shared garden."
    assert notes in system_prompt("register", 10, notes)
    assert notes not in system_prompt("neutral", 10, notes)


def test_a_register_prompt_needs_a_description() -> None:
    for missing in (None, "", "   "):
        with pytest.raises(ConfigurationError, match="register_description"):
            system_prompt("register", 10, missing)


def test_an_unknown_style_is_refused() -> None:
    with pytest.raises(ConfigurationError, match="style"):
        system_prompt("formal", 10, None)  # type: ignore[arg-type]


def test_a_negative_cell_asks_for_the_active_low_end_with_the_concept_definition(
    toy_rubric: Rubric,
) -> None:
    cells = {(c.target, c.level, c.polarity): c for c in build_cells(toy_rubric)}
    prompt = user_prompt(cells[("construct", "mid", "negative")], toy_rubric, 10)
    assert "active LOW end" in prompt
    assert "HIGH end" not in prompt
    assert toy_rubric.concept_definition in prompt
    assert "MODERATELY and recognizably" in prompt
    positive = user_prompt(cells[("construct", "low", "positive")], toy_rubric, 10)
    assert "the HIGH end of this quality" in positive
    assert "FAINTLY but discernibly (a mild, threshold-level)" in positive


def test_an_attribute_cell_names_the_facet_and_its_description(toy_rubric: Rubric) -> None:
    attribute = toy_rubric.attributes[1]
    cells = {(c.target, c.level, c.polarity): c for c in build_cells(toy_rubric)}
    cell = cells[(attribute.attribute_id, "top", "positive")]
    prompt = user_prompt(cell, toy_rubric, 10)
    assert f'specifically the facet "{attribute.name}"' in prompt
    assert attribute.description in prompt
    assert toy_rubric.concept_definition not in prompt
    assert cell.pole_guidance in prompt


def test_the_description_sits_unchanged_between_the_fixed_sentence_and_the_format_rule() -> None:
    prompt = system_prompt("register", 10, DESCRIPTION)
    assert f"might actually produce. {DESCRIPTION}\n\nThe writer is a normal person" in prompt
    assert "Honor the format described above" in prompt


@pytest.mark.parametrize("ending", [".", "!", "?", ".  "])
def test_a_description_with_a_sentence_ending_logs_no_warning(
    caplog: pytest.LogCaptureFixture, ending: str
) -> None:
    with caplog.at_level(logging.WARNING, logger="copernicus.references.prompts"):
        system_prompt("register", 10, "Write each text as a short post" + ending)
    assert caplog.records == []


def test_a_description_without_a_sentence_ending_logs_a_warning_but_is_used(
    caplog: pytest.LogCaptureFixture,
) -> None:
    description = "short written feedback on a draft report"
    with caplog.at_level(logging.WARNING, logger="copernicus.references.prompts"):
        prompt = system_prompt("register", 10, description)
    assert description in prompt
    (record,) = caplog.records
    assert record.levelno == logging.WARNING
    assert description in record.getMessage()
