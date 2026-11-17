"""Synthetic rubrics whose files list the levels of their markers in a chosen order.

The rubric of a short report has one attribute and up to three markers. The descriptions of the
second marker hold an apostrophe, a double quote and both, so a test of the one-stage prompt also
checks the quoting of Python's dictionary notation, character for character. The rubric of a
report adds a second attribute, style, with two markers of its own, so that a test can change the
level order of a later attribute alone.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from copernicus.rubric import Marker, Rubric, load_rubric, rubric_from_dict, toy_rubric_path

ONE_UP = (1, 2, 3, 4, 5)
FIVE_DOWN = (5, 4, 3, 2, 1)
SCATTERED = (3, 1, 5, 2, 4)
LABELS = {1: "Clearly low", 2: "Low", 3: "Mixed", 4: "High", 5: "Clearly high"}
SIGNPOSTS = {
    1: "No phrase announces what comes next.",
    2: "One phrase announces a later part, vaguely.",
    3: "Some parts are announced; others arrive unannounced.",
    4: "Most parts are announced with a clear phrase.",
    5: "Every part is announced with a clear phrase.",
}
SUMMARY = {
    1: 'The text ends without a "so what" of any kind.',
    2: "A closing line restates one point.",
    3: 'The reader\'s "next step" is named for some points.',
    4: "A closing passage restates most points in order.",
    5: "A closing passage restates every point and says why it matters to the reader's work.",
}
MARKERS = (
    ("signposts", "Signposting", "The report announces its parts.", SIGNPOSTS),
    ("summary", "Closing summary", "The report ends with a summary.", SUMMARY),
    ("headings", "Headings", "The report gives its parts headings.", SIGNPOSTS),
)
PLAIN_WORDS = {
    1: "Most words are jargon that the reader must look up.",
    2: "Jargon appears in most sentences.",
    3: "Plain words and jargon are mixed.",
    4: "Most words are plain; the jargon is explained.",
    5: "Every word is plain or explained where it first appears.",
}
SHORT_SENTENCES = {
    1: "Most sentences run over several lines.",
    2: "Long sentences outnumber short ones.",
    3: "Long and short sentences alternate.",
    4: "Most sentences are short.",
    5: "Every sentence makes one point in a few words.",
}
STYLE_MARKERS = (
    ("plain_words", "Plain words", "The report uses plain words.", PLAIN_WORDS),
    (
        "short_sentences",
        "Short sentences",
        "The report keeps its sentences short.",
        SHORT_SENTENCES,
    ),
)


def marker_data(
    definitions: Sequence[tuple[str, str, str, dict[int, str]]], orders: Sequence[Sequence[int]]
) -> list[dict[str, Any]]:
    """Return one marker per order, taken from ``definitions`` in turn, its levels in that order."""
    return [
        {
            "marker_id": marker_id,
            "name": name,
            "description": description,
            "sub_rubric": {
                str(number): {"label": LABELS[number], "description": texts[number]}
                for number in order
            },
        }
        for (marker_id, name, description, texts), order in zip(
            definitions[: len(orders)], orders, strict=True
        )
    ]


def clarity_data(*orders: Sequence[int], rubric_id: str = "clarity_v1") -> dict[str, Any]:
    """Return the rubric of a short report with one marker per order, its levels in that order."""
    markers = marker_data(MARKERS, orders)
    return {
        "rubric_id": rubric_id,
        "concept": "Clarity of a short report",
        "concept_definition": "Clarity is the degree to which a reader can follow a report.",
        "literature_sources": [],
        "attributes": [
            {
                "attribute_id": "structure",
                "name": "Structure",
                "description": "How well the report orders its parts.",
                "literature_sources": [],
                "markers": markers,
            }
        ],
        "definitions": {"signpost": "A phrase that announces what comes next."},
    }


def report_data(
    structure: Sequence[Sequence[int]],
    style: Sequence[Sequence[int]],
    rubric_id: str = "report_v1",
) -> dict[str, Any]:
    """Return the rubric of a report: the attribute of `clarity_data`, then style.

    Args:
        structure: The level orders of the markers of the first attribute, structure.
        style: The level orders of the markers of the second attribute, style.
        rubric_id: The id of the rubric.

    Returns:
        The rubric dictionary, the markers of each attribute listing their levels in order.
    """
    data = clarity_data(*structure, rubric_id=rubric_id)
    data["attributes"].append(
        {
            "attribute_id": "style",
            "name": "Style",
            "description": "How plainly the report is written.",
            "literature_sources": [],
            "markers": marker_data(STYLE_MARKERS, style),
        }
    )
    return data


def write_rubric(folder: Path, data: dict[str, Any], name: str) -> Path:
    """Write a rubric dictionary as a JSON file and return its path."""
    path = folder / f"{name}.json"
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def toy_data(order: Sequence[int] | None = None) -> dict[str, Any]:
    """Return the toy rubric file's dictionary, every marker's levels in ``order`` if given."""
    data: dict[str, Any] = json.loads(toy_rubric_path().read_text(encoding="utf-8"))
    if order is not None:
        for attribute in data["attributes"]:
            for marker in attribute["markers"]:
                levels = marker["sub_rubric"]
                marker["sub_rubric"] = {str(number): levels[str(number)] for number in order}
    return data


def toy_listed(order: Sequence[int]) -> Rubric:
    """Return the toy rubric as read from a file that lists every marker's levels in ``order``."""
    return rubric_from_dict(toy_data(order))


def toy_file() -> Rubric:
    """Return the toy rubric as shipped; its file lists every marker's levels from 1 to 5."""
    return load_rubric(toy_rubric_path())


def built_in_code(rubric: Rubric) -> Rubric:
    """Return the rubric with every marker built in code from its fields, without a level order."""
    attributes = tuple(
        dataclasses.replace(
            attribute,
            markers=tuple(
                Marker(marker.marker_id, marker.name, marker.description, marker.levels)
                for marker in attribute.markers
            ),
        )
        for attribute in rubric.attributes
    )
    return dataclasses.replace(rubric, attributes=attributes)
