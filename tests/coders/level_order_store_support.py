"""Shared settings, texts, answers and refusal messages of the tests of one-stage level orders.

Version 0.1.0 recorded no level order. A store of that version is stood in for by a store of this
version without its ``level_orders`` row (`as_written_by_0_1_0`), whose run_info then holds
exactly `KEYS_OF_0_1_0`, the keys of a one-stage store of version 0.1.0 with the fake backend.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable
from contextlib import closing
from pathlib import Path

import pandas as pd

from copernicus.coders import CodingSettings, export_provenance
from copernicus.data import TextTable
from copernicus.llm import FakeBackend
from copernicus.llm.base import GenerationRequest

Factory = Callable[[], FakeBackend]
KEYS_OF_0_1_0 = frozenset(
    {
        "attribute_ids",
        "backend_kind",
        "backend_max_tokens_parameter",
        "backend_model",
        "backend_reasoning_effort",
        "backend_supports_seed",
        "backend_supports_temperature",
        "backend_supports_top_p",
        "backend_top_p",
        "library_version",
        "marker_ids",
        "max_new_tokens",
        "max_retries",
        "method",
        "min_chars",
        "model_name",
        "rubric_hash",
        "rubric_id",
        "seed",
        "shuffle_markers",
        "temperature",
        "template_hashes",
        "verifier_version",
        "wording_hash",
    }
)
ONE_STAGE = CodingSettings(method="one_stage", runs=1, temperature=0.3)
TWO_STAGE = CodingSettings(runs=1, temperature=0.3)
TEXTS = {
    "a": "The report announces each part before it begins.",
    "b": "A closing passage restates the points in order.",
}
ANSWER = json.dumps({"dimension_result": {"result": 3, "explanation": "fixed"}})
MARKER_NAMES = ("Signposting", "Closing summary", "Headings", "Plain words", "Short sentences")
MISSING = (
    "run_info missing level_orders: the store was coded with the level descriptions of every "
    "marker listed from level 5 down to level 1 in the one-stage prompt, as an earlier version "
    "of the library listed them for every rubric; this version lists them in the order of the "
    "rubric file, and this rubric lists the levels of some marker in another order, so the "
    "prompt of the one-stage coder has changed for this rubric: code the texts into a new store, "
    "or list the levels of every marker from 5 down to 1 in the rubric file to continue the old "
    "coding (the content hash of the rubric stays the same)"
)
DIFFER = (
    "the one-stage prompt lists the level descriptions of each marker in the order of the rubric "
    "file (a marker built in code lists them in its level_order, by default from 5 down to 1), "
    "and this rubric lists the levels in another order than the rubric the store was coded with, "
    "so the prompt of the one-stage coder has changed: code the texts into a new store, or list "
    "the levels in the rubric file in the order that the store records under level_orders (the "
    "content hash of the rubric stays the same)"
)
MERGE = (
    "the stores were coded with one-stage prompts that list the level descriptions of the "
    "markers in different orders: a store records the order of the rubric file under "
    "level_orders, and none when every marker's levels were listed from 5 down to 1, as an "
    "earlier version of the library listed them for every rubric; code the texts of one store "
    "again with the level order of the other, then merge"
)


def respond(request: GenerationRequest) -> str:
    """Answer an extraction prompt with no passage for any of its markers, any other with a 3."""
    if "EXTRACTION ONLY" in request.system:
        names = [name for name in MARKER_NAMES if f'"{name}": {{' in request.system]
        return json.dumps({name: {"evaluable": False, "quotes": []} for name in names})
    return ANSWER


def fail(request: GenerationRequest) -> str:
    """Answer every prompt with text that is not JSON, so every attempt fails."""
    return "not JSON"


def texts() -> TextTable:
    """Return the two texts of the short reports, in one group."""
    frame = pd.DataFrame({"text_id": list(TEXTS), "text": list(TEXTS.values()), "group_id": "g"})
    return TextTable.from_frame(frame, group_id="group_id")


def as_written_by_0_1_0(store: Path) -> None:
    """Remove the level_orders row, the one run_info entry that version 0.1.0 did not write."""
    with closing(sqlite3.connect(store)) as connection, connection:
        removed = connection.execute('DELETE FROM "run_info" WHERE "key" = ?', ("level_orders",))
        assert removed.rowcount == 1
    assert set(export_provenance(store)) == KEYS_OF_0_1_0
