"""A one-stage store records the level orders of its rubric unless every marker is listed 5 to 1.

Version 0.1.0 printed the levels of every marker from 5 down to 1 in the one-stage prompt and
recorded no level order. A one-stage store of a rubric listed from 5 down to 1 therefore holds the
run_info that version wrote (`KEYS_OF_0_1_0`, the keys of a store of version 0.1.0 with the fake
backend) and resumes. For a rubric listed otherwise the prompt has changed, and a store of that
version, which a store of this version without its ``level_orders`` row stands in for, is
refused with a message that says so. The rubrics here have one attribute; the tests of a later
attribute's order are in ``test_one_stage_level_order_attributes.py``.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from level_order_store_support import (
    DIFFER,
    KEYS_OF_0_1_0,
    MERGE,
    MISSING,
    ONE_STAGE,
    TWO_STAGE,
    Factory,
    as_written_by_0_1_0,
    respond,
    texts,
)
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

from copernicus.coders import (
    code_texts,
    export_provenance,
    export_scores,
    merge_stores,
    subrubrics_section,
)
from copernicus.data import TextTable
from copernicus.errors import StoreError
from copernicus.llm import FakeBackend
from copernicus.rubric import Rubric, load_rubric

ALL_ONE_UP = '{"signposts": [1, 2, 3, 4, 5], "summary": [1, 2, 3, 4, 5]}'
ONE_SCATTERED = '{"signposts": [1, 2, 3, 4, 5], "summary": [3, 1, 5, 2, 4]}'
DOWN_UP = '{"signposts": [5, 4, 3, 2, 1], "summary": [1, 2, 3, 4, 5]}'


def clarity(tmp_path: Path, *orders: tuple[int, ...]) -> Rubric:
    """Load the rubric of a short report from a file that lists its markers' levels in order."""
    return load_rubric(write_rubric(tmp_path, clarity_data(*orders), "clarity"))


def test_the_one_stage_coder_sends_each_marker_in_the_level_order_of_its_file(
    tmp_path: Path,
) -> None:
    rubric = clarity(tmp_path, ONE_UP, FIVE_DOWN)
    backend = FakeBackend(respond)
    code_texts(texts(), rubric, backend, tmp_path / "s.sqlite", ONE_STAGE)
    attribute = rubric.attributes[0]
    lines = subrubrics_section(attribute, [m.name for m in attribute.markers]).splitlines()
    assert len(backend.calls) == 2
    for call in backend.calls:
        assert set(lines) <= set(call.system.splitlines())
        assert "Signposting: {1: '" in call.system and "Closing summary: {5: " in call.system


def test_a_one_stage_store_records_the_level_orders_unless_all_are_listed_from_five_down(
    tmp_path: Path,
) -> None:
    recorded = {
        "one_up": (clarity(tmp_path, ONE_UP, ONE_UP), ALL_ONE_UP),
        "scattered": (clarity(tmp_path, ONE_UP, SCATTERED), ONE_SCATTERED),
        "down_up": (clarity(tmp_path, FIVE_DOWN, ONE_UP), DOWN_UP),
    }
    for name, (rubric, orders) in recorded.items():
        store = tmp_path / f"{name}.sqlite"
        code_texts(texts(), rubric, FakeBackend(respond), store, ONE_STAGE)
        info = export_provenance(store)
        assert info["level_orders"] == orders and set(info) == KEYS_OF_0_1_0 | {"level_orders"}
    five_down = clarity(tmp_path, FIVE_DOWN, FIVE_DOWN)
    in_code = built_in_code(clarity(tmp_path, ONE_UP, ONE_UP))
    unrecorded = {"five_down": five_down, "in_code": in_code}
    for name, rubric in unrecorded.items():
        store = tmp_path / f"{name}.sqlite"
        code_texts(texts(), rubric, FakeBackend(respond), store, ONE_STAGE)
        assert set(export_provenance(store)) == KEYS_OF_0_1_0
    for name, (rubric, _) in recorded.items():
        store = tmp_path / f"{name}_two_stage.sqlite"
        code_texts(texts(), rubric, FakeBackend(respond), store, TWO_STAGE)
        assert "level_orders" not in export_provenance(store)


def test_a_one_stage_store_of_0_1_0_resumes_for_a_rubric_listed_from_five_down(
    tmp_path: Path, toy_corpus: TextTable, cue_backend: Factory
) -> None:
    five_down = clarity(tmp_path, FIVE_DOWN, FIVE_DOWN)
    toy = toy_listed(FIVE_DOWN)
    table = toy_corpus.subset(["t001", "t002"])
    cases = [
        (five_down, texts(), lambda: FakeBackend(respond)),
        (built_in_code(five_down), texts(), lambda: FakeBackend(respond)),
        (toy, table, cue_backend),
        (built_in_code(toy), table, cue_backend),
    ]
    for index, (rubric, table_of_case, backend_of) in enumerate(cases):
        store = tmp_path / f"s{index}.sqlite"
        code_texts(table_of_case, rubric, backend_of(), store, ONE_STAGE)
        assert set(export_provenance(store)) == KEYS_OF_0_1_0
        backend = backend_of()
        summary = code_texts(table_of_case, rubric, backend, store, ONE_STAGE)
        assert (summary.resumed, summary.n_attribute_calls, backend.calls) == (2, 0, [])


def test_a_one_stage_store_of_0_1_0_is_refused_for_a_rubric_listed_otherwise(
    tmp_path: Path, toy_corpus: TextTable, cue_backend: Factory
) -> None:
    cases = [
        (clarity(tmp_path, ONE_UP, SCATTERED), texts(), lambda: FakeBackend(respond)),
        (toy_file(), toy_corpus.subset(["t001", "t002"]), cue_backend),
    ]
    for index, (rubric, table, backend_of) in enumerate(cases):
        store = tmp_path / f"s{index}.sqlite"
        code_texts(table, rubric, backend_of(), store, ONE_STAGE)
        as_written_by_0_1_0(store)
        scores = export_scores(store)
        backend = backend_of()
        with pytest.raises(StoreError) as refused:
            code_texts(table, rubric, backend, store, ONE_STAGE)
        assert str(refused.value) == MISSING
        assert backend.calls == []
        pd.testing.assert_frame_equal(export_scores(store), scores)


def test_a_one_stage_store_of_another_level_order_is_refused(tmp_path: Path) -> None:
    store = tmp_path / "s.sqlite"
    code_texts(texts(), clarity(tmp_path, ONE_UP, ONE_UP), FakeBackend(respond), store, ONE_STAGE)
    scattered = clarity(tmp_path, ONE_UP, SCATTERED)
    with pytest.raises(StoreError) as reordered:
        code_texts(texts(), scattered, FakeBackend(respond), store, ONE_STAGE)
    assert str(reordered.value) == (
        f"run_info mismatch on level_orders: stored {ALL_ONE_UP}, requested {ONE_SCATTERED}: "
        f"{DIFFER}"
    )
    five_down = clarity(tmp_path, FIVE_DOWN, FIVE_DOWN)
    for rubric in (five_down, built_in_code(five_down)):
        with pytest.raises(StoreError) as unexpected:
            code_texts(texts(), rubric, FakeBackend(respond), store, ONE_STAGE)
        assert str(unexpected.value) == f"run_info unexpected level_orders: {DIFFER}"


def test_a_two_stage_store_resumes_whatever_the_level_order(
    tmp_path: Path, toy_corpus: TextTable, cue_backend: Factory
) -> None:
    store = tmp_path / "two.sqlite"
    table = toy_corpus.subset(["t001", "t002"])
    code_texts(table, toy_file(), cue_backend(), store, TWO_STAGE)
    for rubric in (toy_listed(FIVE_DOWN), toy_listed(SCATTERED), built_in_code(toy_file())):
        backend = cue_backend()
        summary = code_texts(table, rubric, backend, store, TWO_STAGE)
        assert (summary.resumed, summary.n_attribute_calls, backend.calls) == (2, 0, [])


def test_a_merge_refuses_one_stage_stores_coded_with_different_level_orders(
    tmp_path: Path,
) -> None:
    rubric = clarity(tmp_path, ONE_UP, ONE_UP)
    old, new = tmp_path / "old.sqlite", tmp_path / "new.sqlite"
    code_texts(texts(), rubric, FakeBackend(respond), old, replace(ONE_STAGE, text_range=(0, 1)))
    as_written_by_0_1_0(old)
    code_texts(texts(), rubric, FakeBackend(respond), new, replace(ONE_STAGE, text_range=(1, 2)))
    target = tmp_path / "merged.sqlite"
    with pytest.raises(StoreError) as refused:
        merge_stores([old, new], target)
    assert str(refused.value) == f"run_info missing level_orders in the reference store: {MERGE}"
    with pytest.raises(StoreError) as refused:
        merge_stores([new, old], target)
    assert str(refused.value) == f"run_info missing level_orders in {old}: {MERGE}"
    scattered = tmp_path / "scattered.sqlite"
    rubric_scattered = clarity(tmp_path, ONE_UP, SCATTERED)
    first_text = replace(ONE_STAGE, text_range=(0, 1))
    code_texts(texts(), rubric_scattered, FakeBackend(respond), scattered, first_text)
    with pytest.raises(StoreError) as refused:
        merge_stores([new, scattered], target)
    assert str(refused.value) == (
        f"run_info mismatch on level_orders: stored {ALL_ONE_UP}, requested {ONE_SCATTERED} "
        f"(in {scattered}): {MERGE}"
    )
    assert not target.exists()
    five_down = clarity(tmp_path, FIVE_DOWN, FIVE_DOWN)
    shards = [tmp_path / "first.sqlite", tmp_path / "second.sqlite"]
    for shard, text_range in zip(shards, ((0, 1), (1, 2)), strict=True):
        settings = replace(ONE_STAGE, text_range=text_range)
        code_texts(texts(), five_down, FakeBackend(respond), shard, settings)
    assert merge_stores(shards, target)["results"] == 2


def fields_reversed(data: dict[str, Any]) -> dict[str, Any]:
    """Return a rubric dictionary with the fields of every object in reverse order.

    The fields of the rubric, of each attribute, marker and level are reversed; the entries of
    the glossary and of each sub-rubric keep their order, which the prompts follow.
    """

    def reversed_fields(entry: dict[str, Any]) -> dict[str, Any]:
        return dict(reversed(list(entry.items())))

    rubric = reversed_fields(data)
    rubric["attributes"] = [reversed_fields(attribute) for attribute in data["attributes"]]
    for attribute in rubric["attributes"]:
        attribute["markers"] = [reversed_fields(marker) for marker in attribute["markers"]]
        for marker in attribute["markers"]:
            levels = marker["sub_rubric"].items()
            marker["sub_rubric"] = {key: reversed_fields(level) for key, level in levels}
    return rubric


def test_a_changed_content_needs_a_new_store_and_a_changed_field_order_does_not(
    tmp_path: Path, toy_corpus: TextTable, cue_backend: Factory
) -> None:
    table = toy_corpus.subset(["t001", "t002"])
    reordered = load_rubric(write_rubric(tmp_path, fields_reversed(toy_data()), "fields"))
    assert reordered == toy_file() and reordered.content_hash() == toy_file().content_hash()
    glossary = toy_data()
    glossary["definitions"] = dict(reversed(list(glossary["definitions"].items())))
    content_changed = load_rubric(write_rubric(tmp_path, glossary, "glossary"))
    for settings in (ONE_STAGE, TWO_STAGE):
        store = tmp_path / f"{settings.method}.sqlite"
        code_texts(table, toy_file(), cue_backend(), store, settings)
        backend = cue_backend()
        summary = code_texts(table, reordered, backend, store, settings)
        assert (summary.resumed, summary.n_attribute_calls, backend.calls) == (2, 0, [])
        with pytest.raises(StoreError, match="run_info mismatch on rubric_hash"):
            code_texts(table, content_changed, cue_backend(), store, settings)
