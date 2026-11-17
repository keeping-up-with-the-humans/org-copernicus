"""Every attribute's level orders enter the identity of a one-stage store, not only the first's.

The rubric of a report has two attributes. Its first attribute lists the levels of both markers
from 5 down to 1, the order version 0.1.0 printed for every marker, and only its later attribute
lists them otherwise. A version that recorded the orders of the first attribute alone would record
none for it, and would take a store whose later prompts have changed for one coded with the same
prompts; every case here is refused before any call to the backend and any write to a store.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from level_order_store_support import (
    DIFFER,
    KEYS_OF_0_1_0,
    MERGE,
    MISSING,
    ONE_STAGE,
    as_written_by_0_1_0,
    fail,
    respond,
    texts,
)
from level_order_support import FIVE_DOWN, ONE_UP, SCATTERED, report_data, write_rubric

from copernicus.coders import (
    CodingSettings,
    code_texts,
    export_provenance,
    export_scores,
    merge_stores,
)
from copernicus.coders.level_orders import level_order_settings
from copernicus.errors import StoreError
from copernicus.llm import FakeBackend
from copernicus.rubric import Rubric, load_rubric

FIRST_DOWN = (FIVE_DOWN, FIVE_DOWN)
LATER = (ONE_UP, SCATTERED)
LATER_CHANGED = (SCATTERED, ONE_UP)
COMPLETE = (
    '{"signposts": [5, 4, 3, 2, 1], "summary": [5, 4, 3, 2, 1], '
    '"plain_words": [1, 2, 3, 4, 5], "short_sentences": [3, 1, 5, 2, 4]}'
)
CHANGED = (
    '{"signposts": [5, 4, 3, 2, 1], "summary": [5, 4, 3, 2, 1], '
    '"plain_words": [3, 1, 5, 2, 4], "short_sentences": [1, 2, 3, 4, 5]}'
)
FAILING = replace(ONE_STAGE, max_failure_share=1.0)
RETRY = replace(FAILING, retry_failed=True)


Orders = tuple[tuple[int, ...], ...]


def report(tmp_path: Path, structure: Orders, style: Orders) -> Rubric:
    """Load the rubric of a report from a file that lists each marker's levels in its order."""
    return load_rubric(write_rubric(tmp_path, report_data(structure, style), "report"))


def refused_before_any_call(store: Path, rubric: Rubric, settings: CodingSettings) -> str:
    """Return the refusal of coding into a store, checking that nothing was called or written."""
    before = store.read_bytes()
    backend = FakeBackend(respond)
    with pytest.raises(StoreError) as refused:
        code_texts(texts(), rubric, backend, store, settings)
    assert backend.calls == [] and store.read_bytes() == before
    return str(refused.value)


def test_the_identity_lists_every_marker_of_every_attribute(tmp_path: Path) -> None:
    rubric = report(tmp_path, FIRST_DOWN, LATER)
    every_marker = [marker for attribute in rubric.attributes for marker in attribute.markers]
    assert json.loads(COMPLETE) == {m.marker_id: list(m.level_order) for m in every_marker}
    assert level_order_settings(rubric) == {"level_orders": COMPLETE}
    store = tmp_path / "s.sqlite"
    code_texts(texts(), rubric, FakeBackend(respond), store, ONE_STAGE)
    info = export_provenance(store)
    assert info["level_orders"] == COMPLETE and set(info) == KEYS_OF_0_1_0 | {"level_orders"}


def test_a_change_of_the_later_attribute_alone_is_refused_on_resume_retry_and_merge(
    tmp_path: Path,
) -> None:
    rubric = report(tmp_path, FIRST_DOWN, LATER)
    changed = report(tmp_path, FIRST_DOWN, LATER_CHANGED)
    assert rubric.content_hash() == changed.content_hash()
    mismatch = f"run_info mismatch on level_orders: stored {COMPLETE}, requested {CHANGED}"
    store, failed = tmp_path / "coded.sqlite", tmp_path / "failed.sqlite"
    code_texts(texts(), rubric, FakeBackend(respond), store, ONE_STAGE)
    assert refused_before_any_call(store, changed, ONE_STAGE) == f"{mismatch}: {DIFFER}"
    code_texts(texts(), rubric, FakeBackend(fail), failed, FAILING)
    assert set(export_scores(failed)["status"]) == {"scoring_failed"}
    assert refused_before_any_call(failed, changed, RETRY) == f"{mismatch}: {DIFFER}"
    other, target = tmp_path / "other.sqlite", tmp_path / "merged.sqlite"
    code_texts(texts(), changed, FakeBackend(respond), other, ONE_STAGE)
    with pytest.raises(StoreError) as refused:
        merge_stores([store, other], target)
    assert str(refused.value) == f"{mismatch} (in {other}): {MERGE}"
    assert not target.exists()


def test_an_old_store_is_refused_when_only_a_later_attribute_is_not_listed_from_five_down(
    tmp_path: Path,
) -> None:
    rubric = report(tmp_path, FIRST_DOWN, LATER)
    every_down = report(tmp_path, FIRST_DOWN, FIRST_DOWN)
    assert every_down.content_hash() == rubric.content_hash()
    old, old_failed = tmp_path / "old.sqlite", tmp_path / "old_failed.sqlite"
    code_texts(texts(), every_down, FakeBackend(respond), old, ONE_STAGE)
    code_texts(texts(), every_down, FakeBackend(fail), old_failed, FAILING)
    for store in (old, old_failed):
        assert set(export_provenance(store)) == KEYS_OF_0_1_0
    assert refused_before_any_call(old, rubric, ONE_STAGE) == MISSING
    assert refused_before_any_call(old_failed, rubric, RETRY) == MISSING
    new, target = tmp_path / "new.sqlite", tmp_path / "merged.sqlite"
    code_texts(texts(), rubric, FakeBackend(respond), new, ONE_STAGE)
    with pytest.raises(StoreError) as refused:
        merge_stores([old, new], target)
    assert str(refused.value) == f"run_info missing level_orders in the reference store: {MERGE}"
    assert not target.exists()
    backend = FakeBackend(respond)
    summary = code_texts(texts(), every_down, backend, old, ONE_STAGE)
    assert (summary.resumed, summary.n_attribute_calls, backend.calls) == (2, 0, [])


def test_a_store_of_this_version_without_its_entry_is_refused_as_an_old_one(
    tmp_path: Path,
) -> None:
    rubric = report(tmp_path, FIRST_DOWN, LATER)
    store = tmp_path / "s.sqlite"
    code_texts(texts(), rubric, FakeBackend(respond), store, ONE_STAGE)
    as_written_by_0_1_0(store)
    assert refused_before_any_call(store, rubric, ONE_STAGE) == MISSING
