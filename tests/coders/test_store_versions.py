"""Stores of the published version 0.1.0 under this version, and the number this version records.

A run store records the library version in its run_info and in every result row. Version 0.1.0
wrote a two-stage store, and a one-stage store of a rubric whose file lists every marker's levels
from 5 down to 1, in the layout of this version, so such a store is stood in for by a store of
this version whose version entries say ``0.1.0`` (`as_stored_by_0_1_0`). The version is not part
of the identity of a run: a resume and a merge log a warning that names both numbers and go on,
the exports keep the number of each row, and the hybrid coder reads the store as before.
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Callable
from contextlib import closing
from pathlib import Path

import pandas as pd
import pytest
from hybrid_support import STUB_DIMENSION, STUB_NAME, StubEncoder, unit_references
from level_order_store_support import MISSING, ONE_STAGE, as_written_by_0_1_0

import copernicus
from copernicus.coders import (
    CodingSettings,
    code_texts,
    export_provenance,
    export_scores,
    merge_stores,
)
from copernicus.data import TextTable
from copernicus.errors import StoreError
from copernicus.hybrid import score_hybrid
from copernicus.llm import FakeBackend
from copernicus.rubric.model import Rubric

Factory = Callable[[], FakeBackend]
PUBLISHED = "0.1.0"
TWO_STAGE = CodingSettings(runs=1)


def as_stored_by_0_1_0(store: Path) -> None:
    """Write the number of version 0.1.0 wherever the store records the library version."""
    with closing(sqlite3.connect(store)) as connection, connection:
        connection.execute(
            'UPDATE "run_info" SET "value" = ? WHERE "key" = ?', (PUBLISHED, "library_version")
        )
        connection.execute('UPDATE "results" SET "library_version" = ?', (PUBLISHED,))


def first_texts(corpus: TextTable, count: int) -> TextTable:
    """Return the first texts of the toy corpus."""
    return corpus.subset(corpus.ids()[:count])


def test_a_store_of_this_version_records_its_number_which_is_not_the_published_one(
    toy_rubric: Rubric, toy_corpus: TextTable, cue_backend: Factory, tmp_path: Path
) -> None:
    store = tmp_path / "run.sqlite"
    code_texts(first_texts(toy_corpus, 2), toy_rubric, cue_backend(), store, TWO_STAGE)
    assert export_provenance(store)["library_version"] == copernicus.__version__
    assert set(export_scores(store)["library_version"]) == {copernicus.__version__}
    assert copernicus.__version__ != PUBLISHED


def test_a_store_of_0_1_0_resumes_with_a_warning_that_names_both_numbers(
    toy_rubric: Rubric,
    toy_corpus: TextTable,
    cue_backend: Factory,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    table = first_texts(toy_corpus, 3)
    store = tmp_path / "run.sqlite"
    code_texts(table, toy_rubric, cue_backend(), store, TWO_STAGE)
    as_stored_by_0_1_0(store)
    scores = export_scores(store)
    backend = cue_backend()
    with caplog.at_level(logging.WARNING, logger="copernicus.coders.store"):
        summary = code_texts(table, toy_rubric, backend, store, TWO_STAGE)
    assert (summary.resumed, summary.n_attribute_calls, backend.calls) == (3, 0, [])
    warning = f"store was created with library_version {PUBLISHED}, resumed with "
    assert f"{warning}{copernicus.__version__}" in caplog.text
    pd.testing.assert_frame_equal(export_scores(store), scores)
    assert export_provenance(store)["library_version"] == PUBLISHED


def test_a_store_of_0_1_0_merges_with_a_store_of_this_version_and_each_row_keeps_its_number(
    toy_rubric: Rubric,
    toy_corpus: TextTable,
    cue_backend: Factory,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    ids = toy_corpus.ids()
    old, new, target = tmp_path / "old.sqlite", tmp_path / "new.sqlite", tmp_path / "merged.sqlite"
    code_texts(toy_corpus.subset(ids[:2]), toy_rubric, cue_backend(), old, TWO_STAGE)
    as_stored_by_0_1_0(old)
    code_texts(toy_corpus.subset(ids[1:3]), toy_rubric, cue_backend(), new, TWO_STAGE)
    with caplog.at_level(logging.WARNING, logger="copernicus.coders.store_merge"):
        counts = merge_stores([old, new], target)
    assert (counts["results"], counts["conflicting_cells"]) == (3, 0)
    differs = f"library_version differs: {PUBLISHED} in the reference store, "
    assert f"{differs}{copernicus.__version__} in {new}" in caplog.text
    merged = export_scores(target).drop_duplicates(["text_id", "library_version"])
    numbers = dict(zip(merged["text_id"], merged["library_version"], strict=True))
    assert numbers == {
        str(ids[0]): PUBLISHED,
        str(ids[1]): PUBLISHED,
        str(ids[2]): copernicus.__version__,
    }


def test_the_hybrid_coder_scores_a_store_of_0_1_0_as_the_same_store_of_this_version(
    toy_rubric: Rubric, toy_corpus: TextTable, cue_backend: Factory, tmp_path: Path
) -> None:
    table = first_texts(toy_corpus, 4)
    store = tmp_path / "run.sqlite"
    code_texts(table, toy_rubric, cue_backend(), store, TWO_STAGE)
    refs = unit_references(toy_rubric, STUB_NAME, STUB_DIMENSION)
    current = score_hybrid(store, table, toy_rubric, StubEncoder(), refs, min_words=1)
    as_stored_by_0_1_0(store)
    stored = score_hybrid(store, table, toy_rubric, StubEncoder(), refs, min_words=1)
    pd.testing.assert_frame_equal(stored.frame, current.frame)
    assert stored.provenance["store"]["library_version"] == PUBLISHED
    assert stored.provenance["library_version"] == copernicus.__version__


def test_a_one_stage_store_of_0_1_0_of_a_rubric_listed_upward_is_refused_as_before(
    toy_rubric: Rubric, toy_corpus: TextTable, cue_backend: Factory, tmp_path: Path
) -> None:
    table = first_texts(toy_corpus, 2)
    store = tmp_path / "run.sqlite"
    code_texts(table, toy_rubric, cue_backend(), store, ONE_STAGE)
    as_written_by_0_1_0(store)
    as_stored_by_0_1_0(store)
    scores = export_scores(store)
    backend = cue_backend()
    with pytest.raises(StoreError) as refused:
        code_texts(table, toy_rubric, backend, store, ONE_STAGE)
    assert str(refused.value) == MISSING
    assert backend.calls == []
    pd.testing.assert_frame_equal(export_scores(store), scores)
