"""A resumed run refuses texts that changed under their ids, before any backend call."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from contextlib import closing
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from copernicus.coders import CodingSettings, code_texts, export_evidence, export_scores
from copernicus.data import TextTable
from copernicus.errors import StoreError
from copernicus.llm import FakeBackend
from copernicus.rubric.model import Rubric

Factory = Callable[[], FakeBackend]
NEW_TEXT = "The package arrived on Tuesday near the eastern gate."
REMEDY = (
    "the texts changed since the store was written: a store binds every text id to the text "
    "it was coded on; code the changed texts into a new store, or restore the texts"
)
COLLAPSED_FORM = (
    "coded on the text with every run of whitespace collapsed to one space, as an earlier "
    "version of the library sent every text"
)
COLLAPSED_REMEDY = (
    "this version of the library sends each text as given, with its line breaks, tabs and "
    "runs of spaces, so the prompt of such a text has changed: code these texts into a new "
    "store, or pass each of them with its whitespace collapsed "
    "(copernicus.data.table.collapse_whitespace) to continue the old coding"
)
FAILED = {"extraction_failed", "scoring_failed"}
# Coded once: run 1 of "gained" and "lost" succeeds, run 2 of them and both runs of "failed" fail.
CODED = {
    "failed": "Every attempt at coding this answer failed.",
    "gained": "Two spaces sit between these words.",
    "lost": "Two  spaces sit between these words.",
}
# The same ids, each text with whitespace gained between its words or lost there.
REVISED = {
    "failed": "Every attempt at coding  this answer failed.",
    "gained": "Two  spaces sit between these words.",
    "lost": "Two spaces sit between these words.",
}


def with_texts(table: TextTable, texts: dict[str, str]) -> TextTable:
    """Return the table with the text of each named id replaced."""
    frame = table.frame.copy()
    for text_id, text in texts.items():
        frame.loc[frame["text_id"] == text_id, "text"] = text
    return TextTable.from_frame(frame, group_id="group_id")


def table_of(texts: dict[str, str]) -> TextTable:
    """Return a table of the texts, all in one group."""
    frame = pd.DataFrame({"text_id": list(texts), "text": list(texts.values()), "group_id": "g"})
    return TextTable.from_frame(frame)


def store_with_failed_runs(
    rubric: Rubric, cue_backend: Factory, store: Path, method: str
) -> CodingSettings:
    """Code `CODED` with runs that succeeded and runs that failed; return the retry settings."""
    first = CodingSettings(method=method, runs=1, temperature=0.3)
    succeeded = table_of({text_id: CODED[text_id] for text_id in ("gained", "lost")})
    code_texts(succeeded, rubric, cue_backend(), store, first)
    both = replace(first, runs=2, max_failure_share=1.0)
    code_texts(table_of(CODED), rubric, FakeBackend(lambda request: "not JSON"), store, both)
    statuses = export_scores(store).groupby(["text_id", "run_number"])["status"].agg(set)
    failed = {cell for cell, values in statuses.items() if values & FAILED}
    assert failed == {("failed", 1), ("failed", 2), ("gained", 2), ("lost", 2)}
    return replace(both, retry_failed=True)


def test_a_resume_refuses_texts_that_changed_under_their_ids(
    toy_rubric: Rubric, toy_corpus: TextTable, cue_backend: Factory, tmp_path: Path
) -> None:
    store = tmp_path / "run.sqlite"
    table = toy_corpus.subset(["t052", "t053"])
    settings = CodingSettings(runs=1)
    code_texts(table, toy_rubric, cue_backend(), store, settings)
    scores, evidence = export_scores(store), export_evidence(store)
    backend = cue_backend()
    changed = with_texts(table, {"t052": NEW_TEXT, "t053": NEW_TEXT})
    with pytest.raises(StoreError) as refused:
        code_texts(changed, toy_rubric, backend, store, settings)
    message = str(refused.value)
    assert REMEDY in message and "'t052', 't053'" in message and str(store) in message
    assert backend.calls == []
    pd.testing.assert_frame_equal(export_scores(store), scores)
    pd.testing.assert_frame_equal(export_evidence(store), evidence)
    unchanged = cue_backend()
    resumed = code_texts(table, toy_rubric, unchanged, store, settings)
    assert (resumed.resumed, resumed.n_texts, resumed.n_attribute_calls) == (2, 0, 0)
    assert unchanged.calls == []


def test_only_the_changed_texts_are_named_the_first_ten_of_them(
    toy_rubric: Rubric, toy_corpus: TextTable, cue_backend: Factory, tmp_path: Path
) -> None:
    store = tmp_path / "run.sqlite"
    ids = sorted(str(text_id) for text_id in toy_corpus.ids())[:14]
    table = toy_corpus.subset(ids)
    code_texts(table, toy_rubric, cue_backend(), store, CodingSettings(runs=1))
    one = with_texts(table, {ids[5]: NEW_TEXT})
    with pytest.raises(StoreError, match=REMEDY) as refused:
        code_texts(one, toy_rubric, cue_backend(), store, CodingSettings(runs=1))
    assert f"1 changed text id: '{ids[5]}')" in str(refused.value)
    many = with_texts(table, dict.fromkeys(ids[2:14], NEW_TEXT))
    with pytest.raises(StoreError, match=REMEDY) as refused:
        code_texts(many, toy_rubric, cue_backend(), store, CodingSettings(runs=1))
    message = str(refused.value)
    assert "12 changed text ids: " in message and "and 2 more" in message
    assert all(f"'{text_id}'" in message for text_id in ids[2:12])
    assert not any(f"'{text_id}'" in message for text_id in [*ids[:2], *ids[12:]])


def test_whitespace_at_the_ends_leaves_the_resume_as_it_was_and_between_words_is_refused(
    toy_rubric: Rubric, toy_corpus: TextTable, cue_backend: Factory, tmp_path: Path
) -> None:
    store = tmp_path / "run.sqlite"
    table = toy_corpus.subset(["t001", "t002"])
    code_texts(table, toy_rubric, cue_backend(), store, CodingSettings(runs=1))
    padded = {text_id: "  " + table.text_of(text_id) + " \n\t " for text_id in ("t001", "t002")}
    resumed = code_texts(
        with_texts(table, padded), toy_rubric, cue_backend(), store, CodingSettings(runs=1)
    )
    assert (resumed.resumed, resumed.n_attribute_calls) == (2, 0)
    spaced = {
        text_id: table.text_of(text_id).replace(" ", " \n\t ") for text_id in ("t001", "t002")
    }
    backend = cue_backend()
    with pytest.raises(StoreError) as refused:
        code_texts(with_texts(table, spaced), toy_rubric, backend, store, CodingSettings(runs=1))
    message = str(refused.value)
    assert message == (
        f"2 text ids of the store {store} were {COLLAPSED_FORM}: 't001', 't002'; {COLLAPSED_REMEDY}"
    )
    assert REMEDY not in message and backend.calls == []


@pytest.mark.parametrize("method", ["two_stage", "one_stage"])
def test_a_retry_of_failed_cells_refuses_texts_whose_whitespace_changed_before_any_call(
    toy_rubric: Rubric, cue_backend: Factory, tmp_path: Path, method: str
) -> None:
    store = tmp_path / "run.sqlite"
    retry = store_with_failed_runs(toy_rubric, cue_backend, store, method)
    scores, evidence = export_scores(store), export_evidence(store)
    backend = cue_backend()
    with pytest.raises(StoreError) as refused:
        code_texts(table_of(REVISED), toy_rubric, backend, store, retry)
    assert str(refused.value) == (
        f"{REMEDY} (store {store}; 1 changed text id: 'lost'); 2 text ids of the store {store} "
        f"were {COLLAPSED_FORM}: 'failed', 'gained'; {COLLAPSED_REMEDY}"
    )
    assert backend.calls == []
    pd.testing.assert_frame_equal(export_scores(store), scores)
    pd.testing.assert_frame_equal(export_evidence(store), evidence)


@pytest.mark.parametrize("method", ["two_stage", "one_stage"])
def test_a_retry_of_failed_cells_codes_them_again_when_their_texts_are_unchanged(
    toy_rubric: Rubric, cue_backend: Factory, tmp_path: Path, method: str
) -> None:
    store = tmp_path / "run.sqlite"
    retry = store_with_failed_runs(toy_rubric, cue_backend, store, method)
    backend = cue_backend()
    summary = code_texts(table_of(CODED), toy_rubric, backend, store, retry)
    assert (summary.retried, summary.resumed) == (4, 2) and backend.calls != []
    assert not set(export_scores(store)["status"]) & FAILED


def test_a_changed_text_outside_the_requested_range_is_not_checked(
    toy_rubric: Rubric, toy_corpus: TextTable, cue_backend: Factory, tmp_path: Path
) -> None:
    store = tmp_path / "run.sqlite"
    table = toy_corpus.subset(["t001", "t002", "t003", "t004"])
    code_texts(table, toy_rubric, cue_backend(), store, CodingSettings(runs=1))
    changed = with_texts(table, {"t004": NEW_TEXT})
    first_three = CodingSettings(runs=1, text_range=(0, 3))
    resumed = code_texts(changed, toy_rubric, cue_backend(), store, first_three)
    assert (resumed.resumed, resumed.n_attribute_calls) == (3, 0)
    with pytest.raises(StoreError, match="1 changed text id: 't004'"):
        code_texts(changed, toy_rubric, cue_backend(), store, CodingSettings(runs=1))


def test_every_stored_run_of_a_requested_text_binds_it_not_only_the_requested_runs(
    toy_rubric: Rubric, toy_corpus: TextTable, cue_backend: Factory, tmp_path: Path
) -> None:
    store = tmp_path / "run.sqlite"
    table = toy_corpus.subset(["t001"])
    code_texts(table, toy_rubric, cue_backend(), store, CodingSettings(runs=2, temperature=0.3))
    # The store is edited to hold run 2 of the text only, so run 1 is a missing cell.
    with closing(sqlite3.connect(store)) as connection, connection:
        for name in ("evidence", "results"):
            connection.execute(f'DELETE FROM "{name}" WHERE "run_number" = 1')
    backend = cue_backend()
    with pytest.raises(StoreError, match="1 changed text id: 't001'"):
        code_texts(
            with_texts(table, {"t001": NEW_TEXT}), toy_rubric, backend, store,
            CodingSettings(runs=1, temperature=0.3),
        )  # fmt: skip
    assert backend.calls == []
    assert export_scores(store)["run_number"].unique().tolist() == [2]
