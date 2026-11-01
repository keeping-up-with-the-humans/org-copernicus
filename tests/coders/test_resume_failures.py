"""The failure ceiling judges the requested scope, stored results included, and failed cells
are coded again only on request."""

from __future__ import annotations

import dataclasses
import sqlite3
from collections.abc import Callable
from contextlib import closing
from pathlib import Path

import pandas as pd
import pytest

from copernicus.coders import CodingSettings, code_texts, export_evidence, export_scores
from copernicus.data import TextTable
from copernicus.errors import RunFailureCeilingError
from copernicus.llm import FakeBackend
from copernicus.rubric.model import Rubric

Factory = Callable[[], FakeBackend]
ADVICE = (
    "set retry_failed=True to recode them, raise max_failure_share to accept them, "
    "or code into a new store"
)
STATUSES = ("ok", "no_evidence", "too_short", "extraction_failed", "scoring_failed")


def failing() -> FakeBackend:
    """Return a backend whose every answer is empty, so every extraction fails."""
    return FakeBackend(lambda request: "")


def row_count(path: Path, table: str) -> int:
    with closing(sqlite3.connect(path)) as connection:
        return int(connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])


def failed_store(table: TextTable, rubric: Rubric, path: Path, settings: CodingSettings) -> Path:
    """Code the table with a backend that fails every extraction, as the ceiling stops it."""
    with pytest.raises(RunFailureCeilingError) as stopped:
        code_texts(table, rubric, failing(), path, settings)
    assert stopped.value.summary.status_counts["extraction_failed"] == len(table) * 3
    assert sum(stopped.value.summary.stored_status_counts.values()) == 0
    return path


def test_a_no_op_resume_over_failed_cells_still_stops_at_the_ceiling(
    toy_rubric: Rubric, toy_corpus: TextTable, cue_backend: Factory, tmp_path: Path
) -> None:
    table = toy_corpus.subset(["t052", "t053"])
    settings = CodingSettings(runs=1)
    store = failed_store(table, toy_rubric, tmp_path / "failed.sqlite", settings)
    backend = cue_backend()
    with pytest.raises(RunFailureCeilingError) as again:
        code_texts(table, toy_rubric, backend, store, settings)
    summary = again.value.summary
    assert summary.stored_status_counts["extraction_failed"] == 6
    assert sum(summary.stored_status_counts.values()) == 6
    assert set(summary.status_counts.values()) == {0} and summary.n_attribute_calls == 0
    assert (summary.n_texts, summary.resumed, summary.retried) == (0, 2, 0)
    assert backend.calls == []
    message = str(again.value)
    assert "6 of 6 attribute results in the requested scope failed, 6 of them stored before " \
        "this invocation (share 1.000 above the ceiling 0.1)" in message  # fmt: skip
    assert ADVICE in message and str(store) in message
    assert export_scores(store)["status"].eq("extraction_failed").all()


def test_a_successful_no_op_resume_returns_and_reports_the_stored_statuses(
    toy_rubric: Rubric, toy_corpus: TextTable, cue_backend: Factory, tmp_path: Path
) -> None:
    store = tmp_path / "run.sqlite"
    table = toy_corpus.subset(["t052", "t053"])
    code_texts(table, toy_rubric, cue_backend(), store, CodingSettings(runs=1))
    summary = code_texts(table, toy_rubric, cue_backend(), store, CodingSettings(runs=1))
    stored = export_scores(store)["status"].value_counts()
    expected = {status: int(stored[status]) if status in stored else 0 for status in STATUSES}
    assert dict(summary.stored_status_counts) == expected
    assert set(summary.status_counts.values()) == {0}
    assert (summary.resumed, summary.retried, summary.n_attribute_calls) == (2, 0, 0)


def test_failed_runs_outside_the_requested_runs_are_not_judged(
    toy_rubric: Rubric, toy_corpus: TextTable, cue_backend: Factory, tmp_path: Path
) -> None:
    store = tmp_path / "run.sqlite"
    table = toy_corpus.subset(["t052", "t053"])
    one, two = CodingSettings(runs=1, temperature=0.3), CodingSettings(runs=2, temperature=0.3)
    code_texts(table, toy_rubric, cue_backend(), store, one)
    with pytest.raises(RunFailureCeilingError) as second:
        code_texts(table, toy_rubric, failing(), store, two)
    assert "6 of 12 attribute results in the requested scope failed, 0 of them stored" in str(
        second.value
    )
    first_run = code_texts(table, toy_rubric, cue_backend(), store, one)
    assert first_run.stored_status_counts["extraction_failed"] == 0
    assert sum(first_run.stored_status_counts.values()) == 6
    with pytest.raises(RunFailureCeilingError) as both:
        code_texts(table, toy_rubric, cue_backend(), store, two)
    assert "6 of 12 attribute results in the requested scope failed, 6 of them stored" in str(
        both.value
    )


def test_failed_texts_outside_the_text_range_are_not_judged(
    toy_rubric: Rubric, toy_corpus: TextTable, cue_backend: Factory, tmp_path: Path
) -> None:
    table = toy_corpus.subset(["t001", "t002", "t003"])
    store = failed_store(
        table.subset(["t001"]), toy_rubric, tmp_path / "r.sqlite", CodingSettings(runs=1)
    )
    rest = CodingSettings(runs=1, text_range=(1, 3))
    coded = code_texts(table, toy_rubric, cue_backend(), store, rest)
    assert (coded.n_attribute_calls, sum(coded.stored_status_counts.values())) == (6, 0)
    assert code_texts(table, toy_rubric, cue_backend(), store, rest).resumed == 2
    with pytest.raises(RunFailureCeilingError) as whole:
        code_texts(table, toy_rubric, cue_backend(), store, CodingSettings(runs=1))
    assert "3 of 9 attribute results in the requested scope failed, 3 of them stored" in str(
        whole.value
    )


def test_a_broken_backend_stops_at_the_first_judged_batch_of_a_resume_over_many_stored_cells(
    toy_rubric: Rubric, toy_corpus: TextTable, cue_backend: Factory, tmp_path: Path
) -> None:
    store = tmp_path / "r.sqlite"
    stored = CodingSettings(runs=4, temperature=0.3)
    code_texts(toy_corpus, toy_rubric, cue_backend(), store, stored)
    backend = failing()
    resume = CodingSettings(runs=5, temperature=0.3, save_every=2)
    with pytest.raises(RunFailureCeilingError) as error:
        code_texts(toy_corpus, toy_rubric, backend, store, resume)
    summary = error.value.summary
    # 216 stored cells hold 648 good results. The checks during the run judge this
    # invocation's calls alone, so the run stops at the first judged batch: five batches of
    # two texts and three attributes. Judged with the stored results, 30 of 678 would pass
    # and the stop would come only at 78 calls.
    assert sum(summary.stored_status_counts.values()) == 648
    assert summary.stored_status_counts["extraction_failed"] == 0
    assert (summary.n_texts, summary.n_attribute_calls, summary.resumed) == (10, 30, 216)
    assert len(backend.calls) == 30 * 3
    message = str(error.value)
    assert "30 of 30 attribute calls in this invocation failed (share 1.000 above" in message
    assert ADVICE in message and str(store) in message


def test_the_final_check_counts_the_stored_failures_the_checks_during_the_run_leave_out(
    toy_rubric: Rubric, toy_corpus: TextTable, cue_backend: Factory, tmp_path: Path
) -> None:
    ids = sorted(str(text_id) for text_id in toy_corpus.ids())
    store = failed_store(
        toy_corpus.subset(ids[:10]), toy_rubric, tmp_path / "r.sqlite", CodingSettings(runs=1)
    )
    with pytest.raises(RunFailureCeilingError) as error:
        code_texts(toy_corpus, toy_rubric, cue_backend(), store, CodingSettings(runs=1))
    summary = error.value.summary
    # Every check during the run sees no failed call, so the 44 new texts are coded in full;
    # the check at the end counts the 30 stored failures as well: 30 of 162 is above 0.1.
    assert (summary.n_texts, summary.n_attribute_calls, summary.resumed) == (44, 132, 10)
    assert summary.status_counts["extraction_failed"] == 0
    assert summary.stored_status_counts["extraction_failed"] == 30
    assert "30 of 162 attribute results in the requested scope failed, 30 of them stored" in str(
        error.value
    )


def test_retry_failed_codes_the_failed_cells_again_and_replaces_them(
    toy_rubric: Rubric, toy_corpus: TextTable, cue_backend: Factory, tmp_path: Path
) -> None:
    table = toy_corpus.subset(["t052", "t053"])
    settings = CodingSettings(runs=1)
    store = failed_store(table, toy_rubric, tmp_path / "failed.sqlite", settings)
    backend = cue_backend()
    retry = dataclasses.replace(settings, retry_failed=True)
    summary = code_texts(table, toy_rubric, backend, store, retry)
    assert (summary.retried, summary.resumed, summary.n_texts) == (2, 0, 2)
    assert summary.n_attribute_calls == 6 and summary.status_counts["extraction_failed"] == 0
    assert sum(summary.stored_status_counts.values()) == 0
    fresh, fresh_backend = tmp_path / "fresh.sqlite", cue_backend()
    code_texts(table, toy_rubric, fresh_backend, fresh, settings)
    assert len(backend.calls) == len(fresh_backend.calls) > 6
    pd.testing.assert_frame_equal(export_scores(store), export_scores(fresh))
    pd.testing.assert_frame_equal(export_evidence(store), export_evidence(fresh))
    assert export_scores(store)["score"].notna().all()
    assert row_count(store, "results") == 2
    assert row_count(store, "evidence") == row_count(fresh, "evidence") > 0
    again = code_texts(table, toy_rubric, cue_backend(), store, retry)
    assert (again.retried, again.resumed, again.n_attribute_calls) == (0, 2, 0)


def test_retry_failed_codes_only_the_cells_with_a_failed_attribute(
    toy_rubric: Rubric, toy_corpus: TextTable, cue_backend: Factory, tmp_path: Path
) -> None:
    table = toy_corpus.subset(["t001", "t002", "t003"])
    store = failed_store(
        table.subset(["t002"]), toy_rubric, tmp_path / "r.sqlite", CodingSettings(runs=1)
    )
    backend = cue_backend()
    retry = CodingSettings(runs=1, retry_failed=True)
    summary = code_texts(table, toy_rubric, backend, store, retry)
    assert (summary.retried, summary.resumed, summary.n_texts) == (1, 0, 3)
    assert summary.n_attribute_calls == 9
    before = export_scores(store)
    second = code_texts(table, toy_rubric, cue_backend(), store, retry)
    assert (second.retried, second.resumed, second.n_attribute_calls) == (0, 3, 0)
    pd.testing.assert_frame_equal(export_scores(store), before)
