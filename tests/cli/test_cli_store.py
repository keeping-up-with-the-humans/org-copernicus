"""Tests for the store commands of the command-line interface."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from copernicus.cli import app
from copernicus.rubric.loader import toy_rubric_path

N_TEXTS = 6


def code_shard(runner: CliRunner, texts: Path, out: Path, *extra: str) -> None:
    """Code the texts into a store with the fake backend, failing the test on an error."""
    result = runner.invoke(
        app,
        [
            *("code", "two-stage", "--texts", str(texts), "--rubric", str(toy_rubric_path())),
            *("--out", str(out), "--backend", "fake", "--group-column", "group_id"),
            *extra,
        ],
    )
    assert result.exit_code == 0, result.output


@pytest.fixture
def coded_store(runner: CliRunner, texts_csv: Path, tmp_path: Path) -> Path:
    """Return a store holding all six texts."""
    store = tmp_path / "full.sqlite"
    code_shard(runner, texts_csv, store)
    return store


def test_export_text_scores_to_csv(runner: CliRunner, coded_store: Path, tmp_path: Path) -> None:
    out = tmp_path / "scores.csv"
    result = runner.invoke(
        app, ["store", "export", str(coded_store), "--what", "text", "--out", str(out)]
    )
    assert result.exit_code == 0, result.output
    assert result.stdout.strip() == f"wrote {N_TEXTS} rows to {out}"
    assert len(pd.read_csv(out)) == N_TEXTS


@pytest.mark.parametrize(
    ("what", "minimum_rows"), [("scores", 1), ("construct", 1), ("evidence", 1)]
)
def test_export_other_tables_to_parquet(
    runner: CliRunner, coded_store: Path, tmp_path: Path, what: str, minimum_rows: int
) -> None:
    out = tmp_path / f"{what}.parquet"
    result = runner.invoke(
        app, ["store", "export", str(coded_store), "--what", what, "--out", str(out)]
    )
    assert result.exit_code == 0, result.output
    assert len(pd.read_parquet(out)) >= minimum_rows


def test_export_provenance_writes_key_value_csv(
    runner: CliRunner, coded_store: Path, tmp_path: Path
) -> None:
    out = tmp_path / "provenance.csv"
    result = runner.invoke(
        app, ["store", "export", str(coded_store), "--what", "provenance", "--out", str(out)]
    )
    assert result.exit_code == 0, result.output
    frame = pd.read_csv(out)
    assert list(frame.columns) == ["key", "value"]
    assert frame.set_index("key").loc["method", "value"] == "two_stage"


def test_export_with_unknown_suffix_is_an_error(
    runner: CliRunner, coded_store: Path, tmp_path: Path
) -> None:
    out = tmp_path / "scores.xlsx"
    result = runner.invoke(
        app, ["store", "export", str(coded_store), "--what", "text", "--out", str(out)]
    )
    assert result.exit_code == 1
    assert result.stderr.startswith("error:") and ".xlsx" in result.stderr
    assert not out.exists()


def test_export_of_a_file_that_is_no_store_is_an_error(
    runner: CliRunner, texts_csv: Path, tmp_path: Path
) -> None:
    result = runner.invoke(
        app,
        ["store", "export", str(texts_csv), "--what", "text", "--out", str(tmp_path / "o.csv")],
    )
    assert result.exit_code == 1
    assert result.stderr.startswith("error:")


def test_export_into_a_missing_folder_is_an_error(
    runner: CliRunner, coded_store: Path, tmp_path: Path
) -> None:
    out = tmp_path / "missing" / "scores.csv"
    result = runner.invoke(
        app, ["store", "export", str(coded_store), "--what", "text", "--out", str(out)]
    )
    assert result.exit_code == 1
    assert result.stderr.startswith("error:")


def test_merge_of_two_range_shards_holds_all_texts(
    runner: CliRunner, texts_csv: Path, tmp_path: Path
) -> None:
    first, second, merged = (tmp_path / name for name in ("a.sqlite", "b.sqlite", "m.sqlite"))
    code_shard(runner, texts_csv, first, "--range", "0:3")
    code_shard(runner, texts_csv, second, "--range", "3:6")
    result = runner.invoke(app, ["store", "merge", str(first), str(second), "--out", str(merged)])
    assert result.exit_code == 0, result.output
    counts = dict(line.split(": ") for line in result.stdout.splitlines())
    assert counts["conflicting_cells"] == "0"
    assert int(counts["results"]) == N_TEXTS * 2
    out = tmp_path / "merged.csv"
    exported = runner.invoke(
        app, ["store", "export", str(merged), "--what", "text", "--out", str(out)]
    )
    assert exported.exit_code == 0, exported.output
    assert len(pd.read_csv(out)) == N_TEXTS


def test_merge_with_target_among_sources_is_an_error(runner: CliRunner, coded_store: Path) -> None:
    result = runner.invoke(app, ["store", "merge", str(coded_store), "--out", str(coded_store)])
    assert result.exit_code == 1
    assert result.stderr.startswith("error:")


def test_merge_into_a_file_that_is_no_store_is_an_error(
    runner: CliRunner, coded_store: Path, texts_csv: Path
) -> None:
    before = texts_csv.read_bytes()
    result = runner.invoke(app, ["store", "merge", str(coded_store), "--out", str(texts_csv)])
    assert result.exit_code == 1
    assert result.stderr.startswith("error: file is not a run store")
    assert texts_csv.read_bytes() == before
