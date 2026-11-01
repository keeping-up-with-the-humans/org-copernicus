"""The output files of the commands are written through a partial file, so a failed write keeps
the old file and leaves no partial one."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from copernicus.cli._output import format_rows, write_report, write_table

OLD = pd.DataFrame({"a": [1, 2]})
NEW = pd.DataFrame({"a": [3, 4]})


def half_written(self: pd.DataFrame, target: Any, **_: Any) -> None:
    """Write a fragment to the target, as a writer stopped by a full disk would, then fail."""
    if isinstance(target, str | Path):
        Path(target).write_bytes(b"half")
    else:
        target.write(b"half")
    raise OSError("disk full")


@pytest.mark.parametrize(("suffix", "writer"), [(".csv", "to_csv"), (".parquet", "to_parquet")])
def test_a_failed_table_write_keeps_the_old_table_and_leaves_no_partial_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, suffix: str, writer: str
) -> None:
    out = tmp_path / f"scores{suffix}"
    write_table(OLD, out)
    before = out.read_bytes()
    monkeypatch.setattr(pd.DataFrame, writer, half_written)
    with pytest.raises(OSError, match="disk full"):
        write_table(NEW, out)
    assert out.read_bytes() == before
    assert [path.name for path in tmp_path.iterdir()] == [out.name]


def test_a_failed_report_write_keeps_the_old_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = tmp_path / "panel.txt"
    write_report(report, "old page\n")
    monkeypatch.setattr("copernicus._io.os.replace", half_written_replace)
    with pytest.raises(OSError, match="disk full"):
        write_report(report, "new page\n")
    assert report.read_text(encoding="utf-8") == "old page\n"
    assert [path.name for path in tmp_path.iterdir()] == ["panel.txt"]


def half_written_replace(source: Any, target: Any) -> None:
    """Fail where the partial file would be moved into place."""
    raise OSError("disk full")


def test_no_summary_label_runs_into_its_value() -> None:
    """The values of a summary start in one column, at least one space after the longest label."""
    rows = [
        ("n_texts", "6"),
        ("mean_words_evidence", "57.2"),
        ("n_passages_embedded", "127"),
        ("n_distinct_texts_embedded", "103"),
        ("description file", "hybrid.parquet.json"),
    ]
    lines = format_rows(rows).splitlines()
    assert len({len(line) - len(value) for line, (_, value) in zip(lines, rows, strict=True)}) == 1
    for line, (label, value) in zip(lines, rows, strict=True):
        assert line.startswith(f"{label} ") and line.endswith(value)
        assert line[len(label) :].strip() == value
    assert lines[3] == "n_distinct_texts_embedded 103"
    short = [("texts", "6"), ("completion tokens", "120")]
    assert format_rows(short).splitlines() == [f"{label:<18}{value}" for label, value in short]
