"""Tests of the table hash of a reference set file: the description names the table it describes.

The table is written first and the description last, each atomically. A save stopped between
the two writes leaves a new table beside the old description, of the same size when a rubric
rewritten under the same ids keeps its cells. The description records the digest of its table
(``table_hash``), so the next load refuses such a pair instead of reading the new texts under
the old rubric hash; ``bind_to`` accepts a table only when the description records no digest.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import pandas as pd
import pytest
from references_support import clean_responder, fail_write, texts_json, writer

import copernicus.references.files as reference_files
from copernicus import __version__
from copernicus._digests import table_hash
from copernicus.errors import ConfigurationError
from copernicus.references import GeneratorSettings, ReferenceSet, generate_references
from copernicus.references.reference_set import FRAME_COLUMNS
from copernicus.rubric.model import Rubric

SETTINGS = GeneratorSettings(texts_per_call=1, max_retries=1)
SUFFIXES = [".csv", ".parquet"]
CELL_COUNT = 24
ODD_TEXT = '  A first line,\nthen "a quote", a\ttab, a comma\r\nand outer spaces  '
DIGEST = re.compile("[0-9a-f]{64}")


def description_of(path: Path) -> dict[str, object]:
    """Read the description beside a table."""
    return json.loads(path.with_name(path.name + ".json").read_text(encoding="utf-8"))


def edit_description(path: Path, drop: tuple[str, ...] = (), **values: object) -> None:
    """Drop keys from the description beside a table, or set them, as another version would."""
    kept = {key: value for key, value in description_of(path).items() if key not in drop}
    path.with_name(path.name + ".json").write_text(json.dumps({**kept, **values}), encoding="utf-8")


def table_texts(path: Path) -> list[str]:
    """Read the texts of a table with pandas alone, as the file holds them."""
    if path.suffix == ".parquet":
        return pd.read_parquet(path)["text"].tolist()
    return pd.read_csv(path, keep_default_na=False)["text"].tolist()


@pytest.fixture
def old_set(toy_rubric: Rubric) -> ReferenceSet:
    """One text per cell from one writer, written for the toy rubric."""
    return generate_references(toy_rubric, [writer(clean_responder(1))], SETTINGS)


@pytest.fixture
def revised_set(rewritten_toy_rubric: Rubric) -> ReferenceSet:
    """One other text per cell, written for the toy rubric rewritten under the same ids."""
    revised = writer(lambda request: texts_json(1, f"revised-{request.seed}"))
    return generate_references(rewritten_toy_rubric, [revised], SETTINGS)


@pytest.mark.parametrize("suffix", SUFFIXES)
def test_the_description_names_the_table_it_was_written_with(
    old_set: ReferenceSet, tmp_path: Path, suffix: str
) -> None:
    path = old_set.save(tmp_path / f"references{suffix}")
    recorded = description_of(path)["table_hash"]
    assert isinstance(recorded, str) and DIGEST.fullmatch(recorded)
    assert recorded == table_hash(old_set.frame, FRAME_COLUMNS)
    loaded = ReferenceSet.load(path)
    pd.testing.assert_frame_equal(loaded.frame, old_set.frame)
    assert table_hash(loaded.frame, FRAME_COLUMNS) == recorded
    assert "table_hash" not in old_set.description()


@pytest.mark.parametrize("suffix", SUFFIXES)
def test_an_overwrite_stopped_before_its_description_is_refused_by_the_next_load(
    old_set: ReferenceSet,
    revised_set: ReferenceSet,
    toy_rubric: Rubric,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    suffix: str,
) -> None:
    path = old_set.save(tmp_path / f"references{suffix}")
    assert len(revised_set.frame) == len(old_set.frame) == CELL_COUNT
    fail_write(monkeypatch, reference_files, "write_json")
    with pytest.raises(OSError, match="disk full"):
        revised_set.save(path)
    assert table_texts(path) == revised_set.frame["text"].tolist()
    assert description_of(path)["rubric_hash"] == toy_rubric.content_hash()
    for bind_to in (None, toy_rubric):
        with pytest.raises(ConfigurationError, match="do not belong together") as refused:
            ReferenceSet.load(path, bind_to=bind_to)
        message = str(refused.value)
        assert message.startswith(f"{path} and {path.name}.json do not belong together")
        assert f"the table holds {CELL_COUNT} rows with hash" in message
        assert "a save stopped between its two writes leaves such a pair" in message


def test_a_failed_table_write_keeps_the_old_table_and_its_description(
    old_set: ReferenceSet,
    revised_set: ReferenceSet,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = old_set.save(tmp_path / "references.csv")
    before = {file.name: file.read_bytes() for file in tmp_path.iterdir()}
    fail_write(monkeypatch, reference_files, "write_atomically")
    with pytest.raises(OSError, match="disk full"):
        revised_set.save(path)
    assert {file.name: file.read_bytes() for file in tmp_path.iterdir()} == before
    loaded = ReferenceSet.load(path)
    pd.testing.assert_frame_equal(loaded.frame, old_set.frame)
    assert table_hash(loaded.frame, FRAME_COLUMNS) == description_of(path)["table_hash"]
    assert loaded.rubric_hash == old_set.rubric_hash


@pytest.mark.parametrize(
    ("dropped", "named", "bound_with"),
    [
        (("table_hash",), "table_hash", "generated"),
        (("rubric_hash", "bound_with", "table_hash"), "rubric_hash", "for_rubric"),
    ],
    ids=["no table hash", "no binding and no table hash"],
)
def test_a_set_file_without_a_table_hash_is_read_only_when_bound_on_request(
    old_set: ReferenceSet,
    toy_rubric: Rubric,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    dropped: tuple[str, ...],
    named: str,
    bound_with: str,
) -> None:
    path = old_set.save(tmp_path / "references.csv")
    edit_description(path, drop=dropped)
    with pytest.raises(ConfigurationError) as refused:
        ReferenceSet.load(path)
    assert str(refused.value) == (
        f"{path}.json has no {named}: it was written by an earlier version of copernicus than "
        f"this one ({__version__}); if {path.name} is the table this description was written "
        "for and the rubric is unchanged since its texts were written, bind the set to it with "
        f"ReferenceSet.load(path, bind_to=rubric) and save it again; otherwise regenerate "
        f"{path.name}"
    )
    with caplog.at_level(logging.INFO, logger="copernicus.references.binding"):
        bound = ReferenceSet.load(path, bind_to=toy_rubric)
    accepted = [record for record in caplog.records if "table hash" in record.getMessage()]
    assert [(record.levelno, record.getMessage()) for record in accepted] == [
        (
            logging.INFO,
            f"bound the reference set {path} to rubric {toy_rubric.rubric_id} with hash "
            f"{toy_rubric.content_hash()} and accepted its table of {CELL_COUNT} rows without a "
            "recorded table hash on request",
        )
    ]
    assert (bound.rubric_hash, bound.bound_with) == (toy_rubric.content_hash(), bound_with)
    pd.testing.assert_frame_equal(bound.frame, old_set.frame)
    resaved = bound.save(tmp_path / "resaved.csv")
    assert description_of(resaved)["table_hash"] == table_hash(old_set.frame, FRAME_COLUMNS)
    assert ReferenceSet.load(resaved).rubric_hash == toy_rubric.content_hash()


def test_a_recorded_table_hash_that_differs_is_refused_even_with_bind_to(
    old_set: ReferenceSet, toy_rubric: Rubric, tmp_path: Path
) -> None:
    path = old_set.save(tmp_path / "references.parquet")
    other = "0" * 64
    edit_description(path, table_hash=other)
    for bind_to in (None, toy_rubric):
        with pytest.raises(ConfigurationError, match="do not belong together") as refused:
            ReferenceSet.load(path, bind_to=bind_to)
        assert f"the description records {CELL_COUNT} rows with hash {other}" in str(refused.value)
    edit_description(path, drop=("rubric_hash", "bound_with"))
    with pytest.raises(ConfigurationError, match="do not belong together"):
        ReferenceSet.load(path, bind_to=toy_rubric)


@pytest.mark.parametrize("value", ["", "A" * 64, "0" * 63, "g" * 64, 12345, None])
def test_a_malformed_table_hash_is_refused_naming_the_key(
    old_set: ReferenceSet, toy_rubric: Rubric, tmp_path: Path, value: object
) -> None:
    path = old_set.save(tmp_path / "references.parquet")
    edit_description(path, table_hash=value)
    with pytest.raises(ConfigurationError, match="table_hash must be") as refused:
        ReferenceSet.load(path, bind_to=toy_rubric)
    assert "references.parquet.json" in str(refused.value) and repr(value) in str(refused.value)


def test_a_float_of_seventeen_significant_digits_comes_back_from_csv_exactly(
    old_set: ReferenceSet, toy_rubric: Rubric, tmp_path: Path
) -> None:
    frame = old_set.frame.assign(temperature_applied=0.1 + 0.2)
    path = ReferenceSet.for_rubric(frame, toy_rubric, SETTINGS).save(tmp_path / "references.csv")
    loaded = ReferenceSet.load(path)
    assert loaded.frame["temperature_applied"].tolist() == [0.1 + 0.2] * CELL_COUNT
    digest = table_hash(loaded.frame, FRAME_COLUMNS)
    assert digest == description_of(path)["table_hash"] == table_hash(frame, FRAME_COLUMNS)


@pytest.mark.parametrize("suffix", SUFFIXES)
def test_a_text_with_line_breaks_quotes_commas_tabs_and_outer_spaces_comes_back_exactly(
    old_set: ReferenceSet, toy_rubric: Rubric, tmp_path: Path, suffix: str
) -> None:
    frame = old_set.frame.copy()
    frame.loc[0, "text"] = ODD_TEXT
    path = ReferenceSet.for_rubric(frame, toy_rubric, SETTINGS).save(tmp_path / f"refs{suffix}")
    loaded = ReferenceSet.load(path)
    assert loaded.frame["text"].iloc[0] == ODD_TEXT
    digest = table_hash(loaded.frame, FRAME_COLUMNS)
    assert digest == description_of(path)["table_hash"] == table_hash(frame, FRAME_COLUMNS)


def test_the_row_check_refuses_a_description_without_a_digest_unless_its_caller_allows_it(
    old_set: ReferenceSet, tmp_path: Path
) -> None:
    path = old_set.save(tmp_path / "references.csv")
    sidecar = path.with_name(path.name + ".json")
    undigested = {key: value for key, value in description_of(path).items() if key != "table_hash"}
    with pytest.raises(ConfigurationError) as refused:
        reference_files.check_described_rows(old_set.frame, undigested, path, sidecar)
    assert str(refused.value) == (
        f"{sidecar} has no table_hash, so the rows of {path} cannot be checked against it"
    )
    reference_files.check_described_rows(
        old_set.frame, undigested, path, sidecar, allow_missing_hash=True
    )
    with pytest.raises(ConfigurationError, match="do not belong together.*rows and no hash"):
        reference_files.check_described_rows(
            old_set.frame.iloc[:-1], undigested, path, sidecar, allow_missing_hash=True
        )
