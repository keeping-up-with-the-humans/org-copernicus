"""Tests of the checks between a reference table and the description beside it."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest
from references_support import Responder, clean_responder, writer

from copernicus import __version__
from copernicus.errors import BackendError, ConfigurationError
from copernicus.llm.base import GenerationRequest
from copernicus.references import GeneratorSettings, ReferenceSet, generate_references
from copernicus.rubric.model import Rubric

CELL_COUNT = 24


def description_of(path: Path) -> dict[str, object]:
    return json.loads(path.with_name(path.name + ".json").read_text(encoding="utf-8"))


def rewrite_description(path: Path, **changes: object) -> None:
    """Change keys of the description beside a table; a value of ``None`` removes the key."""
    sidecar = path.with_name(path.name + ".json")
    content = json.loads(sidecar.read_text(encoding="utf-8"))
    for key, value in changes.items():
        if value is None:
            del content[key]
        else:
            content[key] = value
    sidecar.write_text(json.dumps(content), encoding="utf-8")


def rewrite_table(path: Path, drop: str, add: str) -> None:
    """Rewrite the table of a set or resume file with pandas, one column dropped, one added."""
    frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path, dtype=str)
    changed = frame.drop(columns=[drop]).assign(**{add: "x"})
    if path.suffix == ".parquet":
        changed.to_parquet(path, index=False)
    else:
        changed.to_csv(path, index=False)


class StopAfter:
    """Answers like ``inner`` until ``limit`` requests were answered, then fails."""

    def __init__(self, inner: Responder, limit: int) -> None:
        self.inner = inner
        self.limit = limit
        self.answered = 0

    def __call__(self, request: GenerationRequest) -> str:
        if self.answered >= self.limit:
            raise BackendError("the endpoint went away")
        self.answered += 1
        return self.inner(request)


@pytest.fixture
def small_set(toy_rubric: Rubric) -> ReferenceSet:
    settings = GeneratorSettings(texts_per_call=1, max_retries=1)
    return generate_references(toy_rubric, [writer(clean_responder(1))], settings)


@pytest.mark.parametrize("suffix", [".parquet", ".csv"])
def test_every_save_records_the_library_version_and_the_row_count(
    small_set: ReferenceSet, tmp_path: Path, suffix: str
) -> None:
    path = tmp_path / f"references{suffix}"
    small_set.save(path)
    description = description_of(path)
    assert description["written_by"] == __version__
    assert description["n_rows"] == len(small_set.frame) == CELL_COUNT


def test_the_resume_file_records_the_version_and_the_rows_after_every_call(
    toy_rubric: Rubric, tmp_path: Path
) -> None:
    path = tmp_path / "progress.parquet"
    generate_references(toy_rubric, [writer(clean_responder())], GeneratorSettings(), resume=path)
    description = description_of(path)
    assert description["written_by"] == __version__
    assert description["n_rows"] == CELL_COUNT * 10


def test_a_reference_set_file_without_written_by_is_refused_and_named(
    small_set: ReferenceSet, tmp_path: Path
) -> None:
    path = tmp_path / "references.parquet"
    small_set.save(path)
    rewrite_description(path, written_by=None)
    with pytest.raises(ConfigurationError, match="written_by.*regenerate") as error:
        ReferenceSet.load(path)
    assert "references.parquet.json" in str(error.value)


def test_a_resume_file_without_written_by_is_refused_before_any_call(
    toy_rubric: Rubric, tmp_path: Path
) -> None:
    path = tmp_path / "progress.parquet"
    generate_references(toy_rubric, [writer(clean_responder())], GeneratorSettings(), resume=path)
    rewrite_description(path, written_by=None)
    backend = writer(clean_responder())
    with pytest.raises(ConfigurationError, match="progress.parquet.json.*written_by.*regenerate"):
        generate_references(toy_rubric, [backend], GeneratorSettings(), resume=path)
    assert backend.calls == []


def test_a_reference_set_file_without_the_rubric_hash_is_refused_as_an_earlier_version(
    small_set: ReferenceSet, tmp_path: Path
) -> None:
    path = tmp_path / "references.csv"
    small_set.save(path)
    rewrite_description(path, rubric_hash=None)
    with pytest.raises(ConfigurationError) as error:
        ReferenceSet.load(path)
    assert str(error.value) == (
        f"{path}.json has no rubric_hash: it was written by an earlier version of copernicus "
        f"than this one ({__version__}); if the rubric is unchanged since its texts were written, "
        "bind the set to it with ReferenceSet.load(path, bind_to=rubric) and save it again; "
        "otherwise regenerate references.csv"
    )


@pytest.mark.parametrize("value", ["", "4BDEE2EFB58B0003", "0" * 64, 12345, None])
def test_a_reference_set_file_whose_rubric_hash_is_no_content_hash_is_refused(
    small_set: ReferenceSet, tmp_path: Path, value: object
) -> None:
    path = tmp_path / "references.parquet"
    small_set.save(path)
    sidecar = path.with_name(path.name + ".json")
    content = json.loads(sidecar.read_text(encoding="utf-8"))
    sidecar.write_text(json.dumps({**content, "rubric_hash": value}), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="rubric_hash must be the content hash") as error:
        ReferenceSet.load(path)
    assert "references.parquet.json" in str(error.value) and repr(value) in str(error.value)


def test_a_resume_file_without_the_rubric_hash_is_refused_before_any_call(
    toy_rubric: Rubric, tmp_path: Path
) -> None:
    path = tmp_path / "progress.parquet"
    generate_references(toy_rubric, [writer(clean_responder())], GeneratorSettings(), resume=path)
    rewrite_description(path, rubric_hash=None)
    backend = writer(clean_responder())
    with pytest.raises(ConfigurationError, match="progress.parquet.json has no rubric_hash"):
        generate_references(toy_rubric, [backend], GeneratorSettings(), resume=path)
    assert backend.calls == []


def test_a_table_beside_the_description_of_another_save_is_refused(
    toy_rubric: Rubric, small_set: ReferenceSet, tmp_path: Path
) -> None:
    bigger = generate_references(
        toy_rubric,
        [writer(clean_responder(2))],
        GeneratorSettings(texts_per_call=2, max_retries=1),
    )
    first, second = tmp_path / "first.parquet", tmp_path / "second.parquet"
    small_set.save(first)
    bigger.save(second)
    shutil.copy(second, first)
    with pytest.raises(ConfigurationError, match=r"first\.parquet.*48 rows.*24") as error:
        ReferenceSet.load(first)
    assert "first.parquet.json" in str(error.value)


def test_a_description_of_another_rubric_is_refused_naming_the_file(
    small_set: ReferenceSet, tmp_path: Path
) -> None:
    path = tmp_path / "references.csv"
    small_set.save(path)
    rewrite_description(path, rubric_id="another_rubric")
    with pytest.raises(ConfigurationError, match="another_rubric") as error:
        ReferenceSet.load(path)
    assert "references.csv.json" in str(error.value)
    assert small_set.rubric_id in str(error.value)


def test_a_description_with_a_malformed_row_count_is_refused(
    small_set: ReferenceSet, tmp_path: Path
) -> None:
    path = tmp_path / "references.parquet"
    small_set.save(path)
    rewrite_description(path, n_rows="24")
    with pytest.raises(ConfigurationError, match="n_rows"):
        ReferenceSet.load(path)


def test_an_older_resume_table_beside_a_newer_description_is_refused(
    toy_rubric: Rubric, tmp_path: Path
) -> None:
    path = tmp_path / "progress.parquet"
    with pytest.raises(BackendError):
        generate_references(
            toy_rubric, [writer(StopAfter(clean_responder(), 5))], GeneratorSettings(), resume=path
        )
    older = tmp_path / "older.parquet"
    shutil.copy(path, older)
    with pytest.raises(BackendError):
        generate_references(
            toy_rubric, [writer(StopAfter(clean_responder(), 5))], GeneratorSettings(), resume=path
        )
    shutil.copy(older, path)
    backend = writer(clean_responder())
    with pytest.raises(ConfigurationError, match=r"progress\.parquet.*50 rows.*100"):
        generate_references(toy_rubric, [backend], GeneratorSettings(), resume=path)
    assert backend.calls == []


@pytest.mark.parametrize("suffix", [".parquet", ".csv"])
def test_a_table_without_a_column_of_a_reference_set_is_refused_naming_the_file(
    small_set: ReferenceSet, tmp_path: Path, suffix: str
) -> None:
    path = small_set.save(tmp_path / f"references{suffix}")
    rewrite_table(path, drop="top_p_applied", add="note")
    with pytest.raises(ConfigurationError) as error:
        ReferenceSet.load(path)
    message = str(error.value)
    assert message.startswith(f"{path}: a reference set needs the columns ")
    assert "lacks ['top_p_applied']" in message and "unexpected columns ['note']" in message


def test_a_resume_table_without_a_column_is_refused_before_any_call(
    toy_rubric: Rubric, tmp_path: Path
) -> None:
    path = tmp_path / "progress.csv"
    with pytest.raises(BackendError):
        generate_references(
            toy_rubric, [writer(StopAfter(clean_responder(), 5))], GeneratorSettings(), resume=path
        )
    rewrite_table(path, drop="seed", add="note")
    backend = writer(clean_responder())
    with pytest.raises(ConfigurationError, match=r"progress\.csv: .*lacks \['seed'\]"):
        generate_references(toy_rubric, [backend], GeneratorSettings(), resume=path)
    assert backend.calls == []


@pytest.mark.parametrize(
    ("column", "value"),
    [("call_index", "abc"), ("call_index", ""), ("prompt_tokens", "abc")],
    ids=["text in an integer column", "empty integer cell", "text in a float column"],
)
def test_a_csv_cell_that_its_column_cannot_read_is_refused_naming_the_file(
    small_set: ReferenceSet, tmp_path: Path, column: str, value: str
) -> None:
    path = small_set.save(tmp_path / "references.csv")
    table = pd.read_csv(path, dtype=str, keep_default_na=False)
    table.loc[0, column] = value
    table.to_csv(path, index=False)
    with pytest.raises(ConfigurationError) as error:
        ReferenceSet.load(path)
    message = str(error.value)
    assert message.startswith(f"{path}: ") and repr(value) in message
    if column == "prompt_tokens":
        assert message.startswith(f"{path}: column 'prompt_tokens' holds ")


def test_an_empty_csv_file_is_refused_as_empty(small_set: ReferenceSet, tmp_path: Path) -> None:
    path = small_set.save(tmp_path / "references.csv")
    path.write_text("", encoding="utf-8")
    with pytest.raises(ConfigurationError) as error:
        ReferenceSet.load(path)
    assert str(error.value) == f"{path}: the file is empty"


def test_a_csv_file_that_cannot_be_parsed_is_refused_naming_the_file(
    small_set: ReferenceSet, tmp_path: Path
) -> None:
    path = small_set.save(tmp_path / "references.csv")
    path.write_text(path.read_text(encoding="utf-8") + 'text,"unbalanced\n', encoding="utf-8")
    with pytest.raises(ConfigurationError) as error:
        ReferenceSet.load(path)
    assert str(error.value).startswith(f"{path}: the file cannot be parsed as CSV: ")
    assert "dtype of its column" not in str(error.value)


def test_a_csv_file_that_is_not_utf8_is_refused_naming_the_file(
    small_set: ReferenceSet, tmp_path: Path
) -> None:
    path = small_set.save(tmp_path / "references.csv")
    path.write_bytes(path.read_bytes() + "x,café\n".encode("latin-1"))
    with pytest.raises(ConfigurationError) as error:
        ReferenceSet.load(path)
    assert str(error.value) == f"{path}: the file is not UTF-8 text"


def test_an_integer_cell_beyond_the_int64_range_is_refused_naming_the_file(
    small_set: ReferenceSet, tmp_path: Path
) -> None:
    path = small_set.save(tmp_path / "references.csv")
    table = pd.read_csv(path, dtype=str, keep_default_na=False)
    table.loc[0, "call_index"] = str(2**70)
    table.to_csv(path, index=False)
    with pytest.raises(ConfigurationError) as error:
        ReferenceSet.load(path)
    assert str(error.value).startswith(
        f"{path}: a cell is outside the range of its integer column: "
    )
