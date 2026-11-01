"""Tests of the canonical table of a reference set: one form, whatever the caller passed.

A set built by hand may hold its token counts as integers, its floats as float32, its integers
as int32, a float column as an object column of integers and None, or its columns in another
order. Every set holds its table in the canonical form (`canonical_frame`), and the files write
and hash that same form, so such a set saves to CSV and to parquet and loads with the recorded
digest and the values it held. A missing text or screen reason, a value of another type and a
screen flag that is not boolean are refused when the set is built.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from references_support import clean_responder, writer

import copernicus.references.files as reference_files
from copernicus._digests import table_hash
from copernicus.errors import ConfigurationError
from copernicus.references import GeneratorSettings, ReferenceSet, generate_references
from copernicus.references.reference_set import FRAME_COLUMNS
from copernicus.rubric.model import Rubric

SETTINGS = GeneratorSettings(texts_per_call=1, max_retries=1)
SUFFIXES = [".csv", ".parquet"]
CELL_COUNT = 24
TOKENS = ["prompt_tokens", "completion_tokens"]
SAMPLING = ["temperature_applied", "top_p_applied"]
INTEGERS = ["call_index", "attempt", "seed"]
Change = Callable[[pd.DataFrame], pd.DataFrame]


def gaps(values: pd.Series) -> list[int | None]:
    """Return the values as Python integers, every other one None."""
    return [None if row % 2 else int(value) for row, value in enumerate(values)]


def with_gaps(frame: pd.DataFrame) -> pd.DataFrame:
    """Hold the prompt tokens in an object column of Python integers and None."""
    return frame.assign(prompt_tokens=pd.Series(gaps(frame["prompt_tokens"]), dtype=object))


def as_floats_with_gaps(frame: pd.DataFrame) -> pd.DataFrame:
    """Hold the prompt tokens as the floats, NaN for None, that the canonical form makes."""
    values = [np.nan if value is None else float(value) for value in gaps(frame["prompt_tokens"])]
    return frame.assign(prompt_tokens=np.asarray(values, dtype=np.float64))


def float32(frame: pd.DataFrame) -> pd.DataFrame:
    """Hold the applied sampling values as float32."""
    return frame.astype(dict.fromkeys(SAMPLING, "float32"))


# Per variant: the frame a caller passes, and the canonical frame the set then holds.
VARIANTS: dict[str, tuple[Change, Change]] = {
    "int64 token counts": (
        lambda frame: frame.astype(dict.fromkeys(TOKENS, "int64")),
        pd.DataFrame,
    ),
    "float32 floats": (
        float32,
        lambda frame: float32(frame).astype(dict.fromkeys(SAMPLING, "float64")),
    ),
    "int32 integers": (lambda frame: frame.astype(dict.fromkeys(INTEGERS, "int32")), pd.DataFrame),
    "integers and None in a float column": (with_gaps, as_floats_with_gaps),
    "another column order": (lambda frame: frame[list(reversed(FRAME_COLUMNS))], pd.DataFrame),
}


@pytest.fixture
def generated(toy_rubric: Rubric) -> ReferenceSet:
    """One text per cell from one writer: a canonical table."""
    return generate_references(toy_rubric, [writer(clean_responder(1))], SETTINGS)


@pytest.mark.parametrize("suffix", SUFFIXES)
@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_a_frame_of_other_dtypes_or_order_saves_and_loads_with_its_digest_and_values(
    generated: ReferenceSet, toy_rubric: Rubric, tmp_path: Path, variant: str, suffix: str
) -> None:
    passed, expected = (change(generated.frame) for change in VARIANTS[variant])
    built = ReferenceSet.for_rubric(passed, toy_rubric, SETTINGS)
    pd.testing.assert_frame_equal(built.frame, expected)
    path = built.save(tmp_path / f"references{suffix}")
    loaded = ReferenceSet.load(path)
    pd.testing.assert_frame_equal(loaded.frame, expected)
    description = json.loads(path.with_name(path.name + ".json").read_text(encoding="utf-8"))
    digest = table_hash(loaded.frame, FRAME_COLUMNS)
    assert digest == description["table_hash"] == table_hash(expected, FRAME_COLUMNS)


@pytest.mark.parametrize(
    ("column", "value", "remedy"),
    [
        ("screen_reason", None, "write an empty string '' for no screen reason"),
        ("text", np.nan, "a text is never missing: leave out a row that has none"),
    ],
)
def test_a_missing_text_or_screen_reason_is_refused_when_the_set_is_built(
    generated: ReferenceSet, toy_rubric: Rubric, column: str, value: object, remedy: str
) -> None:
    frame = generated.frame.copy()
    frame.loc[3, column] = value
    with pytest.raises(ConfigurationError) as refused:
        ReferenceSet.for_rubric(frame, toy_rubric, SETTINGS)
    assert str(refused.value) == (
        f"column {column!r} of a reference set has a missing value in 1 of {CELL_COUNT} rows; "
        f"{remedy}"
    )


@pytest.mark.parametrize(
    "values", [0, pd.Series([0, 1] * 12, dtype=object)], ids=["int64", "object integers"]
)
def test_a_screen_flag_that_is_not_boolean_is_refused(
    generated: ReferenceSet, toy_rubric: Rubric, values: object
) -> None:
    frame = generated.frame.assign(screened_out=values)
    with pytest.raises(ConfigurationError, match="column 'screened_out' .* must hold booleans"):
        ReferenceSet.for_rubric(frame, toy_rubric, SETTINGS)


@pytest.mark.parametrize(
    "values",
    [pd.array([True, None] * 12, dtype="boolean"), pd.Series([True, None] * 12, dtype=object)],
    ids=["nullable boolean", "object with None"],
)
def test_a_missing_screen_flag_is_refused_naming_the_rows(
    generated: ReferenceSet, toy_rubric: Rubric, values: object
) -> None:
    frame = generated.frame.assign(screened_out=values)
    with pytest.raises(ConfigurationError) as refused:
        ReferenceSet.for_rubric(frame, toy_rubric, SETTINGS)
    assert str(refused.value) == (
        f"column 'screened_out' of a reference set has a missing value in 12 of {CELL_COUNT} "
        "rows; write True or False"
    )


def null_version(table: pd.DataFrame) -> pd.DataFrame:
    """Leave the prompt version of one row missing, as a file of an earlier version may."""
    changed = table.copy()
    changed.loc[2, "prompt_version"] = None
    return changed


def float_call_index(table: pd.DataFrame) -> pd.DataFrame:
    """Hold the call index as floats, as a file of an earlier version may."""
    return table.astype({"call_index": "float64"})


@pytest.mark.parametrize(
    ("change", "column"),
    [(null_version, "prompt_version"), (float_call_index, "call_index")],
    ids=["missing string", "float integers"],
)
def test_an_earlier_file_whose_table_the_set_cannot_take_is_refused_naming_the_file(
    generated: ReferenceSet, toy_rubric: Rubric, tmp_path: Path, change: Change, column: str
) -> None:
    path = generated.save(tmp_path / "references.parquet")
    sidecar = path.with_name(path.name + ".json")
    description = json.loads(sidecar.read_text(encoding="utf-8"))
    del description["table_hash"]
    sidecar.write_text(json.dumps(description), encoding="utf-8")
    change(pd.read_parquet(path)).to_parquet(path, index=False)
    with pytest.raises(ConfigurationError) as refused:
        ReferenceSet.load(path, bind_to=toy_rubric)
    message = str(refused.value)
    assert message.startswith(f"{path}: column {column!r} of a reference set ")
    assert message.endswith(
        "; fill or fix the column with pandas and bind the table with ReferenceSet.for_rubric, "
        f"or regenerate {path.name}"
    )


@pytest.mark.parametrize(
    ("column", "values", "message"),
    [
        ("call_index", 0.0, "column 'call_index' of a reference set must hold integers"),
        ("seed", pd.Series([1, None] * 12, dtype=object), "column 'seed' .* must hold integers"),
        ("attempt", pd.array([0, None] * 12, dtype="Int64"), "'attempt' .* missing value in 12 of"),
        ("prompt_tokens", "580", "column 'prompt_tokens' .* must hold numbers"),
        ("top_p_applied", True, "column 'top_p_applied' .* must hold numbers"),
        ("prompt_hash", 7, "column 'prompt_hash' of a reference set must hold strings"),
    ],
)
def test_a_value_that_its_column_cannot_take_is_refused_naming_the_column(
    generated: ReferenceSet, toy_rubric: Rubric, column: str, values: object, message: str
) -> None:
    frame = generated.frame.assign(**{column: values})
    with pytest.raises(ConfigurationError, match=message):
        ReferenceSet.for_rubric(frame, toy_rubric, SETTINGS)


def test_the_canonical_frame_is_a_new_frame_in_column_order_with_a_range_index(
    generated: ReferenceSet,
) -> None:
    shuffled = generated.frame[list(reversed(FRAME_COLUMNS))].set_axis(range(100, 124))
    canonical = reference_files.canonical_frame(shuffled)
    pd.testing.assert_frame_equal(canonical, generated.frame)
    again = reference_files.canonical_frame(canonical)
    assert again is not canonical
    pd.testing.assert_frame_equal(again, canonical)


@pytest.mark.parametrize("suffix", SUFFIXES)
def test_the_writer_writes_and_hashes_the_canonical_form_of_the_frame_it_is_given(
    generated: ReferenceSet, tmp_path: Path, suffix: str
) -> None:
    passed = with_gaps(float32(generated.frame))
    path = tmp_path / f"references{suffix}"
    reference_files.write_reference_files(path, passed, generated.description())
    frame, description = reference_files.read_reference_files(path, ["rubric_id"])
    canonical = reference_files.canonical_frame(passed)
    pd.testing.assert_frame_equal(frame, canonical)
    assert description["table_hash"] == table_hash(canonical, FRAME_COLUMNS)
