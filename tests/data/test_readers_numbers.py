"""A number of a CSV file is read as the float nearest to its text, on every pandas version.

The default parser of pandas before 3.1 can read a number of seventeen digits back as a
neighbouring float. The reader asks those versions for the exact parser; from pandas 3.1 on the
default parser is exact. Which columns pandas reads as numbers stays its own inference, also for
a text beyond the largest float.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import copernicus.data.readers as readers
from copernicus.data import read_frame
from copernicus.errors import ConfigurationError

pytestmark = pytest.mark.filterwarnings("error")


def _numbers_file(folder: Path, values: np.ndarray) -> Path:
    # repr gives the shortest text that reads back to the same float.
    lines = "".join(f"t{index},{value!r}\n" for index, value in enumerate(values.tolist()))
    path = folder / "numbers.csv"
    path.write_text("text_id,value\n" + lines, encoding="utf-8")
    return path


def test_a_csv_number_is_read_as_the_float_nearest_to_its_text(tmp_path: Path) -> None:
    values = np.random.default_rng(0).normal(scale=5.0, size=2_000)
    frame = read_frame(_numbers_file(tmp_path, values), protected_columns=("text_id",))
    assert frame["value"].to_numpy().tobytes() == values.tobytes()
    assert frame["text_id"].tolist()[:2] == ["t0", "t1"]


def test_integers_missing_values_and_text_columns_are_read_as_before(tmp_path: Path) -> None:
    path = tmp_path / "mixed.csv"
    path.write_text("text_id,count,share,flag\n007,3,0.5,yes\n008,,NA,no\n", encoding="utf-8")
    frame = read_frame(path, protected_columns=("text_id",))
    assert frame["text_id"].tolist() == ["007", "008"]
    assert frame["count"].tolist()[0] == 3.0 and np.isnan(frame["count"].tolist()[1])
    assert frame["share"].tolist()[0] == 0.5 and np.isnan(frame["share"].tolist()[1])
    assert frame["flag"].tolist() == ["yes", "no"]


@pytest.mark.parametrize(
    ("version", "options"),
    [
        ("2.0.0", {"float_precision": "round_trip"}),
        ("2.3.3", {"float_precision": "round_trip"}),
        ("3.0.6", {"float_precision": "round_trip"}),
        ("3.1.0rc1", {}),
        ("3.1.0", {}),
        ("4.0.0.dev0+12.gabc", {}),
    ],
)
def test_the_exact_parser_is_asked_of_every_pandas_before_3_1(
    version: str, options: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(pd, "__version__", version)
    assert readers.exact_float_options() == options


def test_a_pandas_version_that_cannot_be_read_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _numbers_file(tmp_path, np.array([0.5]))
    monkeypatch.setattr(pd, "__version__", "three point one")
    with pytest.raises(ConfigurationError) as refused:
        read_frame(path)
    assert "cannot read the pandas version 'three point one'" in str(refused.value)


@pytest.mark.parametrize(
    ("text", "value"),
    [
        ("1.7976931348623158e308", "1.7976931348623157e+308"),
        ("-1e-9999", "-0.0"),
        ("4.9e-324", "5e-324"),
    ],
)
def test_texts_at_the_edges_of_the_floats_are_read_alike_under_every_pandas(
    text: str, value: str, tmp_path: Path
) -> None:
    # The largest float with one digit more, a negative underflow and the smallest float. With
    # its default parser pandas 2 leaves the first column as text, pandas 3.0 reads infinity
    # there, and both read the second as 0.0; with the exact parser every version reads the
    # nearest float. What a version makes of a text beyond the largest float is not alike.
    path = tmp_path / "edge.csv"
    path.write_text(f"x\n{text}\n0.0\n", encoding="utf-8")
    column = read_frame(path)["x"]
    assert column.dtype == np.float64
    assert [repr(float(entry)) for entry in column.tolist()] == [value, "0.0"]
