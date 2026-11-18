"""The splits of the baseline: the form they check on construction, and the splits file."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from copernicus._version import __version__
from copernicus.baseline import BaselineSplits, read_splits, write_splits
from copernicus.errors import ConfigurationError


def folds_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "seed": [42, 42, 42, 43, 43, 43],
            "text_id": ["a", "b", "c"] * 2,
            "fold": [0, 1, 1, 1, 0, 0],
        }
    )


def draws_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "seed": [42, 42, 43],
            "size": ["2", "full", "2"],
            "fold": [0, 0, 1],
            "text_id": ["b", "c", "a"],
        }
    )


def test_the_tables_take_the_column_order_of_the_splits_and_keep_their_rows() -> None:
    splits = BaselineSplits(
        folds=folds_frame()[["fold", "text_id", "seed"]],
        draws=draws_frame()[["text_id", "fold", "size", "seed"]],
    )
    assert list(splits.folds.columns) == ["seed", "text_id", "fold"]
    assert list(splits.draws.columns) == ["seed", "size", "fold", "text_id"]
    assert splits.folds["text_id"].tolist() == ["a", "b", "c", "a", "b", "c"]
    assert splits.folds["fold"].tolist() == [0, 1, 1, 1, 0, 0]
    assert splits.draws["size"].tolist() == ["2", "full", "2"]
    assert splits.draws["text_id"].tolist() == ["b", "c", "a"]


def test_the_tables_get_integer_columns_of_int64_text_ids_as_strings_and_a_fresh_index() -> None:
    folds = pd.DataFrame(
        {
            "seed": np.array([7, 7], dtype=np.int32),
            "text_id": [11, 12],
            "fold": pd.array([1, 0], dtype="Int64"),
        },
        index=[5, 9],
    )
    splits = BaselineSplits(folds=folds)
    assert splits.draws is None
    assert splits.folds["seed"].dtype == np.int64 and splits.folds["fold"].dtype == np.int64
    assert splits.folds["text_id"].tolist() == ["11", "12"]
    assert pd.api.types.is_string_dtype(splits.folds["text_id"])
    assert list(splits.folds.index) == [0, 1]


def test_the_splits_keep_copies_that_later_changes_of_the_given_tables_do_not_reach() -> None:
    folds, draws = folds_frame(), draws_frame()
    splits = BaselineSplits(folds=folds, draws=draws)
    folds.loc[0, "fold"] = 4
    draws.loc[0, "text_id"] = "z"
    assert splits.folds.loc[0, "fold"] == 0 and splits.draws.loc[0, "text_id"] == "b"


def test_the_splits_are_frozen() -> None:
    splits = BaselineSplits(folds=folds_frame())
    with pytest.raises(dataclasses.FrozenInstanceError):
        splits.draws = draws_frame()  # type: ignore[misc]


def test_empty_draws_are_a_table_and_not_none() -> None:
    splits = BaselineSplits(folds=folds_frame(), draws=draws_frame().iloc[:0])
    assert splits.draws is not None and splits.draws.empty
    assert list(splits.draws.columns) == ["seed", "size", "fold", "text_id"]


def test_empty_tables_of_any_column_type_take_the_types_of_the_columns() -> None:
    typed = BaselineSplits(folds=folds_frame(), draws=draws_frame())
    assert typed.draws is not None
    untyped = pd.DataFrame(columns=["seed", "size", "fold", "text_id"])
    splits = BaselineSplits(folds=pd.DataFrame(columns=["seed", "text_id", "fold"]), draws=untyped)
    assert splits.draws is not None and splits.draws.empty and splits.folds.empty
    assert splits.folds.dtypes.to_dict() == typed.folds.dtypes.to_dict()
    assert splits.draws.dtypes.to_dict() == typed.draws.dtypes.to_dict()


def test_integer_columns_keep_values_beyond_64_bits_and_refuse_other_values() -> None:
    beyond = [2**64 + 5, 2**63, -(2**63) - 1]
    folds = pd.DataFrame({"seed": beyond, "text_id": ["a", "b", "c"], "fold": [0, 1, 1]})
    assert BaselineSplits(folds=folds).folds["seed"].tolist() == beyond
    ends = folds.assign(seed=[2**63 - 1, -(2**63), 0])
    assert BaselineSplits(folds=ends).folds["seed"].dtype == np.int64
    for wrong, shown in (([True, 2**64, 3], "True"), ([1.5, 2**64, 3], "1.5")):
        pattern = rf"column seed must hold integers; 1 value does not, for example {shown}$"
        with pytest.raises(ConfigurationError, match=pattern):
            BaselineSplits(folds=folds.assign(seed=pd.Series(wrong, dtype=object)))


@pytest.mark.parametrize("dtype", [np.uint64, "UInt64"])
def test_unsigned_text_ids_above_the_signed_range_become_their_decimal_strings(
    dtype: Any,
) -> None:
    ids = pd.array([2**63, 2**63 + 1, 2**64 - 1], dtype=dtype)
    folds = pd.DataFrame({"seed": [1, 1, 1], "text_id": ids, "fold": [0, 1, 1]})
    draws = pd.DataFrame({"seed": [1, 1], "size": ["2", "2"], "fold": [0, 0], "text_id": ids[1:]})
    splits = BaselineSplits(folds=folds, draws=draws)
    expected = [str(2**63), str(2**63 + 1), str(2**64 - 1)]
    assert splits.folds["text_id"].tolist() == expected
    assert splits.draws is not None and splits.draws["text_id"].tolist() == expected[1:]


@pytest.mark.parametrize(
    ("field", "value", "pattern"),
    [
        ("folds", {"seed": [1]}, r"folds must be a pandas DataFrame.*got dict"),
        ("draws", [1, 2], r"draws must be None or a pandas DataFrame.*got list"),
    ],
)
def test_a_table_that_is_not_a_data_frame_is_refused(field: str, value: Any, pattern: str) -> None:
    given = {"folds": folds_frame(), "draws": None, field: value}
    with pytest.raises(ConfigurationError, match=pattern):
        BaselineSplits(**given)


def test_a_missing_or_an_unknown_column_is_named() -> None:
    with pytest.raises(ConfigurationError, match=r"folds.*lacks the column\(s\) fold"):
        BaselineSplits(folds=folds_frame().drop(columns="fold"))
    with pytest.raises(ConfigurationError, match=r"folds.*columns that splits do not have: group"):
        BaselineSplits(folds=folds_frame().assign(group="g"))
    with pytest.raises(ConfigurationError, match=r"draws.*lacks the column\(s\) size"):
        BaselineSplits(folds=folds_frame(), draws=draws_frame().drop(columns="size"))
    repeated = pd.concat([folds_frame(), folds_frame()["fold"]], axis=1)
    with pytest.raises(ConfigurationError, match=r"the folds table repeats the column\(s\) fold"):
        BaselineSplits(folds=repeated)


@pytest.mark.parametrize(
    ("table", "column", "value"),
    [
        ("folds", "seed", np.nan),
        ("folds", "text_id", None),
        ("folds", "fold", np.nan),
        ("draws", "size", None),
        ("draws", "text_id", np.nan),
    ],
)
def test_a_missing_value_is_refused_with_its_column_and_row(
    table: str, column: str, value: Any
) -> None:
    frames = {"folds": folds_frame(), "draws": draws_frame()}
    frames[table] = frames[table].astype({column: object})
    frames[table].loc[1, column] = value
    pattern = rf"{table} table has missing values in column {column}, in the rows 1 "
    with pytest.raises(ConfigurationError, match=pattern):
        BaselineSplits(folds=frames["folds"], draws=frames["draws"])


@pytest.mark.parametrize(
    ("table", "column", "values", "pattern"),
    [
        ("folds", "seed", [42.0, 42, 42, 43, 43, 43], r"seed must hold integers.*float64"),
        ("folds", "fold", [True, False] * 3, r"fold must hold integers.*bool"),
        ("folds", "text_id", [0.5, 1.5, 2.5] * 2, r"text_id must hold strings or integers"),
        ("folds", "text_id", [True, False, True] * 2, r"strings or integers; got dtype bool"),
        ("folds", "text_id", ["a", 1.5, "c"] * 2, r"2 values do not, for example 1\.5"),
        ("folds", "text_id", ["a", True, "c"] * 2, r"2 values do not, for example True"),
        ("draws", "size", [2, 2, 2], r"size must hold labels.*'50' or 'full'"),
        ("draws", "size", ["2", 2, "full"], r"size must hold labels .*'full'; got 2$"),
        ("draws", "size", ["050", "full", "2"], r"size.*'050'"),
        ("draws", "size", ["n50", "full", "2"], r"size.*'n50'"),
        ("draws", "size", ["Full", "full", "2"], r"size.*'Full'"),
        ("draws", "size", ["0", "full", "2"], r"size.*'0'"),
        ("draws", "fold", ["0", "0", "1"], r"fold must hold integers"),
    ],
)
def test_a_column_of_the_wrong_type_is_refused(
    table: str, column: str, values: list[Any], pattern: str
) -> None:
    frames = {"folds": folds_frame(), "draws": draws_frame()}
    frames[table] = frames[table].assign(**{column: values})
    with pytest.raises(ConfigurationError, match=pattern):
        BaselineSplits(folds=frames["folds"], draws=frames["draws"])


def test_duplicate_rows_are_refused_and_named() -> None:
    folds = pd.concat([folds_frame(), folds_frame().iloc[[4]]], ignore_index=True)
    with pytest.raises(ConfigurationError, match=r"folds table holds duplicate rows: \(43, b, 0\)"):
        BaselineSplits(folds=folds)
    draws = pd.concat([draws_frame(), draws_frame().iloc[[2]]], ignore_index=True)
    with pytest.raises(ConfigurationError, match=r"draws table.*duplicate rows: \(43, 2, 1, a\)"):
        BaselineSplits(folds=folds_frame(), draws=draws)


# --- the splits file -----------------------------------------------------------------


@pytest.mark.parametrize("with_draws", [True, False])
def test_the_file_round_trips_the_tables(tmp_path: Path, with_draws: bool) -> None:
    splits = BaselineSplits(folds=folds_frame(), draws=draws_frame() if with_draws else None)
    path = tmp_path / "splits.json"
    write_splits(splits, path)
    again = read_splits(str(path))
    pd.testing.assert_frame_equal(again.folds, splits.folds)
    if with_draws:
        assert again.draws is not None and splits.draws is not None
        pd.testing.assert_frame_equal(again.draws, splits.draws)
    else:
        assert again.draws is None


@pytest.mark.parametrize("empty", ["folds", "draws", "both"])
def test_empty_tables_come_back_from_the_file_with_their_types(tmp_path: Path, empty: str) -> None:
    folds = folds_frame().iloc[:0] if empty in ("folds", "both") else folds_frame()
    draws = draws_frame().iloc[:0] if empty in ("draws", "both") else draws_frame()
    splits = BaselineSplits(folds=folds, draws=draws)
    path = tmp_path / "splits.json"
    write_splits(splits, path)
    again = read_splits(path)
    assert again.draws is not None and splits.draws is not None
    pd.testing.assert_frame_equal(again.folds, splits.folds)
    pd.testing.assert_frame_equal(again.draws, splits.draws)


def test_integers_beyond_64_bits_and_unsigned_ids_round_trip_through_the_file(
    tmp_path: Path,
) -> None:
    ids = np.array([2**63, 2**63 + 1, 2**64 - 1], dtype=np.uint64)
    beyond = [2**64 + 5, 2**64 + 5, -(2**63) - 1]
    splits = BaselineSplits(folds=pd.DataFrame({"seed": beyond, "text_id": ids, "fold": [0, 1, 1]}))
    path = tmp_path / "splits.json"
    write_splits(splits, path)
    again = read_splits(path)
    pd.testing.assert_frame_equal(again.folds, splits.folds)
    assert again.folds["seed"].tolist() == beyond
    assert again.folds["text_id"].tolist() == [str(value) for value in ids.tolist()]


def test_the_file_holds_the_splits_block_beside_the_library_version(tmp_path: Path) -> None:
    path = tmp_path / "splits.json"
    write_splits(BaselineSplits(folds=folds_frame()), str(path))
    content = json.loads(path.read_text(encoding="utf-8"))
    assert content["analysis"] == "NLP-ML baseline splits"
    assert content["library_version"] == __version__
    assert content["splits"]["draws"] is None
    assert content["splits"]["folds"] == {
        "seed": [42, 42, 42, 43, 43, 43],
        "text_id": ["a", "b", "c", "a", "b", "c"],
        "fold": [0, 1, 1, 1, 0, 0],
    }


def written(tmp_path: Path, block: Any) -> Path:
    path = tmp_path / "hand.json"
    path.write_text(json.dumps({"analysis": "NLP-ML baseline splits", "splits": block}))
    return path


FOLDS = {"seed": [42, 42], "text_id": ["a", "b"], "fold": [0, 1]}


@pytest.mark.parametrize(
    ("block", "pattern"),
    [
        (None, r"holds no splits: write them with write_splits or --write-splits"),
        ({"folds": FOLDS}, r"the splits lack draws"),
        ({"folds": FOLDS, "draws": None, "notes": 1}, r"fields that no version writes: notes"),
        ({"folds": [1, 2], "draws": None}, r"folds must be an object of columns"),
        (
            {"folds": {**FOLDS, "fold": 3}, "draws": None},
            r"column fold of the folds must be a list",
        ),
        (
            {"folds": {**FOLDS, "fold": [0]}, "draws": None},
            r"columns of the folds differ in length",
        ),
        ({"folds": {**FOLDS, "seed": [1.5, 2]}, "draws": None}, r"seed must hold integers"),
        ({"folds": {"seed": [42], "text_id": ["a"]}, "draws": None}, r"lacks the column\(s\) fold"),
    ],
)
def test_a_file_that_does_not_hold_splits_is_refused_with_its_name(
    tmp_path: Path, block: Any, pattern: str
) -> None:
    path = written(tmp_path, block)
    with pytest.raises(ConfigurationError, match=pattern) as error:
        read_splits(path)
    assert str(path) in str(error.value)


def test_a_file_that_is_not_json_is_refused_with_its_name(tmp_path: Path) -> None:
    path = tmp_path / "broken.json"
    path.write_text("{not json")
    with pytest.raises(ConfigurationError, match="is not valid JSON") as error:
        read_splits(path)
    assert str(path) in str(error.value)
