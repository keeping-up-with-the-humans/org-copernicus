"""Tests for the validated text table, the file readers, and the HTML stripper."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from copernicus.data import TextTable
from copernicus.data.html import strip_html
from copernicus.errors import TextTableError


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def test_from_frame_renames_and_computes_lengths() -> None:
    df = pd.DataFrame({"id": [1, 2], "body": ["Hello  world", "A b c"], "writer": ["w1", "w1"]})
    t = TextTable.from_frame(df, text_id="id", text="body", group_id="writer")
    assert list(t.frame.columns)[:5] == ["text_id", "text", "group_id", "n_chars", "n_words"]
    assert t.frame.n_words.tolist() == [2, 3] and t.text_of(1) == "Hello world"
    assert t.group_defaulted is False


def test_duplicate_or_missing_ids_named() -> None:
    df = pd.DataFrame({"text_id": [1, 1, None], "text": ["a a", "b b", "c c"]})
    with pytest.raises(TextTableError) as e:
        TextTable.from_frame(df)
    assert "duplicate" in str(e.value) and "1" in str(e.value) and "missing" in str(e.value)


def test_empty_text_named_and_group_default_flagged() -> None:
    df = pd.DataFrame({"text_id": ["a", "b"], "text": ["fine text", "   "]})
    with pytest.raises(TextTableError, match="'b'"):
        TextTable.from_frame(df)
    t = TextTable.from_frame(pd.DataFrame({"text_id": ["a"], "text": ["x y"]}))
    assert t.group_defaulted is True and t.frame.group_id.tolist() == ["a"]


def test_a_table_without_a_group_column_warns_that_every_text_is_its_own_group(
    caplog: pytest.LogCaptureFixture,
) -> None:
    frame = pd.DataFrame({"text_id": ["a", "b"], "text": ["x y", "z"], "writer": ["w", "w"]})
    with caplog.at_level(logging.WARNING, logger="copernicus.data.table"):
        TextTable.from_frame(frame)
    (record,) = caplog.records
    assert record.levelno == logging.WARNING
    message = record.getMessage()
    for consequence in ("own group", "independently", "bootstrap", "PPI standard errors"):
        assert consequence in message
    assert "group_id= in Python" in message and "--group-column on the command line" in message
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="copernicus.data.table"):
        TextTable.from_frame(frame, group_id="writer")
        TextTable.from_frame(frame.rename(columns={"writer": "group_id"}))
        TextTable.from_frame(frame, group_id="writer").subset(["a"])
    assert caplog.records == []


def test_read_rejects_unknown_suffix(tmp_path: Path) -> None:
    p = tmp_path / "x.xlsx"
    p.write_text("")
    with pytest.raises(TextTableError, match="xlsx"):
        TextTable.read(p)


def test_read_csv_and_jsonl_roundtrip(tmp_path: Path) -> None:
    df = pd.DataFrame({"text_id": [1, 2], "text": ["a b", "c d"]})
    df.to_csv(tmp_path / "t.csv", index=False)
    df.to_json(tmp_path / "t.jsonl", orient="records", lines=True)
    assert len(TextTable.read(tmp_path / "t.csv")) == 2
    assert len(TextTable.read(tmp_path / "t.jsonl")) == 2


def test_read_parquet_with_column_names(tmp_path: Path) -> None:
    df = pd.DataFrame({"id": ["p", "q"], "body": ["a b", "c d e"], "who": ["g", "g"]})
    df.to_parquet(tmp_path / "t.parquet")
    t = TextTable.read(tmp_path / "t.parquet", text_id="id", text="body", group_id="who")
    assert t.ids() == ["p", "q"] and t.frame.group_id.tolist() == ["g", "g"]


def test_extra_columns_kept_after_guaranteed_ones() -> None:
    df = pd.DataFrame({"text_id": ["a"], "gold": [3], "text": ["x y"], "age": [30]})
    t = TextTable.from_frame(df)
    assert list(t.frame.columns) == [
        "text_id",
        "text",
        "group_id",
        "n_chars",
        "n_words",
        "gold",
        "age",
    ]


def test_normalised_text_drives_lengths() -> None:
    t = TextTable.from_frame(pd.DataFrame({"text_id": ["a"], "text": ["  ab \n\t cd  "]}))
    assert t.text_of("a") == "ab cd"
    assert t.frame.n_chars.tolist() == [5] and t.frame.n_words.tolist() == [2]


def test_input_frame_not_mutated() -> None:
    df = pd.DataFrame({"text_id": ["a"], "text": ["x  y"]})
    TextTable.from_frame(df)
    assert list(df.columns) == ["text_id", "text"] and df.text.tolist() == ["x  y"]


def test_missing_column_named() -> None:
    with pytest.raises(TextTableError, match="body"):
        TextTable.from_frame(pd.DataFrame({"text_id": ["a"], "text": ["x"]}), text="body")


def test_all_problems_reported_in_one_error() -> None:
    df = pd.DataFrame({"text_id": ["a", "a"], "text": ["", "ok"]})
    with pytest.raises(TextTableError) as e:
        TextTable.from_frame(df)
    assert "duplicate" in str(e.value) and "empty" in str(e.value)


def test_missing_text_value_is_empty() -> None:
    df = pd.DataFrame({"text_id": ["a", "b"], "text": ["fine", None]})
    with pytest.raises(TextTableError, match="'b'"):
        TextTable.from_frame(df)


def test_subset_keeps_table_order_and_names_unknown_ids() -> None:
    df = pd.DataFrame({"text_id": [1, 2, 3], "text": ["a a", "b b", "c c"]})
    t = TextTable.from_frame(df)
    sub = t.subset([3, 1])
    assert sub.ids() == [1, 3] and len(sub) == 2 and len(t) == 3
    assert sub.group_defaulted is True
    with pytest.raises(TextTableError, match="99"):
        t.subset([1, 99])


def test_text_of_unknown_id_raises() -> None:
    t = TextTable.from_frame(pd.DataFrame({"text_id": ["a"], "text": ["x"]}))
    with pytest.raises(TextTableError, match="'zz'"):
        t.text_of("zz")


def test_strip_html_blocks_links_entities() -> None:
    html = '<p>Hello <a href="http://x">world</a> &amp; you</p><p>Next<br/>line</p>'
    assert strip_html(html) == "Hello world & you\nNext\nline"
    assert strip_html("") == ""


def test_missing_group_value_named() -> None:
    df = pd.DataFrame({"text_id": ["a", "b"], "text": ["x", "y"], "w": ["g", None]})
    with pytest.raises(TextTableError, match="'w'"):
        TextTable.from_frame(df, group_id="w")


def test_reserved_extra_column_rejected() -> None:
    df = pd.DataFrame({"id": ["a"], "body": ["x"], "text": ["other"]})
    with pytest.raises(TextTableError, match="collide"):
        TextTable.from_frame(df, text_id="id", text="body")


def test_frame_with_own_derived_columns_is_recomputed() -> None:
    df = pd.DataFrame(
        {
            "text_id": ["a", "b"],
            "text": ["x  y", "p q r"],
            "group_id": ["g", "g"],
            "n_chars": [999, 999],
            "n_words": [999, 999],
        }
    )
    t = TextTable.from_frame(df, group_id="group_id")
    assert t.frame.n_chars.tolist() == [3, 5] and t.frame.n_words.tolist() == [2, 3]
    assert list(t.frame.columns) == ["text_id", "text", "group_id", "n_chars", "n_words"]


def test_group_id_column_is_used_without_naming_it() -> None:
    df = pd.DataFrame({"text_id": ["a", "b"], "text": ["x y", "z w"], "group_id": ["g", "h"]})
    t = TextTable.from_frame(df)
    assert t.frame.group_id.tolist() == ["g", "h"] and t.group_defaulted is False


def test_group_id_column_with_missing_value_named() -> None:
    df = pd.DataFrame({"text_id": ["a", "b"], "text": ["x y", "z w"], "group_id": ["g", None]})
    with pytest.raises(TextTableError, match="'group_id'"):
        TextTable.from_frame(df)


def test_table_frame_round_trips_through_csv(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "text_id": ["a", "b"],
            "text": ["NA", "plain words"],
            "group_id": ["g", "h"],
            "gold": [1, 2],
        }
    )
    first = TextTable.from_frame(df)
    first.frame.to_csv(tmp_path / "t.csv", index=False)
    second = TextTable.read(tmp_path / "t.csv")
    pd.testing.assert_frame_equal(first.frame, second.frame)
    assert second.text_of("a") == "NA" and second.group_defaulted is False


def test_csv_reading_protects_text_columns_but_not_numeric_ones(tmp_path: Path) -> None:
    path = tmp_path / "t.csv"
    path.write_text(
        "text_id,text,group_id,gold,note\na,NA,NA,3,NA\nb,null,g,NA,\nc,plain words,g,#N/A,n/a\n",
        encoding="utf-8",
    )
    table = TextTable.read(path)
    assert table.text_of("a") == "NA" and table.frame.group_id.tolist() == ["NA", "g", "g"]
    assert table.frame.gold.isna().tolist() == [False, True, True]
    assert pd.api.types.is_numeric_dtype(table.frame.gold)
    assert table.frame.note.isna().all()


def test_csv_reading_still_treats_an_empty_protected_field_as_missing(tmp_path: Path) -> None:
    path = tmp_path / "t.csv"
    path.write_text("text_id,text\na,fine words\n,more words\n", encoding="utf-8")
    with pytest.raises(TextTableError, match="missing text ids"):
        TextTable.read(path)


def test_csv_reading_protects_the_named_columns(tmp_path: Path) -> None:
    path = tmp_path / "t.csv"
    path.write_text("id,body,who\nNA,NA,NA\n", encoding="utf-8")
    table = TextTable.read(path, text_id="id", text="body", group_id="who")
    assert table.ids() == ["NA"] and table.text_of("NA") == "NA"


def test_ids_and_groups_of_csv_and_jsonl_files_stay_text(tmp_path: Path) -> None:
    csv_path = tmp_path / "t.csv"
    csv_path.write_text(
        "text_id,text,group_id\n007,some words here,0042\n010,other words there,0042\n",
        encoding="utf-8",
    )
    from_csv = TextTable.read(csv_path)
    assert from_csv.ids() == ["007", "010"] and from_csv.text_of("007") == "some words here"
    assert from_csv.frame.group_id.tolist() == ["0042", "0042"]
    rows = [
        {"text_id": "007", "text": "some words here", "group_id": "0042"},
        {"text_id": 10, "text": "other words there", "group_id": 42},
    ]
    from_jsonl = TextTable.read(write_jsonl(tmp_path / "t.jsonl", rows))
    assert from_jsonl.ids() == ["007", "10"]
    assert from_jsonl.frame.group_id.tolist() == ["0042", "42"]


def test_numeric_looking_text_reads_as_text(tmp_path: Path) -> None:
    csv_path = tmp_path / "t.csv"
    csv_path.write_text("text_id,text\na,12345\n", encoding="utf-8")
    assert TextTable.read(csv_path).text_of("a") == "12345"
    rows = [{"text_id": "a", "text": "12345"}, {"text_id": "b", "text": 67890}]
    table = TextTable.read(write_jsonl(tmp_path / "t.jsonl", rows))
    assert (table.text_of("a"), table.text_of("b")) == ("12345", "67890")


def test_jsonl_values_are_neither_dates_nor_retyped(tmp_path: Path) -> None:
    rows = [
        {"text_id": "a", "text": "x y", "created_at": "2024-01-02", "date": "2024-01-03"},
        {"text_id": "b", "text": "z w", "created_at": "2024-02-02", "date": "2024-02-03"},
    ]
    rows[0]["gold"], rows[1]["gold"] = 3, 4
    frame = TextTable.read(write_jsonl(tmp_path / "t.jsonl", rows)).frame
    assert frame.created_at.tolist() == ["2024-01-02", "2024-02-02"]
    assert frame.date.tolist() == ["2024-01-03", "2024-02-03"]
    assert pd.api.types.is_integer_dtype(frame.gold) and frame.gold.tolist() == [3, 4]


def test_jsonl_nulls_stay_missing(tmp_path: Path) -> None:
    rows = [{"text_id": "a", "text": "x y"}, {"text_id": None, "text": "z w"}]
    with pytest.raises(TextTableError, match="missing text ids"):
        TextTable.read(write_jsonl(tmp_path / "ids.jsonl", rows))
    rows = [{"text_id": "a", "text": None}]
    with pytest.raises(TextTableError, match="empty text for ids 'a'"):
        TextTable.read(write_jsonl(tmp_path / "texts.jsonl", rows))


def test_jsonl_lines_that_are_no_json_object_are_named(tmp_path: Path) -> None:
    broken = tmp_path / "broken.jsonl"
    broken.write_text('{"text_id": "a", "text": "x y"}\n\n{broken\n', encoding="utf-8")
    with pytest.raises(TextTableError, match="'broken.jsonl'.*line 3"):
        TextTable.read(broken)
    listed = tmp_path / "listed.jsonl"
    listed.write_text('["a", "x y"]\n', encoding="utf-8")
    with pytest.raises(TextTableError, match="line 1 is not a JSON object"):
        TextTable.read(listed)


def test_parquet_keeps_its_typed_ids(tmp_path: Path) -> None:
    pd.DataFrame({"text_id": [7, 10], "text": ["a b", "c d"]}).to_parquet(tmp_path / "t.parquet")
    assert TextTable.read(tmp_path / "t.parquet").ids() == [7, 10]


@pytest.mark.parametrize(
    ("name", "content"),
    [
        ("quote.csv", b'text_id,text\nt1,"never closed\n'),
        ("latin.csv", b"text_id,text\nt1,caf\xe9 au lait\n"),
        ("latin.jsonl", b'{"text_id": "t1", "text": "caf\xe9"}\n'),
        ("broken.parquet", b"not a parquet file"),
    ],
)
def test_an_unreadable_file_raises_a_text_table_error(
    tmp_path: Path, name: str, content: bytes
) -> None:
    path = tmp_path / name
    path.write_bytes(content)
    with pytest.raises(TextTableError, match=f"cannot read '{name}'"):
        TextTable.read(path)
