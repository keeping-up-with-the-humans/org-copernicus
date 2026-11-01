import json
import sqlite3
from pathlib import Path

import pytest
from store_support import SCORES, make_row, open_store

from copernicus.errors import StoreError
from copernicus.rubric.model import Rubric


def test_insert_writes_the_shuffles_with_the_rows(toy_rubric: Rubric, tmp_path: Path) -> None:
    path = tmp_path / "run.sqlite"
    shuffles = [(7, "tone", ["b", "a"]), (7, "reasoning", ["x", "y"]), (8, "tone", ["a", "b"])]
    with open_store(path, toy_rubric) as store:
        store.insert([make_row(toy_rubric, "t1", 1, SCORES)], shuffles)
        store.insert([], [(7, "tone", ["a", "b"])])
        assert store.counts()["shuffles"] == 3 and store.counts()["results"] == 1
    with sqlite3.connect(path) as connection:
        stored = connection.execute(
            'SELECT "shuffle_seed", "attribute_id", "marker_order_json" FROM "shuffles" '
            "ORDER BY 1, 2"
        ).fetchall()
    assert [(a, b, json.loads(c)) for a, b, c in stored] == [
        (7, "reasoning", ["x", "y"]),
        (7, "tone", ["a", "b"]),
        (8, "tone", ["a", "b"]),
    ]


def test_insert_with_a_bad_shuffle_or_row_writes_nothing(
    toy_rubric: Rubric, tmp_path: Path
) -> None:
    row = make_row(toy_rubric, "t1", 1, SCORES)
    with open_store(tmp_path / "run.sqlite", toy_rubric) as store:
        with pytest.raises(StoreError, match="unknown attribute 'nope'"):
            store.insert([row], [(7, "tone", ["a"]), (7, "nope", [])])
        assert store.counts()["results"] == 0 and store.counts()["shuffles"] == 0
        store.insert([row])
        with pytest.raises(StoreError, match="already stored"):
            store.insert([row], [(9, "tone", ["a"])])
        assert store.counts()["shuffles"] == 0
