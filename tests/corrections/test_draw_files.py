"""The draw file: a draw written to JSON and read back, the way corrections sample writes it."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, get_type_hints

import numpy as np
import pytest

from copernicus.corrections import SampleDraw, draw_human_sample, read_draw, write_draw
from copernicus.corrections.draw_files import DRAW_FIELDS
from copernicus.data import TextTable
from copernicus.errors import ConfigurationError

pytestmark = pytest.mark.filterwarnings("error")

EARLIER = "it was made by an earlier version: draw again"


def _written(table: TextTable, folder: Path) -> tuple[SampleDraw, Path, dict[str, Any]]:
    """A draw of four of the 18 toy groups, written to a draw file, with the file's content."""
    draw = draw_human_sample(table, n=10, seed=11)
    path = folder / "draw.json"
    write_draw(draw, path)
    return draw, path, json.loads(path.read_text(encoding="utf-8"))


def _rewritten(path: Path, content: dict[str, Any]) -> Path:
    path.write_text(json.dumps(content), encoding="utf-8")
    return path


def test_a_written_draw_reads_back_equal_in_every_field(
    toy_corpus: TextTable, tmp_path: Path
) -> None:
    draw, path, content = _written(toy_corpus, tmp_path)
    loaded = read_draw(path)
    assert DRAW_FIELDS == tuple(field.name for field in dataclasses.fields(SampleDraw))
    assert list(content["draw"]) == list(DRAW_FIELDS)
    for name in DRAW_FIELDS:
        assert np.array_equal(getattr(loaded, name), getattr(draw, name)), name
    assert loaded.ids.dtype.kind == "U" and loaded.groups.dtype.kind == "U"
    assert (loaded.m, loaded.n_groups, loaded.pi) == (4, 18, 4 / 18)


def test_the_draw_file_is_named_by_a_str_or_a_path(toy_corpus: TextTable, tmp_path: Path) -> None:
    assert get_type_hints(write_draw)["path"] == str | Path
    assert get_type_hints(read_draw)["path"] == str | Path
    draw = draw_human_sample(toy_corpus, n=10, seed=11)
    text = str(tmp_path / "draw.json")
    write_draw(draw, text)
    loaded = read_draw(text)
    assert np.array_equal(loaded.ids, draw.ids) and loaded.population_hash == draw.population_hash


@pytest.mark.parametrize("missing", ["population_hash", "ids", "groups", "seed"])
def test_a_file_without_a_field_was_made_by_an_earlier_version(
    missing: str, toy_corpus: TextTable, tmp_path: Path
) -> None:
    _, path, content = _written(toy_corpus, tmp_path)
    del content["draw"][missing]
    with pytest.raises(ConfigurationError, match=rf"lacks '{missing}'; {EARLIER}$"):
        read_draw(_rewritten(path, content))


def test_the_description_an_earlier_sample_command_wrote_is_refused(tmp_path: Path) -> None:
    counts = {"m": 4, "n_groups": 18, "n_units": 54, "quota": 10, "pi": 0.22, "seed": 11}
    earlier = {"analysis": "human sample", "draw": {**counts, "n_texts": 12}}
    with pytest.raises(ConfigurationError, match=rf"'ids', 'groups', 'population_hash'; {EARLIER}"):
        read_draw(_rewritten(tmp_path / "drawn.csv.json", earlier))


@pytest.mark.parametrize("value", ["", "abc", "A" * 64, "a" * 63, None, 5])
def test_a_malformed_population_hash_was_made_by_an_earlier_version(
    value: object, toy_corpus: TextTable, tmp_path: Path
) -> None:
    _, path, content = _written(toy_corpus, tmp_path)
    content["draw"]["population_hash"] = value
    with pytest.raises(ConfigurationError, match=rf"population_hash .*{EARLIER}$"):
        read_draw(_rewritten(path, content))


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"m": 5}, r"records m = 5, n_groups = 18 and .* give m = 4, n_groups = 18 and"),
        ({"pi": 0.5}, r"records m = 4, n_groups = 18 and pi = 0\.5; .* and pi = 0\.2"),
        ({"n_groups": 8}, r"records m = 4, n_groups = 8 and pi = 0\.2.* and pi = 0\.5$"),
        ({"groups": ["w03"] * 4}, r"records m = 4, n_groups = 18 and .* give m = 1, n_groups = 18"),
    ],
    ids=["m", "pi", "n_groups", "repeated group"],
)
def test_a_file_whose_counts_contradict_its_groups_is_refused(
    changes: dict[str, object], message: str, toy_corpus: TextTable, tmp_path: Path
) -> None:
    _, path, content = _written(toy_corpus, tmp_path)
    content["draw"].update(changes)
    with pytest.raises(ConfigurationError, match=r"contradict each other: it " + message):
        read_draw(_rewritten(path, content))


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("m", "4", "m must be an integer of at least 1"),
        ("m", True, "m must be an integer of at least 1"),
        ("quota", 0, "quota must be an integer of at least 1"),
        ("seed", 1.5, "seed must be an integer of at least 0"),
        ("pi", "0.2", r"pi must be a number in \(0, 1\]"),
        ("pi", 0.0, r"pi must be a number in \(0, 1\]"),
        ("ids", "t001", "ids must be a non-empty list"),
        ("ids", [], "ids must be a non-empty list"),
        ("groups", [1.5], "groups must hold strings or integers"),
    ],
)
def test_a_field_of_the_wrong_type_is_refused_with_the_file_named(
    name: str, value: object, message: str, toy_corpus: TextTable, tmp_path: Path
) -> None:
    _, path, content = _written(toy_corpus, tmp_path)
    content["draw"][name] = value
    with pytest.raises(ConfigurationError, match=rf"{path.name}: the draw's {message}"):
        read_draw(_rewritten(path, content))


def test_files_that_hold_no_draw_or_more_than_a_draw_are_refused(
    toy_corpus: TextTable, tmp_path: Path
) -> None:
    _, path, content = _written(toy_corpus, tmp_path)
    for not_a_draw in ({"analysis": "human sample"}, {"draw": [1, 2]}):
        with pytest.raises(ConfigurationError, match="holds no draw"):
            read_draw(_rewritten(tmp_path / "other.json", not_a_draw))
    content["draw"]["n_texts"] = 12
    with pytest.raises(ConfigurationError, match="fields that no version writes: 'n_texts'"):
        read_draw(_rewritten(path, content))
    (tmp_path / "list.json").write_text("[]", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="does not hold a JSON object"):
        read_draw(tmp_path / "list.json")
