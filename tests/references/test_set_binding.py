"""Tests of how a reference set is bound to its rubric: recorded, kept, and given on request.

A set records the content hash of its rubric and how it came by it (``bound_with``):
``generated`` when the generator wrote its texts, ``for_rubric`` when they were bound by an
explicit act. A set file of an earlier version, which records neither, is bound only when the
caller names the rubric, and a recorded binding is never overridden.
"""

from __future__ import annotations

import ast
import dataclasses
import json
from pathlib import Path

import pandas as pd
import pytest
from references_support import clean_responder, writer

import copernicus.references.binding as binding
import copernicus.references.files as files
from copernicus import __version__
from copernicus.errors import ConfigurationError
from copernicus.references import GeneratorSettings, ReferenceSet, generate_references
from copernicus.rubric.model import Rubric


@pytest.fixture
def generated(toy_rubric: Rubric) -> ReferenceSet:
    """A small generated set: one text per cell from one writer."""
    settings = GeneratorSettings(texts_per_call=1, max_retries=1)
    return generate_references(toy_rubric, [writer(clean_responder(1))], settings)


def edit_description(path: Path, drop: tuple[str, ...] = (), **values: object) -> None:
    """Drop keys from the description beside a table, or set them, as another version would."""
    sidecar = path.with_name(path.name + ".json")
    content = json.loads(sidecar.read_text(encoding="utf-8"))
    kept = {key: value for key, value in content.items() if key not in drop}
    sidecar.write_text(json.dumps({**kept, **values}), encoding="utf-8")


def earlier_version(path: Path, key: str) -> str:
    """Return the message that refuses a set file of an earlier version without ``key``."""
    return (
        f"{path}.json has no {key}: it was written by an earlier version of copernicus than this "
        f"one ({__version__}); if the rubric is unchanged since its texts were written, bind the "
        "set to it with ReferenceSet.load(path, bind_to=rubric) and save it again; otherwise "
        f"regenerate {path.name}"
    )


def test_how_a_set_came_by_its_hash_is_recorded_and_kept_by_its_files(
    generated: ReferenceSet, toy_rubric: Rubric, tmp_path: Path
) -> None:
    assert generated.bound_with == "generated"
    assert list(generated.description())[:3] == ["rubric_id", "rubric_hash", "bound_with"]
    assert ReferenceSet.load(generated.save(tmp_path / "set.parquet")).bound_with == "generated"
    bound = ReferenceSet.for_rubric(generated.frame, toy_rubric, generated.settings)
    assert bound.bound_with == "for_rubric" and bound.description()["bound_with"] == "for_rubric"
    assert ReferenceSet.load(bound.save(tmp_path / "bound.csv")).bound_with == "for_rubric"


@pytest.mark.parametrize("suffix", [".parquet", ".csv"])
def test_a_set_file_without_a_rubric_hash_is_bound_only_when_the_rubric_is_named(
    generated: ReferenceSet,
    toy_rubric: Rubric,
    rewritten_toy_rubric: Rubric,
    tmp_path: Path,
    suffix: str,
) -> None:
    path = generated.save(tmp_path / f"references{suffix}")
    edit_description(path, drop=("rubric_hash", "bound_with"))
    with pytest.raises(ConfigurationError) as refused:
        ReferenceSet.load(path)
    assert str(refused.value) == earlier_version(path, "rubric_hash")
    bound = ReferenceSet.load(path, bind_to=toy_rubric)
    assert (bound.rubric_id, bound.rubric_hash) == (toy_rubric.rubric_id, toy_rubric.content_hash())
    assert bound.bound_with == "for_rubric"
    pd.testing.assert_frame_equal(bound.frame, generated.frame)
    assert (bound.settings, bound.shortfalls) == (generated.settings, generated.shortfalls)
    resaved = ReferenceSet.load(bound.save(tmp_path / f"resaved{suffix}"))
    assert (resaved.rubric_hash, resaved.bound_with) == (toy_rubric.content_hash(), "for_rubric")
    rebound = ReferenceSet.load(path, bind_to=rewritten_toy_rubric)
    assert rebound.rubric_hash == rewritten_toy_rubric.content_hash()


def test_a_set_file_that_records_its_hash_but_not_its_binding_is_bound_only_on_request(
    generated: ReferenceSet, toy_rubric: Rubric, rewritten_toy_rubric: Rubric, tmp_path: Path
) -> None:
    path = generated.save(tmp_path / "references.parquet")
    edit_description(path, drop=("bound_with",))
    with pytest.raises(ConfigurationError) as refused:
        ReferenceSet.load(path)
    assert str(refused.value) == earlier_version(path, "bound_with")
    bound = ReferenceSet.load(path, bind_to=toy_rubric)
    assert (bound.rubric_hash, bound.bound_with) == (toy_rubric.content_hash(), "for_rubric")
    with pytest.raises(ConfigurationError, match=rewritten_toy_rubric.content_hash()):
        ReferenceSet.load(path, bind_to=rewritten_toy_rubric)


def test_bind_to_never_overrides_what_a_set_file_records(
    generated: ReferenceSet, toy_rubric: Rubric, rewritten_toy_rubric: Rubric, tmp_path: Path
) -> None:
    path = generated.save(tmp_path / "references.parquet")
    kept = ReferenceSet.load(path, bind_to=toy_rubric)
    assert (kept.rubric_hash, kept.bound_with) == (toy_rubric.content_hash(), "generated")
    with pytest.raises(ConfigurationError) as refused:
        ReferenceSet.load(path, bind_to=rewritten_toy_rubric)
    assert str(refused.value) == (
        f"{path}.json was written for rubric {toy_rubric.rubric_id} with hash "
        f"{toy_rubric.content_hash()}, but bind_to has hash {rewritten_toy_rubric.content_hash()}: "
        "the texts describe another definition; regenerate them, or bind texts you checked "
        "against the rubric with ReferenceSet.for_rubric"
    )
    another = dataclasses.replace(toy_rubric, rubric_id="another_rubric")
    with pytest.raises(ConfigurationError, match="is for rubric constructive_feedback_v1, but"):
        ReferenceSet.load(path, bind_to=another)
    edit_description(path, drop=("rubric_hash", "bound_with"))
    with pytest.raises(ConfigurationError, match="bind_to is rubric another_rubric"):
        ReferenceSet.load(path, bind_to=another)


@pytest.mark.parametrize("value", ["", "4BDEE2EFB58B0003", "0" * 64, 12345, None])
def test_a_malformed_rubric_hash_is_refused_even_with_bind_to(
    generated: ReferenceSet, toy_rubric: Rubric, tmp_path: Path, value: object
) -> None:
    path = generated.save(tmp_path / "references.parquet")
    edit_description(path, rubric_hash=value)
    with pytest.raises(ConfigurationError, match="rubric_hash must be the content hash") as error:
        ReferenceSet.load(path, bind_to=toy_rubric)
    assert "references.parquet.json" in str(error.value)


@pytest.mark.parametrize("value", ["guessed", "", 1, None])
def test_a_binding_other_than_the_two_is_refused(
    generated: ReferenceSet, tmp_path: Path, value: object
) -> None:
    with pytest.raises(ConfigurationError, match="bound_with must be one of"):
        dataclasses.replace(generated, bound_with=value)
    path = generated.save(tmp_path / "references.parquet")
    edit_description(path, bound_with=value)
    with pytest.raises(ConfigurationError, match=r"references\.parquet\.json: bound_with must be"):
        ReferenceSet.load(path)


def test_the_key_of_the_rubric_id_has_one_definition() -> None:
    """A string is the same object wherever it is interned, so the sources are read instead."""
    trees = {
        module.__name__: ast.parse(Path(str(module.__file__)).read_text(encoding="utf-8"))
        for module in (binding, files)
    }
    assigned = {
        name: {
            target.id
            for node in tree.body
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name)
        }
        for name, tree in trees.items()
    }
    imported = {
        alias.name
        for node in trees[files.__name__].body
        if isinstance(node, ast.ImportFrom) and node.module == binding.__name__
        for alias in node.names
    }
    assert "RUBRIC_KEY" in assigned[binding.__name__] and "RUBRIC_KEY" in imported
    assert "RUBRIC_KEY" not in assigned[files.__name__]
    assert binding.RUBRIC_KEY == "rubric_id"
