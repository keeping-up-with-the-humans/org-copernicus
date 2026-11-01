"""The identity of a run store, and the ids of a rubric as the names of its columns."""

import dataclasses
import json
import logging
import re
import sqlite3
from pathlib import Path

import pytest
from store_support import make_identity, open_store, unvalidated_rubric

from copernicus.coders import (
    OneStageConfig,
    TwoStageConfig,
    identity_settings,
    load_wording_pool,
)
from copernicus.coders.store_types import wording_hash
from copernicus.errors import RubricValidationError, StoreError
from copernicus.rubric.model import Rubric


def refused_attribute(rubric: Rubric, message: str, **replacements: object) -> Rubric:
    """Change the first attribute so that its rubric breaks the id rule.

    The rubric's constructor refuses the change with ``message``, so a bad id can no longer
    reach the store through a rubric. The result is a stand-in made without the constructor,
    on which the store's own check is shown.
    """
    first, *rest = rubric.attributes
    attributes = (dataclasses.replace(first, **replacements), *rest)
    with pytest.raises(RubricValidationError, match=message):
        dataclasses.replace(rubric, attributes=attributes)
    return unvalidated_rubric(rubric, attributes=attributes)


def test_identity_mismatch_is_loud(toy_rubric: Rubric, tmp_path: Path) -> None:
    path = tmp_path / "run.sqlite"
    open_store(path, toy_rubric).close()
    other = make_identity(toy_rubric, config=TwoStageConfig(verify_threshold=90.0))
    with pytest.raises(
        StoreError, match="run_info mismatch on verify_threshold: stored 88.0, requested 90.0"
    ):
        open_store(path, toy_rubric, other)


def test_model_and_missing_key_are_checked(toy_rubric: Rubric, tmp_path: Path) -> None:
    path = tmp_path / "run.sqlite"
    open_store(path, toy_rubric).close()
    with pytest.raises(StoreError, match="mismatch on model_name: stored model-a, requested m2"):
        open_store(path, toy_rubric, make_identity(toy_rubric, model_name="m2"))
    with pytest.raises(StoreError, match="run_info missing extra_setting"):
        open_store(path, toy_rubric, make_identity(toy_rubric, extra={"extra_setting": "1"}))


def test_different_rubric_hash_raises(toy_rubric: Rubric, tmp_path: Path) -> None:
    path = tmp_path / "run.sqlite"
    open_store(path, toy_rubric).close()
    changed = dataclasses.replace(toy_rubric, concept="Another concept")
    with pytest.raises(StoreError, match="run_info mismatch on rubric_hash"):
        open_store(path, changed, make_identity(changed))


def test_identity_hash_must_fit_the_rubric(toy_rubric: Rubric, tmp_path: Path) -> None:
    wrong = dataclasses.replace(make_identity(toy_rubric), rubric_hash="0" * 16)
    with pytest.raises(StoreError, match="does not match the rubric"):
        open_store(tmp_path / "run.sqlite", toy_rubric, wrong)
    assert not (tmp_path / "run.sqlite").exists()


def test_reserved_setting_key_is_refused(toy_rubric: Rubric, tmp_path: Path) -> None:
    with pytest.raises(StoreError, match="reserved run_info keys: \\['seed'\\]"):
        open_store(
            tmp_path / "run.sqlite", toy_rubric, make_identity(toy_rubric, extra={"seed": "1"})
        )


def test_file_without_run_info_is_not_a_store(toy_rubric: Rubric, tmp_path: Path) -> None:
    path = tmp_path / "other.sqlite"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE t (x INTEGER)")
    with pytest.raises(StoreError, match="not a run store"):
        open_store(path, toy_rubric)


@pytest.mark.parametrize(
    "identifier,message",
    [
        ("bad-id", "cannot be a column name"),
        ("tone\n", "cannot be a column name"),
        ("seed", "reserved column"),
        ("TONE", "another attribute or marker"),
    ],
)
def test_identifiers_must_be_safe_columns(
    toy_rubric: Rubric, tmp_path: Path, identifier: str, message: str
) -> None:
    rubric = refused_attribute(toy_rubric, message, attribute_id=identifier)
    with pytest.raises(StoreError, match=message):
        open_store(tmp_path / "run.sqlite", rubric, make_identity(rubric))


def test_identity_settings_cover_config_and_provenance() -> None:
    pool = load_wording_pool()
    two = identity_settings(TwoStageConfig(), pool)
    one = identity_settings(OneStageConfig(), pool)
    assert two["verify_threshold"] == "88.0" and two["no_evidence_score"] == "1"
    assert "verify_threshold" not in one and one["max_new_tokens"] == "2048"
    assert set(json.loads(two["template_hashes"])) == {"extraction", "scoring", "one_stage"}
    assert re.fullmatch(r"[0-9a-f]{16}", two["wording_hash"])
    assert two["wording_hash"] == one["wording_hash"]
    assert two["verifier_version"].startswith("rapidfuzz") and two["library_version"]


@pytest.mark.parametrize(
    "reserved", ["construct_score", "n_runs", "text_id", "text_hash", "library_version"]
)
def test_reserved_names_are_refused_as_attribute_ids(
    toy_rubric: Rubric, tmp_path: Path, reserved: str
) -> None:
    rubric = refused_attribute(toy_rubric, "reserved column name", attribute_id=reserved)
    with pytest.raises(StoreError, match="reserved column name"):
        open_store(tmp_path / "run.sqlite", rubric, make_identity(rubric))
    assert not (tmp_path / "run.sqlite").exists()


def test_marker_id_equal_to_an_attribute_id_is_refused(toy_rubric: Rubric, tmp_path: Path) -> None:
    first = toy_rubric.attributes[0]
    marker = dataclasses.replace(first.markers[0], marker_id="tone")
    collision = "collides with another attribute or marker id"
    rubric = refused_attribute(toy_rubric, collision, markers=(marker, *first.markers[1:]))
    with pytest.raises(StoreError, match=collision):
        open_store(tmp_path / "run.sqlite", rubric, make_identity(rubric))


def test_library_version_may_differ_on_resume(
    toy_rubric: Rubric, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "run.sqlite"
    open_store(path, toy_rubric).close()
    newer = make_identity(toy_rubric, extra={"library_version": "99.0.0"})
    with caplog.at_level(logging.WARNING, logger="copernicus.coders.store"):
        open_store(path, toy_rubric, newer).close()
    assert "library_version" in caplog.text and "99.0.0" in caplog.text


def test_stored_key_the_identity_does_not_carry_is_refused(
    toy_rubric: Rubric, tmp_path: Path
) -> None:
    path = tmp_path / "run.sqlite"
    open_store(path, toy_rubric, make_identity(toy_rubric, extra={"extra_setting": "1"})).close()
    with pytest.raises(StoreError, match="run_info unexpected extra_setting"):
        open_store(path, toy_rubric)


def test_wording_hash_follows_enabled_flags_and_row_order() -> None:
    pool = load_wording_pool()
    base = wording_hash(pool)
    flipped = dataclasses.replace(
        pool,
        role=(dataclasses.replace(pool.role[0], enabled=not pool.role[0].enabled), *pool.role[1:]),
    )
    reordered = dataclasses.replace(pool, task=tuple(reversed(pool.task)))
    assert len({base, wording_hash(flipped), wording_hash(reordered)}) == 3
    assert wording_hash(pool) == base
