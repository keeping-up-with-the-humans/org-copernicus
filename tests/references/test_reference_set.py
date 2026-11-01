"""Tests for the reference set: files, kept texts, summary, and the balance check."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from references_support import clean_responder, seed_responder, texts_json, writer

from copernicus._digests import table_hash
from copernicus.errors import ConfigurationError
from copernicus.llm.base import GenerationRequest
from copernicus.references import GeneratorSettings, ReferenceSet, Shortfall, generate_references
from copernicus.references.reference_set import FRAME_COLUMNS
from copernicus.rubric.loader import load_rubric, toy_rubric_path
from copernicus.rubric.model import Rubric

CELL_COUNT = 24
MALFORMED = "settings or shortfalls are malformed"
TOY_RUBRIC_HASH = load_rubric(toy_rubric_path()).content_hash()
# Stands for the table hash in a description written here; the test that writes the description
# beside a saved table puts the hash of that table in its place.
TABLE_HASH_MARK = "TABLE-HASH-OF-THE-SAVED-TABLE"
SIDECAR = (
    '{{"rubric_id": "constructive_feedback_v1", "rubric_hash": "{rubric_hash}", '
    '"bound_with": "generated", "settings": {settings}, "shortfalls": {shortfalls}, '
    '"writers": {{}}, "usage": {{}}, '
    '"written_by": "0.1.0", "n_rows": 24, "table_hash": "{table_hash}"}}'
)
# Values that are not the content hash of a rubric: 16 lowercase hexadecimal characters.
NOT_RUBRIC_HASHES = ["", "4BDEE2EFB58B0003", "4bdee2efb58b000", "4bdee2efb58b000g", "0" * 64]


def sidecar(settings: str, shortfalls: str) -> str:
    """Return a description of the toy rubric with the given settings and shortfalls."""
    return SIDECAR.format(
        rubric_hash=TOY_RUBRIC_HASH,
        settings=settings,
        shortfalls=shortfalls,
        table_hash=TABLE_HASH_MARK,
    )


@pytest.fixture
def generated(toy_rubric: Rubric) -> ReferenceSet:
    """A small reference set from one writer whose first text has awkward characters."""
    awkward = '[{"text": "Line one,\\nline two with \\"quotes\\", commas; and a tab\\t."}]'
    responder = seed_responder({42: awkward}, count=1)
    settings = GeneratorSettings(texts_per_call=1, max_retries=1)
    return generate_references(toy_rubric, [writer(responder)], settings)


@pytest.mark.parametrize("suffix", [".parquet", ".csv"])
def test_a_reference_set_survives_a_save_and_load(
    generated: ReferenceSet, tmp_path: Path, suffix: str
) -> None:
    path = tmp_path / f"references{suffix}"
    generated.save(path)
    loaded = ReferenceSet.load(path)
    pd.testing.assert_frame_equal(loaded.frame, generated.frame)
    assert loaded.rubric_id == generated.rubric_id
    assert loaded.rubric_hash == generated.rubric_hash
    assert loaded.settings == generated.settings
    assert loaded.shortfalls == generated.shortfalls
    assert not list(tmp_path.glob("*.partial"))


def test_a_set_with_a_shortfall_keeps_it_through_the_files(
    toy_rubric: Rubric, tmp_path: Path
) -> None:
    short = generate_references(
        toy_rubric,
        [writer(seed_responder({42: "no json"}))],
        GeneratorSettings(max_retries=1),
    )
    assert len(short.shortfalls) == 1
    short.save(tmp_path / "short.csv")
    assert ReferenceSet.load(tmp_path / "short.csv").shortfalls == short.shortfalls


def test_an_unknown_suffix_and_a_missing_file_are_configuration_errors(
    generated: ReferenceSet, tmp_path: Path
) -> None:
    with pytest.raises(ConfigurationError, match="suffix"):
        generated.save(tmp_path / "references.json")
    with pytest.raises(ConfigurationError, match="does not exist"):
        ReferenceSet.load(tmp_path / "absent.parquet")
    generated.save(tmp_path / "references.csv")
    (tmp_path / "references.csv.json").unlink()
    with pytest.raises(ConfigurationError, match="references.csv.json"):
        ReferenceSet.load(tmp_path / "references.csv")


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("{not json", "not valid JSON"),
        ("[1, 2]", "JSON object"),
        ('{"rubric_id": "r"}', "lacks"),
        ('{"rubric_id": "r", "settings": {}, "shortfalls": []}', r"lacks \['writers', 'usage'\]"),
        (sidecar('{"bogus": 1}', "[]"), MALFORMED),
        (sidecar("{}", '[{"bogus": 1}]'), MALFORMED),
        (sidecar('{"texts_per_call": 0}', "[]"), "at least 1"),
        (sidecar('{"style": "loud"}', "[]"), "style"),
    ],
)
def test_a_malformed_sidecar_is_a_configuration_error_naming_the_file(
    generated: ReferenceSet, tmp_path: Path, content: str, message: str
) -> None:
    generated.save(tmp_path / "references.parquet")
    digest = table_hash(generated.frame, FRAME_COLUMNS)
    (tmp_path / "references.parquet.json").write_text(
        content.replace(TABLE_HASH_MARK, digest), encoding="utf-8"
    )
    with pytest.raises(ConfigurationError, match=message) as error:
        ReferenceSet.load(tmp_path / "references.parquet")
    assert "references.parquet.json" in str(error.value)


@pytest.mark.parametrize("suffix", [".parquet", ".csv"])
def test_a_failed_table_write_writes_no_description_and_leaves_no_partial_file(
    generated: ReferenceSet, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, suffix: str
) -> None:
    def fail(*args: object, **kwargs: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fail)
    monkeypatch.setattr(pd.DataFrame, "to_csv", fail)
    with pytest.raises(OSError, match="disk full"):
        generated.save(tmp_path / f"references{suffix}")
    assert list(tmp_path.iterdir()) == []


def test_the_sidecar_holds_the_rubric_the_settings_and_the_version(
    generated: ReferenceSet, toy_rubric: Rubric, tmp_path: Path
) -> None:
    generated.save(tmp_path / "references.parquet")
    description = json.loads((tmp_path / "references.parquet.json").read_text(encoding="utf-8"))
    assert description["rubric_id"] == generated.rubric_id
    assert description["rubric_hash"] == toy_rubric.content_hash()
    assert description["settings"]["texts_per_call"] == 1
    assert description["library_version"]


def test_a_generated_set_is_bound_to_the_content_of_its_rubric(
    generated: ReferenceSet, toy_rubric: Rubric
) -> None:
    assert generated.rubric_hash == toy_rubric.content_hash()
    expected = ["rubric_id", "rubric_hash", "bound_with", "settings"]
    assert list(generated.description())[:4] == expected


@pytest.mark.parametrize("value", NOT_RUBRIC_HASHES)
def test_a_set_without_the_content_hash_of_its_rubric_is_refused(
    generated: ReferenceSet, value: str
) -> None:
    with pytest.raises(ConfigurationError, match="rubric_hash") as error:
        ReferenceSet(generated.frame, generated.rubric_id, value, generated.settings)
    assert repr(value) in str(error.value) and "ReferenceSet.for_rubric" in str(error.value)


def test_for_rubric_binds_rows_to_the_id_and_the_content_hash_of_the_rubric(
    generated: ReferenceSet, toy_rubric: Rubric
) -> None:
    shortfall = Shortfall("writer-a", "tone", "negative", "low", 2, 1)
    writers = {"writer-a": {"backend": "hand"}}
    bound = ReferenceSet.for_rubric(
        generated.frame,
        toy_rubric,
        generated.settings,
        shortfalls=(shortfall,),
        writers=writers,
        usage=generated.usage,
    )
    assert (bound.rubric_id, bound.rubric_hash) == (toy_rubric.rubric_id, toy_rubric.content_hash())
    pd.testing.assert_frame_equal(bound.frame, generated.frame)
    assert bound.settings == generated.settings
    assert bound.shortfalls == (shortfall,) and dict(bound.writers) == writers
    assert dict(bound.usage) == dict(generated.usage)
    plain = ReferenceSet.for_rubric(generated.frame, toy_rubric, generated.settings)
    assert (plain.shortfalls, dict(plain.writers), dict(plain.usage)) == ((), {}, {})


def test_a_frame_with_other_columns_is_refused(generated: ReferenceSet) -> None:
    with pytest.raises(ConfigurationError, match="columns"):
        ReferenceSet(
            generated.frame.drop(columns=["text"]),
            generated.rubric_id,
            generated.rubric_hash,
            generated.settings,
        )


def test_rows_of_another_rubric_are_refused(generated: ReferenceSet, toy_rubric: Rubric) -> None:
    mixed = generated.frame.copy()
    mixed.loc[mixed.index[0], "rubric_id"] = "another_rubric"
    with pytest.raises(ConfigurationError, match="another_rubric") as error:
        ReferenceSet.for_rubric(mixed, toy_rubric, generated.settings)
    assert generated.rubric_id in str(error.value)
    with pytest.raises(ConfigurationError, match="another_rubric"):
        ReferenceSet(
            generated.frame, "another_rubric", toy_rubric.content_hash(), generated.settings
        )


def test_kept_returns_the_rows_that_were_not_screened_out(
    generated: ReferenceSet, toy_rubric: Rubric
) -> None:
    flagged = generated.frame.copy()
    flagged.loc[flagged.index[:3], "screened_out"] = True
    flagged.loc[flagged.index[:3], "screen_reason"] = "names construct term 'x'"
    changed = ReferenceSet.for_rubric(flagged, toy_rubric, generated.settings)
    assert len(changed.kept()) == len(flagged) - 3
    assert not changed.kept()["screened_out"].any()


def test_the_summary_reports_texts_shortfalls_screened_out_duplicates_and_balance(
    toy_rubric: Rubric,
) -> None:
    result = generate_references(
        toy_rubric,
        [writer(clean_responder(), "model-x"), writer(clean_responder(9), "model-y")],
        GeneratorSettings(),
    )
    summary = result.summary()
    assert summary["texts_per_cell"][("construct", "positive", "top")] == 19
    assert len(summary["texts_per_cell"]) == CELL_COUNT
    assert set(summary["shortfalls"].values()) == {1}
    assert {key[0] for key in summary["shortfalls"]} == {"model-y"}
    assert len(summary["shortfalls"]) == CELL_COUNT
    assert summary["screened_out"] == {"total": 0, "by_writer": {"model-x": 0, "model-y": 0}}
    assert summary["duplicates"] == 0
    assert summary["balance"] == {"balanced": True, "complete": True, "problems": []}


def test_the_summary_counts_duplicate_texts_within_a_writer(toy_rubric: Rubric) -> None:
    same = '[{"text": "Same note."}, {"text": " same NOTE. "}]'
    result = generate_references(
        toy_rubric,
        [writer(lambda request: same)],
        GeneratorSettings(texts_per_call=2),
    )
    assert result.summary()["duplicates"] == CELL_COUNT


def test_the_balance_check_names_the_writer_and_the_target(
    generated: ReferenceSet, toy_rubric: Rubric
) -> None:
    frame = generated.frame
    drop = frame[(frame["target"] == "tone") & (frame["polarity"] == "negative")].index[:1]
    unbalanced = ReferenceSet.for_rubric(frame.drop(index=drop), toy_rubric, generated.settings)
    summary = unbalanced.summary()
    assert summary["balance"]["balanced"] is False
    with pytest.raises(ConfigurationError) as error:
        unbalanced.assert_balanced()
    message = str(error.value)
    assert "writer-a" in message
    assert "tone" in message
    assert len(summary["balance"]["problems"]) == 1
    assert summary["balance"]["problems"][0] in message


def test_a_pole_without_kept_texts_makes_the_set_incomplete_but_not_unbalanced(
    toy_rubric: Rubric,
) -> None:
    attribute_seeds = {seed: "no json" for seed in range(48, 54)}
    result = generate_references(
        toy_rubric,
        [writer(seed_responder(attribute_seeds))],
        GeneratorSettings(max_retries=1),
    )
    balance = result.summary()["balance"]
    assert balance == {
        "balanced": True,
        "complete": False,
        "problems": [
            "writer-a/specificity/positive: no kept texts",
            "writer-a/specificity/negative: no kept texts",
        ],
    }
    result.assert_balanced()


def test_screened_out_poles_are_incomplete_and_unbalanced(toy_rubric: Rubric) -> None:
    def leaking(request: GenerationRequest) -> str:
        if "active LOW end" in request.user:
            return json.dumps([{"text": f"The Tone again {index}."} for index in range(10)])
        return texts_json(10, request.seed)

    summary = generate_references(toy_rubric, [writer(leaking)], GeneratorSettings()).summary()
    assert summary["balance"]["balanced"] is False
    assert summary["balance"]["complete"] is False
    assert "writer-a/construct/negative: no kept texts" in summary["balance"]["problems"]
