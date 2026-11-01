"""Tests of what the reference generator records about every call and every writer."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from references_support import clean_responder, seed_responder, texts_json, writer

from copernicus.llm.base import TOP_P, GenerationRequest, GenerationResult
from copernicus.llm.fake import FakeBackend
from copernicus.references import GeneratorSettings, ReferenceSet, generate_references
from copernicus.rubric.model import Rubric

CALL_COLUMNS = ("attempt", "seed", "prompt_tokens", "completion_tokens")
APPLIED_COLUMNS = ("temperature_applied", "top_p_applied")


class SilentBackend(FakeBackend):
    """A fake writer that reports neither tokens nor the sampling it applied."""

    def generate(self, request: GenerationRequest) -> GenerationResult:
        answer = super().generate(request)
        return GenerationResult(answer.text, None, None, None, None, None)


def test_each_row_records_the_attempt_seed_tokens_and_sampling_of_its_call(
    toy_rubric: Rubric,
) -> None:
    backend = writer(seed_responder({42: "not json"}))
    result = generate_references(toy_rubric, [backend], GeneratorSettings())
    frame = result.frame
    first, later = frame.iloc[0], frame.iloc[10]
    assert (first["attempt"], first["seed"]) == (1, 100042)
    assert (later["attempt"], later["seed"]) == (0, 43)
    kept = backend.calls[1]
    assert first["prompt_tokens"] == len(kept.system.split()) + len(kept.user.split())
    assert first["completion_tokens"] == len(texts_json(10, 100042).split())
    assert (first["temperature_applied"], first["top_p_applied"]) == (0.9, TOP_P)
    assert [str(frame[name].dtype) for name in ("call_index", "attempt", "seed")] == ["int64"] * 3
    assert [str(frame[name].dtype) for name in (*CALL_COLUMNS[2:], *APPLIED_COLUMNS)] == [
        "float64"
    ] * 4


@pytest.mark.parametrize("suffix", [".parquet", ".csv"])
def test_values_a_backend_does_not_report_are_missing_and_survive_the_files(
    toy_rubric: Rubric, tmp_path: Path, suffix: str
) -> None:
    silent = SilentBackend(clean_responder(), model_name="silent")
    result = generate_references(toy_rubric, [silent], GeneratorSettings())
    frame = result.frame
    for name in (*CALL_COLUMNS[2:], *APPLIED_COLUMNS):
        assert frame[name].isna().all(), name
    assert (frame["attempt"] == 0).all()
    path = tmp_path / f"references{suffix}"
    result.save(path)
    pd.testing.assert_frame_equal(ReferenceSet.load(path).frame, frame)


@pytest.mark.parametrize("suffix", [".parquet", ".csv"])
def test_the_set_records_the_provenance_of_every_writer(
    toy_rubric: Rubric, tmp_path: Path, suffix: str
) -> None:
    writers = [writer(clean_responder(), "model-x"), writer(clean_responder(), "model-y")]
    result = generate_references(toy_rubric, writers, GeneratorSettings())
    assert result.writers == {
        "model-x": writers[0].provenance(),
        "model-y": writers[1].provenance(),
    }
    path = tmp_path / f"references{suffix}"
    result.save(path)
    description = json.loads(path.with_name(path.name + ".json").read_text(encoding="utf-8"))
    assert description["writers"]["model-y"]["model"] == "model-y"
    assert description["usage"]["model-x"]["calls"] == 24
    loaded = ReferenceSet.load(path)
    assert loaded.writers == result.writers and loaded.usage == result.usage
    pd.testing.assert_frame_equal(loaded.frame, result.frame)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("writers", '["model-x"]'),
        ("writers", '{"model-x": "kind"}'),
        ("writers", '{"model-x": {"kind": 1}}'),
        ("usage", '{"writer-a": {"calls": "many"}}'),
        ("usage", '{"writer-a": {"calls": 1}}'),
        ("usage", '{"writer-a": 3}'),
    ],
)
def test_malformed_writers_or_usage_in_the_sidecar_are_refused(
    toy_rubric: Rubric, tmp_path: Path, key: str, value: str
) -> None:
    from copernicus.errors import ConfigurationError

    result = generate_references(toy_rubric, [writer(clean_responder())], GeneratorSettings())
    path = tmp_path / "references.parquet"
    result.save(path)
    sidecar = path.with_name(path.name + ".json")
    description = json.loads(sidecar.read_text(encoding="utf-8"))
    description[key] = json.loads(value)
    sidecar.write_text(json.dumps(description), encoding="utf-8")
    with pytest.raises(ConfigurationError, match=key) as error:
        ReferenceSet.load(path)
    assert "references.parquet.json" in str(error.value)
