"""Tests of resumable reference generation: the finished calls kept in a resume file."""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest
from references_support import Responder, clean_responder, fail_write, seed_responder, writer

import copernicus.references.files as reference_files
from copernicus.errors import BackendError, ConfigurationError
from copernicus.llm.base import GenerationRequest
from copernicus.llm.fake import FakeBackend
from copernicus.references import GeneratorSettings, ReferenceSet, generate_references
from copernicus.rubric.model import Rubric

CELL_COUNT = 24


class StoppingResponder:
    """Answers like ``inner`` until ``limit`` requests were answered, then fails."""

    def __init__(self, inner: Responder, limit: int) -> None:
        self.inner = inner
        self.limit = limit
        self.answered = 0

    def __call__(self, request: GenerationRequest) -> str:
        if self.answered >= self.limit:
            raise BackendError("the endpoint went away")
        self.answered += 1
        return self.inner(request)


class OtherEndpoint(FakeBackend):
    """A fake writer whose provenance names another endpoint."""

    def provenance(self) -> dict[str, str]:
        return {**super().provenance(), "endpoint": "https://elsewhere.example"}


def resume_description(path: Path) -> dict[str, object]:
    """Read the description beside a resume file."""
    return json.loads(path.with_name(path.name + ".json").read_text(encoding="utf-8"))


def stopped_run(rubric: Rubric, path: Path, limit: int = 10) -> None:
    """Run until the writer fails after ``limit`` answers, writing to the resume file."""
    stopping = writer(StoppingResponder(clean_responder(), limit))
    with pytest.raises(BackendError, match="went away"):
        generate_references(rubric, [stopping], GeneratorSettings(), resume=path)


@pytest.mark.parametrize("suffix", [".parquet", ".csv"])
def test_a_run_with_a_resume_file_equals_one_without_and_keeps_the_file(
    toy_rubric: Rubric, tmp_path: Path, suffix: str
) -> None:
    plain = generate_references(toy_rubric, [writer(clean_responder())], GeneratorSettings())
    path = tmp_path / f"progress{suffix}"
    resumable = generate_references(
        toy_rubric, [writer(clean_responder())], GeneratorSettings(), resume=path
    )
    pd.testing.assert_frame_equal(resumable.frame, plain.frame)
    description = resume_description(path)
    assert len(description["completed"]) == CELL_COUNT  # type: ignore[arg-type]
    assert description["rubric_hash"] == toy_rubric.content_hash()
    assert len(ReferenceSet.load(path).frame) == CELL_COUNT * 10


def test_a_stopped_run_keeps_its_finished_calls_and_the_rerun_makes_only_the_rest(
    toy_rubric: Rubric, tmp_path: Path
) -> None:
    path = tmp_path / "progress.parquet"
    stopped_run(toy_rubric, path)
    assert len(resume_description(path)["completed"]) == 10  # type: ignore[arg-type]
    assert len(ReferenceSet.load(path).frame) == 100
    finishing = writer(clean_responder())
    resumed = generate_references(toy_rubric, [finishing], GeneratorSettings(), resume=path)
    assert len(finishing.calls) == CELL_COUNT - 10
    plain = generate_references(toy_rubric, [writer(clean_responder())], GeneratorSettings())
    pd.testing.assert_frame_equal(resumed.frame, plain.frame)
    assert resumed.summary()["usage"] == plain.summary()["usage"]
    again = writer(clean_responder())
    generate_references(toy_rubric, [again], GeneratorSettings(), resume=path)
    assert again.calls == []


@pytest.mark.parametrize("suffix", [".parquet", ".csv"])
@pytest.mark.parametrize(
    ("failing", "dropped"),
    [("write_atomically", None), ("write_json", "10 rows of calls")],
    ids=["table write", "description write"],
)
def test_a_run_stopped_between_the_two_writes_of_a_call_loses_no_call_for_good(
    toy_rubric: Rubric,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    failing: str,
    dropped: str | None,
    suffix: str,
) -> None:
    path = tmp_path / f"progress{suffix}"
    fail_write(monkeypatch, reference_files, failing, at=5)
    with pytest.raises(OSError, match="disk full"):
        generate_references(
            toy_rubric, [writer(clean_responder())], GeneratorSettings(), resume=path
        )
    finishing = writer(clean_responder())
    with caplog.at_level(logging.INFO, logger="copernicus.references.progress"):
        resumed = generate_references(toy_rubric, [finishing], GeneratorSettings(), resume=path)
    assert len(finishing.calls) == CELL_COUNT - 4
    plain = generate_references(toy_rubric, [writer(clean_responder())], GeneratorSettings())
    pd.testing.assert_frame_equal(resumed.frame, plain.frame)
    assert resumed.shortfalls == () and resumed.summary()["usage"] == plain.summary()["usage"]
    notes = [record for record in caplog.records if "dropped" in record.getMessage()]
    if dropped is None:
        assert notes == []
    else:
        assert len(notes) == 1 and notes[0].levelno == logging.INFO
        assert dropped in notes[0].getMessage() and "['writer-a', 4, 0]" in notes[0].getMessage()


def test_a_run_stopped_in_its_first_write_leaves_a_table_that_a_rerun_refuses(
    toy_rubric: Rubric, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "progress.parquet"
    fail_write(monkeypatch, reference_files, "write_json", at=1)
    with pytest.raises(OSError, match="disk full"):
        generate_references(
            toy_rubric, [writer(clean_responder())], GeneratorSettings(), resume=path
        )
    backend = writer(clean_responder())
    with pytest.raises(ConfigurationError, match="no description .*delete the table"):
        generate_references(toy_rubric, [backend], GeneratorSettings(), resume=path)
    assert backend.calls == []
    path.unlink()
    restarted = generate_references(toy_rubric, [backend], GeneratorSettings(), resume=path)
    assert len(backend.calls) == CELL_COUNT and len(restarted.frame) == CELL_COUNT * 10


def test_a_description_without_its_table_is_refused_before_any_call(
    toy_rubric: Rubric, tmp_path: Path
) -> None:
    path = tmp_path / "progress.parquet"
    stopped_run(toy_rubric, path)
    path.unlink()
    backend = writer(clean_responder())
    with pytest.raises(ConfigurationError, match="progress.parquet does not exist"):
        generate_references(toy_rubric, [backend], GeneratorSettings(), resume=path)
    assert backend.calls == []


def test_a_resume_file_whose_listed_rows_were_replaced_is_refused_before_any_call(
    toy_rubric: Rubric, tmp_path: Path
) -> None:
    path = tmp_path / "progress.parquet"
    stopped_run(toy_rubric, path)
    frame = pd.read_parquet(path)
    frame.loc[3, "text"] = "A text that no call of this run wrote."
    frame.to_parquet(path, index=False)
    backend = writer(clean_responder())
    with pytest.raises(ConfigurationError, match="do not belong together") as refused:
        generate_references(toy_rubric, [backend], GeneratorSettings(), resume=path)
    message = str(refused.value)
    assert message.startswith(f"{path} and progress.parquet.json do not belong together")
    assert "the table holds 100 rows of the listed calls with hash" in message
    assert backend.calls == []


def test_a_resume_table_rewritten_by_pandas_without_a_change_resumes(
    toy_rubric: Rubric, tmp_path: Path
) -> None:
    path = tmp_path / "progress.parquet"
    stopped_run(toy_rubric, path)
    pd.read_parquet(path).to_parquet(path, index=False)
    finishing = writer(clean_responder())
    resumed = generate_references(toy_rubric, [finishing], GeneratorSettings(), resume=path)
    assert len(finishing.calls) == CELL_COUNT - 10 and len(resumed.frame) == CELL_COUNT * 10


def test_a_resume_file_without_a_table_hash_is_refused_before_any_call(
    toy_rubric: Rubric, tmp_path: Path
) -> None:
    path = tmp_path / "progress.parquet"
    stopped_run(toy_rubric, path)
    sidecar = path.with_name(path.name + ".json")
    description = resume_description(path)
    del description["table_hash"]
    sidecar.write_text(json.dumps(description), encoding="utf-8")
    backend = writer(clean_responder())
    with pytest.raises(ConfigurationError) as refused:
        generate_references(toy_rubric, [backend], GeneratorSettings(), resume=path)
    assert str(refused.value).startswith(
        f"{sidecar} has no table_hash: it was written by an earlier version of copernicus"
    )
    assert "start the run with another resume file" in str(refused.value)
    assert backend.calls == []


def test_the_usage_counts_every_attempt_as_the_backend_reported_it(toy_rubric: Rubric) -> None:
    responder = seed_responder({42: "not json"})
    backend = writer(responder)
    usage = generate_references(toy_rubric, [backend], GeneratorSettings()).summary()["usage"]
    prompt = sum(len(call.system.split()) + len(call.user.split()) for call in backend.calls)
    completion = sum(len(responder(call).split()) for call in backend.calls)
    assert usage == {
        "writer-a": {
            "calls": CELL_COUNT,
            "attempts": CELL_COUNT + 1,
            "prompt_tokens": prompt,
            "completion_tokens": completion,
        }
    }


def test_a_resume_file_of_other_settings_or_another_rubric_is_refused(
    toy_rubric: Rubric, tmp_path: Path
) -> None:
    path = tmp_path / "progress.parquet"
    stopped_run(toy_rubric, path)
    with pytest.raises(ConfigurationError, match="texts_per_call"):
        generate_references(
            toy_rubric,
            [writer(clean_responder())],
            GeneratorSettings(texts_per_call=5),
            resume=path,
        )
    changed = replace(toy_rubric, concept_definition=toy_rubric.concept_definition + " More.")
    with pytest.raises(ConfigurationError, match="rubric"):
        generate_references(changed, [writer(clean_responder())], GeneratorSettings(), resume=path)
    resumed = generate_references(
        toy_rubric, [writer(clean_responder())], GeneratorSettings(concurrency=4), resume=path
    )
    assert len(resumed.frame) == CELL_COUNT * 10


def test_a_resume_file_with_a_writer_that_is_not_in_the_run_is_refused(
    toy_rubric: Rubric, tmp_path: Path
) -> None:
    path = tmp_path / "progress.parquet"
    stopped_run(toy_rubric, path)
    with pytest.raises(ConfigurationError, match="writer-a"):
        generate_references(
            toy_rubric, [writer(clean_responder(), "writer-b")], GeneratorSettings(), resume=path
        )


def test_a_writer_whose_provenance_changed_is_refused(toy_rubric: Rubric, tmp_path: Path) -> None:
    path = tmp_path / "progress.parquet"
    stopped_run(toy_rubric, path)
    moved = OtherEndpoint(clean_responder(), model_name="writer-a")
    with pytest.raises(ConfigurationError, match="writer-a.*endpoint"):
        generate_references(toy_rubric, [moved], GeneratorSettings(), resume=path)
    assert moved.calls == []


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("prompt_hash", "0" * 16, "other prompts"),
        ("rubric_id", "another_rubric", "another_rubric"),
        ("call_index", 1, "not a call of this run"),
        ("target", "nonsense", "unknown cell"),
    ],
)
def test_rows_that_do_not_belong_to_the_run_are_refused_before_any_call(
    toy_rubric: Rubric, tmp_path: Path, column: str, value: object, message: str
) -> None:
    path = tmp_path / "progress.csv"
    stopped_run(toy_rubric, path)
    frame = pd.read_csv(path, keep_default_na=False)
    frame.loc[0, column] = value
    frame.to_csv(path, index=False)
    backend = writer(clean_responder())
    with pytest.raises(ConfigurationError, match=message):
        generate_references(toy_rubric, [backend], GeneratorSettings(), resume=path)
    assert backend.calls == []


def test_a_saved_reference_set_is_not_a_resume_file(toy_rubric: Rubric, tmp_path: Path) -> None:
    path = tmp_path / "references.parquet"
    generate_references(toy_rubric, [writer(clean_responder())], GeneratorSettings()).save(path)
    with pytest.raises(ConfigurationError, match="completed"):
        generate_references(
            toy_rubric, [writer(clean_responder())], GeneratorSettings(), resume=path
        )


@pytest.mark.parametrize("name", ["absent/progress.parquet", "progress.json"])
def test_a_resume_path_that_cannot_be_written_is_refused_before_any_call(
    toy_rubric: Rubric, tmp_path: Path, name: str
) -> None:
    backend = writer(clean_responder())
    with pytest.raises(ConfigurationError):
        generate_references(toy_rubric, [backend], GeneratorSettings(), resume=tmp_path / name)
    assert backend.calls == []
