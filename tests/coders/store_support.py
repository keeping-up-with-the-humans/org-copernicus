"""Shared builders for the tests of the run store and its exports."""

import dataclasses
import sqlite3
from collections.abc import Mapping
from pathlib import Path

from copernicus.coders import (
    Assignment,
    AttributeResult,
    EvidenceRow,
    ResultRow,
    RunIdentity,
    RunStore,
    TwoStageConfig,
    hash_text,
    identity_settings,
    load_wording_pool,
)
from copernicus.rubric.model import Rubric

MODEL = "model-a"
TEMPERATURE = 0.3
ASSIGNMENT = Assignment("role_baseline", "task_baseline", "level_independent", 7)
STATUS_OF_SCORE = {None: "scoring_failed", 1: "no_evidence"}
SCORES = {"specificity": 4, "tone": 5, "reasoning": 2}
N_PASSAGES = 2


def make_identity(
    rubric: Rubric,
    model_name: str = MODEL,
    config: TwoStageConfig | None = None,
    extra: Mapping[str, str] | None = None,
) -> RunIdentity:
    settings = identity_settings(
        config if config is not None else TwoStageConfig(), load_wording_pool()
    )
    settings.update(extra or {})
    return RunIdentity(
        method="two_stage",
        model_name=model_name,
        temperature=TEMPERATURE,
        seed=11,
        rubric_hash=rubric.content_hash(),
        settings=settings,
    )


def open_store(path: str | Path, rubric: Rubric, identity: RunIdentity | None = None) -> RunStore:
    return RunStore(path, rubric, identity if identity is not None else make_identity(rubric))


def unvalidated_rubric(rubric: Rubric, **fields: object) -> Rubric:
    """Return a copy of the rubric with some fields replaced, made without its constructor.

    A rubric made through its constructor obeys the id rule. This copy stands in for an object
    that bypassed the constructor, so a test can show that the store applies the rule itself.
    """
    copy = object.__new__(Rubric)
    for field in dataclasses.fields(Rubric):
        value = fields[field.name] if field.name in fields else getattr(rubric, field.name)
        object.__setattr__(copy, field.name, value)
    return copy


def evidence_for(rubric: Rubric, attribute_id: str, n_passages: int) -> tuple[EvidenceRow, ...]:
    marker_id = rubric.attribute(attribute_id).markers[0].marker_id
    return tuple(
        EvidenceRow(
            attribute_id=attribute_id,
            marker_id=marker_id,
            quote_index=index,
            quote_raw=f"raw passage {index}",
            quote_norm=f"raw passage {index}",
            verified=1,
            match_score=100.0,
            char_start=index * 10 if index else None,
            char_end=index * 10 + 9 if index else None,
            duplicate_of=None,
            used_in_stage2=1,
        )
        for index in range(n_passages)
    )


def make_result(
    rubric: Rubric, attribute_id: str, score: int | None, n_passages: int = 1
) -> AttributeResult:
    status = "ok" if score not in STATUS_OF_SCORE else STATUS_OF_SCORE[score]
    first_marker = rubric.attribute(attribute_id).markers[0].marker_id
    return AttributeResult(
        attribute_id=attribute_id,
        score=score,
        status=status,  # type: ignore[arg-type]
        explanation=f"explanation of {attribute_id}",
        stage1_attempt=0,
        stage2_attempt=None if score is None else 0,
        evidence=evidence_for(rubric, attribute_id, n_passages),
        evaluable={first_marker: True},
        justification={first_marker: "raw passage 0"},
        seconds=1.5,
        prompt_tokens=10,
        completion_tokens=5,
    )


def text_of(text_id: str) -> str:
    """Return the text a row of the builders was coded on, unless a test names another."""
    return f"text of {text_id}"


def make_row(
    rubric: Rubric,
    text_id: str,
    run_number: int,
    scores: Mapping[str, int | None],
    n_passages: int = 1,
    text: str | None = None,
) -> ResultRow:
    return ResultRow(
        text_id=text_id,
        text_key=len(text_id),
        text_hash=hash_text(text_of(text_id) if text is None else text),
        run_number=run_number,
        assignment=ASSIGNMENT,
        attributes={
            a.attribute_id: make_result(rubric, a.attribute_id, scores[a.attribute_id], n_passages)
            for a in rubric.attributes
        },
        seconds=4.5,
        prompt_tokens=30,
        completion_tokens=15,
    )


def make_shard(
    rubric: Rubric,
    path: Path,
    texts: list[str],
    model_name: str = MODEL,
    contents: Mapping[str, str] | None = None,
) -> Path:
    """Write a store with one run of each text, with the same scores for every text.

    ``contents`` names the text of an id that was coded on another text than `text_of`.
    """
    changed = contents or {}
    rows = [
        make_row(rubric, t, 1, SCORES, N_PASSAGES, changed[t] if t in changed else None)
        for t in texts
    ]
    with RunStore(path, rubric, make_identity(rubric, model_name=model_name)) as store:
        store.record_wording(load_wording_pool())
        store.insert(rows)
    return path


def table_counts(path: Path) -> dict[str, int]:
    """Count the rows of each table of a store file."""
    with sqlite3.connect(path) as connection:
        return {
            t: connection.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
            for t in ("results", "evidence", "wording", "shuffles", "run_info")
        }
