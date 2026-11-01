"""The column values that a result row writes into the results and evidence tables."""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Mapping, Sequence
from typing import Any

from copernicus._version import __version__
from copernicus.coders.store_schema import (
    ATTRIBUTE_FIELDS,
    MARKER_FIELDS,
    attribute_column,
    marker_column,
)
from copernicus.coders.store_types import ResultRow, RunIdentity
from copernicus.coders.text_hash import TEXT_HASH_PATTERN
from copernicus.coders.two_stage_types import AttributeResult
from copernicus.errors import StoreError

logger = logging.getLogger(__name__)

__all__ = ["row_values"]


def row_values(
    row: ResultRow,
    identity: RunIdentity,
    markers: Mapping[str, Sequence[str]],
    names: tuple[Sequence[str], Sequence[str]],
    created_at: str,
) -> tuple[list[Any], list[tuple[Any, ...]]]:
    """Check a row against the rubric and return the values of its results and evidence rows.

    Args:
        row: The result row.
        identity: The identity of the run, which supplies the model, temperature, and seed.
        markers: The marker ids of each attribute id, in rubric order.
        names: The column names of ``results`` and of ``evidence``, in table order.
        created_at: Time stamp of the write.

    Returns:
        The values of the results row and of each evidence row, in column order.

    Raises:
        StoreError: If the row does not cover exactly the attributes of the rubric, its text
            hash is not 64 lowercase hexadecimal characters, an attribute result refers to an
            id outside the rubric, or the attribute results disagree on the sampling the
            backend applied.
    """
    known, given = set(markers), set(row.attributes)
    if known != given:
        raise StoreError(
            f"text_id={row.text_id!r}: attributes must be exactly {sorted(known)}; "
            f"missing {sorted(known - given)}, unexpected {sorted(given - known)}"
        )
    if not isinstance(row.text_hash, str) or not TEXT_HASH_PATTERN.fullmatch(row.text_hash):
        raise StoreError(
            f"text_id={row.text_id!r}: text_hash must be 64 lowercase hexadecimal "
            f"characters, the hash_text of the coded text; got {row.text_hash!r}"
        )
    result_names, evidence_names = names
    assignment = row.assignment
    temperature_applied, top_p_applied = _applied_sampling(row)
    cells: dict[str, Any] = {
        "text_id": row.text_id,
        "text_key": row.text_key,
        "text_hash": row.text_hash,
        "run_number": row.run_number,
        "model_name": identity.model_name,
        "temperature": identity.temperature,
        "seed": identity.seed,
        "role_id": assignment.role_id,
        "task_id": assignment.task_id,
        "level_id": assignment.level_id,
        "shuffle_seed": assignment.shuffle_seed,
        "seconds": row.seconds,
        "prompt_tokens": row.prompt_tokens,
        "completion_tokens": row.completion_tokens,
        "created_at": created_at,
        "library_version": __version__,
        "temperature_applied": temperature_applied,
        "top_p_applied": top_p_applied,
    }
    evidence_values: list[tuple[Any, ...]] = []
    for attribute_id, result in row.attributes.items():
        _check_result(row, attribute_id, result, markers[attribute_id])
        for field, _ in ATTRIBUTE_FIELDS:
            cells[attribute_column(attribute_id, field)] = getattr(result, field)
        for evidence in result.evidence:
            evidence_cells = {
                "text_id": row.text_id,
                "run_number": row.run_number,
                "model_name": identity.model_name,
                "temperature": identity.temperature,
                **dataclasses.asdict(evidence),
            }
            evidence_values.append(tuple(evidence_cells[n] for n in evidence_names))
        for marker_id in markers[attribute_id]:
            # A marker the result does not mention is stored as NULL in both columns.
            evaluable = result.evaluable[marker_id] if marker_id in result.evaluable else None
            cells[marker_column(marker_id, MARKER_FIELDS[0][0])] = (
                None if evaluable is None else int(evaluable)
            )
            cells[marker_column(marker_id, MARKER_FIELDS[1][0])] = (
                result.justification[marker_id] if marker_id in result.justification else None
            )
    return [cells[name] for name in result_names], evidence_values


def _check_result(
    row: ResultRow, attribute_id: str, result: AttributeResult, marker_ids: Sequence[str]
) -> None:
    """Check that an attribute result only refers to ids of the rubric."""
    where = f"text_id={row.text_id!r}, attribute {attribute_id!r}"
    if result.attribute_id != attribute_id:
        raise StoreError(f"{where}: result belongs to {result.attribute_id!r}")
    allowed = set(marker_ids)
    stray = (set(result.evaluable) | set(result.justification)) - allowed
    stray |= {e.marker_id for e in result.evidence} - allowed
    if stray:
        raise StoreError(f"{where}: unknown markers {sorted(stray)}")
    if any(e.attribute_id != attribute_id for e in result.evidence):
        raise StoreError(f"{where}: evidence belongs to another attribute")


def _applied_sampling(row: ResultRow) -> tuple[float | None, float | None]:
    """Return the temperature and top_p the backend applied in the scoring calls of a row.

    Every attribute of a cell is scored with the same settings, so the attribute results
    that made a scoring call carry the same pair; results without one carry no values.

    Raises:
        StoreError: If two attribute results of the row carry different values.
    """
    pairs = {(r.temperature_applied, r.top_p_applied) for r in row.attributes.values()}
    pairs.discard((None, None))
    if len(pairs) > 1:
        raise StoreError(
            f"text_id={row.text_id!r}: the attribute results disagree on the applied "
            f"sampling (temperature, top_p): {sorted(pairs, key=repr)}"
        )
    return pairs.pop() if pairs else (None, None)
