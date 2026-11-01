"""Verify, deduplicate, and order the passages the extraction stage returns.

The extraction stage of the two-stage LLM coder asks the model for verbatim
passages per marker. Every returned passage is matched against the text before
the scoring stage can see it: an exact substring of the normalised text counts
as verified, a near-exact alignment counts as verified for passages of at
least ``min_fuzzy_words`` words, and everything else is recorded as
unverified. The best score and window are kept for every passage, so a later
analysis can tell a near-miss from an invention.

Only verified passages that are not repeats within their marker survive. The
scoring stage receives them as one list, deduplicated across markers and in the
order they occur in the text.

``rapidfuzz`` is required: a pure-Python fallback would score differently, and
the verification decides which passages the scoring stage sees.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from copernicus.coders.matching import (
    MIN_FUZZY_WORDS,
    VERIFIED_EXACT,
    VERIFIED_FUZZY,
    VERIFIED_NONE,
    VERIFY_THRESHOLD,
    QuoteMatch,
    collapse_whitespace,
    match_quote,
    normalize,
    verifier_version,
)
from copernicus.errors import EvidenceContractError
from copernicus.rubric.model import Attribute

logger = logging.getLogger(__name__)

__all__ = [
    "MIN_FUZZY_WORDS",
    "VERIFIED_EXACT",
    "VERIFIED_FUZZY",
    "VERIFIED_NONE",
    "VERIFY_THRESHOLD",
    "EvidenceRow",
    "ProcessedExtraction",
    "QuoteMatch",
    "collapse_whitespace",
    "match_quote",
    "normalize",
    "process_extraction",
    "validate_extraction_block",
    "verifier_version",
]


@dataclass(frozen=True)
class EvidenceRow:
    """One passage the extraction stage returned, with its verification result.

    Attributes:
        attribute_id: Attribute the passage was extracted for.
        marker_id: Marker the passage was returned under.
        quote_index: Position of the passage in the marker's list.
        quote_raw: The passage as the model returned it.
        quote_norm: The passage with whitespace collapsed.
        verified: ``VERIFIED_EXACT``, ``VERIFIED_FUZZY``, or ``VERIFIED_NONE``.
        match_score: Exact-match score of 100.0 or the best alignment score.
        char_start: Start of the match or best window in the normalised text.
        char_end: End of the match or best window in the normalised text.
        duplicate_of: ``quote_index`` of the first identical passage under the
            same marker; None for the first occurrence.
        used_in_stage2: 1 when the passage is sent to the scoring stage, else 0.
    """

    attribute_id: str
    marker_id: str
    quote_index: int
    quote_raw: str
    quote_norm: str
    verified: int
    match_score: float
    char_start: int | None
    char_end: int | None
    duplicate_of: int | None
    used_in_stage2: int


@dataclass(frozen=True)
class ProcessedExtraction:
    """The evidence of one attribute after verification and deduplication.

    Attributes:
        evidence: Every returned passage, verified or not, in the order the
            model returned them.
        markers_to_rate: Names of the markers that kept at least one passage,
            in marker order.
        passages: The passages for the scoring stage: deduplicated across
            markers and ordered as they occur in the text.
        by_marker: Kept passages per marker name, in text order; a passage kept
            under two markers appears under both.
        evaluable: Per marker id, whether the marker kept any passage.
        justification: Per marker id, the kept passages joined by newlines.
    """

    evidence: tuple[EvidenceRow, ...]
    markers_to_rate: tuple[str, ...]
    passages: tuple[str, ...]
    by_marker: Mapping[str, tuple[str, ...]]
    evaluable: Mapping[str, bool]
    justification: Mapping[str, str]


def _marker_key_map(attribute: Attribute) -> dict[str, tuple[str, str]]:
    """Map every accepted JSON key, marker name or id, to ``(name, id)``."""
    key_map: dict[str, tuple[str, str]] = {}
    for marker in attribute.markers:
        key_map[marker.name] = (marker.name, marker.marker_id)
        key_map[marker.marker_id] = (marker.name, marker.marker_id)
    return key_map


def _validate_entry(key: str, entry: Any) -> tuple[bool, str]:
    """Check one marker entry for ``evaluable`` and ``quotes``."""
    if not isinstance(entry, Mapping):
        return False, f"marker {key!r}: entry is not an object"
    if "evaluable" not in entry or "quotes" not in entry:
        return False, f"marker {key!r}: missing 'evaluable' or 'quotes'"
    quotes = entry["quotes"]
    if not isinstance(quotes, list) or any(
        not isinstance(quote, str) or not quote.strip() for quote in quotes
    ):
        return False, f"marker {key!r}: 'quotes' must be a list of non-empty strings"
    evaluable = entry["evaluable"]
    if not isinstance(evaluable, bool):
        return False, f"marker {key!r}: 'evaluable' must be a bool"
    if evaluable and not quotes:
        return False, f"marker {key!r}: evaluable=true but zero quotes"
    if not evaluable and quotes:
        return False, f"marker {key!r}: evaluable=false but quotes present"
    return True, ""


def validate_extraction_block(block: Any, attribute: Attribute) -> tuple[bool, str]:
    """Validate the per-marker block the extraction stage returned.

    The rules, in order: the block is a JSON object; it has one entry per
    marker; every key is a marker name or id; every entry has a boolean
    ``evaluable`` and a list of non-empty strings ``quotes`` that agree with
    each other; no marker appears under both its name and its id.

    Args:
        block: The parsed per-marker object.
        attribute: The attribute the block was extracted for.

    Returns:
        ``(True, "")`` for a valid block, otherwise ``(False, reason)``. The
        function never raises, so a caller can retry on a bad answer.
    """
    if not isinstance(block, Mapping):
        return False, f"top-level type {type(block).__name__} is not a JSON object"
    expected = len(attribute.markers)
    if len(block) != expected:
        return False, f"extraction output has {len(block)} entries, expected {expected}"
    key_map = _marker_key_map(attribute)
    for key, entry in block.items():
        if key not in key_map:
            return False, f"unknown marker key {key!r} (typo or invented marker)"
        valid, reason = _validate_entry(key, entry)
        if not valid:
            return False, reason
    marker_ids = [key_map[key][1] for key in block]
    if len(set(marker_ids)) != len(marker_ids):
        return False, "a marker appears more than once (name and id used together)"
    return True, ""


@dataclass(frozen=True)
class _KeptQuote:
    """A verified, non-repeated passage of one marker."""

    start: int
    quote_norm: str
    quote_index: int


def _process_marker(
    attribute_id: str,
    marker_id: str,
    quotes: Sequence[str],
    doc_norm: str,
    *,
    threshold: float,
    min_fuzzy_words: int,
) -> tuple[list[EvidenceRow], list[_KeptQuote]]:
    """Record, verify, and deduplicate the passages of one marker.

    Returns:
        Every passage as an evidence row in returned order, and the kept
        passages sorted by their start in the text.
    """
    rows: list[EvidenceRow] = []
    kept: list[_KeptQuote] = []
    first_seen: dict[str, int] = {}
    for index, raw in enumerate(quotes):
        quote_norm = collapse_whitespace(raw)
        match = match_quote(
            quote_norm, doc_norm, threshold=threshold, min_fuzzy_words=min_fuzzy_words
        )
        duplicate_of = first_seen.setdefault(normalize(quote_norm), index)
        is_repeat = duplicate_of != index
        rows.append(
            EvidenceRow(
                attribute_id=attribute_id,
                marker_id=marker_id,
                quote_index=index,
                quote_raw=raw,
                quote_norm=quote_norm,
                verified=match.verified,
                match_score=match.score,
                char_start=match.start,
                char_end=match.end,
                duplicate_of=duplicate_of if is_repeat else None,
                used_in_stage2=0,
            )
        )
        if match.verified != VERIFIED_NONE and not is_repeat and match.start is not None:
            kept.append(_KeptQuote(match.start, quote_norm, index))
    kept.sort(key=lambda item: item.start)
    return rows, kept


def process_extraction(
    parsed: Mapping[str, Any],
    attribute: Attribute,
    text: str,
    marker_order: Sequence[str],
    *,
    threshold: float,
    min_fuzzy_words: int,
) -> ProcessedExtraction:
    """Verify, deduplicate, and order the passages of a validated block.

    A passage is kept when it is verified and not a repeat within its marker.
    The scoring stage receives the union of the kept passages over the markers
    to rate, deduplicated across markers and sorted by position in the text. A
    passage returned under several markers is credited to the one with the
    smallest ``(start, position in marker_order)``.

    Args:
        parsed: The validated per-marker block, keyed by marker name or id.
        attribute: The attribute the block was extracted for.
        text: The text the passages are verified against.
        marker_order: The attribute's marker names in the order of this
            observation; sets the order of the markers to rate.
        threshold: Minimum alignment score for a fuzzy match.
        min_fuzzy_words: Minimum word count for a fuzzy match.

    Returns:
        The processed evidence.

    Raises:
        EvidenceContractError: If a key is not a marker of the attribute, a
            marker appears twice or not at all, or ``marker_order`` is not a
            permutation of the marker names.
    """
    key_map = _marker_key_map(attribute)
    names = [marker.name for marker in attribute.markers]
    if sorted(marker_order) != sorted(names):
        raise EvidenceContractError(
            f"{attribute.attribute_id}: marker_order {list(marker_order)!r} is not a "
            f"permutation of the marker names {names!r}"
        )
    doc_norm = normalize(text)
    evidence: dict[str, list[EvidenceRow]] = {}
    kept_by_name: dict[str, list[_KeptQuote]] = {}
    id_by_name: dict[str, str] = {}
    for key, entry in parsed.items():
        if key not in key_map:
            raise EvidenceContractError(
                f"{attribute.attribute_id}: unknown marker key {key!r} after validation"
            )
        name, marker_id = key_map[key]
        if name in kept_by_name:
            raise EvidenceContractError(
                f"{attribute.attribute_id}: marker {name!r} appears more than once"
            )
        rows, kept = _process_marker(
            attribute.attribute_id,
            marker_id,
            entry["quotes"],
            doc_norm,
            threshold=threshold,
            min_fuzzy_words=min_fuzzy_words,
        )
        evidence[name] = rows
        kept_by_name[name] = kept
        id_by_name[name] = marker_id
    for name in names:
        if name not in kept_by_name:
            raise EvidenceContractError(
                f"{attribute.attribute_id}: marker {name!r} is missing from the extraction output"
            )

    markers_to_rate = tuple(name for name in marker_order if kept_by_name[name])
    chosen = _choose_passages(kept_by_name, marker_order, markers_to_rate)
    used = {(id_by_name[name], quote.quote_index) for name, quote in chosen}
    ordered_rows = tuple(
        dataclasses.replace(row, used_in_stage2=1)
        if (row.marker_id, row.quote_index) in used
        else row
        for rows in evidence.values()
        for row in rows
    )
    logger.debug(
        "%s: %d passages returned, %d sent to the scoring stage",
        attribute.attribute_id,
        len(ordered_rows),
        len(chosen),
    )
    return ProcessedExtraction(
        evidence=ordered_rows,
        markers_to_rate=markers_to_rate,
        passages=tuple(quote.quote_norm for _, quote in chosen),
        by_marker={
            name: tuple(quote.quote_norm for quote in kept) for name, kept in kept_by_name.items()
        },
        evaluable={id_by_name[name]: bool(kept) for name, kept in kept_by_name.items()},
        justification={
            id_by_name[name]: "\n".join(quote.quote_norm for quote in kept)
            for name, kept in kept_by_name.items()
        },
    )


def _choose_passages(
    kept_by_name: Mapping[str, Sequence[_KeptQuote]],
    marker_order: Sequence[str],
    markers_to_rate: Sequence[str],
) -> list[tuple[str, _KeptQuote]]:
    """Pick one marker per distinct passage and order the picks by position.

    Returns:
        ``(marker name, kept passage)`` pairs sorted by
        ``(start, position of the marker in marker_order)``.
    """
    rank = {name: position for position, name in enumerate(marker_order)}
    best: dict[str, tuple[str, _KeptQuote]] = {}
    for name in markers_to_rate:
        for quote in kept_by_name[name]:
            key = normalize(quote.quote_norm)
            current = best.get(key)
            if current is None or (quote.start, rank[name]) < (
                current[1].start,
                rank[current[0]],
            ):
                best[key] = (name, quote)
    return sorted(best.values(), key=lambda pick: (pick[1].start, rank[pick[0]]))
