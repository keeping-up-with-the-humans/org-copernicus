"""Evidence texts: each text rebuilt from the verified passages a two-stage run extracted."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, overload

import pandas as pd

from copernicus.coders import (
    export_evidence,
    export_provenance,
    export_scores,
    hash_text,
    normalize,
)
from copernicus.coders.two_stage_types import EXTRACTION_FAILED
from copernicus.data import TextTable
from copernicus.errors import ConfigurationError, EvidenceContractError, SettingsError
from copernicus.rubric.model import Rubric

logger = logging.getLogger(__name__)

__all__ = ["GRANULARITIES", "HybridTexts", "evidence_texts"]

GRANULARITIES = ("construct", "attribute")
FALLBACK_NONE = ""
FALLBACK_SHORT = "short"
FALLBACK_NO_EVIDENCE = "no_evidence"
MAX_IDS_IN_MESSAGE = 10
TWO_STAGE = "two_stage"
PASSAGE_ORDER = ["char_start", "char_end", "attribute_id", "marker_id", "quote_index"]


@dataclass(frozen=True)
class HybridTexts:
    """The texts the hybrid coder embeds, one per text of the table.

    Attributes:
        ids: Text ids as strings, in table order.
        texts: The evidence text of each text: its verified passages joined by newlines,
            or the raw text when ``fallback`` is not empty.
        fallback: ``""`` when the evidence text is used, ``"short"`` when the raw text has
            fewer than ``min_words`` words, ``"no_evidence"`` when no verified passage
            remains.
        attribute_id: The attribute whose markers supplied the passages; ``None`` for the
            construct.
        summary: ``n_texts``, ``n_short``, ``n_no_evidence``, ``n_evidence`` (texts that
            use their passages), ``mean_words_evidence`` (mean words of those evidence
            texts, 0.0 when there is none), and ``n_passages_embedded`` (passages in them).
    """

    ids: list[str]
    texts: list[str]
    fallback: list[str]
    attribute_id: str | None
    summary: Mapping[str, int | float]


def _listed(values: list[str]) -> str:
    """Return the first ids, quoted, with the number of the others."""
    shown = ", ".join(repr(value) for value in values[:MAX_IDS_IN_MESSAGE])
    rest = len(values) - MAX_IDS_IN_MESSAGE
    return shown + (f" and {rest} more" if rest > 0 else "")


def _check_settings(granularity: str, run_number: int, min_words: int) -> None:
    """Raise if a setting lies outside its range."""
    if granularity not in GRANULARITIES:
        raise SettingsError(f"granularity is {granularity!r}; use one of {GRANULARITIES}")
    if run_number < 1:
        raise SettingsError(f"run_number is {run_number}; it must be at least 1")
    if min_words < 0:
        raise SettingsError(f"min_words is {min_words}; it must not be negative")


def _check_texts(
    store_path: Path, table: TextTable, results: pd.DataFrame, run_number: int
) -> None:
    """Check that every text of the table is the text its results in the run were coded on.

    Args:
        store_path: The run store, for the message.
        table: The texts; every one has results in ``results``.
        results: The long score table of the run.
        run_number: The run, for the message.

    Raises:
        ConfigurationError: If the text hash of a text differs from the stored one; the
            first ten such ids are named, in table order.
    """
    stored: dict[str, set[str]] = {}
    for text_id, text_hash in zip(results["text_id"], results["text_hash"], strict=True):
        stored.setdefault(str(text_id), set()).add(str(text_hash))
    changed = [
        str(text_id)
        for text_id in table.ids()
        if stored[str(text_id)] != {hash_text(table.text_of(text_id))}
    ]
    if changed:
        raise ConfigurationError(
            f"{len(changed)} of {len(table)} texts differ from the texts that run {run_number} "
            f"of the store {store_path} was coded on: {_listed(changed)}; a store binds every "
            "text id to the text it was coded on; pass the texts the store was coded on, or "
            "code the changed texts into a new store"
        )


def _check_extractions(
    store_path: Path, table: TextTable, results: pd.DataFrame, run_number: int, min_words: int
) -> None:
    """Check that every text whose passages are read had all of its extractions done.

    A failed extraction leaves its attribute without passages: the evidence text would lack
    them, and a text left without any passage would be scored as one without evidence. A
    text with fewer than ``min_words`` words keeps its raw text whatever its passages, so a
    failed extraction takes nothing from it. A failed scoring leaves the passages in place.

    Args:
        store_path: The run store, for the message.
        table: The texts; every one has results in ``results``.
        results: The long score table of the run.
        run_number: The run, for the message.
        min_words: Texts with fewer words keep their raw text.

    Raises:
        ConfigurationError: If a text of the table with at least ``min_words`` words has an
            attribute result in ``extraction_failed``; the first ten such ids are named, in
            table order, with the remedies and where the full list is.
    """
    failed_ids = set(results.loc[results["status"] == EXTRACTION_FAILED, "text_id"].astype(str))
    frame = table.frame
    failed = [
        str(text_id)
        for text_id, n_words in zip(frame["text_id"], frame["n_words"], strict=True)
        if str(text_id) in failed_ids and int(n_words) >= min_words
    ]
    if failed:
        raise ConfigurationError(
            f"{len(failed)} of {len(table)} texts have an attribute whose extraction failed in "
            f"run {run_number} of the store {store_path}: {_listed(failed)}; the evidence text "
            "of such a text lacks the passages of that attribute, and a text without any "
            "passage would be scored as one without evidence; recode them with "
            "code_texts(..., CodingSettings(retry_failed=True)) (--retry-failed on the command "
            "line), or score a table without them; export_scores(store) lists every one: its "
            f"rows of run {run_number} with status {EXTRACTION_FAILED!r}"
        )


def _check_store(
    store_path: Path, rubric: Rubric, table: TextTable, run_number: int, min_words: int
) -> None:
    """Check that the store belongs to the rubric, has evidence, and covers every text.

    Raises:
        ConfigurationError: If the rubric hash differs, the store is not a two-stage
            store, a text of the table has no result in the run, a text of the table is
            not the text its results in the run were coded on, a text of the run carries
            more than one model label, or a text with at least ``min_words`` words has an
            attribute whose extraction failed in the run.
    """
    provenance = export_provenance(store_path)
    stored, given = provenance["rubric_hash"], rubric.content_hash()
    if stored != given:
        raise ConfigurationError(
            f"the store {store_path} was coded with rubric hash {stored}, the rubric given "
            f"({rubric.rubric_id}) has hash {given}"
        )
    if provenance["method"] != TWO_STAGE:
        raise ConfigurationError(
            f"the store {store_path} was coded with method {provenance['method']!r}; the "
            f"hybrid coder needs the verified passages of a {TWO_STAGE!r} run"
        )
    results = export_scores(store_path)
    results = results[results["run_number"] == run_number]
    present = set(results["text_id"].astype(str))
    missing = [str(text_id) for text_id in table.ids() if str(text_id) not in present]
    if missing:
        raise ConfigurationError(
            f"{len(missing)} of {len(table)} texts have no result in run {run_number} of the "
            f"store {store_path}: {_listed(missing)}"
        )
    _check_texts(store_path, table, results, run_number)
    labels = results[["text_id", "model_name", "temperature"]].drop_duplicates()
    labelled = labels.groupby("text_id").size()
    if (labelled > 1).any():
        names = sorted(set(labels["model_name"].astype(str)))
        raise ConfigurationError(
            f"run {run_number} of the store {store_path} holds several model labels {names} "
            f"for the same text ({int((labelled > 1).sum())} texts); the evidence of "
            "different models cannot be joined into one text"
        )
    _check_extractions(store_path, table, results, run_number, min_words)


def _passages_by_text(
    frame: pd.DataFrame, run_number: int, markers: set[str] | None
) -> dict[str, list[str]]:
    """Return per text the verified, non-duplicate passages in document order.

    Passages are ordered by start offset, then end offset, attribute id, marker id, and
    passage index, so passages that start at the same place come out in the same order
    whatever the order of the store.

    Args:
        frame: The evidence table of the store.
        run_number: The run whose passages count.
        markers: Marker ids to keep; ``None`` keeps all.

    Raises:
        EvidenceContractError: If a verified passage has no words, or lacks its start or
            its end offset.
    """
    keep = (frame["run_number"] == run_number) & (frame["verified"] > 0)
    keep &= frame["duplicate_of"].isna()
    if markers is not None:
        keep &= frame["marker_id"].isin(markers)
    kept = frame[keep]
    empty = kept["quote_norm"].str.strip() == ""
    unplaced = kept["char_start"].isna() | kept["char_end"].isna()
    if (empty | unplaced).any():
        row = kept[empty | unplaced].iloc[0]
        raise EvidenceContractError(
            f"verified passage {int(row['quote_index'])} of marker {row['marker_id']} for text "
            f"{row['text_id']} has {'no words' if empty.loc[row.name] else 'no match offsets'}"
        )
    ordered = kept.sort_values(PASSAGE_ORDER, kind="stable")
    passages: dict[str, list[str]] = {}
    for text_id, quote in zip(ordered["text_id"], ordered["quote_norm"], strict=True):
        passages.setdefault(str(text_id), []).append(str(quote))
    return passages


def _join_passages(quotes: list[str]) -> list[str]:
    """Keep the first of passages that are equal after normalisation, in order."""
    seen: set[str] = set()
    kept = []
    for quote in quotes:
        key = normalize(quote)
        if key not in seen:
            seen.add(key)
            kept.append(quote)
    return kept


def _assemble(
    table: TextTable, passages: dict[str, list[str]], min_words: int, attribute_id: str | None
) -> HybridTexts:
    """Build the evidence texts of one granularity cell, counting the fallbacks."""
    ids: list[str] = []
    texts: list[str] = []
    fallback: list[str] = []
    n_passages = 0
    words = 0
    frame = table.frame
    for text_id, raw, n_words in zip(
        frame["text_id"], frame["text"], frame["n_words"], strict=True
    ):
        key = str(text_id)
        clean = " ".join(str(raw).split())
        ids.append(key)
        if int(n_words) < min_words:
            texts.append(clean)
            fallback.append(FALLBACK_SHORT)
            continue
        kept = _join_passages(passages.get(key, []))
        if not kept:
            texts.append(clean)
            fallback.append(FALLBACK_NO_EVIDENCE)
            continue
        joined = "\n".join(kept)
        texts.append(joined)
        fallback.append(FALLBACK_NONE)
        n_passages += len(kept)
        words += len(joined.split())
    n_evidence = fallback.count(FALLBACK_NONE)
    summary: dict[str, int | float] = {
        "n_texts": len(ids),
        "n_short": fallback.count(FALLBACK_SHORT),
        "n_no_evidence": fallback.count(FALLBACK_NO_EVIDENCE),
        "n_evidence": n_evidence,
        "mean_words_evidence": words / n_evidence if n_evidence else 0.0,
        "n_passages_embedded": n_passages,
    }
    return HybridTexts(ids, texts, fallback, attribute_id, summary)


@overload
def evidence_texts(
    store_path: Path,
    rubric: Rubric,
    table: TextTable,
    *,
    granularity: Literal["construct"],
    run_number: int = ...,
    min_words: int = ...,
) -> HybridTexts: ...


@overload
def evidence_texts(
    store_path: Path,
    rubric: Rubric,
    table: TextTable,
    *,
    granularity: Literal["attribute"],
    run_number: int = ...,
    min_words: int = ...,
) -> dict[str, HybridTexts]: ...


def evidence_texts(
    store_path: Path,
    rubric: Rubric,
    table: TextTable,
    *,
    granularity: Literal["construct", "attribute"],
    run_number: int = 1,
    min_words: int = 40,
) -> HybridTexts | dict[str, HybridTexts]:
    """Rebuild each text from the verified passages of a two-stage run.

    Per text, in table order: a text with fewer than ``min_words`` words (the ``n_words``
    of the table) keeps its raw text with the fallback ``"short"``. Otherwise the passages
    of the run that were verified and are not repeats within their marker are ordered by
    their position in the text (start offset, then end offset, attribute id, marker id, and
    passage index), reduced to the first of those that are equal after
    `copernicus.coders.normalize`, and joined by newlines. A text left without a
    passage keeps its raw text with the fallback ``"no_evidence"``. At ``"attribute"``
    granularity only the passages of the markers of one attribute count, and every
    attribute gets its own texts. A text that is not short and has an attribute whose
    extraction failed in the run is refused, because its passages are incomplete; a failed
    scoring leaves the passages in place, so such a text is rebuilt as any other.

    Args:
        store_path: Run store of a two-stage run.
        rubric: The rubric the run used; its content hash must equal the stored one.
        table: The texts, as they were coded: the text hash of each must equal the one
            stored with its results in the run.
        granularity: ``"construct"`` for one text per text of the table, ``"attribute"``
            for one per attribute.
        run_number: The run whose passages are read.
        min_words: Texts with fewer words stay raw.

    Returns:
        The texts at ``"construct"`` granularity; at ``"attribute"`` granularity a mapping
        from each attribute id, in rubric order, to its texts.

    Raises:
        SettingsError: If the granularity is unknown, ``run_number`` is below 1, or
            ``min_words`` is negative.
        ConfigurationError: If the rubric differs from the stored one, the store is not a
            two-stage store, texts of the table have no result in the run, differ from the
            texts the run coded, or have at least ``min_words`` words and an attribute whose
            extraction failed in the run (the first ten are listed in each case; recode
            failed extractions with ``CodingSettings(retry_failed=True)``), or a text of the
            run carries several model labels.
        EvidenceContractError: If a verified passage has no words, or lacks its start or
            its end offset.
    """
    _check_settings(granularity, run_number, min_words)
    _check_store(store_path, rubric, table, run_number, min_words)
    evidence = export_evidence(store_path)
    if granularity == "construct":
        passages = _passages_by_text(evidence, run_number, None)
        return _assemble(table, passages, min_words, None)
    result = {}
    for attribute in rubric.attributes:
        markers = {marker.marker_id for marker in attribute.markers}
        passages = _passages_by_text(evidence, run_number, markers)
        result[attribute.attribute_id] = _assemble(
            table, passages, min_words, attribute.attribute_id
        )
    return result
