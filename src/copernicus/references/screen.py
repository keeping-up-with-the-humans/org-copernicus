"""The leakage screen: reference texts that quote the vocabulary of the construct."""

from __future__ import annotations

import logging
import re
from typing import Any

import pandas as pd

from copernicus.rubric.model import Rubric

logger = logging.getLogger(__name__)

__all__ = ["leakage_terms", "screen_report", "screen_texts"]

MIN_TERM_LENGTH = 4
REASON_PREFIX = "names construct term '"
REASON_SUFFIX = "'"
REASON_PATTERN = re.compile(
    f"^{re.escape(REASON_PREFIX)}(.*){re.escape(REASON_SUFFIX)}$", re.DOTALL
)


def leakage_terms(rubric: Rubric) -> list[str]:
    """List the words of the rubric that a reference text must not quote.

    Args:
        rubric: The rubric the reference texts are written for.

    Returns:
        The concept, every attribute name, and every marker name, casefolded and
        deduplicated, without terms shorter than four characters, longest first (equal
        lengths in alphabetical order).
    """
    names = [rubric.concept]
    for attribute in rubric.attributes:
        names.append(attribute.name)
        names.extend(marker.name for marker in attribute.markers)
    terms = {name.casefold().strip() for name in names}
    kept = (term for term in terms if len(term) >= MIN_TERM_LENGTH)
    return sorted(kept, key=lambda term: (-len(term), term))


def _term_pattern(term: str) -> re.Pattern[str]:
    """Compile a casefolded term into a whole-word pattern; spaces match any whitespace."""
    body = r"\s+".join(re.escape(part) for part in term.split())
    return re.compile(rf"(?<!\w){body}(?!\w)")


def screen_texts(frame: pd.DataFrame, terms: list[str]) -> pd.DataFrame:
    """Flag the texts that contain one of the terms.

    Matching is casefolded and respects word boundaries, so ``tone`` does not match
    ``stone``. The first term in ``terms`` that a text contains gives the reason. Only whole
    names are matched, so a head word of a long concept phrase (``constructiveness`` of
    ``constructiveness of written feedback``) is not caught on its own.

    Args:
        frame: A table with a ``text`` column.
        terms: Casefolded terms, for example from `leakage_terms`.

    Returns:
        A copy of ``frame`` with ``screened_out`` (bool) and ``screen_reason`` (empty
        when the text is clean) appended.
    """
    folded = frame["text"].astype(str).str.casefold()
    reasons = pd.Series("", index=frame.index, dtype=object)
    for term in terms:
        pending = reasons == ""
        hit = folded.str.contains(_term_pattern(term), regex=True)
        reasons.loc[pending & hit] = f"{REASON_PREFIX}{term}{REASON_SUFFIX}"
    screened = frame.copy()
    screened["screened_out"] = (reasons != "").astype(bool)
    screened["screen_reason"] = reasons.astype(str)
    return screened


def screen_report(frame: pd.DataFrame) -> dict[str, Any]:
    """Count the hits per term and the share of texts screened out.

    Args:
        frame: A table with ``screened_out`` and ``screen_reason``, from `screen_texts`.

    Returns:
        ``n_texts``, ``n_screened_out``, ``share_screened_out`` (0.0 for an empty table),
        and ``by_term``: the number of texts screened out for each term, most frequent
        first.
    """
    n_texts = len(frame)
    flagged = frame.loc[frame["screened_out"], "screen_reason"]
    counts: dict[str, int] = {}
    for reason in flagged:
        matched = REASON_PATTERN.match(reason)
        term = matched.group(1) if matched else reason
        counts[term] = counts.get(term, 0) + 1
    ordered = dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))
    return {
        "n_texts": n_texts,
        "n_screened_out": len(flagged),
        "share_screened_out": len(flagged) / n_texts if n_texts else 0.0,
        "by_term": ordered,
    }
