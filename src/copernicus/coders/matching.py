"""Match a quoted passage against a text with an exact search and a local alignment.

A passage is verified when it occurs in the normalised text as an exact
substring, or when its best local alignment scores at least the threshold and
the passage has at least ``min_fuzzy_words`` words. The best score and window
are reported for every passage, verified or not, so that a later analysis can
tell a near-miss from an invention.

``rapidfuzz`` is required: a pure-Python fallback would score differently, and
the verification decides which passages the scoring stage sees.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass

import rapidfuzz
from rapidfuzz import fuzz

logger = logging.getLogger(__name__)

__all__ = [
    "MIN_FUZZY_WORDS",
    "VERIFIED_EXACT",
    "VERIFIED_FUZZY",
    "VERIFIED_NONE",
    "VERIFY_THRESHOLD",
    "QuoteMatch",
    "collapse_whitespace",
    "match_quote",
    "normalize",
    "verifier_version",
]

VERIFY_THRESHOLD = 88.0
"""The lowest alignment score, from 0 to 100, at which a near-exact passage is verified."""
MIN_FUZZY_WORDS = 3
"""The fewest words of a passage that may be verified by approximate matching.

A shorter passage must match exactly.
"""
VERIFIED_NONE = 0
"""The ``verified`` value of a passage that is not found in the text."""
VERIFIED_EXACT = 1
"""The ``verified`` value of a passage that is a substring of the normalised text."""
VERIFIED_FUZZY = 2
"""The ``verified`` value of a passage that matches the text approximately, above the threshold."""
EXACT_SCORE = 100.0

_WHITESPACE = re.compile(r"\s+")
_PUNCTUATION_MAP = str.maketrans(
    {
        "’": "'",
        "‘": "'",
        "“": '"',
        "”": '"',
        "–": "-",
        "—": "-",
    }
)


def verifier_version() -> str:
    """Return the name and version of the alignment backend, for run provenance."""
    return f"rapidfuzz {rapidfuzz.__version__}"


def collapse_whitespace(text: str) -> str:
    """Collapse every run of whitespace to one space and strip both ends.

    Args:
        text: Any string.

    Returns:
        The string with single spaces between words.
    """
    return _WHITESPACE.sub(" ", text).strip()


def normalize(text: str) -> str:
    """Casefold, unify quotation marks and dashes, and collapse whitespace.

    Args:
        text: Any string.

    Returns:
        The comparison form used for matching and for deduplication.
    """
    unified = unicodedata.normalize("NFC", text).translate(_PUNCTUATION_MAP)
    return _WHITESPACE.sub(" ", unified).casefold().strip()


@dataclass(frozen=True)
class QuoteMatch:
    """Outcome of matching one passage against the normalised text.

    Attributes:
        verified: ``VERIFIED_EXACT``, ``VERIFIED_FUZZY``, or ``VERIFIED_NONE``.
        score: 100.0 for an exact match, otherwise the best alignment score.
        start: Offset in the normalised text where the match, or the best
            alignment window of an unverified passage, begins; None for an
            empty passage or text.
        end: Offset where that match or window ends; None as for ``start``.
    """

    verified: int
    score: float
    start: int | None
    end: int | None


def match_quote(
    quote: str,
    doc_norm: str,
    *,
    threshold: float = VERIFY_THRESHOLD,
    min_fuzzy_words: int = MIN_FUZZY_WORDS,
) -> QuoteMatch:
    """Match a passage against the normalised text.

    An exact substring is tried first. Otherwise the best local alignment is
    accepted at ``threshold`` or above, but only for passages of at least
    ``min_fuzzy_words`` words; shorter passages must match exactly. A passage
    longer than the text is judged on the full-string ratio, because a local
    alignment would score a passage that contains the whole text plus invented
    additions near 100.

    Args:
        quote: The passage in any form; normalisation is applied here.
        doc_norm: The text after `normalize`.
        threshold: Minimum alignment score for a fuzzy match.
        min_fuzzy_words: Minimum word count for a fuzzy match.

    Returns:
        The match. An empty passage or text is unverified with score 0 and no
        window.
    """
    quote_norm = normalize(quote)
    if not quote_norm or not doc_norm:
        return QuoteMatch(VERIFIED_NONE, 0.0, None, None)
    position = doc_norm.find(quote_norm)
    if position != -1:
        return QuoteMatch(VERIFIED_EXACT, EXACT_SCORE, position, position + len(quote_norm))
    long_enough = len(quote_norm.split()) >= min_fuzzy_words
    if len(quote_norm) > len(doc_norm):
        score = float(fuzz.ratio(quote_norm, doc_norm))
        verified = VERIFIED_FUZZY if score >= threshold and long_enough else VERIFIED_NONE
        return QuoteMatch(verified, score, 0, len(doc_norm))
    alignment = fuzz.partial_ratio_alignment(quote_norm, doc_norm)
    if alignment is None:
        return QuoteMatch(VERIFIED_NONE, 0.0, None, None)
    score = float(alignment.score)
    verified = VERIFIED_FUZZY if score >= threshold and long_enough else VERIFIED_NONE
    return QuoteMatch(verified, score, int(alignment.dest_start), int(alignment.dest_end))
