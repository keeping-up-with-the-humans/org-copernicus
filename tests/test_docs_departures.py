"""Every methods page of a method that the paper's analysis code also has lists its departures."""

from __future__ import annotations

from pathlib import Path

import pytest

METHODS = Path(__file__).resolve().parents[1] / "docs" / "methods"
SECTION = "## Where the library departs from the paper's analysis code"
# The pages of a method that the paper's analysis code also has: each lists its departures.
WITH_COUNTERPART = (
    "two-stage-llm-coder.md",
    "one-stage-llm-coder.md",
    "similarity-coder.md",
    "hybrid-coder.md",
    "nlp-ml-baseline.md",
    "validity-panel.md",
    "corrections/index.md",
    "corrections/regression-calibration.md",
    "corrections/marginal-matching.md",
    "corrections/multiple-imputation.md",
    "corrections/misclassification-matrix.md",
    "corrections/pseudo-values.md",
    "corrections/latent-pooling.md",
    "corrections/supervised-stacking.md",
    "corrections/classical-corrections.md",
    "corrections/embedding-augmented-calibration.md",
    "corrections/correction-ensemble.md",
    "corrections/instrumented-coding.md",
)
# The overview and the bibliography describe no method of their own.
WITHOUT_COUNTERPART = ("index.md", "references.md")
# An entry says what the library does and what the paper's analysis code did, which takes more
# than a few words; a placeholder word, in capitals or not, marks an entry that was never written.
MIN_ENTRY_WORDS = 8
PLACEHOLDERS = frozenset({"TODO", "TBD", "FIXME", "XXX"})
PUNCTUATION = ".,;:!?()[]`*\"'"


def departures(page: str) -> list[str]:
    """The items of a page's departure section: its lines that open with "- "."""
    text = (METHODS / page).read_text(encoding="utf-8")
    assert text.count(SECTION) == 1, f"{page}: the section must appear exactly once"
    body = text.split(SECTION, 1)[1].split("\n## ", 1)[0]
    return [line for line in body.splitlines() if line.startswith("- ")]


def test_every_methods_page_is_classified() -> None:
    found = {path.relative_to(METHODS).as_posix() for path in METHODS.rglob("*.md")}
    assert found == set(WITH_COUNTERPART) | set(WITHOUT_COUNTERPART)


@pytest.mark.parametrize("page", WITH_COUNTERPART)
def test_a_page_with_a_counterpart_lists_its_departures(page: str) -> None:
    entries = departures(page)
    assert entries, f"{page}: the section lists no departure"
    for entry in entries:
        words = entry.removeprefix("- ").split()
        assert len(words) >= MIN_ENTRY_WORDS, (
            f"{page}: an entry holds fewer than {MIN_ENTRY_WORDS} words: {entry!r}"
        )
        found = sorted({word.strip(PUNCTUATION).upper() for word in words} & PLACEHOLDERS)
        assert not found, f"{page}: an entry holds the placeholder {found}: {entry!r}"


@pytest.mark.parametrize("page", WITHOUT_COUNTERPART)
def test_a_page_without_a_counterpart_has_no_section(page: str) -> None:
    assert SECTION not in (METHODS / page).read_text(encoding="utf-8")


def test_the_overview_names_the_section_every_page_has() -> None:
    overview = (METHODS / "index.md").read_text(encoding="utf-8")
    assert f'"{SECTION.removeprefix("## ")}"' in overview
