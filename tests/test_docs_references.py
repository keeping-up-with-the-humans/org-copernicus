"""The literature of the documentation: every citation has its entry and every entry is cited.

``docs/methods/references.md`` holds the entries that the methods pages cite, copied verbatim from
the reference list of the accompanying methods paper (under review). A citation is
``Author (Year)``, or ``Author Year`` inside parentheses, where ``Author`` is one surname, two
surnames joined by ``and``, or one surname followed by ``et al.``, and the year is written as the
entry writes it (``2023a``). Every citation on every page must name an entry, and every entry must
be cited on a methods page. An entry is one line: the authors as surname and initials, separated
by commas, the year in parentheses, and the rest of the reference, ending with a full stop. The
entries are in alphabetical order, and no two of them are cited the same way.
"""

from __future__ import annotations

import re
from pathlib import Path

LIBRARY_ROOT = Path(__file__).resolve().parents[1]
DOCS = LIBRARY_ROOT / "docs"
METHODS = DOCS / "methods"
REFERENCES = METHODS / "references.md"
YEAR = r"(?:1[89]|20)\d\d[a-z]?"
SURNAME = r"[A-Z][A-Za-z'\-]+"
AUTHORS = rf"{SURNAME}(?: and {SURNAME}| et al\.)?"
NARRATIVE = re.compile(rf"\b({AUTHORS}) \(({YEAR})\)")
BARE = re.compile(rf"\b({AUTHORS}) ({YEAR})\b")
ENTRY = re.compile(rf"^(?P<authors>[^()]+?) \((?P<year>{YEAR})\) \S")
# One author of an entry: a surname of one or more capitalised words, then the initials.
ENTRY_AUTHOR = re.compile(r"[A-Z][\w'\-]*(?: [A-Z][\w'\-]*)* [A-Z]{1,3}")


def entry_key(entry: str) -> tuple[str, str]:
    """Return the citation that names an entry, such as ``("Carroll et al.", "2006")``."""
    match = ENTRY.match(entry)
    assert match is not None, f"not a reference entry: {entry[:80]!r}"
    surnames = [author.rsplit(" ", 1)[0] for author in match["authors"].split(", ")]
    if len(surnames) == 1:
        return surnames[0], match["year"]
    if len(surnames) == 2:
        return f"{surnames[0]} and {surnames[1]}", match["year"]
    return f"{surnames[0]} et al.", match["year"]


def reference_entries() -> list[str]:
    """Return the entries of the reference page: its lines that open with authors and a year."""
    return [
        line for line in REFERENCES.read_text(encoding="utf-8").splitlines() if ENTRY.match(line)
    ]


def malformed_entries(entries: list[str]) -> list[str]:
    """Return the opening of every entry whose authors or ending are not written as required."""
    malformed: list[str] = []
    for entry in entries:
        match = ENTRY.match(entry)
        authors = [] if match is None else match["authors"].split(", ")
        well_formed = bool(authors) and all(ENTRY_AUTHOR.fullmatch(name) for name in authors)
        if not (well_formed and entry.endswith(".")):
            malformed.append(entry[:60])
    return malformed


def citations(text: str) -> list[tuple[str, str]]:
    """Return the citations of a text in both forms, as ``(authors, year)`` in text order."""
    flat = " ".join(text.split())
    found = [(match.start(), match[1], match[2]) for match in NARRATIVE.finditer(flat)]
    found += [(match.start(), match[1], match[2]) for match in BARE.finditer(flat)]
    return [(authors, year) for _, authors, year in sorted(found)]


def page_name(page: Path) -> str:
    """Name a page by its path below the docs folder."""
    return page.relative_to(DOCS).as_posix()


def methods_pages() -> list[Path]:
    """List the methods pages, without the reference page."""
    return [page for page in sorted(METHODS.rglob("*.md")) if page != REFERENCES]


def test_the_citation_reader_and_the_entry_key() -> None:
    text = (
        "as Prentice (1982) showed\n(Carroll et al. 2006, Rogan and Gladen 1978); see "
        "Angelopoulos et al. (2023a), Level 5 and 1,000 texts in 2026."
    )
    assert citations(text) == [
        ("Prentice", "1982"),
        ("Carroll et al.", "2006"),
        ("Rogan and Gladen", "1978"),
        ("Angelopoulos et al.", "2023a"),
    ]
    assert entry_key("Fuller WA (1987) *Measurement Error Models* (Wiley).") == ("Fuller", "1987")
    assert entry_key("Bates JM, Granger CWJ (1969) The combination.") == (
        "Bates and Granger",
        "1969",
    )
    assert entry_key("Wei J, Wang X, Le QV (2022a) Chain.") == ("Wei et al.", "2022a")
    assert entry_key("John OP, Srivastava S (1999) Taxonomy. Pervin LA, John OP, eds. Book.") == (
        "John and Srivastava",
        "1999",
    )


def test_every_citation_names_an_entry_of_the_reference_page() -> None:
    keys = {entry_key(entry) for entry in reference_entries()}
    unknown = [
        f"{page_name(page)}: {authors} ({year})"
        for page in sorted(DOCS.rglob("*.md"))
        if page != REFERENCES
        for authors, year in citations(page.read_text(encoding="utf-8"))
        if (authors, year) not in keys
    ]
    assert unknown == []


def test_every_entry_is_cited_on_a_methods_page() -> None:
    cited = {
        citation
        for page in methods_pages()
        for citation in citations(page.read_text(encoding="utf-8"))
    }
    uncited = [entry[:60] for entry in reference_entries() if entry_key(entry) not in cited]
    assert uncited == []


def test_the_entries_are_well_formed_in_alphabetical_order_and_cited_in_distinct_ways() -> None:
    entries = reference_entries()
    assert entries, "the reference page lists no entry"
    assert malformed_entries(entries) == []
    assert entries == sorted(entries, key=str.casefold)
    keys = [entry_key(entry) for entry in entries]
    assert len(keys) == len(set(keys))


def test_the_entry_check_names_a_malformed_author_and_a_missing_full_stop() -> None:
    entries = [
        "Van Durme B, Snyder-Duch J (2019) A title. *Journal* 1(2):3--4.",
        "Bates, Granger CWJ (1969) The combination of forecasts.",
        "Fuller WA (1987) *Measurement Error Models* (Wiley)",
    ]
    assert malformed_entries(entries) == [entries[1][:60], entries[2][:60]]
