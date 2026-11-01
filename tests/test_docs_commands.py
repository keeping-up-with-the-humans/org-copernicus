"""The commands the documentation shows keep the rules the command line applies to their options.

The help check of ``test_docs_build`` runs every documented command with ``--help`` only, so the
rules between options are checked here, on the words of each documented command.
"""

from __future__ import annotations

from pathlib import Path

from docs_support import shell_commands

DOCS = Path(__file__).resolve().parents[1] / "docs"
GENERATE = ("copernicus", "references", "generate")


def option_values(words: tuple[str, ...], option: str) -> list[str]:
    """Return the values given to an option in one command."""
    return [words[index + 1] for index, word in enumerate(words[:-1]) if word == option]


def test_a_documented_generation_names_a_resume_file_other_than_its_output() -> None:
    """The command refuses ``--resume`` naming the ``--out`` file, so no page may show it."""
    shown = []
    for page in sorted(DOCS.rglob("*.md")):
        for line, words in shell_commands(page.read_text(encoding="utf-8")):
            if words[: len(GENERATE)] != GENERATE:
                continue
            shown.append(words)
            clash = set(option_values(words, "--resume")) & set(option_values(words, "--out"))
            assert not clash, f"{page.relative_to(DOCS)}:{line} resumes into its output {clash}"
    assert shown
