"""Readers of the fenced blocks of the documentation pages, shared by the docs tests.

The python blocks are read in ``test_docs_build``; this module holds the fence helpers and the
reader of the shell blocks, whose ``copernicus`` commands the docs tests check against the app.
"""

from __future__ import annotations

import itertools
import shlex

from typer.testing import CliRunner

from copernicus.cli.app import app

FENCE = "```"
NO_RUN_TAG = "<!-- no-run -->"
SHELL_FENCES = ("```bash", "```sh", "```shell")
COMMAND_SEPARATORS = frozenset({"&&", "||", ";", "|"})


def closing_fence(lines: list[str], start: int) -> int:
    """Return the index of the line that closes the fence whose content begins at ``start``."""
    for index in range(start, len(lines)):
        if lines[index].strip() == FENCE:
            return index
    raise AssertionError(f"a code fence opened before line {start} is never closed")


def line_before(lines: list[str], fence: int) -> str:
    """Return the last non-blank line before the fence, stripped (``""`` at the top)."""
    index = fence - 1
    while index >= 0 and not lines[index].strip():
        index -= 1
    return lines[index].strip() if index >= 0 else ""


def shell_commands(text: str) -> list[tuple[int, tuple[str, ...]]]:
    """Extract the commands of the shell blocks of a page, leaving out blocks tagged no-run.

    A line that ends in a backslash continues on the next line; ``&&``, ``||``, ``;`` and ``|``
    separate the commands of a line, and quoted words stay whole.

    Returns:
        One ``(line, words)`` pair per command, with the number of the line it starts on.
    """
    lines = text.splitlines()
    commands: list[tuple[int, tuple[str, ...]]] = []
    index = 0
    while index < len(lines):
        if lines[index].strip() not in SHELL_FENCES:
            index += 1
            continue
        end = closing_fence(lines, index + 1)
        if line_before(lines, index) != NO_RUN_TAG:
            commands.extend(_block_commands(lines, index + 1, end))
        index = end + 1
    return commands


def _block_commands(lines: list[str], start: int, end: int) -> list[tuple[int, tuple[str, ...]]]:
    """Split the lines from ``start`` up to ``end`` of a shell block into commands."""
    commands: list[tuple[int, tuple[str, ...]]] = []
    index = start
    while index < end:
        first = index
        logical = lines[index].rstrip()
        while logical.endswith("\\") and index + 1 < end:
            index += 1
            logical = logical[:-1] + " " + lines[index].strip()
        index += 1
        lexer = shlex.shlex(logical, posix=True, punctuation_chars=True)
        lexer.whitespace_split = True
        for separator, words in itertools.groupby(lexer, key=COMMAND_SEPARATORS.__contains__):
            if not separator:
                commands.append((first + 1, tuple(words)))
    return commands


def cli_invocation(words: tuple[str, ...]) -> tuple[str, ...] | None:
    """Return the group and command that a ``copernicus`` command calls.

    The group and the command are the first two words after ``copernicus``; an option ends them
    early, so ``copernicus --help`` gives ``()``.

    Returns:
        The words, or ``None`` when the command runs another program.
    """
    if not words or words[0] != "copernicus":
        return None
    invocation: list[str] = []
    for word in words[1:3]:
        if word.startswith("-"):
            break
        invocation.append(word)
    return tuple(invocation)


def help_exit_code(invocation: tuple[str, ...]) -> int:
    """Run ``copernicus <invocation> --help`` in process and return its exit code."""
    return CliRunner().invoke(app, [*invocation, "--help"]).exit_code
