"""The documentation site: it builds without a warning, and its code blocks compile and run."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import os
import re
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import yaml
from docs_support import (
    FENCE,
    cli_invocation,
    closing_fence,
    help_exit_code,
    line_before,
    shell_commands,
)
from embeddings_support import MINILM_AVAILABLE, SKIP_REASON

import copernicus
from copernicus.cli.app import app

LIBRARY_ROOT = Path(__file__).resolve().parents[1]
DOCS = LIBRARY_ROOT / "docs"
MKDOCS_CONFIG = LIBRARY_ROOT / "mkdocs.yml"
CHANGELOG = LIBRARY_ROOT / "CHANGELOG.md"
RUN_TAG = "<!-- run -->"
EMBED_TAG = "<!-- run embed -->"
OUTPUT_TAG = "<!-- output -->"
PYTHON_FENCE = "```python"
DOCS_PACKAGES = ("mkdocs", "material", "mkdocstrings")
API_SUBPACKAGES = (
    "rubric",
    "data",
    "llm",
    "coders",
    "embeddings",
    "references",
    "similarity",
    "hybrid",
    "validity",
    "baseline",
    "corrections",
    "errors",
)
# Exported names that have no docstring and so no entry on their API page: none.
KNOWN_UNDOCUMENTED: dict[str, set[str]] = {}


@dataclass(frozen=True)
class Block:
    """A fenced python block of a page.

    Attributes:
        line: Line number of the opening fence.
        code: The code, without the indentation of the fence.
        tag: ``"run"`` for a block under ``<!-- run -->``, ``"embed"`` under
            ``<!-- run embed -->``, ``""`` for an untagged block.
        expected: The text of the block that follows an ``<!-- output -->`` comment, or
            ``None`` when the block has no shown output.
    """

    line: int
    code: str
    tag: str
    expected: str | None


def python_blocks(text: str) -> list[Block]:
    """Extract the python blocks of a page, with their tags and shown outputs."""
    lines = text.splitlines()
    blocks: list[Block] = []
    index = 0
    while index < len(lines):
        stripped = lines[index].strip()
        if stripped != PYTHON_FENCE:
            index += 1
            continue
        end = closing_fence(lines, index + 1)
        code = textwrap.dedent("\n".join(lines[index + 1 : end]))
        blocks.append(Block(index + 1, code, _tag_before(lines, index), _output_after(lines, end)))
        index = end + 1
    return blocks


def _tag_before(lines: list[str], fence: int) -> str:
    """Return the run tag on the last non-blank line before the fence."""
    return {RUN_TAG: "run", EMBED_TAG: "embed"}.get(line_before(lines, fence), "")


def _output_after(lines: list[str], end: int) -> str | None:
    """Return the shown output that follows a block, or ``None`` without an output tag."""
    index = end + 1
    while index < len(lines) and not lines[index].strip():
        index += 1
    if index >= len(lines) or lines[index].strip() != OUTPUT_TAG:
        return None
    opening = index + 1
    assert lines[opening].strip().startswith(FENCE), f"line {opening + 1}: no block after the tag"
    closing = closing_fence(lines, opening + 1)
    return "\n".join(lines[opening + 1 : closing])


def doc_pages() -> list[Path]:
    """List the markdown pages below the docs folder."""
    return sorted(DOCS.rglob("*.md"))


def page_id(page: Path) -> str:
    """Name a page by its path below the docs folder."""
    return page.relative_to(DOCS).as_posix()


def pages_with_blocks(tagged_only: bool = False) -> list[Path]:
    """List the pages that hold python blocks (tagged ones only on request)."""
    return [
        page
        for page in doc_pages()
        if any(
            block.tag or not tagged_only
            for block in python_blocks(page.read_text(encoding="utf-8"))
        )
    ]


def nav_files(nav: Any) -> list[str]:
    """Flatten the ``nav`` entry of the site configuration into its file paths."""
    if isinstance(nav, str):
        return [nav]
    if isinstance(nav, dict):
        return [path for value in nav.values() for path in nav_files(value)]
    return [path for item in nav for path in nav_files(item)]


def built_path(site: Path, name: str) -> Path:
    """Return the file the build writes for a page of the nav."""
    stem = name.removesuffix(".md")
    if stem == "index" or stem.endswith("/index"):
        return site / f"{stem}.html"
    return site / stem / "index.html"


def site_configuration() -> dict[str, Any]:
    """Read the site configuration."""
    return yaml.safe_load(MKDOCS_CONFIG.read_text(encoding="utf-8"))


DOCS_PYTHON = (3, 11)
DOCS_SKIP_REASON = (
    "building the documentation needs the 'docs' extra and Python 3.11 or later; "
    "the library itself runs on 3.10"
)


def docs_packages_installed() -> bool:
    """Tell whether the site can be built: its packages import and Python is recent enough."""
    return sys.version_info >= DOCS_PYTHON and all(
        importlib.util.find_spec(name) is not None for name in DOCS_PACKAGES
    )


def skip_message(page: str, blocks: list[Block], first: int) -> str:
    """Name every block of a page that stays unrun from the block at ``first`` on."""
    unrun = [f"{block.line} ({block.tag})" for block in blocks[first:] if block.tag]
    which = (
        f"the tagged block at line {unrun[0]} is"
        if len(unrun) == 1
        else f"the tagged blocks at lines {', '.join(unrun)} are"
    )
    return (
        f"{page}: {which} not run, because the block at line {blocks[first].line} needs the "
        f"encoder: {SKIP_REASON}"
    )


def docs_cli_invocations() -> list[tuple[str, ...]]:
    """List the distinct ``copernicus`` invocations that the shell blocks of the docs show."""
    found = {
        invocation
        for page in doc_pages()
        for _, words in shell_commands(page.read_text(encoding="utf-8"))
        if (invocation := cli_invocation(words)) is not None
    }
    return sorted(found)


def test_nav_files_exist_and_cover_every_page() -> None:
    files = nav_files(site_configuration()["nav"])
    assert len(files) == len(set(files))
    assert [name for name in files if not (DOCS / name).is_file()] == []
    assert sorted(set(files)) == sorted(page_id(page) for page in doc_pages())


@pytest.mark.slow
@pytest.mark.skipif(not docs_packages_installed(), reason=DOCS_SKIP_REASON)
def test_mkdocs_builds_strict(tmp_path: Path) -> None:
    site = tmp_path / "site"
    environment = {**os.environ, "NO_MKDOCS_2_WARNING": "1"}
    result = subprocess.run(
        [sys.executable, "-m", "mkdocs", "build", "--strict", "-d", str(site)],
        cwd=LIBRARY_ROOT,
        capture_output=True,
        text=True,
        check=False,
        env=environment,
        timeout=600,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    assert "WARNING" not in result.stderr + result.stdout
    for name in nav_files(site_configuration()["nav"]):
        assert built_path(site, name).is_file(), name
    for name in API_SUBPACKAGES:
        module = importlib.import_module(f"copernicus.{name}")
        page = (site / "api" / name / "index.html").read_text(encoding="utf-8")
        anchors = set(re.findall(rf'id="copernicus\.{name}\.([A-Za-z_0-9]+)"', page))
        missing = {exported for exported in module.__all__ if exported not in anchors}
        assert missing == KNOWN_UNDOCUMENTED.get(name, set()), name
    changelog = (site / "changelog" / "index.html").read_text(encoding="utf-8")
    assert "0.1.0" in changelog
    assert CHANGELOG.read_text(encoding="utf-8").splitlines()[2] == "## 0.1.0"


def test_api_pages_exist_for_every_subpackage_and_the_index_names_every_top_level_name() -> None:
    files = nav_files(site_configuration()["nav"])
    for name in API_SUBPACKAGES:
        page = DOCS / "api" / f"{name}.md"
        assert f"api/{name}.md" in files
        assert f"::: copernicus.{name}\n" in page.read_text(encoding="utf-8")
    index = (DOCS / "api" / "index.md").read_text(encoding="utf-8")
    missing = [
        name for name in copernicus.__all__ if name != "__version__" and f"`{name}`" not in index
    ]
    assert missing == []


@pytest.mark.parametrize("page", pages_with_blocks(), ids=page_id)
def test_tutorial_blocks_compile(page: Path) -> None:
    for block in python_blocks(page.read_text(encoding="utf-8")):
        compile(block.code, f"{page_id(page)}:{block.line}", "exec")


@pytest.mark.parametrize("page", pages_with_blocks(tagged_only=True), ids=page_id)
def test_tagged_blocks_run(page: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    namespace: dict[str, Any] = {"__name__": "__docs__"}
    blocks = python_blocks(page.read_text(encoding="utf-8"))
    for position, block in enumerate(blocks):
        if not block.tag:
            continue
        if block.tag == "embed" and not MINILM_AVAILABLE:
            pytest.skip(skip_message(page_id(page), blocks, position))
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            exec(compile(block.code, f"{page_id(page)}:{block.line}", "exec"), namespace)
        if block.expected is not None:
            shown = [" ".join(line.split()) for line in block.expected.strip().splitlines()]
            printed = [" ".join(line.split()) for line in captured.getvalue().strip().splitlines()]
            assert printed == shown, f"{page_id(page)}:{block.line}: the shown output is stale"


def test_the_skip_message_lists_every_block_that_stays_unrun() -> None:
    blocks = [
        Block(10, "a = 1", "run", None),
        Block(20, "b = 2", "embed", None),
        Block(30, "c = 3", "", None),
        Block(40, "d = 4", "run", None),
        Block(50, "e = 5", "embed", None),
    ]
    message = skip_message("page.md", blocks, 1)
    assert "the tagged blocks at lines 20 (embed), 40 (run), 50 (embed) are not run" in message
    assert "10 (" not in message
    assert SKIP_REASON in message
    single = skip_message("page.md", blocks, 4)
    assert single.startswith("page.md: the tagged block at line 50 (embed) is not run")


def test_the_skip_reason_names_torch_exactly_when_torch_is_missing() -> None:
    torch_missing = importlib.util.find_spec("torch") is None
    assert ("torch" in SKIP_REASON) == torch_missing
    if torch_missing:
        assert not MINILM_AVAILABLE


def test_the_block_parser_reads_tags_indentation_and_shown_output() -> None:
    page = "\n".join(
        [
            "text",
            "<!-- run -->",
            "```python",
            "print(1)",
            "```",
            "",
            "<!-- output -->",
            "```text",
            "1",
            "```",
            "<!-- run embed -->",
            "",
            "```python",
            "x = 2",
            "```",
            "```python",
            "y = 3",
            "```",
            "    ```python",
            "    z = 4",
            "    ```",
            "```bash",
            "echo no",
            "```",
        ]
    )
    blocks = python_blocks(page)
    assert [(block.tag, block.code, block.expected) for block in blocks] == [
        ("run", "print(1)", "1"),
        ("embed", "x = 2", None),
        ("", "y = 3", None),
        ("", "z = 4", None),
    ]
    assert [block.line for block in blocks] == [3, 13, 16, 19]


@pytest.mark.parametrize(
    "invocation", docs_cli_invocations(), ids=lambda words: " ".join(("copernicus", *words))
)
def test_every_command_shown_in_the_docs_answers_help(invocation: tuple[str, ...]) -> None:
    assert help_exit_code(invocation) == 0, " ".join(("copernicus", *invocation, "--help"))


def test_the_docs_show_commands_of_every_group() -> None:
    groups = {invocation[0] for invocation in docs_cli_invocations() if invocation}
    assert groups == {group.name for group in app.registered_groups}


def test_the_help_check_fails_for_a_command_that_does_not_exist() -> None:
    assert help_exit_code(("store", "exprot")) != 0
    assert help_exit_code(("stores",)) != 0


def test_the_shell_parser_reads_continuations_separators_quotes_and_the_no_run_tag() -> None:
    page = "\n".join(
        [
            "```bash",
            "pip install .",
            'python -c "import copernicus; print(copernicus.__version__)"',
            "copernicus --help",
            "copernicus store export run.sqlite \\",
            "    --what text --out scores.csv && copernicus corrections list",
            "```",
            "<!-- no-run -->",
            "",
            "```bash",
            "copernicus <group> <command> --help",
            "```",
            "```python",
            "copernicus = 1",
            "```",
            "```sh",
            "echo done | copernicus rubric validate rubric.json",
            "```",
        ]
    )
    commands = shell_commands(page)
    assert [line for line, _ in commands] == [2, 3, 4, 5, 5, 17, 17]
    assert commands[3][1] == (
        *("copernicus", "store", "export", "run.sqlite"),
        *("--what", "text", "--out", "scores.csv"),
    )
    assert [cli_invocation(words) for _, words in commands] == [
        None,
        None,
        (),
        ("store", "export"),
        ("corrections", "list"),
        None,
        ("rubric", "validate"),
    ]
