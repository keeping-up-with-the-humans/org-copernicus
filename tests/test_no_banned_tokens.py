"""Hygiene of the package's files: no address, user path, secret, scheduler flag or planning label.

The files read are those that git tracks, or would add, below the package's root folder; when
the folder is not a git checkout (an unpacked source distribution), every file below it except
the generated and vendored folders. Binary files (those holding a NUL byte) are left out; every
other file must be UTF-8. The ruff checks of the source, the tests and the examples run here too.
"""

from __future__ import annotations

import importlib.util
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

LIBRARY_ROOT = Path(__file__).resolve().parents[1]
# Addresses at the domains reserved for examples and documentation name nobody.
EXAMPLE_DOMAINS = ("example.com", "example.net", "example.org")
EXAMPLE_TOP_LEVEL_DOMAINS = ("example", "invalid", "localhost", "test")
EMAIL = re.compile(
    r"[A-Za-z0-9._%+-]+@(?P<domain>[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,})\b"
)
# Keys of the common providers carry at least 32 characters after their prefix.
PATTERNS: dict[str, re.Pattern[str]] = {
    "absolute user path": re.compile(r"(?:/Users|/home|[A-Za-z]:\\Users)[/\\][A-Za-z0-9._-]+"),
    "secret-shaped string": re.compile(
        r"\b(?:sk-[A-Za-z0-9_-]{32,}|gh[opsur]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{20,}"
        r"|hf_[A-Za-z0-9]{30,}|AKIA[0-9A-Z]{16}|AIza[0-9A-Za-z_-]{35}"
        r"|xox[abposr]-[A-Za-z0-9-]{10,})|-----BEGIN [A-Z ]*PRIVATE KEY-----"
    ),
    "scheduler flag": re.compile(
        r"^\s*#[A-Z]{3,6}\s+-|--(?:partition|gres|qos|account|cpus-per-task|ntasks|nodes"
        r"|mem-per-cpu|nodelist|array|time)\b"
    ),
    "planning label": re.compile(r"\bPlan [A-E]\b|\b[Tt]ask[ -][A-Z]?\d+\b"),
}
SKIPPED_DIRECTORIES = {
    ".git",
    "__pycache__",
    "site",
    "dist",
    "build",
    ".venv",
    "venv",
    ".mypy_cache",
    ".ruff_cache",
    ".pytest_cache",
    ".hypothesis",
    "node_modules",
}
SKIPPED_DIRECTORY_SUFFIX = ".egg-info"
RUFF_TARGETS = ("src", "tests", "examples")


def is_git_checkout(root: Path) -> bool:
    """Tell whether git is installed and the folder lies inside a git work tree."""
    if shutil.which("git") is None:
        return False
    answer = subprocess.run(
        ["git", "rev-parse", "--is-inside-work-tree"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    return answer.returncode == 0 and answer.stdout.strip() == "true"


def walked_files(root: Path) -> list[Path]:
    """List every file below a folder, pruning generated and vendored folders."""
    found: list[Path] = []
    for directory, subdirectories, filenames in os.walk(root):
        subdirectories[:] = sorted(
            name
            for name in subdirectories
            if name not in SKIPPED_DIRECTORIES and not name.endswith(SKIPPED_DIRECTORY_SUFFIX)
        )
        found.extend(Path(directory) / name for name in sorted(filenames))
    return found


def package_files(root: Path) -> list[Path]:
    """List the files to read: tracked or addable ones in a checkout, else every file.

    A tracked file that the working tree has deleted is left out, because it has no content.
    """
    if not is_git_checkout(root):
        return walked_files(root)
    listed = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=root,
        capture_output=True,
        check=True,
    ).stdout.decode("utf-8")
    paths = sorted({root / name for name in listed.split("\0") if name})
    return [path for path in paths if path.is_file()]


def is_example_domain(domain: str) -> bool:
    """Tell whether a mail domain is one reserved for examples and documentation."""
    folded = domain.casefold()
    reserved = any(folded == name or folded.endswith(f".{name}") for name in EXAMPLE_DOMAINS)
    return reserved or folded.rsplit(".", 1)[-1] in EXAMPLE_TOP_LEVEL_DOMAINS


def line_hits(line: str) -> list[str]:
    """Return ``<kind>: <text>`` for every address, path, secret, flag or label of a line."""
    hits = [
        f"e-mail address: {match.group(0)}"
        for match in EMAIL.finditer(line)
        if not is_example_domain(match["domain"])
    ]
    for kind, pattern in PATTERNS.items():
        hits += [f"{kind}: {match.group(0)}" for match in pattern.finditer(line)]
    return hits


def hygiene_hits(root: Path, files: list[Path]) -> list[str]:
    """Return one ``<path>:<line>: <kind>: <text>`` entry per hit in the text files."""
    hits: list[str] = []
    for path in files:
        data = path.read_bytes()
        if b"\0" in data:
            continue
        for number, line in enumerate(data.decode("utf-8").splitlines(), start=1):
            location = f"{path.relative_to(root).as_posix()}:{number}"
            hits += [f"{location}: {hit}" for hit in line_hits(line)]
    return hits


def ruff_command() -> list[str] | None:
    """Locate ruff on the path or as a module of the running interpreter."""
    executable = shutil.which("ruff")
    if executable is not None:
        return [executable]
    if importlib.util.find_spec("ruff") is not None:
        return [sys.executable, "-m", "ruff"]
    return None


def test_no_file_of_the_package_holds_an_address_a_path_a_secret_a_flag_or_a_label() -> None:
    files = package_files(LIBRARY_ROOT)
    assert LIBRARY_ROOT / "pyproject.toml" in files
    hits = hygiene_hits(LIBRARY_ROOT, files)
    assert not hits, "files to clean:\n" + "\n".join(hits)


def test_the_scan_reports_each_kind_and_leaves_examples_and_binary_files_alone(
    tmp_path: Path,
) -> None:
    # The samples are assembled here so that this file holds none of them.
    samples = {
        "address.txt": "write to " + "@".join(("someone", "university.edu")),
        "path.md": "data in " + "/".join(("", "Users", "someone", "corpus.csv")),
        "key.py": "KEY = '" + "sk-" + "a1" * 20 + "'",
        "job.sh": "#" + "JOB -q long\nrun --" + "partition=gpu",
        "notes.md": "see " + "Plan " + "C and " + "Task " + "E6",
        "examples.md": "withheld@example.invalid and user:secret@LLM.example.org; Plan F",
    }
    for name, text in samples.items():
        (tmp_path / name).write_text(text + "\n", encoding="utf-8")
    (tmp_path / "data.bin").write_bytes(b"\0" + samples["address.txt"].encode("utf-8"))
    hits = hygiene_hits(tmp_path, sorted(tmp_path.iterdir()))
    assert [hit.split(": ", 2)[:2] for hit in hits] == [
        ["address.txt:1", "e-mail address"],
        ["job.sh:1", "scheduler flag"],
        ["job.sh:2", "scheduler flag"],
        ["key.py:1", "secret-shaped string"],
        ["notes.md:1", "planning label"],
        ["notes.md:1", "planning label"],
        ["path.md:1", "absolute user path"],
    ]


def test_outside_a_checkout_every_file_but_the_generated_folders_is_read(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "ignored.txt").write_text("", encoding="utf-8")
    (tmp_path / "pkg.egg-info").mkdir()
    (tmp_path / "pkg.egg-info" / "PKG-INFO").write_text("", encoding="utf-8")
    assert walked_files(tmp_path) == [tmp_path / "pyproject.toml"]


def test_ruff_check_and_format_are_clean() -> None:
    command = ruff_command()
    if command is None:
        pytest.skip("ruff is not installed")
    for arguments in (["check", *RUFF_TARGETS], ["format", "--check", *RUFF_TARGETS]):
        completed = subprocess.run(
            [*command, *arguments],
            cwd=LIBRARY_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, (
            f"ruff {' '.join(arguments)} failed:\n{completed.stdout}\n{completed.stderr}"
        )
