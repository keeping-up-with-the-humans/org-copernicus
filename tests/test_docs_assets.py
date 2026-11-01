"""The README, the citation metadata, the export step of the release checklist, and the wheel."""

from __future__ import annotations

import importlib.util
import re
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest
import yaml

import copernicus
from copernicus.cli.app import app

LIBRARY_ROOT = Path(__file__).resolve().parents[1]
README = LIBRARY_ROOT / "README.md"
REPOSITORY_URL = "https://github.com/keeping-up-with-the-humans/org-copernicus"
DOCUMENTATION_URL = "https://keeping-up-with-the-humans.github.io/org-copernicus/"
QUICK_START_LINES = 12
SECTIONS = (
    "What the library does",
    "Installation",
    "Quick start",
    "Module map",
    "Command line",
    "Documentation",
    "How to cite",
    "License",
)
REQUIRED_IN_WHEEL = (
    "copernicus/py.typed",
    "copernicus/assets/examples/toy_corpus.csv",
    "copernicus/assets/examples/toy_cues.json",
    "copernicus/assets/rubrics/constructive_feedback.json",
    "copernicus/rubric/schema.json",
    "copernicus/coders/prompts/extraction.jinja2",
    "copernicus/coders/prompts/scoring.jinja2",
    "copernicus/coders/prompts/one_stage.jinja2",
)
BUILD_TOOLS_MISSING = (
    importlib.util.find_spec("build") is None or importlib.util.find_spec("hatchling") is None
)
BUILD_TOOLS_REASON = "the build and hatchling packages are not installed"
# Generated and vendored folders that a copy of the package's root folder leaves out.
NOT_COPIED = (".venv", "venv", "__pycache__", ".mypy_cache", ".ruff_cache", ".pytest_cache")
NOT_COPIED_BUILDS = ("site", "dist", "build", "*.egg-info", ".git", ".hypothesis")
STRAY_FILES = ("scores.csv", "notes.txt", "outputs/summary.json", "docs_draft.md")
# An install by the wrong package-index name: the distribution is org-copernicus.
INDEX_INSTALL = re.compile(r"""pip install +["']?copernicus(?:\[[\w,]+\])?["']?(?![\w./-])""")


def package_data_files() -> set[str]:
    """Return the non-python files of the source package, as wheel paths."""
    source = LIBRARY_ROOT / "src"
    return {
        path.relative_to(source).as_posix()
        for path in (source / "copernicus").rglob("*")
        if path.is_file() and path.suffix not in {".py", ".pyc"} and "__pycache__" not in path.parts
    }


def quick_start_code() -> str:
    """Return the python block that follows the ``## Quick start`` heading of the README."""
    text = README.read_text(encoding="utf-8")
    section = re.search(r"^## Quick start\n(.*?)(?=^## |\Z)", text, re.DOTALL | re.MULTILINE)
    assert section is not None, "the README has no 'Quick start' section"
    blocks = re.findall(r"^```python\n(.*?)^```", section.group(1), re.DOTALL | re.MULTILINE)
    assert len(blocks) == 1, f"the Quick start holds {len(blocks)} python blocks, not one"
    return blocks[0]


def test_readme_quick_start_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    exec(compile(quick_start_code(), "README.md", "exec"), {"__name__": "__quick_start__"})
    output = capsys.readouterr().out
    assert "rank validity" in output
    assert (tmp_path / "run.sqlite").exists()


def test_readme_quick_start_is_at_most_twelve_lines() -> None:
    lines = quick_start_code().strip("\n").splitlines()
    assert len(lines) <= QUICK_START_LINES


def test_readme_has_the_sections_in_order() -> None:
    text = README.read_text(encoding="utf-8")
    headings = re.findall(r"^## (.+)$", text, re.MULTILINE)
    positions = [headings.index(section) for section in SECTIONS]
    assert positions == sorted(positions)
    assert text.startswith("# copernicus\n")


def test_readme_cites_the_paper_as_under_review() -> None:
    text = README.read_text(encoding="utf-8")
    assert "the accompanying methods paper (under review)" in text
    assert "Author names withheld" in text
    assert f"<{REPOSITORY_URL}>" in text and f"<{DOCUMENTATION_URL}>" in text


def test_citation_file_is_valid_and_withholds_the_authors() -> None:
    cff = yaml.safe_load((LIBRARY_ROOT / "CITATION.cff").read_text(encoding="utf-8"))
    assert cff["cff-version"] == "1.2.0"
    assert cff["type"] == "software"
    assert cff["version"] == "0.1.0"
    assert cff["license"] == "MIT"
    assert cff["authors"] == [{"name": "Author names withheld"}]
    assert cff["repository-code"] == REPOSITORY_URL
    assert cff["url"] == DOCUMENTATION_URL
    for field in ("title", "message"):
        assert isinstance(cff[field], str) and cff[field]
    preferred = cff["preferred-citation"]
    assert preferred["type"] == "article"
    assert preferred["authors"] == [{"name": "Author names withheld"}]
    assert preferred["notes"] == "under review"
    assert "journal" not in preferred and "publisher" not in preferred


def test_the_release_checklist_runs_the_guard_before_the_export_and_counts_the_archive() -> None:
    text = (LIBRARY_ROOT / "RELEASING.md").read_text(encoding="utf-8")
    section = re.search(r"^## \d+\. Anonymity export\n(.*?)(?=^## |\Z)", text, re.S | re.M)
    assert section is not None, "the release checklist has no anonymity export section"
    steps = section.group(1)
    guard = steps.index("-p no:asyncio library_guard")
    export = steps.index("git archive --format=tar --prefix=copernicus/ HEAD:library")
    assert guard < export
    assert "git archive HEAD:library | tar -t | grep -c guard" in steps
    assert "The count must be 0." in steps
    assert "no file named in its lists" in steps


def test_pyproject_carries_the_project_urls_of_the_export() -> None:
    text = (LIBRARY_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "[project.urls]" in text
    assert f'Documentation = "{DOCUMENTATION_URL}"' in text
    assert f'Source = "{REPOSITORY_URL}"' in text
    releasing = (LIBRARY_ROOT / "RELEASING.md").read_text(encoding="utf-8")
    assert "add `[project.urls]`" in releasing


def test_the_description_covers_coding_validity_the_baseline_and_corrections() -> None:
    text = (LIBRARY_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^description = "(.+)"$', text, re.MULTILINE)
    assert match is not None
    for part in ("coders", "validity panel", "NLP-ML baseline", "corrections"):
        assert part in match.group(1), part


def test_changelog_lists_the_release_and_keeps_the_provenance_line() -> None:
    text = (LIBRARY_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert "## 0.1.0" in text
    assert "53d269895adae41edfce7f4df9c9a201f23eff00d6029a3eb5298e8c38a734df" in text


@pytest.mark.skipif(BUILD_TOOLS_MISSING, reason=BUILD_TOOLS_REASON)
def test_built_wheel_holds_the_marker_the_assets_and_the_console_script(tmp_path: Path) -> None:
    command = [
        sys.executable,
        "-m",
        "build",
        "--wheel",
        "--no-isolation",
        "--outdir",
        str(tmp_path),
    ]
    result = subprocess.run(
        command, cwd=LIBRARY_ROOT, capture_output=True, text=True, check=False, timeout=300
    )
    assert result.returncode == 0, result.stderr + result.stdout
    (wheel,) = tmp_path.glob("org_copernicus-*.whl")
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        entry_points = next(name for name in names if name.endswith("entry_points.txt"))
        scripts = archive.read(entry_points).decode("utf-8")
        metadata = next(name for name in names if name.endswith("/METADATA"))
        description = archive.read(metadata).decode("utf-8")
    assert set(REQUIRED_IN_WHEEL) <= package_data_files()
    assert package_data_files() <= set(names)
    assert "copernicus = copernicus.cli.app:app" in scripts
    assert "License-Expression: MIT" in description
    assert "License-File: LICENSE" in description
    assert f"Project-URL: Documentation, {DOCUMENTATION_URL}" in description
    assert f"Project-URL: Source, {REPOSITORY_URL}" in description
    assert "copernicus/py.typed" in names
    assert not any(name.startswith(("tests/", "docs/")) for name in names)


def repository_files() -> tuple[set[str], set[str]] | None:
    """Return the files git tracks below the root folder, and the untracked ones it would add.

    Returns:
        The tracked files and the untracked files that no ignore rule covers, or ``None``
        outside a git checkout; a tracked file that the working tree has deleted is left out.
    """
    if shutil.which("git") is None:
        return None
    found: list[set[str]] = []
    for options in (("--cached",), ("--others", "--exclude-standard")):
        listed = subprocess.run(
            ["git", "ls-files", "-z", *options], cwd=LIBRARY_ROOT, capture_output=True, check=False
        )
        if listed.returncode != 0:
            return None
        names = {name for name in listed.stdout.decode("utf-8").split("\0") if name}
        found.append({name for name in names if (LIBRARY_ROOT / name).is_file()})
    return found[0], found[1]


@pytest.mark.slow
@pytest.mark.skipif(BUILD_TOOLS_MISSING, reason=BUILD_TOOLS_REASON)
def test_a_stray_file_in_the_root_folder_stays_out_of_the_source_distribution(
    tmp_path: Path,
) -> None:
    copy = tmp_path / "package"
    shutil.copytree(
        LIBRARY_ROOT, copy, ignore=shutil.ignore_patterns(*NOT_COPIED, *NOT_COPIED_BUILDS)
    )
    for name in STRAY_FILES:
        (copy / name).parent.mkdir(parents=True, exist_ok=True)
        (copy / name).write_text("left behind by a run\n", encoding="utf-8")
    command = [sys.executable, "-m", "build", "--sdist", "--no-isolation", "--outdir", "dist"]
    result = subprocess.run(command, cwd=copy, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr + result.stdout
    (sdist,) = (copy / "dist").glob("org_copernicus-*.tar.gz")
    with tarfile.open(sdist) as archive:
        members = {name.split("/", 1)[1] for name in archive.getnames() if "/" in name}
    shipped = {name for name in members if not (copy / name).is_dir()}
    assert not shipped & set(STRAY_FILES)
    assert {"pyproject.toml", "README.md", "LICENSE", "src/copernicus/__init__.py"} <= shipped
    files = repository_files()
    if files is not None:
        tracked, addable = files
        assert tracked <= shipped
        assert shipped - {"PKG-INFO"} <= tracked | addable


def test_versions_of_the_citation_file_and_the_readme_follow_the_package() -> None:
    cff = yaml.safe_load((LIBRARY_ROOT / "CITATION.cff").read_text(encoding="utf-8"))
    assert cff["version"] == copernicus.__version__
    for page in (README, LIBRARY_ROOT / "docs" / "citing.md"):
        versions = re.findall(r"Version (\d+\.\d+\.\d+)", page.read_text(encoding="utf-8"))
        assert versions == [copernicus.__version__], page.name


def test_no_page_installs_the_package_by_its_index_name() -> None:
    pages = [
        README,
        LIBRARY_ROOT / "examples" / "README.md",
        *(LIBRARY_ROOT / "docs").rglob("*.md"),
    ]
    hits = [
        f"{page.relative_to(LIBRARY_ROOT)}: {match.group(0)}"
        for page in sorted(pages)
        for match in INDEX_INSTALL.finditer(page.read_text(encoding="utf-8"))
    ]
    assert hits == []
    assert INDEX_INSTALL.search('pip install "copernicus[embed]" and') is not None
    assert INDEX_INSTALL.search("run pip install copernicus") is not None
    assert INDEX_INSTALL.search('pip install ".[embed]"') is None
    assert INDEX_INSTALL.search("pip install copernicus-0.1.0.whl") is None
    assert INDEX_INSTALL.search("pip install org-copernicus") is None
    assert INDEX_INSTALL.search('pip install "org-copernicus[embed]"') is None
    for page in (README, LIBRARY_ROOT / "docs" / "installation.md"):
        text = page.read_text(encoding="utf-8")
        assert "pip install org-copernicus" in text, page.name
        assert "once the first release is published" in text, page.name
        assert "settled at release" not in text, page.name


def test_readme_command_groups_are_the_groups_of_the_app() -> None:
    text = README.read_text(encoding="utf-8")
    section = re.search(r"^## Command line\n(.*?)(?=^## |\Z)", text, re.DOTALL | re.MULTILINE)
    assert section is not None
    listed = re.findall(r"^- `copernicus (\w+)`:", section.group(1), re.MULTILINE)
    registered = [group.name for group in app.registered_groups]
    assert listed == registered
