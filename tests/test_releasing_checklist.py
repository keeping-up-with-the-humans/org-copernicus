"""The release checklist, and the distribution name and file list it describes."""

from __future__ import annotations

import re
import sys

import pytest
from ci_support import LIBRARY_ROOT, RELEASING, releasing_section

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

RELEASING_MENTIONS = (
    "src/copernicus/_version.py",
    "CITATION.cff",
    "CHANGELOG.md",
    "README.md",
    "python -m build",
    "copernicus/assets/",
    "copernicus/py.typed",
    "LICENSE",
    "git archive",
    "pyproject.toml",
    "mkdocs.yml",
    "fresh repository",
)


def test_the_releasing_checklist_names_the_version_files_the_build_and_the_export() -> None:
    text = RELEASING.read_text(encoding="utf-8")
    missing = [mention for mention in RELEASING_MENTIONS if mention not in text]
    assert missing == []


@pytest.mark.parametrize("name", ["RELEASING.md", "docs/installation.md"])
def test_the_docs_say_that_building_them_needs_python_3_11(name: str) -> None:
    text = (LIBRARY_ROOT / name).read_text(encoding="utf-8")
    assert "Building the documentation needs Python 3.11 or later" in text
    assert "the library itself runs on 3.10" in text


def test_the_releasing_checklist_says_what_the_fresh_install_check_runs() -> None:
    section = releasing_section("Build and inspect the distributions")
    assert "/tmp/release-check/bin/copernicus --help" in section
    assert "does not run the quick start" in section


def test_the_distributions_are_built_in_the_export_after_the_placeholders() -> None:
    text = RELEASING.read_text(encoding="utf-8")
    titles = re.findall(r"^## \d+\. (.+)$", text, re.MULTILINE)
    assert titles.index("Anonymity export") < titles.index("Build and inspect the distributions")
    build = releasing_section("Build and inspect the distributions")
    assert "in the export, after the placeholders are replaced, never in the research tree" in build
    assert "cd /tmp/copernicus" in build and "tar -tzf dist/org_copernicus-*.tar.gz" in build


def test_the_first_commit_of_the_export_carries_a_neutral_identity_in_utc() -> None:
    export = releasing_section("Anonymity export")
    for part in (
        'git -c user.name="Author names withheld" -c user.email="withheld@example.invalid"',
        "TZ=UTC",
        'GIT_AUTHOR_DATE="$NOW"',
        'GIT_COMMITTER_DATE="$NOW"',
        "must not carry a person's name",
    ):
        assert part in export, part


def test_the_distribution_is_named_org_copernicus_and_the_sdist_list_is_unchanged() -> None:
    parsed = tomllib.loads((LIBRARY_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert parsed["project"]["name"] == "org-copernicus"
    assert parsed["project"]["scripts"] == {"copernicus": "copernicus.cli.app:app"}
    assert parsed["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"] == ["src/copernicus"]
    assert parsed["tool"]["hatch"]["build"]["targets"]["sdist"]["only-include"] == [
        ".github",
        ".gitignore",
        "CHANGELOG.md",
        "CITATION.cff",
        "LICENSE",
        "README.md",
        "RELEASING.md",
        "docs",
        "examples",
        "mkdocs.yml",
        "pyproject.toml",
        "src/copernicus",
        "tests",
    ]


def test_the_index_name_is_checked_before_the_export_and_the_build() -> None:
    export = releasing_section("Anonymity export")
    assert "`org-copernicus`" in export
    assert "https://pypi.org/simple/org-copernicus/" in export
    assert "answers 404 while the name is free" in export
    assert "free on the index, or already owned by the project's publishing account" in export
    assert "The name `copernicus` itself belongs to a different project" in export
    checked = export.index("https://pypi.org/simple/org-copernicus/")
    assert checked < export.index("library_guard")
    assert checked < export.index("git archive --format=tar --prefix=copernicus/")
    text = RELEASING.read_text(encoding="utf-8")
    assert "working name" not in text
    assert "https://pypi.org/simple/org-copernicus/" not in releasing_section("Tag and publish")


def test_a_taken_name_sends_the_maintainer_to_every_place_that_names_it() -> None:
    export = releasing_section("Anonymity export")
    assert 'grep -rnE "org-copernicus|org_copernicus" library' in export
    for place in (
        "`name` in `pyproject.toml`",
        "`src/copernicus/embeddings/encoder.py`",
        "`src/copernicus/llm/vllm_backend.py`",
        "`src/copernicus/embeddings/__init__.py`",
        "the glob names in the tests",
        "the dist-info names in section 5",
        "the CHANGELOG",
    ):
        assert place in export, place


def test_the_dev_extra_declares_tomli_for_python_3_10() -> None:
    parsed = tomllib.loads((LIBRARY_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert 'tomli>=2; python_version < "3.11"' in parsed["project"]["optional-dependencies"]["dev"]


def test_the_guard_run_checks_the_site_built_in_the_documentation_section() -> None:
    export = releasing_section("Anonymity export")
    command = (
        "COPERNICUS_GUARD_SITE=/tmp/site HF_HUB_OFFLINE=1 "
        "library/.venv/bin/python -m pytest -q -p no:asyncio library_guard"
    )
    assert command in export
    assert "-d /tmp/site" in releasing_section("Check the documentation build")
    assert "`COPERNICUS_GUARD_SITE`" in export
    assert "built site when one is in `library/site`" not in export


def test_the_upload_uses_the_project_account() -> None:
    publish = releasing_section("Tag and publish")
    assert "never a personal one" in publish
