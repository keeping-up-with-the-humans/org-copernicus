"""Shared constants and readers of the CI tests: the workflows, the project file, the checklist."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

LIBRARY_ROOT = Path(__file__).resolve().parents[1]
TESTS = LIBRARY_ROOT / "tests"
WORKFLOW = LIBRARY_ROOT / ".github" / "workflows" / "ci.yml"
DOCS_WORKFLOW = LIBRARY_ROOT / ".github" / "workflows" / "docs.yml"
RELEASING = LIBRARY_ROOT / "RELEASING.md"
EMBED_MARKER = "embed"


def workflow() -> dict[str, Any]:
    """Parse the workflow file."""
    parsed = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    assert isinstance(parsed, dict)
    return parsed


def docs_workflow() -> dict[str, Any]:
    """Parse the workflow file that deploys the documentation site."""
    parsed = yaml.safe_load(DOCS_WORKFLOW.read_text(encoding="utf-8"))
    assert isinstance(parsed, dict)
    return parsed


def pyproject_text() -> str:
    """Read the project file."""
    return (LIBRARY_ROOT / "pyproject.toml").read_text(encoding="utf-8")


def releasing_section(title: str) -> str:
    """Return the text of the section of the release checklist with the given title."""
    text = RELEASING.read_text(encoding="utf-8")
    found = re.search(rf"^## \d+\. {re.escape(title)}\n(.*?)(?=^## |\Z)", text, re.S | re.M)
    assert found is not None, f"the release checklist has no section {title!r}"
    return found.group(1)
