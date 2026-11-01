"""Shared helpers for the command-line tests."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from typer.core import TyperGroup

from copernicus.cli.app import app
from copernicus.rubric.loader import toy_rubric_path


def code_args(method: str, texts: Path, out: Path, *extra: str, backend: str = "fake") -> list[str]:
    """Build the arguments of a ``code`` invocation on the toy rubric."""
    return [
        "code",
        method,
        "--texts",
        str(texts),
        "--rubric",
        str(toy_rubric_path()),
        "--out",
        str(out),
        "--backend",
        backend,
        "--group-column",
        "group_id",
        *extra,
    ]


def summary_value(output: str, label: str) -> str:
    """Return the value printed after a label in the summary table."""
    for line in output.splitlines():
        if line.startswith(label) and line[len(label) :].startswith(" "):
            return line[len(label) :].strip()
    raise AssertionError(f"no summary row {label!r} in:\n{output}")


def with_backend(args: list[str], backend: str) -> list[str]:
    """Return the arguments with the value of ``--backend`` replaced."""
    replaced = list(args)
    replaced[replaced.index("--backend") + 1] = backend
    return replaced


def minimal_rubric_json() -> str:
    """Return a valid rubric with one attribute and one marker."""
    level = {"label": "Level", "description": "A description of this level of the marker."}
    marker = {
        "marker_id": "m_examples",
        "name": "Concrete examples",
        "description": "The feedback illustrates its points with examples from the work.",
        "sub_rubric": {str(n): level for n in range(1, 6)},
    }
    attribute = {
        "attribute_id": "specificity",
        "name": "Specificity",
        "description": "How precisely the feedback points to the work.",
        "literature_sources": ["Source one"],
        "markers": [marker],
    }
    rubric = {
        "rubric_id": "tiny_v1",
        "concept": "Tiny concept",
        "concept_definition": "A concept that is only used in the tests of the command line.",
        "literature_sources": ["Source one"],
        "attributes": [attribute],
        "definitions": {"term": "A glossary entry."},
    }
    return json.dumps(rubric)


def scripted_run(tmp_path: Path, answers: list[str], *extra: str) -> tuple[list[str], Path]:
    """Build a two-stage run of one text on the minimal rubric with scripted fake answers.

    Returns:
        The command-line arguments and the path of the store they write.
    """
    rubric = tmp_path / "rubric.json"
    rubric.write_text(minimal_rubric_json(), encoding="utf-8")
    texts = tmp_path / "one.csv"
    texts.write_text(
        "text_id,text,group_id\nt1,The example in section two shows it.,g1\n", encoding="utf-8"
    )
    script = tmp_path / "script.json"
    script.write_text(json.dumps(answers), encoding="utf-8")
    store = tmp_path / "scripted.sqlite"
    args = [
        *("code", "two-stage", "--texts", str(texts), "--rubric", str(rubric)),
        *("--out", str(store), "--backend", "fake", "--fake-script", str(script)),
        *("--runs", "1", *extra),
    ]
    return args, store


def option_defaults(group: str, command: str) -> dict[str, object]:
    """Return the defaults of the options of a command, by parameter name."""
    root = typer.main.get_command(app)
    assert isinstance(root, TyperGroup)
    group_command = root.commands[group]
    assert isinstance(group_command, TyperGroup)
    return {param.name: param.default for param in group_command.commands[command].params}
