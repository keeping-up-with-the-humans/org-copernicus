"""The CI workflow: its jobs, triggers and extras, and the pytest options and markers."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from typing import Any

import pytest
from ci_support import (
    DOCS_WORKFLOW,
    EMBED_MARKER,
    LIBRARY_ROOT,
    TESTS,
    WORKFLOW,
    docs_workflow,
    pyproject_text,
    workflow,
)

DEFAULT_SELECTION = "not embed and not slow"
SLOW_SELECTION = "slow and not embed"
LINT_COMMANDS = (
    "ruff check src tests examples",
    "ruff format --check src tests examples",
    "mypy src",
)
DOCS_MARKER = '; python_version >= "3.11"'
PAGES_ACTIONS = (
    "actions/configure-pages",
    "actions/upload-pages-artifact",
    "actions/deploy-pages",
)


def triggers(parsed: dict[str, Any]) -> dict[str, Any] | list[str] | str:
    """Return the ``on`` entry, which YAML 1.1 reads as the boolean key ``True``."""
    return parsed["on"] if "on" in parsed else parsed[True]


def run_commands(job: dict[str, Any]) -> list[str]:
    """List the shell commands of a job, one per line of each ``run`` step."""
    return [
        line.strip()
        for step in job["steps"]
        if "run" in step
        for line in step["run"].splitlines()
        if line.strip()
    ]


def uses(job: dict[str, Any]) -> list[str]:
    """List the actions that the steps of a job use."""
    return [step["uses"] for step in job["steps"] if "uses" in step]


def pytest_commands(job: dict[str, Any]) -> list[str]:
    """List the pytest commands of a job."""
    return [command for command in run_commands(job) if command.startswith("pytest")]


def test_the_workflow_runs_on_push_and_pull_request() -> None:
    assert {"push", "pull_request"} <= set(triggers(workflow()))


def test_the_workflow_has_a_default_job_and_a_slow_job() -> None:
    assert set(workflow()["jobs"]) == {"test", "slow"}


def test_the_default_job_runs_both_python_versions_on_ubuntu() -> None:
    job = workflow()["jobs"]["test"]
    assert job["runs-on"] == "ubuntu-latest"
    assert job["strategy"]["matrix"]["python-version"] == ["3.10", "3.12"]
    assert job["strategy"]["fail-fast"] is False


def test_the_extras_differ_by_version_because_the_docs_tools_need_python_3_11() -> None:
    matrix = workflow()["jobs"]["test"]["strategy"]["matrix"]
    extras = {entry["python-version"]: entry["extras"] for entry in matrix["include"]}
    assert extras == {"3.10": "dev", "3.12": "dev,docs"}
    installs = [c for c in run_commands(workflow()["jobs"]["test"]) if "pip install" in c]
    assert installs == ['pip install -e ".[${{ matrix.extras }}]"']


def test_the_documentation_build_step_runs_only_on_python_3_12() -> None:
    steps = workflow()["jobs"]["test"]["steps"]
    build = [step for step in steps if step.get("run") == "mkdocs build --strict"]
    assert len(build) == 1
    assert build[0]["if"] == "matrix.python-version == '3.12'"
    for step in steps:
        if "mkdocs" in step.get("run", "") and step is not build[0]:
            raise AssertionError(f"another step runs mkdocs: {step}")


def test_the_workflow_reads_only_and_cancels_superseded_runs_of_the_same_branch() -> None:
    parsed = workflow()
    assert parsed["permissions"] == {"contents": "read"}
    concurrency = parsed["concurrency"]
    assert concurrency["cancel-in-progress"] is True
    group = concurrency["group"]
    assert "github.workflow" in group and "github.event_name" in group
    assert "github.event.pull_request.head.repo.full_name || github.repository" in group
    assert "github.head_ref" in group and "github.ref_name" in group


def ini_option_keys() -> list[str]:
    """List the keys of the ``[tool.pytest.ini_options]`` table of the project file."""
    table = re.search(
        r"^\[tool\.pytest\.ini_options\]\n(.*?)(?=^\[|\Z)", pyproject_text(), re.S | re.M
    )
    assert table is not None
    return re.findall(r"^(\w+) = ", table.group(1), re.MULTILINE)


def builtin_ini_keys() -> set[str]:
    """List the configuration keys that pytest itself defines, without any plugin."""
    environment = {**os.environ, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"}
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--help"],
        capture_output=True,
        text=True,
        check=True,
        env=environment,
        cwd=LIBRARY_ROOT,
    )
    return set(re.findall(r"^  (\w+) \(", result.stdout, re.MULTILINE))


def test_every_pytest_ini_option_is_one_that_pytest_itself_defines() -> None:
    keys = ini_option_keys()
    assert {"testpaths", "pythonpath", "markers"} <= set(keys)
    assert [key for key in keys if key not in builtin_ini_keys()] == []


def test_the_openai_floor_matches_the_sdk_that_the_stubs_construct_errors_for() -> None:
    match = re.search(r'"openai>=(\d+)', pyproject_text())
    assert match is not None and int(match.group(1)) >= 3
    stubs = (TESTS / "llm" / "openai_support.py").read_text(encoding="utf-8")
    assert "import httpx2" in stubs
    assert not re.search(r"^import httpx$", stubs, re.MULTILINE)


def test_the_docs_extra_names_the_floors_that_work_together_on_python_3_11_and_later() -> None:
    match = re.search(r"^docs = \[(.*)\]$", pyproject_text(), re.MULTILINE)
    assert match is not None
    entries = re.findall(r"'([^']+)'", match.group(1))
    assert entries == [
        "mkdocs-material>=9.8" + DOCS_MARKER,
        "mkdocstrings[python]>=1.0.8" + DOCS_MARKER,
    ]


def test_the_slow_job_runs_python_3_12_on_ubuntu() -> None:
    job = workflow()["jobs"]["slow"]
    assert job["runs-on"] == "ubuntu-latest"
    assert "strategy" not in job
    setup = next(step for step in job["steps"] if step.get("uses", "").startswith("actions/setup"))
    assert setup["with"]["python-version"] == "3.12"


def test_the_default_job_runs_the_checks_the_default_tests_and_the_strict_docs_build() -> None:
    commands = run_commands(workflow()["jobs"]["test"])
    for lint in LINT_COMMANDS:
        assert lint in commands
    assert pytest_commands(workflow()["jobs"]["test"]) == [f'pytest -q -m "{DEFAULT_SELECTION}"']
    assert commands.count("mkdocs build --strict") == 1


def test_the_slow_job_runs_only_the_slow_tests_that_need_no_encoder() -> None:
    job = workflow()["jobs"]["slow"]
    assert pytest_commands(job) == [f'pytest -q -m "{SLOW_SELECTION}"']
    assert not set(LINT_COMMANDS) & set(run_commands(job))


@pytest.mark.parametrize("name", ["test", "slow"])
def test_each_job_checks_out_sets_up_python_and_installs_the_package(name: str) -> None:
    job = workflow()["jobs"][name]
    assert uses(job)[:2] == ["actions/checkout@v4", "actions/setup-python@v5"]
    installs = [command for command in run_commands(job) if "pip install" in command]
    expected = {
        "test": 'pip install -e ".[${{ matrix.extras }}]"',
        "slow": 'pip install -e ".[dev,docs]"',
    }
    assert installs == [expected[name]]
    assert all(re.fullmatch(r"[\w.-]+/[\w.-]+@v\d+", action) for action in uses(job))


def test_the_workflow_sets_the_offline_and_docs_environment_without_secrets() -> None:
    parsed = workflow()
    assert parsed["env"] == {"NO_MKDOCS_2_WARNING": "1", "HF_HUB_OFFLINE": "1"}
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "secrets." not in text and "${{ secrets" not in text
    assert "embed" not in " ".join(
        command for job in parsed["jobs"].values() for command in run_commands(job)
    ).replace(DEFAULT_SELECTION, "").replace(SLOW_SELECTION, "")


def test_the_default_selection_is_the_one_that_skips_the_pages_with_encoder_blocks() -> None:
    commands = pytest_commands(workflow()["jobs"]["test"])
    assert all(f'-m "{DEFAULT_SELECTION}"' in command for command in commands)


def test_both_markers_are_registered_and_the_slow_one_is_not_described_as_default() -> None:
    text = (LIBRARY_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    block = re.search(r"^markers = \[\n(.*?)^\]", text, re.DOTALL | re.MULTILINE)
    assert block is not None, "pyproject.toml registers no markers"
    markers = dict(re.findall(r'^\s+"(\w+): (.*)",$', block.group(1), re.MULTILINE))
    assert set(markers) == {"slow", EMBED_MARKER}
    assert "the default suite runs it" not in markers["slow"]
    assert "slow job" in markers["slow"]


def test_the_help_tests_pass_where_the_runner_forces_the_terminal_rendering_of_typer() -> None:
    """Run a help test with ``GITHUB_ACTIONS`` set, as on the hosting service's runner.

    The test session itself has set the switch that turns the forcing off, so the
    child process starts without it and relies on ``conftest.py`` to set it.
    """
    environment = {
        key: value for key, value in os.environ.items() if key != "_TYPER_FORCE_DISABLE_TERMINAL"
    }
    environment.update({"GITHUB_ACTIONS": "true", "HF_HUB_OFFLINE": "1"})
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            "-p",
            "no:asyncio",
            "tests/cli/test_cli_validity.py::test_the_command_help_keeps_each_paragraph_on_one_line",
        ],
        capture_output=True,
        text=True,
        env=environment,
        cwd=LIBRARY_ROOT,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_the_docs_workflow_runs_on_push_to_main_and_on_manual_dispatch_only() -> None:
    trigger = triggers(docs_workflow())
    assert isinstance(trigger, dict)
    assert set(trigger) == {"push", "workflow_dispatch"}
    assert trigger["push"] == {"branches": ["main"]}


def test_the_docs_workflow_may_only_read_the_code_and_deploy_to_pages() -> None:
    parsed = docs_workflow()
    assert parsed["permissions"] == {"contents": "read", "pages": "write", "id-token": "write"}
    assert all("permissions" not in job for job in parsed["jobs"].values())


def test_the_docs_workflow_has_one_deploy_job_in_the_github_pages_environment() -> None:
    jobs = docs_workflow()["jobs"]
    assert set(jobs) == {"deploy"}
    job = jobs["deploy"]
    assert job["runs-on"] == "ubuntu-latest"
    assert job["environment"] == {
        "name": "github-pages",
        "url": "${{ steps.deployment.outputs.page_url }}",
    }
    deployment = [step for step in job["steps"] if step.get("id") == "deployment"]
    assert [step["uses"].split("@")[0] for step in deployment] == ["actions/deploy-pages"]


def test_the_docs_workflow_builds_the_site_strictly_on_python_3_12_with_the_docs_extra() -> None:
    job = docs_workflow()["jobs"]["deploy"]
    setup = next(step for step in job["steps"] if step.get("uses", "").startswith("actions/setup"))
    assert setup["with"]["python-version"] == "3.12"
    assert run_commands(job) == ['pip install -e ".[docs]"', "mkdocs build --strict"]


def test_the_docs_workflow_uploads_the_site_and_deploys_it_after_the_build() -> None:
    job = docs_workflow()["jobs"]["deploy"]
    steps = job["steps"]
    build = next(i for i, step in enumerate(steps) if step.get("run") == "mkdocs build --strict")
    pages = [(i, s) for i, s in enumerate(steps) if s.get("uses", "").startswith(PAGES_ACTIONS)]
    assert [step["uses"].split("@")[0] for _, step in pages] == list(PAGES_ACTIONS)
    assert all(i > build for i, _ in pages)
    upload = next(step for _, step in pages if "upload-pages-artifact" in step["uses"])
    assert upload["with"] == {"path": "site"}
    assert [action.split("@")[0] for action in uses(job)[:2]] == [
        "actions/checkout",
        "actions/setup-python",
    ]


def test_every_action_of_the_docs_workflow_is_pinned_to_a_major_version() -> None:
    job = docs_workflow()["jobs"]["deploy"]
    assert all(re.fullmatch(r"[\w.-]+/[\w.-]+@v\d+", action) for action in uses(job))
    assert len(uses(job)) == 5


def test_the_docs_workflow_sets_the_same_environment_as_ci_and_references_no_secret() -> None:
    parsed = docs_workflow()
    assert parsed["env"] == workflow()["env"] == {"NO_MKDOCS_2_WARNING": "1", "HF_HUB_OFFLINE": "1"}
    text = DOCS_WORKFLOW.read_text(encoding="utf-8")
    assert "secrets." not in text and "${{ secrets" not in text


def test_a_running_deployment_is_never_cancelled_half_way() -> None:
    assert docs_workflow()["concurrency"] == {"group": "pages", "cancel-in-progress": False}
