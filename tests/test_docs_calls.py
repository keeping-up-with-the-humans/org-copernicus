"""Calls that the documentation shows outside its tagged blocks run as the pages show them.

The tagged python blocks run in ``test_docs_build``, and every documented command answers
``--help`` there. The calls below are shown inline, in an untagged block or as options added to a
documented command. Most tests read the call from its page and run it on the toy data or on a
small synthetic stand-in, so a call that needs an argument the page does not show fails here;
others check what a page shows against the library, such as the counts of the FAQ's dry run.
The instrumented-coding test reads the option names and the instruction about the two
measurements from its page and builds the values of the options from fixtures.
"""

from __future__ import annotations

import re
import shlex
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from corrections_cli_support import estimate_args, write_corpus, write_drawn_gold
from corrections_support import BATTERY_SCORES, embedding_data, make_embedding_set
from docs_support import FENCE, shell_commands
from similarity_support import SIDECAR
from typer.testing import CliRunner

import copernicus as cop
from copernicus.assets import toy_corpus_path
from copernicus.cli._estimate_kinds import REQUIRED
from copernicus.cli.app import app
from copernicus.references import estimate_tokens
from copernicus.rubric.loader import toy_rubric_path

DOCS = Path(__file__).resolve().parents[1] / "docs"
SPAN = re.compile(r"`([^`\n]+)`")
DRY_RUN_PAGES = ("faq.md", "installation.md", "tutorials/similarity-coder.md")


def prose(page: str) -> str:
    """The text of a page without its fenced blocks."""
    kept: list[str] = []
    fenced = False
    for line in (DOCS / page).read_text(encoding="utf-8").splitlines():
        if line.strip().startswith(FENCE):
            fenced = not fenced
            continue
        if not fenced:
            kept.append(line)
    return "\n".join(kept)


def spans(page: str, start: str) -> list[str]:
    """The inline code spans of a page's prose that start with the given text."""
    return [span for span in SPAN.findall(prose(page)) if span.startswith(start)]


def numbers(output: str) -> list[int]:
    """The first integer of each line of a command's output."""
    return [int(re.search(r"\d+", line)[0]) for line in output.splitlines()]  # type: ignore[index]


def test_the_decision_guide_calls_embedding_augmented_calibration_with_one_encoder() -> None:
    shown = spans("decision-guide.md", 'get_correction("embedding_augmented"')
    assert shown, "the decision guide shows no call of embedding-augmented calibration"
    assert [span for span in shown if "encoder=" not in span] == []
    calls = [span for span in shown if "..." not in span]
    assert calls, "the decision guide shows no complete call"
    rng = np.random.default_rng(3)
    gold = rng.normal(size=60)
    vectors = np.column_stack([gold + rng.normal(scale=0.5, size=60), rng.normal(size=(60, 7))])
    data = embedding_data(gold, gold + rng.normal(size=60), {"minilm": vectors})
    for call in calls:
        # The call is read from the page and run as the page shows it.
        correction = eval(call, {"get_correction": cop.get_correction})
        corrected = cop.cross_fit_predictions(correction, data, k=3, seed=1)
        assert len(corrected) == data.n and bool(np.isfinite(corrected.to_numpy()).all())


@pytest.mark.parametrize("page", DRY_RUN_PAGES)
def test_the_shown_dry_run_of_the_reference_generator_runs(
    page: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commands = spans(page, "copernicus references generate")
    assert commands, f"{page} shows no reference generation"
    monkeypatch.chdir(tmp_path)
    shutil.copyfile(toy_rubric_path(), tmp_path / "rubric.json")
    rubric = cop.load_rubric(toy_rubric_path())
    for command in commands:
        words = shlex.split(command)
        assert "--dry-run" in words, command
        result = CliRunner().invoke(app, words[1:])
        assert result.exit_code == 0, result.output
        writers = words.count("--model")
        expected = estimate_tokens(rubric, cop.GeneratorSettings(), n_writers=writers)
        assert numbers(result.stdout) == [
            expected.calls,
            expected.prompt_tokens,
            expected.completion_tokens,
        ]


def test_the_faq_shows_the_counts_of_its_dry_run() -> None:
    # The suite makes tiktoken unimportable, so the estimate counts characters / 4, the counter
    # that the page's output names: every shown number is checked under that counter.
    text = (DOCS / "faq.md").read_text(encoding="utf-8")
    command = spans("faq.md", "copernicus references generate")[0]
    rubric = cop.load_rubric(toy_rubric_path())
    expected = estimate_tokens(rubric, cop.GeneratorSettings(), n_writers=command.count("--model"))
    assert expected.counter == "characters / 4"
    assert f"planned calls: {expected.calls}\n" in text
    assert f"estimated prompt tokens: {expected.prompt_tokens} ({expected.counter})\n" in text
    assert f"estimated completion tokens: {expected.completion_tokens} (" in text


def test_the_similarity_tutorial_dry_run_prints_the_numbers_of_its_python_estimate() -> None:
    command = spans("tutorials/similarity-coder.md", "copernicus references generate")[0]
    page = (DOCS / "tutorials/similarity-coder.md").read_text(encoding="utf-8")
    assert "estimate_tokens(rubric, cop.GeneratorSettings(), n_writers=2)" in page
    assert command.count("--model") == 2


def test_the_coder_score_options_of_the_baseline_command_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    page = "tutorials/nlp-ml-baseline.md"
    shown = spans(page, "--coder-scores")
    assert len(shown) == 1, shown
    baseline = [
        words
        for _, words in shell_commands((DOCS / page).read_text(encoding="utf-8"))
        if words[:3] == ("copernicus", "baseline", "ridge")
    ]
    assert len(baseline) == 1
    monkeypatch.chdir(tmp_path)
    shutil.copyfile(toy_corpus_path(), "toy_corpus.csv")
    shutil.copyfile(toy_rubric_path(), "rubric.json")
    runner = CliRunner()
    for args in (
        "code two-stage --texts toy_corpus.csv --rubric rubric.json --out run.sqlite "
        "--backend fake --group-column group_id",
        "store export run.sqlite --what text --out text_scores.csv",
    ):
        assert runner.invoke(app, shlex.split(args)).exit_code == 0, args
    ids = pd.read_csv("toy_corpus.csv", dtype={"text_id": str})["text_id"].tolist()
    vectors = np.random.default_rng(5).normal(size=(len(ids), 8))
    replace(make_embedding_set(ids, vectors, "minilm"), sidecar=dict(SIDECAR)).save("texts_minilm")
    result = runner.invoke(app, [*baseline[0][1:], *shlex.split(shown[0])])
    assert result.exit_code == 0, result.output
    variants = set(pd.read_csv("baseline.csv")["variant"])
    assert variants == {"embeddings_only", "embeddings_plus_score"}


def test_the_options_the_estimate_tutorial_names_for_oriv_are_enough_to_run_it(
    tmp_path: Path,
) -> None:
    text = " ".join(prose("tutorials/correcting-estimates.md").split())
    assert "the predictor kinds take `--outcome`" in text
    clause = re.search(r"`--kind oriv` ([^;.]+)", text)
    assert clause is not None, "the tutorial names no option of --kind oriv"
    named = set(re.findall(r"`(--[a-z-]+)`", clause[1])) | {"--outcome"}
    assert set(REQUIRED["oriv"]) <= named, clause[1]
    values = {"--outcome": "n_words", "--score-a": "score_a", "--score-b": "score_b"}
    files = write_corpus(tmp_path)
    gold = write_drawn_gold(files, tmp_path)
    options = [word for option in REQUIRED["oriv"] for word in (option, values[option])]
    args = estimate_args(files, gold, tmp_path / "oriv.json", "--kind", "oriv", *options)
    result = CliRunner().invoke(app, [*args, "--n-boot", "50"])
    assert result.exit_code == 0, result.output
    # The page tells the reader to take the two measurements from the --score-columns, and a
    # measurement outside them is refused.
    assert "`--score-a` and `--score-b`, two of the `--score-columns`" in clause[1], clause[1]
    assert values["--score-a"] in BATTERY_SCORES and values["--score-b"] in BATTERY_SCORES
    outside = [word if word != values["--score-b"] else "n_words" for word in options]
    args = estimate_args(files, gold, tmp_path / "outside.json", "--kind", "oriv", *outside)
    result = CliRunner().invoke(app, [*args, "--n-boot", "50"])
    assert result.exit_code != 0, result.output


def test_the_snippet_for_a_server_of_ones_own_builds_without_the_key_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    lines = [
        line.strip().removeprefix("# backend = ")
        for line in (DOCS / "tutorials/two-stage-coder.md").read_text(encoding="utf-8").splitlines()
        if "OpenAICompatibleBackend(" in line and "base_url=" in line
    ]
    assert len(lines) == 1, lines
    # The call is read from the page and run as the page shows it.
    backend = eval(lines[0], {"cop": cop})
    assert isinstance(backend, cop.OpenAICompatibleBackend)
