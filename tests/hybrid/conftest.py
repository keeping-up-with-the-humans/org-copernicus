"""Fixtures of the hybrid coder tests: a coded toy store, built once per module."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from copernicus.assets import load_toy_cues, toy_corpus_path
from copernicus.coders import CodingSettings, code_texts
from copernicus.data import TextTable
from copernicus.llm import FakeBackend, cue_responder
from copernicus.rubric.loader import load_rubric, toy_rubric_path
from copernicus.rubric.model import Rubric

if not os.environ.get("COPERNICUS_TEST_DOWNLOAD"):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

N_TEXTS = 30
N_ONE_STAGE_TEXTS = 3


@pytest.fixture(scope="module")
def rubric() -> Rubric:
    """Return the toy rubric."""
    return load_rubric(toy_rubric_path())


@pytest.fixture(scope="module")
def table() -> TextTable:
    """Return the first texts of the toy corpus."""
    corpus = TextTable.read(toy_corpus_path(), group_id="group_id")
    return corpus.subset(corpus.ids()[:N_TEXTS])


@pytest.fixture(scope="module")
def coded_store(tmp_path_factory: pytest.TempPathFactory, rubric: Rubric, table: TextTable) -> Path:
    """Code the table once with the two-stage coder and the cue backend; return the store."""
    path = tmp_path_factory.mktemp("hybrid") / "two_stage.sqlite"
    backend = FakeBackend(cue_responder(rubric, load_toy_cues()))
    code_texts(table, rubric, backend, path, CodingSettings(runs=1))
    return path


@pytest.fixture(scope="module")
def one_stage_store(
    tmp_path_factory: pytest.TempPathFactory, rubric: Rubric, table: TextTable
) -> Path:
    """Code the first texts with the one-stage coder; such a store holds no evidence."""
    path = tmp_path_factory.mktemp("hybrid") / "one_stage.sqlite"
    backend = FakeBackend(cue_responder(rubric, load_toy_cues()))
    small = table.subset(table.ids()[:N_ONE_STAGE_TEXTS])
    code_texts(small, rubric, backend, path, CodingSettings(runs=1, method="one_stage"))
    return path
