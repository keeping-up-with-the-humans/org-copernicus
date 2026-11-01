"""Fixtures shared by the command-line tests."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from copernicus.data import TextTable

N_TEXTS = 6


@pytest.fixture
def runner() -> CliRunner:
    """Return a runner that captures standard output and standard error separately."""
    return CliRunner()


@pytest.fixture
def texts_csv(toy_corpus: TextTable, tmp_path: Path) -> Path:
    """Write the first six toy texts to a CSV file and return its path."""
    path = tmp_path / "texts.csv"
    toy_corpus.frame.head(N_TEXTS).to_csv(path, index=False)
    return path
