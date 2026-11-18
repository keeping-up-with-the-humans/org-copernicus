"""The summary of the evidence texts of a hybrid run without a text that uses its passages."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
from hybrid_support import STUB_DIMENSION, StubEncoder, unit_references

from copernicus.data import TextTable
from copernicus.hybrid import score_hybrid
from copernicus.rubric.model import Rubric
from copernicus.similarity import SimilaritySettings

MEAN = SimilaritySettings(reduction="mean")


@pytest.mark.parametrize("granularity", ["construct", "attribute"])
def test_a_run_without_evidence_texts_has_no_mean_words_in_its_summary(
    coded_store: Path, rubric: Rubric, table: TextTable, granularity: str
) -> None:
    refs = unit_references(rubric, "stub", STUB_DIMENSION)
    result = score_hybrid(
        coded_store, table, rubric, StubEncoder(), refs, MEAN, granularity=granularity,
        min_words=1000,
    )  # fmt: skip
    summary = result.summary
    cells = [summary] if granularity == "construct" else list(summary["by_attribute"].values())
    assert len(cells) == (1 if granularity == "construct" else len(rubric.attributes))
    for cell in cells:
        counts = (cell["n_short"], cell["n_evidence"], cell["n_passages_embedded"])
        assert counts == (len(table), 0, 0) and math.isnan(cell["mean_words_evidence"])
    assert result.frame["score"].notna().all()
