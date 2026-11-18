"""The hybrid coder names the texts whose evidence text its encoder refuses.

The library's encoder runs on the fake tokenizer and table model of the embedding tests, so no
weights are loaded; the tests need torch and carry the embed marker.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest
from fake_runtime import DIMENSION, DROPPED, DroppingTokenizer, TableModel, fake_runtime
from hybrid_support import unit_references

from copernicus.assets import load_toy_cues
from copernicus.coders import CodingSettings, code_texts
from copernicus.data import TextTable
from copernicus.embeddings import Encoder, EncoderSpec
from copernicus.errors import ConfigurationError
from copernicus.hybrid import score_hybrid
from copernicus.llm import FakeBackend, cue_responder
from copernicus.rubric.model import Rubric
from copernicus.similarity import SimilaritySettings

pytestmark = pytest.mark.embed
SPEC = EncoderSpec("fake", "org/fake", "mean", 6, special_tokens_reserved=0)


@pytest.mark.parametrize("granularity", ["construct", "attribute"])
def test_a_text_whose_window_has_no_token_is_named_at_each_granularity(
    monkeypatch: pytest.MonkeyPatch, rubric: Rubric, tmp_path: Path, granularity: str
) -> None:
    frame = pd.DataFrame(
        {"text_id": ["a", "b"], "text": ["a first text", DROPPED], "group_id": ["g", "h"]}
    )
    table = TextTable.from_frame(frame)
    store = tmp_path / "run.sqlite"
    backend = FakeBackend(cue_responder(rubric, load_toy_cues()))
    code_texts(table, rubric, backend, store, CodingSettings(runs=1))
    runtime = fake_runtime(TableModel(), "mean", DroppingTokenizer())
    monkeypatch.setattr(Encoder, "_load", lambda self: runtime)
    refs = unit_references(rubric, "fake", DIMENSION, identity=SPEC.identity())
    first = rubric.attributes[0].attribute_id
    named = "'b'" if granularity == "construct" else f"'b ({first})'"
    message = (
        f"1 texts have a window that the tokenizer turned into no token, so mean pooling has no "
        f"token vector to average there: {named}; leave these texts out, or change them so that "
        "the tokenizer gives every window a token"
    )
    with pytest.raises(ConfigurationError, match=f"^{re.escape(message)}$"):
        score_hybrid(
            store, table, rubric, Encoder(SPEC), refs, SimilaritySettings(reduction="mean"),
            granularity=granularity,
        )  # fmt: skip
