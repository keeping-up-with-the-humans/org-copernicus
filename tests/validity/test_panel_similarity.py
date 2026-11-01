"""The validity panel on genuine output of the similarity coder at both target levels."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from similarity_support import DIMENSION, make_references, make_texts

from copernicus.errors import ConfigurationError
from copernicus.rubric.loader import load_rubric, toy_rubric_path
from copernicus.rubric.model import Rubric
from copernicus.similarity import SimilaritySettings, score_similarity
from copernicus.validity import association
from copernicus.validity.panel import PanelSettings, validity_panel

N_TEXTS = 60
SETTINGS = PanelSettings(n_boot=50, covariate="length")


@pytest.fixture(scope="module")
def scored() -> tuple[Rubric, pd.DataFrame, pd.DataFrame]:
    """Similarity scores at the construct and the attribute level, and a gold frame."""
    rubric = load_rubric(toy_rubric_path())
    rng = np.random.default_rng(5)
    rows = rng.normal(scale=0.3, size=(N_TEXTS, DIMENSION))
    latent = rng.normal(size=N_TEXTS)
    rows[:, 0] = latent
    texts = make_texts(rows)
    refs = make_references(rubric)
    attribute_level = replace(SimilaritySettings(), target_level="attribute")
    frame = pd.concat(
        [
            score_similarity(texts, refs, rubric, SimilaritySettings()),
            score_similarity(texts, refs, rubric, attribute_level),
        ],
        ignore_index=True,
    )
    gold = pd.DataFrame(
        {
            "text_id": texts.ids,
            "gold": latent + rng.normal(scale=0.5, size=N_TEXTS),
            "length": rng.integers(20, 400, size=N_TEXTS).astype(float),
        }
    )
    return rubric, frame, gold


def test_the_panel_reads_both_target_levels_of_similarity_output(scored):
    rubric, frame, gold = scored
    attribute = rubric.attributes[0].attribute_id
    rollup = f"{rubric.rubric_id}__rollup"
    selected = {
        "construct": ("construct", "construct"),
        rollup: ("attribute", rollup),
        f"attribute:{attribute}": ("attribute", attribute),
        f"construct:{attribute}": ("construct", attribute),
    }
    targets = dict.fromkeys(selected, "gold")
    panel = validity_panel(frame, gold, None, targets=targets, settings=SETTINGS)
    rows = panel.frame.set_index("target")
    assert list(panel.frame["target"]) == list(targets)
    assert (rows["status"] == "ok").all() and (rows["n"] == N_TEXTS).all()
    assert (rows["n_runs"] == 1).all() and rows["noise"].isna().all()
    for key, (level, target_id) in selected.items():
        chosen = frame[(frame["target_level"] == level) & (frame["target_id"] == target_id)]
        score = chosen.set_index("text_id").loc[gold["text_id"], "score"].to_numpy()
        expected = association(score, gold["gold"].to_numpy())
        assert rows.loc[key, "rank_validity"] == pytest.approx(expected.rank_validity, abs=1e-12)
    assert rows.loc["construct", "rank_validity"] != pytest.approx(
        rows.loc[rollup, "rank_validity"]
    )


def test_a_plain_attribute_id_found_at_both_levels_is_refused(scored):
    rubric, frame, gold = scored
    attribute = rubric.attributes[0].attribute_id
    with pytest.raises(ConfigurationError, match=rf"'{attribute}'.*several target levels"):
        validity_panel(frame, gold, None, targets={attribute: "gold"}, settings=SETTINGS)
