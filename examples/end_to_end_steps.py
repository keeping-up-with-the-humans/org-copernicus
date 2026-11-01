"""The steps of the end-to-end example, one function each.

``end_to_end.py`` holds the constants of the run, the order of the steps and the command line;
this module holds what each step does: the hand-written reference texts, the validity panels,
the baseline, the human sample and the corrections. Keep the two files in one folder.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import copernicus as cop
from copernicus.corrections import Estimate
from copernicus.llm import Backend
from copernicus.references.reference_set import FRAME_COLUMNS

logger = logging.getLogger(__name__)

__all__ = [
    "baseline_records",
    "build_reference_set",
    "code_both_ways",
    "correct_with_a_sample",
    "estimate_record",
    "join_construct_scores",
    "labelled_groups",
    "panel_record",
    "validate",
]

GOLD = "gold_constructiveness"
CONSTRUCT = "construct"
TWO_STAGE_RUNS = 2
SAMPLE_UNITS = 18
DRAW_SEED = 39
CROSS_FIT_FOLDS = 3
BASELINE_SIZES = (20, "full")
BASELINE_FLOOR_UNITS = 10
BASELINE_SEEDS = (42,)
BASELINE_FOLDS = 3
SCORE_COLUMNS = ("two_stage", "one_stage")
REFERENCE_LEVELS = ("top", "mid")
PANEL_FIELDS = ("rank_validity", "rank_validity_low", "rank_validity_high")
ESTIMATE_FIELDS = ("point", "se", "ci_low", "ci_high")

# Two generic feedback sentences for each pole of the construct and of each attribute.
REFERENCE_TEXTS: Mapping[tuple[str, str], tuple[str, str]] = {
    ("construct", "positive"): (
        "Thank you for the report. The summary is clear, and I suggest adding the sources "
        "under the second chart, because readers will want to check them.",
        "Good start. The plan would be stronger with a worked example, for instance the March "
        "order, since it shows how the steps fit together.",
    ),
    ("construct", "negative"): (
        "This is poor work. Start again.",
        "I did not enjoy reading it. It is messy and you should have done better.",
    ),
    ("specificity", "positive"): (
        "In the third paragraph on page two, replace the estimate with the figure from the audit.",
        "Move the budget table in section 4 ahead of the paragraph that refers to it.",
    ),
    ("specificity", "negative"): (
        "There are problems here and there.",
        "It could be better in general.",
    ),
    ("tone", "positive"): (
        "I appreciate the work you put in. The layout is neat, and only the closing paragraph "
        "needs another look.",
        "Well done on the opening. One small remark about the method comes next.",
    ),
    ("tone", "negative"): (
        "You were sloppy and did not bother.",
        "I expected better from you. This is lazy.",
    ),
    ("reasoning", "positive"): (
        "The claim about growth needs the sales figures, since without them nobody can verify it.",
        "Rather than report one average, show the two teams separately, because they differ a lot.",
    ),
    ("reasoning", "negative"): (
        "Just fix it.",
        "I dislike the result and have nothing more to say.",
    ),
}


def build_reference_set(rubric: cop.Rubric) -> cop.ReferenceSet:
    """Build the reference set from the hand-written texts.

    Args:
        rubric: The rubric the references belong to.

    Returns:
        A reference set with one row per text, two texts for each pole of each target,
        bound to the content of ``rubric``.
    """
    rows = [
        {
            "rubric_id": rubric.rubric_id,
            "target": target,
            "polarity": polarity,
            "level": level,
            "style": "neutral",
            "writer_model": "hand",
            "text": text,
            "call_index": 0,
            "attempt": 0,
            "seed": 0,
            "prompt_tokens": None,
            "completion_tokens": None,
            "temperature_applied": None,
            "top_p_applied": None,
            "prompt_hash": "hand",
            "prompt_version": "hand",
            "screened_out": False,
            "screen_reason": "",
        }
        for (target, polarity), texts in REFERENCE_TEXTS.items()
        for level, text in zip(REFERENCE_LEVELS, texts, strict=True)
    ]
    frame = pd.DataFrame(rows, columns=list(FRAME_COLUMNS))
    return cop.ReferenceSet.for_rubric(frame, rubric, cop.GeneratorSettings())


def panel_record(panel: cop.ValidityPanel) -> dict[str, Any]:
    """Pick the rank validity, its interval and the sample size from a one-target panel."""
    row = panel.frame.iloc[0]
    record = {name: float(row[name]) for name in PANEL_FIELDS}
    return {**record, "n_used": int(row["n_used"])}


def estimate_record(estimate: Estimate) -> dict[str, Any]:
    """Describe an estimate by its point, standard error and 95 percent interval."""
    record = {name: float(getattr(estimate, name)) for name in ESTIMATE_FIELDS}
    return {**record, "n_labelled": estimate.n_labelled, "method": estimate.method}


def code_both_ways(
    table: cop.TextTable, rubric: cop.Rubric, backend: Backend, out_dir: Path
) -> tuple[Path, Path]:
    """Code the texts with the two-stage coder (two runs) and the one-stage coder (one run).

    Args:
        table: The texts.
        rubric: The rubric.
        backend: The LLM backend both coders call.
        out_dir: Folder for the two run stores.

    Returns:
        The paths of the two-stage and the one-stage store.
    """
    two_stage = out_dir / "two_stage.sqlite"
    one_stage = out_dir / "one_stage.sqlite"
    for path, settings in (
        (two_stage, cop.CodingSettings(runs=TWO_STAGE_RUNS)),
        (one_stage, cop.CodingSettings(runs=1, method="one_stage")),
    ):
        summary = cop.code_texts(table, rubric, backend, path, settings)
        logger.info("%s: statuses %s", path.name, dict(summary.status_counts))
    return two_stage, one_stage


def validate(
    frames: Mapping[str, pd.DataFrame], table: cop.TextTable, n_boot: int
) -> dict[str, cop.ValidityPanel]:
    """Build the validity panel of every coder's construct score against the gold value."""
    settings = cop.PanelSettings(n_boot=n_boot)
    return {
        coder: cop.validity_panel(
            frame, table.frame, table, targets={CONSTRUCT: GOLD}, settings=settings
        )
        for coder, frame in frames.items()
    }


def baseline_records(
    texts: cop.EmbeddingSet, encoder: str, table: cop.TextTable
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Fit the NLP-ML baseline and return its learning curve and a record per size."""
    result = cop.run_ridge_baseline(
        {encoder: texts},
        table.frame,
        table,
        cop.BaselineSettings(
            sizes=BASELINE_SIZES,
            seeds=BASELINE_SEEDS,
            k_folds=BASELINE_FOLDS,
            floor_units=BASELINE_FLOOR_UNITS,
        ),
        gold_column=GOLD,
        unit_id="text_id",
    )
    curve = cop.curves(result)
    records = [
        {"size": str(row["size"]), "rank_validity": float(row["rank_validity"])}
        for _, row in curve.iterrows()
    ]
    return curve, records


def labelled_groups(data: cop.CorrectionData, mask: np.ndarray) -> int:
    """Count the groups that hold a labelled text inside the mask."""
    selected = data.labelled_mask & mask
    return int(pd.Series(data.groups[selected]).nunique())


def join_construct_scores(two_stage: pd.DataFrame, one_stage: pd.DataFrame) -> pd.DataFrame:
    """Join the per-text construct scores of the two coders on ``text_id``.

    Args:
        two_stage: The text scores of the two-stage run, as ``export_text_scores`` returns them.
        one_stage: The text scores of the one-stage run.

    Returns:
        A frame with ``text_id`` and one column per entry of ``SCORE_COLUMNS``, sorted by id.

    Raises:
        ValueError: If the two runs did not score the same texts.
    """
    frames = [
        frame[["text_id", "construct_score"]].rename(columns={"construct_score": name})
        for name, frame in zip(SCORE_COLUMNS, (two_stage, one_stage), strict=True)
    ]
    if set(frames[0]["text_id"]) != set(frames[1]["text_id"]):
        raise ValueError("the two-stage and the one-stage run did not score the same texts")
    joined = frames[0].merge(frames[1], on="text_id", validate="one_to_one")
    return joined.sort_values("text_id", ignore_index=True)


def correct_with_a_sample(
    scores: pd.DataFrame, table: cop.TextTable
) -> tuple[cop.CorrectionData, dict[str, Any]]:
    """Draw the human sample, then estimate a mean, a difference and cross-fitted scores.

    Args:
        scores: Wide frame with ``text_id`` and one column per entry of ``SCORE_COLUMNS``.
        table: The texts, with the gold column that stands in for the human coder.

    Returns:
        The correction data with every gold value (for the battery) and a summary of the
        human sample, the two estimates and the cross-fitted calibration.
    """
    full = cop.CorrectionData.from_scores(
        scores, table, table.frame, gold_column=GOLD, score_columns=SCORE_COLUMNS
    )
    human = table.frame.set_index("text_id")[GOLD]
    draw = cop.draw_human_sample(table, n=SAMPLE_UNITS, seed=DRAW_SEED)
    data = cop.corrections.attach_draw(full, human.loc[draw.ids], draw)
    long_texts = cop.corrections.long_short_split(data)
    mean = cop.mean_ppi(data)
    difference = cop.group_difference_ppi(data, long_texts)
    corrected = cop.cross_fit_predictions(
        cop.get_correction("regression_calibration", "linear"), data, k=CROSS_FIT_FOLDS, seed=42
    )
    truth = human.reindex(corrected.index)
    raw = data.frame.set_index("text_id")[SCORE_COLUMNS[0]]
    summary = {
        "human_sample": {
            "n_texts": len(draw.ids),
            "n_groups": int(draw.m),
            "n_groups_in_corpus": int(draw.n_groups),
            "seed": DRAW_SEED,
            "inclusion_probability": float(draw.pi),
        },
        "gold_mean": float(human.mean()),
        "raw_mean": float(raw.mean()),
        "ppi_mean": {**estimate_record(mean), "clustered": bool(mean.details["clustered"])},
        "ppi_difference": {
            **estimate_record(difference),
            "n_labelled_texts_long": int((data.labelled_mask & long_texts).sum()),
            "n_labelled_texts_short": int((data.labelled_mask & ~long_texts).sum()),
            "n_labelled_groups_long": labelled_groups(data, long_texts),
            "n_labelled_groups_short": labelled_groups(data, ~long_texts),
        },
        "cross_fit": {
            "n_texts": len(corrected),
            "mae_raw": float((raw - truth).abs().mean()),
            "mae_corrected": float((corrected - truth).abs().mean()),
        },
    }
    return full, summary
