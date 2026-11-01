"""Run the whole pipeline of the library on the toy corpus.

The script codes the texts with the two-stage and the one-stage LLM coder, embeds them with
MiniLM, scores them with the similarity coder and the hybrid coder, checks every score against
the gold value with the validity panel, fits the NLP-ML baseline, draws a human sample, and
corrects scores and estimates with it. It needs no API key and no GPU: the default backend
answers from a table of cue words, and the encoder runs on the CPU.

Run it with ``python examples/end_to_end.py OUT_DIR``. The results are written to ``OUT_DIR``.
The steps are in ``end_to_end_steps.py``, which has to stay in the same folder.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd
from end_to_end_steps import (
    baseline_records,
    build_reference_set,
    code_both_ways,
    correct_with_a_sample,
    join_construct_scores,
    panel_record,
    validate,
)

import copernicus as cop
from copernicus.llm import Backend

logger = logging.getLogger(__name__)

__all__ = ["CSV_FILES", "join_construct_scores", "main", "run"]

BATTERY_SIZE = 20
BATTERY_SCHEME = "kfold5"
BATTERY_SEEDS = (42,)
BATTERY_FAMILIES = (
    ("regression_calibration", "linear"),
    ("supervised_stacking", "ridge"),
    ("correction_ensemble", "mean"),
)
CSV_FILES = (
    "two_stage_scores.csv",
    "one_stage_scores.csv",
    "similarity_scores.csv",
    "hybrid_scores.csv",
    "validity_panels.csv",
    "baseline_curves.csv",
    "battery_results.csv",
    "battery_summary.csv",
    "battery_skipped.csv",
)


def run(
    out_dir: Path,
    *,
    backend: Backend | None = None,
    encoder: str = "minilm",
    n_boot: int = 200,
) -> dict[str, Any]:
    """Run every step of the pipeline on the toy corpus and write the results.

    Args:
        out_dir: Folder for the run stores, the embeddings, the CSV files, the validity
            report, and ``summary.json``; created when absent.
        backend: The LLM backend; ``None`` uses the fake backend that answers from the cue
            words of the toy rubric.
        encoder: Name of the registered encoder that embeds the texts and the references.
        n_boot: Number of bootstrap draws of each validity panel.

    Returns:
        The summary that is also written to ``summary.json``.
    """
    started = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    rubric = cop.load_rubric(cop.rubric.loader.toy_rubric_path())
    table = cop.TextTable.read(cop.assets.toy_corpus_path(), group_id="group_id")
    if backend is None:
        backend = cop.FakeBackend(cop.cue_responder(rubric, cop.assets.load_toy_cues()))
    two_stage_path, one_stage_path = code_both_ways(table, rubric, backend, out_dir)
    two_stage_scores = cop.export_scores(two_stage_path)
    one_stage_scores = cop.export_scores(one_stage_path)

    model = cop.Encoder(encoder, device="cpu")
    texts = model.embed(table.frame["text"].tolist(), table.frame["text_id"].tolist())
    texts.save(out_dir / f"texts_{encoder}")
    references = cop.embed_references(build_reference_set(rubric), model)
    references.save(out_dir / f"references_{encoder}")
    similarity = cop.score_similarity(texts, references, rubric, cop.SimilaritySettings())
    hybrid = cop.score_hybrid(
        two_stage_path, table, rubric, model, references, min_words=5, granularity="construct"
    )

    frames = {
        "two_stage": two_stage_scores,
        "one_stage": one_stage_scores,
        "similarity": similarity,
        "hybrid": hybrid.frame,
    }
    panels = validate(frames, table, n_boot)
    report = "\n\n".join(f"== {coder} ==\n{panel.report()}" for coder, panel in panels.items())
    (out_dir / "validity_report.txt").write_text(report + "\n", encoding="utf-8")

    curve, baseline = baseline_records(texts, encoder, table)
    wide = join_construct_scores(
        cop.export_text_scores(two_stage_path), cop.export_text_scores(one_stage_path)
    )
    full, corrections = correct_with_a_sample(wide, table)
    battery = cop.run_battery(
        full,
        cop.BatterySettings(
            families=BATTERY_FAMILIES,
            sizes=(BATTERY_SIZE,),
            schemes=(BATTERY_SCHEME,),
            seeds=BATTERY_SEEDS,
        ),
    )

    tables = {
        "two_stage_scores.csv": two_stage_scores,
        "one_stage_scores.csv": one_stage_scores,
        "similarity_scores.csv": similarity,
        "hybrid_scores.csv": hybrid.frame,
        "validity_panels.csv": pd.concat(
            [panel.frame.assign(coder=coder) for coder, panel in panels.items()]
        ),
        "baseline_curves.csv": curve,
        "battery_results.csv": battery.results,
        "battery_summary.csv": cop.summarize_battery(battery),
        "battery_skipped.csv": battery.skipped,
    }
    for name in CSV_FILES:
        tables[name].to_csv(out_dir / name, index=False)

    summary: dict[str, Any] = {
        "n_texts": len(table),
        "texts_per_group": len(table) // int(table.frame["group_id"].nunique()),
        **{coder: panel_record(panel) for coder, panel in panels.items()},
        "hybrid_fallbacks": int(hybrid.summary["n_no_evidence"]),
        "baseline": baseline,
        **corrections,
        "battery_rows": int((~battery.results["diagnostic"]).sum()),
        "battery_skipped": len(battery.skipped),
        "files": [*CSV_FILES, "validity_report.txt"],
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    logger.info("finished in %.1f seconds", time.perf_counter() - started)
    return summary


def main(arguments: Sequence[str] | None = None) -> None:
    """Run the example with the fake backend and print the validity report.

    Args:
        arguments: Command-line arguments; ``None`` reads ``sys.argv``.
    """
    parser = argparse.ArgumentParser(description="Run the whole pipeline on the toy corpus.")
    parser.add_argument("out_dir", type=Path, help="folder for the results")
    parser.add_argument("--n-boot", type=int, default=200, help="bootstrap draws per panel")
    options = parser.parse_args(arguments)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    run(options.out_dir, n_boot=options.n_boot)
    print((options.out_dir / "validity_report.txt").read_text(encoding="utf-8"))
    print(f"summary written to {options.out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
