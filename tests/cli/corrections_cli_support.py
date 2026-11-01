"""The synthetic corpus of the ``corrections`` command tests, written as files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from corrections_support import BATTERY_SCORES, battery_frame
from typer.testing import CliRunner

from copernicus.cli._embedding_options import read_table
from copernicus.cli._inputs import read_gold
from copernicus.cli.app import app
from copernicus.corrections import CorrectionData

SCORES = ",".join(BATTERY_SCORES)
N_DRAWN = 120


def write_corpus(folder: Path) -> dict[str, Path]:
    """Write the battery corpus as a texts, a scores and a gold file.

    The texts vary in length (so ``log_chars`` varies) and carry three further columns for the
    group-split tests: ``flag`` (boolean, by group), ``half`` (0/1, its complement) and
    ``level`` (0, 1 or 2).

    Returns:
        The paths by name: ``texts``, ``scores`` and ``gold``.
    """
    frame = battery_frame()
    lengths = np.random.default_rng(9).integers(5, 60, size=len(frame))
    group_number = frame["group_id"].str[1:].astype(int)
    texts = pd.DataFrame(
        {
            "text_id": frame["text_id"],
            "text": ["word " * int(length) for length in lengths],
            "group_id": frame["group_id"],
            "flag": (group_number % 2 == 0).to_numpy(),
            "half": (group_number % 2).to_numpy(),
            "level": (group_number % 3).to_numpy(),
        }
    )
    paths = {name: folder / f"{name}.csv" for name in ("texts", "scores", "gold")}
    texts.to_csv(paths["texts"], index=False)
    frame[["text_id", *BATTERY_SCORES]].to_csv(paths["scores"], index=False)
    frame[["text_id", "gold"]].to_csv(paths["gold"], index=False)
    return paths


def write_drawn_gold(files: dict[str, Path], folder: Path) -> Path:
    """Draw a human sample with the sample command and write its gold values with ``pi``."""
    sample_path = folder / "sample.csv"
    args = [
        *("corrections", "sample", "--texts", str(files["texts"]), "--group-column", "group_id"),
        *("--n", str(N_DRAWN), "--seed", "11", "--out", str(sample_path)),
    ]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.stderr
    sample = pd.read_csv(sample_path)
    gold = pd.read_csv(files["gold"]).set_index("text_id")["gold"]
    coded = sample.assign(gold=gold.loc[sample["text_id"]].to_numpy())
    path = folder / "coded.csv"
    coded[["text_id", "gold", "pi"]].to_csv(path, index=False)
    return path


def data_args(files: dict[str, Path], gold: Path | None = None) -> list[str]:
    """The data options of the battery and estimate commands."""
    return [
        *("--scores", str(files["scores"]), "--texts", str(files["texts"])),
        *("--gold", str(gold or files["gold"]), "--gold-column", "gold"),
        *("--score-columns", SCORES, "--group-column", "group_id"),
    ]


def estimate_args(files: dict[str, Path], gold: Path, out: Path, *extra: str) -> list[str]:
    """The arguments of ``corrections estimate`` on the corpus and a drawn sample."""
    return ["corrections", "estimate", *data_args(files, gold), "--out", str(out), *extra]


def library_data(
    files: dict[str, Path], gold: Path | None = None, **options: object
) -> CorrectionData:
    """The correction data the commands build from the same files."""
    table = read_table(files["texts"], "text_id", "text", "group_id")
    return CorrectionData.from_scores(
        pd.read_csv(files["scores"]),
        table,
        read_gold(gold or files["gold"]),
        gold_column="gold",
        score_columns=BATTERY_SCORES,
        **options,  # type: ignore[arg-type]
    )
