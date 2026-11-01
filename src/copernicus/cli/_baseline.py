"""The ``baseline`` command: the NLP-ML baseline, ridge on embeddings."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Any

import pandas as pd
import typer

from copernicus.baseline import BaselineResult, BaselineSettings, curves, run_ridge_baseline
from copernicus.cli._embedding_options import (
    GroupColumn,
    TextColumn,
    TextIdColumn,
    TextsOption,
    read_table,
)
from copernicus.cli._inputs import TEXT_ID, GoldOption, parse_pairs, read_gold
from copernicus.cli._logging import Verbose, cli_logging
from copernicus.cli._options import cli_errors
from copernicus.cli._output import (
    check_table_output,
    format_rows,
    write_description,
    write_table,
)
from copernicus.data import read_frame
from copernicus.embeddings import EmbeddingSet
from copernicus.errors import ConfigurationError, SettingsError
from copernicus.similarity.provenance import embedding_provenance, library_stamp

logger = logging.getLogger(__name__)

__all__ = ["baseline_app"]

DEFAULTS = BaselineSettings()
ANALYSIS = "NLP-ML baseline"
FULL = "full"
SIZES_DEFAULT = ",".join(str(size) for size in DEFAULTS.sizes)
SEEDS_DEFAULT = ",".join(str(seed) for seed in DEFAULTS.seeds)
baseline_app = typer.Typer(
    help="The NLP-ML baseline: ridge regression on embeddings.", no_args_is_help=True
)


def parse_sizes(value: str) -> tuple[int | str, ...]:
    """Parse ``50,100,full`` into training sizes: positive integers and ``"full"``."""
    sizes: list[int | str] = []
    for part in (piece.strip() for piece in value.split(",")):
        if part == FULL:
            sizes.append(FULL)
        elif part.isdecimal():
            sizes.append(int(part))
        else:
            raise SettingsError(f"--sizes expects integers and 'full', got {part!r}")
    return tuple(sizes)


def parse_seeds(value: str) -> tuple[int, ...]:
    """Parse ``42,43,44`` into seeds."""
    parts = [piece.strip() for piece in value.split(",")]
    if not all(part.lstrip("-").isdecimal() for part in parts):
        raise SettingsError(f"--seeds expects integers separated by commas, got {value!r}")
    return tuple(int(part) for part in parts)


def read_coder_scores(path: Path, column: str) -> pd.Series:
    """Read a coder score per text from a file with ``text_id`` and the score column.

    Raises:
        ConfigurationError: If a column is missing or a text id is repeated.
    """
    frame = read_frame(path, protected_columns=(TEXT_ID,))
    absent = [name for name in (TEXT_ID, column) if name not in frame.columns]
    if absent:
        raise ConfigurationError(
            f"{path.name} lacks column(s) {absent}; it has {list(frame.columns)}"
        )
    ids = frame[TEXT_ID].astype(str)
    if ids.duplicated().any():
        raise ConfigurationError(f"{path.name} holds more than one score for a text id")
    return pd.Series(frame[column].to_numpy(), index=pd.Index(ids), name=column)


def sibling(out: Path, label: str) -> Path:
    """Return ``<stem>_<label><suffix>`` beside the results file."""
    return out.with_name(f"{out.stem}_{label}{out.suffix}")


def describe_run(
    result: BaselineResult,
    embedded: Mapping[str, tuple[Path, EmbeddingSet]],
    inputs: Mapping[str, str | None],
    text_columns: Mapping[str, str | None],
    tables: Mapping[str, Path],
) -> dict[str, Any]:
    """Describe a baseline run for the file written beside the results.

    Args:
        result: The baseline result, with its settings and resolved inputs.
        embedded: Embedding name to its stem and the loaded set.
        inputs: Paths of the text, gold and coder-score files, and the score column.
        text_columns: The columns read from the texts file.
        tables: The sibling tables written beside the results, by label.

    Returns:
        JSON values: the settings, the gold and unit columns, the binary decision, the
        inputs, the encoder identity of every embedding set, the tables, the library
        version and the creation time.
    """
    return {
        "analysis": ANALYSIS,
        "settings": asdict(result.settings),
        "gold_column": result.gold_column,
        "unit_id": result.unit_id,
        "binary": result.binary,
        "inputs": dict(inputs),
        "text_columns": dict(text_columns),
        "embeddings": {
            name: {
                "stem": str(stem),
                **embedding_provenance(loaded.sidecar, loaded.encoder, f"embedding set {name}"),
                "n_texts": len(loaded),
            }
            for name, (stem, loaded) in embedded.items()
        },
        "tables": {label: str(path) for label, path in tables.items()},
        "n_results": len(result.results),
        "n_skipped": len(result.skipped),
        **library_stamp(),
    }


def curve_lines(curve_table: pd.DataFrame) -> list[str]:
    """Format the curves, one line per encoder set, variant and size."""
    return [
        f"{row.encoder_set}  {row.variant}  size {row.size}  rank validity "
        f"{row.rank_validity:.3f} (sd {row.rank_validity_sd:.3f}, {row.n_seeds} seeds)"
        for row in curve_table.itertuples(index=False)
    ]


@baseline_app.command("ridge")
def ridge(
    texts: TextsOption,
    gold: GoldOption,
    out: Annotated[Path, typer.Option("--out", help="The results table, .csv or .parquet.")],
    gold_column: Annotated[
        str, typer.Option("--gold-column", help="Numeric gold column to predict.")
    ],
    embeddings: Annotated[
        list[str] | None,
        typer.Option(
            "--embeddings",
            help="NAME=STEM: the embedding set at STEM, made with the encoder NAME; "
            "repeatable, at least one.",
        ),
    ] = None,
    encoder_sets: Annotated[
        list[str] | None,
        typer.Option(
            "--encoder-set",
            help="'single' (one fit per embedding set), 'stacked' (all sets together), or "
            f"the NAME of one set; repeatable; default {', '.join(DEFAULTS.encoder_sets)}.",
        ),
    ] = None,
    sizes: Annotated[
        str,
        typer.Option(
            "--sizes", help="Training sizes in units: integers and 'full', comma-separated."
        ),
    ] = SIZES_DEFAULT,
    seeds: Annotated[
        str, typer.Option("--seeds", help="Seeds of the folds and the draws, comma-separated.")
    ] = SEEDS_DEFAULT,
    k_folds: Annotated[
        int, typer.Option("--k-folds", help="Grouped outer folds.")
    ] = DEFAULTS.k_folds,
    floor_units: Annotated[
        int,
        typer.Option("--floor-units", help="Fewest units a training draw may hold."),
    ] = DEFAULTS.floor_units,
    unit_id: Annotated[
        str | None,
        typer.Option(
            "--unit-id",
            help="Column of the texts file holding the analysis unit; default the group.",
        ),
    ] = None,
    binary: Annotated[
        bool | None,
        typer.Option(
            "--binary/--no-binary",
            help="Whether the gold is binary; inferred from the gold values if omitted.",
        ),
    ] = DEFAULTS.binary,
    coder_scores: Annotated[
        Path | None,
        typer.Option(
            "--coder-scores",
            exists=True,
            dir_okay=False,
            help="File with text_id and a score; adds the embeddings-plus-score variant.",
        ),
    ] = None,
    coder_score_column: Annotated[
        str, typer.Option("--coder-score-column", help="Score column of --coder-scores.")
    ] = "score",
    text_id_column: TextIdColumn = TEXT_ID,
    text_column: TextColumn = "text",
    group_column: GroupColumn = None,
    verbose: Verbose = False,
) -> None:
    """Fit the ridge baseline on embeddings over seeds, training sizes and encoder sets.

    The results go to --out; three more files with its suffix come beside it.

    <stem>_oof: out-of-fold predictions; <stem>_skipped: skipped sizes; <stem>_curves: curves.

    The settings, the inputs and the encoder of each embedding set go to <out>.json.
    """
    with cli_logging(verbose), cli_errors():
        check_table_output(out)
        pairs = parse_pairs(embeddings or [], "--embeddings", "NAME=STEM")
        if not pairs:
            raise SettingsError("give at least one --embeddings NAME=STEM")
        options: dict[str, Any] = {
            "k_folds": k_folds,
            "seeds": parse_seeds(seeds),
            "sizes": parse_sizes(sizes),
            "floor_units": floor_units,
            "encoder_sets": tuple(encoder_sets) if encoder_sets else DEFAULTS.encoder_sets,
            "binary": binary,
        }
        settings = BaselineSettings(**options)
        embedded = {
            name: (Path(stem), EmbeddingSet.load(stem, expect_encoder=name)) for name, stem in pairs
        }
        loaded = {name: embedding_set for name, (_, embedding_set) in embedded.items()}
        table = read_table(texts, text_id_column, text_column, group_column)
        score = read_coder_scores(coder_scores, coder_score_column) if coder_scores else None
        result = run_ridge_baseline(
            loaded,
            read_gold(gold),
            table,
            settings,
            gold_column=gold_column,
            unit_id=unit_id,
            coder_score=score,
        )
        curve_table = curves(result)
        tables = {name: sibling(out, name) for name in ("oof", "skipped", "curves")}
        files = {
            "results": (out, result.results),
            "out-of-fold file": (tables["oof"], result.oof_predictions),
            "skipped file": (tables["skipped"], result.skipped),
            "curves file": (tables["curves"], curve_table),
        }
        for path, frame in files.values():
            write_table(frame, path)
        inputs = {
            "texts": str(texts),
            "gold": str(gold),
            "coder_scores": str(coder_scores) if coder_scores else None,
            "coder_score_column": coder_score_column if coder_scores else None,
        }
        columns = {TEXT_ID: text_id_column, "text": text_column, "group_id": group_column}
        description = write_description(
            out, describe_run(result, embedded, inputs, columns, tables)
        )
        rows = [("results", str(len(result.results))), ("skipped", str(len(result.skipped)))]
        rows += [(label, str(path)) for label, (path, _) in files.items()]
        rows.append(("description file", str(description)))
        typer.echo(format_rows(rows))
        if len(curve_table):
            typer.echo("\n".join(curve_lines(curve_table)))
