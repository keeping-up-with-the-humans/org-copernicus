"""The ``corrections`` commands: the human sample, the size planner, the battery and the list.

``corrections estimate`` lives in ``_corrections_estimate`` and is registered here.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated, Any

import pandas as pd
import typer

from copernicus._io import json_ready
from copernicus.cli._baseline import parse_seeds, sibling
from copernicus.cli._correction_inputs import (
    COVARIATE,
    FAMILY_FORM,
    CovariateOption,
    GoldColumnOption,
    ScoreColumnsOption,
    ScoresOption,
    load_correction_data,
    parse_counts,
    parse_families,
    parse_sizes,
)
from copernicus.cli._corrections_estimate import estimate
from copernicus.cli._embedding_options import (
    GroupColumn,
    TextColumn,
    TextIdColumn,
    TextsOption,
    library_default,
    read_table,
)
from copernicus.cli._inputs import TEXT_ID, GoldOption, parse_pairs
from copernicus.cli._logging import Verbose, cli_logging
from copernicus.cli._options import cli_errors, parse_id_list
from copernicus.cli._output import check_table_output, format_rows, write_description, write_table
from copernicus.corrections import (
    BatteryResult,
    BatterySettings,
    draw_human_sample,
    list_corrections,
    plan_sample_size,
    plan_table,
    run_battery,
    summarize_battery,
)
from copernicus.corrections.draw_files import DRAW_KEY, draw_record
from copernicus.embeddings import EmbeddingSet
from copernicus.errors import SettingsError
from copernicus.similarity.provenance import embedding_provenance, library_stamp

logger = logging.getLogger(__name__)

__all__ = ["corrections_app"]

DEFAULTS = BatterySettings()
SIZES_DEFAULT = ",".join(str(size) for size in DEFAULTS.sizes)
SCHEMES_DEFAULT = ",".join(DEFAULTS.schemes)
SEEDS_DEFAULT = ",".join(str(seed) for seed in DEFAULTS.seeds)
SAMPLE_SEED = library_default(draw_human_sample, "seed")
TABLE_LABELS = ("skipped", "raw", "summary")
corrections_app = typer.Typer(
    help="Corrections of machine scores with a human-coded sample.", no_args_is_help=True
)
corrections_app.command("estimate")(estimate)


@corrections_app.command("sample")
def sample(
    texts: TextsOption,
    out: Annotated[
        Path, typer.Option("--out", help="The drawn texts with group and pi, .csv or .parquet.")
    ],
    n: Annotated[int | None, typer.Option("--n", help="Number of analysis units to code.")] = None,
    share: Annotated[
        float | None,
        typer.Option("--share", help="Share of the analysis units to code, in (0, 1]."),
    ] = None,
    seed: Annotated[
        int, typer.Option("--seed", help=f"Seed of the draw; default {SAMPLE_SEED}.")
    ] = SAMPLE_SEED,
    unit_id: Annotated[
        str | None,
        typer.Option(
            "--unit-id",
            help="Column of the texts file holding the analysis unit; default each text is one.",
        ),
    ] = None,
    text_id_column: TextIdColumn = TEXT_ID,
    text_column: TextColumn = "text",
    group_column: GroupColumn = None,
    verbose: Verbose = False,
) -> None:
    """Draw the texts a human should code: whole groups by simple random sampling.

    Give exactly one of --n and --share. Every text of a drawn group has the inclusion
    probability pi = m / G; code the drawn texts and keep pi beside the gold values.

    The drawn texts go to --out with their group and pi, the draw to <out>.json: its ids,
    groups, pi and the fingerprint of the table's texts and groups. That file binds the coded
    sample to the texts file: pass it to corrections estimate as --draw.
    """
    with cli_logging(verbose), cli_errors():
        check_table_output(out)
        table = read_table(texts, text_id_column, text_column, group_column)
        draw = draw_human_sample(table, n=n, share=share, seed=seed, unit_id=unit_id)
        frame = table.frame[[TEXT_ID, "group_id"]]
        drawn = frame[frame[TEXT_ID].astype(str).isin(draw.ids)].assign(pi=draw.pi)
        write_table(drawn, out)
        description = write_description(
            out,
            {
                "analysis": "human sample",
                "settings": {"n": n, "share": share, "seed": seed, "unit_id": unit_id},
                DRAW_KEY: draw_record(draw),
                "inputs": {"texts": str(texts)},
                "text_columns": {
                    TEXT_ID: text_id_column,
                    "text": text_column,
                    "group_id": group_column,
                },
                **library_stamp(),
            },
        )
        rows = [
            ("groups drawn", f"{draw.m} of {draw.n_groups}"),
            ("pi", f"{draw.pi:.4g}"),
            ("texts drawn", str(len(drawn))),
            ("file", str(out)),
            ("description file", str(description)),
        ]
        typer.echo(format_rows(rows))


@corrections_app.command("plan")
def plan(
    rho: Annotated[
        float, typer.Option("--rho", help="Correlation of the machine score with the human value.")
    ],
    n_human: Annotated[
        str,
        typer.Option("--n-human", help="Human-coded texts; a comma-separated list for a table."),
    ],
    n_machine: Annotated[
        str,
        typer.Option(
            "--n-machine", help="Machine-scored texts; a comma-separated list for a table."
        ),
    ],
    out: Annotated[
        Path | None,
        typer.Option("--out", help="Write the plan table, .csv or .parquet; needed for lists."),
    ] = None,
    verbose: Verbose = False,
) -> None:
    """Plan the human sample: the effective-size factor of a corrected estimate.

    The human sample counts as factor times its size; limit is the factor for unbounded
    machine-scored texts, and uncorrected is 1 / rho^2.
    """
    with cli_logging(verbose), cli_errors():
        if out is not None:
            check_table_output(out)
        humans, machines = (
            parse_counts(n_human, "--n-human"),
            parse_counts(n_machine, "--n-machine"),
        )
        if len(humans) * len(machines) > 1 and out is None:
            raise SettingsError(
                "give --out to write the plan table of the --n-human and --n-machine lists"
            )
        if len(humans) * len(machines) == 1:
            found = plan_sample_size(rho, humans[0], machines[0])
            rows = [
                ("rho", f"{found.rho:g}"),
                ("human texts", str(found.n_human)),
                ("machine texts", str(found.n_machine)),
                ("factor", f"{found.factor:.2f}"),
                ("limit", f"{found.limit:.2f}"),
                ("uncorrected", f"{found.uncorrected_equivalent:.2f}"),
            ]
            typer.echo(format_rows(rows))
        if out is not None:
            frame = plan_table(rho, humans, machines)
            write_table(frame, out)
            typer.echo(format_rows([("plans", str(len(frame))), ("file", str(out))]))


def _describe_battery(
    result: BatteryResult,
    inputs: dict[str, Any],
    text_columns: dict[str, str | None],
    embedded: dict[str, tuple[str, EmbeddingSet]],
    tables: dict[str, Path],
) -> dict[str, Any]:
    """Describe a battery run for the file written beside the results."""
    settings = result.settings
    return json_ready(
        {
            "analysis": "correction battery",
            "settings": {
                "families": settings.families,
                "entries": [entry.key for entry in settings.entries],
                "sizes": settings.sizes,
                "schemes": settings.schemes,
                "seeds": settings.seeds,
                "covariate_scale": settings.covariate_scale,
            },
            "inputs": inputs,
            "text_columns": text_columns,
            "embeddings": {
                name: {
                    "stem": stem,
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
    )


@corrections_app.command("battery")
def battery(
    scores: ScoresOption,
    texts: TextsOption,
    gold: GoldOption,
    gold_column: GoldColumnOption,
    score_columns: ScoreColumnsOption,
    out: Annotated[Path, typer.Option("--out", help="The results table, .csv or .parquet.")],
    embeddings: Annotated[
        list[str] | None,
        typer.Option(
            "--embeddings", help="NAME=STEM: the embedding set at STEM, made with NAME; repeatable."
        ),
    ] = None,
    families: Annotated[
        list[str] | None,
        typer.Option(
            "--families",
            help=f"{FAMILY_FORM} (such as embedding_augmented:embeddings_only:encoder=minilm); "
            "repeatable; default every per-text family and the ensemble, with the embedding "
            "family on the stacked --embeddings sets, on the one set given, or left out.",
        ),
    ] = None,
    sizes: Annotated[
        str, typer.Option("--sizes", help="Counts of texts and shares of a pool, comma-separated.")
    ] = SIZES_DEFAULT,
    schemes: Annotated[
        str, typer.Option("--schemes", help="kfold<k> and holdout, comma-separated.")
    ] = SCHEMES_DEFAULT,
    seeds: Annotated[str, typer.Option("--seeds", help="Seeds, comma-separated.")] = SEEDS_DEFAULT,
    covariate_scale: Annotated[
        str,
        typer.Option(
            "--covariate-scale",
            help=f"'standardized' or 'raw'; default {DEFAULTS.covariate_scale}.",
        ),
    ] = DEFAULTS.covariate_scale,
    covariate: CovariateOption = COVARIATE,
    text_id_column: TextIdColumn = TEXT_ID,
    text_column: TextColumn = "text",
    group_column: GroupColumn = None,
    verbose: Verbose = False,
) -> None:
    """Compare the corrections out of sample over seeds, schemes and sizes.

    The results go to --out; three more files with its suffix come beside it.

    <stem>_skipped: fits not scored, with the reason; <stem>_raw: the raw baseline per fold;
    <stem>_summary: the mean gains per correction.

    The settings and the inputs go to <out>.json.
    """
    with cli_logging(verbose), cli_errors():
        check_table_output(out)
        settings = BatterySettings(
            families=parse_families(families) if families else "all",
            sizes=parse_sizes(sizes),
            schemes=parse_id_list(schemes),
            seeds=parse_seeds(seeds),
            covariate_scale=covariate_scale,
        )
        pairs = parse_pairs(embeddings or [], "--embeddings", "NAME=STEM")
        embedded = {
            name: (stem, EmbeddingSet.load(stem, expect_encoder=name)) for name, stem in pairs
        }
        paths = {"scores": scores, "texts": texts, "gold": gold}
        columns = (text_id_column, text_column, group_column)
        loaded = {name: embedding_set for name, (_, embedding_set) in embedded.items()}
        data = load_correction_data(
            paths, gold_column, score_columns, covariate, columns, embeddings=loaded or None
        )
        result = run_battery(data, settings)
        summary = summarize_battery(result)
        tables = {label: sibling(out, label) for label in TABLE_LABELS}
        write_table(result.results, out)
        for label, frame in zip(TABLE_LABELS, (result.skipped, result.raw, summary), strict=True):
            write_table(frame, tables[label])
        inputs = {name: str(path) for name, path in paths.items()} | {
            "gold_column": gold_column,
            "score_columns": list(data.score_columns),
            "covariate": covariate,
        }
        text_columns = {TEXT_ID: text_id_column, "text": text_column, "group_id": group_column}
        description = write_description(
            out, _describe_battery(result, inputs, text_columns, embedded, tables)
        )
        rows = [("results", str(len(result.results))), ("skipped", str(len(result.skipped)))]
        rows += [("results file", str(out))]
        rows += [(f"{label} file", str(path)) for label, path in tables.items()]
        rows.append(("description file", str(description)))
        typer.echo(format_rows(rows))
        typer.echo(_summary_lines(summary))


def _summary_lines(summary: pd.DataFrame) -> str:
    """One line per correction: its mean gain in rank validity and its counts."""
    return "\n".join(
        f"{row.family}:{row.variant}  gain in rank validity {row.gain_rank_validity:+.3f} "
        f"({row.n_rows} rows, {row.n_skipped} skipped)"
        for row in summary.itertuples(index=False)
    )


def _left_aligned(frame: pd.DataFrame) -> str:
    """A table of text columns with every column padded to its widest entry, left-aligned."""
    columns = [str(column) for column in frame.columns]
    cells = [[str(value) for value in row] for row in frame.itertuples(index=False)]
    widths = [
        max([len(name), *(len(row[index]) for row in cells)]) for index, name in enumerate(columns)
    ]
    lines = [columns, *cells]
    return "\n".join(
        "  ".join(text.ljust(width) for text, width in zip(line, widths, strict=True)).rstrip()
        for line in lines
    )


@corrections_app.command("list")
def list_families(verbose: Verbose = False) -> None:
    """List the correction families, their variants and the errors they address."""
    with cli_logging(verbose), cli_errors():
        listing = list_corrections()
        typer.echo(_left_aligned(listing))
        typer.echo(f"\n{listing['family'].nunique()} families, {len(listing)} variants")
