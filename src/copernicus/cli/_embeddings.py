"""The ``embed``, ``similarity``, and ``hybrid`` commands: embedding sets and their coders."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import typer

from copernicus.cli._embedding_options import (
    BATCH_SIZE,
    CHECKPOINT_EVERY,
    GRANULARITY,
    MIN_WORDS,
    RUN_NUMBER,
    SIMILARITY_DEFAULTS,
    BatchSizeOption,
    CheckpointEveryOption,
    CheckpointOption,
    DeviceOption,
    EncoderOption,
    GroupColumn,
    HybridLongTextOption,
    HybridPolarityOption,
    HybridReductionOption,
    HybridTargetLevelOption,
    HybridWriterPoolingOption,
    LongTextOption,
    MaxTokensOption,
    PolarityOption,
    PoolingOption,
    ReductionOption,
    ReferencesEmbeddingsOption,
    RevisionOption,
    RubricOption,
    TargetLevelOption,
    TextColumn,
    TextIdColumn,
    TextsOption,
    WriterPoolingOption,
    make_encoder,
    read_table,
    scoring_settings,
)
from copernicus.cli._logging import Verbose, cli_logging
from copernicus.cli._options import cli_errors
from copernicus.cli._output import (
    check_stem_output,
    check_table_output,
    format_rows,
    write_description,
    write_table,
)
from copernicus.embeddings import DEFAULT_ENCODER, EmbeddingSet
from copernicus.errors import SettingsError
from copernicus.hybrid import HYBRID_DEFAULTS, score_hybrid
from copernicus.references import ReferenceSet
from copernicus.rubric import load_rubric
from copernicus.similarity import (
    PAPER_GRID,
    ReferenceEmbeddings,
    embed_references,
    score_similarity,
    score_similarity_many,
    scoring_provenance,
)

logger = logging.getLogger(__name__)

__all__ = ["embed_app", "hybrid_app", "similarity_app"]

GRID_PAPER = "paper"
GRID_NOTE = (
    "note: the paper's preset spans 8 (encoder, style) pairs; this run scores the pair of the "
    "given embeddings under all {settings} settings. Run the other pairs the same way, or "
    "assemble the preset in Python with PAPER_KEYS and run_grid."
)
embed_app = typer.Typer(
    help="Embed texts and reference texts with an encoder.", no_args_is_help=True
)
similarity_app = typer.Typer(
    help="Score embedded texts with the similarity coder.", no_args_is_help=True
)
hybrid_app = typer.Typer(
    help="Score texts through the verified passages of a two-stage run with the hybrid coder.",
    no_args_is_help=True,
)


@embed_app.command("texts")
def embed_texts(
    texts: TextsOption,
    out: Annotated[
        Path, typer.Option("--out", help="Stem of the output; writes <stem>.npz/.json.")
    ],
    encoder: EncoderOption = DEFAULT_ENCODER,
    pooling: PoolingOption = None,
    max_tokens: MaxTokensOption = None,
    revision: RevisionOption = None,
    text_id_column: TextIdColumn = "text_id",
    text_column: TextColumn = "text",
    group_column: GroupColumn = None,
    device: DeviceOption = None,
    batch_size: BatchSizeOption = BATCH_SIZE,
    checkpoint: CheckpointOption = None,
    checkpoint_every: CheckpointEveryOption = CHECKPOINT_EVERY,
    verbose: Verbose = False,
) -> None:
    """Embed a table of texts into an embedding set (needs the 'embed' extra)."""
    with cli_logging(verbose), cli_errors():
        check_stem_output(out)
        embedder = make_encoder(encoder, pooling, max_tokens, device, batch_size, revision)
        table = read_table(texts, text_id_column, text_column, group_column)
        logger.info("embedding %d texts with %s", len(table), encoder)
        embedded = embedder.embed_table(
            table, checkpoint=checkpoint, checkpoint_every=checkpoint_every
        )
        array_path, description_path = embedded.save(out)
        typer.echo(
            format_rows(
                [
                    ("texts", str(len(embedded))),
                    ("chunks", str(len(embedded.chunk_owner))),
                    ("array file", str(array_path)),
                    ("description file", str(description_path)),
                ]
            )
        )


@embed_app.command("references")
def embed_reference_texts(
    references: Annotated[
        Path,
        typer.Option("--references", exists=True, dir_okay=False, help="Reference texts table."),
    ],
    out: Annotated[
        Path, typer.Option("--out", help="Stem of the output; writes <stem>.npz/.json.")
    ],
    encoder: EncoderOption = DEFAULT_ENCODER,
    pooling: PoolingOption = None,
    max_tokens: MaxTokensOption = None,
    revision: RevisionOption = None,
    device: DeviceOption = None,
    batch_size: BatchSizeOption = BATCH_SIZE,
    verbose: Verbose = False,
) -> None:
    """Embed the kept reference texts into reference embeddings (needs the 'embed' extra)."""
    with cli_logging(verbose), cli_errors():
        check_stem_output(out)
        embedder = make_encoder(encoder, pooling, max_tokens, device, batch_size, revision)
        reference_set = ReferenceSet.load(references)
        logger.info("embedding the reference texts of %s with %s", references, encoder)
        embedded = embed_references(reference_set, embedder)
        array_path, description_path = embedded.save(out)
        typer.echo(
            format_rows(
                [
                    ("references", str(len(embedded))),
                    ("array file", str(array_path)),
                    ("description file", str(description_path)),
                ]
            )
        )


@similarity_app.command("score")
def score(
    texts_embeddings: Annotated[
        Path, typer.Option("--texts-embeddings", help="Stem of the embedded texts.")
    ],
    references_embeddings: ReferencesEmbeddingsOption,
    rubric: RubricOption,
    out: Annotated[Path, typer.Option("--out", help="Scores, .csv or .parquet.")],
    polarity: PolarityOption = None,
    reduction: ReductionOption = None,
    writer_pooling: WriterPoolingOption = None,
    long_text: LongTextOption = None,
    target_level: TargetLevelOption = None,
    grid: Annotated[
        str | None,
        typer.Option("--grid", help="'paper' scores all 144 settings of the paper's preset."),
    ] = None,
    verbose: Verbose = False,
) -> None:
    """Score embedded texts against the poles of the reference embeddings.

    The setting options default to the paper's configuration of the similarity coder. The
    file <out>.json beside the scores records the settings, the encoder of both inputs, the
    rubric, and the number of rows.
    """
    with cli_logging(verbose), cli_errors():
        check_table_output(out)
        axes = (polarity, reduction, writer_pooling, long_text, target_level)
        if grid is not None and grid != GRID_PAPER:
            raise SettingsError(f"--grid must be '{GRID_PAPER}', got {grid!r}")
        if grid is not None and any(value is not None for value in axes):
            raise SettingsError("--grid sets every setting; do not combine it with setting options")
        scored = PAPER_GRID if grid is not None else (scoring_settings(SIMILARITY_DEFAULTS, *axes),)
        loaded_rubric = load_rubric(rubric)
        texts = EmbeddingSet.load(texts_embeddings)
        refs = ReferenceEmbeddings.load(references_embeddings, expect_encoder=texts.encoder)
        if grid is not None:
            logger.info("scoring %d texts under %d settings", len(texts), len(PAPER_GRID))
            frame = score_similarity_many(texts, refs, loaded_rubric, PAPER_GRID)
            typer.echo(GRID_NOTE.format(settings=len(PAPER_GRID)), err=True)
        else:
            logger.info("scoring %d texts under %s", len(texts), scored[0].label())
            frame = score_similarity(texts, refs, loaded_rubric, scored[0])
        write_table(frame, out)
        description = write_description(
            out, scoring_provenance(texts, refs, loaded_rubric, scored, len(frame))
        )
        rows = [("rows", str(len(frame))), ("file", str(out))]
        typer.echo(format_rows([*rows, ("description file", str(description))]))


@hybrid_app.command("score")
def score_hybrid_command(
    store: Annotated[
        Path,
        typer.Option("--store", exists=True, dir_okay=False, help="Run store of a two-stage run."),
    ],
    texts: TextsOption,
    rubric: RubricOption,
    references_embeddings: ReferencesEmbeddingsOption,
    out: Annotated[Path, typer.Option("--out", help="Scores, .csv or .parquet.")],
    encoder: EncoderOption = DEFAULT_ENCODER,
    pooling: PoolingOption = None,
    max_tokens: MaxTokensOption = None,
    revision: RevisionOption = None,
    granularity: Annotated[
        str, typer.Option("--granularity", help="'construct' or 'attribute' evidence texts.")
    ] = GRANULARITY,
    run_number: Annotated[
        int, typer.Option("--run-number", help="Run of the store whose passages are read.")
    ] = RUN_NUMBER,
    min_words: Annotated[
        int, typer.Option("--min-words", help="Texts with fewer words are embedded as they are.")
    ] = MIN_WORDS,
    polarity: HybridPolarityOption = None,
    reduction: HybridReductionOption = None,
    writer_pooling: HybridWriterPoolingOption = None,
    long_text: HybridLongTextOption = None,
    target_level: HybridTargetLevelOption = None,
    text_id_column: TextIdColumn = "text_id",
    text_column: TextColumn = "text",
    group_column: GroupColumn = None,
    device: DeviceOption = None,
    batch_size: BatchSizeOption = BATCH_SIZE,
    checkpoint: CheckpointOption = None,
    checkpoint_every: CheckpointEveryOption = CHECKPOINT_EVERY,
    verbose: Verbose = False,
) -> None:
    """Embed the verified passages of a two-stage run and score them (needs the 'embed' extra).

    The setting options default to the paper's hybrid coder: the vector of each whole
    evidence text, scored against each pole reduced once over the references of all writers.
    The file <out>.json beside the scores records the settings, the encoder of both inputs,
    the rubric, the store, the evidence summary, and the number of rows.
    """
    with cli_logging(verbose), cli_errors():
        check_table_output(out)
        axes = (polarity, reduction, writer_pooling, long_text, target_level)
        settings = scoring_settings(HYBRID_DEFAULTS, *axes)
        embedder = make_encoder(encoder, pooling, max_tokens, device, batch_size, revision)
        loaded_rubric = load_rubric(rubric)
        table = read_table(texts, text_id_column, text_column, group_column)
        refs = ReferenceEmbeddings.load(references_embeddings, expect_encoder=embedder.spec.name)
        logger.info("scoring %d texts through the passages of %s", len(table), store)
        result = score_hybrid(
            store,
            table,
            loaded_rubric,
            embedder,
            refs,
            settings,
            granularity=granularity,
            run_number=run_number,
            min_words=min_words,
            checkpoint=checkpoint,
            checkpoint_every=checkpoint_every,
        )
        write_table(result.frame, out)
        description = write_description(out, {**result.provenance, "summary": result.summary})
        counts = [
            (key, f"{value:.1f}" if isinstance(value, float) else str(value))
            for key, value in result.summary.items()
            if not isinstance(value, dict)
        ]
        files = [("rows", str(len(result.frame))), ("file", str(out))]
        typer.echo(format_rows([*counts, *files, ("description file", str(description))]))
