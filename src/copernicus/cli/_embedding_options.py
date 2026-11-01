"""Options and helpers shared by the ``embed``, ``similarity``, and ``hybrid`` commands.

Option defaults are read from the library (the settings objects and the signatures of the
functions the commands call), so the command line and the library cannot drift apart.
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Annotated, Any

import typer

from copernicus.data import TextTable
from copernicus.embeddings import Encoder, resolve_encoder
from copernicus.embeddings.registry import POOLING_NAMES
from copernicus.errors import SettingsError
from copernicus.hybrid import HYBRID_DEFAULTS, score_hybrid
from copernicus.similarity import (
    LONG_TEXT,
    POLARITIES,
    REDUCTIONS,
    TARGET_LEVELS,
    WRITER_POOLINGS,
    SimilaritySettings,
)

logger = logging.getLogger(__name__)

__all__ = [
    "BATCH_SIZE",
    "CHECKPOINT_EVERY",
    "GRANULARITY",
    "MIN_WORDS",
    "RUN_NUMBER",
    "SIMILARITY_DEFAULTS",
    "BatchSizeOption",
    "CheckpointEveryOption",
    "CheckpointOption",
    "DeviceOption",
    "EncoderOption",
    "GroupColumn",
    "HybridLongTextOption",
    "HybridPolarityOption",
    "HybridReductionOption",
    "HybridTargetLevelOption",
    "HybridWriterPoolingOption",
    "LongTextOption",
    "MaxTokensOption",
    "PolarityOption",
    "PoolingOption",
    "ReductionOption",
    "ReferencesEmbeddingsOption",
    "RevisionOption",
    "RubricOption",
    "TargetLevelOption",
    "TextColumn",
    "TextIdColumn",
    "TextsOption",
    "WriterPoolingOption",
    "library_default",
    "make_encoder",
    "read_table",
    "scoring_settings",
]

AXIS_OPTIONS = ("polarity", "reduction", "writer_pooling", "long_text", "target_level")
SIMILARITY_DEFAULTS = SimilaritySettings()


def library_default(function: Callable[..., Any], name: str) -> Any:
    """Return the default of a parameter of a library function, for the option of a command."""
    return inspect.signature(function).parameters[name].default


BATCH_SIZE = library_default(Encoder.__init__, "batch_size")
CHECKPOINT_EVERY = library_default(Encoder.embed, "checkpoint_every")
GRANULARITY = library_default(score_hybrid, "granularity")
RUN_NUMBER = library_default(score_hybrid, "run_number")
MIN_WORDS = library_default(score_hybrid, "min_words")


def axis_help(axis: str, allowed: tuple[str, ...], defaults: SimilaritySettings) -> str:
    """Return the help of a setting option: its allowed values and the coder's default."""
    return f"One of {', '.join(allowed)}; default {getattr(defaults, axis)}."


TextsOption = Annotated[
    Path, typer.Option("--texts", exists=True, dir_okay=False, help="CSV, parquet, or JSONL file.")
]
RubricOption = Annotated[
    Path, typer.Option("--rubric", exists=True, dir_okay=False, help="Rubric JSON file.")
]
EncoderOption = Annotated[
    str,
    typer.Option(
        "--encoder",
        help="Encoder: a registry name (minilm, mpnet, bge_m3, gte_qwen2_7b) or a Hugging Face id.",
    ),
]
PoolingOption = Annotated[
    str | None,
    typer.Option(
        "--pooling",
        help=f"Pooling of a Hugging Face id that is not in the registry: one of "
        f"{', '.join(POOLING_NAMES)}; mean, with a warning, if omitted.",
    ),
]
MaxTokensOption = Annotated[
    int | None,
    typer.Option(
        "--max-tokens",
        help="Window of a Hugging Face id that is not in the registry; 512, with a warning, "
        "if omitted.",
    ),
]
RevisionOption = Annotated[
    str | None,
    typer.Option(
        "--revision",
        help="Commit, branch or tag of the model repository to load; the default branch if "
        "omitted. An encoder that runs the repository's own code needs the 40-character "
        "commit hash.",
    ),
]
DeviceOption = Annotated[
    str | None, typer.Option("--device", help="Torch device; chosen automatically if omitted.")
]
BatchSizeOption = Annotated[
    int, typer.Option("--batch-size", help="Chunks per forward pass of the encoder.")
]
CheckpointOption = Annotated[
    Path | None,
    typer.Option("--checkpoint", help="Stem of a checkpoint file that makes the run resumable."),
]
CheckpointEveryOption = Annotated[
    int, typer.Option("--checkpoint-every", help="Chunks between two checkpoint writes.")
]
TextIdColumn = Annotated[str, typer.Option("--text-id-column", help="Column holding the text ids.")]
TextColumn = Annotated[str, typer.Option("--text-column", help="Column holding the texts.")]
GroupColumn = Annotated[
    str | None, typer.Option("--group-column", help="Column holding the group ids.")
]
PolarityOption = Annotated[
    str | None,
    typer.Option("--polarity", help=axis_help("polarity", POLARITIES, SIMILARITY_DEFAULTS)),
]
ReductionOption = Annotated[
    str | None,
    typer.Option("--reduction", help=axis_help("reduction", REDUCTIONS, SIMILARITY_DEFAULTS)),
]
WriterPoolingOption = Annotated[
    str | None,
    typer.Option(
        "--writer-pooling",
        help=axis_help("writer_pooling", WRITER_POOLINGS, SIMILARITY_DEFAULTS),
    ),
]
LongTextOption = Annotated[
    str | None,
    typer.Option("--long-text", help=axis_help("long_text", LONG_TEXT, SIMILARITY_DEFAULTS)),
]
TargetLevelOption = Annotated[
    str | None,
    typer.Option(
        "--target-level", help=axis_help("target_level", TARGET_LEVELS, SIMILARITY_DEFAULTS)
    ),
]
HybridPolarityOption = Annotated[
    str | None,
    typer.Option("--polarity", help=axis_help("polarity", POLARITIES, HYBRID_DEFAULTS)),
]
HybridReductionOption = Annotated[
    str | None,
    typer.Option("--reduction", help=axis_help("reduction", REDUCTIONS, HYBRID_DEFAULTS)),
]
HybridWriterPoolingOption = Annotated[
    str | None,
    typer.Option(
        "--writer-pooling", help=axis_help("writer_pooling", WRITER_POOLINGS, HYBRID_DEFAULTS)
    ),
]
HybridLongTextOption = Annotated[
    str | None,
    typer.Option("--long-text", help=axis_help("long_text", LONG_TEXT, HYBRID_DEFAULTS)),
]
HybridTargetLevelOption = Annotated[
    str | None,
    typer.Option("--target-level", help=axis_help("target_level", TARGET_LEVELS, HYBRID_DEFAULTS)),
]
ReferencesEmbeddingsOption = Annotated[
    Path, typer.Option("--references-embeddings", help="Stem of the reference embeddings.")
]


def read_table(
    path: Path, text_id_column: str, text_column: str, group_column: str | None
) -> TextTable:
    """Read the text table the columns options describe."""
    columns = {"text_id": text_id_column, "text": text_column}
    if group_column is not None:
        columns["group_id"] = group_column
    return TextTable.read(path, **columns)


def scoring_settings(
    base: SimilaritySettings,
    polarity: str | None,
    reduction: str | None,
    writer_pooling: str | None,
    long_text: str | None,
    target_level: str | None,
) -> SimilaritySettings:
    """Build the similarity settings; an option left out keeps the value of ``base``."""
    given = {
        name: value
        for name, value in zip(
            AXIS_OPTIONS,
            (polarity, reduction, writer_pooling, long_text, target_level),
            strict=True,
        )
        if value is not None
    }
    return replace(base, **given)


def make_encoder(
    name: str,
    pooling: str | None,
    max_tokens: int | None,
    device: str | None,
    batch_size: int,
    revision: str | None,
) -> Encoder:
    """Build the encoder the options describe, without loading its model.

    Args:
        name: A registry name or a Hugging Face id.
        pooling: Pooling of an id that is not in the registry.
        max_tokens: Window of an id that is not in the registry.
        device: Torch device, or ``None`` to choose one.
        batch_size: Chunks per forward pass.
        revision: Revision of the model repository to load in place of the specification's
            own; ``None`` keeps it.

    Returns:
        The encoder.

    Raises:
        SettingsError: If the pooling or the window contradicts a registered encoder, the
            batch size is below 1, the revision is blank, or the encoder runs remote code and
            the revision is not a full commit hash.
    """
    if revision is not None and not revision.strip():
        raise SettingsError(
            "--revision needs a commit, branch or tag; leave it out to use the registered revision"
        )
    spec = resolve_encoder(name, pooling=pooling, max_tokens=max_tokens)
    if revision is not None:
        spec = replace(spec, revision=revision)
    return Encoder(spec, device=device, batch_size=batch_size)
