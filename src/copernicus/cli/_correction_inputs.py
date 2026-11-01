"""Options, value parsers and the data readers of the ``corrections`` commands."""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Annotated, Any

import pandas as pd
import typer

from copernicus.cli._embedding_options import library_default, read_table
from copernicus.cli._inputs import TEXT_ID, parse_pairs, read_gold
from copernicus.cli._options import parse_id_list
from copernicus.corrections import CorrectionData, SampleDraw, attach_draw, read_draw
from copernicus.corrections._messages import list_first_ten
from copernicus.corrections._whole_groups import PI_TOLERANCE
from copernicus.corrections.data import PI_COLUMN, numeric_values
from copernicus.data import read_frame
from copernicus.embeddings import EmbeddingSet
from copernicus.errors import ConfigurationError, SettingsError

logger = logging.getLogger(__name__)

__all__ = [
    "COVARIATE",
    "FAMILY_FORM",
    "INTEGER",
    "CovariateOption",
    "GoldColumnOption",
    "ScoreColumnsOption",
    "ScoresOption",
    "bind_to_draw",
    "load_correction_data",
    "parse_counts",
    "parse_families",
    "parse_number",
    "parse_sizes",
]

COVARIATE = library_default(CorrectionData.from_scores, "covariate")
NOT_CARRIED = (TEXT_ID, "text", "group_id", "pi")
FAMILY_FORM = "FAMILY:VARIANT or FAMILY:VARIANT:NAME=VALUE"
INTEGER = re.compile(r"[+-]?[0-9]+")
DECIMAL = re.compile(r"[+-]?([0-9]+\.?[0-9]*|\.[0-9]+)([eE][+-]?[0-9]+)?")

ScoresOption = Annotated[
    Path,
    typer.Option(
        "--scores",
        exists=True,
        dir_okay=False,
        help="CSV or parquet file with text_id and the score columns, one row per text.",
    ),
]
GoldColumnOption = Annotated[
    str, typer.Option("--gold-column", help="Numeric gold column of the gold file.")
]
ScoreColumnsOption = Annotated[
    str,
    typer.Option("--score-columns", help="Score columns of the scores file, comma-separated."),
]
CovariateOption = Annotated[
    str,
    typer.Option(
        "--covariate",
        help=f"'{COVARIATE}' (the log of the character count) or a numeric column of the texts "
        f"or gold file; default {COVARIATE}.",
    ),
]


def parse_number(text: str) -> int | float | None:
    """Read an integer or a decimal number; ``None`` for anything else.

    Args:
        text: The text of an option value.

    Returns:
        An ``int`` for an integer such as ``50``, a ``float`` for a decimal such as ``0.05``
        or ``5e-2``, and ``None`` for other text.
    """
    if INTEGER.fullmatch(text):
        return int(text)
    if DECIMAL.fullmatch(text):
        return float(text)
    return None


def parse_sizes(value: str) -> tuple[int | float, ...]:
    """Parse ``50,100,0.05`` into sizes: integers are counts of texts, decimals are shares.

    Raises:
        SettingsError: If a part is not a number.
    """
    sizes: list[int | float] = []
    for part in (piece.strip() for piece in value.split(",")):
        number = parse_number(part)
        if number is None:
            raise SettingsError(
                f"--sizes expects integers and shares, comma-separated; got {part!r}"
            )
        sizes.append(number)
    return tuple(sizes)


def _param_value(text: str) -> int | float | str:
    """A parameter value: the number it reads as, or the text itself."""
    number = parse_number(text)
    return text if number is None else number


def parse_families(values: Sequence[str]) -> list[tuple[Any, ...]]:
    """Parse ``--families`` values into battery entries.

    ``FAMILY:VARIANT`` gives a ``(family, variant)`` entry and every further ``:NAME=VALUE``
    a keyword parameter of the family's factory; a value that reads as a number is passed as
    one (``m_draws=50``), any other as text (``encoder=minilm``).

    Raises:
        SettingsError: If a value lacks the family or the variant, or a parameter is not of the
            form ``NAME=VALUE`` or is given twice.
    """
    entries: list[tuple[Any, ...]] = []
    for value in values:
        family, _, rest = value.partition(":")
        variant, _, params = rest.partition(":")
        if not family.strip() or not variant.strip():
            raise SettingsError(f"--families expects {FAMILY_FORM}, got {value!r}")
        pairs = parse_pairs(params.split(":") if params else [], "--families", FAMILY_FORM)
        typed = {name: _param_value(text) for name, text in pairs}
        entries.append(
            (family.strip(), variant.strip(), typed) if typed else (family.strip(), variant.strip())
        )
    return entries


def parse_counts(value: str, option: str) -> list[int]:
    """Parse a comma-separated list of integers."""
    parts = [piece.strip() for piece in value.split(",")]
    if not all(INTEGER.fullmatch(part) for part in parts):
        raise SettingsError(f"{option} expects integers separated by commas, got {value!r}")
    return [int(part) for part in parts]


def load_correction_data(
    paths: Mapping[str, Path],
    gold_column: str,
    score_columns: str,
    covariate: str,
    text_columns: tuple[str, str, str | None],
    *,
    embeddings: Mapping[str, EmbeddingSet] | None = None,
    pi_column: str | None = None,
    carry_table_columns: bool = False,
) -> CorrectionData:
    """Read the scores, texts and gold files and join them into correction data.

    Args:
        paths: The ``scores``, ``texts`` and ``gold`` files.
        gold_column: The gold column of the gold file.
        score_columns: The score columns, comma-separated.
        covariate: ``log_chars`` or a numeric column of the texts or gold file.
        text_columns: The text id, text and group columns of the texts file.
        embeddings: Embedding sets by encoder name.
        pi_column: The column of the gold file with the inclusion probability.
        carry_table_columns: Whether to carry every further column of the texts file (such as
            ``n_words`` or an outcome) into the data, where formulas and outcomes read them.

    Returns:
        The correction data.

    Raises:
        SettingsError: If ``score_columns`` names no column.
    """
    names = parse_id_list(score_columns)
    if not names:
        raise SettingsError("--score-columns must name at least one score column")
    table = read_table(paths["texts"], *text_columns)
    roles = {*NOT_CARRIED, *names, gold_column, covariate}
    extras = [str(name) for name in table.frame.columns if str(name) not in roles]
    logger.info("joining %d score columns of %s", len(names), paths["scores"])
    return CorrectionData.from_scores(
        read_frame(paths["scores"], protected_columns=(TEXT_ID,)),
        table,
        read_gold(paths["gold"]),
        gold_column=gold_column,
        score_columns=names,
        covariate=covariate,
        embeddings=embeddings,
        pi_column=pi_column,
        extra_columns=extras if carry_table_columns else (),
    )


def bind_to_draw(
    data: CorrectionData, gold_path: Path, draw_path: Path, pi_column: str | None
) -> CorrectionData:
    """Bind a coded sample to its texts file through the draw file of ``corrections sample``.

    The data pass every check of ``attach_draw``: the fingerprint of the texts and their
    groups first, then the drawn ids and groups, then a finite gold value for every drawn
    text. Then the inclusion probabilities of the gold file, in ``pi_column`` or, without it,
    in a column named ``pi``, must equal the draw's ``pi`` on every drawn text. Only the drawn
    texts belong to the sample, so the gold values of other coded texts are set aside; one
    INFO line counts them.

    Args:
        data: The correction data of the files, built without ``pi``.
        gold_path: The gold file.
        draw_path: The draw file.
        pi_column: The ``--pi-column``, or ``None``.

    Returns:
        The data with the gold values of the drawn texts only and the draw's ``pi`` on every
        row.

    Raises:
        ConfigurationError: If the draw file is refused (see ``read_draw``), the data are not
            the draw's table or do not code the draw (see ``attach_draw``), or the gold file's
            inclusion probabilities differ from the draw's ``pi``.
    """
    draw = read_draw(draw_path)
    bound = attach_draw(data, pd.Series(data.gold, index=data.ids), draw)
    gold = read_gold(gold_path)
    if pi_column is not None:
        _check_gold_pi(gold, pi_column, draw)
    elif PI_COLUMN in gold.columns:
        _check_gold_pi(gold, PI_COLUMN, draw)
    # Every drawn text has a gold value here, so the other coded texts are the difference.
    logger.info(
        "bound the %d drawn texts to the draw in %s; coded texts outside the draw, set aside: %d",
        bound.n_labelled,
        draw_path,
        data.n_labelled - bound.n_labelled,
    )
    return bound


def _check_gold_pi(gold: pd.DataFrame, name: str, draw: SampleDraw) -> None:
    """Refuse inclusion probabilities of the gold file that are not the draw's ``pi``."""
    if name not in gold.columns:
        raise ConfigurationError(f"--pi-column {name!r} is not a column of the gold file")
    values = pd.Series(
        numeric_values(gold[name], f"column {name!r} of the gold file"),
        index=gold[TEXT_ID].astype(str),
    )
    drawn = values.reindex([str(text_id) for text_id in draw.ids])
    differ = [
        text_id
        for text_id, value in drawn.items()
        if not math.isclose(value, draw.pi, rel_tol=PI_TOLERANCE)
    ]
    if differ:
        raise ConfigurationError(
            f"column {name!r} of the gold file gives the drawn texts {list_first_ten(differ)} "
            f"an inclusion probability other than the draw's pi = {draw.pi!r}: code the drawn "
            "texts of this draw file, or give the gold file of this draw"
        )
