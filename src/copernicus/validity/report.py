"""A plain-text page summarising a validity panel."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from copernicus.data.table import ONE_GROUP_PER_TEXT

if TYPE_CHECKING:
    from copernicus.validity.panel import ValidityPanel

logger = logging.getLogger(__name__)

__all__ = ["render_report"]

NOT_AVAILABLE = "n/a"
HEADERS = (
    "target",
    "n",
    "rank validity [95% CI]",
    "MAE",
    "level",
    "calibr.",
    "cov.",
    "noise",
    "share of ceiling",
)
RANK_VALIDITY_NOTE = (
    "Rank validity is an absolute value; its interval is bounded below by 0 and does not test "
    "zero validity."
)


def _number(value: float, digits: int = 2) -> str:
    return NOT_AVAILABLE if not np.isfinite(value) else f"{value:.{digits}f}"


def _rank_validity(row: pd.Series) -> str:
    if not np.isfinite(row["rank_validity"]):
        return NOT_AVAILABLE
    low, high = _number(row["rank_validity_low"]), _number(row["rank_validity_high"])
    return f"{_number(row['rank_validity'])} [{low}, {high}]"


def _share(row: pd.Series) -> str:
    return (
        f"{_number(row['share_of_ceiling'])} of {_number(row['ceiling'])}"
        if np.isfinite(row["ceiling"])
        else NOT_AVAILABLE
    )


def _table_rows(frame: pd.DataFrame) -> list[tuple[str, ...]]:
    rows: list[tuple[str, ...]] = []
    for _, row in frame.iterrows():
        rows.append(
            (
                str(row["target"]),
                str(row["n"]),
                _rank_validity(row),
                _number(row["mae_01"]),
                _number(row["level_displacement"]),
                _number(row["calibration_shortfall"]),
                _number(row["covariate_linked_error"]),
                _number(row["noise"]),
                _share(row),
            )
        )
    return rows


def _fixed_width(rows: list[tuple[str, ...]]) -> list[str]:
    widths = [max(len(line[column]) for line in [HEADERS, *rows]) for column in range(len(HEADERS))]
    lines = []
    for line in [HEADERS, *rows]:
        cells = [
            cell.ljust(width) if column == 0 else cell.rjust(width)
            for column, (cell, width) in enumerate(zip(line, widths, strict=True))
        ]
        lines.append("  ".join(cells).rstrip())
    lines.insert(1, "  ".join("-" * width for width in widths))
    return lines


def _names(targets: pd.Series) -> str:
    return ", ".join(str(target) for target in targets)


def _noise_clause(flagged: pd.DataFrame) -> str:
    """Say for which flagged targets the noise is still reported, or nothing when for none."""
    with_noise = flagged.loc[np.isfinite(flagged["noise"]), "target"]
    if with_noise.empty:
        return ""
    if len(with_noise) == len(flagged):
        return "; the noise is still reported"
    return f"; the noise is still reported for {_names(with_noise)}"


def _constant_score_sentences(flagged: pd.DataFrame) -> str:
    """Describe the errors of the constant-score targets: they exist only against a varying gold."""
    constant = flagged[flagged["status"] == "constant_score"]
    kept = constant.loc[np.isfinite(constant["calibration_shortfall"]), "target"]
    neither = constant.loc[~np.isfinite(constant["calibration_shortfall"]), "target"]
    text = ""
    if not kept.empty:
        text += (
            f" For {_names(kept)}, a constant score still has two errors: a calibration "
            "shortfall of -1 (it does not move with gold) and a covariate-linked error of "
            "minus the slope of gold on the covariate."
        )
    if not neither.empty:
        verb = "has" if len(neither) == 1 else "have"
        text += f" {_names(neither)}, whose gold is constant too, {verb} neither error."
    return text


def _status_paragraph(frame: pd.DataFrame) -> str:
    flagged = frame[frame["status"] != "ok"]
    if flagged.empty:
        return "Every target was scored."
    named = ", ".join(f"{row['target']} ({row['status']})" for _, row in flagged.iterrows())
    paragraph = (
        f"Not scored: {named}. For these targets the association metrics, the score errors "
        f"and the intervals are missing{_noise_clause(flagged)}."
    )
    emptied = flagged.loc[flagged["n_used"] == 0, "target"]
    if not emptied.empty:
        paragraph += f" No text of {_names(emptied)} kept a score, a gold value and a covariate."
    return paragraph + _constant_score_sentences(flagged)


def _direction_lines(frame: pd.DataFrame) -> list[str]:
    """Name the targets whose score was negated, or nothing when none was."""
    negated = frame.loc[frame["direction_applied"] == "negative", "target"]
    if negated.empty:
        return []
    named = ", ".join(str(target) for target in negated)
    return [f"Score negated before every metric (direction negative): {named}."]


def _sign_warnings(frame: pd.DataFrame) -> list[str]:
    """Warn about every target whose score, used as given, falls as the gold rises."""
    falling = frame[(frame["direction_applied"] == "positive") & (frame["spearman"] < 0)]
    return [
        f"Warning: the score falls as the gold rises for {row['target']} (signed Spearman "
        f"{row['spearman']:.2f}); rank validity does not show the sign, so check the direction "
        "of the score (direction negative or auto)."
        for _, row in falling.iterrows()
    ]


def render_report(panel: ValidityPanel) -> str:
    """Render the panel as a plain-text page.

    The page has a header line with the number of texts that entered and the number
    dropped; a line on the intervals and the covariate, which adds ``groups: one per text
    (no group column)`` when every text was its own cluster of the bootstrap; a fixed-width
    table with one row per target (number of texts, rank validity with its interval, mean
    absolute error on the 0-1 scale, level displacement, calibration shortfall,
    covariate-linked error, noise and the share of the reliability ceiling); a note under
    the table that rank validity is an absolute value, whose interval is bounded below by 0
    and does not test zero validity; a paragraph naming the targets whose status is not
    ``ok``; a line naming the targets whose score was negated before every metric
    (``direction_applied`` negative), when there are any; and a warning line for every
    target whose score was used as given (``direction_applied`` positive) although its
    signed Spearman correlation is negative. The paragraph says that the noise is still
    reported for the flagged targets that have one (not for a target that lost every text),
    and that a target with a constant score still has its calibration shortfall of -1 and
    its covariate-linked error, unless the gold is constant too, in which case it has
    neither.

    Args:
        panel: The computed panel.

    Returns:
        The report text.
    """
    frame = panel.frame
    settings = panel.settings
    covariate = frame["covariate"].iloc[0]
    header = (
        f"Validity panel: {panel.n_used} texts entered, {panel.n_dropped_missing} dropped for a "
        f"missing score, gold value or covariate."
    )
    grouping = f"; {ONE_GROUP_PER_TEXT}" if panel.groups_defaulted else ""
    basis = (
        f"Intervals: 95 percent, group-clustered bootstrap, {settings.n_boot} replicates, "
        f"seed {settings.seed}{grouping}. Covariate: {covariate} ({settings.covariate_scale})."
    )
    table = _fixed_width(_table_rows(frame))
    status = _status_paragraph(frame)
    notes = [*_direction_lines(frame), *_sign_warnings(frame)]
    return "\n".join([header, basis, "", *table, RANK_VALIDITY_NOTE, "", status, *notes, ""])
