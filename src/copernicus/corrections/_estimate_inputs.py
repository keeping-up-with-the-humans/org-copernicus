"""Checks of the inputs that the estimate-level estimators share.

Every estimator reads its columns through these helpers, so the refusals read the same
everywhere: an unknown or non-numeric column, a prediction or grouping that does not cover every
text, a labelled sample without inclusion probabilities, too few labelled rows.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from copernicus.corrections._messages import list_first_ten
from copernicus.corrections.data import CorrectionData, numeric_values
from copernicus.corrections.families.pseudo import inclusion_probabilities
from copernicus.errors import ConfigurationError, EstimatorError
from copernicus.validity.metrics import is_constant

logger = logging.getLogger(__name__)

__all__ = [
    "complete_column",
    "equal_probability_sample",
    "group_mask",
    "outcome_values",
    "prediction_values",
    "require_labelled",
    "score_values",
]


def score_values(data: CorrectionData, score_column: str | None) -> tuple[np.ndarray, str]:
    """The values of a score column; ``None`` means the first score column.

    The first column is a fixed choice that never looks at the gold values; choosing the column
    on gold is left to the caller.

    Args:
        data: The correction data.
        score_column: A score column of the data, or ``None``.

    Returns:
        The values in data order and the column's name.

    Raises:
        ConfigurationError: If the column is not a score column of the data.
    """
    name = data.score_columns[0] if score_column is None else score_column
    if name not in data.score_columns:
        raise ConfigurationError(
            f"score column {name!r} is missing from the data; score columns: "
            f"{list_first_ten(list(data.score_columns))}"
        )
    return data.frame[name].to_numpy(dtype=float), name


def complete_column(data: CorrectionData, name: str, role: str) -> np.ndarray:
    """A numeric column of the data's frame that is finite on every text.

    Args:
        data: The correction data.
        name: The column's name.
        role: How the column is named in the messages, such as ``outcome``.

    Returns:
        The values as floats in data order.

    Raises:
        ConfigurationError: If the column is absent, not numeric, or not finite on a text (the
            message names the first ten such texts).
    """
    if name not in data.frame.columns:
        raise ConfigurationError(
            f"{role} {name!r} is not a column of the data; columns: "
            f"{list_first_ten([str(column) for column in data.frame.columns])}"
        )
    values = numeric_values(data.frame[name], f"{role} column {name!r}")
    bad = ~np.isfinite(values)
    if bad.any():
        raise ConfigurationError(
            f"{role} column {name!r} must be finite on every text; non-finite for text ids "
            f"{list_first_ten(data.ids[bad].tolist())}"
        )
    return values


def outcome_values(data: CorrectionData, outcome: str) -> np.ndarray:
    """The outcome column: numeric and finite on every text (see ``complete_column``)."""
    return complete_column(data, outcome, "outcome")


def _aligned(data: CorrectionData, series: pd.Series, name: str) -> pd.Series:
    """A series indexed by text id, put in data order; ids are compared as strings."""
    index = series.index.astype(str)
    repeated = index[index.duplicated()].unique().tolist()
    if repeated:
        raise ConfigurationError(f"{name} repeats text ids {list_first_ten(repeated)}")
    absent = data.ids[~pd.Index(data.ids).isin(index)]
    if absent.size:
        raise ConfigurationError(f"{name} lacks text ids {list_first_ten(absent.tolist())}")
    return pd.Series(series.to_numpy(), index=index).reindex(pd.Index(data.ids))


def prediction_values(data: CorrectionData, prediction: pd.Series | None) -> tuple[np.ndarray, str]:
    """The prediction of every text: a series by text id, or the first score column.

    Args:
        data: The correction data.
        prediction: Values indexed by text id (further ids are ignored), or ``None`` for the
            first score column.

    Returns:
        The values in data order and a name for them: the column's name, the series' name, or
        ``"prediction"`` for an unnamed series.

    Raises:
        ConfigurationError: If ``prediction`` is not a series, is not numeric, repeats an id,
            lacks a text of the data, or is not finite on a text.
    """
    if prediction is None:
        return score_values(data, None)
    if not isinstance(prediction, pd.Series):
        raise ConfigurationError(
            "prediction must be a pandas Series indexed by text id, or None for the first score "
            f"column; got {type(prediction).__name__}"
        )
    numeric = pd.Series(numeric_values(prediction, "prediction"), index=prediction.index)
    values = _aligned(data, numeric, "prediction").to_numpy(dtype=float)
    bad = ~np.isfinite(values)
    if bad.any():
        raise ConfigurationError(
            "prediction must be finite on every text; non-finite for text ids "
            f"{list_first_ten(data.ids[bad].tolist())}"
        )
    return values, "prediction" if prediction.name is None else str(prediction.name)


def group_mask(data: CorrectionData, group: pd.Series | np.ndarray) -> np.ndarray:
    """A two-level grouping of the texts as a boolean array in data order.

    Args:
        data: The correction data.
        group: A boolean array with one entry per text in data order, or a boolean series
            indexed by text id (ids compared as strings; further ids are ignored).

    Returns:
        The boolean mask, True for the texts of the group.

    Raises:
        ConfigurationError: If the array is not boolean or not one entry per text, or the series
            is not boolean, has a missing value, repeats an id or lacks a text of the data.
    """
    if isinstance(group, pd.Series):
        if not pd.api.types.is_bool_dtype(group) or bool(group.isna().any()):
            raise ConfigurationError(
                f"a group series must be boolean without missing values; got dtype {group.dtype}"
            )
        return _aligned(data, group, "group").to_numpy(dtype=bool)
    mask = np.asarray(group)
    if mask.dtype != np.bool_ or mask.shape != (data.n,):
        raise ConfigurationError(
            "group must be a boolean array with one entry per text in data order, or a boolean "
            f"series indexed by text id; got dtype {mask.dtype} and shape {mask.shape} for "
            f"{data.n} texts"
        )
    return mask


def require_labelled(count: int, minimum: int, method: str, where: str = "") -> None:
    """Refuse an estimate that has fewer labelled rows than it needs.

    Args:
        count: The number of labelled rows available.
        minimum: The number needed.
        method: The estimator, for the message.
        where: Where the rows are counted, such as ``" in each group"``.

    Raises:
        EstimatorError: If ``count`` is below ``minimum``.
    """
    if count < minimum:
        raise EstimatorError(
            f"{method} needs at least {minimum} labelled rows{where}; there are {count}"
        )


def equal_probability_sample(data: CorrectionData, method: str) -> np.ndarray:
    """The labelled rows of a human sample in which every text had the same chance.

    The prediction-powered formulas weight every labelled text alike. That is right for the
    group draw of ``draw_human_sample``, which gives every text the inclusion probability
    ``m / G``, and wrong for a sample whose texts had different chances, which ``pseudo_values``
    handles by dividing by each text's ``pi``.

    Args:
        data: The correction data.
        method: The estimator, for the message.

    Returns:
        The boolean mask of the labelled rows.

    Raises:
        EstimatorError: If a labelled row has no finite ``pi`` (the message names ``pi`` and
            points to ``draw_human_sample`` and ``attach_draw``), or ``pi`` differs between
            labelled rows.
    """
    labelled = data.labelled_mask
    pi = inclusion_probabilities(data, labelled, purpose="prediction-powered estimates")[labelled]
    if not is_constant(pi):
        raise EstimatorError(
            f"{method} weights every labelled text alike, so pi must be the same on every "
            f"labelled row (an equal-probability draw such as draw_human_sample's); it ranges "
            f"from {float(pi.min())!r} to {float(pi.max())!r}: use pseudo_values, which divides "
            "by each text's pi"
        )
    return labelled
