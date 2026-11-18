"""Checks of the inputs that the estimate-level estimators share, and the long-short grouping.

Every estimator reads its columns through these helpers, so the refusals read the same
everywhere: an unknown or non-numeric column, a prediction or grouping that does not cover every
text, a labelled sample without inclusion probabilities, too few labelled rows. The plug-in
value of a mean or a difference and its standard error for independent texts, which
``estimates`` exports, live here with the check of their number of values.
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
    "estimand_parts",
    "group_mask",
    "long_short_split",
    "naive_se",
    "outcome_values",
    "plugin_value",
    "prediction_values",
    "require_labelled",
    "require_labelled_sides",
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


def long_short_split(data: CorrectionData) -> np.ndarray:
    """The long side of the long-short difference: the texts at or above the covariate's median.

    With the default covariate (the log of the character count) the mask marks the longer
    texts, against the texts below the median; a text at the median counts as long, as in the
    paper's analysis code. Ties at the median can therefore make the two sides unequal, and a
    constant covariate puts every text on the long side and leaves the short side empty. The
    mask is returned as it is; an estimator of a difference refuses a side with too few texts
    (``group_difference_ppi`` needs two labelled texts on each side, ``repeated_study`` two
    texts).

    Args:
        data: The correction data.

    Returns:
        A boolean array in data order, True on the long side.
    """
    covariate = data.covariate
    return np.asarray(covariate >= np.median(covariate), dtype=bool)


def require_labelled(count: int, minimum: int, method: str) -> None:
    """Refuse an estimate that has fewer labelled rows than it needs.

    Args:
        count: The number of labelled rows available.
        minimum: The number needed.
        method: The estimator, for the message.

    Raises:
        EstimatorError: If ``count`` is below ``minimum``.
    """
    if count < minimum:
        raise EstimatorError(f"{method} needs at least {minimum} labelled rows; there are {count}")


def require_labelled_sides(counts: tuple[int, int], minimum: int, method: str) -> None:
    """Refuse a difference that has fewer labelled rows than it needs on a side of its grouping.

    Args:
        counts: The labelled rows where the grouping is True and where it is False.
        minimum: The number needed on each side.
        method: The estimator, for the message.

    Raises:
        EstimatorError: If a side holds fewer than ``minimum`` labelled rows; the message gives
            the smaller count and what to do.
    """
    count = min(counts)
    if count < minimum:
        raise EstimatorError(
            f"{method} needs at least {minimum} labelled rows on each side; there are {count} on "
            f"one side: label more texts on that side, or use a grouping that puts at least "
            f"{minimum} labelled texts on each side"
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


def estimand_parts(values: np.ndarray, split: np.ndarray | None, minimum: int) -> list[np.ndarray]:
    """The values, or the values on two sides, each with at least ``minimum`` entries."""
    parts = [values] if split is None else [values[split], values[~split]]
    sizes = [int(part.size) for part in parts]
    if min(sizes) < minimum:
        if split is None:
            raise EstimatorError(
                f"the estimand needs at least {minimum} values; there are {sizes[0]}; "
                "provide more values"
            )
        raise EstimatorError(
            f"the estimand needs at least {minimum} values on each side; the sides hold {sizes}; "
            "provide more values on the side with too few, or choose another grouping"
        )
    return parts


def plugin_value(values: np.ndarray, split: np.ndarray | None = None) -> float:
    """The mean of the values, or the mean where ``split`` is True minus the mean elsewhere.

    Args:
        values: One value per text.
        split: ``None`` for the mean, or a boolean mask of the first side.

    Returns:
        The plug-in value of the estimand.

    Raises:
        EstimatorError: If there are no values, or a side holds none.
    """
    parts = estimand_parts(values, split, 1)
    if split is None:
        return float(np.mean(parts[0]))
    return float(np.mean(parts[0]) - np.mean(parts[1]))


def naive_se(values: np.ndarray, split: np.ndarray | None = None) -> float:
    """The standard error of ``plugin_value`` when the values are independent draws.

    Args:
        values: One value per text.
        split: ``None`` for the mean, or a boolean mask of the first side.

    Returns:
        ``sd / sqrt(n)``, or ``sqrt(var_a / n_a + var_b / n_b)``, with sample moments.

    Raises:
        EstimatorError: If there are fewer than two values, or a side holds fewer than two.
    """
    parts = estimand_parts(values, split, 2)
    return float(np.sqrt(sum(np.var(part, ddof=1) / part.size for part in parts)))
