"""Synthetic data for the tests of the estimate-level estimators, and how they compare pins.

Every builder draws from the generator it is given in a fixed order, so a test that seeds the
generator gets the same texts and the same labelled sample on every run.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd
import pytest

from copernicus.corrections._simex import LAMBDAS, rational_extrapolate
from copernicus.corrections.data import CorrectionData
from copernicus.corrections.estimates_oriv import WEAK_INSTRUMENT_FLOOR, stacked_2sls

TRUE_MEAN = 1.0
TRUE_DIFFERENCE = 0.0
TRUE_SLOPE = 0.9
CLASSICAL_SLOPE = 1.0
SPARSE_TEXTS = 300
SPARSE_LABELLED = 20
# The bootstrap of the floor tests: 60 resamples, as the paper's analysis code drew for SIMEX,
# and a seed at which moment subtraction, SIMEX and ORIV each have a defined estimate on 24 of
# them on sparse_covariation_data (fewer than half, and at least 10).
SPARSE_BOOT = 60
SPARSE_SEED = 2
# The defaults of simex: noise replicates for the point and inside each resample.
SIMEX_REPS, SIMEX_BOOT_REPS = 25, 10
FLOOR_ESTIMATORS = ("moment_subtraction", "simex", "oriv")
# Platforms differ in the last digits of floating-point calculations, linear algebra among them,
# so the tests of the corrections compare a value pinned with all its digits to twelve
# significant digits.
RELATIVE_TOLERANCE = 1e-12
# A row of a repeated study as the tests pin it: (estimate_mean, coverage, se_mean, n_undefined).
StudyRow = tuple[float, float, float, int]


def same_study_row(found: StudyRow, pinned: StudyRow) -> bool:
    """Tell whether a row of a repeated study equals its pin.

    The point and the standard error are compared to twelve significant digits, the coverage (a
    share of the draws) and the number of undefined draws exactly.
    """
    point, coverage, se, undefined = found
    pinned_point, pinned_coverage, pinned_se, pinned_undefined = pinned
    close = (point, se) == pytest.approx((pinned_point, pinned_se), rel=RELATIVE_TOLERANCE, abs=0.0)
    return bool(close) and (coverage, undefined) == (pinned_coverage, pinned_undefined)


def text_ids(n: int) -> list[str]:
    """Text ids ``t00000``, ``t00001``, ..."""
    return [f"t{index:05d}" for index in range(n)]


def labels_by_srs(rng: np.random.Generator, n_texts: int, n_labelled: int) -> np.ndarray:
    """A simple random sample of texts as a boolean mask."""
    labelled = np.zeros(n_texts, dtype=bool)
    labelled[rng.choice(n_texts, size=n_labelled, replace=False)] = True
    return labelled


def with_labels(
    frame: pd.DataFrame,
    labelled: np.ndarray,
    *,
    score_columns: Sequence[str],
    pi: float | None,
) -> CorrectionData:
    """Correction data that keep the gold value on the labelled rows only.

    Args:
        frame: Texts with a complete ``gold`` column.
        labelled: Boolean mask of the human-coded texts.
        score_columns: The score columns.
        pi: The inclusion probability written on every row, or ``None`` for no ``pi`` column.

    Returns:
        The correction data.
    """
    coded = frame.copy()
    coded["gold"] = np.where(labelled, coded["gold"].to_numpy(dtype=float), np.nan)
    if pi is not None:
        coded["pi"] = pi
    return CorrectionData(frame=coded, score_columns=tuple(score_columns))


def ppi_frame(rng: np.random.Generator, n_texts: int) -> pd.DataFrame:
    """Texts whose score is the gold value shifted by 0.5 plus an error that grows with length.

    The gold value is ``1 + 0.5 * N(0, 1)`` and independent of the covariate, so the true mean is
    1 and the true long-minus-short difference is 0, while the raw score's difference is about
    ``0.3 * 1.6``. Every text is its own group, so a group draw is a simple random sample.
    """
    covariate = rng.normal(size=n_texts)
    gold = TRUE_MEAN + 0.5 * rng.normal(size=n_texts)
    score = gold + 0.5 + 0.3 * covariate + rng.normal(scale=0.1, size=n_texts)
    ids = text_ids(n_texts)
    return pd.DataFrame(
        {"text_id": ids, "group_id": ids, "score": score, "gold": gold, "log_chars": covariate}
    )


def slope_frame(rng: np.random.Generator, n_texts: int) -> pd.DataFrame:
    """Texts for the construct as predictor: ``y = 0.5 + 0.9 V + 0.4 W + e``.

    ``V`` is the construct (the gold value), ``W`` the covariate ``log_chars`` and the score
    ``0.2 + 0.8 V + 0.3 W + 0.4 u`` measures ``V`` with an error that depends on ``W``.
    """
    covariate = rng.normal(size=n_texts)
    construct = rng.normal(size=n_texts)
    outcome = 0.5 + TRUE_SLOPE * construct + 0.4 * covariate + 0.8 * rng.normal(size=n_texts)
    score = 0.2 + 0.8 * construct + 0.3 * covariate + 0.4 * rng.normal(size=n_texts)
    ids = text_ids(n_texts)
    return pd.DataFrame(
        {
            "text_id": ids,
            "group_id": ids,
            "score": score,
            "gold": construct,
            "outcome": outcome,
            "log_chars": covariate,
        }
    )


def classical_data(
    seed: int,
    *,
    n_texts: int = 2000,
    n_labelled: int = 400,
    error_a: float = 0.4,
    error_b: float = 0.6,
    shared: float = 0.0,
) -> CorrectionData:
    """Two measurements of the construct with classical errors and an outcome with slope 1.

    ``score_a = 0.5 + 0.8 V + error_a * u_a`` and ``score_b = -0.2 + 0.6 V + error_b * u_b``,
    plus ``shared`` times a common error term in both when ``shared`` is not zero; the outcome
    is ``V + 0.5 e``. The labelled texts are a simple random sample with ``pi`` on every row.
    """
    rng = np.random.default_rng(seed)
    construct = rng.normal(size=n_texts)
    outcome = CLASSICAL_SLOPE * construct + 0.5 * rng.normal(size=n_texts)
    score_a = 0.5 + 0.8 * construct + error_a * rng.normal(size=n_texts)
    score_b = -0.2 + 0.6 * construct + error_b * rng.normal(size=n_texts)
    labelled = labels_by_srs(rng, n_texts, n_labelled)
    if shared:
        common = rng.normal(size=n_texts)
        score_a = score_a + shared * common
        score_b = score_b + shared * common
    ids = text_ids(n_texts)
    frame = pd.DataFrame(
        {
            "text_id": ids,
            "group_id": ids,
            "score_a": score_a,
            "score_b": score_b,
            "gold": construct,
            "outcome": outcome,
            "log_chars": rng.normal(size=n_texts),
        }
    )
    return with_labels(
        frame, labelled, score_columns=("score_a", "score_b"), pi=n_labelled / n_texts
    )


def sparse_covariation_data(seed: int) -> CorrectionData:
    """Data on which most stratified resamples leave ``score_a`` without a reverse calibration.

    The texts of ``classical_data`` (300 texts, 20 labelled), with the labelled texts made flat:
    every labelled gold value is 0 but that of the second labelled text in data order, which
    is 1, and every labelled ``score_a`` is 0 but that of the first, which is 1. The line of
    ``score_a`` on the gold values over the labelled texts needs both texts. The data hold both,
    so the point is defined; a resample of the labelled texts draws both with the probability
    ``1 - 2 (1 - 1/20)^20 + (1 - 2/20)^20``, about 0.40, and in every other resample the gold
    values or the scores are constant on its labelled texts. ``score_b`` and the outcome keep
    the values of ``classical_data``.
    """
    data = classical_data(seed, n_texts=SPARSE_TEXTS, n_labelled=SPARSE_LABELLED)
    labelled = np.flatnonzero(data.labelled_mask)
    gold = np.where(data.labelled_mask, 0.0, np.nan)
    gold[labelled[1]] = 1.0
    score_a = data.frame["score_a"].to_numpy(dtype=float, copy=True)
    score_a[labelled] = 0.0
    score_a[labelled[0]] = 1.0
    frame = data.frame.assign(gold=gold, score_a=score_a)
    return CorrectionData(frame=frame, score_columns=data.score_columns)


def stratified_rows(
    rng: np.random.Generator, labelled: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """One resample drawn from the generator as the stratified bootstrap of the library draws it.

    Returns:
        The rows of the resample (its labelled block first) and its labelled mask.
    """
    labelled_rows, other_rows = np.flatnonzero(labelled), np.flatnonzero(~labelled)
    rows = np.concatenate(
        [
            rng.choice(labelled_rows, size=labelled_rows.size, replace=True),
            rng.choice(other_rows, size=other_rows.size, replace=True),
        ]
    )
    sample = np.zeros(labelled.size, dtype=bool)
    sample[: labelled_rows.size] = True
    return rows, sample


def reverse_proxy(score: np.ndarray, sample: np.ndarray, gold: np.ndarray) -> np.ndarray | None:
    """``(score - c) / d`` from the line ``score = c + d gold`` over the sample; ``None`` if flat.

    Flat means constant gold values or constant scores on the sample, the only flat lines that
    ``sparse_covariation_data`` produces.
    """
    score_sample, gold_sample = score[sample], gold[sample]
    if np.ptp(gold_sample) == 0.0 or np.ptp(score_sample) == 0.0:
        return None
    slope = np.cov(score_sample, gold_sample, ddof=1)[0, 1] / np.var(gold_sample, ddof=1)
    return (score - float(score_sample.mean() - slope * gold_sample.mean())) / slope


def defined_resamples(
    value: Callable[[np.ndarray, np.ndarray], float | None],
    labelled: np.ndarray,
    rng: np.random.Generator,
    n_boot: int,
) -> list[float]:
    """The values of the resamples on which ``value`` is defined (not ``None``), in draw order.

    Args:
        value: The estimate on a resample, called with its rows and its labelled mask.
        labelled: The labelled mask of the data.
        rng: The generator, in the state the estimator's bootstrap starts from.
        n_boot: Number of resamples.

    Returns:
        The defined values.
    """
    found: list[float] = []
    for _ in range(n_boot):
        result = value(*stratified_rows(rng, labelled))
        if result is not None:
            found.append(result)
    return found


def moment_value(
    score: np.ndarray, outcome: np.ndarray, sample: np.ndarray, gold: np.ndarray
) -> float | None:
    """``cov(W, y) / (var(W) - sigma2)``; ``None`` without a line or a positive denominator."""
    proxy = reverse_proxy(score, sample, gold)
    if proxy is None:
        return None
    denominator = np.var(proxy, ddof=1) - np.var(proxy[sample] - gold[sample], ddof=1)
    return None if denominator <= 0.0 else float(np.cov(proxy, outcome, ddof=1)[0, 1] / denominator)


def simex_value(
    score: np.ndarray,
    outcome: np.ndarray,
    sample: np.ndarray,
    gold: np.ndarray,
    reps: int,
    rng: np.random.Generator,
) -> float | None:
    """The extrapolated SIMEX slope, drawing the noise from ``rng`` in the estimator's order.

    ``None`` without a line (no noise is drawn then) or when the extrapolation is undefined.
    """
    proxy = reverse_proxy(score, sample, gold)
    if proxy is None:
        return None
    sigma2 = np.var(proxy[sample] - gold[sample], ddof=1)
    centred_outcome = outcome - outcome.mean()
    grid = np.asarray(LAMBDAS)
    curve = np.empty(grid.size)
    for index, multiple in enumerate(LAMBDAS):
        slopes = []
        for _ in range(reps):
            noisy = proxy + rng.normal(0.0, float(np.sqrt(sigma2 * multiple)), size=proxy.size)
            centred = noisy - noisy.mean()
            slopes.append(float((centred @ centred_outcome) / (centred @ centred)))
        curve[index] = float(np.mean(slopes))
    fit = rational_extrapolate(grid, curve)
    return None if fit.status in ("pole_le_1", "no_fit") else fit.value


def oriv_value(
    first: np.ndarray,
    second: np.ndarray,
    outcome: np.ndarray,
    sample: np.ndarray,
    gold: np.ndarray,
) -> float | None:
    """Equation (6) on the two proxies; ``None`` without both lines or with weak instruments."""
    proxy_a, proxy_b = reverse_proxy(first, sample, gold), reverse_proxy(second, sample, gold)
    if proxy_a is None or proxy_b is None:
        return None
    if abs(np.corrcoef(proxy_a, proxy_b)[0, 1]) < WEAK_INSTRUMENT_FLOOR:
        return None
    return stacked_2sls(outcome, proxy_a, proxy_b).slope


def replay(name: str, data: CorrectionData, *, seed: int, n_boot: int) -> tuple[float, list[float]]:
    """The point and the defined resample values of an estimator, recomputed from its generator.

    The estimator reads ``score_a`` (ORIV also ``score_b``) and ``outcome`` and keeps its other
    defaults; its generator is ``default_rng(seed)``, from which SIMEX also draws its noise, for
    the point first and then inside each resample.

    Args:
        name: One of ``FLOOR_ESTIMATORS``.
        data: The correction data; the point must be defined on them.
        seed: The estimator's seed.
        n_boot: Number of resamples.

    Returns:
        The point and the values of the defined resamples, in draw order.
    """
    assert name in FLOOR_ESTIMATORS, name
    first, second = data.frame["score_a"].to_numpy(), data.frame["score_b"].to_numpy()
    outcome, labelled, gold = data.frame["outcome"].to_numpy(), data.labelled_mask, data.gold
    rng = np.random.default_rng(seed)

    def value(rows: np.ndarray, sample: np.ndarray, reps: int = SIMEX_BOOT_REPS) -> float | None:
        if name == "moment_subtraction":
            return moment_value(first[rows], outcome[rows], sample, gold[rows])
        if name == "simex":
            return simex_value(first[rows], outcome[rows], sample, gold[rows], reps, rng)
        return oriv_value(first[rows], second[rows], outcome[rows], sample, gold[rows])

    point = value(np.arange(data.n), labelled, SIMEX_REPS)
    assert point is not None, f"the point of {name} must be defined on these data"
    return point, defined_resamples(value, labelled, rng, n_boot)


def grouped_frame(
    rng: np.random.Generator, *, n_groups: int, group_size: int, level_shift: float = 0.5
) -> pd.DataFrame:
    """Texts in groups with a complete gold column, a biased score and a noise column."""
    n = n_groups * group_size
    covariate = rng.normal(size=n)
    gold = 2.0 + rng.normal(size=n)
    ids = text_ids(n)
    return pd.DataFrame(
        {
            "text_id": ids,
            "group_id": [f"g{index // group_size:04d}" for index in range(n)],
            "score": gold + level_shift + 0.3 * covariate + rng.normal(scale=0.3, size=n),
            "noise": rng.normal(size=n),
            "gold": gold,
            "truth": gold,
            "log_chars": covariate,
        }
    )


def clustered_frame(
    rng: np.random.Generator,
    sizes: Sequence[int],
    *,
    group_error: float = 0.6,
    text_error: float = 0.3,
    shift: float = 0.5,
) -> pd.DataFrame:
    """Texts in groups of the given sizes whose machine error is shared within a group.

    The gold value is ``1 + N(0, 1)`` for every text, so the true mean is 1 and the true difference
    between the groups with an even number and the other groups is 0. The score adds ``shift``,
    an error ``N(0, group_error^2)`` common to the texts of a group, and an error
    ``N(0, text_error^2)`` of each text. ``flag`` marks the groups with an even number.
    """
    counts = np.asarray(sizes, dtype=int)
    groups = np.repeat(np.arange(counts.size), counts)
    n = groups.size
    gold = TRUE_MEAN + rng.normal(size=n)
    shared = rng.normal(scale=group_error, size=counts.size)[groups]
    ids = text_ids(n)
    return pd.DataFrame(
        {
            "text_id": ids,
            "group_id": [f"g{group:04d}" for group in groups],
            "score": gold + shift + shared + text_error * rng.normal(size=n),
            "gold": gold,
            "flag": groups % 2 == 0,
            "log_chars": rng.normal(size=n),
        }
    )
