"""Prediction-powered OLS coefficients with the construct as outcome or as predictor.

Both estimators solve rectified normal equations ``H beta = c`` built from ``M = mean(x x')``
and ``c = mean(x v)`` over sets of texts, and report the influence-function (sandwich) variance

    H^-1 [Cov_base(psi_s) / N_base + Cov_lab(psi_s - psi_g) / n] H^-1

with ``psi = x (v - x'beta)`` the estimating function of a text, sample covariances (divisor
``n - 1``) and no finite-population correction. ``s`` is the prediction of the construct on
every text, ``g`` the gold value on the ``n`` labelled texts.

* ``ols_outcome_ppi``: the construct is the outcome of a regression on a design ``X`` observed
  on every text. ``X`` does not involve the construct, so the moment terms cancel to
  ``H = M_all`` and the correction acts on ``c``: ``M_all beta = c_all(s) - c_lab(s) +
  c_lab(g)``.
* ``ols_predictor_ppi``: the construct is a predictor, ``x = [1, v, W]`` with ``v = s`` on every
  text and ``v = g`` on the labelled ones, ``W`` the covariates, and an outcome ``y`` observed on
  every text: ``[M_base(s) - M_lab(s) + M_lab(g)] beta = c_base(s) - c_lab(s) + c_lab(g)``. The
  ``ppi`` variant takes every text as the base; the ``ludwig`` variant takes the texts outside
  the human sample (the primary sample of the moment correction of Ludwig, Mullainathan and
  Rambachan), which makes its two variance terms independent.

When the estimate clusters (by default, when a group holds more than one labelled text, as in
the whole-group draw of ``draw_human_sample``; see ``_clustering``), each ``Cov(psi) / n`` becomes
``Cov(S) * G / n^2`` over the group sums ``S`` of the term's rows centred at their mean: the
groups of the base texts for the base term, those of the labelled texts for the other.

A rectified moment matrix that is not positive definite leaves the estimate undefined. When every
text is labelled, the ``ppi`` estimates equal OLS on the gold values.
"""

from __future__ import annotations

import logging
from types import MappingProxyType
from typing import Literal

import numpy as np
import pandas as pd

from copernicus.corrections._clustering import mean_covariance, resolve_cluster
from copernicus.corrections._designs import (
    INTERCEPT,
    coefficient_index,
    covariate_design,
    outcome_design,
)
from copernicus.corrections._estimate_inputs import (
    equal_probability_sample,
    outcome_values,
    prediction_values,
    require_labelled,
)
from copernicus.corrections.data import CorrectionData
from copernicus.corrections.estimates import Estimate, make_estimate
from copernicus.errors import EstimatorError, SettingsError
from copernicus.rubric.identifiers import TARGET_CONSTRUCT

logger = logging.getLogger(__name__)

__all__ = ["ols_outcome_ppi", "ols_predictor_ppi"]

VARIANTS = ("ppi", "ludwig")
MIN_ROWS = 2


def _check_positive_definite(matrix: np.ndarray) -> None:
    """Refuse a moment matrix whose smallest eigenvalue is not clearly positive.

    The tolerance is the one ``numpy.linalg.matrix_rank`` uses: the largest absolute eigenvalue
    times the dimension times the machine epsilon.
    """
    eigenvalues = np.linalg.eigvalsh(matrix)
    tolerance = float(np.max(np.abs(eigenvalues))) * matrix.shape[0] * np.finfo(float).eps
    if float(eigenvalues.min()) <= tolerance:
        raise EstimatorError(
            "rectified moment matrix not positive definite (smallest eigenvalue "
            f"{float(eigenvalues.min()):.3g}): the coefficients are not defined on these texts"
        )


def _moments(design: np.ndarray, response: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``mean(x x')`` and ``mean(x v)`` over the rows of the design."""
    n_rows = design.shape[0]
    return design.T @ design / n_rows, design.T @ response / n_rows


def _scores(design: np.ndarray, response: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """The estimating function ``x (v - x'beta)`` of every row."""
    return design * (response - design @ beta)[:, None]


def _report(
    beta: np.ndarray,
    moment: np.ndarray,
    variance: np.ndarray,
    columns: tuple[str, ...],
    position: int,
) -> tuple[float, float, dict[str, object]]:
    """The reported coefficient, its standard error, and every coefficient by name."""
    inverse = np.linalg.inv(moment)
    covariance = inverse @ variance @ inverse
    # The sandwich is positive semi-definite by construction; the clip only removes rounding
    # below zero on a diagonal entry that is zero in exact arithmetic.
    errors = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    details: dict[str, object] = {
        "coefficients": MappingProxyType(dict(zip(columns, beta.tolist(), strict=True))),
        "standard_errors": MappingProxyType(dict(zip(columns, errors.tolist(), strict=True))),
        "columns": columns,
        "coefficient": columns[position],
    }
    return float(beta[position]), float(errors[position]), details


def ols_outcome_ppi(
    data: CorrectionData,
    formula: str | None = None,
    X: np.ndarray | None = None,  # noqa: N803 - the conventional name of a design matrix
    *,
    prediction: pd.Series | None = None,
    coefficient: str | int = 1,
    cluster: bool | None = None,
) -> Estimate:
    """A prediction-powered OLS coefficient of a regression of the construct on a design.

    Power tuning is not implemented for this estimator; for a mean it is available through
    ``mean_ppi(power_tuning=True)`` and ``pseudo_values`` (``mean_rectified_tuned``).

    Args:
        data: The correction data with gold on the labelled rows and the same ``pi`` on every
            labelled row (``draw_human_sample`` and ``attach_draw`` supply it).
        formula: A right-hand side over the data's columns such as ``"1 + log_chars + kind"``
            (``formulaic`` syntax; the intercept comes first unless the formula removes it).
        X: A design matrix with one row per text in data order, instead of ``formula``; its
            columns are named ``x0``, ``x1``, ...
        prediction: The prediction of the construct on every text, indexed by text id and on
            the gold scale; ``None`` for the first score column.
        coefficient: The reported coefficient: a design column name, or a position counted
            from 0 (the default 1 is the first column after the intercept).
        cluster: Whether to sum the estimating functions within each group before the
            covariance; ``None`` does so when a group holds more than one labelled text (as in
            a whole-group draw), ``True`` and ``False`` (numpy booleans too) force the choice.

    Returns:
        The estimate, with method ``ppi_ols_outcome`` and details ``coefficients`` and
        ``standard_errors`` (read-only mappings over the design's columns), ``columns``,
        ``coefficient`` (the reported column), ``prediction``, ``cluster`` (as given),
        ``clustered`` (the choice made) and, when clustered, ``n_clusters`` and
        ``n_clusters_labelled``.

    Raises:
        SettingsError: If not exactly one of ``formula`` and ``X`` is given, the formula is
            malformed, or ``cluster`` is not a boolean or ``None``.
        ConfigurationError: If the design or the prediction does not match the data, or the
            coefficient is not a column of the design.
        EstimatorError: If fewer than two texts are labelled, ``pi`` is missing or unequal on
            the labelled rows, clustering finds fewer than two groups, or the moment matrix
            is not positive definite.
    """
    design, columns = outcome_design(data, formula, X)
    position = coefficient_index(columns, coefficient)
    require_labelled(data.n_labelled, MIN_ROWS, "the prediction-powered regression")
    labelled = equal_probability_sample(data, "the prediction-powered regression")
    clustering = resolve_cluster(data, labelled, cluster)
    predicted, name = prediction_values(data, prediction)
    gold = data.gold[labelled]
    moment, target_all = _moments(design, predicted)
    _, target_scores = _moments(design[labelled], predicted[labelled])
    _, target_gold = _moments(design[labelled], gold)
    _check_positive_definite(moment)
    beta = np.linalg.solve(moment, target_gold + (target_all - target_scores))
    variance = mean_covariance(
        _scores(design, predicted, beta), clustering.groups
    ) + mean_covariance(
        design[labelled] * (predicted[labelled] - gold)[:, None], clustering.labelled_groups
    )
    point, se, details = _report(beta, moment, variance, columns, position)
    details.update(prediction=name, **clustering.details())
    logger.debug(
        "prediction-powered outcome coefficient %r: %.4g (SE %.3g)", columns[position], point, se
    )
    return make_estimate(
        point,
        se,
        n_labelled=int(labelled.sum()),
        n_total=data.n,
        method="ppi_ols_outcome",
        details=details,
    )


def ols_predictor_ppi(
    data: CorrectionData,
    outcome: str,
    covariates: str | None = None,
    X: np.ndarray | None = None,  # noqa: N803 - the conventional name of a design matrix
    *,
    prediction: pd.Series | None = None,
    variant: Literal["ppi", "ludwig"] = "ppi",
    coefficient: int = 1,
    cluster: bool | None = None,
) -> Estimate:
    """A corrected OLS coefficient of an outcome on the construct (and covariates).

    Args:
        data: The correction data with gold on the labelled rows and the same ``pi`` on every
            labelled row.
        outcome: A numeric column of the data, finite on every text.
        covariates: A right-hand side over the data's columns for ``W`` (evaluated with its
            intercept, which is then dropped because the design adds its own), or ``None``.
        X: The covariates ``W`` as a matrix with one row per text in data order, without an
            intercept column, instead of ``covariates``; its columns are named ``x0``, ``x1``,
            ... Give at most one of the two.
        prediction: The prediction of the construct on every text, indexed by text id and on
            the gold scale; ``None`` for the first score column.
        variant: ``ppi`` (every text as the base) or ``ludwig`` (the unlabelled texts).
        coefficient: The reported position in ``[1, construct, W]``: 1 is the construct's slope.
        cluster: As for ``ols_outcome_ppi``; the base term is clustered by the groups of its
            texts.

    Returns:
        The estimate, with method ``ppi_ols_predictor`` or ``ludwig_ols_predictor`` and details
        ``coefficients``, ``standard_errors``, ``columns`` (``Intercept``, ``construct`` and the
        covariate columns), ``coefficient``, ``prediction``, ``variant``, ``outcome``,
        ``n_base`` (the texts of the base term), ``cluster``, ``clustered`` and, when
        clustered, ``n_clusters`` (groups of the base texts) and ``n_clusters_labelled``.

    Raises:
        SettingsError: If the variant is unknown, both ``covariates`` and ``X`` are given, the
            covariate formula is malformed, or ``cluster`` is not a boolean or ``None``.
        ConfigurationError: If the outcome, the covariates or the prediction do not match the
            data, a column of ``X`` is constant, or the position is not a column of the design.
        EstimatorError: If fewer than two texts are labelled, the ``ludwig`` variant has fewer
            than two unlabelled texts, ``pi`` is missing or unequal on the labelled rows,
            clustering finds fewer than two groups, or the rectified moment matrix is not
            positive definite.
    """
    if variant not in VARIANTS:
        raise SettingsError(f"unknown variant {variant!r}; use one of {', '.join(VARIANTS)}")
    response = outcome_values(data, outcome)
    covariate_matrix, covariate_names = covariate_design(data, covariates, X)
    columns = (INTERCEPT, TARGET_CONSTRUCT, *covariate_names)
    position = coefficient_index(columns, coefficient)
    require_labelled(data.n_labelled, MIN_ROWS, "the corrected regression")
    labelled = equal_probability_sample(data, "the corrected regression")
    base = np.ones(data.n, dtype=bool) if variant == "ppi" else ~labelled
    if int(base.sum()) < MIN_ROWS:
        raise EstimatorError(
            f"the ludwig variant needs at least {MIN_ROWS} unlabelled texts as its primary "
            f"sample; there are {int(base.sum())}"
        )
    clustering = resolve_cluster(data, labelled, cluster, base=base)
    predicted, name = prediction_values(data, prediction)
    design = np.column_stack([np.ones(data.n), predicted, covariate_matrix])
    gold_design = design[labelled].copy()
    gold_design[:, 1] = data.gold[labelled]
    moment_base, target_base = _moments(design[base], response[base])
    moment_scores, target_scores = _moments(design[labelled], response[labelled])
    moment_gold, target_gold = _moments(gold_design, response[labelled])
    moment = moment_gold + (moment_base - moment_scores)
    _check_positive_definite(moment)
    beta = np.linalg.solve(moment, target_gold + (target_base - target_scores))
    rectifier = _scores(design[labelled], response[labelled], beta) - _scores(
        gold_design, response[labelled], beta
    )
    variance = mean_covariance(
        _scores(design[base], response[base], beta), clustering.groups
    ) + mean_covariance(rectifier, clustering.labelled_groups)
    point, se, details = _report(beta, moment, variance, columns, position)
    details.update(
        prediction=name,
        variant=variant,
        outcome=outcome,
        n_base=int(base.sum()),
        **clustering.details(),
    )
    logger.debug(
        "%s predictor coefficient %r: %.4g (SE %.3g)", variant, columns[position], point, se
    )
    return make_estimate(
        point,
        se,
        n_labelled=int(labelled.sum()),
        n_total=data.n,
        method=f"{variant}_ols_predictor",
        details=details,
    )
