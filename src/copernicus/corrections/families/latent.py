"""Latent pooling: one factor over the standardised score columns, anchored to gold.

The score columns are standardised with the training rows (population standard deviation,
a constant column kept at standard deviation 1), a single principal-axis factor is
extracted from their correlation matrix, and the weighted-sum factor score is oriented so
that it does not correlate negatively with the row mean of the standardised scores. A
least-squares line of the gold values on the factor score then puts it on the gold scale.

The pooling combines several score columns (different configurations of one coder, or
different coders), so it addresses noise and raises the ceiling that calibration of a
single column can reach. It is not a hierarchical latent-variable model: there is no
separate shared-bias term, and the factor weights are those of the training rows only.
Because it pools several columns it can change the ranks of the texts. It needs at least
two score columns.

The provenance records three choices that the training rows can leave open, each with one
warning per fit: a row mean that is constant leaves the sign of the factor to the
eigendecomposition, and a factor whose computed correlation with the row mean is at most
``1e-12`` in absolute value, as for two columns of unit variance that correlate negatively, is
treated as not oriented by it (``orientation_identified``); the anchor line's slope takes the sign
up; a
round of the iteration whose two largest eigenvalues are equal up to a relative ``1e-12`` makes
the direction of its eigenvector count as not unique (``direction_unique``, see
``top_two_tied``); and an iteration that reaches its cap of 50 rounds without meeting its
tolerance returns the loadings of its last round (``factor_converged``, with the rounds run in
``n_factor_rounds``).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import ClassVar, Literal

import numpy as np
import pandas as pd

from copernicus.corrections._open_cases import (
    warn_not_converged,
    warn_open_orientation,
    warn_tied_rounds,
    warn_uncorrelated_orientation,
)
from copernicus.corrections.data import CorrectionData
from copernicus.corrections.families._factor import (
    ORIENTATION_TOLERANCE,
    PAF_TOLERANCE,
    TIE_RELATIVE_TOLERANCE,
    FactorFit,
    align_loadings,
    orientation_correlation,
    paf_fit,
    standardize_apply,
    standardize_fit,
    weighted_factor_score,
)
from copernicus.corrections.families._fitted import (
    labelled_positions,
    require_columns,
    row_positions,
    score_column_shortage,
)
from copernicus.corrections.families._registry import FamilySpec, register
from copernicus.corrections.families.calibration import fit_ols_line
from copernicus.errors import EstimatorError, NotApplicableError, SettingsError
from copernicus.validity.metrics import is_constant

logger = logging.getLogger(__name__)

__all__ = ["LatentFitted", "LatentPooling", "make_latent_pooling"]

FAMILY = "latent_pooling"
VARIANT = "factor_anchor"
METHOD = "latent pooling"
ERRORS_ADDRESSED = ("noise", "calibration_shortfall")


@dataclass(frozen=True, eq=False)
class LatentFitted:
    """A fitted pooled factor score anchored to the gold values.

    Attributes:
        columns: The score columns pooled, in the order the loadings refer to.
        mean: Training mean of each column.
        sd: Training population standard deviation of each column (1 for a constant one).
        loadings: The oriented first-factor loadings.
        intercept: Intercept of the anchor line.
        slope: Slope of the anchor line.
        provenance: What was fitted; stored as a read-only mapping.
    """

    columns: tuple[str, ...]
    mean: np.ndarray
    sd: np.ndarray
    loadings: np.ndarray
    intercept: float
    slope: float
    provenance: Mapping[str, object]

    def __post_init__(self) -> None:
        """Freeze the provenance."""
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))

    def predict(self, data: CorrectionData, ids: np.ndarray) -> pd.Series:
        """The anchored factor score of the named rows.

        Args:
            data: The correction data holding the rows.
            ids: Text ids of the rows.

        Returns:
            Floats on the gold scale indexed by the ids.

        Raises:
            ConfigurationError: If an id is not in the data or a pooled score column is
                absent from the data.
        """
        require_columns(data, self.columns)
        positions = row_positions(data, ids)
        matrix = data.frame[list(self.columns)].to_numpy(dtype=float)[positions]
        factor = weighted_factor_score(standardize_apply(matrix, self.mean, self.sd), self.loadings)
        return pd.Series(
            self.intercept + self.slope * factor, index=pd.Index(np.asarray(ids).astype(str))
        )


@dataclass(frozen=True)
class LatentPooling:
    """One-factor pooling of all score columns, anchored to the gold values.

    The family addresses noise and the calibration shortfall of a single column (the
    ceiling of calibration is raised by pooling configurations). The variant is
    ``factor_anchor``.
    """

    family: ClassVar[str] = FAMILY
    variant: ClassVar[str] = VARIANT
    target: ClassVar[Literal["A", "B"]] = "A"

    def applies(self, data: CorrectionData) -> tuple[bool, str]:
        """Whether the data hold enough score columns to pool; ``fit`` makes the same check.

        Args:
            data: The correction data.

        Returns:
            ``(True, "")``, or ``(False, reason)`` with fewer than two score columns.
        """
        shortage = score_column_shortage(data, METHOD)
        return (False, shortage) if shortage else (True, "")

    def fit(
        self, data: CorrectionData, train_ids: np.ndarray, *, rng: np.random.Generator
    ) -> LatentFitted:
        """Extract the factor on the training rows and anchor it to the gold values.

        Args:
            data: The correction data.
            train_ids: Ids of the labelled rows to learn from.
            rng: Unused; the fit is deterministic.

        Returns:
            The fitted pooling; its provenance holds ``loadings``, ``anchor_slope``,
            ``anchor_intercept``, ``n_indicators``, ``score_column = "all"``,
            ``orientation_identified`` (False when the row mean of the standardised scores is
            constant on the training rows, so that the loadings keep the sign of the
            eigenvector, or when the factor's computed correlation with it is at most
            ``1e-12`` in absolute value, which is treated as too small to identify the sign),
            ``direction_unique`` (False when a round of
            the iteration had two largest eigenvalues equal up to a relative ``1e-12``),
            ``factor_converged`` (False
            when the iteration reached its cap of 50 rounds without meeting its tolerance)
            and ``n_factor_rounds``; each False logs one warning, which ``run_battery`` counts
            instead and sums up per entry and size (``_open_cases``).

        Raises:
            NotApplicableError: If the data have fewer than two score columns, the check of
                ``applies``.
            ConfigurationError: If an id is unknown, repeated or has no gold value.
            EstimatorError: If the factor score is constant on the training rows.
        """
        del rng
        shortage = score_column_shortage(data, METHOD)
        if shortage:
            raise NotApplicableError(f"{FAMILY}/{VARIANT} does not apply: {shortage}")
        columns = data.score_columns
        positions, gold = labelled_positions(data, train_ids)
        matrix = data.frame[list(columns)].to_numpy(dtype=float)[positions]
        mean, sd = standardize_fit(matrix)
        standardised = standardize_apply(matrix, mean, sd)
        correlation = standardised.T @ standardised / standardised.shape[0]
        factor_fit = paf_fit(correlation)
        reference = standardised.mean(axis=1)
        correlation = orientation_correlation(
            weighted_factor_score(standardised, factor_fit.loadings), reference
        )
        oriented = bool(abs(correlation) > ORIENTATION_TOLERANCE)
        loadings = align_loadings(standardised, factor_fit.loadings, reference)
        factor = weighted_factor_score(standardised, loadings)
        if is_constant(factor):
            raise EstimatorError(
                "the pooled factor score is constant on the training rows: "
                "the score columns carry no variation to pool"
            )
        line = fit_ols_line(factor, gold, "factor score")
        _warn_open_choices(correlation, factor_fit)
        logger.debug("latent pooling over %d columns, slope %.4f", len(columns), line.slope)
        return LatentFitted(
            columns=columns,
            mean=mean,
            sd=sd,
            loadings=loadings,
            intercept=line.intercept,
            slope=line.slope,
            provenance={
                "family": FAMILY,
                "variant": VARIANT,
                "score_column": "all",
                "loadings": [float(value) for value in loadings],
                "anchor_slope": line.slope,
                "anchor_intercept": line.intercept,
                "n_indicators": len(columns),
                "orientation_identified": oriented,
                "direction_unique": factor_fit.direction_unique,
                "factor_converged": factor_fit.converged,
                "n_factor_rounds": factor_fit.rounds,
            },
        )


def _warn_open_choices(correlation: float, factor_fit: FactorFit) -> None:
    """One warning for each choice that the training rows left open, through this logger.

    ``correlation`` is the one of the factor score with the row mean, which sets the sign: NaN
    when the row mean is constant (a constant factor score is refused before), and of the size
    of rounding errors for two columns of unit variance that correlate negatively.
    """
    taken_up = (
        "the anchor line's slope takes up that sign, so the anchored predictions do not depend "
        "on it beyond rounding"
    )
    reference = "the row mean of the standardised score columns"
    if np.isnan(correlation):
        warn_open_orientation(
            logger,
            METHOD,
            f"{reference} is constant on the training rows",
            "the factor",
            "numpy's eigendecomposition",
            taken_up,
        )
    elif abs(correlation) <= ORIENTATION_TOLERANCE:
        values = (correlation, ORIENTATION_TOLERANCE)
        warn_uncorrelated_orientation(logger, METHOD, "the factor", reference, values, taken_up)
    if not factor_fit.direction_unique:
        tied = len(factor_fit.tied_rounds)
        warn_tied_rounds(logger, METHOD, tied, factor_fit.rounds, TIE_RELATIVE_TOLERANCE)
    if not factor_fit.converged:
        warn_not_converged(logger, METHOD, factor_fit.rounds, PAF_TOLERANCE)


def make_latent_pooling(variant: str) -> LatentPooling:
    """Build the pooling correction named by the registry.

    Args:
        variant: Must be ``factor_anchor``.

    Returns:
        The correction.

    Raises:
        SettingsError: If the variant is not ``factor_anchor``.
    """
    if variant != VARIANT:
        raise SettingsError(f"unknown variant {variant!r} of family {FAMILY!r}; use {VARIANT!r}")
    return LatentPooling()


SPEC = register(
    FamilySpec(
        family=FAMILY,
        variants=(VARIANT,),
        target="A",
        level="score",
        errors_addressed=ERRORS_ADDRESSED,
        estimate_errors="none",
        factory=make_latent_pooling,
    )
)
