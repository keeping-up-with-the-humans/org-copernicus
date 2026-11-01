"""Multiple imputation of the gold value: bootstrap-then-impute and predictive mean matching.

Both variants model the gold value of a text as ``1 + score + covariate`` with the best score
column of the training rows, and both refit that model on a bootstrap of the labelled
training rows for every one of ``m_draws`` draws, so that the uncertainty of the model
itself enters the between-draw variance (draws made from one fit with residual noise only
would understate it).

* ``model_draws``: each draw adds ``N(0, sigma_m^2)`` noise to the fitted mean, where
  ``sigma_m^2`` is the residual variance of that draw's bootstrap fit.
* ``pmm``: each draw replaces the fitted mean of a text by the observed gold value of one
  of its ``k_donors`` nearest training donors by fitted mean, chosen at random. Every draw
  therefore lies on the support of the observed gold values. The paper lists predictive mean
  matching while its analysis battery removed the variant, so it ships with that note
  (``REGISTRY["multiple_imputation"].note_for("pmm")``): its battery results have no
  counterpart in the paper.

The corrected value of a text is the mean over the draws, and ``draws`` returns all of
them (shape ``[m, n]``) for pooling with ``rubin_pool``. The draws of a fitted object are
reproducible and belong to the text: the random numbers of a text come from a generator
seeded with a seed stored at fit time and the text's row position in the data, so its
values do not depend on the other ids passed or their order (they do depend on the text
having the same position in the data). The draw streams are not those of the
paper's battery; the model is identical. The family addresses level displacement,
calibration shortfall and noise (the spread is restored in each draw).

Twenty draws serve as a point correction. Inference with Rubin's rules at a labelled share
of 2.5 to 5 percent needs 150 or more draws, because the fraction of missing information is
high.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import ClassVar, Literal

import numpy as np
import pandas as pd

from copernicus.corrections._proxy import check_count
from copernicus.corrections.data import CorrectionData
from copernicus.corrections.families._draws import DrawModels, fit_ols
from copernicus.corrections.families._fitted import (
    require_columns,
    row_positions,
    training_rows,
)
from copernicus.corrections.families._registry import FamilySpec, register
from copernicus.errors import ConfigurationError, EstimatorError, SettingsError

logger = logging.getLogger(__name__)

__all__ = ["ImputationFitted", "MultipleImputation", "PooledEstimate", "rubin_pool"]

FAMILY = "multiple_imputation"
VARIANTS = ("model_draws", "pmm")
ERRORS_ADDRESSED = ("level_displacement", "calibration_shortfall", "noise")
ESTIMATE_ERRORS = "confidence-interval coverage via Rubin's rules"
DESIGN_COLUMNS = 3
MIN_DRAWS_FOR_INFERENCE = 150
INFERENCE_NOTE = (
    f"inference-grade imputation needs m of {MIN_DRAWS_FOR_INFERENCE} or more at 2.5 to 5 "
    "percent labelled; the default draws serve a point correction"
)
PMM_NOTE = (
    "the paper lists predictive mean matching while its analysis battery removed the variant, "
    "so its battery results have no counterpart in the paper"
)


@dataclass(frozen=True)
class PooledEstimate:
    """An estimate pooled over imputed data sets by Rubin's rules.

    Attributes:
        estimate: The mean of the per-draw estimates.
        variance: The total variance ``W + (1 + 1/m) B``.
        se: The square root of the total variance.
        df: Degrees of freedom of the reference t distribution (``inf`` without
            between-draw variance).
        m: The number of draws.
        within: The mean of the per-draw variances, ``W``.
        between: The variance of the per-draw estimates (divisor ``m - 1``), ``B``.
    """

    estimate: float
    variance: float
    se: float
    df: float
    m: int
    within: float
    between: float


def rubin_pool(estimates: Sequence[float], variances: Sequence[float]) -> PooledEstimate:
    """Pool per-draw estimates and their variances by Rubin's rules.

    The pooled estimate is the mean of the estimates, ``W`` the mean of the variances, ``B``
    the variance of the estimates with divisor ``m - 1``, the total variance
    ``T = W + (1 + 1/m) B`` and the degrees of freedom
    ``(m - 1) * (1 + W / ((1 + 1/m) B))^2``. Equal estimates give ``B = 0``, ``T = W`` and
    infinite degrees of freedom.

    Args:
        estimates: The estimate from each imputed data set.
        variances: The variance of each estimate (the squared standard error).

    Returns:
        The pooled estimate.

    Raises:
        ConfigurationError: If the sequences differ in length, hold fewer than two draws,
            hold a value that is not finite, or a variance is negative.
    """
    point = np.asarray(estimates, dtype=float)
    spread = np.asarray(variances, dtype=float)
    if point.shape != spread.shape or point.ndim != 1:
        raise ConfigurationError(
            f"estimates and variances must be one-dimensional of equal length; got shapes "
            f"{point.shape} and {spread.shape}"
        )
    if point.size < 2:
        raise ConfigurationError(f"Rubin's rules need at least two draws; got {point.size}")
    if not (np.isfinite(point).all() and np.isfinite(spread).all()):
        raise ConfigurationError("estimates and variances must be finite")
    if (spread < 0.0).any():
        raise ConfigurationError("variances must not be negative")
    m = int(point.size)
    within = float(spread.mean())
    between = float(point.var(ddof=1))
    total = within + (1.0 + 1.0 / m) * between
    if between == 0.0:
        df = float("inf")
    else:
        df = (m - 1) * (1.0 + within / ((1.0 + 1.0 / m) * between)) ** 2
    return PooledEstimate(
        estimate=float(point.mean()),
        variance=total,
        se=float(np.sqrt(total)),
        df=float(df),
        m=m,
        within=within,
        between=between,
    )


@dataclass(frozen=True, eq=False)
class ImputationFitted:
    """Fitted imputation models, one per draw.

    Attributes:
        column: The score column the models read.
        models: The bootstrap models and the seed of the random part of the draws.
        provenance: What was fitted; stored as a read-only mapping.
    """

    column: str
    models: DrawModels
    provenance: Mapping[str, object]

    def __post_init__(self) -> None:
        """Freeze the provenance."""
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))

    def draws(self, data: CorrectionData, ids: np.ndarray) -> np.ndarray:
        """All completed columns for the named rows.

        Args:
            data: The correction data holding the rows.
            ids: Text ids of the rows.

        Returns:
            Floats of shape ``[m, n]``, identical on every call for the same ``ids``.

        Raises:
            ConfigurationError: If an id is not in the data.
        """
        require_columns(data, (self.column,))
        positions = row_positions(data, ids)
        score = data.frame[self.column].to_numpy(dtype=float)[positions]
        design = np.column_stack([np.ones_like(score), score, data.covariate[positions]])
        return self.models.complete(design, positions)

    def predict(self, data: CorrectionData, ids: np.ndarray) -> pd.Series:
        """The mean of the completed columns of the named rows.

        Args:
            data: The correction data holding the rows.
            ids: Text ids of the rows.

        Returns:
            Floats on the gold scale indexed by the ids.

        Raises:
            ConfigurationError: If an id is not in the data.
        """
        return pd.Series(
            self.draws(data, ids).mean(axis=0), index=pd.Index(np.asarray(ids).astype(str))
        )


@dataclass(frozen=True)
class MultipleImputation:
    """Bootstrap-then-impute draws of the gold value of every text.

    ``pmm`` (predictive mean matching) ships with a note: the paper lists it while its analysis
    battery removed it (see the module docstring).

    Attributes:
        variant: ``model_draws`` or ``pmm``.
        m_draws: Number of draws (an integer of at least 1).
        k_donors: Number of nearest donors for ``pmm`` (an integer of at least 1).
    """

    variant: str
    m_draws: int = 20
    k_donors: int = 5
    family: ClassVar[str] = FAMILY
    target: ClassVar[Literal["A", "B"]] = "B"

    def __post_init__(self) -> None:
        """Refuse an unknown variant and parameters outside their range; store plain integers.

        Raises:
            SettingsError: If the variant is unknown, or ``m_draws`` or ``k_donors`` is not an
                integer (a boolean or a float is refused) or is below 1.
        """
        if self.variant not in VARIANTS:
            raise SettingsError(
                f"unknown variant {self.variant!r} of family {FAMILY!r}; variants: "
                f"{', '.join(VARIANTS)}"
            )
        object.__setattr__(self, "m_draws", check_count("m_draws", self.m_draws, 1))
        object.__setattr__(self, "k_donors", check_count("k_donors", self.k_donors, 1))

    def applies(self, data: CorrectionData) -> tuple[bool, str]:
        """Multiple imputation applies to any data.

        Args:
            data: The correction data.

        Returns:
            ``(True, "")``.
        """
        return True, ""

    def fit(
        self, data: CorrectionData, train_ids: np.ndarray, *, rng: np.random.Generator
    ) -> ImputationFitted:
        """Draw the bootstrap models on the labelled training rows.

        Args:
            data: The correction data.
            train_ids: Ids of the labelled rows to learn from.
            rng: Generator for the bootstraps and for the seed of the later draws.

        Returns:
            The fitted draws; the provenance holds ``within_variance`` (mean residual
            variance), ``between_variance`` (mean over the training rows of the variance
            across draws, divisor ``m - 1``; an in-sample figure computed on the training
            rows, NaN for a single draw), ``m_draws``,
            ``k_donors`` (``None`` for ``model_draws``), ``score_column`` and a note on
            the number of draws inference needs.

        Raises:
            ConfigurationError: If an id is unknown, repeated or has no gold value.
            EstimatorError: If no score column is usable or the design of intercept, score
                and covariate is rank deficient on the training rows.
        """
        rows = training_rows(data, train_ids)
        design = np.column_stack([np.ones_like(rows.score), rows.score, rows.covariate])
        rank = int(np.linalg.matrix_rank(design))
        if rank < DESIGN_COLUMNS:
            raise EstimatorError(
                f"the design of intercept, score {rows.column!r} and covariate has rank {rank} "
                f"of {DESIGN_COLUMNS} on the training rows: the imputation model is not defined"
            )
        n_rows = design.shape[0]
        coefficients = np.empty((self.m_draws, DESIGN_COLUMNS))
        variance = np.empty(self.m_draws)
        donor_means: list[np.ndarray] = []
        donor_gold: list[np.ndarray] = []
        for draw in range(self.m_draws):
            sample = rng.integers(0, n_rows, size=n_rows)
            coefficients[draw], variance[draw] = fit_ols(design[sample], rows.gold[sample])
            if self.variant == "pmm":
                means = design @ coefficients[draw]
                order = np.argsort(means, kind="stable")
                donor_means.append(means[order])
                donor_gold.append(rows.gold[order])
        child_seed = int(rng.integers(0, np.iinfo(np.int64).max))
        pmm = self.variant == "pmm"
        models = DrawModels(
            coefficients=coefficients,
            residual_variance=variance,
            donors=(np.vstack(donor_means), np.vstack(donor_gold)) if pmm else None,
            k_donors=self.k_donors,
            child_seed=child_seed,
        )
        completed = models.complete(design, row_positions(data, train_ids))
        between = (
            float(np.mean(completed.var(axis=0, ddof=1))) if self.m_draws > 1 else float("nan")
        )
        return ImputationFitted(
            column=rows.column,
            models=models,
            provenance={
                "family": FAMILY,
                "variant": self.variant,
                "score_column": rows.column,
                "m_draws": self.m_draws,
                "k_donors": self.k_donors if pmm else None,
                "within_variance": float(variance.mean()),
                "between_variance": between,
                "mean_coefficients": [float(value) for value in coefficients.mean(axis=0)],
                "inference_note": INFERENCE_NOTE,
            },
        )


SPEC = register(
    FamilySpec(
        family=FAMILY,
        variants=VARIANTS,
        target="B",
        level="score",
        errors_addressed=ERRORS_ADDRESSED,
        estimate_errors=ESTIMATE_ERRORS,
        factory=MultipleImputation,
        variant_notes={"pmm": PMM_NOTE},
    )
)
