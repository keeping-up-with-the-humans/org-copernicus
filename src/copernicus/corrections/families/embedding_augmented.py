"""Embedding-augmented calibration: ridge regression on text embeddings, with or without a score.

The embeddings of a text come from one encoder's set, or from ``"stacked"``: every set of the
data concatenated in mapping order, each set z-scored and weighted to the same total variance
with the baseline's ``fit_scaler``. They are standardised on the training rows and enter a
ridge regression with an unpenalised intercept. Three variants:

* ``embeddings_only``: ridge of the gold value on the embeddings. It reads no score, so it is
  the yardstick for the incremental validity of the coders.
* ``embeddings_plus_score``: ridge on the embeddings and the best score column; the score is
  standardised with its mean and population standard deviation over the training rows
  (fixed across the deletions) and then penalised like the embeddings. The original analysis
  appended the raw score, which was harmless because its scores shared one rating scale;
  this library receives scores on arbitrary scales, and a raw column would make the validity
  and the chosen penalty depend on the score's units.
* ``score_anchored_residual``: the least-squares calibration line of the gold value on the
  best score column, plus a ridge of the line's residual on the embeddings. It nests linear
  calibration: when the embeddings carry nothing, the chosen penalty shrinks the residual
  model towards zero and the corrected value stays on the line.

The penalty comes from ``ALPHAS`` by grouped inner deletions: the groups of the training rows
are split into ``min(5, groups)`` grouped folds, each fold is deleted in turn, the pipeline
is refitted on the rest, and the deleted rows' squared error is pooled over all deletions.
Inside each deletion the score column is selected again on the retained rows and the
calibration line refitted there, so neither the column nor the line nor the penalty sees the
gold values of the rows whose error decides them (``embeddings_only`` has nothing to select).
The standardisation of the embeddings stays the one of all training rows: it uses no gold
value. The deletions follow a seed drawn from the fit's generator (``inner_seed``), so the
variants fitted with generators in the same state share their deletions.

The family raises the ceiling of the calibration shortfall with the information in the text,
corrects level displacement and addresses covariate-linked error; with pseudo-values on top
of its predictions it serves estimates (bias and coverage, narrower confidence intervals).
Every embedding set used must cover every text of the data with a finite vector; ids are
compared as strings.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import ClassVar, Literal

import numpy as np
import pandas as pd

from copernicus.baseline.folds import rng_for
from copernicus.baseline.ridge import Scaler, apply_scaler, fit_scaler
from copernicus.corrections.crossfit import inner_k, inner_splits
from copernicus.corrections.data import CorrectionData
from copernicus.corrections.families._embedding_blocks import (
    STACKED,
    encoder_names,
    encoder_problem,
    gather_vectors,
    training_vectors,
)
from copernicus.corrections.families._embedding_search import (
    Predictor,
    fit_variant,
    penalty_provenance,
)
from copernicus.corrections.families._fitted import (
    grouped_training,
    labelled_group_shortage,
    require_columns,
    row_positions,
)
from copernicus.corrections.families._registry import FamilySpec, register
from copernicus.corrections.families._ridge_path import RidgeCoefficients
from copernicus.errors import SettingsError

logger = logging.getLogger(__name__)

__all__ = ["EmbeddingAugmented", "EmbeddingFitted"]

FAMILY = "embedding_augmented"
METHOD = "embedding-augmented calibration"
VARIANTS = ("embeddings_only", "embeddings_plus_score", "score_anchored_residual")
ERRORS_ADDRESSED = ("calibration_shortfall", "level_displacement", "covariate_linked_error")
ESTIMATE_ERRORS = "with pseudo-values on top: bias and coverage, narrower confidence intervals"
DELETION_KEY = "embedding_augmented"
NO_SCORE = "none"


@dataclass(frozen=True, eq=False)
class EmbeddingFitted:
    """A fitted embedding-augmented calibration.

    Attributes:
        encoders: The embedding sets read, in the order of the design.
        dims: The dimension of each set.
        scaler: The standardisation of the embedding blocks, fitted on the training rows.
        score_columns: The score column the variant reads (none for ``embeddings_only``).
        ridge: The ridge at the chosen penalty, on the standardised embeddings (followed by
            the score for ``embeddings_plus_score``). Its weights are not repeated in the
            provenance, which keeps the scalar coefficients only.
        predictor: Maps the standardised embeddings and the score matrix to the prediction.
        provenance: What was fitted; stored as a read-only mapping.
    """

    encoders: tuple[str, ...]
    dims: tuple[int, ...]
    scaler: Scaler
    score_columns: tuple[str, ...]
    ridge: RidgeCoefficients
    predictor: Predictor
    provenance: Mapping[str, object]

    def __post_init__(self) -> None:
        """Freeze the provenance."""
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))

    def predict(self, data: CorrectionData, ids: np.ndarray) -> pd.Series:
        """The corrected values of the named rows, from the data's own embeddings and scores.

        Args:
            data: The correction data holding the rows and the embedding sets of the fit.
            ids: Text ids of the rows.

        Returns:
            Floats on the gold scale indexed by the ids (empty for no id).

        Raises:
            ConfigurationError: If an id is not in the data, the data lack the score column of
                the fit or one of its embedding sets, a set lacks one of the texts or holds a
                non-finite value in the vector of one of them (the message names the first
                ten), or a set has another dimension.
        """
        require_columns(data, self.score_columns)
        positions = row_positions(data, ids)
        wanted = np.asarray(ids).astype(str)
        design = apply_scaler(gather_vectors(data, self.encoders, self.dims, wanted), self.scaler)
        scores = data.frame[list(self.score_columns)].to_numpy(dtype=float)[positions]
        return pd.Series(
            np.asarray(self.predictor(design, scores), dtype=float), index=pd.Index(wanted)
        )


@dataclass(frozen=True)
class EmbeddingAugmented:
    """Ridge calibration on text embeddings, with the penalty from grouped nested deletions.

    Attributes:
        variant: ``embeddings_only``, ``embeddings_plus_score`` or
            ``score_anchored_residual``.
        encoder: The name of an embedding set of the data, or ``"stacked"`` for every set.
    """

    variant: str
    encoder: str = STACKED
    family: ClassVar[str] = FAMILY
    target: ClassVar[Literal["A", "B"]] = "A"

    def __post_init__(self) -> None:
        """Refuse an unknown variant and an empty encoder name.

        Raises:
            SettingsError: If the variant is not one of the three or the encoder name is
                empty.
        """
        if self.variant not in VARIANTS:
            raise SettingsError(
                f"unknown variant {self.variant!r} of family {FAMILY!r}; variants: "
                f"{', '.join(VARIANTS)}"
            )
        if not isinstance(self.encoder, str) or not self.encoder.strip():
            raise SettingsError(
                f"the encoder must name an embedding set, or be {STACKED!r} for every set; "
                f"got {self.encoder!r}"
            )

    def applies(self, data: CorrectionData) -> tuple[bool, str]:
        """Whether the data carry the embeddings and enough labelled groups; ``fit`` checks both.

        Args:
            data: The correction data.

        Returns:
            ``(True, "")``, or ``(False, reason)`` when the data carry no embeddings, no set
            of the encoder's name, fewer than two sets for ``"stacked"``, or fewer than two
            groups among the labelled rows.
        """
        problem = encoder_problem(data, self.encoder) or labelled_group_shortage(data, METHOD)
        return (False, problem) if problem else (True, "")

    def fit(
        self, data: CorrectionData, train_ids: np.ndarray, *, rng: np.random.Generator
    ) -> EmbeddingFitted:
        """Choose the penalty on grouped deletions of the training rows and fit at it.

        Args:
            data: The correction data with its embedding sets.
            train_ids: Ids of the labelled rows to learn from.
            rng: Generator from which the seed of the deletions is drawn.

        Returns:
            The fitted calibration; its provenance holds the penalty fields
            (``penalty_alpha``, ``penalty_selection``, ``k_inner_folds``, ``n_inner_groups``,
            ``penalty_at_grid_edge``, ``penalty_at_lower_edge``, ``penalty_at_upper_edge``,
            ``penalty_deletion_share_lower``, ``penalty_deletion_share_upper``), ``encoder``,
            ``encoders``, ``n_features`` (the columns of the ridge design),
            ``score_column`` (``"none"`` for ``embeddings_only``), ``inner_seed`` and the
            scalar coefficients (for ``embeddings_plus_score`` also ``score_mean`` and
            ``score_sd``, the standardisation of the score, and ``score_coefficient`` per
            standard deviation of the score).

        Raises:
            NotApplicableError: If the data lack the embeddings the encoder names (with the
                reason ``applies`` gives), or the training rows hold fewer than two groups.
            ConfigurationError: If a set does not cover every text of the data or holds a
                non-finite value in the vector of one of them (the message names the first ten
                offending ids), or an id is unknown, repeated or has no gold value.
            EstimatorError: If no score column is usable where the variant needs one.
        """
        names = encoder_names(data, self.encoder, f"{FAMILY}/{self.variant}")
        rows = grouped_training(data, train_ids, rng, METHOD, label=f"{FAMILY}/{self.variant}")
        blocks = training_vectors(data, names, rows.positions)
        scaler = fit_scaler(blocks)
        design = apply_scaler(blocks, scaler)
        splits = inner_splits(
            rows.groups,
            inner_k(rows.n_groups),
            rng_for(rows.inner_seed, DELETION_KEY, "deletions"),
        )
        scores = data.scores.iloc[rows.positions].reset_index(drop=True)
        fitted = fit_variant(self.variant, design, rows.gold, scores, splits, rows.n_groups)
        logger.debug("embedding-augmented %s on %s fitted", self.variant, ", ".join(names))
        return EmbeddingFitted(
            encoders=names,
            dims=tuple(int(block.shape[1]) for block in blocks),
            scaler=scaler,
            score_columns=fitted.score_columns,
            ridge=fitted.ridge,
            predictor=fitted.predictor,
            provenance={
                "family": FAMILY,
                "variant": self.variant,
                "encoder": self.encoder,
                "encoders": list(names),
                "score_column": fitted.score_columns[0] if fitted.score_columns else NO_SCORE,
                "n_features": int(fitted.ridge.weights.shape[0]),
                "inner_seed": rows.inner_seed,
                **penalty_provenance(fitted.choice),
                **fitted.coefficients,
            },
        )


SPEC = register(
    FamilySpec(
        family=FAMILY,
        variants=VARIANTS,
        target="A, then C",
        level="score",
        errors_addressed=ERRORS_ADDRESSED,
        estimate_errors=ESTIMATE_ERRORS,
        factory=EmbeddingAugmented,
    )
)
