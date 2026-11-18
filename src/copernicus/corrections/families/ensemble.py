"""The correction ensemble: corrected columns pooled into one.

The members are corrected score columns of the same texts (the cross-fitted or held-out
predictions of several families), each on its own scale. They are standardised with the
population standard deviation of the rows given and combined by one of four rules:

* ``mean``: the mean of the standardised members.
* ``mode``: the row-wise mode of the members after each is rounded to the nearest gold class
  (a tie goes to the smaller class); it needs the discrete gold classes.
* ``factor``: the regression (Thomson) factor score of a one-factor principal-axis solution
  over the standardised members.
* ``pc1``: the first principal component of the standardised members.

``factor`` and ``pc1`` have an arbitrary sign; it is set so that the result does not
correlate negatively with the mean of the standardised members. Their provenance records the
choices that the members can leave open, each with one warning per call, which ``run_battery``
counts instead and sums up per entry and size (``_open_cases``): a constant mean of the
standardised members leaves the sign to the decomposition, and a combined column whose computed
correlation with that mean is at most ``1e-12`` in absolute value, as for two members of unit
variance that correlate negatively, is treated as not oriented by it
(``orientation_identified``); two
largest eigenvalues of a round of the iteration of ``factor``, or two largest singular values of
``pc1``, that are equal up to a relative ``1e-12`` make the direction count as not unique
(``direction_unique``, see ``top_two_tied``); and the iteration of ``factor`` can reach its cap
of 50 rounds without meeting its tolerance (``factor_converged``, with the rounds run in
``n_factor_rounds``).

The ensemble works on corrected columns, not on texts, so it is not a per-text correction
and has no factory. Its output is rank-only: ``mean``, ``factor`` and ``pc1`` are on the
scale of standardised scores, not on the gold scale, so it addresses the noise of the
members and raises the ceiling of calibration by pooling, and it supports rank validity and
correlation, not the level or spread of the human-coded values. ``mode`` is on the gold scale
but takes only gold classes.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import ClassVar, Literal

import numpy as np
import pandas as pd

from copernicus.corrections._messages import list_first_ten
from copernicus.corrections._open_cases import (
    warn_not_converged,
    warn_open_orientation,
    warn_tied_rounds,
    warn_tied_singular_values,
    warn_uncorrelated_orientation,
)
from copernicus.corrections.families._factor import (
    ORIENTATION_TOLERANCE,
    PAF_TOLERANCE,
    TIE_RELATIVE_TOLERANCE,
    orientation_correlation,
    paf_fit,
    regression_factor_score,
    relation_sign,
    standardize_apply,
    standardize_fit,
    top_two_tied,
)
from copernicus.corrections.families._registry import FamilySpec, register
from copernicus.errors import ConfigurationError, EstimatorError, NotApplicableError, SettingsError
from copernicus.validity.metrics import is_constant

logger = logging.getLogger(__name__)

__all__ = ["CorrectionEnsemble", "EnsembleResult"]

FAMILY = "correction_ensemble"
VARIANTS = ("mean", "mode", "factor", "pc1")
MIN_MEMBERS = 2
ERRORS_ADDRESSED = ("noise", "calibration_shortfall")


@dataclass(frozen=True, eq=False)
class EnsembleResult:
    """The pooled column and what went into it.

    Attributes:
        column: The combined value of every id, indexed by the ids given.
        provenance: A read-only mapping with ``family``, ``variant``, ``member_families``
            (the names of the members used), ``n_members``, ``dropped_members`` (the names
            of members dropped for a missing or non-finite value or no variation) and, for
            ``factor``, ``loadings``, and for ``pc1``, ``direction`` (each a mapping from
            member name to value, signed as applied). ``factor`` and ``pc1`` also hold
            ``orientation_identified`` (False when the mean of the standardised members is
            constant, so that the sign is the one the decomposition gave, or when the combined
            column's computed correlation with it is at most ``1e-12`` in absolute value,
            which is treated as too small to identify the sign) and
            ``direction_unique`` (False when the two largest eigenvalues of a round of the
            iteration, or the two largest singular values, are equal up to a relative
            ``1e-12``), and ``factor`` holds ``factor_converged`` (False when the iteration
            reached its cap of 50 rounds without meeting its tolerance) and
            ``n_factor_rounds``; each False logs a warning.
    """

    column: pd.Series
    provenance: Mapping[str, object]

    def __post_init__(self) -> None:
        """Freeze the provenance."""
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))


def _member_values(members: Mapping[str, pd.Series], ids: np.ndarray) -> dict[str, np.ndarray]:
    """Each member's values in the order of the ids, refusing a member that lacks an id.

    The ids (strings) are looked up in each member's string index through its hash table
    (``Index.get_indexer``), so the cost grows linearly with the number of ids.
    """
    values: dict[str, np.ndarray] = {}
    for name, series in members.items():
        index = pd.Index(series.index.astype(str))
        if index.has_duplicates:
            raise ConfigurationError(f"member {name!r} repeats ids")
        positions = index.get_indexer(ids)
        missing = ids[positions < 0]
        if missing.size:
            raise ConfigurationError(
                f"member {name!r} lacks the ids {list_first_ten(missing.tolist())}"
            )
        values[name] = series.to_numpy(dtype=float)[positions]
    return values


def _signed(names: list[str], flip: float, values: np.ndarray) -> Mapping[str, float]:
    """A read-only mapping from member name to the signed value."""
    return MappingProxyType(
        {name: float(flip * value) for name, value in zip(names, values, strict=True)}
    )


def _row_mode(members: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """Row-wise mode after rounding each entry to the nearest class; a tie goes to the smaller."""
    nearest = np.argmin(np.abs(members[:, :, None] - classes[None, None, :]), axis=2)
    votes = (nearest[:, :, None] == np.arange(classes.size)[None, None, :]).sum(axis=1)
    return classes[np.argmax(votes, axis=1)]


def _classes_of(gold_classes: np.ndarray | None) -> np.ndarray:
    """The sorted distinct gold classes of the mode combiner.

    Raises:
        NotApplicableError: If no classes are given.
        ConfigurationError: If the classes are empty, not one-dimensional or not all finite
            (the message names the non-finite ones).
    """
    if gold_classes is None:
        raise NotApplicableError(
            "the mode combiner needs the discrete gold classes (gold_classes); "
            "continuous gold has none"
        )
    values = np.asarray(gold_classes, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ConfigurationError(
            "gold_classes must be a non-empty one-dimensional array; "
            f"got shape {tuple(values.shape)}"
        )
    non_finite = values[~np.isfinite(values)]
    if non_finite.size:
        raise ConfigurationError(
            "gold_classes must be finite; non-finite classes: "
            f"{list_first_ten(non_finite.tolist())}"
        )
    return np.unique(values)


@dataclass(frozen=True)
class CorrectionEnsemble:
    """One combiner over corrected columns; its output is rank-only.

    Standardised pooled scores support rank validity and correlation, not the level or
    spread of the human-coded values (see the module docstring).

    Attributes:
        variant: ``mean``, ``mode``, ``factor`` or ``pc1``.
    """

    variant: str
    family: ClassVar[str] = FAMILY
    target: ClassVar[Literal["A"]] = "A"

    def __post_init__(self) -> None:
        """Refuse an unknown variant.

        Raises:
            SettingsError: If the variant is not one of the four.
        """
        if self.variant not in VARIANTS:
            raise SettingsError(
                f"unknown variant {self.variant!r} of family {FAMILY!r}; variants: "
                f"{', '.join(VARIANTS)}"
            )

    def combine(
        self,
        members: Mapping[str, pd.Series],
        ids: np.ndarray,
        *,
        gold_classes: np.ndarray | None = None,
    ) -> EnsembleResult:
        """Combine the member columns over the named ids.

        A member that lacks a requested id is an error (a missing id is a bug in the caller,
        not a dropout). A member with a non-finite value or no variation on these ids (zero
        population standard deviation) is dropped and named in ``dropped_members``.

        Args:
            members: Corrected columns by member name, each a series indexed by text id
                (ids are compared as strings).
            ids: The text ids to combine, unique.
            gold_classes: The discrete gold values, needed by ``mode`` only.

        Returns:
            The combined column and its provenance.

        Raises:
            NotApplicableError: If the variant is ``mode`` and no gold classes are given.
            ConfigurationError: If the ids repeat, a member repeats an id or lacks one, or (for
                ``mode``) the gold classes are empty, not one-dimensional or not all finite.
            EstimatorError: If fewer than two members have finite values with some
                variation on these ids.
        """
        classes = _classes_of(gold_classes) if self.variant == "mode" else None
        wanted = np.asarray(ids).astype(str)
        if np.unique(wanted).size != wanted.size:
            raise ConfigurationError("the ids to combine repeat")
        values = _member_values(members, wanted)
        usable = [name for name, v in values.items() if np.isfinite(v).all() and not is_constant(v)]
        dropped = tuple(name for name in values if name not in usable)
        if len(usable) < MIN_MEMBERS:
            raise EstimatorError(
                f"the ensemble needs at least two usable members; usable: {len(usable)}"
                + (f", dropped: {list_first_ten(list(dropped))}" if dropped else "")
            )
        matrix = np.column_stack([values[name] for name in usable])
        mean, sd = standardize_fit(matrix)
        standardised = standardize_apply(matrix, mean, sd)
        column, extra = self._pool(matrix, standardised, usable, classes)
        provenance: dict[str, object] = {
            "family": FAMILY,
            "variant": self.variant,
            "member_families": tuple(usable),
            "n_members": len(usable),
            "dropped_members": dropped,
            **extra,
        }
        logger.debug(
            "ensemble %s over %d members, %d dropped", self.variant, len(usable), len(dropped)
        )
        return EnsembleResult(
            column=pd.Series(column, index=pd.Index(wanted)), provenance=provenance
        )

    def _pool(
        self,
        matrix: np.ndarray,
        standardised: np.ndarray,
        names: list[str],
        classes: np.ndarray | None,
    ) -> tuple[np.ndarray, dict[str, object]]:
        """The combined values and the extra provenance of the variant."""
        reference = standardised.mean(axis=1)
        if self.variant == "mean":
            return reference, {}
        if classes is not None:
            return _row_mode(matrix, classes), {}
        if self.variant == "factor":
            correlation = standardised.T @ standardised / standardised.shape[0]
            fit = paf_fit(correlation)
            score = regression_factor_score(standardised, correlation, fit.loadings)
            flip = relation_sign(score, reference)
            oriented = self._oriented(score, reference, "numpy's eigendecomposition")
            if not fit.direction_unique:
                tied = len(fit.tied_rounds)
                warn_tied_rounds(logger, self._method, tied, fit.rounds, TIE_RELATIVE_TOLERANCE)
            if not fit.converged:
                warn_not_converged(logger, self._method, fit.rounds, PAF_TOLERANCE)
            return flip * score, {
                "loadings": _signed(names, flip, fit.loadings),
                "orientation_identified": oriented,
                "direction_unique": fit.direction_unique,
                "factor_converged": fit.converged,
                "n_factor_rounds": fit.rounds,
            }
        centred = standardised - standardised.mean(axis=0)
        _, singular, vectors = np.linalg.svd(centred, full_matrices=False)
        direction = vectors[0]
        score = centred @ direction
        flip = relation_sign(score, reference)
        oriented = self._oriented(score, reference, "numpy's singular value decomposition")
        unique = not top_two_tied(singular)
        if not unique:
            warn_tied_singular_values(logger, self._method, TIE_RELATIVE_TOLERANCE)
        return flip * score, {
            "direction": _signed(names, flip, direction),
            "orientation_identified": oriented,
            "direction_unique": unique,
        }

    @property
    def _method(self) -> str:
        """How the warnings name the combiner."""
        return f"correction ensemble ({self.variant})"

    def _oriented(self, score: np.ndarray, reference: np.ndarray, source: str) -> bool:
        """Whether the mean of the members fixes the sign, with one warning when it does not."""
        correlation = orientation_correlation(score, reference)
        oriented = bool(abs(correlation) > ORIENTATION_TOLERANCE)
        if oriented:
            return True
        open_direction = (
            "the direction in which the column ranks the texts is then not fixed by the members"
        )
        named = (
            ("the mean of the standardised members", reference),
            ("the combined column", score),
        )
        constant = [name for name, values in named if is_constant(values)]
        if constant:
            warn_open_orientation(
                logger,
                self._method,
                f"{' and '.join(constant)} {'is' if len(constant) == 1 else 'are'} constant",
                "the combined column",
                source,
                open_direction,
            )
        else:
            values = (correlation, ORIENTATION_TOLERANCE)
            warn_uncorrelated_orientation(
                logger, self._method, "the combined column", named[0][0], values, open_direction
            )
        return False


SPEC = register(
    FamilySpec(
        family=FAMILY,
        variants=VARIANTS,
        target="A",
        level="score",
        errors_addressed=ERRORS_ADDRESSED,
        estimate_errors="none",
        factory=None,
    )
)
