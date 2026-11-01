"""The common interface of the correction families and the registry that lists them.

A score-level correction implements ``ScoreCorrection``: ``fit`` learns from the labelled
rows named by ``train_ids`` and returns a ``Fitted`` object whose ``predict`` gives the
corrected value of any rows on the gold scale. Each family module registers a
``FamilySpec`` at import (name, variants, target, level, the errors it addresses and a
factory), and ``get_correction`` builds a correction from the family and variant names.
Estimate-level specs registered by the estimator modules carry no factory, and neither do the
score-level combiners that work on corrected columns or across corpora (the correction ensemble
and blind transport): they are called as functions, not fitted per text.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

import numpy as np
import pandas as pd

from copernicus.errors import ConfigurationError, SettingsError

if TYPE_CHECKING:
    from copernicus.corrections.data import CorrectionData

logger = logging.getLogger(__name__)

__all__ = [
    "REGISTRY",
    "FamilySpec",
    "Fitted",
    "ScoreCorrection",
    "get_correction",
    "list_corrections",
    "register",
]

LEVELS = ("score", "estimate")
SCORE_ERRORS = (
    "level_displacement",
    "calibration_shortfall",
    "covariate_linked_error",
    "noise",
)
LISTING_COLUMNS = [
    "family",
    "variant",
    "target",
    "level",
    "errors_addressed",
    "estimate_errors",
]


@runtime_checkable
class Fitted(Protocol):
    """A fitted correction.

    ``provenance`` is a read-only mapping that holds at least ``family``, ``variant``,
    ``score_column`` (where the correction works on one score column) and the fitted
    coefficients.
    """

    @property
    def provenance(self) -> Mapping[str, object]:
        """What was fitted: family, variant, score column and coefficients."""
        ...

    def predict(self, data: CorrectionData, ids: np.ndarray) -> pd.Series:
        """The corrected values of the named rows.

        Args:
            data: The correction data holding the rows.
            ids: Text ids of the rows to correct.

        Returns:
            Floats on the gold scale, indexed by the string ids in the order given.
        """
        ...


@runtime_checkable
class ScoreCorrection(Protocol):
    """A score-level correction: ``applies`` says whether it can run, ``fit`` learns."""

    @property
    def family(self) -> str:
        """The family name."""
        ...

    @property
    def variant(self) -> str:
        """The variant name within the family."""
        ...

    @property
    def target(self) -> Literal["A", "B"]:
        """The error target the correction addresses."""
        ...

    def applies(self, data: CorrectionData) -> tuple[bool, str]:
        """Whether the correction applies to the data.

        Args:
            data: The correction data.

        Returns:
            ``(True, "")``, or ``(False, reason)`` when it does not apply. It does not
            raise for a mismatch.
        """
        ...

    def fit(
        self, data: CorrectionData, train_ids: np.ndarray, *, rng: np.random.Generator
    ) -> Fitted:
        """Fit on the labelled rows named by ``train_ids``.

        Args:
            data: The correction data.
            train_ids: Ids of the labelled rows to learn from.
            rng: Generator for any randomness in the fit.

        Returns:
            The fitted correction.
        """
        ...


@dataclass(frozen=True)
class FamilySpec:
    """One row of the registry.

    Attributes:
        family: The family name.
        variants: The variant names.
        target: The error target the family addresses.
        level: ``"score"`` for a correction of each text's score, ``"estimate"`` for an
            estimator of a statistic.
        errors_addressed: The score errors the family addresses, a tuple of the words
            ``level_displacement``, ``calibration_shortfall``, ``covariate_linked_error`` and
            ``noise`` (empty for an estimate-level family). Qualifiers such as "spread
            matched to the signal" or "comparability only" belong in the family's docstring.
        estimate_errors: The errors of an estimate that the family addresses, in plain
            words, or ``"none"`` for a family that works on the score only.
        factory: Builds a ``ScoreCorrection`` from a variant name (and keyword parameters);
            ``None`` for an estimate-level family and for a score-level combiner that is
            called as a function rather than fitted per text (the correction ensemble, blind
            transport).
        variant_errors: The score errors of individual variants where they differ from
            the family tuple, by variant name; ``None`` when every variant addresses
            ``errors_addressed``. ``list_corrections`` shows the variant's own tuple for a
            variant named here and the family tuple for the others.
        variant_notes: A note a variant ships with, by variant name (such as why it is kept
            although an analysis left it out); ``None`` when no variant has one.
    """

    family: str
    variants: tuple[str, ...]
    target: str
    level: Literal["score", "estimate"]
    errors_addressed: tuple[str, ...]
    estimate_errors: str
    factory: Callable[..., ScoreCorrection] | None
    variant_errors: Mapping[str, tuple[str, ...]] | None = None
    variant_notes: Mapping[str, str] | None = None

    def note_for(self, variant: str) -> str:
        """The note a variant ships with.

        Args:
            variant: A variant of the family.

        Returns:
            The variant's note from ``variant_notes``, or ``""`` when it has none.
        """
        if self.variant_notes is not None and variant in self.variant_notes:
            return self.variant_notes[variant]
        return ""

    def errors_for(self, variant: str) -> tuple[str, ...]:
        """The score errors a variant addresses.

        Args:
            variant: A variant of the family.

        Returns:
            The variant's own tuple from ``variant_errors`` when it has one, else the
            family's ``errors_addressed``.
        """
        if self.variant_errors is not None and variant in self.variant_errors:
            return self.variant_errors[variant]
        return self.errors_addressed

    def __post_init__(self) -> None:
        """Refuse an entry that is incomplete or inconsistent.

        Raises:
            SettingsError: If the name, target or estimate errors are empty, the level is
                neither ``"score"`` nor ``"estimate"``, the variants are empty or repeat,
                an error word is unknown (in the family tuple or in ``variant_errors``), a
                key of ``variant_errors`` or ``variant_notes`` is not a variant of the family,
                a note is not non-empty text, a score-level family has a factory that is not
                callable, or an estimate-level family has one.
        """
        if not self.family or not self.target or not self.estimate_errors:
            raise SettingsError("a family needs a name, a target and estimate_errors")
        if self.level not in LEVELS:
            raise SettingsError(
                f"family {self.family!r}: level must be one of {', '.join(LEVELS)}; "
                f"got {self.level!r}"
            )
        if not self.variants or len(set(self.variants)) != len(self.variants):
            raise SettingsError(
                f"family {self.family!r}: variants must be non-empty and unique; "
                f"got {self.variants!r}"
            )
        unknown = [word for word in self.errors_addressed if word not in SCORE_ERRORS]
        if unknown:
            raise SettingsError(
                f"family {self.family!r}: unknown error word(s) {unknown!r}; "
                f"use {', '.join(SCORE_ERRORS)}"
            )
        if self.variant_errors is not None:
            self._check_variant_errors(self.variant_errors)
        if self.variant_notes is not None:
            self._check_variant_notes(self.variant_notes)
        if self.level == "score" and self.factory is not None and not callable(self.factory):
            raise SettingsError(
                f"score-level family {self.family!r}: the factory must be a callable factory "
                "or None"
            )
        if self.level == "estimate" and self.factory is not None:
            raise SettingsError(f"estimate-level family {self.family!r} must have factory=None")

    def _check_variant_errors(self, variant_errors: Mapping[str, tuple[str, ...]]) -> None:
        """Refuse per-variant error tuples for unknown variants or with unknown words."""
        stray = [name for name in variant_errors if name not in self.variants]
        if stray:
            raise SettingsError(
                f"family {self.family!r}: variant_errors names {stray!r}, which are not "
                f"variants; variants: {', '.join(self.variants)}"
            )
        for name, words in variant_errors.items():
            unknown = [word for word in words if word not in SCORE_ERRORS]
            if unknown:
                raise SettingsError(
                    f"family {self.family!r}, variant {name!r}: unknown error word(s) "
                    f"{unknown!r}; use {', '.join(SCORE_ERRORS)}"
                )

    def _check_variant_notes(self, variant_notes: Mapping[str, str]) -> None:
        """Refuse notes for unknown variants and notes that are not non-empty text."""
        stray = [name for name in variant_notes if name not in self.variants]
        if stray:
            raise SettingsError(
                f"family {self.family!r}: variant_notes names {stray!r}, which are not "
                f"variants; variants: {', '.join(self.variants)}"
            )
        empty = [
            name for name, note in variant_notes.items() if not isinstance(note, str) or not note
        ]
        if empty:
            raise SettingsError(
                f"family {self.family!r}: the note of variant(s) {empty!r} must be non-empty text"
            )


_REGISTRY: dict[str, FamilySpec] = {}
REGISTRY: Mapping[str, FamilySpec] = MappingProxyType(_REGISTRY)
"""The registered families by name, as a read-only mapping.

Each family module and each estimator module registers its ``FamilySpec`` at import, so the
mapping is complete once ``copernicus.corrections`` is imported; ``list_corrections`` tabulates
it, one row per variant, and ``get_correction`` builds a score-level correction from it.
"""


def register(spec: FamilySpec) -> FamilySpec:
    """Add a family to the registry.

    Args:
        spec: The family's entry.

    Returns:
        The entry, for use in a module-level assignment.

    Raises:
        ConfigurationError: If the family name is already registered.
    """
    if spec.family in _REGISTRY:
        raise ConfigurationError(f"family {spec.family!r} is already registered")
    _REGISTRY[spec.family] = spec
    logger.debug("registered correction family %r (%s)", spec.family, spec.level)
    return spec


def get_correction(family: str, variant: str, **params: object) -> ScoreCorrection:
    """Build a score-level correction from its family and variant names.

    Args:
        family: A registered family name.
        variant: One of the family's variants.
        **params: Keyword parameters for the family's factory.

    Returns:
        The correction.

    Raises:
        SettingsError: If an argument is outside its documented range: the family is unknown
            (the message lists the registered families), the variant is unknown (the message
            lists the family's variants), the family works at the estimate level, or it is a
            score-level combiner that is called as a function (the correction ensemble, blind
            transport), so that there is no per-text correction to build; or the family's
            factory refuses a parameter's value.
        TypeError: If the family's factory does not accept a parameter.
    """
    if family not in _REGISTRY:
        raise SettingsError(
            f"unknown family {family!r}; registered families: {', '.join(_REGISTRY)}"
        )
    spec = _REGISTRY[family]
    if variant not in spec.variants:
        raise SettingsError(
            f"unknown variant {variant!r} of family {family!r}; variants: "
            f"{', '.join(spec.variants)}"
        )
    if spec.factory is None and spec.level == "estimate":
        raise SettingsError(
            f"family {family!r} works at the estimate level and has no score-level correction"
        )
    if spec.factory is None:
        raise SettingsError(
            f"family {family!r} is called as a function, not fitted per text: it has no "
            "score-level correction to build"
        )
    return spec.factory(variant, **params)


def list_corrections() -> pd.DataFrame:
    """The registered families and variants, one row per variant.

    Returns:
        A frame with the columns ``family``, ``variant``, ``target``, ``level``,
        ``errors_addressed`` (the variant's score-error words joined by ``", "``) and
        ``estimate_errors``, in registration order.
    """
    rows = [
        {
            "family": spec.family,
            "variant": variant,
            "target": spec.target,
            "level": spec.level,
            "errors_addressed": ", ".join(spec.errors_for(variant)),
            "estimate_errors": spec.estimate_errors,
        }
        for spec in _REGISTRY.values()
        for variant in spec.variants
    ]
    return pd.DataFrame(rows, columns=LISTING_COLUMNS)
