"""Settings of the correction battery and the entries they expand to.

An entry names one score-level correction to compare: a family, one of its variants and,
optionally, keyword parameters for the family's factory (the encoder of embedding-augmented
calibration, the number of imputation draws). ``families="all"`` expands to every variant of
every registered score-level family that is fitted per text, in registry order, followed by
the four variants of the correction ensemble, which pools the corrected columns of the other
entries. The estimate-level families and blind transport (a cross-corpus function) are not
part of the battery. Under ``"all"`` every entry keeps its factory's defaults, but
``run_battery`` fits the embedding-augmented entries to the data's embedding sets
(``for_embedding_sets``); list per-encoder entries with an ``encoder`` parameter.

Sizes are integers of at least 2 (texts to draw) or shares in ``(0, 0.5]`` of the texts of a
training pool; schemes are ``kfold<k>`` (grouped k-fold, ``k >= 2``) or ``holdout`` (one
grouped 80/20 split).
"""

from __future__ import annotations

import inspect
import logging
import numbers
import re
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Literal

from copernicus.corrections._messages import list_first_ten
from copernicus.corrections.families import REGISTRY, FamilySpec, get_correction
from copernicus.corrections.families._embedding_blocks import MIN_STACKED_SETS
from copernicus.errors import SettingsError
from copernicus.validity.anatomy import COVARIATE_SCALES

logger = logging.getLogger(__name__)

__all__ = [
    "ALL",
    "ENSEMBLE",
    "HOLDOUT",
    "SCHEMES",
    "SEEDS",
    "SIZES",
    "BatteryEntry",
    "BatterySettings",
    "FamilyEntry",
    "scheme_folds",
    "size_label",
]

ALL: Literal["all"] = "all"
ENSEMBLE = "correction_ensemble"
EMBEDDING_FAMILY = "embedding_augmented"
ENCODER = "encoder"
HOLDOUT = "holdout"
SIZES: tuple[int | float, ...] = (50, 100, 250, 500, 1000, 0.01, 0.05, 0.10, 0.25)
SCHEMES: tuple[str, ...] = ("kfold5", "kfold10", HOLDOUT)
SEEDS: tuple[int, ...] = (42, 43, 44)
MIN_SIZE = 2
MAX_SHARE = 0.5
MIN_FOLDS = 2
MIN_ENSEMBLE_MEMBERS = 2
KFOLD = re.compile(r"kfold([1-9][0-9]*)")
KEYWORD_KINDS = (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)

FamilyEntry = tuple[str, str] | tuple[str, str, Mapping[str, object]]


@dataclass(frozen=True)
class BatteryEntry:
    """One correction the battery compares.

    Attributes:
        family: A registered score-level family.
        variant: One of its variants.
        params: Keyword parameters for the family's factory as ``(name, value)`` pairs sorted
            by name; empty for the factory's defaults.
    """

    family: str
    variant: str
    params: tuple[tuple[str, object], ...] = ()

    @property
    def label(self) -> str:
        """The variant, followed by its parameters in parentheses when it has any.

        The tables of the battery name an entry by its family and this label, so two
        entries of one variant with different parameters stay apart.
        """
        if not self.params:
            return self.variant
        return f"{self.variant}({','.join(f'{name}={value}' for name, value in self.params)})"

    @property
    def key(self) -> str:
        """``family:label``; the name of the entry among the members of the ensemble."""
        return f"{self.family}:{self.label}"

    @property
    def kwargs(self) -> dict[str, object]:
        """The parameters as keyword arguments of the family's factory."""
        return dict(self.params)

    @property
    def is_ensemble(self) -> bool:
        """Whether the entry is a variant of the correction ensemble."""
        return self.family == ENSEMBLE


def size_label(size: int | float) -> str:
    """The label of a size in the tables: the digits of a count, the decimal of a share."""
    return str(size) if isinstance(size, float) else str(int(size))


def scheme_folds(scheme: str) -> int | None:
    """The number of folds of a scheme.

    Args:
        scheme: ``kfold<k>`` or ``holdout``.

    Returns:
        ``k`` for ``kfold<k>``, ``None`` for the holdout split.

    Raises:
        SettingsError: If the scheme is neither, or ``k`` is below 2.
    """
    if scheme == HOLDOUT:
        return None
    match = KFOLD.fullmatch(scheme) if isinstance(scheme, str) else None
    if match is None or int(match.group(1)) < MIN_FOLDS:
        raise SettingsError(
            f"each scheme must be kfold<k> with k of at least 2, or holdout; got {scheme!r}"
        )
    return int(match.group(1))


def _sequence(values: object, name: str, what: str) -> tuple[object, ...]:
    """The values of a non-string sequence setting, refusing an empty one."""
    if isinstance(values, str | bytes) or not isinstance(values, Sequence):
        raise SettingsError(f"{name} must be a sequence of {what}; got {values!r}")
    if not values:
        raise SettingsError(f"give at least one {name[:-1]}")
    return tuple(values)


def _checked_size(size: object) -> int | float:
    """A count of at least 2 as an int, or a share in ``(0, 0.5]`` as a float."""
    if isinstance(size, numbers.Integral) and not isinstance(size, bool):
        if int(size) >= MIN_SIZE:
            return int(size)
    elif isinstance(size, numbers.Real) and 0.0 < float(size) <= MAX_SHARE:
        return float(size)
    raise SettingsError(
        "each size must be an integer of at least 2 (texts to draw) or a share in (0, 0.5] of "
        f"the texts of a training pool; got {size!r}"
    )


def _distinct(values: tuple[object, ...], name: str) -> None:
    """Refuse a setting whose values repeat."""
    if len(set(values)) != len(values):
        raise SettingsError(f"{name} must not repeat; got {values!r}")


def _checked_seeds(values: tuple[object, ...]) -> tuple[int, ...]:
    """Distinct integer seeds."""
    seeds: list[int] = []
    for seed in values:
        if not isinstance(seed, numbers.Integral) or isinstance(seed, bool):
            raise SettingsError(f"seeds must be integers; got {values!r}")
        seeds.append(int(seed))
    _distinct(tuple(seeds), "seeds")
    return tuple(seeds)


def _accepted_parameters(factory: Callable[..., object]) -> set[str] | None:
    """The keyword parameters of a factory besides the variant; ``None`` when it takes any."""
    parameters = inspect.signature(factory).parameters.values()
    if any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters):
        return None
    return {
        parameter.name
        for parameter in parameters
        if parameter.kind in KEYWORD_KINDS and parameter.name != "variant"
    }


def _checked_params(spec: FamilySpec, params: object) -> tuple[tuple[str, object], ...]:
    """The parameters of an entry, checked by name against its family's factory."""
    if not isinstance(params, Mapping):
        raise SettingsError(
            f"the parameters of an entry of {spec.family!r} must be a mapping of keyword "
            f"arguments; got {params!r}"
        )
    names = [str(name) for name in params]
    if spec.factory is None:
        if names:
            raise SettingsError(
                f"the correction ensemble takes no parameters; got {list_first_ten(names)}"
            )
        return ()
    accepted = _accepted_parameters(spec.factory)
    unknown = [] if accepted is None else sorted(set(names) - accepted)
    if unknown:
        raise SettingsError(
            f"family {spec.family!r} takes no parameter(s) {list_first_ten(unknown)}; its "
            f"parameters: {list_first_ten(sorted(accepted or set())) or 'none'}"
        )
    unhashable = sorted(
        str(name) for name, value in params.items() if not isinstance(value, Hashable)
    )
    if unhashable:
        raise SettingsError(
            f"the parameter(s) {list_first_ten(unhashable)} of {spec.family!r} must be numbers, "
            "text or other hashable values"
        )
    return tuple(sorted((str(name), value) for name, value in params.items()))


def _entry(item: object) -> BatteryEntry:
    """One entry from a ``(family, variant)`` or ``(family, variant, params)`` tuple."""
    if isinstance(item, str) or not isinstance(item, Sequence) or len(item) not in (2, 3):
        raise SettingsError(
            "each entry must be a (family, variant) or (family, variant, params) tuple; "
            f"got {item!r}"
        )
    family, variant = str(item[0]), str(item[1])
    if family not in REGISTRY:
        raise SettingsError(
            f"unknown family {family!r}; registered families: {', '.join(REGISTRY)}"
        )
    spec = REGISTRY[family]
    if variant not in spec.variants:
        raise SettingsError(
            f"unknown variant {variant!r} of family {family!r}; variants: "
            f"{', '.join(spec.variants)}"
        )
    if spec.level == "estimate":
        raise SettingsError(
            f"family {family!r} works at the estimate level; the battery compares score-level "
            "corrections"
        )
    if spec.factory is None and family != ENSEMBLE:
        raise SettingsError(
            f"family {family!r} is called as a cross-corpus function and is not part of the battery"
        )
    entry = BatteryEntry(family, variant, _checked_params(spec, item[2] if len(item) == 3 else {}))
    if not entry.is_ensemble:
        get_correction(family, variant, **entry.kwargs)
    return entry


def _all_entries() -> tuple[BatteryEntry, ...]:
    """Every per-text score-level variant in registry order, then the ensemble's variants."""
    per_text = [
        BatteryEntry(spec.family, variant)
        for spec in REGISTRY.values()
        if spec.level == "score" and spec.factory is not None
        for variant in spec.variants
    ]
    ensemble = [BatteryEntry(ENSEMBLE, variant) for variant in REGISTRY[ENSEMBLE].variants]
    return (*per_text, *ensemble)


def _checked_families(
    families: object,
) -> tuple[Literal["all"] | tuple[FamilyEntry, ...], tuple[BatteryEntry, ...]]:
    """The families setting as stored, and the entries it expands to."""
    if isinstance(families, str) and families == ALL:
        return ALL, _all_entries()
    if isinstance(families, str) or not isinstance(families, Sequence):
        raise SettingsError(
            "families must be 'all' or a sequence of (family, variant) or (family, variant, "
            f"params) entries; got {families!r}"
        )
    if not families:
        raise SettingsError("families must name at least one (family, variant) entry, or 'all'")
    entries = tuple(_entry(item) for item in families)
    keys = [entry.key for entry in entries]
    repeated = sorted({key for key in keys if keys.count(key) > 1})
    if repeated:
        raise SettingsError(f"families names {list_first_ten(repeated)} twice")
    n_per_text = sum(not entry.is_ensemble for entry in entries)
    if n_per_text < MIN_ENSEMBLE_MEMBERS and any(entry.is_ensemble for entry in entries):
        raise SettingsError(
            "the correction ensemble pools the corrected columns of the other entries and needs "
            f"at least two per-text entries; families lists {n_per_text}"
        )
    return _stored(entries), entries


def _stored(entries: Sequence[BatteryEntry]) -> tuple[FamilyEntry, ...]:
    """The entries as ``(family, variant)`` or ``(family, variant, params)`` tuples."""
    return tuple(
        (entry.family, entry.variant, entry.kwargs)
        if entry.params
        else (entry.family, entry.variant)
        for entry in entries
    )


@dataclass(frozen=True)
class BatterySettings:
    """Settings of the correction battery.

    Attributes:
        families: ``"all"``, or the entries to compare: ``(family, variant)`` tuples, or
            ``(family, variant, params)`` tuples whose params go to the family's factory
            (``get_correction(family, variant, **params)``).
        sizes: Labelled sample sizes: integers of at least 2 (texts) and shares in
            ``(0, 0.5]`` of the texts of each training pool.
        schemes: ``kfold<k>`` (grouped folds) and ``holdout`` (one grouped 80/20 split whose
            training share must lie in 0.75 to 0.85).
        seeds: Seeds of the partitions, the draws and the fits; distinct integers.
        covariate_scale: ``standardized`` (the covariate-linked error per standard deviation
            of the covariate) or ``raw`` (per unit).
        entries: The entries ``families`` expands to (derived, not an argument): per-text
            entries first in the order given, ensemble entries after them.
    """

    families: Sequence[FamilyEntry] | Literal["all"] = ALL
    sizes: Sequence[int | float] = SIZES
    schemes: Sequence[str] = SCHEMES
    seeds: Sequence[int] = SEEDS
    covariate_scale: str = "standardized"
    entries: tuple[BatteryEntry, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Check every setting against its range and expand the families.

        Raises:
            SettingsError: If a setting is outside its documented range: an unknown family or
                variant, an estimate-level family or blind transport, a parameter the
                factory does not take or refuses, a repeated entry, an ensemble without two
                per-text entries, a size that is neither a count of at least 2 nor a share in
                ``(0, 0.5]``, an unknown scheme, repeated or non-integer seeds, or an unknown
                covariate scale.
        """
        sizes = tuple(
            _checked_size(size) for size in _sequence(self.sizes, "sizes", "counts and shares")
        )
        _distinct(sizes, "sizes")
        schemes = tuple(str(scheme) for scheme in _sequence(self.schemes, "schemes", "names"))
        for scheme in schemes:
            scheme_folds(scheme)
        _distinct(schemes, "schemes")
        seeds = _checked_seeds(_sequence(self.seeds, "seeds", "integers"))
        if self.covariate_scale not in COVARIATE_SCALES:
            raise SettingsError(
                f"covariate_scale must be one of {', '.join(COVARIATE_SCALES)}; "
                f"got {self.covariate_scale!r}"
            )
        families, entries = _checked_families(self.families)
        per_text = tuple(entry for entry in entries if not entry.is_ensemble)
        object.__setattr__(self, "families", families)
        object.__setattr__(self, "sizes", sizes)
        object.__setattr__(self, "schemes", schemes)
        object.__setattr__(self, "seeds", seeds)
        object.__setattr__(
            self, "entries", (*per_text, *(entry for entry in entries if entry.is_ensemble))
        )

    def for_embedding_sets(self, names: Sequence[str]) -> BatterySettings:
        """The settings with ``"all"`` fitted to the data's embedding sets.

        Under ``"all"`` the embedding-augmented entries read the ``stacked`` set of two or more
        sets; with one set they read it (``encoder=<name>``), without any they are left out
        (logged once at INFO).

        Args:
            names: The encoder names of the data's embedding sets, in order.

        Returns:
            These settings, or (``"all"`` with fewer than two sets) settings listing its entries.
        """
        if self.families != ALL or len(names) >= MIN_STACKED_SETS:
            return self
        if not names:
            logger.info("the data carry no embedding set, so 'all' leaves out %s", EMBEDDING_FAMILY)
            kept = [entry for entry in self.entries if entry.family != EMBEDDING_FAMILY]
            return replace(self, families=_stored(kept))
        encoder = ((ENCODER, str(names[0])),)
        on_the_set = [
            BatteryEntry(entry.family, entry.variant, encoder)
            if entry.family == EMBEDDING_FAMILY
            else entry
            for entry in self.entries
        ]
        return replace(self, families=_stored(on_the_set))

    @property
    def per_text_entries(self) -> tuple[BatteryEntry, ...]:
        """The entries fitted per text, in order."""
        return tuple(entry for entry in self.entries if not entry.is_ensemble)

    @property
    def ensemble_entries(self) -> tuple[BatteryEntry, ...]:
        """The variants of the correction ensemble, combined after the per-text entries."""
        return tuple(entry for entry in self.entries if entry.is_ensemble)
