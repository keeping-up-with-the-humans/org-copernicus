"""Checks of the repeated study: its settings, and the draws its cross-fitted estimator refuses."""

from __future__ import annotations

import logging
import numbers
from collections.abc import Iterable, Sequence

import numpy as np

from copernicus.corrections._clustering import MIN_CLUSTERS, Clustering, resolve_cluster
from copernicus.corrections._estimate_inputs import require_labelled, require_labelled_sides
from copernicus.corrections.data import CorrectionData
from copernicus.corrections.estimates import MIN_LABELLED_PER_GROUP
from copernicus.errors import EstimatorError, SettingsError

logger = logging.getLogger(__name__)

__all__ = [
    "MIN_SIZE",
    "SIDES",
    "checked_estimators",
    "checked_sizes",
    "crossfit_clustering",
    "require_groups_on_each_side",
]

MIN_SIZE = 2
# The two sides of the long-short difference: the value of the split and how a message names it.
SIDES = (
    (True, "the long side (the texts at or above the covariate's median)"),
    (False, "the short side (the texts below the covariate's median)"),
)


def checked_sizes(sizes: Sequence[int], n_units: int) -> tuple[int, ...]:
    """Distinct sizes, each an integer between ``MIN_SIZE`` and the number of texts.

    Raises:
        SettingsError: If ``sizes`` is not a sequence of distinct integers in that range.
    """
    if isinstance(sizes, str | bytes) or not isinstance(sizes, Iterable):
        raise SettingsError(f"sizes must be a sequence of integers; got {sizes!r}")
    values = tuple(sizes)
    if not values:
        raise SettingsError("sizes must name at least one size")
    for size in values:
        if isinstance(size, bool) or not isinstance(size, numbers.Integral):
            raise SettingsError(f"each size must be an integer; got {size!r}")
        if not MIN_SIZE <= size <= n_units:
            raise SettingsError(
                f"each size must lie between {MIN_SIZE} and the {n_units} texts; got {size}"
            )
    if len(set(values)) != len(values):
        raise SettingsError(f"sizes must be distinct; got {values!r}")
    return tuple(int(size) for size in values)


def checked_estimators(estimators: Sequence[str], known: Sequence[str]) -> tuple[str, ...]:
    """Distinct estimator names among ``known``.

    Raises:
        SettingsError: If the names are not distinct names among ``known``.
    """
    values = () if isinstance(estimators, str) else tuple(estimators)
    if not values or len(set(values)) != len(values) or not set(values) <= set(known):
        raise SettingsError(
            f"estimators must be distinct names among {', '.join(known)}; got {estimators!r}"
        )
    return values


def require_groups_on_each_side(groups: np.ndarray, split: np.ndarray) -> None:
    """Refuse a clustered long-short difference whose texts on one side lie in a single group.

    The clustered standard error of the cross-fitted pseudo-values centres the contributions of
    a side at the side's mean over all its texts and sums them within each group. When every
    text of a side lies in one group those contributions sum to zero within it, and the side
    would add none of its variation to the standard error.

    Args:
        groups: The group of every text of the corpus.
        split: The long side of the difference, in the same order.

    Raises:
        EstimatorError: Naming the side, its group and the remedies.
    """
    for flag, side in SIDES:
        found = np.unique(groups[split == flag])
        if found.size < MIN_CLUSTERS:
            raise EstimatorError(
                "a clustered long-short difference needs texts from at least two groups on each "
                f"side; every text of {side} lies in the group {str(found[0])!r}, so their "
                "contributions cancel within it and that side would add nothing to the "
                "clustered standard error: use a covariate that puts at least two groups on "
                "each side, or pass cluster=False to treat the texts as independent, with a "
                "standard error that ignores the correlation of the texts within a group"
            )


def crossfit_clustering(
    data: CorrectionData, split: np.ndarray | None, cluster: bool | None
) -> Clustering:
    """Refuse a draw as ``ppi_identity`` refuses it, where the reason holds for pseudo-values.

    The checks run before the pseudo-values are cross-fitted, in the order of ``mean_ppi`` and
    ``group_difference_ppi`` and with their messages:

    * at least two labelled texts, and for a difference at least two on each side, under every
      setting of ``cluster`` (the floor ``MIN_LABELLED_PER_GROUP`` of both): a side without
      labelled texts would keep its uncorrected predictions as its pseudo-values;
    * when the draw clusters (``resolve_cluster``), at least two groups among the texts and
      among the labelled texts.

    Two reasons of ``ppi_identity`` do not hold for pseudo-values. It needs one ``pi`` on every
    labelled text, while the pseudo-values divide each labelled text by its own ``pi``
    (``pseudo_values`` refuses one that is missing or not finite), and every draw of the study
    gives its texts one ``pi`` anyway. It refuses a clustered difference whose labelled texts
    on a side lie in one group, while the pseudo-value column covers every text, so such a side
    still adds the variation of its other groups. A side whose texts all lie in one group of
    the corpus adds nothing, and a clustered difference is refused for it instead
    (``require_groups_on_each_side``).

    Args:
        data: The draw's correction data, gold on the labelled texts.
        split: The long side of a difference, or ``None`` for the mean.
        cluster: The study's ``cluster`` option.

    Returns:
        The clustering of the draw's standard error.

    Raises:
        EstimatorError: As described, with the reason.
    """
    labelled = data.labelled_mask
    if split is None:
        require_labelled(
            int(labelled.sum()), MIN_LABELLED_PER_GROUP, "the cross-fitted prediction-powered mean"
        )
    else:
        counts = (int((labelled & split).sum()), int((labelled & ~split).sum()))
        require_labelled_sides(
            counts, MIN_LABELLED_PER_GROUP, "the cross-fitted prediction-powered difference"
        )
    clustering = resolve_cluster(data, labelled, cluster)
    if clustering.groups is not None and split is not None:
        require_groups_on_each_side(clustering.groups, split)
    return clustering
