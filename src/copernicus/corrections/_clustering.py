"""Cluster-robust variances of the prediction-powered estimators.

``draw_human_sample`` draws whole groups, so its labelled texts form a cluster sample: the texts
of a group share the person or the conversation behind them, and their machine errors tend to
move together. Treating them as independent draws then gives intervals that are far too narrow.
The prediction-powered estimators therefore sum the influence contributions ``psi`` of the texts
within each group before the variance. For each of their two terms (the rectifier on the
labelled texts and the prediction on every text, or on the texts outside the human sample for
the Ludwig variant)::

    Cov(mean of psi) = G / (G - 1) * sum_g S_g S_g' / n^2,   S_g = sum_{i in g} (psi_i - mean psi)

with ``G`` the groups and ``n`` the texts of the term. The rows are centred at the term's own
mean before the group sums, which is the linearised variance of a mean over groups of unequal
sizes; with one text per group it is ``Cov(psi) / n``, the independent form.

The option ``cluster`` of the estimators is ``None`` by default: the estimate is clustered by
``group_id`` when any group holds more than one labelled text and treats the texts as
independent otherwise. ``True`` and ``False`` (numpy booleans too) force the choice. Clustering
needs at least two groups among the texts of each term. A difference of two group means needs
labelled texts from at least two groups on each side as well: each side is centred at its own
mean, so the contributions of a side whose labelled texts lie in one group sum to exactly zero
within it, and the side adds nothing to the variance. ``group_difference_ppi`` refuses such a
difference whenever it clusters, under the automatic choice too, rather than report a standard
error without that side's variation between groups; ``cluster=False`` gives the independent
standard error, which ignores the correlation of the texts within a group.

The interval stays the normal ``point +/- 1.96 SE``, and the variance of the labelled term rests
on as many group sums as there are drawn groups (for a difference, the drawn groups on each of
its sides), so with few drawn groups the interval covers less than 95 percent. In simulations
with groups of ten texts sharing a machine error, the interval of a mean covered about 0.88
with five drawn groups, 0.92 with ten and 0.96 with twenty (and 0.59 with twenty when the texts
were treated as independent).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from copernicus.corrections.data import CorrectionData
from copernicus.errors import EstimatorError, SettingsError

logger = logging.getLogger(__name__)

__all__ = [
    "MIN_CLUSTERS",
    "Clustering",
    "checked_cluster",
    "contributions",
    "mean_covariance",
    "resolve_cluster",
]

MIN_CLUSTERS = 2


@dataclass(frozen=True, eq=False)
class Clustering:
    """The clustering an estimate uses.

    Attributes:
        requested: The ``cluster`` option as given (``None`` for the automatic choice).
        groups: The groups of the texts of the base term (every text, or the texts outside the
            human sample), or ``None`` when the texts are treated as independent.
        labelled_groups: The groups of the labelled texts, or ``None`` likewise.
    """

    requested: bool | None
    groups: np.ndarray | None
    labelled_groups: np.ndarray | None

    @property
    def clustered(self) -> bool:
        """Whether the variance sums the contributions within groups."""
        return self.groups is not None

    def details(self) -> dict[str, object]:
        """``cluster`` (as requested), ``clustered`` and, when clustered, the group counts."""
        found: dict[str, object] = {"cluster": self.requested, "clustered": self.clustered}
        if self.groups is not None and self.labelled_groups is not None:
            found["n_clusters"] = int(np.unique(self.groups).size)
            found["n_clusters_labelled"] = int(np.unique(self.labelled_groups).size)
        return found


def checked_cluster(cluster: bool | None) -> bool | None:
    """The ``cluster`` option as ``None`` or a Python ``bool``.

    Args:
        cluster: ``True``, ``False`` (a numpy boolean as well) or ``None``.

    Returns:
        ``None``, ``True`` or ``False``.

    Raises:
        SettingsError: If ``cluster`` is neither ``None`` nor a boolean; an integer such as 1
            is refused.
    """
    if cluster is None:
        return None
    if not isinstance(cluster, bool | np.bool_):
        raise SettingsError(f"cluster must be True, False or None (automatic); got {cluster!r}")
    return bool(cluster)


def resolve_cluster(
    data: CorrectionData,
    labelled: np.ndarray,
    cluster: bool | None,
    *,
    base: np.ndarray | None = None,
    split: np.ndarray | None = None,
) -> Clustering:
    """Decide whether an estimate clusters by group, and with which groups.

    Args:
        data: The correction data.
        labelled: Boolean mask of the labelled texts.
        cluster: ``True`` or ``False`` (a numpy boolean as well) to force the choice; ``None``
            to cluster when a group holds more than one labelled text.
        base: Boolean mask of the texts of the base term; ``None`` for every text.
        split: For a difference, the boolean mask of its first side in data order, with
            labelled texts on both sides; ``None`` for an estimate without sides.

    Returns:
        The clustering, with the groups of the base and of the labelled texts when it clusters.

    Raises:
        SettingsError: If ``cluster`` is neither ``None`` nor a boolean.
        EstimatorError: If the estimate clusters and the base or the labelled texts hold fewer
            than two groups, or the labelled texts of a side of ``split`` lie in one group.
    """
    requested = checked_cluster(cluster)
    labelled_groups = data.groups[labelled]
    shared = int(np.unique(labelled_groups).size) < labelled_groups.size
    if not (shared if requested is None else requested):
        return Clustering(requested=requested, groups=None, labelled_groups=None)
    groups = data.groups if base is None else data.groups[base]
    counts = (int(np.unique(groups).size), int(np.unique(labelled_groups).size))
    if min(counts) < MIN_CLUSTERS:
        raise EstimatorError(
            "clustering needs at least two groups among the texts of each term and among the "
            f"labelled texts; there are {counts[0]} and {counts[1]} (pass cluster=False to treat "
            "the texts as independent)"
        )
    if split is not None:
        _require_groups_on_each_side(labelled_groups, split[labelled])
    logger.debug("clustering by %d groups, %d of them labelled", counts[0], counts[1])
    return Clustering(requested=requested, groups=groups, labelled_groups=labelled_groups)


def _require_groups_on_each_side(labelled_groups: np.ndarray, labelled_split: np.ndarray) -> None:
    """Refuse a clustered difference whose labelled texts on one side lie in a single group.

    Each side's contributions are centred at the side's own mean, so those of a side held by one
    group sum to zero within it and the side adds nothing to the clustered variance. Both sides
    hold labelled texts (``group_difference_ppi`` checks two on each side first).

    Raises:
        EstimatorError: Naming the side, its group and the remedies.
    """
    for flag, side in ((True, labelled_split), (False, ~labelled_split)):
        found = np.unique(labelled_groups[side])
        if found.size < MIN_CLUSTERS:
            raise EstimatorError(
                "a clustered difference needs labelled texts from at least two groups on each "
                f"side; the labelled texts where the grouping is {flag} all lie in the group "
                f"{str(found[0])!r}, so the clustered standard error would leave out that "
                "side's variation between groups: draw more groups, or pass cluster=False "
                "(--no-cluster on the command line) to treat the texts as independent, with a "
                "standard error that ignores the correlation of the texts within a group"
            )


def mean_covariance(psi: np.ndarray, groups: np.ndarray | None) -> np.ndarray:
    """The covariance of the mean of the rows of ``psi``.

    Args:
        psi: One row of contributions per text.
        groups: The group of every row, or ``None`` for independent rows.

    Returns:
        ``Cov(psi) / n`` (sample covariance), or with groups ``Cov(S) * G / n^2`` over the group
        sums ``S`` of the rows centred at their mean.
    """
    n_rows = psi.shape[0]
    if groups is None:
        return np.atleast_2d(np.cov(psi, rowvar=False, ddof=1)) / n_rows
    labels, inverse = np.unique(groups, return_inverse=True)
    sums = np.zeros((labels.size, psi.shape[1]))
    np.add.at(sums, inverse, psi - psi.mean(axis=0))
    return np.atleast_2d(np.cov(sums, rowvar=False, ddof=1)) * labels.size / n_rows**2


def contributions(values: np.ndarray, split: np.ndarray | None) -> np.ndarray:
    """The influence contribution of every text to a mean or a difference of two group means.

    Args:
        values: One value per text.
        split: ``None`` for the mean, or a boolean mask of the first group.

    Returns:
        ``v - mean(v)`` for the mean; for the difference ``n / n_a (v - mean_a)`` in the first
        group and ``-n / n_b (v - mean_b)`` in the second, so that the mean of the contributions
        has the variance of the estimate.
    """
    if split is None:
        return values - values.mean()
    n = values.size
    result = np.empty(n)
    for mask, sign in ((split, 1.0), (~split, -1.0)):
        part = values[mask]
        result[mask] = sign * (part - part.mean()) * n / part.size
    return result
