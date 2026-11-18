"""The rules of the one-factor iteration that its records rest on: a tie, a cap of rounds.

``top_two_tied`` decides whether the two largest of some eigenvalues (or singular values) are
equal up to a relative ``TIE_RELATIVE_TOLERANCE``; ``paf_fit`` returns the loadings of
``paf_one_factor`` with the number of rounds, whether they converged and whether a round met a
tie.
"""

from __future__ import annotations

import numpy as np
import pytest
from traces_support import strong_columns

from copernicus.corrections.families._factor import (
    PAF_MAX_ITER,
    TIE_RELATIVE_TOLERANCE,
    paf_fit,
    paf_one_factor,
    top_two_tied,
)

pytestmark = pytest.mark.filterwarnings("error")


def test_the_tie_rule_compares_the_gap_with_the_largest_eigenvalue() -> None:
    # Exact binary numbers: a gap of 2**-40 (about 9.1e-13) of the largest value is a tie, a gap
    # of 2**-39 (about 1.8e-12) is not.
    assert TIE_RELATIVE_TOLERANCE == 1e-12
    assert top_two_tied(np.array([0.25, 1.0 - 2.0**-40, 1.0]))
    assert not top_two_tied(np.array([0.25, 1.0 - 2.0**-39, 1.0]))
    assert top_two_tied(8.0 * np.array([0.25, 1.0 - 2.0**-40, 1.0]))
    assert not top_two_tied(8.0 * np.array([0.25, 1.0 - 2.0**-39, 1.0]))
    assert top_two_tied(np.array([3.0, 3.0]))
    assert not top_two_tied(np.array([5.0]))


@pytest.mark.parametrize(("off_diagonal", "tied"), [(4e-16, True), (6e-16, False)])
def test_the_iteration_applies_the_tie_rule_to_its_reduced_matrix(
    off_diagonal: float, tied: bool
) -> None:
    # The start puts 0.001 on the diagonal: the eigenvalues 0.001 +/- r differ by about 8e-13
    # of the largest for r = 4e-16 and by about 1.2e-12 for r = 6e-16.
    correlation = np.array([[1.0, off_diagonal], [off_diagonal, 1.0]])
    fit = paf_fit(correlation, n_iter=1)
    assert fit.direction_unique is not tied
    assert fit.rounds == 1
    assert np.array_equal(fit.loadings, paf_one_factor(correlation, n_iter=1))


def test_a_lowered_cap_of_rounds_is_recorded() -> None:
    columns, _ = strong_columns()
    matrix = np.column_stack(list(columns.values()))
    standardised = (matrix - matrix.mean(axis=0)) / matrix.std(axis=0)
    correlation = standardised.T @ standardised / standardised.shape[0]
    capped, full = paf_fit(correlation, n_iter=2), paf_fit(correlation)
    assert (capped.converged, capped.rounds) == (False, 2)
    assert full.converged and 2 < full.rounds < PAF_MAX_ITER and full.direction_unique
    assert np.array_equal(full.loadings, paf_one_factor(correlation))
    assert np.array_equal(capped.loadings, paf_one_factor(correlation, n_iter=2))
    unrun = paf_fit(correlation, n_iter=0)
    assert (unrun.converged, unrun.rounds) == (False, 0) and not unrun.loadings.any()


def test_an_iteration_that_meets_its_tolerance_in_its_last_allowed_round_has_converged() -> None:
    # Three strong columns, whose iteration meets no tie: with a cap of exactly the rounds that
    # it needs, the tolerance is met in the last allowed round; with one round less it is not.
    columns, _ = strong_columns()
    matrix = np.column_stack(list(columns.values()))
    standardised = (matrix - matrix.mean(axis=0)) / matrix.std(axis=0)
    correlation = standardised.T @ standardised / standardised.shape[0]
    full = paf_fit(correlation)
    assert full.converged and full.direction_unique and full.rounds > 2
    short, last, spare = (paf_fit(correlation, n_iter=full.rounds + more) for more in (-1, 0, 1))
    assert (short.rounds, short.converged) == (full.rounds - 1, False)
    assert (last.rounds, last.converged) == (full.rounds, True)
    assert (spare.rounds, spare.converged) == (full.rounds, True)
    assert np.array_equal(last.loadings, full.loadings)
    assert np.array_equal(spare.loadings, full.loadings)


def test_the_first_round_on_an_identity_matrix_is_recorded_as_tied() -> None:
    # Round 1 puts 0.001 on the diagonal of the identity matrix: two equal eigenvalues, whatever
    # eigenvector the decomposition returns for them.
    fit = paf_fit(np.eye(2))
    assert fit.tied_rounds[0] == 1 and not fit.direction_unique
    assert not paf_fit(np.eye(2), n_iter=1).direction_unique
