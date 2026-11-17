"""Reduction of a text-by-reference matrix to one score per text.

scikit-learn is imported by the first principal-component reduction, not with the package,
so that importing copernicus stays fast. The principal component is computed by an exact
decomposition, never by the randomized SVD (see `_pca_solver`), so two reductions of one
matrix give the same scores.
"""

from __future__ import annotations

import logging

import numpy as np

from copernicus._versions import read_version
from copernicus.errors import ConfigurationError

logger = logging.getLogger(__name__)

__all__ = ["REDUCTIONS", "reduce_matrix"]

REDUCTIONS = ("mean", "max", "pca")
"""The reductions of the cosines to the references of a pole: ``mean``, ``max`` and ``pca``."""
MINIMUM_ROWS_FOR_PCA = 2
COVARIANCE_SOLVER_FROM = (1, 5)
COVARIANCE_MAX_COLUMNS = 1_000
COVARIANCE_ROWS_PER_COLUMN = 10
BLOCK_ROWS = 8_192
COMPARISON_BYTES = 8 * 1024 * 1024
DIGEST_SEED = 0


def _pca_solver(n_rows: int, n_columns: int, label: str) -> str:
    """The exact route that scikit-learn's automatic choice takes for one component, if it has one.

    For one component, ``svd_solver="auto"`` of scikit-learn 1.5 to 1.9 takes
    ``covariance_eigh`` for a matrix of at most 1,000 columns and at least ten rows per column,
    the full SVD when neither dimension exceeds 500, and otherwise the randomized SVD, which
    draws its projection from numpy's global random stream, so that two calls on one matrix
    differ. Versions 1.3 and 1.4 have no ``covariance_eigh`` and take the full SVD up to 500
    rows and columns, the randomized SVD beyond. This choice keeps each exact route of the
    automatic one and takes the full SVD where that would be randomized: the scores of the
    decomposition are deterministic, and bitwise those of the automatic choice wherever it was
    exact.

    Raises:
        ConfigurationError: If the version of the installed scikit-learn cannot be read whole
            by the package parser (`copernicus._versions.read_version`); the message says to
            install a released version.
    """
    import sklearn

    version = read_version(sklearn.__version__)
    if version is None:
        raise ConfigurationError(
            f"reduction of {label}: cannot read the scikit-learn version "
            f"{sklearn.__version__!r}; the exact solver of the pca reduction depends on it: "
            "install a released scikit-learn of version 1.3 or later, whose version starts "
            "with its major and minor number (pip install --upgrade scikit-learn)"
        )
    has_covariance = version.release[:2] >= COVARIANCE_SOLVER_FROM
    tall = n_columns <= COVARIANCE_MAX_COLUMNS and n_rows >= COVARIANCE_ROWS_PER_COLUMN * n_columns
    return "covariance_eigh" if has_covariance and tall else "full"


def _row_words(matrix: np.ndarray) -> np.ndarray:
    """Return the bits of the entries of a matrix as unsigned integers, one row per row.

    An entry of 1, 2, 4 or 8 bytes becomes one unsigned integer of that size, in a view of the
    matrix in any memory order. An entry of another size becomes its single bytes, from a
    C-ordered copy of a matrix that is not C-ordered.
    """
    words = matrix
    if matrix.dtype.itemsize not in (1, 2, 4, 8):
        words = np.ascontiguousarray(matrix).view(np.uint8).reshape(matrix.shape[0], -1)
    return words.view(np.dtype(f"u{words.dtype.itemsize}"))


def _row_digest(words: np.ndarray) -> np.ndarray:
    """Return one unsigned integer per row of `_row_words` that rows with equal words share.

    Each word is multiplied by a fixed odd weight of its column and the products of a row are
    summed modulo 2**64, `BLOCK_ROWS` rows at a time. Integer arithmetic is exact, so equal rows
    get equal digests whatever the order of the sum. Two rows that differ can share a digest,
    which costs a comparison of their words, never a wrong group.
    """
    generator = np.random.default_rng(DIGEST_SEED)
    weights = generator.integers(0, 2**63, size=words.shape[1], dtype=np.uint64) * 2 + 1
    digest = np.empty(words.shape[0], dtype=np.uint64)
    for start in range(0, words.shape[0], BLOCK_ROWS):
        block = words[start : start + BLOCK_ROWS].astype(np.uint64, copy=False)
        digest[start : start + BLOCK_ROWS] = block @ weights
    return digest


def _first_equal_row(matrix: np.ndarray) -> np.ndarray:
    """Return, for every row, the index of the first row whose bytes equal its own.

    Two rows are equal when their entries have the same bits, so the rule needs no tolerance:
    a row that differs in one bit, or holds ``-0.0`` where another holds ``0.0``, is a row of
    its own. Only the rows whose digest (`_row_digest`) another row shares are compared. Each
    is compared with the first row of its digest. Each block has at most `BLOCK_ROWS` rows and
    at most half the matrix's rows, with a minimum of one row. The two copies and their
    Boolean comparison fit within `COMPARISON_BYTES` unless comparing a single row already
    needs more bytes. A row that differs from its digest's first row is grouped by its bytes
    with the other such rows.

    Args:
        matrix: A two-dimensional non-object numeric array with at least one row and one
            column.

    Returns:
        Array of shape ``[N]``: for each row, the smallest index among the rows equal to it,
        which is its own index when no earlier row equals it.
    """
    words = _row_words(matrix)
    first = np.arange(words.shape[0])
    digest = _row_digest(words)
    _, earliest, inverse, counts = np.unique(
        digest, return_index=True, return_inverse=True, return_counts=True
    )
    shared = np.flatnonzero(counts[inverse] > 1)
    candidate = earliest[inverse[shared]]
    equal = np.empty(shared.size, dtype=bool)
    bytes_per_row = words.shape[1] * (2 * words.dtype.itemsize + 1)
    block = max(1, min(BLOCK_ROWS, words.shape[0] // 2, COMPARISON_BYTES // bytes_per_row))
    for start in range(0, shared.size, block):
        part = slice(start, start + block)
        equal[part] = (words[shared[part]] == words[candidate[part]]).all(axis=1)
    first[shared[equal]] = candidate[equal]
    seen: dict[bytes, int] = {}
    for row in shared[~equal]:
        first[row] = seen.setdefault(words[row].tobytes(), int(row))
    return first


def _first_component(matrix: np.ndarray, label: str) -> np.ndarray:
    """Return the first principal component of the standardised columns, signed.

    The component comes from the exact decomposition that `_pca_solver` names. Its sign is
    anchored before equal rows are grouped: the decomposition scores are flipped when their
    correlation with the row means is negative. Grouping keeps that sign; near a zero
    correlation, rounding can leave the returned scores with a slightly negative correlation.

    Rows of the matrix with equal bytes then get one score: the score of the first of them.
    The full SVD returns the scores as the
    first left singular vector times the first singular value, and the covariance route
    projects the standardised rows on the component by a matrix product; neither routine is
    bound to give equal rows equal results, so their scores could differ in the last bits,
    and a rank correlation would not see them as tied. The comparison is of the bytes of the
    rows as passed in, not of the standardised copy (`_first_equal_row`). A row that no
    earlier row equals keeps the score of the decomposition bitwise.

    Raises:
        ConfigurationError: If the matrix has fewer than two rows, no column varies, the
            scikit-learn version cannot be read, or the decomposition fails or yields
            non-finite scores.
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    n_rows = matrix.shape[0]
    if n_rows < MINIMUM_ROWS_FOR_PCA:
        raise ConfigurationError(
            f"reduction of {label}: pca needs at least {MINIMUM_ROWS_FOR_PCA} texts, got {n_rows}"
        )
    if not np.any(matrix.var(axis=0) > 0):
        raise ConfigurationError(
            f"reduction of {label}: pca is undefined because every reference column is constant"
        )
    pca = PCA(n_components=1, svd_solver=_pca_solver(n_rows, matrix.shape[1], label))
    try:
        scores = pca.fit_transform(StandardScaler().fit_transform(matrix))[:, 0]
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
        raise ConfigurationError(f"reduction of {label}: pca failed: {error}") from error
    if not np.all(np.isfinite(scores)):
        raise ConfigurationError(f"reduction of {label}: pca produced non-finite scores")
    row_means = matrix.mean(axis=1)
    if np.std(scores) > 0 and np.std(row_means) > 0 and np.corrcoef(scores, row_means)[0, 1] < 0:
        scores = -scores
    return scores[_first_equal_row(matrix)]


def reduce_matrix(mat: np.ndarray, method: str, label: str) -> np.ndarray:
    """Reduce a matrix with one row per text and one column per reference to one score per text.

    A matrix with a single column reduces to that column whatever the method; that is the
    correct reduction of one reference, not a fallback.

    Args:
        mat: Matrix of shape ``[N, K]``: cosines to ``K`` references, or the scores of
            ``K`` attributes in a roll-up.
        method: ``"mean"``, ``"max"``, or ``"pca"`` (first principal component of the
            standardised columns by an exact decomposition, signed by the correlation of the
            scores of the decomposition with the row means, before equal rows are grouped;
            rows whose bytes are equal get one score, that of the first of them).
        label: What is being reduced, named in every error message.

    Returns:
        Array of shape ``[N]``.

    Raises:
        ConfigurationError: If the method is unknown, the matrix is not two-dimensional,
            has no column, or holds a NaN or infinite value, or pca is asked of fewer than
            two texts or of columns without any variance, or with a scikit-learn whose
            version cannot be read.
    """
    if method not in REDUCTIONS:
        raise ConfigurationError(
            f"reduction of {label}: unknown reduction {method!r}; use one of {REDUCTIONS}"
        )
    if mat.ndim != 2:
        raise ConfigurationError(
            f"reduction of {label}: expected a two-dimensional matrix, got shape {mat.shape}"
        )
    if mat.shape[1] == 0:
        raise ConfigurationError(f"reduction of {label}: the matrix has no column")
    if not np.all(np.isfinite(mat)):
        raise ConfigurationError(f"reduction of {label}: the matrix holds a NaN or infinite value")
    if mat.shape[1] == 1:
        return mat[:, 0]
    if method == "mean":
        return mat.mean(axis=1)
    if method == "max":
        return mat.max(axis=1)
    return _first_component(mat, label)
