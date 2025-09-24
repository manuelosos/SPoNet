import numpy as np
from numba import njit
from numpy.typing import NDArray


@njit(cache=True)
def compute_from_simplex_orthogonal_transformation_matrix(d: int) -> NDArray:
    """
    Computes the orthogonal tranformation matrix that isometrically maps
    the d-dimensional standard simplex to the d-1 plane.

    Parameters
    ----------
    d: int

    Returns
    -------
    NDArray
    """
    if d == 3:
        return np.column_stack(
            (np.array([1, -1, 0]) / np.sqrt(2), np.array([1, 1, -2]) / np.sqrt(6))
        )

    # vectors e_k - e_{k+1}
    vecs = np.eye(d)[:-1] - np.eye(d)[1:]
    trans_matrix, _ = np.linalg.qr(vecs.T)
    return trans_matrix[:, : d - 1]


@njit(cache=True)
def project_isometric_simplex_to_plane(
    x: NDArray, trans_matrix: NDArray | None = None
) -> NDArray:
    """
    Projects the vectors from the d-dim standard simplex to the d-1 plane.

    Parameters
    ----------
    x: NDArray
        Single vector with shape (d,)
        or multiple vectors with shape (num_vectors, d).

    trans_matrix: optional NDArray
        Orthogonal transformation matrix of shape (d,d).
        If not provided, compute with compute_from_simplex_orthogonal_transformation_matrix().
        Precompute this matrix and pass as argument to avoid unnecessary computation.

    Returns
    -------
    NDArray
        shape=(d-1,) if single vector was passed
        shape=(num_vectors, d-1) if multiple vectors were passed.

    """
    dim = x.shape[-1]
    barycentric_center = np.ones(dim) / dim
    if trans_matrix is None:
        trans_matrix = compute_from_simplex_orthogonal_transformation_matrix(dim)
    return (trans_matrix.T @ (x - barycentric_center).T).T


# TODO Write test
@njit(cache=True)
def generate_uniform_simplex_grid(dim: int, n_points_per_length_unit: float) -> NDArray:
    """
    Generates a uniform grid on the d-dim standard simplex.

    Parameters
    ----------
    dim: int
    n_points_per_length_unit: int
        Determines how many points are in one length unit.

    Returns
    -------
    NDArray
        shape = (binomial(n_points_per_length_unit+dim-1, dim-1), dim)
    """
    if dim <= 1:
        raise ValueError("dim must be > 1")
    if n_points_per_length_unit <= 1:
        raise ValueError("n_points_per_length_unit must be > 1")

    n_slots = n_points_per_length_unit + dim - 1  # total slots in stars-and-bars
    n_bars = dim - 1  # number of bars

    n_points = _binomial(n_slots, n_bars)
    grid = np.empty((n_points, dim))

    # initialize first combination bar_positions = [0,1,2,...,n_bars-1] (zero-based positions)
    bar_positions = np.empty(n_bars)
    for i in range(n_bars):
        bar_positions[i] = i

    point_index = 0
    while True:

        grid[point_index, :] = np.ediff1d(
            bar_positions,
            to_end=(n_slots - 1) - bar_positions[n_bars - 1],
            to_begin=bar_positions[0],
        )
        grid[point_index, 1:-1] -= 1
        point_index += 1

        i = n_bars - 1
        # check if bar is already at rightmost position
        while i >= 0 and bar_positions[i] == (n_slots - n_bars + i):
            i -= 1
        if i < 0:  # finished if all bars are in rightmost position
            break

        # generate next combination in lexicographic order
        bar_positions[i] += 1
        for j in range(i + 1, n_bars):
            bar_positions[j] = bar_positions[j - 1] + 1

    return grid / n_points_per_length_unit


@njit(cache=True)
def _binomial(n: int, r: int) -> int:
    """
    Computes the binomial coefficient of n over r.

    This function exists because numba does not support any already existing numpy or standard library solution.

    Parameters
    ----------
    n: int
    r:int

    Returns
    -------
    int
    """
    if r < 0 or r > n:
        return 0
    if r == 0 or r == n:
        return 1
    rr = r if r <= n - r else n - r
    res = 1
    for i in range(rr):
        res = (res * (n - i)) // (i + 1)
    return int(res)
