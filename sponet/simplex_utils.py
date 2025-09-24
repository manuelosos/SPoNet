import numpy as np
from numba import njit
from numpy.typing import NDArray


@njit(cache=True)
def compute_from_simplex_orthogonal_transformation_matrix(dim: int) -> NDArray:
    # vectors e_k - e_{k+1}
    if dim == 3:
        return np.column_stack(
            (np.array([1, -1, 0]) / np.sqrt(2), np.array([1, 1, -2]) / np.sqrt(6))
        )
    vecs = np.eye(dim)[:-1] - np.eye(dim)[1:]
    trans_matrix, _ = np.linalg.qr(vecs.T)
    return trans_matrix[:, :dim - 1]


@njit(cache=True)
def project_isometric_simplex_to_plane(x: NDArray, trans_matrix: NDArray | None = None) -> NDArray:
    dim = x.shape[-1]
    barycentric_center = np.ones(dim) / dim
    if trans_matrix is None:
        trans_matrix = compute_from_simplex_orthogonal_transformation_matrix(dim)
    return (trans_matrix.T @ (x - barycentric_center).T).T


@njit(cache=True)
def binomial(n: int, r: int) -> int:
    if r < 0 or r > n:
        return 0
    if r == 0 or r == n:
        return 1
    # compute combinatorially, safe for small sizes
    rr = r if r <= n - r else n - r
    res = 1
    for i in range(rr):
        res = (res * (n - i)) // (i + 1)
    return res


@njit(cache=True)
def simplex_grid(dim: int, resolution: float) -> np.ndarray:
    """
    Uniform grid on the standard M-simplex using spacing ~ resolution.
    Non-recursive, Numba-friendly.
    Returns an array of shape (N, M) with rows summing to 1.
    """
    if dim <= 1:
        raise ValueError("M must be > 1")
    # convert resolution to integer denominator K (round nearest)

    n_points_per_unit = max(1, int(1 / resolution + 0.5))

    nSlots = n_points_per_unit + dim - 1  # total slots in stars-and-bars
    r = dim - 1  # number of separators (bars)

    N = binomial(n_points_per_unit + dim - 1, dim - 1)
    pts = np.empty((N, dim), dtype=np.float64)

    # initialize first combination s = [0,1,2,...,r-1] (zero-based positions)
    s = np.empty(r, dtype=np.int64)
    for i in range(r):
        s[i] = i

    idx = 0
    while True:
        # compute composition from separators s:
        # n0 = s[0]
        # ni = s[i] - s[i-1] - 1  for i=1..r-1
        # n_{M-1} = (nSlots - 1) - s[r-1]
        for j in range(dim):
            if j == 0:
                compj = s[0]
            elif j == dim - 1:
                compj = (nSlots - 1) - s[r - 1]
            else:
                compj = s[j] - s[j - 1] - 1
            pts[idx, j] = compj / n_points_per_unit
        idx += 1

        # generate next combination in lexicographic order
        i = r - 1
        while i >= 0 and s[i] == (nSlots - r + i):
            i -= 1
        if i < 0:
            break
        s[i] += 1
        for j in range(i + 1, r):
            s[j] = s[j - 1] + 1

    return pts
