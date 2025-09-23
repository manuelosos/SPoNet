import numpy as np
from matplotlib.axes import Axes
from numba import njit

from .collective_variables import CollectiveVariable
from .parameters import Parameters
from .cnvm.parameters import CNVMParameters
from .multiprocessing import sample_many_runs
from sponet.cnvm.approximations.chemical_langevin_equation import _drift_and_diffusion

from numpy.typing import NDArray
import matplotlib.pyplot as plt


def plot_trajectories(
    params: Parameters,
    t: float,
    cv: CollectiveVariable,
    ax: Axes,
    num_initial_states: int = 3,
    samples_per_state: int = 5,
) -> Axes:
    """
    Plot some trajectories with uniformly random initial states
    in the provided Axes object.

    The purpose of this function is to quickly inspect how trajectories
    look like for the given set of Parameters.

    Parameters
    ----------
    params : Parameters
    t : float
    cv : CollectiveVariable
    ax : Axes
    num_initial_states : int, optional
    samples_per_state : int, optional

    Returns
    -------
    Axes
    """
    initial_states = np.random.randint(
        0, params.num_opinions, size=(num_initial_states, params.num_agents)
    )

    t, c = sample_many_runs(
        params,
        initial_states,
        t,
        1000,
        samples_per_state,
        collective_variable=cv,
    )

    colors = ["k", "b", "g", "c", "r", "y"]
    linestyles = ["-", "--", "-."]

    for i in range(num_initial_states):
        this_linestyle = linestyles[i % len(linestyles)]
        for j in range(samples_per_state):
            this_color = colors[j % len(colors)]
            ax.plot(t, c[i, j, :, 0], linestyle=this_linestyle, color=this_color)

    ax.set_xlabel("$t$")
    ax.set_ylabel("$c_0$")
    ax.grid()

    return ax


def visualize_mfe_vector_field(
    params: CNVMParameters,
    ax: Axes,
    resolution: int = 15,
):
    """
    Visualizes the vector field given by the RRE/MFE on the simplex.

    Only implemented for n_opinions = 3.


    Parameters
    ----------
    params: CNVMParameters
    resolution: int
        Determines how many points there are on a length of one unit.
    ax: Axes

    Returns
    -------
    Axes
    """
    r = params.r
    r_tilde = params.r_tilde
    n_states = r.shape[0]

    anchor_points = simplex_grid(3, 1 / resolution)

    # TODO Write fast function for isometric projection
    trans_matrix = np.column_stack(
        [np.array([1, -1, 0]) / np.sqrt(2), np.array([1, 1, -2]) / np.sqrt(6)]
    )  # 3x2 matrix
    bary = np.array([1 / 3, 1 / 3, 1 / 3])

    projected_anchor_points = np.empty((anchor_points.shape[0], n_states - 1))
    projected_anchor_vectors = np.empty_like(projected_anchor_points)
    projected_anchor_points = (trans_matrix.T @ (anchor_points - bary).T).T

    for i in range(anchor_points.shape[0]):
        tmp, _ = _drift_and_diffusion(
            anchor_points[i], r, r_tilde, 10
        )  # TODO Use own MFE function
        projected_anchor_vectors[i] = trans_matrix.T @ (tmp - bary)

    ax.quiver(
        projected_anchor_points[:, 0],
        projected_anchor_points[:, 1],
        projected_anchor_vectors[:, 0],
        projected_anchor_vectors[:, 1],
    )

    # Plot triangle frame
    unit_vectors = np.vstack((np.eye(n_states), np.eye(n_states)[0]))
    projected_unit_vectors = (trans_matrix.T @ (unit_vectors - bary).T).T
    ax.plot(projected_unit_vectors[:, 0], projected_unit_vectors[:, 1], zorder=-1)

    ax.set_aspect("equal")

    return ax


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
