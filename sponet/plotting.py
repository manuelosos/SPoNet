import numpy as np
from matplotlib.axes import Axes
from numba import njit, prange

from .collective_variables import CollectiveVariable
from .parameters import Parameters
from .cnvm.parameters import CNVMParameters
from .multiprocessing import sample_many_runs
from sponet.cnvm.approximations.chemical_langevin_equation import _drift_and_diffusion
from .simplex_utils import (
    project_isometric_simplex_to_plane,
    generate_uniform_simplex_grid,
    compute_from_simplex_orthogonal_transformation_matrix,
)
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
    n_points_per_length_unit: int = 15,
    show_only_coordinate: int = -1,
):
    """
    Visualizes the vector field given by the RRE/MFE on the simplex.

    Only implemented for n_opinions = 3.


    Parameters
    ----------
    params: CNVMParameters
    n_points_per_length_unit: int
        Determines how many points there are on a length of one unit.
    ax: Axes

    Returns
    -------
    Axes
    """
    n_states = params.num_opinions

    anchor_points = generate_uniform_simplex_grid(3, n_points_per_length_unit)
    trans_matrix = compute_from_simplex_orthogonal_transformation_matrix(n_states)

    projected_anchor_points = project_isometric_simplex_to_plane(
        anchor_points, trans_matrix
    )

    r = params.r
    r_tilde = params.r_tilde

    @njit(cache=True, parallel=True)
    def compute_anchor_vectors():
        vectors = np.empty_like(anchor_points)

        for i in prange(anchor_points.shape[0]):
            vectors[i, :], _ = _drift_and_diffusion(
                anchor_points[i], r, r_tilde, 10
            )  # TODO Use own MFE function

        if show_only_coordinate != -1:
            vector_mask = np.zeros_like(vectors)
            vector_mask[:, show_only_coordinate] = 1
            vectors[np.logical_not(vector_mask)] = 0

        return vectors

    projected_anchor_vectors = project_isometric_simplex_to_plane(
        compute_anchor_vectors(), trans_matrix
    )

    ax.quiver(
        projected_anchor_points[:, 0],
        projected_anchor_points[:, 1],
        projected_anchor_vectors[:, 0],
        projected_anchor_vectors[:, 1],
    )

    # Plot triangle frame
    unit_vectors = np.vstack((np.eye(n_states), np.eye(n_states)[0]))
    projected_unit_vectors = project_isometric_simplex_to_plane(
        unit_vectors, trans_matrix
    )
    ax.plot(projected_unit_vectors[:, 0], projected_unit_vectors[:, 1], zorder=-1)

    # Label triangle sides
    sides = [
        (projected_unit_vectors[1], projected_unit_vectors[2], "1"),  # v1=0
        (projected_unit_vectors[0], projected_unit_vectors[2], "2"),  # v2=0
        (projected_unit_vectors[0], projected_unit_vectors[1], "3"),  # v3=0
    ]

    offset = 0.05  # Distance from label to triangle side

    for p1, p2, label in sides:
        midpoint = (p1 + p2) / 2
        edge_vec = p2 - p1
        normal = np.array([-edge_vec[1], edge_vec[0]])
        normal /= np.linalg.norm(normal)

        centroid = projected_unit_vectors.mean(axis=0)
        if np.dot(midpoint + normal * offset - centroid, normal) < 0:
            normal = -normal

        mp_shifted = midpoint + offset * normal
        ax.text(
            mp_shifted[0],
            mp_shifted[1],
            label,
            ha="center",
            va="center",
            fontsize=12,
            color="blue",
        )

    ax.set_aspect("equal")
    return ax
