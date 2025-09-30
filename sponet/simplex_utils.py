import numpy as np
from numba import njit
from numpy.typing import NDArray


@njit(cache=True)
def map_to_simplex_facette(x: NDArray, facette_index: int) -> NDArray:
    """
    Maps a vector of dim M to the facette of the standard M+2 simplex with x_{facette_index}=0.

    facette_index starts at 0.
    Map is not an isometry.

    Parameters
    ----------
    x: NDArray
    facette_index: int

    Returns
    -------
    NDArray
        Shape = (x.shape[0]+2,)

    """
    if np.sum(x) > 1:
        raise ValueError(
            "Invalid value for x."
            "Sum of components of of x is larger than 1. x is outside the dim(x)-unit-simplex."
        )

    res = np.empty(x.shape[0] + 2)

    res[facette_index] = 0
    if facette_index == res.shape[0] - 1:
        res[0] = 1 - np.sum(x)
        res[1:-1] = x.copy()
        return res

    res[facette_index + 1] = 1 - np.sum(x)
    res[:facette_index] = x[:facette_index].copy()
    res[facette_index + 2 :] = x[facette_index:].copy()

    return res

    """
    Way cooler code that works with numpy but nUmBa dOeS nOt sUpPoRt nP.iNsErT
        if facette_index == x.shape[0] + 1:
            return np.insert([1 - np.sum(x), 0], 1, x)

        if facette_index == x.shape[0]:
            return np.insert([0, 1 - np.sum(x)], 0, x)

        return np.insert(x, facette_index, [0, 1 - np.sum(x)])
    """


@njit(cache=True)
def map_from_simplex_facette(x: NDArray, facette_index: int) -> NDArray:
    """
    Maps a vector from the facette_index-th facette of the M-dim standard-simplex to M-2-dim unit-simplex.

    facette-index starts at 0.
    Map is not an isometry.

    Parameters
    ----------
    x: NDArray
    facette_index: int

    Returns
    -------
    NDArray
        Shape = (x.shape[0]-2,)
    """
    if not np.isclose(x[facette_index], 0):
        raise ValueError("Invalid value for x. x[facette_index] must be 0!")

    if facette_index == x.shape[0] - 1:
        res = x[1:-1].copy()
        return res

    res = np.empty(x.shape[0] - 2)
    res[:facette_index] = x[:facette_index].copy()
    res[facette_index:] = x[facette_index + 2 :].copy()
    return res
