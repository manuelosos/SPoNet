import pytest
import numpy as np


from sponet.simplex_utils import (
    project_isometric_simplex_to_plane,
    generate_uniform_simplex_grid,
)


# TODO Test fertig schreiben
@pytest.mark.parametrize(
    "value, expected",
    [
        ([1 / 3, 1 / 3, 1 / 3], [0, 0]),
        (
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            [
                [7.07106781e-01, 4.08248290e-01],
                [-7.07106781e-01, 4.08248290e-01],
                [-3.83224528e-18, -8.16496581e-01],
            ],
        ),
    ],
)
def test_project_isometric_simplex_to_plane(value, expected):
    value = np.array(value)
    expected = np.array(expected)
    res = project_isometric_simplex_to_plane(value)
    assert np.allclose(res, expected)


@pytest.mark.parametrize(
    "dim, n_points_per_length_unit, expected",
    [
        (
            2,
            5,
            np.array([[0.0, 1.0], [0.25, 0.75], [0.5, 0.5], [0.75, 0.25], [1.0, 0.0]]),
        ),
        (
            3,
            4,
            np.array(
                [
                    [0.0, 0.0, 1.0],
                    [0.0, 1 / 3, 2 / 3],
                    [0.0, 2 / 3, 1 / 3],
                    [0.0, 1.0, 0.0],
                    [1 / 3, 0.0, 2 / 3],
                    [1 / 3, 1 / 3, 1 / 3],
                    [1 / 3, 2 / 3, 0.0],
                    [2 / 3, 0.0, 1 / 3],
                    [2 / 3, 1 / 3, 0.0],
                    [1.0, 0.0, 0.0],
                ]
            ),
        ),
        (
            4,
            3,
            np.array(
                [
                    [0.0, 0.0, 0.0, 1.0],
                    [0.0, 0.0, 0.5, 0.5],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.5, 0.0, 0.5],
                    [0.0, 0.5, 0.5, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.5, 0.0, 0.0, 0.5],
                    [0.5, 0.0, 0.5, 0.0],
                    [0.5, 0.5, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                ]
            ),
        ),
    ],
)
def test_generate_uniform_simplex_grid(dim, n_points_per_length_unit, expected):
    grid = generate_uniform_simplex_grid(dim, n_points_per_length_unit)
    assert np.allclose(np.sum(grid, axis=-1), 1)
    assert np.allclose(grid, expected)
