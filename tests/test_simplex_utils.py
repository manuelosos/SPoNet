import pytest
import numpy as np


from sponet.simplex_utils import (
    compute_from_simplex_orthogonal_transformation_matrix,
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
