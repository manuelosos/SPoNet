import pytest
import numpy as np


from sponet.simplex_utils import (
    compute_from_simplex_orthogonal_transformation_matrix,
    project_isometric_simplex_to_plane,
    simplex_grid
)

#TODO Test fertig schreiben
@pytest.mark.parametrize(
    "value, expected",
    [
        ([0, 0,1],[0,0])
    ]
)
def test_project_isometric_simplex_to_plane(value, expected):
    value= np.array(value)
    expected = np.array(expected)
    res = project_isometric_simplex_to_plane(value)
    print(res)
    assert np.allclose(project_isometric_simplex_to_plane(value), expected)
