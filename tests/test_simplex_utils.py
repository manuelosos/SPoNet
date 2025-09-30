import pytest
import numpy as np

from sponet.simplex_utils import map_to_simplex_facette, map_from_simplex_facette


@pytest.mark.parametrize(
    "value, facette_index, expected",
    [
        ([0.2], 0, [0, 0.8, 0.2]),
        ([0.7], 1, [0.7, 0, 0.3]),
        ([0.5], 2, [0.5, 0.5, 0]),
        ([0.1, 0.2], 0, [0, 0.7, 0.1, 0.2]),
        ([0.4, 0.5], 1, [0.4, 0, 0.1, 0.5]),
        ([1 / 3, 1 / 3], 2, [1 / 3, 1 / 3, 0, 1 / 3]),
        ([1 / 3, 1 / 3], 3, [1 / 3, 1 / 3, 1 / 3, 0]),
    ],
)
def test_map_to_simplex_facette(value, facette_index, expected):
    value = np.array(value)
    expected = np.array(expected)

    result = map_to_simplex_facette(value, facette_index)
    assert np.isclose(np.sum(result), 1.0)
    assert float(result[facette_index]) == 0.0
    assert np.allclose(result, expected)


@pytest.mark.parametrize(
    "expected, facette_index, value",
    [
        ([0.2], 0, [0, 0.8, 0.2]),
        ([0.7], 1, [0.7, 0, 0.3]),
        ([0.5], 2, [0.5, 0.5, 0]),
        ([0.1, 0.2], 0, [0, 0.7, 0.1, 0.2]),
        ([0.4, 0.5], 1, [0.4, 0, 0.1, 0.5]),
        ([1 / 3, 1 / 3], 2, [1 / 3, 1 / 3, 0, 1 / 3]),
        ([1 / 3, 1 / 3], 3, [1 / 3, 1 / 3, 1 / 3, 0]),
    ],
)
def test_map_from_simplex_facette(value, facette_index, expected):
    value = np.array(value)
    expected = np.array(expected)

    result = map_from_simplex_facette(value, facette_index)
    assert np.allclose(result, expected)
