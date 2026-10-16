"""Geometric invariants for nearest-surface contact mapping."""

import numpy as np
import pytest
import pyvista as pv

from anatplot.anatomy import _project_contacts


def triangle(points):
    return pv.PolyData.from_regular_faces(np.asarray(points, dtype=float), [[0, 1, 2]])


def test_nearest_triangle_mapping_preserves_barycentric_position_and_row_order():
    source = triangle([[0, 0, 0], [2, 0, 0], [0, 2, 0]])
    target = triangle([[10, 0, 2], [14, 0, 2], [10, 6, 2]])
    points = np.array([[0.5, 0.5, -3], [3, 0, 4]], dtype=float)
    original = points.copy()
    cells, closest = source.find_closest_cell(points, return_closest_point=True)
    actual = _project_contacts(source, target, cells, closest, offset=0.25)
    np.testing.assert_allclose(actual, [[11, 1.5, 2.25], [14, 0, 2.25]])
    np.testing.assert_array_equal(points, original)
    np.testing.assert_array_equal(source.points, [[0, 0, 0], [2, 0, 0], [0, 2, 0]])


def test_offset_follows_display_normal_instead_of_source_normal():
    source = triangle([[0, 0, 0], [2, 0, 0], [0, 2, 0]])
    target = triangle([[10, 0, 0], [10, 2, 0], [10, 0, 2]])
    np.testing.assert_allclose(
        _project_contacts(
            source, target, np.array([0]), np.array([[0.5, 0.5, 0]]), offset=0.25
        ),
        [[10.25, 0.5, 0.5]],
    )


def test_thin_float32_source_triangle_preserves_barycentric_position():
    source = pv.PolyData.from_regular_faces(
        np.array([[0, 0, 0], [1, 0, 0], [1, 1e-4, 0]], dtype=np.float32),
        [[0, 1, 2]],
    )
    target = triangle([[10, 0, 2], [11, 0, 2], [11, 2, 2]])
    actual = _project_contacts(
        source, target, np.array([0]), np.array([[0.8, 5e-5, 0]]), offset=0
    )
    np.testing.assert_allclose(actual, [[10.8, 1, 2]], atol=1e-7)


def test_degenerate_display_triangle_is_rejected():
    source = triangle([[0, 0, 0], [2, 0, 0], [0, 2, 0]])
    target = triangle([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
    with pytest.raises(ValueError, match="nondegenerate display"):
        _project_contacts(
            source, target, np.array([0]), np.array([[0.5, 0.5, 0]]), offset=0.25
        )


def test_degenerate_source_triangle_is_rejected():
    source = triangle([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
    target = triangle([[0, 0, 0], [0, 1, 0], [0, 0, 1]])
    with pytest.raises(ValueError, match="nondegenerate source"):
        _project_contacts(
            source, target, np.array([0]), np.array([[0.5, 0, 0]]), offset=0.25
        )
