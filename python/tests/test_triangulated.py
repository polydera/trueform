"""
Tests for triangulated function

Copyright (c) 2025 Žiga Sajovic, XLAB
"""

import sys
import numpy as np
import pytest
import trueform as tf


# Test parameters
REAL_DTYPES = [np.float32, np.float64]
INDEX_DTYPES = [np.int32, np.int64]
DIMS = [2, 3]


# ==============================================================================
# Single Polygon Tests (points array only)
# ==============================================================================

@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_single_polygon_quad_3d(dtype):
    """Triangulate a single quad (4 vertices) in 3D."""
    polygon = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0]
    ], dtype=dtype)

    faces, points = tf.triangulated(polygon)

    # 4-gon -> 2 triangles
    assert faces.shape == (2, 3), f"Expected (2, 3), got {faces.shape}"
    assert points.shape == (4, 3), f"Expected (4, 3), got {points.shape}"
    assert faces.dtype == np.int32
    assert points.dtype == dtype


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_single_polygon_pentagon_3d(dtype):
    """Triangulate a single pentagon (5 vertices) in 3D."""
    # Regular pentagon in XY plane
    angles = np.linspace(0, 2*np.pi, 5, endpoint=False)
    polygon = np.column_stack([
        np.cos(angles),
        np.sin(angles),
        np.zeros(5)
    ]).astype(dtype)

    faces, points = tf.triangulated(polygon)

    # 5-gon -> 3 triangles
    assert faces.shape == (3, 3), f"Expected (3, 3), got {faces.shape}"
    assert points.shape == (5, 3), f"Expected (5, 3), got {points.shape}"


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_single_polygon_2d(dtype):
    """Triangulate a single polygon in 2D."""
    polygon = np.array([
        [0, 0],
        [1, 0],
        [1, 1],
        [0.5, 1.5],
        [0, 1]
    ], dtype=dtype)

    faces, points = tf.triangulated(polygon)

    # 5-gon -> 3 triangles
    assert faces.shape == (3, 3), f"Expected (3, 3), got {faces.shape}"
    assert points.shape == (5, 2), f"Expected (5, 2), got {points.shape}"
    assert points.dtype == dtype


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_single_polygon_triangle(dtype):
    """Triangle input should return single triangle."""
    polygon = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0.5, 1, 0]
    ], dtype=dtype)

    faces, points = tf.triangulated(polygon)

    # Triangle -> 1 triangle
    assert faces.shape == (1, 3), f"Expected (1, 3), got {faces.shape}"
    assert points.shape == (3, 3), f"Expected (3, 3), got {points.shape}"


# ==============================================================================
# Triangle Mesh Tests (ngon=3, no-op)
# ==============================================================================

@pytest.mark.parametrize("dtype", REAL_DTYPES)
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
def test_triangle_tuple_returns_copy(dtype, index_dtype):
    """Triangle tuple should return copies of input."""
    faces_in = np.array([[0, 1, 2], [0, 2, 3]], dtype=index_dtype)
    points_in = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0]
    ], dtype=dtype)

    faces, points = tf.triangulated((faces_in, points_in))

    # Should be same shape and content
    assert faces.shape == faces_in.shape
    assert points.shape == points_in.shape
    np.testing.assert_array_equal(faces, faces_in)
    np.testing.assert_array_equal(points, points_in)

    # Should be copies, not same arrays
    assert faces is not faces_in
    assert points is not points_in


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_triangle_mesh_returns_copy(dtype):
    """Triangle Mesh should return copies of faces and points."""
    faces_in = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    points_in = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0]
    ], dtype=dtype)
    mesh = tf.Mesh(faces_in, points_in)

    faces, points = tf.triangulated(mesh)

    # Should be same shape and content
    assert faces.shape == faces_in.shape
    assert points.shape == points_in.shape
    np.testing.assert_array_equal(faces, faces_in)
    np.testing.assert_array_equal(points, points_in)


# ==============================================================================
# Quad Tuple Tests (ngon=4, converted to dynamic)
# ==============================================================================

@pytest.mark.parametrize("dtype", REAL_DTYPES)
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
def test_quad_tuple_3d(dtype, index_dtype):
    """Triangulate a quad mesh given as tuple."""
    # Single quad
    faces_in = np.array([[0, 1, 2, 3]], dtype=index_dtype)
    points_in = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0]
    ], dtype=dtype)

    faces, points = tf.triangulated((faces_in, points_in))

    # 1 quad -> 2 triangles
    assert faces.shape == (2, 3), f"Expected (2, 3), got {faces.shape}"
    assert points.shape == (4, 3), f"Expected (4, 3), got {points.shape}"
    assert faces.dtype == index_dtype
    assert points.dtype == dtype


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_quad_tuple_2d(dtype):
    """Triangulate quads in 2D."""
    # Two quads sharing an edge
    faces_in = np.array([
        [0, 1, 2, 3],
        [1, 4, 5, 2]
    ], dtype=np.int32)
    points_in = np.array([
        [0, 0],
        [1, 0],
        [1, 1],
        [0, 1],
        [2, 0],
        [2, 1]
    ], dtype=dtype)

    faces, points = tf.triangulated((faces_in, points_in))

    # 2 quads -> 4 triangles
    assert faces.shape == (4, 3), f"Expected (4, 3), got {faces.shape}"
    assert points.shape == (6, 2), f"Expected (6, 2), got {points.shape}"


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_multiple_quads(dtype):
    """Triangulate multiple quads."""
    # 3x3 grid of quads
    n = 4  # 4x4 vertices = 3x3 quads
    x, y = np.meshgrid(np.arange(n), np.arange(n))
    points_in = np.column_stack([
        x.flatten(),
        y.flatten(),
        np.zeros(n*n)
    ]).astype(dtype)

    # Create quad indices
    faces_list = []
    for i in range(n-1):
        for j in range(n-1):
            v0 = i * n + j
            v1 = i * n + j + 1
            v2 = (i + 1) * n + j + 1
            v3 = (i + 1) * n + j
            faces_list.append([v0, v1, v2, v3])
    faces_in = np.array(faces_list, dtype=np.int32)

    faces, points = tf.triangulated((faces_in, points_in))

    # 9 quads -> 18 triangles
    assert faces.shape == (18, 3), f"Expected (18, 3), got {faces.shape}"
    assert points.shape == (16, 3), f"Expected (16, 3), got {points.shape}"


# ==============================================================================
# Dynamic Mesh Tests (OffsetBlockedArray)
# ==============================================================================

@pytest.mark.parametrize("dtype", REAL_DTYPES)
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
def test_dynamic_quads(dtype, index_dtype):
    """Triangulate dynamic mesh with quads."""
    offsets = np.array([0, 4, 8], dtype=index_dtype)
    data = np.array([0, 1, 2, 3, 1, 4, 5, 2], dtype=index_dtype)
    faces_in = tf.OffsetBlockedArray(offsets, data)

    points_in = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [2, 0, 0],
        [2, 1, 0]
    ], dtype=dtype)

    faces, points = tf.triangulated((faces_in, points_in))

    # 2 quads -> 4 triangles
    assert faces.shape == (4, 3), f"Expected (4, 3), got {faces.shape}"
    assert points.shape == (6, 3), f"Expected (6, 3), got {points.shape}"


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_dynamic_mixed_ngons(dtype):
    """Triangulate mesh with mixed polygon sizes."""
    # Triangle (3) + Quad (4) + Pentagon (5)
    offsets = np.array([0, 3, 7, 12], dtype=np.int32)
    data = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], dtype=np.int32)
    faces_in = tf.OffsetBlockedArray(offsets, data)

    # 12 points for our polygons
    angles = np.linspace(0, 2*np.pi, 12, endpoint=False)
    points_in = np.column_stack([
        np.cos(angles) * np.arange(1, 13),
        np.sin(angles) * np.arange(1, 13),
        np.zeros(12)
    ]).astype(dtype)

    faces, points = tf.triangulated((faces_in, points_in))

    # Triangle (1) + Quad (2) + Pentagon (3) = 6 triangles
    assert faces.shape == (6, 3), f"Expected (6, 3), got {faces.shape}"
    assert points.shape == (12, 3), f"Expected (12, 3), got {points.shape}"


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_dynamic_2d(dtype):
    """Triangulate dynamic mesh in 2D."""
    offsets = np.array([0, 4], dtype=np.int32)
    data = np.array([0, 1, 2, 3], dtype=np.int32)
    faces_in = tf.OffsetBlockedArray(offsets, data)

    points_in = np.array([
        [0, 0],
        [1, 0],
        [1, 1],
        [0, 1]
    ], dtype=dtype)

    faces, points = tf.triangulated((faces_in, points_in))

    assert faces.shape == (2, 3), f"Expected (2, 3), got {faces.shape}"
    assert points.shape == (4, 2), f"Expected (4, 2), got {points.shape}"


# ==============================================================================
# Correctness Tests
# ==============================================================================

@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_triangulation_covers_all_area(dtype):
    """Triangulated polygons should cover the same area as original."""
    # Unit square
    polygon = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0]
    ], dtype=dtype)

    faces, points = tf.triangulated(polygon)

    # Compute total area of triangles
    total_area = 0.0
    for face in faces:
        v0, v1, v2 = points[face[0]], points[face[1]], points[face[2]]
        edge1 = v1 - v0
        edge2 = v2 - v0
        area = 0.5 * np.linalg.norm(np.cross(edge1, edge2))
        total_area += area

    # Unit square area = 1.0
    assert abs(total_area - 1.0) < 1e-5, f"Expected area 1.0, got {total_area}"


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_all_indices_valid(dtype):
    """All face indices should be valid point indices."""
    polygon = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0.5, 1.5, 0],
        [0, 1, 0]
    ], dtype=dtype)

    faces, points = tf.triangulated(polygon)

    # All indices should be in range [0, num_points)
    assert np.all(faces >= 0), "Negative indices found"
    assert np.all(faces < len(points)), "Out of bounds indices found"


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_triangles_non_degenerate(dtype):
    """Resulting triangles should be non-degenerate (have positive area)."""
    # L-shaped polygon
    polygon = np.array([
        [0, 0, 0],
        [2, 0, 0],
        [2, 1, 0],
        [1, 1, 0],
        [1, 2, 0],
        [0, 2, 0]
    ], dtype=dtype)

    faces, points = tf.triangulated(polygon)

    for i, face in enumerate(faces):
        v0, v1, v2 = points[face[0]], points[face[1]], points[face[2]]
        edge1 = v1 - v0
        edge2 = v2 - v0
        area = 0.5 * np.linalg.norm(np.cross(edge1, edge2))
        assert area > 1e-10, f"Triangle {i} is degenerate (area={area})"


# ==============================================================================
# Type Preservation Tests
# ==============================================================================

@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_single_polygon_dtype_preserved(dtype):
    """Single polygon output dtype should match input."""
    polygon = np.array([[0, 0], [1, 0], [0.5, 1], [0, 0.5]], dtype=dtype)
    faces, points = tf.triangulated(polygon)
    assert points.dtype == dtype


@pytest.mark.parametrize("dtype", REAL_DTYPES)
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
def test_tuple_dtype_preserved(dtype, index_dtype):
    """Tuple output dtypes should match input."""
    faces_in = np.array([[0, 1, 2, 3]], dtype=index_dtype)
    points_in = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=dtype)

    faces, points = tf.triangulated((faces_in, points_in))

    assert faces.dtype == index_dtype, f"Expected {index_dtype}, got {faces.dtype}"
    assert points.dtype == dtype, f"Expected {dtype}, got {points.dtype}"


# ==============================================================================
# Edge Cases
# ==============================================================================

@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_large_polygon(dtype):
    """Triangulate a polygon with many vertices."""
    n = 100
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    polygon = np.column_stack([
        np.cos(angles),
        np.sin(angles),
        np.zeros(n)
    ]).astype(dtype)

    faces, points = tf.triangulated(polygon)

    # n-gon -> n-2 triangles
    expected_triangles = n - 2
    assert faces.shape == (expected_triangles, 3), \
        f"Expected ({expected_triangles}, 3), got {faces.shape}"


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_convex_polygon(dtype):
    """Triangulate a convex polygon (hexagon)."""
    n = 6
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    polygon = np.column_stack([
        np.cos(angles),
        np.sin(angles),
        np.zeros(n)
    ]).astype(dtype)

    faces, points = tf.triangulated(polygon)

    # 6-gon -> 4 triangles
    assert faces.shape == (4, 3), f"Expected (4, 3), got {faces.shape}"

    # Verify area (regular hexagon with unit radius has area 3*sqrt(3)/2)
    total_area = 0.0
    for face in faces:
        v0, v1, v2 = points[face[0]], points[face[1]], points[face[2]]
        edge1 = v1 - v0
        edge2 = v2 - v0
        total_area += 0.5 * np.linalg.norm(np.cross(edge1, edge2))

    expected_area = 3 * np.sqrt(3) / 2
    assert abs(total_area - expected_area) < 1e-5, \
        f"Expected area {expected_area}, got {total_area}"


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
