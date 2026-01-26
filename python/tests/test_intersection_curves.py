"""
Tests for intersection_curves between two meshes

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/trueform
"""

import sys
import numpy as np
import pytest
import trueform as tf


# Type combinations for intersection_curves (3D only)
INDEX_DTYPES = [np.int32, np.int64]
REAL_DTYPES = [np.float32, np.float64]
MESH_TYPES = ['triangle', 'dynamic']  # triangles and dynamic (variable-size)


# ==============================================================================
# Helper functions for creating test meshes
# ==============================================================================

def create_horizontal_plane_triangles(index_dtype, real_dtype, z=0.0):
    """Create a horizontal plane (z=constant) with triangles"""
    faces = np.array([
        [0, 1, 4],
        [0, 4, 3],
        [1, 2, 5],
        [1, 5, 4]
    ], dtype=index_dtype)
    points = np.array([
        [0, 0, z], [1, 0, z], [2, 0, z],
        [0, 1, z], [1, 1, z], [2, 1, z]
    ], dtype=real_dtype)
    return tf.Mesh(faces, points)


def create_horizontal_plane_dynamic(index_dtype, real_dtype, z=0.0):
    """Create a horizontal plane (z=constant) with dynamic (variable-size) polygons"""
    # Mixed: one quad and two triangles
    offsets = np.array([0, 4, 7, 10], dtype=index_dtype)
    data = np.array([0, 1, 4, 3, 1, 2, 5, 1, 5, 4], dtype=index_dtype)
    faces = tf.OffsetBlockedArray(offsets, data)
    points = np.array([
        [0, 0, z], [1, 0, z], [2, 0, z],
        [0, 1, z], [1, 1, z], [2, 1, z]
    ], dtype=real_dtype)
    return tf.Mesh(faces, points)


def create_vertical_plane_triangles(index_dtype, real_dtype, x=0.5):
    """Create a vertical plane (x=constant) with triangles"""
    faces = np.array([
        [0, 1, 4],
        [0, 4, 3],
        [1, 2, 5],
        [1, 5, 4]
    ], dtype=index_dtype)
    points = np.array([
        [x, 0, 0], [x, 1, 0], [x, 2, 0],
        [x, 0, 1], [x, 1, 1], [x, 2, 1]
    ], dtype=real_dtype)
    return tf.Mesh(faces, points)


def create_vertical_plane_dynamic(index_dtype, real_dtype, x=0.5):
    """Create a vertical plane (x=constant) with dynamic (variable-size) polygons"""
    # Mixed: one quad and two triangles
    offsets = np.array([0, 4, 7, 10], dtype=index_dtype)
    data = np.array([0, 1, 4, 3, 1, 2, 5, 1, 5, 4], dtype=index_dtype)
    faces = tf.OffsetBlockedArray(offsets, data)
    points = np.array([
        [x, 0, 0], [x, 1, 0], [x, 2, 0],
        [x, 0, 1], [x, 1, 1], [x, 2, 1]
    ], dtype=real_dtype)
    return tf.Mesh(faces, points)


# ==============================================================================
# Basic functionality tests
# ==============================================================================

@pytest.mark.parametrize("index_dtype0", INDEX_DTYPES)
@pytest.mark.parametrize("index_dtype1", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
@pytest.mark.parametrize("mesh_type0", MESH_TYPES)
@pytest.mark.parametrize("mesh_type1", MESH_TYPES)
def test_intersection_curves_basic(index_dtype0, index_dtype1, real_dtype, mesh_type0, mesh_type1):
    """Test basic intersection between two perpendicular planes"""
    # Create horizontal plane at z=0.5
    if mesh_type0 == 'triangle':
        mesh0 = create_horizontal_plane_triangles(index_dtype0, real_dtype, z=0.5)
    else:
        mesh0 = create_horizontal_plane_dynamic(index_dtype0, real_dtype, z=0.5)

    # Create vertical plane at x=0.5
    if mesh_type1 == 'triangle':
        mesh1 = create_vertical_plane_triangles(index_dtype1, real_dtype, x=0.5)
    else:
        mesh1 = create_vertical_plane_dynamic(index_dtype1, real_dtype, x=0.5)

    # Compute intersection curves
    paths, points = tf.intersection_curves(mesh0, mesh1)

    # Verify return types
    assert isinstance(paths, tf.OffsetBlockedArray), "paths should be OffsetBlockedArray"
    assert isinstance(points, np.ndarray), "points should be numpy array"

    # Verify points shape
    assert points.ndim == 2, f"Expected 2D array for points, got shape {points.shape}"
    assert points.shape[1] == 3, f"Expected 3D points, got shape {points.shape}"
    assert points.dtype == real_dtype, f"Points dtype should match mesh dtype ({real_dtype})"

    # Verify path indices are valid
    for path_ids in paths:
        assert np.all(path_ids >= 0), "Path indices should be non-negative"
        assert np.all(path_ids < len(points)), f"Path indices should be < {len(points)}"


@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_intersection_curves_iteration(real_dtype):
    """Test iterating over curves in paths"""
    mesh0 = create_horizontal_plane_triangles(np.int32, real_dtype, z=0.5)
    mesh1 = create_vertical_plane_triangles(np.int32, real_dtype, x=0.5)

    paths, curve_points = tf.intersection_curves(mesh0, mesh1)

    # Iterate over paths
    curve_count = 0
    for path_ids in paths:
        curve_count += 1
        # Get points for this curve
        pts = curve_points[path_ids]
        # Should be a valid numpy array
        assert isinstance(pts, np.ndarray)
        assert pts.ndim == 2
        assert pts.shape[1] == 3
        assert pts.dtype == real_dtype

    # Should have iterated over all curves
    assert curve_count == len(paths)


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_intersection_curves_symmetry(index_dtype, real_dtype):
    """Test that intersection is symmetric (mesh0, mesh1) == (mesh1, mesh0)"""
    mesh0 = create_horizontal_plane_triangles(index_dtype, real_dtype, z=0.5)
    mesh1 = create_vertical_plane_triangles(index_dtype, real_dtype, x=0.5)

    # Compute both orders
    paths01, points01 = tf.intersection_curves(mesh0, mesh1)
    paths10, points10 = tf.intersection_curves(mesh1, mesh0)

    # Both should produce valid results
    assert isinstance(paths01, tf.OffsetBlockedArray)
    assert isinstance(paths10, tf.OffsetBlockedArray)
    assert isinstance(points01, np.ndarray)
    assert isinstance(points10, np.ndarray)

    # Point counts should match
    assert len(points01) == len(points10), "Should have same number of curve points"


# ==============================================================================
# Error handling tests
# ==============================================================================

def test_intersection_curves_not_3d_mesh0():
    """Test error when first mesh is not 3D"""
    # Create 2D mesh
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    points = np.array([[0, 0], [1, 0], [0.5, 1]], dtype=np.float32)
    mesh_2d = tf.Mesh(faces, points)

    mesh_3d = create_horizontal_plane_triangles(np.int32, np.float32, z=0.0)

    with pytest.raises(ValueError, match="only supports 3D meshes.*mesh0"):
        tf.intersection_curves(mesh_2d, mesh_3d)


def test_intersection_curves_not_3d_mesh1():
    """Test error when second mesh is not 3D"""
    mesh_3d = create_horizontal_plane_triangles(np.int32, np.float32, z=0.0)

    # Create 2D mesh
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    points = np.array([[0, 0], [1, 0], [0.5, 1]], dtype=np.float32)
    mesh_2d = tf.Mesh(faces, points)

    with pytest.raises(ValueError, match="only supports 3D meshes.*mesh1"):
        tf.intersection_curves(mesh_3d, mesh_2d)


def test_intersection_curves_dtype_mismatch():
    """Test error when mesh dtypes don't match"""
    mesh_float = create_horizontal_plane_triangles(np.int32, np.float32, z=0.0)
    mesh_double = create_vertical_plane_triangles(np.int32, np.float64, x=0.5)

    with pytest.raises(ValueError, match="Mesh dtypes must match"):
        tf.intersection_curves(mesh_float, mesh_double)


def test_intersection_curves_invalid_type_mesh0():
    """Test error when first argument is not a Mesh object"""
    mesh = create_horizontal_plane_triangles(np.int32, np.float32, z=0.0)

    with pytest.raises(TypeError, match="mesh0 must be a Mesh object"):
        tf.intersection_curves("not a mesh", mesh)


def test_intersection_curves_invalid_type_mesh1():
    """Test error when second argument is not a Mesh object"""
    mesh = create_horizontal_plane_triangles(np.int32, np.float32, z=0.0)

    with pytest.raises(TypeError, match="mesh1 must be a Mesh object"):
        tf.intersection_curves(mesh, "not a mesh")


def test_intersection_curves_tuple_input_rejected():
    """Test that tuple input is rejected (topology required)"""
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    points = np.array([[0, 0, 0], [1, 0, 0], [0.5, 1, 0]], dtype=np.float32)

    mesh = create_horizontal_plane_triangles(np.int32, np.float32, z=0.0)

    with pytest.raises(TypeError, match="must be a Mesh object"):
        tf.intersection_curves((faces, points), mesh)


# ==============================================================================
# Transformation tests
# ==============================================================================

@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_intersection_curves_with_transformation(real_dtype):
    """Test intersection curves with transformed meshes"""
    mesh0 = create_horizontal_plane_triangles(np.int32, real_dtype, z=0.5)
    mesh1 = create_vertical_plane_triangles(np.int32, real_dtype, x=0.5)

    # Apply same transformation to both: translate by [10, 5, 2]
    transformation = np.array([
        [1, 0, 0, 10],
        [0, 1, 0,  5],
        [0, 0, 1,  2],
        [0, 0, 0,  1]
    ], dtype=real_dtype)

    mesh0.transformation = transformation
    mesh1.transformation = transformation

    # Should still find intersection curves
    paths, points = tf.intersection_curves(mesh0, mesh1)

    assert isinstance(paths, tf.OffsetBlockedArray), "Should return OffsetBlockedArray"
    assert isinstance(points, np.ndarray), "Should return numpy array for points"
    assert points.dtype == real_dtype, "Points dtype should match mesh dtype"


@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_intersection_curves_different_transformations(real_dtype):
    """Test intersection curves when meshes have different transformations"""
    mesh0 = create_horizontal_plane_triangles(np.int32, real_dtype, z=0.5)
    mesh1 = create_vertical_plane_triangles(np.int32, real_dtype, x=0.5)

    # Apply different transformations
    transformation0 = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=real_dtype)

    transformation1 = np.array([
        [1, 0, 0, 10],  # Translate mesh1 far away
        [0, 1, 0,  0],
        [0, 0, 1,  0],
        [0, 0, 0,  1]
    ], dtype=real_dtype)

    mesh0.transformation = transformation0
    mesh1.transformation = transformation1

    # May or may not intersect depending on translation
    # Just verify function doesn't crash
    paths, points = tf.intersection_curves(mesh0, mesh1)

    assert isinstance(paths, tf.OffsetBlockedArray)
    assert isinstance(points, np.ndarray)


# ==============================================================================
# Output validation tests
# ==============================================================================

@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_intersection_curves_output_types(real_dtype):
    """Test that output has correct types and shapes"""
    mesh0 = create_horizontal_plane_triangles(np.int32, real_dtype, z=0.5)
    mesh1 = create_vertical_plane_triangles(np.int32, real_dtype, x=0.5)

    paths, points = tf.intersection_curves(mesh0, mesh1)

    # Type checks
    assert isinstance(paths, tf.OffsetBlockedArray), \
        f"Expected OffsetBlockedArray, got {type(paths)}"
    assert isinstance(points, np.ndarray), \
        f"Expected numpy array, got {type(points)}"

    # Shape checks
    assert points.ndim == 2, f"Points should be 2D array, got {points.ndim}D"
    assert points.shape[1] == 3, f"Points should have 3 coordinates, got {points.shape[1]}"

    # Dtype checks
    assert points.dtype == real_dtype, \
        f"Points dtype should be {real_dtype}, got {points.dtype}"

    # Verify paths can be iterated
    path_count = sum(1 for _ in paths)
    assert path_count == len(paths), "Iteration count should match len(paths)"


@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_intersection_curves_valid_indices(real_dtype):
    """Test that all path indices are valid"""
    mesh0 = create_horizontal_plane_triangles(np.int32, real_dtype, z=0.5)
    mesh1 = create_vertical_plane_triangles(np.int32, real_dtype, x=0.5)

    paths, points = tf.intersection_curves(mesh0, mesh1)

    num_points = len(points)

    for i, path_ids in enumerate(paths):
        # All indices should be non-negative
        assert np.all(path_ids >= 0), \
            f"Path {i} contains negative indices: {path_ids[path_ids < 0]}"

        # All indices should be less than number of points
        assert np.all(path_ids < num_points), \
            f"Path {i} contains out-of-bounds indices (>= {num_points})"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
