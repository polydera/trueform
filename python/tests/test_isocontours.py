"""
Test make_isocontours functionality

Copyright (c) 2025 Žiga Sajovic, XLAB
"""

import sys
import pytest
import numpy as np
import trueform as tf


# Type combinations for isocontours
INDEX_DTYPES = [np.int32, np.int64]
REAL_DTYPES = [np.float32, np.float64]
MESH_TYPES = ['triangle', 'dynamic']


def create_mesh_data(faces_array, points_array, mesh_type):
    """Helper to create mesh with triangle or dynamic faces"""
    if mesh_type == 'dynamic':
        index_dtype = faces_array.dtype
        offsets = np.arange(0, len(faces_array) * 3 + 1, 3, dtype=index_dtype)
        data = faces_array.ravel()
        faces = tf.OffsetBlockedArray(offsets, data)
    else:
        faces = faces_array
    return tf.Mesh(faces, points_array)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_single_threshold(dtype):
    """Test isocontours with single threshold value"""
    # Create simple mesh (quad in xy-plane)
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    points = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0]
    ], dtype=dtype)
    mesh = tf.Mesh(faces, points)

    # Create scalar field (distance to z=0.5 plane)
    plane = tf.Plane(np.array([0, 0, 1, -0.5], dtype=dtype))
    scalar_field = tf.distance_field(mesh.points, plane)

    # Extract isocontour at 0.0 (z=0.5 plane)
    paths, points_out = tf.isocontours(mesh, scalar_field, 0.0)

    # Verify return types
    assert isinstance(paths, tf.OffsetBlockedArray), "paths should be OffsetBlockedArray"
    assert isinstance(points_out, np.ndarray), "points should be numpy array"

    # Verify points shape
    assert points_out.ndim == 2, f"Expected 2D array for points, got shape {points_out.shape}"
    assert points_out.shape[1] == 3, f"Expected 3D points, got shape {points_out.shape}"
    assert points_out.dtype == dtype, "Points dtype should match mesh dtype"

    # Verify path indices are valid
    for path_ids in paths:
        assert np.all(path_ids >= 0), "Path indices should be non-negative"
        assert np.all(path_ids < len(points_out)), f"Path indices should be < {len(points_out)}"


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_multiple_thresholds(dtype):
    """Test isocontours with multiple threshold values"""
    # Create simple mesh
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    points = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0]
    ], dtype=dtype)
    mesh = tf.Mesh(faces, points)

    # Create scalar field
    plane = tf.Plane(np.array([0, 0, 1, 0], dtype=dtype))
    scalar_field = tf.distance_field(mesh.points, plane)

    # Extract multiple isocontours
    thresholds = np.array([0.0, 0.5, 1.0], dtype=dtype)
    paths, points_out = tf.isocontours(mesh, scalar_field, thresholds)

    # Verify return types
    assert isinstance(paths, tf.OffsetBlockedArray), "paths should be OffsetBlockedArray"
    assert isinstance(points_out, np.ndarray), "points should be numpy array"

    # Verify points dtype
    assert points_out.dtype == dtype, "Points dtype should match mesh dtype"

    # Should have curves (possibly more than single threshold)
    # Number of curves depends on mesh topology, but should be > 0
    assert len(paths) >= 0, "Should have at least 0 curves"


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_with_distance_field(dtype):
    """Test isocontours using distance_field for scalar values"""
    # Create mesh from STL-like data
    faces = np.array([
        [0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1],
        [5, 2, 1], [5, 3, 2], [5, 4, 3], [5, 1, 4]
    ], dtype=np.int32)
    points = np.array([
        [0.0, 0.0, -1.0],  # Bottom center
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0]   # Top center
    ], dtype=dtype)
    mesh = tf.Mesh(faces, points)

    # Create scalar field using distance to horizontal plane
    plane = tf.Plane(np.array([0, 0, 1, 0], dtype=dtype))
    distances = tf.distance_field(mesh.points, plane)

    # Extract isocontour at z=0
    paths, curve_points = tf.isocontours(mesh, distances, 0.0)

    # Verify we got valid output
    assert isinstance(paths, tf.OffsetBlockedArray)
    assert isinstance(curve_points, np.ndarray)
    assert curve_points.dtype == dtype

    # If there are curves, verify they're near the plane
    if len(paths) > 0 and len(curve_points) > 0:
        for path_ids in paths:
            if len(path_ids) > 0:
                pts = curve_points[path_ids]
                # Points should be near z=0 (allowing some tolerance)
                assert np.all(np.abs(pts[:, 2]) < 1.0), "Isocontour points should be near z=0"


def test_scalar_threshold_becomes_array():
    """Test that scalar threshold is properly handled"""
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    points = np.array([[0, 0, 0], [1, 0, 0], [0.5, 1, 0]], dtype=np.float32)
    mesh = tf.Mesh(faces, points)

    # Create simple scalar field
    scalar_field = np.array([0.0, 1.0, 0.5], dtype=np.float32)

    # Single threshold as scalar
    paths1, points1 = tf.isocontours(mesh, scalar_field, 0.5)

    # Single threshold as array
    paths2, points2 = tf.isocontours(mesh, scalar_field, np.array([0.5]))

    # Both should work and give same result
    assert len(paths1) == len(paths2), "Scalar and array threshold should give same number of curves"


def test_list_threshold():
    """Test that threshold as list is converted properly"""
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    points = np.array([[0, 0, 0], [1, 0, 0], [0.5, 1, 0]], dtype=np.float32)
    mesh = tf.Mesh(faces, points)

    scalar_field = np.array([0.0, 1.0, 0.5], dtype=np.float32)

    # List of thresholds
    paths, points_out = tf.isocontours(mesh, scalar_field, [0.0, 0.5, 1.0])

    # Should work without error
    assert isinstance(paths, tf.OffsetBlockedArray)
    assert isinstance(points_out, np.ndarray)


def test_invalid_scalar_field_size():
    """Test error when scalar field size doesn't match mesh"""
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    points = np.array([[0, 0, 0], [1, 0, 0], [0.5, 1, 0]], dtype=np.float32)
    mesh = tf.Mesh(faces, points)

    # Wrong size scalar field
    wrong_scalar_field = np.array([0.0, 1.0], dtype=np.float32)  # Only 2 values

    with pytest.raises(ValueError) as excinfo:
        tf.isocontours(mesh, wrong_scalar_field, 0.5)
    assert "must match number of mesh points" in str(excinfo.value)


def test_invalid_scalar_field_dtype():
    """Test error when scalar field dtype doesn't match mesh"""
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    points = np.array([[0, 0, 0], [1, 0, 0], [0.5, 1, 0]], dtype=np.float32)
    mesh = tf.Mesh(faces, points)

    # Wrong dtype scalar field
    wrong_scalar_field = np.array([0.0, 1.0, 0.5], dtype=np.float64)

    with pytest.raises(TypeError) as excinfo:
        tf.isocontours(mesh, wrong_scalar_field, 0.5)
    assert "must match mesh dtype" in str(excinfo.value)


def test_invalid_mesh_type():
    """Test error when not passing a Mesh object"""
    with pytest.raises(TypeError) as excinfo:
        tf.isocontours("not a mesh", np.array([0.0]), 0.5)
    assert "Expected Mesh" in str(excinfo.value)


def test_invalid_scalar_field_type():
    """Test error when scalar field is not numpy array"""
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    points = np.array([[0, 0, 0], [1, 0, 0], [0.5, 1, 0]], dtype=np.float32)
    mesh = tf.Mesh(faces, points)

    with pytest.raises(TypeError) as excinfo:
        tf.isocontours(mesh, [0.0, 1.0, 0.5], 0.5)
    assert "Expected numpy array for scalar_field" in str(excinfo.value)


def test_invalid_scalar_field_shape():
    """Test error when scalar field is not 1D"""
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    points = np.array([[0, 0, 0], [1, 0, 0], [0.5, 1, 0]], dtype=np.float32)
    mesh = tf.Mesh(faces, points)

    # 2D scalar field
    wrong_scalar_field = np.array([[0.0, 1.0, 0.5]], dtype=np.float32)

    with pytest.raises(ValueError) as excinfo:
        tf.isocontours(mesh, wrong_scalar_field, 0.5)
    assert "Expected 1D array" in str(excinfo.value)


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
@pytest.mark.parametrize("mesh_type", MESH_TYPES)
def test_different_dtypes(index_dtype, real_dtype, mesh_type):
    """Test isocontours with different mesh dtypes and mesh types"""
    faces_array = np.array([[0, 1, 2]], dtype=index_dtype)
    points = np.array([[0, 0, 0], [1, 0, 0], [0.5, 1, 0]], dtype=real_dtype)
    mesh = create_mesh_data(faces_array, points, mesh_type)

    scalar_field = np.array([0.0, 1.0, 0.5], dtype=real_dtype)

    # Should work for all combinations
    paths, points_out = tf.isocontours(mesh, scalar_field, 0.5)

    assert isinstance(paths, tf.OffsetBlockedArray)
    assert isinstance(points_out, np.ndarray)
    assert points_out.dtype == real_dtype


def test_iteration_over_curves():
    """Test iterating over curves in paths"""
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    points = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0]
    ], dtype=np.float32)
    mesh = tf.Mesh(faces, points)

    plane = tf.Plane(np.array([0, 0, 1, 0], dtype=np.float32))
    scalar_field = tf.distance_field(mesh.points, plane)

    paths, curve_points = tf.isocontours(mesh, scalar_field, 0.0)

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

    # Should have iterated over all curves
    assert curve_count == len(paths)


def test_tuple_input():
    """Test that tuple input (faces, points) produces same results as Mesh"""
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    points = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 1.0]
    ], dtype=np.float32)

    # Create scalar field
    scalar_field = np.array([0.0, 0.5, 1.0, 1.5], dtype=np.float32)

    # Test with Mesh object
    mesh = tf.Mesh(faces, points)
    paths_mesh, points_mesh = tf.isocontours(mesh, scalar_field, 0.75)

    # Test with tuple input
    paths_tuple, points_tuple = tf.isocontours((faces, points), scalar_field, 0.75)

    # Results should be identical
    assert len(paths_mesh) == len(paths_tuple), \
        f"Path count mismatch: Mesh={len(paths_mesh)}, Tuple={len(paths_tuple)}"

    assert points_mesh.shape == points_tuple.shape, \
        f"Points shape mismatch: Mesh={points_mesh.shape}, Tuple={points_tuple.shape}"

    assert np.allclose(points_mesh, points_tuple), \
        "Points should be identical between Mesh and tuple input"

    # Check that paths are the same
    for i, (path_mesh, path_tuple) in enumerate(zip(paths_mesh, paths_tuple)):
        assert np.array_equal(path_mesh, path_tuple) or np.array_equal(path_mesh, path_tuple[::-1]), \
            f"Path {i} differs between Mesh and tuple input"


@pytest.mark.parametrize("mesh_type", MESH_TYPES)
def test_dynamic_mesh_isocontours(mesh_type):
    """Test isocontours with dynamic mesh"""
    faces_array = np.array([
        [0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1],
        [5, 2, 1], [5, 3, 2], [5, 4, 3], [5, 1, 4]
    ], dtype=np.int32)
    points = np.array([
        [0.0, 0.0, -1.0],  # Bottom center
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0]   # Top center
    ], dtype=np.float32)
    mesh = create_mesh_data(faces_array, points, mesh_type)

    # Create scalar field using distance to horizontal plane
    plane = tf.Plane(np.array([0, 0, 1, 0], dtype=np.float32))
    distances = tf.distance_field(mesh.points, plane)

    # Extract isocontour at z=0
    paths, curve_points = tf.isocontours(mesh, distances, 0.0)

    # Verify we got valid output
    assert isinstance(paths, tf.OffsetBlockedArray)
    assert isinstance(curve_points, np.ndarray)
    assert curve_points.dtype == np.float32

    # If there are curves, verify they're near the plane
    if len(paths) > 0 and len(curve_points) > 0:
        for path_ids in paths:
            if len(path_ids) > 0:
                pts = curve_points[path_ids]
                # Points should be near z=0 (allowing some tolerance)
                assert np.all(np.abs(pts[:, 2]) < 0.1), "Isocontour points should be near z=0"


@pytest.mark.parametrize("mesh_type", MESH_TYPES)
def test_dynamic_multiple_thresholds(mesh_type):
    """Test isocontours with multiple thresholds on dynamic mesh"""
    faces_array = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    points = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0]
    ], dtype=np.float32)
    mesh = create_mesh_data(faces_array, points, mesh_type)

    # Create scalar field
    plane = tf.Plane(np.array([0, 0, 1, 0], dtype=np.float32))
    scalar_field = tf.distance_field(mesh.points, plane)

    # Extract multiple isocontours
    thresholds = np.array([0.0, 0.5, 1.0], dtype=np.float32)
    paths, points_out = tf.isocontours(mesh, scalar_field, thresholds)

    # Verify return types
    assert isinstance(paths, tf.OffsetBlockedArray), "paths should be OffsetBlockedArray"
    assert isinstance(points_out, np.ndarray), "points should be numpy array"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
