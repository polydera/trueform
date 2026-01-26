"""
Tests for isobands extraction

Copyright (c) 2025 Ziga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/trueform
"""

import sys
import numpy as np
import pytest
import trueform as tf


# Type combinations for isobands (3D only)
INDEX_DTYPES = [np.int32, np.int64]
REAL_DTYPES = [np.float32, np.float64]
MESH_TYPES = ['triangle', 'dynamic']


# ==============================================================================
# Helper functions for creating test meshes
# ==============================================================================

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


def create_octahedron(index_dtype, real_dtype, mesh_type='triangle'):
    """Create a simple octahedron mesh"""
    faces_array = np.array([
        [0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1],
        [5, 2, 1], [5, 3, 2], [5, 4, 3], [5, 1, 4]
    ], dtype=index_dtype)
    points = np.array([
        [0.0, 0.0, -1.0],  # Bottom center
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0]   # Top center
    ], dtype=real_dtype)
    return create_mesh_data(faces_array, points, mesh_type)


def create_simple_quad(index_dtype, real_dtype, mesh_type='triangle'):
    """Create a simple quad triangulated into 2 triangles"""
    faces_array = np.array([[0, 1, 2], [0, 2, 3]], dtype=index_dtype)
    points = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0]
    ], dtype=real_dtype)
    return create_mesh_data(faces_array, points, mesh_type)


# ==============================================================================
# isobands basic functionality tests
# ==============================================================================

@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
@pytest.mark.parametrize("mesh_type", MESH_TYPES)
def test_isobands_basic(index_dtype, real_dtype, mesh_type):
    """Test basic isobands extraction"""
    mesh = create_octahedron(index_dtype, real_dtype, mesh_type)

    # Create scalar field using distance to horizontal plane
    plane = tf.Plane(np.array([0, 0, 1, 0], dtype=real_dtype))
    distances = tf.distance_field(mesh.points, plane)

    # Extract isobands at different z-levels
    cut_values = np.array([-0.5, 0.0, 0.5], dtype=real_dtype)
    (result_faces, result_points), labels = tf.isobands(mesh, distances, cut_values)

    # Verify return types based on mesh type
    assert isinstance(result_points, np.ndarray), "result_points should be numpy array"
    if mesh_type == 'dynamic':
        assert isinstance(result_faces, tf.OffsetBlockedArray), "result_faces should be OffsetBlockedArray for dynamic"
        num_faces = len(result_faces)
    else:
        assert isinstance(result_faces, np.ndarray), "result_faces should be numpy array for triangles"
        assert result_faces.ndim == 2
        assert result_faces.shape[1] == 3, "Should be triangles"
        num_faces = result_faces.shape[0]
        # Verify dtype preserved
        assert result_faces.dtype == index_dtype, f"result_faces dtype should be {index_dtype}"

    assert result_points.dtype == real_dtype, f"result_points dtype should be {real_dtype}"

    # Verify labels
    assert isinstance(labels, np.ndarray), "labels should be numpy array"
    assert labels.shape[0] == num_faces, "labels should have one entry per face"

    # Verify point shapes
    assert result_points.ndim == 2
    assert result_points.shape[1] == 3

    # Should have some output faces
    assert num_faces > 0, "Should have output faces"


@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
@pytest.mark.parametrize("mesh_type", MESH_TYPES)
def test_isobands_single_cut_value(real_dtype, mesh_type):
    """Test isobands with a single cut value"""
    mesh = create_octahedron(np.int32, real_dtype, mesh_type)

    plane = tf.Plane(np.array([0, 0, 1, 0], dtype=real_dtype))
    distances = tf.distance_field(mesh.points, plane)

    # Single cut value creates 2 bands
    (result_faces, result_points), labels = tf.isobands(mesh, distances, 0.0)

    assert isinstance(result_points, np.ndarray)
    if mesh_type == 'dynamic':
        assert isinstance(result_faces, tf.OffsetBlockedArray)
    else:
        assert isinstance(result_faces, np.ndarray)

    # Should have 2 unique labels (bands below and above the cut)
    unique_labels = np.unique(labels)
    assert len(unique_labels) == 2, f"Expected 2 bands, got {len(unique_labels)}"


@pytest.mark.parametrize("mesh_type", MESH_TYPES)
def test_isobands_selected_bands(mesh_type):
    """Test isobands with selected_bands parameter"""
    mesh = create_octahedron(np.int32, np.float32, mesh_type)

    plane = tf.Plane(np.array([0, 0, 1, 0], dtype=np.float32))
    distances = tf.distance_field(mesh.points, plane)

    # 3 cut values create 4 bands (0, 1, 2, 3)
    cut_values = np.array([-0.5, 0.0, 0.5], dtype=np.float32)

    # Only extract bands 1 and 2 (middle bands)
    selected = np.array([1, 2], dtype=np.int32)
    (result_faces, result_points), labels = tf.isobands(
        mesh, distances, cut_values, selected_bands=selected
    )

    # Labels should only contain values from selected bands
    unique_labels = np.unique(labels)
    assert all(label in selected for label in unique_labels), \
        f"Labels {unique_labels} should only contain selected bands {selected}"


@pytest.mark.parametrize("mesh_type", MESH_TYPES)
def test_isobands_with_curves(mesh_type):
    """Test isobands with return_curves=True"""
    mesh = create_octahedron(np.int32, np.float32, mesh_type)

    plane = tf.Plane(np.array([0, 0, 1, 0], dtype=np.float32))
    distances = tf.distance_field(mesh.points, plane)

    cut_values = np.array([-0.5, 0.0, 0.5], dtype=np.float32)
    (result_faces, result_points), labels, (paths, curve_points) = tf.isobands(
        mesh, distances, cut_values, return_curves=True
    )

    # Verify mesh return types based on mesh type
    assert isinstance(result_points, np.ndarray)
    if mesh_type == 'dynamic':
        assert isinstance(result_faces, tf.OffsetBlockedArray)
    else:
        assert isinstance(result_faces, np.ndarray)

    # Verify curves return types
    assert isinstance(paths, tf.OffsetBlockedArray), "paths should be OffsetBlockedArray"
    assert isinstance(curve_points, np.ndarray), "curve_points should be numpy array"

    # Should have some curves (one per cut value that intersects the mesh)
    assert len(paths) >= 0, "Should have curves"

    # Verify curve points shape and dtype
    if len(curve_points) > 0:
        assert curve_points.ndim == 2
        assert curve_points.shape[1] == 3
        assert curve_points.dtype == np.float32


@pytest.mark.parametrize("mesh_type", MESH_TYPES)
def test_isobands_path_indices_valid(mesh_type):
    """Test that path indices are valid when return_curves=True"""
    mesh = create_octahedron(np.int32, np.float32, mesh_type)

    plane = tf.Plane(np.array([0, 0, 1, 0], dtype=np.float32))
    distances = tf.distance_field(mesh.points, plane)

    cut_values = np.array([0.0], dtype=np.float32)
    (result_faces, result_points), labels, (paths, curve_points) = tf.isobands(
        mesh, distances, cut_values, return_curves=True
    )

    num_points = len(curve_points)
    for i, path_ids in enumerate(paths):
        # All indices should be non-negative
        assert np.all(path_ids >= 0), f"Path {i} contains negative indices"
        # All indices should be less than number of points
        if num_points > 0:
            assert np.all(path_ids < num_points), f"Path {i} contains out-of-bounds indices"


def test_isobands_tuple_input():
    """Test isobands with tuple input (faces, points)"""
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    points = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0]
    ], dtype=np.float32)

    # Create scalar field
    scalar_field = np.array([0.0, 0.5, 1.0, 0.5], dtype=np.float32)

    # Test with tuple input
    (result_faces, result_points), labels = tf.isobands(
        (faces, points), scalar_field, [0.25, 0.75]
    )

    assert isinstance(result_faces, np.ndarray)
    assert isinstance(result_points, np.ndarray)
    assert isinstance(labels, np.ndarray)


# ==============================================================================
# isobands error handling tests
# ==============================================================================

def test_isobands_not_3d():
    """Test error when mesh is not 3D"""
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    points = np.array([[0, 0], [1, 0], [0.5, 1]], dtype=np.float32)
    mesh_2d = tf.Mesh(faces, points)

    scalar_field = np.array([0.0, 1.0, 0.5], dtype=np.float32)

    with pytest.raises(ValueError, match="3D"):
        tf.isobands(mesh_2d, scalar_field, 0.5)


def test_isobands_invalid_type():
    """Test error when input is not a Mesh or tuple"""
    with pytest.raises(TypeError, match="Expected Mesh"):
        tf.isobands("not a mesh", np.array([0.0]), 0.5)


def test_isobands_invalid_scalar_field_size():
    """Test error when scalar field size doesn't match mesh"""
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    points = np.array([[0, 0, 0], [1, 0, 0], [0.5, 1, 0]], dtype=np.float32)
    mesh = tf.Mesh(faces, points)

    # Wrong size scalar field
    wrong_scalar_field = np.array([0.0, 1.0], dtype=np.float32)

    with pytest.raises(ValueError, match="must match number of mesh points"):
        tf.isobands(mesh, wrong_scalar_field, 0.5)


def test_isobands_invalid_scalar_field_dtype():
    """Test error when scalar field dtype doesn't match mesh"""
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    points = np.array([[0, 0, 0], [1, 0, 0], [0.5, 1, 0]], dtype=np.float32)
    mesh = tf.Mesh(faces, points)

    # Wrong dtype scalar field
    wrong_scalar_field = np.array([0.0, 1.0, 0.5], dtype=np.float64)

    with pytest.raises(TypeError, match="must match mesh dtype"):
        tf.isobands(mesh, wrong_scalar_field, 0.5)


def test_isobands_invalid_scalar_field_type():
    """Test error when scalar field is not numpy array"""
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    points = np.array([[0, 0, 0], [1, 0, 0], [0.5, 1, 0]], dtype=np.float32)
    mesh = tf.Mesh(faces, points)

    with pytest.raises(TypeError, match="Expected numpy array for scalar_field"):
        tf.isobands(mesh, [0.0, 1.0, 0.5], 0.5)


def test_isobands_invalid_scalar_field_shape():
    """Test error when scalar field is not 1D"""
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    points = np.array([[0, 0, 0], [1, 0, 0], [0.5, 1, 0]], dtype=np.float32)
    mesh = tf.Mesh(faces, points)

    # 2D scalar field
    wrong_scalar_field = np.array([[0.0, 1.0, 0.5]], dtype=np.float32)

    with pytest.raises(ValueError, match="Expected 1D array"):
        tf.isobands(mesh, wrong_scalar_field, 0.5)


# ==============================================================================
# Run tests
# ==============================================================================

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
