"""
Tests for embedded_self_intersection_curves

Copyright (c) 2025 Ziga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/trueform
"""

import sys
import numpy as np
import pytest
import trueform as tf


# Type combinations for embedded_self_intersection_curves (3D only)
INDEX_DTYPES = [np.int32, np.int64]
REAL_DTYPES = [np.float32, np.float64]
MESH_TYPES = ['triangle', 'dynamic']


# ==============================================================================
# Helper functions for creating test meshes
# ==============================================================================

def create_tetrahedron_triangles(index_dtype, real_dtype, mesh_type='triangle'):
    """Create a simple tetrahedron (no self-intersections) with triangles"""
    faces_data = np.array([
        [0, 1, 2],
        [0, 2, 3],
        [0, 3, 1],
        [1, 3, 2]
    ], dtype=index_dtype)
    points = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 1.0, 0.0],
        [0.5, 0.5, 1.0]
    ], dtype=real_dtype)

    if mesh_type == 'dynamic':
        offsets = np.arange(0, len(faces_data) * 3 + 1, 3, dtype=index_dtype)
        data = faces_data.ravel()
        faces = tf.OffsetBlockedArray(offsets, data)
    else:
        faces = faces_data

    return tf.Mesh(faces, points)


def create_single_dynamic(index_dtype, real_dtype):
    """Create a single dynamic polygon (no self-intersections)"""
    offsets = np.array([0, 4], dtype=index_dtype)
    data = np.array([0, 1, 2, 3], dtype=index_dtype)
    faces = tf.OffsetBlockedArray(offsets, data)
    points = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0]
    ], dtype=real_dtype)
    return tf.Mesh(faces, points)


def create_self_intersecting_triangles(index_dtype, real_dtype, mesh_type='triangle'):
    """Create two triangles that cross through each other (has self-intersection)"""
    faces_data = np.array([
        [0, 1, 2],  # Triangle in XY plane
        [3, 4, 5],  # Triangle that passes through the first
    ], dtype=index_dtype)
    points = np.array([
        # First triangle in XY plane
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [1.0, 2.0, 0.0],
        # Second triangle crossing the first
        [1.0, 0.5, -1.0],
        [1.0, 0.5, 1.0],
        [1.0, 1.5, 0.0],
    ], dtype=real_dtype)

    if mesh_type == 'dynamic':
        offsets = np.arange(0, len(faces_data) * 3 + 1, 3, dtype=index_dtype)
        data = faces_data.ravel()
        faces = tf.OffsetBlockedArray(offsets, data)
    else:
        faces = faces_data

    return tf.Mesh(faces, points)


# ==============================================================================
# embedded_self_intersection_curves basic functionality tests
# ==============================================================================

@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
@pytest.mark.parametrize("mesh_type", MESH_TYPES)
def test_embedded_no_intersection(index_dtype, real_dtype, mesh_type):
    """Test embedded_self_intersection_curves for mesh with no self-intersections"""
    mesh = create_tetrahedron_triangles(index_dtype, real_dtype, mesh_type)

    result_faces, result_points = tf.embedded_self_intersection_curves(mesh)

    # Verify return types based on mesh type
    assert isinstance(result_points, np.ndarray), "result_points should be numpy array"
    if mesh_type == 'dynamic':
        assert isinstance(result_faces, tf.OffsetBlockedArray), "result_faces should be OffsetBlockedArray for dynamic"
        num_faces = len(result_faces)
    else:
        assert isinstance(result_faces, np.ndarray), "result_faces should be numpy array"
        # Verify shapes for triangle mesh
        assert result_faces.ndim == 2
        assert result_faces.shape[1] == 3, "Should be triangles"
        num_faces = result_faces.shape[0]
        # Verify dtype preserved
        assert result_faces.dtype == index_dtype, f"result_faces dtype should be {index_dtype}"

    assert result_points.dtype == real_dtype, f"result_points dtype should be {real_dtype}"

    # Verify point shapes
    assert result_points.ndim == 2
    assert result_points.shape[1] == 3

    # Result should have same number of faces (no splitting needed)
    assert num_faces == len(mesh.faces), "Should have same number of faces"


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
@pytest.mark.parametrize("mesh_type", MESH_TYPES)
def test_embedded_with_intersection(index_dtype, real_dtype, mesh_type):
    """Test embedded_self_intersection_curves for mesh with self-intersections"""
    mesh = create_self_intersecting_triangles(index_dtype, real_dtype, mesh_type)

    result_faces, result_points = tf.embedded_self_intersection_curves(mesh)

    # Verify return types based on mesh type
    assert isinstance(result_points, np.ndarray), "result_points should be numpy array"
    if mesh_type == 'dynamic':
        assert isinstance(result_faces, tf.OffsetBlockedArray), "result_faces should be OffsetBlockedArray for dynamic"
        num_faces = len(result_faces)
    else:
        assert isinstance(result_faces, np.ndarray), "result_faces should be numpy array"
        # Verify dtype preserved
        assert result_faces.dtype == index_dtype, f"result_faces dtype should be {index_dtype}"
        # Verify shapes
        assert result_faces.ndim == 2
        num_faces = result_faces.shape[0]

    assert result_points.dtype == real_dtype, f"result_points dtype should be {real_dtype}"

    # Verify shapes
    assert result_points.ndim == 2
    assert result_points.shape[1] == 3

    # Result should have more or equal faces (faces were split along curves)
    assert num_faces >= len(mesh.faces), "Should have at least as many faces"

    # Result should have more or equal points (intersection points added)
    assert len(result_points) >= len(mesh.points), "Should have at least as many points"


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
@pytest.mark.parametrize("mesh_type", MESH_TYPES)
def test_embedded_with_curves(index_dtype, real_dtype, mesh_type):
    """Test embedded_self_intersection_curves with return_curves=True"""
    mesh = create_self_intersecting_triangles(index_dtype, real_dtype, mesh_type)

    (result_faces, result_points), (paths, curve_points) = tf.embedded_self_intersection_curves(
        mesh, return_curves=True
    )

    # Verify mesh return types based on mesh type
    assert isinstance(result_points, np.ndarray), "result_points should be numpy array"
    if mesh_type == 'dynamic':
        assert isinstance(result_faces, tf.OffsetBlockedArray), "result_faces should be OffsetBlockedArray for dynamic"
    else:
        assert isinstance(result_faces, np.ndarray), "result_faces should be numpy array"
        # Verify dtype preserved
        assert result_faces.dtype == index_dtype

    assert result_points.dtype == real_dtype

    # Verify curves return types
    assert isinstance(paths, tf.OffsetBlockedArray), "paths should be OffsetBlockedArray"
    assert isinstance(curve_points, np.ndarray), "curve_points should be numpy array"

    # Should find curves
    assert len(paths) >= 1, "Should find at least one self-intersection curve"
    assert len(curve_points) >= 2, "Should have at least 2 curve points"

    # Verify curve points shape and dtype
    assert curve_points.ndim == 2
    assert curve_points.shape[1] == 3
    assert curve_points.dtype == real_dtype

    # Verify path indices are valid
    for path_ids in paths:
        assert np.all(path_ids >= 0), "Path indices should be non-negative"
        assert np.all(path_ids < len(curve_points)), f"Path indices should be < {len(curve_points)}"


@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_embedded_curves_iteration(real_dtype):
    """Test iterating over curves when return_curves=True"""
    mesh = create_self_intersecting_triangles(np.int32, real_dtype)

    (result_faces, result_points), (paths, curve_points) = tf.embedded_self_intersection_curves(
        mesh, return_curves=True
    )

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


@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_embedded_valid_indices(real_dtype):
    """Test that all path indices are valid when return_curves=True"""
    mesh = create_self_intersecting_triangles(np.int32, real_dtype)

    (result_faces, result_points), (paths, curve_points) = tf.embedded_self_intersection_curves(
        mesh, return_curves=True
    )

    num_points = len(curve_points)

    for i, path_ids in enumerate(paths):
        # All indices should be non-negative
        assert np.all(path_ids >= 0), \
            f"Path {i} contains negative indices: {path_ids[path_ids < 0]}"

        # All indices should be less than number of points
        assert np.all(path_ids < num_points), \
            f"Path {i} contains out-of-bounds indices (>= {num_points})"


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
@pytest.mark.parametrize("mesh_type", MESH_TYPES)
def test_embedded_no_intersection_with_curves(index_dtype, real_dtype, mesh_type):
    """Test embedded_self_intersection_curves with return_curves=True on mesh without self-intersections"""
    mesh = create_tetrahedron_triangles(index_dtype, real_dtype, mesh_type)

    (result_faces, result_points), (paths, curve_points) = tf.embedded_self_intersection_curves(
        mesh, return_curves=True
    )

    # Verify return types based on mesh type
    assert isinstance(result_points, np.ndarray)
    assert isinstance(paths, tf.OffsetBlockedArray)
    assert isinstance(curve_points, np.ndarray)

    if mesh_type == 'dynamic':
        assert isinstance(result_faces, tf.OffsetBlockedArray)
        num_faces = len(result_faces)
    else:
        assert isinstance(result_faces, np.ndarray)
        num_faces = result_faces.shape[0]

    # No curves expected
    assert len(paths) == 0, "Should find no self-intersection curves"

    # Result should have same number of faces
    assert num_faces == len(mesh.faces)


# ==============================================================================
# embedded_self_intersection_curves error handling tests
# ==============================================================================

def test_embedded_not_3d():
    """Test error when mesh is not 3D"""
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    points = np.array([[0, 0], [1, 0], [0.5, 1]], dtype=np.float32)
    mesh_2d = tf.Mesh(faces, points)

    with pytest.raises(ValueError, match="3D"):
        tf.embedded_self_intersection_curves(mesh_2d)


def test_embedded_invalid_type():
    """Test error when input is not a Mesh object"""
    with pytest.raises(TypeError, match="must be a Mesh"):
        tf.embedded_self_intersection_curves("not a mesh")


def test_embedded_tuple_rejected():
    """Test that tuple input is rejected (topology required)"""
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    points = np.array([[0, 0, 0], [1, 0, 0], [0.5, 1, 0]], dtype=np.float32)

    with pytest.raises(TypeError, match="must be a Mesh"):
        tf.embedded_self_intersection_curves((faces, points))


# ==============================================================================
# Run tests
# ==============================================================================

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
