"""
Tests for self_intersection_curves and embedded_self_intersection_curves

Copyright (c) 2025 Ziga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/trueform
"""

import sys
import numpy as np
import pytest
import trueform as tf


# Type combinations for self-intersection operations (3D only)
INDEX_DTYPES = [np.int32, np.int64]
REAL_DTYPES = [np.float32, np.float64]
MESH_TYPES = ['triangle', 'dynamic']  # triangles and dynamic (variable-size)


# ==============================================================================
# Helper functions for creating test meshes
# ==============================================================================

def create_tetrahedron_triangles(index_dtype, real_dtype):
    """Create a simple tetrahedron (no self-intersections) with triangles"""
    faces = np.array([
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


def create_self_intersecting_triangles(index_dtype, real_dtype):
    """Create two triangles that cross through each other (has self-intersection)"""
    faces = np.array([
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
    return tf.Mesh(faces, points)


def create_self_intersecting_dynamic(index_dtype, real_dtype):
    """Create two dynamic polygons that cross through each other (has self-intersection)"""
    offsets = np.array([0, 4, 8], dtype=index_dtype)
    data = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=index_dtype)
    faces = tf.OffsetBlockedArray(offsets, data)
    points = np.array([
        # First quad in XY plane
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [2.0, 2.0, 0.0],
        [0.0, 2.0, 0.0],
        # Second quad crossing the first
        [1.0, 0.0, -1.0],
        [1.0, 0.0, 1.0],
        [1.0, 2.0, 1.0],
        [1.0, 2.0, -1.0],
    ], dtype=real_dtype)
    return tf.Mesh(faces, points)


def create_non_self_intersecting_mesh(index_dtype, real_dtype, mesh_type):
    """Create a mesh with no self-intersections"""
    if mesh_type == 'triangle':
        return create_tetrahedron_triangles(index_dtype, real_dtype)
    else:
        return create_single_dynamic(index_dtype, real_dtype)


def create_self_intersecting_mesh(index_dtype, real_dtype, mesh_type):
    """Create a mesh with self-intersections"""
    if mesh_type == 'triangle':
        return create_self_intersecting_triangles(index_dtype, real_dtype)
    else:
        return create_self_intersecting_dynamic(index_dtype, real_dtype)


# ==============================================================================
# self_intersection_curves basic functionality tests
# ==============================================================================

@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
@pytest.mark.parametrize("mesh_type", MESH_TYPES)
def test_self_intersection_curves_no_intersection(index_dtype, real_dtype, mesh_type):
    """Test self_intersection_curves for mesh with no self-intersections"""
    mesh = create_non_self_intersecting_mesh(index_dtype, real_dtype, mesh_type)

    paths, points = tf.self_intersection_curves(mesh)

    # Verify return types
    assert isinstance(paths, tf.OffsetBlockedArray), "paths should be OffsetBlockedArray"
    assert isinstance(points, np.ndarray), "points should be numpy array"

    # No self-intersections expected
    assert len(paths) == 0, "Should find no self-intersection curves"
    assert len(points) == 0, "Should have no curve points"


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
@pytest.mark.parametrize("mesh_type", MESH_TYPES)
def test_self_intersection_curves_with_intersection(index_dtype, real_dtype, mesh_type):
    """Test self_intersection_curves for mesh with self-intersections"""
    mesh = create_self_intersecting_mesh(index_dtype, real_dtype, mesh_type)

    paths, points = tf.self_intersection_curves(mesh)

    # Verify return types
    assert isinstance(paths, tf.OffsetBlockedArray), "paths should be OffsetBlockedArray"
    assert isinstance(points, np.ndarray), "points should be numpy array"

    # Should find at least one intersection curve
    assert len(paths) >= 1, "Should find at least one self-intersection curve"
    assert len(points) >= 2, "Should have at least 2 curve points"

    # Verify points shape
    assert points.ndim == 2, f"Expected 2D array for points, got shape {points.shape}"
    assert points.shape[1] == 3, f"Expected 3D points, got shape {points.shape}"
    assert points.dtype == real_dtype, f"Points dtype should match mesh dtype ({real_dtype})"

    # Verify path indices are valid
    for path_ids in paths:
        assert np.all(path_ids >= 0), "Path indices should be non-negative"
        assert np.all(path_ids < len(points)), f"Path indices should be < {len(points)}"


@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_self_intersection_curves_iteration(real_dtype):
    """Test iterating over curves in paths"""
    mesh = create_self_intersecting_triangles(np.int32, real_dtype)

    paths, curve_points = tf.self_intersection_curves(mesh)

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
def test_self_intersection_curves_output_types(real_dtype):
    """Test that output has correct types and shapes"""
    mesh = create_self_intersecting_triangles(np.int32, real_dtype)

    paths, points = tf.self_intersection_curves(mesh)

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
def test_self_intersection_curves_valid_indices(real_dtype):
    """Test that all path indices are valid"""
    mesh = create_self_intersecting_triangles(np.int32, real_dtype)

    paths, points = tf.self_intersection_curves(mesh)

    num_points = len(points)

    for i, path_ids in enumerate(paths):
        # All indices should be non-negative
        assert np.all(path_ids >= 0), \
            f"Path {i} contains negative indices: {path_ids[path_ids < 0]}"

        # All indices should be less than number of points
        assert np.all(path_ids < num_points), \
            f"Path {i} contains out-of-bounds indices (>= {num_points})"


# ==============================================================================
# self_intersection_curves error handling tests
# ==============================================================================

def test_self_intersection_curves_not_3d():
    """Test error when mesh is not 3D"""
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    points = np.array([[0, 0], [1, 0], [0.5, 1]], dtype=np.float32)
    mesh_2d = tf.Mesh(faces, points)

    with pytest.raises(ValueError, match="3D"):
        tf.self_intersection_curves(mesh_2d)


def test_self_intersection_curves_invalid_type():
    """Test error when input is not a Mesh object"""
    with pytest.raises(TypeError, match="must be a Mesh"):
        tf.self_intersection_curves("not a mesh")


def test_self_intersection_curves_tuple_rejected():
    """Test that tuple input is rejected (topology required)"""
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    points = np.array([[0, 0, 0], [1, 0, 0], [0.5, 1, 0]], dtype=np.float32)

    with pytest.raises(TypeError, match="must be a Mesh"):
        tf.self_intersection_curves((faces, points))


# ==============================================================================
# Geometric validation tests
# ==============================================================================

@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_self_intersection_matches_intersection_curves(index_dtype, real_dtype):
    """
    Test that self_intersection_curves finds the same curves as intersection_curves
    when two separate meshes are concatenated.
    """
    # Create two triangles that intersect
    # Triangle 1: in XY plane
    faces1 = np.array([
        [0, 1, 2]
    ], dtype=index_dtype)
    points1 = np.array([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [1.0, 2.0, 0.0],
    ], dtype=real_dtype)
    mesh1 = tf.Mesh(faces1, points1)

    # Triangle 2: crosses through triangle 1
    faces2 = np.array([
        [0, 1, 2]
    ], dtype=index_dtype)
    points2 = np.array([
        [1.0, 0.5, -1.0],
        [1.0, 0.5, 1.0],
        [1.0, 1.5, 0.0],
    ], dtype=real_dtype)
    mesh2 = tf.Mesh(faces2, points2)

    # Compute intersection curves between the two meshes
    inter_paths, inter_points = tf.intersection_curves(mesh1, mesh2)

    # Concatenate the two meshes
    concat_faces, concat_points = tf.concatenated([(faces1, points1), (faces2, points2)])
    concat_mesh = tf.Mesh(concat_faces, concat_points)

    # Compute self-intersection curves on concatenated mesh
    self_paths, self_points = tf.self_intersection_curves(concat_mesh)

    # Both should find curves
    assert len(inter_paths) > 0, "intersection_curves should find curves"
    assert len(self_paths) > 0, "self_intersection_curves should find curves"

    # Should have same number of curves
    assert len(inter_paths) == len(self_paths), \
        f"Expected same number of curves: intersection={len(inter_paths)}, self={len(self_paths)}"

    # Compare curve points canonically
    # Get all points from intersection_curves
    inter_all_pts = []
    for path_ids in inter_paths:
        pts = inter_points[path_ids]
        inter_all_pts.append(pts)

    # Get all points from self_intersection_curves
    self_all_pts = []
    for path_ids in self_paths:
        pts = self_points[path_ids]
        self_all_pts.append(pts)

    # For each curve, sort points to create canonical representation
    # (handles potential reversal)
    def canonicalize_curve(pts):
        """Sort curve points lexicographically for comparison"""
        return np.array(sorted(pts.tolist()))

    inter_canonical = [canonicalize_curve(pts) for pts in inter_all_pts]
    self_canonical = [canonicalize_curve(pts) for pts in self_all_pts]

    # Sort curves themselves for comparison
    def curve_key(pts):
        return tuple(pts.flatten().tolist())

    inter_canonical.sort(key=curve_key)
    self_canonical.sort(key=curve_key)

    # Compare
    assert len(inter_canonical) == len(self_canonical)
    for i, (inter_pts, self_pts) in enumerate(zip(inter_canonical, self_canonical)):
        assert inter_pts.shape == self_pts.shape, \
            f"Curve {i}: shape mismatch {inter_pts.shape} vs {self_pts.shape}"
        np.testing.assert_allclose(
            inter_pts, self_pts,
            rtol=1e-5, atol=1e-7,
            err_msg=f"Curve {i} points don't match"
        )


# ==============================================================================
# Run tests
# ==============================================================================

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
