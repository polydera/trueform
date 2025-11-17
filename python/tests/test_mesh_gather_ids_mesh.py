"""
Tests for Mesh × Mesh gather_ids

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via ziga.sajovic@xlab.si.
https://github.com/xlabmedical/trueform
"""
import sys
import os

# Add parent directory to path so we can import trueform
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
import trueform as tf


# Type combinations for Mesh: 2 index types × 2 real types × 2 ngons × 2 dims = 16
INDEX_DTYPES = [np.int32, np.int64]
REAL_DTYPES = [np.float32, np.float64]
NGONS = [3, 4]


def create_tiled_plane_2d_triangles(index_dtype, real_dtype, offset_x=0.0):
    """
    Create a simple 2x1 tiled plane in 2D with triangles
    offset_x shifts the mesh horizontally
    """
    faces = np.array([
        [0, 1, 4],
        [0, 4, 3],
        [1, 2, 5],
        [1, 5, 4]
    ], dtype=index_dtype)
    points = np.array([
        [0 + offset_x, 0], [1 + offset_x, 0], [2 + offset_x, 0],
        [0 + offset_x, 1], [1 + offset_x, 1], [2 + offset_x, 1]
    ], dtype=real_dtype)
    return tf.Mesh(faces, points)


def create_tiled_plane_3d_triangles(index_dtype, real_dtype, offset_z=0.0):
    """Create a simple 2x1 tiled plane in 3D (z=offset_z) with triangles"""
    faces = np.array([
        [0, 1, 4],
        [0, 4, 3],
        [1, 2, 5],
        [1, 5, 4]
    ], dtype=index_dtype)
    points = np.array([
        [0, 0, offset_z], [1, 0, offset_z], [2, 0, offset_z],
        [0, 1, offset_z], [1, 1, offset_z], [2, 1, offset_z]
    ], dtype=real_dtype)
    return tf.Mesh(faces, points)


def create_tiled_plane_2d_quads(index_dtype, real_dtype, offset_x=0.0):
    """Create a simple 2x1 tiled plane in 2D with quads"""
    faces = np.array([
        [0, 1, 4, 3],
        [1, 2, 5, 4]
    ], dtype=index_dtype)
    points = np.array([
        [0 + offset_x, 0], [1 + offset_x, 0], [2 + offset_x, 0],
        [0 + offset_x, 1], [1 + offset_x, 1], [2 + offset_x, 1]
    ], dtype=real_dtype)
    return tf.Mesh(faces, points)


def create_tiled_plane_3d_quads(index_dtype, real_dtype, offset_z=0.0):
    """Create a simple 2x1 tiled plane in 3D (z=offset_z) with quads"""
    faces = np.array([
        [0, 1, 4, 3],
        [1, 2, 5, 4]
    ], dtype=index_dtype)
    points = np.array([
        [0, 0, offset_z], [1, 0, offset_z], [2, 0, offset_z],
        [0, 1, offset_z], [1, 1, offset_z], [2, 1, offset_z]
    ], dtype=real_dtype)
    return tf.Mesh(faces, points)


# ==============================================================================
# 2D Mesh-Mesh gather_ids tests
# ==============================================================================

@pytest.mark.parametrize("index_dtype0", INDEX_DTYPES)
@pytest.mark.parametrize("index_dtype1", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
@pytest.mark.parametrize("ngon0", NGONS)
@pytest.mark.parametrize("ngon1", NGONS)
def test_mesh_gather_ids_mesh_2d_intersects_hit(index_dtype0, index_dtype1, real_dtype, ngon0, ngon1):
    """Test overlapping meshes - find intersecting face pairs"""
    if ngon0 == 3:
        mesh0 = create_tiled_plane_2d_triangles(index_dtype0, real_dtype, offset_x=0.0)
    else:
        mesh0 = create_tiled_plane_2d_quads(index_dtype0, real_dtype, offset_x=0.0)

    if ngon1 == 3:
        # Overlapping mesh at x=1 (overlaps with first mesh)
        mesh1 = create_tiled_plane_2d_triangles(index_dtype1, real_dtype, offset_x=1.0)
    else:
        mesh1 = create_tiled_plane_2d_quads(index_dtype1, real_dtype, offset_x=1.0)

    result = tf.gather_intersecting_ids(mesh0, mesh1)

    # Should return (N, 2) array with N > 0
    assert result.shape[1] == 2
    assert result.shape[0] > 0


@pytest.mark.parametrize("index_dtype0", INDEX_DTYPES)
@pytest.mark.parametrize("index_dtype1", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
@pytest.mark.parametrize("ngon0", NGONS)
@pytest.mark.parametrize("ngon1", NGONS)
def test_mesh_gather_ids_mesh_2d_intersects_miss(index_dtype0, index_dtype1, real_dtype, ngon0, ngon1):
    """Test non-overlapping meshes - no intersections"""
    if ngon0 == 3:
        mesh0 = create_tiled_plane_2d_triangles(index_dtype0, real_dtype, offset_x=0.0)
    else:
        mesh0 = create_tiled_plane_2d_quads(index_dtype0, real_dtype, offset_x=0.0)

    if ngon1 == 3:
        # Non-overlapping mesh at x=5 (far from first mesh)
        mesh1 = create_tiled_plane_2d_triangles(index_dtype1, real_dtype, offset_x=5.0)
    else:
        mesh1 = create_tiled_plane_2d_quads(index_dtype1, real_dtype, offset_x=5.0)

    result = tf.gather_intersecting_ids(mesh0, mesh1)

    # Should return empty (0, 2) array
    assert result.shape == (0, 2)


@pytest.mark.parametrize("index_dtype0", INDEX_DTYPES)
@pytest.mark.parametrize("index_dtype1", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
@pytest.mark.parametrize("ngon0", NGONS)
@pytest.mark.parametrize("ngon1", NGONS)
def test_mesh_gather_ids_mesh_2d_within_distance_hit(index_dtype0, index_dtype1, real_dtype, ngon0, ngon1):
    """Test meshes within distance threshold"""
    if ngon0 == 3:
        mesh0 = create_tiled_plane_2d_triangles(index_dtype0, real_dtype, offset_x=0.0)
    else:
        mesh0 = create_tiled_plane_2d_quads(index_dtype0, real_dtype, offset_x=0.0)

    if ngon1 == 3:
        # Mesh slightly separated at x=2.5 (0.5 units from first mesh edge at x=2)
        mesh1 = create_tiled_plane_2d_triangles(index_dtype1, real_dtype, offset_x=2.5)
    else:
        mesh1 = create_tiled_plane_2d_quads(index_dtype1, real_dtype, offset_x=2.5)

    # With distance=1.0, should find face pairs within distance
    result = tf.gather_ids_within_distance(mesh0, mesh1, distance=1.0)

    assert result.shape[1] == 2
    assert result.shape[0] > 0


@pytest.mark.parametrize("index_dtype0", INDEX_DTYPES)
@pytest.mark.parametrize("index_dtype1", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
@pytest.mark.parametrize("ngon0", NGONS)
@pytest.mark.parametrize("ngon1", NGONS)
def test_mesh_gather_ids_mesh_2d_within_distance_miss(index_dtype0, index_dtype1, real_dtype, ngon0, ngon1):
    """Test meshes outside distance threshold"""
    if ngon0 == 3:
        mesh0 = create_tiled_plane_2d_triangles(index_dtype0, real_dtype, offset_x=0.0)
    else:
        mesh0 = create_tiled_plane_2d_quads(index_dtype0, real_dtype, offset_x=0.0)

    if ngon1 == 3:
        mesh1 = create_tiled_plane_2d_triangles(index_dtype1, real_dtype, offset_x=10.0)
    else:
        mesh1 = create_tiled_plane_2d_quads(index_dtype1, real_dtype, offset_x=10.0)

    result = tf.gather_ids_within_distance(mesh0, mesh1, distance=1.0)

    assert result.shape == (0, 2)


# ==============================================================================
# 3D Mesh-Mesh gather_ids tests
# ==============================================================================

@pytest.mark.parametrize("index_dtype0", INDEX_DTYPES)
@pytest.mark.parametrize("index_dtype1", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
@pytest.mark.parametrize("ngon0", NGONS)
@pytest.mark.parametrize("ngon1", NGONS)
def test_mesh_gather_ids_mesh_3d_intersects_hit(index_dtype0, index_dtype1, real_dtype, ngon0, ngon1):
    """Test overlapping meshes intersect (3D)"""
    if ngon0 == 3:
        mesh0 = create_tiled_plane_3d_triangles(index_dtype0, real_dtype, offset_z=0.0)
    else:
        mesh0 = create_tiled_plane_3d_quads(index_dtype0, real_dtype, offset_z=0.0)

    if ngon1 == 3:
        # Overlapping mesh at same z=0 plane
        mesh1 = create_tiled_plane_3d_triangles(index_dtype1, real_dtype, offset_z=0.0)
    else:
        mesh1 = create_tiled_plane_3d_quads(index_dtype1, real_dtype, offset_z=0.0)

    result = tf.gather_intersecting_ids(mesh0, mesh1)

    assert result.shape[1] == 2
    assert result.shape[0] > 0


@pytest.mark.parametrize("index_dtype0", INDEX_DTYPES)
@pytest.mark.parametrize("index_dtype1", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
@pytest.mark.parametrize("ngon0", NGONS)
@pytest.mark.parametrize("ngon1", NGONS)
def test_mesh_gather_ids_mesh_3d_intersects_miss(index_dtype0, index_dtype1, real_dtype, ngon0, ngon1):
    """Test non-overlapping meshes don't intersect (3D)"""
    if ngon0 == 3:
        mesh0 = create_tiled_plane_3d_triangles(index_dtype0, real_dtype, offset_z=0.0)
    else:
        mesh0 = create_tiled_plane_3d_quads(index_dtype0, real_dtype, offset_z=0.0)

    if ngon1 == 3:
        # Non-overlapping mesh at z=5 (far from first mesh)
        mesh1 = create_tiled_plane_3d_triangles(index_dtype1, real_dtype, offset_z=5.0)
    else:
        mesh1 = create_tiled_plane_3d_quads(index_dtype1, real_dtype, offset_z=5.0)

    result = tf.gather_intersecting_ids(mesh0, mesh1)

    assert result.shape == (0, 2)


@pytest.mark.parametrize("index_dtype0", INDEX_DTYPES)
@pytest.mark.parametrize("index_dtype1", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
@pytest.mark.parametrize("ngon0", NGONS)
@pytest.mark.parametrize("ngon1", NGONS)
def test_mesh_gather_ids_mesh_3d_within_distance(index_dtype0, index_dtype1, real_dtype, ngon0, ngon1):
    """Test 3D meshes within distance threshold"""
    if ngon0 == 3:
        mesh0 = create_tiled_plane_3d_triangles(index_dtype0, real_dtype, offset_z=0.0)
    else:
        mesh0 = create_tiled_plane_3d_quads(index_dtype0, real_dtype, offset_z=0.0)

    if ngon1 == 3:
        # Mesh 0.5 units above
        mesh1 = create_tiled_plane_3d_triangles(index_dtype1, real_dtype, offset_z=0.5)
    else:
        mesh1 = create_tiled_plane_3d_quads(index_dtype1, real_dtype, offset_z=0.5)

    result = tf.gather_ids_within_distance(mesh0, mesh1, distance=1.0)

    assert result.shape[1] == 2
    assert result.shape[0] > 0


# ==============================================================================
# Symmetry tests
# ==============================================================================

@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_mesh_gather_ids_mesh_symmetry(real_dtype):
    """Test that swapping arguments swaps column order in results"""
    mesh0 = create_tiled_plane_2d_triangles(np.int32, real_dtype, offset_x=0.0)
    mesh1 = create_tiled_plane_2d_triangles(np.int32, real_dtype, offset_x=1.0)

    result01 = tf.gather_intersecting_ids(mesh0, mesh1)
    result10 = tf.gather_intersecting_ids(mesh1, mesh0)

    # Both should have same number of matches
    assert result01.shape[0] == result10.shape[0]

    # Columns should be swapped
    if result01.shape[0] > 0:
        pairs01 = set(tuple(row) for row in result01)
        pairs10 = set((row[1], row[0]) for row in result10)
        assert pairs01 == pairs10


# ==============================================================================
# Transformation tests
# ==============================================================================

@pytest.mark.parametrize("index_dtype0", INDEX_DTYPES)
@pytest.mark.parametrize("index_dtype1", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
@pytest.mark.parametrize("ngon0", NGONS)
@pytest.mark.parametrize("ngon1", NGONS)
def test_mesh_gather_ids_mesh_with_transformation_2d(index_dtype0, index_dtype1, real_dtype, ngon0, ngon1):
    """Test gather_ids with transformation (transform both meshes)"""
    if ngon0 == 3:
        mesh0 = create_tiled_plane_2d_triangles(index_dtype0, real_dtype, offset_x=0.0)
    else:
        mesh0 = create_tiled_plane_2d_quads(index_dtype0, real_dtype, offset_x=0.0)

    if ngon1 == 3:
        mesh1 = create_tiled_plane_2d_triangles(index_dtype1, real_dtype, offset_x=1.0)
    else:
        mesh1 = create_tiled_plane_2d_quads(index_dtype1, real_dtype, offset_x=1.0)

    # Verify intersection before transformation
    result_before = tf.gather_intersecting_ids(mesh0, mesh1)
    assert result_before.shape[0] > 0

    # Apply same transformation to both: translate by [5, 3]
    transformation = np.array([
        [1, 0, 5],
        [0, 1, 3],
        [0, 0, 1]
    ], dtype=real_dtype)

    mesh0.transformation = transformation
    mesh1.transformation = transformation

    # Should still intersect with same number of pairs
    result_after = tf.gather_intersecting_ids(mesh0, mesh1)
    assert result_after.shape[0] == result_before.shape[0]


@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_mesh_gather_ids_mesh_different_transformations_2d(real_dtype):
    """Test that different transformations can move meshes apart"""
    mesh0 = create_tiled_plane_2d_triangles(np.int32, real_dtype, offset_x=0.0)
    mesh1 = create_tiled_plane_2d_triangles(np.int32, real_dtype, offset_x=1.0)

    # Verify intersection before transformation
    result_before = tf.gather_intersecting_ids(mesh0, mesh1)
    assert result_before.shape[0] > 0

    # Apply different transformations
    # mesh0: no transformation
    # mesh1: translate far away
    transformation1 = np.array([
        [1, 0, 100],
        [0, 1, 100],
        [0, 0, 1]
    ], dtype=real_dtype)

    mesh1.transformation = transformation1

    # Should NOT intersect anymore
    result_after = tf.gather_intersecting_ids(mesh0, mesh1)
    assert result_after.shape[0] == 0


# ==============================================================================
# Edge cases
# ==============================================================================

def test_mesh_gather_ids_mesh_dimension_mismatch():
    """Test that dimension mismatch raises error"""
    mesh_2d = create_tiled_plane_2d_triangles(np.int32, np.float32)
    mesh_3d = create_tiled_plane_3d_triangles(np.int32, np.float32)

    with pytest.raises(ValueError, match="Dimension mismatch"):
        tf.gather_intersecting_ids(mesh_2d, mesh_3d)


@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_mesh_gather_ids_mesh_self_query(real_dtype):
    """Test mesh querying itself"""
    mesh = create_tiled_plane_2d_triangles(np.int32, real_dtype)

    result = tf.gather_intersecting_ids(mesh, mesh)

    # Should find all self-matches (each face with itself)
    # mesh has 4 faces, so expect at least 4 self-pairs
    assert result.shape[1] == 2
    assert result.shape[0] >= 4


@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
@pytest.mark.parametrize("ngon0", NGONS)
@pytest.mark.parametrize("ngon1", NGONS)
def test_mesh_gather_ids_mesh_return_dtype(real_dtype, ngon0, ngon1):
    """Test that return dtype matches mesh index dtype"""
    if ngon0 == 3:
        mesh0_int32 = create_tiled_plane_2d_triangles(np.int32, real_dtype)
    else:
        mesh0_int32 = create_tiled_plane_2d_quads(np.int32, real_dtype)

    if ngon1 == 3:
        mesh1_int32 = create_tiled_plane_2d_triangles(np.int32, real_dtype, offset_x=1.0)
    else:
        mesh1_int32 = create_tiled_plane_2d_quads(np.int32, real_dtype, offset_x=1.0)

    result = tf.gather_intersecting_ids(mesh0_int32, mesh1_int32)
    assert result.dtype == np.int32

    # Test with int64
    if ngon0 == 3:
        mesh0_int64 = create_tiled_plane_2d_triangles(np.int64, real_dtype)
    else:
        mesh0_int64 = create_tiled_plane_2d_quads(np.int64, real_dtype)

    if ngon1 == 3:
        mesh1_int64 = create_tiled_plane_2d_triangles(np.int64, real_dtype, offset_x=1.0)
    else:
        mesh1_int64 = create_tiled_plane_2d_quads(np.int64, real_dtype, offset_x=1.0)

    result_64 = tf.gather_intersecting_ids(mesh0_int64, mesh1_int64)
    assert result_64.dtype == np.int64


@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_mesh_gather_ids_mesh_partial_overlap(real_dtype):
    """Test meshes with partial overlap - specific face pairs"""
    # mesh0: 4 faces covering x=[0,2], y=[0,1]
    mesh0 = create_tiled_plane_2d_triangles(np.int32, real_dtype, offset_x=0.0)

    # mesh1: 4 faces covering x=[1,3], y=[0,1] (overlaps right half of mesh0)
    mesh1 = create_tiled_plane_2d_triangles(np.int32, real_dtype, offset_x=1.0)

    result = tf.gather_intersecting_ids(mesh0, mesh1)

    # Should find overlapping face pairs
    assert result.shape[1] == 2
    assert result.shape[0] > 0

    # All face indices should be valid
    assert np.all(result[:, 0] >= 0) and np.all(result[:, 0] < 4)
    assert np.all(result[:, 1] >= 0) and np.all(result[:, 1] < 4)


if __name__ == "__main__":
    # Run tests with verbose output
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
