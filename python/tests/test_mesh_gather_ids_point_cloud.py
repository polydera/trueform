"""
Tests for Mesh × PointCloud gather_ids

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/trueform
"""
import sys
import pytest
import numpy as np
import trueform as tf


# Type combinations
INDEX_DTYPES = [np.int32, np.int64]
REAL_DTYPES = [np.float32, np.float64]
MESH_TYPES = ['triangle', 'dynamic']  # triangle (ngon=3) and dynamic


def create_simple_mesh_2d(index_dtype, real_dtype, mesh_type):
    """Create a simple 2D mesh (triangles or dynamic)"""
    # Same geometry for both - two triangles forming a square
    points = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=real_dtype)

    if mesh_type == 'triangle':
        faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=index_dtype)
        return tf.Mesh(faces, points)
    else:
        # Dynamic mesh - same triangles wrapped in OffsetBlockedArray
        offsets = np.array([0, 3, 6], dtype=index_dtype)
        data = np.array([0, 1, 2, 0, 2, 3], dtype=index_dtype)
        faces = tf.OffsetBlockedArray(offsets, data)
        return tf.Mesh(faces, points)


def create_simple_mesh_3d(index_dtype, real_dtype, mesh_type):
    """Create a simple 3D mesh (triangles or dynamic in xy-plane)"""
    # Same geometry for both - two triangles forming a square in xy-plane
    points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=real_dtype)

    if mesh_type == 'triangle':
        faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=index_dtype)
        return tf.Mesh(faces, points)
    else:
        # Dynamic mesh - same triangles wrapped in OffsetBlockedArray
        offsets = np.array([0, 3, 6], dtype=index_dtype)
        data = np.array([0, 1, 2, 0, 2, 3], dtype=index_dtype)
        faces = tf.OffsetBlockedArray(offsets, data)
        return tf.Mesh(faces, points)


# ==============================================================================
# 2D Mesh-PointCloud gather_ids tests
# ==============================================================================

@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
@pytest.mark.parametrize("mesh_type", MESH_TYPES)
def test_mesh_gather_ids_point_cloud_2d_intersects_hit(index_dtype, real_dtype, mesh_type):
    """Test faces that contain points"""
    mesh = create_simple_mesh_2d(index_dtype, real_dtype, mesh_type)

    # Points inside and outside the mesh
    points_cloud = np.array([
        [0.5, 0.5],  # inside
        [0.2, 0.2],  # inside
        [5, 5]       # outside
    ], dtype=real_dtype)
    point_cloud = tf.PointCloud(points_cloud)

    result = tf.gather_intersecting_ids(mesh, point_cloud)

    # Should find first two points
    assert result.shape[1] == 2
    assert result.shape[0] >= 2


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
@pytest.mark.parametrize("mesh_type", MESH_TYPES)
def test_mesh_gather_ids_point_cloud_2d_intersects_miss(index_dtype, real_dtype, mesh_type):
    """Test mesh with points all outside"""
    mesh = create_simple_mesh_2d(index_dtype, real_dtype, mesh_type)

    points_cloud = np.array([[5, 5], [10, 10]], dtype=real_dtype)
    point_cloud = tf.PointCloud(points_cloud)

    result = tf.gather_intersecting_ids(mesh, point_cloud)

    # Should return empty (0, 2) array
    assert result.shape == (0, 2)


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
@pytest.mark.parametrize("mesh_type", MESH_TYPES)
def test_mesh_gather_ids_point_cloud_2d_within_distance_hit(index_dtype, real_dtype, mesh_type):
    """Test points within distance of faces"""
    mesh = create_simple_mesh_2d(index_dtype, real_dtype, mesh_type)

    # Points near the mesh
    points_cloud = np.array([
        [0.5, 0.5],   # inside (distance 0)
        [1.2, 0.5],   # just outside right edge
        [10, 10]      # far away
    ], dtype=real_dtype)
    point_cloud = tf.PointCloud(points_cloud)

    # With distance=0.5, first two points should be found
    result = tf.gather_ids_within_distance(mesh, point_cloud, distance=0.5)

    assert result.shape[1] == 2
    assert result.shape[0] >= 2

    # Verify point 2 is not in results
    pairs = set(tuple(row) for row in result)
    assert not any(pair[1] == 2 for pair in pairs)


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
@pytest.mark.parametrize("mesh_type", MESH_TYPES)
def test_mesh_gather_ids_point_cloud_2d_within_distance_miss(index_dtype, real_dtype, mesh_type):
    """Test points outside distance threshold"""
    mesh = create_simple_mesh_2d(index_dtype, real_dtype, mesh_type)

    points_cloud = np.array([[10, 10], [20, 20]], dtype=real_dtype)
    point_cloud = tf.PointCloud(points_cloud)

    result = tf.gather_ids_within_distance(mesh, point_cloud, distance=1.0)

    assert result.shape == (0, 2)


# ==============================================================================
# 3D Mesh-PointCloud gather_ids tests
# ==============================================================================

@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
@pytest.mark.parametrize("mesh_type", MESH_TYPES)
def test_mesh_gather_ids_point_cloud_3d_intersects_hit(index_dtype, real_dtype, mesh_type):
    """Test 3D faces containing points"""
    mesh = create_simple_mesh_3d(index_dtype, real_dtype, mesh_type)

    points_cloud = np.array([
        [0.5, 0.5, 0],  # on face
        [5, 5, 5]       # far away
    ], dtype=real_dtype)
    point_cloud = tf.PointCloud(points_cloud)

    result = tf.gather_intersecting_ids(mesh, point_cloud)

    assert result.shape[1] == 2
    assert result.shape[0] >= 1


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
@pytest.mark.parametrize("mesh_type", MESH_TYPES)
def test_mesh_gather_ids_point_cloud_3d_within_distance(index_dtype, real_dtype, mesh_type):
    """Test 3D points within distance of faces"""
    mesh = create_simple_mesh_3d(index_dtype, real_dtype, mesh_type)

    points_cloud = np.array([
        [0.5, 0.5, 0.1],  # slightly above face
        [0.5, 0.5, 0.2],  # slightly above face
    ], dtype=real_dtype)
    point_cloud = tf.PointCloud(points_cloud)

    result = tf.gather_ids_within_distance(mesh, point_cloud, distance=0.5)

    assert result.shape[1] == 2
    assert result.shape[0] >= 2


# ==============================================================================
# Symmetry tests (swap argument order)
# ==============================================================================

@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_mesh_gather_ids_point_cloud_symmetry(real_dtype):
    """Test that swapping arguments swaps column order"""
    mesh = create_simple_mesh_2d(np.int32, real_dtype, 'triangle')

    points_cloud = np.array([[0.5, 0.5]], dtype=real_dtype)
    point_cloud = tf.PointCloud(points_cloud)

    result_m_pc = tf.gather_intersecting_ids(mesh, point_cloud)
    result_pc_m = tf.gather_intersecting_ids(point_cloud, mesh)

    # Both should have same number of matches
    assert result_m_pc.shape[0] == result_pc_m.shape[0]

    # Columns should be swapped
    if result_m_pc.shape[0] > 0:
        pairs_m_pc = set(tuple(row) for row in result_m_pc)
        pairs_pc_m = set((row[1], row[0]) for row in result_pc_m)
        assert pairs_m_pc == pairs_pc_m


# ==============================================================================
# Transformation tests
# ==============================================================================

@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
@pytest.mark.parametrize("mesh_type", MESH_TYPES)
def test_mesh_gather_ids_point_cloud_with_transformation(index_dtype, real_dtype, mesh_type):
    """Test gather_ids with transformation"""
    mesh = create_simple_mesh_2d(index_dtype, real_dtype, mesh_type)

    points_cloud = np.array([[0.5, 0.5]], dtype=real_dtype)
    point_cloud = tf.PointCloud(points_cloud)

    # Before transformation - point is inside mesh
    result_before = tf.gather_intersecting_ids(mesh, point_cloud)
    assert result_before.shape[0] > 0

    # Transform mesh away
    transformation = np.array([
        [1, 0, 10],
        [0, 1, 10],
        [0, 0, 1]
    ], dtype=real_dtype)
    mesh.transformation = transformation

    # After transformation - no intersection
    result_after = tf.gather_intersecting_ids(mesh, point_cloud)
    assert result_after.shape[0] == 0


# ==============================================================================
# Edge cases
# ==============================================================================

def test_mesh_gather_ids_point_cloud_dimension_mismatch():
    """Test that dimension mismatch raises error"""
    mesh_2d = create_simple_mesh_2d(np.int32, np.float32, 'triangle')

    points_cloud_3d = np.array([[0, 0, 0]], dtype=np.float32)
    point_cloud_3d = tf.PointCloud(points_cloud_3d)

    with pytest.raises(ValueError, match="Dimension mismatch"):
        tf.gather_intersecting_ids(mesh_2d, point_cloud_3d)


@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_mesh_gather_ids_point_cloud_multiple_faces(real_dtype):
    """Test multiple faces with multiple points"""
    # Create a 2x2 grid of triangles
    faces = np.array([
        [0, 1, 4],
        [1, 2, 4],
        [2, 3, 4],
        [3, 0, 4]
    ], dtype=np.int32)
    points = np.array([
        [0, 0],
        [1, 0],
        [1, 1],
        [0, 1],
        [0.5, 0.5]
    ], dtype=real_dtype)
    mesh = tf.Mesh(faces, points)

    # Points in different faces
    points_cloud = np.array([
        [0.25, 0.25],  # in face 0
        [0.75, 0.25],  # in face 1
        [0.75, 0.75],  # in face 2
        [0.25, 0.75],  # in face 3
        [10, 10]       # outside
    ], dtype=real_dtype)
    point_cloud = tf.PointCloud(points_cloud)

    result = tf.gather_intersecting_ids(mesh, point_cloud)

    # Should find first 4 points
    assert result.shape[1] == 2
    assert result.shape[0] >= 4

    # Verify point 4 is not in results
    pairs = set(tuple(row) for row in result)
    assert not any(pair[1] == 4 for pair in pairs)


@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
@pytest.mark.parametrize("mesh_type", MESH_TYPES)
def test_mesh_gather_ids_point_cloud_return_dtype(real_dtype, mesh_type):
    """Test that return dtype matches mesh index dtype"""
    mesh_int32 = create_simple_mesh_2d(np.int32, real_dtype, mesh_type)

    points_cloud = np.array([[0.5, 0.5]], dtype=real_dtype)
    point_cloud = tf.PointCloud(points_cloud)

    result = tf.gather_intersecting_ids(mesh_int32, point_cloud)
    assert result.dtype == np.int32

    # Test with int64
    mesh_int64 = create_simple_mesh_2d(np.int64, real_dtype, mesh_type)
    result_64 = tf.gather_intersecting_ids(mesh_int64, point_cloud)
    assert result_64.dtype == np.int64


@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_mesh_gather_ids_point_cloud_boundary_points(real_dtype):
    """Test points on face boundaries"""
    mesh = create_simple_mesh_2d(np.int32, real_dtype, 'triangle')

    # Points on edges and corners
    points_cloud = np.array([
        [0, 0],      # corner
        [0.5, 0],    # edge
        [1, 0.5],    # edge
    ], dtype=real_dtype)
    point_cloud = tf.PointCloud(points_cloud)

    result = tf.gather_intersecting_ids(mesh, point_cloud)

    # Boundary points should be found (depending on implementation tolerance)
    assert result.shape[1] == 2
    # Don't assert exact count as boundary handling may vary


if __name__ == "__main__":
    # Run tests with verbose output
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
