"""
Tests for EdgeMesh × PointCloud gather_ids

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


# ==============================================================================
# 2D EdgeMesh-PointCloud gather_ids tests
# ==============================================================================

@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_edge_mesh_gather_ids_point_cloud_2d_intersects_hit(index_dtype, real_dtype):
    """Test edges that contain points"""
    # Horizontal edge from (0, 0) to (2, 0)
    edges = np.array([[0, 1]], dtype=index_dtype)
    points_edge = np.array([[0, 0], [2, 0]], dtype=real_dtype)
    edge_mesh = tf.EdgeMesh(edges, points_edge)

    # Points on the edge
    points_cloud = np.array([[0.5, 0], [1.0, 0], [5, 5]], dtype=real_dtype)
    point_cloud = tf.PointCloud(points_cloud)

    result = tf.gather_intersecting_ids(edge_mesh, point_cloud)

    # Should find points 0 and 1 intersecting edge 0
    assert result.shape[1] == 2
    assert result.shape[0] >= 2


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_edge_mesh_gather_ids_point_cloud_2d_intersects_miss(index_dtype, real_dtype):
    """Test edges with no points on them"""
    edges = np.array([[0, 1]], dtype=index_dtype)
    points_edge = np.array([[0, 0], [1, 0]], dtype=real_dtype)
    edge_mesh = tf.EdgeMesh(edges, points_edge)

    # Points away from edge
    points_cloud = np.array([[5, 5], [10, 10]], dtype=real_dtype)
    point_cloud = tf.PointCloud(points_cloud)

    result = tf.gather_intersecting_ids(edge_mesh, point_cloud)

    # Should return empty (0, 2) array
    assert result.shape == (0, 2)


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_edge_mesh_gather_ids_point_cloud_2d_within_distance_hit(index_dtype, real_dtype):
    """Test points within distance of edges"""
    # Horizontal edge from (0, 0) to (2, 0)
    edges = np.array([[0, 1]], dtype=index_dtype)
    points_edge = np.array([[0, 0], [2, 0]], dtype=real_dtype)
    edge_mesh = tf.EdgeMesh(edges, points_edge)

    # Points near the edge
    points_cloud = np.array([[0.5, 0.2], [1.0, 0.3], [5, 5]], dtype=real_dtype)
    point_cloud = tf.PointCloud(points_cloud)

    # With distance=0.5, first two points should be found
    result = tf.gather_ids_within_distance(edge_mesh, point_cloud, distance=0.5)

    assert result.shape[1] == 2
    assert result.shape[0] >= 2

    # Verify edge 0 paired with points 0 and 1
    pairs = set(tuple(row) for row in result)
    assert (0, 0) in pairs
    assert (0, 1) in pairs


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_edge_mesh_gather_ids_point_cloud_2d_within_distance_miss(index_dtype, real_dtype):
    """Test points outside distance threshold"""
    edges = np.array([[0, 1]], dtype=index_dtype)
    points_edge = np.array([[0, 0], [1, 0]], dtype=real_dtype)
    edge_mesh = tf.EdgeMesh(edges, points_edge)

    # Points far from edge
    points_cloud = np.array([[0, 10], [1, 10]], dtype=real_dtype)
    point_cloud = tf.PointCloud(points_cloud)

    result = tf.gather_ids_within_distance(edge_mesh, point_cloud, distance=1.0)

    assert result.shape == (0, 2)


# ==============================================================================
# 3D EdgeMesh-PointCloud gather_ids tests
# ==============================================================================

@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_edge_mesh_gather_ids_point_cloud_3d_intersects_hit(index_dtype, real_dtype):
    """Test 3D edges containing points"""
    edges = np.array([[0, 1]], dtype=index_dtype)
    points_edge = np.array([[0, 0, 0], [2, 0, 0]], dtype=real_dtype)
    edge_mesh = tf.EdgeMesh(edges, points_edge)

    points_cloud = np.array([[1, 0, 0], [5, 5, 5]], dtype=real_dtype)
    point_cloud = tf.PointCloud(points_cloud)

    result = tf.gather_intersecting_ids(edge_mesh, point_cloud)

    assert result.shape[1] == 2
    assert result.shape[0] >= 1


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_edge_mesh_gather_ids_point_cloud_3d_within_distance(index_dtype, real_dtype):
    """Test 3D points within distance of edges"""
    edges = np.array([[0, 1], [1, 2]], dtype=index_dtype)
    points_edge = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=real_dtype)
    edge_mesh = tf.EdgeMesh(edges, points_edge)

    points_cloud = np.array([[0.5, 0.2, 0], [1.5, 0.3, 0]], dtype=real_dtype)
    point_cloud = tf.PointCloud(points_cloud)

    result = tf.gather_ids_within_distance(edge_mesh, point_cloud, distance=0.5)

    assert result.shape[1] == 2
    assert result.shape[0] >= 2


# ==============================================================================
# Symmetry tests (swap argument order)
# ==============================================================================

@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_edge_mesh_gather_ids_point_cloud_symmetry(real_dtype):
    """Test that swapping arguments swaps column order"""
    edges = np.array([[0, 1]], dtype=np.int32)
    points_edge = np.array([[0, 0], [1, 0]], dtype=real_dtype)
    edge_mesh = tf.EdgeMesh(edges, points_edge)

    points_cloud = np.array([[0.5, 0.1]], dtype=real_dtype)
    point_cloud = tf.PointCloud(points_cloud)

    result_em_pc = tf.gather_ids_within_distance(edge_mesh, point_cloud, distance=0.5)
    result_pc_em = tf.gather_ids_within_distance(point_cloud, edge_mesh, distance=0.5)

    # Both should have same number of matches
    assert result_em_pc.shape[0] == result_pc_em.shape[0]

    # Columns should be swapped
    if result_em_pc.shape[0] > 0:
        pairs_em_pc = set(tuple(row) for row in result_em_pc)
        pairs_pc_em = set((row[1], row[0]) for row in result_pc_em)
        assert pairs_em_pc == pairs_pc_em


# ==============================================================================
# Transformation tests
# ==============================================================================

@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_edge_mesh_gather_ids_point_cloud_with_transformation(index_dtype, real_dtype):
    """Test gather_ids with transformation"""
    edges = np.array([[0, 1]], dtype=index_dtype)
    points_edge = np.array([[0, 0], [2, 0]], dtype=real_dtype)
    edge_mesh = tf.EdgeMesh(edges, points_edge)

    points_cloud = np.array([[1, 0]], dtype=real_dtype)
    point_cloud = tf.PointCloud(points_cloud)

    # Before transformation - point is on edge
    result_before = tf.gather_intersecting_ids(edge_mesh, point_cloud)
    assert result_before.shape[0] > 0

    # Transform point cloud away
    transformation = np.array([
        [1, 0, 0],
        [0, 1, 10],
        [0, 0, 1]
    ], dtype=real_dtype)
    point_cloud.transformation = transformation

    # After transformation - no intersection
    result_after = tf.gather_intersecting_ids(edge_mesh, point_cloud)
    assert result_after.shape[0] == 0

    # But within distance should find it
    result_within = tf.gather_ids_within_distance(edge_mesh, point_cloud, distance=15.0)
    assert result_within.shape[0] > 0


# ==============================================================================
# Edge cases
# ==============================================================================

def test_edge_mesh_gather_ids_point_cloud_dimension_mismatch():
    """Test that dimension mismatch raises error"""
    edges = np.array([[0, 1]], dtype=np.int32)
    points_edge_2d = np.array([[0, 0], [1, 0]], dtype=np.float32)
    edge_mesh_2d = tf.EdgeMesh(edges, points_edge_2d)

    points_cloud_3d = np.array([[0, 0, 0]], dtype=np.float32)
    point_cloud_3d = tf.PointCloud(points_cloud_3d)

    with pytest.raises(ValueError, match="Dimension mismatch"):
        tf.gather_intersecting_ids(edge_mesh_2d, point_cloud_3d)


@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_edge_mesh_gather_ids_point_cloud_multiple_edges(real_dtype):
    """Test multiple edges with multiple points"""
    # Create a square polyline
    edges = np.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=np.int32)
    points_edge = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=real_dtype)
    edge_mesh = tf.EdgeMesh(edges, points_edge)

    # Points on different edges
    points_cloud = np.array([
        [0.5, 0],    # on edge 0
        [1, 0.5],    # on edge 1
        [0.5, 1],    # on edge 2
        [0, 0.5],    # on edge 3
        [5, 5]       # far away
    ], dtype=real_dtype)
    point_cloud = tf.PointCloud(points_cloud)

    result = tf.gather_intersecting_ids(edge_mesh, point_cloud)

    # Should find first 4 points, not the 5th
    assert result.shape[1] == 2
    assert result.shape[0] >= 4

    # Verify point 4 is not in results
    pairs = set(tuple(row) for row in result)
    assert not any(pair[1] == 4 for pair in pairs)


@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_edge_mesh_gather_ids_point_cloud_return_dtype(real_dtype):
    """Test that return dtype matches edge mesh index dtype"""
    edges_int32 = np.array([[0, 1]], dtype=np.int32)
    points_edge = np.array([[0, 0], [1, 0]], dtype=real_dtype)
    edge_mesh_int32 = tf.EdgeMesh(edges_int32, points_edge)

    points_cloud = np.array([[0.5, 0]], dtype=real_dtype)
    point_cloud = tf.PointCloud(points_cloud)

    result = tf.gather_intersecting_ids(edge_mesh_int32, point_cloud)
    assert result.dtype == np.int32

    # Test with int64
    edges_int64 = np.array([[0, 1]], dtype=np.int64)
    edge_mesh_int64 = tf.EdgeMesh(edges_int64, points_edge)

    result_64 = tf.gather_intersecting_ids(edge_mesh_int64, point_cloud)
    assert result_64.dtype == np.int64


if __name__ == "__main__":
    # Run tests with verbose output
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
