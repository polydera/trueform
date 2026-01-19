"""
Tests for PointCloud × EdgeMesh neighbor_search

Copyright (c) 2025 Žiga Sajovic, XLAB
"""
import sys
import pytest
import numpy as np
import trueform as tf


# Type combinations
INDEX_DTYPES = [np.int32, np.int64]
REAL_DTYPES = [np.float32, np.float64]


def create_2d_polyline(index_dtype, real_dtype):
    """
    Create a simple 2D straight polyline along x-axis at y=0.
    Points: [0,0], [1,0], [2,0], [3,0]
    Edges: [0,1], [1,2], [2,3]
    """
    edges = np.array([[0, 1], [1, 2], [2, 3]], dtype=index_dtype)
    points = np.array([[0, 0], [1, 0], [2, 0], [3, 0]], dtype=real_dtype)
    return tf.EdgeMesh(edges, points)


def create_3d_polyline(index_dtype, real_dtype):
    """
    Create a simple 3D straight polyline along x-axis at y=0, z=0.
    Points: [0,0,0], [1,0,0], [2,0,0], [3,0,0]
    Edges: [0,1], [1,2], [2,3]
    """
    edges = np.array([[0, 1], [1, 2], [2, 3]], dtype=index_dtype)
    points = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]], dtype=real_dtype)
    return tf.EdgeMesh(edges, points)


# ==============================================================================
# Basic Functionality Tests - 2D
# ==============================================================================

@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_point_cloud_neighbor_search_edge_mesh_2d_above(index_dtype, real_dtype):
    """Test 2D: point cloud above edge mesh"""
    edge_mesh = create_2d_polyline(index_dtype, real_dtype)

    # PointCloud: points above the edge mesh at y=2
    cloud_points = np.array([[0.5, 2], [1.5, 2], [2.5, 2]], dtype=real_dtype)
    cloud = tf.PointCloud(cloud_points)

    result = tf.neighbor_search(cloud, edge_mesh)
    assert result is not None

    (cloud_idx, edge_idx), (dist, pt_cloud, pt_edge) = result

    # Closest distance should be 2.0 (from y=2 to y=0), squared
    expected_dist2 = 4.0
    assert abs(dist - expected_dist2) < 1e-4

    # pt_cloud should be at y=2
    assert abs(pt_cloud[1] - 2.0) < 1e-4
    # pt_edge should be at y=0 (on the polyline)
    assert abs(pt_edge[1]) < 1e-4


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_point_cloud_neighbor_search_edge_mesh_2d_on_edge(index_dtype, real_dtype):
    """Test 2D: point cloud with point on edge"""
    edge_mesh = create_2d_polyline(index_dtype, real_dtype)

    # PointCloud: point directly on edge 1
    cloud_points = np.array([[1.5, 0.0]], dtype=real_dtype)
    cloud = tf.PointCloud(cloud_points)

    result = tf.neighbor_search(cloud, edge_mesh)
    assert result is not None

    (cloud_idx, edge_idx), (dist, pt_cloud, pt_edge) = result

    # Distance should be 0 (point on edge)
    assert dist < 1e-5
    # Should find edge 1
    assert edge_idx == 1


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_point_cloud_neighbor_search_edge_mesh_2d_specific_edge(index_dtype, real_dtype):
    """Test 2D: verify correct edge is found"""
    edge_mesh = create_2d_polyline(index_dtype, real_dtype)

    # PointCloud: point above edge 2 [2,0]->[3,0]
    cloud_points = np.array([[2.5, 1.0]], dtype=real_dtype)
    cloud = tf.PointCloud(cloud_points)

    result = tf.neighbor_search(cloud, edge_mesh)
    assert result is not None

    (cloud_idx, edge_idx), (dist, pt_cloud, pt_edge) = result

    # Should find edge 2
    assert edge_idx == 2
    # Distance should be 1.0, squared
    assert abs(dist - 1.0) < 1e-4
    # Closest point on edge should be at x=2.5, y=0
    assert abs(pt_edge[0] - 2.5) < 1e-4
    assert abs(pt_edge[1]) < 1e-4


# ==============================================================================
# Basic Functionality Tests - 3D
# ==============================================================================

@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_point_cloud_neighbor_search_edge_mesh_3d_above(index_dtype, real_dtype):
    """Test 3D: point cloud above edge mesh"""
    edge_mesh = create_3d_polyline(index_dtype, real_dtype)

    # PointCloud: points above the edge mesh at z=2
    cloud_points = np.array([[0.5, 0, 2], [1.5, 0, 2], [2.5, 0, 2]], dtype=real_dtype)
    cloud = tf.PointCloud(cloud_points)

    result = tf.neighbor_search(cloud, edge_mesh)
    assert result is not None

    (cloud_idx, edge_idx), (dist, pt_cloud, pt_edge) = result

    # Closest distance should be 2.0 (from z=2 to z=0), squared
    expected_dist2 = 4.0
    assert abs(dist - expected_dist2) < 1e-4

    # pt_cloud should be at z=2
    assert abs(pt_cloud[2] - 2.0) < 1e-4
    # pt_edge should be at z=0
    assert abs(pt_edge[2]) < 1e-4


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_point_cloud_neighbor_search_edge_mesh_3d_offset(index_dtype, real_dtype):
    """Test 3D: point cloud offset in y and z"""
    edge_mesh = create_3d_polyline(index_dtype, real_dtype)

    # PointCloud: point offset in y and z from edge 1
    cloud_points = np.array([[1.5, 1.0, 1.0]], dtype=real_dtype)
    cloud = tf.PointCloud(cloud_points)

    result = tf.neighbor_search(cloud, edge_mesh)
    assert result is not None

    (cloud_idx, edge_idx), (dist, pt_cloud, pt_edge) = result

    # Should find edge 1
    assert edge_idx == 1
    # Distance should be sqrt(1²+1²) = sqrt(2), squared = 2.0
    assert abs(dist - 2.0) < 1e-4
    # Closest point on edge should be at [1.5, 0, 0]
    assert abs(pt_edge[0] - 1.5) < 1e-4
    assert abs(pt_edge[1]) < 1e-4
    assert abs(pt_edge[2]) < 1e-4


# ==============================================================================
# Radius Tests
# ==============================================================================

@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_point_cloud_neighbor_search_edge_mesh_2d_radius_hit(real_dtype):
    """Test 2D with radius - within radius"""
    edge_mesh = create_2d_polyline(np.int32, real_dtype)

    # PointCloud nearby
    cloud_points = np.array([[1.5, 0.3]], dtype=real_dtype)
    cloud = tf.PointCloud(cloud_points)

    # Distance is 0.3, so radius=1.0 should find it
    result = tf.neighbor_search(cloud, edge_mesh, radius=1.0)
    assert result is not None

    ((_, _), (dist, _, _)) = result
    assert dist < 1.0


@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_point_cloud_neighbor_search_edge_mesh_2d_radius_miss(real_dtype):
    """Test 2D with radius - outside radius"""
    edge_mesh = create_2d_polyline(np.int32, real_dtype)

    # PointCloud far away
    cloud_points = np.array([[1.5, 10.0]], dtype=real_dtype)
    cloud = tf.PointCloud(cloud_points)

    result = tf.neighbor_search(cloud, edge_mesh, radius=1.0)
    assert result is None


# ==============================================================================
# Transformation Tests
# ==============================================================================

@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_point_cloud_neighbor_search_edge_mesh_with_edge_mesh_transformation(real_dtype):
    """Test neighbor search with transformed edge mesh"""
    edge_mesh = create_2d_polyline(np.int32, real_dtype)

    # PointCloud
    cloud_points = np.array([[1.5, 4.0]], dtype=real_dtype)
    cloud = tf.PointCloud(cloud_points)

    # Transform edge mesh by offset (0, 3)
    transform = np.eye(3, dtype=real_dtype)
    transform[1, 2] = 3.0  # y offset
    edge_mesh.transformation = transform

    result = tf.neighbor_search(cloud, edge_mesh)
    assert result is not None

    ((_, _), (dist, pt_cloud, pt_edge)) = result

    # pt_edge should be at transformed position (y around 3.0)
    assert abs(pt_edge[1] - 3.0) < 1e-4
    # Distance should be about 1.0, squared
    assert abs(dist - 1.0) < 1e-4


@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_point_cloud_neighbor_search_edge_mesh_with_cloud_transformation(real_dtype):
    """Test neighbor search with transformed point cloud"""
    edge_mesh = create_2d_polyline(np.int32, real_dtype)

    # PointCloud at y=1 (will be transformed to y=4)
    cloud_points = np.array([[1.5, 1.0]], dtype=real_dtype)
    cloud = tf.PointCloud(cloud_points)

    # Transform cloud by offset (0, 3)
    transform = np.eye(3, dtype=real_dtype)
    transform[1, 2] = 3.0  # y offset
    cloud.transformation = transform

    result = tf.neighbor_search(cloud, edge_mesh)
    assert result is not None

    ((_, _), (dist, pt_cloud, pt_edge)) = result

    # pt_cloud should be at transformed position (y around 4.0)
    assert abs(pt_cloud[1] - 4.0) < 1e-4
    # Distance should be about 4.0, squared = 16.0
    assert abs(dist - 16.0) < 1e-4


# ==============================================================================
# Symmetry Tests
# ==============================================================================

@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_point_cloud_neighbor_search_edge_mesh_symmetry(real_dtype):
    """Test that swapping query order gives consistent distances"""
    edge_mesh = create_2d_polyline(np.int32, real_dtype)

    # PointCloud
    cloud_points = np.array([[0.5, 3.0], [1.5, 3.0]], dtype=real_dtype)
    cloud = tf.PointCloud(cloud_points)

    # Query cloud -> edge_mesh
    result1 = tf.neighbor_search(cloud, edge_mesh)
    # Query edge_mesh -> cloud
    result2 = tf.neighbor_search(edge_mesh, cloud)

    assert result1 is not None
    assert result2 is not None

    _, (dist1, _, _) = result1
    _, (dist2, _, _) = result2

    # Distances should be equal
    assert abs(dist1 - dist2) < 1e-5


# ==============================================================================
# Error Handling Tests
# ==============================================================================

def test_point_cloud_neighbor_search_edge_mesh_dimension_mismatch():
    """Test that dimension mismatch raises error"""
    # 2D edge mesh
    edge_mesh = create_2d_polyline(np.int32, np.float32)

    # 3D point cloud
    cloud_points = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)
    cloud = tf.PointCloud(cloud_points)

    with pytest.raises(ValueError, match="Dimension mismatch"):
        tf.neighbor_search(cloud, edge_mesh)


# ==============================================================================
# Multiple Points Tests
# ==============================================================================

@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_point_cloud_neighbor_search_edge_mesh_multiple_points(real_dtype):
    """Test with multiple points in cloud"""
    edge_mesh = create_2d_polyline(np.int32, real_dtype)

    # PointCloud with multiple points at different distances
    cloud_points = np.array([
        [1.5, 5.0],   # Far
        [1.5, 0.5],   # Close (should be chosen)
        [1.5, 10.0],  # Very far
    ], dtype=real_dtype)
    cloud = tf.PointCloud(cloud_points)

    result = tf.neighbor_search(cloud, edge_mesh)
    assert result is not None

    (cloud_idx, edge_idx), (dist, pt_cloud, pt_edge) = result

    # Should find the closest point (index 1)
    assert cloud_idx == 1
    # Distance should be about 0.5, squared = 0.25
    assert abs(dist - 0.25) < 1e-4


@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_point_cloud_neighbor_search_edge_mesh_closest_edge(real_dtype):
    """Test that correct edge is found for multiple edges"""
    edge_mesh = create_2d_polyline(np.int32, real_dtype)

    # Point closest to edge 0
    cloud_points = np.array([[0.5, 1.0]], dtype=real_dtype)
    cloud = tf.PointCloud(cloud_points)

    result = tf.neighbor_search(cloud, edge_mesh)
    assert result is not None

    (_, edge_idx), (_, _, pt_edge) = result

    # Should find edge 0
    assert edge_idx == 0
    # Closest point should be at [0.5, 0]
    assert abs(pt_edge[0] - 0.5) < 1e-4


if __name__ == "__main__":
    # Run tests with verbose output
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
