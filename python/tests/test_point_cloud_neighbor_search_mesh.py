"""
Tests for PointCloud × Mesh neighbor_search

Copyright (c) 2025 Žiga Sajovic, XLAB
"""
import sys
import pytest
import numpy as np
import trueform as tf


# Type combinations
INDEX_DTYPES = [np.int32, np.int64]
REAL_DTYPES = [np.float32, np.float64]
NGONS = [3, 'dyn']


def create_square_mesh_2d(points, index_dtype, ngon):
    """Create a square mesh from 4 points (2D)"""
    if ngon == 3:
        faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=index_dtype)
        return tf.Mesh(faces, points)
    else:  # dynamic - create as one quad
        offsets = np.array([0, 4], dtype=index_dtype)
        data = np.array([0, 1, 2, 3], dtype=index_dtype)
        faces = tf.OffsetBlockedArray(offsets, data)
        return tf.Mesh(faces, points)


def create_square_mesh_3d(points, index_dtype, ngon):
    """Create a square mesh from 4 points (3D)"""
    if ngon == 3:
        faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=index_dtype)
        return tf.Mesh(faces, points)
    else:  # dynamic - create as one quad
        offsets = np.array([0, 4], dtype=index_dtype)
        data = np.array([0, 1, 2, 3], dtype=index_dtype)
        faces = tf.OffsetBlockedArray(offsets, data)
        return tf.Mesh(faces, points)


# ==============================================================================
# Basic Functionality Tests - 2D
# ==============================================================================

@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
@pytest.mark.parametrize("ngon", NGONS)
def test_point_cloud_neighbor_search_mesh_2d_above(index_dtype, real_dtype, ngon):
    """Test 2D: point cloud above mesh"""
    # Mesh: square at y=[0,1]
    mesh_points = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=real_dtype)
    mesh = create_square_mesh_2d(mesh_points, index_dtype, ngon)

    # PointCloud: points above the mesh at y=3
    cloud_points = np.array([[0.5, 3], [0.2, 3], [0.8, 3]], dtype=real_dtype)
    cloud = tf.PointCloud(cloud_points)

    result = tf.neighbor_search(cloud, mesh)
    assert result is not None

    (cloud_idx, mesh_idx), (dist, pt_cloud, pt_mesh) = result

    # Closest distance should be 2.0 (from y=3 to y=1), squared
    expected_dist2 = 4.0
    assert abs(dist - expected_dist2) < 1e-4

    # pt_cloud should be at y=3
    assert abs(pt_cloud[1] - 3.0) < 1e-4
    # pt_mesh should be at y=1 (top edge of mesh)
    assert abs(pt_mesh[1] - 1.0) < 1e-4


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
@pytest.mark.parametrize("ngon", NGONS)
def test_point_cloud_neighbor_search_mesh_2d_inside(index_dtype, real_dtype, ngon):
    """Test 2D: point cloud with point inside mesh"""
    # Mesh: square at origin
    mesh_points = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=real_dtype)
    mesh = create_square_mesh_2d(mesh_points, index_dtype, ngon)

    # PointCloud: one point inside mesh
    cloud_points = np.array([[0.5, 0.5]], dtype=real_dtype)
    cloud = tf.PointCloud(cloud_points)

    result = tf.neighbor_search(cloud, mesh)
    assert result is not None

    (cloud_idx, mesh_idx), (dist, pt_cloud, pt_mesh) = result

    # Distance should be 0 (point inside mesh)
    assert dist < 1e-5


# ==============================================================================
# Basic Functionality Tests - 3D
# ==============================================================================

@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
@pytest.mark.parametrize("ngon", NGONS)
def test_point_cloud_neighbor_search_mesh_3d_above(index_dtype, real_dtype, ngon):
    """Test 3D: point cloud above mesh"""
    # Mesh: square in xy-plane at z=0
    mesh_points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=real_dtype)
    mesh = create_square_mesh_3d(mesh_points, index_dtype, ngon)

    # PointCloud: points above the mesh at z=2
    cloud_points = np.array([[0.5, 0.5, 2], [0.2, 0.2, 2], [0.8, 0.8, 2]], dtype=real_dtype)
    cloud = tf.PointCloud(cloud_points)

    result = tf.neighbor_search(cloud, mesh)
    assert result is not None

    (cloud_idx, mesh_idx), (dist, pt_cloud, pt_mesh) = result

    # Closest distance should be 2.0, squared
    expected_dist2 = 4.0
    assert abs(dist - expected_dist2) < 1e-4

    # pt_cloud should be at z=2
    assert abs(pt_cloud[2] - 2.0) < 1e-4
    # pt_mesh should be at z=0
    assert abs(pt_mesh[2]) < 1e-4


@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_point_cloud_neighbor_search_mesh_3d_closest_point(real_dtype):
    """Test 3D: verify closest point is on mesh face"""
    # Triangle mesh
    mesh_points = np.array([[0, 0, 0], [1, 0, 0], [0.5, 1, 0]], dtype=real_dtype)
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    mesh = tf.Mesh(faces, mesh_points)

    # Point directly above center of triangle
    cloud_points = np.array([[0.5, 0.33, 3.0]], dtype=real_dtype)
    cloud = tf.PointCloud(cloud_points)

    result = tf.neighbor_search(cloud, mesh)
    assert result is not None

    (cloud_idx, mesh_idx), (dist, pt_cloud, pt_mesh) = result

    # Mesh closest point should be at z=0
    assert abs(pt_mesh[2]) < 1e-4
    # Should be roughly below the query point
    assert abs(pt_mesh[0] - 0.5) < 0.2
    assert abs(pt_mesh[1] - 0.33) < 0.2


# ==============================================================================
# Radius Tests
# ==============================================================================

@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_point_cloud_neighbor_search_mesh_2d_radius_hit(real_dtype):
    """Test 2D with radius - within radius"""
    # Triangle mesh
    mesh_points = np.array([[0, 0], [1, 0], [0.5, 1]], dtype=real_dtype)
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    mesh = tf.Mesh(faces, mesh_points)

    # PointCloud nearby
    cloud_points = np.array([[0.5, 1.3]], dtype=real_dtype)
    cloud = tf.PointCloud(cloud_points)

    # Closest distance should be about 0.3, so radius=1.0 should find it
    result = tf.neighbor_search(cloud, mesh, radius=1.0)
    assert result is not None

    ((_, _), (dist, _, _)) = result
    assert dist < 1.0


@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_point_cloud_neighbor_search_mesh_2d_radius_miss(real_dtype):
    """Test 2D with radius - outside radius"""
    # Triangle mesh
    mesh_points = np.array([[0, 0], [1, 0], [0.5, 1]], dtype=real_dtype)
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    mesh = tf.Mesh(faces, mesh_points)

    # PointCloud far away
    cloud_points = np.array([[10, 10]], dtype=real_dtype)
    cloud = tf.PointCloud(cloud_points)

    result = tf.neighbor_search(cloud, mesh, radius=1.0)
    assert result is None


# ==============================================================================
# Transformation Tests
# ==============================================================================

@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_point_cloud_neighbor_search_mesh_with_mesh_transformation(real_dtype):
    """Test neighbor search with transformed mesh"""
    # Triangle mesh at origin
    mesh_points = np.array([[0, 0], [1, 0], [0.5, 1]], dtype=real_dtype)
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    mesh = tf.Mesh(faces, mesh_points)

    # PointCloud
    cloud_points = np.array([[0.5, 4.0]], dtype=real_dtype)
    cloud = tf.PointCloud(cloud_points)

    # Transform mesh by offset (0, 3)
    transform = np.eye(3, dtype=real_dtype)
    transform[1, 2] = 3.0  # y offset
    mesh.transformation = transform

    result = tf.neighbor_search(cloud, mesh)
    assert result is not None

    ((_, _), (dist, pt_cloud, pt_mesh)) = result

    # pt_mesh should be at transformed top of mesh (y around 4.0)
    assert pt_mesh[1] > 3.5


@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_point_cloud_neighbor_search_mesh_with_cloud_transformation(real_dtype):
    """Test neighbor search with transformed point cloud"""
    # Triangle mesh
    mesh_points = np.array([[0, 0], [1, 0], [0.5, 1]], dtype=real_dtype)
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    mesh = tf.Mesh(faces, mesh_points)

    # PointCloud at origin (will be transformed)
    cloud_points = np.array([[0.5, 1.0]], dtype=real_dtype)
    cloud = tf.PointCloud(cloud_points)

    # Transform cloud by offset (0, 3)
    transform = np.eye(3, dtype=real_dtype)
    transform[1, 2] = 3.0  # y offset
    cloud.transformation = transform

    result = tf.neighbor_search(cloud, mesh)
    assert result is not None

    ((_, _), (dist, pt_cloud, pt_mesh)) = result

    # pt_cloud should be at transformed position (y around 4.0)
    assert pt_cloud[1] > 3.5


# ==============================================================================
# Symmetry Tests
# ==============================================================================

@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_point_cloud_neighbor_search_mesh_symmetry(real_dtype):
    """Test that swapping query order gives consistent distances"""
    # Mesh
    mesh_points = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=real_dtype)
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    mesh = tf.Mesh(faces, mesh_points)

    # PointCloud
    cloud_points = np.array([[0.5, 3.0], [1.5, 3.0]], dtype=real_dtype)
    cloud = tf.PointCloud(cloud_points)

    # Query cloud -> mesh
    result1 = tf.neighbor_search(cloud, mesh)
    # Query mesh -> cloud
    result2 = tf.neighbor_search(mesh, cloud)

    assert result1 is not None
    assert result2 is not None

    _, (dist1, _, _) = result1
    _, (dist2, _, _) = result2

    # Distances should be equal
    assert abs(dist1 - dist2) < 1e-5


# ==============================================================================
# Error Handling Tests
# ==============================================================================

def test_point_cloud_neighbor_search_mesh_dimension_mismatch():
    """Test that dimension mismatch raises error"""
    # 2D mesh
    mesh_points = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    mesh = tf.Mesh(faces, mesh_points)

    # 3D point cloud
    cloud_points = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)
    cloud = tf.PointCloud(cloud_points)

    with pytest.raises(ValueError, match="Dimension mismatch"):
        tf.neighbor_search(cloud, mesh)


# ==============================================================================
# Multiple Points Tests
# ==============================================================================

@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_point_cloud_neighbor_search_mesh_multiple_points(real_dtype):
    """Test with multiple points in cloud"""
    # Mesh
    mesh_points = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=real_dtype)
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    mesh = tf.Mesh(faces, mesh_points)

    # PointCloud with multiple points at different distances
    cloud_points = np.array([
        [0.5, 5.0],   # Far
        [0.5, 1.5],   # Close (should be chosen)
        [0.5, 10.0],  # Very far
    ], dtype=real_dtype)
    cloud = tf.PointCloud(cloud_points)

    result = tf.neighbor_search(cloud, mesh)
    assert result is not None

    (cloud_idx, mesh_idx), (dist, pt_cloud, pt_mesh) = result

    # Should find the closest point (index 1)
    assert cloud_idx == 1
    # Distance should be about 0.5, squared = 0.25
    assert abs(dist - 0.25) < 1e-4


if __name__ == "__main__":
    # Run tests with verbose output
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
