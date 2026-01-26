"""
Tests for neighbor_search spatial queries

Copyright (c) 2025 Žiga Sajovic, XLAB
"""
import numpy as np
import pytest
import trueform as tf


def test_neighbor_search_point_2d():
    """Test single nearest neighbor search with point query in 2D"""
    # Create a simple 2D point cloud (float32)
    points = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float32)
    cloud = tf.PointCloud(points)

    # Query with a point close to [0, 0]
    # [0.1, 0.1] becomes float64 from numpy, but dispatch will convert to match cloud
    query = tf.Point([0.1, 0.1])
    idx, dist2, closest_pt = tf.neighbor_search(cloud, query)

    assert idx == 0  # Should find point at [0, 0]
    assert np.isclose(dist2, 0.02)  # Distance squared: 0.1^2 + 0.1^2
    assert np.allclose(closest_pt, [0.0, 0.0])  # Closest point in cloud at index 0


def test_neighbor_search_point_3d():
    """Test single nearest neighbor search with point query in 3D"""
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    cloud = tf.PointCloud(points)

    query = tf.Point([0.2, 0.2, 0.2])

    # Test without radius
    idx, dist2, closest_pt = tf.neighbor_search(cloud, query)
    assert idx == 0
    assert np.isclose(dist2, 0.12)  # 0.2^2 + 0.2^2 + 0.2^2
    assert np.allclose(closest_pt, [0.0, 0.0, 0.0])  # Closest point in cloud at index 0

    # Test with explicit radius=None
    idx2, dist2_2, closest_pt2 = tf.neighbor_search(cloud, query, radius=None)
    assert idx2 == 0
    assert np.isclose(dist2_2, 0.12)
    assert np.allclose(closest_pt2, [0.0, 0.0, 0.0])

    # Test with radius
    idx3, dist2_3, closest_pt3 = tf.neighbor_search(cloud, query, radius=10.0)
    assert idx3 == 0
    assert np.isclose(dist2_3, 0.12)
    assert np.allclose(closest_pt3, [0.0, 0.0, 0.0])


def test_neighbor_search_knn_point():
    """Test k-nearest neighbors search"""
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    cloud = tf.PointCloud(points)

    query = tf.Point([0.1, 0.1, 0.1])

    # Test WITHOUT radius (radius=None)
    results = tf.neighbor_search(cloud, query, k=3)
    assert len(results) == 3
    # Results should be sorted by distance
    for i in range(len(results) - 1):
        assert results[i][1] <= results[i + 1][1]  # distance² is increasing
    # First result should be closest to origin
    assert results[0][0] == 0
    assert np.isclose(results[0][1], 0.03)  # 0.1^2 * 3
    assert np.allclose(results[0][2], [0.0, 0.0, 0.0])  # Cloud point at index 0

    # Verify each result returns the corresponding cloud point
    for idx, dist2, closest_pt in results:
        assert np.allclose(closest_pt, points[idx])

    # Test WITH explicit radius=None
    results2 = tf.neighbor_search(cloud, query, k=3, radius=None)
    assert len(results2) == 3
    for idx, dist2, closest_pt in results2:
        assert np.allclose(closest_pt, points[idx])

    # Test WITH a large radius
    results3 = tf.neighbor_search(cloud, query, k=3, radius=10.0)
    assert len(results3) == 3
    for idx, dist2, closest_pt in results3:
        assert np.allclose(closest_pt, points[idx])


def test_neighbor_search_with_radius():
    """Test neighbor search with radius constraint"""
    points = np.array([[0, 0, 0], [10, 0, 0], [0, 10, 0]], dtype=np.float32)
    cloud = tf.PointCloud(points)

    query = tf.Point([0.5, 0.5, 0.5])

    # Search with small radius - should find the origin
    idx, dist2, closest_pt = tf.neighbor_search(cloud, query, radius=2.0)
    assert idx == 0
    assert dist2 < 2.0

    # Search with very small radius - might not find anything (depends on implementation)
    # If no result within radius, this might return the furthest point or raise


def test_neighbor_search_knn_with_radius():
    """Test KNN search with radius constraint"""
    points = np.array([[0, 0, 0], [1, 0, 0], [10, 0, 0], [0, 10, 0]], dtype=np.float32)
    cloud = tf.PointCloud(points)

    query = tf.Point([0.0, 0.0, 0.0])

    # Request 4 neighbors but limit radius to exclude far points
    results = tf.neighbor_search(cloud, query, radius=5.0, k=4)

    # Should only get points within radius=5.0
    # Points at [0,0,0] and [1,0,0] are within radius
    # Points at [10,0,0] and [0,10,0] are outside
    assert all(r[1] <= 25.0 for r in results)  # All within radius² = 25


def test_neighbor_search_segment_2d():
    """Test neighbor search with segment query in 2D"""
    points = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float32)
    cloud = tf.PointCloud(points)

    # Segment from [0.5, 0] to [0.5, 1]
    query = tf.Segment([[0.5, 0], [0.5, 1]])
    idx, dist2, closest_pt = tf.neighbor_search(cloud, query)

    # Should find one of the points at distance 0.5
    assert dist2 == pytest.approx(0.25)  # 0.5^2


def test_neighbor_search_segment_3d():
    """Test neighbor search with segment query in 3D"""
    points = np.array([[0, 0, 0], [1, 1, 1], [2, 0, 0]], dtype=np.float32)
    cloud = tf.PointCloud(points)

    # Segment along x-axis
    query = tf.Segment([[0, 0.5, 0], [2, 0.5, 0]])
    idx, dist2, closest_pt = tf.neighbor_search(cloud, query)

    # Point at [0,0,0] or [2,0,0] should be nearest
    assert idx in [0, 2]


def test_neighbor_search_polygon_2d():
    """Test neighbor search with polygon query in 2D"""
    points = np.array([[0, 0], [5, 0], [5, 5], [0, 5]], dtype=np.float32)
    cloud = tf.PointCloud(points)

    # Triangle around origin
    query = tf.Polygon([[1, 1], [2, 1], [1.5, 2]])
    idx, dist2, closest_pt = tf.neighbor_search(cloud, query)

    # Origin should be closest
    assert idx == 0


def test_neighbor_search_polygon_3d():
    """Test neighbor search with polygon query in 3D"""
    points = np.array([[0, 0, 0], [5, 0, 0], [0, 5, 0], [0, 0, 5]], dtype=np.float32)
    cloud = tf.PointCloud(points)

    # Triangle in xy-plane
    query = tf.Polygon([[1, 1, 0], [2, 1, 0], [1.5, 2, 0]])
    idx, dist2, closest_pt = tf.neighbor_search(cloud, query)

    # Origin should be closest
    assert idx == 0


def test_neighbor_search_ray_3d():
    """Test neighbor search with ray query"""
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    cloud = tf.PointCloud(points)

    # Ray from [0, 0.5, 0] along x-axis
    # Float lists become float64 from numpy, Ray keeps float64
    query = tf.Ray(origin=[0, 0.5, 0], direction=[1, 0, 0])
    idx, dist2, closest_pt = tf.neighbor_search(cloud, query)

    # Some point should be found
    assert idx >= 0
    # Verify we get back a cloud point
    assert np.allclose(closest_pt, points[idx])


def test_neighbor_search_line_3d():
    """Test neighbor search with line query"""
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    cloud = tf.PointCloud(points)

    # Line through [0, 0.5, 0] along x-axis
    # Float lists become float64 from numpy, Line keeps float64
    query = tf.Line(origin=[0, 0.5, 0], direction=[1, 0, 0])
    idx, dist2, closest_pt = tf.neighbor_search(cloud, query)

    # Some point should be found
    assert idx >= 0
    # Verify we get back a cloud point
    assert np.allclose(closest_pt, points[idx])


def test_neighbor_search_numpy_array():
    """Test neighbor search with raw numpy array (treated as point)"""
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    cloud = tf.PointCloud(points)

    # Query with raw numpy array
    query = np.array([0.1, 0.1, 0.1], dtype=np.float32)
    idx, dist2, closest_pt = tf.neighbor_search(cloud, query)

    assert idx == 0
    assert np.isclose(dist2, 0.03)


def test_neighbor_search_double_precision():
    """Test neighbor search with double precision"""
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    cloud = tf.PointCloud(points)

    # Create query with explicit float64 array - Point will keep float64
    query = tf.Point(np.array([0.1, 0.1, 0.1], dtype=np.float64))
    idx, dist2, closest_pt = tf.neighbor_search(cloud, query)

    assert idx == 0
    assert isinstance(dist2, float)
    assert np.allclose(closest_pt, [0.0, 0.0, 0.0])


def test_neighbor_search_dimension_mismatch():
    """Test that dimension mismatch raises appropriate error"""
    points_3d = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)
    cloud_3d = tf.PointCloud(points_3d)

    # Integer list [0, 0] will be converted to float32 by Point
    query_2d = tf.Point([0, 0])

    with pytest.raises(ValueError, match="Dimension mismatch"):
        tf.neighbor_search(cloud_3d, query_2d)


def test_neighbor_search_invalid_k():
    """Test that invalid k value raises error"""
    points = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)
    cloud = tf.PointCloud(points)
    query = tf.Point([0, 0, 0])

    with pytest.raises(ValueError, match="k must be a positive integer"):
        tf.neighbor_search(cloud, query, k=0)

    with pytest.raises(ValueError, match="k must be a positive integer"):
        tf.neighbor_search(cloud, query, k=-1)


def test_neighbor_search_knn_all_points():
    """Test KNN when k equals number of points"""
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    cloud = tf.PointCloud(points)

    query = tf.Point([0, 0, 0])
    results = tf.neighbor_search(cloud, query, k=3)

    assert len(results) == 3
    # Check that all point indices are present
    indices = [r[0] for r in results]
    assert set(indices) == {0, 1, 2}


def test_neighbor_search_knn_more_than_available():
    """Test KNN when k is larger than number of points"""
    points = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)
    cloud = tf.PointCloud(points)

    query = tf.Point([0, 0, 0])
    results = tf.neighbor_search(cloud, query, k=10)

    # Should return at most 2 results (all available points)
    assert len(results) <= 2


def test_neighbor_search_with_transformation_2d():
    """Test neighbor search with 2D transformation (last column is translation)"""
    # Create point cloud at origin and along x-axis
    # Original points: [0,0], [1,0], [0,1]
    points = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float32)
    cloud = tf.PointCloud(points)

    # Create transformation: identity rotation with translation [5, 3]
    # Format: [r11 r12 tx]
    #         [r21 r22 ty]
    #         [  0   0  1]
    # After transformation, points are at: [5,3], [6,3], [5,4] (world coordinates)
    transformation = np.array([
        [1, 0, 5],
        [0, 1, 3],
        [0, 0, 1]
    ], dtype=np.float32)

    cloud.transformation = transformation

    # Query point at [5.1, 3.1] in world coordinates - should be close to transformed origin [5, 3]
    query = tf.Point([5.1, 3.1])
    idx, dist2, closest_pt = tf.neighbor_search(cloud, query)

    assert idx == 0  # Should find the origin point (index 0)
    # Distance should be small (0.1^2 + 0.1^2 = 0.02)
    assert np.isclose(dist2, 0.02, atol=1e-5)
    # IMPORTANT: closest_pt is in world coordinates (transformed point), not original
    assert np.allclose(closest_pt, [5.0, 3.0], atol=1e-5)


def test_neighbor_search_with_transformation_3d():
    """Test neighbor search with 3D transformation (last column is translation)"""
    # Create point cloud at origin and along axes
    # Original points: [0,0,0], [1,0,0], [0,1,0], [0,0,1]
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    cloud = tf.PointCloud(points)

    # Create transformation: identity rotation with translation [10, 5, 2]
    # Format: [r11 r12 r13 tx]
    #         [r21 r22 r23 ty]
    #         [r31 r32 r33 tz]
    #         [  0   0   0  1]
    # After transformation, points are at: [10,5,2], [11,5,2], [10,6,2], [10,5,3] (world coordinates)
    transformation = np.array([
        [1, 0, 0, 10],
        [0, 1, 0,  5],
        [0, 0, 1,  2],
        [0, 0, 0,  1]
    ], dtype=np.float32)

    cloud.transformation = transformation

    # Query point at [10.2, 5.1, 2.1] in world coordinates - close to transformed origin [10, 5, 2]
    query = tf.Point([10.2, 5.1, 2.1])
    idx, dist2, closest_pt = tf.neighbor_search(cloud, query)

    assert idx == 0  # Should find the origin point (index 0)
    # Distance should be 0.2^2 + 0.1^2 + 0.1^2 = 0.06
    assert np.isclose(dist2, 0.06, atol=1e-5)
    # IMPORTANT: closest_pt is in world coordinates (transformed point)
    assert np.allclose(closest_pt, [10.0, 5.0, 2.0], atol=1e-5)


def test_neighbor_search_with_transformation_knn():
    """Test KNN search with transformation"""
    # Create point cloud
    # Original points: [0,0,0], [1,0,0], [0,1,0], [0,0,1]
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    cloud = tf.PointCloud(points)

    # Apply translation [5, 5, 5]
    # After transformation, points are at: [5,5,5], [6,5,5], [5,6,5], [5,5,6] (world coordinates)
    transformation = np.array([
        [1, 0, 0, 5],
        [0, 1, 0, 5],
        [0, 0, 1, 5],
        [0, 0, 0, 1]
    ], dtype=np.float32)

    cloud.transformation = transformation

    # Query at transformed origin in world coordinates
    query = tf.Point([5.0, 5.0, 5.0])
    results = tf.neighbor_search(cloud, query, k=3)

    assert len(results) == 3
    # First result should be the origin (index 0)
    assert results[0][0] == 0
    assert np.isclose(results[0][1], 0.0, atol=1e-5)  # Distance should be 0
    # IMPORTANT: closest_pt is in world coordinates
    assert np.allclose(results[0][2], [5.0, 5.0, 5.0], atol=1e-5)

    # All results should be sorted by distance
    for i in range(len(results) - 1):
        assert results[i][1] <= results[i + 1][1]


def test_neighbor_search_transformation_clear():
    """Test clearing transformation"""
    points = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)
    cloud = tf.PointCloud(points)

    # Set transformation
    transformation = np.array([
        [1, 0, 0, 10],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)

    cloud.transformation = transformation

    # Query should find point at transformed location [10, 0, 0]
    query1 = tf.Point([10.1, 0, 0])
    idx1, dist2_1, _ = tf.neighbor_search(cloud, query1)
    assert idx1 == 0
    assert np.isclose(dist2_1, 0.01, atol=1e-5)

    # Clear transformation
    cloud.transformation = None

    # Now query at original location [0, 0, 0]
    query2 = tf.Point([0.1, 0, 0])
    idx2, dist2_2, _ = tf.neighbor_search(cloud, query2)
    assert idx2 == 0
    assert np.isclose(dist2_2, 0.01, atol=1e-5)


def test_neighbor_search_with_rotation_transformation():
    """Test neighbor search with rotation transformation"""
    # Create point cloud along x-axis
    # Original points: [1,0,0], [2,0,0]
    points = np.array([[1, 0, 0], [2, 0, 0]], dtype=np.float32)
    cloud = tf.PointCloud(points)

    # Create 90-degree rotation around z-axis (x -> y)
    # After rotation, [1,0,0] -> [0,1,0] and [2,0,0] -> [0,2,0] (world coordinates)
    # Format: [r11 r12 r13 tx]
    #         [r21 r22 r23 ty]
    #         [r31 r32 r33 tz]
    #         [  0   0   0  1]
    transformation = np.array([
        [ 0, -1, 0, 0],
        [ 1,  0, 0, 0],
        [ 0,  0, 1, 0],
        [ 0,  0, 0, 1]
    ], dtype=np.float32)

    cloud.transformation = transformation

    # Query point at [0, 1.1, 0] in world coordinates - close to rotated [1, 0, 0]
    query = tf.Point([0, 1.1, 0])
    idx, dist2, closest_pt = tf.neighbor_search(cloud, query)

    assert idx == 0  # Should find first point
    assert np.isclose(dist2, 0.01, atol=1e-4)  # Distance 0.1^2
    # IMPORTANT: closest_pt is in world coordinates (rotated point)
    assert np.allclose(closest_pt, [0.0, 1.0, 0.0], atol=1e-5)


def test_neighbor_search_with_scale_transformation():
    """Test neighbor search with scaling transformation"""
    # Create point cloud
    # Original points: [1,0,0], [2,0,0]
    points = np.array([[1, 0, 0], [2, 0, 0]], dtype=np.float32)
    cloud = tf.PointCloud(points)

    # Create scaling transformation: scale by 2 in all directions
    # After scaling: [1,0,0] -> [2,0,0] and [2,0,0] -> [4,0,0] (world coordinates)
    transformation = np.array([
        [2, 0, 0, 0],
        [0, 2, 0, 0],
        [0, 0, 2, 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)

    cloud.transformation = transformation

    # Query point at [2.1, 0, 0] in world coordinates - close to scaled [1, 0, 0]
    query = tf.Point([2.1, 0, 0])
    idx, dist2, closest_pt = tf.neighbor_search(cloud, query)

    assert idx == 0  # Should find first point
    assert np.isclose(dist2, 0.01, atol=1e-5)  # Distance 0.1^2
    # IMPORTANT: closest_pt is in world coordinates (scaled point)
    assert np.allclose(closest_pt, [2.0, 0.0, 0.0], atol=1e-5)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
