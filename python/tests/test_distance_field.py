"""
Test distance_field functionality

Copyright (c) 2025 Žiga Sajovic, XLAB
"""

import sys
import pytest
import numpy as np
import trueform as tf


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_distance_field_plane_3d(dtype):
    """Test distance field to plane in 3D (signed distance)"""
    # Plane at z=0 (xy-plane)
    plane = tf.Plane(np.array([0, 0, 1, 0], dtype=dtype))

    # Points above, on, and below the plane
    points = np.array([
        [0.0, 0.0, 2.0],   # 2 units above
        [1.0, 1.0, 0.0],   # on plane
        [0.5, 0.5, -1.0],  # 1 unit below
        [2.0, 2.0, 3.0],   # 3 units above
    ], dtype=dtype)

    distances = tf.distance_field(points, plane)

    # Check shape
    assert distances.shape == (4,), f"Expected shape (4,), got {distances.shape}"

    # Check signed distances (positive above, negative below, zero on plane)
    assert np.isclose(distances[0], 2.0), f"Expected distance 2.0, got {distances[0]}"
    assert np.isclose(distances[1], 0.0), f"Expected distance 0.0, got {distances[1]}"
    assert np.isclose(distances[2], -1.0), f"Expected distance -1.0, got {distances[2]}"
    assert np.isclose(distances[3], 3.0), f"Expected distance 3.0, got {distances[3]}"


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_distance_field_segment_2d(dtype):
    """Test distance field to segment in 2D"""
    # Horizontal segment from (0, 0) to (2, 0)
    segment = tf.Segment(np.array([[0.0, 0.0], [2.0, 0.0]], dtype=dtype))

    # Points at various positions
    points = np.array([
        [1.0, 0.0],   # on segment (midpoint)
        [1.0, 1.0],   # perpendicular distance 1.0
        [3.0, 0.0],   # beyond end, distance 1.0
        [-1.0, 0.0],  # beyond start, distance 1.0
        [1.0, 2.0],   # perpendicular distance 2.0
    ], dtype=dtype)

    distances = tf.distance_field(points, segment)

    # Check shape
    assert distances.shape == (5,), f"Expected shape (5,), got {distances.shape}"

    # Check distances
    assert np.isclose(distances[0], 0.0), f"Expected distance 0.0, got {distances[0]}"
    assert np.isclose(distances[1], 1.0), f"Expected distance 1.0, got {distances[1]}"
    assert np.isclose(distances[2], 1.0), f"Expected distance 1.0, got {distances[2]}"
    assert np.isclose(distances[3], 1.0), f"Expected distance 1.0, got {distances[3]}"
    assert np.isclose(distances[4], 2.0), f"Expected distance 2.0, got {distances[4]}"


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_distance_field_segment_3d(dtype):
    """Test distance field to segment in 3D"""
    # Segment along x-axis from (0, 0, 0) to (2, 0, 0)
    segment = tf.Segment(np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=dtype))

    # Points at various positions
    points = np.array([
        [1.0, 0.0, 0.0],   # on segment (midpoint)
        [1.0, 1.0, 0.0],   # perpendicular distance 1.0
        [1.0, 0.0, 1.0],   # perpendicular distance 1.0
        [1.0, 1.0, 1.0],   # perpendicular distance sqrt(2)
        [3.0, 0.0, 0.0],   # beyond end, distance 1.0
    ], dtype=dtype)

    distances = tf.distance_field(points, segment)

    # Check shape
    assert distances.shape == (5,), f"Expected shape (5,), got {distances.shape}"

    # Check distances
    assert np.isclose(distances[0], 0.0), f"Expected distance 0.0, got {distances[0]}"
    assert np.isclose(distances[1], 1.0), f"Expected distance 1.0, got {distances[1]}"
    assert np.isclose(distances[2], 1.0), f"Expected distance 1.0, got {distances[2]}"
    assert np.isclose(distances[3], np.sqrt(2.0)), f"Expected distance sqrt(2), got {distances[3]}"
    assert np.isclose(distances[4], 1.0), f"Expected distance 1.0, got {distances[4]}"


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_distance_field_polygon_2d(dtype):
    """Test distance field to polygon in 2D"""
    # Unit square
    square = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0]
    ], dtype=dtype)
    poly = tf.Polygon(square)

    # Points inside, on, and outside the square
    points = np.array([
        [0.5, 0.5],   # inside (center)
        [0.0, 0.5],   # on edge
        [2.0, 0.5],   # outside, distance 1.0
        [-1.0, 0.5],  # outside, distance 1.0
        [0.5, 2.0],   # outside, distance 1.0
    ], dtype=dtype)

    distances = tf.distance_field(points, poly)

    # Check shape
    assert distances.shape == (5,), f"Expected shape (5,), got {distances.shape}"

    # All distances should be non-negative (unsigned distance)
    assert np.all(distances >= 0), "All distances should be non-negative"

    # Points on or inside should have distance 0 or small
    assert distances[0] <= 0.5, f"Expected small distance for center point, got {distances[0]}"
    assert np.isclose(distances[1], 0.0), f"Expected distance 0.0 for edge point, got {distances[1]}"

    # Points outside should have positive distance
    assert np.isclose(distances[2], 1.0), f"Expected distance 1.0, got {distances[2]}"
    assert np.isclose(distances[3], 1.0), f"Expected distance 1.0, got {distances[3]}"
    assert np.isclose(distances[4], 1.0), f"Expected distance 1.0, got {distances[4]}"


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_distance_field_polygon_3d(dtype):
    """Test distance field to polygon in 3D"""
    # Triangle in xy-plane at z=0
    triangle = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 1.0, 0.0]
    ], dtype=dtype)
    poly = tf.Polygon(triangle)

    # Points at various positions
    points = np.array([
        [0.5, 0.3, 0.0],   # on triangle
        [0.5, 0.3, 1.0],   # above triangle, distance ~1.0
        [0.5, 0.3, -1.0],  # below triangle, distance ~1.0
        [2.0, 0.0, 0.0],   # outside in plane
    ], dtype=dtype)

    distances = tf.distance_field(points, poly)

    # Check shape
    assert distances.shape == (4,), f"Expected shape (4,), got {distances.shape}"

    # Point on triangle should have distance 0
    assert np.isclose(distances[0], 0.0, atol=1e-5), f"Expected distance 0.0, got {distances[0]}"

    # Points above/below should have positive distance
    assert distances[1] > 0, f"Expected positive distance, got {distances[1]}"
    assert distances[2] > 0, f"Expected positive distance, got {distances[2]}"
    assert distances[3] > 0, f"Expected positive distance, got {distances[3]}"


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_distance_field_line_2d(dtype):
    """Test distance field to line in 2D"""
    # Vertical line at x=1
    line = tf.Line(
        origin=np.array([1.0, 0.0], dtype=dtype),
        direction=np.array([0.0, 1.0], dtype=dtype)
    )

    # Points at various positions
    points = np.array([
        [1.0, 0.0],   # on line
        [1.0, 5.0],   # on line (different y)
        [0.0, 0.0],   # distance 1.0
        [2.0, 0.0],   # distance 1.0
        [3.0, 0.0],   # distance 2.0
    ], dtype=dtype)

    distances = tf.distance_field(points, line)

    # Check shape
    assert distances.shape == (5,), f"Expected shape (5,), got {distances.shape}"

    # Check distances
    assert np.isclose(distances[0], 0.0), f"Expected distance 0.0, got {distances[0]}"
    assert np.isclose(distances[1], 0.0), f"Expected distance 0.0, got {distances[1]}"
    assert np.isclose(distances[2], 1.0), f"Expected distance 1.0, got {distances[2]}"
    assert np.isclose(distances[3], 1.0), f"Expected distance 1.0, got {distances[3]}"
    assert np.isclose(distances[4], 2.0), f"Expected distance 2.0, got {distances[4]}"


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_distance_field_line_3d(dtype):
    """Test distance field to line in 3D"""
    # Line along z-axis through origin
    line = tf.Line(
        origin=np.array([0.0, 0.0, 0.0], dtype=dtype),
        direction=np.array([0.0, 0.0, 1.0], dtype=dtype)
    )

    # Points at various positions
    points = np.array([
        [0.0, 0.0, 0.0],   # on line
        [0.0, 0.0, 5.0],   # on line (different z)
        [1.0, 0.0, 0.0],   # distance 1.0
        [0.0, 1.0, 0.0],   # distance 1.0
        [1.0, 1.0, 0.0],   # distance sqrt(2)
    ], dtype=dtype)

    distances = tf.distance_field(points, line)

    # Check shape
    assert distances.shape == (5,), f"Expected shape (5,), got {distances.shape}"

    # Check distances
    assert np.isclose(distances[0], 0.0), f"Expected distance 0.0, got {distances[0]}"
    assert np.isclose(distances[1], 0.0), f"Expected distance 0.0, got {distances[1]}"
    assert np.isclose(distances[2], 1.0), f"Expected distance 1.0, got {distances[2]}"
    assert np.isclose(distances[3], 1.0), f"Expected distance 1.0, got {distances[3]}"
    assert np.isclose(distances[4], np.sqrt(2.0)), f"Expected distance sqrt(2), got {distances[4]}"


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_distance_field_aabb_2d(dtype):
    """Test distance field to AABB in 2D"""
    # AABB from [0,0] to [1,1]
    aabb = tf.AABB(
        min=np.array([0.0, 0.0], dtype=dtype),
        max=np.array([1.0, 1.0], dtype=dtype)
    )

    # Points inside, on, and outside the AABB
    points = np.array([
        [0.5, 0.5],   # inside (center), distance 0
        [0.0, 0.5],   # on edge, distance 0
        [2.0, 0.5],   # outside, distance 1.0
        [-1.0, 0.5],  # outside, distance 1.0
        [2.0, 2.0],   # outside corner, distance sqrt(2)
    ], dtype=dtype)

    distances = tf.distance_field(points, aabb)

    # Check shape
    assert distances.shape == (5,), f"Expected shape (5,), got {distances.shape}"

    # Check distances
    assert np.isclose(distances[0], 0.0), f"Expected distance 0.0, got {distances[0]}"
    assert np.isclose(distances[1], 0.0), f"Expected distance 0.0, got {distances[1]}"
    assert np.isclose(distances[2], 1.0), f"Expected distance 1.0, got {distances[2]}"
    assert np.isclose(distances[3], 1.0), f"Expected distance 1.0, got {distances[3]}"
    assert np.isclose(distances[4], np.sqrt(2.0)), f"Expected distance sqrt(2), got {distances[4]}"


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_distance_field_aabb_3d(dtype):
    """Test distance field to AABB in 3D"""
    # AABB cube from [0,0,0] to [1,1,1]
    aabb = tf.AABB(
        min=np.array([0.0, 0.0, 0.0], dtype=dtype),
        max=np.array([1.0, 1.0, 1.0], dtype=dtype)
    )

    # Points inside, on, and outside the AABB
    points = np.array([
        [0.5, 0.5, 0.5],   # inside (center), distance 0
        [0.0, 0.5, 0.5],   # on face, distance 0
        [2.0, 0.5, 0.5],   # outside, distance 1.0
        [0.5, 0.5, 2.0],   # outside, distance 1.0
        [2.0, 2.0, 2.0],   # outside corner, distance sqrt(3)
    ], dtype=dtype)

    distances = tf.distance_field(points, aabb)

    # Check shape
    assert distances.shape == (5,), f"Expected shape (5,), got {distances.shape}"

    # Check distances
    assert np.isclose(distances[0], 0.0), f"Expected distance 0.0, got {distances[0]}"
    assert np.isclose(distances[1], 0.0), f"Expected distance 0.0, got {distances[1]}"
    assert np.isclose(distances[2], 1.0), f"Expected distance 1.0, got {distances[2]}"
    assert np.isclose(distances[3], 1.0), f"Expected distance 1.0, got {distances[3]}"
    assert np.isclose(distances[4], np.sqrt(3.0)), f"Expected distance sqrt(3), got {distances[4]}"


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_distance_field_with_point_cloud(dtype):
    """Test distance_field with PointCloud object instead of numpy array"""
    # Create a plane
    plane = tf.Plane(np.array([0, 0, 1, 0], dtype=dtype))

    # Create points as PointCloud
    points_array = np.array([
        [0.0, 0.0, 2.0],
        [1.0, 1.0, 0.0],
        [0.5, 0.5, -1.0],
    ], dtype=dtype)
    point_cloud = tf.PointCloud(points_array)

    # Compute distances using PointCloud
    distances = tf.distance_field(point_cloud, plane)

    # Check shape
    assert distances.shape == (3,), f"Expected shape (3,), got {distances.shape}"

    # Check signed distances
    assert np.isclose(distances[0], 2.0), f"Expected distance 2.0, got {distances[0]}"
    assert np.isclose(distances[1], 0.0), f"Expected distance 0.0, got {distances[1]}"
    assert np.isclose(distances[2], -1.0), f"Expected distance -1.0, got {distances[2]}"


def test_distance_field_dimension_mismatch():
    """Test that dimension mismatch raises an error"""
    points_2d = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    segment_3d = tf.Segment([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

    with pytest.raises(ValueError) as excinfo:
        tf.distance_field(points_2d, segment_3d)
    assert "Dimension mismatch" in str(excinfo.value)


def test_distance_field_dtype_mismatch():
    """Test that dtype mismatch raises an error"""
    points = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    plane = tf.Plane(np.array([0, 0, 1, 0], dtype=np.float64))

    with pytest.raises(TypeError) as excinfo:
        tf.distance_field(points, plane)
    assert "Dtype mismatch" in str(excinfo.value)


def test_distance_field_invalid_input():
    """Test that invalid input raises appropriate errors"""
    plane = tf.Plane([0, 0, 1, 0])

    # Not a numpy array or PointCloud
    with pytest.raises(TypeError) as excinfo:
        tf.distance_field([[0, 0, 0]], plane)
    assert "Expected numpy array or PointCloud" in str(excinfo.value)

    # Wrong shape (1D array)
    points_1d = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    with pytest.raises(ValueError) as excinfo:
        tf.distance_field(points_1d, plane)
    assert "Expected 2D array" in str(excinfo.value)

    # Wrong number of dimensions (4D points)
    points_4d = np.array([[0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    with pytest.raises(ValueError) as excinfo:
        tf.distance_field(points_4d, plane)
    assert "Expected 2D or 3D points" in str(excinfo.value)


def test_distance_field_large_batch():
    """Test distance_field with large number of points (vectorization)"""
    # Create 1000 random points
    np.random.seed(42)
    points = np.random.rand(1000, 3).astype(np.float32)

    # Plane at z=0
    plane = tf.Plane(np.array([0, 0, 1, 0], dtype=np.float32))

    # Compute distances
    distances = tf.distance_field(points, plane)

    # Check shape
    assert distances.shape == (1000,), f"Expected shape (1000,), got {distances.shape}"

    # Check that all distances are equal to z-coordinates (signed)
    expected_distances = points[:, 2]
    assert np.allclose(distances, expected_distances), "Distances should equal z-coordinates"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
