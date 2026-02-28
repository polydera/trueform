"""
Test batched distance (replaces distance_field tests)

Copyright (c) 2025 Žiga Sajovic, XLAB
"""

import sys
import pytest
import numpy as np
import trueform as tf


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_distance_plane_3d(dtype):
    """Test batch distance to plane in 3D (signed distance)"""
    # Plane at z=0 (xy-plane)
    plane = tf.Plane(np.array([0, 0, 1, 0], dtype=dtype))

    # Points above, on, and below the plane
    points = tf.Point(np.array([
        [0.0, 0.0, 2.0],   # 2 units above
        [1.0, 1.0, 0.0],   # on plane
        [0.5, 0.5, -1.0],  # 1 unit below
        [2.0, 2.0, 3.0],   # 3 units above
    ], dtype=dtype))

    distances = tf.distance(points, plane)

    assert distances.shape == (4,)
    assert np.isclose(distances[0], 2.0)
    assert np.isclose(distances[1], 0.0)
    assert np.isclose(distances[2], -1.0)
    assert np.isclose(distances[3], 3.0)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_distance_segment_2d(dtype):
    """Test batch distance to segment in 2D"""
    segment = tf.Segment(np.array([[0.0, 0.0], [2.0, 0.0]], dtype=dtype))

    points = tf.Point(np.array([
        [1.0, 0.0],   # on segment (midpoint)
        [1.0, 1.0],   # perpendicular distance 1.0
        [3.0, 0.0],   # beyond end, distance 1.0
        [-1.0, 0.0],  # beyond start, distance 1.0
        [1.0, 2.0],   # perpendicular distance 2.0
    ], dtype=dtype))

    distances = tf.distance(points, segment)

    assert distances.shape == (5,)
    assert np.isclose(distances[0], 0.0)
    assert np.isclose(distances[1], 1.0)
    assert np.isclose(distances[2], 1.0)
    assert np.isclose(distances[3], 1.0)
    assert np.isclose(distances[4], 2.0)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_distance_segment_3d(dtype):
    """Test batch distance to segment in 3D"""
    segment = tf.Segment(np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=dtype))

    points = tf.Point(np.array([
        [1.0, 0.0, 0.0],   # on segment (midpoint)
        [1.0, 1.0, 0.0],   # perpendicular distance 1.0
        [1.0, 0.0, 1.0],   # perpendicular distance 1.0
        [1.0, 1.0, 1.0],   # perpendicular distance sqrt(2)
        [3.0, 0.0, 0.0],   # beyond end, distance 1.0
    ], dtype=dtype))

    distances = tf.distance(points, segment)

    assert distances.shape == (5,)
    assert np.isclose(distances[0], 0.0)
    assert np.isclose(distances[1], 1.0)
    assert np.isclose(distances[2], 1.0)
    assert np.isclose(distances[3], np.sqrt(2.0))
    assert np.isclose(distances[4], 1.0)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_distance_polygon_2d(dtype):
    """Test batch distance to polygon in 2D"""
    square = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0]
    ], dtype=dtype)
    poly = tf.Polygon(square)

    points = tf.Point(np.array([
        [0.5, 0.5],   # inside (center)
        [0.0, 0.5],   # on edge
        [2.0, 0.5],   # outside, distance 1.0
        [-1.0, 0.5],  # outside, distance 1.0
        [0.5, 2.0],   # outside, distance 1.0
    ], dtype=dtype))

    distances = tf.distance(points, poly)

    assert distances.shape == (5,)
    assert np.all(distances >= 0)
    assert distances[0] <= 0.5
    assert np.isclose(distances[1], 0.0)
    assert np.isclose(distances[2], 1.0)
    assert np.isclose(distances[3], 1.0)
    assert np.isclose(distances[4], 1.0)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_distance_polygon_3d(dtype):
    """Test batch distance to polygon in 3D"""
    triangle = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 1.0, 0.0]
    ], dtype=dtype)
    poly = tf.Polygon(triangle)

    points = tf.Point(np.array([
        [0.5, 0.3, 0.0],   # on triangle
        [0.5, 0.3, 1.0],   # above triangle, distance ~1.0
        [0.5, 0.3, -1.0],  # below triangle, distance ~1.0
        [2.0, 0.0, 0.0],   # outside in plane
    ], dtype=dtype))

    distances = tf.distance(points, poly)

    assert distances.shape == (4,)
    assert np.isclose(distances[0], 0.0, atol=1e-5)
    assert distances[1] > 0
    assert distances[2] > 0
    assert distances[3] > 0


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_distance_line_2d(dtype):
    """Test batch distance to line in 2D"""
    line = tf.Line(
        origin=np.array([1.0, 0.0], dtype=dtype),
        direction=np.array([0.0, 1.0], dtype=dtype))

    points = tf.Point(np.array([
        [1.0, 0.0],   # on line
        [1.0, 5.0],   # on line (different y)
        [0.0, 0.0],   # distance 1.0
        [2.0, 0.0],   # distance 1.0
        [3.0, 0.0],   # distance 2.0
    ], dtype=dtype))

    distances = tf.distance(points, line)

    assert distances.shape == (5,)
    assert np.isclose(distances[0], 0.0)
    assert np.isclose(distances[1], 0.0)
    assert np.isclose(distances[2], 1.0)
    assert np.isclose(distances[3], 1.0)
    assert np.isclose(distances[4], 2.0)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_distance_line_3d(dtype):
    """Test batch distance to line in 3D"""
    line = tf.Line(
        origin=np.array([0.0, 0.0, 0.0], dtype=dtype),
        direction=np.array([0.0, 0.0, 1.0], dtype=dtype))

    points = tf.Point(np.array([
        [0.0, 0.0, 0.0],   # on line
        [0.0, 0.0, 5.0],   # on line (different z)
        [1.0, 0.0, 0.0],   # distance 1.0
        [0.0, 1.0, 0.0],   # distance 1.0
        [1.0, 1.0, 0.0],   # distance sqrt(2)
    ], dtype=dtype))

    distances = tf.distance(points, line)

    assert distances.shape == (5,)
    assert np.isclose(distances[0], 0.0)
    assert np.isclose(distances[1], 0.0)
    assert np.isclose(distances[2], 1.0)
    assert np.isclose(distances[3], 1.0)
    assert np.isclose(distances[4], np.sqrt(2.0))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_distance_aabb_2d(dtype):
    """Test batch distance to AABB in 2D"""
    aabb = tf.AABB(
        min=np.array([0.0, 0.0], dtype=dtype),
        max=np.array([1.0, 1.0], dtype=dtype))

    points = tf.Point(np.array([
        [0.5, 0.5],   # inside (center), distance 0
        [0.0, 0.5],   # on edge, distance 0
        [2.0, 0.5],   # outside, distance 1.0
        [-1.0, 0.5],  # outside, distance 1.0
        [2.0, 2.0],   # outside corner, distance sqrt(2)
    ], dtype=dtype))

    distances = tf.distance(points, aabb)

    assert distances.shape == (5,)
    assert np.isclose(distances[0], 0.0)
    assert np.isclose(distances[1], 0.0)
    assert np.isclose(distances[2], 1.0)
    assert np.isclose(distances[3], 1.0)
    assert np.isclose(distances[4], np.sqrt(2.0))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_distance_aabb_3d(dtype):
    """Test batch distance to AABB in 3D"""
    aabb = tf.AABB(
        min=np.array([0.0, 0.0, 0.0], dtype=dtype),
        max=np.array([1.0, 1.0, 1.0], dtype=dtype))

    points = tf.Point(np.array([
        [0.5, 0.5, 0.5],   # inside (center), distance 0
        [0.0, 0.5, 0.5],   # on face, distance 0
        [2.0, 0.5, 0.5],   # outside, distance 1.0
        [0.5, 0.5, 2.0],   # outside, distance 1.0
        [2.0, 2.0, 2.0],   # outside corner, distance sqrt(3)
    ], dtype=dtype))

    distances = tf.distance(points, aabb)

    assert distances.shape == (5,)
    assert np.isclose(distances[0], 0.0)
    assert np.isclose(distances[1], 0.0)
    assert np.isclose(distances[2], 1.0)
    assert np.isclose(distances[3], 1.0)
    assert np.isclose(distances[4], np.sqrt(3.0))


def test_distance_large_batch():
    """Test batch distance with large number of points"""
    np.random.seed(42)
    points = tf.Point(np.random.rand(1000, 3).astype(np.float32))

    # Plane at z=0
    plane = tf.Plane(np.array([0, 0, 1, 0], dtype=np.float32))

    distances = tf.distance(points, plane)

    assert distances.shape == (1000,)
    # Distances should equal z-coordinates (signed for planes)
    expected = points.data[:, 2]
    assert np.allclose(distances, expected)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
