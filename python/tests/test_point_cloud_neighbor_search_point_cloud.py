"""
Tests for PointCloud × PointCloud neighbor_search

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
REAL_DTYPES = [np.float32, np.float64]


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_point_cloud_neighbor_search_point_cloud_2d_hit(dtype):
    """Test 2D PointCloud neighbor search - hit case"""
    # Cloud 0: points at (0, 0), (1, 0), (2, 0)
    # Cloud 1: points at (0.1, 0.1), (5, 5)
    # Closest pair should be cloud0[0] <-> cloud1[0] at distance ~0.1414

    points0 = np.array([[0, 0], [1, 0], [2, 0]], dtype=dtype)
    points1 = np.array([[0.1, 0.1], [5, 5]], dtype=dtype)

    cloud0 = tf.PointCloud(points0)
    cloud1 = tf.PointCloud(points1)

    result = tf.neighbor_search(cloud0, cloud1)
    assert result is not None

    (idx0, idx1), (dist, pt0, pt1) = result

    # Check indices
    assert idx0 == 0
    assert idx1 == 0

    # Check distance (0.1^2 + 0.1^2 = 0.02)
    expected_dist = 0.1**2 + 0.1**2
    assert abs(dist - expected_dist) < 1e-5

    # Check closest points
    assert np.allclose(pt0, points0[0], atol=1e-5)
    assert np.allclose(pt1, points1[0], atol=1e-5)

    # Test symmetry
    result_sym = tf.neighbor_search(cloud1, cloud0)
    (idx0_sym, idx1_sym), (dist_sym, pt0_sym, pt1_sym) = result_sym

    assert idx0_sym == idx1
    assert idx1_sym == idx0
    assert abs(dist_sym - dist) < 1e-6
    assert np.allclose(pt0_sym, pt1, atol=1e-5)
    assert np.allclose(pt1_sym, pt0, atol=1e-5)


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_point_cloud_neighbor_search_point_cloud_3d_hit(dtype):
    """Test 3D PointCloud neighbor search - hit case"""
    points0 = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=dtype)
    points1 = np.array([[0.2, 0, 0], [5, 5, 5]], dtype=dtype)

    cloud0 = tf.PointCloud(points0)
    cloud1 = tf.PointCloud(points1)

    result = tf.neighbor_search(cloud0, cloud1)
    assert result is not None

    (idx0, idx1), (dist, pt0, pt1) = result

    assert idx0 == 0
    assert idx1 == 0

    expected_dist = 0.2**2
    assert abs(dist - expected_dist) < 1e-5

    assert np.allclose(pt0, points0[0], atol=1e-5)
    assert np.allclose(pt1, points1[0], atol=1e-5)

    # Test symmetry
    result_sym = tf.neighbor_search(cloud1, cloud0)
    (idx0_sym, idx1_sym), (dist_sym, _, _) = result_sym
    assert idx0_sym == 0 and idx1_sym == 0
    assert abs(dist_sym - dist) < 1e-6


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_point_cloud_neighbor_search_point_cloud_2d_radius_hit(dtype):
    """Test 2D with radius - within radius"""
    points0 = np.array([[0, 0], [1, 0]], dtype=dtype)
    points1 = np.array([[0.5, 0], [10, 10]], dtype=dtype)

    cloud0 = tf.PointCloud(points0)
    cloud1 = tf.PointCloud(points1)

    result = tf.neighbor_search(cloud0, cloud1, radius=1.0)
    assert result is not None

    (idx0, idx1), (dist, _, _) = result
    assert idx1 == 0
    assert dist < 1.0


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_point_cloud_neighbor_search_point_cloud_2d_radius_miss(dtype):
    """Test 2D with radius - outside radius"""
    points0 = np.array([[0, 0], [1, 0]], dtype=dtype)
    points1 = np.array([[10, 10], [20, 20]], dtype=dtype)

    cloud0 = tf.PointCloud(points0)
    cloud1 = tf.PointCloud(points1)

    result = tf.neighbor_search(cloud0, cloud1, radius=1.0)
    assert result is None


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_point_cloud_neighbor_search_point_cloud_3d_radius_hit(dtype):
    """Test 3D with radius - within radius"""
    points0 = np.array([[0, 0, 0], [1, 0, 0]], dtype=dtype)
    points1 = np.array([[0.1, 0.1, 0.1], [10, 10, 10]], dtype=dtype)

    cloud0 = tf.PointCloud(points0)
    cloud1 = tf.PointCloud(points1)

    result = tf.neighbor_search(cloud0, cloud1, radius=0.5)
    assert result is not None

    (idx0, idx1), (dist, _, _) = result
    assert idx0 == 0
    assert idx1 == 0
    expected_dist = 0.1**2 + 0.1**2 + 0.1**2
    assert abs(dist - expected_dist) < 1e-5


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_point_cloud_neighbor_search_point_cloud_identical_points(dtype):
    """Test when clouds have identical overlapping points"""
    points0 = np.array([[0, 0], [1, 0], [2, 0]], dtype=dtype)
    points1 = np.array([[1, 0], [5, 5]], dtype=dtype)

    cloud0 = tf.PointCloud(points0)
    cloud1 = tf.PointCloud(points1)

    result = tf.neighbor_search(cloud0, cloud1)
    assert result is not None

    (idx0, idx1), (dist, _, _) = result

    assert dist < 1e-10
    assert idx0 == 1
    assert idx1 == 0


def test_point_cloud_neighbor_search_point_cloud_dimension_mismatch():
    """Test that dimension mismatch raises error"""
    points2d = np.array([[0, 0], [1, 0]], dtype=np.float32)
    points3d = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)

    cloud2d = tf.PointCloud(points2d)
    cloud3d = tf.PointCloud(points3d)

    with pytest.raises(ValueError, match="Dimension mismatch"):
        tf.neighbor_search(cloud2d, cloud3d)


def test_point_cloud_neighbor_search_point_cloud_knn_not_supported():
    """Test that KNN is not supported for form-form"""
    points = np.array([[0, 0], [1, 0]], dtype=np.float32)
    cloud = tf.PointCloud(points)

    with pytest.raises(ValueError, match="KNN .* not supported"):
        tf.neighbor_search(cloud, cloud, k=5)


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_point_cloud_neighbor_search_point_cloud_with_transformation(dtype):
    """Test neighbor search with transformed point clouds"""
    points0 = np.array([[0, 0], [1, 0], [2, 0]], dtype=dtype)
    points1 = np.array([[0, 0], [10, 10]], dtype=dtype)

    cloud0 = tf.PointCloud(points0)
    cloud1 = tf.PointCloud(points1)

    # Transform cloud1 by offset (0.5, 0, 0)
    transform = np.eye(3, dtype=dtype)
    transform[0, 2] = 0.5
    cloud1.transformation = transform

    result = tf.neighbor_search(cloud0, cloud1)
    assert result is not None

    (idx0, idx1), (dist, pt0, pt1) = result

    assert idx0 == 0
    assert idx1 == 0

    expected_dist = 0.5**2
    assert abs(dist - expected_dist) < 1e-5

    assert np.allclose(pt0, [0, 0], atol=1e-5)
    assert np.allclose(pt1, [0.5, 0], atol=1e-5)


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_point_cloud_neighbor_search_point_cloud_separated_clouds(dtype):
    """Test with well-separated clouds"""
    points0 = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=dtype)
    points1 = np.array([[10, 0], [10, 1], [11, 0], [11, 1]], dtype=dtype)

    cloud0 = tf.PointCloud(points0)
    cloud1 = tf.PointCloud(points1)

    result = tf.neighbor_search(cloud0, cloud1)
    assert result is not None

    (idx0, idx1), (dist, pt0, pt1) = result

    assert dist == pytest.approx(81.0, abs=1e-4)
    assert pt0[0] == pytest.approx(1.0, abs=1e-5)
    assert pt1[0] == pytest.approx(10.0, abs=1e-5)


if __name__ == "__main__":
    # Run tests with verbose output
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
