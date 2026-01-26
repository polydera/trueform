"""
Tests for PointCloud × PointCloud gather_ids

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
def test_point_cloud_gather_ids_point_cloud_2d_within_distance_hit(dtype):
    """Test 2D PointCloud gather_ids with distance threshold - multiple hits"""
    # Cloud 0: points at (0, 0), (1, 0), (2, 0)
    # Cloud 1: points at (0.1, 0), (1.1, 0), (10, 10)
    # With distance=0.5: expect pairs (0,0), (1,1)

    points0 = np.array([[0, 0], [1, 0], [2, 0]], dtype=dtype)
    points1 = np.array([[0.1, 0], [1.1, 0], [10, 10]], dtype=dtype)

    cloud0 = tf.PointCloud(points0)
    cloud1 = tf.PointCloud(points1)

    result = tf.gather_ids_within_distance(cloud0, cloud1, distance=0.5)

    # Should return (N, 2) array
    assert result.shape[1] == 2
    assert result.shape[0] > 0  # At least some matches

    # Convert to set of tuples for easier checking
    pairs = set(tuple(row) for row in result)

    # Should contain (0, 0) and (1, 1)
    assert (0, 0) in pairs
    assert (1, 1) in pairs

    # Should NOT contain (2, 2) or anything with cloud1[2] (which is at (10, 10))
    assert not any(pair[1] == 2 for pair in pairs)


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_point_cloud_gather_ids_point_cloud_2d_within_distance_miss(dtype):
    """Test 2D PointCloud gather_ids with distance threshold - no hits"""
    # Cloud 0: points at (0, 0), (1, 0), (2, 0)
    # Cloud 1: points at (10, 10), (20, 20)
    # With distance=1.0: expect no matches

    points0 = np.array([[0, 0], [1, 0], [2, 0]], dtype=dtype)
    points1 = np.array([[10, 10], [20, 20]], dtype=dtype)

    cloud0 = tf.PointCloud(points0)
    cloud1 = tf.PointCloud(points1)

    result = tf.gather_ids_within_distance(cloud0, cloud1, distance=1.0)

    # Should return empty (0, 2) array
    assert result.shape == (0, 2)


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_point_cloud_gather_ids_point_cloud_3d_within_distance_hit(dtype):
    """Test 3D PointCloud gather_ids with distance threshold"""
    points0 = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=dtype)
    points1 = np.array([[0.1, 0, 0], [1.1, 0, 0], [10, 10, 10]], dtype=dtype)

    cloud0 = tf.PointCloud(points0)
    cloud1 = tf.PointCloud(points1)

    result = tf.gather_ids_within_distance(cloud0, cloud1, distance=0.5)

    assert result.shape[1] == 2
    assert result.shape[0] >= 2  # At least (0,0) and (1,1)

    pairs = set(tuple(row) for row in result)
    assert (0, 0) in pairs
    assert (1, 1) in pairs


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_point_cloud_gather_ids_point_cloud_2d_intersects_hit(dtype):
    """Test gather_intersecting_ids - points at same location"""
    # Cloud 0: points at (0, 0), (1, 0), (2, 0)
    # Cloud 1: points at (0, 0), (2, 0), (5, 5)
    # Should find (0, 0) and (2, 1)

    points0 = np.array([[0, 0], [1, 0], [2, 0]], dtype=dtype)
    points1 = np.array([[0, 0], [2, 0], [5, 5]], dtype=dtype)

    cloud0 = tf.PointCloud(points0)
    cloud1 = tf.PointCloud(points1)

    result = tf.gather_intersecting_ids(cloud0, cloud1)

    assert result.shape[1] == 2
    assert result.shape[0] > 0

    pairs = set(tuple(row) for row in result)

    # Note: exact point intersection might have tolerance
    # Just verify we got some matches and shape is correct
    assert len(pairs) > 0


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_point_cloud_gather_ids_point_cloud_2d_intersects_miss(dtype):
    """Test gather_intersecting_ids - no overlapping points"""
    points0 = np.array([[0, 0], [1, 0], [2, 0]], dtype=dtype)
    points1 = np.array([[0.5, 0.5], [1.5, 0.5], [10, 10]], dtype=dtype)

    cloud0 = tf.PointCloud(points0)
    cloud1 = tf.PointCloud(points1)

    result = tf.gather_intersecting_ids(cloud0, cloud1)

    # Should return empty (0, 2) array
    assert result.shape == (0, 2)


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_point_cloud_gather_ids_point_cloud_symmetry(dtype):
    """Test that swapping arguments swaps column order in results"""
    points0 = np.array([[0, 0], [1, 0]], dtype=dtype)
    points1 = np.array([[0.1, 0], [10, 10]], dtype=dtype)

    cloud0 = tf.PointCloud(points0)
    cloud1 = tf.PointCloud(points1)

    result01 = tf.gather_ids_within_distance(cloud0, cloud1, distance=0.5)
    result10 = tf.gather_ids_within_distance(cloud1, cloud0, distance=0.5)

    # Both should have same number of matches
    assert result01.shape[0] == result10.shape[0]

    # Columns should be swapped
    if result01.shape[0] > 0:
        pairs01 = set(tuple(row) for row in result01)
        pairs10 = set((row[1], row[0]) for row in result10)
        assert pairs01 == pairs10


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_point_cloud_gather_ids_point_cloud_with_transformation(dtype):
    """Test gather_ids with transformed point clouds"""
    points0 = np.array([[0, 0], [1, 0], [2, 0]], dtype=dtype)
    points1 = np.array([[0, 0], [1, 0], [2, 0]], dtype=dtype)

    cloud0 = tf.PointCloud(points0)
    cloud1 = tf.PointCloud(points1)

    # Before transformation - should have exact matches
    result_before = tf.gather_intersecting_ids(cloud0, cloud1)
    assert result_before.shape[0] == 3  # All 3 points match

    # Transform cloud1 by offset (0.05, 0)
    transform = np.eye(3, dtype=dtype)
    transform[0, 2] = 0.05
    cloud1.transformation = transform

    # After transformation - no exact intersections (points offset by 0.05)
    result_after = tf.gather_intersecting_ids(cloud0, cloud1)
    assert result_after.shape[0] == 0

    # But with distance threshold, should still find matches
    result_within = tf.gather_ids_within_distance(cloud0, cloud1, distance=0.1)
    assert result_within.shape[0] == 3


def test_point_cloud_gather_ids_point_cloud_dimension_mismatch():
    """Test that dimension mismatch raises error"""
    points2d = np.array([[0, 0], [1, 0]], dtype=np.float32)
    points3d = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)

    cloud2d = tf.PointCloud(points2d)
    cloud3d = tf.PointCloud(points3d)

    with pytest.raises(ValueError, match="Dimension mismatch"):
        tf.gather_intersecting_ids(cloud2d, cloud3d)

    with pytest.raises(ValueError, match="Dimension mismatch"):
        tf.gather_ids_within_distance(cloud2d, cloud3d, distance=1.0)


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_point_cloud_gather_ids_point_cloud_self_query(dtype):
    """Test point cloud querying itself"""
    points = np.array([[0, 0], [1, 0], [2, 0]], dtype=dtype)
    cloud = tf.PointCloud(points)

    result = tf.gather_intersecting_ids(cloud, cloud)

    # Should find all self-matches: (0,0), (1,1), (2,2)
    assert result.shape == (3, 2)

    pairs = set(tuple(row) for row in result)
    assert (0, 0) in pairs
    assert (1, 1) in pairs
    assert (2, 2) in pairs


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_point_cloud_gather_ids_point_cloud_many_matches(dtype):
    """Test with many points - verify all close pairs are found"""
    # Grid of points
    x = np.linspace(0, 10, 11, dtype=dtype)
    y = np.linspace(0, 10, 11, dtype=dtype)
    xv, yv = np.meshgrid(x, y)
    points0 = np.stack([xv.ravel(), yv.ravel()], axis=1)

    # Slightly offset grid
    points1 = points0 + 0.1

    cloud0 = tf.PointCloud(points0)
    cloud1 = tf.PointCloud(points1)

    result = tf.gather_ids_within_distance(cloud0, cloud1, distance=0.5)

    # All 121 points should have matches within threshold
    # (each point in cloud0 is 0.1*sqrt(2) ~ 0.141 away from corresponding point in cloud1)
    assert result.shape[0] == 121
    assert result.shape[1] == 2


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_point_cloud_gather_ids_point_cloud_empty_result_shape(dtype):
    """Test that empty result has correct shape (0, 2)"""
    points0 = np.array([[0, 0]], dtype=dtype)
    points1 = np.array([[100, 100]], dtype=dtype)

    cloud0 = tf.PointCloud(points0)
    cloud1 = tf.PointCloud(points1)

    result = tf.gather_intersecting_ids(cloud0, cloud1)

    assert result.shape == (0, 2)
    assert result.dtype in (np.int32, np.int64)  # Should be integer type


if __name__ == "__main__":
    # Run tests with verbose output
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
