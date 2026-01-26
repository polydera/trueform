"""
Tests for PointCloud ray_cast functionality

Uses pytest parametrization to efficiently test all type combinations:
- Real types: float32, float64
- Dims: 2D, 3D

Copyright (c) 2025 Žiga Sajovic, XLAB
"""

import sys
import numpy as np
import pytest
import trueform as tf

# Type combinations to test
REAL_DTYPES = [np.float32, np.float64]


def create_2d_point_grid(real_dtype):
    """
    Create a simple 2D grid of points.
    Points at: [0,0], [1,0], [2,0], [0,1], [1,1], [2,1]
    """
    points = np.array([
        [0, 0], [1, 0], [2, 0],
        [0, 1], [1, 1], [2, 1]
    ], dtype=real_dtype)
    return tf.PointCloud(points)


def create_3d_point_grid(real_dtype):
    """
    Create a simple 3D grid of points in z=0 plane.
    Points at: [0,0,0], [1,0,0], [2,0,0], [0,1,0], [1,1,0], [2,1,0]
    """
    points = np.array([
        [0, 0, 0], [1, 0, 0], [2, 0, 0],
        [0, 1, 0], [1, 1, 0], [2, 1, 0]
    ], dtype=real_dtype)
    return tf.PointCloud(points)


# ==============================================================================
# Basic Hit/Miss Tests
# ==============================================================================

@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_point_cloud_ray_cast_hit_2d(real_dtype):
    """Test ray casting that should hit a point in 2D"""
    cloud = create_2d_point_grid(real_dtype)

    # Ray passing through point [1, 0]
    ray = tf.Ray(
        origin=np.array([1.0, -1.0], dtype=real_dtype),
        direction=np.array([0.0, 1.0], dtype=real_dtype)
    )

    result = tf.ray_cast(ray, cloud)

    # Should hit point at [1, 0]
    assert result is not None, "Ray should intersect point cloud"
    assert isinstance(result, tuple), "Result should be a tuple"
    assert len(result) == 2, "Result should be (point_idx, t)"

    point_idx, t = result
    assert isinstance(point_idx, (int, np.integer)), "Point index should be an integer"
    assert point_idx == 1, f"Should hit point 1 at [1,0], got {point_idx}"
    assert isinstance(t, (float, np.floating)), "t should be a float"
    assert t > 0, "t should be positive for ray traveling forward"
    assert np.isclose(t, 1.0, atol=1e-5), f"Expected t=1.0, got {t}"


@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_point_cloud_ray_cast_hit_3d(real_dtype):
    """Test ray casting that should hit a point in 3D"""
    cloud = create_3d_point_grid(real_dtype)

    # Ray passing through point [1, 1, 0]
    ray = tf.Ray(
        origin=np.array([1.0, 1.0, 2.0], dtype=real_dtype),
        direction=np.array([0.0, 0.0, -1.0], dtype=real_dtype)
    )

    result = tf.ray_cast(ray, cloud)

    # Should hit point at [1, 1, 0]
    assert result is not None, "Ray should intersect point cloud"
    assert isinstance(result, tuple), "Result should be a tuple"

    point_idx, t = result
    assert point_idx == 4, f"Should hit point 4 at [1,1,0], got {point_idx}"
    assert t > 0, "t should be positive"
    assert np.isclose(t, 2.0, atol=1e-5), f"Expected t=2.0, got {t}"


@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_point_cloud_ray_cast_miss_2d(real_dtype):
    """Test ray casting that should miss all points in 2D"""
    cloud = create_2d_point_grid(real_dtype)

    # Ray that doesn't pass through any grid point
    ray = tf.Ray(
        origin=np.array([0.5, -1.0], dtype=real_dtype),
        direction=np.array([0.0, 1.0], dtype=real_dtype)
    )

    result = tf.ray_cast(ray, cloud)

    # Should miss all points
    assert result is None, "Ray should not intersect any point"


@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_point_cloud_ray_cast_miss_3d(real_dtype):
    """Test ray casting that should miss all points in 3D"""
    cloud = create_3d_point_grid(real_dtype)

    # Ray that doesn't pass through any grid point
    ray = tf.Ray(
        origin=np.array([0.5, 0.5, 2.0], dtype=real_dtype),
        direction=np.array([0.0, 0.0, -1.0], dtype=real_dtype)
    )

    result = tf.ray_cast(ray, cloud)

    # Should miss all points (none are at [0.5, 0.5, z])
    assert result is None, "Ray should not intersect any point"


@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_point_cloud_ray_cast_pointing_away_2d(real_dtype):
    """Test ray pointing away from point cloud in 2D"""
    cloud = create_2d_point_grid(real_dtype)

    # Ray pointing away from point [1, 0]
    ray = tf.Ray(
        origin=np.array([1.0, -1.0], dtype=real_dtype),
        direction=np.array([0.0, -1.0], dtype=real_dtype)  # Pointing away
    )

    result = tf.ray_cast(ray, cloud)

    # Should miss (pointing away)
    assert result is None, "Ray pointing away should not hit"


@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_point_cloud_ray_cast_pointing_away_3d(real_dtype):
    """Test ray pointing away from point cloud in 3D"""
    cloud = create_3d_point_grid(real_dtype)

    # Ray pointing away from point [1, 1, 0]
    ray = tf.Ray(
        origin=np.array([1.0, 1.0, 2.0], dtype=real_dtype),
        direction=np.array([0.0, 0.0, 1.0], dtype=real_dtype)  # Pointing away
    )

    result = tf.ray_cast(ray, cloud)

    # Should miss (pointing away)
    assert result is None, "Ray pointing away should not hit"


# ==============================================================================
# Transformation Tests
# ==============================================================================

@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_point_cloud_ray_cast_with_transformation_2d(real_dtype):
    """Test ray casting with 2D transformation"""
    cloud = create_2d_point_grid(real_dtype)

    # Apply translation [5, 3]
    transformation = np.array([
        [1, 0, 5],
        [0, 1, 3],
        [0, 0, 1]
    ], dtype=real_dtype)

    cloud.transformation = transformation

    # Ray should hit transformed point [1,0] -> [6,3]
    ray = tf.Ray(
        origin=np.array([6.0, 2.0], dtype=real_dtype),
        direction=np.array([0.0, 1.0], dtype=real_dtype)
    )

    result = tf.ray_cast(ray, cloud)
    assert result is not None, "Ray should hit transformed point cloud"

    point_idx, t = result
    assert point_idx == 1, f"Should hit point 1, got {point_idx}"
    assert np.isclose(t, 1.0, atol=1e-5), f"Expected t=1.0, got {t}"


@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_point_cloud_ray_cast_with_transformation_3d(real_dtype):
    """Test ray casting with 3D transformation"""
    cloud = create_3d_point_grid(real_dtype)

    # Apply translation [10, 5, 2]
    transformation = np.array([
        [1, 0, 0, 10],
        [0, 1, 0,  5],
        [0, 0, 1,  2],
        [0, 0, 0,  1]
    ], dtype=real_dtype)

    cloud.transformation = transformation

    # Ray should hit transformed point [1,1,0] -> [11,6,2]
    ray = tf.Ray(
        origin=np.array([11.0, 6.0, 5.0], dtype=real_dtype),
        direction=np.array([0.0, 0.0, -1.0], dtype=real_dtype)
    )

    result = tf.ray_cast(ray, cloud)
    assert result is not None, "Ray should hit transformed point cloud"

    point_idx, t = result
    assert point_idx == 4, f"Should hit point 4, got {point_idx}"
    assert np.isclose(t, 3.0, atol=1e-5), f"Expected t=3.0, got {t}"


@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_point_cloud_ray_cast_transformation_clear(real_dtype):
    """Test clearing transformation"""
    cloud = create_3d_point_grid(real_dtype)

    # Apply transformation
    transformation = np.array([
        [1, 0, 0, 10],
        [0, 1, 0,  5],
        [0, 0, 1,  2],
        [0, 0, 0,  1]
    ], dtype=real_dtype)
    cloud.transformation = transformation

    # Ray at original position should miss
    ray_original = tf.Ray(
        origin=np.array([1.0, 1.0, 2.0], dtype=real_dtype),
        direction=np.array([0.0, 0.0, -1.0], dtype=real_dtype)
    )
    result = tf.ray_cast(ray_original, cloud)
    assert result is None, "Ray at original position should miss transformed cloud"

    # Clear transformation
    cloud.transformation = None

    # Now ray at original position should hit
    result = tf.ray_cast(ray_original, cloud)
    assert result is not None, "Ray should hit after clearing transformation"

    point_idx, t = result
    assert point_idx == 4
    assert np.isclose(t, 2.0, atol=1e-5)


# ==============================================================================
# Error Handling Tests
# ==============================================================================

def test_point_cloud_ray_cast_dimension_mismatch():
    """Test that dimension mismatch raises an error"""
    cloud = create_2d_point_grid(np.float32)  # 2D cloud

    # 3D ray
    ray_3d = tf.Ray(
        origin=np.array([1.0, 1.0, 2.0], dtype=np.float32),
        direction=np.array([0.0, 0.0, -1.0], dtype=np.float32)
    )

    with pytest.raises(ValueError, match="Dimension mismatch"):
        tf.ray_cast(ray_3d, cloud)


def test_point_cloud_ray_cast_dtype_mismatch():
    """Test that dtype mismatch raises an error"""
    cloud = create_3d_point_grid(np.float32)  # float32 cloud

    # float64 ray
    ray_f64 = tf.Ray(
        origin=np.array([1.0, 1.0, 2.0], dtype=np.float64),
        direction=np.array([0.0, 0.0, -1.0], dtype=np.float64)
    )

    with pytest.raises(TypeError, match="Dtype mismatch"):
        tf.ray_cast(ray_f64, cloud)


# ==============================================================================
# Geometric Correctness Tests
# ==============================================================================

@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_point_cloud_ray_cast_multiple_points_inline(real_dtype):
    """Test that ray returns the first (closest) point hit"""
    # Create points along a line: [0,0,0], [0,0,1], [0,0,2]
    points = np.array([
        [0, 0, 0],
        [0, 0, 1],
        [0, 0, 2]
    ], dtype=real_dtype)
    cloud = tf.PointCloud(points)

    # Ray from z=-1 pointing in +z direction
    ray = tf.Ray(
        origin=np.array([0.0, 0.0, -1.0], dtype=real_dtype),
        direction=np.array([0.0, 0.0, 1.0], dtype=real_dtype)
    )

    result = tf.ray_cast(ray, cloud)
    assert result is not None, "Ray should hit point cloud"

    point_idx, t = result
    # Should hit the first point at [0,0,0], which is at t=1
    assert point_idx == 0, f"Should hit first point (idx 0), got {point_idx}"
    assert np.isclose(t, 1.0, atol=1e-5), f"Expected t=1.0, got {t}"


@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_point_cloud_ray_cast_verify_hit_point(real_dtype):
    """Test that hit point calculation is correct"""
    cloud = create_3d_point_grid(real_dtype)

    # Ray passing through [2, 0, 0]
    ray = tf.Ray(
        origin=np.array([2.0, 0.0, 3.0], dtype=real_dtype),
        direction=np.array([0.0, 0.0, -1.0], dtype=real_dtype)
    )

    result = tf.ray_cast(ray, cloud)
    assert result is not None, "Ray should hit point cloud"

    point_idx, t = result

    # Calculate hit point
    hit_point = ray.origin + t * ray.direction

    # Get the actual point in the cloud
    actual_point = cloud.points[point_idx]

    # Hit point should equal the cloud point
    assert np.allclose(hit_point, actual_point, atol=1e-5), \
        f"Hit point {hit_point} should equal cloud point {actual_point}"


if __name__ == "__main__":
    # Run tests with verbose output
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
