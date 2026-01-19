"""
Tests for fit_obb_alignment function (OBB-based alignment)

Copyright (c) 2025 Žiga Sajovic, XLAB
"""

import sys
import numpy as np
import pytest
import trueform as tf


# Test parameters
REAL_DTYPES = [np.float32, np.float64]
DIMS = [2, 3]


# ==============================================================================
# Helper Functions
# ==============================================================================

def create_rotation_2d(angle_degrees, dtype):
    """Create a 2D rotation matrix (3x3 homogeneous)."""
    angle = np.radians(angle_degrees)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    return np.array([
        [cos_a, -sin_a, 0],
        [sin_a,  cos_a, 0],
        [0,      0,     1]
    ], dtype=dtype)


def create_rotation_translation_2d(angle_degrees, tx, ty, dtype):
    """Create a 2D rotation + translation matrix (3x3 homogeneous)."""
    angle = np.radians(angle_degrees)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    return np.array([
        [cos_a, -sin_a, tx],
        [sin_a,  cos_a, ty],
        [0,      0,     1]
    ], dtype=dtype)


def create_rotation_z_3d(angle_degrees, dtype):
    """Create a 3D rotation matrix around Z-axis (4x4 homogeneous)."""
    angle = np.radians(angle_degrees)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    return np.array([
        [cos_a, -sin_a, 0, 0],
        [sin_a,  cos_a, 0, 0],
        [0,      0,     1, 0],
        [0,      0,     0, 1]
    ], dtype=dtype)


def create_rotation_translation_z_3d(angle_degrees, tx, ty, tz, dtype):
    """Create a 3D rotation around Z + translation matrix (4x4 homogeneous)."""
    angle = np.radians(angle_degrees)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    return np.array([
        [cos_a, -sin_a, 0, tx],
        [sin_a,  cos_a, 0, ty],
        [0,      0,     1, tz],
        [0,      0,     0, 1]
    ], dtype=dtype)


def apply_transform_2d(points, T):
    """Apply 3x3 homogeneous transform to 2D points."""
    n = len(points)
    homogeneous = np.hstack([points, np.ones((n, 1), dtype=points.dtype)])
    transformed = (T @ homogeneous.T).T
    return transformed[:, :2]


def apply_transform_3d(points, T):
    """Apply 4x4 homogeneous transform to 3D points."""
    n = len(points)
    homogeneous = np.hstack([points, np.ones((n, 1), dtype=points.dtype)])
    transformed = (T @ homogeneous.T).T
    return transformed[:, :3]


def create_elongated_cloud_2d(n_points, length, width, dtype):
    """Create an elongated 2D point cloud (rectangle)."""
    np.random.seed(123)
    x = np.random.uniform(-length/2, length/2, n_points)
    y = np.random.uniform(-width/2, width/2, n_points)
    return np.column_stack([x, y]).astype(dtype)


def create_elongated_cloud_3d(n_points, length, width, height, dtype):
    """Create an elongated 3D point cloud (box)."""
    np.random.seed(123)
    x = np.random.uniform(-length/2, length/2, n_points)
    y = np.random.uniform(-width/2, width/2, n_points)
    z = np.random.uniform(-height/2, height/2, n_points)
    return np.column_stack([x, y, z]).astype(dtype)


# ==============================================================================
# Basic Functionality Tests - 2D
# ==============================================================================

@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_fit_obb_alignment_2d_identity(dtype):
    """Identical point clouds should give near-identity transformation."""
    pts = create_elongated_cloud_2d(100, 4.0, 1.0, dtype)
    cloud0 = tf.PointCloud(pts)
    cloud1 = tf.PointCloud(pts.copy())

    T = tf.fit_obb_alignment(cloud0, cloud1)

    assert T.shape == (3, 3)
    # Apply transform - result should be close to original
    pts_transformed = apply_transform_2d(pts, T)
    # Check that the transformed cloud overlaps well (low chamfer error)
    cloud_transformed = tf.PointCloud(pts_transformed.astype(dtype))
    error = tf.chamfer_error(cloud_transformed, cloud1)
    assert error < 0.5, f"Alignment error too high: {error}"


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_fit_obb_alignment_2d_translation(dtype):
    """Test alignment with pure translation."""
    pts0 = create_elongated_cloud_2d(100, 4.0, 1.0, dtype)
    translation = np.array([5, 3], dtype=dtype)
    pts1 = pts0 + translation

    cloud0 = tf.PointCloud(pts0)
    cloud1 = tf.PointCloud(pts1)

    T = tf.fit_obb_alignment(cloud0, cloud1)

    # Apply transform and check overlap
    pts0_transformed = apply_transform_2d(pts0, T)
    cloud_transformed = tf.PointCloud(pts0_transformed.astype(dtype))
    error = tf.chamfer_error(cloud_transformed, cloud1)
    assert error < 0.5, f"Alignment error too high: {error}"


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_fit_obb_alignment_2d_rotation(dtype):
    """Test alignment with rotation (OBB can recover up to 90-degree symmetry)."""
    pts0 = create_elongated_cloud_2d(100, 4.0, 1.0, dtype)
    # Use 45 degrees - within OBB's ability to distinguish
    T_true = create_rotation_2d(45, dtype)
    pts1 = apply_transform_2d(pts0, T_true)

    cloud0 = tf.PointCloud(pts0)
    cloud1 = tf.PointCloud(pts1.astype(dtype))

    T = tf.fit_obb_alignment(cloud0, cloud1)

    # Apply transform and check overlap
    pts0_transformed = apply_transform_2d(pts0, T)
    cloud_transformed = tf.PointCloud(pts0_transformed.astype(dtype))
    error = tf.chamfer_error(cloud_transformed, cloud1)
    assert error < 0.5, f"Alignment error too high: {error}"


# ==============================================================================
# Basic Functionality Tests - 3D
# ==============================================================================

@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_fit_obb_alignment_3d_identity(dtype):
    """Identical 3D point clouds should give near-identity transformation."""
    pts = create_elongated_cloud_3d(200, 4.0, 2.0, 1.0, dtype)
    cloud0 = tf.PointCloud(pts)
    cloud1 = tf.PointCloud(pts.copy())

    T = tf.fit_obb_alignment(cloud0, cloud1)

    assert T.shape == (4, 4)
    # Apply transform and check overlap
    pts_transformed = apply_transform_3d(pts, T)
    cloud_transformed = tf.PointCloud(pts_transformed.astype(dtype))
    error = tf.chamfer_error(cloud_transformed, cloud1)
    assert error < 0.5, f"Alignment error too high: {error}"


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_fit_obb_alignment_3d_translation(dtype):
    """Test 3D alignment with pure translation."""
    pts0 = create_elongated_cloud_3d(200, 4.0, 2.0, 1.0, dtype)
    translation = np.array([3, 5, -2], dtype=dtype)
    pts1 = pts0 + translation

    cloud0 = tf.PointCloud(pts0)
    cloud1 = tf.PointCloud(pts1)

    T = tf.fit_obb_alignment(cloud0, cloud1)

    # Apply transform and check overlap
    pts0_transformed = apply_transform_3d(pts0, T)
    cloud_transformed = tf.PointCloud(pts0_transformed.astype(dtype))
    error = tf.chamfer_error(cloud_transformed, cloud1)
    assert error < 0.5, f"Alignment error too high: {error}"


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_fit_obb_alignment_3d_rotation(dtype):
    """Test 3D alignment with rotation."""
    pts0 = create_elongated_cloud_3d(200, 4.0, 2.0, 1.0, dtype)
    T_true = create_rotation_z_3d(30, dtype)
    pts1 = apply_transform_3d(pts0, T_true)

    cloud0 = tf.PointCloud(pts0)
    cloud1 = tf.PointCloud(pts1.astype(dtype))

    T = tf.fit_obb_alignment(cloud0, cloud1)

    # Apply transform and check overlap
    pts0_transformed = apply_transform_3d(pts0, T)
    cloud_transformed = tf.PointCloud(pts0_transformed.astype(dtype))
    error = tf.chamfer_error(cloud_transformed, cloud1)
    assert error < 0.5, f"Alignment error too high: {error}"


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_fit_obb_alignment_3d_rotation_translation(dtype):
    """Test 3D alignment with rotation + translation."""
    pts0 = create_elongated_cloud_3d(200, 4.0, 2.0, 1.0, dtype)
    T_true = create_rotation_translation_z_3d(45, 5, -3, 2, dtype)
    pts1 = apply_transform_3d(pts0, T_true)

    cloud0 = tf.PointCloud(pts0)
    cloud1 = tf.PointCloud(pts1.astype(dtype))

    T = tf.fit_obb_alignment(cloud0, cloud1)

    # Apply transform and check overlap
    pts0_transformed = apply_transform_3d(pts0, T)
    cloud_transformed = tf.PointCloud(pts0_transformed.astype(dtype))
    error = tf.chamfer_error(cloud_transformed, cloud1)
    assert error < 0.5, f"Alignment error too high: {error}"


# ==============================================================================
# Output Properties Tests
# ==============================================================================

@pytest.mark.parametrize("dtype", REAL_DTYPES)
@pytest.mark.parametrize("dims", DIMS)
def test_fit_obb_alignment_output_shape(dtype, dims):
    """Output transformation matrix should have correct shape."""
    if dims == 2:
        pts = create_elongated_cloud_2d(50, 3.0, 1.0, dtype)
    else:
        pts = create_elongated_cloud_3d(100, 3.0, 2.0, 1.0, dtype)

    cloud0 = tf.PointCloud(pts)
    cloud1 = tf.PointCloud(pts + 1.0)

    T = tf.fit_obb_alignment(cloud0, cloud1)

    expected_size = dims + 1
    assert T.shape == (expected_size, expected_size)


@pytest.mark.parametrize("dtype", REAL_DTYPES)
@pytest.mark.parametrize("dims", DIMS)
def test_fit_obb_alignment_output_dtype(dtype, dims):
    """Output transformation should match input dtype."""
    if dims == 2:
        pts = create_elongated_cloud_2d(50, 3.0, 1.0, dtype)
    else:
        pts = create_elongated_cloud_3d(100, 3.0, 2.0, 1.0, dtype)

    cloud0 = tf.PointCloud(pts)
    cloud1 = tf.PointCloud(pts + 1.0)

    T = tf.fit_obb_alignment(cloud0, cloud1)

    assert T.dtype == dtype


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_fit_obb_alignment_3d_is_rigid(dtype):
    """Output should be a valid rigid transformation (det(R) = 1)."""
    pts0 = create_elongated_cloud_3d(200, 4.0, 2.0, 1.0, dtype)
    pts1 = pts0 + np.array([2, 3, 1], dtype=dtype)

    cloud0 = tf.PointCloud(pts0)
    cloud1 = tf.PointCloud(pts1)

    T = tf.fit_obb_alignment(cloud0, cloud1)

    # Extract rotation part and check determinant
    R = T[:3, :3]
    det = np.linalg.det(R)
    assert abs(abs(det) - 1.0) < 1e-4, f"Rotation determinant should be +/-1, got {det}"

    # Check orthogonality: R @ R.T = I
    assert np.allclose(R @ R.T, np.eye(3, dtype=dtype), atol=1e-4)


# ==============================================================================
# Sample Size Parameter Tests
# ==============================================================================

@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_fit_obb_alignment_sample_size_parameter(dtype):
    """Test that sample_size parameter is accepted."""
    pts0 = create_elongated_cloud_3d(200, 4.0, 2.0, 1.0, dtype)
    pts1 = pts0 + np.array([1, 2, 3], dtype=dtype)

    cloud0 = tf.PointCloud(pts0)
    cloud1 = tf.PointCloud(pts1)

    # Test with different sample sizes
    T_small = tf.fit_obb_alignment(cloud0, cloud1, sample_size=10)
    T_large = tf.fit_obb_alignment(cloud0, cloud1, sample_size=200)

    # Both should produce valid transformation matrices
    assert T_small.shape == (4, 4)
    assert T_large.shape == (4, 4)


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_fit_obb_alignment_default_sample_size(dtype):
    """Test with default sample_size (100)."""
    pts0 = create_elongated_cloud_3d(200, 4.0, 2.0, 1.0, dtype)
    pts1 = pts0 + np.array([1, 2, 3], dtype=dtype)

    cloud0 = tf.PointCloud(pts0)
    cloud1 = tf.PointCloud(pts1)

    # Default sample_size
    T = tf.fit_obb_alignment(cloud0, cloud1)

    assert T.shape == (4, 4)


# ==============================================================================
# Different Point Cloud Sizes
# ==============================================================================

@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_fit_obb_alignment_different_sizes(dtype):
    """Point clouds can have different numbers of points."""
    pts0 = create_elongated_cloud_3d(100, 4.0, 2.0, 1.0, dtype)
    # Create second cloud with different number of points but same shape
    np.random.seed(456)  # Different seed
    pts1 = create_elongated_cloud_3d(150, 4.0, 2.0, 1.0, dtype)
    pts1 = pts1 + np.array([2, 3, 1], dtype=dtype)

    cloud0 = tf.PointCloud(pts0)
    cloud1 = tf.PointCloud(pts1)

    # Should work without error
    T = tf.fit_obb_alignment(cloud0, cloud1)

    assert T.shape == (4, 4)


# ==============================================================================
# Error Handling Tests
# ==============================================================================

def test_fit_obb_alignment_dimension_mismatch():
    """Should raise error for mismatched dimensions."""
    pts2d = create_elongated_cloud_2d(50, 3.0, 1.0, np.float32)
    pts3d = create_elongated_cloud_3d(50, 3.0, 2.0, 1.0, np.float32)

    cloud2d = tf.PointCloud(pts2d)
    cloud3d = tf.PointCloud(pts3d)

    with pytest.raises(ValueError, match="Dimension mismatch"):
        tf.fit_obb_alignment(cloud2d, cloud3d)


def test_fit_obb_alignment_dtype_mismatch():
    """Should raise error for mismatched dtypes."""
    pts32 = create_elongated_cloud_3d(50, 3.0, 2.0, 1.0, np.float32)
    pts64 = create_elongated_cloud_3d(50, 3.0, 2.0, 1.0, np.float64)

    cloud32 = tf.PointCloud(pts32)
    cloud64 = tf.PointCloud(pts64)

    with pytest.raises(ValueError, match="Dtype mismatch"):
        tf.fit_obb_alignment(cloud32, cloud64)


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
