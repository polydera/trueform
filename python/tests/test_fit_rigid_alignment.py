"""
Tests for fit_rigid_alignment function (Kabsch/Procrustes algorithm)

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


# ==============================================================================
# Basic Functionality Tests - 2D
# ==============================================================================

@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_fit_rigid_alignment_2d_identity(dtype):
    """Identical point clouds should give identity transformation."""
    pts = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=dtype)
    cloud0 = tf.PointCloud(pts)
    cloud1 = tf.PointCloud(pts.copy())

    T = tf.fit_rigid_alignment(cloud0, cloud1)

    # Should be close to identity
    identity = np.eye(3, dtype=dtype)
    assert T.shape == (3, 3)
    assert np.allclose(T, identity, atol=1e-5)


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_fit_rigid_alignment_2d_translation(dtype):
    """Test recovery of pure translation."""
    pts0 = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=dtype)
    translation = np.array([2, 3], dtype=dtype)
    pts1 = pts0 + translation

    cloud0 = tf.PointCloud(pts0)
    cloud1 = tf.PointCloud(pts1)

    T = tf.fit_rigid_alignment(cloud0, cloud1)

    # Apply recovered transform and check alignment
    pts0_transformed = apply_transform_2d(pts0, T)
    assert np.allclose(pts0_transformed, pts1, atol=1e-4)


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_fit_rigid_alignment_2d_rotation(dtype):
    """Test recovery of pure rotation."""
    pts0 = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=dtype)
    T_true = create_rotation_2d(45, dtype)
    pts1 = apply_transform_2d(pts0, T_true)

    cloud0 = tf.PointCloud(pts0)
    cloud1 = tf.PointCloud(pts1.astype(dtype))

    T = tf.fit_rigid_alignment(cloud0, cloud1)

    # Apply recovered transform and check alignment
    pts0_transformed = apply_transform_2d(pts0, T)
    assert np.allclose(pts0_transformed, pts1, atol=1e-4)


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_fit_rigid_alignment_2d_rotation_translation(dtype):
    """Test recovery of rotation + translation."""
    pts0 = np.array([[0, 0], [2, 0], [2, 1], [0, 1]], dtype=dtype)
    T_true = create_rotation_translation_2d(30, 5, -3, dtype)
    pts1 = apply_transform_2d(pts0, T_true)

    cloud0 = tf.PointCloud(pts0)
    cloud1 = tf.PointCloud(pts1.astype(dtype))

    T = tf.fit_rigid_alignment(cloud0, cloud1)

    # Apply recovered transform and check alignment
    pts0_transformed = apply_transform_2d(pts0, T)
    assert np.allclose(pts0_transformed, pts1, atol=1e-4)


# ==============================================================================
# Basic Functionality Tests - 3D
# ==============================================================================

@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_fit_rigid_alignment_3d_identity(dtype):
    """Identical 3D point clouds should give identity transformation."""
    pts = np.array([
        [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
        [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]
    ], dtype=dtype)
    cloud0 = tf.PointCloud(pts)
    cloud1 = tf.PointCloud(pts.copy())

    T = tf.fit_rigid_alignment(cloud0, cloud1)

    # Should be close to identity
    identity = np.eye(4, dtype=dtype)
    assert T.shape == (4, 4)
    assert np.allclose(T, identity, atol=1e-5)


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_fit_rigid_alignment_3d_translation(dtype):
    """Test recovery of pure 3D translation."""
    pts0 = np.array([
        [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]
    ], dtype=dtype)
    translation = np.array([1, 2, 3], dtype=dtype)
    pts1 = pts0 + translation

    cloud0 = tf.PointCloud(pts0)
    cloud1 = tf.PointCloud(pts1)

    T = tf.fit_rigid_alignment(cloud0, cloud1)

    # Apply recovered transform and check alignment
    pts0_transformed = apply_transform_3d(pts0, T)
    assert np.allclose(pts0_transformed, pts1, atol=1e-4)


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_fit_rigid_alignment_3d_rotation(dtype):
    """Test recovery of pure 3D rotation."""
    pts0 = np.array([
        [1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0],
        [0, 0, 1], [0, 0, -1]
    ], dtype=dtype)
    T_true = create_rotation_z_3d(60, dtype)
    pts1 = apply_transform_3d(pts0, T_true)

    cloud0 = tf.PointCloud(pts0)
    cloud1 = tf.PointCloud(pts1.astype(dtype))

    T = tf.fit_rigid_alignment(cloud0, cloud1)

    # Apply recovered transform and check alignment
    pts0_transformed = apply_transform_3d(pts0, T)
    assert np.allclose(pts0_transformed, pts1, atol=1e-4)


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_fit_rigid_alignment_3d_rotation_translation(dtype):
    """Test recovery of 3D rotation + translation."""
    pts0 = np.array([
        [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
        [1, 1, 1]
    ], dtype=dtype)
    T_true = create_rotation_translation_z_3d(45, 10, -5, 3, dtype)
    pts1 = apply_transform_3d(pts0, T_true)

    cloud0 = tf.PointCloud(pts0)
    cloud1 = tf.PointCloud(pts1.astype(dtype))

    T = tf.fit_rigid_alignment(cloud0, cloud1)

    # Apply recovered transform and check alignment
    pts0_transformed = apply_transform_3d(pts0, T)
    assert np.allclose(pts0_transformed, pts1, atol=1e-4)


# ==============================================================================
# Output Properties Tests
# ==============================================================================

@pytest.mark.parametrize("dtype", REAL_DTYPES)
@pytest.mark.parametrize("dims", DIMS)
def test_fit_rigid_alignment_output_shape(dtype, dims):
    """Output transformation matrix should have correct shape."""
    pts = np.random.rand(10, dims).astype(dtype)
    cloud0 = tf.PointCloud(pts)
    cloud1 = tf.PointCloud(pts + 0.1)

    T = tf.fit_rigid_alignment(cloud0, cloud1)

    expected_size = dims + 1
    assert T.shape == (expected_size, expected_size)


@pytest.mark.parametrize("dtype", REAL_DTYPES)
@pytest.mark.parametrize("dims", DIMS)
def test_fit_rigid_alignment_output_dtype(dtype, dims):
    """Output transformation should match input dtype."""
    pts = np.random.rand(10, dims).astype(dtype)
    cloud0 = tf.PointCloud(pts)
    cloud1 = tf.PointCloud(pts + 0.1)

    T = tf.fit_rigid_alignment(cloud0, cloud1)

    assert T.dtype == dtype


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_fit_rigid_alignment_3d_is_rigid(dtype):
    """Output should be a valid rigid transformation (det(R) = 1)."""
    pts0 = np.random.rand(20, 3).astype(dtype)
    T_true = create_rotation_translation_z_3d(30, 1, 2, 3, dtype)
    pts1 = apply_transform_3d(pts0, T_true)

    cloud0 = tf.PointCloud(pts0)
    cloud1 = tf.PointCloud(pts1.astype(dtype))

    T = tf.fit_rigid_alignment(cloud0, cloud1)

    # Extract rotation part and check determinant
    R = T[:3, :3]
    det = np.linalg.det(R)
    assert abs(det - 1.0) < 1e-4, f"Rotation determinant should be 1, got {det}"

    # Check orthogonality: R @ R.T = I
    assert np.allclose(R @ R.T, np.eye(3, dtype=dtype), atol=1e-4)


# ==============================================================================
# Edge Cases
# ==============================================================================

@pytest.mark.parametrize("dtype", REAL_DTYPES)
@pytest.mark.parametrize("dims", DIMS)
def test_fit_rigid_alignment_minimum_points(dtype, dims):
    """Test with minimum number of points for unique solution."""
    # Need at least dims+1 non-degenerate points for unique solution
    if dims == 2:
        pts0 = np.array([[0, 0], [1, 0], [0, 1]], dtype=dtype)
    else:
        pts0 = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=dtype)

    translation = np.ones(dims, dtype=dtype)
    pts1 = pts0 + translation

    cloud0 = tf.PointCloud(pts0)
    cloud1 = tf.PointCloud(pts1)

    T = tf.fit_rigid_alignment(cloud0, cloud1)

    # Apply and verify
    if dims == 2:
        pts0_transformed = apply_transform_2d(pts0, T)
    else:
        pts0_transformed = apply_transform_3d(pts0, T)

    assert np.allclose(pts0_transformed, pts1, atol=1e-4)


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_fit_rigid_alignment_large_point_cloud(dtype):
    """Test with a larger point cloud."""
    np.random.seed(42)
    pts0 = np.random.rand(500, 3).astype(dtype)
    T_true = create_rotation_translation_z_3d(25, 3, -2, 1, dtype)
    pts1 = apply_transform_3d(pts0, T_true)

    cloud0 = tf.PointCloud(pts0)
    cloud1 = tf.PointCloud(pts1.astype(dtype))

    T = tf.fit_rigid_alignment(cloud0, cloud1)

    # Apply and verify
    pts0_transformed = apply_transform_3d(pts0, T)
    assert np.allclose(pts0_transformed, pts1, atol=1e-3)


# ==============================================================================
# Error Handling Tests
# ==============================================================================

def test_fit_rigid_alignment_dimension_mismatch():
    """Should raise error for mismatched dimensions."""
    pts2d = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float32)
    pts3d = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)

    cloud2d = tf.PointCloud(pts2d)
    cloud3d = tf.PointCloud(pts3d)

    with pytest.raises(ValueError, match="Dimension mismatch"):
        tf.fit_rigid_alignment(cloud2d, cloud3d)


def test_fit_rigid_alignment_dtype_mismatch():
    """Should raise error for mismatched dtypes."""
    pts32 = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    pts64 = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)

    cloud32 = tf.PointCloud(pts32)
    cloud64 = tf.PointCloud(pts64)

    with pytest.raises(ValueError, match="Dtype mismatch"):
        tf.fit_rigid_alignment(cloud32, cloud64)


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
