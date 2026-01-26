"""
Tests for fit_knn_alignment function (k-NN based ICP iteration)

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
def test_fit_knn_alignment_2d_identity(dtype):
    """Identical point clouds with k=1 should give near-identity transformation."""
    np.random.seed(42)
    pts = np.random.rand(50, 2).astype(dtype)
    cloud0 = tf.PointCloud(pts)
    cloud1 = tf.PointCloud(pts.copy())

    T = tf.fit_knn_alignment(cloud0, cloud1, k=1)

    # Should be close to identity
    identity = np.eye(3, dtype=dtype)
    assert T.shape == (3, 3)
    assert np.allclose(T, identity, atol=1e-4)


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_fit_knn_alignment_2d_translation(dtype):
    """Test k=1 alignment with small translation (ICP iteration)."""
    np.random.seed(42)
    pts0 = np.random.rand(100, 2).astype(dtype)
    # Small translation for ICP to work well
    translation = np.array([0.1, 0.1], dtype=dtype)
    pts1 = pts0 + translation

    cloud0 = tf.PointCloud(pts0)
    cloud1 = tf.PointCloud(pts1)

    T = tf.fit_knn_alignment(cloud0, cloud1, k=1)

    # Apply transform
    pts0_transformed = apply_transform_2d(pts0, T)
    cloud_transformed = tf.PointCloud(pts0_transformed.astype(dtype))

    # Should reduce chamfer error significantly
    error_before = tf.chamfer_error(cloud0, cloud1)
    error_after = tf.chamfer_error(cloud_transformed, cloud1)
    assert error_after < error_before, "KNN alignment should reduce error"


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_fit_knn_alignment_2d_rotation(dtype):
    """Test k=1 alignment with small rotation."""
    np.random.seed(42)
    pts0 = np.random.rand(100, 2).astype(dtype) * 2 - 1  # Center around origin
    # Small rotation for ICP
    T_true = create_rotation_2d(5, dtype)  # 5 degrees
    pts1 = apply_transform_2d(pts0, T_true)

    cloud0 = tf.PointCloud(pts0)
    cloud1 = tf.PointCloud(pts1.astype(dtype))

    T = tf.fit_knn_alignment(cloud0, cloud1, k=1)

    # Apply transform
    pts0_transformed = apply_transform_2d(pts0, T)
    cloud_transformed = tf.PointCloud(pts0_transformed.astype(dtype))

    # Should reduce error
    error_before = tf.chamfer_error(cloud0, cloud1)
    error_after = tf.chamfer_error(cloud_transformed, cloud1)
    assert error_after < error_before, "KNN alignment should reduce error"


# ==============================================================================
# Basic Functionality Tests - 3D
# ==============================================================================

@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_fit_knn_alignment_3d_identity(dtype):
    """Identical 3D point clouds with k=1 should give near-identity transformation."""
    np.random.seed(42)
    pts = np.random.rand(100, 3).astype(dtype)
    cloud0 = tf.PointCloud(pts)
    cloud1 = tf.PointCloud(pts.copy())

    T = tf.fit_knn_alignment(cloud0, cloud1, k=1)

    # Should be close to identity
    identity = np.eye(4, dtype=dtype)
    assert T.shape == (4, 4)
    assert np.allclose(T, identity, atol=1e-4)


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_fit_knn_alignment_3d_translation(dtype):
    """Test k=1 3D alignment with small translation."""
    np.random.seed(42)
    pts0 = np.random.rand(200, 3).astype(dtype)
    translation = np.array([0.1, 0.05, 0.1], dtype=dtype)
    pts1 = pts0 + translation

    cloud0 = tf.PointCloud(pts0)
    cloud1 = tf.PointCloud(pts1)

    T = tf.fit_knn_alignment(cloud0, cloud1, k=1)

    # Apply transform
    pts0_transformed = apply_transform_3d(pts0, T)
    cloud_transformed = tf.PointCloud(pts0_transformed.astype(dtype))

    # Should reduce error
    error_before = tf.chamfer_error(cloud0, cloud1)
    error_after = tf.chamfer_error(cloud_transformed, cloud1)
    assert error_after < error_before, "KNN alignment should reduce error"


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_fit_knn_alignment_3d_rotation(dtype):
    """Test k=1 3D alignment with small rotation."""
    np.random.seed(42)
    pts0 = np.random.rand(200, 3).astype(dtype) * 2 - 1  # Center around origin
    T_true = create_rotation_z_3d(5, dtype)  # 5 degrees
    pts1 = apply_transform_3d(pts0, T_true)

    cloud0 = tf.PointCloud(pts0)
    cloud1 = tf.PointCloud(pts1.astype(dtype))

    T = tf.fit_knn_alignment(cloud0, cloud1, k=1)

    # Apply transform
    pts0_transformed = apply_transform_3d(pts0, T)
    cloud_transformed = tf.PointCloud(pts0_transformed.astype(dtype))

    # Should reduce error
    error_before = tf.chamfer_error(cloud0, cloud1)
    error_after = tf.chamfer_error(cloud_transformed, cloud1)
    assert error_after < error_before, "KNN alignment should reduce error"


# ==============================================================================
# K Parameter Tests
# ==============================================================================

@pytest.mark.parametrize("dtype", REAL_DTYPES)
@pytest.mark.parametrize("k", [1, 3, 5, 10])
def test_fit_knn_alignment_different_k_values(dtype, k):
    """Test with different k values."""
    np.random.seed(42)
    pts0 = np.random.rand(100, 3).astype(dtype)
    pts1 = pts0 + np.array([0.1, 0.1, 0.1], dtype=dtype)

    cloud0 = tf.PointCloud(pts0)
    cloud1 = tf.PointCloud(pts1)

    T = tf.fit_knn_alignment(cloud0, cloud1, k=k)

    assert T.shape == (4, 4)
    assert T.dtype == dtype


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_fit_knn_alignment_k1_vs_k5(dtype):
    """Compare k=1 (classic ICP) vs k=5 (soft correspondences)."""
    np.random.seed(42)
    pts0 = np.random.rand(200, 3).astype(dtype)
    # Add noise to make soft correspondences potentially better
    pts1 = pts0 + np.array([0.1, 0.1, 0.1], dtype=dtype)
    pts1 += np.random.randn(200, 3).astype(dtype) * 0.02  # Small noise

    cloud0 = tf.PointCloud(pts0)
    cloud1 = tf.PointCloud(pts1)

    T_k1 = tf.fit_knn_alignment(cloud0, cloud1, k=1)
    T_k5 = tf.fit_knn_alignment(cloud0, cloud1, k=5)

    # Both should produce valid transforms
    assert T_k1.shape == (4, 4)
    assert T_k5.shape == (4, 4)


# ==============================================================================
# Sigma Parameter Tests
# ==============================================================================

@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_fit_knn_alignment_with_sigma(dtype):
    """Test with explicit sigma parameter."""
    np.random.seed(42)
    pts0 = np.random.rand(100, 3).astype(dtype)
    pts1 = pts0 + np.array([0.1, 0.1, 0.1], dtype=dtype)

    cloud0 = tf.PointCloud(pts0)
    cloud1 = tf.PointCloud(pts1)

    # With explicit sigma
    T = tf.fit_knn_alignment(cloud0, cloud1, k=5, sigma=0.1)

    assert T.shape == (4, 4)
    assert T.dtype == dtype


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_fit_knn_alignment_sigma_none_vs_explicit(dtype):
    """Test adaptive sigma (None) vs explicit sigma."""
    np.random.seed(42)
    pts0 = np.random.rand(100, 3).astype(dtype)
    pts1 = pts0 + np.array([0.1, 0.1, 0.1], dtype=dtype)

    cloud0 = tf.PointCloud(pts0)
    cloud1 = tf.PointCloud(pts1)

    # Adaptive sigma (default)
    T_adaptive = tf.fit_knn_alignment(cloud0, cloud1, k=5, sigma=None)

    # Explicit sigma
    T_explicit = tf.fit_knn_alignment(cloud0, cloud1, k=5, sigma=0.5)

    # Both should produce valid transforms
    assert T_adaptive.shape == (4, 4)
    assert T_explicit.shape == (4, 4)


# ==============================================================================
# Output Properties Tests
# ==============================================================================

@pytest.mark.parametrize("dtype", REAL_DTYPES)
@pytest.mark.parametrize("dims", DIMS)
def test_fit_knn_alignment_output_shape(dtype, dims):
    """Output transformation matrix should have correct shape."""
    np.random.seed(42)
    pts = np.random.rand(50, dims).astype(dtype)
    cloud0 = tf.PointCloud(pts)
    cloud1 = tf.PointCloud(pts + 0.1)

    T = tf.fit_knn_alignment(cloud0, cloud1, k=1)

    expected_size = dims + 1
    assert T.shape == (expected_size, expected_size)


@pytest.mark.parametrize("dtype", REAL_DTYPES)
@pytest.mark.parametrize("dims", DIMS)
def test_fit_knn_alignment_output_dtype(dtype, dims):
    """Output transformation should match input dtype."""
    np.random.seed(42)
    pts = np.random.rand(50, dims).astype(dtype)
    cloud0 = tf.PointCloud(pts)
    cloud1 = tf.PointCloud(pts + 0.1)

    T = tf.fit_knn_alignment(cloud0, cloud1, k=1)

    assert T.dtype == dtype


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_fit_knn_alignment_3d_is_rigid(dtype):
    """Output should be a valid rigid transformation."""
    np.random.seed(42)
    pts0 = np.random.rand(100, 3).astype(dtype)
    pts1 = pts0 + np.array([0.1, 0.1, 0.1], dtype=dtype)

    cloud0 = tf.PointCloud(pts0)
    cloud1 = tf.PointCloud(pts1)

    T = tf.fit_knn_alignment(cloud0, cloud1, k=1)

    # Extract rotation part and check determinant
    R = T[:3, :3]
    det = np.linalg.det(R)
    assert abs(det - 1.0) < 1e-4, f"Rotation determinant should be 1, got {det}"

    # Check orthogonality: R @ R.T = I
    assert np.allclose(R @ R.T, np.eye(3, dtype=dtype), atol=1e-4)


# ==============================================================================
# ICP Convergence Tests
# ==============================================================================

@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_fit_knn_alignment_iterated_icp(dtype):
    """Test multiple ICP iterations converge."""
    np.random.seed(42)
    pts0 = np.random.rand(200, 3).astype(dtype) * 2 - 1
    T_true = create_rotation_translation_z_3d(15, 0.3, 0.2, 0.1, dtype)
    pts1 = apply_transform_3d(pts0, T_true)

    cloud1 = tf.PointCloud(pts1.astype(dtype))

    # Run several ICP iterations
    current_pts = pts0.copy()
    for _ in range(10):
        current_cloud = tf.PointCloud(current_pts.astype(dtype))
        T_iter = tf.fit_knn_alignment(current_cloud, cloud1, k=1)
        current_pts = apply_transform_3d(current_pts, T_iter)

    # After iterations, error should be very low
    final_cloud = tf.PointCloud(current_pts.astype(dtype))
    final_error = tf.chamfer_error(final_cloud, cloud1)
    assert final_error < 0.1, f"ICP should converge, got error {final_error}"


# ==============================================================================
# Different Point Cloud Sizes
# ==============================================================================

@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_fit_knn_alignment_different_sizes(dtype):
    """Point clouds can have different numbers of points."""
    np.random.seed(42)
    pts0 = np.random.rand(50, 3).astype(dtype)
    pts1 = np.random.rand(100, 3).astype(dtype) + 0.1

    cloud0 = tf.PointCloud(pts0)
    cloud1 = tf.PointCloud(pts1)

    # Should work without error
    T = tf.fit_knn_alignment(cloud0, cloud1, k=1)

    assert T.shape == (4, 4)


# ==============================================================================
# Error Handling Tests
# ==============================================================================

def test_fit_knn_alignment_dimension_mismatch():
    """Should raise error for mismatched dimensions."""
    pts2d = np.random.rand(50, 2).astype(np.float32)
    pts3d = np.random.rand(50, 3).astype(np.float32)

    cloud2d = tf.PointCloud(pts2d)
    cloud3d = tf.PointCloud(pts3d)

    with pytest.raises(ValueError, match="Dimension mismatch"):
        tf.fit_knn_alignment(cloud2d, cloud3d, k=1)


def test_fit_knn_alignment_dtype_mismatch():
    """Should raise error for mismatched dtypes."""
    pts32 = np.random.rand(50, 3).astype(np.float32)
    pts64 = np.random.rand(50, 3).astype(np.float64)

    cloud32 = tf.PointCloud(pts32)
    cloud64 = tf.PointCloud(pts64)

    with pytest.raises(ValueError, match="Dtype mismatch"):
        tf.fit_knn_alignment(cloud32, cloud64, k=1)


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
