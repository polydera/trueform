"""
Tests for fit_similarity_alignment function (Procrustes with scaling)

Copyright (c) 2025 Žiga Sajovic, XLAB
"""

import sys
import numpy as np
import pytest
import trueform as tf


# Test parameters
REAL_DTYPES = [np.float32, np.float64]


# ==============================================================================
# Helper Functions
# ==============================================================================

def create_similarity_2d(angle_degrees, scale, tx, ty, dtype):
    """Create a 2D similarity matrix (3x3 homogeneous): s*R plus translation."""
    angle = np.radians(angle_degrees)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    return np.array([
        [scale * cos_a, -scale * sin_a, tx],
        [scale * sin_a,  scale * cos_a, ty],
        [0,              0,             1]
    ], dtype=dtype)


def create_similarity_z_3d(angle_degrees, scale, tx, ty, tz, dtype):
    """Create a 3D similarity matrix (4x4 homogeneous): s*Rz plus translation."""
    angle = np.radians(angle_degrees)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    return np.array([
        [scale * cos_a, -scale * sin_a, 0,     tx],
        [scale * sin_a,  scale * cos_a, 0,     ty],
        [0,              0,             scale, tz],
        [0,              0,             0,     1]
    ], dtype=dtype)


def apply_transform(points, T):
    """Apply homogeneous transform to points."""
    n = len(points)
    homogeneous = np.hstack([points, np.ones((n, 1), dtype=points.dtype)])
    transformed = (T @ homogeneous.T).T
    return transformed[:, :points.shape[1]]


def scale_of(T, dims):
    """The uniform scale carried by the linear part of a similarity."""
    return np.linalg.det(T[:dims, :dims]) ** (1.0 / dims)


# ==============================================================================
# Basic Functionality Tests
# ==============================================================================

@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_fit_similarity_alignment_2d_identity(dtype):
    """Identical point clouds should give identity transformation."""
    pts = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=dtype)
    cloud0 = tf.PointCloud(pts)
    cloud1 = tf.PointCloud(pts.copy())

    T = tf.fit_similarity_alignment(cloud0, cloud1)

    assert T.shape == (3, 3)
    assert np.allclose(T, np.eye(3, dtype=dtype), atol=1e-5)
    assert np.isclose(scale_of(T, 2), 1.0, atol=1e-5)


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_fit_similarity_alignment_2d_known(dtype):
    """Recover a known rotation + translation + scale 1.7 in 2D."""
    rng = np.random.default_rng(7)
    pts0 = rng.random((40, 2)).astype(dtype)
    S = create_similarity_2d(30.0, 1.7, 2.0, -1.0, dtype)
    pts1 = apply_transform(pts0, S)

    T = tf.fit_similarity_alignment(tf.PointCloud(pts0), tf.PointCloud(pts1))

    assert T.shape == (3, 3)
    assert np.allclose(apply_transform(pts0, T), pts1, atol=1e-3)
    assert np.isclose(scale_of(T, 2), 1.7, atol=1e-3)


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_fit_similarity_alignment_3d_known(dtype):
    """Recover a known rotation + translation + scale 1.7 in 3D."""
    rng = np.random.default_rng(11)
    pts0 = rng.random((50, 3)).astype(dtype)
    S = create_similarity_z_3d(40.0, 1.7, 1.0, 2.0, -0.5, dtype)
    pts1 = apply_transform(pts0, S)

    T = tf.fit_similarity_alignment(tf.PointCloud(pts0), tf.PointCloud(pts1))

    assert T.shape == (4, 4)
    assert np.allclose(apply_transform(pts0, T), pts1, atol=1e-3)
    assert np.isclose(scale_of(T, 3), 1.7, atol=1e-3)


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_fit_similarity_alignment_3d_shrink(dtype):
    """A scale below one is recovered too."""
    rng = np.random.default_rng(3)
    pts0 = rng.random((30, 3)).astype(dtype)
    S = create_similarity_z_3d(-25.0, 0.4, 0.0, 0.0, 1.0, dtype)
    pts1 = apply_transform(pts0, S)

    T = tf.fit_similarity_alignment(tf.PointCloud(pts0), tf.PointCloud(pts1))

    assert np.allclose(apply_transform(pts0, T), pts1, atol=1e-3)
    assert np.isclose(scale_of(T, 3), 0.4, atol=1e-3)


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_fit_similarity_alignment_transformed_clouds(dtype):
    """Frame-tagged clouds align in world space: the delta composes the
    frames, and a source frame that scales does not scale the answer."""
    rng = np.random.default_rng(7)
    pts = rng.uniform(-1, 1, (40, 3)).astype(dtype)
    T0 = create_similarity_z_3d(30.0, 2.0, 0.5, -0.25, 1.0, dtype)
    T1 = create_similarity_z_3d(-45.0, 0.5, -1.0, 0.75, 0.25, dtype)

    source = tf.PointCloud(pts)
    source.transformation = T0
    target = tf.PointCloud(pts)
    target.transformation = T1

    delta = tf.fit_similarity_alignment(source, target)

    assert np.allclose(delta @ T0, T1, atol=1e-3)
    assert np.isclose(scale_of(delta, 3), 0.25, atol=1e-3)


# ==============================================================================
# Error Handling Tests
# ==============================================================================

def test_fit_similarity_alignment_dims_mismatch():
    pts2 = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float32)
    pts3 = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    with pytest.raises(ValueError):
        tf.fit_similarity_alignment(tf.PointCloud(pts2), tf.PointCloud(pts3))


def test_fit_similarity_alignment_dtype_mismatch():
    pts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    with pytest.raises(ValueError):
        tf.fit_similarity_alignment(
            tf.PointCloud(pts), tf.PointCloud(pts.astype(np.float64)))


def test_fit_similarity_alignment_point_count_mismatch():
    pts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
                   dtype=np.float32)
    with pytest.raises(ValueError, match="Point count mismatch"):
        tf.fit_similarity_alignment(tf.PointCloud(pts), tf.PointCloud(pts[:3]))


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
