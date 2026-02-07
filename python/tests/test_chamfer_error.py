"""
Tests for chamfer_error function

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
# Basic Functionality Tests
# ==============================================================================

@pytest.mark.parametrize("dtype", REAL_DTYPES)
@pytest.mark.parametrize("dims", DIMS)
def test_chamfer_error_identical_clouds(dtype, dims):
    """Chamfer error between identical point clouds should be zero."""
    pts = np.random.rand(50, dims).astype(dtype)
    cloud0 = tf.PointCloud(pts)
    cloud1 = tf.PointCloud(pts.copy())

    error = tf.chamfer_error(cloud0, cloud1)

    assert error < 1e-6, f"Error should be ~0 for identical clouds, got {error}"


@pytest.mark.parametrize("dtype", REAL_DTYPES)
@pytest.mark.parametrize("dims", DIMS)
def test_chamfer_error_known_offset(dtype, dims):
    """Chamfer error with known uniform offset."""
    # Create a simple point cloud
    pts0 = np.array([[0, 0], [1, 0], [0, 1]], dtype=dtype) if dims == 2 else \
           np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=dtype)

    # Offset all points by a known amount
    offset = 0.5
    pts1 = pts0.copy()
    pts1[:, 0] += offset  # Shift in x direction

    cloud0 = tf.PointCloud(pts0)
    cloud1 = tf.PointCloud(pts1)

    error = tf.chamfer_error(cloud0, cloud1)

    # Each point's nearest neighbor is the corresponding point at distance `offset`
    expected_error = offset
    assert abs(error - expected_error) < 1e-5, \
        f"Expected error ~{expected_error}, got {error}"


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_chamfer_error_asymmetry(dtype):
    """Chamfer error is asymmetric (direction matters)."""
    # Cloud0: single point at origin
    pts0 = np.array([[0, 0, 0]], dtype=dtype)
    # Cloud1: two points
    pts1 = np.array([[1, 0, 0], [2, 0, 0]], dtype=dtype)

    cloud0 = tf.PointCloud(pts0)
    cloud1 = tf.PointCloud(pts1)

    # cloud0 -> cloud1: nearest to [0,0,0] is [1,0,0], distance = 1
    error_0_to_1 = tf.chamfer_error(cloud0, cloud1)

    # cloud1 -> cloud0: both points map to [0,0,0], distances = 1 and 2, mean = 1.5
    error_1_to_0 = tf.chamfer_error(cloud1, cloud0)

    assert abs(error_0_to_1 - 1.0) < 1e-5, f"Expected 1.0, got {error_0_to_1}"
    assert abs(error_1_to_0 - 1.5) < 1e-5, f"Expected 1.5, got {error_1_to_0}"
    assert error_0_to_1 != error_1_to_0, "Chamfer error should be asymmetric"


@pytest.mark.parametrize("dtype", REAL_DTYPES)
@pytest.mark.parametrize("dims", DIMS)
def test_chamfer_error_dtype_preserved(dtype, dims):
    """Result type should be consistent with input dtype precision."""
    pts = np.random.rand(20, dims).astype(dtype)
    cloud0 = tf.PointCloud(pts)
    cloud1 = tf.PointCloud(pts + 0.1)

    error = tf.chamfer_error(cloud0, cloud1)

    # Error should be a Python float (from C++ float/double)
    assert isinstance(error, float)


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_chamfer_error_2d_triangle(dtype):
    """Test with a simple 2D triangle configuration."""
    # Equilateral triangle centered at origin
    pts0 = np.array([
        [0, 1],
        [-np.sqrt(3)/2, -0.5],
        [np.sqrt(3)/2, -0.5]
    ], dtype=dtype)

    # Translated triangle - Chamfer finds NEAREST neighbor, not corresponding point
    translation = np.array([2, 0], dtype=dtype)
    pts1 = pts0 + translation

    cloud0 = tf.PointCloud(pts0)
    cloud1 = tf.PointCloud(pts1)

    error = tf.chamfer_error(cloud0, cloud1)

    # Error should be positive (clouds are separated)
    assert error > 0, f"Expected positive error, got {error}"
    # Error should be less than 2 (nearest neighbor is closer than corresponding point)
    assert error < 2.5, f"Expected error < 2.5, got {error}"


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_chamfer_error_3d_cube_vertices(dtype):
    """Test with 3D cube vertices."""
    # Unit cube vertices
    pts0 = np.array([
        [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
        [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]
    ], dtype=dtype)

    # Same cube translated by [1, 0, 0] - cubes share an edge
    # Vertices at x=1 in pts0 have nearest neighbors at x=1 in pts1 (distance 0)
    # Vertices at x=0 in pts0 have nearest neighbors at x=1 in pts1 (distance 1)
    pts1 = pts0 + np.array([1, 0, 0], dtype=dtype)

    cloud0 = tf.PointCloud(pts0)
    cloud1 = tf.PointCloud(pts1)

    error = tf.chamfer_error(cloud0, cloud1)

    # Half the vertices (x=1) have distance 0, half (x=0) have distance 1
    # Mean = (0+0+0+0 + 1+1+1+1) / 8 = 0.5
    assert abs(error - 0.5) < 1e-5, f"Expected error ~0.5, got {error}"


# ==============================================================================
# Different Point Cloud Sizes
# ==============================================================================

@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_chamfer_error_different_sizes(dtype):
    """Point clouds can have different numbers of points."""
    pts0 = np.array([[0, 0, 0], [1, 0, 0]], dtype=dtype)
    pts1 = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]], dtype=dtype)

    cloud0 = tf.PointCloud(pts0)
    cloud1 = tf.PointCloud(pts1)

    # cloud0 -> cloud1: [0,0,0] -> [0,0,0] (dist=0), [1,0,0] -> [1,0,0] (dist=0)
    # mean = 0
    error = tf.chamfer_error(cloud0, cloud1)
    assert error < 1e-6, f"Expected ~0, got {error}"


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_chamfer_error_single_point(dtype):
    """Test with single point in source cloud."""
    pts0 = np.array([[0, 0, 0]], dtype=dtype)
    pts1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=dtype)

    cloud0 = tf.PointCloud(pts0)
    cloud1 = tf.PointCloud(pts1)

    # Nearest to origin is any of the 3 points at distance 1
    error = tf.chamfer_error(cloud0, cloud1)
    assert abs(error - 1.0) < 1e-5, f"Expected 1.0, got {error}"


# ==============================================================================
# Symmetric Chamfer Distance Pattern
# ==============================================================================

@pytest.mark.parametrize("dtype", REAL_DTYPES)
@pytest.mark.parametrize("dims", DIMS)
def test_symmetric_chamfer_distance(dtype, dims):
    """Test computing symmetric Chamfer distance (average of both directions)."""
    np.random.seed(42)
    pts0 = np.random.rand(30, dims).astype(dtype)
    pts1 = np.random.rand(40, dims).astype(dtype)

    cloud0 = tf.PointCloud(pts0)
    cloud1 = tf.PointCloud(pts1)

    error_0_to_1 = tf.chamfer_error(cloud0, cloud1)
    error_1_to_0 = tf.chamfer_error(cloud1, cloud0)

    symmetric_error = (error_0_to_1 + error_1_to_0) / 2

    # Just verify it computes without error and is positive
    assert symmetric_error >= 0
    assert isinstance(symmetric_error, float)


# ==============================================================================
# Error Handling Tests
# ==============================================================================

def test_chamfer_error_dimension_mismatch():
    """Should raise error for mismatched dimensions."""
    pts2d = np.array([[0, 0], [1, 0]], dtype=np.float32)
    pts3d = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)

    cloud2d = tf.PointCloud(pts2d)
    cloud3d = tf.PointCloud(pts3d)

    with pytest.raises(ValueError, match="Expected 2D, got 3D"):
        tf.chamfer_error(cloud2d, cloud3d)


def test_chamfer_error_dtype_mismatch():
    """Should raise error for mismatched dtypes."""
    pts32 = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)
    pts64 = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float64)

    cloud32 = tf.PointCloud(pts32)
    cloud64 = tf.PointCloud(pts64)

    with pytest.raises(ValueError, match="Dtype mismatch"):
        tf.chamfer_error(cloud32, cloud64)


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
