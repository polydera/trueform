"""
Tests for fit_icp_alignment function (Iterative Closest Point)

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


def apply_transform_3d(points, T):
    """Apply 4x4 homogeneous transform to 3D points."""
    n = len(points)
    homogeneous = np.hstack([points, np.ones((n, 1), dtype=points.dtype)])
    transformed = (T @ homogeneous.T).T
    return transformed[:, :3]


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


def apply_transform_2d(points, T):
    """Apply 3x3 homogeneous transform to 2D points."""
    n = len(points)
    homogeneous = np.hstack([points, np.ones((n, 1), dtype=points.dtype)])
    transformed = (T @ homogeneous.T).T
    return transformed[:, :2]


# ==============================================================================
# Basic Functionality Tests - 3D
# ==============================================================================

@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_fit_icp_alignment_3d_identity(dtype):
    """Identical point clouds should give near-identity transformation."""
    np.random.seed(42)
    pts = np.random.rand(100, 3).astype(dtype)
    cloud0 = tf.PointCloud(pts)
    cloud1 = tf.PointCloud(pts.copy())

    T = tf.fit_icp_alignment(cloud0, cloud1, max_iterations=10)

    # Should be close to identity
    identity = np.eye(4, dtype=dtype)
    assert T.shape == (4, 4)
    assert np.allclose(T, identity, atol=1e-4)


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_fit_icp_alignment_3d_translation(dtype):
    """Test ICP convergence with small translation."""
    np.random.seed(42)
    pts0 = np.random.rand(200, 3).astype(dtype)
    translation = np.array([0.1, 0.05, 0.1], dtype=dtype)
    pts1 = pts0 + translation

    cloud0 = tf.PointCloud(pts0)
    cloud1 = tf.PointCloud(pts1)

    T = tf.fit_icp_alignment(cloud0, cloud1, max_iterations=50)

    # Apply transform
    pts0_transformed = apply_transform_3d(pts0, T)
    cloud_transformed = tf.PointCloud(pts0_transformed.astype(dtype))

    # Should achieve very low error
    error = tf.chamfer_error(cloud_transformed, cloud1)
    assert error < 0.01, f"ICP should converge, got error {error}"


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_fit_icp_alignment_3d_rotation(dtype):
    """Test ICP convergence with small rotation."""
    np.random.seed(42)
    pts0 = np.random.rand(200, 3).astype(dtype) * 2 - 1  # Center around origin
    T_true = create_rotation_z_3d(10, dtype)  # 10 degrees
    pts1 = apply_transform_3d(pts0, T_true)

    cloud0 = tf.PointCloud(pts0)
    cloud1 = tf.PointCloud(pts1.astype(dtype))

    T = tf.fit_icp_alignment(cloud0, cloud1, max_iterations=50)

    # Apply transform
    pts0_transformed = apply_transform_3d(pts0, T)
    cloud_transformed = tf.PointCloud(pts0_transformed.astype(dtype))

    # Should achieve low error
    error = tf.chamfer_error(cloud_transformed, cloud1)
    assert error < 0.01, f"ICP should converge, got error {error}"


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_fit_icp_alignment_3d_rotation_translation(dtype):
    """Test ICP convergence with rotation + translation."""
    np.random.seed(42)
    pts0 = np.random.rand(200, 3).astype(dtype) * 2 - 1
    T_true = create_rotation_translation_z_3d(15, 0.2, 0.15, 0.1, dtype)
    pts1 = apply_transform_3d(pts0, T_true)

    cloud0 = tf.PointCloud(pts0)
    cloud1 = tf.PointCloud(pts1.astype(dtype))

    T = tf.fit_icp_alignment(cloud0, cloud1, max_iterations=50)

    # Apply transform
    pts0_transformed = apply_transform_3d(pts0, T)
    cloud_transformed = tf.PointCloud(pts0_transformed.astype(dtype))

    # Should achieve low error
    error = tf.chamfer_error(cloud_transformed, cloud1)
    assert error < 0.05, f"ICP should converge, got error {error}"


# ==============================================================================
# Basic Functionality Tests - 2D
# ==============================================================================

@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_fit_icp_alignment_2d_identity(dtype):
    """Identical 2D point clouds should give near-identity transformation."""
    np.random.seed(42)
    pts = np.random.rand(50, 2).astype(dtype)
    cloud0 = tf.PointCloud(pts)
    cloud1 = tf.PointCloud(pts.copy())

    T = tf.fit_icp_alignment(cloud0, cloud1, max_iterations=10)

    # Should be close to identity
    identity = np.eye(3, dtype=dtype)
    assert T.shape == (3, 3)
    assert np.allclose(T, identity, atol=1e-4)


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_fit_icp_alignment_2d_rotation(dtype):
    """Test 2D ICP convergence with small rotation."""
    np.random.seed(42)
    pts0 = np.random.rand(100, 2).astype(dtype) * 2 - 1  # Center around origin
    T_true = create_rotation_2d(10, dtype)  # 10 degrees
    pts1 = apply_transform_2d(pts0, T_true)

    cloud0 = tf.PointCloud(pts0)
    cloud1 = tf.PointCloud(pts1.astype(dtype))

    T = tf.fit_icp_alignment(cloud0, cloud1, max_iterations=50)

    # Apply transform
    pts0_transformed = apply_transform_2d(pts0, T)
    cloud_transformed = tf.PointCloud(pts0_transformed.astype(dtype))

    # Should achieve low error
    error = tf.chamfer_error(cloud_transformed, cloud1)
    assert error < 0.01, f"ICP should converge, got error {error}"


# ==============================================================================
# Parameter Tests
# ==============================================================================

@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_fit_icp_alignment_with_subsampling(dtype):
    """Test ICP with subsampling (n_samples)."""
    np.random.seed(42)
    pts0 = np.random.rand(500, 3).astype(dtype)
    pts1 = pts0 + np.array([0.1, 0.1, 0.1], dtype=dtype)

    cloud0 = tf.PointCloud(pts0)
    cloud1 = tf.PointCloud(pts1)

    # With subsampling
    T = tf.fit_icp_alignment(cloud0, cloud1, max_iterations=30, n_samples=100)

    assert T.shape == (4, 4)
    assert T.dtype == dtype


@pytest.mark.parametrize("dtype", REAL_DTYPES)
@pytest.mark.parametrize("k", [1, 3, 5])
def test_fit_icp_alignment_different_k(dtype, k):
    """Test ICP with different k values."""
    np.random.seed(42)
    pts0 = np.random.rand(100, 3).astype(dtype)
    pts1 = pts0 + np.array([0.1, 0.1, 0.1], dtype=dtype)

    cloud0 = tf.PointCloud(pts0)
    cloud1 = tf.PointCloud(pts1)

    T = tf.fit_icp_alignment(cloud0, cloud1, max_iterations=20, k=k)

    assert T.shape == (4, 4)
    assert T.dtype == dtype


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_fit_icp_alignment_with_outlier_rejection(dtype):
    """Test ICP with outlier rejection."""
    np.random.seed(42)
    pts0 = np.random.rand(100, 3).astype(dtype)
    pts1 = pts0 + np.array([0.1, 0.1, 0.1], dtype=dtype)

    cloud0 = tf.PointCloud(pts0)
    cloud1 = tf.PointCloud(pts1)

    T = tf.fit_icp_alignment(cloud0, cloud1, max_iterations=20, outlier_proportion=0.1)

    assert T.shape == (4, 4)
    assert T.dtype == dtype


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_fit_icp_alignment_convergence_threshold(dtype):
    """Test ICP with different convergence thresholds."""
    np.random.seed(42)
    pts0 = np.random.rand(100, 3).astype(dtype)
    pts1 = pts0 + np.array([0.05, 0.05, 0.05], dtype=dtype)

    cloud0 = tf.PointCloud(pts0)
    cloud1 = tf.PointCloud(pts1)

    # Tight threshold
    T = tf.fit_icp_alignment(cloud0, cloud1, max_iterations=100, min_relative_improvement=1e-8)

    assert T.shape == (4, 4)


# ==============================================================================
# Point-to-Plane Tests (with normals)
# ==============================================================================

@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_fit_icp_alignment_point_to_plane(dtype):
    """Test point-to-plane ICP with target normals."""
    # Create a simple mesh and compute normals
    faces, points = tf.make_sphere_mesh(1.0, 20, 20, dtype=dtype)
    mesh = tf.Mesh(faces, points)
    target_normals = tf.point_normals(mesh)

    # Transform source
    T_true = create_rotation_translation_z_3d(10, 0.1, 0.1, 0.05, dtype)
    source_pts = apply_transform_3d(points, T_true)

    source = tf.PointCloud(source_pts.astype(dtype))
    target = tf.PointCloud(points)

    # Point-to-plane ICP using (cloud, normals) tuple
    T = tf.fit_icp_alignment(source, (target, target_normals), max_iterations=50)

    # Apply transform
    aligned_pts = apply_transform_3d(source_pts, T)
    aligned = tf.PointCloud(aligned_pts.astype(dtype))

    error = tf.chamfer_error(aligned, target)
    assert error < 0.05, f"Point-to-plane ICP should converge, got error {error}"


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_fit_icp_alignment_normal_weighting(dtype):
    """Test ICP with normal weighting (both have normals)."""
    # Create meshes and compute normals
    faces, points = tf.make_sphere_mesh(1.0, 20, 20, dtype=dtype)
    mesh = tf.Mesh(faces, points)
    target_normals = tf.point_normals(mesh)

    # Transform source
    T_true = create_rotation_translation_z_3d(10, 0.1, 0.1, 0.05, dtype)
    source_pts = apply_transform_3d(points, T_true)

    # For source normals, we use the same normals (approximately correct after small rotation)
    source_normals = target_normals.copy()

    source = tf.PointCloud(source_pts.astype(dtype))
    target = tf.PointCloud(points)

    # Normal weighting ICP using (cloud, normals) tuples for both
    T = tf.fit_icp_alignment(
        (source, source_normals),
        (target, target_normals),
        max_iterations=50
    )

    # Apply transform
    aligned_pts = apply_transform_3d(source_pts, T)
    aligned = tf.PointCloud(aligned_pts.astype(dtype))

    error = tf.chamfer_error(aligned, target)
    assert error < 0.05, f"Normal-weighted ICP should converge, got error {error}"


# ==============================================================================
# Transformation Tagging Tests
# ==============================================================================

@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_fit_icp_alignment_with_initial_transform(dtype):
    """Test ICP with initial transformation set on source."""
    np.random.seed(42)
    pts0 = np.random.rand(200, 3).astype(dtype) * 2 - 1

    # Apply a larger transformation
    T_true = create_rotation_translation_z_3d(30, 0.5, 0.3, 0.2, dtype)
    pts1 = apply_transform_3d(pts0, T_true)

    cloud0 = tf.PointCloud(pts0)
    cloud1 = tf.PointCloud(pts1.astype(dtype))

    # First get OBB alignment as initial guess (returns DELTA)
    T_init = tf.fit_obb_alignment(cloud0, cloud1)
    cloud0.transformation = T_init

    # ICP refinement starting from OBB (returns DELTA)
    T_delta = tf.fit_icp_alignment(cloud0, cloud1, max_iterations=50)

    # Compose delta with initial to get total: total = delta @ initial
    T_total = T_delta @ T_init

    # Apply total transform to original points
    pts0_aligned = apply_transform_3d(pts0, T_total)
    aligned = tf.PointCloud(pts0_aligned.astype(dtype))

    error = tf.chamfer_error(aligned, cloud1)
    assert error < 0.05, f"ICP with initial transform should converge, got error {error}"


# ==============================================================================
# Output Properties Tests
# ==============================================================================

@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_fit_icp_alignment_output_shape_3d(dtype):
    """Output should be 4x4 for 3D."""
    np.random.seed(42)
    pts = np.random.rand(50, 3).astype(dtype)
    cloud0 = tf.PointCloud(pts)
    cloud1 = tf.PointCloud(pts + 0.1)

    T = tf.fit_icp_alignment(cloud0, cloud1, max_iterations=10)

    assert T.shape == (4, 4)


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_fit_icp_alignment_output_shape_2d(dtype):
    """Output should be 3x3 for 2D."""
    np.random.seed(42)
    pts = np.random.rand(50, 2).astype(dtype)
    cloud0 = tf.PointCloud(pts)
    cloud1 = tf.PointCloud(pts + 0.1)

    T = tf.fit_icp_alignment(cloud0, cloud1, max_iterations=10)

    assert T.shape == (3, 3)


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_fit_icp_alignment_output_dtype(dtype):
    """Output dtype should match input dtype."""
    np.random.seed(42)
    pts = np.random.rand(50, 3).astype(dtype)
    cloud0 = tf.PointCloud(pts)
    cloud1 = tf.PointCloud(pts + 0.1)

    T = tf.fit_icp_alignment(cloud0, cloud1, max_iterations=10)

    assert T.dtype == dtype


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_fit_icp_alignment_is_rigid(dtype):
    """Output should be a valid rigid transformation."""
    np.random.seed(42)
    pts0 = np.random.rand(100, 3).astype(dtype) * 2 - 1  # Center around origin
    T_true = create_rotation_translation_z_3d(15, 0.1, 0.1, 0.1, dtype)
    pts1 = apply_transform_3d(pts0, T_true)

    cloud0 = tf.PointCloud(pts0)
    cloud1 = tf.PointCloud(pts1.astype(dtype))

    T = tf.fit_icp_alignment(cloud0, cloud1, max_iterations=50)

    # Check that T is a valid 4x4 matrix
    assert T.shape == (4, 4)

    # Check bottom row is [0, 0, 0, 1]
    assert np.allclose(T[3, :], [0, 0, 0, 1], atol=1e-6)

    # Extract rotation part and check it's orthogonal with det=1
    R = T[:3, :3]
    det = np.linalg.det(R)

    # For a proper rotation, det should be 1
    # Allow some tolerance since ICP may not perfectly converge
    assert abs(det - 1.0) < 0.1, f"Rotation determinant should be ~1, got {det}"

    # Check approximate orthogonality
    assert np.allclose(R @ R.T, np.eye(3, dtype=dtype), atol=0.1)


# ==============================================================================
# Different Point Cloud Sizes
# ==============================================================================

@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_fit_icp_alignment_different_sizes(dtype):
    """Point clouds can have different numbers of points."""
    np.random.seed(42)
    pts0 = np.random.rand(50, 3).astype(dtype)
    pts1 = np.random.rand(100, 3).astype(dtype) + 0.1

    cloud0 = tf.PointCloud(pts0)
    cloud1 = tf.PointCloud(pts1)

    # Should work without error
    T = tf.fit_icp_alignment(cloud0, cloud1, max_iterations=20)

    assert T.shape == (4, 4)


# ==============================================================================
# Transform Combination Tests (4 cases: neither, source, target, both)
# ==============================================================================

@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_fit_icp_alignment_transform_combos_neither(dtype):
    """Case 1: Neither source nor target has a transform (world coords directly)."""
    # Create sphere mesh and extract points
    faces, points = tf.make_sphere_mesh(1.0, 20, 20, dtype=dtype)

    # Define transforms (small offset so ICP converges)
    T_source = create_rotation_translation_z_3d(0, 5.05, -2.05, 1.0, dtype)
    T_target = create_rotation_translation_z_3d(0, 5.0, -2.0, 1.0, dtype)

    # Pre-transform to world coordinates
    source_world = apply_transform_3d(points, T_source)
    target_world = apply_transform_3d(points, T_target)

    cloud_source = tf.PointCloud(source_world.astype(dtype))
    cloud_target = tf.PointCloud(target_world.astype(dtype))

    # Neither has transform set - both in world coords
    T = tf.fit_icp_alignment(cloud_source, cloud_target, max_iterations=50)

    # Apply result to source world points
    aligned = apply_transform_3d(source_world, T)
    cloud_aligned = tf.PointCloud(aligned.astype(dtype))

    error = tf.chamfer_error(cloud_aligned, cloud_target)
    assert error < 0.01, f"ICP should converge, got error {error}"


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_fit_icp_alignment_transform_combos_source_only(dtype):
    """Case 2: Source has transform, target in world coords."""
    # Create sphere mesh and extract points
    faces, points = tf.make_sphere_mesh(1.0, 20, 20, dtype=dtype)

    # Define transforms (small offset so ICP converges)
    T_source = create_rotation_translation_z_3d(0, 5.05, -2.05, 1.0, dtype)
    T_target = create_rotation_translation_z_3d(0, 5.0, -2.0, 1.0, dtype)

    # Target in world coords
    target_world = apply_transform_3d(points, T_target)

    # Source has transform set (local points with transformation)
    cloud_source = tf.PointCloud(points)
    cloud_source.transformation = T_source
    cloud_target = tf.PointCloud(target_world.astype(dtype))

    # ICP returns delta: source_world -> target_world
    T_delta = tf.fit_icp_alignment(cloud_source, cloud_target, max_iterations=50)

    # Total transform: delta @ T_source (maps local points to target world)
    T_total = T_delta @ T_source

    # Apply total to local points
    aligned = apply_transform_3d(points, T_total)
    cloud_aligned = tf.PointCloud(aligned.astype(dtype))

    error = tf.chamfer_error(cloud_aligned, cloud_target)
    assert error < 0.01, f"ICP should converge, got error {error}"


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_fit_icp_alignment_transform_combos_target_only(dtype):
    """Case 3: Target has transform, source in world coords."""
    # Create sphere mesh and extract points
    faces, points = tf.make_sphere_mesh(1.0, 20, 20, dtype=dtype)

    # Define transforms (small offset so ICP converges)
    T_source = create_rotation_translation_z_3d(0, 5.05, -2.05, 1.0, dtype)
    T_target = create_rotation_translation_z_3d(0, 5.0, -2.0, 1.0, dtype)

    # Source in world coords
    source_world = apply_transform_3d(points, T_source)

    # Target has transform set (local points with transformation)
    cloud_source = tf.PointCloud(source_world.astype(dtype))
    cloud_target = tf.PointCloud(points)
    cloud_target.transformation = T_target

    # ICP returns result mapping source_world -> target_world
    T = tf.fit_icp_alignment(cloud_source, cloud_target, max_iterations=50)

    # Apply to source world points, compare to target world
    aligned = apply_transform_3d(source_world, T)
    target_world = apply_transform_3d(points, T_target)
    cloud_aligned = tf.PointCloud(aligned.astype(dtype))
    cloud_target_world = tf.PointCloud(target_world.astype(dtype))

    error = tf.chamfer_error(cloud_aligned, cloud_target_world)
    assert error < 0.01, f"ICP should converge, got error {error}"


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_fit_icp_alignment_transform_combos_both(dtype):
    """Case 4: Both source and target have transforms."""
    # Create sphere mesh and extract points
    faces, points = tf.make_sphere_mesh(1.0, 20, 20, dtype=dtype)

    # Define transforms (small offset so ICP converges)
    T_source = create_rotation_translation_z_3d(0, 5.05, -2.05, 1.0, dtype)
    T_target = create_rotation_translation_z_3d(0, 5.0, -2.0, 1.0, dtype)

    # Both have transforms set (local points with transformations)
    cloud_source = tf.PointCloud(points)
    cloud_source.transformation = T_source
    cloud_target = tf.PointCloud(points)
    cloud_target.transformation = T_target

    # ICP returns delta: source_world -> target_world
    T_delta = tf.fit_icp_alignment(cloud_source, cloud_target, max_iterations=50)

    # Total transform: delta @ T_source (maps local points to target world)
    T_total = T_delta @ T_source

    # Apply total to local points, compare to target world
    aligned = apply_transform_3d(points, T_total)
    target_world = apply_transform_3d(points, T_target)
    cloud_aligned = tf.PointCloud(aligned.astype(dtype))
    cloud_target_world = tf.PointCloud(target_world.astype(dtype))

    error = tf.chamfer_error(cloud_aligned, cloud_target_world)
    assert error < 0.01, f"ICP should converge, got error {error}"


# ==============================================================================
# Transform Combination Tests - Point-to-Plane (with normals)
# ==============================================================================

@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_fit_icp_alignment_p2plane_transform_combos_neither(dtype):
    """Point-to-plane Case 1: Neither has transform (world coords directly)."""
    # Create sphere mesh and compute normals
    faces, points = tf.make_sphere_mesh(1.0, 20, 20, dtype=dtype)
    mesh = tf.Mesh(faces, points)
    normals = tf.point_normals(mesh)

    # Define transforms (small offset so ICP converges)
    T_source = create_rotation_translation_z_3d(0, 5.05, -2.05, 1.0, dtype)
    T_target = create_rotation_translation_z_3d(0, 5.0, -2.0, 1.0, dtype)

    # Pre-transform to world coordinates
    source_world = apply_transform_3d(points, T_source)
    target_world = apply_transform_3d(points, T_target)

    cloud_source = tf.PointCloud(source_world.astype(dtype))
    cloud_target = tf.PointCloud(target_world.astype(dtype))

    # Point-to-plane with target normals (normals unchanged for translation-only)
    T = tf.fit_icp_alignment(cloud_source, (cloud_target, normals), max_iterations=50)

    # Apply result to source world points
    aligned = apply_transform_3d(source_world, T)
    cloud_aligned = tf.PointCloud(aligned.astype(dtype))

    error = tf.chamfer_error(cloud_aligned, cloud_target)
    assert error < 0.01, f"Point-to-plane ICP should converge, got error {error}"


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_fit_icp_alignment_p2plane_transform_combos_source_only(dtype):
    """Point-to-plane Case 2: Source has transform, target in world coords."""
    # Create sphere mesh and compute normals
    faces, points = tf.make_sphere_mesh(1.0, 20, 20, dtype=dtype)
    mesh = tf.Mesh(faces, points)
    normals = tf.point_normals(mesh)

    # Define transforms (small offset so ICP converges)
    T_source = create_rotation_translation_z_3d(0, 5.05, -2.05, 1.0, dtype)
    T_target = create_rotation_translation_z_3d(0, 5.0, -2.0, 1.0, dtype)

    # Target in world coords
    target_world = apply_transform_3d(points, T_target)

    # Source has transform set
    cloud_source = tf.PointCloud(points)
    cloud_source.transformation = T_source
    cloud_target = tf.PointCloud(target_world.astype(dtype))

    # Point-to-plane ICP
    T_delta = tf.fit_icp_alignment(cloud_source, (cloud_target, normals), max_iterations=50)

    # Total transform: delta @ T_source
    T_total = T_delta @ T_source

    # Apply total to local points
    aligned = apply_transform_3d(points, T_total)
    cloud_aligned = tf.PointCloud(aligned.astype(dtype))

    error = tf.chamfer_error(cloud_aligned, cloud_target)
    assert error < 0.01, f"Point-to-plane ICP should converge, got error {error}"


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_fit_icp_alignment_p2plane_transform_combos_target_only(dtype):
    """Point-to-plane Case 3: Target has transform, source in world coords."""
    # Create sphere mesh and compute normals
    faces, points = tf.make_sphere_mesh(1.0, 20, 20, dtype=dtype)
    mesh = tf.Mesh(faces, points)
    normals = tf.point_normals(mesh)

    # Define transforms (small offset so ICP converges)
    T_source = create_rotation_translation_z_3d(0, 5.05, -2.05, 1.0, dtype)
    T_target = create_rotation_translation_z_3d(0, 5.0, -2.0, 1.0, dtype)

    # Source in world coords
    source_world = apply_transform_3d(points, T_source)

    # Target has transform set
    cloud_source = tf.PointCloud(source_world.astype(dtype))
    cloud_target = tf.PointCloud(points)
    cloud_target.transformation = T_target

    # Point-to-plane ICP
    T = tf.fit_icp_alignment(cloud_source, (cloud_target, normals), max_iterations=50)

    # Apply to source world points, compare to target world
    aligned = apply_transform_3d(source_world, T)
    target_world = apply_transform_3d(points, T_target)
    cloud_aligned = tf.PointCloud(aligned.astype(dtype))
    cloud_target_world = tf.PointCloud(target_world.astype(dtype))

    error = tf.chamfer_error(cloud_aligned, cloud_target_world)
    assert error < 0.01, f"Point-to-plane ICP should converge, got error {error}"


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_fit_icp_alignment_p2plane_transform_combos_both(dtype):
    """Point-to-plane Case 4: Both have transforms."""
    # Create sphere mesh and compute normals
    faces, points = tf.make_sphere_mesh(1.0, 20, 20, dtype=dtype)
    mesh = tf.Mesh(faces, points)
    normals = tf.point_normals(mesh)

    # Define transforms (small offset so ICP converges)
    T_source = create_rotation_translation_z_3d(0, 5.05, -2.05, 1.0, dtype)
    T_target = create_rotation_translation_z_3d(0, 5.0, -2.0, 1.0, dtype)

    # Both have transforms set
    cloud_source = tf.PointCloud(points)
    cloud_source.transformation = T_source
    cloud_target = tf.PointCloud(points)
    cloud_target.transformation = T_target

    # Point-to-plane ICP
    T_delta = tf.fit_icp_alignment(cloud_source, (cloud_target, normals), max_iterations=50)

    # Total transform: delta @ T_source
    T_total = T_delta @ T_source

    # Apply total to local points, compare to target world
    aligned = apply_transform_3d(points, T_total)
    target_world = apply_transform_3d(points, T_target)
    cloud_aligned = tf.PointCloud(aligned.astype(dtype))
    cloud_target_world = tf.PointCloud(target_world.astype(dtype))

    error = tf.chamfer_error(cloud_aligned, cloud_target_world)
    assert error < 0.01, f"Point-to-plane ICP should converge, got error {error}"


# ==============================================================================
# Error Handling Tests
# ==============================================================================

def test_fit_icp_alignment_dimension_mismatch():
    """Should raise error for mismatched dimensions."""
    pts2d = np.random.rand(50, 2).astype(np.float32)
    pts3d = np.random.rand(50, 3).astype(np.float32)

    cloud2d = tf.PointCloud(pts2d)
    cloud3d = tf.PointCloud(pts3d)

    with pytest.raises(ValueError, match="Dimension mismatch"):
        tf.fit_icp_alignment(cloud2d, cloud3d, max_iterations=10)


def test_fit_icp_alignment_dtype_mismatch():
    """Should raise error for mismatched dtypes."""
    pts32 = np.random.rand(50, 3).astype(np.float32)
    pts64 = np.random.rand(50, 3).astype(np.float64)

    cloud32 = tf.PointCloud(pts32)
    cloud64 = tf.PointCloud(pts64)

    with pytest.raises(ValueError, match="Dtype mismatch"):
        tf.fit_icp_alignment(cloud32, cloud64, max_iterations=10)


def test_fit_icp_alignment_normals_require_3d():
    """Point-to-plane requires 3D point clouds."""
    pts = np.random.rand(50, 2).astype(np.float32)
    normals = np.random.rand(50, 2).astype(np.float32)

    cloud = tf.PointCloud(pts)

    with pytest.raises(ValueError, match="3D"):
        tf.fit_icp_alignment(cloud, (cloud, normals), max_iterations=10)


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
