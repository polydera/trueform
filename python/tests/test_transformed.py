"""
Tests for transformed function

Copyright (c) 2025 Žiga Sajovic, XLAB
"""

import sys
import numpy as np
import pytest
import trueform as tf


# Test parameters
REAL_DTYPES = [np.float32, np.float64]
DIMS = [2, 3]


def create_rotation_2d(angle_degrees, dtype):
    """Create a 2D rotation matrix"""
    angle = np.radians(angle_degrees)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    return np.array([
        [cos_a, -sin_a, 0],
        [sin_a,  cos_a, 0],
        [0,      0,     1]
    ], dtype=dtype)


def create_rotation_translation_2d(angle_degrees, tx, ty, dtype):
    """Create a 2D rotation + translation matrix"""
    angle = np.radians(angle_degrees)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    return np.array([
        [cos_a, -sin_a, tx],
        [sin_a,  cos_a, ty],
        [0,      0,     1]
    ], dtype=dtype)


def create_rotation_z_3d(angle_degrees, dtype):
    """Create a 3D rotation matrix around Z-axis"""
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
    """Create a 3D rotation around Z + translation matrix"""
    angle = np.radians(angle_degrees)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    return np.array([
        [cos_a, -sin_a, 0, tx],
        [sin_a,  cos_a, 0, ty],
        [0,      0,     1, tz],
        [0,      0,     0, 1]
    ], dtype=dtype)


# ==============================================================================
# Point Tests
# ==============================================================================

@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_transformed_point_2d_rotation(dtype):
    """Test 2D point rotation by 90 degrees"""
    pt = tf.Point(np.array([1, 0], dtype=dtype))
    T = create_rotation_2d(90, dtype)

    pt_transformed = tf.transformed(pt, T)

    # [1, 0] rotated 90° should be approximately [0, 1]
    expected = np.array([0, 1], dtype=dtype)
    assert np.allclose(pt_transformed.data, expected, atol=1e-5)


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_transformed_point_2d_rotation_translation(dtype):
    """Test 2D point rotation + translation"""
    pt = tf.Point(np.array([1, 0], dtype=dtype))
    T = create_rotation_translation_2d(90, 2, 3, dtype)

    pt_transformed = tf.transformed(pt, T)

    # [1, 0] rotated 90° + translated [2, 3] should be [2, 4]
    expected = np.array([2, 4], dtype=dtype)
    assert np.allclose(pt_transformed.data, expected, atol=1e-5)


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_transformed_point_3d_rotation(dtype):
    """Test 3D point rotation around Z by 90 degrees"""
    pt = tf.Point(np.array([1, 0, 0], dtype=dtype))
    T = create_rotation_z_3d(90, dtype)

    pt_transformed = tf.transformed(pt, T)

    # [1, 0, 0] rotated 90° around Z should be [0, 1, 0]
    expected = np.array([0, 1, 0], dtype=dtype)
    assert np.allclose(pt_transformed.data, expected, atol=1e-5)


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_transformed_point_3d_rotation_translation(dtype):
    """Test 3D point rotation + translation"""
    pt = tf.Point(np.array([1, 0, 0], dtype=dtype))
    T = create_rotation_translation_z_3d(90, 2, 3, 4, dtype)

    pt_transformed = tf.transformed(pt, T)

    # [1, 0, 0] rotated 90° around Z + translated [2, 3, 4] should be [2, 4, 4]
    expected = np.array([2, 4, 4], dtype=dtype)
    assert np.allclose(pt_transformed.data, expected, atol=1e-5)


# ==============================================================================
# Segment Tests
# ==============================================================================

@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_transformed_segment_2d(dtype):
    """Test 2D segment transformation"""
    seg = tf.Segment(np.array([[1, 0], [0, 1]], dtype=dtype))
    T = create_rotation_2d(90, dtype)

    seg_transformed = tf.transformed(seg, T)

    # Both endpoints should rotate
    expected_pt0 = np.array([0, 1], dtype=dtype)
    expected_pt1 = np.array([-1, 0], dtype=dtype)
    assert np.allclose(seg_transformed.data[0], expected_pt0, atol=1e-5)
    assert np.allclose(seg_transformed.data[1], expected_pt1, atol=1e-5)


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_transformed_segment_3d(dtype):
    """Test 3D segment transformation"""
    seg = tf.Segment(np.array([[1, 0, 0], [0, 1, 0]], dtype=dtype))
    T = create_rotation_translation_z_3d(90, 2, 3, 4, dtype)

    seg_transformed = tf.transformed(seg, T)

    # Check endpoints
    expected_pt0 = np.array([2, 4, 4], dtype=dtype)
    expected_pt1 = np.array([1, 3, 4], dtype=dtype)
    assert np.allclose(seg_transformed.data[0], expected_pt0, atol=1e-5)
    assert np.allclose(seg_transformed.data[1], expected_pt1, atol=1e-5)


# ==============================================================================
# Polygon Tests
# ==============================================================================

@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_transformed_polygon_2d_triangle(dtype):
    """Test 2D triangle transformation"""
    poly = tf.Polygon(np.array([[1, 0], [0, 1], [-1, 0]], dtype=dtype))
    T = create_rotation_2d(90, dtype)

    poly_transformed = tf.transformed(poly, T)

    # All vertices should rotate
    assert poly_transformed.data.shape == (3, 2)
    expected_pt0 = np.array([0, 1], dtype=dtype)
    assert np.allclose(poly_transformed.data[0], expected_pt0, atol=1e-5)


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_transformed_polygon_3d_triangle(dtype):
    """Test 3D triangle transformation"""
    poly = tf.Polygon(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=dtype))
    T = create_rotation_z_3d(90, dtype)

    poly_transformed = tf.transformed(poly, T)

    assert poly_transformed.data.shape == (3, 3)
    # First vertex [1, 0, 0] -> [0, 1, 0]
    expected_pt0 = np.array([0, 1, 0], dtype=dtype)
    assert np.allclose(poly_transformed.data[0], expected_pt0, atol=1e-5)


# ==============================================================================
# AABB Tests
# ==============================================================================

@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_transformed_aabb_2d(dtype):
    """Test 2D AABB transformation"""
    aabb = tf.AABB(
        min=np.array([0, 0], dtype=dtype),
        max=np.array([1, 1], dtype=dtype)
    )
    T = create_rotation_translation_2d(45, 2, 3, dtype)

    aabb_transformed = tf.transformed(aabb, T)

    # Transformed AABB should still be axis-aligned
    assert aabb_transformed.data.shape == (2, 2)
    # Min should be less than max
    assert np.all(aabb_transformed.data[0] <= aabb_transformed.data[1])


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_transformed_aabb_3d(dtype):
    """Test 3D AABB transformation"""
    aabb = tf.AABB(
        min=np.array([0, 0, 0], dtype=dtype),
        max=np.array([1, 1, 1], dtype=dtype)
    )
    T = create_rotation_translation_z_3d(45, 2, 3, 4, dtype)

    aabb_transformed = tf.transformed(aabb, T)

    # Transformed AABB should still be axis-aligned
    assert aabb_transformed.data.shape == (2, 3)
    # Min should be less than max
    assert np.all(aabb_transformed.data[0] <= aabb_transformed.data[1])


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_transformed_aabb_identity(dtype):
    """Test AABB with identity transformation"""
    aabb = tf.AABB(
        min=np.array([0, 0, 0], dtype=dtype),
        max=np.array([2, 3, 4], dtype=dtype)
    )
    T = np.eye(4, dtype=dtype)

    aabb_transformed = tf.transformed(aabb, T)

    # Should be unchanged
    assert np.allclose(aabb_transformed.data[0], aabb.data[0], atol=1e-5)
    assert np.allclose(aabb_transformed.data[1], aabb.data[1], atol=1e-5)


# ==============================================================================
# Ray Tests
# ==============================================================================

@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_transformed_ray_2d(dtype):
    """Test 2D ray transformation"""
    ray = tf.Ray(
        origin=np.array([0, 0], dtype=dtype),
        direction=np.array([1, 0], dtype=dtype)
    )
    T = create_rotation_2d(90, dtype)

    ray_transformed = tf.transformed(ray, T)

    # Direction should be rotated and normalized
    expected_direction = np.array([0, 1], dtype=dtype)
    assert np.allclose(ray_transformed.data[1], expected_direction, atol=1e-5)
    assert np.allclose(np.linalg.norm(ray_transformed.data[1]), 1.0, atol=1e-5)


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_transformed_ray_3d(dtype):
    """Test 3D ray transformation"""
    ray = tf.Ray(
        origin=np.array([0, 0, 0], dtype=dtype),
        direction=np.array([1, 0, 0], dtype=dtype)
    )
    T = create_rotation_translation_z_3d(90, 2, 3, 4, dtype)

    ray_transformed = tf.transformed(ray, T)

    # Origin should be translated
    expected_origin = np.array([2, 3, 4], dtype=dtype)
    assert np.allclose(ray_transformed.data[0], expected_origin, atol=1e-5)

    # Direction should be rotated and normalized
    expected_direction = np.array([0, 1, 0], dtype=dtype)
    assert np.allclose(ray_transformed.data[1], expected_direction, atol=1e-5)
    assert np.allclose(np.linalg.norm(ray_transformed.data[1]), 1.0, atol=1e-5)


# ==============================================================================
# Line Tests
# ==============================================================================

@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_transformed_line_2d(dtype):
    """Test 2D line transformation"""
    line = tf.Line(
        origin=np.array([0, 0], dtype=dtype),
        direction=np.array([1, 0], dtype=dtype)
    )
    T = create_rotation_translation_2d(90, 1, 2, dtype)

    line_transformed = tf.transformed(line, T)

    # Origin should be transformed
    expected_origin = np.array([1, 2], dtype=dtype)
    assert np.allclose(line_transformed.data[0], expected_origin, atol=1e-5)

    # Direction should be rotated and normalized
    expected_direction = np.array([0, 1], dtype=dtype)
    assert np.allclose(line_transformed.data[1], expected_direction, atol=1e-5)
    assert np.allclose(np.linalg.norm(line_transformed.data[1]), 1.0, atol=1e-5)


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_transformed_line_3d(dtype):
    """Test 3D line transformation"""
    line = tf.Line(
        origin=np.array([1, 0, 0], dtype=dtype),
        direction=np.array([1, 0, 0], dtype=dtype)
    )
    T = create_rotation_z_3d(90, dtype)

    line_transformed = tf.transformed(line, T)

    # Direction should be normalized
    assert np.allclose(np.linalg.norm(line_transformed.data[1]), 1.0, atol=1e-5)


# ==============================================================================
# Plane Tests (3D only)
# ==============================================================================

@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_transformed_plane_3d(dtype):
    """Test 3D plane transformation"""
    # Plane: z = 0 (normal = [0, 0, 1], d = 0)
    plane = tf.Plane(np.array([0, 0, 1, 0], dtype=dtype))
    T = create_rotation_translation_z_3d(0, 0, 0, 5, dtype)  # Translate up by 5

    plane_transformed = tf.transformed(plane, T)

    # Normal should remain [0, 0, 1]
    assert np.allclose(plane_transformed.data[:3], [0, 0, 1], atol=1e-5)
    # d should change to -5 (plane moved up)
    assert np.allclose(plane_transformed.data[3], -5, atol=1e-5)
    # Normal should be unit length
    assert np.allclose(np.linalg.norm(plane_transformed.data[:3]), 1.0, atol=1e-5)


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_transformed_plane_rotation(dtype):
    """Test plane normal rotation"""
    # Plane: z = 0 (normal = [0, 0, 1], d = 0)
    plane = tf.Plane(np.array([0, 0, 1, 0], dtype=dtype))

    # Rotate 90° around Y-axis (swaps X and Z axes)
    T = np.array([
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [-1, 0, 0, 0],
        [0, 0, 0, 1]
    ], dtype=dtype)

    plane_transformed = tf.transformed(plane, T)

    # Normal should be transformed and normalized
    assert np.allclose(np.linalg.norm(plane_transformed.data[:3]), 1.0, atol=1e-5)


# ==============================================================================
# Error Tests
# ==============================================================================

def test_transformed_invalid_matrix_shape():
    """Test that invalid transformation matrix raises error"""
    pt = tf.Point([1, 0, 0])
    T_wrong = np.eye(3)  # 3x3 for 3D point (should be 4x4)

    with pytest.raises(ValueError, match="Transformation matrix must be"):
        tf.transformed(pt, T_wrong)


def test_transformed_dimension_mismatch():
    """Test that 2D transformation on 3D point raises error"""
    pt_3d = tf.Point([1, 0, 0])
    T_2d = np.eye(3)  # 2D transformation

    with pytest.raises(ValueError, match="Transformation matrix must be"):
        tf.transformed(pt_3d, T_2d)


# ==============================================================================
# Batch Tests
# ==============================================================================

@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_transformed_batch_point_3d(dtype):
    """Test batch point transformation"""
    pts = tf.Point(np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ], dtype=dtype))
    assert pts.is_batch
    T = create_rotation_z_3d(90, dtype)

    result = tf.transformed(pts, T)

    assert result.is_batch
    assert result.count == 3
    expected = np.array([
        [0, 1, 0],
        [-1, 0, 0],
        [0, 0, 1]
    ], dtype=dtype)
    assert np.allclose(result.data, expected, atol=1e-5)


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_transformed_batch_point_2d(dtype):
    """Test batch 2D point transformation"""
    pts = tf.Point(np.array([
        [1, 0],
        [0, 1],
    ], dtype=dtype))
    assert pts.is_batch
    T = create_rotation_translation_2d(90, 2, 3, dtype)

    result = tf.transformed(pts, T)

    assert result.is_batch
    assert result.count == 2
    # [1,0] -> [0,1] + [2,3] = [2,4]
    # [0,1] -> [-1,0] + [2,3] = [1,3]
    expected = np.array([[2, 4], [1, 3]], dtype=dtype)
    assert np.allclose(result.data, expected, atol=1e-5)


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_transformed_batch_segment_3d(dtype):
    """Test batch segment transformation"""
    segs = tf.Segment(np.array([
        [[1, 0, 0], [2, 0, 0]],
        [[0, 1, 0], [0, 2, 0]],
    ], dtype=dtype))
    assert segs.is_batch
    T = create_rotation_z_3d(90, dtype)

    result = tf.transformed(segs, T)

    assert result.is_batch
    assert result.count == 2
    # seg1: [1,0,0]->[0,1,0], [2,0,0]->[0,2,0]
    # seg2: [0,1,0]->[-1,0,0], [0,2,0]->[-2,0,0]
    expected = np.array([
        [[0, 1, 0], [0, 2, 0]],
        [[-1, 0, 0], [-2, 0, 0]],
    ], dtype=dtype)
    assert np.allclose(result.data, expected, atol=1e-5)


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_transformed_batch_ray_3d(dtype):
    """Test batch ray transformation"""
    rays = tf.Ray(data=np.array([
        [[0, 0, 0], [1, 0, 0]],
        [[1, 0, 0], [0, 1, 0]],
    ], dtype=dtype))
    assert rays.is_batch
    T = create_rotation_z_3d(90, dtype)

    result = tf.transformed(rays, T)

    assert result.is_batch
    assert result.count == 2
    # ray1: origin [0,0,0]->[0,0,0], dir [1,0,0]->[0,1,0]
    # ray2: origin [1,0,0]->[0,1,0], dir [0,1,0]->[-1,0,0]
    assert np.allclose(result.data[0, 0], [0, 0, 0], atol=1e-5)
    assert np.allclose(result.data[0, 1], [0, 1, 0], atol=1e-5)
    assert np.allclose(result.data[1, 0], [0, 1, 0], atol=1e-5)
    assert np.allclose(result.data[1, 1], [-1, 0, 0], atol=1e-5)
    # Directions should be unit vectors
    assert np.allclose(np.linalg.norm(result.data[:, 1], axis=-1), 1.0, atol=1e-5)


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_transformed_batch_line_3d(dtype):
    """Test batch line transformation"""
    lines = tf.Line(data=np.array([
        [[0, 0, 0], [1, 0, 0]],
        [[0, 0, 0], [0, 0, 1]],
    ], dtype=dtype))
    assert lines.is_batch
    T = create_rotation_z_3d(90, dtype)

    result = tf.transformed(lines, T)

    assert result.is_batch
    assert result.count == 2
    # line1 dir [1,0,0] -> [0,1,0]
    # line2 dir [0,0,1] -> [0,0,1] (z-axis unaffected by z-rotation)
    assert np.allclose(result.data[0, 1], [0, 1, 0], atol=1e-5)
    assert np.allclose(result.data[1, 1], [0, 0, 1], atol=1e-5)


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_transformed_batch_aabb_3d(dtype):
    """Test batch AABB transformation"""
    batch_data = np.array([
        [[0, 0, 0], [1, 1, 1]],
        [[2, 0, 0], [3, 1, 1]],
    ], dtype=dtype)
    boxes = tf.AABB(batch_data)
    assert boxes.is_batch

    T = create_rotation_translation_z_3d(0, 10, 0, 0, dtype)  # translate x+10

    result = tf.transformed(boxes, T)

    assert result.is_batch
    assert result.count == 2
    # box1: [0,0,0]-[1,1,1] -> [10,0,0]-[11,1,1]
    # box2: [2,0,0]-[3,1,1] -> [12,0,0]-[13,1,1]
    assert np.allclose(result.data[0, 0], [10, 0, 0], atol=1e-5)
    assert np.allclose(result.data[0, 1], [11, 1, 1], atol=1e-5)
    assert np.allclose(result.data[1, 0], [12, 0, 0], atol=1e-5)
    assert np.allclose(result.data[1, 1], [13, 1, 1], atol=1e-5)


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_transformed_batch_polygon_3d(dtype):
    """Test batch polygon transformation"""
    polys = tf.Polygon(np.array([
        [[1, 0, 0], [0, 1, 0], [0, 0, 0]],
        [[2, 0, 0], [0, 2, 0], [0, 0, 0]],
    ], dtype=dtype))
    assert polys.is_batch
    T = create_rotation_z_3d(90, dtype)

    result = tf.transformed(polys, T)

    assert result.is_batch
    assert result.count == 2
    # poly1 vertex0: [1,0,0] -> [0,1,0]
    assert np.allclose(result.data[0, 0], [0, 1, 0], atol=1e-5)
    # poly2 vertex0: [2,0,0] -> [0,2,0]
    assert np.allclose(result.data[1, 0], [0, 2, 0], atol=1e-5)


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_transformed_batch_plane_3d(dtype):
    """Test batch plane transformation"""
    planes = tf.Plane(np.array([
        [0, 0, 1, 0],     # z=0 plane
        [0, 0, 1, -5],    # z=5 plane
    ], dtype=dtype))
    assert planes.is_batch

    # Translate up by 10
    T = create_rotation_translation_z_3d(0, 0, 0, 10, dtype)

    result = tf.transformed(planes, T)

    assert result.is_batch
    assert result.count == 2
    # z=0 plane moved up: normal [0,0,1], d = -10
    assert np.allclose(result.data[0, :3], [0, 0, 1], atol=1e-5)
    assert np.isclose(result.data[0, 3], -10, atol=1e-5)
    # z=5 plane moved up: normal [0,0,1], d = -15
    assert np.allclose(result.data[1, :3], [0, 0, 1], atol=1e-5)
    assert np.isclose(result.data[1, 3], -15, atol=1e-5)


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_transformed_batch_preserves_distances(dtype):
    """Test that transformation preserves distances between batch elements."""
    pts = tf.Point(np.array([
        [0, 0, 0],
        [3, 0, 0],
    ], dtype=dtype))
    T = create_rotation_translation_z_3d(45, 10, 20, 30, dtype)

    result = tf.transformed(pts, T)

    # Distance between transformed points should equal original
    original_dist = np.linalg.norm(pts.data[0] - pts.data[1])
    transformed_dist = np.linalg.norm(result.data[0] - result.data[1])
    assert np.isclose(original_dist, transformed_dist, atol=1e-4)


if __name__ == "__main__":
    # Run tests with verbose output
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
