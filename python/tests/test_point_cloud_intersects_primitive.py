"""
Tests for PointCloud intersects with primitives

Copyright (c) 2025 Žiga Sajovic, XLAB
"""
import sys
import numpy as np
import pytest
import trueform as tf


# Type combinations for PointCloud: 2 real types × 2 dims = 4
REAL_DTYPES = [np.float32, np.float64]
DIMS = [2, 3]


def create_point_grid_2d(real_dtype):
    """
    Create a simple 3x3 grid of points in 2D
    Points at: [0,0], [1,0], [2,0], [0,1], [1,1], [2,1], [0,2], [1,2], [2,2]
    """
    points = np.array([
        [0, 0], [1, 0], [2, 0],
        [0, 1], [1, 1], [2, 1],
        [0, 2], [1, 2], [2, 2]
    ], dtype=real_dtype)
    return tf.PointCloud(points)


def create_point_grid_3d(real_dtype):
    """
    Create a simple 3x3 grid of points in 3D (z=0 plane)
    Points at: [0,0,0], [1,0,0], [2,0,0], [0,1,0], [1,1,0], [2,1,0], [0,2,0], [1,2,0], [2,2,0]
    """
    points = np.array([
        [0, 0, 0], [1, 0, 0], [2, 0, 0],
        [0, 1, 0], [1, 1, 0], [2, 1, 0],
        [0, 2, 0], [1, 2, 0], [2, 2, 0]
    ], dtype=real_dtype)
    return tf.PointCloud(points)


# ==============================================================================
# Point intersection tests
# ==============================================================================

@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_point_cloud_intersects_point_2d_hit(real_dtype):
    """Test point coincident with cloud point intersects"""
    cloud = create_point_grid_2d(real_dtype)

    # Point at same location as cloud point [1, 1]
    pt = tf.Point(np.array([1.0, 1.0], dtype=real_dtype))

    assert tf.intersects(cloud, pt)
    assert tf.intersects(pt, cloud)


@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_point_cloud_intersects_point_2d_miss(real_dtype):
    """Test point away from cloud doesn't intersect"""
    cloud = create_point_grid_2d(real_dtype)

    # Point between grid points (not on any cloud point)
    pt = tf.Point(np.array([0.5, 0.5], dtype=real_dtype))

    assert not tf.intersects(cloud, pt)
    assert not tf.intersects(pt, cloud)


@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_point_cloud_intersects_point_3d_hit(real_dtype):
    """Test point coincident with cloud point intersects (3D)"""
    cloud = create_point_grid_3d(real_dtype)

    # Point at same location as cloud point [1, 1, 0]
    pt = tf.Point(np.array([1.0, 1.0, 0.0], dtype=real_dtype))

    assert tf.intersects(cloud, pt)
    assert tf.intersects(pt, cloud)


# ==============================================================================
# Segment intersection tests
# ==============================================================================

@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_point_cloud_intersects_segment_2d_hit(real_dtype):
    """Test segment passing through cloud point intersects"""
    cloud = create_point_grid_2d(real_dtype)

    # Segment that passes through point [1, 1]
    seg = tf.Segment(np.array([[0.5, 1.0], [1.5, 1.0]], dtype=real_dtype))

    assert tf.intersects(cloud, seg)
    assert tf.intersects(seg, cloud)


@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_point_cloud_intersects_segment_2d_miss(real_dtype):
    """Test segment missing all cloud points doesn't intersect"""
    cloud = create_point_grid_2d(real_dtype)

    # Segment that doesn't pass through any cloud point
    seg = tf.Segment(np.array([[0.5, 0.5], [1.5, 0.5]], dtype=real_dtype))

    assert not tf.intersects(cloud, seg)
    assert not tf.intersects(seg, cloud)


@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_point_cloud_intersects_segment_3d_hit(real_dtype):
    """Test segment passing through cloud point intersects (3D)"""
    cloud = create_point_grid_3d(real_dtype)

    # Segment that passes through point [1, 1, 0]
    seg = tf.Segment(np.array([[1.0, 1.0, -1.0], [1.0, 1.0, 1.0]], dtype=real_dtype))

    assert tf.intersects(cloud, seg)
    assert tf.intersects(seg, cloud)


# ==============================================================================
# Polygon intersection tests
# ==============================================================================

@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_point_cloud_intersects_polygon_2d_hit(real_dtype):
    """Test polygon containing cloud point intersects"""
    cloud = create_point_grid_2d(real_dtype)

    # Triangle containing point [1, 1]
    poly = tf.Polygon(np.array([[0.5, 0.5], [1.5, 0.5], [1.0, 1.5]], dtype=real_dtype))

    assert tf.intersects(cloud, poly)
    assert tf.intersects(poly, cloud)


@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_point_cloud_intersects_polygon_2d_miss(real_dtype):
    """Test polygon not containing any cloud point doesn't intersect"""
    cloud = create_point_grid_2d(real_dtype)

    # Triangle that doesn't contain any cloud point
    poly = tf.Polygon(np.array([[0.1, 0.1], [0.4, 0.1], [0.25, 0.4]], dtype=real_dtype))

    assert not tf.intersects(cloud, poly)
    assert not tf.intersects(poly, cloud)


@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_point_cloud_intersects_polygon_3d_hit(real_dtype):
    """Test polygon containing cloud point intersects (3D)"""
    cloud = create_point_grid_3d(real_dtype)

    # Triangle in z=0 plane containing point [1, 1, 0]
    poly = tf.Polygon(np.array([[0.5, 0.5, 0.0], [1.5, 0.5, 0.0], [1.0, 1.5, 0.0]], dtype=real_dtype))

    assert tf.intersects(cloud, poly)
    assert tf.intersects(poly, cloud)


# ==============================================================================
# Ray intersection tests
# ==============================================================================

@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_point_cloud_intersects_ray_2d_hit(real_dtype):
    """Test ray passing through cloud point intersects"""
    cloud = create_point_grid_2d(real_dtype)

    # Ray that passes through point [1, 1]
    ray = tf.Ray(
        origin=np.array([1.0, 0.0], dtype=real_dtype),
        direction=np.array([0.0, 1.0], dtype=real_dtype)
    )

    assert tf.intersects(cloud, ray)
    assert tf.intersects(ray, cloud)


@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_point_cloud_intersects_ray_2d_miss(real_dtype):
    """Test ray missing all cloud points doesn't intersect"""
    cloud = create_point_grid_2d(real_dtype)

    # Ray that doesn't pass through any cloud point
    ray = tf.Ray(
        origin=np.array([0.5, 0.0], dtype=real_dtype),
        direction=np.array([0.0, 1.0], dtype=real_dtype)
    )

    assert not tf.intersects(cloud, ray)
    assert not tf.intersects(ray, cloud)


@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_point_cloud_intersects_ray_3d_hit(real_dtype):
    """Test ray passing through cloud point intersects (3D)"""
    cloud = create_point_grid_3d(real_dtype)

    # Ray pointing down through point [1, 1, 0]
    ray = tf.Ray(
        origin=np.array([1.0, 1.0, 1.0], dtype=real_dtype),
        direction=np.array([0.0, 0.0, -1.0], dtype=real_dtype)
    )

    assert tf.intersects(cloud, ray)
    assert tf.intersects(ray, cloud)


# ==============================================================================
# Line intersection tests
# ==============================================================================

@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_point_cloud_intersects_line_2d_hit(real_dtype):
    """Test line passing through cloud point intersects"""
    cloud = create_point_grid_2d(real_dtype)

    # Horizontal line through y=1 (passes through [0,1], [1,1], [2,1])
    line = tf.Line(
        origin=np.array([1.0, 1.0], dtype=real_dtype),
        direction=np.array([1.0, 0.0], dtype=real_dtype)
    )

    assert tf.intersects(cloud, line)
    assert tf.intersects(line, cloud)


@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_point_cloud_intersects_line_2d_miss(real_dtype):
    """Test line missing all cloud points doesn't intersect"""
    cloud = create_point_grid_2d(real_dtype)

    # Horizontal line at y=0.5 (doesn't pass through any cloud point)
    line = tf.Line(
        origin=np.array([1.0, 0.5], dtype=real_dtype),
        direction=np.array([1.0, 0.0], dtype=real_dtype)
    )

    assert not tf.intersects(cloud, line)
    assert not tf.intersects(line, cloud)


@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_point_cloud_intersects_line_3d_hit(real_dtype):
    """Test line passing through cloud point intersects (3D)"""
    cloud = create_point_grid_3d(real_dtype)

    # Vertical line (in z) through [1, 1, z]
    line = tf.Line(
        origin=np.array([1.0, 1.0, 0.0], dtype=real_dtype),
        direction=np.array([0.0, 0.0, 1.0], dtype=real_dtype)
    )

    assert tf.intersects(cloud, line)
    assert tf.intersects(line, cloud)


# ==============================================================================
# Plane intersection tests (3D only)
# ==============================================================================

@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_point_cloud_intersects_plane_3d_hit(real_dtype):
    """Test plane intersecting point cloud"""
    cloud = create_point_grid_3d(real_dtype)

    # Plane at z=0 (cloud points are at z=0, should intersect)
    plane = tf.Plane(
        normal=np.array([0.0, 0.0, 1.0], dtype=real_dtype),
        origin=np.array([0.0, 0.0, 0.0], dtype=real_dtype)
    )

    assert tf.intersects(cloud, plane)
    assert tf.intersects(plane, cloud)


@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_point_cloud_intersects_plane_3d_miss(real_dtype):
    """Test plane not intersecting point cloud"""
    cloud = create_point_grid_3d(real_dtype)

    # Plane at z=2 (cloud points are at z=0, should not intersect)
    plane = tf.Plane(
        normal=np.array([0.0, 0.0, 1.0], dtype=real_dtype),
        origin=np.array([0.0, 0.0, 2.0], dtype=real_dtype)
    )

    assert not tf.intersects(cloud, plane)
    assert not tf.intersects(plane, cloud)


# ==============================================================================
# Transformation tests
# ==============================================================================

@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_point_cloud_intersects_with_transformation_2d(real_dtype):
    """Test intersects with transformation (transform both cloud and primitive)"""
    cloud = create_point_grid_2d(real_dtype)

    # Point that intersects untransformed cloud
    pt = tf.Point(np.array([1.0, 1.0], dtype=real_dtype))
    assert tf.intersects(cloud, pt)

    # Apply transformation: translate by [5, 3]
    transformation = np.array([
        [1, 0, 5],
        [0, 1, 3],
        [0, 0, 1]
    ], dtype=real_dtype)

    cloud.transformation = transformation

    # Transform the point with same transformation
    pt_homogeneous = np.array([1.0, 1.0, 1.0], dtype=real_dtype)
    pt_transformed_coords = (transformation @ pt_homogeneous)[:2]
    pt_transformed = tf.Point(pt_transformed_coords)

    # Should still intersect
    assert tf.intersects(cloud, pt_transformed)
    assert tf.intersects(pt_transformed, cloud)


@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_point_cloud_intersects_with_transformation_3d(real_dtype):
    """Test intersects with transformation (3D)"""
    cloud = create_point_grid_3d(real_dtype)

    # Segment that intersects untransformed cloud
    seg = tf.Segment(np.array([[1.0, 1.0, -1.0], [1.0, 1.0, 1.0]], dtype=real_dtype))
    assert tf.intersects(cloud, seg)

    # Apply transformation: translate by [10, 5, 2]
    transformation = np.array([
        [1, 0, 0, 10],
        [0, 1, 0,  5],
        [0, 0, 1,  2],
        [0, 0, 0,  1]
    ], dtype=real_dtype)

    cloud.transformation = transformation

    # Transform the segment with same transformation
    seg_data = np.hstack([seg.data, np.ones((2, 1), dtype=real_dtype)])
    seg_transformed_data = (transformation @ seg_data.T).T[:, :3]
    seg_transformed = tf.Segment(seg_transformed_data)

    # Should still intersect
    assert tf.intersects(cloud, seg_transformed)
    assert tf.intersects(seg_transformed, cloud)


# ==============================================================================
# Dimension mismatch tests
# ==============================================================================

def test_point_cloud_intersects_dimension_mismatch():
    """Test that dimension mismatch raises error"""
    cloud_2d = create_point_grid_2d(np.float32)
    pt_3d = tf.Point(np.array([1.0, 1.0, 0.0], dtype=np.float32))

    with pytest.raises(ValueError, match="Dimension mismatch"):
        tf.intersects(cloud_2d, pt_3d)

    with pytest.raises(ValueError, match="Dimension mismatch"):
        tf.intersects(pt_3d, cloud_2d)


if __name__ == "__main__":
    # Run tests with verbose output
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
