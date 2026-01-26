"""
Tests for EdgeMesh intersects with primitives

Copyright (c) 2025 Žiga Sajovic, XLAB
"""
import sys
import numpy as np
import pytest
import trueform as tf


# Type combinations for EdgeMesh: 2 index types × 2 real types × 2 dims = 8
INDEX_DTYPES = [np.int32, np.int64]
REAL_DTYPES = [np.float32, np.float64]
DIMS = [2, 3]


def create_straight_polyline_2d(index_dtype, real_dtype):
    """
    Create a simple straight polyline in 2D
    Points: [0,0], [1,0], [2,0], [3,0]
    Edges: [0,1], [1,2], [2,3]
    Edge 0: [0,0] to [1,0]
    Edge 1: [1,0] to [2,0]
    Edge 2: [2,0] to [3,0]
    """
    edges = np.array([[0, 1], [1, 2], [2, 3]], dtype=index_dtype)
    points = np.array([[0, 0], [1, 0], [2, 0], [3, 0]], dtype=real_dtype)
    return tf.EdgeMesh(edges, points)


def create_straight_polyline_3d(index_dtype, real_dtype):
    """
    Create a simple straight polyline in 3D (along x-axis, z=0)
    Points: [0,0,0], [1,0,0], [2,0,0], [3,0,0]
    Edges: [0,1], [1,2], [2,3]
    """
    edges = np.array([[0, 1], [1, 2], [2, 3]], dtype=index_dtype)
    points = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]], dtype=real_dtype)
    return tf.EdgeMesh(edges, points)


# ==============================================================================
# Point intersection tests
# ==============================================================================

@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_edge_mesh_intersects_point_2d_hit(index_dtype, real_dtype):
    """Test point on edge intersects"""
    edge_mesh = create_straight_polyline_2d(index_dtype, real_dtype)

    # Point on edge 1 at [1.5, 0]
    pt = tf.Point(np.array([1.5, 0.0], dtype=real_dtype))

    assert tf.intersects(edge_mesh, pt)
    assert tf.intersects(pt, edge_mesh)  # Symmetric


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_edge_mesh_intersects_point_2d_miss(index_dtype, real_dtype):
    """Test point away from edge doesn't intersect"""
    edge_mesh = create_straight_polyline_2d(index_dtype, real_dtype)

    # Point above edge mesh at [1.5, 1.0]
    pt = tf.Point(np.array([1.5, 1.0], dtype=real_dtype))

    assert not tf.intersects(edge_mesh, pt)
    assert not tf.intersects(pt, edge_mesh)


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_edge_mesh_intersects_point_3d_hit(index_dtype, real_dtype):
    """Test point on edge intersects (3D)"""
    edge_mesh = create_straight_polyline_3d(index_dtype, real_dtype)

    # Point on edge 1 at [1.5, 0, 0]
    pt = tf.Point(np.array([1.5, 0.0, 0.0], dtype=real_dtype))

    assert tf.intersects(edge_mesh, pt)
    assert tf.intersects(pt, edge_mesh)


# ==============================================================================
# Segment intersection tests
# ==============================================================================

@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_edge_mesh_intersects_segment_2d_hit(index_dtype, real_dtype):
    """Test segment crossing edge intersects"""
    edge_mesh = create_straight_polyline_2d(index_dtype, real_dtype)

    # Vertical segment crossing edge 1 at x=1.5
    seg = tf.Segment(np.array([[1.5, -1.0], [1.5, 1.0]], dtype=real_dtype))

    assert tf.intersects(edge_mesh, seg)
    assert tf.intersects(seg, edge_mesh)


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_edge_mesh_intersects_segment_2d_miss(index_dtype, real_dtype):
    """Test segment above edge mesh doesn't intersect"""
    edge_mesh = create_straight_polyline_2d(index_dtype, real_dtype)

    # Segment above the edge mesh
    seg = tf.Segment(np.array([[1.5, 1.0], [1.5, 2.0]], dtype=real_dtype))

    assert not tf.intersects(edge_mesh, seg)
    assert not tf.intersects(seg, edge_mesh)


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_edge_mesh_intersects_segment_3d_hit(index_dtype, real_dtype):
    """Test segment crossing edge intersects (3D)"""
    edge_mesh = create_straight_polyline_3d(index_dtype, real_dtype)

    # Vertical segment (in z) crossing edge 1 at [1.5, 0, z]
    seg = tf.Segment(np.array([[1.5, 0.0, -1.0], [1.5, 0.0, 1.0]], dtype=real_dtype))

    assert tf.intersects(edge_mesh, seg)
    assert tf.intersects(seg, edge_mesh)


# ==============================================================================
# Polygon intersection tests
# ==============================================================================

@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_edge_mesh_intersects_polygon_2d_hit(index_dtype, real_dtype):
    """Test polygon overlapping edge intersects"""
    edge_mesh = create_straight_polyline_2d(index_dtype, real_dtype)

    # Triangle overlapping edge 1 (straddles y=0)
    poly = tf.Polygon(np.array([[1.3, -0.1], [1.7, -0.1], [1.5, 0.1]], dtype=real_dtype))

    assert tf.intersects(edge_mesh, poly)
    assert tf.intersects(poly, edge_mesh)


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_edge_mesh_intersects_polygon_2d_miss(index_dtype, real_dtype):
    """Test polygon above edge mesh doesn't intersect"""
    edge_mesh = create_straight_polyline_2d(index_dtype, real_dtype)

    # Triangle entirely above edge mesh
    poly = tf.Polygon(np.array([[1.3, 1.0], [1.7, 1.0], [1.5, 1.5]], dtype=real_dtype))

    assert not tf.intersects(edge_mesh, poly)
    assert not tf.intersects(poly, edge_mesh)


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_edge_mesh_intersects_polygon_3d_hit(index_dtype, real_dtype):
    """Test polygon overlapping edge intersects (3D)"""
    edge_mesh = create_straight_polyline_3d(index_dtype, real_dtype)

    # Triangle in xy-plane overlapping edge 1
    poly = tf.Polygon(np.array([[1.3, -0.1, 0.0], [1.7, -0.1, 0.0], [1.5, 0.1, 0.0]], dtype=real_dtype))

    assert tf.intersects(edge_mesh, poly)
    assert tf.intersects(poly, edge_mesh)


# ==============================================================================
# Ray intersection tests
# ==============================================================================

@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_edge_mesh_intersects_ray_2d_hit(index_dtype, real_dtype):
    """Test ray hitting edge intersects"""
    edge_mesh = create_straight_polyline_2d(index_dtype, real_dtype)

    # Ray pointing down from above edge 1
    ray = tf.Ray(
        origin=np.array([1.5, 1.0], dtype=real_dtype),
        direction=np.array([0.0, -1.0], dtype=real_dtype)
    )

    assert tf.intersects(edge_mesh, ray)
    assert tf.intersects(ray, edge_mesh)


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_edge_mesh_intersects_ray_2d_miss(index_dtype, real_dtype):
    """Test ray pointing away doesn't intersect"""
    edge_mesh = create_straight_polyline_2d(index_dtype, real_dtype)

    # Ray pointing up from above edge mesh (away from it)
    ray = tf.Ray(
        origin=np.array([1.5, 1.0], dtype=real_dtype),
        direction=np.array([0.0, 1.0], dtype=real_dtype)
    )

    assert not tf.intersects(edge_mesh, ray)
    assert not tf.intersects(ray, edge_mesh)


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_edge_mesh_intersects_ray_3d_hit(index_dtype, real_dtype):
    """Test ray hitting edge intersects (3D)"""
    edge_mesh = create_straight_polyline_3d(index_dtype, real_dtype)

    # Ray pointing down (in z) from above edge 1
    ray = tf.Ray(
        origin=np.array([1.5, 0.0, 1.0], dtype=real_dtype),
        direction=np.array([0.0, 0.0, -1.0], dtype=real_dtype)
    )

    assert tf.intersects(edge_mesh, ray)
    assert tf.intersects(ray, edge_mesh)


# ==============================================================================
# Line intersection tests
# ==============================================================================

@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_edge_mesh_intersects_line_2d_hit(index_dtype, real_dtype):
    """Test line through edge intersects"""
    edge_mesh = create_straight_polyline_2d(index_dtype, real_dtype)

    # Vertical line through edge 1 at x=1.5
    line = tf.Line(
        origin=np.array([1.5, 0.0], dtype=real_dtype),
        direction=np.array([0.0, 1.0], dtype=real_dtype)
    )

    assert tf.intersects(edge_mesh, line)
    assert tf.intersects(line, edge_mesh)


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_edge_mesh_intersects_line_2d_miss(index_dtype, real_dtype):
    """Test parallel line above edge mesh doesn't intersect"""
    edge_mesh = create_straight_polyline_2d(index_dtype, real_dtype)

    # Horizontal line parallel to edges but above at y=1
    line = tf.Line(
        origin=np.array([1.5, 1.0], dtype=real_dtype),
        direction=np.array([1.0, 0.0], dtype=real_dtype)
    )

    assert not tf.intersects(edge_mesh, line)
    assert not tf.intersects(line, edge_mesh)


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_edge_mesh_intersects_line_3d_hit(index_dtype, real_dtype):
    """Test line through edge intersects (3D)"""
    edge_mesh = create_straight_polyline_3d(index_dtype, real_dtype)

    # Vertical line (in z) through edge 1 at [1.5, 0, z]
    line = tf.Line(
        origin=np.array([1.5, 0.0, 0.0], dtype=real_dtype),
        direction=np.array([0.0, 0.0, 1.0], dtype=real_dtype)
    )

    assert tf.intersects(edge_mesh, line)
    assert tf.intersects(line, edge_mesh)


# ==============================================================================
# Plane intersection tests (3D only)
# ==============================================================================

@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_edge_mesh_intersects_plane_3d_hit(index_dtype, real_dtype):
    """Test plane intersecting edge mesh"""
    edge_mesh = create_straight_polyline_3d(index_dtype, real_dtype)

    # Plane at z=0 (edge mesh is at z=0, should intersect)
    plane = tf.Plane(
        normal=np.array([0.0, 0.0, 1.0], dtype=real_dtype),
        origin=np.array([0.0, 0.0, 0.0], dtype=real_dtype)
    )

    assert tf.intersects(edge_mesh, plane)
    assert tf.intersects(plane, edge_mesh)


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_edge_mesh_intersects_plane_3d_miss(index_dtype, real_dtype):
    """Test plane not intersecting edge mesh"""
    edge_mesh = create_straight_polyline_3d(index_dtype, real_dtype)

    # Plane at z=2 (edge mesh is at z=0, should not intersect)
    plane = tf.Plane(
        normal=np.array([0.0, 0.0, 1.0], dtype=real_dtype),
        origin=np.array([0.0, 0.0, 2.0], dtype=real_dtype)
    )

    assert not tf.intersects(edge_mesh, plane)
    assert not tf.intersects(plane, edge_mesh)


# ==============================================================================
# Transformation tests
# ==============================================================================

@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_edge_mesh_intersects_with_transformation_2d(index_dtype, real_dtype):
    """Test intersects with transformation (transform both form and primitive)"""
    edge_mesh = create_straight_polyline_2d(index_dtype, real_dtype)

    # Point that intersects untransformed edge mesh
    pt = tf.Point(np.array([1.5, 0.0], dtype=real_dtype))
    assert tf.intersects(edge_mesh, pt)

    # Apply transformation: translate by [5, 3]
    transformation = np.array([
        [1, 0, 5],
        [0, 1, 3],
        [0, 0, 1]
    ], dtype=real_dtype)

    edge_mesh.transformation = transformation

    # Transform the point with same transformation
    pt_homogeneous = np.array([1.5, 0.0, 1.0], dtype=real_dtype)
    pt_transformed_coords = (transformation @ pt_homogeneous)[:2]
    pt_transformed = tf.Point(pt_transformed_coords)

    # Should still intersect
    assert tf.intersects(edge_mesh, pt_transformed)
    assert tf.intersects(pt_transformed, edge_mesh)


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_edge_mesh_intersects_with_transformation_3d(index_dtype, real_dtype):
    """Test intersects with transformation (3D)"""
    edge_mesh = create_straight_polyline_3d(index_dtype, real_dtype)

    # Segment that intersects untransformed edge mesh
    seg = tf.Segment(np.array([[1.5, 0.0, -1.0], [1.5, 0.0, 1.0]], dtype=real_dtype))
    assert tf.intersects(edge_mesh, seg)

    # Apply transformation: translate by [10, 5, 2]
    transformation = np.array([
        [1, 0, 0, 10],
        [0, 1, 0,  5],
        [0, 0, 1,  2],
        [0, 0, 0,  1]
    ], dtype=real_dtype)

    edge_mesh.transformation = transformation

    # Transform the segment with same transformation
    seg_data = np.array([[1.5, 0.0, -1.0, 1.0], [1.5, 0.0, 1.0, 1.0]], dtype=real_dtype)
    seg_transformed_data = (transformation @ seg_data.T).T[:, :3]
    seg_transformed = tf.Segment(seg_transformed_data)

    # Should still intersect
    assert tf.intersects(edge_mesh, seg_transformed)
    assert tf.intersects(seg_transformed, edge_mesh)


# ==============================================================================
# Dimension mismatch tests
# ==============================================================================

def test_edge_mesh_intersects_dimension_mismatch():
    """Test that dimension mismatch raises error"""
    edge_mesh_2d = create_straight_polyline_2d(np.int32, np.float32)
    pt_3d = tf.Point(np.array([1.5, 0.0, 0.0], dtype=np.float32))

    with pytest.raises(ValueError, match="Dimension mismatch"):
        tf.intersects(edge_mesh_2d, pt_3d)

    with pytest.raises(ValueError, match="Dimension mismatch"):
        tf.intersects(pt_3d, edge_mesh_2d)


if __name__ == "__main__":
    # Run tests with verbose output
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
