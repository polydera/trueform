"""
Tests for Mesh intersects with primitives

Copyright (c) 2025 Žiga Sajovic, XLAB
"""

import sys
import numpy as np
import pytest
import trueform as tf


# Type combinations for Mesh: 2 index types × 2 real types × 2 mesh types × 2 dims = 16
INDEX_DTYPES = [np.int32, np.int64]
REAL_DTYPES = [np.float32, np.float64]
MESH_TYPES = ['triangle', 'dynamic']  # triangles and dynamic (variable-size)
DIMS = [2, 3]


def create_tiled_plane_2d_triangles(index_dtype, real_dtype):
    """
    Create a simple 2x1 tiled plane in 2D with triangles
    Points (2x3 grid):
      3---4---5
      |  /|  /|
      | / | / |
      |/  |/  |
      0---1---2

    Faces (4 triangles):
      Face 0: [0, 1, 4] - lower-left triangle of first cell
      Face 1: [0, 4, 3] - upper-right triangle of first cell
      Face 2: [1, 2, 5] - lower-left triangle of second cell
      Face 3: [1, 5, 4] - upper-right triangle of second cell
    """
    faces = np.array([
        [0, 1, 4],
        [0, 4, 3],
        [1, 2, 5],
        [1, 5, 4]
    ], dtype=index_dtype)
    points = np.array([
        [0, 0], [1, 0], [2, 0],
        [0, 1], [1, 1], [2, 1]
    ], dtype=real_dtype)
    return tf.Mesh(faces, points)


def create_tiled_plane_3d_triangles(index_dtype, real_dtype):
    """Create a simple 2x1 tiled plane in 3D (z=0) with triangles"""
    faces = np.array([
        [0, 1, 4],
        [0, 4, 3],
        [1, 2, 5],
        [1, 5, 4]
    ], dtype=index_dtype)
    points = np.array([
        [0, 0, 0], [1, 0, 0], [2, 0, 0],
        [0, 1, 0], [1, 1, 0], [2, 1, 0]
    ], dtype=real_dtype)
    return tf.Mesh(faces, points)


def create_tiled_plane_2d_dynamic(index_dtype, real_dtype):
    """Create a simple 2x1 tiled plane in 2D with dynamic (variable-size) polygons.
    Uses a mix of triangles and quads to create a dynamic mesh."""
    # Mixed polygon faces: one quad and two triangles
    # Quad: [0, 1, 4, 3]
    # Triangle: [1, 2, 5]
    # Triangle: [1, 5, 4]
    offsets = np.array([0, 4, 7, 10], dtype=index_dtype)
    data = np.array([0, 1, 4, 3, 1, 2, 5, 1, 5, 4], dtype=index_dtype)
    faces = tf.OffsetBlockedArray(offsets, data)
    points = np.array([
        [0, 0], [1, 0], [2, 0],
        [0, 1], [1, 1], [2, 1]
    ], dtype=real_dtype)
    return tf.Mesh(faces, points)


def create_tiled_plane_3d_dynamic(index_dtype, real_dtype):
    """Create a simple 2x1 tiled plane in 3D (z=0) with dynamic (variable-size) polygons."""
    # Mixed polygon faces: one quad and two triangles
    offsets = np.array([0, 4, 7, 10], dtype=index_dtype)
    data = np.array([0, 1, 4, 3, 1, 2, 5, 1, 5, 4], dtype=index_dtype)
    faces = tf.OffsetBlockedArray(offsets, data)
    points = np.array([
        [0, 0, 0], [1, 0, 0], [2, 0, 0],
        [0, 1, 0], [1, 1, 0], [2, 1, 0]
    ], dtype=real_dtype)
    return tf.Mesh(faces, points)


# ==============================================================================
# Point intersection tests
# ==============================================================================

@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
@pytest.mark.parametrize("mesh_type", MESH_TYPES)
def test_mesh_intersects_point_2d_hit(index_dtype, real_dtype, mesh_type):
    """Test point on face intersects"""
    if mesh_type == 'triangle':
        mesh = create_tiled_plane_2d_triangles(index_dtype, real_dtype)
        # Point in center of face 0 (triangle [0,1,4] = [0,0], [1,0], [1,1])
        # Centroid is approximately [2/3, 1/3]
        pt = tf.Point(np.array([0.5, 0.3], dtype=real_dtype))
    else:
        mesh = create_tiled_plane_2d_dynamic(index_dtype, real_dtype)
        # Point in center of face 0 (quad [0,1,4,3])
        pt = tf.Point(np.array([0.5, 0.5], dtype=real_dtype))

    assert tf.intersects(mesh, pt)
    assert tf.intersects(pt, mesh)


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
@pytest.mark.parametrize("mesh_type", MESH_TYPES)
def test_mesh_intersects_point_2d_miss(index_dtype, real_dtype, mesh_type):
    """Test point outside mesh doesn't intersect"""
    if mesh_type == 'triangle':
        mesh = create_tiled_plane_2d_triangles(index_dtype, real_dtype)
    else:
        mesh = create_tiled_plane_2d_dynamic(index_dtype, real_dtype)

    # Point outside mesh bounds
    pt = tf.Point(np.array([5.0, 5.0], dtype=real_dtype))

    assert not tf.intersects(mesh, pt)
    assert not tf.intersects(pt, mesh)


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
@pytest.mark.parametrize("mesh_type", MESH_TYPES)
def test_mesh_intersects_point_3d_hit(index_dtype, real_dtype, mesh_type):
    """Test point on face intersects (3D)"""
    if mesh_type == 'triangle':
        mesh = create_tiled_plane_3d_triangles(index_dtype, real_dtype)
        pt = tf.Point(np.array([0.5, 0.3, 0.0], dtype=real_dtype))
    else:
        mesh = create_tiled_plane_3d_dynamic(index_dtype, real_dtype)
        pt = tf.Point(np.array([0.5, 0.5, 0.0], dtype=real_dtype))

    assert tf.intersects(mesh, pt)
    assert tf.intersects(pt, mesh)


# ==============================================================================
# Segment intersection tests
# ==============================================================================

@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
@pytest.mark.parametrize("mesh_type", MESH_TYPES)
def test_mesh_intersects_segment_2d_hit(index_dtype, real_dtype, mesh_type):
    """Test segment crossing face intersects"""
    if mesh_type == 'triangle':
        mesh = create_tiled_plane_2d_triangles(index_dtype, real_dtype)
    else:
        mesh = create_tiled_plane_2d_dynamic(index_dtype, real_dtype)

    # Segment crossing the first cell diagonally
    seg = tf.Segment(np.array([[0.2, 0.2], [0.8, 0.8]], dtype=real_dtype))

    assert tf.intersects(mesh, seg)
    assert tf.intersects(seg, mesh)


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
@pytest.mark.parametrize("mesh_type", MESH_TYPES)
def test_mesh_intersects_segment_2d_miss(index_dtype, real_dtype, mesh_type):
    """Test segment outside mesh doesn't intersect"""
    if mesh_type == 'triangle':
        mesh = create_tiled_plane_2d_triangles(index_dtype, real_dtype)
    else:
        mesh = create_tiled_plane_2d_dynamic(index_dtype, real_dtype)

    # Segment outside mesh bounds
    seg = tf.Segment(np.array([[5.0, 5.0], [6.0, 6.0]], dtype=real_dtype))

    assert not tf.intersects(mesh, seg)
    assert not tf.intersects(seg, mesh)


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
@pytest.mark.parametrize("mesh_type", MESH_TYPES)
def test_mesh_intersects_segment_3d_hit(index_dtype, real_dtype, mesh_type):
    """Test segment crossing face intersects (3D)"""
    if mesh_type == 'triangle':
        mesh = create_tiled_plane_3d_triangles(index_dtype, real_dtype)
    else:
        mesh = create_tiled_plane_3d_dynamic(index_dtype, real_dtype)

    # Segment piercing the plane at z=0
    seg = tf.Segment(np.array([[0.5, 0.5, -1.0], [0.5, 0.5, 1.0]], dtype=real_dtype))

    assert tf.intersects(mesh, seg)
    assert tf.intersects(seg, mesh)


# ==============================================================================
# Polygon intersection tests
# ==============================================================================

@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
@pytest.mark.parametrize("mesh_type", MESH_TYPES)
def test_mesh_intersects_polygon_2d_hit(index_dtype, real_dtype, mesh_type):
    """Test polygon overlapping face intersects"""
    if mesh_type == 'triangle':
        mesh = create_tiled_plane_2d_triangles(index_dtype, real_dtype)
    else:
        mesh = create_tiled_plane_2d_dynamic(index_dtype, real_dtype)

    # Small triangle inside first cell
    poly = tf.Polygon(np.array([[0.3, 0.3], [0.7, 0.3], [0.5, 0.7]], dtype=real_dtype))

    assert tf.intersects(mesh, poly)
    assert tf.intersects(poly, mesh)


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
@pytest.mark.parametrize("mesh_type", MESH_TYPES)
def test_mesh_intersects_polygon_2d_miss(index_dtype, real_dtype, mesh_type):
    """Test polygon outside mesh doesn't intersect"""
    if mesh_type == 'triangle':
        mesh = create_tiled_plane_2d_triangles(index_dtype, real_dtype)
    else:
        mesh = create_tiled_plane_2d_dynamic(index_dtype, real_dtype)

    # Triangle outside mesh bounds
    poly = tf.Polygon(np.array([[5.0, 5.0], [6.0, 5.0], [5.5, 6.0]], dtype=real_dtype))

    assert not tf.intersects(mesh, poly)
    assert not tf.intersects(poly, mesh)


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
@pytest.mark.parametrize("mesh_type", MESH_TYPES)
def test_mesh_intersects_polygon_3d_hit(index_dtype, real_dtype, mesh_type):
    """Test polygon overlapping face intersects (3D)"""
    if mesh_type == 'triangle':
        mesh = create_tiled_plane_3d_triangles(index_dtype, real_dtype)
    else:
        mesh = create_tiled_plane_3d_dynamic(index_dtype, real_dtype)

    # Triangle in the plane z=0, inside first cell
    poly = tf.Polygon(np.array([[0.3, 0.3, 0.0], [0.7, 0.3, 0.0], [0.5, 0.7, 0.0]], dtype=real_dtype))

    assert tf.intersects(mesh, poly)
    assert tf.intersects(poly, mesh)


# ==============================================================================
# Ray intersection tests
# ==============================================================================

@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
@pytest.mark.parametrize("mesh_type", MESH_TYPES)
def test_mesh_intersects_ray_2d_hit(index_dtype, real_dtype, mesh_type):
    """Test ray hitting face intersects"""
    if mesh_type == 'triangle':
        mesh = create_tiled_plane_2d_triangles(index_dtype, real_dtype)
    else:
        mesh = create_tiled_plane_2d_dynamic(index_dtype, real_dtype)

    # Ray starting outside pointing into first cell
    ray = tf.Ray(
        origin=np.array([-1.0, 0.5], dtype=real_dtype),
        direction=np.array([1.0, 0.0], dtype=real_dtype)
    )

    assert tf.intersects(mesh, ray)
    assert tf.intersects(ray, mesh)


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
@pytest.mark.parametrize("mesh_type", MESH_TYPES)
def test_mesh_intersects_ray_2d_miss(index_dtype, real_dtype, mesh_type):
    """Test ray pointing away doesn't intersect"""
    if mesh_type == 'triangle':
        mesh = create_tiled_plane_2d_triangles(index_dtype, real_dtype)
    else:
        mesh = create_tiled_plane_2d_dynamic(index_dtype, real_dtype)

    # Ray starting outside pointing away
    ray = tf.Ray(
        origin=np.array([-1.0, 0.5], dtype=real_dtype),
        direction=np.array([-1.0, 0.0], dtype=real_dtype)
    )

    assert not tf.intersects(mesh, ray)
    assert not tf.intersects(ray, mesh)


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
@pytest.mark.parametrize("mesh_type", MESH_TYPES)
def test_mesh_intersects_ray_3d_hit(index_dtype, real_dtype, mesh_type):
    """Test ray hitting face intersects (3D)"""
    if mesh_type == 'triangle':
        mesh = create_tiled_plane_3d_triangles(index_dtype, real_dtype)
    else:
        mesh = create_tiled_plane_3d_dynamic(index_dtype, real_dtype)

    # Ray pointing down from above, toward first cell
    ray = tf.Ray(
        origin=np.array([0.5, 0.5, 1.0], dtype=real_dtype),
        direction=np.array([0.0, 0.0, -1.0], dtype=real_dtype)
    )

    assert tf.intersects(mesh, ray)
    assert tf.intersects(ray, mesh)


# ==============================================================================
# Line intersection tests
# ==============================================================================

@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
@pytest.mark.parametrize("mesh_type", MESH_TYPES)
def test_mesh_intersects_line_2d_hit(index_dtype, real_dtype, mesh_type):
    """Test line through face intersects"""
    if mesh_type == 'triangle':
        mesh = create_tiled_plane_2d_triangles(index_dtype, real_dtype)
    else:
        mesh = create_tiled_plane_2d_dynamic(index_dtype, real_dtype)

    # Horizontal line through first cell at y=0.5
    line = tf.Line(
        origin=np.array([0.5, 0.5], dtype=real_dtype),
        direction=np.array([1.0, 0.0], dtype=real_dtype)
    )

    assert tf.intersects(mesh, line)
    assert tf.intersects(line, mesh)


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
@pytest.mark.parametrize("mesh_type", MESH_TYPES)
def test_mesh_intersects_line_2d_miss(index_dtype, real_dtype, mesh_type):
    """Test line outside mesh doesn't intersect"""
    if mesh_type == 'triangle':
        mesh = create_tiled_plane_2d_triangles(index_dtype, real_dtype)
    else:
        mesh = create_tiled_plane_2d_dynamic(index_dtype, real_dtype)

    # Horizontal line above mesh at y=5
    line = tf.Line(
        origin=np.array([0.5, 5.0], dtype=real_dtype),
        direction=np.array([1.0, 0.0], dtype=real_dtype)
    )

    assert not tf.intersects(mesh, line)
    assert not tf.intersects(line, mesh)


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
@pytest.mark.parametrize("mesh_type", MESH_TYPES)
def test_mesh_intersects_line_3d_hit(index_dtype, real_dtype, mesh_type):
    """Test line through face intersects (3D)"""
    if mesh_type == 'triangle':
        mesh = create_tiled_plane_3d_triangles(index_dtype, real_dtype)
    else:
        mesh = create_tiled_plane_3d_dynamic(index_dtype, real_dtype)

    # Vertical line (in z) through first cell at [0.5, 0.5, z]
    line = tf.Line(
        origin=np.array([0.5, 0.5, 0.0], dtype=real_dtype),
        direction=np.array([0.0, 0.0, 1.0], dtype=real_dtype)
    )

    assert tf.intersects(mesh, line)
    assert tf.intersects(line, mesh)


# ==============================================================================
# Plane intersection tests (3D only)
# ==============================================================================

@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
@pytest.mark.parametrize("mesh_type", MESH_TYPES)
def test_mesh_intersects_plane_3d_hit(index_dtype, real_dtype, mesh_type):
    """Test plane intersecting mesh"""
    if mesh_type == 'triangle':
        mesh = create_tiled_plane_3d_triangles(index_dtype, real_dtype)
    else:
        mesh = create_tiled_plane_3d_dynamic(index_dtype, real_dtype)

    # Plane at z=0 (mesh is at z=0, should intersect)
    plane = tf.Plane(
        normal=np.array([0.0, 0.0, 1.0], dtype=real_dtype),
        origin=np.array([0.0, 0.0, 0.0], dtype=real_dtype)
    )

    assert tf.intersects(mesh, plane)
    assert tf.intersects(plane, mesh)


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
@pytest.mark.parametrize("mesh_type", MESH_TYPES)
def test_mesh_intersects_plane_3d_miss(index_dtype, real_dtype, mesh_type):
    """Test plane not intersecting mesh"""
    if mesh_type == 'triangle':
        mesh = create_tiled_plane_3d_triangles(index_dtype, real_dtype)
    else:
        mesh = create_tiled_plane_3d_dynamic(index_dtype, real_dtype)

    # Plane at z=2 (mesh is at z=0, should not intersect)
    plane = tf.Plane(
        normal=np.array([0.0, 0.0, 1.0], dtype=real_dtype),
        origin=np.array([0.0, 0.0, 2.0], dtype=real_dtype)
    )

    assert not tf.intersects(mesh, plane)
    assert not tf.intersects(plane, mesh)


# ==============================================================================
# Transformation tests
# ==============================================================================

@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
@pytest.mark.parametrize("mesh_type", MESH_TYPES)
def test_mesh_intersects_with_transformation_2d(index_dtype, real_dtype, mesh_type):
    """Test intersects with transformation (transform both form and primitive)"""
    if mesh_type == 'triangle':
        mesh = create_tiled_plane_2d_triangles(index_dtype, real_dtype)
        pt = tf.Point(np.array([0.5, 0.3], dtype=real_dtype))
    else:
        mesh = create_tiled_plane_2d_dynamic(index_dtype, real_dtype)
        pt = tf.Point(np.array([0.5, 0.5], dtype=real_dtype))

    # Verify intersection before transformation
    assert tf.intersects(mesh, pt)

    # Apply transformation: translate by [5, 3]
    transformation = np.array([
        [1, 0, 5],
        [0, 1, 3],
        [0, 0, 1]
    ], dtype=real_dtype)

    mesh.transformation = transformation

    # Transform the point with same transformation
    pt_homogeneous = np.hstack([pt.data, [1.0]]).astype(real_dtype)
    pt_transformed_coords = (transformation @ pt_homogeneous)[:2]
    pt_transformed = tf.Point(pt_transformed_coords)

    # Should still intersect
    assert tf.intersects(mesh, pt_transformed)
    assert tf.intersects(pt_transformed, mesh)


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
@pytest.mark.parametrize("mesh_type", MESH_TYPES)
def test_mesh_intersects_with_transformation_3d(index_dtype, real_dtype, mesh_type):
    """Test intersects with transformation (3D)"""
    if mesh_type == 'triangle':
        mesh = create_tiled_plane_3d_triangles(index_dtype, real_dtype)
    else:
        mesh = create_tiled_plane_3d_dynamic(index_dtype, real_dtype)

    # Segment that intersects untransformed mesh
    seg = tf.Segment(np.array([[0.5, 0.5, -1.0], [0.5, 0.5, 1.0]], dtype=real_dtype))
    assert tf.intersects(mesh, seg)

    # Apply transformation: translate by [10, 5, 2]
    transformation = np.array([
        [1, 0, 0, 10],
        [0, 1, 0,  5],
        [0, 0, 1,  2],
        [0, 0, 0,  1]
    ], dtype=real_dtype)

    mesh.transformation = transformation

    # Transform the segment with same transformation
    seg_data = np.hstack([seg.data, np.ones((2, 1), dtype=real_dtype)])
    seg_transformed_data = (transformation @ seg_data.T).T[:, :3]
    seg_transformed = tf.Segment(seg_transformed_data)

    # Should still intersect
    assert tf.intersects(mesh, seg_transformed)
    assert tf.intersects(seg_transformed, mesh)


# ==============================================================================
# Dimension mismatch tests
# ==============================================================================

def test_mesh_intersects_dimension_mismatch():
    """Test that dimension mismatch raises error"""
    mesh_2d = create_tiled_plane_2d_triangles(np.int32, np.float32)
    pt_3d = tf.Point(np.array([0.5, 0.5, 0.0], dtype=np.float32))

    with pytest.raises(ValueError, match="Dimension mismatch"):
        tf.intersects(mesh_2d, pt_3d)

    with pytest.raises(ValueError, match="Dimension mismatch"):
        tf.intersects(pt_3d, mesh_2d)


if __name__ == "__main__":
    # Run tests with verbose output
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
