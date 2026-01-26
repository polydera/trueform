"""
Tests for gather_ids functionality

Copyright (c) 2025 Žiga Sajovic, XLAB
"""
import numpy as np
import pytest
import trueform as tf


# Type combinations
INDEX_DTYPES = [np.int32, np.int64]
REAL_DTYPES = [np.float32, np.float64]
DIMS = [2, 3]


# ==============================================================================
# Mesh gather_ids tests
# ==============================================================================

def create_simple_mesh_2d(index_dtype, real_dtype):
    """
    Create a simple 2D mesh with 2 triangles
    Points:
      2---3
      |  /|
      | / |
      |/  |
      0---1
    Faces:
      Face 0: [0, 1, 2]
      Face 1: [1, 3, 2]
    """
    faces = np.array([[0, 1, 2], [1, 3, 2]], dtype=index_dtype)
    points = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=real_dtype)
    return tf.Mesh(faces, points)


def create_simple_mesh_3d(index_dtype, real_dtype):
    """
    Create a simple 3D mesh with 2 triangles at z=0
    """
    faces = np.array([[0, 1, 2], [1, 3, 2]], dtype=index_dtype)
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=real_dtype)
    return tf.Mesh(faces, points)


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_mesh_gather_ids_point_intersects_2d(index_dtype, real_dtype):
    """Test mesh gather_ids with point (intersects predicate) in 2D"""
    mesh = create_simple_mesh_2d(index_dtype, real_dtype)

    # Point inside face 0
    pt = tf.Point(np.array([0.3, 0.3], dtype=real_dtype))
    ids = tf.gather_intersecting_ids(mesh, pt)
    assert len(ids) == 1, "Point should intersect 1 face"
    assert 0 in ids, "Should intersect face 0"

    # Point inside face 1
    pt = tf.Point(np.array([0.7, 0.7], dtype=real_dtype))
    ids = tf.gather_intersecting_ids(mesh, pt)
    assert len(ids) == 1, "Point should intersect 1 face"
    assert 1 in ids, "Should intersect face 1"

    # Point outside mesh
    pt = tf.Point(np.array([2.0, 2.0], dtype=real_dtype))
    ids = tf.gather_intersecting_ids(mesh, pt)
    assert len(ids) == 0, "Point outside should intersect no faces"

    # Test swap order
    pt = tf.Point(np.array([0.3, 0.3], dtype=real_dtype))
    ids = tf.gather_intersecting_ids(pt, mesh)
    assert len(ids) == 1, "Swapped order should work"
    assert 0 in ids, "Should intersect face 0"


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_mesh_gather_ids_point_within_distance_2d(index_dtype, real_dtype):
    """Test mesh gather_ids with point (within_distance predicate) in 2D"""
    mesh = create_simple_mesh_2d(index_dtype, real_dtype)

    # Point close to mesh
    pt = tf.Point(np.array([0.5, -0.1], dtype=real_dtype))
    ids = tf.gather_ids_within_distance(mesh, pt, distance=0.2)
    assert len(ids) > 0, "Point should be within distance of some faces"

    # Point far from mesh
    pt = tf.Point(np.array([10.0, 10.0], dtype=real_dtype))
    ids = tf.gather_ids_within_distance(mesh, pt, distance=0.5)
    assert len(ids) == 0, "Far point should not be within distance"


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_mesh_gather_ids_segment_intersects_2d(index_dtype, real_dtype):
    """Test mesh gather_ids with segment (intersects predicate) in 2D"""
    mesh = create_simple_mesh_2d(index_dtype, real_dtype)

    # Segment crossing through face 0
    seg = tf.Segment(np.array([[0.2, 0.1], [0.4, 0.3]], dtype=real_dtype))
    ids = tf.gather_intersecting_ids(mesh, seg)
    assert len(ids) >= 1, "Segment should intersect at least 1 face"
    assert 0 in ids, "Should intersect face 0"

    # Segment far from mesh
    seg = tf.Segment(np.array([[10.0, 10.0], [11.0, 11.0]], dtype=real_dtype))
    ids = tf.gather_intersecting_ids(mesh, seg)
    assert len(ids) == 0, "Far segment should not intersect"


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_mesh_gather_ids_polygon_intersects_2d(index_dtype, real_dtype):
    """Test mesh gather_ids with polygon (intersects predicate) in 2D"""
    mesh = create_simple_mesh_2d(index_dtype, real_dtype)

    # Small square overlapping face 0
    poly = tf.Polygon(np.array([[0.2, 0.2], [0.4, 0.2], [0.4, 0.4], [0.2, 0.4]], dtype=real_dtype))
    ids = tf.gather_intersecting_ids(mesh, poly)
    assert len(ids) >= 1, "Polygon should intersect at least 1 face"
    assert 0 in ids, "Should intersect face 0"


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_mesh_gather_ids_ray_intersects_3d(index_dtype, real_dtype):
    """Test mesh gather_ids with ray (intersects predicate) in 3D"""
    mesh = create_simple_mesh_3d(index_dtype, real_dtype)

    # Ray shooting down at face 0
    ray = tf.Ray(
        origin=np.array([0.3, 0.3, 1.0], dtype=real_dtype),
        direction=np.array([0.0, 0.0, -1.0], dtype=real_dtype)
    )
    ids = tf.gather_intersecting_ids(mesh, ray)
    assert len(ids) == 1, "Ray should intersect 1 face"
    assert 0 in ids, "Should intersect face 0"

    # Ray missing mesh
    ray = tf.Ray(
        origin=np.array([5.0, 5.0, 1.0], dtype=real_dtype),
        direction=np.array([0.0, 0.0, -1.0], dtype=real_dtype)
    )
    ids = tf.gather_intersecting_ids(mesh, ray)
    assert len(ids) == 0, "Ray should not intersect any face"


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_mesh_gather_ids_line_intersects_2d(index_dtype, real_dtype):
    """Test mesh gather_ids with line (intersects predicate) in 2D"""
    mesh = create_simple_mesh_2d(index_dtype, real_dtype)

    # Line crossing through mesh
    line = tf.Line(
        origin=np.array([0.5, -1.0], dtype=real_dtype),
        direction=np.array([0.0, 1.0], dtype=real_dtype)
    )
    ids = tf.gather_intersecting_ids(mesh, line)
    assert len(ids) >= 1, "Line should intersect at least 1 face"

    # Line missing mesh
    line = tf.Line(
        origin=np.array([5.0, 0.0], dtype=real_dtype),
        direction=np.array([0.0, 1.0], dtype=real_dtype)
    )
    ids = tf.gather_intersecting_ids(mesh, line)
    assert len(ids) == 0, "Line should not intersect any face"


# ==============================================================================
# EdgeMesh gather_ids tests
# ==============================================================================

def create_simple_edge_mesh_2d(index_dtype, real_dtype):
    """
    Create a simple 2D edge mesh
    Points: 0---1---2
    Edges:
      Edge 0: [0, 1]
      Edge 1: [1, 2]
    """
    edges = np.array([[0, 1], [1, 2]], dtype=index_dtype)
    points = np.array([[0, 0], [1, 0], [2, 0]], dtype=real_dtype)
    return tf.EdgeMesh(edges, points)


def create_simple_edge_mesh_3d(index_dtype, real_dtype):
    """Create a simple 3D edge mesh"""
    edges = np.array([[0, 1], [1, 2]], dtype=index_dtype)
    points = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=real_dtype)
    return tf.EdgeMesh(edges, points)


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_edge_mesh_gather_ids_point_intersects_2d(index_dtype, real_dtype):
    """Test edge mesh gather_ids with point (intersects predicate) in 2D"""
    edge_mesh = create_simple_edge_mesh_2d(index_dtype, real_dtype)

    # Point on edge 0
    pt = tf.Point(np.array([0.5, 0.0], dtype=real_dtype))
    ids = tf.gather_intersecting_ids(edge_mesh, pt)
    assert len(ids) == 1, "Point should intersect 1 edge"
    assert 0 in ids, "Should intersect edge 0"

    # Point on edge 1
    pt = tf.Point(np.array([1.5, 0.0], dtype=real_dtype))
    ids = tf.gather_intersecting_ids(edge_mesh, pt)
    assert len(ids) == 1, "Point should intersect 1 edge"
    assert 1 in ids, "Should intersect edge 1"

    # Point off edges
    pt = tf.Point(np.array([0.5, 1.0], dtype=real_dtype))
    ids = tf.gather_intersecting_ids(edge_mesh, pt)
    assert len(ids) == 0, "Point off edges should not intersect"


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_edge_mesh_gather_ids_point_within_distance_2d(index_dtype, real_dtype):
    """Test edge mesh gather_ids with point (within_distance predicate) in 2D"""
    edge_mesh = create_simple_edge_mesh_2d(index_dtype, real_dtype)

    # Point close to edge 0
    pt = tf.Point(np.array([0.5, 0.1], dtype=real_dtype))
    ids = tf.gather_ids_within_distance(edge_mesh, pt, distance=0.2)
    assert len(ids) >= 1, "Point should be within distance of at least 1 edge"
    assert 0 in ids, "Should be close to edge 0"

    # Point far from all edges
    pt = tf.Point(np.array([10.0, 10.0], dtype=real_dtype))
    ids = tf.gather_ids_within_distance(edge_mesh, pt, distance=0.5)
    assert len(ids) == 0, "Far point should not be within distance"


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_edge_mesh_gather_ids_segment_intersects_2d(index_dtype, real_dtype):
    """Test edge mesh gather_ids with segment (intersects predicate) in 2D"""
    edge_mesh = create_simple_edge_mesh_2d(index_dtype, real_dtype)

    # Segment crossing edge 0
    seg = tf.Segment(np.array([[0.5, -0.5], [0.5, 0.5]], dtype=real_dtype))
    ids = tf.gather_intersecting_ids(edge_mesh, seg)
    assert len(ids) == 1, "Segment should intersect 1 edge"
    assert 0 in ids, "Should intersect edge 0"


# ==============================================================================
# PointCloud gather_ids tests
# ==============================================================================

def create_simple_point_cloud_2d(real_dtype):
    """
    Create a simple 2D point cloud
    Points: [0, 0], [1, 0], [0, 1], [1, 1]
    """
    points = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=real_dtype)
    return tf.PointCloud(points)


def create_simple_point_cloud_3d(real_dtype):
    """Create a simple 3D point cloud"""
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=real_dtype)
    return tf.PointCloud(points)


@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_point_cloud_gather_ids_point_intersects_2d(real_dtype):
    """Test point cloud gather_ids with point (intersects predicate) in 2D"""
    pc = create_simple_point_cloud_2d(real_dtype)

    # Point coinciding with point 0
    pt = tf.Point(np.array([0.0, 0.0], dtype=real_dtype))
    ids = tf.gather_intersecting_ids(pc, pt)
    assert len(ids) == 1, "Point should intersect 1 point"
    assert 0 in ids, "Should intersect point 0"

    # Point not coinciding with any point
    pt = tf.Point(np.array([0.5, 0.5], dtype=real_dtype))
    ids = tf.gather_intersecting_ids(pc, pt)
    assert len(ids) == 0, "Non-coinciding point should not intersect"


@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_point_cloud_gather_ids_point_within_distance_2d(real_dtype):
    """Test point cloud gather_ids with point (within_distance predicate) in 2D"""
    pc = create_simple_point_cloud_2d(real_dtype)

    # Point close to point 0
    pt = tf.Point(np.array([0.1, 0.1], dtype=real_dtype))
    ids = tf.gather_ids_within_distance(pc, pt, distance=0.2)
    assert len(ids) >= 1, "Point should be within distance of at least 1 point"
    assert 0 in ids, "Should be close to point 0"

    # Point in center, close to all 4 points
    pt = tf.Point(np.array([0.5, 0.5], dtype=real_dtype))
    ids = tf.gather_ids_within_distance(pc, pt, distance=1.0)
    assert len(ids) == 4, "Should be close to all 4 points"


@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_point_cloud_gather_ids_segment_within_distance_2d(real_dtype):
    """Test point cloud gather_ids with segment (within_distance predicate) in 2D"""
    pc = create_simple_point_cloud_2d(real_dtype)

    # Segment close to point 0 and 1
    seg = tf.Segment(np.array([[0.0, 0.0], [1.0, 0.0]], dtype=real_dtype))
    ids = tf.gather_ids_within_distance(pc, seg, distance=0.1)
    assert len(ids) >= 2, "Segment should be close to at least 2 points"
    assert 0 in ids and 1 in ids, "Should be close to points 0 and 1"


# ==============================================================================
# Validation tests
# ==============================================================================

def test_gather_ids_dimension_mismatch():
    """Test that dimension mismatch raises an error"""
    mesh_2d = create_simple_mesh_2d(np.int32, np.float32)
    pt_3d = tf.Point([0.0, 0.0, 0.0])

    with pytest.raises(ValueError, match="Dimension mismatch"):
        tf.gather_intersecting_ids(mesh_2d, pt_3d)


def test_gather_ids_missing_threshold():
    """Test that missing threshold for within_distance raises an error"""
    mesh = create_simple_mesh_2d(np.int32, np.float32)
    pt = tf.Point([0.5, 0.5])

    with pytest.raises(ValueError, match="distance is required"):
        tf.gather_ids_within_distance(mesh, pt, distance=None)


def test_gather_ids_invalid_form_type():
    """Test that invalid form type raises an error"""
    pt1 = tf.Point([0.0, 0.0])
    pt2 = tf.Point([1.0, 1.0])

    with pytest.raises(TypeError):
        tf.gather_intersecting_ids(pt1, pt2)


def test_gather_ids_return_dtype():
    """Test that gather_ids returns correct dtype based on form index type"""
    # Mesh with int32 indices
    mesh_int32 = create_simple_mesh_2d(np.int32, np.float32)
    pt = tf.Point([0.3, 0.3])
    ids = tf.gather_intersecting_ids(mesh_int32, pt)
    assert ids.dtype == np.int32, "Should return int32 for int32-indexed mesh"

    # Mesh with int64 indices
    mesh_int64 = create_simple_mesh_2d(np.int64, np.float32)
    ids = tf.gather_intersecting_ids(mesh_int64, pt)
    assert ids.dtype == np.int64, "Should return int64 for int64-indexed mesh"


# ==============================================================================
# Main runner
# ==============================================================================

if __name__ == "__main__":
    import sys
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
