"""
Tests for edge_mesh neighbor_search functionality

Uses simple straight polylines where we know exactly which edge is closest.

Copyright (c) 2025 Žiga Sajovic, XLAB
"""

import sys
import numpy as np
import pytest
import trueform as tf

# Type combinations to test
INDEX_DTYPES = [np.int32, np.int64]
REAL_DTYPES = [np.float32, np.float64]


def create_2d_straight_polyline(index_dtype, real_dtype):
    """
    Create a simple 2D straight polyline along x-axis.

    Edges: [0,1], [1,2], [2,3]
    Points: [0,0], [1,0], [2,0], [3,0]

    Edge 0: [0,0] -> [1,0]
    Edge 1: [1,0] -> [2,0]
    Edge 2: [2,0] -> [3,0]
    """
    edges = np.array([[0, 1], [1, 2], [2, 3]], dtype=index_dtype)
    points = np.array([[0, 0], [1, 0], [2, 0], [3, 0]], dtype=real_dtype)
    return tf.EdgeMesh(edges, points)


def create_3d_straight_polyline(index_dtype, real_dtype):
    """
    Create a simple 3D straight polyline along x-axis.

    Edges: [0,1], [1,2], [2,3]
    Points: [0,0,0], [1,0,0], [2,0,0], [3,0,0]

    Edge 0: [0,0,0] -> [1,0,0]
    Edge 1: [1,0,0] -> [2,0,0]
    Edge 2: [2,0,0] -> [3,0,0]
    """
    edges = np.array([[0, 1], [1, 2], [2, 3]], dtype=index_dtype)
    points = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]], dtype=real_dtype)
    return tf.EdgeMesh(edges, points)


# ==============================================================================
# Point queries
# ==============================================================================

@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_edge_mesh_neighbor_search_point_2d(index_dtype, real_dtype):
    """Test neighbor search with point queries in 2D"""
    edge_mesh = create_2d_straight_polyline(index_dtype, real_dtype)

    # Query point above the middle of edge 0 [0,0]->[1,0]
    query = tf.Point(np.array([0.5, 1.0], dtype=real_dtype))
    edge_idx, dist_sq, closest_pt = tf.neighbor_search(edge_mesh, query)

    assert edge_idx == 0, f"Should be closest to edge 0, got {edge_idx}"
    assert np.isclose(dist_sq, 1.0, atol=1e-5), f"Distance² should be 1.0, got {dist_sq}"
    assert np.allclose(closest_pt, [0.5, 0.0], atol=1e-5), f"Closest point should be [0.5, 0.0], got {closest_pt}"

    # Query point above the middle of edge 1 [1,0]->[2,0]
    query = tf.Point(np.array([1.5, 2.0], dtype=real_dtype))
    edge_idx, dist_sq, closest_pt = tf.neighbor_search(edge_mesh, query)

    assert edge_idx == 1, f"Should be closest to edge 1, got {edge_idx}"
    assert np.isclose(dist_sq, 4.0, atol=1e-5), f"Distance² should be 4.0, got {dist_sq}"
    assert np.allclose(closest_pt, [1.5, 0.0], atol=1e-5), f"Closest point should be [1.5, 0.0], got {closest_pt}"

    # Query point above the middle of edge 2 [2,0]->[3,0]
    query = tf.Point(np.array([2.5, 0.5], dtype=real_dtype))
    edge_idx, dist_sq, closest_pt = tf.neighbor_search(edge_mesh, query)

    assert edge_idx == 2, f"Should be closest to edge 2, got {edge_idx}"
    assert np.isclose(dist_sq, 0.25, atol=1e-5), f"Distance² should be 0.25, got {dist_sq}"
    assert np.allclose(closest_pt, [2.5, 0.0], atol=1e-5), f"Closest point should be [2.5, 0.0], got {closest_pt}"


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_edge_mesh_neighbor_search_point_3d(index_dtype, real_dtype):
    """Test neighbor search with point queries in 3D"""
    edge_mesh = create_3d_straight_polyline(index_dtype, real_dtype)

    # Query point above the middle of edge 0
    query = tf.Point(np.array([0.5, 1.0, 0.0], dtype=real_dtype))
    edge_idx, dist_sq, closest_pt = tf.neighbor_search(edge_mesh, query)

    assert edge_idx == 0, f"Should be closest to edge 0, got {edge_idx}"
    assert np.isclose(dist_sq, 1.0, atol=1e-5), f"Distance² should be 1.0, got {dist_sq}"
    assert np.allclose(closest_pt, [0.5, 0.0, 0.0], atol=1e-5)

    # Query point above the middle of edge 1
    query = tf.Point(np.array([1.5, 0.0, 2.0], dtype=real_dtype))
    edge_idx, dist_sq, closest_pt = tf.neighbor_search(edge_mesh, query)

    assert edge_idx == 1, f"Should be closest to edge 1, got {edge_idx}"
    assert np.isclose(dist_sq, 4.0, atol=1e-5), f"Distance² should be 4.0, got {dist_sq}"
    assert np.allclose(closest_pt, [1.5, 0.0, 0.0], atol=1e-5)

    # Query point above the middle of edge 2
    query = tf.Point(np.array([2.5, 1.0, 1.0], dtype=real_dtype))
    edge_idx, dist_sq, closest_pt = tf.neighbor_search(edge_mesh, query)

    assert edge_idx == 2, f"Should be closest to edge 2, got {edge_idx}"
    assert np.isclose(dist_sq, 2.0, atol=1e-5), f"Distance² should be 2.0 (1²+1²), got {dist_sq}"
    assert np.allclose(closest_pt, [2.5, 0.0, 0.0], atol=1e-5)


# ==============================================================================
# Segment queries
# ==============================================================================

@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_edge_mesh_neighbor_search_segment_2d(index_dtype, real_dtype):
    """Test neighbor search with segment queries in 2D"""
    edge_mesh = create_2d_straight_polyline(index_dtype, real_dtype)

    # Segment parallel to edge 0, offset above
    query = tf.Segment(np.array([[0.2, 1.0], [0.8, 1.0]], dtype=real_dtype))
    edge_idx, dist_sq, closest_pt = tf.neighbor_search(edge_mesh, query)

    assert edge_idx == 0, f"Should be closest to edge 0, got {edge_idx}"
    assert np.isclose(dist_sq, 1.0, atol=1e-5), f"Distance² should be 1.0, got {dist_sq}"
    # Closest point should be on edge 0 (at y=0)
    assert 0.2 <= closest_pt[0] <= 0.8, f"x should be in [0.2, 0.8], got {closest_pt[0]}"
    assert np.isclose(closest_pt[1], 0.0, atol=1e-5), f"y should be 0.0 (on edge), got {closest_pt[1]}"


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_edge_mesh_neighbor_search_segment_3d(index_dtype, real_dtype):
    """Test neighbor search with segment queries in 3D"""
    edge_mesh = create_3d_straight_polyline(index_dtype, real_dtype)

    # Segment parallel to edge 1, offset above
    query = tf.Segment(np.array([[1.2, 1.0, 0.0], [1.8, 1.0, 0.0]], dtype=real_dtype))
    edge_idx, dist_sq, closest_pt = tf.neighbor_search(edge_mesh, query)

    assert edge_idx == 1, f"Should be closest to edge 1, got {edge_idx}"
    assert np.isclose(dist_sq, 1.0, atol=1e-5), f"Distance² should be 1.0, got {dist_sq}"
    # Closest point should be on edge 1 (at y=0, z=0)
    assert 1.2 <= closest_pt[0] <= 1.8, f"x should be in [1.2, 1.8], got {closest_pt[0]}"
    assert np.isclose(closest_pt[1], 0.0, atol=1e-5), f"y should be 0.0 (on edge), got {closest_pt[1]}"
    assert np.isclose(closest_pt[2], 0.0, atol=1e-5), f"z should be 0.0 (on edge), got {closest_pt[2]}"


# ==============================================================================
# Polygon queries
# ==============================================================================

@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_edge_mesh_neighbor_search_polygon_2d(index_dtype, real_dtype):
    """Test neighbor search with polygon queries in 2D"""
    edge_mesh = create_2d_straight_polyline(index_dtype, real_dtype)

    # Triangle above edge 0
    query = tf.Polygon(np.array([[0.3, 0.5], [0.7, 0.5], [0.5, 0.8]], dtype=real_dtype))
    edge_idx, dist_sq, closest_pt = tf.neighbor_search(edge_mesh, query)

    assert edge_idx == 0, f"Should be closest to edge 0, got {edge_idx}"
    assert dist_sq >= 0.0, "Distance² should be non-negative"
    # Closest point should be on edge 0 (at y=0)
    assert 0.3 <= closest_pt[0] <= 0.7, f"x should be in [0.3, 0.7], got {closest_pt[0]}"
    assert np.isclose(closest_pt[1], 0.0, atol=1e-5), f"y should be 0.0 (on edge), got {closest_pt[1]}"


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_edge_mesh_neighbor_search_polygon_3d(index_dtype, real_dtype):
    """Test neighbor search with polygon queries in 3D"""
    edge_mesh = create_3d_straight_polyline(index_dtype, real_dtype)

    # Triangle above edge 2
    query = tf.Polygon(np.array([[2.3, 0.5, 0.0], [2.7, 0.5, 0.0], [2.5, 0.8, 0.0]], dtype=real_dtype))
    edge_idx, dist_sq, closest_pt = tf.neighbor_search(edge_mesh, query)

    assert edge_idx == 2, f"Should be closest to edge 2, got {edge_idx}"
    assert dist_sq >= 0.0, "Distance² should be non-negative"
    # Closest point should be on edge 2 (at y=0, z=0)
    assert 2.3 - 1e-5 <= closest_pt[0] <= 2.7 + 1e-5, f"x should be in [2.3, 2.7], got {closest_pt[0]}"
    assert np.isclose(closest_pt[1], 0.0, atol=1e-5), f"y should be 0.0 (on edge), got {closest_pt[1]}"
    assert np.isclose(closest_pt[2], 0.0, atol=1e-5), f"z should be 0.0 (on edge), got {closest_pt[2]}"


# ==============================================================================
# Ray queries
# ==============================================================================

@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_edge_mesh_neighbor_search_ray_2d(index_dtype, real_dtype):
    """Test neighbor search with ray queries in 2D"""
    edge_mesh = create_2d_straight_polyline(index_dtype, real_dtype)

    # Ray starting above edge 1, pointing down
    query = tf.Ray(
        origin=np.array([1.5, 3.0], dtype=real_dtype),
        direction=np.array([0.0, -1.0], dtype=real_dtype)
    )
    edge_idx, dist_sq, closest_pt = tf.neighbor_search(edge_mesh, query)

    assert edge_idx == 1, f"Should be closest to edge 1, got {edge_idx}"
    # Ray passes through edge 1, so distance should be 0
    assert np.isclose(dist_sq, 0.0, atol=1e-5), f"Distance² should be 0.0, got {dist_sq}"
    # Closest point should be on edge 1 (the intersection point at y=0)
    assert np.allclose(closest_pt, [1.5, 0.0], atol=1e-5), f"Closest point should be [1.5, 0.0], got {closest_pt}"


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_edge_mesh_neighbor_search_ray_3d(index_dtype, real_dtype):
    """Test neighbor search with ray queries in 3D"""
    edge_mesh = create_3d_straight_polyline(index_dtype, real_dtype)

    # Ray starting to the side of edge 0, pointing parallel
    query = tf.Ray(
        origin=np.array([0.5, 1.0, 0.0], dtype=real_dtype),
        direction=np.array([1.0, 0.0, 0.0], dtype=real_dtype)
    )
    edge_idx, dist_sq, closest_pt = tf.neighbor_search(edge_mesh, query)

    assert edge_idx == 0, f"Should be closest to edge 0, got {edge_idx}"
    assert np.isclose(dist_sq, 1.0, atol=1e-5), f"Distance² should be 1.0, got {dist_sq}"
    # Closest point should be on edge 0 (at y=0, z=0)
    assert np.isclose(closest_pt[0], 0.5, atol=1e-5), f"x should be 0.5, got {closest_pt[0]}"
    assert np.isclose(closest_pt[1], 0.0, atol=1e-5), f"y should be 0.0 (on edge), got {closest_pt[1]}"
    assert np.isclose(closest_pt[2], 0.0, atol=1e-5), f"z should be 0.0 (on edge), got {closest_pt[2]}"


# ==============================================================================
# Line queries
# ==============================================================================

@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_edge_mesh_neighbor_search_line_2d(index_dtype, real_dtype):
    """Test neighbor search with line queries in 2D"""
    edge_mesh = create_2d_straight_polyline(index_dtype, real_dtype)

    # Line parallel to polyline, offset above
    # All edges have same distance since line is parallel, so any edge is valid
    query = tf.Line(
        origin=np.array([2.5, 0.5], dtype=real_dtype),
        direction=np.array([1.0, 0.0], dtype=real_dtype)
    )
    edge_idx, dist_sq, closest_pt = tf.neighbor_search(edge_mesh, query)

    # All edges have same distance (0.25), so any edge is acceptable
    assert edge_idx in [0, 1, 2], f"Should find one of the edges, got {edge_idx}"
    assert np.isclose(dist_sq, 0.25, atol=1e-5), f"Distance² should be 0.25, got {dist_sq}"
    # Closest point should be on the returned edge (at y=0)
    assert np.isclose(closest_pt[1], 0.0, atol=1e-5), f"y should be 0.0 (on edge), got {closest_pt[1]}"


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_edge_mesh_neighbor_search_line_3d(index_dtype, real_dtype):
    """Test neighbor search with line queries in 3D"""
    edge_mesh = create_3d_straight_polyline(index_dtype, real_dtype)

    # Line parallel to polyline, offset from edges
    # All edges have same distance since line is parallel, so any edge is valid
    query = tf.Line(
        origin=np.array([1.5, 1.0, 1.0], dtype=real_dtype),
        direction=np.array([1.0, 0.0, 0.0], dtype=real_dtype)
    )
    edge_idx, dist_sq, closest_pt = tf.neighbor_search(edge_mesh, query)

    # All edges have same distance (2.0), so any edge is acceptable
    assert edge_idx in [0, 1, 2], f"Should find one of the edges, got {edge_idx}"
    assert np.isclose(dist_sq, 2.0, atol=1e-5), f"Distance² should be 2.0, got {dist_sq}"
    # Closest point should be on the returned edge (at y=0, z=0)
    assert np.isclose(closest_pt[1], 0.0, atol=1e-5), f"y should be 0.0 (on edge), got {closest_pt[1]}"
    assert np.isclose(closest_pt[2], 0.0, atol=1e-5), f"z should be 0.0 (on edge), got {closest_pt[2]}"


# ==============================================================================
# Radius tests
# ==============================================================================

@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_edge_mesh_neighbor_search_with_radius_2d(index_dtype, real_dtype):
    """Test neighbor search with radius constraint in 2D"""
    edge_mesh = create_2d_straight_polyline(index_dtype, real_dtype)

    # Query point far from polyline
    query = tf.Point(np.array([0.5, 10.0], dtype=real_dtype))

    # With no radius constraint, should find edge 0
    edge_idx, dist_sq, closest_pt = tf.neighbor_search(edge_mesh, query)
    assert edge_idx == 0
    assert np.isclose(dist_sq, 100.0, atol=1e-5)

    # With small radius, should find nothing
    result = tf.neighbor_search(edge_mesh, query, radius=5.0)
    assert result is None, "Should return None when no edges within radius"

    # With large enough radius, should find edge 0
    edge_idx, dist_sq, closest_pt = tf.neighbor_search(edge_mesh, query, radius=15.0)
    assert edge_idx == 0
    assert np.isclose(dist_sq, 100.0, atol=1e-5)


# ==============================================================================
# Transformation tests
# ==============================================================================

@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_edge_mesh_neighbor_search_with_transformation_2d(real_dtype):
    """Test neighbor search with 2D transformation"""
    edge_mesh = create_2d_straight_polyline(np.int32, real_dtype)

    # Query point above edge 0 in untransformed space
    query_untransformed = tf.Point(np.array([0.5, 1.0], dtype=real_dtype))

    # Verify it finds edge 0 untransformed
    edge_idx, dist_sq, closest_pt = tf.neighbor_search(edge_mesh, query_untransformed)
    assert edge_idx == 0
    assert np.isclose(dist_sq, 1.0, atol=1e-5)

    # Apply transformation [translation by (10, 5)]
    transformation = np.array([
        [1, 0, 10],
        [0, 1, 5],
        [0, 0, 1]
    ], dtype=real_dtype)
    edge_mesh.transformation = transformation

    # Transform the query point
    query_homogeneous = np.array([query_untransformed.data[0], query_untransformed.data[1], 1.0], dtype=real_dtype)
    transformed_query_data = (transformation @ query_homogeneous)[:2]
    query_transformed = tf.Point(transformed_query_data)

    # Should still find edge 0 with same distance
    edge_idx, dist_sq, closest_pt = tf.neighbor_search(edge_mesh, query_transformed)
    assert edge_idx == 0, f"Should find edge 0, got {edge_idx}"
    assert np.isclose(dist_sq, 1.0, atol=1e-4), f"Distance² should still be 1.0, got {dist_sq}"


@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_edge_mesh_neighbor_search_with_transformation_3d(real_dtype):
    """Test neighbor search with 3D transformation"""
    edge_mesh = create_3d_straight_polyline(np.int32, real_dtype)

    # Query point above edge 1 in untransformed space
    query_untransformed = tf.Point(np.array([1.5, 2.0, 0.0], dtype=real_dtype))

    # Verify it finds edge 1 untransformed
    edge_idx, dist_sq, closest_pt = tf.neighbor_search(edge_mesh, query_untransformed)
    assert edge_idx == 1
    assert np.isclose(dist_sq, 4.0, atol=1e-5)

    # Apply transformation [translation by (5, 3, 2)]
    transformation = np.array([
        [1, 0, 0, 5],
        [0, 1, 0, 3],
        [0, 0, 1, 2],
        [0, 0, 0, 1]
    ], dtype=real_dtype)
    edge_mesh.transformation = transformation

    # Transform the query point
    query_homogeneous = np.array([
        query_untransformed.data[0],
        query_untransformed.data[1],
        query_untransformed.data[2],
        1.0
    ], dtype=real_dtype)
    transformed_query_data = (transformation @ query_homogeneous)[:3]
    query_transformed = tf.Point(transformed_query_data)

    # Should still find edge 1 with same distance
    edge_idx, dist_sq, closest_pt = tf.neighbor_search(edge_mesh, query_transformed)
    assert edge_idx == 1, f"Should find edge 1, got {edge_idx}"
    assert np.isclose(dist_sq, 4.0, atol=1e-4), f"Distance² should still be 4.0, got {dist_sq}"


# ==============================================================================
# Dimension/dtype mismatch tests
# ==============================================================================

def test_edge_mesh_neighbor_search_dimension_mismatch():
    """Test that dimension mismatch raises an error"""
    edge_mesh = create_2d_straight_polyline(np.int32, np.float32)  # 2D

    # 3D query
    query = tf.Point(np.array([0.5, 0.5, 2.0], dtype=np.float32))

    with pytest.raises(ValueError, match="Dimension mismatch"):
        tf.neighbor_search(edge_mesh, query)


if __name__ == "__main__":
    # Run tests with verbose output
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
