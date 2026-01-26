"""
Tests for EdgeMesh × EdgeMesh gather_ids

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/trueform
"""
import sys
import pytest
import numpy as np
import trueform as tf


# Type combinations for EdgeMesh: 2 index types × 2 real types × 2 dims = 8
INDEX_DTYPES = [np.int32, np.int64]
REAL_DTYPES = [np.float32, np.float64]


def create_straight_polyline_2d(index_dtype, real_dtype, offset_x=0.0):
    """
    Create a simple straight polyline in 2D
    offset_x shifts the polyline horizontally
    Points: [0+offset,0], [1+offset,0], [2+offset,0], [3+offset,0]
    Edges: [0,1], [1,2], [2,3]
    """
    edges = np.array([[0, 1], [1, 2], [2, 3]], dtype=index_dtype)
    points = np.array([[0 + offset_x, 0], [1 + offset_x, 0], [2 + offset_x, 0], [3 + offset_x, 0]], dtype=real_dtype)
    return tf.EdgeMesh(edges, points)


def create_straight_polyline_3d(index_dtype, real_dtype, offset_z=0.0):
    """
    Create a simple straight polyline in 3D (along x-axis, z=offset_z)
    Points: [0,0,offset_z], [1,0,offset_z], [2,0,offset_z], [3,0,offset_z]
    Edges: [0,1], [1,2], [2,3]
    """
    edges = np.array([[0, 1], [1, 2], [2, 3]], dtype=index_dtype)
    points = np.array([[0, 0, offset_z], [1, 0, offset_z], [2, 0, offset_z], [3, 0, offset_z]], dtype=real_dtype)
    return tf.EdgeMesh(edges, points)


# ==============================================================================
# 2D EdgeMesh-EdgeMesh gather_ids tests
# ==============================================================================

@pytest.mark.parametrize("index_dtype0", INDEX_DTYPES)
@pytest.mark.parametrize("index_dtype1", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_edge_mesh_gather_ids_edge_mesh_2d_intersects_hit(index_dtype0, index_dtype1, real_dtype):
    """Test overlapping edge meshes - find intersecting edges"""
    # edge_mesh0: edges 0,1,2 at x=0 to 3
    # edge_mesh1: edges 0,1,2 at x=1.5 to 4.5 (overlaps with edges 1,2 of mesh0)
    edge_mesh0 = create_straight_polyline_2d(index_dtype0, real_dtype, offset_x=0.0)
    edge_mesh1 = create_straight_polyline_2d(index_dtype1, real_dtype, offset_x=1.5)

    result = tf.gather_intersecting_ids(edge_mesh0, edge_mesh1)

    # Should return (N, 2) array with N > 0
    assert result.shape[1] == 2
    assert result.shape[0] > 0

    # Edges should overlap since both are horizontal along same line
    # Specific pairs depend on implementation details


@pytest.mark.parametrize("index_dtype0", INDEX_DTYPES)
@pytest.mark.parametrize("index_dtype1", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_edge_mesh_gather_ids_edge_mesh_2d_intersects_miss(index_dtype0, index_dtype1, real_dtype):
    """Test non-overlapping edge meshes - no intersections"""
    edge_mesh0 = create_straight_polyline_2d(index_dtype0, real_dtype, offset_x=0.0)
    edge_mesh1 = create_straight_polyline_2d(index_dtype1, real_dtype, offset_x=10.0)

    result = tf.gather_intersecting_ids(edge_mesh0, edge_mesh1)

    # Should return empty (0, 2) array
    assert result.shape == (0, 2)


@pytest.mark.parametrize("index_dtype0", INDEX_DTYPES)
@pytest.mark.parametrize("index_dtype1", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_edge_mesh_gather_ids_edge_mesh_2d_within_distance_hit(index_dtype0, index_dtype1, real_dtype):
    """Test edge meshes within distance threshold"""
    # Create two parallel horizontal polylines 0.5 units apart
    edges0 = np.array([[0, 1], [1, 2]], dtype=index_dtype0)
    points0 = np.array([[0, 0], [1, 0], [2, 0]], dtype=real_dtype)

    edges1 = np.array([[0, 1], [1, 2]], dtype=index_dtype1)
    points1 = np.array([[0, 0.5], [1, 0.5], [2, 0.5]], dtype=real_dtype)

    edge_mesh0 = tf.EdgeMesh(edges0, points0)
    edge_mesh1 = tf.EdgeMesh(edges1, points1)

    # With distance=1.0, should find all edge pairs
    result = tf.gather_ids_within_distance(edge_mesh0, edge_mesh1, distance=1.0)

    assert result.shape[1] == 2
    assert result.shape[0] > 0


@pytest.mark.parametrize("index_dtype0", INDEX_DTYPES)
@pytest.mark.parametrize("index_dtype1", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_edge_mesh_gather_ids_edge_mesh_2d_within_distance_miss(index_dtype0, index_dtype1, real_dtype):
    """Test edge meshes outside distance threshold"""
    edges0 = np.array([[0, 1]], dtype=index_dtype0)
    points0 = np.array([[0, 0], [1, 0]], dtype=real_dtype)

    edges1 = np.array([[0, 1]], dtype=index_dtype1)
    points1 = np.array([[0, 10], [1, 10]], dtype=real_dtype)

    edge_mesh0 = tf.EdgeMesh(edges0, points0)
    edge_mesh1 = tf.EdgeMesh(edges1, points1)

    # Distance is 10, distance=1.0 should miss
    result = tf.gather_ids_within_distance(edge_mesh0, edge_mesh1, distance=1.0)

    assert result.shape == (0, 2)


# ==============================================================================
# 3D EdgeMesh-EdgeMesh gather_ids tests
# ==============================================================================

@pytest.mark.parametrize("index_dtype0", INDEX_DTYPES)
@pytest.mark.parametrize("index_dtype1", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_edge_mesh_gather_ids_edge_mesh_3d_intersects_hit(index_dtype0, index_dtype1, real_dtype):
    """Test overlapping edge meshes intersect (3D)"""
    edge_mesh0 = create_straight_polyline_3d(index_dtype0, real_dtype, offset_z=0.0)
    edge_mesh1 = create_straight_polyline_3d(index_dtype1, real_dtype, offset_z=0.0)

    result = tf.gather_intersecting_ids(edge_mesh0, edge_mesh1)

    assert result.shape[1] == 2
    assert result.shape[0] > 0


@pytest.mark.parametrize("index_dtype0", INDEX_DTYPES)
@pytest.mark.parametrize("index_dtype1", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_edge_mesh_gather_ids_edge_mesh_3d_intersects_miss(index_dtype0, index_dtype1, real_dtype):
    """Test non-overlapping edge meshes don't intersect (3D)"""
    edge_mesh0 = create_straight_polyline_3d(index_dtype0, real_dtype, offset_z=0.0)
    edge_mesh1 = create_straight_polyline_3d(index_dtype1, real_dtype, offset_z=10.0)

    result = tf.gather_intersecting_ids(edge_mesh0, edge_mesh1)

    assert result.shape == (0, 2)


@pytest.mark.parametrize("index_dtype0", INDEX_DTYPES)
@pytest.mark.parametrize("index_dtype1", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_edge_mesh_gather_ids_edge_mesh_3d_within_distance(index_dtype0, index_dtype1, real_dtype):
    """Test 3D edge meshes within distance"""
    edges0 = np.array([[0, 1], [1, 2]], dtype=index_dtype0)
    points0 = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=real_dtype)

    edges1 = np.array([[0, 1], [1, 2]], dtype=index_dtype1)
    points1 = np.array([[0, 0, 0.5], [1, 0, 0.5], [2, 0, 0.5]], dtype=real_dtype)

    edge_mesh0 = tf.EdgeMesh(edges0, points0)
    edge_mesh1 = tf.EdgeMesh(edges1, points1)

    result = tf.gather_ids_within_distance(edge_mesh0, edge_mesh1, distance=1.0)

    assert result.shape[1] == 2
    assert result.shape[0] > 0


# ==============================================================================
# Symmetry and transformation tests
# ==============================================================================

@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_edge_mesh_gather_ids_edge_mesh_symmetry(real_dtype):
    """Test that swapping arguments swaps column order in results"""
    edges0 = np.array([[0, 1]], dtype=np.int32)
    points0 = np.array([[0, 0], [1, 0]], dtype=real_dtype)

    edges1 = np.array([[0, 1]], dtype=np.int32)
    points1 = np.array([[0, 0.3], [1, 0.3]], dtype=real_dtype)

    edge_mesh0 = tf.EdgeMesh(edges0, points0)
    edge_mesh1 = tf.EdgeMesh(edges1, points1)

    result01 = tf.gather_ids_within_distance(edge_mesh0, edge_mesh1, distance=1.0)
    result10 = tf.gather_ids_within_distance(edge_mesh1, edge_mesh0, distance=1.0)

    # Both should have same number of matches
    assert result01.shape[0] == result10.shape[0]

    # Columns should be swapped
    if result01.shape[0] > 0:
        for i in range(result01.shape[0]):
            # Find matching row in result10
            found = False
            for j in range(result10.shape[0]):
                if result01[i, 0] == result10[j, 1] and result01[i, 1] == result10[j, 0]:
                    found = True
                    break
            assert found, f"Row {result01[i]} not found with swapped columns in result10"


@pytest.mark.parametrize("index_dtype0", INDEX_DTYPES)
@pytest.mark.parametrize("index_dtype1", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_edge_mesh_gather_ids_edge_mesh_with_transformation_2d(index_dtype0, index_dtype1, real_dtype):
    """Test gather_ids with transformation"""
    edge_mesh0 = create_straight_polyline_2d(index_dtype0, real_dtype, offset_x=0.0)
    edge_mesh1 = create_straight_polyline_2d(index_dtype1, real_dtype, offset_x=1.5)

    # Before transformation - should intersect
    result_before = tf.gather_intersecting_ids(edge_mesh0, edge_mesh1)
    assert result_before.shape[0] > 0

    # Apply transformation to edge_mesh1: move far away
    transformation = np.array([
        [1, 0, 100],
        [0, 1, 100],
        [0, 0, 1]
    ], dtype=real_dtype)

    edge_mesh1.transformation = transformation

    # After transformation - should not intersect
    result_after = tf.gather_intersecting_ids(edge_mesh0, edge_mesh1)
    assert result_after.shape[0] == 0


# ==============================================================================
# Edge cases
# ==============================================================================

def test_edge_mesh_gather_ids_edge_mesh_dimension_mismatch():
    """Test that dimension mismatch raises error"""
    edges2d = np.array([[0, 1]], dtype=np.int32)
    points2d = np.array([[0, 0], [1, 0]], dtype=np.float32)

    edges3d = np.array([[0, 1]], dtype=np.int32)
    points3d = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)

    edge_mesh2d = tf.EdgeMesh(edges2d, points2d)
    edge_mesh3d = tf.EdgeMesh(edges3d, points3d)

    with pytest.raises(ValueError, match="Dimension mismatch"):
        tf.gather_intersecting_ids(edge_mesh2d, edge_mesh3d)


@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_edge_mesh_gather_ids_edge_mesh_self_query(real_dtype):
    """Test edge mesh querying itself"""
    edges = np.array([[0, 1], [1, 2]], dtype=np.int32)
    points = np.array([[0, 0], [1, 0], [2, 0]], dtype=real_dtype)
    edge_mesh = tf.EdgeMesh(edges, points)

    result = tf.gather_intersecting_ids(edge_mesh, edge_mesh)

    # Should find all self-matches
    assert result.shape[1] == 2
    assert result.shape[0] >= 2  # At least (0,0) and (1,1)


@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_edge_mesh_gather_ids_edge_mesh_crossing_edges(real_dtype):
    """Test perpendicular crossing edges"""
    # Horizontal edge
    edges0 = np.array([[0, 1]], dtype=np.int32)
    points0 = np.array([[0, 0.5], [2, 0.5]], dtype=real_dtype)

    # Vertical edge crossing at x=1
    edges1 = np.array([[0, 1]], dtype=np.int32)
    points1 = np.array([[1, 0], [1, 1]], dtype=real_dtype)

    edge_mesh0 = tf.EdgeMesh(edges0, points0)
    edge_mesh1 = tf.EdgeMesh(edges1, points1)

    result = tf.gather_intersecting_ids(edge_mesh0, edge_mesh1)

    # Should find the intersection
    assert result.shape[0] > 0
    assert result.shape[1] == 2


@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_edge_mesh_gather_ids_edge_mesh_return_dtype(real_dtype):
    """Test that return dtype matches index dtype"""
    # Test with int32 indices
    edges0 = np.array([[0, 1]], dtype=np.int32)
    points0 = np.array([[0, 0], [1, 0]], dtype=real_dtype)

    edges1 = np.array([[0, 1]], dtype=np.int32)
    points1 = np.array([[0, 0], [1, 0]], dtype=real_dtype)

    edge_mesh0 = tf.EdgeMesh(edges0, points0)
    edge_mesh1 = tf.EdgeMesh(edges1, points1)

    result = tf.gather_intersecting_ids(edge_mesh0, edge_mesh1)

    assert result.dtype == np.int32

    # Test with int64 indices
    edges0_64 = np.array([[0, 1]], dtype=np.int64)
    edges1_64 = np.array([[0, 1]], dtype=np.int64)

    edge_mesh0_64 = tf.EdgeMesh(edges0_64, points0)
    edge_mesh1_64 = tf.EdgeMesh(edges1_64, points0)

    result_64 = tf.gather_intersecting_ids(edge_mesh0_64, edge_mesh1_64)

    assert result_64.dtype == np.int64


if __name__ == "__main__":
    # Run tests with verbose output
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
