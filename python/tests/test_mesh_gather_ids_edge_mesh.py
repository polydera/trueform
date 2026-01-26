"""
Tests for Mesh × EdgeMesh gather_ids

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/trueform
"""
import sys
import pytest
import numpy as np
import trueform as tf


# Type combinations
INDEX_DTYPES = [np.int32, np.int64]
REAL_DTYPES = [np.float32, np.float64]
MESH_TYPES = ['triangle', 'dynamic']  # triangle (ngon=3) and dynamic


def create_simple_mesh_2d(index_dtype, real_dtype, mesh_type):
    """Create a simple 2D mesh (triangles or dynamic)"""
    # Same geometry for both - two triangles forming a square
    points = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=real_dtype)

    if mesh_type == 'triangle':
        faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=index_dtype)
        return tf.Mesh(faces, points)
    else:
        # Dynamic mesh - same triangles wrapped in OffsetBlockedArray
        offsets = np.array([0, 3, 6], dtype=index_dtype)
        data = np.array([0, 1, 2, 0, 2, 3], dtype=index_dtype)
        faces = tf.OffsetBlockedArray(offsets, data)
        return tf.Mesh(faces, points)


def create_simple_mesh_3d(index_dtype, real_dtype, mesh_type):
    """Create a simple 3D mesh (triangles or dynamic in xy-plane)"""
    # Same geometry for both - two triangles forming a square in xy-plane
    points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=real_dtype)

    if mesh_type == 'triangle':
        faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=index_dtype)
        return tf.Mesh(faces, points)
    else:
        # Dynamic mesh - same triangles wrapped in OffsetBlockedArray
        offsets = np.array([0, 3, 6], dtype=index_dtype)
        data = np.array([0, 1, 2, 0, 2, 3], dtype=index_dtype)
        faces = tf.OffsetBlockedArray(offsets, data)
        return tf.Mesh(faces, points)


# ==============================================================================
# 2D Mesh-EdgeMesh gather_ids tests
# ==============================================================================

@pytest.mark.parametrize("index_dtype0", INDEX_DTYPES)
@pytest.mark.parametrize("index_dtype1", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
@pytest.mark.parametrize("mesh_type", MESH_TYPES)
def test_mesh_gather_ids_edge_mesh_2d_intersects_hit(index_dtype0, index_dtype1, real_dtype, mesh_type):
    """Test faces that intersect edges"""
    mesh = create_simple_mesh_2d(index_dtype0, real_dtype, mesh_type)

    # Edge crossing through the mesh
    edges = np.array([[0, 1]], dtype=index_dtype1)
    points = np.array([[0.5, -0.5], [0.5, 1.5]], dtype=real_dtype)
    edge_mesh = tf.EdgeMesh(edges, points)

    result = tf.gather_intersecting_ids(mesh, edge_mesh)

    # Should find intersection
    assert result.shape[1] == 2
    assert result.shape[0] > 0


@pytest.mark.parametrize("index_dtype0", INDEX_DTYPES)
@pytest.mark.parametrize("index_dtype1", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
@pytest.mark.parametrize("mesh_type", MESH_TYPES)
def test_mesh_gather_ids_edge_mesh_2d_intersects_miss(index_dtype0, index_dtype1, real_dtype, mesh_type):
    """Test mesh and edge mesh that don't intersect"""
    mesh = create_simple_mesh_2d(index_dtype0, real_dtype, mesh_type)

    # Edge far from mesh
    edges = np.array([[0, 1]], dtype=index_dtype1)
    points = np.array([[10, 10], [20, 20]], dtype=real_dtype)
    edge_mesh = tf.EdgeMesh(edges, points)

    result = tf.gather_intersecting_ids(mesh, edge_mesh)

    # Should return empty (0, 2) array
    assert result.shape == (0, 2)


@pytest.mark.parametrize("index_dtype0", INDEX_DTYPES)
@pytest.mark.parametrize("index_dtype1", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
@pytest.mark.parametrize("mesh_type", MESH_TYPES)
def test_mesh_gather_ids_edge_mesh_2d_within_distance_hit(index_dtype0, index_dtype1, real_dtype, mesh_type):
    """Test edges within distance of faces"""
    mesh = create_simple_mesh_2d(index_dtype0, real_dtype, mesh_type)

    # Edge parallel to and near the mesh
    edges = np.array([[0, 1]], dtype=index_dtype1)
    points = np.array([[0.5, -0.3], [0.5, -0.2]], dtype=real_dtype)
    edge_mesh = tf.EdgeMesh(edges, points)

    # With distance=0.5, should find edges near faces
    result = tf.gather_ids_within_distance(mesh, edge_mesh, distance=0.5)

    assert result.shape[1] == 2
    assert result.shape[0] > 0


@pytest.mark.parametrize("index_dtype0", INDEX_DTYPES)
@pytest.mark.parametrize("index_dtype1", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
@pytest.mark.parametrize("mesh_type", MESH_TYPES)
def test_mesh_gather_ids_edge_mesh_2d_within_distance_miss(index_dtype0, index_dtype1, real_dtype, mesh_type):
    """Test edges outside distance threshold"""
    mesh = create_simple_mesh_2d(index_dtype0, real_dtype, mesh_type)

    # Edge far from mesh
    edges = np.array([[0, 1]], dtype=index_dtype1)
    points = np.array([[10, 10], [20, 20]], dtype=real_dtype)
    edge_mesh = tf.EdgeMesh(edges, points)

    result = tf.gather_ids_within_distance(mesh, edge_mesh, distance=1.0)

    assert result.shape == (0, 2)


# ==============================================================================
# 3D Mesh-EdgeMesh gather_ids tests
# ==============================================================================

@pytest.mark.parametrize("index_dtype0", INDEX_DTYPES)
@pytest.mark.parametrize("index_dtype1", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
@pytest.mark.parametrize("mesh_type", MESH_TYPES)
def test_mesh_gather_ids_edge_mesh_3d_intersects_hit(index_dtype0, index_dtype1, real_dtype, mesh_type):
    """Test 3D faces intersecting edges"""
    mesh = create_simple_mesh_3d(index_dtype0, real_dtype, mesh_type)

    # Edge perpendicular to mesh, crossing through it
    edges = np.array([[0, 1]], dtype=index_dtype1)
    points = np.array([[0.5, 0.5, -1], [0.5, 0.5, 1]], dtype=real_dtype)
    edge_mesh = tf.EdgeMesh(edges, points)

    result = tf.gather_intersecting_ids(mesh, edge_mesh)

    assert result.shape[1] == 2
    assert result.shape[0] > 0


@pytest.mark.parametrize("index_dtype0", INDEX_DTYPES)
@pytest.mark.parametrize("index_dtype1", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
@pytest.mark.parametrize("mesh_type", MESH_TYPES)
def test_mesh_gather_ids_edge_mesh_3d_intersects_miss(index_dtype0, index_dtype1, real_dtype, mesh_type):
    """Test 3D mesh and edge mesh that don't intersect"""
    mesh = create_simple_mesh_3d(index_dtype0, real_dtype, mesh_type)

    # Edge parallel to mesh but above it
    edges = np.array([[0, 1]], dtype=index_dtype1)
    points = np.array([[0, 0, 5], [1, 0, 5]], dtype=real_dtype)
    edge_mesh = tf.EdgeMesh(edges, points)

    result = tf.gather_intersecting_ids(mesh, edge_mesh)

    assert result.shape == (0, 2)


@pytest.mark.parametrize("index_dtype0", INDEX_DTYPES)
@pytest.mark.parametrize("index_dtype1", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
@pytest.mark.parametrize("mesh_type", MESH_TYPES)
def test_mesh_gather_ids_edge_mesh_3d_within_distance(index_dtype0, index_dtype1, real_dtype, mesh_type):
    """Test 3D edges within distance of faces"""
    mesh = create_simple_mesh_3d(index_dtype0, real_dtype, mesh_type)

    # Edge parallel to mesh, slightly above
    edges = np.array([[0, 1]], dtype=index_dtype1)
    points = np.array([[0.5, 0.5, 0.3], [0.5, 0.8, 0.3]], dtype=real_dtype)
    edge_mesh = tf.EdgeMesh(edges, points)

    result = tf.gather_ids_within_distance(mesh, edge_mesh, distance=0.5)

    assert result.shape[1] == 2
    assert result.shape[0] > 0


# ==============================================================================
# Symmetry tests (swap argument order)
# ==============================================================================

@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_mesh_gather_ids_edge_mesh_symmetry(real_dtype):
    """Test that swapping arguments swaps column order"""
    mesh = create_simple_mesh_2d(np.int32, real_dtype, 'triangle')

    edges = np.array([[0, 1]], dtype=np.int32)
    points = np.array([[0.5, -0.5], [0.5, 1.5]], dtype=real_dtype)
    edge_mesh = tf.EdgeMesh(edges, points)

    result_m_em = tf.gather_intersecting_ids(mesh, edge_mesh)
    result_em_m = tf.gather_intersecting_ids(edge_mesh, mesh)

    # Both should have same number of matches
    assert result_m_em.shape[0] == result_em_m.shape[0]

    # Columns should be swapped
    if result_m_em.shape[0] > 0:
        pairs_m_em = set(tuple(row) for row in result_m_em)
        pairs_em_m = set((row[1], row[0]) for row in result_em_m)
        assert pairs_m_em == pairs_em_m


# ==============================================================================
# Transformation tests
# ==============================================================================

@pytest.mark.parametrize("index_dtype0", INDEX_DTYPES)
@pytest.mark.parametrize("index_dtype1", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
@pytest.mark.parametrize("mesh_type", MESH_TYPES)
def test_mesh_gather_ids_edge_mesh_with_transformation(index_dtype0, index_dtype1, real_dtype, mesh_type):
    """Test gather_ids with transformation"""
    mesh = create_simple_mesh_2d(index_dtype0, real_dtype, mesh_type)

    edges = np.array([[0, 1]], dtype=index_dtype1)
    points = np.array([[0.5, -0.5], [0.5, 1.5]], dtype=real_dtype)
    edge_mesh = tf.EdgeMesh(edges, points)

    # Before transformation - should intersect
    result_before = tf.gather_intersecting_ids(mesh, edge_mesh)
    assert result_before.shape[0] > 0

    # Transform edge mesh far away
    transformation = np.array([
        [1, 0, 100],
        [0, 1, 100],
        [0, 0, 1]
    ], dtype=real_dtype)
    edge_mesh.transformation = transformation

    # After transformation - no intersection
    result_after = tf.gather_intersecting_ids(mesh, edge_mesh)
    assert result_after.shape[0] == 0


# ==============================================================================
# Edge cases
# ==============================================================================

def test_mesh_gather_ids_edge_mesh_dimension_mismatch():
    """Test that dimension mismatch raises error"""
    mesh_2d = create_simple_mesh_2d(np.int32, np.float32, 'triangle')

    edges_3d = np.array([[0, 1]], dtype=np.int32)
    points_3d = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)
    edge_mesh_3d = tf.EdgeMesh(edges_3d, points_3d)

    with pytest.raises(ValueError, match="Dimension mismatch"):
        tf.gather_intersecting_ids(mesh_2d, edge_mesh_3d)


@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_mesh_gather_ids_edge_mesh_multiple_faces_edges(real_dtype):
    """Test multiple faces with multiple edges"""
    # Create 2 triangular faces
    faces = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)
    points = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=real_dtype)
    mesh = tf.Mesh(faces, points)

    # Create edges that intersect different faces
    edges = np.array([
        [0, 1],  # crosses face 0
        [2, 3],  # crosses face 1
        [4, 5]   # far away
    ], dtype=np.int32)
    edge_points = np.array([
        [0.3, -0.5], [0.3, 0.5],  # edge 0 crosses face 0
        [0.7, 0.5], [0.7, 1.5],   # edge 1 crosses face 1
        [10, 10], [20, 20]        # edge 2 is far
    ], dtype=real_dtype)
    edge_mesh = tf.EdgeMesh(edges, edge_points)

    result = tf.gather_intersecting_ids(mesh, edge_mesh)

    # Should find some intersections
    assert result.shape[1] == 2
    assert result.shape[0] > 0

    # Verify edge 2 is not in results
    pairs = set(tuple(row) for row in result)
    assert not any(pair[1] == 2 for pair in pairs)


@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
@pytest.mark.parametrize("mesh_type", MESH_TYPES)
def test_mesh_gather_ids_edge_mesh_return_dtype(real_dtype, mesh_type):
    """Test that return dtype matches mesh index dtype"""
    mesh_int32 = create_simple_mesh_2d(np.int32, real_dtype, mesh_type)

    edges = np.array([[0, 1]], dtype=np.int32)
    points = np.array([[0.5, -0.5], [0.5, 1.5]], dtype=real_dtype)
    edge_mesh = tf.EdgeMesh(edges, points)

    result = tf.gather_intersecting_ids(mesh_int32, edge_mesh)
    assert result.dtype == np.int32

    # Test with int64
    mesh_int64 = create_simple_mesh_2d(np.int64, real_dtype, mesh_type)
    result_64 = tf.gather_intersecting_ids(mesh_int64, edge_mesh)
    assert result_64.dtype == np.int64


@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_mesh_gather_ids_edge_mesh_edge_on_boundary(real_dtype):
    """Test edge lying on face boundary"""
    mesh = create_simple_mesh_2d(np.int32, real_dtype, 'triangle')

    # Edge along the bottom edge of the mesh
    edges = np.array([[0, 1]], dtype=np.int32)
    points = np.array([[0, 0], [1, 0]], dtype=real_dtype)
    edge_mesh = tf.EdgeMesh(edges, points)

    result = tf.gather_intersecting_ids(mesh, edge_mesh)

    # Should find intersection on boundary (implementation dependent)
    assert result.shape[1] == 2
    # Don't assert exact count as boundary handling may vary


if __name__ == "__main__":
    # Run tests with verbose output
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
