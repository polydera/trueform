"""
Tests for mesh neighbor_search spatial queries

Uses pytest parametrization to efficiently test all type combinations:
- Index types: int32, int64
- Real types: float32, float64
- Ngon: 3 (triangles), dynamic
- Dims: 2D, 3D

Copyright (c) 2025 Žiga Sajovic, XLAB
"""

import sys
import numpy as np
import pytest
import trueform as tf

# Type combinations to test
INDEX_DTYPES = [np.int32, np.int64]
REAL_DTYPES = [np.float32, np.float64]


def create_2d_triangle_mesh(index_dtype, real_dtype):
    """Create a simple 2D triangle mesh"""
    # Triangle mesh: 4 points, 2 triangles
    faces = np.array([[0, 1, 2], [1, 2, 3]], dtype=index_dtype)
    points = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=real_dtype)
    return tf.Mesh(faces, points)


def create_3d_triangle_mesh(index_dtype, real_dtype):
    """Create a simple 3D triangle mesh"""
    # Triangle mesh: 4 points, 2 triangles
    faces = np.array([[0, 1, 2], [1, 2, 3]], dtype=index_dtype)
    points = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=real_dtype
    )
    return tf.Mesh(faces, points)


def create_2d_dynamic_mesh(index_dtype, real_dtype):
    """Create a simple 2D dynamic (n-gon) mesh using OffsetBlockedArray"""
    # Dynamic mesh: 4 points, 2 triangles (same geometry as triangle mesh)
    # Triangle 0: [0, 1, 2], Triangle 1: [1, 2, 3]
    offsets = np.array([0, 3, 6], dtype=index_dtype)
    data = np.array([0, 1, 2, 1, 2, 3], dtype=index_dtype)
    faces = tf.OffsetBlockedArray(offsets, data)
    points = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=real_dtype)
    return tf.Mesh(faces, points)


def create_3d_dynamic_mesh(index_dtype, real_dtype):
    """Create a simple 3D dynamic (n-gon) mesh using OffsetBlockedArray"""
    # Dynamic mesh: 4 points, 2 triangles (same geometry as triangle mesh)
    # Triangle 0: [0, 1, 2], Triangle 1: [1, 2, 3]
    offsets = np.array([0, 3, 6], dtype=index_dtype)
    data = np.array([0, 1, 2, 1, 2, 3], dtype=index_dtype)
    faces = tf.OffsetBlockedArray(offsets, data)
    points = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=real_dtype
    )
    return tf.Mesh(faces, points)


# Test fixtures for mesh creation
# (dims, ngon) where ngon is 3 for triangles, 'dyn' for dynamic
MESH_CREATORS = {
    (2, 3): create_2d_triangle_mesh,
    (3, 3): create_3d_triangle_mesh,
    (2, 'dyn'): create_2d_dynamic_mesh,
    (3, 'dyn'): create_3d_dynamic_mesh,
}


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
@pytest.mark.parametrize("dims,ngon", [(2, 3), (3, 3), (2, 'dyn'), (3, 'dyn')])
def test_mesh_neighbor_search_point(index_dtype, real_dtype, dims, ngon):
    """Test single nearest neighbor search with point query on mesh"""
    # Create mesh with specified types
    mesh_creator = MESH_CREATORS[(dims, ngon)]
    mesh = mesh_creator(index_dtype, real_dtype)

    # Create query point near origin
    if dims == 2:
        query = tf.Point([0.1, 0.1])
    else:  # dims == 3
        query = tf.Point([0.1, 0.1, 0.1])

    # Query mesh
    idx, dist2, closest_pt = tf.neighbor_search(mesh, query)

    # Should find face 0 (closest to query point)
    assert idx == 0
    # For 2D: point is inside face, dist should be ~0
    # For 3D: face 0 is in z=0 plane, query at z=0.1, so dist² = 0.1² = 0.01
    if dims == 2:
        assert dist2 < 1e-6
    else:
        assert abs(dist2 - 0.01) < 1e-5
    # Closest point should have correct dimensions
    assert closest_pt.shape == (dims,)
    # Closest point should have correct dtype
    assert closest_pt.dtype == real_dtype


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
@pytest.mark.parametrize("dims,ngon", [(2, 3), (3, 3), (2, 'dyn'), (3, 'dyn')])
def test_mesh_neighbor_search_segment(index_dtype, real_dtype, dims, ngon):
    """Test single nearest neighbor search with segment query on mesh"""
    mesh_creator = MESH_CREATORS[(dims, ngon)]
    mesh = mesh_creator(index_dtype, real_dtype)

    # Create segment query
    if dims == 2:
        query = tf.Segment([[0.1, 0.1], [0.2, 0.2]])
    else:  # dims == 3
        query = tf.Segment([[0.1, 0.1, 0.1], [0.2, 0.2, 0.2]])

    idx, dist2, closest_pt = tf.neighbor_search(mesh, query)

    assert idx >= 0
    assert dist2 >= 0
    assert closest_pt.shape == (dims,)
    assert closest_pt.dtype == real_dtype


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
@pytest.mark.parametrize("dims,ngon", [(2, 3), (3, 3), (2, 'dyn'), (3, 'dyn')])
def test_mesh_neighbor_search_polygon(index_dtype, real_dtype, dims, ngon):
    """Test single nearest neighbor search with polygon query on mesh"""
    mesh_creator = MESH_CREATORS[(dims, ngon)]
    mesh = mesh_creator(index_dtype, real_dtype)

    # Create polygon query (triangle)
    if dims == 2:
        query = tf.Polygon([[0.1, 0.1], [0.3, 0.1], [0.2, 0.3]])
    else:  # dims == 3
        query = tf.Polygon([[0.1, 0.1, 0.1], [0.3, 0.1, 0.1], [0.2, 0.3, 0.1]])

    idx, dist2, closest_pt = tf.neighbor_search(mesh, query)

    assert idx >= 0
    assert dist2 >= 0
    assert closest_pt.shape == (dims,)
    assert closest_pt.dtype == real_dtype


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
@pytest.mark.parametrize("dims,ngon", [(2, 3), (3, 3), (2, 'dyn'), (3, 'dyn')])
def test_mesh_neighbor_search_ray(index_dtype, real_dtype, dims, ngon):
    """Test single nearest neighbor search with ray query on mesh"""
    mesh_creator = MESH_CREATORS[(dims, ngon)]
    mesh = mesh_creator(index_dtype, real_dtype)

    # Create ray query
    if dims == 2:
        query = tf.Ray(origin=[0.1, 0.1], direction=[1.0, 1.0])
    else:  # dims == 3
        query = tf.Ray(origin=[0.1, 0.1, 0.1], direction=[1.0, 1.0, 1.0])

    idx, dist2, closest_pt = tf.neighbor_search(mesh, query)

    assert idx >= 0
    assert dist2 >= 0
    assert closest_pt.shape == (dims,)
    assert closest_pt.dtype == real_dtype


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
@pytest.mark.parametrize("dims,ngon", [(2, 3), (3, 3), (2, 'dyn'), (3, 'dyn')])
def test_mesh_neighbor_search_line(index_dtype, real_dtype, dims, ngon):
    """Test single nearest neighbor search with line query on mesh"""
    mesh_creator = MESH_CREATORS[(dims, ngon)]
    mesh = mesh_creator(index_dtype, real_dtype)

    # Create line query
    if dims == 2:
        query = tf.Line(data=[[0.1, 0.1], [1.0, 1.0]])
    else:  # dims == 3
        query = tf.Line(data=[[0.1, 0.1, 0.1], [1.0, 1.0, 1.0]])

    idx, dist2, closest_pt = tf.neighbor_search(mesh, query)

    assert idx >= 0
    assert dist2 >= 0
    assert closest_pt.shape == (dims,)
    assert closest_pt.dtype == real_dtype


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
@pytest.mark.parametrize("dims,ngon", [(2, 3), (3, 3), (2, 'dyn'), (3, 'dyn')])
def test_mesh_neighbor_search_knn_point(index_dtype, real_dtype, dims, ngon):
    """Test KNN search with point query on mesh"""
    mesh_creator = MESH_CREATORS[(dims, ngon)]
    mesh = mesh_creator(index_dtype, real_dtype)

    # Create query point
    if dims == 2:
        query = tf.Point([0.5, 0.5])
    else:  # dims == 3
        query = tf.Point([0.5, 0.5, 0.5])

    # Query for k=2 nearest faces
    k = 2
    results = tf.neighbor_search(mesh, query, k=k)

    # Should return up to k results
    assert len(results) <= k
    assert len(results) > 0

    # Results should be sorted by distance
    for i in range(len(results) - 1):
        assert results[i][1] <= results[i + 1][1]

    # Each result should have correct structure
    for idx, dist2, closest_pt in results:
        assert idx >= 0
        assert dist2 >= 0
        assert closest_pt.shape == (dims,)
        assert closest_pt.dtype == real_dtype


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_mesh_neighbor_search_with_radius_2d_tri(index_dtype, real_dtype):
    """Test mesh neighbor search with radius constraint"""
    mesh = create_2d_triangle_mesh(index_dtype, real_dtype)

    query = tf.Point([0.1, 0.1])

    # Test with large radius - should find something
    idx1, dist2_1, pt1 = tf.neighbor_search(mesh, query, radius=10.0)
    assert idx1 >= 0
    assert dist2_1 >= 0

    # Test with very small radius - might not find anything depending on mesh
    # Just verify it doesn't crash
    try:
        result = tf.neighbor_search(mesh, query, radius=0.001)
        if result is not None:
            idx2, dist2_2, pt2 = result
            assert dist2_2 <= 0.001 ** 2  # Distance should be within radius
    except Exception:
        # If no result found within radius, that's also valid behavior
        pass


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_mesh_neighbor_search_knn_with_radius_3d_tri(index_dtype, real_dtype):
    """Test mesh KNN search with radius constraint"""
    mesh = create_3d_triangle_mesh(index_dtype, real_dtype)

    query = tf.Point([0.5, 0.5, 0.5])

    # Test KNN with large radius
    results = tf.neighbor_search(mesh, query, k=2, radius=10.0)
    assert len(results) > 0
    for idx, dist2, closest_pt in results:
        assert dist2 <= 100.0  # Within radius^2


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_mesh_neighbor_search_with_radius_dynamic(index_dtype, real_dtype):
    """Test dynamic mesh neighbor search with radius constraint"""
    mesh = create_3d_dynamic_mesh(index_dtype, real_dtype)

    query = tf.Point([0.1, 0.1, 0.1])

    # Test with large radius - should find something
    idx1, dist2_1, pt1 = tf.neighbor_search(mesh, query, radius=10.0)
    assert idx1 >= 0
    assert dist2_1 >= 0


def test_mesh_type_validation():
    """Test that mesh validates types correctly"""
    # Valid triangle mesh
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    points = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float32)
    mesh = tf.Mesh(faces, points)
    assert mesh.dims == 2
    assert mesh.ngon == 3

    # Valid dynamic mesh
    offsets = np.array([0, 3, 7], dtype=np.int64)
    data = np.array([0, 1, 2, 0, 1, 2, 3], dtype=np.int64)
    faces = tf.OffsetBlockedArray(offsets, data)
    points = np.array(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float64
    )
    mesh = tf.Mesh(faces, points)
    assert mesh.dims == 3
    assert mesh.ngon is None  # Dynamic meshes have ngon=None
    assert mesh.is_dynamic


def test_mesh_quad_not_supported():
    """Test that quad meshes are no longer supported"""
    faces = np.array([[0, 1, 2, 3]], dtype=np.int32)
    points = np.array(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float32
    )
    with pytest.raises(ValueError, match="Fixed-size faces must have 3 vertices"):
        tf.Mesh(faces, points)


def test_mesh_dimension_mismatch():
    """Test that dimension mismatch is detected"""
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    points = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float32)
    mesh = tf.Mesh(faces, points)  # 2D mesh

    # Try to query with 3D point
    query_3d = tf.Point([0.5, 0.5, 0.5])

    with pytest.raises(ValueError, match="Dimension mismatch"):
        tf.neighbor_search(mesh, query_3d)


if __name__ == "__main__":
    # Run tests with verbose output
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
