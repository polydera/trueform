"""
Tests for EdgeMesh × EdgeMesh neighbor_search

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


@pytest.mark.parametrize("index_dtype0", INDEX_DTYPES)
@pytest.mark.parametrize("index_dtype1", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_edge_mesh_neighbor_search_edge_mesh_2d_parallel_lines(index_dtype0, index_dtype1, real_dtype):
    """Test 2D parallel horizontal lines"""
    # EdgeMesh0: horizontal line from (0, 0) to (2, 0)
    # EdgeMesh1: horizontal line from (0, 1) to (2, 1)
    # Distance should be 1.0

    points0 = np.array([[0, 0], [1, 0], [2, 0]], dtype=real_dtype)
    edges0 = np.array([[0, 1], [1, 2]], dtype=index_dtype0)

    points1 = np.array([[0, 1], [1, 1], [2, 1]], dtype=real_dtype)
    edges1 = np.array([[0, 1], [1, 2]], dtype=index_dtype1)

    mesh0 = tf.EdgeMesh(edges0, points0)
    mesh1 = tf.EdgeMesh(edges1, points1)

    result = tf.neighbor_search(mesh0, mesh1)
    assert result is not None

    (idx0, idx1), (dist, pt0, pt1) = result

    # Distance should be 1.0 (vertical separation)
    assert abs(dist - 1.0) < 1e-5

    # Points should have same x coord, differ by 1 in y
    assert abs(pt0[0] - pt1[0]) < 1e-5
    assert abs(pt0[1] - pt1[1] - (-1.0)) < 1e-5

    # Test symmetry
    result_sym = tf.neighbor_search(mesh1, mesh0)
    (idx0_sym, idx1_sym), (dist_sym, pt0_sym, pt1_sym) = result_sym

    assert abs(dist_sym - dist) < 1e-6
    assert np.allclose(pt0_sym, pt1, atol=1e-5)
    assert np.allclose(pt1_sym, pt0, atol=1e-5)


@pytest.mark.parametrize("index_dtype0", INDEX_DTYPES)
@pytest.mark.parametrize("index_dtype1", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_edge_mesh_neighbor_search_edge_mesh_3d_perpendicular_lines(index_dtype0, index_dtype1, real_dtype):
    """Test 3D perpendicular lines - known closest point"""
    # EdgeMesh0: line along x-axis from (0, 0, 0) to (2, 0, 0)
    # EdgeMesh1: line along z-axis from (1, 1, 0) to (1, 1, 2)
    # Closest points: (1, 0, 0) on mesh0[0], (1, 1, 0) on mesh1[0]
    # Distance = 1.0

    points0 = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=real_dtype)
    edges0 = np.array([[0, 1], [1, 2]], dtype=index_dtype0)

    points1 = np.array([[1, 1, 0], [1, 1, 1], [1, 1, 2]], dtype=real_dtype)
    edges1 = np.array([[0, 1], [1, 2]], dtype=index_dtype1)

    mesh0 = tf.EdgeMesh(edges0, points0)
    mesh1 = tf.EdgeMesh(edges1, points1)

    result = tf.neighbor_search(mesh0, mesh1)
    assert result is not None

    (idx0, idx1), (dist, pt0, pt1) = result

    # Distance should be 1.0
    assert abs(dist - 1.0) < 1e-5

    # pt0 should be at (1, 0, 0)
    assert np.allclose(pt0, [1, 0, 0], atol=1e-5)
    # pt1 should be at (1, 1, 0)
    assert np.allclose(pt1, [1, 1, 0], atol=1e-5)

    # idx1 should be 0 (first edge of mesh1 contains the endpoint)
    assert idx1 == 0

    # Test symmetry
    result_sym = tf.neighbor_search(mesh1, mesh0)
    ((_, _), (dist_sym, pt0_sym, pt1_sym)) = result_sym
    assert abs(dist_sym - 1.0) < 1e-6
    assert np.allclose(pt0_sym, pt1, atol=1e-5)
    assert np.allclose(pt1_sym, pt0, atol=1e-5)


@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_edge_mesh_neighbor_search_edge_mesh_2d_radius_hit(real_dtype):
    """Test 2D with radius - within radius"""
    points0 = np.array([[0, 0], [1, 0]], dtype=real_dtype)
    edges0 = np.array([[0, 1]], dtype=np.int32)

    points1 = np.array([[0.5, 0.3], [1.5, 0.3]], dtype=real_dtype)
    edges1 = np.array([[0, 1]], dtype=np.int32)

    mesh0 = tf.EdgeMesh(edges0, points0)
    mesh1 = tf.EdgeMesh(edges1, points1)

    # Distance is 0.3, so radius=1.0 should find it
    result = tf.neighbor_search(mesh0, mesh1, radius=1.0)
    assert result is not None

    ((_, _), (dist, _, _)) = result
    assert dist < 1.0


@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_edge_mesh_neighbor_search_edge_mesh_2d_radius_miss(real_dtype):
    """Test 2D with radius - outside radius"""
    points0 = np.array([[0, 0], [1, 0]], dtype=real_dtype)
    edges0 = np.array([[0, 1]], dtype=np.int32)

    points1 = np.array([[0, 10], [1, 10]], dtype=real_dtype)
    edges1 = np.array([[0, 1]], dtype=np.int32)

    mesh0 = tf.EdgeMesh(edges0, points0)
    mesh1 = tf.EdgeMesh(edges1, points1)

    # Distance is 10, radius=1.0 should miss
    result = tf.neighbor_search(mesh0, mesh1, radius=1.0)
    assert result is None


@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_edge_mesh_neighbor_search_edge_mesh_index_canonicalization(real_dtype):
    """Test that int64×int32 is handled correctly via canonicalization"""
    points0 = np.array([[0, 0], [1, 0]], dtype=real_dtype)
    edges0 = np.array([[0, 1]], dtype=np.int64)

    points1 = np.array([[0, 0.5], [1, 0.5]], dtype=real_dtype)
    edges1 = np.array([[0, 1]], dtype=np.int32)

    mesh0 = tf.EdgeMesh(edges0, points0)
    mesh1 = tf.EdgeMesh(edges1, points1)

    # Should work despite different index types
    result = tf.neighbor_search(mesh0, mesh1)
    assert result is not None

    ((_, _), (dist, _, _)) = result
    assert abs(dist - 0.25) < 1e-5


@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_edge_mesh_neighbor_search_edge_mesh_with_transformation(real_dtype):
    """Test neighbor search with transformed edge meshes"""
    points0 = np.array([[0, 0], [2, 0]], dtype=real_dtype)
    edges0 = np.array([[0, 1]], dtype=np.int32)

    points1 = np.array([[0, 0], [2, 0]], dtype=real_dtype)
    edges1 = np.array([[0, 1]], dtype=np.int32)

    mesh0 = tf.EdgeMesh(edges0, points0)
    mesh1 = tf.EdgeMesh(edges1, points1)

    # Transform mesh1 by offset (0, 1)
    transform = np.eye(3, dtype=real_dtype)
    transform[1, 2] = 1.0
    mesh1.transformation = transform

    result = tf.neighbor_search(mesh0, mesh1)
    assert result is not None

    ((_, _), (dist, pt0, pt1)) = result

    # Distance should be 1.0 (vertical offset)
    assert abs(dist - 1.0) < 1e-5

    # pt0 should be on mesh0 (y=0)
    # pt1 should be on transformed mesh1 (y=1)
    assert pt0[1] == pytest.approx(0.0, abs=1e-5)
    assert pt1[1] == pytest.approx(1.0, abs=1e-5)


def test_edge_mesh_neighbor_search_edge_mesh_dimension_mismatch():
    """Test that dimension mismatch raises error"""
    points2d = np.array([[0, 0], [1, 0]], dtype=np.float32)
    edges2d = np.array([[0, 1]], dtype=np.int32)

    points3d = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)
    edges3d = np.array([[0, 1]], dtype=np.int32)

    mesh2d = tf.EdgeMesh(edges2d, points2d)
    mesh3d = tf.EdgeMesh(edges3d, points3d)

    with pytest.raises(ValueError, match="Dimension mismatch"):
        tf.neighbor_search(mesh2d, mesh3d)


@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_edge_mesh_neighbor_search_edge_mesh_offset_parallel_segments(real_dtype):
    """Test with offset parallel segments - unambiguous closest point"""
    # Mesh0: single segment from (0, 0) to (1, 0)
    # Mesh1: single segment from (2, 1) to (3, 1)
    # Closest points: (1, 0) to (2, 1), distance = sqrt(2)

    points0 = np.array([[0, 0], [1, 0]], dtype=real_dtype)
    edges0 = np.array([[0, 1]], dtype=np.int32)

    points1 = np.array([[2, 1], [3, 1]], dtype=real_dtype)
    edges1 = np.array([[0, 1]], dtype=np.int32)

    mesh0 = tf.EdgeMesh(edges0, points0)
    mesh1 = tf.EdgeMesh(edges1, points1)

    result = tf.neighbor_search(mesh0, mesh1)
    assert result is not None

    (idx0, idx1), (dist, pt0, pt1) = result

    # Both should be edge 0
    assert idx0 == 0
    assert idx1 == 0

    # Distance = 1^2 + 1^2 = 2
    assert abs(dist - 2.0) < 1e-5

    # Closest point on mesh0 should be at (1, 0)
    assert np.allclose(pt0, [1, 0], atol=1e-5)
    # Closest point on mesh1 should be at (2, 1)
    assert np.allclose(pt1, [2, 1], atol=1e-5)


if __name__ == "__main__":
    # Run tests with verbose output
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
