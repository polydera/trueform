"""
Tests for Mesh × EdgeMesh neighbor_search

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
# ngon: 3 for triangles, 'dyn' for dynamic
NGONS = [3, 'dyn']


def create_square_mesh_2d(points, index_dtype, ngon):
    """Create a square mesh from 4 points (2D)"""
    if ngon == 3:
        faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=index_dtype)
        return tf.Mesh(faces, points)
    else:  # dynamic - create as one quad
        offsets = np.array([0, 4], dtype=index_dtype)
        data = np.array([0, 1, 2, 3], dtype=index_dtype)
        faces = tf.OffsetBlockedArray(offsets, data)
        return tf.Mesh(faces, points)


def create_square_mesh_3d(points, index_dtype, ngon):
    """Create a square mesh from 4 points (3D)"""
    if ngon == 3:
        faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=index_dtype)
        return tf.Mesh(faces, points)
    else:  # dynamic - create as one quad
        offsets = np.array([0, 4], dtype=index_dtype)
        data = np.array([0, 1, 2, 3], dtype=index_dtype)
        faces = tf.OffsetBlockedArray(offsets, data)
        return tf.Mesh(faces, points)


@pytest.mark.parametrize("mesh_index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("edge_index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
@pytest.mark.parametrize("ngon", NGONS)
def test_mesh_neighbor_search_edge_mesh_2d_planar_mesh_and_line(mesh_index_dtype, edge_index_dtype, real_dtype, ngon):
    """Test 2D: planar mesh and a line"""
    points = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=real_dtype)
    mesh = create_square_mesh_2d(points, mesh_index_dtype, ngon)

    # EdgeMesh: horizontal line from (0.5, 1.5) to (0.5, 2.5)
    edge_points = np.array([[0.5, 1.5], [0.5, 2.5]], dtype=real_dtype)
    edges = np.array([[0, 1]], dtype=edge_index_dtype)

    edge_mesh = tf.EdgeMesh(edges, edge_points)

    result = tf.neighbor_search(mesh, edge_mesh)
    assert result is not None

    (mesh_idx, edge_idx), (dist, pt_mesh, pt_edge) = result

    # Closest point on mesh should be around (0.5, 1)
    # Closest point on edge should be around (0.5, 1.5)
    # Distance = 0.5^2 = 0.25
    expected_dist = 0.5**2
    assert abs(dist - expected_dist) < 1e-4

    # Test symmetry
    result_sym = tf.neighbor_search(edge_mesh, mesh)
    (edge_idx_sym, mesh_idx_sym), (dist_sym, pt_edge_sym, pt_mesh_sym) = result_sym

    assert abs(dist_sym - dist) < 1e-6
    assert np.allclose(pt_edge_sym, pt_edge, atol=1e-5)
    assert np.allclose(pt_mesh_sym, pt_mesh, atol=1e-5)


@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
@pytest.mark.parametrize("ngon", NGONS)
def test_mesh_neighbor_search_edge_mesh_3d_planar_mesh_and_line(real_dtype, ngon):
    """Test 3D: planar mesh in xy-plane and a line parallel to z-axis"""
    # Mesh: Simple planar tiling in xy-plane at z=0
    # EdgeMesh: Vertical line at (0.5, 0.5, z) from z=1 to z=2
    # Closest points: (0.5, 0.5, 0) on mesh, (0.5, 0.5, 1) on line
    # Distance = 1.0

    points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=real_dtype)
    mesh = create_square_mesh_3d(points, np.int32, ngon)

    edge_points = np.array([[0.5, 0.5, 1], [0.5, 0.5, 2]], dtype=real_dtype)
    edges = np.array([[0, 1]], dtype=np.int32)

    edge_mesh = tf.EdgeMesh(edges, edge_points)

    result = tf.neighbor_search(mesh, edge_mesh)
    assert result is not None

    (mesh_idx, edge_idx), (dist, pt_mesh, pt_edge) = result

    # Distance should be 1.0
    assert abs(dist - 1.0) < 1e-5

    # pt_mesh should be at (0.5, 0.5, 0)
    assert np.allclose(pt_mesh, [0.5, 0.5, 0], atol=1e-4)
    # pt_edge should be at (0.5, 0.5, 1)
    assert np.allclose(pt_edge, [0.5, 0.5, 1], atol=1e-4)

    # edge_idx should be 0 (only one edge)
    assert edge_idx == 0

    # Test symmetry
    result_sym = tf.neighbor_search(edge_mesh, mesh)
    ((_, _), (dist_sym, pt_edge_sym, pt_mesh_sym)) = result_sym
    assert abs(dist_sym - 1.0) < 1e-6
    assert np.allclose(pt_edge_sym, pt_edge, atol=1e-5)
    assert np.allclose(pt_mesh_sym, pt_mesh, atol=1e-5)


@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_mesh_neighbor_search_edge_mesh_2d_radius_hit(real_dtype):
    """Test 2D with radius - within radius"""
    # Triangle mesh
    points = np.array([[0, 0], [1, 0], [0.5, 1]], dtype=real_dtype)
    faces = np.array([[0, 1, 2]], dtype=np.int32)

    # Line near the mesh
    edge_points = np.array([[0.5, -0.5], [0.5, -0.3]], dtype=real_dtype)
    edges = np.array([[0, 1]], dtype=np.int32)

    mesh = tf.Mesh(faces, points)
    edge_mesh = tf.EdgeMesh(edges, edge_points)

    # Closest distance should be around 0.3, so radius=1.0 should find it
    result = tf.neighbor_search(mesh, edge_mesh, radius=1.0)
    assert result is not None

    ((_, _), (dist, _, _)) = result
    assert dist < 1.0


@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_mesh_neighbor_search_edge_mesh_2d_radius_miss(real_dtype):
    """Test 2D with radius - outside radius"""
    points = np.array([[0, 0], [1, 0], [0.5, 1]], dtype=real_dtype)
    faces = np.array([[0, 1, 2]], dtype=np.int32)

    # Line far from mesh
    edge_points = np.array([[10, 10], [11, 11]], dtype=real_dtype)
    edges = np.array([[0, 1]], dtype=np.int32)

    mesh = tf.Mesh(faces, points)
    edge_mesh = tf.EdgeMesh(edges, edge_points)

    result = tf.neighbor_search(mesh, edge_mesh, radius=1.0)
    assert result is None


@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_mesh_neighbor_search_edge_mesh_with_transformation(real_dtype):
    """Test neighbor search with transformed mesh"""
    # Triangle mesh at origin
    points = np.array([[0, 0], [1, 0], [0, 1]], dtype=real_dtype)
    faces = np.array([[0, 1, 2]], dtype=np.int32)

    # Line segment
    edge_points = np.array([[0.5, 0], [1.5, 0]], dtype=real_dtype)
    edges = np.array([[0, 1]], dtype=np.int32)

    mesh = tf.Mesh(faces, points)
    edge_mesh = tf.EdgeMesh(edges, edge_points)

    # Transform mesh by offset (0, 1)
    transform = np.eye(3, dtype=real_dtype)
    transform[1, 2] = 1.0  # y offset
    mesh.transformation = transform

    result = tf.neighbor_search(mesh, edge_mesh)
    assert result is not None

    ((_, _), (dist, pt_mesh, pt_edge)) = result

    # pt_mesh should be on the transformed mesh (y around 1.0)
    # pt_edge should be on the edge (y = 0)
    assert pt_mesh[1] > 0.9
    assert abs(pt_edge[1]) < 0.1


def test_mesh_neighbor_search_edge_mesh_dimension_mismatch():
    """Test that dimension mismatch raises error"""
    points2d = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float32)
    faces2d = np.array([[0, 1, 2]], dtype=np.int32)

    points3d = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)
    edges3d = np.array([[0, 1]], dtype=np.int32)

    mesh2d = tf.Mesh(faces2d, points2d)
    edge3d = tf.EdgeMesh(edges3d, points3d)

    with pytest.raises(ValueError, match="Dimension mismatch"):
        tf.neighbor_search(mesh2d, edge3d)


@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_mesh_neighbor_search_edge_mesh_edge_above_mesh_corner(real_dtype):
    """Test edge positioned directly above a mesh corner - unambiguous result"""
    # Dynamic mesh (quad)
    points = np.array([[0, 0], [2, 0], [2, 2], [0, 2]], dtype=real_dtype)
    offsets = np.array([0, 4], dtype=np.int32)
    data = np.array([0, 1, 2, 3], dtype=np.int32)
    faces = tf.OffsetBlockedArray(offsets, data)

    # Edge from (1, 3) to (1, 4) - above the mesh
    edge_points = np.array([[1, 3], [1, 4]], dtype=real_dtype)
    edges = np.array([[0, 1]], dtype=np.int32)

    mesh = tf.Mesh(faces, points)
    edge_mesh = tf.EdgeMesh(edges, edge_points)

    result = tf.neighbor_search(mesh, edge_mesh)
    assert result is not None

    ((_, edge_idx), (dist, pt_mesh, pt_edge)) = result

    # Closest point on mesh should be (1, 2) - top edge of quad
    # Closest point on edge should be (1, 3) - bottom of edge
    # Distance = 1.0^2 = 1.0

    assert abs(dist - 1.0) < 1e-5
    assert np.allclose(pt_mesh, [1, 2], atol=1e-4)
    assert np.allclose(pt_edge, [1, 3], atol=1e-4)
    assert edge_idx == 0


@pytest.mark.parametrize("mesh_index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("edge_index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_mesh_neighbor_search_edge_mesh_all_index_combinations(mesh_index_dtype, edge_index_dtype, real_dtype):
    """Test all combinations of index types work correctly"""
    # Simple triangle
    points = np.array([[0, 0], [1, 0], [0, 1]], dtype=real_dtype)
    faces = np.array([[0, 1, 2]], dtype=mesh_index_dtype)

    # Simple edge
    edge_points = np.array([[0.5, -0.5], [0.5, -0.2]], dtype=real_dtype)
    edges = np.array([[0, 1]], dtype=edge_index_dtype)

    mesh = tf.Mesh(faces, points)
    edge_mesh = tf.EdgeMesh(edges, edge_points)

    # Should work for all index type combinations
    result = tf.neighbor_search(mesh, edge_mesh)
    assert result is not None

    # Test symmetry also works
    result_sym = tf.neighbor_search(edge_mesh, mesh)
    assert result_sym is not None


if __name__ == "__main__":
    # Run tests with verbose output
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
