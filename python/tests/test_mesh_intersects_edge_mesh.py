"""
Tests for Mesh intersects with EdgeMesh

Copyright (c) 2025 Žiga Sajovic, XLAB
"""

import sys
import numpy as np
import pytest
import trueform as tf


# Type combinations
INDEX_DTYPES = [np.int32, np.int64]
REAL_DTYPES = [np.float32, np.float64]
MESH_TYPES = ['triangle', 'dynamic']  # triangles and dynamic (variable-size)
DIMS = [2, 3]


def create_tiled_plane_2d_triangles(index_dtype, real_dtype, offset_x=0.0):
    """Create a simple 2x1 tiled plane in 2D with triangles"""
    faces = np.array([
        [0, 1, 4],
        [0, 4, 3],
        [1, 2, 5],
        [1, 5, 4]
    ], dtype=index_dtype)
    points = np.array([
        [0 + offset_x, 0], [1 + offset_x, 0], [2 + offset_x, 0],
        [0 + offset_x, 1], [1 + offset_x, 1], [2 + offset_x, 1]
    ], dtype=real_dtype)
    return tf.Mesh(faces, points)


def create_tiled_plane_3d_triangles(index_dtype, real_dtype, offset_z=0.0):
    """Create a simple 2x1 tiled plane in 3D (z=offset_z) with triangles"""
    faces = np.array([
        [0, 1, 4],
        [0, 4, 3],
        [1, 2, 5],
        [1, 5, 4]
    ], dtype=index_dtype)
    points = np.array([
        [0, 0, offset_z], [1, 0, offset_z], [2, 0, offset_z],
        [0, 1, offset_z], [1, 1, offset_z], [2, 1, offset_z]
    ], dtype=real_dtype)
    return tf.Mesh(faces, points)


def create_tiled_plane_2d_dynamic(index_dtype, real_dtype, offset_x=0.0):
    """Create a simple 2x1 tiled plane in 2D with dynamic (variable-size) polygons."""
    # Mixed polygon faces: one quad and two triangles
    offsets = np.array([0, 4, 7, 10], dtype=index_dtype)
    data = np.array([0, 1, 4, 3, 1, 2, 5, 1, 5, 4], dtype=index_dtype)
    faces = tf.OffsetBlockedArray(offsets, data)
    points = np.array([
        [0 + offset_x, 0], [1 + offset_x, 0], [2 + offset_x, 0],
        [0 + offset_x, 1], [1 + offset_x, 1], [2 + offset_x, 1]
    ], dtype=real_dtype)
    return tf.Mesh(faces, points)


def create_tiled_plane_3d_dynamic(index_dtype, real_dtype, offset_z=0.0):
    """Create a simple 2x1 tiled plane in 3D (z=offset_z) with dynamic (variable-size) polygons."""
    # Mixed polygon faces: one quad and two triangles
    offsets = np.array([0, 4, 7, 10], dtype=index_dtype)
    data = np.array([0, 1, 4, 3, 1, 2, 5, 1, 5, 4], dtype=index_dtype)
    faces = tf.OffsetBlockedArray(offsets, data)
    points = np.array([
        [0, 0, offset_z], [1, 0, offset_z], [2, 0, offset_z],
        [0, 1, offset_z], [1, 1, offset_z], [2, 1, offset_z]
    ], dtype=real_dtype)
    return tf.Mesh(faces, points)


def create_straight_polyline_2d(index_dtype, real_dtype, offset_x=0.0):
    """Create a simple straight polyline in 2D"""
    edges = np.array([[0, 1], [1, 2], [2, 3]], dtype=index_dtype)
    points = np.array([[0 + offset_x, 0], [1 + offset_x, 0], [2 + offset_x, 0], [3 + offset_x, 0]], dtype=real_dtype)
    return tf.EdgeMesh(edges, points)


def create_straight_polyline_3d(index_dtype, real_dtype, offset_z=0.0):
    """Create a simple straight polyline in 3D (along x-axis, z=offset_z)"""
    edges = np.array([[0, 1], [1, 2], [2, 3]], dtype=index_dtype)
    points = np.array([[0, 0, offset_z], [1, 0, offset_z], [2, 0, offset_z], [3, 0, offset_z]], dtype=real_dtype)
    return tf.EdgeMesh(edges, points)


# ==============================================================================
# 2D Mesh-EdgeMesh intersection tests
# ==============================================================================

@pytest.mark.parametrize("mesh_index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("edge_index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
@pytest.mark.parametrize("mesh_type", MESH_TYPES)
def test_mesh_intersects_edge_mesh_2d_hit(mesh_index_dtype, edge_index_dtype, real_dtype, mesh_type):
    """Test mesh and edge mesh that overlap intersect"""
    if mesh_type == 'triangle':
        mesh = create_tiled_plane_2d_triangles(mesh_index_dtype, real_dtype, offset_x=0.0)
    else:
        mesh = create_tiled_plane_2d_dynamic(mesh_index_dtype, real_dtype, offset_x=0.0)

    # EdgeMesh overlapping with mesh (at y=0, x from 0.5 to 3.5)
    edge_mesh = create_straight_polyline_2d(edge_index_dtype, real_dtype, offset_x=0.5)

    assert tf.intersects(mesh, edge_mesh)
    assert tf.intersects(edge_mesh, mesh)  # Symmetric


@pytest.mark.parametrize("mesh_index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("edge_index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
@pytest.mark.parametrize("mesh_type", MESH_TYPES)
def test_mesh_intersects_edge_mesh_2d_miss(mesh_index_dtype, edge_index_dtype, real_dtype, mesh_type):
    """Test non-overlapping mesh and edge mesh don't intersect"""
    if mesh_type == 'triangle':
        mesh = create_tiled_plane_2d_triangles(mesh_index_dtype, real_dtype, offset_x=0.0)
    else:
        mesh = create_tiled_plane_2d_dynamic(mesh_index_dtype, real_dtype, offset_x=0.0)

    # EdgeMesh far from mesh (at x from 5 to 8)
    edge_mesh = create_straight_polyline_2d(edge_index_dtype, real_dtype, offset_x=5.0)

    assert not tf.intersects(mesh, edge_mesh)
    assert not tf.intersects(edge_mesh, mesh)


# ==============================================================================
# 3D Mesh-EdgeMesh intersection tests
# ==============================================================================

@pytest.mark.parametrize("mesh_index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("edge_index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
@pytest.mark.parametrize("mesh_type", MESH_TYPES)
def test_mesh_intersects_edge_mesh_3d_hit(mesh_index_dtype, edge_index_dtype, real_dtype, mesh_type):
    """Test mesh and edge mesh that overlap intersect (3D)"""
    if mesh_type == 'triangle':
        mesh = create_tiled_plane_3d_triangles(mesh_index_dtype, real_dtype, offset_z=0.0)
    else:
        mesh = create_tiled_plane_3d_dynamic(mesh_index_dtype, real_dtype, offset_z=0.0)

    # EdgeMesh at same z=0 plane (overlaps with mesh)
    edge_mesh = create_straight_polyline_3d(edge_index_dtype, real_dtype, offset_z=0.0)

    assert tf.intersects(mesh, edge_mesh)
    assert tf.intersects(edge_mesh, mesh)


@pytest.mark.parametrize("mesh_index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("edge_index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
@pytest.mark.parametrize("mesh_type", MESH_TYPES)
def test_mesh_intersects_edge_mesh_3d_miss(mesh_index_dtype, edge_index_dtype, real_dtype, mesh_type):
    """Test non-overlapping mesh and edge mesh don't intersect (3D)"""
    if mesh_type == 'triangle':
        mesh = create_tiled_plane_3d_triangles(mesh_index_dtype, real_dtype, offset_z=0.0)
    else:
        mesh = create_tiled_plane_3d_dynamic(mesh_index_dtype, real_dtype, offset_z=0.0)

    # EdgeMesh far from mesh (at z=5)
    edge_mesh = create_straight_polyline_3d(edge_index_dtype, real_dtype, offset_z=5.0)

    assert not tf.intersects(mesh, edge_mesh)
    assert not tf.intersects(edge_mesh, mesh)


# ==============================================================================
# Transformation tests
# ==============================================================================

@pytest.mark.parametrize("mesh_index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("edge_index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
@pytest.mark.parametrize("mesh_type", MESH_TYPES)
def test_mesh_intersects_edge_mesh_with_transformation_2d(mesh_index_dtype, edge_index_dtype, real_dtype, mesh_type):
    """Test intersects with transformation (transform both mesh and edge mesh)"""
    if mesh_type == 'triangle':
        mesh = create_tiled_plane_2d_triangles(mesh_index_dtype, real_dtype, offset_x=0.0)
    else:
        mesh = create_tiled_plane_2d_dynamic(mesh_index_dtype, real_dtype, offset_x=0.0)

    edge_mesh = create_straight_polyline_2d(edge_index_dtype, real_dtype, offset_x=0.5)

    # Verify intersection before transformation
    assert tf.intersects(mesh, edge_mesh)

    # Apply same transformation to both: translate by [5, 3]
    transformation = np.array([
        [1, 0, 5],
        [0, 1, 3],
        [0, 0, 1]
    ], dtype=real_dtype)

    mesh.transformation = transformation
    edge_mesh.transformation = transformation

    # Should still intersect
    assert tf.intersects(mesh, edge_mesh)
    assert tf.intersects(edge_mesh, mesh)


@pytest.mark.parametrize("mesh_index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("edge_index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
@pytest.mark.parametrize("mesh_type", MESH_TYPES)
def test_mesh_intersects_edge_mesh_with_transformation_3d(mesh_index_dtype, edge_index_dtype, real_dtype, mesh_type):
    """Test intersects with transformation (3D)"""
    if mesh_type == 'triangle':
        mesh = create_tiled_plane_3d_triangles(mesh_index_dtype, real_dtype, offset_z=0.0)
    else:
        mesh = create_tiled_plane_3d_dynamic(mesh_index_dtype, real_dtype, offset_z=0.0)

    edge_mesh = create_straight_polyline_3d(edge_index_dtype, real_dtype, offset_z=0.0)

    # Verify intersection before transformation
    assert tf.intersects(mesh, edge_mesh)

    # Apply same transformation to both: translate by [10, 5, 2]
    transformation = np.array([
        [1, 0, 0, 10],
        [0, 1, 0,  5],
        [0, 0, 1,  2],
        [0, 0, 0,  1]
    ], dtype=real_dtype)

    mesh.transformation = transformation
    edge_mesh.transformation = transformation

    # Should still intersect
    assert tf.intersects(mesh, edge_mesh)
    assert tf.intersects(edge_mesh, mesh)


# ==============================================================================
# Different transformation tests
# ==============================================================================

@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_mesh_intersects_edge_mesh_different_transformations_2d(real_dtype):
    """Test that different transformations can move forms apart"""
    mesh = create_tiled_plane_2d_triangles(np.int32, real_dtype, offset_x=0.0)
    edge_mesh = create_straight_polyline_2d(np.int32, real_dtype, offset_x=0.5)

    # Verify intersection before transformation
    assert tf.intersects(mesh, edge_mesh)

    # Apply different transformations
    # mesh: no transformation
    # edge_mesh: translate far away
    transformation = np.array([
        [1, 0, 100],
        [0, 1, 100],
        [0, 0, 1]
    ], dtype=real_dtype)

    edge_mesh.transformation = transformation

    # Should NOT intersect anymore
    assert not tf.intersects(mesh, edge_mesh)
    assert not tf.intersects(edge_mesh, mesh)


# ==============================================================================
# Dimension mismatch tests
# ==============================================================================

def test_mesh_intersects_edge_mesh_dimension_mismatch():
    """Test that dimension mismatch raises error"""
    mesh_2d = create_tiled_plane_2d_triangles(np.int32, np.float32)
    edge_mesh_3d = create_straight_polyline_3d(np.int32, np.float32)

    with pytest.raises(ValueError, match="Dimension mismatch"):
        tf.intersects(mesh_2d, edge_mesh_3d)

    with pytest.raises(ValueError, match="Dimension mismatch"):
        tf.intersects(edge_mesh_3d, mesh_2d)


# ==============================================================================
# Real type mismatch tests
# ==============================================================================

def test_mesh_intersects_edge_mesh_real_type_mismatch():
    """Test that real type mismatch raises error"""
    mesh_float = create_tiled_plane_2d_triangles(np.int32, np.float32)
    edge_mesh_double = create_straight_polyline_2d(np.int32, np.float64)

    # This should raise an error because real types don't match
    with pytest.raises((TypeError, AttributeError)):
        tf.intersects(mesh_float, edge_mesh_double)


# ==============================================================================
# Edge cases
# ==============================================================================

@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
@pytest.mark.parametrize("mesh_type", MESH_TYPES)
def test_mesh_intersects_edge_mesh_edge_on_face(real_dtype, mesh_type):
    """Test edge mesh lying on mesh face"""
    if mesh_type == 'triangle':
        mesh = create_tiled_plane_2d_triangles(np.int32, real_dtype, offset_x=0.0)
    else:
        mesh = create_tiled_plane_2d_dynamic(np.int32, real_dtype, offset_x=0.0)

    # EdgeMesh at y=0 (on the bottom edge of the mesh)
    edge_mesh = create_straight_polyline_2d(np.int32, real_dtype, offset_x=0.5)

    # Should intersect
    result = tf.intersects(mesh, edge_mesh)
    assert isinstance(result, (bool, np.bool_))


if __name__ == "__main__":
    # Run tests with verbose output
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
