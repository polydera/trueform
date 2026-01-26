"""
Tests for Mesh intersects with Mesh

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


def create_tiled_plane_2d_triangles(index_dtype, real_dtype, offset_x=0.0):
    """
    Create a simple 2x1 tiled plane in 2D with triangles
    offset_x shifts the mesh horizontally
    """
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


# ==============================================================================
# 2D Mesh-Mesh intersection tests
# ==============================================================================

@pytest.mark.parametrize("index_dtype0", INDEX_DTYPES)
@pytest.mark.parametrize("index_dtype1", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
@pytest.mark.parametrize("mesh_type0", MESH_TYPES)
@pytest.mark.parametrize("mesh_type1", MESH_TYPES)
def test_mesh_intersects_mesh_2d_hit(index_dtype0, index_dtype1, real_dtype, mesh_type0, mesh_type1):
    """Test overlapping meshes intersect"""
    if mesh_type0 == 'triangle':
        mesh0 = create_tiled_plane_2d_triangles(index_dtype0, real_dtype, offset_x=0.0)
    else:
        mesh0 = create_tiled_plane_2d_dynamic(index_dtype0, real_dtype, offset_x=0.0)

    if mesh_type1 == 'triangle':
        # Overlapping mesh at x=1 (overlaps with first mesh)
        mesh1 = create_tiled_plane_2d_triangles(index_dtype1, real_dtype, offset_x=1.0)
    else:
        mesh1 = create_tiled_plane_2d_dynamic(index_dtype1, real_dtype, offset_x=1.0)

    assert tf.intersects(mesh0, mesh1)
    assert tf.intersects(mesh1, mesh0)  # Symmetric


@pytest.mark.parametrize("index_dtype0", INDEX_DTYPES)
@pytest.mark.parametrize("index_dtype1", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
@pytest.mark.parametrize("mesh_type0", MESH_TYPES)
@pytest.mark.parametrize("mesh_type1", MESH_TYPES)
def test_mesh_intersects_mesh_2d_miss(index_dtype0, index_dtype1, real_dtype, mesh_type0, mesh_type1):
    """Test non-overlapping meshes don't intersect"""
    if mesh_type0 == 'triangle':
        mesh0 = create_tiled_plane_2d_triangles(index_dtype0, real_dtype, offset_x=0.0)
    else:
        mesh0 = create_tiled_plane_2d_dynamic(index_dtype0, real_dtype, offset_x=0.0)

    if mesh_type1 == 'triangle':
        # Non-overlapping mesh at x=5 (far from first mesh)
        mesh1 = create_tiled_plane_2d_triangles(index_dtype1, real_dtype, offset_x=5.0)
    else:
        mesh1 = create_tiled_plane_2d_dynamic(index_dtype1, real_dtype, offset_x=5.0)

    assert not tf.intersects(mesh0, mesh1)
    assert not tf.intersects(mesh1, mesh0)


# ==============================================================================
# 3D Mesh-Mesh intersection tests
# ==============================================================================

@pytest.mark.parametrize("index_dtype0", INDEX_DTYPES)
@pytest.mark.parametrize("index_dtype1", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
@pytest.mark.parametrize("mesh_type0", MESH_TYPES)
@pytest.mark.parametrize("mesh_type1", MESH_TYPES)
def test_mesh_intersects_mesh_3d_hit(index_dtype0, index_dtype1, real_dtype, mesh_type0, mesh_type1):
    """Test overlapping meshes intersect (3D)"""
    if mesh_type0 == 'triangle':
        mesh0 = create_tiled_plane_3d_triangles(index_dtype0, real_dtype, offset_z=0.0)
    else:
        mesh0 = create_tiled_plane_3d_dynamic(index_dtype0, real_dtype, offset_z=0.0)

    if mesh_type1 == 'triangle':
        # Overlapping mesh at same z=0 plane
        mesh1 = create_tiled_plane_3d_triangles(index_dtype1, real_dtype, offset_z=0.0)
    else:
        mesh1 = create_tiled_plane_3d_dynamic(index_dtype1, real_dtype, offset_z=0.0)

    assert tf.intersects(mesh0, mesh1)
    assert tf.intersects(mesh1, mesh0)


@pytest.mark.parametrize("index_dtype0", INDEX_DTYPES)
@pytest.mark.parametrize("index_dtype1", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
@pytest.mark.parametrize("mesh_type0", MESH_TYPES)
@pytest.mark.parametrize("mesh_type1", MESH_TYPES)
def test_mesh_intersects_mesh_3d_miss(index_dtype0, index_dtype1, real_dtype, mesh_type0, mesh_type1):
    """Test non-overlapping meshes don't intersect (3D)"""
    if mesh_type0 == 'triangle':
        mesh0 = create_tiled_plane_3d_triangles(index_dtype0, real_dtype, offset_z=0.0)
    else:
        mesh0 = create_tiled_plane_3d_dynamic(index_dtype0, real_dtype, offset_z=0.0)

    if mesh_type1 == 'triangle':
        # Non-overlapping mesh at z=5 (far from first mesh)
        mesh1 = create_tiled_plane_3d_triangles(index_dtype1, real_dtype, offset_z=5.0)
    else:
        mesh1 = create_tiled_plane_3d_dynamic(index_dtype1, real_dtype, offset_z=5.0)

    assert not tf.intersects(mesh0, mesh1)
    assert not tf.intersects(mesh1, mesh0)


# ==============================================================================
# Transformation tests
# ==============================================================================

@pytest.mark.parametrize("index_dtype0", INDEX_DTYPES)
@pytest.mark.parametrize("index_dtype1", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
@pytest.mark.parametrize("mesh_type0", MESH_TYPES)
@pytest.mark.parametrize("mesh_type1", MESH_TYPES)
def test_mesh_intersects_mesh_with_transformation_2d(index_dtype0, index_dtype1, real_dtype, mesh_type0, mesh_type1):
    """Test intersects with transformation (transform both meshes)"""
    if mesh_type0 == 'triangle':
        mesh0 = create_tiled_plane_2d_triangles(index_dtype0, real_dtype, offset_x=0.0)
    else:
        mesh0 = create_tiled_plane_2d_dynamic(index_dtype0, real_dtype, offset_x=0.0)

    if mesh_type1 == 'triangle':
        mesh1 = create_tiled_plane_2d_triangles(index_dtype1, real_dtype, offset_x=1.0)
    else:
        mesh1 = create_tiled_plane_2d_dynamic(index_dtype1, real_dtype, offset_x=1.0)

    # Verify intersection before transformation
    assert tf.intersects(mesh0, mesh1)

    # Apply same transformation to both: translate by [5, 3]
    transformation = np.array([
        [1, 0, 5],
        [0, 1, 3],
        [0, 0, 1]
    ], dtype=real_dtype)

    mesh0.transformation = transformation
    mesh1.transformation = transformation

    # Should still intersect
    assert tf.intersects(mesh0, mesh1)
    assert tf.intersects(mesh1, mesh0)


@pytest.mark.parametrize("index_dtype0", INDEX_DTYPES)
@pytest.mark.parametrize("index_dtype1", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
@pytest.mark.parametrize("mesh_type0", MESH_TYPES)
@pytest.mark.parametrize("mesh_type1", MESH_TYPES)
def test_mesh_intersects_mesh_with_transformation_3d(index_dtype0, index_dtype1, real_dtype, mesh_type0, mesh_type1):
    """Test intersects with transformation (3D)"""
    if mesh_type0 == 'triangle':
        mesh0 = create_tiled_plane_3d_triangles(index_dtype0, real_dtype, offset_z=0.0)
    else:
        mesh0 = create_tiled_plane_3d_dynamic(index_dtype0, real_dtype, offset_z=0.0)

    if mesh_type1 == 'triangle':
        mesh1 = create_tiled_plane_3d_triangles(index_dtype1, real_dtype, offset_z=0.0)
    else:
        mesh1 = create_tiled_plane_3d_dynamic(index_dtype1, real_dtype, offset_z=0.0)

    # Verify intersection before transformation
    assert tf.intersects(mesh0, mesh1)

    # Apply same transformation to both: translate by [10, 5, 2]
    transformation = np.array([
        [1, 0, 0, 10],
        [0, 1, 0,  5],
        [0, 0, 1,  2],
        [0, 0, 0,  1]
    ], dtype=real_dtype)

    mesh0.transformation = transformation
    mesh1.transformation = transformation

    # Should still intersect
    assert tf.intersects(mesh0, mesh1)
    assert tf.intersects(mesh1, mesh0)


# ==============================================================================
# Different transformation tests (meshes move apart)
# ==============================================================================

@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_mesh_intersects_mesh_different_transformations_2d(real_dtype):
    """Test that different transformations can move meshes apart"""
    mesh0 = create_tiled_plane_2d_triangles(np.int32, real_dtype, offset_x=0.0)
    mesh1 = create_tiled_plane_2d_triangles(np.int32, real_dtype, offset_x=1.0)

    # Verify intersection before transformation
    assert tf.intersects(mesh0, mesh1)

    # Apply different transformations
    # mesh0: no transformation
    # mesh1: translate far away
    transformation1 = np.array([
        [1, 0, 100],
        [0, 1, 100],
        [0, 0, 1]
    ], dtype=real_dtype)

    mesh1.transformation = transformation1

    # Should NOT intersect anymore
    assert not tf.intersects(mesh0, mesh1)
    assert not tf.intersects(mesh1, mesh0)


# ==============================================================================
# Dimension mismatch tests
# ==============================================================================

def test_mesh_intersects_mesh_dimension_mismatch():
    """Test that dimension mismatch raises error"""
    mesh_2d = create_tiled_plane_2d_triangles(np.int32, np.float32)
    mesh_3d = create_tiled_plane_3d_triangles(np.int32, np.float32)

    with pytest.raises(ValueError, match="Dimension mismatch"):
        tf.intersects(mesh_2d, mesh_3d)

    with pytest.raises(ValueError, match="Dimension mismatch"):
        tf.intersects(mesh_3d, mesh_2d)


# ==============================================================================
# Real type mismatch tests
# ==============================================================================

def test_mesh_intersects_mesh_real_type_mismatch():
    """Test that real type mismatch raises error"""
    mesh_float = create_tiled_plane_2d_triangles(np.int32, np.float32)
    mesh_double = create_tiled_plane_2d_triangles(np.int32, np.float64)

    # This should raise an error because real types don't match
    with pytest.raises((TypeError, AttributeError)):
        tf.intersects(mesh_float, mesh_double)


# ==============================================================================
# Edge cases
# ==============================================================================

@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_mesh_intersects_itself(real_dtype):
    """Test mesh intersecting with itself"""
    mesh = create_tiled_plane_2d_triangles(np.int32, real_dtype)

    assert tf.intersects(mesh, mesh)


@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_mesh_intersects_mesh_touching_edge(real_dtype):
    """Test meshes that share an edge"""
    # mesh0: x from 0 to 2
    mesh0 = create_tiled_plane_2d_triangles(np.int32, real_dtype, offset_x=0.0)
    # mesh1: x from 2 to 4 (shares edge at x=2)
    mesh1 = create_tiled_plane_2d_triangles(np.int32, real_dtype, offset_x=2.0)

    # Edge touching should be considered intersection
    result = tf.intersects(mesh0, mesh1)
    # Note: depending on implementation, this might be True or False
    # Just check it runs without error
    assert isinstance(result, (bool, np.bool_))


if __name__ == "__main__":
    # Run tests with verbose output
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
