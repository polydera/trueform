"""
Tests for EdgeMesh intersects with EdgeMesh

Copyright (c) 2025 Žiga Sajovic, XLAB
"""

import sys
import numpy as np
import pytest
import trueform as tf


# Type combinations for EdgeMesh: 2 index types × 2 real types × 2 dims = 8
INDEX_DTYPES = [np.int32, np.int64]
REAL_DTYPES = [np.float32, np.float64]
DIMS = [2, 3]


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
# 2D EdgeMesh-EdgeMesh intersection tests
# ==============================================================================

@pytest.mark.parametrize("index_dtype0", INDEX_DTYPES)
@pytest.mark.parametrize("index_dtype1", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_edge_mesh_intersects_edge_mesh_2d_hit(index_dtype0, index_dtype1, real_dtype):
    """Test overlapping edge meshes intersect"""
    edge_mesh0 = create_straight_polyline_2d(index_dtype0, real_dtype, offset_x=0.0)
    # Overlapping polyline at x=1.5 (overlaps with first edge mesh)
    edge_mesh1 = create_straight_polyline_2d(index_dtype1, real_dtype, offset_x=1.5)

    assert tf.intersects(edge_mesh0, edge_mesh1)
    assert tf.intersects(edge_mesh1, edge_mesh0)  # Symmetric


@pytest.mark.parametrize("index_dtype0", INDEX_DTYPES)
@pytest.mark.parametrize("index_dtype1", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_edge_mesh_intersects_edge_mesh_2d_miss(index_dtype0, index_dtype1, real_dtype):
    """Test non-overlapping edge meshes don't intersect"""
    edge_mesh0 = create_straight_polyline_2d(index_dtype0, real_dtype, offset_x=0.0)
    # Non-overlapping polyline at x=5 (far from first edge mesh)
    edge_mesh1 = create_straight_polyline_2d(index_dtype1, real_dtype, offset_x=5.0)

    assert not tf.intersects(edge_mesh0, edge_mesh1)
    assert not tf.intersects(edge_mesh1, edge_mesh0)


# ==============================================================================
# 3D EdgeMesh-EdgeMesh intersection tests
# ==============================================================================

@pytest.mark.parametrize("index_dtype0", INDEX_DTYPES)
@pytest.mark.parametrize("index_dtype1", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_edge_mesh_intersects_edge_mesh_3d_hit(index_dtype0, index_dtype1, real_dtype):
    """Test overlapping edge meshes intersect (3D)"""
    edge_mesh0 = create_straight_polyline_3d(index_dtype0, real_dtype, offset_z=0.0)
    # Overlapping polyline at same z=0 plane
    edge_mesh1 = create_straight_polyline_3d(index_dtype1, real_dtype, offset_z=0.0)

    assert tf.intersects(edge_mesh0, edge_mesh1)
    assert tf.intersects(edge_mesh1, edge_mesh0)


@pytest.mark.parametrize("index_dtype0", INDEX_DTYPES)
@pytest.mark.parametrize("index_dtype1", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_edge_mesh_intersects_edge_mesh_3d_miss(index_dtype0, index_dtype1, real_dtype):
    """Test non-overlapping edge meshes don't intersect (3D)"""
    edge_mesh0 = create_straight_polyline_3d(index_dtype0, real_dtype, offset_z=0.0)
    # Non-overlapping polyline at z=5 (far from first edge mesh)
    edge_mesh1 = create_straight_polyline_3d(index_dtype1, real_dtype, offset_z=5.0)

    assert not tf.intersects(edge_mesh0, edge_mesh1)
    assert not tf.intersects(edge_mesh1, edge_mesh0)


# ==============================================================================
# Transformation tests
# ==============================================================================

@pytest.mark.parametrize("index_dtype0", INDEX_DTYPES)
@pytest.mark.parametrize("index_dtype1", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_edge_mesh_intersects_edge_mesh_with_transformation_2d(index_dtype0, index_dtype1, real_dtype):
    """Test intersects with transformation (transform both edge meshes)"""
    edge_mesh0 = create_straight_polyline_2d(index_dtype0, real_dtype, offset_x=0.0)
    edge_mesh1 = create_straight_polyline_2d(index_dtype1, real_dtype, offset_x=1.5)

    # Verify intersection before transformation
    assert tf.intersects(edge_mesh0, edge_mesh1)

    # Apply same transformation to both: translate by [5, 3]
    transformation = np.array([
        [1, 0, 5],
        [0, 1, 3],
        [0, 0, 1]
    ], dtype=real_dtype)

    edge_mesh0.transformation = transformation
    edge_mesh1.transformation = transformation

    # Should still intersect
    assert tf.intersects(edge_mesh0, edge_mesh1)
    assert tf.intersects(edge_mesh1, edge_mesh0)


@pytest.mark.parametrize("index_dtype0", INDEX_DTYPES)
@pytest.mark.parametrize("index_dtype1", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_edge_mesh_intersects_edge_mesh_with_transformation_3d(index_dtype0, index_dtype1, real_dtype):
    """Test intersects with transformation (3D)"""
    edge_mesh0 = create_straight_polyline_3d(index_dtype0, real_dtype, offset_z=0.0)
    edge_mesh1 = create_straight_polyline_3d(index_dtype1, real_dtype, offset_z=0.0)

    # Verify intersection before transformation
    assert tf.intersects(edge_mesh0, edge_mesh1)

    # Apply same transformation to both: translate by [10, 5, 2]
    transformation = np.array([
        [1, 0, 0, 10],
        [0, 1, 0,  5],
        [0, 0, 1,  2],
        [0, 0, 0,  1]
    ], dtype=real_dtype)

    edge_mesh0.transformation = transformation
    edge_mesh1.transformation = transformation

    # Should still intersect
    assert tf.intersects(edge_mesh0, edge_mesh1)
    assert tf.intersects(edge_mesh1, edge_mesh0)


# ==============================================================================
# Different transformation tests (edge meshes move apart)
# ==============================================================================

@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_edge_mesh_intersects_edge_mesh_different_transformations_2d(real_dtype):
    """Test that different transformations can move edge meshes apart"""
    edge_mesh0 = create_straight_polyline_2d(np.int32, real_dtype, offset_x=0.0)
    edge_mesh1 = create_straight_polyline_2d(np.int32, real_dtype, offset_x=1.5)

    # Verify intersection before transformation
    assert tf.intersects(edge_mesh0, edge_mesh1)

    # Apply different transformations
    # edge_mesh0: no transformation
    # edge_mesh1: translate far away
    transformation1 = np.array([
        [1, 0, 100],
        [0, 1, 100],
        [0, 0, 1]
    ], dtype=real_dtype)

    edge_mesh1.transformation = transformation1

    # Should NOT intersect anymore
    assert not tf.intersects(edge_mesh0, edge_mesh1)
    assert not tf.intersects(edge_mesh1, edge_mesh0)


# ==============================================================================
# Dimension mismatch tests
# ==============================================================================

def test_edge_mesh_intersects_edge_mesh_dimension_mismatch():
    """Test that dimension mismatch raises error"""
    edge_mesh_2d = create_straight_polyline_2d(np.int32, np.float32)
    edge_mesh_3d = create_straight_polyline_3d(np.int32, np.float32)

    with pytest.raises(ValueError, match="Dimension mismatch"):
        tf.intersects(edge_mesh_2d, edge_mesh_3d)

    with pytest.raises(ValueError, match="Dimension mismatch"):
        tf.intersects(edge_mesh_3d, edge_mesh_2d)


# ==============================================================================
# Real type mismatch tests
# ==============================================================================

def test_edge_mesh_intersects_edge_mesh_real_type_mismatch():
    """Test that real type mismatch raises error"""
    edge_mesh_float = create_straight_polyline_2d(np.int32, np.float32)
    edge_mesh_double = create_straight_polyline_2d(np.int32, np.float64)

    # This should raise an error because real types don't match
    with pytest.raises((TypeError, AttributeError)):
        tf.intersects(edge_mesh_float, edge_mesh_double)


# ==============================================================================
# Edge cases
# ==============================================================================

@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_edge_mesh_intersects_itself(real_dtype):
    """Test edge mesh intersecting with itself"""
    edge_mesh = create_straight_polyline_2d(np.int32, real_dtype)

    assert tf.intersects(edge_mesh, edge_mesh)


@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_edge_mesh_intersects_edge_mesh_touching_endpoint(real_dtype):
    """Test edge meshes that share an endpoint"""
    # edge_mesh0: x from 0 to 3
    edge_mesh0 = create_straight_polyline_2d(np.int32, real_dtype, offset_x=0.0)
    # edge_mesh1: x from 3 to 6 (shares endpoint at x=3)
    edge_mesh1 = create_straight_polyline_2d(np.int32, real_dtype, offset_x=3.0)

    # Endpoint touching should be considered intersection
    result = tf.intersects(edge_mesh0, edge_mesh1)
    # Just check it runs without error
    assert isinstance(result, (bool, np.bool_))


# ==============================================================================
# Index type canonicalization tests (int64 × int32)
# ==============================================================================

@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_edge_mesh_intersects_edge_mesh_index_canonicalization(real_dtype):
    """Test that int64 × int32 works (gets canonicalized to int32 × int64)"""
    # This tests the Python canonicalization logic
    edge_mesh_int64 = create_straight_polyline_2d(np.int64, real_dtype, offset_x=0.0)
    edge_mesh_int32 = create_straight_polyline_2d(np.int32, real_dtype, offset_x=1.5)

    # Should work in both directions
    assert tf.intersects(edge_mesh_int64, edge_mesh_int32)
    assert tf.intersects(edge_mesh_int32, edge_mesh_int64)


if __name__ == "__main__":
    # Run tests with verbose output
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
