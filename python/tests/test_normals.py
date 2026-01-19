"""
Test normals and point_normals functionality

Copyright (c) 2025 Ziga Sajovic, XLAB
"""

import sys

import pytest
import numpy as np
import trueform as tf

# Parameter sets
INDEX_DTYPES = [np.int32, np.int64]
REAL_DTYPES = [np.float32, np.float64]


# ==============================================================================
# Test data generators
# ==============================================================================

def create_triangle_mesh_3d(index_dtype, real_dtype):
    """Create a simple 3D triangle mesh (2 triangles sharing an edge)."""
    faces = np.array([
        [0, 1, 2],
        [1, 3, 2]
    ], dtype=index_dtype)
    points = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 1.0, 0.0],
        [1.5, 1.0, 0.0]
    ], dtype=real_dtype)
    return faces, points


def create_triangle_mesh_2d(index_dtype, real_dtype):
    """Create a simple 2D triangle mesh."""
    faces = np.array([
        [0, 1, 2],
        [1, 3, 2]
    ], dtype=index_dtype)
    points = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, 1.0],
        [1.5, 1.0]
    ], dtype=real_dtype)
    return faces, points


def create_dynamic_mesh_3d(index_dtype, real_dtype):
    """Create a dynamic 3D mesh with mixed polygon sizes (triangle + quad)."""
    offsets = np.array([0, 3, 7], dtype=index_dtype)
    data = np.array([0, 1, 2, 1, 3, 4, 2], dtype=index_dtype)
    faces = tf.OffsetBlockedArray(offsets, data)
    points = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 1.0, 0.0],
        [2.0, 0.0, 0.0],
        [1.5, 1.0, 0.0]
    ], dtype=real_dtype)
    return faces, points


def create_cube_mesh(index_dtype, real_dtype):
    """Create a cube mesh for testing normals pointing outward."""
    # Simple cube with 8 vertices and 12 triangles (2 per face)
    points = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],  # bottom face z=0
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],  # top face z=1
    ], dtype=real_dtype)
    # CCW winding for outward normals
    faces = np.array([
        # bottom (z=0, normal pointing -z)
        [0, 2, 1], [0, 3, 2],
        # top (z=1, normal pointing +z)
        [4, 5, 6], [4, 6, 7],
        # front (y=0, normal pointing -y)
        [0, 1, 5], [0, 5, 4],
        # back (y=1, normal pointing +y)
        [2, 3, 7], [2, 7, 6],
        # left (x=0, normal pointing -x)
        [0, 4, 7], [0, 7, 3],
        # right (x=1, normal pointing +x)
        [1, 2, 6], [1, 6, 5],
    ], dtype=index_dtype)
    return faces, points


# ==============================================================================
# normals Tests - Basic Functionality
# ==============================================================================

@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_normals_basic(index_dtype, real_dtype):
    """Test normals computation for simple 3D triangle mesh."""
    faces, points = create_triangle_mesh_3d(index_dtype, real_dtype)
    mesh = tf.Mesh(faces, points)

    normals = mesh.normals

    # Should have one normal per face
    assert normals.shape == (len(faces), 3)
    assert normals.dtype == real_dtype


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_normals_unit_vectors(index_dtype, real_dtype):
    """Test that normals are unit vectors."""
    faces, points = create_triangle_mesh_3d(index_dtype, real_dtype)
    mesh = tf.Mesh(faces, points)

    normals = mesh.normals

    # Check that each normal has unit length
    lengths = np.linalg.norm(normals, axis=1)
    np.testing.assert_allclose(lengths, 1.0, atol=1e-6)


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_normals_planar_faces(index_dtype, real_dtype):
    """Test normals for coplanar triangles (all in xy-plane)."""
    faces, points = create_triangle_mesh_3d(index_dtype, real_dtype)
    mesh = tf.Mesh(faces, points)

    normals = mesh.normals

    # Both triangles lie in xy-plane, so normals should be +/- z
    for i in range(len(faces)):
        # Normal should be parallel to z-axis
        assert abs(normals[i, 0]) < 1e-6, "Normal x component should be ~0"
        assert abs(normals[i, 1]) < 1e-6, "Normal y component should be ~0"
        assert abs(abs(normals[i, 2]) - 1.0) < 1e-6, "Normal z component should be ~1"


# ==============================================================================
# point_normals Tests - Basic Functionality
# ==============================================================================

@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_point_normals_basic(index_dtype, real_dtype):
    """Test point normals computation for simple 3D triangle mesh."""
    faces, points = create_triangle_mesh_3d(index_dtype, real_dtype)
    mesh = tf.Mesh(faces, points)

    point_normals = mesh.point_normals

    # Should have one normal per vertex
    assert point_normals.shape == (len(points), 3)
    assert point_normals.dtype == real_dtype


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_point_normals_unit_vectors(index_dtype, real_dtype):
    """Test that point normals are unit vectors."""
    faces, points = create_triangle_mesh_3d(index_dtype, real_dtype)
    mesh = tf.Mesh(faces, points)

    point_normals = mesh.point_normals

    # Check that each normal has unit length
    lengths = np.linalg.norm(point_normals, axis=1)
    np.testing.assert_allclose(lengths, 1.0, atol=1e-6)


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_point_normals_planar_mesh(index_dtype, real_dtype):
    """Test point normals for coplanar triangles."""
    faces, points = create_triangle_mesh_3d(index_dtype, real_dtype)
    mesh = tf.Mesh(faces, points)

    point_normals = mesh.point_normals

    # All vertices are in xy-plane, so all normals should be +/- z
    for i in range(len(points)):
        assert abs(point_normals[i, 0]) < 1e-6
        assert abs(point_normals[i, 1]) < 1e-6
        assert abs(abs(point_normals[i, 2]) - 1.0) < 1e-6


# ==============================================================================
# Dynamic Mesh Tests
# ==============================================================================

@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_normals_dynamic_mesh(index_dtype, real_dtype):
    """Test normals for dynamic (variable polygon size) 3D mesh."""
    faces, points = create_dynamic_mesh_3d(index_dtype, real_dtype)
    mesh = tf.Mesh(faces, points)

    normals = mesh.normals

    # Should have one normal per face
    assert normals.shape == (len(faces), 3)
    assert normals.dtype == real_dtype


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_point_normals_dynamic_mesh(index_dtype, real_dtype):
    """Test point normals for dynamic (variable polygon size) 3D mesh."""
    faces, points = create_dynamic_mesh_3d(index_dtype, real_dtype)
    mesh = tf.Mesh(faces, points)

    point_normals = mesh.point_normals

    # Should have one normal per vertex
    assert point_normals.shape == (len(points), 3)
    assert point_normals.dtype == real_dtype


# ==============================================================================
# Cube Mesh Tests - Verify Correct Direction
# ==============================================================================

@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_normals_cube(index_dtype, real_dtype):
    """Test normals for cube mesh have correct directions."""
    faces, points = create_cube_mesh(index_dtype, real_dtype)
    mesh = tf.Mesh(faces, points)

    normals = mesh.normals

    assert normals.shape == (12, 3)

    # Bottom face (faces 0,1): normal should point -z
    for i in [0, 1]:
        np.testing.assert_allclose(normals[i], [0, 0, -1], atol=1e-6)

    # Top face (faces 2,3): normal should point +z
    for i in [2, 3]:
        np.testing.assert_allclose(normals[i], [0, 0, 1], atol=1e-6)


# ==============================================================================
# Setter Tests
# ==============================================================================

@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_normals_setter(index_dtype, real_dtype):
    """Test setting normals on mesh."""
    faces, points = create_triangle_mesh_3d(index_dtype, real_dtype)
    mesh = tf.Mesh(faces, points)

    # Create custom normals
    custom_normals = np.array([
        [0, 0, 1],
        [0, 0, 1]
    ], dtype=real_dtype)

    mesh.normals = custom_normals

    # Get them back
    retrieved = mesh.normals

    np.testing.assert_allclose(retrieved, custom_normals)


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_point_normals_setter(index_dtype, real_dtype):
    """Test setting point normals on mesh."""
    faces, points = create_triangle_mesh_3d(index_dtype, real_dtype)
    mesh = tf.Mesh(faces, points)

    # Create custom point normals
    custom_normals = np.array([
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 1]
    ], dtype=real_dtype)

    mesh.point_normals = custom_normals

    # Get them back
    retrieved = mesh.point_normals

    np.testing.assert_allclose(retrieved, custom_normals)


# ==============================================================================
# Lazy Computation Tests
# ==============================================================================

@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_normals_lazy_build(index_dtype, real_dtype):
    """Test that normals are lazily computed on first access."""
    faces, points = create_triangle_mesh_3d(index_dtype, real_dtype)
    mesh = tf.Mesh(faces, points)

    # First access triggers build
    normals1 = mesh.normals
    # Second access should return cached result
    normals2 = mesh.normals

    # Should be the same array (or at least equal)
    np.testing.assert_array_equal(normals1, normals2)


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_point_normals_lazy_build(index_dtype, real_dtype):
    """Test that point normals are lazily computed on first access."""
    faces, points = create_triangle_mesh_3d(index_dtype, real_dtype)
    mesh = tf.Mesh(faces, points)

    # First access triggers build
    pn1 = mesh.point_normals
    # Second access should return cached result
    pn2 = mesh.point_normals

    np.testing.assert_array_equal(pn1, pn2)


# ==============================================================================
# 2D Mesh Error Tests
# ==============================================================================

@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_normals_2d_mesh_raises(index_dtype, real_dtype):
    """Test that accessing normals on 2D mesh raises ValueError."""
    faces, points = create_triangle_mesh_2d(index_dtype, real_dtype)
    mesh = tf.Mesh(faces, points)

    with pytest.raises(ValueError, match="Normals only supported for 3D meshes"):
        _ = mesh.normals


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_point_normals_2d_mesh_raises(index_dtype, real_dtype):
    """Test that accessing point_normals on 2D mesh raises ValueError."""
    faces, points = create_triangle_mesh_2d(index_dtype, real_dtype)
    mesh = tf.Mesh(faces, points)

    with pytest.raises(ValueError, match="Point normals only supported for 3D meshes"):
        _ = mesh.point_normals


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_build_normals_2d_mesh_raises(index_dtype, real_dtype):
    """Test that build_normals on 2D mesh raises ValueError."""
    faces, points = create_triangle_mesh_2d(index_dtype, real_dtype)
    mesh = tf.Mesh(faces, points)

    with pytest.raises(ValueError, match="Normals only supported for 3D meshes"):
        mesh.build_normals()


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_build_point_normals_2d_mesh_raises(index_dtype, real_dtype):
    """Test that build_point_normals on 2D mesh raises ValueError."""
    faces, points = create_triangle_mesh_2d(index_dtype, real_dtype)
    mesh = tf.Mesh(faces, points)

    with pytest.raises(ValueError, match="Point normals only supported for 3D meshes"):
        mesh.build_point_normals()


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_set_normals_2d_mesh_raises(index_dtype, real_dtype):
    """Test that setting normals on 2D mesh raises ValueError."""
    faces, points = create_triangle_mesh_2d(index_dtype, real_dtype)
    mesh = tf.Mesh(faces, points)

    custom_normals = np.array([[0, 0, 1], [0, 0, 1]], dtype=real_dtype)

    with pytest.raises(ValueError, match="Normals only supported for 3D meshes"):
        mesh.normals = custom_normals


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_set_point_normals_2d_mesh_raises(index_dtype, real_dtype):
    """Test that setting point_normals on 2D mesh raises ValueError."""
    faces, points = create_triangle_mesh_2d(index_dtype, real_dtype)
    mesh = tf.Mesh(faces, points)

    custom_normals = np.array([[0, 0, 1]] * 4, dtype=real_dtype)

    with pytest.raises(ValueError, match="Point normals only supported for 3D meshes"):
        mesh.point_normals = custom_normals


# ==============================================================================
# Main runner
# ==============================================================================

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
