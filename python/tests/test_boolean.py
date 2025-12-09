"""
Tests for boolean operations on meshes

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via ziga.sajovic@xlab.si.
https://github.com/xlabmedical/trueform
"""

import sys
import os

# Add parent directory to path so we can import trueform
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
import trueform as tf


# Type combinations for boolean operations (3D only)
INDEX_DTYPES = [np.int32, np.int64]
REAL_DTYPES = [np.float32, np.float64]
MESH_TYPES = ['triangle', 'dynamic']


# ==============================================================================
# Helper functions for creating test meshes
# ==============================================================================

def create_cube(index_dtype, real_dtype, center, size=1.0, mesh_type='triangle'):
    """Create a cube mesh with specified center and size"""
    half = size / 2.0
    cx, cy, cz = center

    # 8 vertices
    points = np.array([
        [cx - half, cy - half, cz - half],  # 0
        [cx + half, cy - half, cz - half],  # 1
        [cx + half, cy + half, cz - half],  # 2
        [cx - half, cy + half, cz - half],  # 3
        [cx - half, cy - half, cz + half],  # 4
        [cx + half, cy - half, cz + half],  # 5
        [cx + half, cy + half, cz + half],  # 6
        [cx - half, cy + half, cz + half],  # 7
    ], dtype=real_dtype)

    # 12 triangles (2 per face)
    faces_data = np.array([
        # Bottom (z-)
        [0, 1, 2], [0, 2, 3],
        # Top (z+)
        [4, 7, 6], [4, 6, 5],
        # Front (y-)
        [0, 5, 1], [0, 4, 5],
        # Back (y+)
        [2, 7, 3], [2, 6, 7],
        # Left (x-)
        [0, 3, 7], [0, 7, 4],
        # Right (x+)
        [1, 6, 2], [1, 5, 6],
    ], dtype=index_dtype)

    if mesh_type == 'dynamic':
        # Use OffsetBlockedArray for dynamic mesh
        offsets = np.arange(0, len(faces_data) * 3 + 1, 3, dtype=index_dtype)
        data = faces_data.ravel()
        faces = tf.OffsetBlockedArray(offsets, data)
    else:
        faces = faces_data

    return tf.Mesh(faces, points)


# ==============================================================================
# Basic functionality tests
# ==============================================================================

@pytest.mark.parametrize("index_dtype0", INDEX_DTYPES)
@pytest.mark.parametrize("index_dtype1", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
@pytest.mark.parametrize("mesh_type0", MESH_TYPES)
@pytest.mark.parametrize("mesh_type1", MESH_TYPES)
def test_boolean_union_basic(index_dtype0, index_dtype1, real_dtype, mesh_type0, mesh_type1):
    """Test basic union of two cubes"""
    # Create two overlapping cubes
    mesh0 = create_cube(index_dtype0, real_dtype, center=(0, 0, 0), size=1.0, mesh_type=mesh_type0)
    mesh1 = create_cube(index_dtype1, real_dtype, center=(0.5, 0, 0), size=1.0, mesh_type=mesh_type1)

    # Build required structures
    mesh0.build_tree()
    mesh0.build_face_membership()
    mesh0.build_manifold_edge_link()
    mesh1.build_tree()
    mesh1.build_face_membership()
    mesh1.build_manifold_edge_link()

    # Compute union
    (result_faces, result_points), labels = tf.boolean_union(mesh0, mesh1)

    # Validate output
    assert isinstance(result_points, np.ndarray)
    assert isinstance(labels, np.ndarray)

    # Result faces type depends on input mesh types
    result_is_dynamic = mesh_type0 == 'dynamic' or mesh_type1 == 'dynamic'
    if result_is_dynamic:
        assert isinstance(result_faces, tf.OffsetBlockedArray)
        num_faces = len(result_faces)
    else:
        assert isinstance(result_faces, np.ndarray)
        # Faces should be triangles
        assert result_faces.ndim == 2
        assert result_faces.shape[1] == 3
        num_faces = result_faces.shape[0]

    # Points should be 3D
    assert result_points.ndim == 2
    assert result_points.shape[1] == 3

    # Labels should match number of faces
    assert labels.shape == (num_faces,)

    # Labels should only contain 0 or 1
    assert np.all((labels == 0) | (labels == 1))

    # Union should have some faces (non-empty)
    assert num_faces > 0


@pytest.mark.parametrize("index_dtype0", INDEX_DTYPES)
@pytest.mark.parametrize("index_dtype1", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
@pytest.mark.parametrize("mesh_type0", MESH_TYPES)
@pytest.mark.parametrize("mesh_type1", MESH_TYPES)
def test_boolean_intersection_basic(index_dtype0, index_dtype1, real_dtype, mesh_type0, mesh_type1):
    """Test basic intersection of two cubes"""
    # Create two overlapping cubes
    mesh0 = create_cube(index_dtype0, real_dtype, center=(0, 0, 0), size=1.0, mesh_type=mesh_type0)
    mesh1 = create_cube(index_dtype1, real_dtype, center=(0.5, 0, 0), size=1.0, mesh_type=mesh_type1)

    # Build required structures
    mesh0.build_tree()
    mesh0.build_face_membership()
    mesh0.build_manifold_edge_link()
    mesh1.build_tree()
    mesh1.build_face_membership()
    mesh1.build_manifold_edge_link()

    # Compute intersection
    (result_faces, result_points), labels = tf.boolean_intersection(mesh0, mesh1)

    # Validate output
    assert isinstance(result_points, np.ndarray)
    assert isinstance(labels, np.ndarray)

    # Result faces type depends on input mesh types
    result_is_dynamic = mesh_type0 == 'dynamic' or mesh_type1 == 'dynamic'
    if result_is_dynamic:
        assert isinstance(result_faces, tf.OffsetBlockedArray)
        num_faces = len(result_faces)
    else:
        assert isinstance(result_faces, np.ndarray)
        # Faces should be triangles
        assert result_faces.ndim == 2
        assert result_faces.shape[1] == 3
        num_faces = result_faces.shape[0]

    # Points should be 3D
    assert result_points.ndim == 2
    assert result_points.shape[1] == 3

    # Labels should match number of faces
    assert labels.shape == (num_faces,)

    # Labels should only contain 0 or 1
    assert np.all((labels == 0) | (labels == 1))

    # Intersection should have some faces (non-empty for overlapping cubes)
    assert num_faces > 0


@pytest.mark.parametrize("index_dtype0", INDEX_DTYPES)
@pytest.mark.parametrize("index_dtype1", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
@pytest.mark.parametrize("mesh_type0", MESH_TYPES)
@pytest.mark.parametrize("mesh_type1", MESH_TYPES)
def test_boolean_difference_basic(index_dtype0, index_dtype1, real_dtype, mesh_type0, mesh_type1):
    """Test basic difference of two cubes"""
    # Create two overlapping cubes
    mesh0 = create_cube(index_dtype0, real_dtype, center=(0, 0, 0), size=1.0, mesh_type=mesh_type0)
    mesh1 = create_cube(index_dtype1, real_dtype, center=(0.5, 0, 0), size=1.0, mesh_type=mesh_type1)

    # Build required structures
    mesh0.build_tree()
    mesh0.build_face_membership()
    mesh0.build_manifold_edge_link()
    mesh1.build_tree()
    mesh1.build_face_membership()
    mesh1.build_manifold_edge_link()

    # Compute difference mesh0 - mesh1
    (result_faces, result_points), labels = tf.boolean_difference(mesh0, mesh1)

    # Validate output
    assert isinstance(result_points, np.ndarray)
    assert isinstance(labels, np.ndarray)

    # Result faces type depends on input mesh types
    result_is_dynamic = mesh_type0 == 'dynamic' or mesh_type1 == 'dynamic'
    if result_is_dynamic:
        assert isinstance(result_faces, tf.OffsetBlockedArray)
        num_faces = len(result_faces)
    else:
        assert isinstance(result_faces, np.ndarray)
        # Faces should be triangles
        assert result_faces.ndim == 2
        assert result_faces.shape[1] == 3
        num_faces = result_faces.shape[0]

    # Points should be 3D
    assert result_points.ndim == 2
    assert result_points.shape[1] == 3

    # Labels should match number of faces
    assert labels.shape == (num_faces,)

    # Labels should only contain 0 or 1
    assert np.all((labels == 0) | (labels == 1))

    # Difference should have some faces (non-empty)
    assert num_faces > 0


@pytest.mark.parametrize("index_dtype0", INDEX_DTYPES)
@pytest.mark.parametrize("index_dtype1", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
@pytest.mark.parametrize("mesh_type0", MESH_TYPES)
@pytest.mark.parametrize("mesh_type1", MESH_TYPES)
def test_boolean_with_curves(index_dtype0, index_dtype1, real_dtype, mesh_type0, mesh_type1):
    """Test boolean operations with return_curves=True"""
    # Create two overlapping cubes
    mesh0 = create_cube(index_dtype0, real_dtype, center=(0, 0, 0), size=1.0, mesh_type=mesh_type0)
    mesh1 = create_cube(index_dtype1, real_dtype, center=(0.5, 0, 0), size=1.0, mesh_type=mesh_type1)

    # Build required structures
    mesh0.build_tree()
    mesh0.build_face_membership()
    mesh0.build_manifold_edge_link()
    mesh1.build_tree()
    mesh1.build_face_membership()
    mesh1.build_manifold_edge_link()

    # Compute union with curves
    (result_faces, result_points), labels, (paths, curve_points) = tf.boolean_union(
        mesh0, mesh1, return_curves=True
    )

    # Validate mesh output
    assert isinstance(result_points, np.ndarray)
    assert isinstance(labels, np.ndarray)

    # Result faces type depends on input mesh types
    result_is_dynamic = mesh_type0 == 'dynamic' or mesh_type1 == 'dynamic'
    if result_is_dynamic:
        assert isinstance(result_faces, tf.OffsetBlockedArray)
    else:
        assert isinstance(result_faces, np.ndarray)

    # Validate curves output
    assert isinstance(paths, tf.OffsetBlockedArray)
    assert isinstance(curve_points, np.ndarray)

    # Curve points should be 3D
    if len(curve_points) > 0:
        assert curve_points.ndim == 2
        assert curve_points.shape[1] == 3


# ==============================================================================
# Index type symmetry tests
# ==============================================================================

def test_index_type_symmetry():
    """Test that swapping meshes with different index types produces correct labels"""
    # Create two cubes with different index types
    mesh_int32 = create_cube(np.int32, np.float32, center=(0, 0, 0), size=1.0)
    mesh_int64 = create_cube(np.int64, np.float32, center=(0.5, 0, 0), size=1.0)

    # Build required structures
    for mesh in [mesh_int32, mesh_int64]:
        mesh.build_tree()
        mesh.build_face_membership()
        mesh.build_manifold_edge_link()

    # Compute union both ways
    (faces1, points1), labels1 = tf.boolean_union(mesh_int32, mesh_int64)
    (faces2, points2), labels2 = tf.boolean_union(mesh_int64, mesh_int32)

    # Both should produce valid results
    assert len(faces1) > 0
    assert len(faces2) > 0

    # Labels should be valid (0 or 1)
    assert np.all((labels1 == 0) | (labels1 == 1))
    assert np.all((labels2 == 0) | (labels2 == 1))


# ==============================================================================
# Validation tests
# ==============================================================================

def test_rejects_non_mesh_input():
    """Test that boolean operations reject non-Mesh inputs"""
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    mesh = tf.Mesh(faces, points)
    mesh.build_tree()
    mesh.build_face_membership()
    mesh.build_manifold_edge_link()

    # Try with tuple instead of Mesh
    with pytest.raises(TypeError, match="must be a Mesh object"):
        tf.boolean_union((faces, points), mesh)

    with pytest.raises(TypeError, match="must be a Mesh object"):
        tf.boolean_union(mesh, (faces, points))


def test_rejects_2d_meshes():
    """Test that boolean operations reject 2D meshes"""
    # Create 2D triangle
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    points = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float32)
    mesh_2d = tf.Mesh(faces, points)

    # Create 3D triangle
    faces_3d = np.array([[0, 1, 2]], dtype=np.int32)
    points_3d = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    mesh_3d = tf.Mesh(faces_3d, points_3d)
    mesh_3d.build_tree()
    mesh_3d.build_face_membership()
    mesh_3d.build_manifold_edge_link()

    # Should reject 2D meshes
    with pytest.raises(ValueError, match="only support 3D meshes"):
        tf.boolean_union(mesh_2d, mesh_3d)

    with pytest.raises(ValueError, match="only support 3D meshes"):
        tf.boolean_union(mesh_3d, mesh_2d)


def test_rejects_non_triangle_fixed_meshes():
    """Test that boolean operations reject non-triangle fixed-size meshes"""
    # Create triangle mesh
    tri_faces = np.array([[0, 1, 2]], dtype=np.int32)
    tri_points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    tri_mesh = tf.Mesh(tri_faces, tri_points)
    tri_mesh.build_tree()
    tri_mesh.build_face_membership()
    tri_mesh.build_manifold_edge_link()

    # Creating a quad mesh (faces with 4 vertices) should fail at Mesh creation
    # since only triangles (ngon=3) and dynamic meshes are supported
    quad_faces = np.array([[0, 1, 2, 3]], dtype=np.int32)
    quad_points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float32)

    # Quad mesh creation should fail
    with pytest.raises((ValueError, TypeError)):
        tf.Mesh(quad_faces, quad_points)


def test_rejects_mismatched_dtypes():
    """Test that boolean operations reject meshes with different real dtypes"""
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    points_f32 = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    points_f64 = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)

    mesh_f32 = tf.Mesh(faces, points_f32)
    mesh_f64 = tf.Mesh(faces, points_f64)

    mesh_f32.build_tree()
    mesh_f32.build_face_membership()
    mesh_f32.build_manifold_edge_link()
    mesh_f64.build_tree()
    mesh_f64.build_face_membership()
    mesh_f64.build_manifold_edge_link()

    # Should reject mismatched dtypes
    with pytest.raises(ValueError, match="Mesh dtypes must match"):
        tf.boolean_union(mesh_f32, mesh_f64)


# ==============================================================================
# Run tests
# ==============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
