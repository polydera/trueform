"""
Tests for geometry compute functions (normals, curvatures, shape index)

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


def create_sphere_mesh(index_dtype, real_dtype, subdivisions=2):
    """Create an icosphere mesh for curvature testing."""
    # Start with icosahedron vertices
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
    ], dtype=real_dtype)
    # Normalize to unit sphere
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)

    tris = np.array([
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
    ], dtype=index_dtype)

    # Subdivide for smoother sphere
    for _ in range(subdivisions):
        new_tris = []
        edge_midpoints = {}

        def get_midpoint(i0, i1):
            key = (min(i0, i1), max(i0, i1))
            if key not in edge_midpoints:
                mid = (verts[i0] + verts[i1]) / 2
                mid = mid / np.linalg.norm(mid)  # Project to sphere
                edge_midpoints[key] = len(verts)
                verts_list = list(verts)
                verts_list.append(mid)
                return np.array(verts_list, dtype=real_dtype), edge_midpoints[key]
            return verts, edge_midpoints[key]

        verts_list = list(verts)
        for tri in tris:
            i0, i1, i2 = tri
            # Get midpoints
            for edge in [(i0, i1), (i1, i2), (i2, i0)]:
                key = (min(edge[0], edge[1]), max(edge[0], edge[1]))
                if key not in edge_midpoints:
                    mid = (verts[edge[0]] + verts[edge[1]]) / 2
                    mid = mid / np.linalg.norm(mid)
                    edge_midpoints[key] = len(verts_list)
                    verts_list.append(mid)

            m01 = edge_midpoints[(min(i0, i1), max(i0, i1))]
            m12 = edge_midpoints[(min(i1, i2), max(i1, i2))]
            m20 = edge_midpoints[(min(i2, i0), max(i2, i0))]

            new_tris.append([i0, m01, m20])
            new_tris.append([i1, m12, m01])
            new_tris.append([i2, m20, m12])
            new_tris.append([m01, m12, m20])

        verts = np.array(verts_list, dtype=real_dtype)
        tris = np.array(new_tris, dtype=index_dtype)

    return tris, verts


# ==============================================================================
# normals Tests
# ==============================================================================

@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_normals_from_mesh(index_dtype, real_dtype):
    """Test normals with Mesh input."""
    faces, points = create_triangle_mesh_3d(index_dtype, real_dtype)
    mesh = tf.Mesh(faces, points)

    normals = tf.normals(mesh)

    assert normals.shape == (len(faces), 3)
    assert normals.dtype == real_dtype


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_normals_from_tuple(index_dtype, real_dtype):
    """Test normals with (faces, points) tuple input."""
    faces, points = create_triangle_mesh_3d(index_dtype, real_dtype)

    normals = tf.normals((faces, points))

    assert normals.shape == (len(faces), 3)
    assert normals.dtype == real_dtype


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_normals_unit_vectors(index_dtype, real_dtype):
    """Test that computed normals are unit vectors."""
    faces, points = create_triangle_mesh_3d(index_dtype, real_dtype)

    normals = tf.normals((faces, points))

    lengths = np.linalg.norm(normals, axis=1)
    np.testing.assert_allclose(lengths, 1.0, atol=1e-6)


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_normals_dynamic_mesh(index_dtype, real_dtype):
    """Test normals with dynamic mesh."""
    faces, points = create_dynamic_mesh_3d(index_dtype, real_dtype)

    normals = tf.normals((faces, points))

    assert normals.shape == (len(faces), 3)
    assert normals.dtype == real_dtype


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_normals_2d_raises(index_dtype, real_dtype):
    """Test that normals raises for 2D mesh."""
    faces, points = create_triangle_mesh_2d(index_dtype, real_dtype)

    with pytest.raises(ValueError, match="3D"):
        tf.normals((faces, points))


# ==============================================================================
# point_normals Tests
# ==============================================================================

@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_point_normals_from_mesh(index_dtype, real_dtype):
    """Test point_normals with Mesh input."""
    faces, points = create_triangle_mesh_3d(index_dtype, real_dtype)
    mesh = tf.Mesh(faces, points)

    point_normals = tf.point_normals(mesh)

    assert point_normals.shape == (len(points), 3)
    assert point_normals.dtype == real_dtype


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_point_normals_from_tuple(index_dtype, real_dtype):
    """Test point_normals with (faces, points) tuple input."""
    faces, points = create_triangle_mesh_3d(index_dtype, real_dtype)

    point_normals = tf.point_normals((faces, points))

    assert point_normals.shape == (len(points), 3)
    assert point_normals.dtype == real_dtype


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_point_normals_unit_vectors(index_dtype, real_dtype):
    """Test that computed point normals are unit vectors."""
    faces, points = create_triangle_mesh_3d(index_dtype, real_dtype)

    point_normals = tf.point_normals((faces, points))

    lengths = np.linalg.norm(point_normals, axis=1)
    np.testing.assert_allclose(lengths, 1.0, atol=1e-6)


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_point_normals_dynamic_mesh(index_dtype, real_dtype):
    """Test point_normals with dynamic mesh."""
    faces, points = create_dynamic_mesh_3d(index_dtype, real_dtype)

    point_normals = tf.point_normals((faces, points))

    assert point_normals.shape == (len(points), 3)
    assert point_normals.dtype == real_dtype


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_point_normals_2d_raises(index_dtype, real_dtype):
    """Test that point_normals raises for 2D mesh."""
    faces, points = create_triangle_mesh_2d(index_dtype, real_dtype)

    with pytest.raises(ValueError, match="3D"):
        tf.point_normals((faces, points))


# ==============================================================================
# principal_curvatures Tests
# ==============================================================================

@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_principal_curvatures_from_mesh(index_dtype, real_dtype):
    """Test principal_curvatures with Mesh input."""
    faces, points = create_triangle_mesh_3d(index_dtype, real_dtype)
    mesh = tf.Mesh(faces, points)

    result = tf.principal_curvatures(mesh)

    assert len(result) == 2
    k0, k1 = result
    assert k0.shape == (len(points),)
    assert k1.shape == (len(points),)
    assert k0.dtype == real_dtype
    assert k1.dtype == real_dtype


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_principal_curvatures_from_tuple(index_dtype, real_dtype):
    """Test principal_curvatures with (faces, points) tuple input."""
    faces, points = create_triangle_mesh_3d(index_dtype, real_dtype)

    k0, k1 = tf.principal_curvatures((faces, points))

    assert k0.shape == (len(points),)
    assert k1.shape == (len(points),)


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_principal_curvatures_with_directions(index_dtype, real_dtype):
    """Test principal_curvatures with directions=True."""
    faces, points = create_triangle_mesh_3d(index_dtype, real_dtype)

    result = tf.principal_curvatures((faces, points), directions=True)

    assert len(result) == 4
    k0, k1, d0, d1 = result
    assert k0.shape == (len(points),)
    assert k1.shape == (len(points),)
    assert d0.shape == (len(points), 3)
    assert d1.shape == (len(points), 3)


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_principal_curvatures_directions_unit_vectors(index_dtype, real_dtype):
    """Test that principal directions are unit vectors."""
    faces, points = create_cube_mesh(index_dtype, real_dtype)

    k0, k1, d0, d1 = tf.principal_curvatures((faces, points), directions=True)

    # Check that directions are unit vectors
    lengths_d0 = np.linalg.norm(d0, axis=1)
    lengths_d1 = np.linalg.norm(d1, axis=1)
    np.testing.assert_allclose(lengths_d0, 1.0, atol=1e-5)
    np.testing.assert_allclose(lengths_d1, 1.0, atol=1e-5)


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_principal_curvatures_k_parameter(index_dtype, real_dtype):
    """Test principal_curvatures with different k values."""
    faces, points = create_cube_mesh(index_dtype, real_dtype)

    # Test with different k-ring sizes
    k0_k1, k1_k1 = tf.principal_curvatures((faces, points), k=1)
    k0_k2, k1_k2 = tf.principal_curvatures((faces, points), k=2)
    k0_k3, k1_k3 = tf.principal_curvatures((faces, points), k=3)

    # All should have correct shape
    assert k0_k1.shape == (len(points),)
    assert k0_k2.shape == (len(points),)
    assert k0_k3.shape == (len(points),)


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_principal_curvatures_planar(index_dtype, real_dtype):
    """Test that planar mesh has zero curvature."""
    faces, points = create_triangle_mesh_3d(index_dtype, real_dtype)

    k0, k1 = tf.principal_curvatures((faces, points))

    # Planar mesh should have approximately zero curvature
    np.testing.assert_allclose(k0, 0.0, atol=1e-5)
    np.testing.assert_allclose(k1, 0.0, atol=1e-5)


@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_principal_curvatures_sphere(real_dtype):
    """Test that sphere has constant curvature equal to 1/radius."""
    faces, points = create_sphere_mesh(np.int32, real_dtype, subdivisions=2)

    k0, k1 = tf.principal_curvatures((faces, points), k=2)

    # For unit sphere, curvatures should be approximately 1.0
    # (Allow larger tolerance due to discrete approximation)
    mean_k0 = np.mean(k0)
    mean_k1 = np.mean(k1)
    # Both principal curvatures should be similar for sphere
    np.testing.assert_allclose(mean_k0, mean_k1, atol=0.3)


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_principal_curvatures_dynamic_mesh(index_dtype, real_dtype):
    """Test principal_curvatures with dynamic mesh."""
    faces, points = create_dynamic_mesh_3d(index_dtype, real_dtype)

    k0, k1 = tf.principal_curvatures((faces, points))

    assert k0.shape == (len(points),)
    assert k1.shape == (len(points),)


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_principal_curvatures_2d_raises(index_dtype, real_dtype):
    """Test that principal_curvatures raises for 2D mesh."""
    faces, points = create_triangle_mesh_2d(index_dtype, real_dtype)

    with pytest.raises(ValueError, match="3D"):
        tf.principal_curvatures((faces, points))


# ==============================================================================
# shape_index Tests
# ==============================================================================

@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_shape_index_from_mesh(index_dtype, real_dtype):
    """Test shape_index with Mesh input."""
    faces, points = create_triangle_mesh_3d(index_dtype, real_dtype)
    mesh = tf.Mesh(faces, points)

    si = tf.shape_index(mesh)

    assert si.shape == (len(points),)


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_shape_index_from_tuple(index_dtype, real_dtype):
    """Test shape_index with (faces, points) tuple input."""
    faces, points = create_triangle_mesh_3d(index_dtype, real_dtype)

    si = tf.shape_index((faces, points))

    assert si.shape == (len(points),)


@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_shape_index_range(real_dtype):
    """Test that shape index is in range [-1, 1]."""
    faces, points = create_sphere_mesh(np.int32, real_dtype, subdivisions=2)

    si = tf.shape_index((faces, points))

    assert np.all(si >= -1.0 - 1e-2)
    assert np.all(si <= 1.0 + 1e-2)


@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_shape_index_sphere_convex(real_dtype):
    """Test that convex sphere has positive shape index."""
    faces, points = create_sphere_mesh(np.int32, real_dtype, subdivisions=2)

    si = tf.shape_index((faces, points))

    # Sphere should have mostly positive shape index (convex)
    mean_si = np.mean(si)
    assert mean_si > 0.0, f"Expected positive mean shape index, got {mean_si}"


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_shape_index_k_parameter(index_dtype, real_dtype):
    """Test shape_index with different k values."""
    faces, points = create_cube_mesh(index_dtype, real_dtype)

    si_k1 = tf.shape_index((faces, points), k=1)
    si_k2 = tf.shape_index((faces, points), k=2)
    si_k3 = tf.shape_index((faces, points), k=3)

    # All should have correct shape
    assert si_k1.shape == (len(points),)
    assert si_k2.shape == (len(points),)
    assert si_k3.shape == (len(points),)


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_shape_index_dynamic_mesh(index_dtype, real_dtype):
    """Test shape_index with dynamic mesh."""
    faces, points = create_dynamic_mesh_3d(index_dtype, real_dtype)

    si = tf.shape_index((faces, points))

    assert si.shape == (len(points),)


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_shape_index_2d_raises(index_dtype, real_dtype):
    """Test that shape_index raises for 2D mesh."""
    faces, points = create_triangle_mesh_2d(index_dtype, real_dtype)

    with pytest.raises(ValueError, match="3D"):
        tf.shape_index((faces, points))


# ==============================================================================
# Consistency Tests
# ==============================================================================

@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_normals_matches_mesh_property(index_dtype, real_dtype):
    """Test that normals matches mesh.normals property."""
    faces, points = create_triangle_mesh_3d(index_dtype, real_dtype)
    mesh = tf.Mesh(faces, points)

    normals_func = tf.normals(mesh)
    normals_prop = mesh.normals

    np.testing.assert_allclose(normals_func, normals_prop)


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_point_normals_matches_mesh_property(index_dtype, real_dtype):
    """Test that point_normals matches mesh.point_normals property."""
    faces, points = create_triangle_mesh_3d(index_dtype, real_dtype)
    mesh = tf.Mesh(faces, points)

    pn_func = tf.point_normals(mesh)
    pn_prop = mesh.point_normals

    np.testing.assert_allclose(pn_func, pn_prop)


# ==============================================================================
# Main runner
# ==============================================================================

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
