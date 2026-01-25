"""
Tests for mesh primitive generation functions

Copyright (c) 2025 Žiga Sajovic, XLAB
"""

import sys
import numpy as np
import pytest
import trueform as tf


# Test parameters
REAL_DTYPES = [np.float32, np.float64]
INDEX_DTYPES = [np.int32, np.int64]


# ==============================================================================
# Topology Validation Helpers
# ==============================================================================

def compute_signed_volume(faces: np.ndarray, points: np.ndarray) -> float:
    """
    Compute signed volume of a closed mesh using the divergence theorem.

    For a closed mesh with outward normals, returns positive volume.
    """
    volume = 0.0
    for face in faces:
        v0, v1, v2 = points[face[0]], points[face[1]], points[face[2]]
        # Signed volume of tetrahedron with origin
        volume += np.dot(v0, np.cross(v1, v2)) / 6.0
    return volume


def check_topology(faces: np.ndarray, points: np.ndarray, name: str = "mesh"):
    """
    Validate mesh topology. Raises AssertionError on failure.

    Checks:
    - Valid indices (in range, non-negative)
    - Watertight (no boundary edges)
    - Manifold (each edge has exactly 2 adjacent faces)
    - Non-degenerate triangles (positive area)
    """
    mesh = tf.Mesh(faces, points)

    # Valid indices
    assert np.all(faces >= 0), f"{name}: negative indices found"
    assert np.all(faces < len(points)), f"{name}: out of bounds indices found"

    # Watertight
    boundary = tf.boundary_edges(mesh)
    assert len(boundary) == 0, f"{name}: has {len(boundary)} boundary edges, expected 0"

    # Manifold (no non-manifold edges)
    non_manifold = tf.non_manifold_edges(mesh)
    assert len(non_manifold) == 0, f"{name}: has {len(non_manifold)} non-manifold edges"

    # Non-degenerate triangles
    for i, face in enumerate(faces):
        v0, v1, v2 = points[face[0]], points[face[1]], points[face[2]]
        area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
        assert area > 1e-10, f"{name}: triangle {i} is degenerate (area={area})"


def check_bounds(points: np.ndarray, expected_extents: tuple, name: str = "mesh"):
    """
    Validate mesh bounds are centered at origin with expected extents.

    expected_extents: (width, height, depth) - full sizes along x, y, z
    """
    width, height, depth = expected_extents

    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    y_min, y_max = points[:, 1].min(), points[:, 1].max()
    z_min, z_max = points[:, 2].min(), points[:, 2].max()

    # Check extents
    np.testing.assert_allclose(x_max - x_min, width, rtol=1e-5,
        err_msg=f"{name}: width mismatch")
    np.testing.assert_allclose(y_max - y_min, height, rtol=1e-5,
        err_msg=f"{name}: height mismatch")
    np.testing.assert_allclose(z_max - z_min, depth, rtol=1e-5,
        err_msg=f"{name}: depth mismatch")

    # Check centered at origin
    np.testing.assert_allclose(x_min, -width/2, rtol=1e-5,
        err_msg=f"{name}: not centered on x")
    np.testing.assert_allclose(y_min, -height/2, rtol=1e-5,
        err_msg=f"{name}: not centered on y")
    np.testing.assert_allclose(z_min, -depth/2, rtol=1e-5,
        err_msg=f"{name}: not centered on z")


# ==============================================================================
# Sphere Mesh Tests
# ==============================================================================

@pytest.mark.parametrize("dtype", REAL_DTYPES)
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
def test_sphere_dtypes(dtype, index_dtype):
    """Sphere mesh respects dtype parameters."""
    faces, points = tf.make_sphere_mesh(
        1.0, stacks=10, segments=10, dtype=dtype, index_dtype=index_dtype
    )
    assert faces.dtype == index_dtype
    assert points.dtype == dtype
    assert faces.shape[1] == 3
    assert points.shape[1] == 3


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_sphere_topology(dtype):
    """Sphere mesh has valid topology."""
    faces, points = tf.make_sphere_mesh(1.0, stacks=10, segments=10, dtype=dtype)
    check_topology(faces, points, "sphere")


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_sphere_radius(dtype):
    """All sphere vertices lie on the sphere surface."""
    radius = 3.5
    faces, points = tf.make_sphere_mesh(radius, stacks=10, segments=10, dtype=dtype)

    distances = np.linalg.norm(points, axis=1)
    np.testing.assert_allclose(distances, radius, rtol=1e-5,
        err_msg="Not all vertices at specified radius")


@pytest.mark.parametrize("radius", [0.5, 1.0, 2.0, 5.0])
def test_sphere_volume(radius):
    """Sphere volume matches (4/3)πr³."""
    # Use high resolution for accuracy
    faces, points = tf.make_sphere_mesh(radius, stacks=40, segments=40, dtype=np.float64)
    check_topology(faces, points, f"sphere(r={radius})")

    computed = compute_signed_volume(faces, points)
    expected = (4.0 / 3.0) * np.pi * radius**3

    # Allow 1% error due to discretization
    np.testing.assert_allclose(computed, expected, rtol=0.01,
        err_msg=f"Sphere volume mismatch: computed={computed}, expected={expected}")


@pytest.mark.parametrize("stacks,segments", [(5, 5), (10, 10), (20, 20), (10, 30)])
def test_sphere_resolution_topology(stacks, segments):
    """Sphere topology is valid at various resolutions."""
    faces, points = tf.make_sphere_mesh(1.0, stacks=stacks, segments=segments)
    check_topology(faces, points, f"sphere(stacks={stacks}, segments={segments})")


# ==============================================================================
# Cylinder Mesh Tests
# ==============================================================================

@pytest.mark.parametrize("dtype", REAL_DTYPES)
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
def test_cylinder_dtypes(dtype, index_dtype):
    """Cylinder mesh respects dtype parameters."""
    faces, points = tf.make_cylinder_mesh(
        1.0, 2.0, segments=16, dtype=dtype, index_dtype=index_dtype
    )
    assert faces.dtype == index_dtype
    assert points.dtype == dtype
    assert faces.shape[1] == 3
    assert points.shape[1] == 3


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_cylinder_topology(dtype):
    """Cylinder mesh has valid topology."""
    faces, points = tf.make_cylinder_mesh(1.0, 2.0, segments=16, dtype=dtype)
    check_topology(faces, points, "cylinder")


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_cylinder_bounds(dtype):
    """Cylinder has correct dimensions."""
    radius, height = 2.0, 4.0
    faces, points = tf.make_cylinder_mesh(radius, height, segments=32, dtype=dtype)

    # Z bounds
    z_min, z_max = points[:, 2].min(), points[:, 2].max()
    np.testing.assert_allclose(z_min, -height/2, rtol=1e-5)
    np.testing.assert_allclose(z_max, height/2, rtol=1e-5)

    # XY radius (ring vertices should be at radius, caps at 0)
    xy_distances = np.linalg.norm(points[:, :2], axis=1)
    # Max should be radius, min should be 0 (cap centers)
    np.testing.assert_allclose(xy_distances.max(), radius, rtol=1e-5)
    np.testing.assert_allclose(xy_distances.min(), 0.0, atol=1e-7)


@pytest.mark.parametrize("radius,height", [(0.5, 1.0), (1.0, 2.0), (2.0, 3.0)])
def test_cylinder_volume(radius, height):
    """Cylinder volume matches πr²h."""
    # Use high segment count for accuracy
    faces, points = tf.make_cylinder_mesh(radius, height, segments=64, dtype=np.float64)
    check_topology(faces, points, f"cylinder(r={radius}, h={height})")

    computed = compute_signed_volume(faces, points)
    expected = np.pi * radius**2 * height

    # Allow 1% error due to discretization
    np.testing.assert_allclose(computed, expected, rtol=0.01,
        err_msg=f"Cylinder volume mismatch: computed={computed}, expected={expected}")


@pytest.mark.parametrize("segments", [8, 16, 32, 64])
def test_cylinder_resolution_topology(segments):
    """Cylinder topology is valid at various resolutions."""
    faces, points = tf.make_cylinder_mesh(1.0, 2.0, segments=segments)
    check_topology(faces, points, f"cylinder(segments={segments})")


# ==============================================================================
# Box Mesh Tests
# ==============================================================================

@pytest.mark.parametrize("dtype", REAL_DTYPES)
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
def test_box_dtypes(dtype, index_dtype):
    """Box mesh respects dtype parameters."""
    faces, points = tf.make_box_mesh(
        2.0, 1.0, 3.0, dtype=dtype, index_dtype=index_dtype
    )
    assert faces.dtype == index_dtype
    assert points.dtype == dtype
    assert faces.shape == (12, 3)  # 6 faces × 2 triangles
    assert points.shape == (8, 3)   # 8 corners


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_box_topology(dtype):
    """Simple box mesh has valid topology."""
    faces, points = tf.make_box_mesh(2.0, 1.0, 3.0, dtype=dtype)
    check_topology(faces, points, "box")


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_box_bounds(dtype):
    """Box has correct dimensions centered at origin."""
    width, height, depth = 4.0, 2.0, 6.0
    faces, points = tf.make_box_mesh(width, height, depth, dtype=dtype)
    check_bounds(points, (width, height, depth), "box")


@pytest.mark.parametrize("width,height,depth", [(1.0, 1.0, 1.0), (2.0, 3.0, 4.0), (0.5, 1.5, 2.5)])
def test_box_volume(width, height, depth):
    """Box volume matches w×h×d."""
    faces, points = tf.make_box_mesh(width, height, depth, dtype=np.float64)
    check_topology(faces, points, f"box({width}×{height}×{depth})")

    computed = compute_signed_volume(faces, points)
    expected = width * height * depth

    np.testing.assert_allclose(computed, expected, rtol=1e-5,
        err_msg=f"Box volume mismatch: computed={computed}, expected={expected}")


# Subdivided box tests with various tick combinations
@pytest.mark.parametrize("ticks", [
    (1, 1, 1),  # Simple box
    (2, 2, 2),  # Uniform subdivision
    (3, 1, 1),  # Subdivide only width
    (1, 4, 1),  # Subdivide only height
    (1, 1, 5),  # Subdivide only depth
    (2, 3, 4),  # Different subdivisions
    (5, 5, 5),  # High subdivision
])
def test_box_subdivided_topology(ticks):
    """Subdivided box has valid topology for various tick combinations."""
    wt, ht, dt = ticks
    faces, points = tf.make_box_mesh(2.0, 3.0, 4.0, wt, ht, dt, dtype=np.float64)
    check_topology(faces, points, f"box(ticks={ticks})")


@pytest.mark.parametrize("ticks", [
    (1, 1, 1),
    (2, 2, 2),
    (3, 4, 5),
])
def test_box_subdivided_volume_invariant(ticks):
    """Subdivided box volume equals simple box volume."""
    width, height, depth = 2.0, 3.0, 4.0
    wt, ht, dt = ticks

    faces, points = tf.make_box_mesh(width, height, depth, wt, ht, dt, dtype=np.float64)
    computed = compute_signed_volume(faces, points)
    expected = width * height * depth

    np.testing.assert_allclose(computed, expected, rtol=1e-5,
        err_msg=f"Subdivided box volume mismatch at ticks={ticks}")


@pytest.mark.parametrize("ticks", [(2, 2, 2), (3, 3, 3)])
def test_box_subdivided_bounds(ticks):
    """Subdivided box has same bounds as simple box."""
    width, height, depth = 4.0, 2.0, 6.0
    wt, ht, dt = ticks

    faces, points = tf.make_box_mesh(width, height, depth, wt, ht, dt, dtype=np.float64)
    check_bounds(points, (width, height, depth), f"subdivided_box(ticks={ticks})")


# ==============================================================================
# Plane Mesh Tests
# ==============================================================================

@pytest.mark.parametrize("dtype", REAL_DTYPES)
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
def test_plane_dtypes(dtype, index_dtype):
    """Plane mesh respects dtype parameters."""
    faces, points = tf.make_plane_mesh(
        10.0, 5.0, dtype=dtype, index_dtype=index_dtype
    )
    assert faces.dtype == index_dtype
    assert points.dtype == dtype
    assert faces.shape == (2, 3)   # 2 triangles
    assert points.shape == (4, 3)  # 4 corners


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_plane_bounds(dtype):
    """Plane has correct dimensions."""
    width, height = 8.0, 4.0
    faces, points = tf.make_plane_mesh(width, height, dtype=dtype)

    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    y_min, y_max = points[:, 1].min(), points[:, 1].max()

    np.testing.assert_allclose(x_max - x_min, width, rtol=1e-5)
    np.testing.assert_allclose(y_max - y_min, height, rtol=1e-5)
    np.testing.assert_allclose(points[:, 2], 0.0, atol=1e-7)  # All at z=0


def compute_triangle_area_sum(faces: np.ndarray, points: np.ndarray) -> float:
    """Compute sum of triangle areas."""
    total = 0.0
    for face in faces:
        v0, v1, v2 = points[face[0]], points[face[1]], points[face[2]]
        total += 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
    return total


@pytest.mark.parametrize("width,height", [(1.0, 1.0), (4.0, 3.0), (10.0, 5.0)])
def test_plane_area(width, height):
    """Plane area matches w×h."""
    faces, points = tf.make_plane_mesh(width, height, dtype=np.float64)

    computed = compute_triangle_area_sum(faces, points)
    expected = width * height

    np.testing.assert_allclose(computed, expected, rtol=1e-5,
        err_msg=f"Plane area mismatch: computed={computed}, expected={expected}")


@pytest.mark.parametrize("ticks", [
    (1, 1),
    (2, 2),
    (5, 3),
    (10, 10),
])
def test_plane_subdivided_area_invariant(ticks):
    """Subdivided plane area equals simple plane area."""
    width, height = 6.0, 4.0
    wt, ht = ticks

    faces, points = tf.make_plane_mesh(width, height, wt, ht, dtype=np.float64)
    computed = compute_triangle_area_sum(faces, points)
    expected = width * height

    np.testing.assert_allclose(computed, expected, rtol=1e-5,
        err_msg=f"Subdivided plane area mismatch at ticks={ticks}")


@pytest.mark.parametrize("ticks", [(1, 1), (5, 3), (10, 10)])
def test_plane_subdivided_vertex_count(ticks):
    """Subdivided plane has expected vertex count."""
    wt, ht = ticks
    faces, points = tf.make_plane_mesh(4.0, 3.0, wt, ht)

    expected_vertices = (wt + 1) * (ht + 1)
    expected_faces = 2 * wt * ht

    assert points.shape[0] == expected_vertices, \
        f"Expected {expected_vertices} vertices, got {points.shape[0]}"
    assert faces.shape[0] == expected_faces, \
        f"Expected {expected_faces} faces, got {faces.shape[0]}"


@pytest.mark.parametrize("ticks", [(1, 1), (3, 3), (5, 5)])
def test_plane_valid_indices(ticks):
    """Plane has valid face indices."""
    wt, ht = ticks
    faces, points = tf.make_plane_mesh(4.0, 3.0, wt, ht)

    assert np.all(faces >= 0), "Negative indices found"
    assert np.all(faces < len(points)), "Out of bounds indices found"


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
