"""
Tests for mesh measurement functions (volume, area)

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
# Manual computation helpers (for verification)
# ==============================================================================

def manual_signed_volume(faces, points):
    """Compute signed volume manually using divergence theorem."""
    volume = 0.0
    for face in faces:
        v0, v1, v2 = points[face[0]], points[face[1]], points[face[2]]
        volume += np.dot(v0, np.cross(v1, v2)) / 6.0
    return volume


def manual_triangle_area(v0, v1, v2):
    """Compute area of a single triangle."""
    return 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))


def manual_mesh_area(faces, points):
    """Compute total surface area of a triangle mesh."""
    total = 0.0
    for face in faces:
        v0, v1, v2 = points[face[0]], points[face[1]], points[face[2]]
        total += manual_triangle_area(v0, v1, v2)
    return total


def manual_polygon_area_3d(polygon):
    """Compute area of a 3D polygon using triangulation from first vertex."""
    total = 0.0
    v0 = polygon[0]
    for i in range(1, len(polygon) - 1):
        v1 = polygon[i]
        v2 = polygon[i + 1]
        total += manual_triangle_area(v0, v1, v2)
    return total


# ==============================================================================
# Single Polygon Area Tests
# ==============================================================================

@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_area_unit_square_3d(dtype):
    """Unit square polygon has area 1."""
    polygon = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0]
    ], dtype=dtype)

    computed = tf.area(polygon)
    np.testing.assert_allclose(computed, 1.0, rtol=1e-5)


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_area_2d_polygon(dtype):
    """2D polygon area."""
    polygon = np.array([
        [0, 0],
        [2, 0],
        [2, 3],
        [0, 3]
    ], dtype=dtype)

    computed = tf.area(polygon)
    np.testing.assert_allclose(computed, 6.0, rtol=1e-5)


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_area_triangle(dtype):
    """Triangle area = 0.5 * base * height."""
    polygon = np.array([
        [0, 0, 0],
        [2, 0, 0],
        [1, 2, 0]
    ], dtype=dtype)

    computed = tf.area(polygon)
    # base=2, height=2 -> area=2
    np.testing.assert_allclose(computed, 2.0, rtol=1e-5)


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_area_regular_hexagon(dtype):
    """Regular hexagon area matches (3*sqrt(3)/2)*r^2."""
    n = 6
    r = 2.0
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    polygon = np.column_stack([
        r * np.cos(angles),
        r * np.sin(angles),
        np.zeros(n)
    ]).astype(dtype)

    computed = tf.area(polygon)
    expected = (3 * np.sqrt(3) / 2) * r**2
    np.testing.assert_allclose(computed, expected, rtol=1e-5)


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_area_polygon_object(dtype):
    """Area with tf.Polygon object."""
    poly = tf.Polygon(np.array([
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0]
    ], dtype=dtype))

    computed = tf.area(poly)
    np.testing.assert_allclose(computed, 1.0, rtol=1e-5)


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_area_polygon_object_2d(dtype):
    """Area with 2D tf.Polygon object."""
    poly = tf.Polygon(np.array([
        [0, 0],
        [2, 0],
        [2, 3],
        [0, 3]
    ], dtype=dtype))

    computed = tf.area(poly)
    np.testing.assert_allclose(computed, 6.0, rtol=1e-5)


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_area_matches_manual(dtype):
    """Area matches manual computation for irregular polygon."""
    # L-shaped polygon
    polygon = np.array([
        [0, 0, 0],
        [2, 0, 0],
        [2, 1, 0],
        [1, 1, 0],
        [1, 2, 0],
        [0, 2, 0]
    ], dtype=dtype)

    computed = tf.area(polygon)
    expected = manual_polygon_area_3d(polygon)
    np.testing.assert_allclose(computed, expected, rtol=1e-5)


# ==============================================================================
# Mesh Area (Surface Area) Tests
# ==============================================================================

@pytest.mark.parametrize("dtype", REAL_DTYPES)
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
def test_mesh_area_box(dtype, index_dtype):
    """Box surface area matches 2(wh + hd + wd)."""
    w, h, d = 2.0, 3.0, 4.0
    faces, points = tf.make_box_mesh(
        w, h, d, dtype=dtype, index_dtype=index_dtype)

    computed = tf.area((faces, points))
    expected = 2 * (w * h + h * d + w * d)
    np.testing.assert_allclose(computed, expected, rtol=1e-5)


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_mesh_area_with_mesh_object(dtype):
    """Area works with Mesh object."""
    w, h, d = 2.0, 3.0, 4.0
    faces, points = tf.make_box_mesh(w, h, d, dtype=dtype)
    mesh = tf.Mesh(faces, points)

    computed = tf.area(mesh)
    expected = 2 * (w * h + h * d + w * d)
    np.testing.assert_allclose(computed, expected, rtol=1e-5)


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_mesh_area_unit_cube(dtype):
    """Unit cube has surface area 6."""
    faces, points = tf.make_box_mesh(1.0, 1.0, 1.0, dtype=dtype)

    computed = tf.area((faces, points))
    np.testing.assert_allclose(computed, 6.0, rtol=1e-5)


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_mesh_area_sphere(dtype):
    """Sphere surface area matches 4*pi*r^2."""
    radius = 2.0
    faces, points = tf.make_sphere_mesh(
        radius, stacks=40, segments=40, dtype=dtype)

    computed = tf.area((faces, points))
    expected = 4 * np.pi * radius**2
    # Approximate due to triangulation
    np.testing.assert_allclose(computed, expected, rtol=0.01)


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_mesh_area_cylinder(dtype):
    """Cylinder surface area matches 2*pi*r^2 + 2*pi*r*h."""
    radius, height = 1.5, 3.0
    faces, points = tf.make_cylinder_mesh(
        radius, height, segments=64, dtype=dtype)

    computed = tf.area((faces, points))
    expected = 2 * np.pi * radius**2 + 2 * np.pi * radius * height
    # Approximate due to triangulation
    np.testing.assert_allclose(computed, expected, rtol=0.02)


@pytest.mark.parametrize("dtype", REAL_DTYPES)
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
def test_mesh_area_matches_manual(dtype, index_dtype):
    """Mesh area matches manual computation."""
    faces, points = tf.make_box_mesh(
        2.0, 3.0, 4.0, dtype=dtype, index_dtype=index_dtype)

    computed = tf.area((faces, points))
    expected = manual_mesh_area(faces, points)
    np.testing.assert_allclose(computed, expected, rtol=1e-5)


# ==============================================================================
# Volume Tests
# ==============================================================================

@pytest.mark.parametrize("dtype", REAL_DTYPES)
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
def test_volume_box(dtype, index_dtype):
    """Box volume matches w*h*d."""
    w, h, d = 2.0, 3.0, 4.0
    faces, points = tf.make_box_mesh(
        w, h, d, dtype=dtype, index_dtype=index_dtype)

    computed = tf.volume((faces, points))
    expected = w * h * d
    np.testing.assert_allclose(computed, expected, rtol=1e-5)


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_volume_with_mesh_object(dtype):
    """Volume works with Mesh object."""
    w, h, d = 2.0, 3.0, 4.0
    faces, points = tf.make_box_mesh(w, h, d, dtype=dtype)
    mesh = tf.Mesh(faces, points)

    computed = tf.volume(mesh)
    expected = w * h * d
    np.testing.assert_allclose(computed, expected, rtol=1e-5)


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_volume_unit_cube(dtype):
    """Unit cube has volume 1."""
    faces, points = tf.make_box_mesh(1.0, 1.0, 1.0, dtype=dtype)

    computed = tf.volume((faces, points))
    np.testing.assert_allclose(computed, 1.0, rtol=1e-5)


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_volume_sphere(dtype):
    """Sphere volume matches (4/3)*pi*r^3."""
    radius = 2.0
    faces, points = tf.make_sphere_mesh(
        radius, stacks=40, segments=40, dtype=dtype)

    computed = tf.volume((faces, points))
    expected = (4 / 3) * np.pi * radius**3
    # Approximate due to triangulation
    np.testing.assert_allclose(computed, expected, rtol=0.01)


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_volume_cylinder(dtype):
    """Cylinder volume matches pi*r^2*h."""
    radius, height = 1.5, 3.0
    faces, points = tf.make_cylinder_mesh(
        radius, height, segments=64, dtype=dtype)

    computed = tf.volume((faces, points))
    expected = np.pi * radius**2 * height
    # Approximate due to triangulation
    np.testing.assert_allclose(computed, expected, rtol=0.01)


@pytest.mark.parametrize("dtype", REAL_DTYPES)
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
def test_volume_matches_manual(dtype, index_dtype):
    """Volume matches manual signed volume computation."""
    faces, points = tf.make_box_mesh(
        2.0, 3.0, 4.0, dtype=dtype, index_dtype=index_dtype)

    computed = tf.volume((faces, points))
    expected = abs(manual_signed_volume(faces, points))
    np.testing.assert_allclose(computed, expected, rtol=1e-5)


# ==============================================================================
# Signed Volume Tests
# ==============================================================================

@pytest.mark.parametrize("dtype", REAL_DTYPES)
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
def test_signed_volume_positive(dtype, index_dtype):
    """Outward-facing normals give positive volume."""
    faces, points = tf.make_box_mesh(
        1.0, 1.0, 1.0, dtype=dtype, index_dtype=index_dtype)

    sv = tf.signed_volume((faces, points))
    assert sv > 0, "Outward normals should give positive volume"


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_signed_volume_flipped(dtype):
    """Flipped faces give negative volume."""
    faces, points = tf.make_box_mesh(1.0, 1.0, 1.0, dtype=dtype)

    sv = tf.signed_volume((faces, points))
    # Flip faces by reversing vertex order
    sv_flipped = tf.signed_volume((faces[:, ::-1], points))

    assert sv > 0, "Original should be positive"
    assert sv_flipped < 0, "Flipped should be negative"
    np.testing.assert_allclose(abs(sv), abs(sv_flipped), rtol=1e-5)


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_volume_is_abs_signed_volume(dtype):
    """volume() equals abs(signed_volume())."""
    faces, points = tf.make_box_mesh(2.0, 3.0, 4.0, dtype=dtype)

    sv = tf.signed_volume((faces, points))
    v = tf.volume((faces, points))

    np.testing.assert_allclose(v, abs(sv), rtol=1e-10)


@pytest.mark.parametrize("dtype", REAL_DTYPES)
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
def test_signed_volume_matches_manual(dtype, index_dtype):
    """Signed volume matches manual computation."""
    faces, points = tf.make_box_mesh(
        2.0, 3.0, 4.0, dtype=dtype, index_dtype=index_dtype)

    computed = tf.signed_volume((faces, points))
    expected = manual_signed_volume(faces, points)
    np.testing.assert_allclose(computed, expected, rtol=1e-5)


# ==============================================================================
# Error Handling Tests
# ==============================================================================

def test_signed_volume_requires_3d():
    """signed_volume raises error for 2D mesh."""
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    points = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float32)

    with pytest.raises(ValueError, match="3D"):
        tf.signed_volume((faces, points))


def test_volume_requires_3d():
    """volume raises error for 2D mesh."""
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    points = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float32)

    with pytest.raises(ValueError, match="3D"):
        tf.volume((faces, points))


# ==============================================================================
# Dynamic Mesh (OffsetBlockedArray) Tests
# ==============================================================================

@pytest.mark.parametrize("dtype", REAL_DTYPES)
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
def test_area_dynamic_mesh_tuple(dtype, index_dtype):
    """Area works with (OffsetBlockedArray, points) tuple."""
    # Create a simple mesh: two triangles forming a square
    offsets = np.array([0, 3, 6], dtype=index_dtype)
    data = np.array([0, 1, 2, 0, 2, 3], dtype=index_dtype)
    dyn_faces = tf.OffsetBlockedArray(offsets, data)

    points = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0]
    ], dtype=dtype)

    computed = tf.area((dyn_faces, points))
    expected = 1.0  # Unit square
    np.testing.assert_allclose(computed, expected, rtol=1e-5)


@pytest.mark.parametrize("dtype", REAL_DTYPES)
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
def test_volume_dynamic_mesh_tuple(dtype, index_dtype):
    """Volume works with (OffsetBlockedArray, points) tuple."""
    # Create box mesh and convert to dynamic
    faces, points = tf.make_box_mesh(
        2.0, 3.0, 4.0, dtype=dtype, index_dtype=index_dtype)

    # Convert to OffsetBlockedArray
    dyn_faces = tf.as_offset_blocked(faces)

    computed = tf.volume((dyn_faces, points))
    expected = 2.0 * 3.0 * 4.0
    np.testing.assert_allclose(computed, expected, rtol=1e-5)


@pytest.mark.parametrize("dtype", REAL_DTYPES)
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
def test_signed_volume_dynamic_mesh_tuple(dtype, index_dtype):
    """Signed volume works with (OffsetBlockedArray, points) tuple."""
    # Create box mesh and convert to dynamic
    faces, points = tf.make_box_mesh(
        2.0, 3.0, 4.0, dtype=dtype, index_dtype=index_dtype)

    # Convert to OffsetBlockedArray
    dyn_faces = tf.as_offset_blocked(faces)

    computed = tf.signed_volume((dyn_faces, points))
    expected = 2.0 * 3.0 * 4.0
    np.testing.assert_allclose(computed, expected, rtol=1e-5)
    assert computed > 0, "Outward normals should give positive volume"


@pytest.mark.parametrize("dtype", REAL_DTYPES)
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
def test_area_dynamic_mesh_object(dtype, index_dtype):
    """Area works with Mesh object created from OffsetBlockedArray."""
    faces, points = tf.make_box_mesh(
        2.0, 3.0, 4.0, dtype=dtype, index_dtype=index_dtype)
    dyn_faces = tf.as_offset_blocked(faces)
    mesh = tf.Mesh(dyn_faces, points)

    computed = tf.area(mesh)
    expected = 2 * (2.0 * 3.0 + 3.0 * 4.0 + 2.0 * 4.0)
    np.testing.assert_allclose(computed, expected, rtol=1e-5)


@pytest.mark.parametrize("dtype", REAL_DTYPES)
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
def test_volume_dynamic_mesh_object(dtype, index_dtype):
    """Volume works with Mesh object created from OffsetBlockedArray."""
    faces, points = tf.make_box_mesh(
        2.0, 3.0, 4.0, dtype=dtype, index_dtype=index_dtype)
    dyn_faces = tf.as_offset_blocked(faces)
    mesh = tf.Mesh(dyn_faces, points)

    computed = tf.volume(mesh)
    expected = 2.0 * 3.0 * 4.0
    np.testing.assert_allclose(computed, expected, rtol=1e-5)


@pytest.mark.parametrize("dtype", REAL_DTYPES)
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
def test_signed_volume_dynamic_mesh_object(dtype, index_dtype):
    """Signed volume works with Mesh object created from OffsetBlockedArray."""
    faces, points = tf.make_box_mesh(
        2.0, 3.0, 4.0, dtype=dtype, index_dtype=index_dtype)
    dyn_faces = tf.as_offset_blocked(faces)
    mesh = tf.Mesh(dyn_faces, points)

    computed = tf.signed_volume(mesh)
    expected = 2.0 * 3.0 * 4.0
    np.testing.assert_allclose(computed, expected, rtol=1e-5)
    assert computed > 0, "Outward normals should give positive volume"


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_area_dynamic_mesh_mixed_ngons(dtype):
    """Area works with mixed n-gon dynamic mesh."""
    # Triangle + quad + pentagon (all in z=0 plane)
    offsets = np.array([0, 3, 7, 12], dtype=np.int32)
    data = np.array([
        0, 1, 2,           # triangle
        3, 4, 5, 6,        # quad
        7, 8, 9, 10, 11    # pentagon
    ], dtype=np.int32)
    dyn_faces = tf.OffsetBlockedArray(offsets, data)

    # Triangle: vertices at (0,0), (1,0), (0.5, 0.866) -> area ≈ 0.433
    # Quad: unit square -> area = 1.0
    # Pentagon: regular pentagon with r=1 -> area ≈ 2.377
    points = np.array([
        # Triangle (equilateral, side=1)
        [0, 0, 0],
        [1, 0, 0],
        [0.5, np.sqrt(3)/2, 0],
        # Quad (unit square)
        [2, 0, 0],
        [3, 0, 0],
        [3, 1, 0],
        [2, 1, 0],
        # Pentagon (regular, r=1)
        [5 + np.cos(0), np.sin(0), 0],
        [5 + np.cos(2*np.pi/5), np.sin(2*np.pi/5), 0],
        [5 + np.cos(4*np.pi/5), np.sin(4*np.pi/5), 0],
        [5 + np.cos(6*np.pi/5), np.sin(6*np.pi/5), 0],
        [5 + np.cos(8*np.pi/5), np.sin(8*np.pi/5), 0],
    ], dtype=dtype)

    computed = tf.area((dyn_faces, points))

    # Manual calculation
    triangle_area = np.sqrt(3) / 4  # equilateral triangle side=1
    quad_area = 1.0
    pentagon_area = (5/2) * 1**2 * np.sin(2*np.pi/5)  # regular pentagon r=1

    expected = triangle_area + quad_area + pentagon_area
    np.testing.assert_allclose(computed, expected, rtol=1e-4)


# ==============================================================================
# 2D Mesh Area Tests
# ==============================================================================

@pytest.mark.parametrize("dtype", REAL_DTYPES)
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
def test_area_2d_mesh(dtype, index_dtype):
    """Area works for 2D mesh (plane)."""
    faces, points = tf.make_plane_mesh(
        4.0, 3.0, dtype=dtype, index_dtype=index_dtype)

    computed = tf.area((faces, points))
    expected = 4.0 * 3.0
    np.testing.assert_allclose(computed, expected, rtol=1e-5)


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
