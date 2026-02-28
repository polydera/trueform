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
# Transformation Helpers
# ==============================================================================

def make_uniform_scale(s, dtype=np.float64):
    """Create a 4x4 uniform scale matrix."""
    m = np.eye(4, dtype=dtype)
    m[0, 0] = s
    m[1, 1] = s
    m[2, 2] = s
    return m


def make_rotation_z(angle_deg, dtype=np.float64):
    """Create a 4x4 rotation matrix around Z axis."""
    a = np.radians(angle_deg)
    c, s = np.cos(a), np.sin(a)
    m = np.eye(4, dtype=dtype)
    m[0, 0] = c
    m[0, 1] = -s
    m[1, 0] = s
    m[1, 1] = c
    return m


def make_translation(tx, ty, tz, dtype=np.float64):
    """Create a 4x4 translation matrix."""
    m = np.eye(4, dtype=dtype)
    m[0, 3] = tx
    m[1, 3] = ty
    m[2, 3] = tz
    return m


# ==============================================================================
# Transformation-Aware Measurement Tests (on Spheres)
# ==============================================================================

SPHERE_RADIUS = 1.0
SPHERE_STACKS = 40
SPHERE_SEGMENTS = 40


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_area_sphere_with_uniform_scale(dtype):
    """Area of a scaled sphere equals 4*pi*(s*r)^2."""
    scale = 2.0
    faces, points = tf.make_sphere_mesh(
        SPHERE_RADIUS, stacks=SPHERE_STACKS, segments=SPHERE_SEGMENTS,
        dtype=dtype)
    mesh = tf.Mesh(faces, points)
    mesh.transformation = make_uniform_scale(scale, dtype)

    computed = tf.area(mesh)
    expected = 4 * np.pi * (scale * SPHERE_RADIUS) ** 2
    np.testing.assert_allclose(computed, expected, rtol=0.01)


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_area_sphere_with_rotation(dtype):
    """Area is unchanged by rotation."""
    faces, points = tf.make_sphere_mesh(
        SPHERE_RADIUS, stacks=SPHERE_STACKS, segments=SPHERE_SEGMENTS,
        dtype=dtype)

    area_no_transform = tf.area((faces, points))

    mesh = tf.Mesh(faces, points)
    mesh.transformation = make_rotation_z(45.0, dtype)

    area_rotated = tf.area(mesh)
    np.testing.assert_allclose(area_rotated, area_no_transform, rtol=1e-5)


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_volume_sphere_with_uniform_scale(dtype):
    """Volume of a scaled sphere equals (4/3)*pi*(s*r)^3."""
    scale = 2.0
    faces, points = tf.make_sphere_mesh(
        SPHERE_RADIUS, stacks=SPHERE_STACKS, segments=SPHERE_SEGMENTS,
        dtype=dtype)
    mesh = tf.Mesh(faces, points)
    mesh.transformation = make_uniform_scale(scale, dtype)

    computed = tf.volume(mesh)
    expected = (4 / 3) * np.pi * (scale * SPHERE_RADIUS) ** 3
    np.testing.assert_allclose(computed, expected, rtol=0.01)


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_volume_sphere_with_rotation(dtype):
    """Volume is unchanged by rotation."""
    faces, points = tf.make_sphere_mesh(
        SPHERE_RADIUS, stacks=SPHERE_STACKS, segments=SPHERE_SEGMENTS,
        dtype=dtype)

    vol_no_transform = tf.volume((faces, points))

    mesh = tf.Mesh(faces, points)
    mesh.transformation = make_rotation_z(45.0, dtype)

    vol_rotated = tf.volume(mesh)
    np.testing.assert_allclose(vol_rotated, vol_no_transform, rtol=1e-5)


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_signed_volume_sphere_with_uniform_scale(dtype):
    """Signed volume scales by s^3."""
    scale = 3.0
    faces, points = tf.make_sphere_mesh(
        SPHERE_RADIUS, stacks=SPHERE_STACKS, segments=SPHERE_SEGMENTS,
        dtype=dtype)

    sv_base = tf.signed_volume((faces, points))

    mesh = tf.Mesh(faces, points)
    mesh.transformation = make_uniform_scale(scale, dtype)

    sv_scaled = tf.signed_volume(mesh)
    np.testing.assert_allclose(sv_scaled, sv_base * scale ** 3, rtol=0.01)


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_mean_edge_length_sphere_with_uniform_scale(dtype):
    """Mean edge length scales linearly with uniform scale."""
    scale = 2.0
    faces, points = tf.make_sphere_mesh(
        SPHERE_RADIUS, stacks=SPHERE_STACKS, segments=SPHERE_SEGMENTS,
        dtype=dtype)

    mel_base = tf.mean_edge_length((faces, points))

    mesh = tf.Mesh(faces, points)
    mesh.transformation = make_uniform_scale(scale, dtype)

    mel_scaled = tf.mean_edge_length(mesh)
    np.testing.assert_allclose(mel_scaled, mel_base * scale, rtol=1e-5)


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_mean_edge_length_sphere_with_rotation(dtype):
    """Mean edge length is unchanged by rotation."""
    faces, points = tf.make_sphere_mesh(
        SPHERE_RADIUS, stacks=SPHERE_STACKS, segments=SPHERE_SEGMENTS,
        dtype=dtype)

    mel_base = tf.mean_edge_length((faces, points))

    mesh = tf.Mesh(faces, points)
    mesh.transformation = make_rotation_z(45.0, dtype)

    mel_rotated = tf.mean_edge_length(mesh)
    np.testing.assert_allclose(mel_rotated, mel_base, rtol=1e-5)


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_measurements_sphere_translation_invariant(dtype):
    """Translation does not change area, volume, or mean edge length."""
    faces, points = tf.make_sphere_mesh(
        SPHERE_RADIUS, stacks=SPHERE_STACKS, segments=SPHERE_SEGMENTS,
        dtype=dtype)

    area_base = tf.area((faces, points))
    vol_base = tf.volume((faces, points))
    mel_base = tf.mean_edge_length((faces, points))

    mesh = tf.Mesh(faces, points)
    mesh.transformation = make_translation(100.0, -50.0, 25.0, dtype)

    np.testing.assert_allclose(tf.area(mesh), area_base, rtol=1e-5)
    np.testing.assert_allclose(tf.volume(mesh), vol_base, rtol=1e-5)
    np.testing.assert_allclose(tf.mean_edge_length(mesh), mel_base, rtol=1e-5)


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_measurements_sphere_composed_transform(dtype):
    """Scale + rotation gives correct scaled measurements."""
    scale = 1.5
    faces, points = tf.make_sphere_mesh(
        SPHERE_RADIUS, stacks=SPHERE_STACKS, segments=SPHERE_SEGMENTS,
        dtype=dtype)

    area_base = tf.area((faces, points))
    vol_base = tf.volume((faces, points))
    mel_base = tf.mean_edge_length((faces, points))

    mesh = tf.Mesh(faces, points)
    mesh.transformation = make_rotation_z(30.0, dtype) @ make_uniform_scale(scale, dtype)

    np.testing.assert_allclose(
        tf.area(mesh), area_base * scale ** 2, rtol=0.01)
    np.testing.assert_allclose(
        tf.volume(mesh), vol_base * scale ** 3, rtol=0.01)
    np.testing.assert_allclose(
        tf.mean_edge_length(mesh), mel_base * scale, rtol=1e-4)


# ==============================================================================
# Triangle Primitive Area Tests
# ==============================================================================

@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_area_triangle_primitive_single_3d(dtype):
    """Area of a single 3D Triangle primitive."""
    tri = tf.Triangle(np.array([
        [0, 0, 0],
        [2, 0, 0],
        [0, 2, 0]
    ], dtype=dtype))

    computed = tf.area(tri)
    assert isinstance(computed, float)
    np.testing.assert_allclose(computed, 2.0, rtol=1e-5)


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_area_triangle_primitive_single_2d(dtype):
    """Area of a single 2D Triangle primitive."""
    tri = tf.Triangle(np.array([
        [0, 0],
        [1, 0],
        [0, 1]
    ], dtype=dtype))

    computed = tf.area(tri)
    assert isinstance(computed, float)
    np.testing.assert_allclose(computed, 0.5, rtol=1e-5)


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_area_triangle_primitive_batch(dtype):
    """Area of a batch of Triangle primitives returns array."""
    data = np.array([
        [[0, 0, 0], [1, 0, 0], [0, 1, 0]],   # area = 0.5
        [[0, 0, 0], [2, 0, 0], [0, 2, 0]],   # area = 2.0
        [[0, 0, 0], [3, 0, 0], [0, 3, 0]],   # area = 4.5
    ], dtype=dtype)
    tris = tf.Triangle(data)

    computed = tf.area(tris)
    assert isinstance(computed, np.ndarray)
    assert computed.shape == (3,)
    np.testing.assert_allclose(computed, [0.5, 2.0, 4.5], rtol=1e-5)


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_area_triangle_primitive_batch_2d(dtype):
    """Area of a batch of 2D Triangle primitives."""
    data = np.array([
        [[0, 0], [1, 0], [0, 1]],   # area = 0.5
        [[0, 0], [2, 0], [0, 2]],   # area = 2.0
    ], dtype=dtype)
    tris = tf.Triangle(data)

    computed = tf.area(tris)
    assert isinstance(computed, np.ndarray)
    assert computed.shape == (2,)
    np.testing.assert_allclose(computed, [0.5, 2.0], rtol=1e-5)


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_area_triangle_primitive_unit_right(dtype):
    """Unit right triangle has area 0.5."""
    tri = tf.Triangle(a=np.array([0, 0, 0], dtype=dtype),
                      b=np.array([1, 0, 0], dtype=dtype),
                      c=np.array([0, 1, 0], dtype=dtype))

    computed = tf.area(tri)
    np.testing.assert_allclose(computed, 0.5, rtol=1e-5)


# ==============================================================================
# Polygon Primitive Area Tests (batch)
# ==============================================================================

@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_area_polygon_primitive_batch_3d(dtype):
    """Area of a batch of 3D Polygon primitives."""
    # Two unit squares
    data = np.array([
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],   # area = 1.0
        [[0, 0, 0], [2, 0, 0], [2, 2, 0], [0, 2, 0]],   # area = 4.0
    ], dtype=dtype)
    polys = tf.Polygon(data)

    computed = tf.area(polys)
    assert isinstance(computed, np.ndarray)
    assert computed.shape == (2,)
    np.testing.assert_allclose(computed, [1.0, 4.0], rtol=1e-5)


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_area_polygon_primitive_batch_2d(dtype):
    """Area of a batch of 2D Polygon primitives."""
    data = np.array([
        [[0, 0], [1, 0], [1, 1], [0, 1]],   # area = 1.0
        [[0, 0], [3, 0], [3, 2], [0, 2]],   # area = 6.0
    ], dtype=dtype)
    polys = tf.Polygon(data)

    computed = tf.area(polys)
    assert isinstance(computed, np.ndarray)
    assert computed.shape == (2,)
    np.testing.assert_allclose(computed, [1.0, 6.0], rtol=1e-5)


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_area_polygon_primitive_single_still_works(dtype):
    """Single Polygon still works after refactor."""
    poly = tf.Polygon(np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]
    ], dtype=dtype))

    computed = tf.area(poly)
    np.testing.assert_allclose(computed, 1.0, rtol=1e-5)


# ==============================================================================
# Triangle/Polygon Primitive mean_edge_length Tests
# ==============================================================================

@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_mean_edge_length_triangle_single(dtype):
    """Mean edge length of a unit right triangle."""
    tri = tf.Triangle(np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0]
    ], dtype=dtype))

    computed = tf.mean_edge_length(tri)
    # Edges: 1, 1, sqrt(2) → mean = (2 + sqrt(2)) / 3
    expected = (2.0 + np.sqrt(2)) / 3.0
    np.testing.assert_allclose(computed, expected, rtol=1e-5)


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_mean_edge_length_equilateral_triangle(dtype):
    """Equilateral triangle with side=2 has mean edge length 2."""
    tri = tf.Triangle(np.array([
        [0, 0, 0],
        [2, 0, 0],
        [1, np.sqrt(3), 0]
    ], dtype=dtype))

    computed = tf.mean_edge_length(tri)
    np.testing.assert_allclose(computed, 2.0, rtol=1e-5)


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_mean_edge_length_triangle_batch(dtype):
    """Mean edge length across a batch of triangles."""
    data = np.array([
        [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
        [[0, 0, 0], [2, 0, 0], [0, 2, 0]],
    ], dtype=dtype)
    tris = tf.Triangle(data)

    computed = tf.mean_edge_length(tris)
    # Tri 1: edges 1, 1, sqrt(2). Tri 2: edges 2, 2, 2*sqrt(2)
    # Total 6 edges, mean = (1 + 1 + sqrt(2) + 2 + 2 + 2*sqrt(2)) / 6
    expected = (6.0 + 3 * np.sqrt(2)) / 6.0
    np.testing.assert_allclose(computed, expected, rtol=1e-5)


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_mean_edge_length_polygon_single(dtype):
    """Mean edge length of a unit square polygon."""
    poly = tf.Polygon(np.array([
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0]
    ], dtype=dtype))

    computed = tf.mean_edge_length(poly)
    np.testing.assert_allclose(computed, 1.0, rtol=1e-5)


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_mean_edge_length_polygon_batch(dtype):
    """Mean edge length across a batch of polygons."""
    data = np.array([
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],   # unit square, all edges = 1
        [[0, 0, 0], [2, 0, 0], [2, 2, 0], [0, 2, 0]],   # 2x2 square, all edges = 2
    ], dtype=dtype)
    polys = tf.Polygon(data)

    computed = tf.mean_edge_length(polys)
    # 4 edges of length 1 + 4 edges of length 2 → mean = 12/8 = 1.5
    np.testing.assert_allclose(computed, 1.5, rtol=1e-5)


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_mean_edge_length_triangle_2d(dtype):
    """Mean edge length works for 2D triangles."""
    tri = tf.Triangle(np.array([
        [0, 0],
        [1, 0],
        [0, 1]
    ], dtype=dtype))

    computed = tf.mean_edge_length(tri)
    expected = (2.0 + np.sqrt(2)) / 3.0
    np.testing.assert_allclose(computed, expected, rtol=1e-5)


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
