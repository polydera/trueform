"""
Tests for boolean operations on meshes

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/trueform
"""

import sys
import numpy as np
import pytest
import trueform as tf


# Test parameters
REAL_DTYPES = [np.float32, np.float64]
INDEX_DTYPES = [np.int32, np.int64]
MESH_TYPES = ['triangle', 'dynamic']


# ==============================================================================
# Helper functions
# ==============================================================================

def prepare_mesh(mesh):
    """Build required structures for boolean operations."""
    mesh.build_tree()
    mesh.build_face_membership()
    mesh.build_manifold_edge_link()
    return mesh


def make_mesh(faces, points, mesh_type='triangle'):
    """Create mesh with specified type."""
    # If faces is already OffsetBlockedArray, use directly
    if isinstance(faces, tf.OffsetBlockedArray):
        return tf.Mesh(faces, points)

    if mesh_type == 'dynamic':
        dyn_faces = tf.as_offset_blocked(faces)
        return tf.Mesh(dyn_faces, points)
    return tf.Mesh(faces, points)


def verify_topology(faces, points):
    """Verify result is manifold and closed."""
    mesh = tf.Mesh(faces, points)
    mesh.build_face_membership()

    boundaries = tf.boundary_paths(mesh)
    non_manifold = tf.non_manifold_edges(mesh)

    assert len(boundaries) == 0, f"Result should be closed, found {len(boundaries)} boundary loops"
    assert len(non_manifold) == 0, f"Result should be manifold, found {len(non_manifold)} non-manifold edges"


def get_num_faces(faces):
    """Get number of faces for both ndarray and OffsetBlockedArray."""
    if isinstance(faces, tf.OffsetBlockedArray):
        return len(faces)
    return faces.shape[0]


# ==============================================================================
# Test 1: Bicylinder (Steinmetz Solid) - Intersection of Perpendicular Cylinders
# ==============================================================================

@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("dtype", REAL_DTYPES)
@pytest.mark.parametrize("mesh_type", MESH_TYPES)
def test_bicylinder_intersection(index_dtype, dtype, mesh_type):
    """Intersection of two perpendicular cylinders creates a Steinmetz solid."""
    radius = dtype(1.0)
    height = dtype(4.0)

    # Create vertical cylinder
    faces1, points1 = tf.make_cylinder_mesh(
        radius, height, segments=100, dtype=dtype, index_dtype=index_dtype)
    mesh1 = make_mesh(faces1, points1, mesh_type)
    new_faces1 = tf.ensure_positive_orientation(mesh1)
    mesh1 = prepare_mesh(make_mesh(new_faces1, points1, mesh_type))

    # Create horizontal cylinder (rotated 90 degrees around X)
    faces2, points2 = tf.make_cylinder_mesh(
        radius, height, segments=100, dtype=dtype, index_dtype=index_dtype)
    rotated_points = np.column_stack([
        points2[:, 0],
        -points2[:, 2],
        points2[:, 1]
    ]).astype(dtype)
    mesh2 = make_mesh(faces2, rotated_points, mesh_type)
    new_faces2 = tf.ensure_positive_orientation(mesh2)
    mesh2 = prepare_mesh(make_mesh(new_faces2, rotated_points, mesh_type))

    # Boolean intersection
    (result_faces, result_points), labels = tf.boolean_intersection(mesh1, mesh2)

    # Verify topology
    verify_topology(result_faces, result_points)

    # Steinmetz solid volume = 16r³/3
    expected_volume = 16 * float(radius)**3 / 3
    result_volume = tf.volume((result_faces, result_points))

    np.testing.assert_allclose(result_volume, expected_volume, rtol=0.02)


# ==============================================================================
# Test 2: Nested Spheres - Boolean Operations with Volume Verification
# ==============================================================================

@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("dtype", REAL_DTYPES)
@pytest.mark.parametrize("mesh_type", MESH_TYPES)
def test_nested_spheres_merge(index_dtype, dtype, mesh_type):
    """Merge of nested spheres equals outer sphere."""
    outer_radius = dtype(2.0)
    inner_radius = dtype(1.0)

    faces1, points1 = tf.make_sphere_mesh(
        outer_radius, stacks=60, segments=60, dtype=dtype, index_dtype=index_dtype)
    outer = prepare_mesh(make_mesh(faces1, points1, mesh_type))

    faces2, points2 = tf.make_sphere_mesh(
        inner_radius, stacks=40, segments=40, dtype=dtype, index_dtype=index_dtype)
    inner = prepare_mesh(make_mesh(faces2, points2, mesh_type))

    # Merge
    (result_faces, result_points), labels = tf.boolean_union(outer, inner)

    # Verify topology
    verify_topology(result_faces, result_points)

    # Result should be just the outer sphere
    assert get_num_faces(result_faces) == get_num_faces(faces1)
    assert result_points.shape[0] == points1.shape[0]

    # Volume = outer sphere
    outer_volume = (4/3) * np.pi * float(outer_radius)**3
    result_volume = tf.volume((result_faces, result_points))

    np.testing.assert_allclose(result_volume, outer_volume, rtol=0.01)


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("dtype", REAL_DTYPES)
@pytest.mark.parametrize("mesh_type", MESH_TYPES)
def test_nested_spheres_difference(index_dtype, dtype, mesh_type):
    """Difference of nested spheres creates hollow sphere."""
    outer_radius = dtype(2.0)
    inner_radius = dtype(1.0)

    faces1, points1 = tf.make_sphere_mesh(
        outer_radius, stacks=60, segments=60, dtype=dtype, index_dtype=index_dtype)
    outer = prepare_mesh(make_mesh(faces1, points1, mesh_type))

    faces2, points2 = tf.make_sphere_mesh(
        inner_radius, stacks=40, segments=40, dtype=dtype, index_dtype=index_dtype)
    inner = prepare_mesh(make_mesh(faces2, points2, mesh_type))

    # Difference
    (result_faces, result_points), labels = tf.boolean_difference(outer, inner)

    # Verify topology
    verify_topology(result_faces, result_points)

    # Result has both surfaces
    assert get_num_faces(result_faces) == get_num_faces(faces1) + get_num_faces(faces2)
    assert result_points.shape[0] == points1.shape[0] + points2.shape[0]

    # Volume = outer - inner
    outer_volume = (4/3) * np.pi * float(outer_radius)**3
    inner_volume = (4/3) * np.pi * float(inner_radius)**3
    expected_volume = outer_volume - inner_volume
    result_volume = tf.volume((result_faces, result_points))

    np.testing.assert_allclose(result_volume, expected_volume, rtol=0.01)


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("dtype", REAL_DTYPES)
@pytest.mark.parametrize("mesh_type", MESH_TYPES)
def test_nested_spheres_intersection(index_dtype, dtype, mesh_type):
    """Intersection of nested spheres equals inner sphere."""
    outer_radius = dtype(2.0)
    inner_radius = dtype(1.0)

    faces1, points1 = tf.make_sphere_mesh(
        outer_radius, stacks=60, segments=60, dtype=dtype, index_dtype=index_dtype)
    outer = prepare_mesh(make_mesh(faces1, points1, mesh_type))

    faces2, points2 = tf.make_sphere_mesh(
        inner_radius, stacks=40, segments=40, dtype=dtype, index_dtype=index_dtype)
    inner = prepare_mesh(make_mesh(faces2, points2, mesh_type))

    # Intersection
    (result_faces, result_points), labels = tf.boolean_intersection(outer, inner)

    # Verify topology
    verify_topology(result_faces, result_points)

    # Result is just the inner sphere
    assert get_num_faces(result_faces) == get_num_faces(faces2)
    assert result_points.shape[0] == points2.shape[0]

    # Volume = inner sphere
    inner_volume = (4/3) * np.pi * float(inner_radius)**3
    result_volume = tf.volume((result_faces, result_points))

    np.testing.assert_allclose(result_volume, inner_volume, rtol=0.01)


# ==============================================================================
# Test 3: Overlapping Boxes - All Boolean Operations
# ==============================================================================

@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("dtype", REAL_DTYPES)
@pytest.mark.parametrize("mesh_type", MESH_TYPES)
def test_overlapping_boxes_union(index_dtype, dtype, mesh_type):
    """Union of overlapping boxes."""
    faces1, points1 = tf.make_box_mesh(1.0, 1.0, 1.0, dtype=dtype, index_dtype=index_dtype)
    box1 = prepare_mesh(make_mesh(faces1, points1, mesh_type))

    faces2, points2 = tf.make_box_mesh(1.0, 1.0, 1.0, dtype=dtype, index_dtype=index_dtype)
    points2_translated = points2.copy()
    points2_translated[:, 0] += 0.5
    box2 = prepare_mesh(make_mesh(faces2, points2_translated, mesh_type))

    # Union
    (result_faces, result_points), labels = tf.boolean_union(box1, box2)

    # Verify topology
    verify_topology(result_faces, result_points)

    # Volume = box1 + box2 - overlap = 1 + 1 - 0.5 = 1.5
    expected_volume = 1.5
    result_volume = tf.volume((result_faces, result_points))

    np.testing.assert_allclose(result_volume, expected_volume, rtol=0.01)


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("dtype", REAL_DTYPES)
@pytest.mark.parametrize("mesh_type", MESH_TYPES)
def test_overlapping_boxes_intersection(index_dtype, dtype, mesh_type):
    """Intersection of overlapping boxes."""
    faces1, points1 = tf.make_box_mesh(1.0, 1.0, 1.0, dtype=dtype, index_dtype=index_dtype)
    box1 = prepare_mesh(make_mesh(faces1, points1, mesh_type))

    faces2, points2 = tf.make_box_mesh(1.0, 1.0, 1.0, dtype=dtype, index_dtype=index_dtype)
    points2_translated = points2.copy()
    points2_translated[:, 0] += 0.5
    box2 = prepare_mesh(make_mesh(faces2, points2_translated, mesh_type))

    # Intersection
    (result_faces, result_points), labels = tf.boolean_intersection(box1, box2)

    # Verify topology
    verify_topology(result_faces, result_points)

    # Volume = overlap = 0.5 * 1 * 1 = 0.5
    expected_volume = 0.5
    result_volume = tf.volume((result_faces, result_points))

    np.testing.assert_allclose(result_volume, expected_volume, rtol=0.01)


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("dtype", REAL_DTYPES)
@pytest.mark.parametrize("mesh_type", MESH_TYPES)
def test_overlapping_boxes_difference(index_dtype, dtype, mesh_type):
    """Difference of overlapping boxes."""
    faces1, points1 = tf.make_box_mesh(1.0, 1.0, 1.0, dtype=dtype, index_dtype=index_dtype)
    box1 = prepare_mesh(make_mesh(faces1, points1, mesh_type))

    faces2, points2 = tf.make_box_mesh(1.0, 1.0, 1.0, dtype=dtype, index_dtype=index_dtype)
    points2_translated = points2.copy()
    points2_translated[:, 0] += 0.5
    box2 = prepare_mesh(make_mesh(faces2, points2_translated, mesh_type))

    # Difference (box1 - box2)
    (result_faces, result_points), labels = tf.boolean_difference(box1, box2)

    # Verify topology
    verify_topology(result_faces, result_points)

    # Volume = box1 - overlap = 1 - 0.5 = 0.5
    expected_volume = 0.5
    result_volume = tf.volume((result_faces, result_points))

    np.testing.assert_allclose(result_volume, expected_volume, rtol=0.01)


# ==============================================================================
# Test 4: Non-Overlapping Meshes
# ==============================================================================

@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("dtype", REAL_DTYPES)
@pytest.mark.parametrize("mesh_type", MESH_TYPES)
def test_non_overlapping_union(index_dtype, dtype, mesh_type):
    """Union of non-overlapping boxes."""
    faces1, points1 = tf.make_box_mesh(1.0, 1.0, 1.0, dtype=dtype, index_dtype=index_dtype)
    box1 = prepare_mesh(make_mesh(faces1, points1, mesh_type))

    faces2, points2 = tf.make_box_mesh(1.0, 1.0, 1.0, dtype=dtype, index_dtype=index_dtype)
    points2_translated = points2.copy()
    points2_translated[:, 0] += 5.0
    box2 = prepare_mesh(make_mesh(faces2, points2_translated, mesh_type))

    # Union
    (result_faces, result_points), labels = tf.boolean_union(box1, box2)

    # Volume = 2 boxes
    expected_volume = 2.0
    result_volume = tf.volume((result_faces, result_points))

    np.testing.assert_allclose(result_volume, expected_volume, rtol=0.01)


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("dtype", REAL_DTYPES)
@pytest.mark.parametrize("mesh_type", MESH_TYPES)
def test_non_overlapping_intersection(index_dtype, dtype, mesh_type):
    """Intersection of non-overlapping boxes is empty."""
    faces1, points1 = tf.make_box_mesh(1.0, 1.0, 1.0, dtype=dtype, index_dtype=index_dtype)
    box1 = prepare_mesh(make_mesh(faces1, points1, mesh_type))

    faces2, points2 = tf.make_box_mesh(1.0, 1.0, 1.0, dtype=dtype, index_dtype=index_dtype)
    points2_translated = points2.copy()
    points2_translated[:, 0] += 5.0
    box2 = prepare_mesh(make_mesh(faces2, points2_translated, mesh_type))

    # Intersection
    (result_faces, result_points), labels = tf.boolean_intersection(box1, box2)

    # Empty result
    assert get_num_faces(result_faces) == 0


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("dtype", REAL_DTYPES)
@pytest.mark.parametrize("mesh_type", MESH_TYPES)
def test_non_overlapping_difference(index_dtype, dtype, mesh_type):
    """Difference with non-overlapping box leaves original unchanged."""
    faces1, points1 = tf.make_box_mesh(1.0, 1.0, 1.0, dtype=dtype, index_dtype=index_dtype)
    box1 = prepare_mesh(make_mesh(faces1, points1, mesh_type))

    faces2, points2 = tf.make_box_mesh(1.0, 1.0, 1.0, dtype=dtype, index_dtype=index_dtype)
    points2_translated = points2.copy()
    points2_translated[:, 0] += 5.0
    box2 = prepare_mesh(make_mesh(faces2, points2_translated, mesh_type))

    # Difference
    (result_faces, result_points), labels = tf.boolean_difference(box1, box2)

    # Volume unchanged
    expected_volume = 1.0
    result_volume = tf.volume((result_faces, result_points))

    np.testing.assert_allclose(result_volume, expected_volume, rtol=0.01)


# ==============================================================================
# Test 5: Overlapping Spheres - Two Spheres with Intersecting Surfaces
# ==============================================================================

@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("dtype", REAL_DTYPES)
@pytest.mark.parametrize("mesh_type", MESH_TYPES)
def test_overlapping_spheres_union(index_dtype, dtype, mesh_type):
    """Union of overlapping spheres."""
    radius = dtype(1.0)
    separation = dtype(1.0)

    faces1, points1 = tf.make_sphere_mesh(
        radius, stacks=50, segments=50, dtype=dtype, index_dtype=index_dtype)
    sphere1 = prepare_mesh(make_mesh(faces1, points1, mesh_type))

    faces2, points2 = tf.make_sphere_mesh(
        radius, stacks=50, segments=50, dtype=dtype, index_dtype=index_dtype)
    points2_translated = points2.copy()
    points2_translated[:, 0] += float(separation)
    sphere2 = prepare_mesh(make_mesh(faces2, points2_translated, mesh_type))

    # Union
    (result_faces, result_points), labels = tf.boolean_union(sphere1, sphere2)

    # Verify topology
    verify_topology(result_faces, result_points)

    # Lens volume formula
    r = float(radius)
    d = float(separation)
    h = 2 * r - d
    lens_volume = (np.pi * h**2 / 12) * (6 * r - h)
    sphere_volume = (4/3) * np.pi * r**3
    expected_volume = 2 * sphere_volume - lens_volume

    result_volume = tf.volume((result_faces, result_points))
    np.testing.assert_allclose(result_volume, expected_volume, rtol=0.02)


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("dtype", REAL_DTYPES)
@pytest.mark.parametrize("mesh_type", MESH_TYPES)
def test_overlapping_spheres_intersection(index_dtype, dtype, mesh_type):
    """Intersection of overlapping spheres creates lens shape."""
    radius = dtype(1.0)
    separation = dtype(1.0)

    faces1, points1 = tf.make_sphere_mesh(
        radius, stacks=50, segments=50, dtype=dtype, index_dtype=index_dtype)
    sphere1 = prepare_mesh(make_mesh(faces1, points1, mesh_type))

    faces2, points2 = tf.make_sphere_mesh(
        radius, stacks=50, segments=50, dtype=dtype, index_dtype=index_dtype)
    points2_translated = points2.copy()
    points2_translated[:, 0] += float(separation)
    sphere2 = prepare_mesh(make_mesh(faces2, points2_translated, mesh_type))

    # Intersection
    (result_faces, result_points), labels = tf.boolean_intersection(sphere1, sphere2)

    # Verify topology
    verify_topology(result_faces, result_points)

    # Lens volume
    r = float(radius)
    d = float(separation)
    h = 2 * r - d
    expected_volume = (np.pi * h**2 / 12) * (6 * r - h)

    result_volume = tf.volume((result_faces, result_points))
    np.testing.assert_allclose(result_volume, expected_volume, rtol=0.02)


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("dtype", REAL_DTYPES)
@pytest.mark.parametrize("mesh_type", MESH_TYPES)
def test_overlapping_spheres_difference(index_dtype, dtype, mesh_type):
    """Difference of overlapping spheres."""
    radius = dtype(1.0)
    separation = dtype(1.0)

    faces1, points1 = tf.make_sphere_mesh(
        radius, stacks=50, segments=50, dtype=dtype, index_dtype=index_dtype)
    sphere1 = prepare_mesh(make_mesh(faces1, points1, mesh_type))

    faces2, points2 = tf.make_sphere_mesh(
        radius, stacks=50, segments=50, dtype=dtype, index_dtype=index_dtype)
    points2_translated = points2.copy()
    points2_translated[:, 0] += float(separation)
    sphere2 = prepare_mesh(make_mesh(faces2, points2_translated, mesh_type))

    # Difference
    (result_faces, result_points), labels = tf.boolean_difference(sphere1, sphere2)

    # Verify topology
    verify_topology(result_faces, result_points)

    # Volume = sphere - lens
    r = float(radius)
    d = float(separation)
    h = 2 * r - d
    lens_volume = (np.pi * h**2 / 12) * (6 * r - h)
    sphere_volume = (4/3) * np.pi * r**3
    expected_volume = sphere_volume - lens_volume

    result_volume = tf.volume((result_faces, result_points))
    np.testing.assert_allclose(result_volume, expected_volume, rtol=0.02)


# ==============================================================================
# Test 6: Mixed Mesh Types (triangle + dynamic)
# ==============================================================================

@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_mixed_mesh_types(index_dtype, dtype):
    """Boolean between triangle and dynamic mesh."""
    faces1, points1 = tf.make_box_mesh(1.0, 1.0, 1.0, dtype=dtype, index_dtype=index_dtype)
    box1 = prepare_mesh(make_mesh(faces1, points1, 'triangle'))

    faces2, points2 = tf.make_box_mesh(1.0, 1.0, 1.0, dtype=dtype, index_dtype=index_dtype)
    points2_translated = points2.copy()
    points2_translated[:, 0] += 0.5
    box2 = prepare_mesh(make_mesh(faces2, points2_translated, 'dynamic'))

    # Union
    (result_faces, result_points), labels = tf.boolean_union(box1, box2)

    # Result should be dynamic when one input is dynamic
    assert isinstance(result_faces, tf.OffsetBlockedArray)

    # Verify topology
    verify_topology(result_faces, result_points)

    # Volume
    expected_volume = 1.5
    result_volume = tf.volume((result_faces, result_points))
    np.testing.assert_allclose(result_volume, expected_volume, rtol=0.01)


# ==============================================================================
# Test 7: Error Handling
# ==============================================================================

def test_rejects_non_mesh_input():
    """Boolean operations reject non-Mesh inputs."""
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    mesh = prepare_mesh(tf.Mesh(faces, points))

    with pytest.raises(TypeError, match="must be a Mesh"):
        tf.boolean_union((faces, points), mesh)


def test_rejects_2d_meshes():
    """Boolean operations reject 2D meshes."""
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    points_2d = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float32)
    mesh_2d = tf.Mesh(faces, points_2d)

    points_3d = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    mesh_3d = prepare_mesh(tf.Mesh(faces, points_3d))

    with pytest.raises(ValueError, match="3D"):
        tf.boolean_union(mesh_2d, mesh_3d)


def test_rejects_mismatched_dtypes():
    """Boolean operations reject meshes with different real dtypes."""
    faces = np.array([[0, 1, 2]], dtype=np.int32)

    mesh_f32 = prepare_mesh(tf.Mesh(faces, np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)))
    mesh_f64 = prepare_mesh(tf.Mesh(faces, np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)))

    with pytest.raises(ValueError, match="dtype"):
        tf.boolean_union(mesh_f32, mesh_f64)


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
