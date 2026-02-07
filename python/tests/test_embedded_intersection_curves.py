"""
Tests for embedded_intersection_curves

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/trueform
"""

import sys
import numpy as np
import pytest
import trueform as tf


REAL_DTYPES = [np.float32, np.float64]
INDEX_DTYPES = [np.int32, np.int64]
MESH_TYPES = ['triangle', 'dynamic']


def prepare_mesh(mesh):
    """Build required structures for operations."""
    mesh.build_tree()
    mesh.build_face_membership()
    mesh.build_manifold_edge_link()
    return mesh


def make_mesh(faces, points, mesh_type='triangle'):
    """Create mesh with specified type."""
    if isinstance(faces, tf.OffsetBlockedArray):
        return tf.Mesh(faces, points)

    if mesh_type == 'dynamic':
        dyn_faces = tf.as_offset_blocked(faces)
        return tf.Mesh(dyn_faces, points)
    return tf.Mesh(faces, points)


def get_num_faces(faces):
    """Get number of faces for both ndarray and OffsetBlockedArray."""
    if isinstance(faces, tf.OffsetBlockedArray):
        return len(faces)
    return faces.shape[0]


# ==============================================================================
# Test 1: Overlapping Spheres - Basic Embedding
# ==============================================================================

@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("dtype", REAL_DTYPES)
@pytest.mark.parametrize("mesh_type", MESH_TYPES)
def test_overlapping_spheres(index_dtype, dtype, mesh_type):
    """Embed intersection curve between overlapping spheres."""
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

    original_volume = tf.volume((faces1, points1))
    original_area = tf.area((faces1, points1))

    result_faces, result_points = tf.embedded_intersection_curves(sphere1, sphere2)

    # Topology preserved
    result_mesh = tf.Mesh(result_faces, result_points)
    assert tf.is_manifold(result_mesh)
    assert tf.is_closed(result_mesh)

    # Volume preserved
    result_volume = tf.volume((result_faces, result_points))
    np.testing.assert_allclose(result_volume, original_volume, rtol=0.01)

    # Area preserved
    result_area = tf.area((result_faces, result_points))
    np.testing.assert_allclose(result_area, original_area, rtol=0.01)

    # More faces than original (some were split)
    assert get_num_faces(result_faces) >= get_num_faces(faces1)


# ==============================================================================
# Test 2: Overlapping Spheres - With Curves Return
# ==============================================================================

@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("dtype", REAL_DTYPES)
@pytest.mark.parametrize("mesh_type", MESH_TYPES)
def test_overlapping_spheres_with_curves(index_dtype, dtype, mesh_type):
    """Embed intersection curve and return the curve geometry."""
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

    (result_faces, result_points), (paths, curve_points) = tf.embedded_intersection_curves(
        sphere1, sphere2, return_curves=True)

    # Topology preserved
    result_mesh = tf.Mesh(result_faces, result_points)
    assert tf.is_manifold(result_mesh)
    assert tf.is_closed(result_mesh)

    # Curves should form a closed loop (intersection circle)
    assert len(paths) == 1
    assert len(curve_points) > 0

    # The curve should be closed
    path = paths[0]
    assert path[0] == path[-1]


# ==============================================================================
# Test 3: Non-Overlapping Meshes - No Intersection
# ==============================================================================

@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("dtype", REAL_DTYPES)
@pytest.mark.parametrize("mesh_type", MESH_TYPES)
def test_non_overlapping(index_dtype, dtype, mesh_type):
    """Non-overlapping meshes return mesh unchanged."""
    faces1, points1 = tf.make_box_mesh(1.0, 1.0, 1.0, dtype=dtype, index_dtype=index_dtype)
    box1 = prepare_mesh(make_mesh(faces1, points1, mesh_type))

    faces2, points2 = tf.make_box_mesh(1.0, 1.0, 1.0, dtype=dtype, index_dtype=index_dtype)
    points2_translated = points2.copy()
    points2_translated[:, 0] += 5.0
    box2 = prepare_mesh(make_mesh(faces2, points2_translated, mesh_type))

    original_num_faces = get_num_faces(faces1)
    original_num_points = points1.shape[0]
    original_volume = tf.volume((faces1, points1))

    result_faces, result_points = tf.embedded_intersection_curves(box1, box2)

    # Same face and point count
    assert get_num_faces(result_faces) == original_num_faces
    assert result_points.shape[0] == original_num_points

    # Volume unchanged
    result_volume = tf.volume((result_faces, result_points))
    np.testing.assert_allclose(result_volume, original_volume, rtol=0.001)

    # Topology preserved
    result_mesh = tf.Mesh(result_faces, result_points)
    assert tf.is_manifold(result_mesh)
    assert tf.is_closed(result_mesh)


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_non_overlapping_with_curves(index_dtype, dtype):
    """Non-overlapping meshes return no curves."""
    faces1, points1 = tf.make_box_mesh(1.0, 1.0, 1.0, dtype=dtype, index_dtype=index_dtype)
    box1 = prepare_mesh(make_mesh(faces1, points1))

    faces2, points2 = tf.make_box_mesh(1.0, 1.0, 1.0, dtype=dtype, index_dtype=index_dtype)
    points2_translated = points2.copy()
    points2_translated[:, 0] += 5.0
    box2 = prepare_mesh(make_mesh(faces2, points2_translated))

    (result_faces, result_points), (paths, curve_points) = tf.embedded_intersection_curves(
        box1, box2, return_curves=True)

    assert len(paths) == 0
    assert len(curve_points) == 0


# ==============================================================================
# Test 4: Overlapping Boxes
# ==============================================================================

@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("dtype", REAL_DTYPES)
@pytest.mark.parametrize("mesh_type", MESH_TYPES)
def test_overlapping_boxes(index_dtype, dtype, mesh_type):
    """Embed intersection curves between overlapping boxes."""
    faces1, points1 = tf.make_box_mesh(1.0, 1.0, 1.0, dtype=dtype, index_dtype=index_dtype)
    box1 = prepare_mesh(make_mesh(faces1, points1, mesh_type))

    faces2, points2 = tf.make_box_mesh(1.0, 1.0, 1.0, dtype=dtype, index_dtype=index_dtype)
    points2_translated = points2.copy()
    points2_translated[:, 0] += 0.5
    box2 = prepare_mesh(make_mesh(faces2, points2_translated, mesh_type))

    original_volume = tf.volume((faces1, points1))
    original_area = tf.area((faces1, points1))

    (result_faces, result_points), (paths, curve_points) = tf.embedded_intersection_curves(
        box1, box2, return_curves=True)

    # Topology preserved
    result_mesh = tf.Mesh(result_faces, result_points)
    assert tf.is_manifold(result_mesh)
    assert tf.is_closed(result_mesh)

    # Volume preserved
    result_volume = tf.volume((result_faces, result_points))
    np.testing.assert_allclose(result_volume, original_volume, rtol=0.01)

    # Area preserved
    result_area = tf.area((result_faces, result_points))
    np.testing.assert_allclose(result_area, original_area, rtol=0.01)

    # Box-box intersection forms closed loop(s)
    assert len(paths) >= 1


# ==============================================================================
# Test 5: Nested Spheres - No Surface Intersection
# ==============================================================================

@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("dtype", REAL_DTYPES)
@pytest.mark.parametrize("mesh_type", MESH_TYPES)
def test_nested_spheres(index_dtype, dtype, mesh_type):
    """Nested spheres have no surface intersection."""
    outer_radius = dtype(2.0)
    inner_radius = dtype(1.0)

    faces1, points1 = tf.make_sphere_mesh(
        outer_radius, stacks=40, segments=40, dtype=dtype, index_dtype=index_dtype)
    outer = prepare_mesh(make_mesh(faces1, points1, mesh_type))

    faces2, points2 = tf.make_sphere_mesh(
        inner_radius, stacks=30, segments=30, dtype=dtype, index_dtype=index_dtype)
    inner = prepare_mesh(make_mesh(faces2, points2, mesh_type))

    original_num_faces = get_num_faces(faces1)
    original_num_points = points1.shape[0]
    original_volume = tf.volume((faces1, points1))

    (result_faces, result_points), (paths, curve_points) = tf.embedded_intersection_curves(
        outer, inner, return_curves=True)

    # Same face and point count
    assert get_num_faces(result_faces) == original_num_faces
    assert result_points.shape[0] == original_num_points

    # Volume unchanged
    result_volume = tf.volume((result_faces, result_points))
    np.testing.assert_allclose(result_volume, original_volume, rtol=0.01)

    # Topology preserved
    result_mesh = tf.Mesh(result_faces, result_points)
    assert tf.is_manifold(result_mesh)
    assert tf.is_closed(result_mesh)

    # No intersection curves
    assert len(paths) == 0


# ==============================================================================
# Test 6: Mixed Index Types (int32 x int64 and int64 x int32)
# ==============================================================================

@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_mixed_index_types_int32_int64(dtype):
    """int32 mesh with int64 mesh works."""
    faces1, points1 = tf.make_sphere_mesh(1.0, stacks=30, segments=30, dtype=dtype, index_dtype=np.int32)
    sphere1 = prepare_mesh(tf.Mesh(faces1, points1))

    faces2, points2 = tf.make_sphere_mesh(1.0, stacks=30, segments=30, dtype=dtype, index_dtype=np.int64)
    points2[:, 0] += 1.0
    sphere2 = prepare_mesh(tf.Mesh(faces2, points2))

    result_faces, result_points = tf.embedded_intersection_curves(sphere1, sphere2)

    result_mesh = tf.Mesh(result_faces, result_points)
    assert tf.is_manifold(result_mesh)
    assert tf.is_closed(result_mesh)


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_mixed_index_types_int64_int32(dtype):
    """int64 mesh with int32 mesh works (asymmetric case)."""
    faces1, points1 = tf.make_sphere_mesh(1.0, stacks=30, segments=30, dtype=dtype, index_dtype=np.int64)
    sphere1 = prepare_mesh(tf.Mesh(faces1, points1))

    faces2, points2 = tf.make_sphere_mesh(1.0, stacks=30, segments=30, dtype=dtype, index_dtype=np.int32)
    points2[:, 0] += 1.0
    sphere2 = prepare_mesh(tf.Mesh(faces2, points2))

    result_faces, result_points = tf.embedded_intersection_curves(sphere1, sphere2)

    result_mesh = tf.Mesh(result_faces, result_points)
    assert tf.is_manifold(result_mesh)
    assert tf.is_closed(result_mesh)


# ==============================================================================
# Test 7: Mixed Mesh Types
# ==============================================================================

@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_mixed_mesh_types_triangle_dynamic(index_dtype, dtype):
    """Triangle mesh0 with dynamic mesh1 returns triangle result."""
    faces1, points1 = tf.make_box_mesh(1.0, 1.0, 1.0, dtype=dtype, index_dtype=index_dtype)
    box1 = prepare_mesh(make_mesh(faces1, points1, 'triangle'))

    faces2, points2 = tf.make_box_mesh(1.0, 1.0, 1.0, dtype=dtype, index_dtype=index_dtype)
    points2_translated = points2.copy()
    points2_translated[:, 0] += 0.5
    box2 = prepare_mesh(make_mesh(faces2, points2_translated, 'dynamic'))

    result_faces, result_points = tf.embedded_intersection_curves(box1, box2)

    # Result type depends only on mesh0 (triangle), not mesh1
    assert isinstance(result_faces, np.ndarray)

    result_mesh = tf.Mesh(result_faces, result_points)
    assert tf.is_manifold(result_mesh)
    assert tf.is_closed(result_mesh)


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_mixed_mesh_types_dynamic_triangle(index_dtype, dtype):
    """Dynamic mesh0 with triangle mesh1 returns dynamic result."""
    faces1, points1 = tf.make_box_mesh(1.0, 1.0, 1.0, dtype=dtype, index_dtype=index_dtype)
    box1 = prepare_mesh(make_mesh(faces1, points1, 'dynamic'))

    faces2, points2 = tf.make_box_mesh(1.0, 1.0, 1.0, dtype=dtype, index_dtype=index_dtype)
    points2_translated = points2.copy()
    points2_translated[:, 0] += 0.5
    box2 = prepare_mesh(make_mesh(faces2, points2_translated, 'triangle'))

    result_faces, result_points = tf.embedded_intersection_curves(box1, box2)

    # Result type depends only on mesh0 (dynamic), not mesh1
    assert isinstance(result_faces, tf.OffsetBlockedArray)

    result_mesh = tf.Mesh(result_faces, result_points)
    assert tf.is_manifold(result_mesh)
    assert tf.is_closed(result_mesh)


# ==============================================================================
# Test 8: Error Handling
# ==============================================================================

def test_rejects_non_mesh_input():
    """Rejects non-Mesh inputs."""
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    mesh = prepare_mesh(tf.Mesh(faces, points))

    with pytest.raises(TypeError, match="must be a Mesh"):
        tf.embedded_intersection_curves((faces, points), mesh)


def test_rejects_2d_meshes():
    """Rejects 2D meshes."""
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    points_2d = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float32)
    mesh_2d = tf.Mesh(faces, points_2d)

    points_3d = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    mesh_3d = prepare_mesh(tf.Mesh(faces, points_3d))

    with pytest.raises(ValueError, match="3D"):
        tf.embedded_intersection_curves(mesh_2d, mesh_3d)


def test_rejects_mismatched_dtypes():
    """Rejects meshes with different real dtypes."""
    faces = np.array([[0, 1, 2]], dtype=np.int32)

    mesh_f32 = prepare_mesh(tf.Mesh(faces, np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)))
    mesh_f64 = prepare_mesh(tf.Mesh(faces, np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)))

    with pytest.raises(ValueError, match="dtype"):
        tf.embedded_intersection_curves(mesh_f32, mesh_f64)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
