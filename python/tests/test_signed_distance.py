"""
Tests for signed_distance (winding number: negative inside, positive outside)

Copyright (c) 2025 Žiga Sajovic, XLAB
"""

import sys
import numpy as np
import pytest
import trueform as tf


INDEX_DTYPES = [np.int32, np.int64]
REAL_DTYPES = [np.float32, np.float64]


# ==============================================================================
# Helper Functions
# ==============================================================================

def create_unit_cube(index_dtype, real_dtype):
    """A closed cube spanning [-0.5, 0.5]^3 with outward orientation."""
    points = np.array(
        [[x, y, z] for x in (-0.5, 0.5) for y in (-0.5, 0.5)
         for z in (-0.5, 0.5)],
        dtype=real_dtype,
    )
    faces = np.array(
        [[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
         [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]],
        dtype=index_dtype,
    )
    return tf.Mesh(faces, points)


def create_open_cube(index_dtype, real_dtype):
    """The same cube with its +x face dropped: a mesh with a boundary."""
    mesh = create_unit_cube(index_dtype, real_dtype)
    faces = np.array(
        [f for f in mesh.faces if not all(mesh.points[v][0] > 0.0 for v in f)],
        dtype=index_dtype,
    )
    return tf.Mesh(faces, mesh.points)


def create_triangle_mesh_2d(index_dtype, real_dtype):
    """A 2D mesh, which has no winding moments to build."""
    faces = np.array([[0, 1, 2]], dtype=index_dtype)
    points = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]], dtype=real_dtype)
    return tf.Mesh(faces, points)


# ==============================================================================
# Sign and magnitude oracles
# ==============================================================================

@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_signed_distance_single(index_dtype, real_dtype):
    mesh = create_unit_cube(index_dtype, real_dtype)

    assert tf.signed_distance(mesh, [0.0, 0.0, 0.0]) == -0.5
    assert tf.signed_distance(mesh, [2.0, 0.0, 0.0]) == 1.5


@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_signed_distance_point_query(real_dtype):
    mesh = create_unit_cube(np.int32, real_dtype)
    pt = tf.Point(np.array([2.0, 0.0, 0.0], dtype=real_dtype))
    assert tf.signed_distance(mesh, pt) == 1.5
    # either argument order
    assert tf.signed_distance(pt, mesh) == 1.5


@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_signed_distance_batch(real_dtype):
    mesh = create_unit_cube(np.int32, real_dtype)
    queries = tf.Point(
        np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=real_dtype))

    d = tf.signed_distance(mesh, queries)
    assert d.shape == (2,)
    assert d.dtype == real_dtype
    assert np.array_equal(d, np.array([-0.5, 1.5], dtype=real_dtype))


@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_signed_distance_batch_ndarray(real_dtype):
    mesh = create_unit_cube(np.int32, real_dtype)
    queries = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])

    d = tf.signed_distance(mesh, queries)
    assert d.dtype == real_dtype
    assert np.array_equal(d, np.array([-0.5, 1.5], dtype=real_dtype))


def test_signed_distance_dynamic_mesh():
    mesh = create_unit_cube(np.int32, np.float32)
    dyn = tf.Mesh(tf.as_offset_blocked(mesh.faces), mesh.points)
    assert tf.signed_distance(dyn, [0.0, 0.0, 0.0]) == -0.5
    assert tf.signed_distance(dyn, [2.0, 0.0, 0.0]) == 1.5


def test_signed_distance_transformed_mesh():
    mesh = create_unit_cube(np.int32, np.float32)
    T = np.eye(4, dtype=np.float32)
    T[:3, 3] = [10.0, 0.0, 0.0]
    mesh.transformation = T
    assert tf.signed_distance(mesh, [10.0, 0.0, 0.0]) == -0.5
    assert tf.signed_distance(mesh, [12.0, 0.0, 0.0]) == 1.5


@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_signed_distance_open_mesh(real_dtype):
    """A cube missing a face: the winding number still signs it."""
    mesh = create_open_cube(np.int32, real_dtype)
    assert mesh.number_of_faces == 10

    assert tf.signed_distance(mesh, [0.0, 0.0, 3.0]) == 2.5
    assert tf.signed_distance(mesh, [-3.0, 0.0, 0.0]) == 2.5
    # the boundary takes a sixth of the solid angle, so the inside still wins
    assert tf.signed_distance(mesh, [0.0, 0.0, 0.0]) == -0.5


# ==============================================================================
# Cached structure reuse and invalidation
# ==============================================================================

def test_signed_distance_builds_moments_once_and_agrees_across_calls():
    mesh = create_unit_cube(np.int32, np.float32)
    assert not mesh._wrapper.has_winding_moments()

    first = tf.signed_distance(mesh, [0.0, 0.0, 0.0])
    assert mesh._wrapper.has_winding_moments()
    second = tf.signed_distance(mesh, [0.0, 0.0, 0.0])
    assert first == second == -0.5


def test_signed_distance_build_winding_moments():
    mesh = create_unit_cube(np.int32, np.float32)
    mesh.build_winding_moments()
    assert mesh._wrapper.has_winding_moments()
    assert tf.signed_distance(mesh, [0.0, 0.0, 0.0]) == -0.5
    assert tf.signed_distance(mesh, [2.0, 0.0, 0.0]) == 1.5


def test_signed_distance_shared_view_shares_the_moments():
    """A shared view is one cache: one build, and one mutation stales both."""
    mesh = create_unit_cube(np.int32, np.float32)
    view = mesh.shared_view()

    assert tf.signed_distance(mesh, [0.0, 0.0, 0.0]) == -0.5
    assert view._wrapper.has_winding_moments()
    assert tf.signed_distance(view, [0.0, 0.0, 0.0]) == -0.5

    mesh.points = mesh.points * np.float32(4.0)
    assert not mesh._wrapper.has_winding_moments()
    assert not view._wrapper.has_winding_moments()


def test_signed_distance_after_points_change():
    """Moving the points stales the moments with the tree they mirror."""
    mesh = create_unit_cube(np.int32, np.float32)
    assert tf.signed_distance(mesh, [0.0, 0.0, 0.0]) == -0.5

    mesh.points = mesh.points * np.float32(4.0)
    assert not mesh._wrapper.has_winding_moments()
    assert tf.signed_distance(mesh, [0.0, 0.0, 0.0]) == -2.0
    assert tf.signed_distance(mesh, [4.0, 0.0, 0.0]) == 2.0


# ==============================================================================
# Error Handling Tests
# ==============================================================================

def test_signed_distance_rejects_non_mesh():
    pt = tf.Point([0.0, 0.0, 0.0])
    with pytest.raises(TypeError):
        tf.signed_distance(pt, pt)


def test_signed_distance_rejects_non_point_primitive():
    mesh = create_unit_cube(np.int32, np.float32)
    seg = tf.Segment([[0, 0, 0], [1, 1, 1]])
    with pytest.raises(TypeError):
        tf.signed_distance(mesh, seg)


def test_signed_distance_rejects_bad_shape():
    mesh = create_unit_cube(np.int32, np.float32)
    with pytest.raises(ValueError):
        tf.signed_distance(mesh, [0.0, 0.0])


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_build_winding_moments_2d_mesh_raises(index_dtype, real_dtype):
    mesh = create_triangle_mesh_2d(index_dtype, real_dtype)
    with pytest.raises(
            ValueError, match="Winding moments only supported for 3D meshes"):
        mesh.build_winding_moments()
    # the wrapper refuses it too, so no empty structure is ever stated
    with pytest.raises(ValueError, match="winding moments require a 3D mesh"):
        mesh._wrapper.build_winding_moments()
    assert not mesh._wrapper.has_winding_moments()


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
