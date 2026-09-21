"""
Tests for winding_number (the generalized winding number: ~1 inside, ~0 outside)

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

def create_winding_sphere(index_dtype, real_dtype, radius=1.0):
    """A closed UV sphere, outward wound, centered at the origin."""
    faces, points = tf.make_sphere_mesh(
        radius, 24, 24, dtype=real_dtype, index_dtype=index_dtype)
    return tf.Mesh(faces, points)


def create_winding_sheet(index_dtype, real_dtype):
    """An open square sheet in z = 0: a mesh no volume encloses."""
    faces, points = tf.make_plane_mesh(
        2.0, 2.0, 8, 8, dtype=real_dtype, index_dtype=index_dtype)
    return tf.Mesh(faces, points)


def create_winding_triangle_2d(index_dtype, real_dtype):
    """A 2D mesh, which has no winding moments to build."""
    faces = np.array([[0, 1, 2]], dtype=index_dtype)
    points = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]], dtype=real_dtype)
    return tf.Mesh(faces, points)


# ==============================================================================
# Inside / outside oracles
# ==============================================================================

@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_winding_number_single(index_dtype, real_dtype):
    mesh = create_winding_sphere(index_dtype, real_dtype)

    inside = tf.winding_number(mesh, [0.0, 0.0, 0.0])
    outside = tf.winding_number(mesh, [3.0, 0.0, 0.0])

    assert isinstance(inside, float)
    assert inside == pytest.approx(1.0, abs=0.1)
    assert outside == pytest.approx(0.0, abs=0.1)


@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_winding_number_point_query(real_dtype):
    mesh = create_winding_sphere(np.int32, real_dtype)
    pt = tf.Point(np.array([0.0, 0.0, 0.0], dtype=real_dtype))
    assert tf.winding_number(mesh, pt) == pytest.approx(1.0, abs=0.1)


@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_winding_number_batch_matches_single(real_dtype):
    """The batch overload is the single query at every position."""
    mesh = create_winding_sphere(np.int32, real_dtype)
    coords = np.array(
        [[0.0, 0.0, 0.0], [0.2, -0.3, 0.1], [3.0, 0.0, 0.0],
         [0.0, -4.0, 2.0]],
        dtype=real_dtype,
    )
    queries = tf.Point(coords)

    w = tf.winding_number(mesh, queries)
    assert w.shape == (4,)
    # the number is dimensionless: float64 for either mesh dtype
    assert w.dtype == np.float64

    singles = np.array([tf.winding_number(mesh, c) for c in coords])
    assert np.array_equal(w, singles)

    assert w[0] == pytest.approx(1.0, abs=0.1)
    assert w[1] == pytest.approx(1.0, abs=0.1)
    assert w[2] == pytest.approx(0.0, abs=0.1)
    assert w[3] == pytest.approx(0.0, abs=0.1)


@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_winding_number_batch_ndarray(real_dtype):
    mesh = create_winding_sphere(np.int32, real_dtype)
    queries = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])

    w = tf.winding_number(mesh, queries)
    assert w.dtype == np.float64
    assert w[0] == pytest.approx(1.0, abs=0.1)
    assert w[1] == pytest.approx(0.0, abs=0.1)


def test_winding_number_empty_batch():
    mesh = create_winding_sphere(np.int32, np.float32)
    w = tf.winding_number(mesh, np.zeros((0, 3), dtype=np.float32))
    assert w.shape == (0,)
    assert w.dtype == np.float64


def test_winding_number_dynamic_mesh():
    mesh = create_winding_sphere(np.int32, np.float32)
    dyn = tf.Mesh(tf.as_offset_blocked(mesh.faces), mesh.points)
    assert tf.winding_number(dyn, [0.0, 0.0, 0.0]) == pytest.approx(1.0, abs=0.1)
    assert tf.winding_number(dyn, [3.0, 0.0, 0.0]) == pytest.approx(0.0, abs=0.1)


def test_winding_number_transformed_mesh():
    """The query is moved into the mesh's own frame."""
    mesh = create_winding_sphere(np.int32, np.float32)
    T = np.eye(4, dtype=np.float32)
    T[:3, 3] = [10.0, 0.0, 0.0]
    mesh.transformation = T

    assert tf.winding_number(mesh, [10.0, 0.0, 0.0]) == pytest.approx(1.0, abs=0.1)
    assert tf.winding_number(mesh, [0.0, 0.0, 0.0]) == pytest.approx(0.0, abs=0.1)


@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_winding_number_open_sheet_is_fractional(real_dtype):
    """An open sheet encloses nothing, so it reads neither 1 nor 0."""
    mesh = create_winding_sheet(np.int32, real_dtype)

    above = tf.winding_number(mesh, [0.0, 0.0, 0.01])
    below = tf.winding_number(mesh, [0.0, 0.0, -0.01])
    far = tf.winding_number(mesh, [0.0, 0.0, 10.0])

    assert np.isfinite(above) and np.isfinite(below) and np.isfinite(far)
    assert 0.3 < abs(above) < 0.7
    assert 0.3 < abs(below) < 0.7
    assert abs(far) < 0.1


def test_winding_number_beta_sharpens_toward_the_exact_sum():
    """A large beta descends to the exact solid-angle sum."""
    mesh = create_winding_sphere(np.int32, np.float64)

    exact_inside = tf.winding_number(mesh, [0.0, 0.0, 0.0], beta=1e9)
    exact_outside = tf.winding_number(mesh, [3.0, 0.0, 0.0], beta=1e9)

    assert exact_inside == pytest.approx(1.0, abs=1e-6)
    assert exact_outside == pytest.approx(0.0, abs=1e-6)


# ==============================================================================
# Cached structure reuse and invalidation
# ==============================================================================

def test_winding_number_builds_moments_once_and_agrees_across_calls():
    mesh = create_winding_sphere(np.int32, np.float32)
    assert not mesh._wrapper.has_winding_moments()

    first = tf.winding_number(mesh, [0.0, 0.0, 0.0])
    assert mesh._wrapper.has_winding_moments()
    second = tf.winding_number(mesh, [0.0, 0.0, 0.0])
    assert first == second


def test_winding_number_reuses_the_moments_signed_distance_built():
    """One cache: the second query pays for nothing."""
    mesh = create_winding_sphere(np.int32, np.float32)
    tf.signed_distance(mesh, [0.0, 0.0, 0.0])
    assert mesh._wrapper.has_winding_moments()
    assert tf.winding_number(mesh, [0.0, 0.0, 0.0]) == pytest.approx(1.0, abs=0.1)


def test_winding_number_shared_view_shares_the_moments():
    mesh = create_winding_sphere(np.int32, np.float32)
    view = mesh.shared_view()

    tf.winding_number(mesh, [0.0, 0.0, 0.0])
    assert view._wrapper.has_winding_moments()
    assert tf.winding_number(view, [0.0, 0.0, 0.0]) == pytest.approx(1.0, abs=0.1)

    mesh.points = mesh.points * np.float32(4.0)
    assert not mesh._wrapper.has_winding_moments()
    assert not view._wrapper.has_winding_moments()


def test_winding_number_after_points_change():
    """Moving the points stales the moments with the tree they mirror."""
    mesh = create_winding_sphere(np.int32, np.float32)
    assert tf.winding_number(mesh, [2.0, 0.0, 0.0]) == pytest.approx(0.0, abs=0.1)

    mesh.points = mesh.points * np.float32(4.0)
    assert not mesh._wrapper.has_winding_moments()
    assert tf.winding_number(mesh, [2.0, 0.0, 0.0]) == pytest.approx(1.0, abs=0.1)


# ==============================================================================
# Error Handling Tests
# ==============================================================================

def test_winding_number_rejects_non_mesh():
    pt = tf.Point([0.0, 0.0, 0.0])
    with pytest.raises(TypeError):
        tf.winding_number(pt, pt)


def test_winding_number_rejects_non_point_primitive():
    mesh = create_winding_sphere(np.int32, np.float32)
    seg = tf.Segment([[0, 0, 0], [1, 1, 1]])
    with pytest.raises(TypeError):
        tf.winding_number(mesh, seg)


def test_winding_number_rejects_bad_shape():
    mesh = create_winding_sphere(np.int32, np.float32)
    with pytest.raises(ValueError):
        tf.winding_number(mesh, [0.0, 0.0])


def test_winding_number_rejects_bad_beta():
    mesh = create_winding_sphere(np.int32, np.float32)
    with pytest.raises(ValueError):
        tf.winding_number(mesh, [0.0, 0.0, 0.0], beta=0.0)


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_winding_number_rejects_2d_mesh(index_dtype, real_dtype):
    mesh = create_winding_triangle_2d(index_dtype, real_dtype)
    with pytest.raises(ValueError, match="only supports 3D meshes"):
        tf.winding_number(mesh, [0.0, 0.0])


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
