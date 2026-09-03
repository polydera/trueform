"""
Tests for signed_distance (pseudonormal: negative inside, positive outside)

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


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
