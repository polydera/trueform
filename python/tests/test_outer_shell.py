"""Tests for tf.outer_shell.

Copyright (c) 2025 Žiga Sajovic, XLAB
"""
import sys

import numpy as np
import pytest

import trueform as tf

INDEX_DTYPES = [np.int32, np.int64]
REAL_DTYPES = [np.float32, np.float64]

SPHERE_VOLUME = 4.0 / 3.0 * np.pi


def _two_spheres_soup(index_dtype, real_dtype, mesh_type="triangle"):
    """Two overlapping unit spheres concatenated into one soup."""
    f0, p0 = tf.make_sphere_mesh(
        1.0, stacks=24, segments=24, dtype=real_dtype, index_dtype=index_dtype
    )
    p1 = p0 + np.array([1.0, 0.0, 0.0], dtype=real_dtype)
    faces, points = tf.concatenated([(f0, p0), (f0, p1)])
    faces = faces.astype(index_dtype, copy=False)
    points = points.astype(real_dtype, copy=False)
    if mesh_type == "dynamic":
        faces = tf.as_offset_blocked(faces)
    return tf.Mesh(faces, points)


def _assert_closed(faces):
    faces = np.asarray(faces)
    edges = np.concatenate(
        [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]
    )
    edges = np.sort(edges, axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    assert (counts == 2).all()


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_outer_shell_two_spheres(index_dtype, real_dtype):
    soup = _two_spheres_soup(index_dtype, real_dtype)
    shell = tf.outer_shell(soup)

    assert isinstance(shell, tf.Mesh)
    assert shell.faces.dtype == index_dtype
    assert shell.points.dtype == real_dtype

    # The shell bounds the union: closed, outward-oriented, and the
    # overlap membrane is gone.
    _assert_closed(shell.faces)
    volume = tf.signed_volume(shell)
    assert volume > SPHERE_VOLUME
    assert volume < 2.0 * SPHERE_VOLUME
    assert len(shell.faces) < len(soup.faces)


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_outer_shell_dynamic(index_dtype, real_dtype):
    soup = _two_spheres_soup(index_dtype, real_dtype, mesh_type="dynamic")
    shell = tf.outer_shell(soup)

    assert isinstance(shell, tf.Mesh)
    assert shell.is_dynamic
    assert shell.points.dtype == real_dtype
    volume = tf.signed_volume(shell)
    assert volume > SPHERE_VOLUME
    assert volume < 2.0 * SPHERE_VOLUME


def test_outer_shell_removes_cavity():
    # A small sphere buried inside a big one: the cavity is internal
    # structure and must be removed.
    f0, p0 = tf.make_sphere_mesh(1.0, stacks=24, segments=24)
    f1, p1 = tf.make_sphere_mesh(0.4, stacks=24, segments=24)
    soup = tf.Mesh(*tf.concatenated([(f0, p0), (f1, p1)]))
    shell = tf.outer_shell(soup)

    _assert_closed(shell.faces)
    outer = tf.signed_volume(tf.Mesh(f0, p0))
    # rel=1e-4: float32 points re-quantize through the arrangement; a
    # retained cavity would subtract ~0.27 (6% of the volume).
    assert tf.signed_volume(shell) == pytest.approx(outer, rel=1e-4)


def test_outer_shell_rejects_non_mesh():
    faces, points = tf.make_sphere_mesh(1.0)
    with pytest.raises(TypeError):
        tf.outer_shell((faces, points))


def _cube_mesh(center):
    c = np.asarray(center, dtype=np.float32)
    points = np.array(
        [[x, y, z] for x in (-0.5, 0.5) for y in (-0.5, 0.5)
         for z in (-0.5, 0.5)],
        dtype=np.float32,
    ) + c
    faces = np.array(
        [[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
         [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]],
        dtype=np.int32,
    )
    return tf.Mesh(faces, points)


def test_graph_outer_shell_matches_union():
    # The graph's own shell read: on two overlapping cubes it is the
    # boundary of their union.
    a = _cube_mesh((0.0, 0.0, 0.0))
    b = _cube_mesh((0.5, 0.5, 0.5))
    graph = tf.CsgGraph([a, b])

    faces, points = graph.outer_shell()
    assert faces.dtype == np.int32
    assert points.dtype == np.float32
    _assert_closed(faces)

    union = tf.Mesh(*graph.mesh(tf.op(0) | tf.op(1)))
    assert tf.signed_volume(tf.Mesh(faces, points)) == pytest.approx(
        tf.signed_volume(union), rel=1e-6)
    assert tf.signed_volume(tf.Mesh(faces, points)) == pytest.approx(
        1.875, rel=1e-5)


# ==============================================================================
# Main runner
# ==============================================================================

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
