"""
Test face_quality and dihedral_angles

Copyright (c) 2025 Ziga Sajovic, XLAB
"""

import pytest
import numpy as np
import trueform as tf

# Parameter sets
INDEX_DTYPES = [np.int32, np.int64]
REAL_DTYPES = [np.float32, np.float64]

TOLERANCES = {np.float32: 1e-5, np.float64: 1e-12}


# ==============================================================================
# face_quality
# ==============================================================================

@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_face_quality_equilateral(index_dtype, real_dtype):
    """An equilateral triangle measures 1 everywhere."""
    faces = np.array([[0, 1, 2]], dtype=index_dtype)
    points = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, np.sqrt(3.0) / 2.0, 0.0]
    ], dtype=real_dtype)
    mesh = tf.Mesh(faces, points)

    quality, min_angle, max_angle, aspect_ratio = tf.face_quality(mesh)

    tol = TOLERANCES[real_dtype]
    assert quality.dtype == real_dtype
    assert quality.shape == (1,)
    assert quality[0] == pytest.approx(1.0, abs=tol)
    assert min_angle[0] == pytest.approx(np.pi / 3.0, abs=tol)
    assert max_angle[0] == pytest.approx(np.pi / 3.0, abs=tol)
    assert aspect_ratio[0] == pytest.approx(1.0, abs=tol)


@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_face_quality_right_isosceles(real_dtype):
    """A right isosceles triangle: 45/45/90 corners, sqrt(2) aspect."""
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    points = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0]
    ], dtype=real_dtype)
    mesh = tf.Mesh(faces, points)

    quality, min_angle, max_angle, aspect_ratio = tf.face_quality(mesh)

    tol = TOLERANCES[real_dtype]
    # quality = (2 / sqrt(3)) * 2A / L^2 with 2A = 1 and L^2 = 2.
    assert quality[0] == pytest.approx(1.0 / np.sqrt(3.0), abs=tol)
    assert min_angle[0] == pytest.approx(np.pi / 4.0, abs=tol)
    assert max_angle[0] == pytest.approx(np.pi / 2.0, abs=tol)
    assert aspect_ratio[0] == pytest.approx(np.sqrt(2.0), abs=tol)


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
def test_face_quality_dynamic_quad(index_dtype):
    """A quad has corner angles and an aspect ratio, but no quality."""
    offsets = np.array([0, 4], dtype=index_dtype)
    data = np.array([0, 1, 2, 3], dtype=index_dtype)
    faces = tf.OffsetBlockedArray(offsets, data)
    points = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0]
    ], dtype=np.float32)
    mesh = tf.Mesh(faces, points)

    quality, min_angle, max_angle, aspect_ratio = tf.face_quality(mesh)

    tol = TOLERANCES[np.float32]
    assert quality[0] == pytest.approx(-1.0, abs=tol)
    assert min_angle[0] == pytest.approx(np.pi / 2.0, abs=tol)
    assert max_angle[0] == pytest.approx(np.pi / 2.0, abs=tol)
    assert aspect_ratio[0] == pytest.approx(1.0, abs=tol)


def test_face_quality_tuple_input():
    """The (faces, points) entry shape reaches the same measures."""
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    points = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, np.sqrt(3.0) / 2.0, 0.0]
    ], dtype=np.float32)

    quality, _, _, _ = tf.face_quality((faces, points))

    assert quality[0] == pytest.approx(1.0, abs=TOLERANCES[np.float32])


# ==============================================================================
# dihedral_angles
# ==============================================================================

@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_dihedral_angles_flat_pair(index_dtype, real_dtype):
    """Two coplanar triangles share one edge, and it is flat."""
    faces = np.array([
        [0, 1, 2],
        [1, 3, 2]
    ], dtype=index_dtype)
    points = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0]
    ], dtype=real_dtype)
    mesh = tf.Mesh(faces, points)

    edges, angles = tf.dihedral_angles(mesh)

    assert edges.dtype == index_dtype
    assert angles.dtype == real_dtype
    np.testing.assert_array_equal(
        edges, np.array([[1, 2]], dtype=index_dtype))
    assert angles[0] == pytest.approx(0.0, abs=TOLERANCES[real_dtype])


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_dihedral_angles_box(index_dtype, real_dtype):
    """A box turns through pi/2 on its edges and 0 on its face diagonals."""
    faces, points = tf.make_box_mesh(
        1.0, 1.0, 1.0, dtype=real_dtype, index_dtype=index_dtype)
    mesh = tf.Mesh(faces, points)

    edges, angles = tf.dihedral_angles(mesh)

    # 12 triangles, closed: 18 edges, each stated once, pairs ascending.
    assert edges.shape == (18, 2)
    assert angles.shape == (18,)
    assert np.all(edges[:, 0] < edges[:, 1])
    assert len(set(map(tuple, edges))) == 18

    tol = 1e-4 if real_dtype == np.float32 else 1e-10
    n_right = int(np.sum(np.isclose(angles, np.pi / 2.0, atol=tol)))
    n_flat = int(np.sum(np.isclose(angles, 0.0, atol=tol)))
    assert n_right == 12
    assert n_flat == 6


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
def test_dihedral_angles_boundary_edge_not_stated(index_dtype):
    """A boundary edge joins no pair of faces and is not stated."""
    faces = np.array([[0, 1, 2]], dtype=index_dtype)
    points = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0]
    ], dtype=np.float32)
    mesh = tf.Mesh(faces, points)

    edges, angles = tf.dihedral_angles(mesh)

    assert edges.shape == (0, 2)
    assert angles.shape == (0,)
