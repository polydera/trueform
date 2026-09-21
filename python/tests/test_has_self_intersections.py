"""
Test has_self_intersections

Copyright (c) 2025 Ziga Sajovic, XLAB
"""

import pytest
import numpy as np
import trueform as tf

# Parameter sets
INDEX_DTYPES = [np.int32, np.int64]
REAL_DTYPES = [np.float32, np.float64]


# ==============================================================================
# Test data generators
# ==============================================================================

def create_crossing_triangles(index_dtype, real_dtype):
    """Two triangles whose interiors genuinely cross."""
    faces = np.array([
        [0, 1, 2],
        [3, 4, 5]
    ], dtype=index_dtype)
    points = np.array([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.2, 0.2, -0.5],
        [0.8, 0.2, 0.5],
        [0.2, 0.8, 0.5]
    ], dtype=real_dtype)
    return faces, points


def create_bowtie_mesh(index_dtype, real_dtype):
    """Two triangles meeting at a single vertex: neighbours, not contacts."""
    faces = np.array([
        [0, 1, 2],
        [2, 3, 4]
    ], dtype=index_dtype)
    points = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 1.0, 0.0],
        [1.5, 2.0, 0.0],
        [0.5, 2.0, 0.0]
    ], dtype=real_dtype)
    return faces, points


# ==============================================================================
# Tests
# ==============================================================================

@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_sphere_has_none(index_dtype, real_dtype):
    """A sphere does not meet itself."""
    faces, points = tf.make_sphere_mesh(
        1.0, stacks=12, segments=12, dtype=real_dtype, index_dtype=index_dtype)
    mesh = tf.Mesh(faces, points)

    assert tf.has_self_intersections(mesh) is False


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_crossing_triangles(index_dtype, real_dtype):
    """Two crossing triangles are a contact."""
    faces, points = create_crossing_triangles(index_dtype, real_dtype)
    mesh = tf.Mesh(faces, points)

    assert tf.has_self_intersections(mesh) is True


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
def test_shared_features_are_neighbours(index_dtype):
    """Faces sharing a vertex or an edge are neighbours, not contacts."""
    faces, points = create_bowtie_mesh(index_dtype, np.float32)
    assert tf.has_self_intersections(tf.Mesh(faces, points)) is False

    # Three triangles on one edge: non-manifold, but still only neighbours.
    faces = np.array([
        [0, 1, 2],
        [0, 1, 3],
        [0, 1, 4]
    ], dtype=index_dtype)
    points = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 1.0, 0.0],
        [0.5, -1.0, 0.0],
        [0.5, 0.0, 1.0]
    ], dtype=np.float32)
    assert tf.has_self_intersections(tf.Mesh(faces, points)) is False


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
def test_crossing_triangles_dynamic(index_dtype):
    """The dynamic entry states the same verdict."""
    tri_faces, points = create_crossing_triangles(index_dtype, np.float32)
    offsets = np.array([0, 3, 6], dtype=index_dtype)
    data = tri_faces.reshape(-1).copy()
    faces = tf.OffsetBlockedArray(offsets, data)
    mesh = tf.Mesh(faces, points)

    assert tf.has_self_intersections(mesh) is True


def test_tagged_mesh_reuses_structures():
    """A mesh whose structures are already built answers without rebuilding."""
    faces, points = tf.make_sphere_mesh(1.0, stacks=8, segments=8)
    mesh = tf.Mesh(faces, points)
    mesh.build_tree()
    mesh.build_face_membership()
    mesh.build_manifold_edge_link()
    fm_before = mesh._wrapper.face_membership_array()
    fm_data_before = fm_before.data_array().__array_interface__["data"][0]

    assert tf.has_self_intersections(mesh) is False

    fm_after = mesh._wrapper.face_membership_array()
    assert fm_after.data_array().__array_interface__["data"][0] == fm_data_before
