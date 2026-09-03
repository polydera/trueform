"""
Tests for CsgGraph over dynamic (n-gon) meshes.

Copyright (c) 2026 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/trueform
"""

import numpy as np
import pytest
import trueform as tf


REAL_DTYPES = [np.float32, np.float64]
INDEX_DTYPES = [np.int32, np.int64]


# ==============================================================================
# Helper functions
# ==============================================================================

def _box_points(center, size, real_dtype):
    c = np.asarray(center, dtype=real_dtype)
    h = size / 2
    return np.array(
        [[x, y, z] for x in (-h, h) for y in (-h, h) for z in (-h, h)],
        dtype=real_dtype,
    ) + c


_QUAD_FACES = [
    [0, 1, 3, 2], [4, 6, 7, 5], [0, 4, 5, 1],
    [2, 3, 7, 6], [0, 2, 6, 4], [1, 5, 7, 3],
]


def make_quad_box(center, size=1.0, real_dtype=np.float32,
                  index_dtype=np.int32):
    """A cube of six outward quads as a dynamic mesh."""
    data = np.array([v for quad in _QUAD_FACES for v in quad],
                    dtype=index_dtype)
    offsets = np.arange(0, 4 * len(_QUAD_FACES) + 1, 4, dtype=index_dtype)
    return tf.Mesh(tf.OffsetBlockedArray(offsets, data),
                   _box_points(center, size, real_dtype))


def make_tri_box(center, size=1.0, real_dtype=np.float32,
                 index_dtype=np.int32):
    faces = np.array(
        [[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
         [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]],
        dtype=index_dtype,
    )
    return tf.Mesh(faces, _box_points(center, size, real_dtype))


def signed_volume_blocks(faces, points):
    """Signed volume of a dynamic mesh, each face fanned from its first
    vertex."""
    p = np.asarray(points, dtype=np.float64)
    total = 0.0
    for block in faces:
        for k in range(1, len(block) - 1):
            a, b, c = p[block[0]], p[block[k]], p[block[k + 1]]
            total += float(np.dot(np.cross(a, b), c))
    return total / 6.0


def two_quad_boxes(real_dtype=np.float32, index_dtype=np.int32):
    a = make_quad_box((0.0, 0.0, 0.0), 1.0, real_dtype, index_dtype)
    b = make_quad_box((0.5, 0.0, 0.0), 1.0, real_dtype, index_dtype)
    return a, b


# ==============================================================================
# Boolean volumes through the graph
# ==============================================================================

@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_dyn_boolean_volumes(index_dtype, real_dtype):
    a, b = two_quad_boxes(real_dtype, index_dtype)
    graph = tf.CsgGraph([a, b])

    faces, points = graph.mesh(tf.op(0) | tf.op(1))
    assert isinstance(faces, tf.OffsetBlockedArray)
    assert faces.dtype == np.dtype(index_dtype)
    assert points.dtype == np.dtype(real_dtype)
    assert abs(abs(signed_volume_blocks(faces, points)) - 1.5) < 1e-5

    faces, points = graph.mesh(tf.op(0) & tf.op(1))
    assert abs(abs(signed_volume_blocks(faces, points)) - 0.5) < 1e-5

    faces, points = graph.mesh(tf.op(0) - tf.op(1))
    assert abs(abs(signed_volume_blocks(faces, points)) - 0.5) < 1e-5


def test_dyn_mesh_with_source_ids():
    a, b = two_quad_boxes()
    graph = tf.CsgGraph([a, b])
    (faces, points), tag_labels, face_labels = graph.mesh(
        tf.op(0) | tf.op(1), return_source_ids=True)
    assert isinstance(faces, tf.OffsetBlockedArray)
    assert len(tag_labels) == len(faces)
    assert len(face_labels) == len(faces)
    assert set(np.unique(tag_labels)) <= {0, 1}


def test_dyn_mesh_index_map():
    a, b = two_quad_boxes()
    graph = tf.CsgGraph([a, b])
    (faces, points), imap = graph.mesh(tf.op(0) | tf.op(1),
                                       return_index_map=True)
    assert isinstance(faces, tf.OffsetBlockedArray)
    assert len(imap.point_tag_labels) == len(points)
    assert len(imap.point_labels) == len(points)
    assert len(imap.face_tag_labels) == len(faces)
    assert len(imap.face_labels) == len(faces)
    assert len(imap.point_f) == imap.n_tags == 2
    for t, p in zip(imap.point_tag_labels, imap.point_labels):
        if t < imap.n_tags:
            assert 0 <= p < len(graph.forms[t].points)
        else:
            assert p == imap.n_output_points
    assert abs(abs(signed_volume_blocks(faces, points)) - 1.5) < 1e-5


# ==============================================================================
# Domains
# ==============================================================================

def test_dyn_domains():
    a, b = two_quad_boxes()
    graph = tf.CsgGraph([a, b])
    cells, ids = graph.domains()
    assert len(cells) == 3
    assert len(ids) == 3
    total = 0.0
    for faces, points in cells:
        assert isinstance(faces, tf.OffsetBlockedArray)
        total += abs(signed_volume_blocks(faces, points))
    assert abs(total - 1.5) < 1e-5


def test_dyn_domains_index_map():
    a, b = two_quad_boxes()
    graph = tf.CsgGraph([a, b])
    cells, ids, imap = graph.domains(return_index_map=True)
    assert len(cells) == 3
    for (f, p), ftb, ptb in zip(cells, imap.face_tag_blocks,
                                imap.point_tag_blocks):
        assert isinstance(f, tf.OffsetBlockedArray)
        assert len(ftb) == len(f)
        assert len(ptb) == len(p)
    assert imap.inclusion.shape == (len(cells), 2)
    only_a = imap.inclusion[:, 0] & ~imap.inclusion[:, 1]
    _, d_ids = graph.domains(tf.op(0) - tf.op(1))
    assert (set(np.asarray(ids)[only_a].tolist())
            == set(np.asarray(d_ids).tolist()))


# ==============================================================================
# Sheets
# ==============================================================================

def test_dyn_sheet_cuts_box():
    """A dynamic sheet operand cuts a dynamic volume into capped halves."""
    box = make_quad_box((0.0, 0.0, 0.0))
    sheet_points = np.array(
        [[-2, -2, 0], [2, -2, 0], [2, 2, 0], [-2, 2, 0]], dtype=np.float32)
    sheet = tf.Mesh(
        tf.OffsetBlockedArray(np.array([0, 4], dtype=np.int32),
                              np.array([0, 1, 2, 3], dtype=np.int32)),
        sheet_points)

    graph = tf.CsgGraph([sheet, box], sheets=[0])
    assert graph.sheets == (0,)

    lower_faces, lower_points = graph.mesh(tf.op(1) & tf.op(0))
    upper_faces, upper_points = graph.mesh(tf.op(1) - tf.op(0))
    assert isinstance(lower_faces, tf.OffsetBlockedArray)
    lower = abs(signed_volume_blocks(lower_faces, lower_points))
    upper = abs(signed_volume_blocks(upper_faces, upper_points))
    assert abs(lower - 0.5) < 1e-5
    assert abs(upper - 0.5) < 1e-5


# ==============================================================================
# Outer shell and curves
# ==============================================================================

def test_dyn_outer_shell():
    a, b = two_quad_boxes()
    graph = tf.CsgGraph([a, b])
    faces, points = graph.outer_shell()
    assert isinstance(faces, tf.OffsetBlockedArray)
    assert abs(abs(signed_volume_blocks(faces, points)) - 1.5) < 1e-5

    paths, curve_points = graph.intersection_curves()
    assert len(paths) > 0
    assert curve_points.shape[1] == 3


# ==============================================================================
# Facade acceptance matrix
# ==============================================================================

def test_facade_all_triangle_accepted():
    a = make_tri_box((0.0, 0.0, 0.0))
    b = make_tri_box((0.5, 0.0, 0.0))
    graph = tf.CsgGraph([a, b])
    faces, points = graph.mesh(tf.op(0) | tf.op(1))
    assert isinstance(faces, np.ndarray)
    assert faces.shape[1] == 3


def test_facade_all_dynamic_accepted():
    a, b = two_quad_boxes()
    graph = tf.CsgGraph([a, b])
    faces, _ = graph.mesh()
    assert isinstance(faces, tf.OffsetBlockedArray)


def test_facade_mixed_refused():
    tri = make_tri_box((0.0, 0.0, 0.0))
    dyn = make_quad_box((0.5, 0.0, 0.0))
    with pytest.raises(ValueError, match="all triangle or all dynamic"):
        tf.CsgGraph([tri, dyn])
    with pytest.raises(ValueError, match="all triangle or all dynamic"):
        tf.CsgGraph([dyn, tri])
