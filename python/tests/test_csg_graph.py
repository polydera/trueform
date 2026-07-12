"""
Tests for CsgGraph: build once, query many.

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


# ==============================================================================
# Helper functions
# ==============================================================================

def make_box(center, size=1.0, real_dtype=np.float32, index_dtype=np.int32):
    c = np.asarray(center, dtype=real_dtype)
    h = size / 2
    points = np.array(
        [[x, y, z] for x in (-h, h) for y in (-h, h) for z in (-h, h)],
        dtype=real_dtype,
    ) + c
    faces = np.array(
        [[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
         [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]],
        dtype=index_dtype,
    )
    return tf.Mesh(faces, points)


def signed_volume(faces, points):
    p = np.asarray(points, dtype=np.float64)
    f = np.asarray(faces)
    return float(
        np.einsum("ij,ij->", np.cross(p[f[:, 0]], p[f[:, 1]]), p[f[:, 2]])
    ) / 6.0


def is_closed(faces):
    edges = {}
    for tri in np.asarray(faces):
        for e in range(3):
            u, v = int(tri[e]), int(tri[(e + 1) % 3])
            edges[(min(u, v), max(u, v))] = edges.get((min(u, v), max(u, v)), 0) + 1
    return all(c == 2 for c in edges.values())


# ==============================================================================
# Expression building
# ==============================================================================

def test_expr_operators():
    e = tf.op(0) - tf.op(1)
    assert e.program() == [0, 1, -3]
    e = (tf.op(0) | tf.op(1)) & tf.op(2)
    assert e.program() == [0, 1, -1, 2, -2]
    e = ~tf.op(0)
    assert e.program() == [0, -4]
    # ints promote to operands
    assert (tf.op(0) - 1).program() == [0, 1, -3]
    assert (0 | tf.op(1)).program() == [0, 1, -1]


def test_expr_rejects_bad_operand():
    with pytest.raises(ValueError):
        tf.op(-1)
    with pytest.raises(TypeError):
        tf.op(0) - "nope"


# ==============================================================================
# Build + query across the dtype matrix
# ==============================================================================

@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
def test_boolean_matches_pairwise(real_dtype, index_dtype):
    a = make_box((0, 0, 0), real_dtype=real_dtype, index_dtype=index_dtype)
    b = make_box((0.4, 0.3, 0.2), real_dtype=real_dtype,
                 index_dtype=index_dtype)
    graph = tf.CsgGraph([a, b])

    for expr, op_fn in [
        (tf.op(0) - tf.op(1), tf.boolean_difference),
        (tf.op(0) | tf.op(1), tf.boolean_union),
        (tf.op(0) & tf.op(1), tf.boolean_intersection),
    ]:
        faces, points = graph.mesh(expr)
        assert faces.dtype == index_dtype
        assert points.dtype == real_dtype
        assert is_closed(faces)
        (bf, bp), _, _ = op_fn(a, b)
        assert abs(abs(signed_volume(faces, points)) -
                   abs(signed_volume(bf, bp))) < 1e-5


@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_build_once_query_many(real_dtype):
    a = make_box((0, 0, 0), real_dtype=real_dtype)
    b = make_box((0.4, 0.3, 0.2), real_dtype=real_dtype)
    graph = tf.CsgGraph([a, b])
    v_union = abs(signed_volume(*graph.mesh(tf.op(0) | tf.op(1))))
    v_inter = abs(signed_volume(*graph.mesh(tf.op(0) & tf.op(1))))
    v_a_only = abs(signed_volume(*graph.mesh(tf.op(0) - tf.op(1))))
    v_b_only = abs(signed_volume(*graph.mesh(tf.op(1) - tf.op(0))))
    # inclusion-exclusion over the same build
    assert abs(v_union - (v_a_only + v_b_only + v_inter)) < 1e-5
    assert abs((v_a_only + v_inter) - 1.0) < 1e-5  # |A| = unit box


def test_mesh_labels_resolve():
    a = make_box((0, 0, 0))
    b = make_box((0.4, 0.3, 0.2))
    graph = tf.CsgGraph([a, b])
    (faces, points), tags, face_labels = graph.mesh(
        tf.op(0) | tf.op(1), return_source_ids=True)
    assert len(tags) == len(faces)
    assert len(face_labels) == len(faces)
    assert set(tags.tolist()) == {0, 1}
    for t, f in zip(tags, face_labels):
        assert 0 <= f < len(graph.forms[t].faces)


def test_domains_partition():
    a = make_box((0, 0, 0))
    b = make_box((0.4, 0.3, 0.2))
    graph = tf.CsgGraph([a, b])
    cells, ids = graph.domains()
    # two overlapping boxes: A-only, B-only, A-and-B
    assert len(cells) == 3
    assert len(ids) == 3
    total = sum(abs(signed_volume(f, p)) for f, p in cells)
    v_union = abs(signed_volume(*graph.mesh(tf.op(0) | tf.op(1))))
    assert abs(total - v_union) < 1e-5
    for f, p in cells:
        assert is_closed(f)

    # expression-restricted domains
    inter_cells, _ = graph.domains(tf.op(0) & tf.op(1))
    assert len(inter_cells) == 1

    # provenance blocks parallel to cells
    cells_l, ids_l, tag_blocks, face_blocks = graph.domains(
        return_source_ids=True)
    assert len(cells_l) == len(cells)
    assert len(tag_blocks) == len(cells)
    assert len(face_blocks) == len(cells)
    for (f, p), tb in zip(cells_l, tag_blocks):
        assert len(tb) == len(f)


@pytest.mark.parametrize("triangulation", ["cdt", "refined_cdt"])
def test_triangulation_types(triangulation):
    a = make_box((0, 0, 0))
    b = make_box((0.4, 0.3, 0.2))
    graph = tf.CsgGraph([a, b], triangulation=triangulation)
    faces, points = graph.mesh(tf.op(0) - tf.op(1))
    assert is_closed(faces)
    assert abs(abs(signed_volume(faces, points)) - (1.0 - 0.6 * 0.7 * 0.8)) \
        < 1e-4
    cells, ids = graph.domains()
    assert len(cells) == 3


def test_refined_creates_more_points():
    a = make_box((0, 0, 0))
    b = make_box((0.4, 0.3, 0.2))
    stock = tf.CsgGraph([a, b])
    refined = tf.CsgGraph([a, b], triangulation="refined_cdt")
    assert stock.created_points.shape[1] == 3
    assert refined.created_points.shape[0] > stock.created_points.shape[0]
    assert stock.created_points.dtype == np.float32


def test_mesh_index_map():
    a = make_box((0, 0, 0))
    b = make_box((0.4, 0.3, 0.2))
    graph = tf.CsgGraph([a, b])
    (faces, points), imap = graph.mesh(tf.op(0) | tf.op(1),
                                       return_index_map=True)
    assert len(imap.point_tag_labels) == len(points)
    assert len(imap.point_labels) == len(points)
    assert len(imap.face_tag_labels) == len(faces)
    assert len(imap.face_labels) == len(faces)
    assert len(imap.point_f) == imap.n_tags == 2
    # every original entry resolves into its input mesh
    for t, p in zip(imap.point_tag_labels, imap.point_labels):
        if t < imap.n_tags:
            assert 0 <= p < len(graph.forms[t].points)
        else:
            assert p == imap.n_output_points
    with pytest.raises(ValueError):
        graph.mesh(tf.op(0), return_source_ids=True, return_index_map=True)


def test_domains_index_map():
    a = make_box((0, 0, 0))
    b = make_box((0.4, 0.3, 0.2))
    graph = tf.CsgGraph([a, b])
    cells, ids, imap = graph.domains(return_index_map=True)
    assert len(imap.face_tag_blocks) == len(cells)
    assert len(imap.point_tag_blocks) == len(cells)
    for (f, p), ftb, ptb in zip(cells, imap.face_tag_blocks,
                                imap.point_tag_blocks):
        assert len(ftb) == len(f)
        assert len(ptb) == len(p)
    assert imap.n_tags == 2

    # inclusion matrix: rows parallel to cells, columns per form; masks
    # must agree with the expression-filtered queries (ids are stable)
    assert imap.inclusion.shape == (len(cells), 2)
    assert imap.inclusion.dtype == bool
    ids = np.asarray(ids)
    for col, expr in [(0, tf.op(0)), (1, tf.op(1))]:
        _, q_ids = graph.domains(expr)
        assert (set(ids[imap.inclusion[:, col]].tolist())
                == set(np.asarray(q_ids).tolist()))
    only_a = imap.inclusion[:, 0] & ~imap.inclusion[:, 1]
    _, d_ids = graph.domains(tf.op(0) - tf.op(1))
    assert set(ids[only_a].tolist()) == set(np.asarray(d_ids).tolist())


def test_domains_inclusion_matches_query():
    # three mutually overlapping spheres with a common core: selecting
    # "inside all three" by hand from the inclusion matrix must return
    # the same cells as the expression query
    def sphere(center):
        fp = tf.make_sphere_mesh(1.0, 24, 24, dtype=np.float32,
                                 index_dtype=np.int32)
        return tf.Mesh(fp[0], np.asarray(fp[1]) + np.asarray(center,
                                                             np.float32))

    graph = tf.CsgGraph([sphere((0, 0, 0)), sphere((0.7, 0, 0)),
                         sphere((0.35, 0.6, 0))])
    cells, ids, imap = graph.domains(return_index_map=True)
    ids = np.asarray(ids)

    mask = imap.inclusion.all(axis=1)
    assert mask.any()  # the triple-overlap core exists

    q_cells, q_ids = graph.domains(tf.op(0) & tf.op(1) & tf.op(2))
    q_ids = np.asarray(q_ids)
    assert sorted(ids[mask].tolist()) == sorted(q_ids.tolist())

    by_id = {int(i): signed_volume(f, p)
             for (f, p), i, m in zip(cells, ids, mask) if m}
    for (f, p), i in zip(q_cells, q_ids):
        assert abs(signed_volume(f, p) - by_id[int(i)]) < 1e-12


def test_full_arrangement_mesh():
    a = make_box((0, 0, 0))
    b = make_box((0.4, 0.3, 0.2))
    graph = tf.CsgGraph([a, b])
    # no expression: the full arrangement mesh (both surfaces, cut)
    faces, points = graph.mesh()
    v_union = abs(signed_volume(*graph.mesh(tf.op(0) | tf.op(1))))
    assert len(faces) > 0
    # full arrangement carries every surface, so more faces than the union
    assert len(faces) > len(graph.mesh(tf.op(0) | tf.op(1))[0])

    # no-expression labels path
    (ff, fp), tags, flabels = graph.mesh(return_source_ids=True)
    assert len(tags) == len(ff)
    assert len(flabels) == len(ff)
    assert set(tags.tolist()) == {0, 1}

    # index map requires an expression
    with pytest.raises(RuntimeError):
        graph.mesh(return_index_map=True)
    del v_union


def test_every_return_path():
    a = make_box((0, 0, 0))
    b = make_box((0.4, 0.3, 0.2))
    graph = tf.CsgGraph([a, b])
    e = tf.op(0) - tf.op(1)

    # mesh: plain / labels / index map
    fp = graph.mesh(e)
    (f1, p1), t1, l1 = graph.mesh(e, return_source_ids=True)
    (f2, p2), im = graph.mesh(e, return_index_map=True)
    assert len(f1) == len(fp[0]) == len(f2)
    assert len(t1) == len(f1) and len(l1) == len(f1)
    assert len(im.face_tag_labels) == len(f2)
    assert np.array_equal(im.face_tag_labels, t1)
    assert np.array_equal(im.face_labels, l1)

    # domains: plain / labels / index map, with and without expression
    for expr in (None, e):
        cells, ids = graph.domains(expr)
        cells_l, ids_l, tb, fb = graph.domains(expr, return_source_ids=True)
        cells_m, ids_m, dmap = graph.domains(expr, return_index_map=True)
        assert len(cells) == len(cells_l) == len(cells_m)
        assert np.array_equal(ids, ids_l) and np.array_equal(ids, ids_m)
        assert len(tb) == len(cells) and len(fb) == len(cells)
        assert len(dmap.face_tag_blocks) == len(cells)
        assert len(dmap.point_blocks) == len(cells)
        for (fc, pc), ftb, ptb in zip(cells_m, dmap.face_tag_blocks,
                                      dmap.point_tag_blocks):
            assert len(ftb) == len(fc)
            assert len(ptb) == len(pc)

    # exclusivity on both methods
    with pytest.raises(ValueError):
        graph.mesh(e, return_source_ids=True, return_index_map=True)
    with pytest.raises(ValueError):
        graph.domains(e, return_source_ids=True, return_index_map=True)


def test_intersection_curves():
    a = make_box((0, 0, 0))
    b = make_box((0.4, 0.3, 0.2))
    graph = tf.CsgGraph([a, b])
    paths, curve_points = graph.intersection_curves()
    assert curve_points.shape[1] == 3
    assert curve_points.shape[0] > 0
    assert len(paths) > 0
    # every path indexes into curve_points
    for path in paths:
        for idx in path:
            assert 0 <= idx < len(curve_points)


def test_deterministic():
    a = make_box((0, 0, 0))
    b = make_box((0.4, 0.3, 0.2))
    f0, p0 = tf.CsgGraph([a, b]).mesh(tf.op(0) - tf.op(1))
    f1, p1 = tf.CsgGraph([a, b]).mesh(tf.op(0) - tf.op(1))
    assert np.array_equal(f0, f1)
    assert np.array_equal(p0, p1)


# ==============================================================================
# Remembered construction state
# ==============================================================================

def test_language_side_state():
    a = make_box((0, 0, 0))
    b = make_box((0.4, 0.3, 0.2))
    c = make_box((1.2, 0, 0))
    graph = tf.CsgGraph([a, b, c], sheets=[2], tolerance=0.0,
                        triangulation="cdt")
    assert graph.forms[0] is a
    assert graph.forms[1] is b
    assert len(graph.forms) == 3
    assert graph.sheets == (2,)
    assert graph.mode == "primitives"
    assert graph.tolerance == 0.0
    assert graph.triangulation == "cdt"


def test_input_validation():
    a = make_box((0, 0, 0))
    b64 = make_box((0.4, 0.3, 0.2), real_dtype=np.float64)
    with pytest.raises(ValueError):
        tf.CsgGraph([a])
    with pytest.raises(ValueError):
        tf.CsgGraph([a, b64])
    with pytest.raises(ValueError):
        tf.CsgGraph([a, make_box((0.4, 0, 0))], sheets=[5])


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
