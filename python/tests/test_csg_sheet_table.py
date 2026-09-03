"""
The sheets table of docs/content/py/2.modules/09.csg.md, executed row by row.

Scene as the docs state it: A = plane at z = 0 (normal +z, operand 0,
declared a sheet), B = unit box (operand 1). Every row's claim is pinned:
face counts, tags, windings, signed volumes, and openness — a row saying
"open along the sheet's rim" must measure exactly the rim's boundary
edges, a row saying "closed and capped" must measure zero.

Copyright (c) 2026 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/trueform
"""

import numpy as np
import pytest
import trueform as tf


BOX_CENTER = np.zeros(3)


def sheet_table_box():
    h = 0.5
    points = np.array(
        [[x, y, z] for x in (-h, h) for y in (-h, h) for z in (-h, h)],
        dtype=np.float32,
    )
    faces = np.array(
        [[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
         [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]],
        dtype=np.int32,
    )
    return tf.Mesh(faces, points)


def sheet_table_volume(faces, points):
    p = np.asarray(points, dtype=np.float64)
    f = np.asarray(faces)
    if len(f) == 0:
        return 0.0
    return float(np.linalg.det(p[f]).sum() / 6.0)


def sheet_table_boundary_edges(faces):
    f = np.asarray(faces)
    if len(f) == 0:
        return np.zeros((0, 2), dtype=int)
    e = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
    uniq, cnt = np.unique(e, axis=0, return_counts=True)
    return uniq[cnt == 1]


def sheet_table_normals(faces, points):
    p = np.asarray(points, dtype=np.float64)
    f = np.asarray(faces)
    return np.cross(p[f[:, 1]] - p[f[:, 0]], p[f[:, 2]] - p[f[:, 0]])


@pytest.fixture(scope="module")
def sheet_graph():
    plane = tf.Mesh(*tf.make_plane_mesh(2.0, 2.0, 2, 2))    # z = 0, +z
    return tf.CsgGraph([plane, sheet_table_box()], sheets=[0])


def test_row_volume_minus_sheet_is_capped_half(sheet_graph):
    # "op(1) - op(0) | the box's upper half, closed and capped by the sheet"
    (faces, points), tags, _ = sheet_graph.mesh(
        tf.op(1) - tf.op(0), return_source_ids=True)
    assert len(sheet_table_boundary_edges(faces)) == 0
    assert sheet_table_volume(faces, points) == pytest.approx(0.5, rel=1e-6)
    cap = np.asarray(tags) == 0
    assert cap.sum() == 8
    n = sheet_table_normals(faces, points)
    assert np.all(n[cap][:, 2] < 0)                # capped: outward is down
    c = np.asarray(points, dtype=np.float64)[np.asarray(faces)].mean(axis=1)
    assert np.all(c[:, 2] >= -1e-6)                # the upper half


def test_row_volume_and_sheet_is_capped_half(sheet_graph):
    # "op(1) & op(0) | the box's lower half, closed and capped by the sheet"
    (faces, points), tags, _ = sheet_graph.mesh(
        tf.op(1) & tf.op(0), return_source_ids=True)
    assert len(sheet_table_boundary_edges(faces)) == 0
    assert sheet_table_volume(faces, points) == pytest.approx(0.5, rel=1e-6)
    cap = np.asarray(tags) == 0
    assert cap.sum() == 8
    n = sheet_table_normals(faces, points)
    assert np.all(n[cap][:, 2] > 0)                # capped: outward is up
    c = np.asarray(points, dtype=np.float64)[np.asarray(faces)].mean(axis=1)
    assert np.all(c[:, 2] <= 1e-6)                 # the lower half


def test_row_sheet_minus_volume_is_open_along_rim(sheet_graph):
    # "op(0) - op(1) | the boundary of the unbounded region below the sheet
    #  and outside the box: the sheet's annulus plus the box's lower walls,
    #  open along the sheet's rim"
    (faces, points), tags, _ = sheet_graph.mesh(
        tf.op(0) - tf.op(1), return_source_ids=True)
    t = np.asarray(tags)
    assert (t == 0).sum() == 16                    # the annulus
    assert (t == 1).sum() == 14                    # the box's lower walls
    open_edges = sheet_table_boundary_edges(faces)
    assert len(open_edges) == 8                    # open along the sheet's rim
    p = np.asarray(points, dtype=np.float64)
    mids = (p[open_edges[:, 0]] + p[open_edges[:, 1]]) / 2
    assert np.all(np.isclose(np.abs(mids[:, :2]).max(axis=1), 1.0))
    n = sheet_table_normals(faces, points)
    assert np.all(n[t == 0][:, 2] > 0)             # outward from below: up
    c = p[np.asarray(faces)].mean(axis=1)
    into_box = np.einsum("ij,ij->i", n[t == 1], BOX_CENTER - c[t == 1])
    assert np.all(into_box > 0)                    # outward from the region


def test_row_selection_reads_annulus_wound_away(sheet_graph):
    # "~op(0) & ~op(1), selection=[0] | the sheet outside the box (the
    #  annulus), wound away from that region"
    (faces, points), tags, _ = sheet_graph.mesh(
        ~tf.op(0) & ~tf.op(1), selection=[0], return_source_ids=True)
    t = np.asarray(tags)
    assert len(faces) == 16 and np.all(t == 0)
    n = sheet_table_normals(faces, points)
    assert np.all(n[:, 2] < 0)                     # away from the region above


def test_row_inside_reads_cap_own_winding(sheet_graph):
    # "op(1), inside=[0] | the sheet inside the box (the cap), the sheet's
    #  own winding"
    (faces, points), tags, _ = sheet_graph.mesh(
        tf.op(1), inside=[0], return_source_ids=True)
    t = np.asarray(tags)
    assert len(faces) == 8 and np.all(t == 0)
    n = sheet_table_normals(faces, points)
    assert np.all(n[:, 2] > 0)                     # the sheet's own +z


def test_row_inside_reads_annulus_own_winding(sheet_graph):
    # "~op(1), inside=[0] | the annulus, the sheet's own winding"
    (faces, points), tags, _ = sheet_graph.mesh(
        ~tf.op(1), inside=[0], return_source_ids=True)
    t = np.asarray(tags)
    assert len(faces) == 16 and np.all(t == 0)
    n = sheet_table_normals(faces, points)
    assert np.all(n[:, 2] > 0)                     # the sheet's own +z


def test_row_no_sheet_piece_bounds_the_volume(sheet_graph):
    # "op(1), selection=[0] | nothing: both sides of the cap are inside the
    #  box, so no piece of the sheet bounds op(1)"
    (faces, _), _, _ = sheet_graph.mesh(
        tf.op(1), selection=[0], return_source_ids=True)
    assert len(faces) == 0


def test_row_volume_walls_below_sheet_outward(sheet_graph):
    # "op(0), inside=[1] | the box's walls below the sheet, outward"
    (faces, points), tags, _ = sheet_graph.mesh(
        tf.op(0), inside=[1], return_source_ids=True)
    t = np.asarray(tags)
    assert len(faces) == 14 and np.all(t == 1)
    p = np.asarray(points, dtype=np.float64)
    c = p[np.asarray(faces)].mean(axis=1)
    assert np.all(c[:, 2] <= 1e-6)                 # below the sheet
    n = sheet_table_normals(faces, points)
    outward = np.einsum("ij,ij->i", n, c - BOX_CENTER)
    assert np.all(outward > 0)


def test_row_sheets_own_bit_is_empty(sheet_graph):
    # "op(0), inside=[0] | nothing: a sheet's own bit differs across each
    #  of its pieces"
    (faces, _), _, _ = sheet_graph.mesh(
        tf.op(0), inside=[0], return_source_ids=True)
    assert len(faces) == 0
