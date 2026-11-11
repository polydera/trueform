"""
Test tet_mesh

Copyright (c) 2026 Ziga Sajovic, XLAB
"""

import gc

import pytest
import numpy as np
import trueform as tf

INDEX_DTYPES = [np.int32, np.int64]
REAL_DTYPES = [np.float32, np.float64]


# ==============================================================================
# Helpers
# ==============================================================================

def box(index_dtype, real_dtype, ticks=2):
    faces, points = tf.make_box_mesh(
        1.0, 2.0, 3.0, ticks, ticks, ticks, dtype=real_dtype,
        index_dtype=index_dtype)
    return faces, points


def overlapping_boxes(index_dtype, real_dtype):
    """Two boxes overlapping in one surface: a surface that crosses itself."""
    f0, p0 = tf.make_box_mesh(2.0, 2.0, 2.0, dtype=real_dtype,
                              index_dtype=index_dtype)
    p1 = p0 + np.array([0.5, 0.75, 1.25], dtype=real_dtype)
    faces = np.concatenate([f0, f0 + len(p0)]).astype(index_dtype)
    return faces, np.ascontiguousarray(np.concatenate([p0, p1]))


def volume(tets, points):
    a, b, c, d = (points[tets[:, k]] for k in range(4))
    return float(np.einsum('ij,ij->i', b - a, np.cross(c - a, d - a)).sum() / 6)


# ==============================================================================
# Tests
# ==============================================================================

@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_tet_mesh_fills_a_box(index_dtype, real_dtype):
    """A box meshes whole, in float64 points, to its own volume."""
    faces, points = box(index_dtype, real_dtype)
    tets, out, report = tf.tet_mesh(tf.Mesh(faces, points), return_report=True)
    assert report.status == tf.TetMeshStatus.MESHED
    assert report.drains == 0
    assert tets.shape[1] == 4 and len(tets) > 0
    assert tets.dtype == index_dtype
    assert out.dtype == np.float64 and out.shape[1] == 3
    assert abs(volume(tets, out) - 6.0) < 1e-9
    assert isinstance(report.census, dict)
    assert "faces" in report.census


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_tet_mesh_index_map(index_dtype, real_dtype):
    """Every input face lies in a facet, and the facets' triangles cover the
    box's area."""
    faces, points = box(index_dtype, real_dtype)
    tets, out, status, imap = tf.tet_mesh(
        tf.Mesh(faces, points), return_index_map=True)
    assert status == tf.TetMeshStatus.MESHED
    assert len(imap.face_facets) == len(faces)
    for f in range(len(faces)):
        assert len(imap.face_facets[f]) == 1
    assert imap.facet_offsets[-1] == len(imap.facet_triangles)
    tri = out[imap.facet_triangles]
    area = np.linalg.norm(
        np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1).sum() / 2
    assert abs(area - 22.0) < 1e-9
    plain, _, _ = tf.tet_mesh(tf.Mesh(faces, points))
    assert np.array_equal(plain, tets)


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
def test_tet_mesh_as_given(index_dtype):
    """A clean surface gives the same cells as given; a crossing refuses."""
    faces, points = box(index_dtype, np.float32)
    resolved, rp, _ = tf.tet_mesh(tf.Mesh(faces, points))
    given, gp, report = tf.tet_mesh(
        tf.Mesh(faces, points), split_constraints=False, return_report=True)
    assert report.status == tf.TetMeshStatus.MESHED
    assert report.drains == 0
    assert np.array_equal(resolved, given)
    assert np.array_equal(rp, gp)
    faces, points = overlapping_boxes(index_dtype, np.float32)
    _, _, meshed = tf.tet_mesh(tf.Mesh(faces, points))
    _, _, refused = tf.tet_mesh(tf.Mesh(faces, points), split_constraints=False)
    assert meshed == tf.TetMeshStatus.MESHED
    assert refused == tf.TetMeshStatus.REFUSED


def test_tet_mesh_config_fields():
    """A zero tolerance and the stated default budget are the default call."""
    faces, points = overlapping_boxes(np.int32, np.float64)
    tets, out, _ = tf.tet_mesh(tf.Mesh(faces, points))
    again, again_points, _ = tf.tet_mesh(
        tf.Mesh(faces, points), tolerance=0.0, site_budget_ratio=32)
    assert np.array_equal(tets, again)
    assert np.array_equal(out, again_points)
    with pytest.raises(TypeError):
        tf.tet_mesh(tf.Mesh(faces, points), site_budget_ratio=32.0)


@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_tet_mesh_transformation(real_dtype):
    """A mesh's transformation places it: a translated box's points are the
    box's points moved, its cells the same."""
    faces, points = box(np.int32, real_dtype)
    tets, out, _ = tf.tet_mesh(tf.Mesh(faces, points))
    mesh = tf.Mesh(faces, points)
    t = np.eye(4, dtype=real_dtype)
    t[:3, 3] = [5.0, -2.0, 7.0]
    mesh.transformation = t
    moved, moved_points, status = tf.tet_mesh(mesh)
    assert status == tf.TetMeshStatus.MESHED
    assert np.array_equal(tets, moved)
    assert np.abs(moved_points - (out + [5.0, -2.0, 7.0])).max() < 1e-6


def test_tet_mesh_rejects_other_carriers():
    faces, points = box(np.int32, np.float32)
    offsets = np.arange(0, 3 * len(faces) + 1, 3, dtype=np.int32)
    dynamic = tf.Mesh(tf.OffsetBlockedArray(offsets, faces.reshape(-1).copy()),
                      points)
    with pytest.raises(ValueError):
        tf.tet_mesh(dynamic)
    with pytest.raises(TypeError):
        tf.tet_mesh((faces, points))


def test_tet_mesh_outputs_outlive_the_mesh():
    faces, points = box(np.int32, np.float32)
    mesh = tf.Mesh(faces, points)
    tets, out, _ = tf.tet_mesh(mesh)
    expected = volume(tets, out)
    del mesh, faces, points
    gc.collect()
    assert abs(volume(tets, out) - expected) < 1e-12


def test_tet_mesh_report_is_a_request():
    """The default call answers the status; return_report answers the report
    in its place, the same cells either way, and composes with the map."""
    faces, points = box(np.int32, np.float64)
    tets, out, status = tf.tet_mesh(tf.Mesh(faces, points))
    assert isinstance(status, tf.TetMeshStatus)
    rtets, rout, report = tf.tet_mesh(tf.Mesh(faces, points), return_report=True)
    assert isinstance(report, tf.TetMeshReport)
    assert report.status == status
    assert np.array_equal(tets, rtets)
    assert np.array_equal(out, rout)
    _, _, both, imap = tf.tet_mesh(
        tf.Mesh(faces, points), return_report=True, return_index_map=True)
    assert isinstance(both, tf.TetMeshReport)
    assert len(imap.face_facets) == len(faces)
