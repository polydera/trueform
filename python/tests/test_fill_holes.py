"""
Test fill_holes and filled_mesh

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

def punched_sphere(index_dtype, real_dtype, cap=0.8):
    """A sphere with one polar cap removed: one closed rim."""
    faces, points = tf.make_sphere_mesh(
        1.0, 24, 24, dtype=real_dtype, index_dtype=index_dtype)
    centroids = points[faces].mean(axis=1)
    faces = faces[centroids[:, 2] < cap]
    used = np.unique(faces)
    remap = np.full(len(points), -1, dtype=index_dtype)
    remap[used] = np.arange(len(used), dtype=index_dtype)
    return remap[faces], np.ascontiguousarray(points[used])


def punched_box(index_dtype, real_dtype):
    """A box with one face's two triangles removed: a four-vertex rim."""
    faces, points = tf.make_box_mesh(
        1.0, 1.0, 1.0, dtype=real_dtype, index_dtype=index_dtype)
    centroids = points[faces].mean(axis=1)
    keep = centroids[:, 2] < centroids[:, 2].max() - 1e-6
    return np.ascontiguousarray(faces[keep]), points


def as_dynamic(faces):
    offsets = np.arange(0, 3 * len(faces) + 1, 3, dtype=faces.dtype)
    return tf.OffsetBlockedArray(offsets, faces.reshape(-1).copy())


def assert_closed(faces, points):
    mesh = tf.Mesh(faces, points)
    assert tf.is_closed(mesh)
    assert len(tf.boundary_edges(mesh)) == 0


# ==============================================================================
# Tests
# ==============================================================================

@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_fill_holes_closes_a_sphere(index_dtype, real_dtype):
    """One rim, one group; the chart continues the sphere and the replay closes it."""
    faces, points = punched_sphere(index_dtype, real_dtype)
    mesh = tf.Mesh(faces, points)
    fill = tf.fill_holes(mesh, tf.boundary_rims(mesh))

    assert len(fill) == 1
    assert fill.status[0] == tf.HoleFillStatus.FILLED
    assert fill.charted[0] == tf.HoleChartStatus.ADMITTED
    assert fill.triangle_offsets[1] > 0
    assert fill.group_offsets.tolist() == [0, 1]
    assert fill.offending.shape == (1, 2)
    assert fill.offending[0].tolist() == [-1, -1]

    out_faces, out_points = tf.filled_mesh(mesh, fill)
    assert out_faces.dtype == index_dtype
    assert out_points.dtype == real_dtype
    assert out_points.shape == (len(points) + len(fill.minted_points), 3)
    assert len(out_faces) == len(faces) + len(fill.triangles)
    assert_closed(out_faces, out_points)

    # The chart continues the surface: every minted point stands near the sphere.
    minted = fill.minted_points
    if len(minted):
        assert np.abs(np.linalg.norm(minted, axis=1) - 1.0).max() < 0.05


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
def test_fill_holes_planar_rim(index_dtype):
    """A box missing one side fills flat and closes."""
    faces, points = punched_box(index_dtype, np.float32)
    mesh = tf.Mesh(faces, points)
    fill = tf.fill_holes(mesh, tf.boundary_rims(mesh))

    assert fill.status.tolist() == [tf.HoleFillStatus.FILLED]
    out_faces, out_points = tf.filled_mesh(mesh, fill)
    assert_closed(out_faces, out_points)


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
def test_fill_holes_dynamic(index_dtype):
    """The dynamic entry states the same fill and replays variable faces."""
    faces, points = punched_sphere(index_dtype, np.float64)
    tri = tf.fill_holes(tf.Mesh(faces, points),
                        tf.boundary_rims(tf.Mesh(faces, points)))
    mesh = tf.Mesh(as_dynamic(faces), points)
    dyn = tf.fill_holes(mesh, tf.boundary_rims(mesh))

    assert dyn.status.tolist() == tri.status.tolist()
    assert dyn.charted.tolist() == tri.charted.tolist()
    assert np.array_equal(dyn.triangles, tri.triangles)
    out_faces, out_points = tf.filled_mesh(mesh, dyn)
    assert isinstance(out_faces, tf.OffsetBlockedArray)
    assert len(out_faces) == len(faces) + len(dyn.triangles)


@pytest.mark.parametrize("split_rim", [True, False])
def test_fill_holes_config(split_rim):
    """The config reaches the native fill; without splits nothing is published."""
    faces, points = punched_sphere(np.int32, np.float32)
    mesh = tf.Mesh(faces, points)
    fill = tf.fill_holes(mesh, tf.boundary_rims(mesh), min_quality=0.0,
                         fairing=False, split_rim=split_rim)
    assert fill.splits.shape[1] == 4
    assert len(fill.split_parameters) == len(fill.splits)
    if not split_rim:
        assert len(fill.splits) == 0
        assert len(fill.plan_faces) == 0
    assert fill.status[0] != tf.HoleFillStatus.REFUSED_RIM_SPLIT or not split_rim
    out_faces, out_points = tf.filled_mesh(mesh, fill)
    assert_closed(out_faces, out_points)


def test_fill_holes_no_rims():
    """A closed mesh has no rims and the fill holds no groups."""
    faces, points = tf.make_sphere_mesh(1.0, 12, 12)
    mesh = tf.Mesh(faces, points)
    fill = tf.fill_holes(mesh, tf.boundary_rims(mesh))
    assert len(fill) == 0
    assert fill.group_offsets.tolist() == [0]
    out_faces, out_points = tf.filled_mesh(mesh, fill)
    assert np.array_equal(out_faces, faces)
    assert np.array_equal(out_points, points)


def test_fill_holes_result_outlives_inputs():
    """The result owns its arrays; dropping the mesh and rims keeps them valid."""
    faces, points = punched_sphere(np.int32, np.float32)
    mesh = tf.Mesh(faces, points)
    rims = tf.boundary_rims(mesh)
    fill = tf.fill_holes(mesh, rims)
    expected = fill.triangles.copy()
    del mesh, rims, faces, points
    gc.collect()
    assert np.array_equal(fill.triangles, expected)


def test_fill_holes_status_enums_match_native():
    """The Python enums carry the native order."""
    assert tf.HoleFillStatus.REFUSED_RIM_SPLIT == 7
    assert tf.HoleChartStatus.RIM_PLANE == 2
    assert tf.HoleChartStatus.OUT_OF_LATTICE == 12
    assert tf.HoleRefineStatus.BUDGET_EXHAUSTED == 3
    assert tf.HoleFairStatus.FAIRING_REFUSED == 3


def test_fill_holes_invalid_inputs():
    faces, points = punched_sphere(np.int32, np.float32)
    mesh = tf.Mesh(faces, points)
    with pytest.raises(TypeError):
        tf.fill_holes((faces, points), tf.boundary_rims(mesh))
    with pytest.raises(TypeError):
        tf.fill_holes(mesh, [1, 2, 3])
    other = tf.Mesh(faces.astype(np.int64), points)
    with pytest.raises(TypeError):
        tf.fill_holes(mesh, tf.boundary_rims(other))
