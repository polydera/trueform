"""
Tests for preserve_regions on the remesh ops (decimated, simplified,
isotropic_remeshed).

Region labels are carried through the remesh as a per-face attribute: when
preserve_regions is given the op returns a third value, the per-face labels of
the output mesh (int32, one per output face). When omitted the op returns the
usual (faces, points).

Copyright (c) 2025 Žiga Sajovic, XLAB
"""

import numpy as np
import pytest
import trueform as tf


REAL_DTYPES = [np.float32, np.float64]
INDEX_DTYPES = [np.int32, np.int64]


def _box_with_two_regions(dtype, index_dtype):
    """A subdivided box plus a per-face label array splitting it in two by the
    sign of each face's centroid x."""
    faces, points = tf.make_box_mesh(
        2, 3, 4, 6, 6, 6, dtype=dtype, index_dtype=index_dtype
    )
    centroid_x = points[faces].mean(axis=1)[:, 0]
    labels = (centroid_x > 0).astype(np.int32)
    return faces, points, labels


def _ops(faces, points):
    """(name, callable) for each remesh op, callable takes keyword options."""
    mel = tf.mean_edge_length((faces, points))
    return [
        ("simplified", lambda **k: tf.simplified((faces, points), **k)),
        ("decimated", lambda **k: tf.decimated((faces, points), 0.5, **k)),
        (
            "isotropic_remeshed",
            lambda **k: tf.isotropic_remeshed((faces, points), 2.0 * mel, **k),
        ),
    ]


# ==============================================================================
# Contract: with preserve_regions -> (faces, points, labels)
# ==============================================================================

@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_returns_labels_one_per_output_face(dtype, index_dtype):
    faces, points, labels = _box_with_two_regions(dtype, index_dtype)
    for name, call in _ops(faces, points):
        out = call(preserve_regions=labels)
        assert len(out) == 3, f"{name}: expected (faces, points, labels)"
        out_faces, out_points, out_labels = out
        assert out_faces.shape[0] > 0, f"{name}: should have faces"
        assert out_labels.shape[0] == out_faces.shape[0], (
            f"{name}: one label per output face"
        )
        assert out_labels.ndim == 1, f"{name}: labels are 1-D"
        # labels round-trip as int32 regardless of the mesh index dtype
        assert out_labels.dtype == np.int32, f"{name}: labels are int32"
        # the two input regions are both still present
        assert set(np.unique(out_labels)).issubset({0, 1})


# ==============================================================================
# Contract: without preserve_regions -> (faces, points)
# ==============================================================================

@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_no_regions_returns_two_tuple(dtype, index_dtype):
    faces, points, _ = _box_with_two_regions(dtype, index_dtype)
    for name, call in _ops(faces, points):
        out = call()
        assert len(out) == 2, f"{name}: no regions -> (faces, points)"
        out_none = call(preserve_regions=None)
        assert len(out_none) == 2, f"{name}: preserve_regions=None -> 2-tuple"


# ==============================================================================
# A single region is preserved end to end (no region is dropped)
# ==============================================================================

@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_both_regions_survive(dtype, index_dtype):
    faces, points, labels = _box_with_two_regions(dtype, index_dtype)
    for name, call in _ops(faces, points):
        _, _, out_labels = call(preserve_regions=labels)
        present = set(np.unique(out_labels))
        assert present == {0, 1}, (
            f"{name}: both regions should survive, got {present}"
        )


# ==============================================================================
# Validation: wrong-length label array raises
# ==============================================================================

@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_wrong_length_labels_raises(dtype, index_dtype):
    faces, points, labels = _box_with_two_regions(dtype, index_dtype)
    bad = labels[:-1]  # one short (and not empty)
    for name, call in _ops(faces, points):
        with pytest.raises(ValueError):
            call(preserve_regions=bad)


# ==============================================================================
# An empty preserve_regions buffer is also wrong-sized: the binding forbids it.
# (The empty-range "no regions" convenience is a C++-library detail; a Python
# caller omits preserve_regions instead of passing an empty array.)
# ==============================================================================

@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_empty_regions_raises(dtype, index_dtype):
    faces, points, _ = _box_with_two_regions(dtype, index_dtype)
    empty = np.empty(0, dtype=np.int32)
    for name, call in _ops(faces, points):
        with pytest.raises(ValueError):
            call(preserve_regions=empty)


# ==============================================================================
# Labels accept any int input dtype (cast to int32 internally)
# ==============================================================================

@pytest.mark.parametrize("label_dtype", [np.int8, np.int32, np.int64, np.uint16])
def test_label_input_dtype_is_accepted(label_dtype):
    faces, points, labels = _box_with_two_regions(np.float64, np.int32)
    labels = labels.astype(label_dtype)
    for name, call in _ops(faces, points):
        _, _, out_labels = call(preserve_regions=labels)
        assert out_labels.dtype == np.int32, f"{name}: output labels int32"


# ==============================================================================
# feature_angle still works on the no-regions path (preserve_regions omitted)
# ==============================================================================

_CUBE_CORNERS = np.array(
    [[sx, sy, sz] for sx in (-1.0, 1.0) for sy in (-1.0, 1.0)
     for sz in (-1.0, 1.0)], dtype=np.float64
)


def _count_corners(points):
    """How many of the 8 (+-1,+-1,+-1) cube corners survive exactly."""
    return sum(
        bool(np.any(np.all(np.isclose(points, c, atol=1e-9), axis=1)))
        for c in _CUBE_CORNERS
    )


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_feature_angle_locks_corners_without_regions(dtype, index_dtype):
    """feature_angle must preserve sharp corners even with no preserve_regions.

    A box has 8 sharp corners (3 feature edges each). They are interior
    vertices of a closed mesh, so the relaxation cleanup drifts them off the
    exact (+-1,+-1,+-1) positions unless feature_angle marks them as corners
    and freezes them. With feature_angle set (and NO regions) all 8 must
    survive for every op.
    """
    faces, points = tf.make_box_mesh(
        2, 2, 2, 16, 16, 16, dtype=dtype, index_dtype=index_dtype
    )
    ops = [
        ("simplified",
         lambda fa: tf.simplified((faces, points), error_rel=0.05,
                                  parallel=False, feature_angle=fa)),
        ("decimated",
         lambda fa: tf.decimated((faces, points), 0.05, parallel=False,
                                 feature_angle=fa)),
        ("isotropic_remeshed",
         lambda fa: tf.isotropic_remeshed(
             (faces, points), 2.0 * tf.mean_edge_length((faces, points)),
             parallel=False, feature_angle=fa)),
    ]
    for name, call in ops:
        _, p_feat = call(30.0)  # feature_angle on, no preserve_regions
        assert _count_corners(p_feat) == 8, (
            f"{name}: feature_angle (no regions) must keep all 8 box corners"
        )


def test_feature_angle_is_not_a_noop_without_regions():
    """Discriminating check: simplify's relaxation drops the corners without
    feature_angle, so feature_angle demonstrably changes the result on the
    no-regions path (it is not silently ignored)."""
    faces, points = tf.make_box_mesh(
        2, 2, 2, 16, 16, 16, dtype=np.float64, index_dtype=np.int32
    )
    _, p_plain = tf.simplified((faces, points), error_rel=0.05, parallel=False)
    _, p_feat = tf.simplified((faces, points), error_rel=0.05, parallel=False,
                              feature_angle=30.0)
    assert _count_corners(p_feat) == 8
    assert _count_corners(p_plain) < 8, (
        "without feature_angle the relaxation should move the corners"
    )
