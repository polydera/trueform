"""Tests for tf.split_into_domains.

Copyright (c) 2025 Žiga Sajovic, XLAB
"""
import sys

import numpy as np
import pytest

import trueform as tf

INDEX_DTYPES = [np.int32, np.int64]
REAL_DTYPES = [np.float32, np.float64]


def _box_mesh(index_dtype, real_dtype):
    faces, points = tf.make_box_mesh(1.0, 1.0, 1.0)
    return tf.Mesh(
        faces.astype(index_dtype, copy=False),
        points.astype(real_dtype, copy=False),
    )


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_split_into_domains_box(index_dtype, real_dtype):
    mesh = _box_mesh(index_dtype, real_dtype)
    dl = tf.domain_labels(mesh)
    comps, ids = tf.split_into_domains(mesh, dl)

    # 2 domains → 2 submeshes.
    assert len(comps) == 2
    assert len(ids) == 2

    n_faces = mesh.faces.shape[0]
    for (f, p), _ in zip(comps, ids):
        # Each face contributes to both domains (one per side).
        assert f.shape == (n_faces, 3)
        assert f.dtype == index_dtype
        assert p.dtype == real_dtype


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_split_into_domains_exclude_outer_shell(index_dtype, real_dtype):
    mesh = _box_mesh(index_dtype, real_dtype)
    dl = tf.domain_labels(mesh, exclude_outer_shell=True)
    comps, ids = tf.split_into_domains(mesh, dl)

    # Outer shell excluded → only 1 bounded interior remains.
    assert len(comps) == 1
    f, _ = comps[0]
    assert f.shape == (mesh.faces.shape[0], 3)


def test_split_into_domains_tuple_input():
    faces, points = tf.make_box_mesh(1.0, 1.0, 1.0)
    dl = tf.domain_labels((faces, points))
    comps, ids = tf.split_into_domains((faces, points), dl)
    assert len(comps) == 2


def test_split_into_domains_rejects_mismatched_dtype():
    faces, points = tf.make_box_mesh(1.0, 1.0, 1.0)
    mesh = tf.Mesh(faces, points)
    labels, n_domains, outer_shell_label = tf.domain_labels(mesh)
    # Cast labels to wrong dtype.
    other_dtype = np.int64 if labels.dtype == np.int32 else np.int32
    mismatched = (labels.astype(other_dtype), n_domains, outer_shell_label)
    with pytest.raises(TypeError):
        tf.split_into_domains(mesh, mismatched)


def test_split_into_domains_accepts_2_tuple():
    # 2-tuple (labels, n_domains) — outer_shell_label optional.
    faces, points = tf.make_box_mesh(1.0, 1.0, 1.0)
    mesh = tf.Mesh(faces, points)
    labels, n_domains, _ = tf.domain_labels(mesh)
    comps, ids = tf.split_into_domains(mesh, (labels, n_domains))
    assert len(comps) == 2


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_split_into_domains_dynamic_box(index_dtype, real_dtype):
    """Exercise the V=dyn dispatch path: wrap faces in OffsetBlockedArray
    on the way in, expect OffsetBlockedArray faces on the way out."""
    from trueform import OffsetBlockedArray

    faces, points = tf.make_box_mesh(1.0, 1.0, 1.0)
    faces_dyn = tf.as_offset_blocked(faces.astype(index_dtype, copy=False))
    points = points.astype(real_dtype, copy=False)
    dl = tf.domain_labels((faces_dyn, points))
    comps, ids = tf.split_into_domains((faces_dyn, points), dl)

    assert len(comps) == 2
    assert len(ids) == 2
    for sub_faces, sub_points in comps:
        # Output mirrors input layout — dynamic in, dynamic out.
        assert isinstance(sub_faces, OffsetBlockedArray)
        assert sub_faces.dtype == index_dtype
        assert sub_points.dtype == real_dtype
        # Box has 12 triangle face-sides per domain.
        assert sub_points.shape[1] == 3


# ==============================================================================
# Main runner
# ==============================================================================

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
