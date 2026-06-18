"""Tests for tf.domain_labels.

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
def test_domain_labels_box_default(index_dtype, real_dtype):
    mesh = _box_mesh(index_dtype, real_dtype)
    labels, n_domains, outer_shell_label = tf.domain_labels(mesh)

    # Closed cube: 1 bounded interior + 1 outer shell = 2 domains.
    assert n_domains == 2
    # Include mode: outer_shell_label is the index of the outer-shell
    # (unbounded) domain — a real label in range that faces carry, not
    # the out-of-range sentinel.
    assert 0 <= outer_shell_label < n_domains
    assert (labels == outer_shell_label).any()
    assert labels.shape == (mesh.faces.shape[0], 2)
    assert labels.dtype == index_dtype
    assert (labels >= 0).all() and (labels < n_domains).all()


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_domain_labels_box_exclude_outer_shell(index_dtype, real_dtype):
    mesh = _box_mesh(index_dtype, real_dtype)
    labels, n_domains, outer_shell_label = tf.domain_labels(
        mesh, exclude_outer_shell=True,
    )

    # With exclude_outer_shell, the universe maps to the sentinel label
    # (= n_domains). Only the bounded interior remains.
    assert n_domains == 1
    assert outer_shell_label == n_domains
    assert (labels == n_domains).any()


def test_domain_labels_default_no_sentinel():
    faces, points = tf.make_box_mesh(1.0, 1.0, 1.0)
    labels, n_domains, _ = tf.domain_labels(tf.Mesh(faces, points))
    # Default config: no side carries the sentinel.
    assert not (labels == n_domains).any()


def test_domain_labels_tuple_input():
    faces, points = tf.make_box_mesh(1.0, 1.0, 1.0)
    labels, n_domains, _ = tf.domain_labels((faces, points))
    assert n_domains == 2
    assert labels.shape == (faces.shape[0], 2)


def test_domain_labels_rejects_2d_mesh():
    # 2D points should be rejected — domains are a 3D concept.
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    points = np.zeros((3, 2), dtype=np.float32)
    with pytest.raises((ValueError, TypeError)):
        tf.domain_labels((faces, points))


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_domain_labels_dynamic_box(index_dtype, real_dtype):
    """Exercise the V=dyn dispatch path by wrapping faces in OffsetBlockedArray."""
    faces, points = tf.make_box_mesh(1.0, 1.0, 1.0)
    faces_dyn = tf.as_offset_blocked(faces.astype(index_dtype, copy=False))
    points = points.astype(real_dtype, copy=False)
    labels, n_domains, outer_shell_label = tf.domain_labels(
        (faces_dyn, points),
    )
    assert n_domains == 2
    assert 0 <= outer_shell_label < n_domains
    assert (labels == outer_shell_label).any()
    assert labels.shape == (faces.shape[0], 2)
    assert labels.dtype == index_dtype


# ==============================================================================
# Main runner
# ==============================================================================

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
