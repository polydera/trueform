"""
Tests for tf.cdt — constrained Delaunay triangulation.

Copyright (c) 2025 Žiga Sajovic, XLAB
"""

import sys
import numpy as np
import pytest
import trueform as tf


REAL_DTYPES = [np.float32, np.float64]


# ==============================================================================
# No-edges (convex hull) variants
# ==============================================================================

@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_cdt_points_only(dtype):
    """Convex-hull Delaunay triangulation of a small point set."""
    rng = np.random.default_rng(seed=42)
    points = rng.uniform(0, 1, (50, 2)).astype(dtype)

    faces, out_points = tf.cdt(points)

    assert faces.ndim == 2 and faces.shape[1] == 3
    assert out_points.shape[1] == 2
    assert out_points.dtype == dtype
    assert faces.dtype == np.int32
    assert np.all(faces >= 0) and np.all(faces < out_points.shape[0])


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_cdt_points_only_with_index_map(dtype):
    """No edges, return_index_map=True returns ((faces, points), (f, kept_ids))."""
    rng = np.random.default_rng(seed=42)
    points = rng.uniform(0, 1, (50, 2)).astype(dtype)

    (faces, out_points), (f, kept_ids) = tf.cdt(points, return_index_map=True)

    assert faces.shape[1] == 3
    assert f.shape == (points.shape[0],)
    assert kept_ids.shape == (out_points.shape[0],)


# ==============================================================================
# With-edges variants
# ==============================================================================

def _circle_outline(n_boundary, dtype, n_steiner=0, seed=0):
    """First n_boundary points form a regular polygon outline; remaining are
    random Steiner points inside the unit square. Returns (points, edges)."""
    cx, cy, r = 0.5, 0.5, 0.45
    theta = np.linspace(0, 2 * np.pi, n_boundary, endpoint=False)
    boundary = np.column_stack(
        [cx + r * np.cos(theta), cy + r * np.sin(theta)]
    ).astype(dtype)

    if n_steiner > 0:
        rng = np.random.default_rng(seed)
        steiner = rng.uniform(0, 1, (n_steiner, 2)).astype(dtype)
        points = np.vstack([boundary, steiner])
    else:
        points = boundary

    edges = np.column_stack(
        [np.arange(n_boundary), (np.arange(n_boundary) + 1) % n_boundary]
    ).astype(np.int32)

    return points, edges


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_cdt_with_edges(dtype):
    """Constrained Delaunay with closed polygon outline + Steiner points."""
    points, edges = _circle_outline(n_boundary=32, dtype=dtype, n_steiner=64)

    faces, out_points = tf.cdt(points, edges)

    assert faces.shape[1] == 3
    assert out_points.dtype == dtype
    assert faces.dtype == np.int32
    # The interior parity filter keeps only triangles inside the outline,
    # so the output triangle count should be strictly less than the
    # convex-hull triangulation.
    full_faces, _ = tf.cdt(points)
    assert faces.shape[0] < full_faces.shape[0]


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_cdt_with_edges_and_mask(dtype):
    """edge_mask=False everywhere should drop interior filtering — every edge
    becomes a non-boundary constraint, so parity stays at 0 and the output
    is empty (no parity-1 triangles)."""
    points, edges = _circle_outline(n_boundary=16, dtype=dtype)
    edge_mask = np.zeros(edges.shape[0], dtype=bool)

    faces, _ = tf.cdt(points, edges, edge_mask=edge_mask)
    # All non-boundary constraints + no boundary edges => parity never flips
    # => no triangles labelled 1 => empty output.
    assert faces.shape[0] == 0


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_cdt_with_edges_and_index_map(dtype):
    """With edges + return_index_map returns ((faces, points), (f, kept_ids))."""
    points, edges = _circle_outline(n_boundary=16, dtype=dtype, n_steiner=32)

    (faces, out_points), (f, kept_ids) = tf.cdt(
        points, edges, return_index_map=True
    )

    assert faces.shape[1] == 3
    assert f.shape == (points.shape[0],)
    assert kept_ids.shape == (out_points.shape[0],)


# ==============================================================================
# Error handling
# ==============================================================================

def test_cdt_rejects_3d_points():
    points = np.zeros((10, 3), dtype=np.float32)
    with pytest.raises(ValueError):
        tf.cdt(points)


def test_cdt_rejects_int_points():
    points = np.zeros((10, 2), dtype=np.int32)
    with pytest.raises(TypeError):
        tf.cdt(points)


def test_cdt_edge_mask_requires_edges():
    points = np.zeros((10, 2), dtype=np.float32)
    with pytest.raises(ValueError):
        tf.cdt(points, edge_mask=np.zeros(0, dtype=bool))


def test_cdt_edge_mask_length_mismatch():
    points, edges = _circle_outline(n_boundary=8, dtype=np.float32)
    bad_mask = np.zeros(edges.shape[0] + 1, dtype=bool)
    with pytest.raises(ValueError):
        tf.cdt(points, edges, edge_mask=bad_mask)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
