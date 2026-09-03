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
# split_constraints
# ==============================================================================

def _crossing_constraints(dtype):
    """Unit square corners; the two diagonals cross at the centre."""
    points = np.array(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=dtype
    )
    edges = np.array([[0, 2], [1, 3]], dtype=np.int32)
    return points, edges


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_cdt_split_constraints_creates_crossing_vertex(dtype):
    """Default split_constraints resolves the crossing by adding a point."""
    points, edges = _crossing_constraints(dtype)

    faces, out_points = tf.cdt(points, edges)

    assert faces.shape[0] > 0
    assert out_points.shape[0] > points.shape[0]
    for corner in points:
        assert np.any(np.all(np.isclose(out_points, corner), axis=1))


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_cdt_split_constraints_false_refuses_crossing(dtype):
    """split_constraints=False refuses the crossing: the empty result is the
    answer, not an error. Both faces and points come back empty."""
    points, edges = _crossing_constraints(dtype)

    faces, out_points = tf.cdt(points, edges, split_constraints=False)

    assert faces.shape[0] == 0
    assert out_points.shape[0] == 0


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_cdt_split_constraints_false_matches_default_without_crossings(dtype):
    """Non-crossing constraints are unaffected by the flag."""
    points, edges = _circle_outline(n_boundary=16, dtype=dtype, n_steiner=32)

    faces, out_points = tf.cdt(points, edges, split_constraints=False)
    ref_faces, ref_points = tf.cdt(points, edges)

    assert faces.shape[0] > 0
    assert np.array_equal(faces, ref_faces)
    assert np.array_equal(out_points, ref_points)


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_cdt_split_constraints_with_index_map(dtype):
    """return_index_map accepts split_constraints and refuses the same way."""
    points, edges = _crossing_constraints(dtype)

    (faces, out_points), (f, kept_ids) = tf.cdt(
        points, edges, split_constraints=False, return_index_map=True
    )

    assert faces.shape[0] == 0
    assert out_points.shape[0] == 0
    assert kept_ids.shape == (out_points.shape[0],)

    (ok_faces, _), _ = tf.cdt(points, edges, return_index_map=True)
    assert ok_faces.shape[0] > 0
    assert f.shape == (points.shape[0],)


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_cdt_split_constraints_with_edge_mask(dtype):
    """The flag threads through the explicit edge_mask overload too."""
    points, edges = _crossing_constraints(dtype)
    edge_mask = np.ones(edges.shape[0], dtype=bool)

    faces, _ = tf.cdt(
        points, edges, edge_mask=edge_mask, split_constraints=False
    )
    assert faces.shape[0] == 0


# ==============================================================================
# region_labels
# ==============================================================================

def _holed_square(dtype):
    """Points 0-3 the outer wall, 4-7 the hole wall, 8-11 unconstrained hull
    corners beyond both, so the hull-exterior band owns triangles."""
    points = np.array(
        [[2, 2], [8, 2], [8, 8], [2, 8],
         [4, 4], [6, 4], [6, 6], [4, 6],
         [0, 0], [10, 0], [10, 10], [0, 10]], dtype=dtype
    )
    edges = np.array(
        [[0, 1], [1, 2], [2, 3], [3, 0],
         [4, 5], [5, 6], [6, 7], [7, 4]], dtype=np.int32
    )
    return points, edges


def _ringed_island(dtype):
    """A ring inside a hole inside a ring; the outermost wall is the hull."""
    points = np.array(
        [[0, 0], [20, 0], [20, 20], [0, 20],
         [4, 4], [16, 4], [16, 16], [4, 16],
         [8, 8], [12, 8], [12, 12], [8, 12],
         [9, 9], [11, 9], [11, 11], [9, 11]], dtype=dtype
    )
    outlines = [(0, 1, 2, 3), (4, 5, 6, 7), (8, 9, 10, 11), (12, 13, 14, 15)]
    edges = np.array(
        [[o[i], o[(i + 1) % 4]] for o in outlines for i in range(4)],
        dtype=np.int32,
    )
    return points, edges


def _centroid_in(centroids, lo, hi):
    return (
        (centroids[:, 0] > lo) & (centroids[:, 0] < hi)
        & (centroids[:, 1] > lo) & (centroids[:, 1] < hi)
    )


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_cdt_region_labels_nesting(dtype):
    """Nesting labels are the wall-crossing parity: hole and hull band 0,
    annulus 1; every triangle is returned, region 0 included."""
    points, edges = _holed_square(dtype)

    faces, out_points, labels = tf.cdt(points, edges, region_labels="nesting")

    assert labels.shape == (faces.shape[0],)
    assert labels.dtype == np.int32
    assert set(labels.tolist()) == {0, 1}

    centroids = out_points[faces].mean(axis=1)
    hole = _centroid_in(centroids, 4, 6)
    annulus = _centroid_in(centroids, 2, 8) & ~hole
    band = ~hole & ~annulus
    assert hole.any() and annulus.any() and band.any()
    assert np.all(labels[hole] == 0)
    assert np.all(labels[annulus] == 1)
    assert np.all(labels[band] == 0)


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_cdt_region_labels_components(dtype):
    """Component labels tell the hole from the exterior: three wall-cut
    components, 0 the hull-exterior band."""
    points, edges = _holed_square(dtype)

    faces, out_points, labels = tf.cdt(
        points, edges, region_labels="components"
    )

    assert set(labels.tolist()) == {0, 1, 2}
    centroids = out_points[faces].mean(axis=1)
    hole = _centroid_in(centroids, 4, 6)
    annulus = _centroid_in(centroids, 2, 8) & ~hole
    band = ~hole & ~annulus
    assert np.all(labels[band] == 0)
    hole_labels = set(labels[hole].tolist())
    annulus_labels = set(labels[annulus].tolist())
    assert len(hole_labels) == 1 and len(annulus_labels) == 1
    assert hole_labels != annulus_labels
    assert 0 not in hole_labels and 0 not in annulus_labels


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_cdt_region_labels_depth2_island(dtype):
    """The depth-2 island: nesting flattens it to the same parities as the
    outer ring and the holes, components give it its own id."""
    points, edges = _ringed_island(dtype)

    def regions(faces, out_points):
        centroids = out_points[faces].mean(axis=1)
        inner_hole = _centroid_in(centroids, 9, 11)
        island = _centroid_in(centroids, 8, 12) & ~inner_hole
        hole = _centroid_in(centroids, 4, 16) & ~island & ~inner_hole
        outer = ~inner_hole & ~island & ~hole
        return outer, hole, island, inner_hole

    faces, out_points, labels = tf.cdt(points, edges, region_labels="nesting")
    outer, hole, island, inner_hole = regions(faces, out_points)
    assert island.any()
    assert np.all(labels[outer] == 1)
    assert np.all(labels[hole] == 0)
    assert np.all(labels[island] == 1)
    assert np.all(labels[inner_hole] == 0)

    faces, out_points, labels = tf.cdt(
        points, edges, region_labels="components"
    )
    region_labels_seen = []
    for mask in regions(faces, out_points):
        distinct = set(labels[mask].tolist())
        assert len(distinct) == 1
        region_labels_seen.append(distinct.pop())
    assert len(set(region_labels_seen)) == 4
    # The hull outline is walled, so the hull-exterior component 0 owns
    # no triangle here.
    assert all(label > 0 for label in region_labels_seen)


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_cdt_region_labels_preserve_refusal(dtype):
    """A preserve-mode refusal answers with empty faces AND empty labels."""
    points, edges = _crossing_constraints(dtype)

    faces, out_points, labels = tf.cdt(
        points, edges, split_constraints=False, region_labels="nesting"
    )

    assert faces.shape[0] == 0
    assert out_points.shape[0] == 0
    assert labels.shape[0] == 0


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_cdt_region_labels_default_byte_compat(dtype):
    """The default entry is exactly the labels read filtered by nesting
    parity, points compacted in ascending welded order."""
    points, edges = _holed_square(dtype)

    def_faces, def_points = tf.cdt(points, edges)
    faces, out_points, labels = tf.cdt(points, edges, region_labels="nesting")

    kept_faces = faces[labels % 2 == 1]
    kept_ids = np.unique(kept_faces)
    remap = np.full(out_points.shape[0], -1, dtype=np.int32)
    remap[kept_ids] = np.arange(kept_ids.shape[0], dtype=np.int32)

    assert np.array_equal(def_faces, remap[kept_faces])
    assert np.array_equal(def_points, out_points[kept_ids])


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_cdt_region_labels_with_index_map(dtype):
    """With return_index_map the shape is ((faces, points), labels, (f, kept_ids))."""
    points, edges = _holed_square(dtype)

    (faces, out_points), labels, (f, kept_ids) = tf.cdt(
        points, edges, region_labels="nesting", return_index_map=True
    )

    assert labels.shape == (faces.shape[0],)
    assert f.shape == (points.shape[0],)
    assert kept_ids.shape == (out_points.shape[0],)


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_cdt_region_labels_with_edge_mask(dtype):
    """region_labels composes with edge_mask: an all-True mask equals the
    maskless call."""
    points, edges = _holed_square(dtype)
    edge_mask = np.ones(edges.shape[0], dtype=bool)

    faces, out_points, labels = tf.cdt(
        points, edges, edge_mask=edge_mask, region_labels="nesting"
    )
    ref_faces, ref_points, ref_labels = tf.cdt(
        points, edges, region_labels="nesting"
    )

    assert np.array_equal(faces, ref_faces)
    assert np.array_equal(out_points, ref_points)
    assert np.array_equal(labels, ref_labels)


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_cdt_region_labels_with_split_constraints(dtype):
    """region_labels composes with the default split path: the crossing is
    resolved and every quadrant is labelled."""
    points, edges = _crossing_constraints(dtype)

    faces, out_points, labels = tf.cdt(points, edges, region_labels="nesting")

    assert faces.shape[0] == 4
    assert out_points.shape[0] == 5
    assert sorted(labels.tolist()) == [0, 0, 1, 1]


def test_cdt_region_labels_requires_edges():
    points = np.zeros((10, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="region_labels requires edges"):
        tf.cdt(points, region_labels="nesting")


def test_cdt_region_labels_rejects_unknown_mode():
    points, edges = _holed_square(np.float32)
    with pytest.raises(ValueError, match="region_labels must be"):
        tf.cdt(points, edges, region_labels="bogus")


# ==============================================================================
# Contiguity normalization
# ==============================================================================

@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_cdt_non_contiguous_points_match_contiguous(dtype):
    """A sliced (N, 2) view of a wider array triangulates like its copy."""
    points, edges = _circle_outline(n_boundary=16, dtype=dtype, n_steiner=32)
    wide = np.concatenate(
        [points, np.ones((points.shape[0], 1), dtype=dtype)], axis=1
    )
    view = wide[:, :2]
    assert not view.flags["C_CONTIGUOUS"]

    faces_v, pts_v = tf.cdt(view, edges)
    faces_c, pts_c = tf.cdt(np.ascontiguousarray(view), edges)

    assert np.array_equal(faces_v, faces_c)
    assert np.array_equal(pts_v, pts_c)


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_cdt_non_contiguous_edges_and_mask_match_contiguous(dtype):
    """Strided int32 edges and a strided bool mask match their copies."""
    points, edges = _circle_outline(n_boundary=16, dtype=dtype, n_steiner=32)
    edges_view = np.repeat(edges, 2, axis=0)[::2]
    mask_view = np.ones(2 * edges.shape[0], dtype=np.bool_)[::2]
    assert not edges_view.flags["C_CONTIGUOUS"]
    assert not mask_view.flags["C_CONTIGUOUS"]

    faces_v, pts_v = tf.cdt(points, edges_view, edge_mask=mask_view)
    faces_c, pts_c = tf.cdt(points, edges)

    assert np.array_equal(faces_v, faces_c)
    assert np.array_equal(pts_v, pts_c)


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


def test_cdt_split_constraints_requires_edges():
    points = np.zeros((10, 2), dtype=np.float32)
    with pytest.raises(ValueError):
        tf.cdt(points, split_constraints=False)


def test_cdt_edge_mask_length_mismatch():
    points, edges = _circle_outline(n_boundary=8, dtype=np.float32)
    bad_mask = np.zeros(edges.shape[0] + 1, dtype=bool)
    with pytest.raises(ValueError):
        tf.cdt(points, edges, edge_mask=bad_mask)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
