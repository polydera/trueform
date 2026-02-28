"""
Test batch/broadcasting for primitive and form operations

Tests all 4 broadcast combinations for prim x prim:
  1. single x single -> scalar
  2. batch x single -> ndarray
  3. single x batch -> ndarray
  4. batch x batch -> ndarray (element-wise)

Tests form x batch prim for:
  - distance, distance2, intersects, ray_cast
  - neighbor_search (1-nn batch, k-nn batch)

Copyright (c) 2025 Žiga Sajovic, XLAB
"""

import sys
import pytest
import numpy as np
import trueform as tf


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_mesh_3d(dtype):
    """Simple 3D triangle mesh (two triangles forming a square in xy-plane at z=0)."""
    points = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]
    ], dtype=dtype)
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    return tf.Mesh(faces, points)


def _make_point_cloud_3d(dtype):
    """Simple 3D point cloud."""
    points = np.array([
        [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]
    ], dtype=dtype)
    return tf.PointCloud(points)


# ---------------------------------------------------------------------------
# prim x prim: distance / distance2
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", [np.float32, np.float64])
class TestBatchDistance:

    def test_single_x_single(self, dtype):
        """Baseline: single point x single segment -> scalar."""
        pt = tf.Point(np.array([0.0, 1.0, 0.0], dtype=dtype))
        seg = tf.Segment(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=dtype))
        d = tf.distance(pt, seg)
        d2 = tf.distance2(pt, seg)
        assert isinstance(d, float), "single x single should return scalar"
        assert abs(d - 1.0) < 1e-5
        assert abs(d2 - 1.0) < 1e-5

    def test_batch_x_single(self, dtype):
        """Batch points x single segment -> ndarray."""
        pts_data = np.array([
            [0.0, 1.0, 0.0],
            [0.5, 2.0, 0.0],
            [1.0, 0.0, 0.0],
        ], dtype=dtype)
        pts = tf.Point(pts_data)
        seg = tf.Segment(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=dtype))

        d = tf.distance(pts, seg)
        d2 = tf.distance2(pts, seg)
        assert isinstance(d, np.ndarray), "batch x single should return ndarray"
        assert d.shape == (3,)
        assert d2.shape == (3,)
        assert abs(d[0] - 1.0) < 1e-5   # (0,1,0) -> seg = 1
        assert abs(d[1] - 2.0) < 1e-5   # (0.5,2,0) -> seg = 2
        assert abs(d[2] - 0.0) < 1e-5   # (1,0,0) on seg

    def test_single_x_batch(self, dtype):
        """Single point x batch segments -> ndarray."""
        pt = tf.Point(np.array([0.0, 1.0, 0.0], dtype=dtype))
        segs_data = np.array([
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        ], dtype=dtype)
        segs = tf.Segment(segs_data)

        d = tf.distance(pt, segs)
        d2 = tf.distance2(pt, segs)
        assert isinstance(d, np.ndarray), "single x batch should return ndarray"
        assert d.shape == (2,)
        assert abs(d[0] - 1.0) < 1e-5   # dist to x-axis seg = 1
        assert abs(d[1] - 1.0) < 1e-5  # dist to z-axis seg (closest pt is origin)

    def test_batch_x_batch(self, dtype):
        """Batch points x batch segments (element-wise) -> ndarray."""
        pts_data = np.array([
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 2.0],
        ], dtype=dtype)
        segs_data = np.array([
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        ], dtype=dtype)
        pts = tf.Point(pts_data)
        segs = tf.Segment(segs_data)

        d = tf.distance(pts, segs)
        d2 = tf.distance2(pts, segs)
        assert isinstance(d, np.ndarray), "batch x batch should return ndarray"
        assert d.shape == (2,)
        assert abs(d[0] - 1.0) < 1e-5   # pt0 -> seg0
        assert abs(d[1] - 1.0) < 1e-5   # pt1 (0,0,2) -> seg1 (z-axis 0..1)

    def test_swap_order(self, dtype):
        """Swapping argument order should give the same result."""
        pts = tf.Point(np.array([[0.0, 1.0, 0.0], [0.5, 2.0, 0.0]], dtype=dtype))
        seg = tf.Segment(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=dtype))

        d_forward = tf.distance(pts, seg)
        d_backward = tf.distance(seg, pts)
        np.testing.assert_allclose(d_forward, d_backward, atol=1e-5)


# ---------------------------------------------------------------------------
# prim x prim: intersects
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", [np.float32, np.float64])
class TestBatchIntersects:

    def test_single_x_single(self, dtype):
        """Baseline: single point x single AABB -> bool."""
        pt = tf.Point(np.array([0.5, 0.5], dtype=dtype))
        box = tf.AABB(
            min=np.array([0.0, 0.0], dtype=dtype),
            max=np.array([1.0, 1.0], dtype=dtype))
        result = tf.intersects(pt, box)
        assert result is True or result == True

    def test_batch_x_single(self, dtype):
        """Batch points x single AABB -> ndarray."""
        pts = tf.Point(np.array([
            [0.5, 0.5],   # inside
            [2.0, 2.0],   # outside
            [1.0, 1.0],   # on corner
        ], dtype=dtype))
        box = tf.AABB(
            min=np.array([0.0, 0.0], dtype=dtype),
            max=np.array([1.0, 1.0], dtype=dtype))

        result = tf.intersects(pts, box)
        assert isinstance(result, np.ndarray), "batch x single should return ndarray"
        assert result.shape == (3,)
        assert result[0] == 1   # inside
        assert result[1] == 0   # outside
        assert result[2] == 1   # on corner

    def test_single_x_batch(self, dtype):
        """Single point x batch AABBs -> ndarray."""
        pt = tf.Point(np.array([0.5, 0.5], dtype=dtype))
        boxes = tf.AABB(np.array([
            [[0.0, 0.0], [1.0, 1.0]],   # contains point
            [[2.0, 2.0], [3.0, 3.0]],   # doesn't contain
        ], dtype=dtype))

        result = tf.intersects(pt, boxes)
        assert isinstance(result, np.ndarray), "single x batch should return ndarray"
        assert result.shape == (2,)
        assert result[0] == 1
        assert result[1] == 0

    def test_batch_x_batch(self, dtype):
        """Batch points x batch AABBs (element-wise) -> ndarray."""
        pts = tf.Point(np.array([
            [0.5, 0.5],
            [2.5, 2.5],
        ], dtype=dtype))
        boxes = tf.AABB(np.array([
            [[0.0, 0.0], [1.0, 1.0]],   # pt0 inside
            [[2.0, 2.0], [3.0, 3.0]],   # pt1 inside
        ], dtype=dtype))

        result = tf.intersects(pts, boxes)
        assert isinstance(result, np.ndarray), "batch x batch should return ndarray"
        assert result.shape == (2,)
        assert result[0] == 1
        assert result[1] == 1

    def test_swap_order(self, dtype):
        """Swapping argument order should give the same result."""
        pts = tf.Point(np.array([[0.5, 0.5], [2.0, 2.0]], dtype=dtype))
        box = tf.AABB(
            min=np.array([0.0, 0.0], dtype=dtype),
            max=np.array([1.0, 1.0], dtype=dtype))

        r1 = tf.intersects(pts, box)
        r2 = tf.intersects(box, pts)
        np.testing.assert_array_equal(r1, r2)


# ---------------------------------------------------------------------------
# prim x prim: closest_point_pair
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", [np.float32, np.float64])
class TestBatchClosestPointPair:

    def test_single_x_single(self, dtype):
        """Baseline: single point x single segment -> scalar metric + 2 points."""
        pt = tf.Point(np.array([0.5, 1.0, 0.0], dtype=dtype))
        seg = tf.Segment(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=dtype))

        metric, p0, p1 = tf.closest_point_pair(pt, seg)
        assert isinstance(metric, float), "single x single metric should be scalar"
        assert abs(metric - 1.0) < 1e-5  # distance² = 1
        assert p0.shape == (3,)
        assert p1.shape == (3,)
        # p0 should be the point itself
        np.testing.assert_allclose(p0, [0.5, 1.0, 0.0], atol=1e-5)
        # p1 should be closest on segment
        np.testing.assert_allclose(p1, [0.5, 0.0, 0.0], atol=1e-5)

    def test_batch_x_single(self, dtype):
        """Batch points x single segment -> ndarray metric + batch points."""
        pts = tf.Point(np.array([
            [0.0, 1.0, 0.0],
            [0.5, 2.0, 0.0],
        ], dtype=dtype))
        seg = tf.Segment(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=dtype))

        metric, p0, p1 = tf.closest_point_pair(pts, seg)
        assert isinstance(metric, np.ndarray)
        assert metric.shape == (2,)
        assert p0.shape == (2, 3)
        assert p1.shape == (2, 3)
        assert abs(metric[0] - 1.0) < 1e-5
        assert abs(metric[1] - 4.0) < 1e-5

    def test_single_x_batch(self, dtype):
        """Single point x batch segments -> ndarray metric + batch points."""
        pt = tf.Point(np.array([0.0, 1.0, 0.0], dtype=dtype))
        segs = tf.Segment(np.array([
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        ], dtype=dtype))

        metric, p0, p1 = tf.closest_point_pair(pt, segs)
        assert isinstance(metric, np.ndarray)
        assert metric.shape == (2,)
        assert p0.shape == (2, 3)
        assert p1.shape == (2, 3)

    def test_batch_x_batch(self, dtype):
        """Batch points x batch segments (element-wise) -> ndarray."""
        pts = tf.Point(np.array([
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 2.0],
        ], dtype=dtype))
        segs = tf.Segment(np.array([
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        ], dtype=dtype))

        metric, p0, p1 = tf.closest_point_pair(pts, segs)
        assert isinstance(metric, np.ndarray)
        assert metric.shape == (2,)
        assert abs(metric[0] - 1.0) < 1e-5  # pt0 -> seg0
        assert abs(metric[1] - 1.0) < 1e-5  # pt1 (0,0,2) -> seg1 (z: 0..1)


# ---------------------------------------------------------------------------
# prim x prim: ray_cast
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", [np.float32, np.float64])
class TestBatchRayCast:

    def test_single_x_single(self, dtype):
        """Baseline: single ray x single segment -> scalar t or None."""
        ray = tf.Ray(
            origin=np.array([0.0, 0.5], dtype=dtype),
            direction=np.array([1.0, 0.0], dtype=dtype))
        seg = tf.Segment(np.array([[1.0, 0.0], [1.0, 1.0]], dtype=dtype))

        t = tf.ray_cast(ray, seg)
        assert t is not None
        assert abs(t - 1.0) < 1e-5

    def test_batch_ray_x_single(self, dtype):
        """Batch rays x single segment -> ndarray of t values."""
        rays = tf.Ray(np.array([
            [[0.0, 0.5], [1.0, 0.0]],   # hits seg at t=1
            [[0.0, 0.5], [-1.0, 0.0]],  # misses (away)
            [[0.0, 0.25], [1.0, 0.0]],  # hits seg at t=1
        ], dtype=dtype))
        seg = tf.Segment(np.array([[1.0, 0.0], [1.0, 1.0]], dtype=dtype))

        result = tf.ray_cast(rays, seg)
        assert isinstance(result, np.ndarray), "batch ray x single should return ndarray"
        assert result.shape == (3,)
        assert abs(result[0] - 1.0) < 1e-5
        assert np.isnan(result[1]), "miss should be NaN"
        assert abs(result[2] - 1.0) < 1e-5

    def test_single_ray_x_batch(self, dtype):
        """Single ray x batch segments -> ndarray of t values."""
        ray = tf.Ray(
            origin=np.array([0.0, 0.5], dtype=dtype),
            direction=np.array([1.0, 0.0], dtype=dtype))
        segs = tf.Segment(np.array([
            [[1.0, 0.0], [1.0, 1.0]],   # hit at t=1
            [[3.0, 0.0], [3.0, 1.0]],   # hit at t=3
        ], dtype=dtype))

        result = tf.ray_cast(ray, segs)
        assert isinstance(result, np.ndarray), "single ray x batch should return ndarray"
        assert result.shape == (2,)
        assert abs(result[0] - 1.0) < 1e-5
        assert abs(result[1] - 3.0) < 1e-5

    def test_batch_ray_x_batch(self, dtype):
        """Batch rays x batch segments (element-wise) -> ndarray of t values."""
        rays = tf.Ray(np.array([
            [[0.0, 0.5], [1.0, 0.0]],   # hits seg0 at t=1
            [[0.0, 0.5], [1.0, 0.0]],   # hits seg1 at t=3
        ], dtype=dtype))
        segs = tf.Segment(np.array([
            [[1.0, 0.0], [1.0, 1.0]],   # x=1
            [[3.0, 0.0], [3.0, 1.0]],   # x=3
        ], dtype=dtype))

        result = tf.ray_cast(rays, segs)
        assert isinstance(result, np.ndarray), "batch x batch should return ndarray"
        assert result.shape == (2,)
        assert abs(result[0] - 1.0) < 1e-5
        assert abs(result[1] - 3.0) < 1e-5

    def test_scalar_config(self, dtype):
        """Scalar config tuple still works for batch rays."""
        rays = tf.Ray(np.array([
            [[0.0, 0.5], [1.0, 0.0]],   # hits seg at t=1
            [[0.0, 0.5], [1.0, 0.0]],   # hits seg at t=1
        ], dtype=dtype))
        seg = tf.Segment(np.array([[1.0, 0.0], [1.0, 1.0]], dtype=dtype))

        result = tf.ray_cast(rays, seg, config=(0.0, 0.5))
        assert result.shape == (2,)
        # max_t=0.5 is too short to reach t=1 → both miss
        assert np.isnan(result[0])
        assert np.isnan(result[1])

    def test_per_ray_config_arrays(self, dtype):
        """Per-ray config arrays for batch prim x prim."""
        rays = tf.Ray(np.array([
            [[0.0, 0.5], [1.0, 0.0]],   # hits seg at t=1
            [[0.0, 0.5], [1.0, 0.0]],   # hits seg at t=1
            [[0.0, 0.5], [1.0, 0.0]],   # hits seg at t=1
        ], dtype=dtype))
        seg = tf.Segment(np.array([[1.0, 0.0], [1.0, 1.0]], dtype=dtype))

        min_ts = np.array([0.0, 0.0, 0.5], dtype=dtype)
        max_ts = np.array([2.0, 0.5, 2.0], dtype=dtype)

        result = tf.ray_cast(rays, seg, config=(min_ts, max_ts))
        assert result.shape == (3,)
        assert abs(result[0] - 1.0) < 1e-5   # hit: max_t=2 covers t=1
        assert np.isnan(result[1])             # miss: max_t=0.5 < t=1
        assert abs(result[2] - 1.0) < 1e-5   # hit: min_t=0.5 < t=1


# ---------------------------------------------------------------------------
# form x prim: distance, distance2, intersects
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", [np.float32, np.float64])
class TestFormPrimBatch:

    def test_form_distance_single(self, dtype):
        """Form x single point -> scalar."""
        mesh = _make_mesh_3d(dtype)
        pt = tf.Point(np.array([0.5, 0.5, 1.0], dtype=dtype))

        d = tf.distance(mesh, pt)
        assert isinstance(d, float)
        assert abs(d - 1.0) < 1e-5

    def test_form_distance_batch(self, dtype):
        """Form x batch points -> ndarray."""
        mesh = _make_mesh_3d(dtype)
        pts = tf.Point(np.array([
            [0.5, 0.5, 1.0],
            [0.5, 0.5, 2.0],
            [0.5, 0.5, 0.0],  # on mesh
        ], dtype=dtype))

        d = tf.distance(mesh, pts)
        assert isinstance(d, np.ndarray)
        assert d.shape == (3,)
        assert abs(d[0] - 1.0) < 1e-5
        assert abs(d[1] - 2.0) < 1e-5
        assert abs(d[2]) < 1e-5

    def test_form_distance2_batch(self, dtype):
        """Form x batch points -> ndarray (squared distance)."""
        mesh = _make_mesh_3d(dtype)
        pts = tf.Point(np.array([
            [0.5, 0.5, 1.0],
            [0.5, 0.5, 2.0],
        ], dtype=dtype))

        d2 = tf.distance2(mesh, pts)
        assert isinstance(d2, np.ndarray)
        assert d2.shape == (2,)
        assert abs(d2[0] - 1.0) < 1e-5
        assert abs(d2[1] - 4.0) < 1e-5

    def test_form_distance_swap(self, dtype):
        """distance(prim, form) should equal distance(form, prim)."""
        mesh = _make_mesh_3d(dtype)
        pts = tf.Point(np.array([
            [0.5, 0.5, 1.0],
            [0.5, 0.5, 2.0],
        ], dtype=dtype))

        d1 = tf.distance(mesh, pts)
        d2 = tf.distance(pts, mesh)
        np.testing.assert_allclose(d1, d2, atol=1e-5)

    def test_form_intersects_single(self, dtype):
        """Form x single point -> bool."""
        mesh = _make_mesh_3d(dtype)
        pt_on = tf.Point(np.array([0.5, 0.5, 0.0], dtype=dtype))
        pt_off = tf.Point(np.array([0.5, 0.5, 1.0], dtype=dtype))

        assert tf.intersects(mesh, pt_on)
        assert not tf.intersects(mesh, pt_off)

    def test_form_intersects_batch(self, dtype):
        """Form x batch points -> ndarray."""
        mesh = _make_mesh_3d(dtype)
        pts = tf.Point(np.array([
            [0.5, 0.5, 0.0],   # on mesh
            [0.5, 0.5, 1.0],   # off mesh
            [0.0, 0.0, 0.0],   # on vertex
        ], dtype=dtype))

        result = tf.intersects(mesh, pts)
        assert isinstance(result, np.ndarray)
        assert result.shape == (3,)
        assert result[0] == 1
        assert result[1] == 0
        assert result[2] == 1

    def test_form_intersects_swap(self, dtype):
        """intersects(prim, form) should equal intersects(form, prim)."""
        mesh = _make_mesh_3d(dtype)
        pts = tf.Point(np.array([
            [0.5, 0.5, 0.0],
            [0.5, 0.5, 1.0],
        ], dtype=dtype))

        r1 = tf.intersects(mesh, pts)
        r2 = tf.intersects(pts, mesh)
        np.testing.assert_array_equal(r1, r2)


# ---------------------------------------------------------------------------
# form x prim: ray_cast (batch rays against form)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", [np.float32, np.float64])
class TestFormRayCastBatch:

    def test_form_ray_cast_single(self, dtype):
        """Single ray x form -> (idx, t) or None."""
        mesh = _make_mesh_3d(dtype)
        ray = tf.Ray(
            origin=np.array([0.5, 0.5, 2.0], dtype=dtype),
            direction=np.array([0.0, 0.0, -1.0], dtype=dtype))

        result = tf.ray_cast(ray, mesh)
        assert result is not None
        idx, t = result
        assert isinstance(idx, (int, np.integer))
        assert abs(t - 2.0) < 1e-5

    def test_form_ray_cast_batch(self, dtype):
        """Batch rays x form -> (ids, ts) tuple."""
        mesh = _make_mesh_3d(dtype)
        rays = tf.Ray(np.array([
            [[0.5, 0.5, 2.0], [0.0, 0.0, -1.0]],  # hits mesh at t=2
            [[5.0, 5.0, 2.0], [0.0, 0.0, -1.0]],  # misses mesh
            [[0.25, 0.25, 1.0], [0.0, 0.0, -1.0]], # hits mesh at t=1
        ], dtype=dtype))

        ids, ts = tf.ray_cast(rays, mesh)
        assert ids.shape == (3,)
        assert ts.shape == (3,)
        # ray 0: hit at t=2
        assert ids[0] >= 0
        assert abs(ts[0] - 2.0) < 1e-5
        # ray 1: miss -> -1 / NaN
        assert np.isnan(ts[1])
        # ray 2: hit at t=1
        assert ids[2] >= 0
        assert abs(ts[2] - 1.0) < 1e-5

    def test_form_ray_cast_scalar_config(self, dtype):
        """Scalar config tuple works for batch rays x form."""
        mesh = _make_mesh_3d(dtype)
        rays = tf.Ray(np.array([
            [[0.5, 0.5, 2.0], [0.0, 0.0, -1.0]],  # hits at t=2
            [[0.25, 0.25, 1.0], [0.0, 0.0, -1.0]], # hits at t=1
        ], dtype=dtype))

        # max_t=0.5 is too short for either hit
        ids, ts = tf.ray_cast(rays, mesh, config=(0.0, 0.5))
        assert np.isnan(ts[0])
        assert np.isnan(ts[1])

    def test_form_ray_cast_per_ray_config(self, dtype):
        """Per-ray config arrays for batch rays x form."""
        mesh = _make_mesh_3d(dtype)
        rays = tf.Ray(np.array([
            [[0.5, 0.5, 2.0], [0.0, 0.0, -1.0]],  # hits at t=2
            [[0.5, 0.5, 2.0], [0.0, 0.0, -1.0]],  # hits at t=2
            [[0.25, 0.25, 1.0], [0.0, 0.0, -1.0]], # hits at t=1
        ], dtype=dtype))

        min_ts = np.array([0.0, 0.0, 0.0], dtype=dtype)
        max_ts = np.array([3.0, 1.0, 3.0], dtype=dtype)

        ids, ts = tf.ray_cast(rays, mesh, config=(min_ts, max_ts))
        # ray 0: max_t=3 covers t=2 → hit
        assert ids[0] >= 0
        assert abs(ts[0] - 2.0) < 1e-5
        # ray 1: max_t=1 < t=2 → miss
        assert np.isnan(ts[1])
        # ray 2: max_t=3 covers t=1 → hit
        assert ids[2] >= 0
        assert abs(ts[2] - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# form x prim: neighbor_search (1-nn batch and k-nn batch)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", [np.float32, np.float64])
class TestFormNeighborSearchBatch:

    def test_1nn_single(self, dtype):
        """Single query -> (idx, dist2, point)."""
        cloud = _make_point_cloud_3d(dtype)
        query = tf.Point(np.array([0.1, 0.1, 0.0], dtype=dtype))

        result = tf.neighbor_search(cloud, query)
        assert result is not None
        idx, dist2, pt = result
        assert isinstance(idx, (int, np.integer))
        assert idx == 0  # closest to [0,0,0]
        assert pt.shape == (3,)

    def test_1nn_batch(self, dtype):
        """Batch query -> (ids[N], dists[N], pts[N,D])."""
        cloud = _make_point_cloud_3d(dtype)
        queries = tf.Point(np.array([
            [0.1, 0.0, 0.0],  # closest to [0,0,0]
            [0.9, 0.0, 0.0],  # closest to [1,0,0]
            [0.0, 0.9, 0.0],  # closest to [0,1,0]
        ], dtype=dtype))

        result = tf.neighbor_search(cloud, queries)
        ids, dists, pts = result
        assert ids.shape == (3,)
        assert dists.shape == (3,)
        assert pts.shape == (3, 3)
        assert ids[0] == 0
        assert ids[1] == 1
        assert ids[2] == 2

    def test_1nn_batch_with_radius(self, dtype):
        """Batch 1-nn with radius: some queries may not find neighbors."""
        cloud = _make_point_cloud_3d(dtype)
        queries = tf.Point(np.array([
            [0.1, 0.0, 0.0],   # within radius of [0,0,0]
            [10.0, 10.0, 10.0], # far away
        ], dtype=dtype))

        result = tf.neighbor_search(cloud, queries, radius=0.5)
        ids, dists, pts = result
        assert ids.shape == (2,)
        assert ids[0] == 0
        # id[1] should be -1 (not found)
        assert ids[1] == -1 or ids[1] == np.iinfo(ids.dtype).max

    def test_knn_single(self, dtype):
        """Single k-nn query -> list of tuples."""
        cloud = _make_point_cloud_3d(dtype)
        query = tf.Point(np.array([0.0, 0.0, 0.0], dtype=dtype))

        result = tf.neighbor_search(cloud, query, k=2)
        assert isinstance(result, list)
        assert len(result) <= 2
        # First result should be exact match (idx=0, dist2≈0)
        idx0, dist0, pt0 = result[0]
        assert idx0 == 0
        assert abs(dist0) < 1e-5

    def test_knn_batch(self, dtype):
        """Batch k-nn query -> (ids[N,K], dists[N,K], pts[N,K,D], counts[N])."""
        cloud = _make_point_cloud_3d(dtype)
        queries = tf.Point(np.array([
            [0.0, 0.0, 0.0],  # on point 0
            [0.5, 0.0, 0.0],  # between point 0 and 1
        ], dtype=dtype))

        result = tf.neighbor_search(cloud, queries, k=3)
        ids, dists, pts, counts = result
        assert ids.shape == (2, 3)
        assert dists.shape == (2, 3)
        assert pts.shape == (2, 3, 3)
        assert counts.shape == (2,)
        # Both queries should find at least 3 neighbors
        assert counts[0] == 3
        assert counts[1] == 3

    def test_knn_batch_with_radius(self, dtype):
        """Batch k-nn with radius limiting results."""
        cloud = _make_point_cloud_3d(dtype)
        queries = tf.Point(np.array([
            [0.0, 0.0, 0.0],   # right on a point, very small radius
            [0.5, 0.0, 0.0],   # between points
        ], dtype=dtype))

        result = tf.neighbor_search(cloud, queries, k=3, radius=0.01)
        ids, dists, pts, counts = result
        assert counts[0] >= 1  # should find at least the exact point
        # Unfilled slots should have id == -1
        for i in range(2):
            for j in range(counts[i], 3):
                assert ids[i, j] == -1 or ids[i, j] == np.iinfo(ids.dtype).max

    def test_mesh_1nn_batch(self, dtype):
        """Batch 1-nn against mesh."""
        mesh = _make_mesh_3d(dtype)
        queries = tf.Point(np.array([
            [0.5, 0.5, 1.0],
            [0.5, 0.5, 2.0],
        ], dtype=dtype))

        result = tf.neighbor_search(mesh, queries)
        ids, dists, pts = result
        assert ids.shape == (2,)
        assert dists.shape == (2,)
        assert pts.shape == (2, 3)
        # points project onto mesh at z=0
        assert abs(pts[0, 2]) < 1e-5
        assert abs(pts[1, 2]) < 1e-5


# ---------------------------------------------------------------------------
# edge cases
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", [np.float32, np.float64])
class TestBatchEdgeCases:

    def test_batch_size_one(self, dtype):
        """Batch of size 1 should still return ndarray."""
        pts = tf.Point(np.array([[0.5, 0.5, 0.0]], dtype=dtype))
        seg = tf.Segment(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=dtype))

        d = tf.distance(pts, seg)
        assert isinstance(d, np.ndarray)
        assert d.shape == (1,)

    def test_2d_batch(self, dtype):
        """Batch operations in 2D."""
        pts = tf.Point(np.array([
            [0.5, 0.5],
            [2.0, 2.0],
        ], dtype=dtype))
        seg = tf.Segment(np.array([[0.0, 0.0], [1.0, 0.0]], dtype=dtype))

        d = tf.distance(pts, seg)
        assert isinstance(d, np.ndarray)
        assert d.shape == (2,)
        assert abs(d[0] - 0.5) < 1e-5
        assert abs(d[1] - np.sqrt(5.0)) < 1e-5  # (2,2) to closest pt (1,0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
