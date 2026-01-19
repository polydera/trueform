"""
Test closest_metric_point_pair functionality

Copyright (c) 2025 Žiga Sajovic, XLAB
"""

import sys
import pytest
import numpy as np
import trueform as tf


def test_point_polygon_2d_inside():
    """Test closest metric point pair with point inside polygon in 2D"""
    square = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0]
    ], dtype=np.float32)

    pt_inside = tf.Point([0.5, 0.5])
    poly = tf.Polygon(square)

    dist2, p0, p1 = tf.closest_metric_point_pair(pt_inside, poly)
    assert dist2 == 0.0, f"Expected 0.0, got {dist2}"


def test_point_polygon_2d_outside():
    """Test closest metric point pair with point outside polygon in 2D"""
    square = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0]
    ], dtype=np.float32)

    pt_outside = tf.Point([2.0, 0.5])
    poly = tf.Polygon(square)

    dist2, p0, p1 = tf.closest_metric_point_pair(pt_outside, poly)
    assert np.isclose(dist2, 1.0), f"Expected 1.0, got {dist2}"
    assert np.allclose(p1, [1.0, 0.5], atol=1e-5), f"Expected [1.0, 0.5], got {p1}"


def test_point_polygon_3d_inside():
    """Test closest metric point pair with point inside polygon in 3D"""
    triangle = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 1.0, 0.0]
    ], dtype=np.float64)

    pt_inside = tf.Point([0.5, 0.3, 0.0])
    poly = tf.Polygon(triangle)

    dist2, p0, p1 = tf.closest_metric_point_pair(pt_inside, poly)
    assert np.isclose(0, dist2)


def test_point_polygon_3d_above():
    """Test closest metric point pair with point above polygon in 3D"""
    triangle = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 1.0, 0.0]
    ], dtype=np.float64)

    pt_above = tf.Point([0.5, 0.3, 2.0])
    poly = tf.Polygon(triangle)

    dist2, p0, p1 = tf.closest_metric_point_pair(pt_above, poly)
    assert np.isclose(dist2, 4.0), f"Expected 4.0, got {dist2}"
    assert np.allclose(p1, [0.5, 0.3, 0.0], atol=1e-5), f"Expected [0.5, 0.3, 0.0], got {p1}"


def test_polygon_polygon_2d_separate():
    """Test closest metric point pair between two separate polygons in 2D"""
    square1 = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0]
    ], dtype=np.float32)

    square2 = np.array([
        [2.0, 0.0],
        [3.0, 0.0],
        [3.0, 1.0],
        [2.0, 1.0]
    ], dtype=np.float32)

    poly1 = tf.Polygon(square1)
    poly2 = tf.Polygon(square2)

    dist2, p0, p1 = tf.closest_metric_point_pair(poly1, poly2)
    assert np.isclose(dist2, 1.0), f"Expected 1.0, got {dist2}"


def test_polygon_polygon_2d_overlapping():
    """Test closest metric point pair between two overlapping polygons in 2D"""
    square1 = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0]
    ], dtype=np.float32)

    square3 = np.array([
        [0.5, 0.5],
        [1.5, 0.5],
        [1.5, 1.5],
        [0.5, 1.5]
    ], dtype=np.float32)

    poly1 = tf.Polygon(square1)
    poly3 = tf.Polygon(square3)

    dist2, p0, p1 = tf.closest_metric_point_pair(poly1, poly3)
    assert dist2 == 0.0, f"Expected 0.0, got {dist2}"


def test_polygon_polygon_3d():
    """Test closest metric point pair between two polygons in 3D"""
    triangle1 = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 1.0, 0.0]
    ], dtype=np.float64)

    triangle2 = np.array([
        [0.0, 0.0, 2.0],
        [1.0, 0.0, 2.0],
        [0.5, 1.0, 2.0]
    ], dtype=np.float64)

    poly1 = tf.Polygon(triangle1)
    poly2 = tf.Polygon(triangle2)

    dist2, p0, p1 = tf.closest_metric_point_pair(poly1, poly2)
    assert np.isclose(dist2, 4.0), f"Expected 4.0, got {dist2}"


def test_segment_polygon_2d_intersecting():
    """Test closest metric point pair with segment intersecting polygon in 2D"""
    square = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0]
    ], dtype=np.float32)

    seg_intersect = tf.Segment([[0.5, -0.5], [0.5, 1.5]])
    poly = tf.Polygon(square)

    dist2, p0, p1 = tf.closest_metric_point_pair(seg_intersect, poly)
    assert dist2 == 0.0, f"Expected 0.0, got {dist2}"


def test_segment_polygon_2d_outside():
    """Test closest metric point pair with segment outside polygon in 2D"""
    square = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0]
    ], dtype=np.float32)

    seg_outside = tf.Segment([[2.0, 0.0], [3.0, 0.0]])
    poly = tf.Polygon(square)

    dist2, p0, p1 = tf.closest_metric_point_pair(seg_outside, poly)
    assert np.isclose(dist2, 1.0), f"Expected 1.0, got {dist2}"


def test_ray_polygon_3d_hitting():
    """Test closest metric point pair with ray hitting polygon in 3D"""
    triangle = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 1.0, 0.0]
    ], dtype=np.float32)

    ray_hit = tf.Ray(origin=[0.5, 0.3, 2.0], direction=[0.0, 0.0, -1.0])
    poly = tf.Polygon(triangle)

    dist2, p0, p1 = tf.closest_metric_point_pair(ray_hit, poly)
    assert dist2 == 0.0, f"Expected 0.0, got {dist2}"


def test_ray_polygon_3d_missing():
    """Test closest metric point pair with ray missing polygon in 3D"""
    triangle = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 1.0, 0.0]
    ], dtype=np.float32)

    ray_miss = tf.Ray(origin=[0.5, 0.3, 2.0], direction=[0.0, 0.0, 1.0])
    poly = tf.Polygon(triangle)

    dist2, p0, p1 = tf.closest_metric_point_pair(ray_miss, poly)
    assert np.isclose(dist2, 4.0), f"Expected 4.0, got {dist2}"


def test_line_polygon_2d_intersecting():
    """Test closest metric point pair with line intersecting polygon in 2D"""
    square = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0]
    ], dtype=np.float32)

    line_intersect = tf.Line(origin=[0.5, -1.0], direction=[0.0, 1.0])
    poly = tf.Polygon(square)

    dist2, p0, p1 = tf.closest_metric_point_pair(line_intersect, poly)
    assert dist2 == 0.0, f"Expected 0.0, got {dist2}"


def test_line_polygon_2d_parallel():
    """Test closest metric point pair with line parallel to polygon in 2D"""
    square = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0]
    ], dtype=np.float32)

    line_parallel = tf.Line(origin=[2.0, 0.0], direction=[0.0, 1.0])
    poly = tf.Polygon(square)

    dist2, p0, p1 = tf.closest_metric_point_pair(line_parallel, poly)
    assert np.isclose(dist2, 1.0), f"Expected 1.0, got {dist2}"


def test_dimension_mismatch_point():
    """Test that dimension mismatch raises an error for points"""
    pt_2d = tf.Point([0.0, 0.0])
    pt_3d = tf.Point([0.0, 0.0, 0.0])

    with pytest.raises(ValueError) as exc_info:
        dist2, p0, p1 = tf.closest_metric_point_pair(pt_2d, pt_3d)
    assert "Dimension mismatch" in str(exc_info.value)
    assert "2D" in str(exc_info.value)
    assert "3D" in str(exc_info.value)


def test_dimension_mismatch_polygon():
    """Test that dimension mismatch raises an error for polygon"""
    triangle_2d = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, 1.0]
    ], dtype=np.float32)
    poly_2d = tf.Polygon(triangle_2d)
    pt_3d = tf.Point([0.0, 0.0, 0.0])

    with pytest.raises(ValueError) as exc_info:
        dist2, p0, p1 = tf.closest_metric_point_pair(pt_3d, poly_2d)
    assert "Dimension mismatch" in str(exc_info.value)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_point_plane_on_plane(dtype):
    """Test closest metric point pair with point on plane"""
    plane = tf.Plane(np.array([0.0, 0.0, 1.0, 0.0], dtype=dtype))
    pt_on = tf.Point(np.array([1.0, 2.0, 0.0], dtype=dtype))

    dist2, p0, p1 = tf.closest_metric_point_pair(pt_on, plane)
    assert abs(dist2) < 1e-5, f"Distance should be 0 ({dtype})"
    assert np.allclose(p0, pt_on.data), f"p0 should be the original point ({dtype})"
    assert np.allclose(p1, pt_on.data), f"p1 should be on plane ({dtype})"


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_point_plane_above(dtype):
    """Test closest metric point pair with point above plane"""
    plane = tf.Plane(np.array([0.0, 0.0, 1.0, 0.0], dtype=dtype))
    pt_above = tf.Point(np.array([1.0, 2.0, 5.0], dtype=dtype))

    dist2, p0, p1 = tf.closest_metric_point_pair(pt_above, plane)
    assert abs(dist2 - 25.0) < 1e-4, f"Distance2 should be 25 ({dtype})"
    assert np.allclose(p0, pt_above.data), f"p0 should be the original point ({dtype})"
    assert np.allclose(p1, [1.0, 2.0, 0.0]), f"p1 should be projected on plane ({dtype})"


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_plane_point_swap(dtype):
    """Test that swapping plane and point works"""
    plane = tf.Plane(np.array([0.0, 0.0, 1.0, 0.0], dtype=dtype))
    pt_above = tf.Point(np.array([1.0, 2.0, 5.0], dtype=dtype))

    dist2_swap, p0_swap, p1_swap = tf.closest_metric_point_pair(plane, pt_above)
    assert abs(dist2_swap - 25.0) < 1e-4, f"Swapped distance should match ({dtype})"
    assert np.allclose(p0_swap, [1.0, 2.0, 0.0]), f"p0 should be on plane after swap ({dtype})"
    assert np.allclose(p1_swap, pt_above.data), f"p1 should be the point after swap ({dtype})"


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_segment_plane_parallel(dtype):
    """Test closest metric point pair with segment parallel to plane"""
    plane = tf.Plane(np.array([0.0, 0.0, 1.0, 0.0], dtype=dtype))
    seg = tf.Segment(np.array([[0.0, 0.0, 3.0], [1.0, 0.0, 3.0]], dtype=dtype))

    dist2, p0, p1 = tf.closest_metric_point_pair(seg, plane)
    assert abs(dist2 - 9.0) < 1e-4, f"Distance2 should be 9 ({dtype})"


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_segment_plane_crossing(dtype):
    """Test closest metric point pair with segment crossing plane"""
    plane = tf.Plane(np.array([0.0, 0.0, 1.0, 0.0], dtype=dtype))
    seg_cross = tf.Segment(np.array([[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]], dtype=dtype))

    dist2, p0, p1 = tf.closest_metric_point_pair(seg_cross, plane)
    assert abs(dist2) < 1e-5, f"Distance should be 0 for intersecting segment ({dtype})"


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_ray_plane_toward(dtype):
    """Test closest metric point pair with ray pointing toward plane"""
    plane = tf.Plane(np.array([0.0, 0.0, 1.0, 0.0], dtype=dtype))
    ray = tf.Ray(
        origin=np.array([0.0, 0.0, 5.0], dtype=dtype),
        direction=np.array([0.0, 0.0, -1.0], dtype=dtype)
    )

    dist2, p0, p1 = tf.closest_metric_point_pair(ray, plane)
    assert abs(dist2) < 1e-5, f"Distance should be 0 for ray hitting plane ({dtype})"


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_ray_plane_parallel(dtype):
    """Test closest metric point pair with ray parallel to plane"""
    plane = tf.Plane(np.array([0.0, 0.0, 1.0, 0.0], dtype=dtype))
    ray_parallel = tf.Ray(
        origin=np.array([0.0, 0.0, 5.0], dtype=dtype),
        direction=np.array([1.0, 0.0, 0.0], dtype=dtype)
    )

    dist2, p0, p1 = tf.closest_metric_point_pair(ray_parallel, plane)
    assert abs(dist2 - 25.0) < 1e-4, f"Distance2 should be 25 ({dtype})"


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_line_plane_intersecting(dtype):
    """Test closest metric point pair with line intersecting plane"""
    plane = tf.Plane(np.array([0.0, 0.0, 1.0, 0.0], dtype=dtype))
    line = tf.Line(
        origin=np.array([0.0, 0.0, 5.0], dtype=dtype),
        direction=np.array([0.0, 0.0, 1.0], dtype=dtype)
    )

    dist2, p0, p1 = tf.closest_metric_point_pair(line, plane)
    assert abs(dist2) < 1e-5, f"Distance should be 0 for line intersecting plane ({dtype})"


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_line_plane_parallel(dtype):
    """Test closest metric point pair with line parallel to plane"""
    plane = tf.Plane(np.array([0.0, 0.0, 1.0, 0.0], dtype=dtype))
    line_parallel = tf.Line(
        origin=np.array([0.0, 0.0, 3.0], dtype=dtype),
        direction=np.array([1.0, 0.0, 0.0], dtype=dtype)
    )

    dist2, p0, p1 = tf.closest_metric_point_pair(line_parallel, plane)
    assert abs(dist2 - 9.0) < 1e-4, f"Distance2 should be 9 ({dtype})"


# ==============================================================================
# Full Matrix Tests - Point combinations
# ==============================================================================

@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("dims", [2, 3])
def test_point_point_separated(dtype, dims):
    """Test Point-Point with known separation."""
    # Points separated by 3 units along x-axis
    p0 = tf.Point(np.array([0.0] + [0.0] * (dims - 1), dtype=dtype))
    p1 = tf.Point(np.array([3.0] + [0.0] * (dims - 1), dtype=dtype))

    dist2, c0, c1 = tf.closest_metric_point_pair(p0, p1)

    assert np.isclose(dist2, 9.0), f"Expected dist2=9.0, got {dist2}"
    assert np.allclose(c0, p0.data), "c0 should be p0"
    assert np.allclose(c1, p1.data), "c1 should be p1"
    assert np.isclose(tf.distance2(p0, p1), dist2), "distance2 should match"


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("dims", [2, 3])
def test_point_segment_perpendicular(dtype, dims):
    """Test Point-Segment with point perpendicular to segment midpoint."""
    # Segment along x-axis from (0,0) to (4,0), point at (2, 3)
    # Closest point on segment is (2, 0), distance = 3
    seg_pts = np.zeros((2, dims), dtype=dtype)
    seg_pts[0, 0] = 0.0
    seg_pts[1, 0] = 4.0
    seg = tf.Segment(seg_pts)

    pt_coords = np.zeros(dims, dtype=dtype)
    pt_coords[0] = 2.0
    pt_coords[1] = 3.0
    pt = tf.Point(pt_coords)

    dist2, c0, c1 = tf.closest_metric_point_pair(pt, seg)

    expected_closest = np.zeros(dims, dtype=dtype)
    expected_closest[0] = 2.0

    assert np.isclose(dist2, 9.0), f"Expected dist2=9.0, got {dist2}"
    assert np.allclose(c0, pt.data), "c0 should be the point"
    assert np.allclose(c1, expected_closest, atol=1e-5), f"c1 should be (2,0,...), got {c1}"
    assert np.isclose(tf.distance2(pt, seg), dist2), "distance2 should match"


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("dims", [2, 3])
def test_point_segment_endpoint(dtype, dims):
    """Test Point-Segment with closest point at segment endpoint."""
    # Segment from (0,0) to (4,0), point at (-2, 0)
    # Closest point is endpoint (0, 0), distance = 2
    seg_pts = np.zeros((2, dims), dtype=dtype)
    seg_pts[0, 0] = 0.0
    seg_pts[1, 0] = 4.0
    seg = tf.Segment(seg_pts)

    pt_coords = np.zeros(dims, dtype=dtype)
    pt_coords[0] = -2.0
    pt = tf.Point(pt_coords)

    dist2, c0, c1 = tf.closest_metric_point_pair(pt, seg)

    expected_closest = np.zeros(dims, dtype=dtype)

    assert np.isclose(dist2, 4.0), f"Expected dist2=4.0, got {dist2}"
    assert np.allclose(c1, expected_closest, atol=1e-5), f"c1 should be (0,0,...), got {c1}"
    assert np.isclose(tf.distance2(pt, seg), dist2), "distance2 should match"


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("dims", [2, 3])
def test_point_ray_perpendicular(dtype, dims):
    """Test Point-Ray with point perpendicular to ray."""
    # Ray from (0,0) along x-axis, point at (3, 4)
    # Closest point on ray is (3, 0), distance = 4
    origin = np.zeros(dims, dtype=dtype)
    direction = np.zeros(dims, dtype=dtype)
    direction[0] = 1.0
    ray = tf.Ray(origin=origin, direction=direction)

    pt_coords = np.zeros(dims, dtype=dtype)
    pt_coords[0] = 3.0
    pt_coords[1] = 4.0
    pt = tf.Point(pt_coords)

    dist2, c0, c1 = tf.closest_metric_point_pair(pt, ray)

    expected_closest = np.zeros(dims, dtype=dtype)
    expected_closest[0] = 3.0

    assert np.isclose(dist2, 16.0), f"Expected dist2=16.0, got {dist2}"
    assert np.allclose(c1, expected_closest, atol=1e-5), f"c1 should be (3,0,...), got {c1}"
    assert np.isclose(tf.distance2(pt, ray), dist2), "distance2 should match"


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("dims", [2, 3])
def test_point_ray_behind_origin(dtype, dims):
    """Test Point-Ray with point behind ray origin."""
    # Ray from (0,0) along positive x, point at (-3, 4)
    # Closest point is ray origin (0, 0), distance = 5
    origin = np.zeros(dims, dtype=dtype)
    direction = np.zeros(dims, dtype=dtype)
    direction[0] = 1.0
    ray = tf.Ray(origin=origin, direction=direction)

    pt_coords = np.zeros(dims, dtype=dtype)
    pt_coords[0] = -3.0
    pt_coords[1] = 4.0
    pt = tf.Point(pt_coords)

    dist2, c0, c1 = tf.closest_metric_point_pair(pt, ray)

    assert np.isclose(dist2, 25.0), f"Expected dist2=25.0, got {dist2}"
    assert np.allclose(c1, origin, atol=1e-5), f"c1 should be ray origin, got {c1}"
    assert np.isclose(tf.distance2(pt, ray), dist2), "distance2 should match"


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("dims", [2, 3])
def test_point_line_perpendicular(dtype, dims):
    """Test Point-Line with perpendicular projection."""
    # Line through (0,0) along x-axis, point at (5, 12)
    # Closest point on line is (5, 0), distance = 12
    origin = np.zeros(dims, dtype=dtype)
    direction = np.zeros(dims, dtype=dtype)
    direction[0] = 1.0
    line = tf.Line(origin=origin, direction=direction)

    pt_coords = np.zeros(dims, dtype=dtype)
    pt_coords[0] = 5.0
    pt_coords[1] = 12.0
    pt = tf.Point(pt_coords)

    dist2, c0, c1 = tf.closest_metric_point_pair(pt, line)

    expected_closest = np.zeros(dims, dtype=dtype)
    expected_closest[0] = 5.0

    assert np.isclose(dist2, 144.0), f"Expected dist2=144.0, got {dist2}"
    assert np.allclose(c1, expected_closest, atol=1e-5), f"c1 should be (5,0,...), got {c1}"
    assert np.isclose(tf.distance2(pt, line), dist2), "distance2 should match"


# ==============================================================================
# Full Matrix Tests - Segment combinations
# ==============================================================================

@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("dims", [2, 3])
def test_segment_segment_parallel(dtype, dims):
    """Test Segment-Segment parallel segments."""
    # Segment 1: (0,0) to (4,0)
    # Segment 2: (0,3) to (4,3)
    # Closest distance = 3 (any point along the parallel sections)
    seg1_pts = np.zeros((2, dims), dtype=dtype)
    seg1_pts[0, 0] = 0.0
    seg1_pts[1, 0] = 4.0
    seg1 = tf.Segment(seg1_pts)

    seg2_pts = np.zeros((2, dims), dtype=dtype)
    seg2_pts[0, 0] = 0.0
    seg2_pts[0, 1] = 3.0
    seg2_pts[1, 0] = 4.0
    seg2_pts[1, 1] = 3.0
    seg2 = tf.Segment(seg2_pts)

    dist2, c0, c1 = tf.closest_metric_point_pair(seg1, seg2)

    assert np.isclose(dist2, 9.0), f"Expected dist2=9.0, got {dist2}"
    # Verify closest points are at same x-coordinate
    assert np.isclose(c0[0], c1[0], atol=1e-5), "x-coordinates should match"
    assert np.isclose(tf.distance2(seg1, seg2), dist2), "distance2 should match"


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("dims", [2, 3])
def test_segment_segment_endpoint_to_midpoint(dtype, dims):
    """Test Segment-Segment with endpoint closest to midpoint."""
    # Segment 1: (0,0) to (4,0)
    # Segment 2: (2,3) to (2,6) - vertical segment above midpoint of seg1
    # Closest: (2,0) on seg1 to (2,3) on seg2, distance = 3
    seg1_pts = np.zeros((2, dims), dtype=dtype)
    seg1_pts[0, 0] = 0.0
    seg1_pts[1, 0] = 4.0
    seg1 = tf.Segment(seg1_pts)

    seg2_pts = np.zeros((2, dims), dtype=dtype)
    seg2_pts[0, 0] = 2.0
    seg2_pts[0, 1] = 3.0
    seg2_pts[1, 0] = 2.0
    seg2_pts[1, 1] = 6.0
    seg2 = tf.Segment(seg2_pts)

    dist2, c0, c1 = tf.closest_metric_point_pair(seg1, seg2)

    expected_c0 = np.zeros(dims, dtype=dtype)
    expected_c0[0] = 2.0
    expected_c1 = np.zeros(dims, dtype=dtype)
    expected_c1[0] = 2.0
    expected_c1[1] = 3.0

    assert np.isclose(dist2, 9.0), f"Expected dist2=9.0, got {dist2}"
    assert np.allclose(c0, expected_c0, atol=1e-5), f"c0 should be (2,0,...), got {c0}"
    assert np.allclose(c1, expected_c1, atol=1e-5), f"c1 should be (2,3,...), got {c1}"
    assert np.isclose(tf.distance2(seg1, seg2), dist2), "distance2 should match"


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("dims", [2, 3])
def test_segment_ray_separated(dtype, dims):
    """Test Segment-Ray with clear separation."""
    # Segment: (0,0) to (2,0)
    # Ray: from (4,3) pointing in +y direction
    # Closest: (2,0) on segment to (4,3) on ray, distance = sqrt(4+9) = sqrt(13)
    seg_pts = np.zeros((2, dims), dtype=dtype)
    seg_pts[0, 0] = 0.0
    seg_pts[1, 0] = 2.0
    seg = tf.Segment(seg_pts)

    ray_origin = np.zeros(dims, dtype=dtype)
    ray_origin[0] = 4.0
    ray_origin[1] = 3.0
    ray_dir = np.zeros(dims, dtype=dtype)
    ray_dir[1] = 1.0
    ray = tf.Ray(origin=ray_origin, direction=ray_dir)

    dist2, c0, c1 = tf.closest_metric_point_pair(seg, ray)

    expected_c0 = np.zeros(dims, dtype=dtype)
    expected_c0[0] = 2.0
    expected_c1 = ray_origin.copy()

    assert np.isclose(dist2, 13.0), f"Expected dist2=13.0, got {dist2}"
    assert np.allclose(c0, expected_c0, atol=1e-5), f"c0 should be (2,0,...), got {c0}"
    assert np.allclose(c1, expected_c1, atol=1e-5), f"c1 should be (4,3,...), got {c1}"
    assert np.isclose(tf.distance2(seg, ray), dist2), "distance2 should match"


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("dims", [2, 3])
def test_segment_line_perpendicular(dtype, dims):
    """Test Segment-Line perpendicular configuration."""
    # Segment: (0,5) to (4,5) - horizontal at y=5
    # Line: through origin, along y-axis
    # Closest: (0,5) on segment to (0,5) on line, distance = 0
    seg_pts = np.zeros((2, dims), dtype=dtype)
    seg_pts[0, 0] = 0.0
    seg_pts[0, 1] = 5.0
    seg_pts[1, 0] = 4.0
    seg_pts[1, 1] = 5.0
    seg = tf.Segment(seg_pts)

    line_origin = np.zeros(dims, dtype=dtype)
    line_dir = np.zeros(dims, dtype=dtype)
    line_dir[1] = 1.0
    line = tf.Line(origin=line_origin, direction=line_dir)

    dist2, c0, c1 = tf.closest_metric_point_pair(seg, line)

    expected_c0 = np.zeros(dims, dtype=dtype)
    expected_c0[1] = 5.0
    expected_c1 = expected_c0.copy()

    assert np.isclose(dist2, 0.0, atol=1e-5), f"Expected dist2=0.0, got {dist2}"
    assert np.allclose(c0, expected_c0, atol=1e-5), f"c0 should be (0,5,...), got {c0}"
    assert np.allclose(c1, expected_c1, atol=1e-5), f"c1 should be (0,5,...), got {c1}"
    assert np.isclose(tf.distance2(seg, line), dist2, atol=1e-5), "distance2 should match"


# ==============================================================================
# Full Matrix Tests - Ray combinations
# ==============================================================================

@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("dims", [2, 3])
def test_ray_ray_diverging(dtype, dims):
    """Test Ray-Ray diverging rays."""
    # Ray 1: from (0,0) along +x
    # Ray 2: from (0,4) along +x
    # Parallel rays, distance = 4, closest at origins
    ray1_origin = np.zeros(dims, dtype=dtype)
    ray1_dir = np.zeros(dims, dtype=dtype)
    ray1_dir[0] = 1.0
    ray1 = tf.Ray(origin=ray1_origin, direction=ray1_dir)

    ray2_origin = np.zeros(dims, dtype=dtype)
    ray2_origin[1] = 4.0
    ray2_dir = np.zeros(dims, dtype=dtype)
    ray2_dir[0] = 1.0
    ray2 = tf.Ray(origin=ray2_origin, direction=ray2_dir)

    dist2, c0, c1 = tf.closest_metric_point_pair(ray1, ray2)

    assert np.isclose(dist2, 16.0), f"Expected dist2=16.0, got {dist2}"
    # Closest points should have same x-coordinate
    assert np.isclose(c0[0], c1[0], atol=1e-5), "x-coordinates should match"
    assert np.isclose(tf.distance2(ray1, ray2), dist2), "distance2 should match"


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("dims", [2, 3])
def test_ray_line_perpendicular(dtype, dims):
    """Test Ray-Line perpendicular."""
    # Ray: from (3,0) along +y
    # Line: through origin along x-axis
    # Closest: (3,0) on ray to (3,0) on line, distance = 0
    ray_origin = np.zeros(dims, dtype=dtype)
    ray_origin[0] = 3.0
    ray_dir = np.zeros(dims, dtype=dtype)
    ray_dir[1] = 1.0
    ray = tf.Ray(origin=ray_origin, direction=ray_dir)

    line_origin = np.zeros(dims, dtype=dtype)
    line_dir = np.zeros(dims, dtype=dtype)
    line_dir[0] = 1.0
    line = tf.Line(origin=line_origin, direction=line_dir)

    dist2, c0, c1 = tf.closest_metric_point_pair(ray, line)

    expected_pt = np.zeros(dims, dtype=dtype)
    expected_pt[0] = 3.0

    assert np.isclose(dist2, 0.0, atol=1e-5), f"Expected dist2=0.0, got {dist2}"
    assert np.allclose(c0, expected_pt, atol=1e-5), f"c0 should be (3,0,...), got {c0}"
    assert np.allclose(c1, expected_pt, atol=1e-5), f"c1 should be (3,0,...), got {c1}"
    assert np.isclose(tf.distance2(ray, line), dist2, atol=1e-5), "distance2 should match"


# ==============================================================================
# Full Matrix Tests - Line combinations
# ==============================================================================

@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("dims", [2, 3])
def test_line_line_parallel(dtype, dims):
    """Test Line-Line parallel lines."""
    # Line 1: through origin along x-axis
    # Line 2: through (0,5) along x-axis
    # Distance = 5 everywhere
    line1_origin = np.zeros(dims, dtype=dtype)
    line1_dir = np.zeros(dims, dtype=dtype)
    line1_dir[0] = 1.0
    line1 = tf.Line(origin=line1_origin, direction=line1_dir)

    line2_origin = np.zeros(dims, dtype=dtype)
    line2_origin[1] = 5.0
    line2_dir = np.zeros(dims, dtype=dtype)
    line2_dir[0] = 1.0
    line2 = tf.Line(origin=line2_origin, direction=line2_dir)

    dist2, c0, c1 = tf.closest_metric_point_pair(line1, line2)

    assert np.isclose(dist2, 25.0), f"Expected dist2=25.0, got {dist2}"
    # Closest points should have same x-coordinate
    assert np.isclose(c0[0], c1[0], atol=1e-5), "x-coordinates should match"
    assert np.isclose(tf.distance2(line1, line2), dist2), "distance2 should match"


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_line_line_skew_3d(dtype):
    """Test Line-Line skew lines in 3D."""
    # Line 1: through origin along x-axis
    # Line 2: through (0,0,4) along y-axis
    # Closest points: (0,0,0) and (0,0,4), distance = 4
    line1_origin = np.array([0.0, 0.0, 0.0], dtype=dtype)
    line1_dir = np.array([1.0, 0.0, 0.0], dtype=dtype)
    line1 = tf.Line(origin=line1_origin, direction=line1_dir)

    line2_origin = np.array([0.0, 0.0, 4.0], dtype=dtype)
    line2_dir = np.array([0.0, 1.0, 0.0], dtype=dtype)
    line2 = tf.Line(origin=line2_origin, direction=line2_dir)

    dist2, c0, c1 = tf.closest_metric_point_pair(line1, line2)

    expected_c0 = np.array([0.0, 0.0, 0.0], dtype=dtype)
    expected_c1 = np.array([0.0, 0.0, 4.0], dtype=dtype)

    assert np.isclose(dist2, 16.0), f"Expected dist2=16.0, got {dist2}"
    assert np.allclose(c0, expected_c0, atol=1e-5), f"c0 should be (0,0,0), got {c0}"
    assert np.allclose(c1, expected_c1, atol=1e-5), f"c1 should be (0,0,4), got {c1}"
    assert np.isclose(tf.distance2(line1, line2), dist2), "distance2 should match"


# ==============================================================================
# Full Matrix Tests - Polygon combinations
# ==============================================================================

@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("dims", [2, 3])
def test_segment_polygon_separated(dtype, dims):
    """Test Segment-Polygon with clear separation."""
    # Polygon: unit square at origin
    # Segment: from (3,0.5) to (5,0.5) - to the right of square
    # Closest: (1,0.5) on polygon edge to (3,0.5) on segment, distance = 2
    square = np.zeros((4, dims), dtype=dtype)
    square[0] = [0.0, 0.0] + [0.0] * (dims - 2)
    square[1] = [1.0, 0.0] + [0.0] * (dims - 2)
    square[2] = [1.0, 1.0] + [0.0] * (dims - 2)
    square[3] = [0.0, 1.0] + [0.0] * (dims - 2)
    poly = tf.Polygon(square)

    seg_pts = np.zeros((2, dims), dtype=dtype)
    seg_pts[0, 0] = 3.0
    seg_pts[0, 1] = 0.5
    seg_pts[1, 0] = 5.0
    seg_pts[1, 1] = 0.5
    seg = tf.Segment(seg_pts)

    dist2, c0, c1 = tf.closest_metric_point_pair(seg, poly)

    expected_c0 = np.zeros(dims, dtype=dtype)
    expected_c0[0] = 3.0
    expected_c0[1] = 0.5
    expected_c1 = np.zeros(dims, dtype=dtype)
    expected_c1[0] = 1.0
    expected_c1[1] = 0.5

    assert np.isclose(dist2, 4.0), f"Expected dist2=4.0, got {dist2}"
    assert np.allclose(c0, expected_c0, atol=1e-5), f"c0 should be (3,0.5,...), got {c0}"
    assert np.allclose(c1, expected_c1, atol=1e-5), f"c1 should be (1,0.5,...), got {c1}"
    assert np.isclose(tf.distance2(seg, poly), dist2), "distance2 should match"


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_polygon_ray_hitting_3d(dtype):
    """Test Polygon-Ray with ray hitting polygon in 3D."""
    # Triangle in XY plane at z=0
    # Ray from (0.25, 0.25, 5) pointing in -z direction
    # Hits polygon at (0.25, 0.25, 0), distance = 0
    triangle = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 1.0, 0.0]
    ], dtype=dtype)
    poly = tf.Polygon(triangle)

    ray = tf.Ray(
        origin=np.array([0.25, 0.25, 5.0], dtype=dtype),
        direction=np.array([0.0, 0.0, -1.0], dtype=dtype)
    )

    dist2, c0, c1 = tf.closest_metric_point_pair(poly, ray)

    assert np.isclose(dist2, 0.0, atol=1e-5), f"Expected dist2=0.0, got {dist2}"
    assert np.isclose(tf.distance2(poly, ray), dist2, atol=1e-5), "distance2 should match"


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_polygon_line_parallel_3d(dtype):
    """Test Polygon-Line parallel to polygon in 3D."""
    # Triangle in XY plane at z=0
    # Line at z=3, parallel to x-axis at y=0.5
    # Closest point on polygon is on the triangle at y=0.5
    triangle = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 1.0, 0.0]
    ], dtype=dtype)
    poly = tf.Polygon(triangle)

    line = tf.Line(
        origin=np.array([0.0, 0.5, 3.0], dtype=dtype),
        direction=np.array([1.0, 0.0, 0.0], dtype=dtype)
    )

    dist2, c0, c1 = tf.closest_metric_point_pair(poly, line)

    # Distance is 3 (z-difference)
    assert np.isclose(dist2, 9.0), f"Expected dist2=9.0, got {dist2}"
    # Both closest points should be at z=0 and z=3 respectively
    assert np.isclose(c0[2], 0.0, atol=1e-5), f"c0 z should be 0, got {c0[2]}"
    assert np.isclose(c1[2], 3.0, atol=1e-5), f"c1 z should be 3, got {c1[2]}"
    assert np.isclose(tf.distance2(poly, line), dist2), "distance2 should match"


# ==============================================================================
# Full Matrix Tests - Plane combinations (3D only)
# ==============================================================================

@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_polygon_plane_parallel(dtype):
    """Test Polygon-Plane parallel configuration."""
    # Triangle in XY plane at z=0
    # Plane at z=5 (normal = [0,0,1], d=5)
    triangle = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 1.0, 0.0]
    ], dtype=dtype)
    poly = tf.Polygon(triangle)

    plane = tf.Plane(np.array([0.0, 0.0, 1.0, -5.0], dtype=dtype))

    dist2, c0, c1 = tf.closest_metric_point_pair(poly, plane)

    # Distance is 5
    assert np.isclose(dist2, 25.0), f"Expected dist2=25.0, got {dist2}"
    assert np.isclose(c0[2], 0.0, atol=1e-5), f"c0 z should be 0, got {c0[2]}"
    assert np.isclose(c1[2], 5.0, atol=1e-5), f"c1 z should be 5, got {c1[2]}"
    assert np.isclose(tf.distance2(poly, plane), dist2), "distance2 should match"


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_plane_plane_parallel(dtype):
    """Test Plane-Plane parallel planes."""
    # Plane 1 at z=0, Plane 2 at z=7
    plane1 = tf.Plane(np.array([0.0, 0.0, 1.0, 0.0], dtype=dtype))
    plane2 = tf.Plane(np.array([0.0, 0.0, 1.0, -7.0], dtype=dtype))

    dist2, c0, c1 = tf.closest_metric_point_pair(plane1, plane2)

    # Distance is 7
    assert np.isclose(dist2, 49.0), f"Expected dist2=49.0, got {dist2}"
    assert np.isclose(c0[2], 0.0, atol=1e-5), f"c0 z should be 0, got {c0[2]}"
    assert np.isclose(c1[2], 7.0, atol=1e-5), f"c1 z should be 7, got {c1[2]}"
    assert np.isclose(tf.distance2(plane1, plane2), dist2), "distance2 should match"


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_plane_plane_intersecting(dtype):
    """Test Plane-Plane intersecting planes."""
    # Plane 1: z=0, Plane 2: x=0
    # These intersect along y-axis, distance = 0
    plane1 = tf.Plane(np.array([0.0, 0.0, 1.0, 0.0], dtype=dtype))
    plane2 = tf.Plane(np.array([1.0, 0.0, 0.0, 0.0], dtype=dtype))

    dist2, c0, c1 = tf.closest_metric_point_pair(plane1, plane2)

    assert np.isclose(dist2, 0.0, atol=1e-5), f"Expected dist2=0.0, got {dist2}"
    assert np.isclose(tf.distance2(plane1, plane2), dist2, atol=1e-5), "distance2 should match"


# ==============================================================================
# Swap symmetry tests
# ==============================================================================

@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_swap_symmetry_point_segment(dtype):
    """Test that swapping arguments gives consistent results."""
    seg_pts = np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]], dtype=dtype)
    seg = tf.Segment(seg_pts)
    pt = tf.Point(np.array([2.0, 3.0, 0.0], dtype=dtype))

    dist2_a, c0_a, c1_a = tf.closest_metric_point_pair(pt, seg)
    dist2_b, c0_b, c1_b = tf.closest_metric_point_pair(seg, pt)

    assert np.isclose(dist2_a, dist2_b), "Distances should match after swap"
    assert np.allclose(c0_a, c1_b, atol=1e-5), "c0 of A should equal c1 of B"
    assert np.allclose(c1_a, c0_b, atol=1e-5), "c1 of A should equal c0 of B"


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_swap_symmetry_segment_polygon(dtype):
    """Test swap symmetry for Segment-Polygon."""
    square = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0]
    ], dtype=dtype)
    poly = tf.Polygon(square)

    seg_pts = np.array([[3.0, 0.5, 0.0], [5.0, 0.5, 0.0]], dtype=dtype)
    seg = tf.Segment(seg_pts)

    dist2_a, c0_a, c1_a = tf.closest_metric_point_pair(seg, poly)
    dist2_b, c0_b, c1_b = tf.closest_metric_point_pair(poly, seg)

    assert np.isclose(dist2_a, dist2_b), "Distances should match after swap"
    assert np.allclose(c0_a, c1_b, atol=1e-5), "c0 of A should equal c1 of B"
    assert np.allclose(c1_a, c0_b, atol=1e-5), "c1 of A should equal c0 of B"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
