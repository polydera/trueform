"""
Test closest_metric_point_pair functionality

Copyright (c) 2025 Žiga Sajovic, XLAB
"""

import sys
import os

# Add parent directory to path so we can import trueform
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
