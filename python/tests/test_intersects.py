"""
Test intersects functionality

Copyright (c) 2025 Žiga Sajovic, XLAB
"""

import sys
import pytest
import numpy as np
import trueform as tf


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_intersects_point_point(dtype):
    """Test intersects between points"""
    # Same points should intersect
    pt1 = tf.Point(np.array([1.0, 2.0, 3.0], dtype=dtype))
    pt2 = tf.Point(np.array([1.0, 2.0, 3.0], dtype=dtype))
    result = tf.intersects(pt1, pt2)
    assert result, f"Same points should intersect ({dtype})"

    # Different points should not intersect
    pt3 = tf.Point(np.array([5.0, 6.0, 7.0], dtype=dtype))
    result = tf.intersects(pt1, pt3)
    assert not result, f"Different points should not intersect ({dtype})"


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_intersects_point_aabb(dtype):
    """Test intersects between point and AABB"""
    # Point inside AABB
    pt_inside = tf.Point(np.array([0.5, 0.5], dtype=dtype))
    box = tf.AABB(
        min=np.array([0.0, 0.0], dtype=dtype),
        max=np.array([1.0, 1.0], dtype=dtype)
    )
    result = tf.intersects(pt_inside, box)
    assert result, f"Point inside AABB should intersect ({dtype})"

    # Point outside AABB
    pt_outside = tf.Point(np.array([2.0, 2.0], dtype=dtype))
    result = tf.intersects(pt_outside, box)
    assert not result, f"Point outside AABB should not intersect ({dtype})"

    # Test swap order
    result = tf.intersects(box, pt_inside)
    assert result, f"Swapped order should work ({dtype})"


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_intersects_aabb_aabb(dtype):
    """Test intersects between two AABBs"""
    # Overlapping AABBs
    box1 = tf.AABB(
        min=np.array([0.0, 0.0, 0.0], dtype=dtype),
        max=np.array([2.0, 2.0, 2.0], dtype=dtype)
    )
    box2 = tf.AABB(
        min=np.array([1.0, 1.0, 1.0], dtype=dtype),
        max=np.array([3.0, 3.0, 3.0], dtype=dtype)
    )
    result = tf.intersects(box1, box2)
    assert result, f"Overlapping AABBs should intersect ({dtype})"

    # Non-overlapping AABBs
    box3 = tf.AABB(
        min=np.array([5.0, 5.0, 5.0], dtype=dtype),
        max=np.array([6.0, 6.0, 6.0], dtype=dtype)
    )
    result = tf.intersects(box1, box3)
    assert not result, f"Non-overlapping AABBs should not intersect ({dtype})"


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_intersects_point_segment(dtype):
    """Test intersects between point and segment"""
    # Point on segment
    pt_on = tf.Point(np.array([0.5, 0.5], dtype=dtype))
    seg = tf.Segment(np.array([[0.0, 0.0], [1.0, 1.0]], dtype=dtype))
    result = tf.intersects(pt_on, seg)
    assert result, f"Point on segment should intersect ({dtype})"

    # Point off segment
    pt_off = tf.Point(np.array([0.5, 0.0], dtype=dtype))
    result = tf.intersects(pt_off, seg)
    assert not result, f"Point off segment should not intersect ({dtype})"

    # Test swap
    result = tf.intersects(seg, pt_on)
    assert result, f"Swapped order should work ({dtype})"


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_intersects_segment_segment(dtype):
    """Test intersects between two segments"""
    # Intersecting segments (2D)
    seg1 = tf.Segment(np.array([[0.0, 0.0], [1.0, 1.0]], dtype=dtype))
    seg2 = tf.Segment(np.array([[0.0, 1.0], [1.0, 0.0]], dtype=dtype))
    result = tf.intersects(seg1, seg2)
    assert result, f"Intersecting segments should intersect ({dtype})"

    # Non-intersecting segments (2D)
    seg3 = tf.Segment(np.array([[2.0, 2.0], [3.0, 3.0]], dtype=dtype))
    result = tf.intersects(seg1, seg3)
    assert not result, f"Non-intersecting segments should not intersect ({dtype})"


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_intersects_point_polygon(dtype):
    """Test intersects between point and polygon"""
    # Square polygon
    square = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0]
    ], dtype=dtype)
    poly = tf.Polygon(square)

    # Point inside polygon
    pt_inside = tf.Point(np.array([0.5, 0.5], dtype=dtype))
    result = tf.intersects(pt_inside, poly)
    assert result, f"Point inside polygon should intersect ({dtype})"

    # Point outside polygon
    pt_outside = tf.Point(np.array([2.0, 2.0], dtype=dtype))
    result = tf.intersects(pt_outside, poly)
    assert not result, f"Point outside polygon should not intersect ({dtype})"

    # Test swap
    result = tf.intersects(poly, pt_inside)
    assert result, f"Swapped order should work ({dtype})"


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_intersects_ray_segment(dtype):
    """Test intersects between ray and segment"""
    # Ray intersecting segment
    ray = tf.Ray(
        origin=np.array([0.0, 0.5], dtype=dtype),
        direction=np.array([1.0, 0.0], dtype=dtype)
    )
    seg = tf.Segment(np.array([[0.5, 0.0], [0.5, 1.0]], dtype=dtype))
    result = tf.intersects(ray, seg)
    assert result, f"Ray should intersect segment ({dtype})"

    # Ray missing segment
    ray_miss = tf.Ray(
        origin=np.array([0.0, 2.0], dtype=dtype),
        direction=np.array([1.0, 0.0], dtype=dtype)
    )
    result = tf.intersects(ray_miss, seg)
    assert not result, f"Ray should not intersect segment ({dtype})"

    # Test swap
    result = tf.intersects(seg, ray)
    assert result, f"Swapped order should work ({dtype})"


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_intersects_ray_polygon(dtype):
    """Test intersects between ray and polygon"""
    # Triangle
    triangle = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 1.0, 0.0]
    ], dtype=dtype)
    poly = tf.Polygon(triangle)

    # Ray hitting polygon
    ray = tf.Ray(
        origin=np.array([0.5, 0.3, 2.0], dtype=dtype),
        direction=np.array([0.0, 0.0, -1.0], dtype=dtype)
    )
    result = tf.intersects(ray, poly)
    assert result, f"Ray should intersect polygon ({dtype})"

    # Ray missing polygon
    ray_miss = tf.Ray(
        origin=np.array([5.0, 5.0, 2.0], dtype=dtype),
        direction=np.array([0.0, 0.0, -1.0], dtype=dtype)
    )
    result = tf.intersects(ray_miss, poly)
    assert not result, f"Ray should not intersect polygon ({dtype})"

    # Test swap
    result = tf.intersects(poly, ray)
    assert result, f"Swapped order should work ({dtype})"


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_intersects_line_line(dtype):
    """Test intersects between two lines"""
    # Intersecting lines (2D)
    line1 = tf.Line(
        origin=np.array([0.0, 0.0], dtype=dtype),
        direction=np.array([1.0, 0.0], dtype=dtype)
    )
    line2 = tf.Line(
        origin=np.array([0.5, -1.0], dtype=dtype),
        direction=np.array([0.0, 1.0], dtype=dtype)
    )
    result = tf.intersects(line1, line2)
    assert result, f"Intersecting lines should intersect ({dtype})"

    # Parallel lines (2D)
    line3 = tf.Line(
        origin=np.array([0.0, 1.0], dtype=dtype),
        direction=np.array([1.0, 0.0], dtype=dtype)
    )
    result = tf.intersects(line1, line3)
    assert not result, f"Parallel lines should not intersect ({dtype})"


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_intersects_plane_primitives(dtype):
    """Test intersects between plane and various primitives (3D only)"""
    # Plane at z=0
    plane = tf.Plane(np.array([0.0, 0.0, 1.0, 0.0], dtype=dtype))

    # Point on plane
    pt_on = tf.Point(np.array([1.0, 1.0, 0.0], dtype=dtype))
    result = tf.intersects(plane, pt_on)
    assert result, f"Point on plane should intersect ({dtype})"

    # Point off plane
    pt_off = tf.Point(np.array([1.0, 1.0, 5.0], dtype=dtype))
    result = tf.intersects(plane, pt_off)
    assert not result, f"Point off plane should not intersect ({dtype})"

    # Ray hitting plane
    ray = tf.Ray(
        origin=np.array([0.0, 0.0, 2.0], dtype=dtype),
        direction=np.array([0.0, 0.0, -1.0], dtype=dtype)
    )
    result = tf.intersects(plane, ray)
    assert result, f"Ray should intersect plane ({dtype})"

    # Ray parallel to plane
    ray_parallel = tf.Ray(
        origin=np.array([0.0, 0.0, 2.0], dtype=dtype),
        direction=np.array([1.0, 0.0, 0.0], dtype=dtype)
    )
    result = tf.intersects(plane, ray_parallel)
    assert not result, f"Parallel ray should not intersect plane ({dtype})"

    # Test swap with AABB
    box = tf.AABB(
        min=np.array([-1.0, -1.0, -1.0], dtype=dtype),
        max=np.array([1.0, 1.0, 1.0], dtype=dtype)
    )
    result = tf.intersects(plane, box)
    assert result, f"Plane should intersect AABB ({dtype})"

    result = tf.intersects(box, plane)
    assert result, f"Swapped order should work ({dtype})"


def test_intersects_dimension_mismatch():
    """Test that dimension mismatch raises an error"""
    pt_2d = tf.Point([0.0, 0.0])
    pt_3d = tf.Point([0.0, 0.0, 0.0])

    try:
        _result = tf.intersects(pt_2d, pt_3d)
        assert False, "Expected ValueError was not raised"
    except ValueError as e:
        assert "Dimension mismatch" in str(e)


def test_segment_aabb_2d():
    """Test segment-AABB intersection in 2D"""
    # AABB from (0,0) to (1,1)
    aabb = tf.AABB(min=[0.0, 0.0], max=[1.0, 1.0])

    # Segment that crosses through AABB
    seg_hit = tf.Segment([[0.5, -0.5], [0.5, 1.5]])
    assert tf.intersects(seg_hit, aabb), "Segment should intersect AABB"
    assert tf.intersects(aabb, seg_hit), "AABB should intersect segment (swapped)"

    # Segment that misses AABB
    seg_miss = tf.Segment([[2.0, 0.5], [3.0, 0.5]])
    assert not tf.intersects(seg_miss, aabb), "Segment should not intersect AABB"
    assert not tf.intersects(aabb, seg_miss), "AABB should not intersect segment (swapped)"


def test_ray_aabb_3d():
    """Test ray-AABB intersection in 3D"""
    # AABB from (0,0,0) to (1,1,1)
    aabb = tf.AABB(min=[0.0, 0.0, 0.0], max=[1.0, 1.0, 1.0])

    # Ray that hits AABB
    ray_hit = tf.Ray(origin=[-1.0, 0.5, 0.5], direction=[1.0, 0.0, 0.0])
    assert tf.intersects(ray_hit, aabb), "Ray should intersect AABB"
    assert tf.intersects(aabb, ray_hit), "AABB should intersect ray (swapped)"

    # Ray that misses AABB
    ray_miss = tf.Ray(origin=[-1.0, 2.0, 0.5], direction=[1.0, 0.0, 0.0])
    assert not tf.intersects(ray_miss, aabb), "Ray should not intersect AABB"
    assert not tf.intersects(aabb, ray_miss), "AABB should not intersect ray (swapped)"


def test_line_aabb_2d():
    """Test line-AABB intersection in 2D"""
    # AABB from (0,0) to (1,1)
    aabb = tf.AABB(min=[0.0, 0.0], max=[1.0, 1.0])

    # Line that crosses through AABB
    line_hit = tf.Line(origin=[0.5, -1.0], direction=[0.0, 1.0])
    assert tf.intersects(line_hit, aabb), "Line should intersect AABB"
    assert tf.intersects(aabb, line_hit), "AABB should intersect line (swapped)"

    # Line that misses AABB
    line_miss = tf.Line(origin=[2.0, 0.0], direction=[0.0, 1.0])
    assert not tf.intersects(line_miss, aabb), "Line should not intersect AABB"
    assert not tf.intersects(aabb, line_miss), "AABB should not intersect line (swapped)"


def test_polygon_aabb_3d():
    """Test polygon-AABB intersection in 3D"""
    # AABB from (0,0,0) to (1,1,1)
    aabb = tf.AABB(min=[0.0, 0.0, 0.0], max=[1.0, 1.0, 1.0])

    # Triangle that crosses through AABB
    poly_hit = tf.Polygon([[0.5, 0.5, -0.5], [0.5, 0.5, 1.5], [1.5, 0.5, 0.5]])
    assert tf.intersects(poly_hit, aabb), "Polygon should intersect AABB"
    assert tf.intersects(aabb, poly_hit), "AABB should intersect polygon (swapped)"

    # Triangle that misses AABB
    poly_miss = tf.Polygon([[2.0, 2.0, 2.0], [3.0, 2.0, 2.0], [2.5, 3.0, 2.0]])
    assert not tf.intersects(poly_miss, aabb), "Polygon should not intersect AABB"
    assert not tf.intersects(aabb, poly_miss), "AABB should not intersect polygon (swapped)"


def test_mixed_dtypes():
    """Test with both float32 and float64"""
    # Test with float64
    aabb_double = tf.AABB(min=np.array([0.0, 0.0], dtype=np.float64),
                          max=np.array([1.0, 1.0], dtype=np.float64))
    seg_double = tf.Segment(np.array([[0.5, -0.5], [0.5, 1.5]], dtype=np.float64))
    assert tf.intersects(seg_double, aabb_double), "Should work with float64"

    # Test with float32
    aabb_float = tf.AABB(min=np.array([0.0, 0.0], dtype=np.float32),
                         max=np.array([1.0, 1.0], dtype=np.float32))
    seg_float = tf.Segment(np.array([[0.5, -0.5], [0.5, 1.5]], dtype=np.float32))
    assert tf.intersects(seg_float, aabb_float), "Should work with float32"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
