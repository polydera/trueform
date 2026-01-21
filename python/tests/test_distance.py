"""
Test distance and distance2 functionality

Copyright (c) 2025 Žiga Sajovic, XLAB
"""

import sys
import pytest
import numpy as np
import trueform as tf


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_distance_point_point(dtype):
    """Test distance between points"""
    # Same points
    pt1 = tf.Point(np.array([1.0, 2.0, 3.0], dtype=dtype))
    pt2 = tf.Point(np.array([1.0, 2.0, 3.0], dtype=dtype))
    dist = tf.distance(pt1, pt2)
    dist2 = tf.distance2(pt1, pt2)
    assert abs(dist) < 1e-5, f"Distance should be 0 ({dtype})"
    assert abs(dist2) < 1e-5, f"Distance2 should be 0 ({dtype})"

    # Different points (distance = sqrt(3))
    pt3 = tf.Point(np.array([2.0, 3.0, 4.0], dtype=dtype))
    dist = tf.distance(pt1, pt3)
    dist2 = tf.distance2(pt1, pt3)
    expected = np.sqrt(3.0)
    assert abs(dist - expected) < 1e-5, f"Distance should be sqrt(3) ({dtype})"
    assert abs(dist2 - 3.0) < 1e-5, f"Distance2 should be 3 ({dtype})"


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_distance_point_aabb(dtype):
    """Test distance between point and AABB"""
    # Point inside AABB (distance = 0)
    pt_inside = tf.Point(np.array([0.5, 0.5], dtype=dtype))
    box = tf.AABB(
        min=np.array([0.0, 0.0], dtype=dtype),
        max=np.array([1.0, 1.0], dtype=dtype)
    )
    dist = tf.distance(pt_inside, box)
    dist2 = tf.distance2(pt_inside, box)
    assert abs(dist) < 1e-5, f"Distance should be 0 for point inside AABB ({dtype})"
    assert abs(dist2) < 1e-5, f"Distance2 should be 0 for point inside AABB ({dtype})"

    # Point outside AABB (distance = sqrt(2))
    pt_outside = tf.Point(np.array([2.0, 2.0], dtype=dtype))
    dist = tf.distance(pt_outside, box)
    dist2 = tf.distance2(pt_outside, box)
    expected = np.sqrt(2.0)
    assert abs(dist - expected) < 1e-5, f"Distance should be sqrt(2) ({dtype})"
    assert abs(dist2 - 2.0) < 1e-5, f"Distance2 should be 2 ({dtype})"

    # Test swap order
    dist_swap = tf.distance(box, pt_outside)
    assert abs(dist_swap - expected) < 1e-5, f"Swapped order should work ({dtype})"


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_distance_aabb_aabb(dtype):
    """Test distance between two AABBs"""
    # Overlapping AABBs (distance = 0)
    box1 = tf.AABB(
        min=np.array([0.0, 0.0, 0.0], dtype=dtype),
        max=np.array([2.0, 2.0, 2.0], dtype=dtype)
    )
    box2 = tf.AABB(
        min=np.array([1.0, 1.0, 1.0], dtype=dtype),
        max=np.array([3.0, 3.0, 3.0], dtype=dtype)
    )
    dist = tf.distance(box1, box2)
    dist2 = tf.distance2(box1, box2)
    assert abs(dist) < 1e-5, f"Distance should be 0 for overlapping AABBs ({dtype})"
    assert abs(dist2) < 1e-5, f"Distance2 should be 0 for overlapping AABBs ({dtype})"

    # Non-overlapping AABBs
    box3 = tf.AABB(
        min=np.array([5.0, 5.0, 5.0], dtype=dtype),
        max=np.array([6.0, 6.0, 6.0], dtype=dtype)
    )
    dist = tf.distance(box1, box3)
    dist2 = tf.distance2(box1, box3)
    expected = np.sqrt(27.0)  # sqrt(3^2 + 3^2 + 3^2)
    assert abs(dist - expected) < 1e-5, f"Distance calculation failed ({dtype})"
    assert abs(dist2 - 27.0) < 1e-5, f"Distance2 should be 27 ({dtype})"


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_distance_point_segment(dtype):
    """Test distance between point and segment"""
    # Point on segment
    pt_on = tf.Point(np.array([0.5, 0.5], dtype=dtype))
    seg = tf.Segment(np.array([[0.0, 0.0], [1.0, 1.0]], dtype=dtype))
    dist = tf.distance(pt_on, seg)
    dist2 = tf.distance2(pt_on, seg)
    assert abs(dist) < 1e-5, f"Distance should be 0 for point on segment ({dtype})"
    assert abs(dist2) < 1e-5, f"Distance2 should be 0 for point on segment ({dtype})"

    # Point off segment
    pt_off = tf.Point(np.array([0.5, 0.0], dtype=dtype))
    dist = tf.distance(pt_off, seg)
    dist2 = tf.distance2(pt_off, seg)
    expected = np.sqrt(0.125)  # perpendicular distance to line y=x
    assert abs(dist - expected) < 1e-4, f"Distance calculation failed ({dtype})"
    assert abs(dist2 - 0.125) < 1e-4, f"Distance2 calculation failed ({dtype})"

    # Test swap
    dist_swap = tf.distance(seg, pt_on)
    assert abs(dist_swap) < 1e-5, f"Swapped order should work ({dtype})"


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_distance_segment_segment(dtype):
    """Test distance between two segments"""
    # Parallel segments in 2D
    seg1 = tf.Segment(np.array([[0.0, 0.0], [1.0, 0.0]], dtype=dtype))
    seg2 = tf.Segment(np.array([[0.0, 2.0], [1.0, 2.0]], dtype=dtype))
    dist = tf.distance(seg1, seg2)
    dist2 = tf.distance2(seg1, seg2)
    assert abs(dist - 2.0) < 1e-5, f"Distance should be 2 ({dtype})"
    assert abs(dist2 - 4.0) < 1e-5, f"Distance2 should be 4 ({dtype})"

    # Skew segments in 3D
    seg3 = tf.Segment(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=dtype))
    seg4 = tf.Segment(np.array([[0.0, 1.0, 1.0], [1.0, 1.0, 1.0]], dtype=dtype))
    dist = tf.distance(seg3, seg4)
    dist2 = tf.distance2(seg3, seg4)
    expected = np.sqrt(2.0)  # sqrt(1^2 + 1^2)
    assert abs(dist - expected) < 1e-5, f"Distance calculation failed ({dtype})"
    assert abs(dist2 - 2.0) < 1e-5, f"Distance2 should be 2 ({dtype})"


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_distance_point_polygon(dtype):
    """Test distance between point and polygon"""
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
    dist = tf.distance(pt_inside, poly)
    dist2 = tf.distance2(pt_inside, poly)
    # Point inside polygon has distance 0
    assert abs(dist) < 1e-5, f"Distance should be 0 ({dtype})"
    assert abs(dist2) < 1e-5, f"Distance2 should be 0 ({dtype})"

    # Point outside polygon
    pt_outside = tf.Point(np.array([2.0, 2.0], dtype=dtype))
    dist = tf.distance(pt_outside, poly)
    dist2 = tf.distance2(pt_outside, poly)
    expected = np.sqrt(2.0)  # distance to corner (1,1)
    assert abs(dist - expected) < 1e-5, f"Distance calculation failed ({dtype})"
    assert abs(dist2 - 2.0) < 1e-5, f"Distance2 should be 2 ({dtype})"

    # Test swap
    dist_swap = tf.distance(poly, pt_inside)
    assert abs(dist_swap) < 1e-5, f"Swapped order should work ({dtype})"


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_distance_point_line(dtype):
    """Test distance between point and line"""
    # Line along x-axis
    line = tf.Line(
        origin=np.array([0.0, 0.0], dtype=dtype),
        direction=np.array([1.0, 0.0], dtype=dtype)
    )

    # Point on line
    pt_on = tf.Point(np.array([5.0, 0.0], dtype=dtype))
    dist = tf.distance(pt_on, line)
    dist2 = tf.distance2(pt_on, line)
    assert abs(dist) < 1e-5, f"Distance should be 0 ({dtype})"
    assert abs(dist2) < 1e-5, f"Distance2 should be 0 ({dtype})"

    # Point off line
    pt_off = tf.Point(np.array([0.0, 3.0], dtype=dtype))
    dist = tf.distance(pt_off, line)
    dist2 = tf.distance2(pt_off, line)
    assert abs(dist - 3.0) < 1e-5, f"Distance should be 3 ({dtype})"
    assert abs(dist2 - 9.0) < 1e-5, f"Distance2 should be 9 ({dtype})"

    # Test swap
    dist_swap = tf.distance(line, pt_off)
    assert abs(dist_swap - 3.0) < 1e-5, f"Swapped order should work ({dtype})"


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_distance_point_ray(dtype):
    """Test distance between point and ray"""
    # Ray along positive x-axis
    ray = tf.Ray(
        origin=np.array([0.0, 0.0], dtype=dtype),
        direction=np.array([1.0, 0.0], dtype=dtype)
    )

    # Point on ray
    pt_on = tf.Point(np.array([5.0, 0.0], dtype=dtype))
    dist = tf.distance(pt_on, ray)
    dist2 = tf.distance2(pt_on, ray)
    assert abs(dist) < 1e-5, f"Distance should be 0 ({dtype})"
    assert abs(dist2) < 1e-5, f"Distance2 should be 0 ({dtype})"

    # Point behind ray origin
    pt_behind = tf.Point(np.array([-2.0, 0.0], dtype=dtype))
    dist = tf.distance(pt_behind, ray)
    dist2 = tf.distance2(pt_behind, ray)
    assert abs(dist - 2.0) < 1e-5, f"Distance should be 2 ({dtype})"
    assert abs(dist2 - 4.0) < 1e-5, f"Distance2 should be 4 ({dtype})"

    # Test swap
    dist_swap = tf.distance(ray, pt_on)
    assert abs(dist_swap) < 1e-5, f"Swapped order should work ({dtype})"


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_distance_point_plane(dtype):
    """Test distance between point and plane (3D only)"""
    # Plane at z=0 (normal = [0,0,1], d=0)
    plane = tf.Plane(np.array([0.0, 0.0, 1.0, 0.0], dtype=dtype))

    # Point on plane
    pt_on = tf.Point(np.array([1.0, 1.0, 0.0], dtype=dtype))
    dist = tf.distance(pt_on, plane)
    dist2 = tf.distance2(pt_on, plane)
    assert abs(dist) < 1e-5, f"Distance should be 0 ({dtype})"
    assert abs(dist2) < 1e-5, f"Distance2 should be 0 ({dtype})"

    # Point above plane
    pt_above = tf.Point(np.array([1.0, 1.0, 5.0], dtype=dtype))
    dist = tf.distance(pt_above, plane)
    dist2 = tf.distance2(pt_above, plane)
    assert abs(dist - 5.0) < 1e-5, f"Distance should be 5 ({dtype})"
    assert abs(dist2 - 25.0) < 1e-5, f"Distance2 should be 25 ({dtype})"

    # Test swap
    dist_swap = tf.distance(plane, pt_above)
    assert abs(dist_swap - 5.0) < 1e-5, f"Swapped order should work ({dtype})"


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_distance_line_line(dtype):
    """Test distance between two lines"""
    # Parallel lines in 2D
    line1 = tf.Line(
        origin=np.array([0.0, 0.0], dtype=dtype),
        direction=np.array([1.0, 0.0], dtype=dtype)
    )
    line2 = tf.Line(
        origin=np.array([0.0, 3.0], dtype=dtype),
        direction=np.array([1.0, 0.0], dtype=dtype)
    )
    dist = tf.distance(line1, line2)
    dist2 = tf.distance2(line1, line2)
    assert abs(dist - 3.0) < 1e-5, f"Distance should be 3 ({dtype})"
    assert abs(dist2 - 9.0) < 1e-5, f"Distance2 should be 9 ({dtype})"

    # Intersecting lines in 2D
    line3 = tf.Line(
        origin=np.array([0.0, 0.0], dtype=dtype),
        direction=np.array([0.0, 1.0], dtype=dtype)
    )
    dist = tf.distance(line1, line3)
    dist2 = tf.distance2(line1, line3)
    assert abs(dist) < 1e-5, f"Distance should be 0 ({dtype})"
    assert abs(dist2) < 1e-5, f"Distance2 should be 0 ({dtype})"


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_distance_ray_ray(dtype):
    """Test distance between two rays"""
    # Parallel rays
    ray1 = tf.Ray(
        origin=np.array([0.0, 0.0], dtype=dtype),
        direction=np.array([1.0, 0.0], dtype=dtype)
    )
    ray2 = tf.Ray(
        origin=np.array([0.0, 2.0], dtype=dtype),
        direction=np.array([1.0, 0.0], dtype=dtype)
    )
    dist = tf.distance(ray1, ray2)
    dist2 = tf.distance2(ray1, ray2)
    assert abs(dist - 2.0) < 1e-5, f"Distance should be 2 ({dtype})"
    assert abs(dist2 - 4.0) < 1e-5, f"Distance2 should be 4 ({dtype})"


def test_distance_dimension_mismatch():
    """Test that dimension mismatch raises an error"""
    pt_2d = tf.Point([0.0, 0.0])
    pt_3d = tf.Point([0.0, 0.0, 0.0])

    try:
        _result = tf.distance(pt_2d, pt_3d)
        assert False, "Expected ValueError was not raised"
    except ValueError as e:
        assert "Dimension mismatch" in str(e)

    try:
        _result = tf.distance2(pt_2d, pt_3d)
        assert False, "Expected ValueError was not raised"
    except ValueError as e:
        assert "Dimension mismatch" in str(e)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
