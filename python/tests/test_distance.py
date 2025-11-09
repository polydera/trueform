"""
Test distance and distance2 functionality

Copyright (c) 2025 Žiga Sajovic, XLAB
"""

import sys
import os

# Add parent directory to path so we can import trueform
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import trueform as tf


def test_distance_point_point():
    """Test distance between points"""
    print("\n=== Test: Point to Point ===")

    for dtype in [np.float32, np.float64]:
        dtype_name = "float32" if dtype == np.float32 else "float64"
        print(f"\n  Testing with {dtype_name}:")

        # Same points
        pt1 = tf.Point(np.array([1.0, 2.0, 3.0], dtype=dtype))
        pt2 = tf.Point(np.array([1.0, 2.0, 3.0], dtype=dtype))
        dist = tf.distance(pt1, pt2)
        dist2 = tf.distance2(pt1, pt2)
        print(f"    Same points: distance={dist}, distance2={dist2}")
        assert abs(dist) < 1e-5, f"Distance should be 0 ({dtype_name})"
        assert abs(dist2) < 1e-5, f"Distance2 should be 0 ({dtype_name})"

        # Different points (distance = sqrt(3))
        pt3 = tf.Point(np.array([2.0, 3.0, 4.0], dtype=dtype))
        dist = tf.distance(pt1, pt3)
        dist2 = tf.distance2(pt1, pt3)
        expected = np.sqrt(3.0)
        print(f"    Different points: distance={dist}, distance2={dist2}")
        assert abs(dist - expected) < 1e-5, f"Distance should be sqrt(3) ({dtype_name})"
        assert abs(dist2 - 3.0) < 1e-5, f"Distance2 should be 3 ({dtype_name})"

    print("✓ Point to Point tests passed")


def test_distance_point_aabb():
    """Test distance between point and AABB"""
    print("\n=== Test: Point to AABB ===")

    for dtype in [np.float32, np.float64]:
        dtype_name = "float32" if dtype == np.float32 else "float64"
        print(f"\n  Testing with {dtype_name}:")

        # Point inside AABB (distance = 0)
        pt_inside = tf.Point(np.array([0.5, 0.5], dtype=dtype))
        box = tf.AABB(
            min=np.array([0.0, 0.0], dtype=dtype),
            max=np.array([1.0, 1.0], dtype=dtype)
        )
        dist = tf.distance(pt_inside, box)
        dist2 = tf.distance2(pt_inside, box)
        print(f"    Point inside: distance={dist}, distance2={dist2}")
        assert abs(dist) < 1e-5, f"Distance should be 0 for point inside AABB ({dtype_name})"
        assert abs(dist2) < 1e-5, f"Distance2 should be 0 for point inside AABB ({dtype_name})"

        # Point outside AABB (distance = sqrt(2))
        pt_outside = tf.Point(np.array([2.0, 2.0], dtype=dtype))
        dist = tf.distance(pt_outside, box)
        dist2 = tf.distance2(pt_outside, box)
        expected = np.sqrt(2.0)
        print(f"    Point outside: distance={dist}, distance2={dist2}")
        assert abs(dist - expected) < 1e-5, f"Distance should be sqrt(2) ({dtype_name})"
        assert abs(dist2 - 2.0) < 1e-5, f"Distance2 should be 2 ({dtype_name})"

        # Test swap order
        dist_swap = tf.distance(box, pt_outside)
        print(f"    AABB to point (swapped): distance={dist_swap}")
        assert abs(dist_swap - expected) < 1e-5, f"Swapped order should work ({dtype_name})"

    print("✓ Point to AABB tests passed")


def test_distance_aabb_aabb():
    """Test distance between two AABBs"""
    print("\n=== Test: AABB to AABB ===")

    for dtype in [np.float32, np.float64]:
        dtype_name = "float32" if dtype == np.float32 else "float64"
        print(f"\n  Testing with {dtype_name}:")

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
        print(f"    Overlapping: distance={dist}, distance2={dist2}")
        assert abs(dist) < 1e-5, f"Distance should be 0 for overlapping AABBs ({dtype_name})"
        assert abs(dist2) < 1e-5, f"Distance2 should be 0 for overlapping AABBs ({dtype_name})"

        # Non-overlapping AABBs
        box3 = tf.AABB(
            min=np.array([5.0, 5.0, 5.0], dtype=dtype),
            max=np.array([6.0, 6.0, 6.0], dtype=dtype)
        )
        dist = tf.distance(box1, box3)
        dist2 = tf.distance2(box1, box3)
        expected = np.sqrt(27.0)  # sqrt(3^2 + 3^2 + 3^2)
        print(f"    Non-overlapping: distance={dist}, distance2={dist2}")
        assert abs(dist - expected) < 1e-5, f"Distance calculation failed ({dtype_name})"
        assert abs(dist2 - 27.0) < 1e-5, f"Distance2 should be 27 ({dtype_name})"

    print("✓ AABB to AABB tests passed")


def test_distance_point_segment():
    """Test distance between point and segment"""
    print("\n=== Test: Point to Segment ===")

    for dtype in [np.float32, np.float64]:
        dtype_name = "float32" if dtype == np.float32 else "float64"
        print(f"\n  Testing with {dtype_name}:")

        # Point on segment
        pt_on = tf.Point(np.array([0.5, 0.5], dtype=dtype))
        seg = tf.Segment(np.array([[0.0, 0.0], [1.0, 1.0]], dtype=dtype))
        dist = tf.distance(pt_on, seg)
        dist2 = tf.distance2(pt_on, seg)
        print(f"    Point on segment: distance={dist}, distance2={dist2}")
        assert abs(dist) < 1e-5, f"Distance should be 0 for point on segment ({dtype_name})"
        assert abs(dist2) < 1e-5, f"Distance2 should be 0 for point on segment ({dtype_name})"

        # Point off segment
        pt_off = tf.Point(np.array([0.5, 0.0], dtype=dtype))
        dist = tf.distance(pt_off, seg)
        dist2 = tf.distance2(pt_off, seg)
        expected = np.sqrt(0.125)  # perpendicular distance to line y=x
        print(f"    Point off segment: distance={dist}, distance2={dist2}")
        assert abs(dist - expected) < 1e-4, f"Distance calculation failed ({dtype_name})"
        assert abs(dist2 - 0.125) < 1e-4, f"Distance2 calculation failed ({dtype_name})"

        # Test swap
        dist_swap = tf.distance(seg, pt_on)
        print(f"    Segment to point (swapped): distance={dist_swap}")
        assert abs(dist_swap) < 1e-5, f"Swapped order should work ({dtype_name})"

    print("✓ Point to Segment tests passed")


def test_distance_segment_segment():
    """Test distance between two segments"""
    print("\n=== Test: Segment to Segment ===")

    for dtype in [np.float32, np.float64]:
        dtype_name = "float32" if dtype == np.float32 else "float64"
        print(f"\n  Testing with {dtype_name}:")

        # Parallel segments in 2D
        seg1 = tf.Segment(np.array([[0.0, 0.0], [1.0, 0.0]], dtype=dtype))
        seg2 = tf.Segment(np.array([[0.0, 2.0], [1.0, 2.0]], dtype=dtype))
        dist = tf.distance(seg1, seg2)
        dist2 = tf.distance2(seg1, seg2)
        print(f"    Parallel segments: distance={dist}, distance2={dist2}")
        assert abs(dist - 2.0) < 1e-5, f"Distance should be 2 ({dtype_name})"
        assert abs(dist2 - 4.0) < 1e-5, f"Distance2 should be 4 ({dtype_name})"

        # Skew segments in 3D
        seg3 = tf.Segment(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=dtype))
        seg4 = tf.Segment(np.array([[0.0, 1.0, 1.0], [1.0, 1.0, 1.0]], dtype=dtype))
        dist = tf.distance(seg3, seg4)
        dist2 = tf.distance2(seg3, seg4)
        expected = np.sqrt(2.0)  # sqrt(1^2 + 1^2)
        print(f"    Skew segments: distance={dist}, distance2={dist2}")
        assert abs(dist - expected) < 1e-5, f"Distance calculation failed ({dtype_name})"
        assert abs(dist2 - 2.0) < 1e-5, f"Distance2 should be 2 ({dtype_name})"

    print("✓ Segment to Segment tests passed")


def test_distance_point_polygon():
    """Test distance between point and polygon"""
    print("\n=== Test: Point to Polygon ===")

    for dtype in [np.float32, np.float64]:
        dtype_name = "float32" if dtype == np.float32 else "float64"
        print(f"\n  Testing with {dtype_name}:")

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
        print(f"    Point inside: distance={dist}, distance2={dist2}")
        # Point inside polygon has distance 0
        assert abs(dist) < 1e-5, f"Distance should be 0 ({dtype_name})"
        assert abs(dist2) < 1e-5, f"Distance2 should be 0 ({dtype_name})"

        # Point outside polygon
        pt_outside = tf.Point(np.array([2.0, 2.0], dtype=dtype))
        dist = tf.distance(pt_outside, poly)
        dist2 = tf.distance2(pt_outside, poly)
        expected = np.sqrt(2.0)  # distance to corner (1,1)
        print(f"    Point outside: distance={dist}, distance2={dist2}")
        assert abs(dist - expected) < 1e-5, f"Distance calculation failed ({dtype_name})"
        assert abs(dist2 - 2.0) < 1e-5, f"Distance2 should be 2 ({dtype_name})"

        # Test swap
        dist_swap = tf.distance(poly, pt_inside)
        print(f"    Polygon to point (swapped): distance={dist_swap}")
        assert abs(dist_swap) < 1e-5, f"Swapped order should work ({dtype_name})"

    print("✓ Point to Polygon tests passed")


def test_distance_point_line():
    """Test distance between point and line"""
    print("\n=== Test: Point to Line ===")

    for dtype in [np.float32, np.float64]:
        dtype_name = "float32" if dtype == np.float32 else "float64"
        print(f"\n  Testing with {dtype_name}:")

        # Line along x-axis
        line = tf.Line(
            origin=np.array([0.0, 0.0], dtype=dtype),
            direction=np.array([1.0, 0.0], dtype=dtype)
        )

        # Point on line
        pt_on = tf.Point(np.array([5.0, 0.0], dtype=dtype))
        dist = tf.distance(pt_on, line)
        dist2 = tf.distance2(pt_on, line)
        print(f"    Point on line: distance={dist}, distance2={dist2}")
        assert abs(dist) < 1e-5, f"Distance should be 0 ({dtype_name})"
        assert abs(dist2) < 1e-5, f"Distance2 should be 0 ({dtype_name})"

        # Point off line
        pt_off = tf.Point(np.array([0.0, 3.0], dtype=dtype))
        dist = tf.distance(pt_off, line)
        dist2 = tf.distance2(pt_off, line)
        print(f"    Point off line: distance={dist}, distance2={dist2}")
        assert abs(dist - 3.0) < 1e-5, f"Distance should be 3 ({dtype_name})"
        assert abs(dist2 - 9.0) < 1e-5, f"Distance2 should be 9 ({dtype_name})"

        # Test swap
        dist_swap = tf.distance(line, pt_off)
        print(f"    Line to point (swapped): distance={dist_swap}")
        assert abs(dist_swap - 3.0) < 1e-5, f"Swapped order should work ({dtype_name})"

    print("✓ Point to Line tests passed")


def test_distance_point_ray():
    """Test distance between point and ray"""
    print("\n=== Test: Point to Ray ===")

    for dtype in [np.float32, np.float64]:
        dtype_name = "float32" if dtype == np.float32 else "float64"
        print(f"\n  Testing with {dtype_name}:")

        # Ray along positive x-axis
        ray = tf.Ray(
            origin=np.array([0.0, 0.0], dtype=dtype),
            direction=np.array([1.0, 0.0], dtype=dtype)
        )

        # Point on ray
        pt_on = tf.Point(np.array([5.0, 0.0], dtype=dtype))
        dist = tf.distance(pt_on, ray)
        dist2 = tf.distance2(pt_on, ray)
        print(f"    Point on ray: distance={dist}, distance2={dist2}")
        assert abs(dist) < 1e-5, f"Distance should be 0 ({dtype_name})"
        assert abs(dist2) < 1e-5, f"Distance2 should be 0 ({dtype_name})"

        # Point behind ray origin
        pt_behind = tf.Point(np.array([-2.0, 0.0], dtype=dtype))
        dist = tf.distance(pt_behind, ray)
        dist2 = tf.distance2(pt_behind, ray)
        print(f"    Point behind ray: distance={dist}, distance2={dist2}")
        assert abs(dist - 2.0) < 1e-5, f"Distance should be 2 ({dtype_name})"
        assert abs(dist2 - 4.0) < 1e-5, f"Distance2 should be 4 ({dtype_name})"

        # Test swap
        dist_swap = tf.distance(ray, pt_on)
        print(f"    Ray to point (swapped): distance={dist_swap}")
        assert abs(dist_swap) < 1e-5, f"Swapped order should work ({dtype_name})"

    print("✓ Point to Ray tests passed")


def test_distance_point_plane():
    """Test distance between point and plane (3D only)"""
    print("\n=== Test: Point to Plane ===")

    for dtype in [np.float32, np.float64]:
        dtype_name = "float32" if dtype == np.float32 else "float64"
        print(f"\n  Testing with {dtype_name}:")

        # Plane at z=0 (normal = [0,0,1], d=0)
        plane = tf.Plane(np.array([0.0, 0.0, 1.0, 0.0], dtype=dtype))

        # Point on plane
        pt_on = tf.Point(np.array([1.0, 1.0, 0.0], dtype=dtype))
        dist = tf.distance(pt_on, plane)
        dist2 = tf.distance2(pt_on, plane)
        print(f"    Point on plane: distance={dist}, distance2={dist2}")
        assert abs(dist) < 1e-5, f"Distance should be 0 ({dtype_name})"
        assert abs(dist2) < 1e-5, f"Distance2 should be 0 ({dtype_name})"

        # Point above plane
        pt_above = tf.Point(np.array([1.0, 1.0, 5.0], dtype=dtype))
        dist = tf.distance(pt_above, plane)
        dist2 = tf.distance2(pt_above, plane)
        print(f"    Point above plane: distance={dist}, distance2={dist2}")
        assert abs(dist - 5.0) < 1e-5, f"Distance should be 5 ({dtype_name})"
        assert abs(dist2 - 25.0) < 1e-5, f"Distance2 should be 25 ({dtype_name})"

        # Test swap
        dist_swap = tf.distance(plane, pt_above)
        print(f"    Plane to point (swapped): distance={dist_swap}")
        assert abs(dist_swap - 5.0) < 1e-5, f"Swapped order should work ({dtype_name})"

    print("✓ Point to Plane tests passed")


def test_distance_line_line():
    """Test distance between two lines"""
    print("\n=== Test: Line to Line ===")

    for dtype in [np.float32, np.float64]:
        dtype_name = "float32" if dtype == np.float32 else "float64"
        print(f"\n  Testing with {dtype_name}:")

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
        print(f"    Parallel lines: distance={dist}, distance2={dist2}")
        assert abs(dist - 3.0) < 1e-5, f"Distance should be 3 ({dtype_name})"
        assert abs(dist2 - 9.0) < 1e-5, f"Distance2 should be 9 ({dtype_name})"

        # Intersecting lines in 2D
        line3 = tf.Line(
            origin=np.array([0.0, 0.0], dtype=dtype),
            direction=np.array([0.0, 1.0], dtype=dtype)
        )
        dist = tf.distance(line1, line3)
        dist2 = tf.distance2(line1, line3)
        print(f"    Intersecting lines: distance={dist}, distance2={dist2}")
        assert abs(dist) < 1e-5, f"Distance should be 0 ({dtype_name})"
        assert abs(dist2) < 1e-5, f"Distance2 should be 0 ({dtype_name})"

    print("✓ Line to Line tests passed")


def test_distance_ray_ray():
    """Test distance between two rays"""
    print("\n=== Test: Ray to Ray ===")

    for dtype in [np.float32, np.float64]:
        dtype_name = "float32" if dtype == np.float32 else "float64"
        print(f"\n  Testing with {dtype_name}:")

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
        print(f"    Parallel rays: distance={dist}, distance2={dist2}")
        assert abs(dist - 2.0) < 1e-5, f"Distance should be 2 ({dtype_name})"
        assert abs(dist2 - 4.0) < 1e-5, f"Distance2 should be 4 ({dtype_name})"

    print("✓ Ray to Ray tests passed")


def test_distance_dimension_mismatch():
    """Test that dimension mismatch raises an error"""
    print("\n=== Test: Dimension Mismatch Validation ===")

    pt_2d = tf.Point([0.0, 0.0])
    pt_3d = tf.Point([0.0, 0.0, 0.0])

    try:
        result = tf.distance(pt_2d, pt_3d)
        print("✗ ERROR: Should have raised ValueError for dimension mismatch!")
        assert False, "Expected ValueError was not raised"
    except ValueError as e:
        print(f"  Correctly raised ValueError: {e}")
        assert "Dimension mismatch" in str(e)

    try:
        result = tf.distance2(pt_2d, pt_3d)
        print("✗ ERROR: Should have raised ValueError for dimension mismatch!")
        assert False, "Expected ValueError was not raised"
    except ValueError as e:
        print(f"  Correctly raised ValueError for distance2: {e}")
        assert "Dimension mismatch" in str(e)

    print("✓ Dimension mismatch validation tests passed")


if __name__ == "__main__":
    print("Testing distance and distance2 functionality\n")
    print("=" * 60)

    try:
        test_distance_point_point()
        test_distance_point_aabb()
        test_distance_aabb_aabb()
        test_distance_point_segment()
        test_distance_segment_segment()
        test_distance_point_polygon()
        test_distance_point_line()
        test_distance_point_ray()
        test_distance_point_plane()
        test_distance_line_line()
        test_distance_ray_ray()
        test_distance_dimension_mismatch()

        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED!")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
