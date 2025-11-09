"""
Test intersects functionality

Copyright (c) 2025 Žiga Sajovic, XLAB
"""

import numpy as np
import trueform as tf


def test_intersects_point_point():
    """Test intersects between points"""
    print("\n=== Test: Point to Point ===")

    for dtype in [np.float32, np.float64]:
        dtype_name = "float32" if dtype == np.float32 else "float64"
        print(f"\n  Testing with {dtype_name}:")

        # Same points should intersect
        pt1 = tf.Point(np.array([1.0, 2.0, 3.0], dtype=dtype))
        pt2 = tf.Point(np.array([1.0, 2.0, 3.0], dtype=dtype))
        result = tf.intersects(pt1, pt2)
        print(f"    Same points: {result}")
        assert result == True, f"Same points should intersect ({dtype_name})"

        # Different points should not intersect
        pt3 = tf.Point(np.array([5.0, 6.0, 7.0], dtype=dtype))
        result = tf.intersects(pt1, pt3)
        print(f"    Different points: {result}")
        assert result == False, f"Different points should not intersect ({dtype_name})"

    print("✓ Point to Point tests passed")


def test_intersects_point_aabb():
    """Test intersects between point and AABB"""
    print("\n=== Test: Point to AABB ===")

    for dtype in [np.float32, np.float64]:
        dtype_name = "float32" if dtype == np.float32 else "float64"
        print(f"\n  Testing with {dtype_name}:")

        # Point inside AABB
        pt_inside = tf.Point(np.array([0.5, 0.5], dtype=dtype))
        box = tf.AABB(
            min=np.array([0.0, 0.0], dtype=dtype),
            max=np.array([1.0, 1.0], dtype=dtype)
        )
        result = tf.intersects(pt_inside, box)
        print(f"    Point inside: {result}")
        assert result == True, f"Point inside AABB should intersect ({dtype_name})"

        # Point outside AABB
        pt_outside = tf.Point(np.array([2.0, 2.0], dtype=dtype))
        result = tf.intersects(pt_outside, box)
        print(f"    Point outside: {result}")
        assert result == False, f"Point outside AABB should not intersect ({dtype_name})"

        # Test swap order
        result = tf.intersects(box, pt_inside)
        print(f"    AABB to point (swapped): {result}")
        assert result == True, f"Swapped order should work ({dtype_name})"

    print("✓ Point to AABB tests passed")


def test_intersects_aabb_aabb():
    """Test intersects between two AABBs"""
    print("\n=== Test: AABB to AABB ===")

    for dtype in [np.float32, np.float64]:
        dtype_name = "float32" if dtype == np.float32 else "float64"
        print(f"\n  Testing with {dtype_name}:")

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
        print(f"    Overlapping: {result}")
        assert result == True, f"Overlapping AABBs should intersect ({dtype_name})"

        # Non-overlapping AABBs
        box3 = tf.AABB(
            min=np.array([5.0, 5.0, 5.0], dtype=dtype),
            max=np.array([6.0, 6.0, 6.0], dtype=dtype)
        )
        result = tf.intersects(box1, box3)
        print(f"    Non-overlapping: {result}")
        assert result == False, f"Non-overlapping AABBs should not intersect ({dtype_name})"

    print("✓ AABB to AABB tests passed")


def test_intersects_point_segment():
    """Test intersects between point and segment"""
    print("\n=== Test: Point to Segment ===")

    for dtype in [np.float32, np.float64]:
        dtype_name = "float32" if dtype == np.float32 else "float64"
        print(f"\n  Testing with {dtype_name}:")

        # Point on segment
        pt_on = tf.Point(np.array([0.5, 0.5], dtype=dtype))
        seg = tf.Segment(np.array([[0.0, 0.0], [1.0, 1.0]], dtype=dtype))
        result = tf.intersects(pt_on, seg)
        print(f"    Point on segment: {result}")
        assert result == True, f"Point on segment should intersect ({dtype_name})"

        # Point off segment
        pt_off = tf.Point(np.array([0.5, 0.0], dtype=dtype))
        result = tf.intersects(pt_off, seg)
        print(f"    Point off segment: {result}")
        assert result == False, f"Point off segment should not intersect ({dtype_name})"

        # Test swap
        result = tf.intersects(seg, pt_on)
        print(f"    Segment to point (swapped): {result}")
        assert result == True, f"Swapped order should work ({dtype_name})"

    print("✓ Point to Segment tests passed")


def test_intersects_segment_segment():
    """Test intersects between two segments"""
    print("\n=== Test: Segment to Segment ===")

    for dtype in [np.float32, np.float64]:
        dtype_name = "float32" if dtype == np.float32 else "float64"
        print(f"\n  Testing with {dtype_name}:")

        # Intersecting segments (2D)
        seg1 = tf.Segment(np.array([[0.0, 0.0], [1.0, 1.0]], dtype=dtype))
        seg2 = tf.Segment(np.array([[0.0, 1.0], [1.0, 0.0]], dtype=dtype))
        result = tf.intersects(seg1, seg2)
        print(f"    Intersecting 2D: {result}")
        assert result == True, f"Intersecting segments should intersect ({dtype_name})"

        # Non-intersecting segments (2D)
        seg3 = tf.Segment(np.array([[2.0, 2.0], [3.0, 3.0]], dtype=dtype))
        result = tf.intersects(seg1, seg3)
        print(f"    Non-intersecting 2D: {result}")
        assert result == False, f"Non-intersecting segments should not intersect ({dtype_name})"

    print("✓ Segment to Segment tests passed")


def test_intersects_point_polygon():
    """Test intersects between point and polygon"""
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
        result = tf.intersects(pt_inside, poly)
        print(f"    Point inside: {result}")
        assert result == True, f"Point inside polygon should intersect ({dtype_name})"

        # Point outside polygon
        pt_outside = tf.Point(np.array([2.0, 2.0], dtype=dtype))
        result = tf.intersects(pt_outside, poly)
        print(f"    Point outside: {result}")
        assert result == False, f"Point outside polygon should not intersect ({dtype_name})"

        # Test swap
        result = tf.intersects(poly, pt_inside)
        print(f"    Polygon to point (swapped): {result}")
        assert result == True, f"Swapped order should work ({dtype_name})"

    print("✓ Point to Polygon tests passed")


def test_intersects_ray_segment():
    """Test intersects between ray and segment"""
    print("\n=== Test: Ray to Segment ===")

    for dtype in [np.float32, np.float64]:
        dtype_name = "float32" if dtype == np.float32 else "float64"
        print(f"\n  Testing with {dtype_name}:")

        # Ray intersecting segment
        ray = tf.Ray(
            origin=np.array([0.0, 0.5], dtype=dtype),
            direction=np.array([1.0, 0.0], dtype=dtype)
        )
        seg = tf.Segment(np.array([[0.5, 0.0], [0.5, 1.0]], dtype=dtype))
        result = tf.intersects(ray, seg)
        print(f"    Ray intersecting segment: {result}")
        assert result == True, f"Ray should intersect segment ({dtype_name})"

        # Ray missing segment
        ray_miss = tf.Ray(
            origin=np.array([0.0, 2.0], dtype=dtype),
            direction=np.array([1.0, 0.0], dtype=dtype)
        )
        result = tf.intersects(ray_miss, seg)
        print(f"    Ray missing segment: {result}")
        assert result == False, f"Ray should not intersect segment ({dtype_name})"

        # Test swap
        result = tf.intersects(seg, ray)
        print(f"    Segment to ray (swapped): {result}")
        assert result == True, f"Swapped order should work ({dtype_name})"

    print("✓ Ray to Segment tests passed")


def test_intersects_ray_polygon():
    """Test intersects between ray and polygon"""
    print("\n=== Test: Ray to Polygon ===")

    for dtype in [np.float32, np.float64]:
        dtype_name = "float32" if dtype == np.float32 else "float64"
        print(f"\n  Testing with {dtype_name}:")

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
        print(f"    Ray hitting polygon: {result}")
        assert result == True, f"Ray should intersect polygon ({dtype_name})"

        # Ray missing polygon
        ray_miss = tf.Ray(
            origin=np.array([5.0, 5.0, 2.0], dtype=dtype),
            direction=np.array([0.0, 0.0, -1.0], dtype=dtype)
        )
        result = tf.intersects(ray_miss, poly)
        print(f"    Ray missing polygon: {result}")
        assert result == False, f"Ray should not intersect polygon ({dtype_name})"

        # Test swap
        result = tf.intersects(poly, ray)
        print(f"    Polygon to ray (swapped): {result}")
        assert result == True, f"Swapped order should work ({dtype_name})"

    print("✓ Ray to Polygon tests passed")


def test_intersects_line_line():
    """Test intersects between two lines"""
    print("\n=== Test: Line to Line ===")

    for dtype in [np.float32, np.float64]:
        dtype_name = "float32" if dtype == np.float32 else "float64"
        print(f"\n  Testing with {dtype_name}:")

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
        print(f"    Intersecting lines: {result}")
        assert result == True, f"Intersecting lines should intersect ({dtype_name})"

        # Parallel lines (2D)
        line3 = tf.Line(
            origin=np.array([0.0, 1.0], dtype=dtype),
            direction=np.array([1.0, 0.0], dtype=dtype)
        )
        result = tf.intersects(line1, line3)
        print(f"    Parallel lines: {result}")
        assert result == False, f"Parallel lines should not intersect ({dtype_name})"

    print("✓ Line to Line tests passed")


def test_intersects_plane_primitives():
    """Test intersects between plane and various primitives (3D only)"""
    print("\n=== Test: Plane to Primitives ===")

    for dtype in [np.float32, np.float64]:
        dtype_name = "float32" if dtype == np.float32 else "float64"
        print(f"\n  Testing with {dtype_name}:")

        # Plane at z=0
        plane = tf.Plane(np.array([0.0, 0.0, 1.0, 0.0], dtype=dtype))

        # Point on plane
        pt_on = tf.Point(np.array([1.0, 1.0, 0.0], dtype=dtype))
        result = tf.intersects(plane, pt_on)
        print(f"    Point on plane: {result}")
        assert result == True, f"Point on plane should intersect ({dtype_name})"

        # Point off plane
        pt_off = tf.Point(np.array([1.0, 1.0, 5.0], dtype=dtype))
        result = tf.intersects(plane, pt_off)
        print(f"    Point off plane: {result}")
        assert result == False, f"Point off plane should not intersect ({dtype_name})"

        # Ray hitting plane
        ray = tf.Ray(
            origin=np.array([0.0, 0.0, 2.0], dtype=dtype),
            direction=np.array([0.0, 0.0, -1.0], dtype=dtype)
        )
        result = tf.intersects(plane, ray)
        print(f"    Ray hitting plane: {result}")
        assert result == True, f"Ray should intersect plane ({dtype_name})"

        # Ray parallel to plane
        ray_parallel = tf.Ray(
            origin=np.array([0.0, 0.0, 2.0], dtype=dtype),
            direction=np.array([1.0, 0.0, 0.0], dtype=dtype)
        )
        result = tf.intersects(plane, ray_parallel)
        print(f"    Ray parallel to plane: {result}")
        assert result == False, f"Parallel ray should not intersect plane ({dtype_name})"

        # Test swap with AABB
        box = tf.AABB(
            min=np.array([-1.0, -1.0, -1.0], dtype=dtype),
            max=np.array([1.0, 1.0, 1.0], dtype=dtype)
        )
        result = tf.intersects(plane, box)
        print(f"    Plane to AABB: {result}")
        assert result == True, f"Plane should intersect AABB ({dtype_name})"

        result = tf.intersects(box, plane)
        print(f"    AABB to plane (swapped): {result}")
        assert result == True, f"Swapped order should work ({dtype_name})"

    print("✓ Plane to Primitives tests passed")


def test_intersects_dimension_mismatch():
    """Test that dimension mismatch raises an error"""
    print("\n=== Test: Dimension Mismatch Validation ===")

    pt_2d = tf.Point([0.0, 0.0])
    pt_3d = tf.Point([0.0, 0.0, 0.0])

    try:
        result = tf.intersects(pt_2d, pt_3d)
        print("✗ ERROR: Should have raised ValueError for dimension mismatch!")
        assert False, "Expected ValueError was not raised"
    except ValueError as e:
        print(f"  Correctly raised ValueError: {e}")
        assert "Dimension mismatch" in str(e)

    print("✓ Dimension mismatch validation tests passed")


if __name__ == "__main__":
    print("Testing intersects functionality\n")
    print("=" * 60)

    try:
        test_intersects_point_point()
        test_intersects_point_aabb()
        test_intersects_aabb_aabb()
        test_intersects_point_segment()
        test_intersects_segment_segment()
        test_intersects_point_polygon()
        test_intersects_ray_segment()
        test_intersects_ray_polygon()
        test_intersects_line_line()
        test_intersects_plane_primitives()
        test_intersects_dimension_mismatch()

        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED!")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
