"""
Test closest_metric_point_pair functionality

Copyright (c) 2025 Žiga Sajovic, XLAB
"""

import sys
import os

# Add parent directory to path so we can import trueform
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import trueform as tf


def test_point_polygon_2d():
    """Test closest metric point pair between point and polygon in 2D"""
    print("\n=== Test: Point to Polygon (2D) ===")

    # Create a square polygon centered at origin
    square = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0]
    ], dtype=np.float32)

    # Point inside the polygon - distance should be 0
    pt_inside = tf.Point([0.5, 0.5])
    poly = tf.Polygon(square)

    dist2, p0, p1 = tf.closest_metric_point_pair(pt_inside, poly)
    print(f"Point inside polygon: {pt_inside.data}")
    print(f"  Distance²: {dist2}")
    print(f"  Point on point: {p0}")
    print(f"  Point on polygon: {p1}")
    print(f"  Expected distance²: 0.0")
    assert dist2 == 0.0, f"Expected 0.0, got {dist2}"

    # Point outside the polygon
    pt_outside = tf.Point([2.0, 0.5])
    dist2, p0, p1 = tf.closest_metric_point_pair(pt_outside, poly)
    print(f"\nPoint outside polygon: {pt_outside.data}")
    print(f"  Distance²: {dist2}")
    print(f"  Point on point: {p0}")
    print(f"  Point on polygon: {p1}")
    print(f"  Expected distance²: 1.0 (distance from [2,0.5] to [1,0.5])")
    assert np.isclose(dist2, 1.0), f"Expected 1.0, got {dist2}"
    assert np.allclose(p1, [1.0, 0.5], atol=1e-5), f"Expected [1.0, 0.5], got {p1}"

    print("✓ Point to Polygon 2D tests passed")


def test_point_polygon_3d():
    """Test closest metric point pair between point and polygon in 3D"""
    print("\n=== Test: Point to Polygon (3D) ===")

    # Create a triangle in the xy-plane at z=0
    triangle = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 1.0, 0.0]
    ], dtype=np.float64)

    # Point inside the triangle on the same plane
    pt_inside = tf.Point([0.5, 0.3, 0.0])
    poly = tf.Polygon(triangle)

    dist2, p0, p1 = tf.closest_metric_point_pair(pt_inside, poly)
    print(f"Point inside triangle: {pt_inside.data}")
    print(f"  Distance²: {dist2}")
    print(f"  Point on point: {p0}")
    print(f"  Point on polygon: {p1}")
    print(f"  Expected distance²: 0.0")
    assert np.isclose(0, dist2)

    # Point above the triangle
    pt_above = tf.Point([0.5, 0.3, 2.0])
    dist2, p0, p1 = tf.closest_metric_point_pair(pt_above, poly)
    print(f"\nPoint above triangle: {pt_above.data}")
    print(f"  Distance²: {dist2}")
    print(f"  Point on point: {p0}")
    print(f"  Point on polygon: {p1}")
    print(f"  Expected distance²: 4.0 (vertical distance = 2.0)")
    assert np.isclose(dist2, 4.0), f"Expected 4.0, got {dist2}"
    assert np.allclose(p1, [0.5, 0.3, 0.0], atol=1e-5), f"Expected [0.5, 0.3, 0.0], got {p1}"

    print("✓ Point to Polygon 3D tests passed")


def test_polygon_polygon_2d():
    """Test closest metric point pair between two polygons in 2D"""
    print("\n=== Test: Polygon to Polygon (2D) ===")

    # Two squares that don't overlap
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
    print(f"Square 1: corners at [0,0], [1,0], [1,1], [0,1]")
    print(f"Square 2: corners at [2,0], [3,0], [3,1], [2,1]")
    print(f"  Distance²: {dist2}")
    print(f"  Point on polygon1: {p0}")
    print(f"  Point on polygon2: {p1}")
    print(f"  Expected distance²: 1.0 (horizontal gap from [1,y] to [2,y])")
    assert np.isclose(dist2, 1.0), f"Expected 1.0, got {dist2}"

    # Two overlapping squares
    square3 = np.array([
        [0.5, 0.5],
        [1.5, 0.5],
        [1.5, 1.5],
        [0.5, 1.5]
    ], dtype=np.float32)

    poly3 = tf.Polygon(square3)
    dist2, p0, p1 = tf.closest_metric_point_pair(poly1, poly3)
    print(f"\nOverlapping squares:")
    print(f"  Distance²: {dist2}")
    print(f"  Point on polygon1: {p0}")
    print(f"  Point on polygon3: {p1}")
    print(f"  Expected distance²: 0.0 (polygons overlap)")
    assert dist2 == 0.0, f"Expected 0.0, got {dist2}"

    print("✓ Polygon to Polygon 2D tests passed")


def test_polygon_polygon_3d():
    """Test closest metric point pair between two polygons in 3D"""
    print("\n=== Test: Polygon to Polygon (3D) ===")

    # Two triangles in parallel planes
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
    print(f"Triangle 1 at z=0")
    print(f"Triangle 2 at z=2 (parallel, directly above)")
    print(f"  Distance²: {dist2}")
    print(f"  Point on polygon1: {p0}")
    print(f"  Point on polygon2: {p1}")
    print(f"  Expected distance²: 4.0 (vertical distance = 2.0)")
    assert np.isclose(dist2, 4.0), f"Expected 4.0, got {dist2}"

    print("✓ Polygon to Polygon 3D tests passed")


def test_segment_polygon_2d():
    """Test closest metric point pair between segment and polygon in 2D"""
    print("\n=== Test: Segment to Polygon (2D) ===")

    # Square polygon
    square = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0]
    ], dtype=np.float32)

    # Segment that intersects the square
    seg_intersect = tf.Segment([[0.5, -0.5], [0.5, 1.5]])
    poly = tf.Polygon(square)

    dist2, p0, p1 = tf.closest_metric_point_pair(seg_intersect, poly)
    print(f"Segment intersecting square:")
    print(f"  Distance²: {dist2}")
    print(f"  Point on segment: {p0}")
    print(f"  Point on polygon: {p1}")
    print(f"  Expected distance²: 0.0 (segment intersects polygon)")
    assert dist2 == 0.0, f"Expected 0.0, got {dist2}"

    # Segment outside the square
    seg_outside = tf.Segment([[2.0, 0.0], [3.0, 0.0]])
    dist2, p0, p1 = tf.closest_metric_point_pair(seg_outside, poly)
    print(f"\nSegment outside square:")
    print(f"  Distance²: {dist2}")
    print(f"  Point on segment: {p0}")
    print(f"  Point on polygon: {p1}")
    print(f"  Expected distance²: 1.0")
    assert np.isclose(dist2, 1.0), f"Expected 1.0, got {dist2}"

    print("✓ Segment to Polygon 2D tests passed")


def test_ray_polygon_3d():
    """Test closest metric point pair between ray and polygon in 3D"""
    print("\n=== Test: Ray to Polygon (3D) ===")

    # Triangle in xy-plane at z=0
    triangle = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 1.0, 0.0]
    ], dtype=np.float32)

    # Ray pointing down that hits the triangle
    ray_hit = tf.Ray(origin=[0.5, 0.3, 2.0], direction=[0.0, 0.0, -1.0])
    poly = tf.Polygon(triangle)

    dist2, p0, p1 = tf.closest_metric_point_pair(ray_hit, poly)
    print(f"Ray hitting triangle from above:")
    print(f"  Distance²: {dist2}")
    print(f"  Point on ray: {p0}")
    print(f"  Point on polygon: {p1}")
    print(f"  Expected distance²: 0.0 (ray intersects triangle)")
    assert dist2 == 0.0, f"Expected 0.0, got {dist2}"

    # Ray pointing away from triangle
    ray_miss = tf.Ray(origin=[0.5, 0.3, 2.0], direction=[0.0, 0.0, 1.0])
    dist2, p0, p1 = tf.closest_metric_point_pair(ray_miss, poly)
    print(f"\nRay pointing away from triangle:")
    print(f"  Distance²: {dist2}")
    print(f"  Point on ray: {p0}")
    print(f"  Point on polygon: {p1}")
    print(f"  Expected distance²: 4.0 (distance from ray origin to triangle)")
    assert np.isclose(dist2, 4.0), f"Expected 4.0, got {dist2}"

    print("✓ Ray to Polygon 3D tests passed")


def test_line_polygon_2d():
    """Test closest metric point pair between line and polygon in 2D"""
    print("\n=== Test: Line to Polygon (2D) ===")

    # Square polygon
    square = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0]
    ], dtype=np.float32)

    # Line that intersects the square
    line_intersect = tf.Line(origin=[0.5, -1.0], direction=[0.0, 1.0])
    poly = tf.Polygon(square)

    dist2, p0, p1 = tf.closest_metric_point_pair(line_intersect, poly)
    print(f"Line intersecting square:")
    print(f"  Distance²: {dist2}")
    print(f"  Point on line: {p0}")
    print(f"  Point on polygon: {p1}")
    print(f"  Expected distance²: 0.0 (line intersects polygon)")
    assert dist2 == 0.0, f"Expected 0.0, got {dist2}"

    # Line parallel to square but outside
    line_parallel = tf.Line(origin=[2.0, 0.0], direction=[0.0, 1.0])
    dist2, p0, p1 = tf.closest_metric_point_pair(line_parallel, poly)
    print(f"\nLine parallel to square (outside):")
    print(f"  Distance²: {dist2}")
    print(f"  Point on line: {p0}")
    print(f"  Point on polygon: {p1}")
    print(f"  Expected distance²: 1.0")
    assert np.isclose(dist2, 1.0), f"Expected 1.0, got {dist2}"

    print("✓ Line to Polygon 2D tests passed")


def test_dimension_mismatch():
    """Test that dimension mismatch raises an error"""
    print("\n=== Test: Dimension Mismatch Validation ===")

    # Create 2D and 3D points
    pt_2d = tf.Point([0.0, 0.0])
    pt_3d = tf.Point([0.0, 0.0, 0.0])

    print(f"Point 2D: {pt_2d.data}")
    print(f"Point 3D: {pt_3d.data}")

    try:
        dist2, p0, p1 = tf.closest_metric_point_pair(pt_2d, pt_3d)
        print("✗ ERROR: Should have raised ValueError for dimension mismatch!")
        assert False, "Expected ValueError was not raised"
    except ValueError as e:
        print(f"  Correctly raised ValueError: {e}")
        assert "Dimension mismatch" in str(e)
        assert "2D" in str(e)
        assert "3D" in str(e)

    # Test with polygon and point
    triangle_2d = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, 1.0]
    ], dtype=np.float32)
    poly_2d = tf.Polygon(triangle_2d)

    try:
        dist2, p0, p1 = tf.closest_metric_point_pair(pt_3d, poly_2d)
        print("✗ ERROR: Should have raised ValueError for dimension mismatch!")
        assert False, "Expected ValueError was not raised"
    except ValueError as e:
        print(f"  Correctly raised ValueError: {e}")
        assert "Dimension mismatch" in str(e)

    print("✓ Dimension mismatch validation tests passed")


def test_point_plane():
    """Test closest_metric_point_pair between point and plane"""
    print("\n=== Test: Point to Plane ===")

    for dtype in [np.float32, np.float64]:
        dtype_name = "float32" if dtype == np.float32 else "float64"
        print(f"\n  Testing with {dtype_name}:")

        # Plane at z=0 (normal = [0,0,1], d=0)
        plane = tf.Plane(np.array([0.0, 0.0, 1.0, 0.0], dtype=dtype))

        # Point on plane
        pt_on = tf.Point(np.array([1.0, 2.0, 0.0], dtype=dtype))
        dist2, p0, p1 = tf.closest_metric_point_pair(pt_on, plane)
        print(f"    Point on plane: dist2={dist2}, p0={p0}, p1={p1}")
        assert abs(dist2) < 1e-5, f"Distance should be 0 ({dtype_name})"
        assert np.allclose(p0, pt_on.data), f"p0 should be the original point ({dtype_name})"
        assert np.allclose(p1, pt_on.data), f"p1 should be on plane ({dtype_name})"

        # Point above plane
        pt_above = tf.Point(np.array([1.0, 2.0, 5.0], dtype=dtype))
        dist2, p0, p1 = tf.closest_metric_point_pair(pt_above, plane)
        print(f"    Point above plane: dist2={dist2}, p0={p0}, p1={p1}")
        assert abs(dist2 - 25.0) < 1e-4, f"Distance2 should be 25 ({dtype_name})"
        assert np.allclose(p0, pt_above.data), f"p0 should be the original point ({dtype_name})"
        assert np.allclose(p1, [1.0, 2.0, 0.0]), f"p1 should be projected on plane ({dtype_name})"

        # Test swap order
        dist2_swap, p0_swap, p1_swap = tf.closest_metric_point_pair(plane, pt_above)
        print(f"    Plane to point (swapped): dist2={dist2_swap}")
        assert abs(dist2_swap - 25.0) < 1e-4, f"Swapped distance should match ({dtype_name})"
        # Note: After swap, p0 should be on plane, p1 should be the point
        assert np.allclose(p0_swap, [1.0, 2.0, 0.0]), f"p0 should be on plane after swap ({dtype_name})"
        assert np.allclose(p1_swap, pt_above.data), f"p1 should be the point after swap ({dtype_name})"

    print("✓ Point to Plane tests passed")


def test_segment_plane():
    """Test closest_metric_point_pair between segment and plane"""
    print("\n=== Test: Segment to Plane ===")

    for dtype in [np.float32, np.float64]:
        dtype_name = "float32" if dtype == np.float32 else "float64"
        print(f"\n  Testing with {dtype_name}:")

        # Plane at z=0
        plane = tf.Plane(np.array([0.0, 0.0, 1.0, 0.0], dtype=dtype))

        # Segment parallel to plane above it
        seg = tf.Segment(np.array([[0.0, 0.0, 3.0], [1.0, 0.0, 3.0]], dtype=dtype))
        dist2, p0, p1 = tf.closest_metric_point_pair(seg, plane)
        print(f"    Segment parallel above: dist2={dist2}, p0={p0}, p1={p1}")
        assert abs(dist2 - 9.0) < 1e-4, f"Distance2 should be 9 ({dtype_name})"

        # Segment crossing plane
        seg_cross = tf.Segment(np.array([[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]], dtype=dtype))
        dist2, p0, p1 = tf.closest_metric_point_pair(seg_cross, plane)
        print(f"    Segment crossing plane: dist2={dist2}")
        assert abs(dist2) < 1e-5, f"Distance should be 0 for intersecting segment ({dtype_name})"

    print("✓ Segment to Plane tests passed")


def test_ray_plane():
    """Test closest_metric_point_pair between ray and plane"""
    print("\n=== Test: Ray to Plane ===")

    for dtype in [np.float32, np.float64]:
        dtype_name = "float32" if dtype == np.float32 else "float64"
        print(f"\n  Testing with {dtype_name}:")

        # Plane at z=0
        plane = tf.Plane(np.array([0.0, 0.0, 1.0, 0.0], dtype=dtype))

        # Ray pointing toward plane
        ray = tf.Ray(
            origin=np.array([0.0, 0.0, 5.0], dtype=dtype),
            direction=np.array([0.0, 0.0, -1.0], dtype=dtype)
        )
        dist2, p0, p1 = tf.closest_metric_point_pair(ray, plane)
        print(f"    Ray toward plane: dist2={dist2}")
        assert abs(dist2) < 1e-5, f"Distance should be 0 for ray hitting plane ({dtype_name})"

        # Ray parallel to plane
        ray_parallel = tf.Ray(
            origin=np.array([0.0, 0.0, 5.0], dtype=dtype),
            direction=np.array([1.0, 0.0, 0.0], dtype=dtype)
        )
        dist2, p0, p1 = tf.closest_metric_point_pair(ray_parallel, plane)
        print(f"    Ray parallel to plane: dist2={dist2}")
        assert abs(dist2 - 25.0) < 1e-4, f"Distance2 should be 25 ({dtype_name})"

    print("✓ Ray to Plane tests passed")


def test_line_plane():
    """Test closest_metric_point_pair between line and plane"""
    print("\n=== Test: Line to Plane ===")

    for dtype in [np.float32, np.float64]:
        dtype_name = "float32" if dtype == np.float32 else "float64"
        print(f"\n  Testing with {dtype_name}:")

        # Plane at z=0
        plane = tf.Plane(np.array([0.0, 0.0, 1.0, 0.0], dtype=dtype))

        # Line intersecting plane
        line = tf.Line(
            origin=np.array([0.0, 0.0, 5.0], dtype=dtype),
            direction=np.array([0.0, 0.0, 1.0], dtype=dtype)
        )
        dist2, p0, p1 = tf.closest_metric_point_pair(line, plane)
        print(f"    Line intersecting plane: dist2={dist2}")
        assert abs(dist2) < 1e-5, f"Distance should be 0 for line intersecting plane ({dtype_name})"

        # Line parallel to plane
        line_parallel = tf.Line(
            origin=np.array([0.0, 0.0, 3.0], dtype=dtype),
            direction=np.array([1.0, 0.0, 0.0], dtype=dtype)
        )
        dist2, p0, p1 = tf.closest_metric_point_pair(line_parallel, plane)
        print(f"    Line parallel to plane: dist2={dist2}")
        assert abs(dist2 - 9.0) < 1e-4, f"Distance2 should be 9 ({dtype_name})"

    print("✓ Line to Plane tests passed")


if __name__ == "__main__":
    print("Testing closest_metric_point_pair functionality\n")
    print("=" * 60)

    try:
        test_dimension_mismatch()
        test_point_polygon_2d()
        test_point_polygon_3d()
        test_polygon_polygon_2d()
        test_polygon_polygon_3d()
        test_segment_polygon_2d()
        test_ray_polygon_3d()
        test_line_polygon_2d()
        test_point_plane()
        test_segment_plane()
        test_ray_plane()
        test_line_plane()

        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED!")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
