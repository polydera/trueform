"""
Test ray_cast functionality

Copyright (c) 2025 Žiga Sajovic, XLAB
"""

import numpy as np
import trueform as tf


def test_ray_plane_3d():
    """Test ray casting against plane in 3D"""
    print("\n=== Test: Ray to Plane (3D) ===")

    for dtype in [np.float32, np.float64]:
        dtype_name = "float32" if dtype == np.float32 else "float64"
        print(f"\n  Testing with {dtype_name}:")

        # Plane at z=0 (xy-plane)
        plane = tf.Plane(np.array([0, 0, 1, 0], dtype=dtype))

        # Ray pointing down from above - should hit
        ray_hit = tf.Ray(
            origin=np.array([0.5, 0.5, 2.0], dtype=dtype),
            direction=np.array([0.0, 0.0, -1.0], dtype=dtype)
        )
        t = tf.ray_cast(ray_hit, plane)
        print(f"    Ray pointing down: t = {t}")
        assert t is not None, f"Expected intersection ({dtype_name})"
        assert np.isclose(t, 2.0), f"Expected t=2.0, got {t} ({dtype_name})"

        # Verify hit point
        hit_point = ray_hit.origin + t * ray_hit.direction
        assert np.isclose(hit_point[2], 0.0), f"Hit point should be on plane ({dtype_name})"

        # Ray pointing away - should not hit
        ray_miss = tf.Ray(
            origin=np.array([0.5, 0.5, 2.0], dtype=dtype),
            direction=np.array([0.0, 0.0, 1.0], dtype=dtype)
        )
        t = tf.ray_cast(ray_miss, plane)
        print(f"    Ray pointing away: t = {t}")
        assert t is None, f"Expected no intersection ({dtype_name})"

        # Ray parallel to plane - should not hit
        ray_parallel = tf.Ray(
            origin=np.array([0.5, 0.5, 2.0], dtype=dtype),
            direction=np.array([1.0, 0.0, 0.0], dtype=dtype)
        )
        t = tf.ray_cast(ray_parallel, plane)
        print(f"    Ray parallel: t = {t}")
        assert t is None, f"Expected no intersection for parallel ray ({dtype_name})"

    print("✓ Ray to Plane 3D tests passed")


def test_ray_polygon_2d():
    """Test ray casting against polygon in 2D"""
    print("\n=== Test: Ray to Polygon (2D) ===")

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

        # Ray from left pointing right - should hit
        ray_hit = tf.Ray(
            origin=np.array([-1.0, 0.5], dtype=dtype),
            direction=np.array([1.0, 0.0], dtype=dtype)
        )
        t = tf.ray_cast(ray_hit, poly)
        print(f"    Ray from left: t = {t}")
        assert t is not None, f"Expected intersection ({dtype_name})"
        assert np.isclose(t, 1.0), f"Expected t=1.0, got {t} ({dtype_name})"

        # Ray from right pointing away - should not hit
        ray_miss = tf.Ray(
            origin=np.array([2.0, 0.5], dtype=dtype),
            direction=np.array([1.0, 0.0], dtype=dtype)
        )
        t = tf.ray_cast(ray_miss, poly)
        print(f"    Ray pointing away: t = {t}")
        assert t is None, f"Expected no intersection ({dtype_name})"

        # Ray starting inside - should hit
        ray_inside = tf.Ray(
            origin=np.array([0.5, 0.5], dtype=dtype),
            direction=np.array([1.0, 0.0], dtype=dtype)
        )
        t = tf.ray_cast(ray_inside, poly)
        print(f"    Ray from inside: t = {t}")
        assert t is not None, f"Expected intersection from inside ({dtype_name})"

    print("✓ Ray to Polygon 2D tests passed")


def test_ray_polygon_3d():
    """Test ray casting against polygon in 3D"""
    print("\n=== Test: Ray to Polygon (3D) ===")

    for dtype in [np.float32, np.float64]:
        dtype_name = "float32" if dtype == np.float32 else "float64"
        print(f"\n  Testing with {dtype_name}:")

        # Triangle in xy-plane at z=0
        triangle = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0]
        ], dtype=dtype)
        poly = tf.Polygon(triangle)

        # Ray pointing down from above - should hit
        ray_hit = tf.Ray(
            origin=np.array([0.5, 0.3, 2.0], dtype=dtype),
            direction=np.array([0.0, 0.0, -1.0], dtype=dtype)
        )
        t = tf.ray_cast(ray_hit, poly)
        print(f"    Ray from above: t = {t}")
        assert t is not None, f"Expected intersection ({dtype_name})"
        assert np.isclose(t, 2.0), f"Expected t=2.0, got {t} ({dtype_name})"

        # Ray from above but offset (outside triangle) - should not hit
        ray_miss = tf.Ray(
            origin=np.array([2.0, 2.0, 2.0], dtype=dtype),
            direction=np.array([0.0, 0.0, -1.0], dtype=dtype)
        )
        t = tf.ray_cast(ray_miss, poly)
        print(f"    Ray outside triangle: t = {t}")
        assert t is None, f"Expected no intersection ({dtype_name})"

    print("✓ Ray to Polygon 3D tests passed")


def test_ray_segment_2d():
    """Test ray casting against segment in 2D"""
    print("\n=== Test: Ray to Segment (2D) ===")

    for dtype in [np.float32, np.float64]:
        dtype_name = "float32" if dtype == np.float32 else "float64"
        print(f"\n  Testing with {dtype_name}:")

        # Vertical segment
        segment = tf.Segment(np.array([[1.0, 0.0], [1.0, 2.0]], dtype=dtype))

        # Ray from left pointing right - should hit
        ray_hit = tf.Ray(
            origin=np.array([0.0, 1.0], dtype=dtype),
            direction=np.array([1.0, 0.0], dtype=dtype)
        )
        t = tf.ray_cast(ray_hit, segment)
        print(f"    Ray hitting segment: t = {t}")
        assert t is not None, f"Expected intersection ({dtype_name})"
        assert np.isclose(t, 1.0), f"Expected t=1.0, got {t} ({dtype_name})"

        # Ray from left but above segment - should not hit
        ray_miss = tf.Ray(
            origin=np.array([0.0, 3.0], dtype=dtype),
            direction=np.array([1.0, 0.0], dtype=dtype)
        )
        t = tf.ray_cast(ray_miss, segment)
        print(f"    Ray missing segment: t = {t}")
        assert t is None, f"Expected no intersection ({dtype_name})"

        # Ray pointing away - should not hit
        ray_away = tf.Ray(
            origin=np.array([0.0, 1.0], dtype=dtype),
            direction=np.array([-1.0, 0.0], dtype=dtype)
        )
        t = tf.ray_cast(ray_away, segment)
        print(f"    Ray pointing away: t = {t}")
        assert t is None, f"Expected no intersection ({dtype_name})"

    print("✓ Ray to Segment 2D tests passed")


def test_ray_segment_3d():
    """Test ray casting against segment in 3D"""
    print("\n=== Test: Ray to Segment (3D) ===")

    for dtype in [np.float32, np.float64]:
        dtype_name = "float32" if dtype == np.float32 else "float64"
        print(f"\n  Testing with {dtype_name}:")

        # Segment along x-axis from (0, 0.5, 0.5) to (2, 0.5, 0.5)
        segment = tf.Segment(np.array([[0.0, 0.5, 0.5], [2.0, 0.5, 0.5]], dtype=dtype))

        # Ray from below pointing up - should hit at y=0.5
        ray_hit = tf.Ray(
            origin=np.array([1.0, 0.0, 0.5], dtype=dtype),
            direction=np.array([0.0, 1.0, 0.0], dtype=dtype)
        )
        t = tf.ray_cast(ray_hit, segment)
        print(f"    Ray hitting segment: t = {t}")
        assert t is not None, f"Expected intersection ({dtype_name})"
        assert np.isclose(t, 0.5), f"Expected t=0.5, got {t} ({dtype_name})"

        # Ray parallel but offset - should not hit
        ray_miss = tf.Ray(
            origin=np.array([0.0, 1.5, 0.5], dtype=dtype),
            direction=np.array([1.0, 0.0, 0.0], dtype=dtype)
        )
        t = tf.ray_cast(ray_miss, segment)
        print(f"    Ray parallel but offset: t = {t}")
        assert t is None, f"Expected no intersection ({dtype_name})"

    print("✓ Ray to Segment 3D tests passed")


def test_ray_line_2d():
    """Test ray casting against line in 2D"""
    print("\n=== Test: Ray to Line (2D) ===")

    for dtype in [np.float32, np.float64]:
        dtype_name = "float32" if dtype == np.float32 else "float64"
        print(f"\n  Testing with {dtype_name}:")

        # Vertical line at x=1
        line = tf.Line(
            origin=np.array([1.0, 0.0], dtype=dtype),
            direction=np.array([0.0, 1.0], dtype=dtype)
        )

        # Ray from left pointing right - should hit
        ray_hit = tf.Ray(
            origin=np.array([0.0, 0.5], dtype=dtype),
            direction=np.array([1.0, 0.0], dtype=dtype)
        )
        t = tf.ray_cast(ray_hit, line)
        print(f"    Ray hitting line: t = {t}")
        assert t is not None, f"Expected intersection ({dtype_name})"
        assert np.isclose(t, 1.0), f"Expected t=1.0, got {t} ({dtype_name})"

        # Ray parallel to line - should not hit
        ray_parallel = tf.Ray(
            origin=np.array([0.0, 0.5], dtype=dtype),
            direction=np.array([0.0, 1.0], dtype=dtype)
        )
        t = tf.ray_cast(ray_parallel, line)
        print(f"    Ray parallel: t = {t}")
        assert t is None, f"Expected no intersection for parallel rays ({dtype_name})"

        # Ray pointing away - should not hit
        ray_away = tf.Ray(
            origin=np.array([0.0, 0.5], dtype=dtype),
            direction=np.array([-1.0, 0.0], dtype=dtype)
        )
        t = tf.ray_cast(ray_away, line)
        print(f"    Ray pointing away: t = {t}")
        assert t is None, f"Expected no intersection ({dtype_name})"

    print("✓ Ray to Line 2D tests passed")


def test_ray_line_3d():
    """Test ray casting against line in 3D"""
    print("\n=== Test: Ray to Line (3D) ===")

    for dtype in [np.float32, np.float64]:
        dtype_name = "float32" if dtype == np.float32 else "float64"
        print(f"\n  Testing with {dtype_name}:")

        # Line along z-axis through origin
        line = tf.Line(
            origin=np.array([0.0, 0.0, 0.0], dtype=dtype),
            direction=np.array([0.0, 0.0, 1.0], dtype=dtype)
        )

        # Ray in xy-plane pointing at line - should hit
        ray_hit = tf.Ray(
            origin=np.array([1.0, 0.0, 0.5], dtype=dtype),
            direction=np.array([-1.0, 0.0, 0.0], dtype=dtype)
        )
        t = tf.ray_cast(ray_hit, line)
        print(f"    Ray hitting line: t = {t}")
        assert t is not None, f"Expected intersection ({dtype_name})"
        assert np.isclose(t, 1.0), f"Expected t=1.0, got {t} ({dtype_name})"

        # Ray skew to line (non-intersecting in 3D) - should not hit
        ray_skew = tf.Ray(
            origin=np.array([1.0, 1.0, 0.0], dtype=dtype),
            direction=np.array([1.0, 0.0, 0.0], dtype=dtype)
        )
        t = tf.ray_cast(ray_skew, line)
        print(f"    Ray skew to line: t = {t}")
        assert t is None, f"Expected no intersection for skew rays ({dtype_name})"

    print("✓ Ray to Line 3D tests passed")


def test_ray_aabb_2d():
    """Test ray casting against AABB in 2D"""
    print("\n=== Test: Ray to AABB (2D) ===")

    for dtype in [np.float32, np.float64]:
        dtype_name = "float32" if dtype == np.float32 else "float64"
        print(f"\n  Testing with {dtype_name}:")

        # AABB from [0,0] to [1,1]
        aabb = tf.AABB(
            min=np.array([0.0, 0.0], dtype=dtype),
            max=np.array([1.0, 1.0], dtype=dtype)
        )

        # Ray from left pointing right - should hit
        ray_hit = tf.Ray(
            origin=np.array([-1.0, 0.5], dtype=dtype),
            direction=np.array([1.0, 0.0], dtype=dtype)
        )
        t = tf.ray_cast(ray_hit, aabb)
        print(f"    Ray from left: t = {t}")
        assert t is not None, f"Expected intersection ({dtype_name})"
        assert np.isclose(t, 1.0), f"Expected t=1.0, got {t} ({dtype_name})"

        # Ray from right pointing away - should not hit
        ray_miss = tf.Ray(
            origin=np.array([2.0, 0.5], dtype=dtype),
            direction=np.array([1.0, 0.0], dtype=dtype)
        )
        t = tf.ray_cast(ray_miss, aabb)
        print(f"    Ray pointing away: t = {t}")
        assert t is None, f"Expected no intersection ({dtype_name})"

        # Ray starting inside - should hit at t=0
        ray_inside = tf.Ray(
            origin=np.array([0.5, 0.5], dtype=dtype),
            direction=np.array([1.0, 0.0], dtype=dtype)
        )
        t = tf.ray_cast(ray_inside, aabb)
        print(f"    Ray from inside: t = {t}")
        assert t is not None, f"Expected intersection ({dtype_name})"

    print("✓ Ray to AABB 2D tests passed")


def test_ray_aabb_3d():
    """Test ray casting against AABB in 3D"""
    print("\n=== Test: Ray to AABB (3D) ===")

    for dtype in [np.float32, np.float64]:
        dtype_name = "float32" if dtype == np.float32 else "float64"
        print(f"\n  Testing with {dtype_name}:")

        # AABB cube from [0,0,0] to [1,1,1]
        aabb = tf.AABB(
            min=np.array([0.0, 0.0, 0.0], dtype=dtype),
            max=np.array([1.0, 1.0, 1.0], dtype=dtype)
        )

        # Ray from above pointing down - should hit
        ray_hit = tf.Ray(
            origin=np.array([0.5, 0.5, 2.0], dtype=dtype),
            direction=np.array([0.0, 0.0, -1.0], dtype=dtype)
        )
        t = tf.ray_cast(ray_hit, aabb)
        print(f"    Ray from above: t = {t}")
        assert t is not None, f"Expected intersection ({dtype_name})"
        assert np.isclose(t, 1.0), f"Expected t=1.0, got {t} ({dtype_name})"

        # Ray from above but outside AABB - should not hit
        ray_miss = tf.Ray(
            origin=np.array([2.0, 2.0, 2.0], dtype=dtype),
            direction=np.array([0.0, 0.0, -1.0], dtype=dtype)
        )
        t = tf.ray_cast(ray_miss, aabb)
        print(f"    Ray outside AABB: t = {t}")
        assert t is None, f"Expected no intersection ({dtype_name})"

        # Ray pointing away - should not hit
        ray_away = tf.Ray(
            origin=np.array([0.5, 0.5, 2.0], dtype=dtype),
            direction=np.array([0.0, 0.0, 1.0], dtype=dtype)
        )
        t = tf.ray_cast(ray_away, aabb)
        print(f"    Ray pointing away: t = {t}")
        assert t is None, f"Expected no intersection ({dtype_name})"

    print("✓ Ray to AABB 3D tests passed")


def test_dimension_mismatch():
    """Test that dimension mismatch raises an error"""
    print("\n=== Test: Dimension Mismatch Validation ===")

    ray_2d = tf.Ray(origin=[0.0, 0.0], direction=[1.0, 0.0])
    segment_3d = tf.Segment([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

    try:
        t = tf.ray_cast(ray_2d, segment_3d)
        print("✗ ERROR: Should have raised ValueError for dimension mismatch!")
        assert False, "Expected ValueError was not raised"
    except ValueError as e:
        print(f"  Correctly raised ValueError: {e}")
        assert "Dimension mismatch" in str(e)

    print("✓ Dimension mismatch validation tests passed")


if __name__ == "__main__":
    print("Testing ray_cast functionality\n")
    print("=" * 60)

    try:
        test_ray_plane_3d()
        test_ray_polygon_2d()
        test_ray_polygon_3d()
        test_ray_segment_2d()
        test_ray_segment_3d()
        test_ray_line_2d()
        test_ray_line_3d()
        test_ray_aabb_2d()
        test_ray_aabb_3d()
        test_dimension_mismatch()

        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED!")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
