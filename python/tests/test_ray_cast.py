"""
Test ray_cast functionality

Copyright (c) 2025 Žiga Sajovic, XLAB
"""

import sys
import pytest
import numpy as np
import trueform as tf


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_ray_plane_3d(dtype):
    """Test ray casting against plane in 3D"""
    # Plane at z=0 (xy-plane)
    plane = tf.Plane(np.array([0, 0, 1, 0], dtype=dtype))

    # Ray pointing down from above - should hit
    ray_hit = tf.Ray(
        origin=np.array([0.5, 0.5, 2.0], dtype=dtype),
        direction=np.array([0.0, 0.0, -1.0], dtype=dtype)
    )
    t = tf.ray_cast(ray_hit, plane)
    assert t is not None, f"Expected intersection ({dtype})"
    assert np.isclose(t, 2.0), f"Expected t=2.0, got {t} ({dtype})"

    # Verify hit point
    hit_point = ray_hit.origin + t * ray_hit.direction
    assert np.isclose(hit_point[2], 0.0), f"Hit point should be on plane ({dtype})"

    # Ray pointing away - should not hit
    ray_miss = tf.Ray(
        origin=np.array([0.5, 0.5, 2.0], dtype=dtype),
        direction=np.array([0.0, 0.0, 1.0], dtype=dtype)
    )
    t = tf.ray_cast(ray_miss, plane)
    assert t is None, f"Expected no intersection ({dtype})"

    # Ray parallel to plane - should not hit
    ray_parallel = tf.Ray(
        origin=np.array([0.5, 0.5, 2.0], dtype=dtype),
        direction=np.array([1.0, 0.0, 0.0], dtype=dtype)
    )
    t = tf.ray_cast(ray_parallel, plane)
    assert t is None, f"Expected no intersection for parallel ray ({dtype})"


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_ray_polygon_2d(dtype):
    """Test ray casting against polygon in 2D"""
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
    assert t is not None, f"Expected intersection ({dtype})"
    assert np.isclose(t, 1.0), f"Expected t=1.0, got {t} ({dtype})"

    # Ray from right pointing away - should not hit
    ray_miss = tf.Ray(
        origin=np.array([2.0, 0.5], dtype=dtype),
        direction=np.array([1.0, 0.0], dtype=dtype)
    )
    t = tf.ray_cast(ray_miss, poly)
    assert t is None, f"Expected no intersection ({dtype})"

    # Ray starting inside - should hit
    ray_inside = tf.Ray(
        origin=np.array([0.5, 0.5], dtype=dtype),
        direction=np.array([1.0, 0.0], dtype=dtype)
    )
    t = tf.ray_cast(ray_inside, poly)
    assert t is not None, f"Expected intersection from inside ({dtype})"


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_ray_polygon_3d(dtype):
    """Test ray casting against polygon in 3D"""
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
    assert t is not None, f"Expected intersection ({dtype})"
    assert np.isclose(t, 2.0), f"Expected t=2.0, got {t} ({dtype})"

    # Ray from above but offset (outside triangle) - should not hit
    ray_miss = tf.Ray(
        origin=np.array([2.0, 2.0, 2.0], dtype=dtype),
        direction=np.array([0.0, 0.0, -1.0], dtype=dtype)
    )
    t = tf.ray_cast(ray_miss, poly)
    assert t is None, f"Expected no intersection ({dtype})"


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_ray_segment_2d(dtype):
    """Test ray casting against segment in 2D"""
    # Vertical segment
    segment = tf.Segment(np.array([[1.0, 0.0], [1.0, 2.0]], dtype=dtype))

    # Ray from left pointing right - should hit
    ray_hit = tf.Ray(
        origin=np.array([0.0, 1.0], dtype=dtype),
        direction=np.array([1.0, 0.0], dtype=dtype)
    )
    t = tf.ray_cast(ray_hit, segment)
    assert t is not None, f"Expected intersection ({dtype})"
    assert np.isclose(t, 1.0), f"Expected t=1.0, got {t} ({dtype})"

    # Ray from left but above segment - should not hit
    ray_miss = tf.Ray(
        origin=np.array([0.0, 3.0], dtype=dtype),
        direction=np.array([1.0, 0.0], dtype=dtype)
    )
    t = tf.ray_cast(ray_miss, segment)
    assert t is None, f"Expected no intersection ({dtype})"

    # Ray pointing away - should not hit
    ray_away = tf.Ray(
        origin=np.array([0.0, 1.0], dtype=dtype),
        direction=np.array([-1.0, 0.0], dtype=dtype)
    )
    t = tf.ray_cast(ray_away, segment)
    assert t is None, f"Expected no intersection ({dtype})"


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_ray_segment_3d(dtype):
    """Test ray casting against segment in 3D"""
    # Segment along x-axis from (0, 0.5, 0.5) to (2, 0.5, 0.5)
    segment = tf.Segment(np.array([[0.0, 0.5, 0.5], [2.0, 0.5, 0.5]], dtype=dtype))

    # Ray from below pointing up - should hit at y=0.5
    ray_hit = tf.Ray(
        origin=np.array([1.0, 0.0, 0.5], dtype=dtype),
        direction=np.array([0.0, 1.0, 0.0], dtype=dtype)
    )
    t = tf.ray_cast(ray_hit, segment)
    assert t is not None, f"Expected intersection ({dtype})"
    assert np.isclose(t, 0.5), f"Expected t=0.5, got {t} ({dtype})"

    # Ray parallel but offset - should not hit
    ray_miss = tf.Ray(
        origin=np.array([0.0, 1.5, 0.5], dtype=dtype),
        direction=np.array([1.0, 0.0, 0.0], dtype=dtype)
    )
    t = tf.ray_cast(ray_miss, segment)
    assert t is None, f"Expected no intersection ({dtype})"


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_ray_line_2d(dtype):
    """Test ray casting against line in 2D"""
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
    assert t is not None, f"Expected intersection ({dtype})"
    assert np.isclose(t, 1.0), f"Expected t=1.0, got {t} ({dtype})"

    # Ray parallel to line - should not hit
    ray_parallel = tf.Ray(
        origin=np.array([0.0, 0.5], dtype=dtype),
        direction=np.array([0.0, 1.0], dtype=dtype)
    )
    t = tf.ray_cast(ray_parallel, line)
    assert t is None, f"Expected no intersection for parallel rays ({dtype})"

    # Ray pointing away - should not hit
    ray_away = tf.Ray(
        origin=np.array([0.0, 0.5], dtype=dtype),
        direction=np.array([-1.0, 0.0], dtype=dtype)
    )
    t = tf.ray_cast(ray_away, line)
    assert t is None, f"Expected no intersection ({dtype})"


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_ray_line_3d(dtype):
    """Test ray casting against line in 3D"""
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
    assert t is not None, f"Expected intersection ({dtype})"
    assert np.isclose(t, 1.0), f"Expected t=1.0, got {t} ({dtype})"

    # Ray skew to line (non-intersecting in 3D) - should not hit
    ray_skew = tf.Ray(
        origin=np.array([1.0, 1.0, 0.0], dtype=dtype),
        direction=np.array([1.0, 0.0, 0.0], dtype=dtype)
    )
    t = tf.ray_cast(ray_skew, line)
    assert t is None, f"Expected no intersection for skew rays ({dtype})"


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_ray_aabb_2d(dtype):
    """Test ray casting against AABB in 2D"""
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
    assert t is not None, f"Expected intersection ({dtype})"
    assert np.isclose(t, 1.0), f"Expected t=1.0, got {t} ({dtype})"

    # Ray from right pointing away - should not hit
    ray_miss = tf.Ray(
        origin=np.array([2.0, 0.5], dtype=dtype),
        direction=np.array([1.0, 0.0], dtype=dtype)
    )
    t = tf.ray_cast(ray_miss, aabb)
    assert t is None, f"Expected no intersection ({dtype})"

    # Ray starting inside - should hit at t=0
    ray_inside = tf.Ray(
        origin=np.array([0.5, 0.5], dtype=dtype),
        direction=np.array([1.0, 0.0], dtype=dtype)
    )
    t = tf.ray_cast(ray_inside, aabb)
    assert t is not None, f"Expected intersection ({dtype})"


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_ray_aabb_3d(dtype):
    """Test ray casting against AABB in 3D"""
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
    assert t is not None, f"Expected intersection ({dtype})"
    assert np.isclose(t, 1.0), f"Expected t=1.0, got {t} ({dtype})"

    # Ray from above but outside AABB - should not hit
    ray_miss = tf.Ray(
        origin=np.array([2.0, 2.0, 2.0], dtype=dtype),
        direction=np.array([0.0, 0.0, -1.0], dtype=dtype)
    )
    t = tf.ray_cast(ray_miss, aabb)
    assert t is None, f"Expected no intersection ({dtype})"

    # Ray pointing away - should not hit
    ray_away = tf.Ray(
        origin=np.array([0.5, 0.5, 2.0], dtype=dtype),
        direction=np.array([0.0, 0.0, 1.0], dtype=dtype)
    )
    t = tf.ray_cast(ray_away, aabb)
    assert t is None, f"Expected no intersection ({dtype})"


def test_dimension_mismatch():
    """Test that dimension mismatch raises an error"""
    ray_2d = tf.Ray(origin=[0.0, 0.0], direction=[1.0, 0.0])
    segment_3d = tf.Segment([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

    try:
        _t = tf.ray_cast(ray_2d, segment_3d)
        assert False, "Expected ValueError was not raised"
    except ValueError as e:
        assert "Dimension mismatch" in str(e)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
