/**
 * @file test_intersects.cpp
 * @brief Tests for intersects functionality on core primitives
 *
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <trueform/core.hpp>
#include <array>

// =============================================================================
// Point-Point tests
// =============================================================================

TEMPLATE_TEST_CASE("point_point_intersects", "[core][intersects]",
    float, double)
{
    using real_t = TestType;

    SECTION("2D same point") {
        auto p0 = tf::make_point(real_t(1), real_t(2));
        auto p1 = tf::make_point(real_t(1), real_t(2));
        REQUIRE(tf::intersects(p0, p1));
    }

    SECTION("2D different points") {
        auto p0 = tf::make_point(real_t(0), real_t(0));
        auto p1 = tf::make_point(real_t(1), real_t(1));
        REQUIRE_FALSE(tf::intersects(p0, p1));
    }

    SECTION("3D same point") {
        auto p0 = tf::make_point(real_t(1), real_t(2), real_t(3));
        auto p1 = tf::make_point(real_t(1), real_t(2), real_t(3));
        REQUIRE(tf::intersects(p0, p1));
    }

    SECTION("3D different points") {
        auto p0 = tf::make_point(real_t(0), real_t(0), real_t(0));
        auto p1 = tf::make_point(real_t(1), real_t(1), real_t(1));
        REQUIRE_FALSE(tf::intersects(p0, p1));
    }
}

// =============================================================================
// Point-Segment tests
// =============================================================================

TEMPLATE_TEST_CASE("point_segment_intersects", "[core][intersects]",
    float, double)
{
    using real_t = TestType;

    SECTION("2D point on segment") {
        auto seg = tf::make_segment_between_points(
            tf::make_point(real_t(0), real_t(0)),
            tf::make_point(real_t(4), real_t(0))
        );
        auto pt = tf::make_point(real_t(2), real_t(0));
        REQUIRE(tf::intersects(pt, seg));
        REQUIRE(tf::intersects(seg, pt));
    }

    SECTION("2D point at endpoint") {
        auto seg = tf::make_segment_between_points(
            tf::make_point(real_t(0), real_t(0)),
            tf::make_point(real_t(4), real_t(0))
        );
        auto pt = tf::make_point(real_t(0), real_t(0));
        REQUIRE(tf::intersects(pt, seg));
    }

    SECTION("2D point off segment") {
        auto seg = tf::make_segment_between_points(
            tf::make_point(real_t(0), real_t(0)),
            tf::make_point(real_t(4), real_t(0))
        );
        auto pt = tf::make_point(real_t(2), real_t(1));
        REQUIRE_FALSE(tf::intersects(pt, seg));
    }

    SECTION("3D point on segment") {
        auto seg = tf::make_segment_between_points(
            tf::make_point(real_t(0), real_t(0), real_t(0)),
            tf::make_point(real_t(4), real_t(0), real_t(0))
        );
        auto pt = tf::make_point(real_t(2), real_t(0), real_t(0));
        REQUIRE(tf::intersects(pt, seg));
    }

    SECTION("3D point off segment") {
        auto seg = tf::make_segment_between_points(
            tf::make_point(real_t(0), real_t(0), real_t(0)),
            tf::make_point(real_t(4), real_t(0), real_t(0))
        );
        auto pt = tf::make_point(real_t(2), real_t(1), real_t(0));
        REQUIRE_FALSE(tf::intersects(pt, seg));
    }
}

// =============================================================================
// Point-Ray tests
// =============================================================================

TEMPLATE_TEST_CASE("point_ray_intersects", "[core][intersects]",
    float, double)
{
    using real_t = TestType;

    SECTION("2D point on ray") {
        auto ray = tf::make_ray(
            tf::make_point(real_t(0), real_t(0)),
            tf::make_vector(real_t(1), real_t(0))
        );
        auto pt = tf::make_point(real_t(5), real_t(0));
        REQUIRE(tf::intersects(pt, ray));
        REQUIRE(tf::intersects(ray, pt));
    }

    SECTION("2D point at origin") {
        auto ray = tf::make_ray(
            tf::make_point(real_t(0), real_t(0)),
            tf::make_vector(real_t(1), real_t(0))
        );
        auto pt = tf::make_point(real_t(0), real_t(0));
        REQUIRE(tf::intersects(pt, ray));
    }

    SECTION("2D point behind ray") {
        auto ray = tf::make_ray(
            tf::make_point(real_t(0), real_t(0)),
            tf::make_vector(real_t(1), real_t(0))
        );
        auto pt = tf::make_point(real_t(-1), real_t(0));
        REQUIRE_FALSE(tf::intersects(pt, ray));
    }

    SECTION("2D point off ray") {
        auto ray = tf::make_ray(
            tf::make_point(real_t(0), real_t(0)),
            tf::make_vector(real_t(1), real_t(0))
        );
        auto pt = tf::make_point(real_t(2), real_t(1));
        REQUIRE_FALSE(tf::intersects(pt, ray));
    }
}

// =============================================================================
// Point-Line tests
// =============================================================================

TEMPLATE_TEST_CASE("point_line_intersects", "[core][intersects]",
    float, double)
{
    using real_t = TestType;

    SECTION("2D point on line") {
        auto line = tf::make_line_like(
            tf::make_point(real_t(0), real_t(0)),
            tf::make_vector(real_t(1), real_t(0))
        );
        auto pt = tf::make_point(real_t(5), real_t(0));
        REQUIRE(tf::intersects(pt, line));
        REQUIRE(tf::intersects(line, pt));
    }

    SECTION("2D point on line (negative direction)") {
        auto line = tf::make_line_like(
            tf::make_point(real_t(0), real_t(0)),
            tf::make_vector(real_t(1), real_t(0))
        );
        auto pt = tf::make_point(real_t(-5), real_t(0));
        REQUIRE(tf::intersects(pt, line));
    }

    SECTION("2D point off line") {
        auto line = tf::make_line_like(
            tf::make_point(real_t(0), real_t(0)),
            tf::make_vector(real_t(1), real_t(0))
        );
        auto pt = tf::make_point(real_t(2), real_t(1));
        REQUIRE_FALSE(tf::intersects(pt, line));
    }
}

// =============================================================================
// Point-Polygon tests
// =============================================================================

TEMPLATE_TEST_CASE("point_polygon_intersects", "[core][intersects]",
    float, double)
{
    using real_t = TestType;

    SECTION("2D point inside polygon") {
        std::array<tf::point<real_t, 2>, 4> square_pts = {{
            tf::make_point(real_t(0), real_t(0)),
            tf::make_point(real_t(2), real_t(0)),
            tf::make_point(real_t(2), real_t(2)),
            tf::make_point(real_t(0), real_t(2))
        }};
        auto poly = tf::make_polygon(square_pts);
        auto pt = tf::make_point(real_t(1), real_t(1));
        REQUIRE(tf::intersects(pt, poly));
        REQUIRE(tf::intersects(poly, pt));
    }

    SECTION("2D point outside polygon") {
        std::array<tf::point<real_t, 2>, 4> square_pts = {{
            tf::make_point(real_t(0), real_t(0)),
            tf::make_point(real_t(2), real_t(0)),
            tf::make_point(real_t(2), real_t(2)),
            tf::make_point(real_t(0), real_t(2))
        }};
        auto poly = tf::make_polygon(square_pts);
        auto pt = tf::make_point(real_t(3), real_t(1));
        REQUIRE_FALSE(tf::intersects(pt, poly));
    }

    SECTION("3D point on polygon plane inside") {
        std::array<tf::point<real_t, 3>, 3> triangle_pts = {{
            tf::make_point(real_t(0), real_t(0), real_t(0)),
            tf::make_point(real_t(2), real_t(0), real_t(0)),
            tf::make_point(real_t(1), real_t(2), real_t(0))
        }};
        auto poly = tf::make_polygon(triangle_pts);
        auto pt = tf::make_point(real_t(1), real_t(0.5), real_t(0));
        REQUIRE(tf::intersects(pt, poly));
    }

    SECTION("3D point above polygon") {
        std::array<tf::point<real_t, 3>, 3> triangle_pts = {{
            tf::make_point(real_t(0), real_t(0), real_t(0)),
            tf::make_point(real_t(2), real_t(0), real_t(0)),
            tf::make_point(real_t(1), real_t(2), real_t(0))
        }};
        auto poly = tf::make_polygon(triangle_pts);
        auto pt = tf::make_point(real_t(1), real_t(0.5), real_t(1));
        REQUIRE_FALSE(tf::intersects(pt, poly));
    }
}

// =============================================================================
// Point-Plane tests
// =============================================================================

TEMPLATE_TEST_CASE("point_plane_intersects", "[core][intersects]",
    float, double)
{
    using real_t = TestType;

    SECTION("point on plane") {
        auto plane = tf::make_plane(
            tf::make_unit_vector(real_t(0), real_t(0), real_t(1)),
            real_t(0)
        );
        auto pt = tf::make_point(real_t(5), real_t(3), real_t(0));
        REQUIRE(tf::intersects(pt, plane));
        REQUIRE(tf::intersects(plane, pt));
    }

    SECTION("point above plane") {
        auto plane = tf::make_plane(
            tf::make_unit_vector(real_t(0), real_t(0), real_t(1)),
            real_t(0)
        );
        auto pt = tf::make_point(real_t(5), real_t(3), real_t(1));
        REQUIRE_FALSE(tf::intersects(pt, plane));
    }
}

// =============================================================================
// Point-AABB tests
// =============================================================================

TEMPLATE_TEST_CASE("point_aabb_intersects", "[core][intersects]",
    float, double)
{
    using real_t = TestType;

    SECTION("2D point inside aabb") {
        auto aabb = tf::make_aabb(
            tf::make_point(real_t(0), real_t(0)),
            tf::make_point(real_t(2), real_t(2))
        );
        auto pt = tf::make_point(real_t(1), real_t(1));
        REQUIRE(tf::intersects(pt, aabb));
        REQUIRE(tf::intersects(aabb, pt));
    }

    SECTION("2D point outside aabb") {
        auto aabb = tf::make_aabb(
            tf::make_point(real_t(0), real_t(0)),
            tf::make_point(real_t(2), real_t(2))
        );
        auto pt = tf::make_point(real_t(3), real_t(1));
        REQUIRE_FALSE(tf::intersects(pt, aabb));
    }

    SECTION("3D point inside aabb") {
        auto aabb = tf::make_aabb(
            tf::make_point(real_t(0), real_t(0), real_t(0)),
            tf::make_point(real_t(2), real_t(2), real_t(2))
        );
        auto pt = tf::make_point(real_t(1), real_t(1), real_t(1));
        REQUIRE(tf::intersects(pt, aabb));
    }

    SECTION("3D point outside aabb") {
        auto aabb = tf::make_aabb(
            tf::make_point(real_t(0), real_t(0), real_t(0)),
            tf::make_point(real_t(2), real_t(2), real_t(2))
        );
        auto pt = tf::make_point(real_t(3), real_t(1), real_t(1));
        REQUIRE_FALSE(tf::intersects(pt, aabb));
    }
}

// =============================================================================
// Segment-Segment tests
// =============================================================================

TEMPLATE_TEST_CASE("segment_segment_intersects", "[core][intersects]",
    float, double)
{
    using real_t = TestType;

    SECTION("2D crossing segments") {
        auto seg1 = tf::make_segment_between_points(
            tf::make_point(real_t(0), real_t(0)),
            tf::make_point(real_t(4), real_t(4))
        );
        auto seg2 = tf::make_segment_between_points(
            tf::make_point(real_t(0), real_t(4)),
            tf::make_point(real_t(4), real_t(0))
        );
        REQUIRE(tf::intersects(seg1, seg2));
    }

    SECTION("2D parallel non-intersecting") {
        auto seg1 = tf::make_segment_between_points(
            tf::make_point(real_t(0), real_t(0)),
            tf::make_point(real_t(4), real_t(0))
        );
        auto seg2 = tf::make_segment_between_points(
            tf::make_point(real_t(0), real_t(2)),
            tf::make_point(real_t(4), real_t(2))
        );
        REQUIRE_FALSE(tf::intersects(seg1, seg2));
    }

    SECTION("2D T-junction") {
        auto seg1 = tf::make_segment_between_points(
            tf::make_point(real_t(0), real_t(0)),
            tf::make_point(real_t(4), real_t(0))
        );
        auto seg2 = tf::make_segment_between_points(
            tf::make_point(real_t(2), real_t(0)),
            tf::make_point(real_t(2), real_t(2))
        );
        REQUIRE(tf::intersects(seg1, seg2));
    }

    SECTION("3D skew non-intersecting") {
        auto seg1 = tf::make_segment_between_points(
            tf::make_point(real_t(0), real_t(0), real_t(0)),
            tf::make_point(real_t(4), real_t(0), real_t(0))
        );
        auto seg2 = tf::make_segment_between_points(
            tf::make_point(real_t(2), real_t(0), real_t(1)),
            tf::make_point(real_t(2), real_t(2), real_t(1))
        );
        REQUIRE_FALSE(tf::intersects(seg1, seg2));
    }

    SECTION("3D intersecting") {
        auto seg1 = tf::make_segment_between_points(
            tf::make_point(real_t(0), real_t(0), real_t(0)),
            tf::make_point(real_t(4), real_t(0), real_t(0))
        );
        auto seg2 = tf::make_segment_between_points(
            tf::make_point(real_t(2), real_t(-1), real_t(0)),
            tf::make_point(real_t(2), real_t(1), real_t(0))
        );
        REQUIRE(tf::intersects(seg1, seg2));
    }
}

// =============================================================================
// Segment-Ray tests
// =============================================================================

TEMPLATE_TEST_CASE("segment_ray_intersects", "[core][intersects]",
    float, double)
{
    using real_t = TestType;

    SECTION("2D ray hitting segment") {
        auto seg = tf::make_segment_between_points(
            tf::make_point(real_t(2), real_t(-1)),
            tf::make_point(real_t(2), real_t(1))
        );
        auto ray = tf::make_ray(
            tf::make_point(real_t(0), real_t(0)),
            tf::make_vector(real_t(1), real_t(0))
        );
        REQUIRE(tf::intersects(seg, ray));
        REQUIRE(tf::intersects(ray, seg));
    }

    SECTION("2D ray missing segment") {
        auto seg = tf::make_segment_between_points(
            tf::make_point(real_t(2), real_t(2)),
            tf::make_point(real_t(2), real_t(4))
        );
        auto ray = tf::make_ray(
            tf::make_point(real_t(0), real_t(0)),
            tf::make_vector(real_t(1), real_t(0))
        );
        REQUIRE_FALSE(tf::intersects(seg, ray));
    }

    SECTION("2D ray pointing away from segment") {
        auto seg = tf::make_segment_between_points(
            tf::make_point(real_t(2), real_t(-1)),
            tf::make_point(real_t(2), real_t(1))
        );
        auto ray = tf::make_ray(
            tf::make_point(real_t(0), real_t(0)),
            tf::make_vector(real_t(-1), real_t(0))
        );
        REQUIRE_FALSE(tf::intersects(seg, ray));
    }
}

// =============================================================================
// Segment-Line tests
// =============================================================================

TEMPLATE_TEST_CASE("segment_line_intersects", "[core][intersects]",
    float, double)
{
    using real_t = TestType;

    SECTION("2D line crossing segment") {
        auto seg = tf::make_segment_between_points(
            tf::make_point(real_t(2), real_t(-1)),
            tf::make_point(real_t(2), real_t(1))
        );
        auto line = tf::make_line_like(
            tf::make_point(real_t(0), real_t(0)),
            tf::make_vector(real_t(1), real_t(0))
        );
        REQUIRE(tf::intersects(seg, line));
        REQUIRE(tf::intersects(line, seg));
    }

    SECTION("2D parallel non-intersecting") {
        auto seg = tf::make_segment_between_points(
            tf::make_point(real_t(0), real_t(2)),
            tf::make_point(real_t(4), real_t(2))
        );
        auto line = tf::make_line_like(
            tf::make_point(real_t(0), real_t(0)),
            tf::make_vector(real_t(1), real_t(0))
        );
        REQUIRE_FALSE(tf::intersects(seg, line));
    }
}

// =============================================================================
// Segment-Polygon tests
// =============================================================================

TEMPLATE_TEST_CASE("segment_polygon_intersects", "[core][intersects]",
    float, double)
{
    using real_t = TestType;

    SECTION("2D segment crossing polygon") {
        std::array<tf::point<real_t, 2>, 4> square_pts = {{
            tf::make_point(real_t(0), real_t(0)),
            tf::make_point(real_t(2), real_t(0)),
            tf::make_point(real_t(2), real_t(2)),
            tf::make_point(real_t(0), real_t(2))
        }};
        auto poly = tf::make_polygon(square_pts);
        auto seg = tf::make_segment_between_points(
            tf::make_point(real_t(-1), real_t(1)),
            tf::make_point(real_t(3), real_t(1))
        );
        REQUIRE(tf::intersects(seg, poly));
        REQUIRE(tf::intersects(poly, seg));
    }

    SECTION("2D segment inside polygon") {
        std::array<tf::point<real_t, 2>, 4> square_pts = {{
            tf::make_point(real_t(0), real_t(0)),
            tf::make_point(real_t(4), real_t(0)),
            tf::make_point(real_t(4), real_t(4)),
            tf::make_point(real_t(0), real_t(4))
        }};
        auto poly = tf::make_polygon(square_pts);
        auto seg = tf::make_segment_between_points(
            tf::make_point(real_t(1), real_t(1)),
            tf::make_point(real_t(3), real_t(3))
        );
        REQUIRE(tf::intersects(seg, poly));
    }

    SECTION("2D segment outside polygon") {
        std::array<tf::point<real_t, 2>, 4> square_pts = {{
            tf::make_point(real_t(0), real_t(0)),
            tf::make_point(real_t(2), real_t(0)),
            tf::make_point(real_t(2), real_t(2)),
            tf::make_point(real_t(0), real_t(2))
        }};
        auto poly = tf::make_polygon(square_pts);
        auto seg = tf::make_segment_between_points(
            tf::make_point(real_t(3), real_t(0)),
            tf::make_point(real_t(3), real_t(2))
        );
        REQUIRE_FALSE(tf::intersects(seg, poly));
    }

    SECTION("3D segment crossing polygon") {
        std::array<tf::point<real_t, 3>, 3> triangle_pts = {{
            tf::make_point(real_t(0), real_t(0), real_t(0)),
            tf::make_point(real_t(2), real_t(0), real_t(0)),
            tf::make_point(real_t(1), real_t(2), real_t(0))
        }};
        auto poly = tf::make_polygon(triangle_pts);
        auto seg = tf::make_segment_between_points(
            tf::make_point(real_t(1), real_t(0.5), real_t(-1)),
            tf::make_point(real_t(1), real_t(0.5), real_t(1))
        );
        REQUIRE(tf::intersects(seg, poly));
    }
}

// =============================================================================
// Segment-Plane tests
// =============================================================================

TEMPLATE_TEST_CASE("segment_plane_intersects", "[core][intersects]",
    float, double)
{
    using real_t = TestType;

    SECTION("segment crossing plane") {
        auto plane = tf::make_plane(
            tf::make_unit_vector(real_t(0), real_t(0), real_t(1)),
            real_t(0)
        );
        auto seg = tf::make_segment_between_points(
            tf::make_point(real_t(0), real_t(0), real_t(-1)),
            tf::make_point(real_t(0), real_t(0), real_t(1))
        );
        REQUIRE(tf::intersects(seg, plane));
        REQUIRE(tf::intersects(plane, seg));
    }

    SECTION("segment parallel to plane") {
        auto plane = tf::make_plane(
            tf::make_unit_vector(real_t(0), real_t(0), real_t(1)),
            real_t(0)
        );
        auto seg = tf::make_segment_between_points(
            tf::make_point(real_t(0), real_t(0), real_t(1)),
            tf::make_point(real_t(2), real_t(0), real_t(1))
        );
        REQUIRE_FALSE(tf::intersects(seg, plane));
    }

    SECTION("segment on plane") {
        auto plane = tf::make_plane(
            tf::make_unit_vector(real_t(0), real_t(0), real_t(1)),
            real_t(0)
        );
        auto seg = tf::make_segment_between_points(
            tf::make_point(real_t(0), real_t(0), real_t(0)),
            tf::make_point(real_t(2), real_t(0), real_t(0))
        );
        REQUIRE(tf::intersects(seg, plane));
    }
}

// =============================================================================
// Segment-AABB tests
// =============================================================================

TEMPLATE_TEST_CASE("segment_aabb_intersects", "[core][intersects]",
    float, double)
{
    using real_t = TestType;

    SECTION("2D segment crossing aabb") {
        auto aabb = tf::make_aabb(
            tf::make_point(real_t(0), real_t(0)),
            tf::make_point(real_t(2), real_t(2))
        );
        auto seg = tf::make_segment_between_points(
            tf::make_point(real_t(-1), real_t(1)),
            tf::make_point(real_t(3), real_t(1))
        );
        REQUIRE(tf::intersects(seg, aabb));
        REQUIRE(tf::intersects(aabb, seg));
    }

    SECTION("2D segment inside aabb") {
        auto aabb = tf::make_aabb(
            tf::make_point(real_t(0), real_t(0)),
            tf::make_point(real_t(4), real_t(4))
        );
        auto seg = tf::make_segment_between_points(
            tf::make_point(real_t(1), real_t(1)),
            tf::make_point(real_t(2), real_t(2))
        );
        REQUIRE(tf::intersects(seg, aabb));
    }

    SECTION("2D segment outside aabb") {
        auto aabb = tf::make_aabb(
            tf::make_point(real_t(0), real_t(0)),
            tf::make_point(real_t(2), real_t(2))
        );
        auto seg = tf::make_segment_between_points(
            tf::make_point(real_t(3), real_t(0)),
            tf::make_point(real_t(3), real_t(2))
        );
        REQUIRE_FALSE(tf::intersects(seg, aabb));
    }
}

// =============================================================================
// Ray-Ray tests
// =============================================================================

TEMPLATE_TEST_CASE("ray_ray_intersects", "[core][intersects]",
    float, double)
{
    using real_t = TestType;

    SECTION("2D intersecting rays") {
        auto ray1 = tf::make_ray(
            tf::make_point(real_t(0), real_t(0)),
            tf::make_vector(real_t(1), real_t(1))
        );
        auto ray2 = tf::make_ray(
            tf::make_point(real_t(2), real_t(0)),
            tf::make_vector(real_t(-1), real_t(1))
        );
        REQUIRE(tf::intersects(ray1, ray2));
    }

    SECTION("2D parallel rays") {
        auto ray1 = tf::make_ray(
            tf::make_point(real_t(0), real_t(0)),
            tf::make_vector(real_t(1), real_t(0))
        );
        auto ray2 = tf::make_ray(
            tf::make_point(real_t(0), real_t(2)),
            tf::make_vector(real_t(1), real_t(0))
        );
        REQUIRE_FALSE(tf::intersects(ray1, ray2));
    }

    SECTION("2D diverging rays") {
        auto ray1 = tf::make_ray(
            tf::make_point(real_t(0), real_t(0)),
            tf::make_vector(real_t(1), real_t(0))
        );
        auto ray2 = tf::make_ray(
            tf::make_point(real_t(0), real_t(2)),
            tf::make_vector(real_t(-1), real_t(0))
        );
        REQUIRE_FALSE(tf::intersects(ray1, ray2));
    }
}

// =============================================================================
// Ray-Line tests
// =============================================================================

TEMPLATE_TEST_CASE("ray_line_intersects", "[core][intersects]",
    float, double)
{
    using real_t = TestType;

    SECTION("2D ray toward line") {
        auto ray = tf::make_ray(
            tf::make_point(real_t(0), real_t(0)),
            tf::make_vector(real_t(0), real_t(1))
        );
        auto line = tf::make_line_like(
            tf::make_point(real_t(0), real_t(5)),
            tf::make_vector(real_t(1), real_t(0))
        );
        REQUIRE(tf::intersects(ray, line));
        REQUIRE(tf::intersects(line, ray));
    }

    SECTION("2D ray pointing away from line") {
        auto ray = tf::make_ray(
            tf::make_point(real_t(0), real_t(0)),
            tf::make_vector(real_t(0), real_t(-1))
        );
        auto line = tf::make_line_like(
            tf::make_point(real_t(0), real_t(5)),
            tf::make_vector(real_t(1), real_t(0))
        );
        REQUIRE_FALSE(tf::intersects(ray, line));
    }

    SECTION("2D parallel") {
        auto ray = tf::make_ray(
            tf::make_point(real_t(0), real_t(0)),
            tf::make_vector(real_t(1), real_t(0))
        );
        auto line = tf::make_line_like(
            tf::make_point(real_t(0), real_t(5)),
            tf::make_vector(real_t(1), real_t(0))
        );
        REQUIRE_FALSE(tf::intersects(ray, line));
    }
}

// =============================================================================
// Ray-Polygon tests
// =============================================================================

TEMPLATE_TEST_CASE("ray_polygon_intersects", "[core][intersects]",
    float, double)
{
    using real_t = TestType;

    SECTION("3D ray hitting polygon") {
        std::array<tf::point<real_t, 3>, 3> triangle_pts = {{
            tf::make_point(real_t(0), real_t(0), real_t(0)),
            tf::make_point(real_t(2), real_t(0), real_t(0)),
            tf::make_point(real_t(1), real_t(2), real_t(0))
        }};
        auto poly = tf::make_polygon(triangle_pts);
        auto ray = tf::make_ray(
            tf::make_point(real_t(1), real_t(0.5), real_t(2)),
            tf::make_vector(real_t(0), real_t(0), real_t(-1))
        );
        REQUIRE(tf::intersects(ray, poly));
        REQUIRE(tf::intersects(poly, ray));
    }

    SECTION("3D ray missing polygon") {
        std::array<tf::point<real_t, 3>, 3> triangle_pts = {{
            tf::make_point(real_t(0), real_t(0), real_t(0)),
            tf::make_point(real_t(2), real_t(0), real_t(0)),
            tf::make_point(real_t(1), real_t(2), real_t(0))
        }};
        auto poly = tf::make_polygon(triangle_pts);
        auto ray = tf::make_ray(
            tf::make_point(real_t(5), real_t(5), real_t(2)),
            tf::make_vector(real_t(0), real_t(0), real_t(-1))
        );
        REQUIRE_FALSE(tf::intersects(ray, poly));
    }

    SECTION("3D ray pointing away") {
        std::array<tf::point<real_t, 3>, 3> triangle_pts = {{
            tf::make_point(real_t(0), real_t(0), real_t(0)),
            tf::make_point(real_t(2), real_t(0), real_t(0)),
            tf::make_point(real_t(1), real_t(2), real_t(0))
        }};
        auto poly = tf::make_polygon(triangle_pts);
        auto ray = tf::make_ray(
            tf::make_point(real_t(1), real_t(0.5), real_t(2)),
            tf::make_vector(real_t(0), real_t(0), real_t(1))
        );
        REQUIRE_FALSE(tf::intersects(ray, poly));
    }
}

// =============================================================================
// Ray-Plane tests
// =============================================================================

TEMPLATE_TEST_CASE("ray_plane_intersects", "[core][intersects]",
    float, double)
{
    using real_t = TestType;

    SECTION("ray toward plane") {
        auto plane = tf::make_plane(
            tf::make_unit_vector(real_t(0), real_t(0), real_t(1)),
            real_t(0)
        );
        auto ray = tf::make_ray(
            tf::make_point(real_t(0), real_t(0), real_t(5)),
            tf::make_vector(real_t(0), real_t(0), real_t(-1))
        );
        REQUIRE(tf::intersects(ray, plane));
        REQUIRE(tf::intersects(plane, ray));
    }

    SECTION("ray away from plane") {
        auto plane = tf::make_plane(
            tf::make_unit_vector(real_t(0), real_t(0), real_t(1)),
            real_t(0)
        );
        auto ray = tf::make_ray(
            tf::make_point(real_t(0), real_t(0), real_t(5)),
            tf::make_vector(real_t(0), real_t(0), real_t(1))
        );
        REQUIRE_FALSE(tf::intersects(ray, plane));
    }

    SECTION("ray parallel to plane") {
        auto plane = tf::make_plane(
            tf::make_unit_vector(real_t(0), real_t(0), real_t(1)),
            real_t(0)
        );
        auto ray = tf::make_ray(
            tf::make_point(real_t(0), real_t(0), real_t(5)),
            tf::make_vector(real_t(1), real_t(0), real_t(0))
        );
        REQUIRE_FALSE(tf::intersects(ray, plane));
    }
}

// =============================================================================
// Ray-AABB tests
// =============================================================================

TEMPLATE_TEST_CASE("ray_aabb_intersects", "[core][intersects]",
    float, double)
{
    using real_t = TestType;

    SECTION("2D ray hitting aabb") {
        auto aabb = tf::make_aabb(
            tf::make_point(real_t(2), real_t(-1)),
            tf::make_point(real_t(4), real_t(1))
        );
        auto ray = tf::make_ray(
            tf::make_point(real_t(0), real_t(0)),
            tf::make_vector(real_t(1), real_t(0))
        );
        REQUIRE(tf::intersects(ray, aabb));
        REQUIRE(tf::intersects(aabb, ray));
    }

    SECTION("2D ray missing aabb") {
        auto aabb = tf::make_aabb(
            tf::make_point(real_t(2), real_t(2)),
            tf::make_point(real_t(4), real_t(4))
        );
        auto ray = tf::make_ray(
            tf::make_point(real_t(0), real_t(0)),
            tf::make_vector(real_t(1), real_t(0))
        );
        REQUIRE_FALSE(tf::intersects(ray, aabb));
    }

    SECTION("3D ray hitting aabb") {
        auto aabb = tf::make_aabb(
            tf::make_point(real_t(-1), real_t(-1), real_t(-1)),
            tf::make_point(real_t(1), real_t(1), real_t(1))
        );
        auto ray = tf::make_ray(
            tf::make_point(real_t(0), real_t(0), real_t(5)),
            tf::make_vector(real_t(0), real_t(0), real_t(-1))
        );
        REQUIRE(tf::intersects(ray, aabb));
    }
}

// =============================================================================
// Line-Line tests
// =============================================================================

TEMPLATE_TEST_CASE("line_line_intersects", "[core][intersects]",
    float, double)
{
    using real_t = TestType;

    SECTION("2D crossing lines") {
        auto line1 = tf::make_line_like(
            tf::make_point(real_t(0), real_t(0)),
            tf::make_vector(real_t(1), real_t(1))
        );
        auto line2 = tf::make_line_like(
            tf::make_point(real_t(2), real_t(0)),
            tf::make_vector(real_t(-1), real_t(1))
        );
        REQUIRE(tf::intersects(line1, line2));
    }

    SECTION("2D parallel lines") {
        auto line1 = tf::make_line_like(
            tf::make_point(real_t(0), real_t(0)),
            tf::make_vector(real_t(1), real_t(0))
        );
        auto line2 = tf::make_line_like(
            tf::make_point(real_t(0), real_t(2)),
            tf::make_vector(real_t(1), real_t(0))
        );
        REQUIRE_FALSE(tf::intersects(line1, line2));
    }

    SECTION("3D skew lines") {
        auto line1 = tf::make_line_like(
            tf::make_point(real_t(0), real_t(0), real_t(0)),
            tf::make_vector(real_t(1), real_t(0), real_t(0))
        );
        auto line2 = tf::make_line_like(
            tf::make_point(real_t(0), real_t(0), real_t(1)),
            tf::make_vector(real_t(0), real_t(1), real_t(0))
        );
        REQUIRE_FALSE(tf::intersects(line1, line2));
    }

    SECTION("3D intersecting lines") {
        auto line1 = tf::make_line_like(
            tf::make_point(real_t(0), real_t(0), real_t(0)),
            tf::make_vector(real_t(1), real_t(0), real_t(0))
        );
        auto line2 = tf::make_line_like(
            tf::make_point(real_t(2), real_t(0), real_t(0)),
            tf::make_vector(real_t(0), real_t(1), real_t(0))
        );
        REQUIRE(tf::intersects(line1, line2));
    }
}

// =============================================================================
// Line-Polygon tests
// =============================================================================

TEMPLATE_TEST_CASE("line_polygon_intersects", "[core][intersects]",
    float, double)
{
    using real_t = TestType;

    SECTION("3D line crossing polygon") {
        std::array<tf::point<real_t, 3>, 3> triangle_pts = {{
            tf::make_point(real_t(0), real_t(0), real_t(0)),
            tf::make_point(real_t(2), real_t(0), real_t(0)),
            tf::make_point(real_t(1), real_t(2), real_t(0))
        }};
        auto poly = tf::make_polygon(triangle_pts);
        auto line = tf::make_line_like(
            tf::make_point(real_t(1), real_t(0.5), real_t(5)),
            tf::make_vector(real_t(0), real_t(0), real_t(1))
        );
        REQUIRE(tf::intersects(line, poly));
        REQUIRE(tf::intersects(poly, line));
    }

    SECTION("3D line parallel to polygon") {
        std::array<tf::point<real_t, 3>, 3> triangle_pts = {{
            tf::make_point(real_t(0), real_t(0), real_t(0)),
            tf::make_point(real_t(2), real_t(0), real_t(0)),
            tf::make_point(real_t(1), real_t(2), real_t(0))
        }};
        auto poly = tf::make_polygon(triangle_pts);
        auto line = tf::make_line_like(
            tf::make_point(real_t(0), real_t(0), real_t(5)),
            tf::make_vector(real_t(1), real_t(0), real_t(0))
        );
        REQUIRE_FALSE(tf::intersects(line, poly));
    }
}

// =============================================================================
// Line-Plane tests
// =============================================================================

TEMPLATE_TEST_CASE("line_plane_intersects", "[core][intersects]",
    float, double)
{
    using real_t = TestType;

    SECTION("line crossing plane") {
        auto plane = tf::make_plane(
            tf::make_unit_vector(real_t(0), real_t(0), real_t(1)),
            real_t(0)
        );
        auto line = tf::make_line_like(
            tf::make_point(real_t(0), real_t(0), real_t(5)),
            tf::make_vector(real_t(0), real_t(0), real_t(1))
        );
        REQUIRE(tf::intersects(line, plane));
        REQUIRE(tf::intersects(plane, line));
    }

    SECTION("line parallel to plane") {
        auto plane = tf::make_plane(
            tf::make_unit_vector(real_t(0), real_t(0), real_t(1)),
            real_t(0)
        );
        auto line = tf::make_line_like(
            tf::make_point(real_t(0), real_t(0), real_t(5)),
            tf::make_vector(real_t(1), real_t(0), real_t(0))
        );
        REQUIRE_FALSE(tf::intersects(line, plane));
    }

    SECTION("line on plane") {
        auto plane = tf::make_plane(
            tf::make_unit_vector(real_t(0), real_t(0), real_t(1)),
            real_t(0)
        );
        auto line = tf::make_line_like(
            tf::make_point(real_t(0), real_t(0), real_t(0)),
            tf::make_vector(real_t(1), real_t(0), real_t(0))
        );
        REQUIRE(tf::intersects(line, plane));
    }
}

// =============================================================================
// Line-AABB tests
// =============================================================================

TEMPLATE_TEST_CASE("line_aabb_intersects", "[core][intersects]",
    float, double)
{
    using real_t = TestType;

    SECTION("2D line through aabb") {
        auto aabb = tf::make_aabb(
            tf::make_point(real_t(-1), real_t(-1)),
            tf::make_point(real_t(1), real_t(1))
        );
        auto line = tf::make_line_like(
            tf::make_point(real_t(-5), real_t(0)),
            tf::make_vector(real_t(1), real_t(0))
        );
        REQUIRE(tf::intersects(line, aabb));
        REQUIRE(tf::intersects(aabb, line));
    }

    SECTION("2D line missing aabb") {
        auto aabb = tf::make_aabb(
            tf::make_point(real_t(-1), real_t(-1)),
            tf::make_point(real_t(1), real_t(1))
        );
        auto line = tf::make_line_like(
            tf::make_point(real_t(-5), real_t(5)),
            tf::make_vector(real_t(1), real_t(0))
        );
        REQUIRE_FALSE(tf::intersects(line, aabb));
    }
}

// =============================================================================
// Polygon-Polygon tests
// =============================================================================

TEMPLATE_TEST_CASE("polygon_polygon_intersects", "[core][intersects]",
    float, double)
{
    using real_t = TestType;

    SECTION("2D overlapping polygons") {
        std::array<tf::point<real_t, 2>, 4> square1_pts = {{
            tf::make_point(real_t(0), real_t(0)),
            tf::make_point(real_t(2), real_t(0)),
            tf::make_point(real_t(2), real_t(2)),
            tf::make_point(real_t(0), real_t(2))
        }};
        std::array<tf::point<real_t, 2>, 4> square2_pts = {{
            tf::make_point(real_t(1), real_t(1)),
            tf::make_point(real_t(3), real_t(1)),
            tf::make_point(real_t(3), real_t(3)),
            tf::make_point(real_t(1), real_t(3))
        }};
        auto poly1 = tf::make_polygon(square1_pts);
        auto poly2 = tf::make_polygon(square2_pts);
        REQUIRE(tf::intersects(poly1, poly2));
    }

    SECTION("2D separated polygons") {
        std::array<tf::point<real_t, 2>, 4> square1_pts = {{
            tf::make_point(real_t(0), real_t(0)),
            tf::make_point(real_t(1), real_t(0)),
            tf::make_point(real_t(1), real_t(1)),
            tf::make_point(real_t(0), real_t(1))
        }};
        std::array<tf::point<real_t, 2>, 4> square2_pts = {{
            tf::make_point(real_t(2), real_t(0)),
            tf::make_point(real_t(3), real_t(0)),
            tf::make_point(real_t(3), real_t(1)),
            tf::make_point(real_t(2), real_t(1))
        }};
        auto poly1 = tf::make_polygon(square1_pts);
        auto poly2 = tf::make_polygon(square2_pts);
        REQUIRE_FALSE(tf::intersects(poly1, poly2));
    }

    SECTION("3D coplanar overlapping polygons") {
        std::array<tf::point<real_t, 3>, 3> tri1_pts = {{
            tf::make_point(real_t(0), real_t(0), real_t(0)),
            tf::make_point(real_t(2), real_t(0), real_t(0)),
            tf::make_point(real_t(1), real_t(2), real_t(0))
        }};
        std::array<tf::point<real_t, 3>, 3> tri2_pts = {{
            tf::make_point(real_t(1), real_t(0), real_t(0)),
            tf::make_point(real_t(3), real_t(0), real_t(0)),
            tf::make_point(real_t(2), real_t(2), real_t(0))
        }};
        auto poly1 = tf::make_polygon(tri1_pts);
        auto poly2 = tf::make_polygon(tri2_pts);
        REQUIRE(tf::intersects(poly1, poly2));
    }

    SECTION("3D non-coplanar intersecting") {
        std::array<tf::point<real_t, 3>, 4> square_xy = {{
            tf::make_point(real_t(-1), real_t(-1), real_t(0)),
            tf::make_point(real_t(1), real_t(-1), real_t(0)),
            tf::make_point(real_t(1), real_t(1), real_t(0)),
            tf::make_point(real_t(-1), real_t(1), real_t(0))
        }};
        std::array<tf::point<real_t, 3>, 4> square_xz = {{
            tf::make_point(real_t(-1), real_t(0), real_t(-1)),
            tf::make_point(real_t(1), real_t(0), real_t(-1)),
            tf::make_point(real_t(1), real_t(0), real_t(1)),
            tf::make_point(real_t(-1), real_t(0), real_t(1))
        }};
        auto poly1 = tf::make_polygon(square_xy);
        auto poly2 = tf::make_polygon(square_xz);
        REQUIRE(tf::intersects(poly1, poly2));
    }
}

// =============================================================================
// Polygon-Plane tests
// =============================================================================

TEMPLATE_TEST_CASE("polygon_plane_intersects", "[core][intersects]",
    float, double)
{
    using real_t = TestType;

    SECTION("polygon crossing plane") {
        std::array<tf::point<real_t, 3>, 3> triangle_pts = {{
            tf::make_point(real_t(0), real_t(0), real_t(-1)),
            tf::make_point(real_t(2), real_t(0), real_t(1)),
            tf::make_point(real_t(1), real_t(2), real_t(0))
        }};
        auto poly = tf::make_polygon(triangle_pts);
        auto plane = tf::make_plane(
            tf::make_unit_vector(real_t(0), real_t(0), real_t(1)),
            real_t(0)
        );
        REQUIRE(tf::intersects(poly, plane));
        REQUIRE(tf::intersects(plane, poly));
    }

    SECTION("polygon parallel to plane") {
        std::array<tf::point<real_t, 3>, 3> triangle_pts = {{
            tf::make_point(real_t(0), real_t(0), real_t(5)),
            tf::make_point(real_t(2), real_t(0), real_t(5)),
            tf::make_point(real_t(1), real_t(2), real_t(5))
        }};
        auto poly = tf::make_polygon(triangle_pts);
        auto plane = tf::make_plane(
            tf::make_unit_vector(real_t(0), real_t(0), real_t(1)),
            real_t(0)
        );
        REQUIRE_FALSE(tf::intersects(poly, plane));
    }

    SECTION("polygon on plane") {
        std::array<tf::point<real_t, 3>, 3> triangle_pts = {{
            tf::make_point(real_t(0), real_t(0), real_t(0)),
            tf::make_point(real_t(2), real_t(0), real_t(0)),
            tf::make_point(real_t(1), real_t(2), real_t(0))
        }};
        auto poly = tf::make_polygon(triangle_pts);
        auto plane = tf::make_plane(
            tf::make_unit_vector(real_t(0), real_t(0), real_t(1)),
            real_t(0)
        );
        REQUIRE(tf::intersects(poly, plane));
    }
}

// =============================================================================
// Polygon-AABB tests
// =============================================================================

TEMPLATE_TEST_CASE("polygon_aabb_intersects", "[core][intersects]",
    float, double)
{
    using real_t = TestType;

    SECTION("3D polygon crossing aabb") {
        std::array<tf::point<real_t, 3>, 3> triangle_pts = {{
            tf::make_point(real_t(-2), real_t(0), real_t(0)),
            tf::make_point(real_t(2), real_t(0), real_t(0)),
            tf::make_point(real_t(0), real_t(2), real_t(0))
        }};
        auto poly = tf::make_polygon(triangle_pts);
        auto aabb = tf::make_aabb(
            tf::make_point(real_t(-1), real_t(-1), real_t(-1)),
            tf::make_point(real_t(1), real_t(1), real_t(1))
        );
        REQUIRE(tf::intersects(poly, aabb));
        REQUIRE(tf::intersects(aabb, poly));
    }

    SECTION("3D polygon outside aabb") {
        std::array<tf::point<real_t, 3>, 3> triangle_pts = {{
            tf::make_point(real_t(5), real_t(0), real_t(0)),
            tf::make_point(real_t(7), real_t(0), real_t(0)),
            tf::make_point(real_t(6), real_t(2), real_t(0))
        }};
        auto poly = tf::make_polygon(triangle_pts);
        auto aabb = tf::make_aabb(
            tf::make_point(real_t(-1), real_t(-1), real_t(-1)),
            tf::make_point(real_t(1), real_t(1), real_t(1))
        );
        REQUIRE_FALSE(tf::intersects(poly, aabb));
    }
}

// =============================================================================
// Plane-Plane tests
// =============================================================================

TEMPLATE_TEST_CASE("plane_plane_intersects", "[core][intersects]",
    float, double)
{
    using real_t = TestType;

    SECTION("perpendicular planes") {
        auto plane1 = tf::make_plane(
            tf::make_unit_vector(real_t(0), real_t(0), real_t(1)),
            real_t(0)
        );
        auto plane2 = tf::make_plane(
            tf::make_unit_vector(real_t(1), real_t(0), real_t(0)),
            real_t(0)
        );
        REQUIRE(tf::intersects(plane1, plane2));
    }

    SECTION("parallel non-coincident planes") {
        auto plane1 = tf::make_plane(
            tf::make_unit_vector(real_t(0), real_t(0), real_t(1)),
            real_t(0)
        );
        auto plane2 = tf::make_plane(
            tf::make_unit_vector(real_t(0), real_t(0), real_t(1)),
            real_t(-5)
        );
        REQUIRE_FALSE(tf::intersects(plane1, plane2));
    }

    SECTION("coincident planes") {
        auto plane1 = tf::make_plane(
            tf::make_unit_vector(real_t(0), real_t(0), real_t(1)),
            real_t(0)
        );
        auto plane2 = tf::make_plane(
            tf::make_unit_vector(real_t(0), real_t(0), real_t(1)),
            real_t(0)
        );
        REQUIRE(tf::intersects(plane1, plane2));
    }
}

// =============================================================================
// Plane-AABB tests
// =============================================================================

TEMPLATE_TEST_CASE("plane_aabb_intersects", "[core][intersects]",
    float, double)
{
    using real_t = TestType;

    SECTION("plane through aabb center") {
        auto aabb = tf::make_aabb(
            tf::make_point(real_t(-1), real_t(-1), real_t(-1)),
            tf::make_point(real_t(1), real_t(1), real_t(1))
        );
        auto plane = tf::make_plane(
            tf::make_unit_vector(real_t(0), real_t(0), real_t(1)),
            real_t(0)
        );
        REQUIRE(tf::intersects(plane, aabb));
        REQUIRE(tf::intersects(aabb, plane));
    }

    SECTION("plane above aabb") {
        auto aabb = tf::make_aabb(
            tf::make_point(real_t(-1), real_t(-1), real_t(-1)),
            tf::make_point(real_t(1), real_t(1), real_t(1))
        );
        auto plane = tf::make_plane(
            tf::make_unit_vector(real_t(0), real_t(0), real_t(1)),
            real_t(-5)
        );
        REQUIRE_FALSE(tf::intersects(plane, aabb));
    }

    SECTION("plane touching aabb face") {
        auto aabb = tf::make_aabb(
            tf::make_point(real_t(-1), real_t(-1), real_t(-1)),
            tf::make_point(real_t(1), real_t(1), real_t(1))
        );
        auto plane = tf::make_plane(
            tf::make_unit_vector(real_t(0), real_t(0), real_t(1)),
            real_t(-1)
        );
        REQUIRE(tf::intersects(plane, aabb));
    }
}

// =============================================================================
// AABB-AABB tests
// =============================================================================

TEMPLATE_TEST_CASE("aabb_aabb_intersects", "[core][intersects]",
    float, double)
{
    using real_t = TestType;

    SECTION("2D overlapping aabbs") {
        auto aabb1 = tf::make_aabb(
            tf::make_point(real_t(0), real_t(0)),
            tf::make_point(real_t(2), real_t(2))
        );
        auto aabb2 = tf::make_aabb(
            tf::make_point(real_t(1), real_t(1)),
            tf::make_point(real_t(3), real_t(3))
        );
        REQUIRE(tf::intersects(aabb1, aabb2));
    }

    SECTION("2D separated aabbs") {
        auto aabb1 = tf::make_aabb(
            tf::make_point(real_t(0), real_t(0)),
            tf::make_point(real_t(1), real_t(1))
        );
        auto aabb2 = tf::make_aabb(
            tf::make_point(real_t(2), real_t(0)),
            tf::make_point(real_t(3), real_t(1))
        );
        REQUIRE_FALSE(tf::intersects(aabb1, aabb2));
    }

    SECTION("2D touching aabbs") {
        auto aabb1 = tf::make_aabb(
            tf::make_point(real_t(0), real_t(0)),
            tf::make_point(real_t(1), real_t(1))
        );
        auto aabb2 = tf::make_aabb(
            tf::make_point(real_t(1), real_t(0)),
            tf::make_point(real_t(2), real_t(1))
        );
        REQUIRE(tf::intersects(aabb1, aabb2));
    }

    SECTION("3D overlapping aabbs") {
        auto aabb1 = tf::make_aabb(
            tf::make_point(real_t(0), real_t(0), real_t(0)),
            tf::make_point(real_t(2), real_t(2), real_t(2))
        );
        auto aabb2 = tf::make_aabb(
            tf::make_point(real_t(1), real_t(1), real_t(1)),
            tf::make_point(real_t(3), real_t(3), real_t(3))
        );
        REQUIRE(tf::intersects(aabb1, aabb2));
    }

    SECTION("3D separated aabbs") {
        auto aabb1 = tf::make_aabb(
            tf::make_point(real_t(0), real_t(0), real_t(0)),
            tf::make_point(real_t(1), real_t(1), real_t(1))
        );
        auto aabb2 = tf::make_aabb(
            tf::make_point(real_t(5), real_t(5), real_t(5)),
            tf::make_point(real_t(6), real_t(6), real_t(6))
        );
        REQUIRE_FALSE(tf::intersects(aabb1, aabb2));
    }
}

// =============================================================================
// OBB tests
// =============================================================================

TEMPLATE_TEST_CASE("obb_point_intersects", "[core][intersects]",
    float, double)
{
    using real_t = TestType;

    SECTION("3D point inside obb") {
        std::array<tf::unit_vector<real_t, 3>, 3> axes = {{
            tf::make_unit_vector(real_t(1), real_t(0), real_t(0)),
            tf::make_unit_vector(real_t(0), real_t(1), real_t(0)),
            tf::make_unit_vector(real_t(0), real_t(0), real_t(1))
        }};
        std::array<real_t, 3> extent = {{real_t(1), real_t(1), real_t(1)}};
        auto obb = tf::make_obb(
            tf::make_point(real_t(0), real_t(0), real_t(0)),
            axes,
            extent
        );
        auto pt = tf::make_point(real_t(0.5), real_t(0.5), real_t(0.5));
        REQUIRE(tf::intersects(obb, pt));
        REQUIRE(tf::intersects(pt, obb));
    }

    SECTION("3D point outside obb") {
        std::array<tf::unit_vector<real_t, 3>, 3> axes = {{
            tf::make_unit_vector(real_t(1), real_t(0), real_t(0)),
            tf::make_unit_vector(real_t(0), real_t(1), real_t(0)),
            tf::make_unit_vector(real_t(0), real_t(0), real_t(1))
        }};
        std::array<real_t, 3> extent = {{real_t(1), real_t(1), real_t(1)}};
        auto obb = tf::make_obb(
            tf::make_point(real_t(0), real_t(0), real_t(0)),
            axes,
            extent
        );
        auto pt = tf::make_point(real_t(2), real_t(2), real_t(2));
        REQUIRE_FALSE(tf::intersects(obb, pt));
    }
}

TEMPLATE_TEST_CASE("obb_obb_intersects", "[core][intersects]",
    float, double)
{
    using real_t = TestType;

    SECTION("3D overlapping obbs") {
        std::array<tf::unit_vector<real_t, 3>, 3> axes = {{
            tf::make_unit_vector(real_t(1), real_t(0), real_t(0)),
            tf::make_unit_vector(real_t(0), real_t(1), real_t(0)),
            tf::make_unit_vector(real_t(0), real_t(0), real_t(1))
        }};
        std::array<real_t, 3> extent = {{real_t(1), real_t(1), real_t(1)}};
        auto obb1 = tf::make_obb(
            tf::make_point(real_t(0), real_t(0), real_t(0)),
            axes,
            extent
        );
        auto obb2 = tf::make_obb(
            tf::make_point(real_t(0.5), real_t(0.5), real_t(0.5)),
            axes,
            extent
        );
        REQUIRE(tf::intersects(obb1, obb2));
    }

    SECTION("3D separated obbs") {
        std::array<tf::unit_vector<real_t, 3>, 3> axes = {{
            tf::make_unit_vector(real_t(1), real_t(0), real_t(0)),
            tf::make_unit_vector(real_t(0), real_t(1), real_t(0)),
            tf::make_unit_vector(real_t(0), real_t(0), real_t(1))
        }};
        std::array<real_t, 3> extent = {{real_t(1), real_t(1), real_t(1)}};
        auto obb1 = tf::make_obb(
            tf::make_point(real_t(0), real_t(0), real_t(0)),
            axes,
            extent
        );
        auto obb2 = tf::make_obb(
            tf::make_point(real_t(5), real_t(5), real_t(5)),
            axes,
            extent
        );
        REQUIRE_FALSE(tf::intersects(obb1, obb2));
    }
}

TEMPLATE_TEST_CASE("obb_aabb_intersects", "[core][intersects]",
    float, double)
{
    using real_t = TestType;

    SECTION("3D obb inside aabb") {
        std::array<tf::unit_vector<real_t, 3>, 3> axes = {{
            tf::make_unit_vector(real_t(1), real_t(0), real_t(0)),
            tf::make_unit_vector(real_t(0), real_t(1), real_t(0)),
            tf::make_unit_vector(real_t(0), real_t(0), real_t(1))
        }};
        std::array<real_t, 3> extent = {{real_t(0.5), real_t(0.5), real_t(0.5)}};
        auto obb = tf::make_obb(
            tf::make_point(real_t(0), real_t(0), real_t(0)),
            axes,
            extent
        );
        auto aabb = tf::make_aabb(
            tf::make_point(real_t(-1), real_t(-1), real_t(-1)),
            tf::make_point(real_t(1), real_t(1), real_t(1))
        );
        REQUIRE(tf::intersects(obb, aabb));
        REQUIRE(tf::intersects(aabb, obb));
    }

    SECTION("3D obb outside aabb") {
        std::array<tf::unit_vector<real_t, 3>, 3> axes = {{
            tf::make_unit_vector(real_t(1), real_t(0), real_t(0)),
            tf::make_unit_vector(real_t(0), real_t(1), real_t(0)),
            tf::make_unit_vector(real_t(0), real_t(0), real_t(1))
        }};
        std::array<real_t, 3> extent = {{real_t(0.5), real_t(0.5), real_t(0.5)}};
        auto obb = tf::make_obb(
            tf::make_point(real_t(5), real_t(5), real_t(5)),
            axes,
            extent
        );
        auto aabb = tf::make_aabb(
            tf::make_point(real_t(-1), real_t(-1), real_t(-1)),
            tf::make_point(real_t(1), real_t(1), real_t(1))
        );
        REQUIRE_FALSE(tf::intersects(obb, aabb));
    }
}

// =============================================================================
// Colinear segment-segment tests
// =============================================================================

TEMPLATE_TEST_CASE("segment_segment_colinear", "[core][intersects]",
    float, double)
{
    using real_t = TestType;

    SECTION("2D colinear overlapping segments") {
        auto seg1 = tf::make_segment_between_points(
            tf::make_point(real_t(0), real_t(0)),
            tf::make_point(real_t(4), real_t(0))
        );
        auto seg2 = tf::make_segment_between_points(
            tf::make_point(real_t(2), real_t(0)),
            tf::make_point(real_t(6), real_t(0))
        );
        REQUIRE(tf::intersects(seg1, seg2));
    }

    SECTION("2D colinear non-overlapping segments") {
        auto seg1 = tf::make_segment_between_points(
            tf::make_point(real_t(0), real_t(0)),
            tf::make_point(real_t(2), real_t(0))
        );
        auto seg2 = tf::make_segment_between_points(
            tf::make_point(real_t(4), real_t(0)),
            tf::make_point(real_t(6), real_t(0))
        );
        REQUIRE_FALSE(tf::intersects(seg1, seg2));
    }

    SECTION("2D colinear touching at endpoint") {
        auto seg1 = tf::make_segment_between_points(
            tf::make_point(real_t(0), real_t(0)),
            tf::make_point(real_t(2), real_t(0))
        );
        auto seg2 = tf::make_segment_between_points(
            tf::make_point(real_t(2), real_t(0)),
            tf::make_point(real_t(4), real_t(0))
        );
        REQUIRE(tf::intersects(seg1, seg2));
    }

    SECTION("2D colinear one inside other") {
        auto seg1 = tf::make_segment_between_points(
            tf::make_point(real_t(0), real_t(0)),
            tf::make_point(real_t(6), real_t(0))
        );
        auto seg2 = tf::make_segment_between_points(
            tf::make_point(real_t(2), real_t(0)),
            tf::make_point(real_t(4), real_t(0))
        );
        REQUIRE(tf::intersects(seg1, seg2));
    }

    SECTION("3D colinear overlapping segments") {
        auto seg1 = tf::make_segment_between_points(
            tf::make_point(real_t(0), real_t(0), real_t(0)),
            tf::make_point(real_t(4), real_t(4), real_t(4))
        );
        auto seg2 = tf::make_segment_between_points(
            tf::make_point(real_t(2), real_t(2), real_t(2)),
            tf::make_point(real_t(6), real_t(6), real_t(6))
        );
        REQUIRE(tf::intersects(seg1, seg2));
    }

    SECTION("3D colinear non-overlapping segments") {
        auto seg1 = tf::make_segment_between_points(
            tf::make_point(real_t(0), real_t(0), real_t(0)),
            tf::make_point(real_t(1), real_t(1), real_t(1))
        );
        auto seg2 = tf::make_segment_between_points(
            tf::make_point(real_t(3), real_t(3), real_t(3)),
            tf::make_point(real_t(4), real_t(4), real_t(4))
        );
        REQUIRE_FALSE(tf::intersects(seg1, seg2));
    }
}

// =============================================================================
// Colinear ray-segment tests
// =============================================================================

TEMPLATE_TEST_CASE("ray_segment_colinear", "[core][intersects]",
    float, double)
{
    using real_t = TestType;

    SECTION("2D ray colinear pointing toward segment") {
        auto ray = tf::make_ray(
            tf::make_point(real_t(0), real_t(0)),
            tf::make_vector(real_t(1), real_t(0))
        );
        auto seg = tf::make_segment_between_points(
            tf::make_point(real_t(2), real_t(0)),
            tf::make_point(real_t(4), real_t(0))
        );
        REQUIRE(tf::intersects(ray, seg));
        REQUIRE(tf::intersects(seg, ray));
    }

    SECTION("2D ray colinear pointing away from segment") {
        auto ray = tf::make_ray(
            tf::make_point(real_t(0), real_t(0)),
            tf::make_vector(real_t(-1), real_t(0))
        );
        auto seg = tf::make_segment_between_points(
            tf::make_point(real_t(2), real_t(0)),
            tf::make_point(real_t(4), real_t(0))
        );
        REQUIRE_FALSE(tf::intersects(ray, seg));
    }

    SECTION("2D ray origin inside colinear segment") {
        auto ray = tf::make_ray(
            tf::make_point(real_t(3), real_t(0)),
            tf::make_vector(real_t(1), real_t(0))
        );
        auto seg = tf::make_segment_between_points(
            tf::make_point(real_t(2), real_t(0)),
            tf::make_point(real_t(4), real_t(0))
        );
        REQUIRE(tf::intersects(ray, seg));
    }

    SECTION("2D ray colinear past segment") {
        auto ray = tf::make_ray(
            tf::make_point(real_t(5), real_t(0)),
            tf::make_vector(real_t(1), real_t(0))
        );
        auto seg = tf::make_segment_between_points(
            tf::make_point(real_t(2), real_t(0)),
            tf::make_point(real_t(4), real_t(0))
        );
        REQUIRE_FALSE(tf::intersects(ray, seg));
    }
}

// =============================================================================
// Colinear ray-ray tests
// =============================================================================

TEMPLATE_TEST_CASE("ray_ray_colinear", "[core][intersects]",
    float, double)
{
    using real_t = TestType;

    SECTION("2D colinear rays same direction, first behind") {
        auto ray1 = tf::make_ray(
            tf::make_point(real_t(0), real_t(0)),
            tf::make_vector(real_t(1), real_t(0))
        );
        auto ray2 = tf::make_ray(
            tf::make_point(real_t(5), real_t(0)),
            tf::make_vector(real_t(1), real_t(0))
        );
        REQUIRE(tf::intersects(ray1, ray2));
    }

    SECTION("2D colinear rays opposite directions, converging") {
        auto ray1 = tf::make_ray(
            tf::make_point(real_t(0), real_t(0)),
            tf::make_vector(real_t(1), real_t(0))
        );
        auto ray2 = tf::make_ray(
            tf::make_point(real_t(5), real_t(0)),
            tf::make_vector(real_t(-1), real_t(0))
        );
        REQUIRE(tf::intersects(ray1, ray2));
    }

    SECTION("2D colinear rays opposite directions, diverging") {
        auto ray1 = tf::make_ray(
            tf::make_point(real_t(0), real_t(0)),
            tf::make_vector(real_t(-1), real_t(0))
        );
        auto ray2 = tf::make_ray(
            tf::make_point(real_t(5), real_t(0)),
            tf::make_vector(real_t(1), real_t(0))
        );
        REQUIRE_FALSE(tf::intersects(ray1, ray2));
    }

    SECTION("2D colinear rays same origin") {
        auto ray1 = tf::make_ray(
            tf::make_point(real_t(0), real_t(0)),
            tf::make_vector(real_t(1), real_t(0))
        );
        auto ray2 = tf::make_ray(
            tf::make_point(real_t(0), real_t(0)),
            tf::make_vector(real_t(-1), real_t(0))
        );
        REQUIRE(tf::intersects(ray1, ray2));
    }
}

// =============================================================================
// Colinear line-segment tests
// =============================================================================

TEMPLATE_TEST_CASE("line_segment_colinear", "[core][intersects]",
    float, double)
{
    using real_t = TestType;

    SECTION("2D colinear line and segment") {
        auto line = tf::make_line_like(
            tf::make_point(real_t(0), real_t(0)),
            tf::make_vector(real_t(1), real_t(0))
        );
        auto seg = tf::make_segment_between_points(
            tf::make_point(real_t(5), real_t(0)),
            tf::make_point(real_t(10), real_t(0))
        );
        // Colinear always intersects (segment lies on line)
        REQUIRE(tf::intersects(line, seg));
        REQUIRE(tf::intersects(seg, line));
    }

    SECTION("3D colinear line and segment") {
        auto line = tf::make_line_like(
            tf::make_point(real_t(0), real_t(0), real_t(0)),
            tf::make_vector(real_t(1), real_t(1), real_t(1))
        );
        auto seg = tf::make_segment_between_points(
            tf::make_point(real_t(5), real_t(5), real_t(5)),
            tf::make_point(real_t(10), real_t(10), real_t(10))
        );
        REQUIRE(tf::intersects(line, seg));
    }
}

// =============================================================================
// Colinear line-line tests
// =============================================================================

TEMPLATE_TEST_CASE("line_line_colinear", "[core][intersects]",
    float, double)
{
    using real_t = TestType;

    SECTION("2D colinear lines (same line)") {
        auto line1 = tf::make_line_like(
            tf::make_point(real_t(0), real_t(0)),
            tf::make_vector(real_t(1), real_t(0))
        );
        auto line2 = tf::make_line_like(
            tf::make_point(real_t(5), real_t(0)),
            tf::make_vector(real_t(2), real_t(0))
        );
        REQUIRE(tf::intersects(line1, line2));
    }

    SECTION("3D colinear lines") {
        auto line1 = tf::make_line_like(
            tf::make_point(real_t(0), real_t(0), real_t(0)),
            tf::make_vector(real_t(1), real_t(1), real_t(1))
        );
        auto line2 = tf::make_line_like(
            tf::make_point(real_t(5), real_t(5), real_t(5)),
            tf::make_vector(real_t(-1), real_t(-1), real_t(-1))
        );
        REQUIRE(tf::intersects(line1, line2));
    }
}

// =============================================================================
// Coplanar ray-polygon tests
// =============================================================================

TEMPLATE_TEST_CASE("ray_polygon_coplanar", "[core][intersects]",
    float, double)
{
    using real_t = TestType;

    // Triangle in XY plane at z=0
    std::array<tf::point<real_t, 3>, 3> triangle_pts = {{
        tf::make_point(real_t(0), real_t(0), real_t(0)),
        tf::make_point(real_t(4), real_t(0), real_t(0)),
        tf::make_point(real_t(2), real_t(4), real_t(0))
    }};
    auto poly = tf::make_polygon(triangle_pts);

    SECTION("ray coplanar, starting inside, pointing outward") {
        auto ray = tf::make_ray(
            tf::make_point(real_t(2), real_t(1), real_t(0)),
            tf::make_vector(real_t(1), real_t(0), real_t(0))
        );
        REQUIRE(tf::intersects(ray, poly));
    }

    SECTION("ray coplanar, starting outside, pointing toward polygon") {
        auto ray = tf::make_ray(
            tf::make_point(real_t(-2), real_t(1), real_t(0)),
            tf::make_vector(real_t(1), real_t(0), real_t(0))
        );
        REQUIRE(tf::intersects(ray, poly));
    }

    SECTION("ray coplanar, starting outside, pointing away") {
        auto ray = tf::make_ray(
            tf::make_point(real_t(-2), real_t(1), real_t(0)),
            tf::make_vector(real_t(-1), real_t(0), real_t(0))
        );
        REQUIRE_FALSE(tf::intersects(ray, poly));
    }

    SECTION("ray coplanar, missing polygon entirely") {
        auto ray = tf::make_ray(
            tf::make_point(real_t(-2), real_t(10), real_t(0)),
            tf::make_vector(real_t(1), real_t(0), real_t(0))
        );
        REQUIRE_FALSE(tf::intersects(ray, poly));
    }
}

// =============================================================================
// Coplanar line-polygon tests
// =============================================================================

TEMPLATE_TEST_CASE("line_polygon_coplanar", "[core][intersects]",
    float, double)
{
    using real_t = TestType;

    // Triangle in XY plane at z=0
    std::array<tf::point<real_t, 3>, 3> triangle_pts = {{
        tf::make_point(real_t(0), real_t(0), real_t(0)),
        tf::make_point(real_t(4), real_t(0), real_t(0)),
        tf::make_point(real_t(2), real_t(4), real_t(0))
    }};
    auto poly = tf::make_polygon(triangle_pts);

    SECTION("line coplanar, passing through polygon") {
        auto line = tf::make_line_like(
            tf::make_point(real_t(2), real_t(1), real_t(0)),
            tf::make_vector(real_t(1), real_t(0), real_t(0))
        );
        REQUIRE(tf::intersects(line, poly));
        REQUIRE(tf::intersects(poly, line));
    }

    SECTION("line coplanar, missing polygon") {
        auto line = tf::make_line_like(
            tf::make_point(real_t(0), real_t(10), real_t(0)),
            tf::make_vector(real_t(1), real_t(0), real_t(0))
        );
        REQUIRE_FALSE(tf::intersects(line, poly));
    }
}

// =============================================================================
// Coplanar segment-polygon tests
// =============================================================================

TEMPLATE_TEST_CASE("segment_polygon_coplanar", "[core][intersects]",
    float, double)
{
    using real_t = TestType;

    // Square in XY plane at z=0
    std::array<tf::point<real_t, 3>, 4> square_pts = {{
        tf::make_point(real_t(0), real_t(0), real_t(0)),
        tf::make_point(real_t(4), real_t(0), real_t(0)),
        tf::make_point(real_t(4), real_t(4), real_t(0)),
        tf::make_point(real_t(0), real_t(4), real_t(0))
    }};
    auto poly = tf::make_polygon(square_pts);

    SECTION("segment coplanar, fully inside polygon") {
        auto seg = tf::make_segment_between_points(
            tf::make_point(real_t(1), real_t(1), real_t(0)),
            tf::make_point(real_t(3), real_t(3), real_t(0))
        );
        REQUIRE(tf::intersects(seg, poly));
        REQUIRE(tf::intersects(poly, seg));
    }

    SECTION("segment coplanar, crossing polygon boundary") {
        auto seg = tf::make_segment_between_points(
            tf::make_point(real_t(-1), real_t(2), real_t(0)),
            tf::make_point(real_t(2), real_t(2), real_t(0))
        );
        REQUIRE(tf::intersects(seg, poly));
    }

    SECTION("segment coplanar, fully outside polygon") {
        auto seg = tf::make_segment_between_points(
            tf::make_point(real_t(5), real_t(0), real_t(0)),
            tf::make_point(real_t(6), real_t(0), real_t(0))
        );
        REQUIRE_FALSE(tf::intersects(seg, poly));
    }

    SECTION("segment coplanar, touching polygon edge") {
        auto seg = tf::make_segment_between_points(
            tf::make_point(real_t(2), real_t(0), real_t(0)),
            tf::make_point(real_t(2), real_t(-2), real_t(0))
        );
        REQUIRE(tf::intersects(seg, poly));
    }
}

// =============================================================================
// Coplanar polygon-polygon tests (non-overlapping)
// =============================================================================

TEMPLATE_TEST_CASE("polygon_polygon_coplanar_separated", "[core][intersects]",
    float, double)
{
    using real_t = TestType;

    SECTION("3D coplanar triangles, separated") {
        std::array<tf::point<real_t, 3>, 3> tri1_pts = {{
            tf::make_point(real_t(0), real_t(0), real_t(0)),
            tf::make_point(real_t(1), real_t(0), real_t(0)),
            tf::make_point(real_t(0.5), real_t(1), real_t(0))
        }};
        std::array<tf::point<real_t, 3>, 3> tri2_pts = {{
            tf::make_point(real_t(5), real_t(0), real_t(0)),
            tf::make_point(real_t(6), real_t(0), real_t(0)),
            tf::make_point(real_t(5.5), real_t(1), real_t(0))
        }};
        auto poly1 = tf::make_polygon(tri1_pts);
        auto poly2 = tf::make_polygon(tri2_pts);
        REQUIRE_FALSE(tf::intersects(poly1, poly2));
    }

    SECTION("3D coplanar triangles, touching at vertex") {
        std::array<tf::point<real_t, 3>, 3> tri1_pts = {{
            tf::make_point(real_t(0), real_t(0), real_t(0)),
            tf::make_point(real_t(2), real_t(0), real_t(0)),
            tf::make_point(real_t(1), real_t(2), real_t(0))
        }};
        std::array<tf::point<real_t, 3>, 3> tri2_pts = {{
            tf::make_point(real_t(2), real_t(0), real_t(0)),
            tf::make_point(real_t(4), real_t(0), real_t(0)),
            tf::make_point(real_t(3), real_t(2), real_t(0))
        }};
        auto poly1 = tf::make_polygon(tri1_pts);
        auto poly2 = tf::make_polygon(tri2_pts);
        REQUIRE(tf::intersects(poly1, poly2));
    }

    SECTION("3D coplanar triangles, sharing edge") {
        std::array<tf::point<real_t, 3>, 3> tri1_pts = {{
            tf::make_point(real_t(0), real_t(0), real_t(0)),
            tf::make_point(real_t(2), real_t(0), real_t(0)),
            tf::make_point(real_t(1), real_t(2), real_t(0))
        }};
        std::array<tf::point<real_t, 3>, 3> tri2_pts = {{
            tf::make_point(real_t(0), real_t(0), real_t(0)),
            tf::make_point(real_t(2), real_t(0), real_t(0)),
            tf::make_point(real_t(1), real_t(-2), real_t(0))
        }};
        auto poly1 = tf::make_polygon(tri1_pts);
        auto poly2 = tf::make_polygon(tri2_pts);
        REQUIRE(tf::intersects(poly1, poly2));
    }
}

// =============================================================================
// Segment colinear with polygon edge tests
// =============================================================================

TEMPLATE_TEST_CASE("segment_colinear_with_polygon_edge", "[core][intersects]",
    float, double)
{
    using real_t = TestType;

    // Square in XY plane at z=0, edges along x and y axes
    std::array<tf::point<real_t, 3>, 4> square_pts = {{
        tf::make_point(real_t(0), real_t(0), real_t(0)),
        tf::make_point(real_t(4), real_t(0), real_t(0)),
        tf::make_point(real_t(4), real_t(4), real_t(0)),
        tf::make_point(real_t(0), real_t(4), real_t(0))
    }};
    auto poly = tf::make_polygon(square_pts);

    SECTION("segment overlapping polygon edge") {
        // Segment on bottom edge, overlapping
        auto seg = tf::make_segment_between_points(
            tf::make_point(real_t(1), real_t(0), real_t(0)),
            tf::make_point(real_t(3), real_t(0), real_t(0))
        );
        REQUIRE(tf::intersects(seg, poly));
    }

    SECTION("segment colinear with edge but non-overlapping") {
        // Segment on line of bottom edge, but outside polygon
        auto seg = tf::make_segment_between_points(
            tf::make_point(real_t(5), real_t(0), real_t(0)),
            tf::make_point(real_t(7), real_t(0), real_t(0))
        );
        REQUIRE_FALSE(tf::intersects(seg, poly));
    }

    SECTION("segment extending beyond polygon edge") {
        // Segment on bottom edge line, partially overlapping
        auto seg = tf::make_segment_between_points(
            tf::make_point(real_t(-2), real_t(0), real_t(0)),
            tf::make_point(real_t(2), real_t(0), real_t(0))
        );
        REQUIRE(tf::intersects(seg, poly));
    }

    SECTION("segment exactly matching polygon edge") {
        auto seg = tf::make_segment_between_points(
            tf::make_point(real_t(0), real_t(0), real_t(0)),
            tf::make_point(real_t(4), real_t(0), real_t(0))
        );
        REQUIRE(tf::intersects(seg, poly));
    }
}

// =============================================================================
// Ray colinear with polygon edge tests
// =============================================================================

TEMPLATE_TEST_CASE("ray_colinear_with_polygon_edge", "[core][intersects]",
    float, double)
{
    using real_t = TestType;

    // Square in XY plane at z=0
    std::array<tf::point<real_t, 3>, 4> square_pts = {{
        tf::make_point(real_t(0), real_t(0), real_t(0)),
        tf::make_point(real_t(4), real_t(0), real_t(0)),
        tf::make_point(real_t(4), real_t(4), real_t(0)),
        tf::make_point(real_t(0), real_t(4), real_t(0))
    }};
    auto poly = tf::make_polygon(square_pts);

    SECTION("ray origin on edge, pointing along edge into polygon range") {
        auto ray = tf::make_ray(
            tf::make_point(real_t(2), real_t(0), real_t(0)),
            tf::make_vector(real_t(1), real_t(0), real_t(0))
        );
        REQUIRE(tf::intersects(ray, poly));
    }

    SECTION("ray origin before edge, pointing toward edge") {
        auto ray = tf::make_ray(
            tf::make_point(real_t(-2), real_t(0), real_t(0)),
            tf::make_vector(real_t(1), real_t(0), real_t(0))
        );
        REQUIRE(tf::intersects(ray, poly));
    }

    SECTION("ray origin after edge, pointing away") {
        auto ray = tf::make_ray(
            tf::make_point(real_t(5), real_t(0), real_t(0)),
            tf::make_vector(real_t(1), real_t(0), real_t(0))
        );
        REQUIRE_FALSE(tf::intersects(ray, poly));
    }

    SECTION("ray origin on edge, pointing away from polygon") {
        auto ray = tf::make_ray(
            tf::make_point(real_t(2), real_t(0), real_t(0)),
            tf::make_vector(real_t(-1), real_t(0), real_t(0))
        );
        REQUIRE(tf::intersects(ray, poly));
    }
}

// =============================================================================
// Line colinear with polygon edge tests
// =============================================================================

TEMPLATE_TEST_CASE("line_colinear_with_polygon_edge", "[core][intersects]",
    float, double)
{
    using real_t = TestType;

    // Square in XY plane at z=0
    std::array<tf::point<real_t, 3>, 4> square_pts = {{
        tf::make_point(real_t(0), real_t(0), real_t(0)),
        tf::make_point(real_t(4), real_t(0), real_t(0)),
        tf::make_point(real_t(4), real_t(4), real_t(0)),
        tf::make_point(real_t(0), real_t(4), real_t(0))
    }};
    auto poly = tf::make_polygon(square_pts);

    SECTION("line colinear with polygon edge") {
        auto line = tf::make_line_like(
            tf::make_point(real_t(10), real_t(0), real_t(0)),
            tf::make_vector(real_t(1), real_t(0), real_t(0))
        );
        REQUIRE(tf::intersects(line, poly));
    }

    SECTION("line parallel to edge but offset") {
        auto line = tf::make_line_like(
            tf::make_point(real_t(0), real_t(-1), real_t(0)),
            tf::make_vector(real_t(1), real_t(0), real_t(0))
        );
        REQUIRE_FALSE(tf::intersects(line, poly));
    }
}

// =============================================================================
// Point on polygon edge/vertex tests
// =============================================================================

TEMPLATE_TEST_CASE("point_on_polygon_edge_vertex", "[core][intersects]",
    float, double)
{
    using real_t = TestType;

    // Triangle in XY plane at z=0
    std::array<tf::point<real_t, 3>, 3> triangle_pts = {{
        tf::make_point(real_t(0), real_t(0), real_t(0)),
        tf::make_point(real_t(4), real_t(0), real_t(0)),
        tf::make_point(real_t(2), real_t(4), real_t(0))
    }};
    auto poly = tf::make_polygon(triangle_pts);

    SECTION("point exactly on vertex") {
        auto pt = tf::make_point(real_t(0), real_t(0), real_t(0));
        REQUIRE(tf::intersects(pt, poly));
    }

    SECTION("point on edge midpoint") {
        auto pt = tf::make_point(real_t(2), real_t(0), real_t(0));
        REQUIRE(tf::intersects(pt, poly));
    }

    SECTION("point on diagonal edge") {
        // Point on edge from (4,0,0) to (2,4,0): midpoint is (3,2,0)
        auto pt = tf::make_point(real_t(3), real_t(2), real_t(0));
        REQUIRE(tf::intersects(pt, poly));
    }

    SECTION("point near edge but off") {
        auto pt = tf::make_point(real_t(2), real_t(0.001), real_t(0));
        // This is inside the polygon, should intersect
        REQUIRE(tf::intersects(pt, poly));
    }
}

// =============================================================================
// Swap symmetry tests
// =============================================================================

TEMPLATE_TEST_CASE("intersects_swap_symmetry", "[core][intersects]",
    float, double)
{
    using real_t = TestType;

    SECTION("point-segment symmetry") {
        auto seg = tf::make_segment_between_points(
            tf::make_point(real_t(0), real_t(0)),
            tf::make_point(real_t(4), real_t(0))
        );
        auto pt = tf::make_point(real_t(2), real_t(0));
        REQUIRE(tf::intersects(pt, seg) == tf::intersects(seg, pt));
    }

    SECTION("ray-polygon symmetry") {
        std::array<tf::point<real_t, 3>, 3> triangle_pts = {{
            tf::make_point(real_t(0), real_t(0), real_t(0)),
            tf::make_point(real_t(2), real_t(0), real_t(0)),
            tf::make_point(real_t(1), real_t(2), real_t(0))
        }};
        auto poly = tf::make_polygon(triangle_pts);
        auto ray = tf::make_ray(
            tf::make_point(real_t(1), real_t(0.5), real_t(2)),
            tf::make_vector(real_t(0), real_t(0), real_t(-1))
        );
        REQUIRE(tf::intersects(ray, poly) == tf::intersects(poly, ray));
    }

    SECTION("plane-segment symmetry") {
        auto plane = tf::make_plane(
            tf::make_unit_vector(real_t(0), real_t(0), real_t(1)),
            real_t(0)
        );
        auto seg = tf::make_segment_between_points(
            tf::make_point(real_t(0), real_t(0), real_t(-1)),
            tf::make_point(real_t(0), real_t(0), real_t(1))
        );
        REQUIRE(tf::intersects(plane, seg) == tf::intersects(seg, plane));
    }

    SECTION("aabb-aabb symmetry") {
        auto aabb1 = tf::make_aabb(
            tf::make_point(real_t(0), real_t(0)),
            tf::make_point(real_t(2), real_t(2))
        );
        auto aabb2 = tf::make_aabb(
            tf::make_point(real_t(1), real_t(1)),
            tf::make_point(real_t(3), real_t(3))
        );
        REQUIRE(tf::intersects(aabb1, aabb2) == tf::intersects(aabb2, aabb1));
    }
}
