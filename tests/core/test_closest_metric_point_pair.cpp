/**
 * @file test_closest_metric_point_pair.cpp
 * @brief Tests for closest_metric_point_pair functionality
 *
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <trueform/core.hpp>
#include <array>
#include <cmath>

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

// =============================================================================
// Helper functions
// =============================================================================

template <typename real_t>
auto approx_zero(real_t value, real_t tol = real_t(1e-5)) -> bool
{
    return std::abs(value) < tol;
}

template <typename real_t>
auto approx_equal(real_t a, real_t b, real_t tol = real_t(1e-5)) -> bool
{
    return std::abs(a - b) < tol;
}

template <typename point_t, typename real_t>
auto points_close(const point_t& a, const point_t& b, real_t tol = real_t(1e-5)) -> bool
{
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (std::abs(a[i] - b[i]) > tol) return false;
    }
    return true;
}

// =============================================================================
// Point-Polygon tests
// =============================================================================

TEST_CASE("point_polygon_2d_inside", "[core][closest_metric_point_pair]")
{
    std::array<tf::point<float, 2>, 4> square_pts = {{
        tf::make_point(0.0f, 0.0f),
        tf::make_point(1.0f, 0.0f),
        tf::make_point(1.0f, 1.0f),
        tf::make_point(0.0f, 1.0f)
    }};
    auto poly = tf::make_polygon(square_pts);

    auto pt_inside = tf::make_point(0.5f, 0.5f);

    auto [dist2, p0, p1] = tf::closest_metric_point_pair(pt_inside, poly);
    REQUIRE(dist2 == 0.0f);
}

TEST_CASE("point_polygon_2d_outside", "[core][closest_metric_point_pair]")
{
    std::array<tf::point<float, 2>, 4> square_pts = {{
        tf::make_point(0.0f, 0.0f),
        tf::make_point(1.0f, 0.0f),
        tf::make_point(1.0f, 1.0f),
        tf::make_point(0.0f, 1.0f)
    }};
    auto poly = tf::make_polygon(square_pts);

    auto pt_outside = tf::make_point(2.0f, 0.5f);

    auto [dist2, p0, p1] = tf::closest_metric_point_pair(pt_outside, poly);
    REQUIRE_THAT(dist2, WithinAbs(1.0f, 1e-5f));
    REQUIRE(approx_equal(p1[0], 1.0f));
    REQUIRE(approx_equal(p1[1], 0.5f));
}

TEST_CASE("point_polygon_3d_inside", "[core][closest_metric_point_pair]")
{
    std::array<tf::point<double, 3>, 3> triangle_pts = {{
        tf::make_point(0.0, 0.0, 0.0),
        tf::make_point(1.0, 0.0, 0.0),
        tf::make_point(0.5, 1.0, 0.0)
    }};
    auto poly = tf::make_polygon(triangle_pts);

    auto pt_inside = tf::make_point(0.5, 0.3, 0.0);

    auto [dist2, p0, p1] = tf::closest_metric_point_pair(pt_inside, poly);
    REQUIRE(approx_zero(dist2));
}

TEST_CASE("point_polygon_3d_above", "[core][closest_metric_point_pair]")
{
    std::array<tf::point<double, 3>, 3> triangle_pts = {{
        tf::make_point(0.0, 0.0, 0.0),
        tf::make_point(1.0, 0.0, 0.0),
        tf::make_point(0.5, 1.0, 0.0)
    }};
    auto poly = tf::make_polygon(triangle_pts);

    auto pt_above = tf::make_point(0.5, 0.3, 2.0);

    auto [dist2, p0, p1] = tf::closest_metric_point_pair(pt_above, poly);
    REQUIRE_THAT(dist2, WithinAbs(4.0, 1e-5));
    REQUIRE(approx_equal(p1[0], 0.5));
    REQUIRE(approx_equal(p1[1], 0.3));
    REQUIRE(approx_equal(p1[2], 0.0));
}

// =============================================================================
// Polygon-Polygon tests
// =============================================================================

TEST_CASE("polygon_polygon_2d_separate", "[core][closest_metric_point_pair]")
{
    std::array<tf::point<float, 2>, 4> square1_pts = {{
        tf::make_point(0.0f, 0.0f),
        tf::make_point(1.0f, 0.0f),
        tf::make_point(1.0f, 1.0f),
        tf::make_point(0.0f, 1.0f)
    }};
    auto poly1 = tf::make_polygon(square1_pts);

    std::array<tf::point<float, 2>, 4> square2_pts = {{
        tf::make_point(2.0f, 0.0f),
        tf::make_point(3.0f, 0.0f),
        tf::make_point(3.0f, 1.0f),
        tf::make_point(2.0f, 1.0f)
    }};
    auto poly2 = tf::make_polygon(square2_pts);

    auto [dist2, p0, p1] = tf::closest_metric_point_pair(poly1, poly2);
    REQUIRE_THAT(dist2, WithinAbs(1.0f, 1e-5f));
}

TEST_CASE("polygon_polygon_2d_overlapping", "[core][closest_metric_point_pair]")
{
    std::array<tf::point<float, 2>, 4> square1_pts = {{
        tf::make_point(0.0f, 0.0f),
        tf::make_point(1.0f, 0.0f),
        tf::make_point(1.0f, 1.0f),
        tf::make_point(0.0f, 1.0f)
    }};
    auto poly1 = tf::make_polygon(square1_pts);

    std::array<tf::point<float, 2>, 4> square3_pts = {{
        tf::make_point(0.5f, 0.5f),
        tf::make_point(1.5f, 0.5f),
        tf::make_point(1.5f, 1.5f),
        tf::make_point(0.5f, 1.5f)
    }};
    auto poly3 = tf::make_polygon(square3_pts);

    auto [dist2, p0, p1] = tf::closest_metric_point_pair(poly1, poly3);
    REQUIRE(dist2 == 0.0f);
}

TEST_CASE("polygon_polygon_3d", "[core][closest_metric_point_pair]")
{
    std::array<tf::point<double, 3>, 3> triangle1_pts = {{
        tf::make_point(0.0, 0.0, 0.0),
        tf::make_point(1.0, 0.0, 0.0),
        tf::make_point(0.5, 1.0, 0.0)
    }};
    auto poly1 = tf::make_polygon(triangle1_pts);

    std::array<tf::point<double, 3>, 3> triangle2_pts = {{
        tf::make_point(0.0, 0.0, 2.0),
        tf::make_point(1.0, 0.0, 2.0),
        tf::make_point(0.5, 1.0, 2.0)
    }};
    auto poly2 = tf::make_polygon(triangle2_pts);

    auto [dist2, p0, p1] = tf::closest_metric_point_pair(poly1, poly2);
    REQUIRE_THAT(dist2, WithinAbs(4.0, 1e-5));
}

// =============================================================================
// Segment-Polygon tests
// =============================================================================

TEST_CASE("segment_polygon_2d_intersecting", "[core][closest_metric_point_pair]")
{
    std::array<tf::point<float, 2>, 4> square_pts = {{
        tf::make_point(0.0f, 0.0f),
        tf::make_point(1.0f, 0.0f),
        tf::make_point(1.0f, 1.0f),
        tf::make_point(0.0f, 1.0f)
    }};
    auto poly = tf::make_polygon(square_pts);

    auto seg = tf::make_segment_between_points(
        tf::make_point(0.5f, -0.5f),
        tf::make_point(0.5f, 1.5f)
    );

    auto [dist2, p0, p1] = tf::closest_metric_point_pair(seg, poly);
    REQUIRE(dist2 == 0.0f);
}

TEST_CASE("segment_polygon_2d_outside", "[core][closest_metric_point_pair]")
{
    std::array<tf::point<float, 2>, 4> square_pts = {{
        tf::make_point(0.0f, 0.0f),
        tf::make_point(1.0f, 0.0f),
        tf::make_point(1.0f, 1.0f),
        tf::make_point(0.0f, 1.0f)
    }};
    auto poly = tf::make_polygon(square_pts);

    auto seg = tf::make_segment_between_points(
        tf::make_point(2.0f, 0.0f),
        tf::make_point(3.0f, 0.0f)
    );

    auto [dist2, p0, p1] = tf::closest_metric_point_pair(seg, poly);
    REQUIRE_THAT(dist2, WithinAbs(1.0f, 1e-5f));
}

// =============================================================================
// Ray-Polygon tests
// =============================================================================

TEST_CASE("ray_polygon_3d_hitting", "[core][closest_metric_point_pair]")
{
    std::array<tf::point<float, 3>, 3> triangle_pts = {{
        tf::make_point(0.0f, 0.0f, 0.0f),
        tf::make_point(1.0f, 0.0f, 0.0f),
        tf::make_point(0.5f, 1.0f, 0.0f)
    }};
    auto poly = tf::make_polygon(triangle_pts);

    auto ray = tf::make_ray(
        tf::make_point(0.5f, 0.3f, 2.0f),
        tf::make_vector(0.0f, 0.0f, -1.0f)
    );

    auto [dist2, p0, p1] = tf::closest_metric_point_pair(ray, poly);
    REQUIRE(dist2 == 0.0f);
}

TEST_CASE("ray_polygon_3d_missing", "[core][closest_metric_point_pair]")
{
    std::array<tf::point<float, 3>, 3> triangle_pts = {{
        tf::make_point(0.0f, 0.0f, 0.0f),
        tf::make_point(1.0f, 0.0f, 0.0f),
        tf::make_point(0.5f, 1.0f, 0.0f)
    }};
    auto poly = tf::make_polygon(triangle_pts);

    auto ray = tf::make_ray(
        tf::make_point(0.5f, 0.3f, 2.0f),
        tf::make_vector(0.0f, 0.0f, 1.0f)
    );

    auto [dist2, p0, p1] = tf::closest_metric_point_pair(ray, poly);
    REQUIRE_THAT(dist2, WithinAbs(4.0f, 1e-5f));
}

// =============================================================================
// Line-Polygon tests
// =============================================================================

TEST_CASE("line_polygon_2d_intersecting", "[core][closest_metric_point_pair]")
{
    std::array<tf::point<float, 2>, 4> square_pts = {{
        tf::make_point(0.0f, 0.0f),
        tf::make_point(1.0f, 0.0f),
        tf::make_point(1.0f, 1.0f),
        tf::make_point(0.0f, 1.0f)
    }};
    auto poly = tf::make_polygon(square_pts);

    auto line = tf::make_line_like(
        tf::make_point(0.5f, -1.0f),
        tf::make_vector(0.0f, 1.0f)
    );

    auto [dist2, p0, p1] = tf::closest_metric_point_pair(line, poly);
    REQUIRE(dist2 == 0.0f);
}

TEST_CASE("line_polygon_2d_parallel", "[core][closest_metric_point_pair]")
{
    std::array<tf::point<float, 2>, 4> square_pts = {{
        tf::make_point(0.0f, 0.0f),
        tf::make_point(1.0f, 0.0f),
        tf::make_point(1.0f, 1.0f),
        tf::make_point(0.0f, 1.0f)
    }};
    auto poly = tf::make_polygon(square_pts);

    auto line = tf::make_line_like(
        tf::make_point(2.0f, 0.0f),
        tf::make_vector(0.0f, 1.0f)
    );

    auto [dist2, p0, p1] = tf::closest_metric_point_pair(line, poly);
    REQUIRE_THAT(dist2, WithinAbs(1.0f, 1e-5f));
}

// =============================================================================
// Point-Plane tests (3D only)
// =============================================================================

TEMPLATE_TEST_CASE("point_plane_on_plane", "[core][closest_metric_point_pair]",
    float, double)
{
    using real_t = TestType;

    auto plane = tf::make_plane(
        tf::make_unit_vector(real_t(0), real_t(0), real_t(1)),
        real_t(0)
    );
    auto pt_on = tf::make_point(real_t(1), real_t(2), real_t(0));

    auto [dist2, p0, p1] = tf::closest_metric_point_pair(pt_on, plane);
    REQUIRE(approx_zero(dist2, real_t(1e-5)));
}

TEMPLATE_TEST_CASE("point_plane_above", "[core][closest_metric_point_pair]",
    float, double)
{
    using real_t = TestType;

    auto plane = tf::make_plane(
        tf::make_unit_vector(real_t(0), real_t(0), real_t(1)),
        real_t(0)
    );
    auto pt_above = tf::make_point(real_t(1), real_t(2), real_t(5));

    auto [dist2, p0, p1] = tf::closest_metric_point_pair(pt_above, plane);
    REQUIRE(approx_equal(dist2, real_t(25), real_t(1e-4)));
    REQUIRE(approx_equal(p1[0], real_t(1)));
    REQUIRE(approx_equal(p1[1], real_t(2)));
    REQUIRE(approx_equal(p1[2], real_t(0)));
}

TEMPLATE_TEST_CASE("plane_point_swap", "[core][closest_metric_point_pair]",
    float, double)
{
    using real_t = TestType;

    auto plane = tf::make_plane(
        tf::make_unit_vector(real_t(0), real_t(0), real_t(1)),
        real_t(0)
    );
    auto pt_above = tf::make_point(real_t(1), real_t(2), real_t(5));

    auto [dist2_swap, p0_swap, p1_swap] = tf::closest_metric_point_pair(plane, pt_above);
    REQUIRE(approx_equal(dist2_swap, real_t(25), real_t(1e-4)));
    REQUIRE(approx_equal(p0_swap[2], real_t(0)));
    REQUIRE(approx_equal(p1_swap[2], real_t(5)));
}

// =============================================================================
// Segment-Plane tests
// =============================================================================

TEMPLATE_TEST_CASE("segment_plane_parallel", "[core][closest_metric_point_pair]",
    float, double)
{
    using real_t = TestType;

    auto plane = tf::make_plane(
        tf::make_unit_vector(real_t(0), real_t(0), real_t(1)),
        real_t(0)
    );
    auto seg = tf::make_segment_between_points(
        tf::make_point(real_t(0), real_t(0), real_t(3)),
        tf::make_point(real_t(1), real_t(0), real_t(3))
    );

    auto [dist2, p0, p1] = tf::closest_metric_point_pair(seg, plane);
    REQUIRE(approx_equal(dist2, real_t(9), real_t(1e-4)));
}

TEMPLATE_TEST_CASE("segment_plane_crossing", "[core][closest_metric_point_pair]",
    float, double)
{
    using real_t = TestType;

    auto plane = tf::make_plane(
        tf::make_unit_vector(real_t(0), real_t(0), real_t(1)),
        real_t(0)
    );
    auto seg_cross = tf::make_segment_between_points(
        tf::make_point(real_t(0), real_t(0), real_t(-1)),
        tf::make_point(real_t(0), real_t(0), real_t(1))
    );

    auto [dist2, p0, p1] = tf::closest_metric_point_pair(seg_cross, plane);
    REQUIRE(approx_zero(dist2, real_t(1e-5)));
}

// =============================================================================
// Ray-Plane tests
// =============================================================================

TEMPLATE_TEST_CASE("ray_plane_toward", "[core][closest_metric_point_pair]",
    float, double)
{
    using real_t = TestType;

    auto plane = tf::make_plane(
        tf::make_unit_vector(real_t(0), real_t(0), real_t(1)),
        real_t(0)
    );
    auto ray = tf::make_ray(
        tf::make_point(real_t(0), real_t(0), real_t(5)),
        tf::make_vector(real_t(0), real_t(0), real_t(-1))
    );

    auto [dist2, p0, p1] = tf::closest_metric_point_pair(ray, plane);
    REQUIRE(approx_zero(dist2, real_t(1e-5)));
}

TEMPLATE_TEST_CASE("ray_plane_parallel", "[core][closest_metric_point_pair]",
    float, double)
{
    using real_t = TestType;

    auto plane = tf::make_plane(
        tf::make_unit_vector(real_t(0), real_t(0), real_t(1)),
        real_t(0)
    );
    auto ray_parallel = tf::make_ray(
        tf::make_point(real_t(0), real_t(0), real_t(5)),
        tf::make_vector(real_t(1), real_t(0), real_t(0))
    );

    auto [dist2, p0, p1] = tf::closest_metric_point_pair(ray_parallel, plane);
    REQUIRE(approx_equal(dist2, real_t(25), real_t(1e-4)));
}

// =============================================================================
// Line-Plane tests
// =============================================================================

TEMPLATE_TEST_CASE("line_plane_intersecting", "[core][closest_metric_point_pair]",
    float, double)
{
    using real_t = TestType;

    auto plane = tf::make_plane(
        tf::make_unit_vector(real_t(0), real_t(0), real_t(1)),
        real_t(0)
    );
    auto line = tf::make_line_like(
        tf::make_point(real_t(0), real_t(0), real_t(5)),
        tf::make_vector(real_t(0), real_t(0), real_t(1))
    );

    auto [dist2, p0, p1] = tf::closest_metric_point_pair(line, plane);
    REQUIRE(approx_zero(dist2, real_t(1e-5)));
}

TEMPLATE_TEST_CASE("line_plane_parallel", "[core][closest_metric_point_pair]",
    float, double)
{
    using real_t = TestType;

    auto plane = tf::make_plane(
        tf::make_unit_vector(real_t(0), real_t(0), real_t(1)),
        real_t(0)
    );
    auto line_parallel = tf::make_line_like(
        tf::make_point(real_t(0), real_t(0), real_t(3)),
        tf::make_vector(real_t(1), real_t(0), real_t(0))
    );

    auto [dist2, p0, p1] = tf::closest_metric_point_pair(line_parallel, plane);
    REQUIRE(approx_equal(dist2, real_t(9), real_t(1e-4)));
}

// =============================================================================
// Point-Point tests
// =============================================================================

TEMPLATE_TEST_CASE("point_point_separated", "[core][closest_metric_point_pair]",
    float, double)
{
    using real_t = TestType;

    SECTION("2D") {
        auto p0 = tf::make_point(real_t(0), real_t(0));
        auto p1 = tf::make_point(real_t(3), real_t(0));

        auto [dist2, c0, c1] = tf::closest_metric_point_pair(p0, p1);
        REQUIRE(approx_equal(dist2, real_t(9)));
        REQUIRE(approx_equal(tf::distance2(p0, p1), dist2));
    }

    SECTION("3D") {
        auto p0 = tf::make_point(real_t(0), real_t(0), real_t(0));
        auto p1 = tf::make_point(real_t(3), real_t(0), real_t(0));

        auto [dist2, c0, c1] = tf::closest_metric_point_pair(p0, p1);
        REQUIRE(approx_equal(dist2, real_t(9)));
        REQUIRE(approx_equal(tf::distance2(p0, p1), dist2));
    }
}

// =============================================================================
// Point-Segment tests
// =============================================================================

TEMPLATE_TEST_CASE("point_segment_perpendicular", "[core][closest_metric_point_pair]",
    float, double)
{
    using real_t = TestType;

    SECTION("2D") {
        auto seg = tf::make_segment_between_points(
            tf::make_point(real_t(0), real_t(0)),
            tf::make_point(real_t(4), real_t(0))
        );
        auto pt = tf::make_point(real_t(2), real_t(3));

        auto [dist2, c0, c1] = tf::closest_metric_point_pair(pt, seg);
        REQUIRE(approx_equal(dist2, real_t(9)));
        REQUIRE(approx_equal(c1[0], real_t(2)));
        REQUIRE(approx_equal(c1[1], real_t(0)));
        REQUIRE(approx_equal(tf::distance2(pt, seg), dist2));
    }

    SECTION("3D") {
        auto seg = tf::make_segment_between_points(
            tf::make_point(real_t(0), real_t(0), real_t(0)),
            tf::make_point(real_t(4), real_t(0), real_t(0))
        );
        auto pt = tf::make_point(real_t(2), real_t(3), real_t(0));

        auto [dist2, c0, c1] = tf::closest_metric_point_pair(pt, seg);
        REQUIRE(approx_equal(dist2, real_t(9)));
        REQUIRE(approx_equal(c1[0], real_t(2)));
        REQUIRE(approx_equal(c1[1], real_t(0)));
        REQUIRE(approx_equal(tf::distance2(pt, seg), dist2));
    }
}

TEMPLATE_TEST_CASE("point_segment_endpoint", "[core][closest_metric_point_pair]",
    float, double)
{
    using real_t = TestType;

    auto seg = tf::make_segment_between_points(
        tf::make_point(real_t(0), real_t(0)),
        tf::make_point(real_t(4), real_t(0))
    );
    auto pt = tf::make_point(real_t(-2), real_t(0));

    auto [dist2, c0, c1] = tf::closest_metric_point_pair(pt, seg);
    REQUIRE(approx_equal(dist2, real_t(4)));
    REQUIRE(approx_equal(c1[0], real_t(0)));
    REQUIRE(approx_equal(c1[1], real_t(0)));
    REQUIRE(approx_equal(tf::distance2(pt, seg), dist2));
}

// =============================================================================
// Point-Ray tests
// =============================================================================

TEMPLATE_TEST_CASE("point_ray_perpendicular", "[core][closest_metric_point_pair]",
    float, double)
{
    using real_t = TestType;

    SECTION("2D") {
        auto ray = tf::make_ray(
            tf::make_point(real_t(0), real_t(0)),
            tf::make_vector(real_t(1), real_t(0))
        );
        auto pt = tf::make_point(real_t(3), real_t(4));

        auto [dist2, c0, c1] = tf::closest_metric_point_pair(pt, ray);
        REQUIRE(approx_equal(dist2, real_t(16)));
        REQUIRE(approx_equal(c1[0], real_t(3)));
        REQUIRE(approx_equal(c1[1], real_t(0)));
        REQUIRE(approx_equal(tf::distance2(pt, ray), dist2));
    }
}

TEMPLATE_TEST_CASE("point_ray_behind_origin", "[core][closest_metric_point_pair]",
    float, double)
{
    using real_t = TestType;

    auto ray = tf::make_ray(
        tf::make_point(real_t(0), real_t(0)),
        tf::make_vector(real_t(1), real_t(0))
    );
    auto pt = tf::make_point(real_t(-3), real_t(4));

    auto [dist2, c0, c1] = tf::closest_metric_point_pair(pt, ray);
    REQUIRE(approx_equal(dist2, real_t(25)));
    REQUIRE(approx_equal(c1[0], real_t(0)));
    REQUIRE(approx_equal(c1[1], real_t(0)));
    REQUIRE(approx_equal(tf::distance2(pt, ray), dist2));
}

// =============================================================================
// Point-Line tests
// =============================================================================

TEMPLATE_TEST_CASE("point_line_perpendicular", "[core][closest_metric_point_pair]",
    float, double)
{
    using real_t = TestType;

    auto line = tf::make_line_like(
        tf::make_point(real_t(0), real_t(0)),
        tf::make_vector(real_t(1), real_t(0))
    );
    auto pt = tf::make_point(real_t(5), real_t(12));

    auto [dist2, c0, c1] = tf::closest_metric_point_pair(pt, line);
    REQUIRE(approx_equal(dist2, real_t(144)));
    REQUIRE(approx_equal(c1[0], real_t(5)));
    REQUIRE(approx_equal(c1[1], real_t(0)));
    REQUIRE(approx_equal(tf::distance2(pt, line), dist2));
}

// =============================================================================
// Segment-Segment tests
// =============================================================================

TEMPLATE_TEST_CASE("segment_segment_parallel", "[core][closest_metric_point_pair]",
    float, double)
{
    using real_t = TestType;

    auto seg1 = tf::make_segment_between_points(
        tf::make_point(real_t(0), real_t(0)),
        tf::make_point(real_t(4), real_t(0))
    );
    auto seg2 = tf::make_segment_between_points(
        tf::make_point(real_t(0), real_t(3)),
        tf::make_point(real_t(4), real_t(3))
    );

    auto [dist2, c0, c1] = tf::closest_metric_point_pair(seg1, seg2);
    REQUIRE(approx_equal(dist2, real_t(9)));
    REQUIRE(approx_equal(c0[0], c1[0]));  // x-coordinates should match
    REQUIRE(approx_equal(tf::distance2(seg1, seg2), dist2));
}

TEMPLATE_TEST_CASE("segment_segment_endpoint_to_midpoint", "[core][closest_metric_point_pair]",
    float, double)
{
    using real_t = TestType;

    auto seg1 = tf::make_segment_between_points(
        tf::make_point(real_t(0), real_t(0)),
        tf::make_point(real_t(4), real_t(0))
    );
    auto seg2 = tf::make_segment_between_points(
        tf::make_point(real_t(2), real_t(3)),
        tf::make_point(real_t(2), real_t(6))
    );

    auto [dist2, c0, c1] = tf::closest_metric_point_pair(seg1, seg2);
    REQUIRE(approx_equal(dist2, real_t(9)));
    REQUIRE(approx_equal(c0[0], real_t(2)));
    REQUIRE(approx_equal(c0[1], real_t(0)));
    REQUIRE(approx_equal(c1[0], real_t(2)));
    REQUIRE(approx_equal(c1[1], real_t(3)));
    REQUIRE(approx_equal(tf::distance2(seg1, seg2), dist2));
}

// =============================================================================
// Ray-Ray tests
// =============================================================================

TEMPLATE_TEST_CASE("ray_ray_diverging", "[core][closest_metric_point_pair]",
    float, double)
{
    using real_t = TestType;

    auto ray1 = tf::make_ray(
        tf::make_point(real_t(0), real_t(0)),
        tf::make_vector(real_t(1), real_t(0))
    );
    auto ray2 = tf::make_ray(
        tf::make_point(real_t(0), real_t(4)),
        tf::make_vector(real_t(1), real_t(0))
    );

    auto [dist2, c0, c1] = tf::closest_metric_point_pair(ray1, ray2);
    REQUIRE(approx_equal(dist2, real_t(16)));
    REQUIRE(approx_equal(c0[0], c1[0]));  // x-coordinates should match
    REQUIRE(approx_equal(tf::distance2(ray1, ray2), dist2));
}

// =============================================================================
// Line-Line tests
// =============================================================================

TEMPLATE_TEST_CASE("line_line_parallel", "[core][closest_metric_point_pair]",
    float, double)
{
    using real_t = TestType;

    auto line1 = tf::make_line_like(
        tf::make_point(real_t(0), real_t(0)),
        tf::make_vector(real_t(1), real_t(0))
    );
    auto line2 = tf::make_line_like(
        tf::make_point(real_t(0), real_t(5)),
        tf::make_vector(real_t(1), real_t(0))
    );

    auto [dist2, c0, c1] = tf::closest_metric_point_pair(line1, line2);
    REQUIRE(approx_equal(dist2, real_t(25)));
    REQUIRE(approx_equal(c0[0], c1[0]));  // x-coordinates should match
    REQUIRE(approx_equal(tf::distance2(line1, line2), dist2));
}

TEMPLATE_TEST_CASE("line_line_skew_3d", "[core][closest_metric_point_pair]",
    float, double)
{
    using real_t = TestType;

    auto line1 = tf::make_line_like(
        tf::make_point(real_t(0), real_t(0), real_t(0)),
        tf::make_vector(real_t(1), real_t(0), real_t(0))
    );
    auto line2 = tf::make_line_like(
        tf::make_point(real_t(0), real_t(0), real_t(4)),
        tf::make_vector(real_t(0), real_t(1), real_t(0))
    );

    auto [dist2, c0, c1] = tf::closest_metric_point_pair(line1, line2);
    REQUIRE(approx_equal(dist2, real_t(16)));
    REQUIRE(approx_equal(c0[2], real_t(0)));
    REQUIRE(approx_equal(c1[2], real_t(4)));
    REQUIRE(approx_equal(tf::distance2(line1, line2), dist2));
}

// =============================================================================
// Plane-Plane tests
// =============================================================================

TEMPLATE_TEST_CASE("plane_plane_parallel", "[core][closest_metric_point_pair]",
    float, double)
{
    using real_t = TestType;

    auto plane1 = tf::make_plane(
        tf::make_unit_vector(real_t(0), real_t(0), real_t(1)),
        real_t(0)
    );
    auto plane2 = tf::make_plane(
        tf::make_unit_vector(real_t(0), real_t(0), real_t(1)),
        real_t(-7)
    );

    auto [dist2, c0, c1] = tf::closest_metric_point_pair(plane1, plane2);
    REQUIRE(approx_equal(dist2, real_t(49)));
    REQUIRE(approx_equal(c0[2], real_t(0)));
    REQUIRE(approx_equal(c1[2], real_t(7)));
    REQUIRE(approx_equal(tf::distance2(plane1, plane2), dist2));
}

TEMPLATE_TEST_CASE("plane_plane_intersecting", "[core][closest_metric_point_pair]",
    float, double)
{
    using real_t = TestType;

    auto plane1 = tf::make_plane(
        tf::make_unit_vector(real_t(0), real_t(0), real_t(1)),
        real_t(0)
    );
    auto plane2 = tf::make_plane(
        tf::make_unit_vector(real_t(1), real_t(0), real_t(0)),
        real_t(0)
    );

    auto [dist2, c0, c1] = tf::closest_metric_point_pair(plane1, plane2);
    REQUIRE(approx_zero(dist2, real_t(1e-5)));
    REQUIRE(approx_zero(tf::distance2(plane1, plane2), real_t(1e-5)));
}

// =============================================================================
// Segment-Ray tests
// =============================================================================

TEMPLATE_TEST_CASE("segment_ray_separated", "[core][closest_metric_point_pair]",
    float, double)
{
    using real_t = TestType;

    SECTION("2D ray pointing away") {
        auto seg = tf::make_segment_between_points(
            tf::make_point(real_t(0), real_t(0)),
            tf::make_point(real_t(2), real_t(0))
        );
        auto ray = tf::make_ray(
            tf::make_point(real_t(5), real_t(0)),
            tf::make_vector(real_t(1), real_t(0))
        );

        auto [dist2, c0, c1] = tf::closest_metric_point_pair(seg, ray);
        REQUIRE(approx_equal(dist2, real_t(9)));
        REQUIRE(approx_equal(c0[0], real_t(2)));
        REQUIRE(approx_equal(c1[0], real_t(5)));
    }

    SECTION("3D parallel separated") {
        auto seg = tf::make_segment_between_points(
            tf::make_point(real_t(0), real_t(0), real_t(0)),
            tf::make_point(real_t(4), real_t(0), real_t(0))
        );
        auto ray = tf::make_ray(
            tf::make_point(real_t(0), real_t(3), real_t(0)),
            tf::make_vector(real_t(1), real_t(0), real_t(0))
        );

        auto [dist2, c0, c1] = tf::closest_metric_point_pair(seg, ray);
        REQUIRE(approx_equal(dist2, real_t(9)));
        REQUIRE(approx_equal(tf::distance2(seg, ray), dist2));
    }
}

TEMPLATE_TEST_CASE("segment_ray_intersecting", "[core][closest_metric_point_pair]",
    float, double)
{
    using real_t = TestType;

    SECTION("2D perpendicular intersection") {
        auto seg = tf::make_segment_between_points(
            tf::make_point(real_t(0), real_t(0)),
            tf::make_point(real_t(4), real_t(0))
        );
        auto ray = tf::make_ray(
            tf::make_point(real_t(2), real_t(-2)),
            tf::make_vector(real_t(0), real_t(1))
        );

        auto [dist2, c0, c1] = tf::closest_metric_point_pair(seg, ray);
        REQUIRE(approx_zero(dist2, real_t(1e-5)));
    }

    SECTION("3D skew but close") {
        auto seg = tf::make_segment_between_points(
            tf::make_point(real_t(0), real_t(0), real_t(0)),
            tf::make_point(real_t(4), real_t(0), real_t(0))
        );
        auto ray = tf::make_ray(
            tf::make_point(real_t(2), real_t(0), real_t(-1)),
            tf::make_vector(real_t(0), real_t(0), real_t(1))
        );

        auto [dist2, c0, c1] = tf::closest_metric_point_pair(seg, ray);
        REQUIRE(approx_zero(dist2, real_t(1e-5)));
    }
}

TEMPLATE_TEST_CASE("ray_segment_swap_symmetry", "[core][closest_metric_point_pair]",
    float, double)
{
    using real_t = TestType;

    auto seg = tf::make_segment_between_points(
        tf::make_point(real_t(0), real_t(0)),
        tf::make_point(real_t(4), real_t(0))
    );
    auto ray = tf::make_ray(
        tf::make_point(real_t(2), real_t(5)),
        tf::make_vector(real_t(0), real_t(1))
    );

    auto [dist2_a, c0_a, c1_a] = tf::closest_metric_point_pair(seg, ray);
    auto [dist2_b, c0_b, c1_b] = tf::closest_metric_point_pair(ray, seg);

    REQUIRE(approx_equal(dist2_a, dist2_b));
    REQUIRE(points_close(c0_a, c1_b, real_t(1e-5)));
    REQUIRE(points_close(c1_a, c0_b, real_t(1e-5)));
}

// =============================================================================
// Segment-Line tests
// =============================================================================

TEMPLATE_TEST_CASE("segment_line_perpendicular", "[core][closest_metric_point_pair]",
    float, double)
{
    using real_t = TestType;

    SECTION("2D perpendicular intersection") {
        auto seg = tf::make_segment_between_points(
            tf::make_point(real_t(0), real_t(0)),
            tf::make_point(real_t(4), real_t(0))
        );
        auto line = tf::make_line_like(
            tf::make_point(real_t(2), real_t(0)),
            tf::make_vector(real_t(0), real_t(1))
        );

        auto [dist2, c0, c1] = tf::closest_metric_point_pair(seg, line);
        REQUIRE(approx_zero(dist2, real_t(1e-5)));
    }

    SECTION("3D perpendicular separated") {
        auto seg = tf::make_segment_between_points(
            tf::make_point(real_t(0), real_t(0), real_t(0)),
            tf::make_point(real_t(4), real_t(0), real_t(0))
        );
        auto line = tf::make_line_like(
            tf::make_point(real_t(2), real_t(0), real_t(5)),
            tf::make_vector(real_t(0), real_t(1), real_t(0))
        );

        auto [dist2, c0, c1] = tf::closest_metric_point_pair(seg, line);
        REQUIRE(approx_equal(dist2, real_t(25)));
        REQUIRE(approx_equal(c0[0], real_t(2)));
        REQUIRE(approx_equal(c0[2], real_t(0)));
        REQUIRE(approx_equal(c1[2], real_t(5)));
    }
}

TEMPLATE_TEST_CASE("segment_line_parallel", "[core][closest_metric_point_pair]",
    float, double)
{
    using real_t = TestType;

    SECTION("2D parallel") {
        auto seg = tf::make_segment_between_points(
            tf::make_point(real_t(0), real_t(0)),
            tf::make_point(real_t(4), real_t(0))
        );
        auto line = tf::make_line_like(
            tf::make_point(real_t(0), real_t(3)),
            tf::make_vector(real_t(1), real_t(0))
        );

        auto [dist2, c0, c1] = tf::closest_metric_point_pair(seg, line);
        REQUIRE(approx_equal(dist2, real_t(9)));
        REQUIRE(approx_equal(c0[1], real_t(0)));
        REQUIRE(approx_equal(c1[1], real_t(3)));
    }

    SECTION("3D parallel offset") {
        auto seg = tf::make_segment_between_points(
            tf::make_point(real_t(0), real_t(0), real_t(0)),
            tf::make_point(real_t(4), real_t(0), real_t(0))
        );
        auto line = tf::make_line_like(
            tf::make_point(real_t(0), real_t(4), real_t(0)),
            tf::make_vector(real_t(1), real_t(0), real_t(0))
        );

        auto [dist2, c0, c1] = tf::closest_metric_point_pair(seg, line);
        REQUIRE(approx_equal(dist2, real_t(16)));
        REQUIRE(approx_equal(tf::distance2(seg, line), dist2));
    }
}

TEMPLATE_TEST_CASE("line_segment_swap_symmetry", "[core][closest_metric_point_pair]",
    float, double)
{
    using real_t = TestType;

    auto seg = tf::make_segment_between_points(
        tf::make_point(real_t(0), real_t(0)),
        tf::make_point(real_t(4), real_t(0))
    );
    auto line = tf::make_line_like(
        tf::make_point(real_t(10), real_t(6)),
        tf::make_vector(real_t(1), real_t(0))
    );

    auto [dist2_a, c0_a, c1_a] = tf::closest_metric_point_pair(seg, line);
    auto [dist2_b, c0_b, c1_b] = tf::closest_metric_point_pair(line, seg);

    REQUIRE(approx_equal(dist2_a, dist2_b));
    REQUIRE(points_close(c0_a, c1_b, real_t(1e-5)));
    REQUIRE(points_close(c1_a, c0_b, real_t(1e-5)));
}

// =============================================================================
// Ray-Line tests
// =============================================================================

TEMPLATE_TEST_CASE("ray_line_perpendicular", "[core][closest_metric_point_pair]",
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

        auto [dist2, c0, c1] = tf::closest_metric_point_pair(ray, line);
        REQUIRE(approx_zero(dist2, real_t(1e-5)));
    }

    SECTION("3D skew ray and line") {
        auto ray = tf::make_ray(
            tf::make_point(real_t(0), real_t(0), real_t(0)),
            tf::make_vector(real_t(1), real_t(0), real_t(0))
        );
        auto line = tf::make_line_like(
            tf::make_point(real_t(5), real_t(0), real_t(4)),
            tf::make_vector(real_t(0), real_t(1), real_t(0))
        );

        auto [dist2, c0, c1] = tf::closest_metric_point_pair(ray, line);
        REQUIRE(approx_equal(dist2, real_t(16)));
        REQUIRE(approx_equal(c0[0], real_t(5)));
        REQUIRE(approx_equal(c0[2], real_t(0)));
        REQUIRE(approx_equal(c1[2], real_t(4)));
    }
}

TEMPLATE_TEST_CASE("ray_line_parallel", "[core][closest_metric_point_pair]",
    float, double)
{
    using real_t = TestType;

    SECTION("2D parallel") {
        auto ray = tf::make_ray(
            tf::make_point(real_t(0), real_t(0)),
            tf::make_vector(real_t(1), real_t(0))
        );
        auto line = tf::make_line_like(
            tf::make_point(real_t(0), real_t(7)),
            tf::make_vector(real_t(1), real_t(0))
        );

        auto [dist2, c0, c1] = tf::closest_metric_point_pair(ray, line);
        REQUIRE(approx_equal(dist2, real_t(49)));
        REQUIRE(approx_equal(tf::distance2(ray, line), dist2));
    }

    SECTION("3D parallel offset") {
        auto ray = tf::make_ray(
            tf::make_point(real_t(0), real_t(0), real_t(0)),
            tf::make_vector(real_t(1), real_t(0), real_t(0))
        );
        auto line = tf::make_line_like(
            tf::make_point(real_t(0), real_t(3), real_t(4)),
            tf::make_vector(real_t(1), real_t(0), real_t(0))
        );

        auto [dist2, c0, c1] = tf::closest_metric_point_pair(ray, line);
        REQUIRE(approx_equal(dist2, real_t(25)));  // 3^2 + 4^2 = 25
        REQUIRE(approx_equal(tf::distance2(ray, line), dist2));
    }
}

TEMPLATE_TEST_CASE("line_ray_swap_symmetry", "[core][closest_metric_point_pair]",
    float, double)
{
    using real_t = TestType;

    auto ray = tf::make_ray(
        tf::make_point(real_t(0), real_t(0)),
        tf::make_vector(real_t(1), real_t(0))
    );
    auto line = tf::make_line_like(
        tf::make_point(real_t(3), real_t(8)),
        tf::make_vector(real_t(0), real_t(1))
    );

    auto [dist2_a, c0_a, c1_a] = tf::closest_metric_point_pair(ray, line);
    auto [dist2_b, c0_b, c1_b] = tf::closest_metric_point_pair(line, ray);

    REQUIRE(approx_equal(dist2_a, dist2_b));
    REQUIRE(points_close(c0_a, c1_b, real_t(1e-5)));
    REQUIRE(points_close(c1_a, c0_b, real_t(1e-5)));
}

// =============================================================================
// Polygon-Plane tests
// =============================================================================

TEMPLATE_TEST_CASE("polygon_plane_parallel", "[core][closest_metric_point_pair]",
    float, double)
{
    using real_t = TestType;

    std::array<tf::point<real_t, 3>, 3> triangle_pts = {{
        tf::make_point(real_t(0), real_t(0), real_t(3)),
        tf::make_point(real_t(1), real_t(0), real_t(3)),
        tf::make_point(real_t(0.5), real_t(1), real_t(3))
    }};
    auto poly = tf::make_polygon(triangle_pts);

    auto plane = tf::make_plane(
        tf::make_unit_vector(real_t(0), real_t(0), real_t(1)),
        real_t(0)
    );

    auto [dist2, c0, c1] = tf::closest_metric_point_pair(poly, plane);
    REQUIRE(approx_equal(dist2, real_t(9)));
    REQUIRE(approx_equal(c0[2], real_t(3)));
    REQUIRE(approx_equal(c1[2], real_t(0)));
}

TEMPLATE_TEST_CASE("polygon_plane_intersecting", "[core][closest_metric_point_pair]",
    float, double)
{
    using real_t = TestType;

    std::array<tf::point<real_t, 3>, 3> triangle_pts = {{
        tf::make_point(real_t(0), real_t(0), real_t(-1)),
        tf::make_point(real_t(1), real_t(0), real_t(1)),
        tf::make_point(real_t(0.5), real_t(1), real_t(0))
    }};
    auto poly = tf::make_polygon(triangle_pts);

    auto plane = tf::make_plane(
        tf::make_unit_vector(real_t(0), real_t(0), real_t(1)),
        real_t(0)
    );

    auto [dist2, c0, c1] = tf::closest_metric_point_pair(poly, plane);
    REQUIRE(approx_zero(dist2, real_t(1e-5)));
}

TEMPLATE_TEST_CASE("polygon_plane_above", "[core][closest_metric_point_pair]",
    float, double)
{
    using real_t = TestType;

    // Quad polygon in xy-plane at z=6
    std::array<tf::point<real_t, 3>, 4> quad_pts = {{
        tf::make_point(real_t(0), real_t(0), real_t(6)),
        tf::make_point(real_t(2), real_t(0), real_t(6)),
        tf::make_point(real_t(2), real_t(2), real_t(6)),
        tf::make_point(real_t(0), real_t(2), real_t(6))
    }};
    auto poly = tf::make_polygon(quad_pts);

    auto plane = tf::make_plane(
        tf::make_unit_vector(real_t(0), real_t(0), real_t(1)),
        real_t(0)
    );

    auto [dist2, c0, c1] = tf::closest_metric_point_pair(poly, plane);
    REQUIRE(approx_equal(dist2, real_t(36)));
    REQUIRE(approx_equal(c0[2], real_t(6)));
    REQUIRE(approx_equal(c1[2], real_t(0)));
}

TEMPLATE_TEST_CASE("plane_polygon_swap_symmetry", "[core][closest_metric_point_pair]",
    float, double)
{
    using real_t = TestType;

    std::array<tf::point<real_t, 3>, 3> triangle_pts = {{
        tf::make_point(real_t(0), real_t(0), real_t(4)),
        tf::make_point(real_t(1), real_t(0), real_t(4)),
        tf::make_point(real_t(0.5), real_t(1), real_t(4))
    }};
    auto poly = tf::make_polygon(triangle_pts);

    auto plane = tf::make_plane(
        tf::make_unit_vector(real_t(0), real_t(0), real_t(1)),
        real_t(0)
    );

    auto [dist2_a, c0_a, c1_a] = tf::closest_metric_point_pair(poly, plane);
    auto [dist2_b, c0_b, c1_b] = tf::closest_metric_point_pair(plane, poly);

    REQUIRE(approx_equal(dist2_a, dist2_b));
    REQUIRE(points_close(c0_a, c1_b, real_t(1e-5)));
    REQUIRE(points_close(c1_a, c0_b, real_t(1e-5)));
}

// =============================================================================
// Swap symmetry tests
// =============================================================================

TEMPLATE_TEST_CASE("swap_symmetry_point_segment", "[core][closest_metric_point_pair]",
    float, double)
{
    using real_t = TestType;

    auto seg = tf::make_segment_between_points(
        tf::make_point(real_t(0), real_t(0), real_t(0)),
        tf::make_point(real_t(4), real_t(0), real_t(0))
    );
    auto pt = tf::make_point(real_t(2), real_t(3), real_t(0));

    auto [dist2_a, c0_a, c1_a] = tf::closest_metric_point_pair(pt, seg);
    auto [dist2_b, c0_b, c1_b] = tf::closest_metric_point_pair(seg, pt);

    REQUIRE(approx_equal(dist2_a, dist2_b));
    REQUIRE(points_close(c0_a, c1_b, real_t(1e-5)));
    REQUIRE(points_close(c1_a, c0_b, real_t(1e-5)));
}

TEMPLATE_TEST_CASE("swap_symmetry_segment_polygon", "[core][closest_metric_point_pair]",
    float, double)
{
    using real_t = TestType;

    std::array<tf::point<real_t, 3>, 4> square_pts = {{
        tf::make_point(real_t(0), real_t(0), real_t(0)),
        tf::make_point(real_t(1), real_t(0), real_t(0)),
        tf::make_point(real_t(1), real_t(1), real_t(0)),
        tf::make_point(real_t(0), real_t(1), real_t(0))
    }};
    auto poly = tf::make_polygon(square_pts);

    auto seg = tf::make_segment_between_points(
        tf::make_point(real_t(3), real_t(0.5), real_t(0)),
        tf::make_point(real_t(5), real_t(0.5), real_t(0))
    );

    auto [dist2_a, c0_a, c1_a] = tf::closest_metric_point_pair(seg, poly);
    auto [dist2_b, c0_b, c1_b] = tf::closest_metric_point_pair(poly, seg);

    REQUIRE(approx_equal(dist2_a, dist2_b));
    REQUIRE(points_close(c0_a, c1_b, real_t(1e-5)));
    REQUIRE(points_close(c1_a, c0_b, real_t(1e-5)));
}

// =============================================================================
// AABB-Point tests
// =============================================================================

TEMPLATE_TEST_CASE("aabb_point_inside", "[core][closest_metric_point_pair]",
    float, double)
{
    using real_t = TestType;

    auto aabb = tf::make_aabb(
        tf::make_point(real_t(0), real_t(0), real_t(0)),
        tf::make_point(real_t(2), real_t(2), real_t(2))
    );
    auto pt = tf::make_point(real_t(1), real_t(1), real_t(1));

    auto [dist2, c0, c1] = tf::closest_metric_point_pair(aabb, pt);
    REQUIRE(approx_zero(dist2, real_t(1e-5)));
    REQUIRE(points_close(c0, pt, real_t(1e-5)));
    REQUIRE(points_close(c1, pt, real_t(1e-5)));
}

TEMPLATE_TEST_CASE("aabb_point_outside_one_axis", "[core][closest_metric_point_pair]",
    float, double)
{
    using real_t = TestType;

    auto aabb = tf::make_aabb(
        tf::make_point(real_t(0), real_t(0), real_t(0)),
        tf::make_point(real_t(2), real_t(2), real_t(2))
    );

    SECTION("beyond +x") {
        auto pt = tf::make_point(real_t(5), real_t(1), real_t(1));
        auto [dist2, c0, c1] = tf::closest_metric_point_pair(aabb, pt);
        REQUIRE(approx_equal(dist2, real_t(9)));
        REQUIRE(approx_equal(c0[0], real_t(2)));
        REQUIRE(approx_equal(c0[1], real_t(1)));
        REQUIRE(approx_equal(c0[2], real_t(1)));
        REQUIRE(points_close(c1, pt, real_t(1e-5)));
    }

    SECTION("below -y") {
        auto pt = tf::make_point(real_t(1), real_t(-3), real_t(1));
        auto [dist2, c0, c1] = tf::closest_metric_point_pair(aabb, pt);
        REQUIRE(approx_equal(dist2, real_t(9)));
        REQUIRE(approx_equal(c0[0], real_t(1)));
        REQUIRE(approx_equal(c0[1], real_t(0)));
        REQUIRE(approx_equal(c0[2], real_t(1)));
    }

    SECTION("beyond +z") {
        auto pt = tf::make_point(real_t(1), real_t(1), real_t(6));
        auto [dist2, c0, c1] = tf::closest_metric_point_pair(aabb, pt);
        REQUIRE(approx_equal(dist2, real_t(16)));
        REQUIRE(approx_equal(c0[2], real_t(2)));
    }
}

TEMPLATE_TEST_CASE("aabb_point_outside_corner", "[core][closest_metric_point_pair]",
    float, double)
{
    using real_t = TestType;

    auto aabb = tf::make_aabb(
        tf::make_point(real_t(0), real_t(0), real_t(0)),
        tf::make_point(real_t(1), real_t(1), real_t(1))
    );
    auto pt = tf::make_point(real_t(3), real_t(3), real_t(3));

    auto [dist2, c0, c1] = tf::closest_metric_point_pair(aabb, pt);
    // Corner (1,1,1) to (3,3,3): dist2 = 4+4+4 = 12
    REQUIRE(approx_equal(dist2, real_t(12)));
    REQUIRE(approx_equal(c0[0], real_t(1)));
    REQUIRE(approx_equal(c0[1], real_t(1)));
    REQUIRE(approx_equal(c0[2], real_t(1)));
    REQUIRE(approx_equal(tf::distance2(aabb, pt), dist2));
}

TEMPLATE_TEST_CASE("aabb_point_on_face", "[core][closest_metric_point_pair]",
    float, double)
{
    using real_t = TestType;

    auto aabb = tf::make_aabb(
        tf::make_point(real_t(0), real_t(0), real_t(0)),
        tf::make_point(real_t(2), real_t(2), real_t(2))
    );
    auto pt = tf::make_point(real_t(2), real_t(1), real_t(1));

    auto [dist2, c0, c1] = tf::closest_metric_point_pair(aabb, pt);
    REQUIRE(approx_zero(dist2, real_t(1e-5)));
    REQUIRE(points_close(c0, pt, real_t(1e-5)));
}

TEMPLATE_TEST_CASE("aabb_point_on_edge", "[core][closest_metric_point_pair]",
    float, double)
{
    using real_t = TestType;

    auto aabb = tf::make_aabb(
        tf::make_point(real_t(0), real_t(0), real_t(0)),
        tf::make_point(real_t(2), real_t(2), real_t(2))
    );
    // Point on the edge where x=2, z=2, y varies
    auto pt = tf::make_point(real_t(2), real_t(1), real_t(2));

    auto [dist2, c0, c1] = tf::closest_metric_point_pair(aabb, pt);
    REQUIRE(approx_zero(dist2, real_t(1e-5)));
    REQUIRE(points_close(c0, pt, real_t(1e-5)));
}

TEMPLATE_TEST_CASE("aabb_point_negative_quadrant", "[core][closest_metric_point_pair]",
    float, double)
{
    using real_t = TestType;

    auto aabb = tf::make_aabb(
        tf::make_point(real_t(-1), real_t(-1), real_t(-1)),
        tf::make_point(real_t(1), real_t(1), real_t(1))
    );
    auto pt = tf::make_point(real_t(-4), real_t(-4), real_t(-4));

    auto [dist2, c0, c1] = tf::closest_metric_point_pair(aabb, pt);
    // Corner (-1,-1,-1) to (-4,-4,-4): dist2 = 9+9+9 = 27
    REQUIRE(approx_equal(dist2, real_t(27)));
    REQUIRE(approx_equal(c0[0], real_t(-1)));
    REQUIRE(approx_equal(c0[1], real_t(-1)));
    REQUIRE(approx_equal(c0[2], real_t(-1)));
}

// =============================================================================
// AABB-Segment tests
// =============================================================================

TEMPLATE_TEST_CASE("aabb_segment_through", "[core][closest_metric_point_pair]",
    float, double)
{
    using real_t = TestType;

    auto aabb = tf::make_aabb(
        tf::make_point(real_t(0), real_t(0), real_t(0)),
        tf::make_point(real_t(2), real_t(2), real_t(2))
    );
    // Segment passing through the box
    auto seg = tf::make_segment_between_points(
        tf::make_point(real_t(-1), real_t(1), real_t(1)),
        tf::make_point(real_t(3), real_t(1), real_t(1))
    );

    auto [dist2, c0, c1] = tf::closest_metric_point_pair(aabb, seg);
    REQUIRE(approx_zero(dist2, real_t(1e-5)));
}

TEMPLATE_TEST_CASE("aabb_segment_below", "[core][closest_metric_point_pair]",
    float, double)
{
    using real_t = TestType;

    auto aabb = tf::make_aabb(
        tf::make_point(real_t(0), real_t(0), real_t(0)),
        tf::make_point(real_t(2), real_t(2), real_t(2))
    );
    // Segment entirely below the box on z-axis
    auto seg = tf::make_segment_between_points(
        tf::make_point(real_t(0), real_t(1), real_t(-3)),
        tf::make_point(real_t(2), real_t(1), real_t(-3))
    );

    auto [dist2, c0, c1] = tf::closest_metric_point_pair(aabb, seg);
    REQUIRE(approx_equal(dist2, real_t(9)));
    REQUIRE(approx_equal(c0[2], real_t(0)));
    REQUIRE(approx_equal(c1[2], real_t(-3)));
}

TEMPLATE_TEST_CASE("aabb_segment_parallel_offset", "[core][closest_metric_point_pair]",
    float, double)
{
    using real_t = TestType;

    auto aabb = tf::make_aabb(
        tf::make_point(real_t(0), real_t(0), real_t(0)),
        tf::make_point(real_t(4), real_t(4), real_t(4))
    );
    // Segment parallel to x-axis, offset by 2 in y, within x range
    auto seg = tf::make_segment_between_points(
        tf::make_point(real_t(1), real_t(6), real_t(2)),
        tf::make_point(real_t(3), real_t(6), real_t(2))
    );

    auto [dist2, c0, c1] = tf::closest_metric_point_pair(aabb, seg);
    // Perpendicular distance in y only: (6-4)^2 = 4
    REQUIRE(approx_equal(dist2, real_t(4)));
    REQUIRE(approx_equal(c0[1], real_t(4)));
    REQUIRE(approx_equal(c1[1], real_t(6)));
    REQUIRE(approx_equal(tf::distance2(aabb, seg), dist2));
}

TEMPLATE_TEST_CASE("aabb_segment_endpoint_nearest", "[core][closest_metric_point_pair]",
    float, double)
{
    using real_t = TestType;

    auto aabb = tf::make_aabb(
        tf::make_point(real_t(0), real_t(0), real_t(0)),
        tf::make_point(real_t(2), real_t(2), real_t(2))
    );
    // Segment pointing away from box, closest at t=0 endpoint
    auto seg = tf::make_segment_between_points(
        tf::make_point(real_t(3), real_t(1), real_t(1)),
        tf::make_point(real_t(10), real_t(1), real_t(1))
    );

    auto [dist2, c0, c1] = tf::closest_metric_point_pair(aabb, seg);
    // Closest: aabb face at x=2 to segment start at x=3 -> dist2 = 1
    REQUIRE(approx_equal(dist2, real_t(1)));
    REQUIRE(approx_equal(c0[0], real_t(2)));
    REQUIRE(approx_equal(c1[0], real_t(3)));
}

TEMPLATE_TEST_CASE("aabb_segment_diagonal_near_corner", "[core][closest_metric_point_pair]",
    float, double)
{
    using real_t = TestType;

    auto aabb = tf::make_aabb(
        tf::make_point(real_t(0), real_t(0), real_t(0)),
        tf::make_point(real_t(1), real_t(1), real_t(1))
    );
    // Segment diagonal in space near corner (2,2,2)
    auto seg = tf::make_segment_between_points(
        tf::make_point(real_t(2), real_t(2), real_t(2)),
        tf::make_point(real_t(4), real_t(4), real_t(4))
    );

    auto [dist2, c0, c1] = tf::closest_metric_point_pair(aabb, seg);
    // Closest: corner (1,1,1) to segment start (2,2,2), dist2 = 3
    REQUIRE(approx_equal(dist2, real_t(3)));
    REQUIRE(approx_equal(c0[0], real_t(1)));
    REQUIRE(approx_equal(c0[1], real_t(1)));
    REQUIRE(approx_equal(c0[2], real_t(1)));
    REQUIRE(approx_equal(c1[0], real_t(2)));
}

TEMPLATE_TEST_CASE("aabb_segment_inside_box", "[core][closest_metric_point_pair]",
    float, double)
{
    using real_t = TestType;

    auto aabb = tf::make_aabb(
        tf::make_point(real_t(0), real_t(0), real_t(0)),
        tf::make_point(real_t(10), real_t(10), real_t(10))
    );
    // Segment entirely inside the box
    auto seg = tf::make_segment_between_points(
        tf::make_point(real_t(2), real_t(3), real_t(4)),
        tf::make_point(real_t(5), real_t(6), real_t(7))
    );

    auto [dist2, c0, c1] = tf::closest_metric_point_pair(aabb, seg);
    REQUIRE(approx_zero(dist2, real_t(1e-5)));
}

TEMPLATE_TEST_CASE("aabb_segment_diagonal_mid_segment", "[core][closest_metric_point_pair]",
    float, double)
{
    using real_t = TestType;

    auto aabb = tf::make_aabb(
        tf::make_point(real_t(0), real_t(0), real_t(0)),
        tf::make_point(real_t(1), real_t(1), real_t(1))
    );
    // Segment from (-2,4,0.5) to (4,-2,0.5), midpoint at (1,1,0.5)
    // which is on the AABB face. The closest point is at the interior
    // of the segment, not at an endpoint, exercising the quadratic minimum.
    auto seg = tf::make_segment_between_points(
        tf::make_point(real_t(-2), real_t(4), real_t(0.5)),
        tf::make_point(real_t(4), real_t(-2), real_t(0.5))
    );

    auto [dist2, c0, c1] = tf::closest_metric_point_pair(aabb, seg);
    // At t=0.5: seg midpoint = (1,1,0.5) which is on the AABB -> dist2 = 0
    REQUIRE(approx_zero(dist2, real_t(1e-5)));
}

TEMPLATE_TEST_CASE("aabb_segment_diagonal_near_face", "[core][closest_metric_point_pair]",
    float, double)
{
    using real_t = TestType;

    auto aabb = tf::make_aabb(
        tf::make_point(real_t(0), real_t(0), real_t(0)),
        tf::make_point(real_t(1), real_t(1), real_t(1))
    );
    // Segment from (-2,4,3) to (4,-2,3), midpoint at (1,1,3).
    // Closest AABB point to midpoint: (1,1,1), dist2 = 4.
    // At endpoints: (-2,4,3) -> clamp (0,1,1) dist2=4+9+4=17;
    //               (4,-2,3) -> clamp (1,0,1) dist2=9+4+4=17.
    // The interior minimum at t=0.5 gives the best result.
    auto seg = tf::make_segment_between_points(
        tf::make_point(real_t(-2), real_t(4), real_t(3)),
        tf::make_point(real_t(4), real_t(-2), real_t(3))
    );

    auto [dist2, c0, c1] = tf::closest_metric_point_pair(aabb, seg);
    REQUIRE(approx_equal(dist2, real_t(4)));
    REQUIRE(approx_equal(c0[0], real_t(1)));
    REQUIRE(approx_equal(c0[1], real_t(1)));
    REQUIRE(approx_equal(c0[2], real_t(1)));
    REQUIRE(approx_equal(c1[0], real_t(1)));
    REQUIRE(approx_equal(c1[1], real_t(1)));
    REQUIRE(approx_equal(c1[2], real_t(3)));
}

// =============================================================================
// AABB-Line tests
// =============================================================================

TEMPLATE_TEST_CASE("aabb_line_through", "[core][closest_metric_point_pair]",
    float, double)
{
    using real_t = TestType;

    auto aabb = tf::make_aabb(
        tf::make_point(real_t(0), real_t(0), real_t(0)),
        tf::make_point(real_t(2), real_t(2), real_t(2))
    );
    // Line through the box center along x
    auto line = tf::make_line_like(
        tf::make_point(real_t(1), real_t(1), real_t(1)),
        tf::make_vector(real_t(1), real_t(0), real_t(0))
    );

    auto [dist2, c0, c1] = tf::closest_metric_point_pair(aabb, line);
    REQUIRE(approx_zero(dist2, real_t(1e-5)));
}

TEMPLATE_TEST_CASE("aabb_line_parallel_offset", "[core][closest_metric_point_pair]",
    float, double)
{
    using real_t = TestType;

    auto aabb = tf::make_aabb(
        tf::make_point(real_t(0), real_t(0), real_t(0)),
        tf::make_point(real_t(2), real_t(2), real_t(2))
    );
    // Line parallel to x-axis at y=5, z=1
    auto line = tf::make_line_like(
        tf::make_point(real_t(0), real_t(5), real_t(1)),
        tf::make_vector(real_t(1), real_t(0), real_t(0))
    );

    auto [dist2, c0, c1] = tf::closest_metric_point_pair(aabb, line);
    // Perpendicular distance in y: (5-2)^2 = 9
    REQUIRE(approx_equal(dist2, real_t(9)));
    REQUIRE(approx_equal(c0[1], real_t(2)));
    REQUIRE(approx_equal(c1[1], real_t(5)));
    REQUIRE(approx_equal(tf::distance2(aabb, line), dist2));
}

TEMPLATE_TEST_CASE("aabb_line_skew", "[core][closest_metric_point_pair]",
    float, double)
{
    using real_t = TestType;

    auto aabb = tf::make_aabb(
        tf::make_point(real_t(0), real_t(0), real_t(0)),
        tf::make_point(real_t(2), real_t(2), real_t(2))
    );
    // Line along z at x=5, y=1 -> closest approach is face x=2, y=1
    auto line = tf::make_line_like(
        tf::make_point(real_t(5), real_t(1), real_t(0)),
        tf::make_vector(real_t(0), real_t(0), real_t(1))
    );

    auto [dist2, c0, c1] = tf::closest_metric_point_pair(aabb, line);
    // Distance in x only: (5-2)^2 = 9
    REQUIRE(approx_equal(dist2, real_t(9)));
    REQUIRE(approx_equal(c0[0], real_t(2)));
    REQUIRE(approx_equal(c0[1], real_t(1)));
    REQUIRE(approx_equal(c1[0], real_t(5)));
    REQUIRE(approx_equal(c1[1], real_t(1)));
}

TEMPLATE_TEST_CASE("aabb_line_diagonal_approach", "[core][closest_metric_point_pair]",
    float, double)
{
    using real_t = TestType;

    auto aabb = tf::make_aabb(
        tf::make_point(real_t(0), real_t(0), real_t(0)),
        tf::make_point(real_t(1), real_t(1), real_t(1))
    );
    // Line along (1,1,0) passing through (5,5,3) -> offset in z=3 from box
    // Closest approach: line at z=3 always outside on z by 2
    // Line sweeps x,y equally; closest when x,y in [0,1]
    // At t such that origin+t*dir has x=1,y=1 -> t=-4 from origin (5,5,3)
    // -> point on line (1,1,3), point on aabb (1,1,1), dist2=4
    auto line = tf::make_line_like(
        tf::make_point(real_t(5), real_t(5), real_t(3)),
        tf::make_vector(real_t(1), real_t(1), real_t(0))
    );

    auto [dist2, c0, c1] = tf::closest_metric_point_pair(aabb, line);
    REQUIRE(approx_equal(dist2, real_t(4)));
    REQUIRE(approx_equal(c0[2], real_t(1)));
    REQUIRE(approx_equal(c1[2], real_t(3)));
}

// =============================================================================
// AABB-Ray tests
// =============================================================================

TEMPLATE_TEST_CASE("aabb_ray_origin_inside", "[core][closest_metric_point_pair]",
    float, double)
{
    using real_t = TestType;

    auto aabb = tf::make_aabb(
        tf::make_point(real_t(0), real_t(0), real_t(0)),
        tf::make_point(real_t(4), real_t(4), real_t(4))
    );
    auto ray = tf::make_ray(
        tf::make_point(real_t(2), real_t(2), real_t(2)),
        tf::make_vector(real_t(1), real_t(0), real_t(0))
    );

    auto [dist2, c0, c1] = tf::closest_metric_point_pair(aabb, ray);
    REQUIRE(approx_zero(dist2, real_t(1e-5)));
}

TEMPLATE_TEST_CASE("aabb_ray_pointing_away", "[core][closest_metric_point_pair]",
    float, double)
{
    using real_t = TestType;

    auto aabb = tf::make_aabb(
        tf::make_point(real_t(0), real_t(0), real_t(0)),
        tf::make_point(real_t(2), real_t(2), real_t(2))
    );
    // Ray origin at (5,1,1) pointing away from box (+x)
    auto ray = tf::make_ray(
        tf::make_point(real_t(5), real_t(1), real_t(1)),
        tf::make_vector(real_t(1), real_t(0), real_t(0))
    );

    auto [dist2, c0, c1] = tf::closest_metric_point_pair(aabb, ray);
    // Closest is at t=0 (origin), so (5,1,1) to face at x=2 -> dist2 = 9
    REQUIRE(approx_equal(dist2, real_t(9)));
    REQUIRE(approx_equal(c0[0], real_t(2)));
    REQUIRE(approx_equal(c1[0], real_t(5)));
}

TEMPLATE_TEST_CASE("aabb_ray_pointing_toward", "[core][closest_metric_point_pair]",
    float, double)
{
    using real_t = TestType;

    auto aabb = tf::make_aabb(
        tf::make_point(real_t(0), real_t(0), real_t(0)),
        tf::make_point(real_t(2), real_t(2), real_t(2))
    );
    // Ray origin at (5,1,1) pointing toward box (-x)
    auto ray = tf::make_ray(
        tf::make_point(real_t(5), real_t(1), real_t(1)),
        tf::make_vector(real_t(-1), real_t(0), real_t(0))
    );

    auto [dist2, c0, c1] = tf::closest_metric_point_pair(aabb, ray);
    // Ray hits the box at t=3 -> point (2,1,1) which is on the box -> dist2 = 0
    REQUIRE(approx_zero(dist2, real_t(1e-5)));
}

TEMPLATE_TEST_CASE("aabb_ray_parallel_above", "[core][closest_metric_point_pair]",
    float, double)
{
    using real_t = TestType;

    auto aabb = tf::make_aabb(
        tf::make_point(real_t(0), real_t(0), real_t(0)),
        tf::make_point(real_t(4), real_t(4), real_t(4))
    );
    // Ray parallel to x, at y=7, z=2, origin at x=-5
    auto ray = tf::make_ray(
        tf::make_point(real_t(-5), real_t(7), real_t(2)),
        tf::make_vector(real_t(1), real_t(0), real_t(0))
    );

    auto [dist2, c0, c1] = tf::closest_metric_point_pair(aabb, ray);
    // Ray closest to box when ray x in [0,4]. At x=0, y gap = 7-4=3, dist2=9
    REQUIRE(approx_equal(dist2, real_t(9)));
    REQUIRE(approx_equal(c0[1], real_t(4)));
    REQUIRE(approx_equal(c1[1], real_t(7)));
    REQUIRE(approx_equal(tf::distance2(aabb, ray), dist2));
}

TEMPLATE_TEST_CASE("aabb_ray_behind_corner", "[core][closest_metric_point_pair]",
    float, double)
{
    using real_t = TestType;

    auto aabb = tf::make_aabb(
        tf::make_point(real_t(0), real_t(0), real_t(0)),
        tf::make_point(real_t(1), real_t(1), real_t(1))
    );
    // Ray at corner region pointing away diagonally
    auto ray = tf::make_ray(
        tf::make_point(real_t(3), real_t(3), real_t(3)),
        tf::make_vector(real_t(1), real_t(1), real_t(1))
    );

    auto [dist2, c0, c1] = tf::closest_metric_point_pair(aabb, ray);
    // Closest at t=0: corner (1,1,1) to (3,3,3), dist2 = 12
    REQUIRE(approx_equal(dist2, real_t(12)));
    REQUIRE(approx_equal(c0[0], real_t(1)));
    REQUIRE(approx_equal(c1[0], real_t(3)));
}

TEMPLATE_TEST_CASE("aabb_ray_diagonal_interior_min", "[core][closest_metric_point_pair]",
    float, double)
{
    using real_t = TestType;

    auto aabb = tf::make_aabb(
        tf::make_point(real_t(0), real_t(0), real_t(0)),
        tf::make_point(real_t(1), real_t(1), real_t(1))
    );
    // Ray origin at (-5,-5,3), direction (1,1,0).
    // D(t) = 4 for all t in [5,6] (flat minimum region), z gap is always 2.
    // At t=0: ray at (-5,-5,3), dist2 = 25+25+4 = 54.
    // Any t in [5,6] gives the global min dist2 = 4.
    auto ray = tf::make_ray(
        tf::make_point(real_t(-5), real_t(-5), real_t(3)),
        tf::make_vector(real_t(1), real_t(1), real_t(0))
    );

    auto [dist2, c0, c1] = tf::closest_metric_point_pair(aabb, ray);
    REQUIRE(approx_equal(dist2, real_t(4)));
    // z is invariant across the flat region
    REQUIRE(approx_equal(c0[2], real_t(1)));
    REQUIRE(approx_equal(c1[2], real_t(3)));
    // x,y can be anywhere in [0,1] — just verify consistency
    REQUIRE(approx_equal(tf::distance2(c0, c1), dist2, real_t(1e-4)));
}

// =============================================================================
// AABB-Plane tests
// =============================================================================

TEMPLATE_TEST_CASE("aabb_plane_intersects", "[core][closest_metric_point_pair]",
    float, double)
{
    using real_t = TestType;

    auto aabb = tf::make_aabb(
        tf::make_point(real_t(0), real_t(0), real_t(0)),
        tf::make_point(real_t(2), real_t(2), real_t(2))
    );
    // Plane z=1 cuts through the middle
    auto plane = tf::make_plane(
        tf::make_unit_vector(real_t(0), real_t(0), real_t(1)),
        real_t(-1)
    );

    auto [dist2, c0, c1] = tf::closest_metric_point_pair(aabb, plane);
    REQUIRE(approx_zero(dist2, real_t(1e-5)));
}

TEMPLATE_TEST_CASE("aabb_plane_above", "[core][closest_metric_point_pair]",
    float, double)
{
    using real_t = TestType;

    auto aabb = tf::make_aabb(
        tf::make_point(real_t(0), real_t(0), real_t(0)),
        tf::make_point(real_t(2), real_t(2), real_t(2))
    );
    // Plane z=5 above the box (normal pointing +z, d = -5)
    auto plane = tf::make_plane(
        tf::make_unit_vector(real_t(0), real_t(0), real_t(1)),
        real_t(-5)
    );

    auto [dist2, c0, c1] = tf::closest_metric_point_pair(aabb, plane);
    // Support vertex: AABB on negative side (d_center < 0), closest vertex
    // maximizes dot with normal -> (x,y,2). d = 0*x + 0*y + 1*2 + (-5) = -3
    // dist2 = 9
    REQUIRE(approx_equal(dist2, real_t(9)));
    REQUIRE(approx_equal(c0[2], real_t(2)));
    REQUIRE(approx_equal(c1[2], real_t(5)));
}

TEMPLATE_TEST_CASE("aabb_plane_below", "[core][closest_metric_point_pair]",
    float, double)
{
    using real_t = TestType;

    // This is the bug-catching case: AABB above the plane
    auto aabb = tf::make_aabb(
        tf::make_point(real_t(0), real_t(0), real_t(0)),
        tf::make_point(real_t(2), real_t(2), real_t(2))
    );
    // Plane z=-3 below the box (normal pointing +z, d = 3)
    auto plane = tf::make_plane(
        tf::make_unit_vector(real_t(0), real_t(0), real_t(1)),
        real_t(3)
    );

    auto [dist2, c0, c1] = tf::closest_metric_point_pair(aabb, plane);
    // AABB center at (1,1,1), d_center = 0+0+1+3 = 4 > 0 (positive side)
    // Support vertex minimizes dot -> (x,y,0). d = 0+0+0+3 = 3
    // dist2 = 9
    REQUIRE(approx_equal(dist2, real_t(9)));
    REQUIRE(approx_equal(c0[2], real_t(0)));
    REQUIRE(approx_equal(c1[2], real_t(-3)));
}

TEMPLATE_TEST_CASE("aabb_plane_45_degree", "[core][closest_metric_point_pair]",
    float, double)
{
    using real_t = TestType;

    auto aabb = tf::make_aabb(
        tf::make_point(real_t(0), real_t(0), real_t(0)),
        tf::make_point(real_t(2), real_t(2), real_t(2))
    );
    // Plane with normal (1,0,1)/sqrt(2), passing through (5,0,5)
    // normal . (5,0,5) = (5+5)/sqrt(2) = 10/sqrt(2)
    // d = -10/sqrt(2)
    real_t s = real_t(1) / std::sqrt(real_t(2));
    auto plane = tf::make_plane(
        tf::make_unit_vector(s, real_t(0), s),
        -real_t(10) * s
    );

    auto [dist2, c0, c1] = tf::closest_metric_point_pair(aabb, plane);
    // AABB center (1,1,1): d_center = s*1+0+s*1 - 10*s = s*(2-10) = -8*s < 0
    // Negative side -> maximize dot -> support = (2,y,2)
    // d = s*2 + s*2 - 10*s = s*(4-10) = -6*s
    // dist2 = (6*s)^2 = 36/2 = 18
    REQUIRE(approx_equal(dist2, real_t(18), real_t(1e-4)));
    REQUIRE(approx_equal(c0[0], real_t(2), real_t(1e-4)));
    REQUIRE(approx_equal(c0[2], real_t(2), real_t(1e-4)));
}

TEMPLATE_TEST_CASE("aabb_plane_negative_normal", "[core][closest_metric_point_pair]",
    float, double)
{
    using real_t = TestType;

    auto aabb = tf::make_aabb(
        tf::make_point(real_t(0), real_t(0), real_t(0)),
        tf::make_point(real_t(2), real_t(2), real_t(2))
    );
    // Plane z=5 but with normal pointing -z (d=5)
    auto plane = tf::make_plane(
        tf::make_unit_vector(real_t(0), real_t(0), real_t(-1)),
        real_t(5)
    );

    auto [dist2, c0, c1] = tf::closest_metric_point_pair(aabb, plane);
    // d_center = 0+0+(-1)*1+5 = 4 > 0 -> minimize dot (which means maximize z)
    // support = (x,y,2), d = -2+5 = 3, dist2 = 9
    REQUIRE(approx_equal(dist2, real_t(9)));
    REQUIRE(approx_equal(c0[2], real_t(2)));
}

TEMPLATE_TEST_CASE("aabb_plane_touching", "[core][closest_metric_point_pair]",
    float, double)
{
    using real_t = TestType;

    auto aabb = tf::make_aabb(
        tf::make_point(real_t(0), real_t(0), real_t(0)),
        tf::make_point(real_t(2), real_t(2), real_t(2))
    );
    // Plane exactly at z=2 touching the top face
    auto plane = tf::make_plane(
        tf::make_unit_vector(real_t(0), real_t(0), real_t(1)),
        real_t(-2)
    );

    auto [dist2, c0, c1] = tf::closest_metric_point_pair(aabb, plane);
    REQUIRE(approx_zero(dist2, real_t(1e-5)));
    REQUIRE(approx_equal(c0[2], real_t(2), real_t(1e-5)));
}

TEMPLATE_TEST_CASE("aabb_plane_cuts_through_off_center", "[core][closest_metric_point_pair]",
    float, double)
{
    using real_t = TestType;

    auto aabb = tf::make_aabb(
        tf::make_point(real_t(0), real_t(0), real_t(0)),
        tf::make_point(real_t(2), real_t(2), real_t(2))
    );
    // Plane z=0.5, center at z=1 is above the plane but plane still
    // cuts through the box. This is the case where d_center != 0 but
    // the support vertex crosses to the other side.
    auto plane = tf::make_plane(
        tf::make_unit_vector(real_t(0), real_t(0), real_t(1)),
        real_t(-0.5)
    );

    auto [dist2, c0, c1] = tf::closest_metric_point_pair(aabb, plane);
    REQUIRE(approx_zero(dist2, real_t(1e-5)));
}

TEMPLATE_TEST_CASE("aabb_plane_cuts_through_diagonal", "[core][closest_metric_point_pair]",
    float, double)
{
    using real_t = TestType;

    auto aabb = tf::make_aabb(
        tf::make_point(real_t(0), real_t(0), real_t(0)),
        tf::make_point(real_t(2), real_t(2), real_t(2))
    );
    // Diagonal plane through (1,1,1) with normal (1,1,1)/sqrt(3)
    real_t s = real_t(1) / std::sqrt(real_t(3));
    // dot(normal, (1,1,1)) = 3/sqrt(3) = sqrt(3), d = -sqrt(3)
    auto plane = tf::make_plane(
        tf::make_unit_vector(s, s, s),
        -std::sqrt(real_t(3))
    );

    auto [dist2, c0, c1] = tf::closest_metric_point_pair(aabb, plane);
    // Plane passes through center of the AABB -> intersects
    REQUIRE(approx_zero(dist2, real_t(1e-4)));
}

// =============================================================================
// AABB-AABB tests
// =============================================================================

TEMPLATE_TEST_CASE("aabb_aabb_overlapping", "[core][closest_metric_point_pair]",
    float, double)
{
    using real_t = TestType;

    auto a = tf::make_aabb(
        tf::make_point(real_t(0), real_t(0), real_t(0)),
        tf::make_point(real_t(3), real_t(3), real_t(3))
    );
    auto b = tf::make_aabb(
        tf::make_point(real_t(1), real_t(1), real_t(1)),
        tf::make_point(real_t(4), real_t(4), real_t(4))
    );

    auto [dist2, c0, c1] = tf::closest_metric_point_pair(a, b);
    REQUIRE(approx_zero(dist2, real_t(1e-5)));
}

TEMPLATE_TEST_CASE("aabb_aabb_separated_one_axis", "[core][closest_metric_point_pair]",
    float, double)
{
    using real_t = TestType;

    auto a = tf::make_aabb(
        tf::make_point(real_t(0), real_t(0), real_t(0)),
        tf::make_point(real_t(1), real_t(1), real_t(1))
    );
    auto b = tf::make_aabb(
        tf::make_point(real_t(3), real_t(0), real_t(0)),
        tf::make_point(real_t(4), real_t(1), real_t(1))
    );

    auto [dist2, c0, c1] = tf::closest_metric_point_pair(a, b);
    // Gap on x only: 3-1=2, dist2 = 4
    REQUIRE(approx_equal(dist2, real_t(4)));
    REQUIRE(approx_equal(c0[0], real_t(1)));
    REQUIRE(approx_equal(c1[0], real_t(3)));
    REQUIRE(approx_equal(tf::distance2(a, b), dist2));
}

TEMPLATE_TEST_CASE("aabb_aabb_separated_all_axes", "[core][closest_metric_point_pair]",
    float, double)
{
    using real_t = TestType;

    auto a = tf::make_aabb(
        tf::make_point(real_t(0), real_t(0), real_t(0)),
        tf::make_point(real_t(1), real_t(1), real_t(1))
    );
    auto b = tf::make_aabb(
        tf::make_point(real_t(3), real_t(3), real_t(3)),
        tf::make_point(real_t(4), real_t(4), real_t(4))
    );

    auto [dist2, c0, c1] = tf::closest_metric_point_pair(a, b);
    // Gaps: x=2, y=2, z=2, dist2 = 4+4+4 = 12
    REQUIRE(approx_equal(dist2, real_t(12)));
    REQUIRE(approx_equal(c0[0], real_t(1)));
    REQUIRE(approx_equal(c0[1], real_t(1)));
    REQUIRE(approx_equal(c0[2], real_t(1)));
    REQUIRE(approx_equal(c1[0], real_t(3)));
    REQUIRE(approx_equal(c1[1], real_t(3)));
    REQUIRE(approx_equal(c1[2], real_t(3)));
}

TEMPLATE_TEST_CASE("aabb_aabb_touching", "[core][closest_metric_point_pair]",
    float, double)
{
    using real_t = TestType;

    auto a = tf::make_aabb(
        tf::make_point(real_t(0), real_t(0), real_t(0)),
        tf::make_point(real_t(1), real_t(1), real_t(1))
    );
    auto b = tf::make_aabb(
        tf::make_point(real_t(1), real_t(0), real_t(0)),
        tf::make_point(real_t(2), real_t(1), real_t(1))
    );

    auto [dist2, c0, c1] = tf::closest_metric_point_pair(a, b);
    REQUIRE(approx_zero(dist2, real_t(1e-5)));
    REQUIRE(approx_equal(c0[0], real_t(1), real_t(1e-5)));
    REQUIRE(approx_equal(c1[0], real_t(1), real_t(1e-5)));
}

TEMPLATE_TEST_CASE("aabb_aabb_contained", "[core][closest_metric_point_pair]",
    float, double)
{
    using real_t = TestType;

    auto outer = tf::make_aabb(
        tf::make_point(real_t(0), real_t(0), real_t(0)),
        tf::make_point(real_t(10), real_t(10), real_t(10))
    );
    auto inner = tf::make_aabb(
        tf::make_point(real_t(2), real_t(3), real_t(4)),
        tf::make_point(real_t(5), real_t(6), real_t(7))
    );

    auto [dist2, c0, c1] = tf::closest_metric_point_pair(outer, inner);
    REQUIRE(approx_zero(dist2, real_t(1e-5)));
}

// =============================================================================
// AABB-Polygon tests
// =============================================================================

TEMPLATE_TEST_CASE("aabb_polygon_face_projection", "[core][closest_metric_point_pair]",
    float, double)
{
    using real_t = TestType;

    auto aabb = tf::make_aabb(
        tf::make_point(real_t(0), real_t(0), real_t(0)),
        tf::make_point(real_t(2), real_t(2), real_t(2))
    );
    // Triangle directly above the AABB, centered over it
    std::array<tf::point<real_t, 3>, 3> tri_pts = {{
        tf::make_point(real_t(0), real_t(0), real_t(5)),
        tf::make_point(real_t(2), real_t(0), real_t(5)),
        tf::make_point(real_t(1), real_t(2), real_t(5))
    }};
    auto poly = tf::make_polygon(tri_pts);

    auto [dist2, c0, c1] = tf::closest_metric_point_pair(aabb, poly);
    // Perpendicular distance in z: (5-2)^2 = 9
    REQUIRE(approx_equal(dist2, real_t(9)));
    REQUIRE(approx_equal(c0[2], real_t(2)));
    REQUIRE(approx_equal(c1[2], real_t(5)));
    REQUIRE(approx_equal(tf::distance2(aabb, poly), dist2));
}

TEMPLATE_TEST_CASE("aabb_polygon_edge_closest", "[core][closest_metric_point_pair]",
    float, double)
{
    using real_t = TestType;

    auto aabb = tf::make_aabb(
        tf::make_point(real_t(0), real_t(0), real_t(0)),
        tf::make_point(real_t(1), real_t(1), real_t(1))
    );
    // Triangle to the side with no face projection onto the box
    std::array<tf::point<real_t, 3>, 3> tri_pts = {{
        tf::make_point(real_t(3), real_t(0), real_t(0)),
        tf::make_point(real_t(5), real_t(0), real_t(0)),
        tf::make_point(real_t(4), real_t(0), real_t(2))
    }};
    auto poly = tf::make_polygon(tri_pts);

    auto [dist2, c0, c1] = tf::closest_metric_point_pair(aabb, poly);
    // Closest: aabb face x=1 to triangle edge at x=3
    // The closest point on triangle is (3,0,0), on aabb is (1,0,0), dist2=4
    REQUIRE(approx_equal(dist2, real_t(4)));
    REQUIRE(approx_equal(c0[0], real_t(1)));
    REQUIRE(approx_equal(c1[0], real_t(3)));
}

TEMPLATE_TEST_CASE("aabb_polygon_vertex_closest", "[core][closest_metric_point_pair]",
    float, double)
{
    using real_t = TestType;

    auto aabb = tf::make_aabb(
        tf::make_point(real_t(0), real_t(0), real_t(0)),
        tf::make_point(real_t(1), real_t(1), real_t(1))
    );
    // Triangle with nearest vertex at (3,3,3)
    std::array<tf::point<real_t, 3>, 3> tri_pts = {{
        tf::make_point(real_t(3), real_t(3), real_t(3)),
        tf::make_point(real_t(5), real_t(5), real_t(3)),
        tf::make_point(real_t(5), real_t(3), real_t(5))
    }};
    auto poly = tf::make_polygon(tri_pts);

    auto [dist2, c0, c1] = tf::closest_metric_point_pair(aabb, poly);
    // Corner (1,1,1) to vertex (3,3,3), dist2 = 4+4+4 = 12
    REQUIRE(approx_equal(dist2, real_t(12)));
    REQUIRE(approx_equal(c0[0], real_t(1)));
    REQUIRE(approx_equal(c0[1], real_t(1)));
    REQUIRE(approx_equal(c0[2], real_t(1)));
    REQUIRE(approx_equal(c1[0], real_t(3)));
}

TEMPLATE_TEST_CASE("aabb_polygon_intersecting", "[core][closest_metric_point_pair]",
    float, double)
{
    using real_t = TestType;

    auto aabb = tf::make_aabb(
        tf::make_point(real_t(0), real_t(0), real_t(0)),
        tf::make_point(real_t(4), real_t(4), real_t(4))
    );
    // Triangle cutting through the box
    std::array<tf::point<real_t, 3>, 3> tri_pts = {{
        tf::make_point(real_t(-1), real_t(2), real_t(2)),
        tf::make_point(real_t(5), real_t(2), real_t(2)),
        tf::make_point(real_t(2), real_t(2), real_t(-1))
    }};
    auto poly = tf::make_polygon(tri_pts);

    auto [dist2, c0, c1] = tf::closest_metric_point_pair(aabb, poly);
    REQUIRE(approx_zero(dist2, real_t(1e-5)));
}

TEMPLATE_TEST_CASE("aabb_polygon_plane_intersects_face_overlaps", "[core][closest_metric_point_pair]",
    float, double)
{
    using real_t = TestType;

    auto aabb = tf::make_aabb(
        tf::make_point(real_t(0), real_t(0), real_t(0)),
        tf::make_point(real_t(4), real_t(4), real_t(4))
    );
    // Triangle in z=1 plane, entirely inside AABB's xy footprint.
    // The polygon's plane cuts through the box AND the face overlaps
    // with the box's interior -> distance must be 0.
    std::array<tf::point<real_t, 3>, 3> tri_pts = {{
        tf::make_point(real_t(1), real_t(1), real_t(1)),
        tf::make_point(real_t(3), real_t(1), real_t(1)),
        tf::make_point(real_t(2), real_t(3), real_t(1))
    }};
    auto poly = tf::make_polygon(tri_pts);

    auto [dist2, c0, c1] = tf::closest_metric_point_pair(aabb, poly);
    REQUIRE(approx_zero(dist2, real_t(1e-5)));
}

TEMPLATE_TEST_CASE("aabb_polygon_plane_intersects_face_no_overlap", "[core][closest_metric_point_pair]",
    float, double)
{
    using real_t = TestType;

    auto aabb = tf::make_aabb(
        tf::make_point(real_t(0), real_t(0), real_t(0)),
        tf::make_point(real_t(1), real_t(1), real_t(1))
    );
    // Triangle in z=0.5 plane, but far away in x,y.
    // The polygon's plane cuts through the box, BUT the polygon face
    // does NOT overlap the box footprint. Distance should be > 0.
    std::array<tf::point<real_t, 3>, 3> tri_pts = {{
        tf::make_point(real_t(5), real_t(5), real_t(0.5)),
        tf::make_point(real_t(7), real_t(5), real_t(0.5)),
        tf::make_point(real_t(6), real_t(7), real_t(0.5))
    }};
    auto poly = tf::make_polygon(tri_pts);

    auto [dist2, c0, c1] = tf::closest_metric_point_pair(aabb, poly);
    // Nearest: AABB corner (1,1,0.5) to triangle vertex (5,5,0.5)
    // dist2 = 16+16+0 = 32
    REQUIRE(dist2 > real_t(0.1));
    REQUIRE(approx_equal(c0[2], real_t(0.5), real_t(1e-4)));
    REQUIRE(approx_equal(c1[2], real_t(0.5), real_t(1e-4)));
}

// =============================================================================
// AABB swap symmetry tests
// =============================================================================

TEMPLATE_TEST_CASE("swap_symmetry_aabb_point", "[core][closest_metric_point_pair]",
    float, double)
{
    using real_t = TestType;

    auto aabb = tf::make_aabb(
        tf::make_point(real_t(0), real_t(0), real_t(0)),
        tf::make_point(real_t(2), real_t(2), real_t(2))
    );
    auto pt = tf::make_point(real_t(5), real_t(3), real_t(-1));

    auto [dist2_a, c0_a, c1_a] = tf::closest_metric_point_pair(aabb, pt);
    auto [dist2_b, c0_b, c1_b] = tf::closest_metric_point_pair(pt, aabb);

    REQUIRE(approx_equal(dist2_a, dist2_b));
    REQUIRE(points_close(c0_a, c1_b, real_t(1e-5)));
    REQUIRE(points_close(c1_a, c0_b, real_t(1e-5)));
}

TEMPLATE_TEST_CASE("swap_symmetry_aabb_segment", "[core][closest_metric_point_pair]",
    float, double)
{
    using real_t = TestType;

    auto aabb = tf::make_aabb(
        tf::make_point(real_t(0), real_t(0), real_t(0)),
        tf::make_point(real_t(2), real_t(2), real_t(2))
    );
    auto seg = tf::make_segment_between_points(
        tf::make_point(real_t(5), real_t(1), real_t(1)),
        tf::make_point(real_t(8), real_t(3), real_t(-1))
    );

    auto [dist2_a, c0_a, c1_a] = tf::closest_metric_point_pair(aabb, seg);
    auto [dist2_b, c0_b, c1_b] = tf::closest_metric_point_pair(seg, aabb);

    REQUIRE(approx_equal(dist2_a, dist2_b));
    REQUIRE(points_close(c0_a, c1_b, real_t(1e-5)));
    REQUIRE(points_close(c1_a, c0_b, real_t(1e-5)));
}

TEMPLATE_TEST_CASE("swap_symmetry_aabb_line", "[core][closest_metric_point_pair]",
    float, double)
{
    using real_t = TestType;

    auto aabb = tf::make_aabb(
        tf::make_point(real_t(0), real_t(0), real_t(0)),
        tf::make_point(real_t(2), real_t(2), real_t(2))
    );
    auto line = tf::make_line_like(
        tf::make_point(real_t(5), real_t(5), real_t(1)),
        tf::make_vector(real_t(1), real_t(-1), real_t(0))
    );

    auto [dist2_a, c0_a, c1_a] = tf::closest_metric_point_pair(aabb, line);
    auto [dist2_b, c0_b, c1_b] = tf::closest_metric_point_pair(line, aabb);

    REQUIRE(approx_equal(dist2_a, dist2_b));
    REQUIRE(points_close(c0_a, c1_b, real_t(1e-5)));
    REQUIRE(points_close(c1_a, c0_b, real_t(1e-5)));
}

TEMPLATE_TEST_CASE("swap_symmetry_aabb_ray", "[core][closest_metric_point_pair]",
    float, double)
{
    using real_t = TestType;

    auto aabb = tf::make_aabb(
        tf::make_point(real_t(0), real_t(0), real_t(0)),
        tf::make_point(real_t(2), real_t(2), real_t(2))
    );
    auto ray = tf::make_ray(
        tf::make_point(real_t(5), real_t(5), real_t(5)),
        tf::make_vector(real_t(1), real_t(1), real_t(1))
    );

    auto [dist2_a, c0_a, c1_a] = tf::closest_metric_point_pair(aabb, ray);
    auto [dist2_b, c0_b, c1_b] = tf::closest_metric_point_pair(ray, aabb);

    REQUIRE(approx_equal(dist2_a, dist2_b));
    REQUIRE(points_close(c0_a, c1_b, real_t(1e-5)));
    REQUIRE(points_close(c1_a, c0_b, real_t(1e-5)));
}

TEMPLATE_TEST_CASE("swap_symmetry_aabb_plane", "[core][closest_metric_point_pair]",
    float, double)
{
    using real_t = TestType;

    auto aabb = tf::make_aabb(
        tf::make_point(real_t(0), real_t(0), real_t(0)),
        tf::make_point(real_t(2), real_t(2), real_t(2))
    );
    auto plane = tf::make_plane(
        tf::make_unit_vector(real_t(0), real_t(0), real_t(1)),
        real_t(-5)
    );

    auto [dist2_a, c0_a, c1_a] = tf::closest_metric_point_pair(aabb, plane);
    auto [dist2_b, c0_b, c1_b] = tf::closest_metric_point_pair(plane, aabb);

    REQUIRE(approx_equal(dist2_a, dist2_b));
    REQUIRE(points_close(c0_a, c1_b, real_t(1e-5)));
    REQUIRE(points_close(c1_a, c0_b, real_t(1e-5)));
}

TEMPLATE_TEST_CASE("swap_symmetry_aabb_polygon", "[core][closest_metric_point_pair]",
    float, double)
{
    using real_t = TestType;

    auto aabb = tf::make_aabb(
        tf::make_point(real_t(0), real_t(0), real_t(0)),
        tf::make_point(real_t(1), real_t(1), real_t(1))
    );
    std::array<tf::point<real_t, 3>, 3> tri_pts = {{
        tf::make_point(real_t(3), real_t(0), real_t(0)),
        tf::make_point(real_t(5), real_t(0), real_t(0)),
        tf::make_point(real_t(4), real_t(2), real_t(0))
    }};
    auto poly = tf::make_polygon(tri_pts);

    auto [dist2_a, c0_a, c1_a] = tf::closest_metric_point_pair(aabb, poly);
    auto [dist2_b, c0_b, c1_b] = tf::closest_metric_point_pair(poly, aabb);

    REQUIRE(approx_equal(dist2_a, dist2_b));
    REQUIRE(points_close(c0_a, c1_b, real_t(1e-5)));
    REQUIRE(points_close(c1_a, c0_b, real_t(1e-5)));
}

TEMPLATE_TEST_CASE("swap_symmetry_aabb_aabb", "[core][closest_metric_point_pair]",
    float, double)
{
    using real_t = TestType;

    auto a = tf::make_aabb(
        tf::make_point(real_t(0), real_t(0), real_t(0)),
        tf::make_point(real_t(1), real_t(1), real_t(1))
    );
    auto b = tf::make_aabb(
        tf::make_point(real_t(3), real_t(4), real_t(5)),
        tf::make_point(real_t(6), real_t(7), real_t(8))
    );

    auto [dist2_a, c0_a, c1_a] = tf::closest_metric_point_pair(a, b);
    auto [dist2_b, c0_b, c1_b] = tf::closest_metric_point_pair(b, a);

    REQUIRE(approx_equal(dist2_a, dist2_b));
    REQUIRE(points_close(c0_a, c1_b, real_t(1e-5)));
    REQUIRE(points_close(c1_a, c0_b, real_t(1e-5)));
}
