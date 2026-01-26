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
