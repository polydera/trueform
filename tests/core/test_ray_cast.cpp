/**
 * @file test_ray_cast.cpp
 * @brief Tests for ray_cast functionality on core primitives
 *
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <trueform/core.hpp>
#include <array>

// =============================================================================
// Helper functions
// =============================================================================

template <typename real_t>
auto approx_equal(real_t a, real_t b, real_t tol = real_t(1e-5)) -> bool
{
    return std::abs(a - b) < tol;
}

// =============================================================================
// Ray-Plane tests
// =============================================================================

TEMPLATE_TEST_CASE("ray_plane_3d", "[core][ray_cast]",
    float, double)
{
    using real_t = TestType;

    // Plane at z=0 (xy-plane)
    auto plane = tf::make_plane(
        tf::make_unit_vector(real_t(0), real_t(0), real_t(1)),
        real_t(0)
    );

    SECTION("ray pointing down from above - should hit") {
        auto ray = tf::make_ray(
            tf::make_point(real_t(0.5), real_t(0.5), real_t(2)),
            tf::make_vector(real_t(0), real_t(0), real_t(-1))
        );
        auto result = tf::ray_cast(ray, plane);
        REQUIRE(result);
        REQUIRE(approx_equal(result.t, real_t(2)));

        // Verify hit point
        auto hit_point = ray.origin + result.t * ray.direction;
        REQUIRE(approx_equal(hit_point[2], real_t(0)));
    }

    SECTION("ray pointing away - should not hit") {
        auto ray = tf::make_ray(
            tf::make_point(real_t(0.5), real_t(0.5), real_t(2)),
            tf::make_vector(real_t(0), real_t(0), real_t(1))
        );
        auto result = tf::ray_cast(ray, plane);
        REQUIRE_FALSE(result);
    }

    SECTION("ray parallel to plane - should not hit") {
        auto ray = tf::make_ray(
            tf::make_point(real_t(0.5), real_t(0.5), real_t(2)),
            tf::make_vector(real_t(1), real_t(0), real_t(0))
        );
        auto result = tf::ray_cast(ray, plane);
        REQUIRE_FALSE(result);
    }
}

// =============================================================================
// Ray-Polygon tests (2D)
// =============================================================================

TEMPLATE_TEST_CASE("ray_polygon_2d", "[core][ray_cast]",
    float, double)
{
    using real_t = TestType;

    // Square polygon
    std::array<tf::point<real_t, 2>, 4> square_pts = {{
        tf::make_point(real_t(0), real_t(0)),
        tf::make_point(real_t(1), real_t(0)),
        tf::make_point(real_t(1), real_t(1)),
        tf::make_point(real_t(0), real_t(1))
    }};
    auto poly = tf::make_polygon(square_pts);

    SECTION("ray from left pointing right - should hit") {
        auto ray = tf::make_ray(
            tf::make_point(real_t(-1), real_t(0.5)),
            tf::make_vector(real_t(1), real_t(0))
        );
        auto result = tf::ray_cast(ray, poly);
        REQUIRE(result);
        REQUIRE(approx_equal(result.t, real_t(1)));
    }

    SECTION("ray from right pointing away - should not hit") {
        auto ray = tf::make_ray(
            tf::make_point(real_t(2), real_t(0.5)),
            tf::make_vector(real_t(1), real_t(0))
        );
        auto result = tf::ray_cast(ray, poly);
        REQUIRE_FALSE(result);
    }

    SECTION("ray starting inside - should hit") {
        auto ray = tf::make_ray(
            tf::make_point(real_t(0.5), real_t(0.5)),
            tf::make_vector(real_t(1), real_t(0))
        );
        auto result = tf::ray_cast(ray, poly);
        REQUIRE(result);
    }
}

// =============================================================================
// Ray-Polygon tests (3D)
// =============================================================================

TEMPLATE_TEST_CASE("ray_polygon_3d", "[core][ray_cast]",
    float, double)
{
    using real_t = TestType;

    // Triangle in xy-plane at z=0
    std::array<tf::point<real_t, 3>, 3> triangle_pts = {{
        tf::make_point(real_t(0), real_t(0), real_t(0)),
        tf::make_point(real_t(1), real_t(0), real_t(0)),
        tf::make_point(real_t(0.5), real_t(1), real_t(0))
    }};
    auto poly = tf::make_polygon(triangle_pts);

    SECTION("ray pointing down from above - should hit") {
        auto ray = tf::make_ray(
            tf::make_point(real_t(0.5), real_t(0.3), real_t(2)),
            tf::make_vector(real_t(0), real_t(0), real_t(-1))
        );
        auto result = tf::ray_cast(ray, poly);
        REQUIRE(result);
        REQUIRE(approx_equal(result.t, real_t(2)));
    }

    SECTION("ray from above but offset (outside triangle) - should not hit") {
        auto ray = tf::make_ray(
            tf::make_point(real_t(2), real_t(2), real_t(2)),
            tf::make_vector(real_t(0), real_t(0), real_t(-1))
        );
        auto result = tf::ray_cast(ray, poly);
        REQUIRE_FALSE(result);
    }

    SECTION("ray pointing away - should not hit") {
        auto ray = tf::make_ray(
            tf::make_point(real_t(0.5), real_t(0.3), real_t(2)),
            tf::make_vector(real_t(0), real_t(0), real_t(1))
        );
        auto result = tf::ray_cast(ray, poly);
        REQUIRE_FALSE(result);
    }
}

// =============================================================================
// Ray-Segment tests (2D)
// =============================================================================

TEMPLATE_TEST_CASE("ray_segment_2d", "[core][ray_cast]",
    float, double)
{
    using real_t = TestType;

    // Vertical segment
    auto segment = tf::make_segment_between_points(
        tf::make_point(real_t(1), real_t(0)),
        tf::make_point(real_t(1), real_t(2))
    );

    SECTION("ray from left pointing right - should hit") {
        auto ray = tf::make_ray(
            tf::make_point(real_t(0), real_t(1)),
            tf::make_vector(real_t(1), real_t(0))
        );
        auto result = tf::ray_cast(ray, segment);
        REQUIRE(result);
        REQUIRE(approx_equal(result.t, real_t(1)));
    }

    SECTION("ray from left but above segment - should not hit") {
        auto ray = tf::make_ray(
            tf::make_point(real_t(0), real_t(3)),
            tf::make_vector(real_t(1), real_t(0))
        );
        auto result = tf::ray_cast(ray, segment);
        REQUIRE_FALSE(result);
    }

    SECTION("ray pointing away - should not hit") {
        auto ray = tf::make_ray(
            tf::make_point(real_t(0), real_t(1)),
            tf::make_vector(real_t(-1), real_t(0))
        );
        auto result = tf::ray_cast(ray, segment);
        REQUIRE_FALSE(result);
    }
}

// =============================================================================
// Ray-Segment tests (3D)
// =============================================================================

TEMPLATE_TEST_CASE("ray_segment_3d", "[core][ray_cast]",
    float, double)
{
    using real_t = TestType;

    // Segment along x-axis
    auto segment = tf::make_segment_between_points(
        tf::make_point(real_t(0), real_t(0.5), real_t(0.5)),
        tf::make_point(real_t(2), real_t(0.5), real_t(0.5))
    );

    SECTION("ray from below pointing up - should hit") {
        auto ray = tf::make_ray(
            tf::make_point(real_t(1), real_t(0), real_t(0.5)),
            tf::make_vector(real_t(0), real_t(1), real_t(0))
        );
        auto result = tf::ray_cast(ray, segment);
        REQUIRE(result);
        REQUIRE(approx_equal(result.t, real_t(0.5)));
    }

    SECTION("ray parallel but offset - should not hit") {
        auto ray = tf::make_ray(
            tf::make_point(real_t(0), real_t(1.5), real_t(0.5)),
            tf::make_vector(real_t(1), real_t(0), real_t(0))
        );
        auto result = tf::ray_cast(ray, segment);
        REQUIRE_FALSE(result);
    }
}

// =============================================================================
// Ray-Line tests (2D)
// =============================================================================

TEMPLATE_TEST_CASE("ray_line_2d", "[core][ray_cast]",
    float, double)
{
    using real_t = TestType;

    // Vertical line at x=1
    auto line = tf::make_line_like(
        tf::make_point(real_t(1), real_t(0)),
        tf::make_vector(real_t(0), real_t(1))
    );

    SECTION("ray from left pointing right - should hit") {
        auto ray = tf::make_ray(
            tf::make_point(real_t(0), real_t(0.5)),
            tf::make_vector(real_t(1), real_t(0))
        );
        auto result = tf::ray_cast(ray, line);
        REQUIRE(result);
        REQUIRE(approx_equal(result.t, real_t(1)));
    }

    SECTION("ray parallel to line - should not hit") {
        auto ray = tf::make_ray(
            tf::make_point(real_t(0), real_t(0.5)),
            tf::make_vector(real_t(0), real_t(1))
        );
        auto result = tf::ray_cast(ray, line);
        REQUIRE_FALSE(result);
    }

    SECTION("ray pointing away - should not hit") {
        auto ray = tf::make_ray(
            tf::make_point(real_t(0), real_t(0.5)),
            tf::make_vector(real_t(-1), real_t(0))
        );
        auto result = tf::ray_cast(ray, line);
        REQUIRE_FALSE(result);
    }
}

// =============================================================================
// Ray-Line tests (3D)
// =============================================================================

TEMPLATE_TEST_CASE("ray_line_3d", "[core][ray_cast]",
    float, double)
{
    using real_t = TestType;

    // Line along z-axis through origin
    auto line = tf::make_line_like(
        tf::make_point(real_t(0), real_t(0), real_t(0)),
        tf::make_vector(real_t(0), real_t(0), real_t(1))
    );

    SECTION("ray in xy-plane pointing at line - should hit") {
        auto ray = tf::make_ray(
            tf::make_point(real_t(1), real_t(0), real_t(0.5)),
            tf::make_vector(real_t(-1), real_t(0), real_t(0))
        );
        auto result = tf::ray_cast(ray, line);
        REQUIRE(result);
        REQUIRE(approx_equal(result.t, real_t(1)));
    }

    SECTION("ray skew to line - should not hit") {
        auto ray = tf::make_ray(
            tf::make_point(real_t(1), real_t(1), real_t(0)),
            tf::make_vector(real_t(1), real_t(0), real_t(0))
        );
        auto result = tf::ray_cast(ray, line);
        REQUIRE_FALSE(result);
    }
}

// =============================================================================
// Ray-AABB tests (2D)
// =============================================================================

TEMPLATE_TEST_CASE("ray_aabb_2d", "[core][ray_cast]",
    float, double)
{
    using real_t = TestType;

    // AABB from [0,0] to [1,1]
    auto aabb = tf::make_aabb(
        tf::make_point(real_t(0), real_t(0)),
        tf::make_point(real_t(1), real_t(1))
    );

    SECTION("ray from left pointing right - should hit") {
        auto ray = tf::make_ray(
            tf::make_point(real_t(-1), real_t(0.5)),
            tf::make_vector(real_t(1), real_t(0))
        );
        auto result = tf::ray_cast(ray, aabb);
        REQUIRE(result);
        REQUIRE(approx_equal(result.t, real_t(1)));
    }

    SECTION("ray from right pointing away - should not hit") {
        auto ray = tf::make_ray(
            tf::make_point(real_t(2), real_t(0.5)),
            tf::make_vector(real_t(1), real_t(0))
        );
        auto result = tf::ray_cast(ray, aabb);
        REQUIRE_FALSE(result);
    }

    SECTION("ray starting inside - should hit") {
        auto ray = tf::make_ray(
            tf::make_point(real_t(0.5), real_t(0.5)),
            tf::make_vector(real_t(1), real_t(0))
        );
        auto result = tf::ray_cast(ray, aabb);
        REQUIRE(result);
    }
}

// =============================================================================
// Ray-AABB tests (3D)
// =============================================================================

TEMPLATE_TEST_CASE("ray_aabb_3d", "[core][ray_cast]",
    float, double)
{
    using real_t = TestType;

    // AABB cube from [0,0,0] to [1,1,1]
    auto aabb = tf::make_aabb(
        tf::make_point(real_t(0), real_t(0), real_t(0)),
        tf::make_point(real_t(1), real_t(1), real_t(1))
    );

    SECTION("ray from above pointing down - should hit") {
        auto ray = tf::make_ray(
            tf::make_point(real_t(0.5), real_t(0.5), real_t(2)),
            tf::make_vector(real_t(0), real_t(0), real_t(-1))
        );
        auto result = tf::ray_cast(ray, aabb);
        REQUIRE(result);
        REQUIRE(approx_equal(result.t, real_t(1)));
    }

    SECTION("ray from above but outside AABB - should not hit") {
        auto ray = tf::make_ray(
            tf::make_point(real_t(2), real_t(2), real_t(2)),
            tf::make_vector(real_t(0), real_t(0), real_t(-1))
        );
        auto result = tf::ray_cast(ray, aabb);
        REQUIRE_FALSE(result);
    }

    SECTION("ray pointing away - should not hit") {
        auto ray = tf::make_ray(
            tf::make_point(real_t(0.5), real_t(0.5), real_t(2)),
            tf::make_vector(real_t(0), real_t(0), real_t(1))
        );
        auto result = tf::ray_cast(ray, aabb);
        REQUIRE_FALSE(result);
    }

    SECTION("diagonal ray through cube") {
        auto ray = tf::make_ray(
            tf::make_point(real_t(-1), real_t(-1), real_t(-1)),
            tf::make_vector(real_t(1), real_t(1), real_t(1))
        );
        auto result = tf::ray_cast(ray, aabb);
        REQUIRE(result);
    }
}

// =============================================================================
// Ray-Ray tests (2D)
// =============================================================================

TEMPLATE_TEST_CASE("ray_ray_2d", "[core][ray_cast]",
    float, double)
{
    using real_t = TestType;

    SECTION("intersecting rays") {
        auto ray1 = tf::make_ray(
            tf::make_point(real_t(0), real_t(0)),
            tf::make_vector(real_t(1), real_t(0))
        );
        auto ray2 = tf::make_ray(
            tf::make_point(real_t(2), real_t(-1)),
            tf::make_vector(real_t(0), real_t(1))
        );
        auto result = tf::ray_cast(ray1, ray2);
        REQUIRE(result);
        REQUIRE(approx_equal(result.t, real_t(2)));
    }

    SECTION("parallel rays - should not hit") {
        auto ray1 = tf::make_ray(
            tf::make_point(real_t(0), real_t(0)),
            tf::make_vector(real_t(1), real_t(0))
        );
        auto ray2 = tf::make_ray(
            tf::make_point(real_t(0), real_t(1)),
            tf::make_vector(real_t(1), real_t(0))
        );
        auto result = tf::ray_cast(ray1, ray2);
        REQUIRE_FALSE(result);
    }

    SECTION("diverging rays - should not hit") {
        auto ray1 = tf::make_ray(
            tf::make_point(real_t(0), real_t(0)),
            tf::make_vector(real_t(1), real_t(0))
        );
        auto ray2 = tf::make_ray(
            tf::make_point(real_t(-1), real_t(1)),
            tf::make_vector(real_t(-1), real_t(0))
        );
        auto result = tf::ray_cast(ray1, ray2);
        REQUIRE_FALSE(result);
    }
}

// =============================================================================
// Ray-Point tests
// =============================================================================

TEMPLATE_TEST_CASE("ray_point_2d", "[core][ray_cast]",
    float, double)
{
    using real_t = TestType;

    SECTION("ray hitting point") {
        auto ray = tf::make_ray(
            tf::make_point(real_t(0), real_t(0)),
            tf::make_vector(real_t(1), real_t(0))
        );
        auto pt = tf::make_point(real_t(3), real_t(0));
        auto result = tf::ray_cast(ray, pt);
        REQUIRE(result);
        REQUIRE(approx_equal(result.t, real_t(3)));
    }

    SECTION("ray missing point") {
        auto ray = tf::make_ray(
            tf::make_point(real_t(0), real_t(0)),
            tf::make_vector(real_t(1), real_t(0))
        );
        auto pt = tf::make_point(real_t(3), real_t(1));
        auto result = tf::ray_cast(ray, pt);
        REQUIRE_FALSE(result);
    }

    SECTION("point behind ray origin") {
        auto ray = tf::make_ray(
            tf::make_point(real_t(0), real_t(0)),
            tf::make_vector(real_t(1), real_t(0))
        );
        auto pt = tf::make_point(real_t(-3), real_t(0));
        auto result = tf::ray_cast(ray, pt);
        REQUIRE_FALSE(result);
    }
}

TEMPLATE_TEST_CASE("ray_point_3d", "[core][ray_cast]",
    float, double)
{
    using real_t = TestType;

    SECTION("ray hitting point") {
        auto ray = tf::make_ray(
            tf::make_point(real_t(0), real_t(0), real_t(0)),
            tf::make_vector(real_t(0), real_t(0), real_t(1))
        );
        auto pt = tf::make_point(real_t(0), real_t(0), real_t(5));
        auto result = tf::ray_cast(ray, pt);
        REQUIRE(result);
        REQUIRE(approx_equal(result.t, real_t(5)));
    }

    SECTION("ray missing point") {
        auto ray = tf::make_ray(
            tf::make_point(real_t(0), real_t(0), real_t(0)),
            tf::make_vector(real_t(0), real_t(0), real_t(1))
        );
        auto pt = tf::make_point(real_t(1), real_t(0), real_t(5));
        auto result = tf::ray_cast(ray, pt);
        REQUIRE_FALSE(result);
    }
}

// =============================================================================
// Ray config tests
// =============================================================================

TEMPLATE_TEST_CASE("ray_cast_with_config", "[core][ray_cast]",
    float, double)
{
    using real_t = TestType;

    // Plane at z=0
    auto plane = tf::make_plane(
        tf::make_unit_vector(real_t(0), real_t(0), real_t(1)),
        real_t(0)
    );

    SECTION("ray with min_t > hit - should not hit") {
        auto ray = tf::make_ray(
            tf::make_point(real_t(0), real_t(0), real_t(2)),
            tf::make_vector(real_t(0), real_t(0), real_t(-1))
        );
        // Hit would be at t=2, but min_t=3
        auto config = tf::make_ray_config(real_t(3), std::numeric_limits<real_t>::max());
        auto result = tf::ray_cast(ray, plane, config);
        REQUIRE_FALSE(result);
    }

    SECTION("ray with max_t < hit - should not hit") {
        auto ray = tf::make_ray(
            tf::make_point(real_t(0), real_t(0), real_t(2)),
            tf::make_vector(real_t(0), real_t(0), real_t(-1))
        );
        // Hit would be at t=2, but max_t=1
        auto config = tf::make_ray_config(real_t(0), real_t(1));
        auto result = tf::ray_cast(ray, plane, config);
        REQUIRE_FALSE(result);
    }

    SECTION("ray with config including hit - should hit") {
        auto ray = tf::make_ray(
            tf::make_point(real_t(0), real_t(0), real_t(2)),
            tf::make_vector(real_t(0), real_t(0), real_t(-1))
        );
        // Hit at t=2, config allows [1, 3]
        auto config = tf::make_ray_config(real_t(1), real_t(3));
        auto result = tf::ray_cast(ray, plane, config);
        REQUIRE(result);
        REQUIRE(approx_equal(result.t, real_t(2)));
    }
}

// =============================================================================
// Intersect status tests
// =============================================================================

TEMPLATE_TEST_CASE("ray_cast_status", "[core][ray_cast]",
    float, double)
{
    using real_t = TestType;

    SECTION("intersection status") {
        auto plane = tf::make_plane(
            tf::make_unit_vector(real_t(0), real_t(0), real_t(1)),
            real_t(0)
        );
        auto ray = tf::make_ray(
            tf::make_point(real_t(0), real_t(0), real_t(2)),
            tf::make_vector(real_t(0), real_t(0), real_t(-1))
        );
        auto result = tf::ray_cast(ray, plane);
        REQUIRE(result.status == tf::intersect_status::intersection);
    }

    SECTION("parallel status") {
        auto plane = tf::make_plane(
            tf::make_unit_vector(real_t(0), real_t(0), real_t(1)),
            real_t(0)
        );
        auto ray = tf::make_ray(
            tf::make_point(real_t(0), real_t(0), real_t(2)),
            tf::make_vector(real_t(1), real_t(0), real_t(0))
        );
        auto result = tf::ray_cast(ray, plane);
        REQUIRE(result.status == tf::intersect_status::parallel);
    }
}

// =============================================================================
// Coplanar/Colinear tests
// =============================================================================

TEMPLATE_TEST_CASE("ray_plane_coplanar", "[core][ray_cast]",
    float, double)
{
    using real_t = TestType;

    // Plane at z=0
    auto plane = tf::make_plane(
        tf::make_unit_vector(real_t(0), real_t(0), real_t(1)),
        real_t(0)
    );

    SECTION("ray on plane - coplanar") {
        auto ray = tf::make_ray(
            tf::make_point(real_t(0), real_t(0), real_t(0)),
            tf::make_vector(real_t(1), real_t(0), real_t(0))
        );
        auto result = tf::ray_cast(ray, plane);
        REQUIRE(result.status == tf::intersect_status::coplanar);
    }
}

TEMPLATE_TEST_CASE("ray_polygon_coplanar", "[core][ray_cast]",
    float, double)
{
    using real_t = TestType;

    // Triangle in xy-plane at z=0
    std::array<tf::point<real_t, 3>, 3> triangle_pts = {{
        tf::make_point(real_t(0), real_t(0), real_t(0)),
        tf::make_point(real_t(2), real_t(0), real_t(0)),
        tf::make_point(real_t(1), real_t(2), real_t(0))
    }};
    auto poly = tf::make_polygon(triangle_pts);

    SECTION("ray coplanar with polygon, pointing toward it") {
        auto ray = tf::make_ray(
            tf::make_point(real_t(-1), real_t(0.5), real_t(0)),
            tf::make_vector(real_t(1), real_t(0), real_t(0))
        );
        auto result = tf::ray_cast(ray, poly);
        // Should either intersect or be coplanar depending on impl
        REQUIRE(tf::core::does_intersect_any(result));
    }

    SECTION("ray coplanar with polygon, origin inside") {
        auto ray = tf::make_ray(
            tf::make_point(real_t(1), real_t(0.5), real_t(0)),
            tf::make_vector(real_t(1), real_t(0), real_t(0))
        );
        auto result = tf::ray_cast(ray, poly);
        REQUIRE(tf::core::does_intersect_any(result));
    }

    SECTION("ray coplanar with polygon, pointing away") {
        auto ray = tf::make_ray(
            tf::make_point(real_t(-1), real_t(0.5), real_t(0)),
            tf::make_vector(real_t(-1), real_t(0), real_t(0))
        );
        auto result = tf::ray_cast(ray, poly);
        // Status is coplanar (geometric relationship), but no actual hit
        REQUIRE(result.status == tf::intersect_status::coplanar);
        REQUIRE_FALSE(static_cast<bool>(result));
    }
}

TEMPLATE_TEST_CASE("ray_segment_colinear", "[core][ray_cast]",
    float, double)
{
    using real_t = TestType;

    // Segment along x-axis from x=2 to x=4
    auto segment = tf::make_segment_between_points(
        tf::make_point(real_t(2), real_t(0)),
        tf::make_point(real_t(4), real_t(0))
    );

    SECTION("ray colinear with segment, pointing toward it") {
        auto ray = tf::make_ray(
            tf::make_point(real_t(0), real_t(0)),
            tf::make_vector(real_t(1), real_t(0))
        );
        auto result = tf::ray_cast(ray, segment);
        REQUIRE(tf::core::does_intersect_any(result));
    }

    SECTION("ray colinear with segment, starting inside") {
        auto ray = tf::make_ray(
            tf::make_point(real_t(3), real_t(0)),
            tf::make_vector(real_t(1), real_t(0))
        );
        auto result = tf::ray_cast(ray, segment);
        REQUIRE(tf::core::does_intersect_any(result));
    }

    SECTION("ray colinear with segment, pointing away") {
        auto ray = tf::make_ray(
            tf::make_point(real_t(0), real_t(0)),
            tf::make_vector(real_t(-1), real_t(0))
        );
        auto result = tf::ray_cast(ray, segment);
        // Status is colinear (geometric relationship), but no actual hit
        REQUIRE(result.status == tf::intersect_status::colinear);
        REQUIRE_FALSE(static_cast<bool>(result));
    }

    SECTION("ray colinear but past segment") {
        auto ray = tf::make_ray(
            tf::make_point(real_t(5), real_t(0)),
            tf::make_vector(real_t(1), real_t(0))
        );
        auto result = tf::ray_cast(ray, segment);
        // Status is colinear (geometric relationship), but no actual hit
        REQUIRE(result.status == tf::intersect_status::colinear);
        REQUIRE_FALSE(static_cast<bool>(result));
    }
}

TEMPLATE_TEST_CASE("ray_line_colinear", "[core][ray_cast]",
    float, double)
{
    using real_t = TestType;

    // Line along x-axis
    auto line = tf::make_line_like(
        tf::make_point(real_t(0), real_t(0)),
        tf::make_vector(real_t(1), real_t(0))
    );

    SECTION("ray colinear with line") {
        auto ray = tf::make_ray(
            tf::make_point(real_t(5), real_t(0)),
            tf::make_vector(real_t(1), real_t(0))
        );
        auto result = tf::ray_cast(ray, line);
        // Colinear - should report colinear status
        REQUIRE(result.status == tf::intersect_status::colinear);
    }

    SECTION("ray colinear, opposite direction") {
        auto ray = tf::make_ray(
            tf::make_point(real_t(5), real_t(0)),
            tf::make_vector(real_t(-1), real_t(0))
        );
        auto result = tf::ray_cast(ray, line);
        REQUIRE(result.status == tf::intersect_status::colinear);
    }
}

TEMPLATE_TEST_CASE("ray_ray_colinear", "[core][ray_cast]",
    float, double)
{
    using real_t = TestType;

    SECTION("colinear rays, same direction") {
        auto ray1 = tf::make_ray(
            tf::make_point(real_t(0), real_t(0)),
            tf::make_vector(real_t(1), real_t(0))
        );
        auto ray2 = tf::make_ray(
            tf::make_point(real_t(5), real_t(0)),
            tf::make_vector(real_t(1), real_t(0))
        );
        auto result = tf::ray_cast(ray1, ray2);
        REQUIRE(tf::core::does_intersect_any(result));
    }

    SECTION("colinear rays, opposite directions, converging") {
        auto ray1 = tf::make_ray(
            tf::make_point(real_t(0), real_t(0)),
            tf::make_vector(real_t(1), real_t(0))
        );
        auto ray2 = tf::make_ray(
            tf::make_point(real_t(5), real_t(0)),
            tf::make_vector(real_t(-1), real_t(0))
        );
        auto result = tf::ray_cast(ray1, ray2);
        REQUIRE(tf::core::does_intersect_any(result));
    }

    SECTION("colinear rays, opposite directions, diverging") {
        auto ray1 = tf::make_ray(
            tf::make_point(real_t(0), real_t(0)),
            tf::make_vector(real_t(-1), real_t(0))
        );
        auto ray2 = tf::make_ray(
            tf::make_point(real_t(5), real_t(0)),
            tf::make_vector(real_t(1), real_t(0))
        );
        auto result = tf::ray_cast(ray1, ray2);
        // Status is colinear (geometric relationship), but no actual hit
        REQUIRE(result.status == tf::intersect_status::colinear);
        REQUIRE_FALSE(static_cast<bool>(result));
    }
}

// =============================================================================
// Line vs Flat AABB (degenerate case)
// =============================================================================

TEMPLATE_TEST_CASE("line_flat_aabb_3d", "[core][ray_cast]",
    float, double)
{
    using real_t = TestType;

    // Flat AABB at z=0 (zero thickness in z)
    auto flat_aabb = tf::make_aabb(
        tf::make_point(real_t(0), real_t(0), real_t(0)),
        tf::make_point(real_t(4), real_t(4), real_t(0))
    );

    SECTION("line through flat aabb - should intersect") {
        // Line at (2, 2, 5) with direction (0, 0, 1) passes through (2, 2, 0)
        auto line = tf::make_line_like(
            tf::make_point(real_t(2), real_t(2), real_t(5)),
            tf::make_vector(real_t(0), real_t(0), real_t(1))
        );
        REQUIRE(tf::intersects(line, flat_aabb));
        REQUIRE(tf::intersects(flat_aabb, line));
    }

    SECTION("line missing flat aabb - should not intersect") {
        // Line at (10, 10, 5) with direction (0, 0, 1) - outside xy bounds
        auto line = tf::make_line_like(
            tf::make_point(real_t(10), real_t(10), real_t(5)),
            tf::make_vector(real_t(0), real_t(0), real_t(1))
        );
        REQUIRE_FALSE(tf::intersects(line, flat_aabb));
    }

    SECTION("line parallel to flat aabb but not touching") {
        // Line parallel to xy plane at z=1
        auto line = tf::make_line_like(
            tf::make_point(real_t(2), real_t(2), real_t(1)),
            tf::make_vector(real_t(1), real_t(0), real_t(0))
        );
        REQUIRE_FALSE(tf::intersects(line, flat_aabb));
    }

    SECTION("line on the flat aabb surface") {
        // Line on z=0 plane within xy bounds
        auto line = tf::make_line_like(
            tf::make_point(real_t(2), real_t(2), real_t(0)),
            tf::make_vector(real_t(1), real_t(0), real_t(0))
        );
        REQUIRE(tf::intersects(line, flat_aabb));
    }
}
