/**
 * @file test_classify.cpp
 * @brief Tests for point classification functionality (sidedness and containment)
 *
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <trueform/core.hpp>
#include <array>

// =============================================================================
// Point-Plane classification tests (3D) - returns sidedness
// =============================================================================

TEMPLATE_TEST_CASE("classify_point_plane", "[core][classify]",
    float, double)
{
    using real_t = TestType;

    // Plane at z=0 with normal pointing up (+z)
    auto plane = tf::make_plane(
        tf::make_unit_vector(real_t(0), real_t(0), real_t(1)),
        real_t(0)
    );

    SECTION("point above plane (positive side)") {
        auto pt = tf::make_point(real_t(0), real_t(0), real_t(5));
        auto result = tf::classify(pt, plane);
        REQUIRE(result == tf::sidedness::on_positive_side);
    }

    SECTION("point below plane (negative side)") {
        auto pt = tf::make_point(real_t(0), real_t(0), real_t(-5));
        auto result = tf::classify(pt, plane);
        REQUIRE(result == tf::sidedness::on_negative_side);
    }

    SECTION("point on plane (boundary)") {
        auto pt = tf::make_point(real_t(5), real_t(3), real_t(0));
        auto result = tf::classify(pt, plane);
        REQUIRE(result == tf::sidedness::on_boundary);
    }

    SECTION("point very close to plane (boundary)") {
        auto pt = tf::make_point(real_t(0), real_t(0), real_t(0));
        auto result = tf::classify(pt, plane);
        REQUIRE(result == tf::sidedness::on_boundary);
    }
}

TEMPLATE_TEST_CASE("classify_point_plane_offset", "[core][classify]",
    float, double)
{
    using real_t = TestType;

    // Plane at z=5 with normal pointing up (+z)
    auto plane = tf::make_plane(
        tf::make_unit_vector(real_t(0), real_t(0), real_t(1)),
        real_t(-5)  // d = -dot(normal, point_on_plane) = -5
    );

    SECTION("point above offset plane") {
        auto pt = tf::make_point(real_t(0), real_t(0), real_t(10));
        auto result = tf::classify(pt, plane);
        REQUIRE(result == tf::sidedness::on_positive_side);
    }

    SECTION("point below offset plane") {
        auto pt = tf::make_point(real_t(0), real_t(0), real_t(0));
        auto result = tf::classify(pt, plane);
        REQUIRE(result == tf::sidedness::on_negative_side);
    }

    SECTION("point on offset plane") {
        auto pt = tf::make_point(real_t(0), real_t(0), real_t(5));
        auto result = tf::classify(pt, plane);
        REQUIRE(result == tf::sidedness::on_boundary);
    }
}

TEMPLATE_TEST_CASE("classify_point_plane_tilted", "[core][classify]",
    float, double)
{
    using real_t = TestType;

    // Tilted plane with normal (1, 1, 1) normalized, passing through origin
    auto plane = tf::make_plane(
        tf::make_unit_vector(real_t(1), real_t(1), real_t(1)),
        real_t(0)
    );

    SECTION("point on positive side of tilted plane") {
        auto pt = tf::make_point(real_t(1), real_t(1), real_t(1));
        auto result = tf::classify(pt, plane);
        REQUIRE(result == tf::sidedness::on_positive_side);
    }

    SECTION("point on negative side of tilted plane") {
        auto pt = tf::make_point(real_t(-1), real_t(-1), real_t(-1));
        auto result = tf::classify(pt, plane);
        REQUIRE(result == tf::sidedness::on_negative_side);
    }

    SECTION("point on tilted plane") {
        // Point (1, -1, 0) has dot product with (1,1,1) = 1-1+0 = 0
        auto pt = tf::make_point(real_t(1), real_t(-1), real_t(0));
        auto result = tf::classify(pt, plane);
        REQUIRE(result == tf::sidedness::on_boundary);
    }
}

// =============================================================================
// Point-Line classification tests (2D) - returns sidedness
// =============================================================================

TEMPLATE_TEST_CASE("classify_point_line_2d", "[core][classify]",
    float, double)
{
    using real_t = TestType;

    // Line along x-axis through origin, direction (1, 0)
    auto line = tf::make_line_like(
        tf::make_point(real_t(0), real_t(0)),
        tf::make_vector(real_t(1), real_t(0))
    );

    SECTION("point above line (positive side / left)") {
        auto pt = tf::make_point(real_t(5), real_t(3));
        auto result = tf::classify(pt, line);
        // Convention: positive = left of direction vector
        REQUIRE(result == tf::sidedness::on_positive_side);
    }

    SECTION("point below line (negative side / right)") {
        auto pt = tf::make_point(real_t(5), real_t(-3));
        auto result = tf::classify(pt, line);
        REQUIRE(result == tf::sidedness::on_negative_side);
    }

    SECTION("point on line (boundary)") {
        auto pt = tf::make_point(real_t(100), real_t(0));
        auto result = tf::classify(pt, line);
        REQUIRE(result == tf::sidedness::on_boundary);
    }

    SECTION("point at line origin") {
        auto pt = tf::make_point(real_t(0), real_t(0));
        auto result = tf::classify(pt, line);
        REQUIRE(result == tf::sidedness::on_boundary);
    }
}

TEMPLATE_TEST_CASE("classify_point_line_2d_diagonal", "[core][classify]",
    float, double)
{
    using real_t = TestType;

    // Diagonal line through origin, direction (1, 1)
    auto line = tf::make_line_like(
        tf::make_point(real_t(0), real_t(0)),
        tf::make_vector(real_t(1), real_t(1))
    );

    SECTION("point left of diagonal line") {
        // Point (0, 2) is to the left of line y=x
        auto pt = tf::make_point(real_t(0), real_t(2));
        auto result = tf::classify(pt, line);
        REQUIRE(result == tf::sidedness::on_positive_side);
    }

    SECTION("point right of diagonal line") {
        // Point (2, 0) is to the right of line y=x
        auto pt = tf::make_point(real_t(2), real_t(0));
        auto result = tf::classify(pt, line);
        REQUIRE(result == tf::sidedness::on_negative_side);
    }

    SECTION("point on diagonal line") {
        auto pt = tf::make_point(real_t(5), real_t(5));
        auto result = tf::classify(pt, line);
        REQUIRE(result == tf::sidedness::on_boundary);
    }
}

// =============================================================================
// Point-Ray classification tests (2D) - returns sidedness
// =============================================================================

TEMPLATE_TEST_CASE("classify_point_ray_2d", "[core][classify]",
    float, double)
{
    using real_t = TestType;

    // Ray from origin, direction (1, 0)
    auto ray = tf::make_ray(
        tf::make_point(real_t(0), real_t(0)),
        tf::make_vector(real_t(1), real_t(0))
    );

    SECTION("point above ray (positive side)") {
        auto pt = tf::make_point(real_t(5), real_t(3));
        auto result = tf::classify(pt, ray);
        REQUIRE(result == tf::sidedness::on_positive_side);
    }

    SECTION("point below ray (negative side)") {
        auto pt = tf::make_point(real_t(5), real_t(-3));
        auto result = tf::classify(pt, ray);
        REQUIRE(result == tf::sidedness::on_negative_side);
    }

    SECTION("point on ray (boundary)") {
        auto pt = tf::make_point(real_t(10), real_t(0));
        auto result = tf::classify(pt, ray);
        REQUIRE(result == tf::sidedness::on_boundary);
    }

    SECTION("point at ray origin") {
        auto pt = tf::make_point(real_t(0), real_t(0));
        auto result = tf::classify(pt, ray);
        REQUIRE(result == tf::sidedness::on_boundary);
    }

    SECTION("point behind ray (colinear but behind origin)") {
        auto pt = tf::make_point(real_t(-5), real_t(0));
        auto result = tf::classify(pt, ray);
        // For rays, points behind the origin are classified as on_negative_side
        REQUIRE(result == tf::sidedness::on_negative_side);
    }
}

// =============================================================================
// Point-Segment classification tests (2D) - returns sidedness
// =============================================================================

TEMPLATE_TEST_CASE("classify_point_segment_2d", "[core][classify]",
    float, double)
{
    using real_t = TestType;

    // Horizontal segment from (0,0) to (4,0)
    auto seg = tf::make_segment_between_points(
        tf::make_point(real_t(0), real_t(0)),
        tf::make_point(real_t(4), real_t(0))
    );

    SECTION("point above segment (positive side)") {
        auto pt = tf::make_point(real_t(2), real_t(3));
        auto result = tf::classify(pt, seg);
        REQUIRE(result == tf::sidedness::on_positive_side);
    }

    SECTION("point below segment (negative side)") {
        auto pt = tf::make_point(real_t(2), real_t(-3));
        auto result = tf::classify(pt, seg);
        REQUIRE(result == tf::sidedness::on_negative_side);
    }

    SECTION("point on segment (boundary)") {
        auto pt = tf::make_point(real_t(2), real_t(0));
        auto result = tf::classify(pt, seg);
        REQUIRE(result == tf::sidedness::on_boundary);
    }

    SECTION("point at segment endpoint") {
        auto pt = tf::make_point(real_t(0), real_t(0));
        auto result = tf::classify(pt, seg);
        REQUIRE(result == tf::sidedness::on_boundary);
    }

    SECTION("point colinear but outside segment") {
        // For segments, points outside the segment bounds are classified as on_negative_side
        auto pt = tf::make_point(real_t(10), real_t(0));
        auto result = tf::classify(pt, seg);
        REQUIRE(result == tf::sidedness::on_negative_side);
    }
}

TEMPLATE_TEST_CASE("classify_point_segment_2d_vertical", "[core][classify]",
    float, double)
{
    using real_t = TestType;

    // Vertical segment from (0,0) to (0,4)
    auto seg = tf::make_segment_between_points(
        tf::make_point(real_t(0), real_t(0)),
        tf::make_point(real_t(0), real_t(4))
    );

    SECTION("point right of vertical segment (negative side)") {
        auto pt = tf::make_point(real_t(3), real_t(2));
        auto result = tf::classify(pt, seg);
        REQUIRE(result == tf::sidedness::on_negative_side);
    }

    SECTION("point left of vertical segment (positive side)") {
        auto pt = tf::make_point(real_t(-3), real_t(2));
        auto result = tf::classify(pt, seg);
        REQUIRE(result == tf::sidedness::on_positive_side);
    }

    SECTION("point on vertical segment") {
        auto pt = tf::make_point(real_t(0), real_t(2));
        auto result = tf::classify(pt, seg);
        REQUIRE(result == tf::sidedness::on_boundary);
    }
}

// =============================================================================
// Point-Polygon classification tests (2D) - returns containment
// =============================================================================

TEMPLATE_TEST_CASE("classify_point_polygon_2d", "[core][classify]",
    float, double)
{
    using real_t = TestType;

    // Square polygon: (0,0), (4,0), (4,4), (0,4)
    std::array<tf::point<real_t, 2>, 4> square_pts = {{
        tf::make_point(real_t(0), real_t(0)),
        tf::make_point(real_t(4), real_t(0)),
        tf::make_point(real_t(4), real_t(4)),
        tf::make_point(real_t(0), real_t(4))
    }};
    auto poly = tf::make_polygon(square_pts);

    SECTION("point inside polygon") {
        auto pt = tf::make_point(real_t(2), real_t(2));
        auto result = tf::classify(pt, poly);
        REQUIRE(result == tf::containment::inside);
    }

    SECTION("point outside polygon") {
        auto pt = tf::make_point(real_t(10), real_t(10));
        auto result = tf::classify(pt, poly);
        REQUIRE(result == tf::containment::outside);
    }

    SECTION("point on polygon edge") {
        auto pt = tf::make_point(real_t(2), real_t(0));
        auto result = tf::classify(pt, poly);
        REQUIRE(result == tf::containment::on_boundary);
    }

    SECTION("point at polygon vertex") {
        auto pt = tf::make_point(real_t(0), real_t(0));
        auto result = tf::classify(pt, poly);
        REQUIRE(result == tf::containment::on_boundary);
    }

    SECTION("point just outside polygon") {
        auto pt = tf::make_point(real_t(-0.1), real_t(2));
        auto result = tf::classify(pt, poly);
        REQUIRE(result == tf::containment::outside);
    }
}

TEMPLATE_TEST_CASE("classify_point_polygon_2d_triangle", "[core][classify]",
    float, double)
{
    using real_t = TestType;

    // Triangle: (0,0), (4,0), (2,4)
    std::array<tf::point<real_t, 2>, 3> triangle_pts = {{
        tf::make_point(real_t(0), real_t(0)),
        tf::make_point(real_t(4), real_t(0)),
        tf::make_point(real_t(2), real_t(4))
    }};
    auto poly = tf::make_polygon(triangle_pts);

    SECTION("point inside triangle") {
        auto pt = tf::make_point(real_t(2), real_t(1));
        auto result = tf::classify(pt, poly);
        REQUIRE(result == tf::containment::inside);
    }

    SECTION("point outside triangle") {
        auto pt = tf::make_point(real_t(0), real_t(4));
        auto result = tf::classify(pt, poly);
        REQUIRE(result == tf::containment::outside);
    }

    SECTION("point on triangle edge") {
        auto pt = tf::make_point(real_t(1), real_t(0));
        auto result = tf::classify(pt, poly);
        REQUIRE(result == tf::containment::on_boundary);
    }

    SECTION("point at centroid") {
        auto pt = tf::make_point(real_t(2), real_t(4) / real_t(3));
        auto result = tf::classify(pt, poly);
        REQUIRE(result == tf::containment::inside);
    }
}

TEMPLATE_TEST_CASE("classify_point_polygon_2d_concave", "[core][classify]",
    float, double)
{
    using real_t = TestType;

    // L-shaped concave polygon
    std::array<tf::point<real_t, 2>, 6> l_pts = {{
        tf::make_point(real_t(0), real_t(0)),
        tf::make_point(real_t(2), real_t(0)),
        tf::make_point(real_t(2), real_t(2)),
        tf::make_point(real_t(4), real_t(2)),
        tf::make_point(real_t(4), real_t(4)),
        tf::make_point(real_t(0), real_t(4))
    }};
    auto poly = tf::make_polygon(l_pts);

    SECTION("point inside L-shape") {
        auto pt = tf::make_point(real_t(1), real_t(3));
        auto result = tf::classify(pt, poly);
        REQUIRE(result == tf::containment::inside);
    }

    SECTION("point in concave region (outside)") {
        auto pt = tf::make_point(real_t(3), real_t(1));
        auto result = tf::classify(pt, poly);
        REQUIRE(result == tf::containment::outside);
    }

    SECTION("point in lower left corner (inside)") {
        auto pt = tf::make_point(real_t(1), real_t(1));
        auto result = tf::classify(pt, poly);
        REQUIRE(result == tf::containment::inside);
    }
}

// =============================================================================
// Point-Polygon classification tests (3D) - returns containment
// =============================================================================

TEMPLATE_TEST_CASE("classify_point_polygon_3d", "[core][classify]",
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

    SECTION("point inside polygon (on plane)") {
        auto pt = tf::make_point(real_t(2), real_t(1), real_t(0));
        auto result = tf::classify(pt, poly);
        REQUIRE(result == tf::containment::inside);
    }

    SECTION("point outside polygon (on plane)") {
        auto pt = tf::make_point(real_t(0), real_t(4), real_t(0));
        auto result = tf::classify(pt, poly);
        REQUIRE(result == tf::containment::outside);
    }

    SECTION("point on polygon edge") {
        auto pt = tf::make_point(real_t(2), real_t(0), real_t(0));
        auto result = tf::classify(pt, poly);
        REQUIRE(result == tf::containment::on_boundary);
    }

    SECTION("point above polygon (off plane)") {
        auto pt = tf::make_point(real_t(2), real_t(1), real_t(5));
        auto result = tf::classify(pt, poly);
        REQUIRE(result == tf::containment::outside);
    }

    SECTION("point below polygon (off plane)") {
        auto pt = tf::make_point(real_t(2), real_t(1), real_t(-5));
        auto result = tf::classify(pt, poly);
        REQUIRE(result == tf::containment::outside);
    }
}

TEMPLATE_TEST_CASE("classify_point_polygon_3d_tilted", "[core][classify]",
    float, double)
{
    using real_t = TestType;

    // Triangle in tilted plane
    std::array<tf::point<real_t, 3>, 3> triangle_pts = {{
        tf::make_point(real_t(0), real_t(0), real_t(0)),
        tf::make_point(real_t(4), real_t(0), real_t(0)),
        tf::make_point(real_t(2), real_t(2), real_t(2))
    }};
    auto poly = tf::make_polygon(triangle_pts);

    SECTION("point inside tilted polygon") {
        // Centroid of the triangle
        auto pt = tf::make_point(real_t(2), real_t(2) / real_t(3), real_t(2) / real_t(3));
        auto result = tf::classify(pt, poly);
        REQUIRE(result == tf::containment::inside);
    }

    SECTION("point on tilted polygon edge") {
        auto pt = tf::make_point(real_t(2), real_t(0), real_t(0));
        auto result = tf::classify(pt, poly);
        REQUIRE(result == tf::containment::on_boundary);
    }
}

TEMPLATE_TEST_CASE("classify_point_polygon_3d_quad", "[core][classify]",
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

    SECTION("point inside quad") {
        auto pt = tf::make_point(real_t(2), real_t(2), real_t(0));
        auto result = tf::classify(pt, poly);
        REQUIRE(result == tf::containment::inside);
    }

    SECTION("point outside quad") {
        auto pt = tf::make_point(real_t(5), real_t(2), real_t(0));
        auto result = tf::classify(pt, poly);
        REQUIRE(result == tf::containment::outside);
    }

    SECTION("point at quad vertex") {
        auto pt = tf::make_point(real_t(4), real_t(4), real_t(0));
        auto result = tf::classify(pt, poly);
        REQUIRE(result == tf::containment::on_boundary);
    }
}

// =============================================================================
// Edge cases
// =============================================================================

TEMPLATE_TEST_CASE("classify_edge_cases", "[core][classify]",
    float, double)
{
    using real_t = TestType;

    SECTION("degenerate segment (zero length)") {
        auto seg = tf::make_segment_between_points(
            tf::make_point(real_t(2), real_t(2)),
            tf::make_point(real_t(2), real_t(2))
        );
        auto pt = tf::make_point(real_t(2), real_t(2));
        auto result = tf::classify(pt, seg);
        REQUIRE(result == tf::sidedness::on_boundary);
    }

    SECTION("point at origin with plane through origin") {
        auto plane = tf::make_plane(
            tf::make_unit_vector(real_t(0), real_t(1), real_t(0)),
            real_t(0)
        );
        auto pt = tf::make_point(real_t(0), real_t(0), real_t(0));
        auto result = tf::classify(pt, plane);
        REQUIRE(result == tf::sidedness::on_boundary);
    }
}
