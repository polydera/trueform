/**
 * @file test_transformed.cpp
 * @brief Tests for transformed functionality and policy propagation
 *
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <trueform/core.hpp>
#include <cmath>

// =============================================================================
// Helper functions
// =============================================================================

template <typename real_t>
auto approx_equal(real_t a, real_t b, real_t tol = real_t(1e-5)) -> bool
{
    return std::abs(a - b) < tol;
}

template <std::size_t Dims, typename T0, typename T1, typename real_t>
auto vectors_approx_equal(const T0& a, const T1& b, real_t tol = real_t(1e-5)) -> bool
{
    for (std::size_t i = 0; i < Dims; ++i) {
        if (!approx_equal(a[i], b[i], tol))
            return false;
    }
    return true;
}

// =============================================================================
// Basic primitive transformations
// =============================================================================

TEMPLATE_TEST_CASE("transformed_point", "[core][transformed]",
    float, double)
{
    using real_t = TestType;

    auto pt = tf::make_point(real_t(1), real_t(0), real_t(0));

    SECTION("rotation 90 degrees around z-axis") {
        auto rotation = tf::make_rotation(tf::deg(real_t(90)), tf::axis<2>);
        auto frame = tf::make_frame(rotation);
        auto result = tf::transformed(pt, frame);

        REQUIRE(approx_equal(result[0], real_t(0)));
        REQUIRE(approx_equal(result[1], real_t(1)));
        REQUIRE(approx_equal(result[2], real_t(0)));
    }

    SECTION("translation") {
        auto translation = tf::make_transformation_from_translation(
            tf::make_vector(real_t(1), real_t(2), real_t(3)));
        auto frame = tf::make_frame(translation);
        auto result = tf::transformed(pt, frame);

        REQUIRE(approx_equal(result[0], real_t(2)));
        REQUIRE(approx_equal(result[1], real_t(2)));
        REQUIRE(approx_equal(result[2], real_t(3)));
    }

    SECTION("identity transformation is no-op") {
        auto identity = tf::identity_frame<real_t, 3>{};
        auto result = tf::transformed(pt, identity);

        REQUIRE(approx_equal(result[0], pt[0]));
        REQUIRE(approx_equal(result[1], pt[1]));
        REQUIRE(approx_equal(result[2], pt[2]));
    }
}

TEMPLATE_TEST_CASE("transformed_vector", "[core][transformed]",
    float, double)
{
    using real_t = TestType;

    auto vec = tf::make_vector(real_t(1), real_t(0), real_t(0));

    SECTION("rotation 90 degrees around z-axis") {
        auto rotation = tf::make_rotation(tf::deg(real_t(90)), tf::axis<2>);
        auto frame = tf::make_frame(rotation);
        auto result = tf::transformed(vec, frame);

        REQUIRE(approx_equal(result[0], real_t(0)));
        REQUIRE(approx_equal(result[1], real_t(1)));
        REQUIRE(approx_equal(result[2], real_t(0)));
    }

    SECTION("translation does not affect vectors") {
        auto translation = tf::make_transformation_from_translation(
            tf::make_vector(real_t(1), real_t(2), real_t(3)));
        auto frame = tf::make_frame(translation);
        auto result = tf::transformed(vec, frame);

        // Vectors are unaffected by translation
        REQUIRE(approx_equal(result[0], real_t(1)));
        REQUIRE(approx_equal(result[1], real_t(0)));
        REQUIRE(approx_equal(result[2], real_t(0)));
    }
}

// =============================================================================
// transformed_normal - uses inverse transpose
// =============================================================================

TEMPLATE_TEST_CASE("transformed_normal_rotation", "[core][transformed]",
    float, double)
{
    using real_t = TestType;

    // Normal pointing in +X direction
    auto normal = tf::make_unit_vector(real_t(1), real_t(0), real_t(0));

    SECTION("rotation 90 degrees around z-axis") {
        auto rotation = tf::make_rotation(tf::deg(real_t(90)), tf::axis<2>);
        auto frame = tf::make_frame(rotation);
        auto result = tf::transformed_normal(normal, frame);

        // Normal should rotate the same way for pure rotation
        REQUIRE(approx_equal(result[0], real_t(0)));
        REQUIRE(approx_equal(result[1], real_t(1)));
        REQUIRE(approx_equal(result[2], real_t(0)));
    }

    SECTION("identity transformation is no-op") {
        auto identity = tf::identity_frame<real_t, 3>{};
        auto result = tf::transformed_normal(normal, identity);

        REQUIRE(approx_equal(result[0], normal[0]));
        REQUIRE(approx_equal(result[1], normal[1]));
        REQUIRE(approx_equal(result[2], normal[2]));
    }
}

TEMPLATE_TEST_CASE("transformed_normal_stays_perpendicular", "[core][transformed]",
    float, double)
{
    using real_t = TestType;

    // Create a surface normal and a tangent vector
    // Normal pointing in +Z
    auto normal = tf::make_unit_vector(real_t(0), real_t(0), real_t(1));
    // Tangent in XY plane
    auto tangent = tf::make_vector(real_t(1), real_t(1), real_t(0));

    SECTION("after rotation, normal and tangent stay perpendicular") {
        auto rotation = tf::make_rotation(tf::deg(real_t(45)), tf::axis<0>);
        auto frame = tf::make_frame(rotation);

        auto transformed_n = tf::transformed_normal(normal, frame);
        auto transformed_t = tf::transformed(tangent, frame);

        // Dot product should be ~0 (perpendicular)
        auto dot = tf::dot(transformed_n, transformed_t);
        REQUIRE(approx_equal(dot, real_t(0)));
    }
}

// =============================================================================
// Policy propagation through transformed
// =============================================================================

TEMPLATE_TEST_CASE("transformed_tag_id_preserved", "[core][transformed][policy]",
    float, double)
{
    using real_t = TestType;

    auto pt = tf::make_point(real_t(1), real_t(0), real_t(0))
            | tf::tag_id(42);

    auto rotation = tf::make_rotation(tf::deg(real_t(90)), tf::axis<2>);
    auto frame = tf::make_frame(rotation);
    auto result = tf::transformed(pt, frame);

    // ID should be preserved (copied, not transformed)
    REQUIRE(result.id() == 42);

    // Point should be transformed
    REQUIRE(approx_equal(result[0], real_t(0)));
    REQUIRE(approx_equal(result[1], real_t(1)));
}

TEMPLATE_TEST_CASE("transformed_tag_normal_uses_inverse_transpose", "[core][transformed][policy]",
    float, double)
{
    using real_t = TestType;

    // Point with tagged normal
    auto normal = tf::make_unit_vector(real_t(1), real_t(0), real_t(0));
    auto pt = tf::make_point(real_t(0), real_t(0), real_t(0))
            | tf::tag_normal(normal);

    SECTION("rotation transforms normal correctly") {
        auto rotation = tf::make_rotation(tf::deg(real_t(90)), tf::axis<2>);
        auto frame = tf::make_frame(rotation);
        auto result = tf::transformed(pt, frame);

        auto result_normal = result.normal();

        // Normal should be rotated
        REQUIRE(approx_equal(result_normal[0], real_t(0)));
        REQUIRE(approx_equal(result_normal[1], real_t(1)));
        REQUIRE(approx_equal(result_normal[2], real_t(0)));
    }
}

TEMPLATE_TEST_CASE("transformed_tag_plane_correct", "[core][transformed][policy]",
    float, double)
{
    using real_t = TestType;

    // Create a plane and tag a point with it
    auto plane = tf::make_plane(
        tf::make_unit_vector(real_t(0), real_t(0), real_t(1)),  // normal in +Z
        real_t(0)  // d = 0, plane at origin
    );
    auto pt = tf::make_point(real_t(0), real_t(0), real_t(0))
            | tf::tag_plane(plane);

    SECTION("rotation transforms plane normal") {
        // Rotate 90 degrees around X axis
        auto rotation = tf::make_rotation(tf::deg(real_t(90)), tf::axis<0>);
        auto frame = tf::make_frame(rotation);
        auto result = tf::transformed(pt, frame);

        auto result_plane = result.plane();

        // Normal (0,0,1) rotated 90° around X -> (0,-1,0)
        REQUIRE(approx_equal(result_plane.normal[0], real_t(0)));
        REQUIRE(approx_equal(result_plane.normal[1], real_t(-1)));
        REQUIRE(approx_equal(result_plane.normal[2], real_t(0)));
    }

    SECTION("translation affects plane d") {
        // Translate in +Z direction
        auto translation = tf::make_transformation_from_translation(
            tf::make_vector(real_t(0), real_t(0), real_t(5)));
        auto frame = tf::make_frame(translation);
        auto result = tf::transformed(pt, frame);

        auto result_plane = result.plane();

        // Normal unchanged
        REQUIRE(approx_equal(result_plane.normal[0], real_t(0)));
        REQUIRE(approx_equal(result_plane.normal[1], real_t(0)));
        REQUIRE(approx_equal(result_plane.normal[2], real_t(1)));

        // d should change: original d=0, translated by 5 in normal direction
        // new d = old_d - dot(normal, translation) = 0 - 5 = -5
        REQUIRE(approx_equal(result_plane.d, real_t(-5)));
    }
}

TEMPLATE_TEST_CASE("transformed_tag_state_with_geometry", "[core][transformed][policy]",
    float, double)
{
    using real_t = TestType;

    // Point with state containing: a color (non-geometric), a vector, and a point
    std::array<real_t, 3> color{real_t(1), real_t(0), real_t(0)};
    auto state_vec = tf::make_vector(real_t(1), real_t(0), real_t(0));
    auto state_pt = tf::make_point(real_t(2), real_t(0), real_t(0));

    auto pt = tf::make_point(real_t(1), real_t(0), real_t(0))
            | tf::tag_state(color, state_vec, state_pt);

    auto rotation = tf::make_rotation(tf::deg(real_t(90)), tf::axis<2>);
    auto frame = tf::make_frame(rotation);
    auto result = tf::transformed(pt, frame);

    const auto& [result_color, result_vec, result_pt] = result.state();

    // Color (non-geometric std::array) should be preserved unchanged
    REQUIRE(approx_equal(result_color[0], real_t(1)));
    REQUIRE(approx_equal(result_color[1], real_t(0)));
    REQUIRE(approx_equal(result_color[2], real_t(0)));

    // Vector in state should be transformed (rotated)
    REQUIRE(approx_equal(result_vec[0], real_t(0)));
    REQUIRE(approx_equal(result_vec[1], real_t(1)));
    REQUIRE(approx_equal(result_vec[2], real_t(0)));

    // Point in state should be transformed (rotated)
    REQUIRE(approx_equal(result_pt[0], real_t(0)));
    REQUIRE(approx_equal(result_pt[1], real_t(2)));
    REQUIRE(approx_equal(result_pt[2], real_t(0)));

    // Outer point should be transformed
    REQUIRE(approx_equal(result[0], real_t(0)));
    REQUIRE(approx_equal(result[1], real_t(1)));
}

TEMPLATE_TEST_CASE("transformed_multiple_policies", "[core][transformed][policy]",
    float, double)
{
    using real_t = TestType;

    // Point with multiple policies
    auto normal = tf::make_unit_vector(real_t(1), real_t(0), real_t(0));
    auto pt = tf::make_point(real_t(1), real_t(0), real_t(0))
            | tf::tag_id(123)
            | tf::tag_normal(normal);

    auto rotation = tf::make_rotation(tf::deg(real_t(90)), tf::axis<2>);
    auto frame = tf::make_frame(rotation);
    auto result = tf::transformed(pt, frame);

    // ID preserved
    REQUIRE(result.id() == 123);

    // Normal transformed
    auto result_normal = result.normal();
    REQUIRE(approx_equal(result_normal[0], real_t(0)));
    REQUIRE(approx_equal(result_normal[1], real_t(1)));

    // Point transformed
    REQUIRE(approx_equal(result[0], real_t(0)));
    REQUIRE(approx_equal(result[1], real_t(1)));
}

TEMPLATE_TEST_CASE("transformed_deeply_nested_policies", "[core][transformed][policy]",
    float, double)
{
    using real_t = TestType;

    // Create a point with a normal (this will be in state)
    auto inner_normal = tf::make_unit_vector(real_t(0), real_t(0), real_t(1));
    auto inner_pt = tf::make_point(real_t(3), real_t(0), real_t(0))
                  | tf::tag_normal(inner_normal);

    // Outer point with: id, state containing (int, inner_point_with_normal)
    auto outer_pt = tf::make_point(real_t(1), real_t(0), real_t(0))
                  | tf::tag_id(999)
                  | tf::tag_state(42, inner_pt);

    auto rotation = tf::make_rotation(tf::deg(real_t(90)), tf::axis<2>);
    auto frame = tf::make_frame(rotation);
    auto result = tf::transformed(outer_pt, frame);

    // ID preserved
    REQUIRE(result.id() == 999);

    const auto& [result_int, result_inner_pt] = result.state();

    // Int in state preserved
    REQUIRE(result_int == 42);

    // Inner point transformed (rotated from (3,0,0) to (0,3,0))
    REQUIRE(approx_equal(result_inner_pt[0], real_t(0)));
    REQUIRE(approx_equal(result_inner_pt[1], real_t(3)));
    REQUIRE(approx_equal(result_inner_pt[2], real_t(0)));

    // Inner point's normal transformed (was (0,0,1), rotation around z doesn't change it)
    auto result_inner_normal = result_inner_pt.normal();
    REQUIRE(approx_equal(result_inner_normal[0], real_t(0)));
    REQUIRE(approx_equal(result_inner_normal[1], real_t(0)));
    REQUIRE(approx_equal(result_inner_normal[2], real_t(1)));

    // Outer point transformed (rotated from (1,0,0) to (0,1,0))
    REQUIRE(approx_equal(result[0], real_t(0)));
    REQUIRE(approx_equal(result[1], real_t(1)));
    REQUIRE(approx_equal(result[2], real_t(0)));
}

TEMPLATE_TEST_CASE("transformed_nested_normal_rotates", "[core][transformed][policy]",
    float, double)
{
    using real_t = TestType;

    // Inner point with normal pointing in +X
    auto inner_normal = tf::make_unit_vector(real_t(1), real_t(0), real_t(0));
    auto inner_pt = tf::make_point(real_t(0), real_t(0), real_t(0))
                  | tf::tag_normal(inner_normal);

    // Outer point with state containing the inner point
    auto outer_pt = tf::make_point(real_t(0), real_t(0), real_t(0))
                  | tf::tag_state(inner_pt);

    // Rotate 90 degrees around Z - normal should go from +X to +Y
    auto rotation = tf::make_rotation(tf::deg(real_t(90)), tf::axis<2>);
    auto frame = tf::make_frame(rotation);
    auto result = tf::transformed(outer_pt, frame);

    const auto& result_inner_pt = result.state();

    // Inner point's normal should be rotated from (1,0,0) to (0,1,0)
    auto result_inner_normal = result_inner_pt.normal();
    REQUIRE(approx_equal(result_inner_normal[0], real_t(0)));
    REQUIRE(approx_equal(result_inner_normal[1], real_t(1)));
    REQUIRE(approx_equal(result_inner_normal[2], real_t(0)));
}

// =============================================================================
// Normal with state - state transforms with inverse transpose frame
// =============================================================================

TEMPLATE_TEST_CASE("transformed_normal_with_state_point", "[core][transformed][policy]",
    float, double)
{
    using real_t = TestType;

    // A normal with a point in its state
    // When transformed_normal is called, the point should transform
    // using the same inverse transpose frame as the normal
    auto state_pt = tf::make_point(real_t(1), real_t(0), real_t(0));
    auto normal = tf::make_unit_vector(real_t(1), real_t(0), real_t(0))
                | tf::tag_state(state_pt);

    // Rotate 90 degrees around Z
    auto rotation = tf::make_rotation(tf::deg(real_t(90)), tf::axis<2>);
    auto frame = tf::make_frame(rotation);
    auto result = tf::transformed_normal(normal, frame);

    // Normal should be rotated from (1,0,0) to (0,1,0)
    REQUIRE(approx_equal(result[0], real_t(0)));
    REQUIRE(approx_equal(result[1], real_t(1)));
    REQUIRE(approx_equal(result[2], real_t(0)));

    // Point in state should also be transformed with the inverse transpose frame
    // For pure rotation, this is the same as regular rotation
    const auto& result_pt = result.state();
    REQUIRE(approx_equal(result_pt[0], real_t(0)));
    REQUIRE(approx_equal(result_pt[1], real_t(1)));
    REQUIRE(approx_equal(result_pt[2], real_t(0)));
}

TEMPLATE_TEST_CASE("transformed_normal_with_nested_normal", "[core][transformed][policy]",
    float, double)
{
    using real_t = TestType;

    // A normal with another normal in its state
    // Both should transform using inverse transpose
    auto inner_normal = tf::make_unit_vector(real_t(0), real_t(0), real_t(1));
    auto outer_normal = tf::make_unit_vector(real_t(1), real_t(0), real_t(0))
                      | tf::tag_state(inner_normal);

    // Rotate 90 degrees around X
    // Outer normal (1,0,0) stays (1,0,0) - X axis unchanged
    // Inner normal (0,0,1) becomes (0,-1,0)
    auto rotation = tf::make_rotation(tf::deg(real_t(90)), tf::axis<0>);
    auto frame = tf::make_frame(rotation);
    auto result = tf::transformed_normal(outer_normal, frame);

    // Outer normal unchanged (rotation around X doesn't affect X direction)
    REQUIRE(approx_equal(result[0], real_t(1)));
    REQUIRE(approx_equal(result[1], real_t(0)));
    REQUIRE(approx_equal(result[2], real_t(0)));

    // Inner normal (0,0,1) rotated 90° around X -> (0,-1,0)
    const auto& result_inner = result.state();
    REQUIRE(approx_equal(result_inner[0], real_t(0)));
    REQUIRE(approx_equal(result_inner[1], real_t(-1)));
    REQUIRE(approx_equal(result_inner[2], real_t(0)));
}

// =============================================================================
// Plane transformation
// =============================================================================

TEMPLATE_TEST_CASE("transformed_plane", "[core][transformed]",
    float, double)
{
    using real_t = TestType;

    // Plane at z=1, normal pointing up
    auto plane = tf::make_plane(
        tf::make_unit_vector(real_t(0), real_t(0), real_t(1)),
        real_t(-1)  // d = -1 means plane at z=1
    );

    SECTION("rotation 90 degrees around x-axis") {
        auto rotation = tf::make_rotation(tf::deg(real_t(90)), tf::axis<0>);
        auto frame = tf::make_frame(rotation);
        auto result = tf::transformed(plane, frame);

        // Normal (0,0,1) rotated 90° around X -> (0,-1,0)
        REQUIRE(approx_equal(result.normal[0], real_t(0)));
        REQUIRE(approx_equal(result.normal[1], real_t(-1)));
        REQUIRE(approx_equal(result.normal[2], real_t(0)));
    }

    SECTION("translation in normal direction") {
        auto translation = tf::make_transformation_from_translation(
            tf::make_vector(real_t(0), real_t(0), real_t(2)));
        auto frame = tf::make_frame(translation);
        auto result = tf::transformed(plane, frame);

        // Normal unchanged
        REQUIRE(approx_equal(result.normal[0], real_t(0)));
        REQUIRE(approx_equal(result.normal[1], real_t(0)));
        REQUIRE(approx_equal(result.normal[2], real_t(1)));

        // d should change: plane moves from z=1 to z=3
        // new d = old_d - dot(normal, translation) = -1 - 2 = -3
        REQUIRE(approx_equal(result.d, real_t(-3)));
    }
}

// =============================================================================
// Segment and polygon transformation
// =============================================================================

TEMPLATE_TEST_CASE("transformed_segment", "[core][transformed]",
    float, double)
{
    using real_t = TestType;

    auto seg = tf::make_segment_between_points(
        tf::make_point(real_t(0), real_t(0), real_t(0)),
        tf::make_point(real_t(1), real_t(0), real_t(0))
    );

    auto rotation = tf::make_rotation(tf::deg(real_t(90)), tf::axis<2>);
    auto frame = tf::make_frame(rotation);
    auto result = tf::transformed(seg, frame);

    auto [p0, p1] = result;

    // First point at origin stays at origin
    REQUIRE(approx_equal(p0[0], real_t(0)));
    REQUIRE(approx_equal(p0[1], real_t(0)));

    // Second point rotated from (1,0,0) to (0,1,0)
    REQUIRE(approx_equal(p1[0], real_t(0)));
    REQUIRE(approx_equal(p1[1], real_t(1)));
}
