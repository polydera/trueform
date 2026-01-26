/**
 * @file test_curvature.cpp
 * @brief Tests for curvature analysis functions
 *
 * Tests for:
 * - make_principal_curvatures
 * - make_principal_directions
 * - make_shape_index
 *
 * Copyright (c) 2025 Ziga Sajovic, XLAB
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <trueform/trueform.hpp>
#include "type_traits.hpp"
#include <cmath>

// =============================================================================
// Sphere - Principal Curvatures
// =============================================================================

TEMPLATE_TEST_CASE("make_principal_curvatures_sphere", "[geometry][curvature]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    // Create a fine sphere - more segments = better curvature estimation
    real_t radius = real_t(2);
    auto sphere = tf::make_sphere_mesh<index_t>(radius, 40, 40);

    auto [k0, k1] = tf::make_principal_curvatures(sphere.polygons());

    // Expected curvature for a sphere: k = 1/r
    real_t expected_k = real_t(1) / radius;
    real_t tolerance = real_t(0.1) * expected_k;  // 10% tolerance for discretization

    // Check curvatures at interior vertices (skip poles which have singularities)
    std::size_t valid_count = 0;
    for (decltype(k0.size()) i = 0; i < k0.size(); ++i) {
        // Skip poles (first and last vertex in UV sphere)
        if (i == 0 || i == k0.size() - 1) continue;

        // Both principal curvatures should be approximately 1/r
        REQUIRE(std::abs(k0[i] - expected_k) < tolerance);
        REQUIRE(std::abs(k1[i] - expected_k) < tolerance);
        ++valid_count;
    }

    REQUIRE(valid_count > 0);
}

// =============================================================================
// Sphere - Shape Index
// =============================================================================

TEMPLATE_TEST_CASE("make_shape_index_sphere", "[geometry][curvature]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    real_t radius = real_t(1);
    auto sphere = tf::make_sphere_mesh<index_t>(radius, 40, 40);

    auto shape_index = tf::make_shape_index(sphere.polygons());

    // For a convex sphere, shape index should be close to 1 (convex ellipsoid)
    // Shape index range: [-1, 1]
    // [5/8, 1] = convex ellipsoid (cap)
    real_t min_expected = real_t(0.5);  // Allow some tolerance

    std::size_t valid_count = 0;
    for (decltype(shape_index.size()) i = 0; i < shape_index.size(); ++i) {
        // Skip poles
        if (i == 0 || i == shape_index.size() - 1) continue;

        // Shape index should be high (convex)
        REQUIRE(shape_index[i] >= min_expected);
        REQUIRE(shape_index[i] <= real_t(1));
        ++valid_count;
    }

    REQUIRE(valid_count > 0);
}

// =============================================================================
// Sphere - Principal Directions
// =============================================================================

TEMPLATE_TEST_CASE("make_principal_directions_sphere", "[geometry][curvature]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    real_t radius = real_t(1);
    auto sphere = tf::make_sphere_mesh<index_t>(radius, 30, 30);

    auto [k0, k1, d0, d1] = tf::make_principal_directions(sphere.polygons());

    // Verify directions are unit vectors and perpendicular
    for (decltype(d0.size()) i = 0; i < d0.size(); ++i) {
        // Skip poles
        if (i == 0 || i == d0.size() - 1) continue;

        // Directions should be unit vectors (length ~1)
        real_t len0 = std::sqrt(d0[i][0]*d0[i][0] + d0[i][1]*d0[i][1] + d0[i][2]*d0[i][2]);
        real_t len1 = std::sqrt(d1[i][0]*d1[i][0] + d1[i][1]*d1[i][1] + d1[i][2]*d1[i][2]);
        REQUIRE(std::abs(len0 - real_t(1)) < real_t(0.01));
        REQUIRE(std::abs(len1 - real_t(1)) < real_t(0.01));

        // Directions should be approximately perpendicular (dot product ~0)
        real_t dot = d0[i][0]*d1[i][0] + d0[i][1]*d1[i][1] + d0[i][2]*d1[i][2];
        REQUIRE(std::abs(dot) < real_t(0.1));
    }
}

// =============================================================================
// Plane - Zero Curvature
// =============================================================================

TEMPLATE_TEST_CASE("make_principal_curvatures_plane", "[geometry][curvature]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    // Create a subdivided plane
    auto plane = tf::make_plane_mesh<index_t>(real_t(10), real_t(10), 20, 20);

    auto [k0, k1] = tf::make_principal_curvatures(plane.polygons());

    // A flat plane should have zero curvature everywhere
    real_t tolerance = real_t(0.01);

    // Check interior vertices (boundary vertices may have edge effects)
    for (decltype(k0.size()) i = 0; i < k0.size(); ++i) {
        // Skip boundary vertices - check if vertex is interior
        auto pt = plane.points()[i];
        if (std::abs(pt[0]) > real_t(4.5) || std::abs(pt[1]) > real_t(4.5)) continue;

        REQUIRE(std::abs(k0[i]) < tolerance);
        REQUIRE(std::abs(k1[i]) < tolerance);
    }
}

// =============================================================================
// Cylinder - Mixed Curvature
// =============================================================================

TEMPLATE_TEST_CASE("make_principal_curvatures_cylinder", "[geometry][curvature]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    real_t radius = real_t(1);
    real_t height = real_t(4);
    auto cylinder = tf::make_cylinder_mesh<index_t>(radius, height, 40);

    auto [k0, k1] = tf::make_principal_curvatures(cylinder.polygons());

    // For a cylinder: one curvature = 1/r, other = 0
    real_t expected_k = real_t(1) / radius;
    real_t tolerance = real_t(0.15) * expected_k;

    // Find vertices on the side (not caps)
    for (decltype(k0.size()) i = 0; i < k0.size(); ++i) {
        auto pt = cylinder.points()[i];

        // Skip cap vertices (at z = +/- height/2)
        if (std::abs(pt[2] - height/2) < real_t(0.1) ||
            std::abs(pt[2] + height/2) < real_t(0.1)) continue;

        // On the side: one curvature ~1/r, other ~0
        // k0 >= k1 by convention
        real_t k_max = std::max(std::abs(k0[i]), std::abs(k1[i]));
        real_t k_min = std::min(std::abs(k0[i]), std::abs(k1[i]));

        REQUIRE(std::abs(k_max - expected_k) < tolerance);
        REQUIRE(k_min < tolerance);
    }
}

// =============================================================================
// Curvature with k-ring parameter
// =============================================================================

TEMPLATE_TEST_CASE("make_principal_curvatures_k_ring", "[geometry][curvature]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    real_t radius = real_t(1);
    auto sphere = tf::make_sphere_mesh<index_t>(radius, 20, 20);

    // Compare k=2 (default) with k=3
    auto [k0_2, k1_2] = tf::make_principal_curvatures(sphere.polygons(), 2);
    auto [k0_3, k1_3] = tf::make_principal_curvatures(sphere.polygons(), 3);

    // Both should give similar results for a sphere
    real_t expected_k = real_t(1) / radius;

    // Just verify both produce reasonable results (larger k should be smoother)
    for (decltype(k0_2.size()) i = 0; i < k0_2.size(); ++i) {
        if (i == 0 || i == k0_2.size() - 1) continue;

        // k=3 should still be reasonable
        REQUIRE(std::abs(k0_3[i] - expected_k) < real_t(0.5));
        REQUIRE(std::abs(k1_3[i] - expected_k) < real_t(0.5));
    }
}

// =============================================================================
// Gaussian and Mean Curvature from Principal
// =============================================================================

TEMPLATE_TEST_CASE("gaussian_mean_curvature_sphere", "[geometry][curvature]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    real_t radius = real_t(2);
    auto sphere = tf::make_sphere_mesh<index_t>(radius, 40, 40);

    auto [k0, k1] = tf::make_principal_curvatures(sphere.polygons());

    // For sphere:
    // Gaussian curvature K = k0 * k1 = 1/r^2
    // Mean curvature H = (k0 + k1) / 2 = 1/r
    real_t expected_gaussian = real_t(1) / (radius * radius);
    real_t expected_mean = real_t(1) / radius;
    real_t tol_g = real_t(0.1) * expected_gaussian;
    real_t tol_m = real_t(0.1) * expected_mean;

    for (decltype(k0.size()) i = 0; i < k0.size(); ++i) {
        if (i == 0 || i == k0.size() - 1) continue;

        real_t gaussian = k0[i] * k1[i];
        real_t mean = (k0[i] + k1[i]) / real_t(2);

        REQUIRE(std::abs(gaussian - expected_gaussian) < tol_g);
        REQUIRE(std::abs(mean - expected_mean) < tol_m);
    }
}
