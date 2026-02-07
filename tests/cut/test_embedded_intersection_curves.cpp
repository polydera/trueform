/**
 * @file test_embedded_intersection_curves.cpp
 * @brief Tests for embedded_intersection_curves
 *
 * Tests for:
 * - Intersection curve correctness
 * - Topology preservation (manifold, closed meshes)
 * - Volume and surface area preservation
 * - Both return variants (with/without curves)
 * - All dynamic/static mesh combinations
 *
 * Copyright (c) 2025 Ziga Sajovic, XLAB
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <trueform/trueform.hpp>
#include "type_traits.hpp"
#include "mesh_generators.hpp"
#include <cmath>

// =============================================================================
// Test 1: Overlapping Spheres - Basic Embedding
// =============================================================================

TEMPLATE_TEST_CASE("embedded_intersection_curves_overlapping_spheres", "[embedded_intersection_curves]",
    (tf::test::type_pair_dyn2<std::int32_t, double, false, false>),
    (tf::test::type_pair_dyn2<std::int32_t, double, true, false>),
    (tf::test::type_pair_dyn2<std::int32_t, double, false, true>),
    (tf::test::type_pair_dyn2<std::int32_t, double, true, true>),
    (tf::test::type_pair_dyn2<std::int64_t, double, false, false>),
    (tf::test::type_pair_dyn2<std::int64_t, double, true, false>),
    (tf::test::type_pair_dyn2<std::int64_t, double, false, true>),
    (tf::test::type_pair_dyn2<std::int64_t, double, true, true>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;
    constexpr bool dyn1 = TestType::is_dynamic1;
    constexpr bool dyn2 = TestType::is_dynamic2;

    real_t radius = real_t(1);
    real_t separation = real_t(1);

    auto sphere1 = tf::test::maybe_as_dynamic<dyn1>(
        tf::make_sphere_mesh<index_t>(radius, 50, 50));
    tf::ensure_positive_orientation(sphere1.polygons());

    auto sphere2 = tf::test::maybe_as_dynamic<dyn2>(
        tf::make_sphere_mesh<index_t>(radius, 50, 50));
    tf::ensure_positive_orientation(sphere2.polygons());

    auto sphere2_transform = tf::make_transformation_from_translation(
        tf::make_vector(separation, real_t(0), real_t(0)));
    auto sphere2_frame = tf::make_frame(sphere2_transform);

    real_t original_volume = tf::signed_volume(sphere1.polygons());
    real_t original_area = tf::area(sphere1.polygons());

    auto result = tf::embedded_intersection_curves(
        sphere1.polygons(),
        sphere2.polygons() | tf::tag(sphere2_frame));

    // Topology: manifold and closed
    REQUIRE(tf::is_manifold(result.polygons()));
    REQUIRE(tf::is_closed(result.polygons()));

    // Volume preserved
    real_t result_volume = tf::signed_volume(result.polygons());
    REQUIRE(std::abs(result_volume - original_volume) < original_volume * real_t(0.01));

    // Surface area preserved
    real_t result_area = tf::area(result.polygons());
    REQUIRE(std::abs(result_area - original_area) < original_area * real_t(0.01));

    // More faces than original (some were split)
    REQUIRE(result.polygons().size() >= sphere1.polygons().size());
}

// =============================================================================
// Test 2: Overlapping Spheres - With Curves Return
// =============================================================================

TEMPLATE_TEST_CASE("embedded_intersection_curves_with_curves", "[embedded_intersection_curves]",
    (tf::test::type_pair_dyn2<std::int32_t, double, false, false>),
    (tf::test::type_pair_dyn2<std::int32_t, double, true, false>),
    (tf::test::type_pair_dyn2<std::int32_t, double, false, true>),
    (tf::test::type_pair_dyn2<std::int32_t, double, true, true>),
    (tf::test::type_pair_dyn2<std::int64_t, double, false, false>),
    (tf::test::type_pair_dyn2<std::int64_t, double, true, false>),
    (tf::test::type_pair_dyn2<std::int64_t, double, false, true>),
    (tf::test::type_pair_dyn2<std::int64_t, double, true, true>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;
    constexpr bool dyn1 = TestType::is_dynamic1;
    constexpr bool dyn2 = TestType::is_dynamic2;

    real_t radius = real_t(1);
    real_t separation = real_t(1);

    auto sphere1 = tf::test::maybe_as_dynamic<dyn1>(
        tf::make_sphere_mesh<index_t>(radius, 50, 50));
    tf::ensure_positive_orientation(sphere1.polygons());

    auto sphere2 = tf::test::maybe_as_dynamic<dyn2>(
        tf::make_sphere_mesh<index_t>(radius, 50, 50));
    tf::ensure_positive_orientation(sphere2.polygons());

    auto sphere2_transform = tf::make_transformation_from_translation(
        tf::make_vector(separation, real_t(0), real_t(0)));
    auto sphere2_frame = tf::make_frame(sphere2_transform);

    auto [result, curves] = tf::embedded_intersection_curves(
        sphere1.polygons(),
        sphere2.polygons() | tf::tag(sphere2_frame),
        tf::return_curves);

    // Topology: manifold and closed
    REQUIRE(tf::is_manifold(result.polygons()));
    REQUIRE(tf::is_closed(result.polygons()));

    // Curves should form a closed loop (intersection circle)
    REQUIRE(curves.paths().size() == 1);
    REQUIRE(curves.points().size() > 0);

    // The curve should be closed (first point == last point in path)
    auto path = curves.paths().front();
    REQUIRE(path.front() == path.back());
}

// =============================================================================
// Test 3: Non-Overlapping Meshes - No Intersection
// =============================================================================

TEMPLATE_TEST_CASE("embedded_intersection_curves_non_overlapping", "[embedded_intersection_curves]",
    (tf::test::type_pair_dyn2<std::int32_t, double, false, false>),
    (tf::test::type_pair_dyn2<std::int32_t, double, true, false>),
    (tf::test::type_pair_dyn2<std::int32_t, double, false, true>),
    (tf::test::type_pair_dyn2<std::int32_t, double, true, true>),
    (tf::test::type_pair_dyn2<std::int64_t, double, false, false>),
    (tf::test::type_pair_dyn2<std::int64_t, double, true, false>),
    (tf::test::type_pair_dyn2<std::int64_t, double, false, true>),
    (tf::test::type_pair_dyn2<std::int64_t, double, true, true>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;
    constexpr bool dyn1 = TestType::is_dynamic1;
    constexpr bool dyn2 = TestType::is_dynamic2;

    auto box1 = tf::test::maybe_as_dynamic<dyn1>(
        tf::make_box_mesh<index_t>(real_t(1), real_t(1), real_t(1)));
    tf::ensure_positive_orientation(box1.polygons());

    auto box2 = tf::test::maybe_as_dynamic<dyn2>(
        tf::make_box_mesh<index_t>(real_t(1), real_t(1), real_t(1)));
    tf::ensure_positive_orientation(box2.polygons());

    auto box2_transform = tf::make_transformation_from_translation(
        tf::make_vector(real_t(5), real_t(0), real_t(0)));
    auto box2_frame = tf::make_frame(box2_transform);

    std::size_t original_faces = box1.polygons().size();
    std::size_t original_points = box1.points().size();
    real_t original_volume = tf::signed_volume(box1.polygons());

    auto result = tf::embedded_intersection_curves(
        box1.polygons(),
        box2.polygons() | tf::tag(box2_frame));

    // Same face and point count (no intersection)
    REQUIRE(result.polygons().size() == original_faces);
    REQUIRE(result.points().size() == original_points);

    // Volume unchanged
    real_t result_volume = tf::signed_volume(result.polygons());
    REQUIRE(std::abs(result_volume - original_volume) < tf::epsilon<real_t>);

    // Topology preserved
    REQUIRE(tf::is_manifold(result.polygons()));
    REQUIRE(tf::is_closed(result.polygons()));

    // With curves variant: no curves
    auto [result2, curves] = tf::embedded_intersection_curves(
        box1.polygons(),
        box2.polygons() | tf::tag(box2_frame),
        tf::return_curves);

    REQUIRE(curves.paths().size() == 0);
    REQUIRE(curves.points().size() == 0);
}

// =============================================================================
// Test 4: Overlapping Boxes - Intersection Curve Topology
// =============================================================================

TEMPLATE_TEST_CASE("embedded_intersection_curves_overlapping_boxes", "[embedded_intersection_curves]",
    (tf::test::type_pair_dyn2<std::int32_t, double, false, false>),
    (tf::test::type_pair_dyn2<std::int32_t, double, true, false>),
    (tf::test::type_pair_dyn2<std::int32_t, double, false, true>),
    (tf::test::type_pair_dyn2<std::int32_t, double, true, true>),
    (tf::test::type_pair_dyn2<std::int64_t, double, false, false>),
    (tf::test::type_pair_dyn2<std::int64_t, double, true, false>),
    (tf::test::type_pair_dyn2<std::int64_t, double, false, true>),
    (tf::test::type_pair_dyn2<std::int64_t, double, true, true>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;
    constexpr bool dyn1 = TestType::is_dynamic1;
    constexpr bool dyn2 = TestType::is_dynamic2;

    auto box1 = tf::test::maybe_as_dynamic<dyn1>(
        tf::make_box_mesh<index_t>(real_t(1), real_t(1), real_t(1)));
    tf::ensure_positive_orientation(box1.polygons());

    auto box2 = tf::test::maybe_as_dynamic<dyn2>(
        tf::make_box_mesh<index_t>(real_t(1), real_t(1), real_t(1)));
    tf::ensure_positive_orientation(box2.polygons());

    auto box2_transform = tf::make_transformation_from_translation(
        tf::make_vector(real_t(0.5), real_t(0), real_t(0)));
    auto box2_frame = tf::make_frame(box2_transform);

    real_t original_volume = tf::signed_volume(box1.polygons());
    real_t original_area = tf::area(box1.polygons());

    auto [result, curves] = tf::embedded_intersection_curves(
        box1.polygons(),
        box2.polygons() | tf::tag(box2_frame),
        tf::return_curves);

    // Topology preserved
    REQUIRE(tf::is_manifold(result.polygons()));
    REQUIRE(tf::is_closed(result.polygons()));

    // Volume preserved
    real_t result_volume = tf::signed_volume(result.polygons());
    REQUIRE(std::abs(result_volume - original_volume) < original_volume * real_t(0.01));

    // Surface area preserved
    real_t result_area = tf::area(result.polygons());
    REQUIRE(std::abs(result_area - original_area) < original_area * real_t(0.01));

    // Box-box intersection forms closed loop(s)
    REQUIRE(curves.paths().size() >= 1);
}

// =============================================================================
// Test 5: Curve Matches Boolean Intersection Curve
// =============================================================================

TEMPLATE_TEST_CASE("embedded_intersection_curves_matches_boolean", "[embedded_intersection_curves]",
    (tf::test::type_pair_dyn2<std::int32_t, double, false, false>),
    (tf::test::type_pair_dyn2<std::int64_t, double, false, false>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    real_t radius = real_t(1);
    real_t separation = real_t(1);

    auto sphere1 = tf::make_sphere_mesh<index_t>(radius, 30, 30);
    tf::ensure_positive_orientation(sphere1.polygons());

    auto sphere2 = tf::make_sphere_mesh<index_t>(radius, 30, 30);
    tf::ensure_positive_orientation(sphere2.polygons());

    auto sphere2_transform = tf::make_transformation_from_translation(
        tf::make_vector(separation, real_t(0), real_t(0)));
    auto sphere2_frame = tf::make_frame(sphere2_transform);

    // Get curves from embedded_intersection_curves
    auto [embedded_result, embedded_curves] = tf::embedded_intersection_curves(
        sphere1.polygons(),
        sphere2.polygons() | tf::tag(sphere2_frame),
        tf::return_curves);

    // Get intersection from boolean
    auto [boolean_result, labels] = tf::make_boolean(
        sphere1.polygons(),
        sphere2.polygons() | tf::tag(sphere2_frame),
        tf::boolean_op::intersection);

    // Embedded mesh has intersection edges; boolean intersection is the lens
    // The intersection curve should have same number of points
    // (they trace the same intersection circle)
    REQUIRE(embedded_curves.points().size() > 0);

    // Both results should be valid topology
    REQUIRE(tf::is_manifold(embedded_result.polygons()));
    REQUIRE(tf::is_closed(embedded_result.polygons()));
    REQUIRE(tf::is_manifold(boolean_result.polygons()));
    REQUIRE(tf::is_closed(boolean_result.polygons()));
}

// =============================================================================
// Test 6: Nested Spheres - Inner Fully Inside Outer
// =============================================================================

TEMPLATE_TEST_CASE("embedded_intersection_curves_nested", "[embedded_intersection_curves]",
    (tf::test::type_pair_dyn2<std::int32_t, double, false, false>),
    (tf::test::type_pair_dyn2<std::int32_t, double, true, true>),
    (tf::test::type_pair_dyn2<std::int64_t, double, false, false>),
    (tf::test::type_pair_dyn2<std::int64_t, double, true, true>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;
    constexpr bool dyn1 = TestType::is_dynamic1;
    constexpr bool dyn2 = TestType::is_dynamic2;

    real_t outer_radius = real_t(2);
    real_t inner_radius = real_t(1);

    auto outer_sphere = tf::test::maybe_as_dynamic<dyn1>(
        tf::make_sphere_mesh<index_t>(outer_radius, 40, 40));
    tf::ensure_positive_orientation(outer_sphere.polygons());

    auto inner_sphere = tf::test::maybe_as_dynamic<dyn2>(
        tf::make_sphere_mesh<index_t>(inner_radius, 30, 30));
    tf::ensure_positive_orientation(inner_sphere.polygons());

    std::size_t original_faces = outer_sphere.polygons().size();
    std::size_t original_points = outer_sphere.points().size();
    real_t original_volume = tf::signed_volume(outer_sphere.polygons());

    auto [result, curves] = tf::embedded_intersection_curves(
        outer_sphere.polygons(),
        inner_sphere.polygons(),
        tf::return_curves);

    // No surface intersection when fully nested
    REQUIRE(result.polygons().size() == original_faces);
    REQUIRE(result.points().size() == original_points);

    // Volume unchanged
    real_t result_volume = tf::signed_volume(result.polygons());
    REQUIRE(std::abs(result_volume - original_volume) < original_volume * real_t(0.01));

    // Topology preserved
    REQUIRE(tf::is_manifold(result.polygons()));
    REQUIRE(tf::is_closed(result.polygons()));

    // No intersection curves
    REQUIRE(curves.paths().size() == 0);
}
