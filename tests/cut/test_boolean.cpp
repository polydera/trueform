/**
 * @file test_boolean.cpp
 * @brief Tests for boolean operations on meshes
 *
 * Tests for:
 * - make_boolean (merge, intersection, left_difference, right_difference)
 * - Topology preservation (manifold, closed meshes)
 * - Volume correctness
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
// Test 1: Boolean Topology - Repeated Operations Preserve Manifold Property
// =============================================================================

TEMPLATE_TEST_CASE("boolean_topology_repeated_ops", "[boolean]",
    (tf::test::type_pair<std::int32_t, double>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    // Create base sphere
    auto big_sphere = tf::make_sphere_mesh<index_t>(real_t(10), 40, 40);
    tf::ensure_positive_orientation(big_sphere.polygons());

    // Create small sphere for operation
    auto small_sphere = tf::make_sphere_mesh<index_t>(real_t(0.5), 20, 20);
    tf::ensure_positive_orientation(small_sphere.polygons());

    // Place small sphere at north pole of big sphere
    auto merge_point = big_sphere.points()[0];
    auto transform = tf::make_transformation_from_translation(merge_point.as_vector());
    auto frame = tf::make_frame(transform);

    // Test all boolean operations
    std::array<tf::boolean_op, 3> ops = {
        tf::boolean_op::merge,
        tf::boolean_op::intersection,
        tf::boolean_op::left_difference
    };

    for (auto op : ops) {
        // First boolean
        auto [current, labels] = tf::make_boolean(
            big_sphere.polygons(),
            small_sphere.polygons() | tf::tag(frame),
            op);

        auto baseline_points = current.points().size();
        auto baseline_faces = current.polygons().size();

        // Check first result is manifold and closed
        auto boundaries = tf::make_boundary_paths(current.polygons());
        auto non_manifold = tf::make_non_manifold_edges(current.polygons());

        REQUIRE(boundaries.size() == 0);
        REQUIRE(non_manifold.size() == 0);

        // Repeated boolean at same point (coplanarity test)
        for (int i = 2; i <= 4; ++i) {
            auto [next, next_labels] = tf::make_boolean(
                current.polygons(),
                small_sphere.polygons() | tf::tag(frame),
                op);

            boundaries = tf::make_boundary_paths(next.polygons());
            non_manifold = tf::make_non_manifold_edges(next.polygons());

            REQUIRE(boundaries.size() == 0);
            REQUIRE(non_manifold.size() == 0);

            // Repeated ops give same result
            REQUIRE(next.points().size() == baseline_points);
            REQUIRE(next.polygons().size() == baseline_faces);

            current = std::move(next);
        }
    }
}

// =============================================================================
// Test 2: Steinmetz Solid (Bicylinder) - Intersection of Perpendicular Cylinders
// =============================================================================

TEMPLATE_TEST_CASE("boolean_bicylinder_intersection", "[boolean]",
    (tf::test::type_pair<std::int32_t, double>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    // Create a cylinder
    real_t radius = real_t(1);
    real_t height = real_t(4);
    auto cylinder = tf::make_cylinder_mesh<index_t>(radius, height, 400);
    tf::ensure_positive_orientation(cylinder.polygons());
    auto triangulated = tf::triangulated(cylinder.polygons());

    // Rotate 90 degrees around X-axis, centered at cylinder's centroid
    auto center = tf::centroid(triangulated.polygons());
    auto horizontal_cylinder = triangulated.polygons() |
        tf::tag(tf::make_rotation(tf::deg(real_t(90)), tf::axis<0>, center));

    // Boolean intersection creates Steinmetz solid (bicylinder)
    auto [bicylinder, labels] = tf::make_boolean(
        triangulated.polygons(),
        horizontal_cylinder,
        tf::boolean_op::intersection);

    // Result is watertight manifold
    auto boundaries = tf::make_boundary_paths(bicylinder.polygons());
    auto non_manifold = tf::make_non_manifold_edges(bicylinder.polygons());

    REQUIRE(boundaries.size() == 0);
    REQUIRE(non_manifold.size() == 0);

    // Steinmetz solid volume = 16r³/3
    real_t expected_volume = real_t(16) * radius * radius * radius / real_t(3);
    real_t bicylinder_volume = tf::signed_volume(bicylinder.polygons());

    REQUIRE(std::abs(bicylinder_volume - expected_volume) <
            expected_volume * real_t(0.01));
}

// =============================================================================
// Test 3: Nested Spheres - Boolean Operations with Volume Verification
// =============================================================================

TEMPLATE_TEST_CASE("boolean_nested_spheres", "[boolean]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    // Outer sphere radius 2, inner sphere radius 1
    real_t outer_radius = real_t(2);
    real_t inner_radius = real_t(1);

    auto outer_sphere = tf::make_sphere_mesh<index_t>(outer_radius, 300, 300);
    tf::ensure_positive_orientation(outer_sphere.polygons());

    auto inner_sphere = tf::make_sphere_mesh<index_t>(inner_radius, 200, 200);
    tf::ensure_positive_orientation(inner_sphere.polygons());

    // Volume formula: V = (4/3) * pi * r^3
    constexpr real_t pi = real_t(3.14159265358979323846);
    real_t outer_volume_expected = (real_t(4) / real_t(3)) * pi * outer_radius * outer_radius * outer_radius;
    real_t inner_volume_expected = (real_t(4) / real_t(3)) * pi * inner_radius * inner_radius * inner_radius;

    // Test 3.1: Merge (union) - inner fully inside outer, result is outer sphere
    {
        auto [merged, labels] = tf::make_boolean(
            outer_sphere.polygons(),
            inner_sphere.polygons(),
            tf::boolean_op::merge);

        auto boundaries = tf::make_boundary_paths(merged.polygons());
        auto non_manifold = tf::make_non_manifold_edges(merged.polygons());

        REQUIRE(boundaries.size() == 0);
        REQUIRE(non_manifold.size() == 0);

        // Result is just the outer sphere
        REQUIRE(merged.polygons().size() == outer_sphere.polygons().size());
        REQUIRE(merged.points().size() == outer_sphere.points().size());

        // Union of nested spheres = outer sphere volume
        real_t merged_volume = tf::signed_volume(merged.polygons());
        REQUIRE(std::abs(merged_volume - outer_volume_expected) <
                outer_volume_expected * real_t(0.01));
    }

    // Test 3.2: Left difference - hollow sphere (outer minus inner)
    {
        auto [hollow, labels] = tf::make_boolean(
            outer_sphere.polygons(),
            inner_sphere.polygons(),
            tf::boolean_op::left_difference);

        auto boundaries = tf::make_boundary_paths(hollow.polygons());
        auto non_manifold = tf::make_non_manifold_edges(hollow.polygons());

        REQUIRE(boundaries.size() == 0);
        REQUIRE(non_manifold.size() == 0);

        // Result has both outer and inner surfaces
        REQUIRE(hollow.polygons().size() == outer_sphere.polygons().size() + inner_sphere.polygons().size());
        REQUIRE(hollow.points().size() == outer_sphere.points().size() + inner_sphere.points().size());

        // Volume of hollow sphere = outer - inner
        real_t hollow_volume = tf::signed_volume(hollow.polygons());
        real_t expected_hollow_volume = outer_volume_expected - inner_volume_expected;

        REQUIRE(std::abs(hollow_volume - expected_hollow_volume) <
                expected_hollow_volume * real_t(0.01));
    }

    // Test 3.3: Intersection - inner sphere is fully inside outer, result is inner
    {
        auto [intersection, labels] = tf::make_boolean(
            outer_sphere.polygons(),
            inner_sphere.polygons(),
            tf::boolean_op::intersection);

        auto boundaries = tf::make_boundary_paths(intersection.polygons());
        auto non_manifold = tf::make_non_manifold_edges(intersection.polygons());

        REQUIRE(boundaries.size() == 0);
        REQUIRE(non_manifold.size() == 0);

        // Result is just the inner sphere
        REQUIRE(intersection.polygons().size() == inner_sphere.polygons().size());
        REQUIRE(intersection.points().size() == inner_sphere.points().size());

        // Volume of intersection = inner sphere volume
        real_t intersection_volume = tf::signed_volume(intersection.polygons());

        REQUIRE(std::abs(intersection_volume - inner_volume_expected) <
                inner_volume_expected * real_t(0.01));
    }
}

// =============================================================================
// Test 4: Overlapping Boxes - All Boolean Operations
// =============================================================================

TEMPLATE_TEST_CASE("boolean_overlapping_boxes", "[boolean]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    // Two unit boxes, second translated by (0.5, 0, 0)
    auto box1 = tf::make_box_mesh<index_t>(real_t(1), real_t(1), real_t(1));
    tf::ensure_positive_orientation(box1.polygons());

    auto box2 = tf::make_box_mesh<index_t>(real_t(1), real_t(1), real_t(1));
    tf::ensure_positive_orientation(box2.polygons());

    auto box2_transform = tf::make_transformation_from_translation(
        tf::make_vector(real_t(0.5), real_t(0), real_t(0)));
    auto box2_frame = tf::make_frame(box2_transform);

    // Box volumes: each is 1 cubic unit
    real_t box_volume = real_t(1);
    // Overlap volume: 0.5 * 1 * 1 = 0.5 cubic units
    real_t overlap_volume = real_t(0.5);

    // Test merge (union)
    {
        auto [merged, labels] = tf::make_boolean(
            box1.polygons(),
            box2.polygons() | tf::tag(box2_frame),
            tf::boolean_op::merge);

        auto boundaries = tf::make_boundary_paths(merged.polygons());
        auto non_manifold = tf::make_non_manifold_edges(merged.polygons());

        REQUIRE(boundaries.size() == 0);
        REQUIRE(non_manifold.size() == 0);

        // Union volume = box1 + box2 - overlap
        real_t merged_volume = tf::signed_volume(merged.polygons());
        real_t expected = box_volume + box_volume - overlap_volume;

        REQUIRE(std::abs(merged_volume - expected) <
                std::max(tf::epsilon<real_t>, real_t(0.01)));
    }

    // Test intersection
    {
        auto [intersection, labels] = tf::make_boolean(
            box1.polygons(),
            box2.polygons() | tf::tag(box2_frame),
            tf::boolean_op::intersection);

        auto boundaries = tf::make_boundary_paths(intersection.polygons());
        auto non_manifold = tf::make_non_manifold_edges(intersection.polygons());

        REQUIRE(boundaries.size() == 0);
        REQUIRE(non_manifold.size() == 0);

        // Intersection volume = overlap
        real_t intersection_volume = tf::signed_volume(intersection.polygons());

        REQUIRE(std::abs(intersection_volume - overlap_volume) <
                std::max(tf::epsilon<real_t>, real_t(0.01)));
    }

    // Test left difference (box1 - box2)
    {
        auto [diff, labels] = tf::make_boolean(
            box1.polygons(),
            box2.polygons() | tf::tag(box2_frame),
            tf::boolean_op::left_difference);

        auto boundaries = tf::make_boundary_paths(diff.polygons());
        auto non_manifold = tf::make_non_manifold_edges(diff.polygons());

        REQUIRE(boundaries.size() == 0);
        REQUIRE(non_manifold.size() == 0);

        // Left difference volume = box1 - overlap
        real_t diff_volume = tf::signed_volume(diff.polygons());
        real_t expected = box_volume - overlap_volume;

        REQUIRE(std::abs(diff_volume - expected) <
                std::max(tf::epsilon<real_t>, real_t(0.01)));
    }

    // Test right difference (box2 - box1)
    {
        auto [diff, labels] = tf::make_boolean(
            box1.polygons(),
            box2.polygons() | tf::tag(box2_frame),
            tf::boolean_op::right_difference);

        auto boundaries = tf::make_boundary_paths(diff.polygons());
        auto non_manifold = tf::make_non_manifold_edges(diff.polygons());

        REQUIRE(boundaries.size() == 0);
        REQUIRE(non_manifold.size() == 0);

        // Right difference volume = box2 - overlap
        real_t diff_volume = tf::signed_volume(diff.polygons());
        real_t expected = box_volume - overlap_volume;

        REQUIRE(std::abs(diff_volume - expected) <
                std::max(tf::epsilon<real_t>, real_t(0.01)));
    }
}

// =============================================================================
// Test 5: Non-Overlapping Meshes
// =============================================================================

TEMPLATE_TEST_CASE("boolean_non_overlapping", "[boolean]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto box1 = tf::make_box_mesh<index_t>(real_t(1), real_t(1), real_t(1));
    tf::ensure_positive_orientation(box1.polygons());

    auto box2 = tf::make_box_mesh<index_t>(real_t(1), real_t(1), real_t(1));
    tf::ensure_positive_orientation(box2.polygons());

    // Translate box2 far away (no overlap)
    auto box2_transform = tf::make_transformation_from_translation(
        tf::make_vector(real_t(5), real_t(0), real_t(0)));
    auto box2_frame = tf::make_frame(box2_transform);

    real_t box_volume = real_t(1);

    // Merge of non-overlapping = sum of volumes
    {
        auto [merged, labels] = tf::make_boolean(
            box1.polygons(),
            box2.polygons() | tf::tag(box2_frame),
            tf::boolean_op::merge);

        real_t merged_volume = tf::signed_volume(merged.polygons());
        REQUIRE(std::abs(merged_volume - real_t(2) * box_volume) <
                std::max(tf::epsilon<real_t>, real_t(0.01)));
    }

    // Intersection of non-overlapping = empty (0 volume)
    {
        auto [intersection, labels] = tf::make_boolean(
            box1.polygons(),
            box2.polygons() | tf::tag(box2_frame),
            tf::boolean_op::intersection);

        // Empty result has 0 faces
        REQUIRE(intersection.polygons().size() == 0);
    }

    // Left difference of non-overlapping = box1 unchanged
    {
        auto [diff, labels] = tf::make_boolean(
            box1.polygons(),
            box2.polygons() | tf::tag(box2_frame),
            tf::boolean_op::left_difference);

        real_t diff_volume = tf::signed_volume(diff.polygons());
        REQUIRE(std::abs(diff_volume - box_volume) <
                std::max(tf::epsilon<real_t>, real_t(0.01)));
    }
}
