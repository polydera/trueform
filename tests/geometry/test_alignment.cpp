/**
 * @file test_alignment.cpp
 * @brief Tests for point cloud alignment functions
 *
 * Tests for:
 * - fit_rigid_alignment
 * - fit_obb_alignment
 * - fit_knn_alignment
 * - chamfer_error
 *
 * Copyright (c) 2025 Ziga Sajovic, XLAB
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <trueform/trueform.hpp>
#include "type_traits.hpp"
#include <cmath>

namespace {

/**
 * @brief Compute RMS error between two point sets with optional transformation
 */
template <typename PointsA, typename PointsB, typename Transform>
auto compute_rms_error(const PointsA& A, const PointsB& B, const Transform& T) {
    using real_t = std::decay_t<decltype(A[0][0])>;
    real_t sum_sq = real_t(0);
    for (decltype(A.size()) i = 0; i < A.size(); ++i) {
        auto a_transformed = tf::transformed(A[i], T);
        real_t dx = a_transformed[0] - B[i][0];
        real_t dy = a_transformed[1] - B[i][1];
        real_t dz = a_transformed[2] - B[i][2];
        sum_sq += dx*dx + dy*dy + dz*dz;
    }
    return std::sqrt(sum_sq / real_t(A.size()));
}

} // anonymous namespace

// =============================================================================
// fit_rigid_alignment - Identity (same point clouds)
// =============================================================================

TEMPLATE_TEST_CASE("fit_rigid_alignment_identity", "[geometry][alignment]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto box = tf::make_box_mesh<index_t>(real_t(2), real_t(1), real_t(3));

    auto T = tf::fit_rigid_alignment(box.points(), box.points());

    // Should be close to identity - RMS error should be ~0
    real_t rms = compute_rms_error(box.points(), box.points(), T);
    REQUIRE(rms < real_t(1e-5));
}

// =============================================================================
// fit_rigid_alignment - Translation
// =============================================================================

TEMPLATE_TEST_CASE("fit_rigid_alignment_translation", "[geometry][alignment]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto box = tf::make_box_mesh<index_t>(real_t(2), real_t(1), real_t(3));

    // Create translation
    auto T_true = tf::make_transformation_from_translation(
        tf::vector<real_t, 3>{real_t(5), real_t(-3), real_t(2)});

    // Transform source points
    tf::points_buffer<real_t, 3> source;
    source.allocate(box.points().size());
    for (decltype(box.points().size()) i = 0; i < box.points().size(); ++i) {
        source[i] = tf::transformed(box.points()[i], T_true);
    }

    auto T_recovered = tf::fit_rigid_alignment(source.points(), box.points());

    // Recovered transform should align source back to target
    real_t rms = compute_rms_error(source.points(), box.points(), T_recovered);
    REQUIRE(rms < real_t(1e-4));
}

// =============================================================================
// fit_rigid_alignment - Rotation
// =============================================================================

TEMPLATE_TEST_CASE("fit_rigid_alignment_rotation", "[geometry][alignment]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto sphere = tf::make_sphere_mesh<index_t>(real_t(1), 20, 20);

    // Create rotation around Z axis (45 degrees)
    real_t angle = real_t(3.14159265358979323846) / real_t(4);
    real_t cos_a = std::cos(angle);
    real_t sin_a = std::sin(angle);

    tf::transformation<real_t, 3> T_true;
    T_true(0, 0) = cos_a;  T_true(0, 1) = -sin_a; T_true(0, 2) = real_t(0); T_true(0, 3) = real_t(0);
    T_true(1, 0) = sin_a;  T_true(1, 1) = cos_a;  T_true(1, 2) = real_t(0); T_true(1, 3) = real_t(0);
    T_true(2, 0) = real_t(0); T_true(2, 1) = real_t(0); T_true(2, 2) = real_t(1); T_true(2, 3) = real_t(0);
    T_true(3, 0) = real_t(0); T_true(3, 1) = real_t(0); T_true(3, 2) = real_t(0); T_true(3, 3) = real_t(1);

    // Transform source points
    tf::points_buffer<real_t, 3> source;
    source.allocate(sphere.points().size());
    for (decltype(sphere.points().size()) i = 0; i < sphere.points().size(); ++i) {
        source[i] = tf::transformed(sphere.points()[i], T_true);
    }

    auto T_recovered = tf::fit_rigid_alignment(source.points(), sphere.points());

    real_t rms = compute_rms_error(source.points(), sphere.points(), T_recovered);
    REQUIRE(rms < real_t(1e-4));
}

// =============================================================================
// fit_rigid_alignment - Rotation + Translation
// =============================================================================

TEMPLATE_TEST_CASE("fit_rigid_alignment_rotation_translation", "[geometry][alignment]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto box = tf::make_box_mesh<index_t>(real_t(2), real_t(1), real_t(3));

    // Create rotation around Z (30 degrees) + translation
    real_t angle = real_t(3.14159265358979323846) / real_t(6);
    real_t cos_a = std::cos(angle);
    real_t sin_a = std::sin(angle);

    tf::transformation<real_t, 3> T_true;
    T_true(0, 0) = cos_a;  T_true(0, 1) = -sin_a; T_true(0, 2) = real_t(0); T_true(0, 3) = real_t(10);
    T_true(1, 0) = sin_a;  T_true(1, 1) = cos_a;  T_true(1, 2) = real_t(0); T_true(1, 3) = real_t(-5);
    T_true(2, 0) = real_t(0); T_true(2, 1) = real_t(0); T_true(2, 2) = real_t(1); T_true(2, 3) = real_t(3);
    T_true(3, 0) = real_t(0); T_true(3, 1) = real_t(0); T_true(3, 2) = real_t(0); T_true(3, 3) = real_t(1);

    // Transform source points
    tf::points_buffer<real_t, 3> source;
    source.allocate(box.points().size());
    for (decltype(box.points().size()) i = 0; i < box.points().size(); ++i) {
        source[i] = tf::transformed(box.points()[i], T_true);
    }

    auto T_recovered = tf::fit_rigid_alignment(source.points(), box.points());

    real_t rms = compute_rms_error(source.points(), box.points(), T_recovered);
    REQUIRE(rms < real_t(1e-4));
}

// =============================================================================
// fit_obb_alignment - Basic alignment
// =============================================================================

TEMPLATE_TEST_CASE("fit_obb_alignment_basic", "[geometry][alignment]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto box = tf::make_box_mesh<index_t>(real_t(2), real_t(1), real_t(3));

    // Create rotation + translation
    real_t angle = real_t(3.14159265358979323846) / real_t(4);
    real_t cos_a = std::cos(angle);
    real_t sin_a = std::sin(angle);

    tf::transformation<real_t, 3> T_true;
    T_true(0, 0) = cos_a;  T_true(0, 1) = -sin_a; T_true(0, 2) = real_t(0); T_true(0, 3) = real_t(5);
    T_true(1, 0) = sin_a;  T_true(1, 1) = cos_a;  T_true(1, 2) = real_t(0); T_true(1, 3) = real_t(-2);
    T_true(2, 0) = real_t(0); T_true(2, 1) = real_t(0); T_true(2, 2) = real_t(1); T_true(2, 3) = real_t(1);
    T_true(3, 0) = real_t(0); T_true(3, 1) = real_t(0); T_true(3, 2) = real_t(0); T_true(3, 3) = real_t(1);

    // Transform source points
    tf::points_buffer<real_t, 3> source;
    source.allocate(box.points().size());
    for (decltype(box.points().size()) i = 0; i < box.points().size(); ++i) {
        source[i] = tf::transformed(box.points()[i], T_true);
    }

    // Build tree on target for disambiguation
    tf::aabb_tree<index_t, real_t, 3> tree(box.points(), tf::config_tree(4, 4));
    auto target_with_tree = box.points() | tf::tag(tree);

    auto T_recovered = tf::fit_obb_alignment(source.points(), target_with_tree);

    // OBB alignment should get close (may have 180 degree ambiguity without tree)
    real_t chamfer = tf::chamfer_error(source.points() | tf::tag(T_recovered), target_with_tree);

    // Chamfer error should be small relative to box size
    REQUIRE(chamfer < real_t(0.5));
}

// =============================================================================
// chamfer_error - Identical point clouds
// =============================================================================

TEMPLATE_TEST_CASE("chamfer_error_identical", "[geometry][alignment]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto sphere = tf::make_sphere_mesh<index_t>(real_t(1), 20, 20);

    tf::aabb_tree<index_t, real_t, 3> tree(sphere.points(), tf::config_tree(4, 4));
    auto target_with_tree = sphere.points() | tf::tag(tree);

    real_t chamfer = tf::chamfer_error(sphere.points(), target_with_tree);

    // Identical point clouds should have ~0 chamfer error
    REQUIRE(chamfer < real_t(1e-5));
}

// =============================================================================
// chamfer_error - Known displacement
// =============================================================================

TEMPLATE_TEST_CASE("chamfer_error_displaced", "[geometry][alignment]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto sphere = tf::make_sphere_mesh<index_t>(real_t(1), 20, 20);

    // Translate source by known amount
    auto T_offset = tf::make_transformation_from_translation(
        tf::vector<real_t, 3>{real_t(0.1), real_t(0), real_t(0)});

    tf::points_buffer<real_t, 3> source;
    source.allocate(sphere.points().size());
    for (decltype(sphere.points().size()) i = 0; i < sphere.points().size(); ++i) {
        source[i] = tf::transformed(sphere.points()[i], T_offset);
    }

    tf::aabb_tree<index_t, real_t, 3> tree(sphere.points(), tf::config_tree(4, 4));
    auto target_with_tree = sphere.points() | tf::tag(tree);

    real_t chamfer = tf::chamfer_error(source.points(), target_with_tree);

    // Chamfer error should be approximately the displacement (0.1)
    // For a sphere, it's not exactly 0.1 due to surface curvature
    REQUIRE(chamfer > real_t(0.05));
    REQUIRE(chamfer < real_t(0.15));
}

// =============================================================================
// fit_knn_alignment - Single ICP iteration
// =============================================================================

TEMPLATE_TEST_CASE("fit_knn_alignment_basic", "[geometry][alignment]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto sphere = tf::make_sphere_mesh<index_t>(real_t(1), 20, 20);

    // Small translation
    auto T_true = tf::make_transformation_from_translation(
        tf::vector<real_t, 3>{real_t(0.05), real_t(0.05), real_t(0)});

    tf::points_buffer<real_t, 3> source;
    source.allocate(sphere.points().size());
    for (decltype(sphere.points().size()) i = 0; i < sphere.points().size(); ++i) {
        source[i] = tf::transformed(sphere.points()[i], T_true);
    }

    tf::aabb_tree<index_t, real_t, 3> tree(sphere.points(), tf::config_tree(4, 4));
    auto target_with_tree = sphere.points() | tf::tag(tree);

    // Initial chamfer
    real_t chamfer_before = tf::chamfer_error(source.points(), target_with_tree);

    // One KNN alignment iteration
    auto T_iter = tf::fit_knn_alignment(source.points(), target_with_tree, 1);

    // Chamfer after alignment should be smaller
    real_t chamfer_after = tf::chamfer_error(source.points() | tf::tag(T_iter), target_with_tree);

    REQUIRE(chamfer_after < chamfer_before);
}

// =============================================================================
// fit_rigid_alignment - Different resolutions (sphere)
// =============================================================================

TEMPLATE_TEST_CASE("fit_rigid_alignment_different_resolutions", "[geometry][alignment]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    // Two spheres with different resolutions
    auto sphere_low = tf::make_sphere_mesh<index_t>(real_t(1), 10, 10);
    auto sphere_high = tf::make_sphere_mesh<index_t>(real_t(1), 30, 30);

    // Transform low-res sphere
    real_t angle = real_t(3.14159265358979323846) / real_t(3);
    real_t cos_a = std::cos(angle);
    real_t sin_a = std::sin(angle);

    tf::transformation<real_t, 3> T_true;
    T_true(0, 0) = cos_a;  T_true(0, 1) = -sin_a; T_true(0, 2) = real_t(0); T_true(0, 3) = real_t(3);
    T_true(1, 0) = sin_a;  T_true(1, 1) = cos_a;  T_true(1, 2) = real_t(0); T_true(1, 3) = real_t(-2);
    T_true(2, 0) = real_t(0); T_true(2, 1) = real_t(0); T_true(2, 2) = real_t(1); T_true(2, 3) = real_t(1);
    T_true(3, 0) = real_t(0); T_true(3, 1) = real_t(0); T_true(3, 2) = real_t(0); T_true(3, 3) = real_t(1);

    tf::points_buffer<real_t, 3> source;
    source.allocate(sphere_low.points().size());
    for (decltype(sphere_low.points().size()) i = 0; i < sphere_low.points().size(); ++i) {
        source[i] = tf::transformed(sphere_low.points()[i], T_true);
    }

    // Build tree on high-res target
    tf::aabb_tree<index_t, real_t, 3> tree(sphere_high.points(), tf::config_tree(4, 4));
    auto target_with_tree = sphere_high.points() | tf::tag(tree);

    // Initial chamfer (far apart)
    real_t chamfer_before = tf::chamfer_error(source.points(), target_with_tree);

    // OBB alignment
    auto T_obb = tf::fit_obb_alignment(source.points(), target_with_tree);
    real_t chamfer_after = tf::chamfer_error(source.points() | tf::tag(T_obb), target_with_tree);

    // Should be much closer after alignment
    REQUIRE(chamfer_after < chamfer_before);
    // Should be close to 0 since both are unit spheres
    REQUIRE(chamfer_after < real_t(0.2));
}
