/**
 * @file test_alignment.cpp
 * @brief Tests for point cloud alignment functions
 *
 * Tests for:
 * - fit_rigid_alignment
 * - fit_obb_alignment
 * - fit_knn_alignment
 * - fit_icp_alignment
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
    auto T_iter = tf::fit_knn_alignment(source.points(), target_with_tree, {1});

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

// =============================================================================
// fit_icp_alignment - Identity (same point clouds)
// =============================================================================

TEMPLATE_TEST_CASE("fit_icp_alignment_identity", "[geometry][alignment][icp]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto sphere = tf::make_sphere_mesh<index_t>(real_t(1), 20, 20);

    tf::aabb_tree<index_t, real_t, 3> tree(sphere.points(), tf::config_tree(4, 4));
    auto target_with_tree = sphere.points() | tf::tag(tree);

    tf::icp_config config;
    config.max_iterations = 10;

    auto T = tf::fit_icp_alignment(sphere.points(), target_with_tree, config);

    // Should be close to identity - RMS error should be ~0
    real_t rms = compute_rms_error(sphere.points(), sphere.points(), T);
    REQUIRE(rms < real_t(1e-4));
}

// =============================================================================
// fit_icp_alignment - Translation
// =============================================================================

TEMPLATE_TEST_CASE("fit_icp_alignment_translation", "[geometry][alignment][icp]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto sphere = tf::make_sphere_mesh<index_t>(real_t(1), 20, 20);

    // Small translation
    auto T_true = tf::make_transformation_from_translation(
        tf::vector<real_t, 3>{real_t(0.1), real_t(0.05), real_t(0.1)});

    tf::points_buffer<real_t, 3> source;
    source.allocate(sphere.points().size());
    for (decltype(sphere.points().size()) i = 0; i < sphere.points().size(); ++i) {
        source[i] = tf::transformed(sphere.points()[i], T_true);
    }

    tf::aabb_tree<index_t, real_t, 3> tree(sphere.points(), tf::config_tree(4, 4));
    auto target_with_tree = sphere.points() | tf::tag(tree);

    tf::icp_config config;
    config.max_iterations = 50;

    auto T_recovered = tf::fit_icp_alignment(source.points(), target_with_tree, config);

    // Chamfer error should be small after alignment
    real_t chamfer = tf::chamfer_error(source.points() | tf::tag(T_recovered), target_with_tree);
    REQUIRE(chamfer < real_t(0.01));
}

// =============================================================================
// fit_icp_alignment - Rotation
// =============================================================================

TEMPLATE_TEST_CASE("fit_icp_alignment_rotation", "[geometry][alignment][icp]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto box = tf::make_box_mesh<index_t>(real_t(2), real_t(1), real_t(3));

    // 10 degree rotation around Z
    real_t angle = real_t(3.14159265358979323846) / real_t(18);
    real_t cos_a = std::cos(angle);
    real_t sin_a = std::sin(angle);

    tf::transformation<real_t, 3> T_true;
    T_true(0, 0) = cos_a;  T_true(0, 1) = -sin_a; T_true(0, 2) = real_t(0); T_true(0, 3) = real_t(0);
    T_true(1, 0) = sin_a;  T_true(1, 1) = cos_a;  T_true(1, 2) = real_t(0); T_true(1, 3) = real_t(0);
    T_true(2, 0) = real_t(0); T_true(2, 1) = real_t(0); T_true(2, 2) = real_t(1); T_true(2, 3) = real_t(0);
    T_true(3, 0) = real_t(0); T_true(3, 1) = real_t(0); T_true(3, 2) = real_t(0); T_true(3, 3) = real_t(1);

    tf::points_buffer<real_t, 3> source;
    source.allocate(box.points().size());
    for (decltype(box.points().size()) i = 0; i < box.points().size(); ++i) {
        source[i] = tf::transformed(box.points()[i], T_true);
    }

    tf::aabb_tree<index_t, real_t, 3> tree(box.points(), tf::config_tree(4, 4));
    auto target_with_tree = box.points() | tf::tag(tree);

    tf::icp_config config;
    config.max_iterations = 50;

    auto T_recovered = tf::fit_icp_alignment(source.points(), target_with_tree, config);

    real_t chamfer = tf::chamfer_error(source.points() | tf::tag(T_recovered), target_with_tree);
    REQUIRE(chamfer < real_t(0.01));
}

// =============================================================================
// fit_icp_alignment - Rotation + Translation
// =============================================================================

TEMPLATE_TEST_CASE("fit_icp_alignment_rotation_translation", "[geometry][alignment][icp]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto sphere = tf::make_sphere_mesh<index_t>(real_t(1), 20, 20);

    // 15 degree rotation + small translation
    real_t angle = real_t(3.14159265358979323846) / real_t(12);
    real_t cos_a = std::cos(angle);
    real_t sin_a = std::sin(angle);

    tf::transformation<real_t, 3> T_true;
    T_true(0, 0) = cos_a;  T_true(0, 1) = -sin_a; T_true(0, 2) = real_t(0); T_true(0, 3) = real_t(0.2);
    T_true(1, 0) = sin_a;  T_true(1, 1) = cos_a;  T_true(1, 2) = real_t(0); T_true(1, 3) = real_t(0.15);
    T_true(2, 0) = real_t(0); T_true(2, 1) = real_t(0); T_true(2, 2) = real_t(1); T_true(2, 3) = real_t(0.1);
    T_true(3, 0) = real_t(0); T_true(3, 1) = real_t(0); T_true(3, 2) = real_t(0); T_true(3, 3) = real_t(1);

    tf::points_buffer<real_t, 3> source;
    source.allocate(sphere.points().size());
    for (decltype(sphere.points().size()) i = 0; i < sphere.points().size(); ++i) {
        source[i] = tf::transformed(sphere.points()[i], T_true);
    }

    tf::aabb_tree<index_t, real_t, 3> tree(sphere.points(), tf::config_tree(4, 4));
    auto target_with_tree = sphere.points() | tf::tag(tree);

    tf::icp_config config;
    config.max_iterations = 50;

    auto T_recovered = tf::fit_icp_alignment(source.points(), target_with_tree, config);

    real_t chamfer = tf::chamfer_error(source.points() | tf::tag(T_recovered), target_with_tree);
    REQUIRE(chamfer < real_t(0.05));
}

// =============================================================================
// fit_icp_alignment - Point-to-Plane (with normals)
// =============================================================================

TEMPLATE_TEST_CASE("fit_icp_alignment_point_to_plane", "[geometry][alignment][icp]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto sphere = tf::make_sphere_mesh<index_t>(real_t(1), 20, 20);

    // Compute normals on target
    auto target_normals = tf::compute_point_normals(sphere.polygons());

    // 10 degree rotation + translation
    real_t angle = real_t(3.14159265358979323846) / real_t(18);
    real_t cos_a = std::cos(angle);
    real_t sin_a = std::sin(angle);

    tf::transformation<real_t, 3> T_true;
    T_true(0, 0) = cos_a;  T_true(0, 1) = -sin_a; T_true(0, 2) = real_t(0); T_true(0, 3) = real_t(0.1);
    T_true(1, 0) = sin_a;  T_true(1, 1) = cos_a;  T_true(1, 2) = real_t(0); T_true(1, 3) = real_t(0.1);
    T_true(2, 0) = real_t(0); T_true(2, 1) = real_t(0); T_true(2, 2) = real_t(1); T_true(2, 3) = real_t(0.05);
    T_true(3, 0) = real_t(0); T_true(3, 1) = real_t(0); T_true(3, 2) = real_t(0); T_true(3, 3) = real_t(1);

    tf::points_buffer<real_t, 3> source;
    source.allocate(sphere.points().size());
    for (decltype(sphere.points().size()) i = 0; i < sphere.points().size(); ++i) {
        source[i] = tf::transformed(sphere.points()[i], T_true);
    }

    // Build tree and attach normals for point-to-plane
    tf::aabb_tree<index_t, real_t, 3> tree(sphere.points(), tf::config_tree(4, 4));
    auto target_with_tree_and_normals = sphere.points() | tf::tag(tree) | tf::tag_normals(target_normals);

    tf::icp_config config;
    config.max_iterations = 50;

    auto T_recovered = tf::fit_icp_alignment(source.points(), target_with_tree_and_normals, config);

    // Point-to-plane should converge
    real_t chamfer = tf::chamfer_error(source.points() | tf::tag(T_recovered),
                                       sphere.points() | tf::tag(tree));
    REQUIRE(chamfer < real_t(0.05));
}

// =============================================================================
// fit_icp_alignment - With initial frame on source
// =============================================================================

TEMPLATE_TEST_CASE("fit_icp_alignment_with_initial_frame", "[geometry][alignment][icp]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto sphere = tf::make_sphere_mesh<index_t>(real_t(1), 20, 20);

    // Larger transformation
    real_t angle = real_t(3.14159265358979323846) / real_t(6);  // 30 degrees
    real_t cos_a = std::cos(angle);
    real_t sin_a = std::sin(angle);

    tf::transformation<real_t, 3> T_true;
    T_true(0, 0) = cos_a;  T_true(0, 1) = -sin_a; T_true(0, 2) = real_t(0); T_true(0, 3) = real_t(0.5);
    T_true(1, 0) = sin_a;  T_true(1, 1) = cos_a;  T_true(1, 2) = real_t(0); T_true(1, 3) = real_t(0.3);
    T_true(2, 0) = real_t(0); T_true(2, 1) = real_t(0); T_true(2, 2) = real_t(1); T_true(2, 3) = real_t(0.2);
    T_true(3, 0) = real_t(0); T_true(3, 1) = real_t(0); T_true(3, 2) = real_t(0); T_true(3, 3) = real_t(1);

    tf::points_buffer<real_t, 3> source;
    source.allocate(sphere.points().size());
    for (decltype(sphere.points().size()) i = 0; i < sphere.points().size(); ++i) {
        source[i] = tf::transformed(sphere.points()[i], T_true);
    }

    tf::aabb_tree<index_t, real_t, 3> tree(sphere.points(), tf::config_tree(4, 4));
    auto target_with_tree = sphere.points() | tf::tag(tree);

    // First get OBB alignment as initial guess
    // OBB returns delta (source_world -> target_world)
    // Since source has no frame, source_world = source_local, so T_init is also total
    auto T_init = tf::fit_obb_alignment(source.points(), target_with_tree);
    auto source_with_frame = source.points() | tf::tag(T_init);

    tf::icp_config config;
    config.max_iterations = 50;

    // ICP returns delta (source_world -> target_world)
    // Now source_world = T_init @ source_local
    auto T_delta = tf::fit_icp_alignment(source_with_frame, target_with_tree, config);

    // To get total: compose T_init with T_delta
    auto T_total = tf::transformed(T_init, T_delta);

    // Verify: T_total @ source_local ≈ target_world
    real_t chamfer = tf::chamfer_error(source.points() | tf::tag(T_total), target_with_tree);
    REQUIRE(chamfer < real_t(0.05));
}

// =============================================================================
// fit_icp_alignment - Convergence check (error decreases)
// =============================================================================

TEMPLATE_TEST_CASE("fit_icp_alignment_convergence", "[geometry][alignment][icp]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto sphere = tf::make_sphere_mesh<index_t>(real_t(1), 20, 20);

    // Moderate transformation
    real_t angle = real_t(3.14159265358979323846) / real_t(12);
    real_t cos_a = std::cos(angle);
    real_t sin_a = std::sin(angle);

    tf::transformation<real_t, 3> T_true;
    T_true(0, 0) = cos_a;  T_true(0, 1) = -sin_a; T_true(0, 2) = real_t(0); T_true(0, 3) = real_t(0.15);
    T_true(1, 0) = sin_a;  T_true(1, 1) = cos_a;  T_true(1, 2) = real_t(0); T_true(1, 3) = real_t(0.1);
    T_true(2, 0) = real_t(0); T_true(2, 1) = real_t(0); T_true(2, 2) = real_t(1); T_true(2, 3) = real_t(0.05);
    T_true(3, 0) = real_t(0); T_true(3, 1) = real_t(0); T_true(3, 2) = real_t(0); T_true(3, 3) = real_t(1);

    tf::points_buffer<real_t, 3> source;
    source.allocate(sphere.points().size());
    for (decltype(sphere.points().size()) i = 0; i < sphere.points().size(); ++i) {
        source[i] = tf::transformed(sphere.points()[i], T_true);
    }

    tf::aabb_tree<index_t, real_t, 3> tree(sphere.points(), tf::config_tree(4, 4));
    auto target_with_tree = sphere.points() | tf::tag(tree);

    // Initial error
    real_t chamfer_before = tf::chamfer_error(source.points(), target_with_tree);

    tf::icp_config config;
    config.max_iterations = 50;

    auto T_recovered = tf::fit_icp_alignment(source.points(), target_with_tree, config);

    real_t chamfer_after = tf::chamfer_error(source.points() | tf::tag(T_recovered), target_with_tree);

    // Error should decrease significantly
    REQUIRE(chamfer_after < chamfer_before);
    REQUIRE(chamfer_after < real_t(0.05));
}

// =============================================================================
// TARGET WITH TRANSFORMATION TESTS
// These tests verify alignment works when the target has a non-identity transform.
// Strategy: Use SAME points for source and target, apply known transforms,
// then verify that after alignment, corresponding points match exactly.
// =============================================================================

// =============================================================================
// fit_rigid_alignment - Target has transformation (known correspondences)
// =============================================================================

TEMPLATE_TEST_CASE("fit_rigid_alignment_target_transform", "[geometry][alignment][target_transform]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto box = tf::make_box_mesh<index_t>(real_t(2), real_t(1), real_t(3));

    // Target transformation
    auto T_target = tf::make_transformation_from_translation(
        tf::vector<real_t, 3>{real_t(10), real_t(-5), real_t(3)});

    // Source transformation (same local points, different world position)
    real_t angle = real_t(3.14159265358979323846) / real_t(6);  // 30 degrees
    real_t cos_a = std::cos(angle);
    real_t sin_a = std::sin(angle);

    tf::transformation<real_t, 3> T_source;
    T_source(0, 0) = cos_a;  T_source(0, 1) = -sin_a; T_source(0, 2) = real_t(0); T_source(0, 3) = real_t(12);
    T_source(1, 0) = sin_a;  T_source(1, 1) = cos_a;  T_source(1, 2) = real_t(0); T_source(1, 3) = real_t(-3);
    T_source(2, 0) = real_t(0); T_source(2, 1) = real_t(0); T_source(2, 2) = real_t(1); T_source(2, 3) = real_t(4);
    T_source(3, 0) = real_t(0); T_source(3, 1) = real_t(0); T_source(3, 2) = real_t(0); T_source(3, 3) = real_t(1);

    // Both use same local points with different transforms
    auto source_with_transform = box.points() | tf::tag(T_source);
    auto target_with_transform = box.points() | tf::tag(T_target);

    // DELTA convention: result maps source_world -> target_world
    auto T_delta = tf::fit_rigid_alignment(source_with_transform, target_with_transform);

    // To map source_local -> target_world, compose with source frame
    auto T_total = tf::transformed(T_source, T_delta);

    // Verify: T_total @ local_pt should equal T_target @ local_pt
    real_t max_error = real_t(0);
    for (std::size_t i = 0; i < box.points().size(); ++i) {
        auto src_world = tf::transformed(box.points()[i], T_total);
        auto tgt_world = tf::transformed(box.points()[i], T_target);
        real_t dx = src_world[0] - tgt_world[0];
        real_t dy = src_world[1] - tgt_world[1];
        real_t dz = src_world[2] - tgt_world[2];
        real_t err = std::sqrt(dx*dx + dy*dy + dz*dz);
        max_error = std::max(max_error, err);
    }
    REQUIRE(max_error < real_t(1e-4));
}

// =============================================================================
// fit_obb_alignment - Target has transformation (SHUFFLED - no correspondence)
// =============================================================================

TEMPLATE_TEST_CASE("fit_obb_alignment_target_transform", "[geometry][alignment][target_transform]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto box = tf::make_box_mesh<index_t>(real_t(2), real_t(1), real_t(3));

    // Target transformation (rotation + translation)
    real_t target_angle = real_t(3.14159265358979323846) / real_t(3);  // 60 degrees
    real_t cos_t = std::cos(target_angle);
    real_t sin_t = std::sin(target_angle);

    tf::transformation<real_t, 3> T_target;
    T_target(0, 0) = cos_t;  T_target(0, 1) = -sin_t; T_target(0, 2) = real_t(0); T_target(0, 3) = real_t(10);
    T_target(1, 0) = sin_t;  T_target(1, 1) = cos_t;  T_target(1, 2) = real_t(0); T_target(1, 3) = real_t(-5);
    T_target(2, 0) = real_t(0); T_target(2, 1) = real_t(0); T_target(2, 2) = real_t(1); T_target(2, 3) = real_t(3);
    T_target(3, 0) = real_t(0); T_target(3, 1) = real_t(0); T_target(3, 2) = real_t(0); T_target(3, 3) = real_t(1);

    // Source transformation (different from target)
    real_t source_angle = real_t(3.14159265358979323846) / real_t(4);  // 45 degrees
    real_t cos_s = std::cos(source_angle);
    real_t sin_s = std::sin(source_angle);

    tf::transformation<real_t, 3> T_source;
    T_source(0, 0) = cos_s;  T_source(0, 1) = -sin_s; T_source(0, 2) = real_t(0); T_source(0, 3) = real_t(5);
    T_source(1, 0) = sin_s;  T_source(1, 1) = cos_s;  T_source(1, 2) = real_t(0); T_source(1, 3) = real_t(-2);
    T_source(2, 0) = real_t(0); T_source(2, 1) = real_t(0); T_source(2, 2) = real_t(1); T_source(2, 3) = real_t(1);
    T_source(3, 0) = real_t(0); T_source(3, 1) = real_t(0); T_source(3, 2) = real_t(0); T_source(3, 3) = real_t(1);

    // Create REVERSED source points (no index correspondence with target)
    tf::points_buffer<real_t, 3> source_reversed;
    source_reversed.allocate(box.points().size());
    for (std::size_t i = 0; i < box.points().size(); ++i) {
        source_reversed[i] = box.points()[box.points().size() - 1 - i];
    }

    auto source_with_transform = source_reversed.points() | tf::tag(T_source);

    tf::aabb_tree<index_t, real_t, 3> tree(box.points(), tf::config_tree(4, 4));
    auto target_with_tree_and_transform = box.points() | tf::tag(tree) | tf::tag(T_target);

    // DELTA convention: result maps source_world -> target_world
    auto T_delta = tf::fit_obb_alignment(source_with_transform, target_with_tree_and_transform, 50);

    // To map source_local -> target_world, compose with source frame
    auto T_total = tf::transformed(T_source, T_delta);

    // Verify with known correspondences: source_reversed[i] corresponds to target[n-1-i]
    // After alignment: T_total @ source_reversed[i] ≈ T_target @ target[n-1-i]
    real_t max_error = real_t(0);
    const auto n = box.points().size();
    for (std::size_t i = 0; i < n; ++i) {
        auto src_world = tf::transformed(source_reversed[i], T_total);
        auto tgt_world = tf::transformed(box.points()[n - 1 - i], T_target);
        real_t dx = src_world[0] - tgt_world[0];
        real_t dy = src_world[1] - tgt_world[1];
        real_t dz = src_world[2] - tgt_world[2];
        max_error = std::max(max_error, std::sqrt(dx*dx + dy*dy + dz*dz));
    }
    REQUIRE(max_error < real_t(0.5));  // OBB has some tolerance due to symmetry
}

// =============================================================================
// fit_knn_alignment - Target has transformation (known correspondences)
// =============================================================================

TEMPLATE_TEST_CASE("fit_knn_alignment_target_transform", "[geometry][alignment][target_transform]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto sphere = tf::make_sphere_mesh<index_t>(real_t(1), 20, 20);

    // Target transformation
    auto T_target = tf::make_transformation_from_translation(
        tf::vector<real_t, 3>{real_t(5), real_t(3), real_t(-2)});

    // Source: same points but with small offset transform (close to target)
    auto T_source = tf::make_transformation_from_translation(
        tf::vector<real_t, 3>{real_t(5.05), real_t(3.05), real_t(-2)});

    auto source_with_transform = sphere.points() | tf::tag(T_source);

    tf::aabb_tree<index_t, real_t, 3> tree(sphere.points(), tf::config_tree(4, 4));
    auto target_with_tree_and_transform = sphere.points() | tf::tag(tree) | tf::tag(T_target);

    // Compute error before
    real_t error_before = real_t(0);
    for (std::size_t i = 0; i < sphere.points().size(); ++i) {
        auto src_world = tf::transformed(sphere.points()[i], T_source);
        auto tgt_world = tf::transformed(sphere.points()[i], T_target);
        real_t dx = src_world[0] - tgt_world[0];
        real_t dy = src_world[1] - tgt_world[1];
        real_t dz = src_world[2] - tgt_world[2];
        error_before += std::sqrt(dx*dx + dy*dy + dz*dz);
    }
    error_before /= real_t(sphere.points().size());

    // One KNN alignment iteration
    tf::knn_alignment_config config;
    config.k = 1;
    // DELTA convention: result maps source_world -> target_world
    auto T_delta = tf::fit_knn_alignment(source_with_transform, target_with_tree_and_transform, config);

    // To map source_local -> target_world, compose with source frame
    auto T_total = tf::transformed(T_source, T_delta);

    // Compute error after
    real_t error_after = real_t(0);
    for (std::size_t i = 0; i < sphere.points().size(); ++i) {
        auto src_world = tf::transformed(sphere.points()[i], T_total);
        auto tgt_world = tf::transformed(sphere.points()[i], T_target);
        real_t dx = src_world[0] - tgt_world[0];
        real_t dy = src_world[1] - tgt_world[1];
        real_t dz = src_world[2] - tgt_world[2];
        error_after += std::sqrt(dx*dx + dy*dy + dz*dz);
    }
    error_after /= real_t(sphere.points().size());

    REQUIRE(error_after < error_before);
}

// =============================================================================
// fit_icp_alignment - Target has transformation (SHUFFLED - no correspondence)
// =============================================================================

TEMPLATE_TEST_CASE("fit_icp_alignment_target_transform", "[geometry][alignment][icp][target_transform]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto sphere = tf::make_sphere_mesh<index_t>(real_t(1), 20, 20);

    // Target transformation (rotation + translation)
    real_t target_angle = real_t(3.14159265358979323846) / real_t(4);  // 45 degrees
    real_t cos_t = std::cos(target_angle);
    real_t sin_t = std::sin(target_angle);

    tf::transformation<real_t, 3> T_target;
    T_target(0, 0) = cos_t;  T_target(0, 1) = -sin_t; T_target(0, 2) = real_t(0); T_target(0, 3) = real_t(5);
    T_target(1, 0) = sin_t;  T_target(1, 1) = cos_t;  T_target(1, 2) = real_t(0); T_target(1, 3) = real_t(3);
    T_target(2, 0) = real_t(0); T_target(2, 1) = real_t(0); T_target(2, 2) = real_t(1); T_target(2, 3) = real_t(-2);

    // Source transformation (small perturbation from target)
    real_t source_angle = target_angle + real_t(0.1);  // ~5.7 degree difference
    real_t cos_s = std::cos(source_angle);
    real_t sin_s = std::sin(source_angle);

    tf::transformation<real_t, 3> T_source;
    T_source(0, 0) = cos_s;  T_source(0, 1) = -sin_s; T_source(0, 2) = real_t(0); T_source(0, 3) = real_t(5.1);
    T_source(1, 0) = sin_s;  T_source(1, 1) = cos_s;  T_source(1, 2) = real_t(0); T_source(1, 3) = real_t(3.1);
    T_source(2, 0) = real_t(0); T_source(2, 1) = real_t(0); T_source(2, 2) = real_t(1); T_source(2, 3) = real_t(-1.9);

    // Create REVERSED source points (no index correspondence)
    tf::points_buffer<real_t, 3> source_reversed;
    source_reversed.allocate(sphere.points().size());
    for (std::size_t i = 0; i < sphere.points().size(); ++i) {
        source_reversed[i] = sphere.points()[sphere.points().size() - 1 - i];
    }

    // Also create pre-transformed source points (same world positions, no frame tag)
    tf::points_buffer<real_t, 3> source_world;
    source_world.allocate(sphere.points().size());
    for (std::size_t i = 0; i < sphere.points().size(); ++i) {
        source_world[i] = tf::transformed(source_reversed[i], T_source);
    }

    tf::aabb_tree<index_t, real_t, 3> tree(sphere.points(), tf::config_tree(4, 4));
    auto target_with_tree_and_transform = sphere.points() | tf::tag(tree) | tf::tag(T_target);

    tf::icp_config config;
    config.max_iterations = 50;
    config.n_samples = 200;

    // Test 1: Source WITH transform tag (local points + T_source)
    auto source_with_transform = source_reversed.points() | tf::tag(T_source);
    auto T_delta_tagged = tf::fit_icp_alignment(source_with_transform, target_with_tree_and_transform, config);

    // Test 2: Source WITHOUT transform tag (pre-transformed world points)
    auto T_delta_world = tf::fit_icp_alignment(source_world.points(), target_with_tree_and_transform, config);

    // Compute total for tagged case
    auto T_total_tagged = tf::transformed(T_source, T_delta_tagged);

    // For world case, delta IS the total (no source frame to compose)

    // Both totals should give same result when applied to their respective sources
    real_t chamfer_tagged = tf::chamfer_error(source_reversed.points() | tf::tag(T_total_tagged), target_with_tree_and_transform);
    real_t chamfer_world = tf::chamfer_error(source_world.points() | tf::tag(T_delta_world), target_with_tree_and_transform);

    // The chamfers should be similar
    REQUIRE(chamfer_tagged < real_t(0.1));
    REQUIRE(chamfer_world < real_t(0.1));
}

// =============================================================================
// fit_icp_alignment - Both source AND target have transformations (SHUFFLED)
// =============================================================================

TEMPLATE_TEST_CASE("fit_icp_alignment_both_transforms", "[geometry][alignment][icp][target_transform]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto box = tf::make_box_mesh<index_t>(real_t(2), real_t(1), real_t(3));

    // Target transformation
    real_t target_angle = real_t(3.14159265358979323846) / real_t(3);  // 60 degrees
    real_t cos_t = std::cos(target_angle);
    real_t sin_t = std::sin(target_angle);

    tf::transformation<real_t, 3> T_target;
    T_target(0, 0) = cos_t;  T_target(0, 1) = -sin_t; T_target(0, 2) = real_t(0); T_target(0, 3) = real_t(10);
    T_target(1, 0) = sin_t;  T_target(1, 1) = cos_t;  T_target(1, 2) = real_t(0); T_target(1, 3) = real_t(-5);
    T_target(2, 0) = real_t(0); T_target(2, 1) = real_t(0); T_target(2, 2) = real_t(1); T_target(2, 3) = real_t(3);
    T_target(3, 0) = real_t(0); T_target(3, 1) = real_t(0); T_target(3, 2) = real_t(0); T_target(3, 3) = real_t(1);

    // Source transformation (small offset from target)
    real_t source_angle = target_angle + real_t(0.05);
    real_t cos_s = std::cos(source_angle);
    real_t sin_s = std::sin(source_angle);

    tf::transformation<real_t, 3> T_source;
    T_source(0, 0) = cos_s;  T_source(0, 1) = -sin_s; T_source(0, 2) = real_t(0); T_source(0, 3) = real_t(10.1);
    T_source(1, 0) = sin_s;  T_source(1, 1) = cos_s;  T_source(1, 2) = real_t(0); T_source(1, 3) = real_t(-4.9);
    T_source(2, 0) = real_t(0); T_source(2, 1) = real_t(0); T_source(2, 2) = real_t(1); T_source(2, 3) = real_t(3.1);
    T_source(3, 0) = real_t(0); T_source(3, 1) = real_t(0); T_source(3, 2) = real_t(0); T_source(3, 3) = real_t(1);

    // Create REVERSED source points (no index correspondence)
    tf::points_buffer<real_t, 3> source_reversed;
    source_reversed.allocate(box.points().size());
    for (std::size_t i = 0; i < box.points().size(); ++i) {
        source_reversed[i] = box.points()[box.points().size() - 1 - i];
    }

    auto source_with_transform = source_reversed.points() | tf::tag(T_source);

    tf::aabb_tree<index_t, real_t, 3> tree(box.points(), tf::config_tree(4, 4));
    auto target_with_tree_and_transform = box.points() | tf::tag(tree) | tf::tag(T_target);

    tf::icp_config config;
    config.max_iterations = 50;
    config.n_samples = 0;  // Use all points

    // DELTA convention: result maps source_world -> target_world
    auto T_delta = tf::fit_icp_alignment(source_with_transform, target_with_tree_and_transform, config);

    // To map source_local -> target_world, compose with source frame
    auto T_total = tf::transformed(T_source, T_delta);

    // Verify with known correspondences: source_reversed[i] corresponds to target[n-1-i]
    real_t max_error = real_t(0);
    const auto n = box.points().size();
    for (std::size_t i = 0; i < n; ++i) {
        auto src_world = tf::transformed(source_reversed[i], T_total);
        auto tgt_world = tf::transformed(box.points()[n - 1 - i], T_target);
        real_t dx = src_world[0] - tgt_world[0];
        real_t dy = src_world[1] - tgt_world[1];
        real_t dz = src_world[2] - tgt_world[2];
        max_error = std::max(max_error, std::sqrt(dx*dx + dy*dy + dz*dz));
    }
    REQUIRE(max_error < real_t(0.1));
}

// =============================================================================
// chamfer_error - Both have same transformation (should be ~0)
// =============================================================================

TEMPLATE_TEST_CASE("chamfer_error_both_transforms", "[geometry][alignment][target_transform]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto sphere = tf::make_sphere_mesh<index_t>(real_t(1), 20, 20);

    // Same transformation for both
    auto T = tf::make_transformation_from_translation(
        tf::vector<real_t, 3>{real_t(5), real_t(3), real_t(-2)});

    auto source_with_transform = sphere.points() | tf::tag(T);

    tf::aabb_tree<index_t, real_t, 3> tree(sphere.points(), tf::config_tree(4, 4));
    auto target_with_tree_and_transform = sphere.points() | tf::tag(tree) | tf::tag(T);

    real_t chamfer = tf::chamfer_error(source_with_transform, target_with_tree_and_transform);

    // Same transform on both = same world positions = ~0 chamfer
    REQUIRE(chamfer < real_t(1e-5));
}

// =============================================================================
// fit_icp_alignment - Point-to-Plane with target transformation (SHUFFLED)
// =============================================================================

TEMPLATE_TEST_CASE("fit_icp_alignment_p2plane_target_transform", "[geometry][alignment][icp][target_transform]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto sphere = tf::make_sphere_mesh<index_t>(real_t(1), 20, 20);

    // Target transformation
    real_t target_angle = real_t(3.14159265358979323846) / real_t(4);
    real_t cos_t = std::cos(target_angle);
    real_t sin_t = std::sin(target_angle);

    tf::transformation<real_t, 3> T_target;
    T_target(0, 0) = cos_t;  T_target(0, 1) = -sin_t; T_target(0, 2) = real_t(0); T_target(0, 3) = real_t(5);
    T_target(1, 0) = sin_t;  T_target(1, 1) = cos_t;  T_target(1, 2) = real_t(0); T_target(1, 3) = real_t(3);
    T_target(2, 0) = real_t(0); T_target(2, 1) = real_t(0); T_target(2, 2) = real_t(1); T_target(2, 3) = real_t(-2);

    // Compute normals (in local coords)
    auto target_normals = tf::compute_point_normals(sphere.polygons());

    // Source: small perturbation from target
    real_t source_angle = target_angle + real_t(0.1);
    real_t cos_s = std::cos(source_angle);
    real_t sin_s = std::sin(source_angle);

    tf::transformation<real_t, 3> T_source;
    T_source(0, 0) = cos_s;  T_source(0, 1) = -sin_s; T_source(0, 2) = real_t(0); T_source(0, 3) = real_t(5.1);
    T_source(1, 0) = sin_s;  T_source(1, 1) = cos_s;  T_source(1, 2) = real_t(0); T_source(1, 3) = real_t(3.1);
    T_source(2, 0) = real_t(0); T_source(2, 1) = real_t(0); T_source(2, 2) = real_t(1); T_source(2, 3) = real_t(-1.9);

    // Create REVERSED source points (no index correspondence)
    tf::points_buffer<real_t, 3> source_reversed;
    source_reversed.allocate(sphere.points().size());
    for (std::size_t i = 0; i < sphere.points().size(); ++i) {
        source_reversed[i] = sphere.points()[sphere.points().size() - 1 - i];
    }

    // Also create pre-transformed source points (same world positions, no frame tag)
    tf::points_buffer<real_t, 3> source_world;
    source_world.allocate(sphere.points().size());
    for (std::size_t i = 0; i < sphere.points().size(); ++i) {
        source_world[i] = tf::transformed(source_reversed[i], T_source);
    }

    // Target with normals and transformation
    tf::aabb_tree<index_t, real_t, 3> tree(sphere.points(), tf::config_tree(4, 4));
    auto target_with_all = sphere.points() | tf::tag(tree) | tf::tag(T_target) | tf::tag_normals(target_normals);

    tf::icp_config config;
    config.max_iterations = 50;
    config.n_samples = 200;

    // Test 1: Source WITH transform tag (local points + T_source)
    auto source_with_transform = source_reversed.points() | tf::tag(T_source);
    auto T_delta_tagged = tf::fit_icp_alignment(source_with_transform, target_with_all, config);

    // Test 2: Source WITHOUT transform tag (pre-transformed world points)
    auto T_delta_world = tf::fit_icp_alignment(source_world.points(), target_with_all, config);

    // Compute total for tagged case
    auto T_total_tagged = tf::transformed(T_source, T_delta_tagged);

    // For chamfer, use target without normals
    auto target_for_chamfer = sphere.points() | tf::tag(tree) | tf::tag(T_target);

    // Both totals should give same result when applied to their respective sources
    real_t chamfer_tagged = tf::chamfer_error(source_reversed.points() | tf::tag(T_total_tagged), target_for_chamfer);
    real_t chamfer_world = tf::chamfer_error(source_world.points() | tf::tag(T_delta_world), target_for_chamfer);

    // The chamfers should be similar and small
    REQUIRE(chamfer_tagged < real_t(0.1));
    REQUIRE(chamfer_world < real_t(0.1));
}

// =============================================================================
// chamfer_error - Consistency with different transform combinations
// All 4 combinations should give same result for same world positions
// =============================================================================

TEMPLATE_TEST_CASE("chamfer_error_transform_combinations", "[geometry][alignment][target_transform]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto sphere = tf::make_sphere_mesh<index_t>(real_t(1), 20, 20);

    // Define a transform
    auto T = tf::make_transformation_from_translation(
        tf::vector<real_t, 3>{real_t(5), real_t(3), real_t(-2)});

    // Pre-transform points
    tf::points_buffer<real_t, 3> world_points;
    world_points.allocate(sphere.points().size());
    for (std::size_t i = 0; i < sphere.points().size(); ++i) {
        world_points[i] = tf::transformed(sphere.points()[i], T);
    }

    // Build trees
    tf::aabb_tree<index_t, real_t, 3> tree_local(sphere.points(), tf::config_tree(4, 4));
    tf::aabb_tree<index_t, real_t, 3> tree_world(world_points.points(), tf::config_tree(4, 4));

    // Case 1: Neither has transform (both in local coords)
    real_t chamfer_local = tf::chamfer_error(
        sphere.points(),
        sphere.points() | tf::tag(tree_local));

    // Case 2: Source has transform, target local
    // Source world vs target local -> different positions, so use world-world
    // Actually for this test we need consistent world positions

    // Case 3: Both have same transform
    real_t chamfer_both = tf::chamfer_error(
        sphere.points() | tf::tag(T),
        sphere.points() | tf::tag(tree_local) | tf::tag(T));

    // Case 4: Neither has transform, but using pre-transformed points
    real_t chamfer_world = tf::chamfer_error(
        world_points.points(),
        world_points.points() | tf::tag(tree_world));

    // All should be ~0 (same points, same positions)
    REQUIRE(chamfer_local < real_t(1e-5));
    REQUIRE(chamfer_both < real_t(1e-5));
    REQUIRE(chamfer_world < real_t(1e-5));
}

// =============================================================================
// chamfer_error - Shuffled points should give same result
// =============================================================================

TEMPLATE_TEST_CASE("chamfer_error_shuffled", "[geometry][alignment][target_transform]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto sphere = tf::make_sphere_mesh<index_t>(real_t(1), 20, 20);

    // Create reversed source
    tf::points_buffer<real_t, 3> source_reversed;
    source_reversed.allocate(sphere.points().size());
    for (std::size_t i = 0; i < sphere.points().size(); ++i) {
        source_reversed[i] = sphere.points()[sphere.points().size() - 1 - i];
    }

    tf::aabb_tree<index_t, real_t, 3> tree(sphere.points(), tf::config_tree(4, 4));

    // Both should give ~0 chamfer (same set of points, different order)
    real_t chamfer_normal = tf::chamfer_error(sphere.points(), sphere.points() | tf::tag(tree));
    real_t chamfer_reversed = tf::chamfer_error(source_reversed.points(), sphere.points() | tf::tag(tree));

    REQUIRE(chamfer_normal < real_t(1e-5));
    REQUIRE(chamfer_reversed < real_t(1e-5));
}

// =============================================================================
// fit_rigid_alignment - All 4 transform combinations
// =============================================================================

TEMPLATE_TEST_CASE("fit_rigid_alignment_all_transform_combos", "[geometry][alignment][target_transform]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto box = tf::make_box_mesh<index_t>(real_t(2), real_t(1), real_t(3));

    // Define transforms
    auto T_source = tf::make_transformation_from_translation(
        tf::vector<real_t, 3>{real_t(5), real_t(-2), real_t(1)});
    auto T_target = tf::make_transformation_from_translation(
        tf::vector<real_t, 3>{real_t(10), real_t(-5), real_t(3)});

    // Pre-transform points for "world" versions
    tf::points_buffer<real_t, 3> source_world, target_world;
    source_world.allocate(box.points().size());
    target_world.allocate(box.points().size());
    for (std::size_t i = 0; i < box.points().size(); ++i) {
        source_world[i] = tf::transformed(box.points()[i], T_source);
        target_world[i] = tf::transformed(box.points()[i], T_target);
    }

    // DELTA convention: result maps source_world -> target_world
    // To verify: compose source frame with delta, then apply to source_local

    auto verify_alignment = [&](const auto& T_total, const auto& source_pts) {
        real_t max_error = real_t(0);
        for (std::size_t i = 0; i < box.points().size(); ++i) {
            auto src_world = tf::transformed(source_pts[i], T_total);
            auto tgt_world = tf::transformed(box.points()[i], T_target);
            real_t dx = src_world[0] - tgt_world[0];
            real_t dy = src_world[1] - tgt_world[1];
            real_t dz = src_world[2] - tgt_world[2];
            max_error = std::max(max_error, std::sqrt(dx*dx + dy*dy + dz*dz));
        }
        return max_error;
    };

    // Case 1: Neither has transform (using pre-transformed points)
    // source_local = source_world, so delta = total
    auto T1 = tf::fit_rigid_alignment(source_world.points(), target_world.points());
    REQUIRE(verify_alignment(T1, source_world.points()) < real_t(1e-4));

    // Case 2: Source has transform only
    // delta maps source_world -> target_world
    // total = delta @ T_source = tf::transformed(T_source, delta)
    auto T2_delta = tf::fit_rigid_alignment(
        box.points() | tf::tag(T_source),
        target_world.points());
    auto T2_total = tf::transformed(T_source, T2_delta);
    REQUIRE(verify_alignment(T2_total, box.points()) < real_t(1e-4));

    // Case 3: Target has transform only
    // source_local = source_world, so delta = total
    auto T3 = tf::fit_rigid_alignment(
        source_world.points(),
        box.points() | tf::tag(T_target));
    REQUIRE(verify_alignment(T3, source_world.points()) < real_t(1e-4));

    // Case 4: Both have transforms
    // delta maps source_world -> target_world
    // total = delta @ T_source = tf::transformed(T_source, delta)
    auto T4_delta = tf::fit_rigid_alignment(
        box.points() | tf::tag(T_source),
        box.points() | tf::tag(T_target));
    auto T4_total = tf::transformed(T_source, T4_delta);
    REQUIRE(verify_alignment(T4_total, box.points()) < real_t(1e-4));
}

// =============================================================================
// fit_obb_alignment - All 4 transform combinations
// =============================================================================

TEMPLATE_TEST_CASE("fit_obb_alignment_all_transform_combos", "[geometry][alignment][target_transform]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto box = tf::make_box_mesh<index_t>(real_t(2), real_t(1), real_t(3));

    // Define transforms
    real_t angle = real_t(3.14159265358979323846) / real_t(4);  // 45 degrees
    real_t cos_a = std::cos(angle);
    real_t sin_a = std::sin(angle);

    tf::transformation<real_t, 3> T_source;
    T_source(0, 0) = cos_a;  T_source(0, 1) = -sin_a; T_source(0, 2) = real_t(0); T_source(0, 3) = real_t(5);
    T_source(1, 0) = sin_a;  T_source(1, 1) = cos_a;  T_source(1, 2) = real_t(0); T_source(1, 3) = real_t(-2);
    T_source(2, 0) = real_t(0); T_source(2, 1) = real_t(0); T_source(2, 2) = real_t(1); T_source(2, 3) = real_t(1);
    T_source(3, 0) = real_t(0); T_source(3, 1) = real_t(0); T_source(3, 2) = real_t(0); T_source(3, 3) = real_t(1);

    tf::transformation<real_t, 3> T_target;
    T_target(0, 0) = real_t(1); T_target(0, 1) = real_t(0); T_target(0, 2) = real_t(0); T_target(0, 3) = real_t(10);
    T_target(1, 0) = real_t(0); T_target(1, 1) = real_t(1); T_target(1, 2) = real_t(0); T_target(1, 3) = real_t(-5);
    T_target(2, 0) = real_t(0); T_target(2, 1) = real_t(0); T_target(2, 2) = real_t(1); T_target(2, 3) = real_t(3);
    T_target(3, 0) = real_t(0); T_target(3, 1) = real_t(0); T_target(3, 2) = real_t(0); T_target(3, 3) = real_t(1);

    // Pre-transform points
    tf::points_buffer<real_t, 3> source_world, target_world;
    source_world.allocate(box.points().size());
    target_world.allocate(box.points().size());
    for (std::size_t i = 0; i < box.points().size(); ++i) {
        source_world[i] = tf::transformed(box.points()[i], T_source);
        target_world[i] = tf::transformed(box.points()[i], T_target);
    }

    // Build trees
    tf::aabb_tree<index_t, real_t, 3> tree_local(box.points(), tf::config_tree(4, 4));
    tf::aabb_tree<index_t, real_t, 3> tree_world(target_world.points(), tf::config_tree(4, 4));

    // DELTA convention: result maps source_world -> target_world
    // For cases with source transform, compose with source frame to get total

    // Case 1: Neither has transform (source_local = source_world)
    auto T1 = tf::fit_obb_alignment(source_world.points(), target_world.points() | tf::tag(tree_world), 50);
    real_t chamfer1 = tf::chamfer_error(source_world.points() | tf::tag(T1), target_world.points() | tf::tag(tree_world));
    REQUIRE(chamfer1 < real_t(0.5));

    // Case 2: Source has transform only
    auto T2_delta = tf::fit_obb_alignment(
        box.points() | tf::tag(T_source),
        target_world.points() | tf::tag(tree_world), 50);
    auto T2_total = tf::transformed(T_source, T2_delta);
    real_t chamfer2 = tf::chamfer_error(box.points() | tf::tag(T2_total), target_world.points() | tf::tag(tree_world));
    REQUIRE(chamfer2 < real_t(0.5));

    // Case 3: Target has transform only (source_local = source_world)
    auto T3 = tf::fit_obb_alignment(
        source_world.points(),
        box.points() | tf::tag(tree_local) | tf::tag(T_target), 50);
    real_t chamfer3 = tf::chamfer_error(source_world.points() | tf::tag(T3), box.points() | tf::tag(tree_local) | tf::tag(T_target));
    REQUIRE(chamfer3 < real_t(0.5));

    // Case 4: Both have transforms
    auto T4_delta = tf::fit_obb_alignment(
        box.points() | tf::tag(T_source),
        box.points() | tf::tag(tree_local) | tf::tag(T_target), 50);
    auto T4_total = tf::transformed(T_source, T4_delta);
    real_t chamfer4 = tf::chamfer_error(box.points() | tf::tag(T4_total), box.points() | tf::tag(tree_local) | tf::tag(T_target));
    REQUIRE(chamfer4 < real_t(0.5));
}

// =============================================================================
// fit_knn_alignment - All 4 transform combinations
// =============================================================================

TEMPLATE_TEST_CASE("fit_knn_alignment_all_transform_combos", "[geometry][alignment][target_transform]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto sphere = tf::make_sphere_mesh<index_t>(real_t(1), 20, 20);

    // Define transforms (small offset so KNN improves)
    auto T_source = tf::make_transformation_from_translation(
        tf::vector<real_t, 3>{real_t(5.05), real_t(-2.05), real_t(1)});
    auto T_target = tf::make_transformation_from_translation(
        tf::vector<real_t, 3>{real_t(5), real_t(-2), real_t(1)});

    // Pre-transform points
    tf::points_buffer<real_t, 3> source_world, target_world;
    source_world.allocate(sphere.points().size());
    target_world.allocate(sphere.points().size());
    for (std::size_t i = 0; i < sphere.points().size(); ++i) {
        source_world[i] = tf::transformed(sphere.points()[i], T_source);
        target_world[i] = tf::transformed(sphere.points()[i], T_target);
    }

    // Build trees
    tf::aabb_tree<index_t, real_t, 3> tree_local(sphere.points(), tf::config_tree(4, 4));
    tf::aabb_tree<index_t, real_t, 3> tree_world(target_world.points(), tf::config_tree(4, 4));

    tf::knn_alignment_config config;
    config.k = 1;

    auto compute_chamfer_before = [&](const auto& src, const auto& tgt) {
        return tf::chamfer_error(src, tgt);
    };

    // DELTA convention: result maps source_world -> target_world
    // For cases with source transform, compose with source frame to get total

    // Case 1: Neither has transform (source_local = source_world)
    real_t before1 = compute_chamfer_before(source_world.points(), target_world.points() | tf::tag(tree_world));
    auto T1 = tf::fit_knn_alignment(source_world.points(), target_world.points() | tf::tag(tree_world), config);
    real_t after1 = tf::chamfer_error(source_world.points() | tf::tag(T1), target_world.points() | tf::tag(tree_world));
    REQUIRE(after1 < before1);

    // Case 2: Source has transform only
    real_t before2 = compute_chamfer_before(sphere.points() | tf::tag(T_source), target_world.points() | tf::tag(tree_world));
    auto T2_delta = tf::fit_knn_alignment(sphere.points() | tf::tag(T_source), target_world.points() | tf::tag(tree_world), config);
    auto T2_total = tf::transformed(T_source, T2_delta);
    real_t after2 = tf::chamfer_error(sphere.points() | tf::tag(T2_total), target_world.points() | tf::tag(tree_world));
    REQUIRE(after2 < before2);

    // Case 3: Target has transform only (source_local = source_world)
    real_t before3 = compute_chamfer_before(source_world.points(), sphere.points() | tf::tag(tree_local) | tf::tag(T_target));
    auto T3 = tf::fit_knn_alignment(source_world.points(), sphere.points() | tf::tag(tree_local) | tf::tag(T_target), config);
    real_t after3 = tf::chamfer_error(source_world.points() | tf::tag(T3), sphere.points() | tf::tag(tree_local) | tf::tag(T_target));
    REQUIRE(after3 < before3);

    // Case 4: Both have transforms
    real_t before4 = compute_chamfer_before(sphere.points() | tf::tag(T_source), sphere.points() | tf::tag(tree_local) | tf::tag(T_target));
    auto T4_delta = tf::fit_knn_alignment(sphere.points() | tf::tag(T_source), sphere.points() | tf::tag(tree_local) | tf::tag(T_target), config);
    auto T4_total = tf::transformed(T_source, T4_delta);
    real_t after4 = tf::chamfer_error(sphere.points() | tf::tag(T4_total), sphere.points() | tf::tag(tree_local) | tf::tag(T_target));
    REQUIRE(after4 < before4);
}

// =============================================================================
// fit_icp_alignment - All 4 transform combinations (Point-to-Point)
// =============================================================================

TEMPLATE_TEST_CASE("fit_icp_alignment_all_transform_combos", "[geometry][alignment][icp][target_transform]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto sphere = tf::make_sphere_mesh<index_t>(real_t(1), 20, 20);

    // Define transforms (small offset so ICP converges)
    auto T_source = tf::make_transformation_from_translation(
        tf::vector<real_t, 3>{real_t(5.05), real_t(-2.05), real_t(1)});
    auto T_target = tf::make_transformation_from_translation(
        tf::vector<real_t, 3>{real_t(5), real_t(-2), real_t(1)});

    // Pre-transform points
    tf::points_buffer<real_t, 3> source_world, target_world;
    source_world.allocate(sphere.points().size());
    target_world.allocate(sphere.points().size());
    for (std::size_t i = 0; i < sphere.points().size(); ++i) {
        source_world[i] = tf::transformed(sphere.points()[i], T_source);
        target_world[i] = tf::transformed(sphere.points()[i], T_target);
    }

    // Build trees
    tf::aabb_tree<index_t, real_t, 3> tree_local(sphere.points(), tf::config_tree(4, 4));
    tf::aabb_tree<index_t, real_t, 3> tree_world(target_world.points(), tf::config_tree(4, 4));

    tf::icp_config config;
    config.max_iterations = 50;
    config.n_samples = 100;

    // DELTA convention: result maps source_world -> target_world
    // For cases with source transform, compose with source frame to get total

    // Case 1: Neither has transform (source_local = source_world)
    auto T1 = tf::fit_icp_alignment(source_world.points(), target_world.points() | tf::tag(tree_world), config);
    real_t chamfer1 = tf::chamfer_error(source_world.points() | tf::tag(T1), target_world.points() | tf::tag(tree_world));
    REQUIRE(chamfer1 < real_t(0.01));

    // Case 2: Source has transform only
    auto T2_delta = tf::fit_icp_alignment(
        sphere.points() | tf::tag(T_source),
        target_world.points() | tf::tag(tree_world), config);
    auto T2_total = tf::transformed(T_source, T2_delta);
    real_t chamfer2 = tf::chamfer_error(sphere.points() | tf::tag(T2_total), target_world.points() | tf::tag(tree_world));
    REQUIRE(chamfer2 < real_t(0.01));

    // Case 3: Target has transform only (source_local = source_world)
    auto T3 = tf::fit_icp_alignment(
        source_world.points(),
        sphere.points() | tf::tag(tree_local) | tf::tag(T_target), config);
    real_t chamfer3 = tf::chamfer_error(source_world.points() | tf::tag(T3), sphere.points() | tf::tag(tree_local) | tf::tag(T_target));
    REQUIRE(chamfer3 < real_t(0.01));

    // Case 4: Both have transforms
    auto T4_delta = tf::fit_icp_alignment(
        sphere.points() | tf::tag(T_source),
        sphere.points() | tf::tag(tree_local) | tf::tag(T_target), config);
    auto T4_total = tf::transformed(T_source, T4_delta);
    real_t chamfer4 = tf::chamfer_error(sphere.points() | tf::tag(T4_total), sphere.points() | tf::tag(tree_local) | tf::tag(T_target));
    REQUIRE(chamfer4 < real_t(0.01));
}

// =============================================================================
// fit_icp_alignment - All 4 transform combinations (Point-to-Plane with normals)
// =============================================================================

TEMPLATE_TEST_CASE("fit_icp_alignment_p2plane_all_transform_combos", "[geometry][alignment][icp][target_transform]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto sphere = tf::make_sphere_mesh<index_t>(real_t(1), 20, 20);

    // Compute normals on local points
    auto normals_local = tf::compute_point_normals(sphere.polygons());

    // Define transforms (small offset so ICP converges)
    auto T_source = tf::make_transformation_from_translation(
        tf::vector<real_t, 3>{real_t(5.05), real_t(-2.05), real_t(1)});
    auto T_target = tf::make_transformation_from_translation(
        tf::vector<real_t, 3>{real_t(5), real_t(-2), real_t(1)});

    // Pre-transform points
    tf::points_buffer<real_t, 3> source_world, target_world;
    source_world.allocate(sphere.points().size());
    target_world.allocate(sphere.points().size());
    for (std::size_t i = 0; i < sphere.points().size(); ++i) {
        source_world[i] = tf::transformed(sphere.points()[i], T_source);
        target_world[i] = tf::transformed(sphere.points()[i], T_target);
    }

    // Pre-transform normals for world target (normals transform by inverse-transpose,
    // but for pure translation/rotation, just apply rotation part)
    // For translation-only transforms, normals stay the same
    auto normals_world = normals_local;

    // Build trees
    tf::aabb_tree<index_t, real_t, 3> tree_local(sphere.points(), tf::config_tree(4, 4));
    tf::aabb_tree<index_t, real_t, 3> tree_world(target_world.points(), tf::config_tree(4, 4));

    tf::icp_config config;
    config.max_iterations = 50;
    config.n_samples = 100;

    // DELTA convention: result maps source_world -> target_world
    // For cases with source transform, compose with source frame to get total

    // Case 1: Neither has transform (source_local = source_world)
    // Target: world points + tree + normals (no transform tag)
    auto target1 = target_world.points() | tf::tag(tree_world) | tf::tag_normals(normals_world);
    auto T1 = tf::fit_icp_alignment(source_world.points(), target1, config);
    real_t chamfer1 = tf::chamfer_error(source_world.points() | tf::tag(T1), target_world.points() | tf::tag(tree_world));
    REQUIRE(chamfer1 < real_t(0.01));

    // Case 2: Source has transform only
    // Target: world points + tree + normals (no transform tag)
    auto target2 = target_world.points() | tf::tag(tree_world) | tf::tag_normals(normals_world);
    auto T2_delta = tf::fit_icp_alignment(
        sphere.points() | tf::tag(T_source),
        target2, config);
    auto T2_total = tf::transformed(T_source, T2_delta);
    real_t chamfer2 = tf::chamfer_error(sphere.points() | tf::tag(T2_total), target_world.points() | tf::tag(tree_world));
    REQUIRE(chamfer2 < real_t(0.01));

    // Case 3: Target has transform only (source_local = source_world)
    // Target: local points + tree + transform + normals
    auto target3 = sphere.points() | tf::tag(tree_local) | tf::tag(T_target) | tf::tag_normals(normals_local);
    auto T3 = tf::fit_icp_alignment(source_world.points(), target3, config);
    real_t chamfer3 = tf::chamfer_error(source_world.points() | tf::tag(T3), sphere.points() | tf::tag(tree_local) | tf::tag(T_target));
    REQUIRE(chamfer3 < real_t(0.01));

    // Case 4: Both have transforms
    // Target: local points + tree + transform + normals
    auto target4 = sphere.points() | tf::tag(tree_local) | tf::tag(T_target) | tf::tag_normals(normals_local);
    auto T4_delta = tf::fit_icp_alignment(
        sphere.points() | tf::tag(T_source),
        target4, config);
    auto T4_total = tf::transformed(T_source, T4_delta);
    real_t chamfer4 = tf::chamfer_error(sphere.points() | tf::tag(T4_total), sphere.points() | tf::tag(tree_local) | tf::tag(T_target));
    REQUIRE(chamfer4 < real_t(0.01));
}
