/**
 * @file test_isocontours.cpp
 * @brief Tests for scalar field isocontour extraction
 *
 * Tests for:
 * - make_isocontours
 *
 * Copyright (c) 2025 Ziga Sajovic, XLAB
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <trueform/trueform.hpp>
#include "type_traits.hpp"
#include <cmath>
#include <vector>

// =============================================================================
// Helper functions
// =============================================================================

template <typename Index, typename Real>
auto create_horizontal_plane(Real z_height) -> tf::polygons_buffer<Index, Real, 3, 4> {
    tf::polygons_buffer<Index, Real, 3, 4> result;

    result.points_buffer().emplace_back(Real(-2), Real(-2), z_height);
    result.points_buffer().emplace_back(Real(2), Real(-2), z_height);
    result.points_buffer().emplace_back(Real(2), Real(2), z_height);
    result.points_buffer().emplace_back(Real(-2), Real(2), z_height);

    result.faces_buffer().emplace_back(Index(0), Index(1), Index(2), Index(3));

    return result;
}

// =============================================================================
// Test 3.1: Sphere Latitude Lines
// =============================================================================

TEMPLATE_TEST_CASE("isocontours_sphere_latitude", "[isocontours]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto sphere = tf::make_sphere_mesh<index_t>(real_t(1), 100, 100);

    // Scalar field: z-coordinate
    std::vector<real_t> scalar_z;
    scalar_z.reserve(sphere.points().size());
    for (decltype(sphere.points().size()) i = 0; i < sphere.points().size(); ++i) {
        scalar_z.push_back(sphere.points()[i][2]);
    }

    // Isocontour at z=0.4 (latitude circle away from poles)
    auto contours = tf::make_isocontours(sphere.polygons(), tf::make_range(scalar_z), real_t(0.4));

    // 1 closed curve (latitude circle)
    REQUIRE(contours.paths().size() == 1);

    // Curve is closed
    const auto& path = contours.paths()[0];
    REQUIRE(path.front() == path.back());

    // Expected: z=0.4, radius^2 = 1 - 0.4^2 = 0.84
    real_t expected_z = real_t(0.4);
    real_t expected_r2 = real_t(1) - expected_z * expected_z;

    for (const auto& pt : contours.points()) {
        REQUIRE(std::abs(pt[2] - expected_z) < tf::epsilon<real_t>);
        real_t r2 = pt[0] * pt[0] + pt[1] * pt[1];
        REQUIRE(std::abs(r2 - expected_r2) < std::max(tf::epsilon<real_t>, real_t(0.002)));
    }
}

// =============================================================================
// Test 3.1b: Sphere Multiple Latitude Lines
// =============================================================================

TEMPLATE_TEST_CASE("isocontours_sphere_multiple_latitudes", "[isocontours]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto sphere = tf::make_sphere_mesh<index_t>(real_t(1), 100, 100);

    // Scalar field: z-coordinate
    std::vector<real_t> scalar_z;
    scalar_z.reserve(sphere.points().size());
    for (decltype(sphere.points().size()) i = 0; i < sphere.points().size(); ++i) {
        scalar_z.push_back(sphere.points()[i][2]);
    }

    // Multiple thresholds for latitude circles
    std::array<real_t, 3> thresholds = {real_t(-0.5), real_t(0), real_t(0.5)};
    auto contours = tf::make_isocontours(
        sphere.polygons(),
        tf::make_range(scalar_z),
        tf::make_range(thresholds));

    // 3 closed curves (one per latitude)
    REQUIRE(contours.paths().size() == 3);

    // Expected z-values and radii squared: r^2 = 1 - z^2
    std::array<real_t, 3> expected_z = {real_t(-0.5), real_t(0), real_t(0.5)};
    std::array<real_t, 3> expected_r2 = {
        real_t(1) - real_t(0.25),  // 0.75 at z=-0.5
        real_t(1),                  // 1.0 at z=0 (equator)
        real_t(1) - real_t(0.25)   // 0.75 at z=0.5
    };

    // Collect average z per curve to sort them
    std::vector<std::pair<real_t, std::size_t>> curve_z;
    for (std::size_t i = 0; i < contours.paths().size(); ++i) {
        real_t avg_z = real_t(0);
        for (auto idx : contours.paths()[i]) {
            avg_z += contours.points()[idx][2];
        }
        avg_z /= contours.paths()[i].size();
        curve_z.emplace_back(avg_z, i);
    }
    std::sort(curve_z.begin(), curve_z.end());

    // Verify each curve
    for (std::size_t i = 0; i < 3; ++i) {
        const auto& path = contours.paths()[curve_z[i].second];

        // Curve is closed
        REQUIRE(path.front() == path.back());

        // All points at correct z and radius
        for (auto idx : path) {
            const auto& pt = contours.points()[idx];
            REQUIRE(std::abs(pt[2] - expected_z[i]) < tf::epsilon<real_t>);
            real_t r2 = pt[0] * pt[0] + pt[1] * pt[1];
            REQUIRE(std::abs(r2 - expected_r2[i]) < std::max(tf::epsilon<real_t>, real_t(0.002)));
        }
    }
}

// =============================================================================
// Test 3.2: Sphere Distance Field
// =============================================================================

TEMPLATE_TEST_CASE("isocontours_sphere_distance_field", "[isocontours]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto sphere = tf::make_sphere_mesh<index_t>(real_t(1), 100, 100);

    // Scalar field: distance from plane z=0 (signed)
    std::vector<real_t> distance_z;
    distance_z.reserve(sphere.points().size());
    for (decltype(sphere.points().size()) i = 0; i < sphere.points().size(); ++i) {
        distance_z.push_back(sphere.points()[i][2]);
    }

    // Isocontour at z=0.3 (well inside the sphere, away from poles)
    auto contours = tf::make_isocontours(sphere.polygons(), tf::make_range(distance_z), real_t(0.3));

    // 1 closed curve (latitude circle)
    REQUIRE(contours.paths().size() == 1);

    // Curve is closed
    const auto& path = contours.paths()[0];
    REQUIRE(path.front() == path.back());

    // Expected: z=0.3, radius^2 = 1 - 0.3^2 = 0.91
    real_t expected_z = real_t(0.3);
    real_t expected_r2 = real_t(1) - expected_z * expected_z;

    for (const auto& pt : contours.points()) {
        REQUIRE(std::abs(pt[2] - expected_z) < tf::epsilon<real_t>);
        real_t r2 = pt[0] * pt[0] + pt[1] * pt[1];
        REQUIRE(std::abs(r2 - expected_r2) < std::max(tf::epsilon<real_t>, real_t(0.002)));
    }
}

// =============================================================================
// Test 3.3: Threshold Outside Range
// =============================================================================

TEMPLATE_TEST_CASE("isocontours_threshold_outside_range", "[isocontours]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto grid = tf::make_plane_mesh<index_t>(real_t(2), real_t(2), 10, 10);

    // Scalar field: x-coordinate (x in [-1, 1])
    std::vector<real_t> scalar_x;
    scalar_x.reserve(grid.points().size());
    for (decltype(grid.points().size()) i = 0; i < grid.points().size(); ++i) {
        scalar_x.push_back(grid.points()[i][0]);
    }

    // Threshold outside the scalar range
    auto contours = tf::make_isocontours(grid.polygons(), tf::make_range(scalar_x), real_t(5.0));

    // 0 curves since 5.0 is outside [-1, 1]
    REQUIRE(contours.paths().size() == 0);
}

// =============================================================================
// Test 3.4: Cross-Verification - Intersection Curves vs Isocontours
// =============================================================================

TEMPLATE_TEST_CASE("isocontours_cross_verify_with_intersection", "[isocontours]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto sphere = tf::make_sphere_mesh<index_t>(real_t(1), 100, 100);

    // Create horizontal plane at z=0.5
    auto h_plane = create_horizontal_plane<index_t, real_t>(real_t(0.5));

    // Method 1: Intersection curves
    auto curves_intersect = tf::make_intersection_curves(sphere.polygons(), h_plane.polygons());

    // Method 2: Distance field isocontour at threshold 0
    // Distance from plane z=0.5 is (z - 0.5)
    std::vector<real_t> distance_field;
    distance_field.reserve(sphere.points().size());
    for (decltype(sphere.points().size()) i = 0; i < sphere.points().size(); ++i) {
        distance_field.push_back(sphere.points()[i][2] - real_t(0.5));
    }
    auto curves_iso = tf::make_isocontours(sphere.polygons(), tf::make_range(distance_field), real_t(0));

    // Both methods produce 1 curve
    REQUIRE(curves_intersect.paths().size() == 1);
    REQUIRE(curves_iso.paths().size() == 1);

    // Expected values: z=0.5, radius^2 = 1 - 0.5^2 = 0.75
    real_t expected_z = real_t(0.5);
    real_t expected_r2 = real_t(0.75);

    // Verify intersection curves
    for (const auto& pt : curves_intersect.points()) {
        REQUIRE(std::abs(pt[2] - expected_z) < tf::epsilon<real_t>);
        real_t r2 = pt[0] * pt[0] + pt[1] * pt[1];
        REQUIRE(std::abs(r2 - expected_r2) < std::max(tf::epsilon<real_t>, real_t(0.002)));
    }

    // Verify isocontours
    for (const auto& pt : curves_iso.points()) {
        REQUIRE(std::abs(pt[2] - expected_z) < tf::epsilon<real_t>);
        real_t r2 = pt[0] * pt[0] + pt[1] * pt[1];
        REQUIRE(std::abs(r2 - expected_r2) < std::max(tf::epsilon<real_t>, real_t(0.002)));
    }
}

// =============================================================================
// Test 3.5: Single Threshold Value
// =============================================================================

TEMPLATE_TEST_CASE("isocontours_single_threshold", "[isocontours]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto sphere = tf::make_sphere_mesh<index_t>(real_t(1), 100, 100);

    // Scalar field: z-coordinate
    std::vector<real_t> scalar_z;
    scalar_z.reserve(sphere.points().size());
    for (decltype(sphere.points().size()) i = 0; i < sphere.points().size(); ++i) {
        scalar_z.push_back(sphere.points()[i][2]);
    }

    // Single threshold at z=0 (equator)
    auto contours = tf::make_isocontours(sphere.polygons(), tf::make_range(scalar_z), real_t(0));

    // 1 curve (the equator)
    REQUIRE(contours.paths().size() == 1);

    // Expected: z=0, radius=1
    for (const auto& pt : contours.points()) {
        REQUIRE(std::abs(pt[2]) < tf::epsilon<real_t>);
        real_t r2 = pt[0] * pt[0] + pt[1] * pt[1];
        REQUIRE(std::abs(r2 - real_t(1)) < std::max(tf::epsilon<real_t>, real_t(0.002)));
    }
}

// =============================================================================
// Test 3.6: Isocontours at Boundary Values
// =============================================================================

TEMPLATE_TEST_CASE("isocontours_at_boundaries", "[isocontours]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto grid = tf::make_plane_mesh<index_t>(real_t(2), real_t(2), 10, 10);

    // Scalar field: x-coordinate (x in [-1, 1])
    std::vector<real_t> scalar_x;
    scalar_x.reserve(grid.points().size());
    for (decltype(grid.points().size()) i = 0; i < grid.points().size(); ++i) {
        scalar_x.push_back(grid.points()[i][0]);
    }

    // Threshold at the minimum boundary (-1)
    auto contours_min = tf::make_isocontours(grid.polygons(), tf::make_range(scalar_x), real_t(-1));

    // Threshold at the maximum boundary (1)
    auto contours_max = tf::make_isocontours(grid.polygons(), tf::make_range(scalar_x), real_t(1));

    // At exact boundary values, contours exist at the edges
    if (contours_min.paths().size() > 0) {
        for (const auto& pt : contours_min.points()) {
            REQUIRE(std::abs(pt[0] - real_t(-1)) < tf::epsilon<real_t>);
        }
    }

    if (contours_max.paths().size() > 0) {
        for (const auto& pt : contours_max.points()) {
            REQUIRE(std::abs(pt[0] - real_t(1)) < tf::epsilon<real_t>);
        }
    }
}

// =============================================================================
// Test 3.7: On-grid single threshold (cut value lands exactly on vertices)
// =============================================================================

TEMPLATE_TEST_CASE("isocontours_on_grid_single_threshold", "[isocontours]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    // Plane mesh: x in [-2, 2], y in [-2, 2], z = 0
    // 20 subdivisions → x-coords: -2.0, -1.8, ..., 0.0, ..., 1.8, 2.0
    auto grid = tf::make_plane_mesh<index_t>(real_t(4), real_t(4), 20, 20);

    // Scalar field: x-coordinate
    std::vector<real_t> scalar_x;
    scalar_x.reserve(grid.points().size());
    for (decltype(grid.points().size()) i = 0; i < grid.points().size(); ++i) {
        scalar_x.push_back(grid.points()[i][0]);
    }

    // Threshold at x=0 lands exactly on grid vertices
    auto contours = tf::make_isocontours(grid.polygons(), tf::make_range(scalar_x), real_t(0));

    // Should produce curves (previously returned 0 with strict inequalities)
    REQUIRE(contours.paths().size() >= 1);
    REQUIRE(contours.points().size() > 0);

    // All intersection points should be at x ≈ 0
    for (const auto& pt : contours.points()) {
        REQUIRE(std::abs(pt[0]) < tf::epsilon<real_t>);
    }

    // Points should span the full y range
    real_t min_y = std::numeric_limits<real_t>::max();
    real_t max_y = std::numeric_limits<real_t>::lowest();
    for (const auto& pt : contours.points()) {
        min_y = std::min(min_y, pt[1]);
        max_y = std::max(max_y, pt[1]);
    }
    REQUIRE(min_y <= real_t(-1.9));
    REQUIRE(max_y >= real_t(1.9));
}

// =============================================================================
// Test 3.8: On-grid multiple thresholds
// =============================================================================

TEMPLATE_TEST_CASE("isocontours_on_grid_multiple_thresholds", "[isocontours]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto grid = tf::make_plane_mesh<index_t>(real_t(4), real_t(4), 20, 20);

    std::vector<real_t> scalar_x;
    scalar_x.reserve(grid.points().size());
    for (decltype(grid.points().size()) i = 0; i < grid.points().size(); ++i) {
        scalar_x.push_back(grid.points()[i][0]);
    }

    // Multiple on-grid thresholds at x = -1, 0, 1
    std::array<real_t, 3> thresholds = {real_t(-1), real_t(0), real_t(1)};
    auto contours = tf::make_isocontours(
        grid.polygons(),
        tf::make_range(scalar_x),
        tf::make_range(thresholds));

    // Should produce 3 curves (one per threshold)
    REQUIRE(contours.paths().size() == 3);

    // Collect average x per curve to sort them
    std::vector<std::pair<real_t, std::size_t>> curve_x;
    for (std::size_t i = 0; i < contours.paths().size(); ++i) {
        real_t avg_x = real_t(0);
        for (auto idx : contours.paths()[i]) {
            avg_x += contours.points()[idx][0];
        }
        avg_x /= contours.paths()[i].size();
        curve_x.emplace_back(avg_x, i);
    }
    std::sort(curve_x.begin(), curve_x.end());

    // Each curve should be at its respective x threshold
    std::array<real_t, 3> expected_x = {real_t(-1), real_t(0), real_t(1)};
    for (std::size_t i = 0; i < 3; ++i) {
        for (auto idx : contours.paths()[curve_x[i].second]) {
            const auto& pt = contours.points()[idx];
            REQUIRE(std::abs(pt[0] - expected_x[i]) < tf::epsilon<real_t>);
        }
    }
}

// =============================================================================
// Test 3.9: Half-open boundary convention
// =============================================================================

TEMPLATE_TEST_CASE("isocontours_half_open_boundary", "[isocontours]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto grid = tf::make_plane_mesh<index_t>(real_t(4), real_t(4), 20, 20);

    // Scalar field: x-coordinate (range [-2, 2])
    std::vector<real_t> scalar_x;
    scalar_x.reserve(grid.points().size());
    for (decltype(grid.points().size()) i = 0; i < grid.points().size(); ++i) {
        scalar_x.push_back(grid.points()[i][0]);
    }

    // Threshold at minimum (-2): half-open [min, max) should fire on edges
    // going from x=-2 to x>-2
    auto contours_min = tf::make_isocontours(grid.polygons(), tf::make_range(scalar_x), real_t(-2));
    REQUIRE(contours_min.paths().size() >= 1);
    for (const auto& pt : contours_min.points()) {
        REQUIRE(std::abs(pt[0] - real_t(-2)) < tf::epsilon<real_t>);
    }

    // Threshold at maximum (2): cut < max fails since max == cut → 0 curves
    auto contours_max = tf::make_isocontours(grid.polygons(), tf::make_range(scalar_x), real_t(2));
    REQUIRE(contours_max.paths().size() == 0);
}

// =============================================================================
// Test 3.10: Basic isobands extraction
// =============================================================================

TEMPLATE_TEST_CASE("isobands_basic", "[isobands]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto grid = tf::make_plane_mesh<index_t>(real_t(4), real_t(4), 20, 20);

    // Scalar field: x-coordinate (range [-2, 2])
    std::vector<real_t> scalar_x;
    scalar_x.reserve(grid.points().size());
    for (decltype(grid.points().size()) i = 0; i < grid.points().size(); ++i) {
        scalar_x.push_back(grid.points()[i][0]);
    }

    // Cut values at -0.5 and 0.5 → 3 bands:
    //   band 0: x <= -0.5
    //   band 1: -0.5 < x <= 0.5
    //   band 2: x > 0.5
    std::array<real_t, 2> cut_values = {real_t(-0.5), real_t(0.5)};
    std::array<index_t, 1> selected = {index_t(1)}; // middle band

    auto [mesh, labels] = tf::make_isobands(
        grid.polygons(),
        tf::make_range(scalar_x),
        tf::make_range(cut_values),
        tf::make_range(selected));

    // Output mesh should have faces
    REQUIRE(mesh.polygons().size() > 0);

    // Labels size matches face count
    REQUIRE(labels.size() == mesh.polygons().size());

    // All labels should be 1 (the selected band)
    for (decltype(labels.size()) i = 0; i < labels.size(); ++i) {
        REQUIRE(labels[i] == index_t(1));
    }

    // All output vertices should have x in [-0.5, 0.5] (within tolerance)
    for (const auto& pt : mesh.points()) {
        REQUIRE(pt[0] >= real_t(-0.5) - tf::epsilon<real_t>);
        REQUIRE(pt[0] <= real_t(0.5) + tf::epsilon<real_t>);
    }
}

// =============================================================================
// Test 3.11: Isobands with on-grid thresholds
// =============================================================================

TEMPLATE_TEST_CASE("isobands_on_grid_thresholds", "[isobands]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto grid = tf::make_plane_mesh<index_t>(real_t(4), real_t(4), 20, 20);

    // Scalar field: x-coordinate
    std::vector<real_t> scalar_x;
    scalar_x.reserve(grid.points().size());
    for (decltype(grid.points().size()) i = 0; i < grid.points().size(); ++i) {
        scalar_x.push_back(grid.points()[i][0]);
    }

    // On-grid cut values at -1, 0, 1 → 4 bands:
    //   band 0: x <= -1
    //   band 1: -1 < x <= 0
    //   band 2: 0 < x <= 1
    //   band 3: x > 1
    std::array<real_t, 3> cut_values = {real_t(-1), real_t(0), real_t(1)};
    std::array<index_t, 2> selected = {index_t(1), index_t(2)};

    auto [mesh, labels] = tf::make_isobands(
        grid.polygons(),
        tf::make_range(scalar_x),
        tf::make_range(cut_values),
        tf::make_range(selected));

    REQUIRE(mesh.polygons().size() > 0);
    REQUIRE(labels.size() == mesh.polygons().size());

    // Labels should be either 1 or 2
    for (decltype(labels.size()) i = 0; i < labels.size(); ++i) {
        REQUIRE((labels[i] == index_t(1) || labels[i] == index_t(2)));
    }

    // All output vertices should have x in [-1, 1] (within tolerance)
    for (const auto& pt : mesh.points()) {
        REQUIRE(pt[0] >= real_t(-1) - tf::epsilon<real_t>);
        REQUIRE(pt[0] <= real_t(1) + tf::epsilon<real_t>);
    }
}

// =============================================================================
// Test 3.12: Isobands with boundary curves
// =============================================================================

TEMPLATE_TEST_CASE("isobands_with_curves", "[isobands]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto grid = tf::make_plane_mesh<index_t>(real_t(4), real_t(4), 20, 20);

    std::vector<real_t> scalar_x;
    scalar_x.reserve(grid.points().size());
    for (decltype(grid.points().size()) i = 0; i < grid.points().size(); ++i) {
        scalar_x.push_back(grid.points()[i][0]);
    }

    std::array<real_t, 2> cut_values = {real_t(-0.5), real_t(0.5)};
    std::array<index_t, 1> selected = {index_t(1)};

    auto [mesh, labels, curves] = tf::make_isobands(
        grid.polygons(),
        tf::make_range(scalar_x),
        tf::make_range(cut_values),
        tf::make_range(selected),
        tf::return_curves);

    REQUIRE(mesh.polygons().size() > 0);

    // Boundary curves should exist (two cut boundaries)
    REQUIRE(curves.paths().size() >= 1);
    REQUIRE(curves.points().size() > 0);

    // Curve points should be at the cut value boundaries
    for (const auto& pt : curves.points()) {
        bool at_left = std::abs(pt[0] - real_t(-0.5)) < tf::epsilon<real_t>;
        bool at_right = std::abs(pt[0] - real_t(0.5)) < tf::epsilon<real_t>;
        REQUIRE((at_left || at_right));
    }
}
