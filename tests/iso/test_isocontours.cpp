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
#include <array>
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
// Test 3.9b: Dense cut values (multiple values crossing one face)
// =============================================================================

TEMPLATE_TEST_CASE("isocontours_dense_values_closed", "[isocontours]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto sphere = tf::make_sphere_mesh<index_t>(real_t(1), 64, 64);

    // Tilted plane distance: faces span several cut values at this density
    std::vector<real_t> scalars;
    scalars.reserve(sphere.points().size());
    for (decltype(sphere.points().size()) i = 0; i < sphere.points().size(); ++i) {
        const auto& pt = sphere.points()[i];
        scalars.push_back(real_t(0.62) * pt[0] + real_t(0.31) * pt[1] +
                          real_t(0.72) * pt[2] + real_t(2));
    }
    auto [min_it, max_it] = std::minmax_element(scalars.begin(), scalars.end());

    std::vector<real_t> values;
    for (int k = 0; k < 32; ++k)
        values.push_back(real_t(1) + real_t(2) * (real_t(k) + real_t(0.5)) / real_t(32));

    std::size_t expected = 0;
    for (auto v : values)
        expected += (v > *min_it && v < *max_it);

    auto contours = tf::make_isocontours(
        sphere.polygons(), tf::make_range(scalars), tf::make_range(values));

    REQUIRE(contours.paths().size() == expected);
    for (const auto& path : contours.paths())
        REQUIRE(path.front() == path.back());

    // Every intersection point sits on the plane of some cut value
    for (const auto& pt : contours.points()) {
        real_t s = real_t(0.62) * pt[0] + real_t(0.31) * pt[1] +
                   real_t(0.72) * pt[2] + real_t(2);
        real_t best = std::numeric_limits<real_t>::max();
        for (auto v : values)
            best = std::min(best, std::abs(s - v));
        REQUIRE(best < std::max(tf::epsilon<real_t>, real_t(1e-4)));
    }
}

// =============================================================================
// Test 3.9b2: Duplicate cut values collapse to one contour
// =============================================================================

TEMPLATE_TEST_CASE("isocontours_duplicate_values", "[isocontours]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto sphere = tf::make_sphere_mesh<index_t>(real_t(1), 100, 100);

    std::vector<real_t> scalar_z;
    scalar_z.reserve(sphere.points().size());
    for (decltype(sphere.points().size()) i = 0; i < sphere.points().size(); ++i) {
        scalar_z.push_back(sphere.points()[i][2]);
    }

    std::array<real_t, 3> values = {real_t(0.4), real_t(0.4), real_t(-0.3)};
    auto contours = tf::make_isocontours(
        sphere.polygons(), tf::make_range(scalar_z), tf::make_range(values));

    REQUIRE(contours.paths().size() == 2);
    for (const auto& path : contours.paths())
        REQUIRE(path.front() == path.back());
}

// =============================================================================
// Test 3.9c: Cut values landing exactly on vertices
// =============================================================================

TEMPLATE_TEST_CASE("isocontours_on_vertex_values_closed", "[isocontours]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto sphere = tf::make_sphere_mesh<index_t>(real_t(1), 64, 64);

    std::vector<real_t> scalars;
    scalars.reserve(sphere.points().size());
    for (decltype(sphere.points().size()) i = 0; i < sphere.points().size(); ++i) {
        const auto& pt = sphere.points()[i];
        scalars.push_back(real_t(0.62) * pt[0] + real_t(0.31) * pt[1] +
                          real_t(0.72) * pt[2] + real_t(2));
    }

    // Cut exactly at vertex scalars spread over the sphere
    std::vector<real_t> values;
    for (int k = 0; k < 16; ++k)
        values.push_back(scalars[std::size_t(k * 97 + 11) % scalars.size()]);
    std::sort(values.begin(), values.end());
    values.erase(std::unique(values.begin(), values.end()), values.end());

    auto contours = tf::make_isocontours(
        sphere.polygons(), tf::make_range(scalars), tf::make_range(values));

    REQUIRE(contours.paths().size() == values.size());
    for (const auto& path : contours.paths())
        REQUIRE(path.front() == path.back());
}

// =============================================================================
// Test 3.9d: Whole mesh edges lying on the isovalue (ring of a sphere)
// =============================================================================

TEMPLATE_TEST_CASE("isocontours_edge_on_isovalue", "[isocontours]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto sphere = tf::make_sphere_mesh<index_t>(real_t(1), 100, 100);

    // Scalar field: z-coordinate; latitude rings share one exact z value
    std::vector<real_t> scalar_z;
    scalar_z.reserve(sphere.points().size());
    for (decltype(sphere.points().size()) i = 0; i < sphere.points().size(); ++i) {
        scalar_z.push_back(sphere.points()[i][2]);
    }

    // Pick a mid-latitude vertex: the whole ring of edges lies on this value
    real_t ring_z = scalar_z[scalar_z.size() / 2];
    auto contours = tf::make_isocontours(
        sphere.polygons(), tf::make_range(scalar_z), ring_z);

    REQUIRE(contours.paths().size() == 1);
    const auto& path = contours.paths()[0];
    REQUIRE(path.front() == path.back());

    for (const auto& pt : contours.points()) {
        REQUIRE(std::abs(pt[2] - ring_z) < tf::epsilon<real_t>);
    }
}

// =============================================================================
// Test 3.9e: Isovalue edge with both incident faces above (valley line)
// =============================================================================

TEMPLATE_TEST_CASE("isocontours_valley_edge_on_isovalue", "[isocontours]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    // Edge (A,B) sits exactly on the isovalue; both apexes are above, so
    // both faces emit the same segment — it must come out once, not doubled.
    tf::polygons_buffer<index_t, real_t, 3, 3> mesh;
    mesh.points_buffer().emplace_back(real_t(0), real_t(0), real_t(0));
    mesh.points_buffer().emplace_back(real_t(1), real_t(0), real_t(0));
    mesh.points_buffer().emplace_back(real_t(0.5), real_t(1), real_t(0));
    mesh.points_buffer().emplace_back(real_t(0.5), real_t(-1), real_t(0));
    mesh.faces_buffer().emplace_back(index_t(0), index_t(1), index_t(2));
    mesh.faces_buffer().emplace_back(index_t(1), index_t(0), index_t(3));

    std::array<real_t, 4> scalars = {real_t(1), real_t(1), real_t(2), real_t(2)};

    auto contours = tf::make_isocontours(
        mesh.polygons(), tf::make_range(scalars), real_t(1));

    REQUIRE(contours.paths().size() == 1);
    REQUIRE(contours.paths()[0].size() == 2);
    for (const auto& pt : contours.points()) {
        REQUIRE(std::abs(pt[1]) < tf::epsilon<real_t>);
    }
}

// =============================================================================
// Test 3.9f: Quad with a vertex touch — the touch joins no chord
// =============================================================================

TEMPLATE_TEST_CASE("isocontours_quad_vertex_touch", "[isocontours]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    // v3 sits exactly on the cut with both neighbors above: the level set
    // touches the face there but the contour chord runs P01 - P12.
    tf::polygons_buffer<index_t, real_t, 3, 4> mesh;
    mesh.points_buffer().emplace_back(real_t(0), real_t(0), real_t(0));
    mesh.points_buffer().emplace_back(real_t(1), real_t(0), real_t(0));
    mesh.points_buffer().emplace_back(real_t(1), real_t(1), real_t(0));
    mesh.points_buffer().emplace_back(real_t(0), real_t(1), real_t(0));
    mesh.faces_buffer().emplace_back(index_t(0), index_t(1), index_t(2),
                                     index_t(3));
    std::array<real_t, 4> scalars = {real_t(2), real_t(0), real_t(2),
                                     real_t(1)};

    auto contours = tf::make_isocontours(
        mesh.polygons(), tf::make_range(scalars), real_t(1));

    REQUIRE(contours.paths().size() == 1);
    REQUIRE(contours.paths()[0].size() == 2);
    for (const auto& pt : contours.points()) {
        bool at_p01 = std::abs(pt[0] - real_t(0.5)) < tf::epsilon<real_t> &&
                      std::abs(pt[1]) < tf::epsilon<real_t>;
        bool at_p12 = std::abs(pt[0] - real_t(1)) < tf::epsilon<real_t> &&
                      std::abs(pt[1] - real_t(0.5)) < tf::epsilon<real_t>;
        REQUIRE((at_p01 || at_p12));
    }
}

// =============================================================================
// Test 3.9g: Pentagon with a flat edge on the cut plus a transversal chord
// =============================================================================

TEMPLATE_TEST_CASE("isocontours_pentagon_flat_edge_and_chord", "[isocontours]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    // Edge (v0,v1) lies on the cut; v3 dips below: the contour is the flat
    // edge plus the chord P34 - P23 cutting off the v3 corner.
    tf::polygons_buffer<index_t, real_t, 3, tf::dynamic_size> mesh;
    mesh.points_buffer().emplace_back(real_t(0), real_t(0), real_t(0));
    mesh.points_buffer().emplace_back(real_t(0), real_t(1), real_t(0));
    mesh.points_buffer().emplace_back(real_t(2), real_t(1), real_t(0));
    mesh.points_buffer().emplace_back(real_t(3), real_t(0.5), real_t(0));
    mesh.points_buffer().emplace_back(real_t(2), real_t(0), real_t(0));
    std::array<index_t, 5> face = {index_t(0), index_t(1), index_t(2),
                                   index_t(3), index_t(4)};
    mesh.faces_buffer().push_back(
        tf::make_range(face.begin(), face.end()));
    std::array<real_t, 5> scalars = {real_t(1), real_t(1), real_t(2),
                                     real_t(0), real_t(2)};

    auto contours = tf::make_isocontours(
        mesh.polygons(), tf::make_range(scalars), real_t(1));

    REQUIRE(contours.paths().size() == 2);
    for (const auto& path : contours.paths())
        REQUIRE(path.size() == 2);
    for (const auto& pt : contours.points()) {
        bool on_flat_edge = pt[0] == real_t(0);
        bool on_chord = std::abs(pt[0] - real_t(2.5)) < tf::epsilon<real_t>;
        REQUIRE((on_flat_edge || on_chord));
    }
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

    auto [mesh, labels, fl_] = tf::make_isobands(
        grid.polygons(),
        tf::make_range(scalar_x),
        tf::make_range(cut_values),
        tf::make_range(selected));

    // Output mesh should have faces
    REQUIRE(mesh.polygons().size() > 0);

    // Labels size matches face count
    REQUIRE(labels.size() == mesh.polygons().size());
    REQUIRE(fl_.size() == mesh.polygons().size());

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

    auto [mesh, labels, fl_] = tf::make_isobands(
        grid.polygons(),
        tf::make_range(scalar_x),
        tf::make_range(cut_values),
        tf::make_range(selected));

    REQUIRE(mesh.polygons().size() > 0);
    REQUIRE(labels.size() == mesh.polygons().size());
    REQUIRE(fl_.size() == mesh.polygons().size());

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
// Test 3.11b: Embedded isocurves with dense values (multi-cut faces)
// =============================================================================

TEMPLATE_TEST_CASE("embedded_isocurves_dense_values", "[isobands]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto grid = tf::make_plane_mesh<index_t>(real_t(4), real_t(4), 20, 20);

    // Tilted field: faces span several of the 16 cut values
    auto field = [](const auto& pt) {
        return pt[0] + real_t(0.25) * pt[1] + real_t(2);
    };
    std::vector<real_t> scalars;
    scalars.reserve(grid.points().size());
    for (decltype(grid.points().size()) i = 0; i < grid.points().size(); ++i) {
        scalars.push_back(field(grid.points()[i]));
    }

    std::vector<real_t> values;
    for (int k = 0; k < 16; ++k)
        values.push_back(real_t(0.5) + real_t(3) * (real_t(k) + real_t(0.5)) / real_t(16));

    auto [mesh, labels, fl_] = tf::embedded_isocurves(
        grid.polygons(), tf::make_range(scalars), tf::make_range(values));

    REQUIRE(mesh.polygons().size() > 0);
    REQUIRE(labels.size() == mesh.polygons().size());

    // The recut mesh covers the grid exactly
    real_t total_area = real_t(0);
    for (const auto& poly : mesh.polygons())
        total_area += tf::area(poly);
    REQUIRE(std::abs(total_area - real_t(16)) < real_t(1e-3));

    // Every face lies inside its band: all vertices within the band's range
    real_t tol = real_t(1e-4);
    for (decltype(mesh.polygons().size()) i = 0; i < mesh.polygons().size(); ++i) {
        auto band = labels[i];
        real_t lo = band == 0 ? std::numeric_limits<real_t>::lowest()
                              : values[std::size_t(band) - 1];
        real_t hi = std::size_t(band) == values.size()
                        ? std::numeric_limits<real_t>::max()
                        : values[std::size_t(band)];
        const auto& poly = mesh.polygons()[i];
        for (std::size_t j = 0; j < std::size_t(poly.size()); ++j) {
            real_t s = field(poly[j]);
            REQUIRE(s >= lo - tol);
            REQUIRE(s <= hi + tol);
        }
    }
}

// =============================================================================
// Test 3.11c: Isobands with dense values select the right region
// =============================================================================

TEMPLATE_TEST_CASE("isobands_dense_values", "[isobands]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto grid = tf::make_plane_mesh<index_t>(real_t(4), real_t(4), 20, 20);

    auto field = [](const auto& pt) {
        return pt[0] + real_t(0.25) * pt[1] + real_t(2);
    };
    std::vector<real_t> scalars;
    scalars.reserve(grid.points().size());
    for (decltype(grid.points().size()) i = 0; i < grid.points().size(); ++i) {
        scalars.push_back(field(grid.points()[i]));
    }

    std::vector<real_t> values;
    for (int k = 0; k < 16; ++k)
        values.push_back(real_t(0.5) + real_t(3) * (real_t(k) + real_t(0.5)) / real_t(16));

    // One band in the middle: values[7] < s <= values[8]
    std::array<index_t, 1> selected = {index_t(8)};
    auto [mesh, labels, fl_] = tf::make_isobands(
        grid.polygons(), tf::make_range(scalars), tf::make_range(values),
        tf::make_range(selected));

    REQUIRE(mesh.polygons().size() > 0);

    // Analytic area of the strip values[7] < x + 0.25y + 2 <= values[8]
    // inside the square: full-height strip, width = dv / |d s/d x| per y row
    // integrated: dv * 4 / 1 with the 0.25y shift staying inside x range.
    real_t dv = values[8] - values[7];
    real_t expected_area = dv * real_t(4);

    real_t total_area = real_t(0);
    for (const auto& poly : mesh.polygons())
        total_area += tf::area(poly);
    REQUIRE(std::abs(total_area - expected_area) < real_t(1e-3));

    real_t tol = real_t(1e-4);
    for (const auto& pt : mesh.points()) {
        real_t s = field(pt);
        REQUIRE(s >= values[7] - tol);
        REQUIRE(s <= values[8] + tol);
    }
}

// =============================================================================
// Test 3.11d: Dense values on a quad grid (multi-chord quad cutting)
// =============================================================================

TEMPLATE_TEST_CASE("embedded_isocurves_dense_values_quads", "[isobands]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    // 10x10 quad grid over [-2,2]^2
    tf::polygons_buffer<index_t, real_t, 3, 4> grid;
    const int n = 10;
    for (int j = 0; j <= n; ++j)
        for (int i = 0; i <= n; ++i)
            grid.points_buffer().emplace_back(
                real_t(-2) + real_t(4) * real_t(i) / real_t(n),
                real_t(-2) + real_t(4) * real_t(j) / real_t(n), real_t(0));
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) {
            index_t v = index_t(j * (n + 1) + i);
            grid.faces_buffer().emplace_back(v, v + 1, v + index_t(n) + 2,
                                             v + index_t(n) + 1);
        }

    auto field = [](const auto& pt) {
        return pt[0] + real_t(0.25) * pt[1] + real_t(2);
    };
    std::vector<real_t> scalars;
    scalars.reserve(grid.points().size());
    for (decltype(grid.points().size()) i = 0; i < grid.points().size(); ++i) {
        scalars.push_back(field(grid.points()[i]));
    }

    // Cell scalar span is ~0.5; spacing ~0.19 puts 2-3 cuts through a quad
    std::vector<real_t> values;
    for (int k = 0; k < 16; ++k)
        values.push_back(real_t(0.5) + real_t(3) * (real_t(k) + real_t(0.5)) / real_t(16));

    auto [mesh, labels, fl_] = tf::embedded_isocurves(
        grid.polygons(), tf::make_range(scalars), tf::make_range(values));

    real_t total_area = real_t(0);
    for (const auto& poly : mesh.polygons())
        total_area += tf::area(poly);
    REQUIRE(std::abs(total_area - real_t(16)) < real_t(1e-3));

    real_t tol = real_t(1e-4);
    for (decltype(mesh.polygons().size()) i = 0; i < mesh.polygons().size(); ++i) {
        auto band = labels[i];
        real_t lo = band == 0 ? std::numeric_limits<real_t>::lowest()
                              : values[std::size_t(band) - 1];
        real_t hi = std::size_t(band) == values.size()
                        ? std::numeric_limits<real_t>::max()
                        : values[std::size_t(band)];
        const auto& poly = mesh.polygons()[i];
        for (std::size_t j = 0; j < std::size_t(poly.size()); ++j) {
            real_t s = field(poly[j]);
            REQUIRE(s >= lo - tol);
            REQUIRE(s <= hi + tol);
        }
    }
}

// =============================================================================
// Test 3.11e: Dense values on a closed mesh preserve total area
// =============================================================================

TEMPLATE_TEST_CASE("embedded_isocurves_dense_values_sphere", "[isobands]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto sphere = tf::make_sphere_mesh<index_t>(real_t(1), 48, 48);

    std::vector<real_t> scalars;
    scalars.reserve(sphere.points().size());
    for (decltype(sphere.points().size()) i = 0; i < sphere.points().size(); ++i) {
        const auto& pt = sphere.points()[i];
        scalars.push_back(real_t(0.62) * pt[0] + real_t(0.31) * pt[1] +
                          real_t(0.72) * pt[2] + real_t(2));
    }

    std::vector<real_t> values;
    for (int k = 0; k < 24; ++k)
        values.push_back(real_t(1) + real_t(2) * (real_t(k) + real_t(0.5)) / real_t(24));

    real_t input_area = real_t(0);
    for (const auto& poly : sphere.polygons())
        input_area += tf::area(poly);

    auto [mesh, labels, fl_] = tf::embedded_isocurves(
        sphere.polygons(), tf::make_range(scalars), tf::make_range(values));

    real_t output_area = real_t(0);
    for (const auto& poly : mesh.polygons())
        output_area += tf::area(poly);
    REQUIRE(std::abs(output_area - input_area) < input_area * real_t(1e-4));

    for (decltype(labels.size()) i = 0; i < labels.size(); ++i) {
        REQUIRE(labels[i] >= 0);
        REQUIRE(std::size_t(labels[i]) <= values.size());
    }
}

// =============================================================================
// Test 3.11e2: Embedded isocurves with curve output on dense values
// =============================================================================

TEMPLATE_TEST_CASE("embedded_isocurves_dense_values_with_curves", "[isobands]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto sphere = tf::make_sphere_mesh<index_t>(real_t(1), 48, 48);

    std::vector<real_t> scalars;
    scalars.reserve(sphere.points().size());
    for (decltype(sphere.points().size()) i = 0; i < sphere.points().size(); ++i) {
        const auto& pt = sphere.points()[i];
        scalars.push_back(real_t(0.62) * pt[0] + real_t(0.31) * pt[1] +
                          real_t(0.72) * pt[2] + real_t(2));
    }

    std::vector<real_t> values;
    for (int k = 0; k < 16; ++k)
        values.push_back(real_t(1) + real_t(2) * (real_t(k) + real_t(0.5)) / real_t(16));

    auto [mesh, labels, fl_, curves] = tf::embedded_isocurves(
        sphere.polygons(), tf::make_range(scalars), tf::make_range(values),
        tf::return_curves);

    REQUIRE(mesh.polygons().size() > 0);

    // One closed boundary circle per cut value on the sphere
    REQUIRE(curves.paths().size() == values.size());
    for (const auto& path : curves.paths())
        REQUIRE(path.front() == path.back());
}

// =============================================================================
// Test 3.11f: Vertex-exact cut values classify touching faces correctly
// =============================================================================

TEMPLATE_TEST_CASE("embedded_isocurves_on_vertex_values", "[isobands]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto sphere = tf::make_sphere_mesh<index_t>(real_t(1), 48, 48);

    auto field = [](const auto& pt) {
        return real_t(0.62) * pt[0] + real_t(0.31) * pt[1] +
               real_t(0.72) * pt[2] + real_t(2);
    };
    std::vector<real_t> scalars;
    scalars.reserve(sphere.points().size());
    for (decltype(sphere.points().size()) i = 0; i < sphere.points().size(); ++i) {
        scalars.push_back(field(sphere.points()[i]));
    }

    // Cut exactly through vertex scalars
    std::vector<real_t> values;
    for (int k = 0; k < 8; ++k)
        values.push_back(scalars[std::size_t(k * 97 + 11) % scalars.size()]);
    std::sort(values.begin(), values.end());
    values.erase(std::unique(values.begin(), values.end()), values.end());

    auto [mesh, labels, fl_] = tf::embedded_isocurves(
        sphere.polygons(), tf::make_range(scalars), tf::make_range(values));

    real_t tol = real_t(1e-4);
    for (decltype(mesh.polygons().size()) i = 0; i < mesh.polygons().size(); ++i) {
        auto band = labels[i];
        real_t lo = band == 0 ? std::numeric_limits<real_t>::lowest()
                              : values[std::size_t(band) - 1];
        real_t hi = std::size_t(band) == values.size()
                        ? std::numeric_limits<real_t>::max()
                        : values[std::size_t(band)];
        const auto& poly = mesh.polygons()[i];
        for (std::size_t j = 0; j < std::size_t(poly.size()); ++j) {
            real_t s = field(poly[j]);
            REQUIRE(s >= lo - tol);
            REQUIRE(s <= hi + tol);
        }
    }
}

// =============================================================================
// Test 3.11g: A closed solid whose caps one cut value chords twice
// =============================================================================

namespace {

/// A U extruded to depth 1: two U caps of area 20 and eight side quads over a
/// perimeter of 24, so the surface is 64 and closed. A cap is non-convex, so
/// a level set above the notch floor meets its boundary four times and ONE
/// cut value states TWO chords on it. Every cap edge is shared with a side
/// quad, which is what makes the recut closed only if both carriers name a
/// field point identically.
template <typename Index, typename Real>
auto create_u_prism() -> tf::polygons_buffer<Index, Real, 3, tf::dynamic_size> {
    tf::polygons_buffer<Index, Real, 3, tf::dynamic_size> mesh;
    const std::array<std::array<Real, 2>, 8> corners{
        {{0, 0}, {6, 0}, {6, 4}, {4, 4}, {4, 2}, {2, 2}, {2, 4}, {0, 4}}};
    for (const auto& c : corners)
        mesh.points_buffer().emplace_back(c[0], c[1], Real(0));
    for (const auto& c : corners)
        mesh.points_buffer().emplace_back(c[0], c[1], Real(1));

    std::array<Index, 8> cap;
    for (int i = 0; i < 8; ++i)
        cap[std::size_t(i)] = Index(8 + i);
    mesh.faces_buffer().push_back(tf::make_range(cap.begin(), cap.end()));
    for (int i = 0; i < 8; ++i)
        cap[std::size_t(i)] = Index(7 - i);
    mesh.faces_buffer().push_back(tf::make_range(cap.begin(), cap.end()));

    for (int i = 0; i < 8; ++i) {
        const Index j = Index((i + 1) % 8);
        std::array<Index, 4> side{Index(i), j, Index(j + 8), Index(i + 8)};
        mesh.faces_buffer().push_back(
            tf::make_range(side.begin(), side.end()));
    }
    return mesh;
}

} // anonymous namespace

TEMPLATE_TEST_CASE("embedded_isocurves_multi_chord_face_closed", "[isobands]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto prism = create_u_prism<index_t, real_t>();
    REQUIRE(tf::is_closed(prism.polygons()));

    // The field is the height, so y = 3 chords each prong of a cap and y = 1
    // chords the bar: three chords from two values on one face.
    std::vector<real_t> scalars;
    scalars.reserve(prism.points().size());
    for (decltype(prism.points().size()) i = 0; i < prism.points().size(); ++i)
        scalars.push_back(prism.points()[i][1]);

    std::array<real_t, 2> values = {real_t(1), real_t(3)};

    auto [mesh, labels, face_labels, refused] = tf::embedded_isocurves(
        prism.polygons(), tf::make_range(scalars), tf::make_range(values),
        tf::return_refused);

    REQUIRE(refused.size() == 0);
    REQUIRE(labels.size() == mesh.polygons().size());
    REQUIRE(face_labels.size() == mesh.polygons().size());

    // A cap's field points sit on edges it shares with side quads, so the
    // recut closes only when one crossing is one id in both faces.
    REQUIRE(tf::is_closed(mesh.polygons()));

    // Band 0 is y <= 1, band 1 is 1 < y <= 3, band 2 is y > 3.
    std::array<real_t, 3> band_area{};
    for (decltype(mesh.polygons().size()) i = 0; i < mesh.polygons().size(); ++i) {
        REQUIRE(labels[i] >= index_t(0));
        REQUIRE(labels[i] <= index_t(2));
        band_area[std::size_t(labels[i])] += tf::area(mesh.polygons()[i]);
    }
    REQUIRE(std::abs(band_area[0] - real_t(20)) < real_t(1e-3));
    REQUIRE(std::abs(band_area[1] - real_t(28)) < real_t(1e-3));
    REQUIRE(std::abs(band_area[2] - real_t(16)) < real_t(1e-3));

    // The notch is a pocket of the cap's hull and survives in no band.
    REQUIRE(std::abs(band_area[0] + band_area[1] + band_area[2] - real_t(64)) <
            real_t(1e-3));

    real_t tol = real_t(1e-4);
    for (decltype(mesh.polygons().size()) i = 0; i < mesh.polygons().size(); ++i) {
        auto band = labels[i];
        real_t lo = band == 0 ? std::numeric_limits<real_t>::lowest()
                              : values[std::size_t(band) - 1];
        real_t hi = std::size_t(band) == values.size()
                        ? std::numeric_limits<real_t>::max()
                        : values[std::size_t(band)];
        const auto& poly = mesh.polygons()[i];
        for (std::size_t j = 0; j < std::size_t(poly.size()); ++j) {
            REQUIRE(poly[j][1] >= lo - tol);
            REQUIRE(poly[j][1] <= hi + tol);
        }
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

    auto [mesh, labels, fl_, curves] = tf::make_isobands(
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

// =============================================================================
// return_refused - the surface names whose emptiness it was
// =============================================================================

namespace {

/// A quad grid whose field crosses every cell, so every cut face declines the
/// one-chord split and takes the constrained build — the only state that can
/// refuse at all.
template <typename Index, typename Real>
auto create_field_grid(int n) -> tf::polygons_buffer<Index, Real, 3, 4> {
    tf::polygons_buffer<Index, Real, 3, 4> grid;
    for (int j = 0; j <= n; ++j)
        for (int i = 0; i <= n; ++i)
            grid.points_buffer().emplace_back(Real(i), Real(j), Real(0));
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) {
            Index v = Index(j * (n + 1) + i);
            grid.faces_buffer().emplace_back(v, v + 1, v + Index(n) + 2,
                                             v + Index(n) + 1);
        }
    return grid;
}

/// A face whose projection is degenerate, a face with coincident corners and a
/// face whose loop crosses itself — every shape that declines the one-chord
/// split — beside a healthy face, all of them crossed by the field.
template <typename Index, typename Real>
auto create_degenerate_ladder()
    -> tf::polygons_buffer<Index, Real, 3, tf::dynamic_size> {
    tf::polygons_buffer<Index, Real, 3, tf::dynamic_size> mesh;

    mesh.points_buffer().emplace_back(Real(0), Real(0), Real(0));   // 0
    mesh.points_buffer().emplace_back(Real(1), Real(0), Real(0));   // 1
    mesh.points_buffer().emplace_back(Real(2), Real(0), Real(0));   // 2
    mesh.points_buffer().emplace_back(Real(0), Real(2), Real(0));   // 3
    mesh.points_buffer().emplace_back(Real(4), Real(0), Real(0));   // 4
    mesh.points_buffer().emplace_back(Real(5), Real(0), Real(0));   // 5
    mesh.points_buffer().emplace_back(Real(4), Real(1), Real(0));   // 6
    mesh.points_buffer().emplace_back(Real(7), Real(0), Real(0));   // 7
    mesh.points_buffer().emplace_back(Real(9), Real(0), Real(0));   // 8
    mesh.points_buffer().emplace_back(Real(7), Real(1), Real(0));   // 9
    mesh.points_buffer().emplace_back(Real(8), Real(1), Real(0));   // 10

    mesh.faces_buffer().push_back({Index(0), Index(1), Index(2)});  // collinear
    mesh.faces_buffer().push_back({Index(0), Index(2), Index(3)});  // healthy
    mesh.faces_buffer().push_back(
        {Index(4), Index(5), Index(5), Index(6)});                  // coincident
    mesh.faces_buffer().push_back(
        {Index(7), Index(8), Index(9), Index(10)});                 // crossing

    return mesh;
}

} // anonymous namespace

TEMPLATE_TEST_CASE("embedded_isocurves_refused_empty_on_clean_input", "[isobands]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto grid = create_field_grid<index_t, real_t>(8);

    std::vector<real_t> scalars;
    scalars.reserve(grid.points().size());
    for (decltype(grid.points().size()) i = 0; i < grid.points().size(); ++i)
        scalars.push_back(grid.points()[i][0] + real_t(0.25) * grid.points()[i][1]);

    std::vector<real_t> values;
    for (int k = 0; k < 12; ++k)
        values.push_back(real_t(0.3) + real_t(0.71) * real_t(k));

    auto [plain, plain_labels, plain_face_labels] = tf::embedded_isocurves(
        grid.polygons(), tf::make_range(scalars), tf::make_range(values));
    auto [tagged, tagged_labels, tagged_face_labels, refused] =
        tf::embedded_isocurves(grid.polygons(), tf::make_range(scalars),
                               tf::make_range(values), tf::return_refused);

    REQUIRE(refused.size() == 0);

    // The tagged call answers exactly what the untagged one does.
    REQUIRE(tagged.faces().size() == plain.faces().size());
    REQUIRE(tagged.points().size() == plain.points().size());
    for (decltype(plain.faces().size()) f = 0; f < plain.faces().size(); ++f) {
        REQUIRE(tagged.faces()[f].size() == plain.faces()[f].size());
        for (std::size_t c = 0; c < std::size_t(plain.faces()[f].size()); ++c)
            REQUIRE(tagged.faces()[f][c] == plain.faces()[f][c]);
    }
    for (decltype(plain.points().size()) p = 0; p < plain.points().size(); ++p)
        for (std::size_t d = 0; d < 3; ++d)
            REQUIRE(tagged.points()[p][d] == plain.points()[p][d]);
    for (decltype(plain_labels.size()) i = 0; i < plain_labels.size(); ++i)
        REQUIRE(tagged_labels[i] == plain_labels[i]);
    for (decltype(plain_face_labels.size()) i = 0; i < plain_face_labels.size(); ++i)
        REQUIRE(tagged_face_labels[i] == plain_face_labels[i]);
}

TEMPLATE_TEST_CASE("make_isobands_refused_empty_on_clean_input", "[isobands]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto grid = create_field_grid<index_t, real_t>(6);

    std::vector<real_t> scalars;
    scalars.reserve(grid.points().size());
    for (decltype(grid.points().size()) i = 0; i < grid.points().size(); ++i)
        scalars.push_back(grid.points()[i][0] + grid.points()[i][1]);

    std::vector<real_t> values{real_t(2.5), real_t(5.5), real_t(8.5)};
    std::vector<index_t> bands{index_t(1), index_t(2)};

    auto [plain, plain_labels, plain_face_labels] = tf::make_isobands(
        grid.polygons(), tf::make_range(scalars), tf::make_range(values),
        tf::make_range(bands));
    auto [tagged, tagged_labels, tagged_face_labels, refused] = tf::make_isobands(
        grid.polygons(), tf::make_range(scalars), tf::make_range(values),
        tf::make_range(bands), tf::return_refused);

    REQUIRE(refused.size() == 0);
    REQUIRE(tagged.faces().size() == plain.faces().size());
    REQUIRE(tagged.points().size() == plain.points().size());
    for (decltype(plain.faces().size()) f = 0; f < plain.faces().size(); ++f)
        for (std::size_t c = 0; c < std::size_t(plain.faces()[f].size()); ++c)
            REQUIRE(tagged.faces()[f][c] == plain.faces()[f][c]);
    for (decltype(plain_labels.size()) i = 0; i < plain_labels.size(); ++i)
        REQUIRE(tagged_labels[i] == plain_labels[i]);
    for (decltype(plain_face_labels.size()) i = 0; i < plain_face_labels.size(); ++i)
        REQUIRE(tagged_face_labels[i] == plain_face_labels[i]);
}

TEMPLATE_TEST_CASE("embedded_isocurves_refused_empty_on_degenerate_faces", "[isobands]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    // The list speaks only on a build the constraint set defeats. This tier
    // states each face's own chords, which are level sets of the field on it:
    // they meet nothing to cross, so the resolving build admits every set it is
    // handed. A face whose projection is degenerate is therefore admitted and
    // states its emptiness — it holds no piece, and it is not refused.
    auto mesh = create_degenerate_ladder<index_t, real_t>();

    std::vector<real_t> scalars{real_t(0), real_t(1), real_t(2), real_t(2),
                                real_t(0), real_t(2), real_t(1),
                                real_t(0), real_t(2), real_t(0), real_t(1)};
    std::vector<real_t> values{real_t(0.5), real_t(1.5)};

    auto [tagged, labels, face_labels, refused] =
        tf::embedded_isocurves(mesh.polygons(), tf::make_range(scalars),
                               tf::make_range(values), tf::return_refused);

    REQUIRE(refused.size() == 0);
    REQUIRE(tagged.faces().size() > 0);
}

// =============================================================================
// The cut lattice is the one the input real resolves to
// =============================================================================

TEST_CASE("iso cut carries a band finer than a 31-bit lattice", "[isobands]")
{
    using index_t = int;
    using real_t = double;

    // The two cut values lie 1e-11 apart in a mesh of extent 1, so a 31-bit
    // lattice snaps both of the face's crossing pairs onto one site each and
    // the band between them cannot exist. Double input resolves to the 63-bit
    // lattice, where the two chords stand ~9e7 sites apart.
    const real_t lo = real_t(0.5);
    const real_t hi = lo + real_t(1e-11);

    tf::polygons_buffer<index_t, real_t, 3, 3> mesh;
    mesh.points_buffer().emplace_back(real_t(0), real_t(0), real_t(0));
    mesh.points_buffer().emplace_back(real_t(1), real_t(0), real_t(0));
    mesh.points_buffer().emplace_back(real_t(0), real_t(1), real_t(0));
    mesh.faces_buffer().emplace_back(index_t(0), index_t(1), index_t(2));

    // The field is x, so a cut value chords the face at its own abscissa.
    std::array<real_t, 3> scalars = {real_t(0), real_t(1), real_t(0)};
    std::array<real_t, 2> values = {lo, hi};

    auto [full, labels, face_labels] = tf::embedded_isocurves(
        mesh.polygons(), tf::make_range(scalars), tf::make_range(values));

    REQUIRE(labels.size() == full.polygons().size());
    REQUIRE(face_labels.size() == full.polygons().size());

    real_t full_area = real_t(0);
    for (const auto& poly : full.polygons())
        full_area += tf::area(poly);
    REQUIRE(std::abs(full_area - real_t(0.5)) < real_t(1e-12));

    bool holds_sliver = false;
    for (decltype(labels.size()) i = 0; i < labels.size(); ++i) {
        REQUIRE(labels[i] >= index_t(0));
        REQUIRE(labels[i] <= index_t(2));
        holds_sliver = holds_sliver || labels[i] == index_t(1);
    }
    REQUIRE(holds_sliver);

    std::array<index_t, 1> selected = {index_t(1)};
    auto [band, band_labels, band_face_labels] = tf::make_isobands(
        mesh.polygons(), tf::make_range(scalars), tf::make_range(values),
        tf::make_range(selected));

    REQUIRE(band.polygons().size() > 0);
    REQUIRE(band_face_labels.size() == band.polygons().size());
    for (decltype(band_labels.size()) i = 0; i < band_labels.size(); ++i)
        REQUIRE(band_labels[i] == index_t(1));

    // The strip lo < x <= hi inside the triangle x >= 0, y >= 0, x + y <= 1.
    const real_t expected_area = (hi - lo) * (real_t(1) - (lo + hi) / real_t(2));
    real_t band_area = real_t(0);
    for (const auto& poly : band.polygons())
        band_area += tf::area(poly);
    REQUIRE(std::abs(band_area - expected_area) < expected_area * real_t(0.01));

    for (const auto& pt : band.points()) {
        REQUIRE(pt[0] >= lo - real_t(1e-13));
        REQUIRE(pt[0] <= hi + real_t(1e-13));
    }
}
