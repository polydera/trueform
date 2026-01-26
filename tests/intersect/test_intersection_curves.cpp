/**
 * @file test_intersection_curves.cpp
 * @brief Tests for mesh-mesh intersection curve extraction
 *
 * Tests for:
 * - make_intersection_curves
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
// Helper functions to create test meshes
// =============================================================================

template <typename Index, typename Real>
auto create_three_horizontal_planes() -> tf::polygons_buffer<Index, Real, 3, 4> {
    tf::polygons_buffer<Index, Real, 3, 4> result;

    // Plane at z=0: vertices 0-3
    result.points_buffer().emplace_back(Real(-1), Real(-1), Real(0));
    result.points_buffer().emplace_back(Real(1), Real(-1), Real(0));
    result.points_buffer().emplace_back(Real(1), Real(1), Real(0));
    result.points_buffer().emplace_back(Real(-1), Real(1), Real(0));

    // Plane at z=1: vertices 4-7
    result.points_buffer().emplace_back(Real(-1), Real(-1), Real(1));
    result.points_buffer().emplace_back(Real(1), Real(-1), Real(1));
    result.points_buffer().emplace_back(Real(1), Real(1), Real(1));
    result.points_buffer().emplace_back(Real(-1), Real(1), Real(1));

    // Plane at z=2: vertices 8-11
    result.points_buffer().emplace_back(Real(-1), Real(-1), Real(2));
    result.points_buffer().emplace_back(Real(1), Real(-1), Real(2));
    result.points_buffer().emplace_back(Real(1), Real(1), Real(2));
    result.points_buffer().emplace_back(Real(-1), Real(1), Real(2));

    // Faces (quads)
    result.faces_buffer().emplace_back(Index(0), Index(1), Index(2), Index(3));
    result.faces_buffer().emplace_back(Index(4), Index(5), Index(6), Index(7));
    result.faces_buffer().emplace_back(Index(8), Index(9), Index(10), Index(11));

    return result;
}

template <typename Index, typename Real>
auto create_vertical_plane_y0() -> tf::polygons_buffer<Index, Real, 3, 4> {
    tf::polygons_buffer<Index, Real, 3, 4> result;

    // Vertical plane at y=0, spanning x=[-1,1], z=[-0.5, 2.5]
    result.points_buffer().emplace_back(Real(-1), Real(0), Real(-0.5));
    result.points_buffer().emplace_back(Real(1), Real(0), Real(-0.5));
    result.points_buffer().emplace_back(Real(1), Real(0), Real(2.5));
    result.points_buffer().emplace_back(Real(-1), Real(0), Real(2.5));

    result.faces_buffer().emplace_back(Index(0), Index(1), Index(2), Index(3));

    return result;
}

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
// Test 1.1: Three Horizontal Planes vs Vertical Plane
// =============================================================================

TEMPLATE_TEST_CASE("intersection_curves_three_planes_vs_vertical", "[intersection_curves]",
    (tf::test::type_pair_dyn2<std::int32_t, float, false, false>),
    (tf::test::type_pair_dyn2<std::int32_t, float, true, false>),
    (tf::test::type_pair_dyn2<std::int32_t, float, false, true>),
    (tf::test::type_pair_dyn2<std::int32_t, float, true, true>),
    (tf::test::type_pair_dyn2<std::int64_t, double, false, false>),
    (tf::test::type_pair_dyn2<std::int64_t, double, true, false>),
    (tf::test::type_pair_dyn2<std::int64_t, double, false, true>),
    (tf::test::type_pair_dyn2<std::int64_t, double, true, true>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;
    constexpr bool dyn1 = TestType::is_dynamic1;
    constexpr bool dyn2 = TestType::is_dynamic2;

    auto planes_h = tf::test::maybe_as_dynamic<dyn1>(create_three_horizontal_planes<index_t, real_t>());
    auto plane_v = tf::test::maybe_as_dynamic<dyn2>(create_vertical_plane_y0<index_t, real_t>());

    auto curves = tf::make_intersection_curves(planes_h.polygons(), plane_v.polygons());

    // 3 intersection curves (one per horizontal plane)
    REQUIRE(curves.paths().size() == 3);

    // Collect points by their z-coordinate to identify which curve corresponds to which plane
    std::vector<std::vector<real_t>> z_values_per_curve(curves.paths().size());

    for (std::size_t path_idx = 0; path_idx < curves.paths().size(); ++path_idx) {
        const auto& path = curves.paths()[path_idx];
        for (auto pt_idx : path) {
            z_values_per_curve[path_idx].push_back(curves.points()[pt_idx][2]);
        }
    }

    // Sort curves by their z-values
    std::vector<std::size_t> sorted_indices = {0, 1, 2};
    std::sort(sorted_indices.begin(), sorted_indices.end(), [&](std::size_t a, std::size_t b) {
        real_t avg_a = real_t(0);
        for (auto z : z_values_per_curve[a]) avg_a += z;
        avg_a /= z_values_per_curve[a].size();

        real_t avg_b = real_t(0);
        for (auto z : z_values_per_curve[b]) avg_b += z;
        avg_b /= z_values_per_curve[b].size();

        return avg_a < avg_b;
    });

    // Verify each curve's z-coordinate and y-coordinate
    real_t expected_z[] = {real_t(0), real_t(1), real_t(2)};

    for (std::size_t i = 0; i < 3; ++i) {
        const auto& path = curves.paths()[sorted_indices[i]];
        REQUIRE(path.size() >= 2);  // At least 2 points per curve

        for (auto pt_idx : path) {
            const auto& pt = curves.points()[pt_idx];
            REQUIRE(std::abs(pt[2] - expected_z[i]) < tf::epsilon<real_t>);
            REQUIRE(std::abs(pt[1]) < tf::epsilon<real_t>);
        }
    }
}

// =============================================================================
// Test 1.2: Sphere vs Horizontal Plane
// =============================================================================

TEMPLATE_TEST_CASE("intersection_curves_sphere_vs_plane", "[intersection_curves]",
    (tf::test::type_pair_dyn2<std::int32_t, float, false, false>),
    (tf::test::type_pair_dyn2<std::int32_t, float, true, false>),
    (tf::test::type_pair_dyn2<std::int32_t, float, false, true>),
    (tf::test::type_pair_dyn2<std::int32_t, float, true, true>),
    (tf::test::type_pair_dyn2<std::int64_t, double, false, false>),
    (tf::test::type_pair_dyn2<std::int64_t, double, true, false>),
    (tf::test::type_pair_dyn2<std::int64_t, double, false, true>),
    (tf::test::type_pair_dyn2<std::int64_t, double, true, true>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;
    constexpr bool dyn1 = TestType::is_dynamic1;
    constexpr bool dyn2 = TestType::is_dynamic2;

    // Unit sphere centered at origin
    auto sphere = tf::test::maybe_as_dynamic<dyn1>(tf::make_sphere_mesh<index_t>(real_t(1), 30, 30));

    // Horizontal plane at z=0.5
    auto plane = tf::test::maybe_as_dynamic<dyn2>(create_horizontal_plane<index_t, real_t>(real_t(0.5)));

    auto curves = tf::make_intersection_curves(sphere.polygons(), plane.polygons());

    // 1 intersection curve (closed circle)
    REQUIRE(curves.paths().size() == 1);
    REQUIRE(curves.points().size() >= 3);

    // Curve is closed: first index equals last index
    const auto& path = curves.paths()[0];
    REQUIRE(path.front() == path.back());

    // Expected radius at z=0.5: sqrt(1 - 0.5^2) = sqrt(0.75) ~= 0.866
    real_t expected_r2 = real_t(0.75);
    real_t expected_z = real_t(0.5);

    for (const auto& pt : curves.points()) {
        REQUIRE(std::abs(pt[2] - expected_z) < tf::epsilon<real_t>);
        real_t r2 = pt[0] * pt[0] + pt[1] * pt[1];
        REQUIRE(std::abs(r2 - expected_r2) < tf::epsilon<real_t>);
    }
}

// =============================================================================
// Test 1.2b: Sphere vs Multiple Horizontal Planes
// =============================================================================

TEMPLATE_TEST_CASE("intersection_curves_sphere_vs_multiple_planes", "[intersection_curves]",
    (tf::test::type_pair_dyn2<std::int32_t, float, false, false>),
    (tf::test::type_pair_dyn2<std::int32_t, float, true, false>),
    (tf::test::type_pair_dyn2<std::int32_t, float, false, true>),
    (tf::test::type_pair_dyn2<std::int32_t, float, true, true>),
    (tf::test::type_pair_dyn2<std::int64_t, double, false, false>),
    (tf::test::type_pair_dyn2<std::int64_t, double, true, false>),
    (tf::test::type_pair_dyn2<std::int64_t, double, false, true>),
    (tf::test::type_pair_dyn2<std::int64_t, double, true, true>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;
    constexpr bool dyn1 = TestType::is_dynamic1;
    constexpr bool dyn2 = TestType::is_dynamic2;

    // Unit sphere centered at origin
    auto sphere = tf::test::maybe_as_dynamic<dyn1>(tf::make_sphere_mesh<index_t>(real_t(1), 50, 50));

    // Three horizontal planes at z = -0.5, 0, 0.5
    auto plane1 = create_horizontal_plane<index_t, real_t>(real_t(-0.5));
    auto plane2 = create_horizontal_plane<index_t, real_t>(real_t(0));
    auto plane3 = create_horizontal_plane<index_t, real_t>(real_t(0.5));

    // Concatenate planes into single mesh
    auto planes = tf::test::maybe_as_dynamic<dyn2>(tf::concatenated(
        plane1.polygons(),
        plane2.polygons(),
        plane3.polygons()));

    auto curves = tf::make_intersection_curves(sphere.polygons(), planes.polygons());

    // 3 intersection curves (one per plane)
    REQUIRE(curves.paths().size() == 3);

    // Expected z-values and radii squared
    std::array<real_t, 3> expected_z = {real_t(-0.5), real_t(0), real_t(0.5)};
    std::array<real_t, 3> expected_r2 = {
        real_t(1) - real_t(0.25),  // 0.75 at z=-0.5
        real_t(1),                  // 1.0 at z=0 (equator)
        real_t(1) - real_t(0.25)   // 0.75 at z=0.5
    };

    // Collect average z per curve to sort them
    std::vector<std::pair<real_t, std::size_t>> curve_z;
    for (std::size_t i = 0; i < curves.paths().size(); ++i) {
        real_t avg_z = real_t(0);
        for (auto idx : curves.paths()[i]) {
            avg_z += curves.points()[idx][2];
        }
        avg_z /= curves.paths()[i].size();
        curve_z.emplace_back(avg_z, i);
    }
    std::sort(curve_z.begin(), curve_z.end());

    // Verify each curve
    for (std::size_t i = 0; i < 3; ++i) {
        const auto& path = curves.paths()[curve_z[i].second];

        // Curve is closed
        REQUIRE(path.front() == path.back());

        // All points at correct z and radius
        for (auto idx : path) {
            const auto& pt = curves.points()[idx];
            REQUIRE(std::abs(pt[2] - expected_z[i]) < tf::epsilon<real_t>);
            real_t r2 = pt[0] * pt[0] + pt[1] * pt[1];
            REQUIRE(std::abs(r2 - expected_r2[i]) < std::max(tf::epsilon<real_t>, real_t(0.004)));
        }
    }
}

// =============================================================================
// Test 1.3: Non-Intersecting Meshes
// =============================================================================

TEMPLATE_TEST_CASE("intersection_curves_non_intersecting", "[intersection_curves]",
    (tf::test::type_pair_dyn2<std::int32_t, float, false, false>),
    (tf::test::type_pair_dyn2<std::int32_t, float, true, false>),
    (tf::test::type_pair_dyn2<std::int32_t, float, false, true>),
    (tf::test::type_pair_dyn2<std::int32_t, float, true, true>),
    (tf::test::type_pair_dyn2<std::int64_t, double, false, false>),
    (tf::test::type_pair_dyn2<std::int64_t, double, true, false>),
    (tf::test::type_pair_dyn2<std::int64_t, double, false, true>),
    (tf::test::type_pair_dyn2<std::int64_t, double, true, true>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;
    constexpr bool dyn1 = TestType::is_dynamic1;
    constexpr bool dyn2 = TestType::is_dynamic2;

    auto box1_fixed = tf::make_box_mesh<index_t>(real_t(1), real_t(1), real_t(1));
    auto box2_fixed = tf::make_box_mesh<index_t>(real_t(1), real_t(1), real_t(1));

    // Translate box2 far away
    for (decltype(box2_fixed.points().size()) i = 0; i < box2_fixed.points().size(); ++i) {
        box2_fixed.points_buffer()[i][0] += real_t(10);
    }

    auto box1 = tf::test::maybe_as_dynamic<dyn1>(std::move(box1_fixed));
    auto box2 = tf::test::maybe_as_dynamic<dyn2>(std::move(box2_fixed));

    auto curves = tf::make_intersection_curves(box1.polygons(), box2.polygons());

    // 0 intersection curves
    REQUIRE(curves.paths().size() == 0);
}

// =============================================================================
// Test 1.4: Two Overlapping Boxes
// =============================================================================

TEMPLATE_TEST_CASE("intersection_curves_overlapping_boxes", "[intersection_curves]",
    (tf::test::type_pair_dyn2<std::int32_t, float, false, false>),
    (tf::test::type_pair_dyn2<std::int32_t, float, true, false>),
    (tf::test::type_pair_dyn2<std::int32_t, float, false, true>),
    (tf::test::type_pair_dyn2<std::int32_t, float, true, true>),
    (tf::test::type_pair_dyn2<std::int64_t, double, false, false>),
    (tf::test::type_pair_dyn2<std::int64_t, double, true, false>),
    (tf::test::type_pair_dyn2<std::int64_t, double, false, true>),
    (tf::test::type_pair_dyn2<std::int64_t, double, true, true>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;
    constexpr bool dyn1 = TestType::is_dynamic1;
    constexpr bool dyn2 = TestType::is_dynamic2;

    auto box1_fixed = tf::make_box_mesh<index_t>(real_t(1), real_t(1), real_t(1));
    auto box2_fixed = tf::make_box_mesh<index_t>(real_t(1), real_t(1), real_t(1));

    // Translate box2 by (0.5, 0.5, 0.5) to create partial overlap
    for (decltype(box2_fixed.points().size()) i = 0; i < box2_fixed.points().size(); ++i) {
        box2_fixed.points_buffer()[i][0] += real_t(0.5);
        box2_fixed.points_buffer()[i][1] += real_t(0.5);
        box2_fixed.points_buffer()[i][2] += real_t(0.5);
    }

    auto box1 = tf::test::maybe_as_dynamic<dyn1>(std::move(box1_fixed));
    auto box2 = tf::test::maybe_as_dynamic<dyn2>(std::move(box2_fixed));

    auto curves = tf::make_intersection_curves(box1.polygons(), box2.polygons());

    // Multiple intersection curves exist
    REQUIRE(curves.paths().size() > 0);
    REQUIRE(curves.points().size() > 0);

    // Box1 spans [-0.5, 0.5] in all dimensions
    // Box2 spans [0, 1] in all dimensions
    for (const auto& pt : curves.points()) {
        // Points are on the surface of both boxes
        bool on_box1_surface = (std::abs(std::abs(pt[0]) - real_t(0.5)) < tf::epsilon<real_t>) ||
                               (std::abs(std::abs(pt[1]) - real_t(0.5)) < tf::epsilon<real_t>) ||
                               (std::abs(std::abs(pt[2]) - real_t(0.5)) < tf::epsilon<real_t>);

        bool on_box2_surface = (std::abs(pt[0]) < tf::epsilon<real_t>) || (std::abs(pt[0] - real_t(1)) < tf::epsilon<real_t>) ||
                               (std::abs(pt[1]) < tf::epsilon<real_t>) || (std::abs(pt[1] - real_t(1)) < tf::epsilon<real_t>) ||
                               (std::abs(pt[2]) < tf::epsilon<real_t>) || (std::abs(pt[2] - real_t(1)) < tf::epsilon<real_t>);

        REQUIRE((on_box1_surface && on_box2_surface));
    }
}
