/**
 * @file test_self_intersection.cpp
 * @brief Tests for mesh self-intersection detection
 *
 * Tests for:
 * - make_self_intersection_curves
 * - Concatenation equivalence: self_intersect(A+B) == intersect(A,B)
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
auto create_horizontal_plane(Real z_height) -> tf::polygons_buffer<Index, Real, 3, 4> {
    tf::polygons_buffer<Index, Real, 3, 4> result;

    result.points_buffer().emplace_back(Real(-1), Real(-1), z_height);
    result.points_buffer().emplace_back(Real(1), Real(-1), z_height);
    result.points_buffer().emplace_back(Real(1), Real(1), z_height);
    result.points_buffer().emplace_back(Real(-1), Real(1), z_height);

    result.faces_buffer().emplace_back(Index(0), Index(1), Index(2), Index(3));

    return result;
}

template <typename Index, typename Real>
auto create_vertical_plane_y0() -> tf::polygons_buffer<Index, Real, 3, 4> {
    tf::polygons_buffer<Index, Real, 3, 4> result;

    // Vertical plane at y=0, spanning x=[-1,1], z=[-0.5, 0.5]
    result.points_buffer().emplace_back(Real(-1), Real(0), Real(-0.5));
    result.points_buffer().emplace_back(Real(1), Real(0), Real(-0.5));
    result.points_buffer().emplace_back(Real(1), Real(0), Real(0.5));
    result.points_buffer().emplace_back(Real(-1), Real(0), Real(0.5));

    result.faces_buffer().emplace_back(Index(0), Index(1), Index(2), Index(3));

    return result;
}

// =============================================================================
// Test 2.1: Concatenation Equivalence
// =============================================================================

TEMPLATE_TEST_CASE("self_intersection_concatenation_equivalence", "[self_intersection]",
    (tf::test::type_pair_dyn<std::int32_t, float, false>),
    (tf::test::type_pair_dyn<std::int32_t, float, true>),
    (tf::test::type_pair_dyn<std::int64_t, double, false>),
    (tf::test::type_pair_dyn<std::int64_t, double, true>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;
    constexpr bool dyn = TestType::is_dynamic;

    auto plane_h = tf::test::maybe_as_dynamic<dyn>(
        create_horizontal_plane<index_t, real_t>(real_t(0)));
    auto plane_v = tf::test::maybe_as_dynamic<dyn>(
        create_vertical_plane_y0<index_t, real_t>());

    // Method 1: Intersection between two separate meshes
    auto curves_ab = tf::make_intersection_curves(plane_h.polygons(), plane_v.polygons());

    // Method 2: Self-intersection of concatenated mesh
    auto combined = tf::concatenated(plane_h.polygons(), plane_v.polygons());
    auto curves_self = tf::make_self_intersection_curves(combined.polygons());

    // Same number of curves
    REQUIRE(curves_ab.paths().size() == curves_self.paths().size());

    // Exactly 1 curve (the intersection line)
    REQUIRE(curves_ab.paths().size() == 1);

    // Both curves have points at y=0, z=0
    for (const auto& pt : curves_ab.points()) {
        REQUIRE(std::abs(pt[1]) < tf::epsilon<real_t>);
        REQUIRE(std::abs(pt[2]) < tf::epsilon<real_t>);
    }

    for (const auto& pt : curves_self.points()) {
        REQUIRE(std::abs(pt[1]) < tf::epsilon<real_t>);
        REQUIRE(std::abs(pt[2]) < tf::epsilon<real_t>);
    }
}

// =============================================================================
// Test 2.2: Non-Self-Intersecting Sphere
// =============================================================================

TEMPLATE_TEST_CASE("self_intersection_sphere_clean", "[self_intersection]",
    (tf::test::type_pair_dyn<std::int32_t, float, false>),
    (tf::test::type_pair_dyn<std::int32_t, float, true>),
    (tf::test::type_pair_dyn<std::int64_t, double, false>),
    (tf::test::type_pair_dyn<std::int64_t, double, true>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;
    constexpr bool dyn = TestType::is_dynamic;

    auto sphere = tf::test::maybe_as_dynamic<dyn>(
        tf::make_sphere_mesh<index_t>(real_t(1), 20, 20));
    auto curves = tf::make_self_intersection_curves(sphere.polygons());

    REQUIRE(curves.paths().size() == 0);
}

// =============================================================================
// Test 2.3: Non-Self-Intersecting Box
// =============================================================================

TEMPLATE_TEST_CASE("self_intersection_box_clean", "[self_intersection]",
    (tf::test::type_pair_dyn<std::int32_t, float, false>),
    (tf::test::type_pair_dyn<std::int32_t, float, true>),
    (tf::test::type_pair_dyn<std::int64_t, double, false>),
    (tf::test::type_pair_dyn<std::int64_t, double, true>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;
    constexpr bool dyn = TestType::is_dynamic;

    auto box = tf::test::maybe_as_dynamic<dyn>(
        tf::make_box_mesh<index_t>(real_t(2), real_t(1), real_t(3)));
    auto curves = tf::make_self_intersection_curves(box.polygons());

    REQUIRE(curves.paths().size() == 0);
}

// =============================================================================
// Test 2.4: Non-Self-Intersecting Cylinder
// =============================================================================

TEMPLATE_TEST_CASE("self_intersection_cylinder_clean", "[self_intersection]",
    (tf::test::type_pair_dyn<std::int32_t, float, false>),
    (tf::test::type_pair_dyn<std::int32_t, float, true>),
    (tf::test::type_pair_dyn<std::int64_t, double, false>),
    (tf::test::type_pair_dyn<std::int64_t, double, true>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;
    constexpr bool dyn = TestType::is_dynamic;

    auto cylinder = tf::test::maybe_as_dynamic<dyn>(
        tf::make_cylinder_mesh<index_t>(real_t(1), real_t(2), 20));
    auto curves = tf::make_self_intersection_curves(cylinder.polygons());

    REQUIRE(curves.paths().size() == 0);
}

// =============================================================================
// Test 2.5: Non-Self-Intersecting Plane
// =============================================================================

TEMPLATE_TEST_CASE("self_intersection_plane_clean", "[self_intersection]",
    (tf::test::type_pair_dyn<std::int32_t, float, false>),
    (tf::test::type_pair_dyn<std::int32_t, float, true>),
    (tf::test::type_pair_dyn<std::int64_t, double, false>),
    (tf::test::type_pair_dyn<std::int64_t, double, true>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;
    constexpr bool dyn = TestType::is_dynamic;

    auto plane = tf::test::maybe_as_dynamic<dyn>(
        tf::make_plane_mesh<index_t>(real_t(2), real_t(2), 10, 10));
    auto curves = tf::make_self_intersection_curves(plane.polygons());

    REQUIRE(curves.paths().size() == 0);
}

// =============================================================================
// Test 2.6: Self-Intersection with Concatenated Overlapping Planes
// =============================================================================

TEMPLATE_TEST_CASE("self_intersection_overlapping_planes", "[self_intersection]",
    (tf::test::type_pair_dyn<std::int32_t, float, false>),
    (tf::test::type_pair_dyn<std::int32_t, float, true>),
    (tf::test::type_pair_dyn<std::int64_t, double, false>),
    (tf::test::type_pair_dyn<std::int64_t, double, true>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;
    constexpr bool dyn = TestType::is_dynamic;

    // Create two crossing planes
    auto plane_h = tf::test::maybe_as_dynamic<dyn>(
        create_horizontal_plane<index_t, real_t>(real_t(0)));
    auto plane_v = tf::test::maybe_as_dynamic<dyn>(
        create_vertical_plane_y0<index_t, real_t>());

    // Concatenate them into a single mesh
    auto combined = tf::concatenated(plane_h.polygons(), plane_v.polygons());

    auto curves = tf::make_self_intersection_curves(combined.polygons());

    // Exactly 1 self-intersection curve
    REQUIRE(curves.paths().size() == 1);
    REQUIRE(curves.points().size() >= 2);

    // The intersection is along the x-axis at y=0, z=0
    for (const auto& pt : curves.points()) {
        REQUIRE(std::abs(pt[1]) < tf::epsilon<real_t>);
        REQUIRE(std::abs(pt[2]) < tf::epsilon<real_t>);
    }
}

// =============================================================================
// Test 2.7: Sphere + Plane Concatenated (mirrors intersection_curves_sphere_vs_plane)
// =============================================================================

TEMPLATE_TEST_CASE("self_intersection_sphere_plane_concatenated", "[self_intersection]",
    (tf::test::type_pair_dyn<std::int32_t, float, false>),
    (tf::test::type_pair_dyn<std::int32_t, float, true>),
    (tf::test::type_pair_dyn<std::int64_t, double, false>),
    (tf::test::type_pair_dyn<std::int64_t, double, true>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;
    constexpr bool dyn = TestType::is_dynamic;

    // Unit sphere centered at origin
    auto sphere = tf::test::maybe_as_dynamic<dyn>(
        tf::make_sphere_mesh<index_t>(real_t(1), 30, 30));

    // Horizontal plane at z=0.5
    auto plane = tf::test::maybe_as_dynamic<dyn>(
        create_horizontal_plane<index_t, real_t>(real_t(0.5)));

    // Concatenate into single mesh
    auto combined = tf::concatenated(sphere.polygons(), plane.polygons());
    auto curves = tf::make_self_intersection_curves(combined.polygons());

    // 1 intersection curve (closed circle)
    REQUIRE(curves.paths().size() == 1);
    REQUIRE(curves.points().size() >= 3);

    // Curve is closed
    const auto& path = curves.paths()[0];
    REQUIRE(path.front() == path.back());

    // Expected radius at z=0.5: sqrt(1 - 0.5^2) = sqrt(0.75)
    real_t expected_r2 = real_t(0.75);
    real_t expected_z = real_t(0.5);

    for (const auto& pt : curves.points()) {
        REQUIRE(std::abs(pt[2] - expected_z) < tf::epsilon<real_t>);
        real_t r2 = pt[0] * pt[0] + pt[1] * pt[1];
        REQUIRE(std::abs(r2 - expected_r2) < tf::epsilon<real_t>);
    }
}

// =============================================================================
// Test 2.8: Sphere + Multiple Planes Concatenated
// =============================================================================

TEMPLATE_TEST_CASE("self_intersection_sphere_multiple_planes_concatenated", "[self_intersection]",
    (tf::test::type_pair_dyn<std::int32_t, float, false>),
    (tf::test::type_pair_dyn<std::int32_t, float, true>),
    (tf::test::type_pair_dyn<std::int64_t, double, false>),
    (tf::test::type_pair_dyn<std::int64_t, double, true>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;
    constexpr bool dyn = TestType::is_dynamic;

    // Unit sphere centered at origin
    auto sphere = tf::test::maybe_as_dynamic<dyn>(
        tf::make_sphere_mesh<index_t>(real_t(1), 50, 50));

    // Three horizontal planes at z = -0.5, 0, 0.5
    auto plane1 = tf::test::maybe_as_dynamic<dyn>(
        create_horizontal_plane<index_t, real_t>(real_t(-0.5)));
    auto plane2 = tf::test::maybe_as_dynamic<dyn>(
        create_horizontal_plane<index_t, real_t>(real_t(0)));
    auto plane3 = tf::test::maybe_as_dynamic<dyn>(
        create_horizontal_plane<index_t, real_t>(real_t(0.5)));

    // Concatenate all into single mesh
    auto combined = tf::concatenated(
        sphere.polygons(),
        plane1.polygons(),
        plane2.polygons(),
        plane3.polygons());

    auto curves = tf::make_self_intersection_curves(combined.polygons());

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
// Test 2.9: Three Horizontal Planes + Vertical Plane Concatenated
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
auto create_tall_vertical_plane_y0() -> tf::polygons_buffer<Index, Real, 3, 4> {
    tf::polygons_buffer<Index, Real, 3, 4> result;

    // Vertical plane at y=0, spanning x=[-1,1], z=[-0.5, 2.5]
    result.points_buffer().emplace_back(Real(-1), Real(0), Real(-0.5));
    result.points_buffer().emplace_back(Real(1), Real(0), Real(-0.5));
    result.points_buffer().emplace_back(Real(1), Real(0), Real(2.5));
    result.points_buffer().emplace_back(Real(-1), Real(0), Real(2.5));

    result.faces_buffer().emplace_back(Index(0), Index(1), Index(2), Index(3));

    return result;
}

TEMPLATE_TEST_CASE("self_intersection_three_planes_vs_vertical_concatenated", "[self_intersection]",
    (tf::test::type_pair_dyn<std::int32_t, float, false>),
    (tf::test::type_pair_dyn<std::int32_t, float, true>),
    (tf::test::type_pair_dyn<std::int64_t, double, false>),
    (tf::test::type_pair_dyn<std::int64_t, double, true>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;
    constexpr bool dyn = TestType::is_dynamic;

    auto planes_h = tf::test::maybe_as_dynamic<dyn>(
        create_three_horizontal_planes<index_t, real_t>());
    auto plane_v = tf::test::maybe_as_dynamic<dyn>(
        create_tall_vertical_plane_y0<index_t, real_t>());

    // Concatenate into single mesh
    auto combined = tf::concatenated(planes_h.polygons(), plane_v.polygons());
    auto curves = tf::make_self_intersection_curves(combined.polygons());

    // 3 intersection curves (one per horizontal plane)
    REQUIRE(curves.paths().size() == 3);

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

    // Verify each curve's z-coordinate and y-coordinate
    real_t expected_z[] = {real_t(0), real_t(1), real_t(2)};

    for (std::size_t i = 0; i < 3; ++i) {
        const auto& path = curves.paths()[curve_z[i].second];
        REQUIRE(path.size() >= 2);

        for (auto pt_idx : path) {
            const auto& pt = curves.points()[pt_idx];
            REQUIRE(std::abs(pt[2] - expected_z[i]) < tf::epsilon<real_t>);
            REQUIRE(std::abs(pt[1]) < tf::epsilon<real_t>);
        }
    }
}
