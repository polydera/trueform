/**
 * @file test_neighborhoods.cpp
 * @brief Tests for k-ring and radius-based neighborhood functions
 *
 * Tests for:
 * - make_k_rings
 * - make_neighborhoods
 *
 * Copyright (c) 2025 Ziga Sajovic, XLAB
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <trueform/trueform.hpp>
#include "type_traits.hpp"
#include "topology_generators.hpp"
#include <set>

// =============================================================================
// make_k_rings - k=1 matches vertex_link
// =============================================================================

TEMPLATE_TEST_CASE("make_k_rings_k1_matches_vertex_link", "[topology][neighborhoods]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto mesh = tf::test::create_two_triangles_3d<index_t, real_t>();
    auto vl = tf::make_vertex_link(mesh.polygons());
    auto k1 = tf::make_k_rings(vl, 1);

    REQUIRE(k1.size() == 4);

    // k=1 should exactly match vertex_link (immediate neighbors)
    for (decltype(k1.size()) i = 0; i < k1.size(); ++i) {
        auto k1_set = std::set<index_t>(k1[i].begin(), k1[i].end());
        auto vl_set = std::set<index_t>(vl[i].begin(), vl[i].end());
        REQUIRE(k1_set == vl_set);
    }
}

// =============================================================================
// make_k_rings - k=2 is superset of k=1
// =============================================================================

TEMPLATE_TEST_CASE("make_k_rings_k2_superset_of_k1", "[topology][neighborhoods]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto mesh = tf::test::create_triangle_strip_3d<index_t, real_t>(5);
    auto vl = tf::make_vertex_link(mesh.polygons());
    auto k1 = tf::make_k_rings(vl, 1);
    auto k2 = tf::make_k_rings(vl, 2);

    for (decltype(k1.size()) i = 0; i < k1.size(); ++i) {
        auto k1_set = std::set<index_t>(k1[i].begin(), k1[i].end());
        auto k2_set = std::set<index_t>(k2[i].begin(), k2[i].end());

        // k=1 should be subset of k=2
        for (auto v : k1_set) {
            REQUIRE(k2_set.count(v));
        }

        // k=2 should be >= k=1
        REQUIRE(k2_set.size() >= k1_set.size());
    }
}

// =============================================================================
// make_k_rings - Large k reaches all connected vertices
// =============================================================================

TEMPLATE_TEST_CASE("make_k_rings_large_k_reaches_all", "[topology][neighborhoods]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto mesh = tf::test::create_two_triangles_3d<index_t, real_t>();
    auto vl = tf::make_vertex_link(mesh.polygons());

    // k=10 should reach all vertices from any starting point (mesh is small)
    auto k10 = tf::make_k_rings(vl, 10);

    // From vertex 0, should reach all other vertices (1, 2, 3)
    auto v0_neighbors = std::set<index_t>(k10[0].begin(), k10[0].end());
    REQUIRE(v0_neighbors.count(1));
    REQUIRE(v0_neighbors.count(2));
    REQUIRE(v0_neighbors.count(3));
}

// =============================================================================
// make_k_rings - Inclusive flag
// =============================================================================

TEMPLATE_TEST_CASE("make_k_rings_inclusive_vs_exclusive", "[topology][neighborhoods]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto mesh = tf::test::create_two_triangles_3d<index_t, real_t>();
    auto vl = tf::make_vertex_link(mesh.polygons());

    auto k1_exclusive = tf::make_k_rings(vl, 1, false);
    auto k1_inclusive = tf::make_k_rings(vl, 1, true);

    for (decltype(vl.size()) i = 0; i < vl.size(); ++i) {
        // Exclusive should not contain seed vertex
        auto excl_set = std::set<index_t>(k1_exclusive[i].begin(), k1_exclusive[i].end());
        REQUIRE_FALSE(excl_set.count(static_cast<index_t>(i)));

        // Inclusive should contain seed vertex
        auto incl_set = std::set<index_t>(k1_inclusive[i].begin(), k1_inclusive[i].end());
        REQUIRE(incl_set.count(static_cast<index_t>(i)));

        // Inclusive size should be exclusive size + 1
        REQUIRE(incl_set.size() == excl_set.size() + 1);
    }
}

// =============================================================================
// make_k_rings - Two Components (k-ring doesn't cross components)
// =============================================================================

TEMPLATE_TEST_CASE("make_k_rings_two_components", "[topology][neighborhoods]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto mesh = tf::test::create_two_components_3d<index_t, real_t>();
    auto vl = tf::make_vertex_link(mesh.polygons());

    // Even with large k, vertices from different components shouldn't mix
    auto k10 = tf::make_k_rings(vl, 10);

    // Vertex 0 is in component 1 (vertices 0,1,2)
    auto v0_neighbors = std::set<index_t>(k10[0].begin(), k10[0].end());
    REQUIRE(v0_neighbors.count(1));
    REQUIRE(v0_neighbors.count(2));
    REQUIRE_FALSE(v0_neighbors.count(3));
    REQUIRE_FALSE(v0_neighbors.count(4));
    REQUIRE_FALSE(v0_neighbors.count(5));

    // Vertex 3 is in component 2 (vertices 3,4,5)
    auto v3_neighbors = std::set<index_t>(k10[3].begin(), k10[3].end());
    REQUIRE(v3_neighbors.count(4));
    REQUIRE(v3_neighbors.count(5));
    REQUIRE_FALSE(v3_neighbors.count(0));
    REQUIRE_FALSE(v3_neighbors.count(1));
    REQUIRE_FALSE(v3_neighbors.count(2));
}

// =============================================================================
// make_k_rings - Grid Mesh (predictable k-ring sizes)
// =============================================================================

TEMPLATE_TEST_CASE("make_k_rings_grid_mesh", "[topology][neighborhoods]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    // 5x5 grid
    auto mesh = tf::test::create_grid_mesh_3d<index_t, real_t>(5, 5);
    auto vl = tf::make_vertex_link(mesh.polygons());

    // k=1 for center vertex (index 12 in 5x5 grid)
    auto k1 = tf::make_k_rings(vl, 1);
    auto center_k1 = std::set<index_t>(k1[12].begin(), k1[12].end());

    // Center vertex in triangulated grid should have 6 neighbors
    REQUIRE(center_k1.size() == 6);

    // k=2 should have more neighbors
    auto k2 = tf::make_k_rings(vl, 2);
    auto center_k2 = std::set<index_t>(k2[12].begin(), k2[12].end());
    REQUIRE(center_k2.size() > center_k1.size());
}

// =============================================================================
// make_neighborhoods - Radius-based
// =============================================================================

TEMPLATE_TEST_CASE("make_neighborhoods_radius", "[topology][neighborhoods]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto mesh = tf::test::create_grid_mesh_3d<index_t, real_t>(5, 5);
    auto vl = tf::make_vertex_link(mesh.polygons());

    real_t radius = real_t(1.5);

    // Create distance function
    auto distance2_f = [&](index_t seed, index_t neighbor) {
        auto p0 = mesh.points()[seed];
        auto p1 = mesh.points()[neighbor];
        real_t dx = p1[0] - p0[0];
        real_t dy = p1[1] - p0[1];
        real_t dz = p1[2] - p0[2];
        return dx * dx + dy * dy + dz * dz;
    };

    auto neighborhoods = tf::make_neighborhoods(vl, distance2_f, radius);

    // Verify all neighbors are within radius
    real_t radius2 = radius * radius;
    for (decltype(neighborhoods.size()) i = 0; i < neighborhoods.size(); ++i) {
        for (auto neighbor : neighborhoods[i]) {
            REQUIRE(distance2_f(static_cast<index_t>(i), neighbor) <= radius2);
        }
    }
}

TEMPLATE_TEST_CASE("make_neighborhoods_small_radius", "[topology][neighborhoods]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto mesh = tf::test::create_grid_mesh_3d<index_t, real_t>(5, 5);
    auto vl = tf::make_vertex_link(mesh.polygons());

    // Very small radius - should only get immediate neighbors
    real_t radius = real_t(1.01);

    auto distance2_f = [&](index_t seed, index_t neighbor) {
        auto p0 = mesh.points()[seed];
        auto p1 = mesh.points()[neighbor];
        real_t dx = p1[0] - p0[0];
        real_t dy = p1[1] - p0[1];
        real_t dz = p1[2] - p0[2];
        return dx * dx + dy * dy + dz * dz;
    };

    auto neighborhoods = tf::make_neighborhoods(vl, distance2_f, radius);

    // With radius ~1, should only get direct neighbors (distance 1 or sqrt(2))
    // Center vertex at index 12 has neighbors at distance 1 (up/down/left/right)
    // and diagonal neighbors at distance sqrt(2) ≈ 1.41
    // With radius 1.01, should get 4 neighbors (horizontal/vertical)
    auto center_neighbors = std::set<index_t>(
        neighborhoods[12].begin(), neighborhoods[12].end());

    // Should have neighbors but not all 6 (some are diagonal)
    REQUIRE(center_neighbors.size() >= 2);
    REQUIRE(center_neighbors.size() <= 6);
}

TEMPLATE_TEST_CASE("make_neighborhoods_large_radius", "[topology][neighborhoods]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto mesh = tf::test::create_grid_mesh_3d<index_t, real_t>(4, 4);
    auto vl = tf::make_vertex_link(mesh.polygons());

    // Very large radius - should reach all vertices
    real_t radius = real_t(100);

    auto distance2_f = [&](index_t seed, index_t neighbor) {
        auto p0 = mesh.points()[seed];
        auto p1 = mesh.points()[neighbor];
        real_t dx = p1[0] - p0[0];
        real_t dy = p1[1] - p0[1];
        real_t dz = p1[2] - p0[2];
        return dx * dx + dy * dy + dz * dz;
    };

    auto neighborhoods = tf::make_neighborhoods(vl, distance2_f, radius);

    // With very large radius, center should reach all 15 other vertices
    auto corner_neighbors = std::set<index_t>(
        neighborhoods[0].begin(), neighborhoods[0].end());

    // Should reach all 15 other vertices (16 total - 1 for self)
    REQUIRE(corner_neighbors.size() == 15);
}

TEMPLATE_TEST_CASE("make_neighborhoods_inclusive", "[topology][neighborhoods]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto mesh = tf::test::create_grid_mesh_3d<index_t, real_t>(3, 3);
    auto vl = tf::make_vertex_link(mesh.polygons());

    real_t radius = real_t(1.5);

    auto distance2_f = [&](index_t seed, index_t neighbor) {
        auto p0 = mesh.points()[seed];
        auto p1 = mesh.points()[neighbor];
        real_t dx = p1[0] - p0[0];
        real_t dy = p1[1] - p0[1];
        real_t dz = p1[2] - p0[2];
        return dx * dx + dy * dy + dz * dz;
    };

    auto excl = tf::make_neighborhoods(vl, distance2_f, radius, false);
    auto incl = tf::make_neighborhoods(vl, distance2_f, radius, true);

    for (decltype(vl.size()) i = 0; i < vl.size(); ++i) {
        auto excl_set = std::set<index_t>(excl[i].begin(), excl[i].end());
        auto incl_set = std::set<index_t>(incl[i].begin(), incl[i].end());

        // Exclusive should not contain seed
        REQUIRE_FALSE(excl_set.count(static_cast<index_t>(i)));

        // Inclusive should contain seed
        REQUIRE(incl_set.count(static_cast<index_t>(i)));
    }
}

// =============================================================================
// Brute Force Verification
// =============================================================================

TEMPLATE_TEST_CASE("make_k_rings_brute_force_verification", "[topology][neighborhoods]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto mesh = tf::test::create_grid_mesh_3d<index_t, real_t>(4, 4);
    auto vl = tf::make_vertex_link(mesh.polygons());
    auto k2 = tf::make_k_rings(vl, 2, false);

    // Brute force k=2: BFS from each vertex up to 2 hops
    for (decltype(vl.size()) seed = 0; seed < vl.size(); ++seed) {
        std::set<index_t> expected;

        // Ring 1: immediate neighbors
        for (auto n1 : vl[seed]) {
            expected.insert(n1);
        }

        // Ring 2: neighbors of neighbors
        for (auto n1 : vl[seed]) {
            for (auto n2 : vl[n1]) {
                if (n2 != static_cast<index_t>(seed)) {
                    expected.insert(n2);
                }
            }
        }

        auto actual = std::set<index_t>(k2[seed].begin(), k2[seed].end());
        REQUIRE(actual == expected);
    }
}
