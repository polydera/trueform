/**
 * @file test_boundary.cpp
 * @brief Tests for boundary edge and path extraction functions
 *
 * Tests for:
 * - make_boundary_edges
 * - make_boundary_paths
 *
 * Copyright (c) 2025 Ziga Sajovic, XLAB
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <trueform/trueform.hpp>
#include "type_traits.hpp"
#include "topology_generators.hpp"
#include <map>
#include <set>

namespace {

/**
 * @brief Canonicalize an edge by ordering vertex indices
 */
template <typename Index>
auto canonicalize_edge(Index a, Index b) -> std::pair<Index, Index> {
    return a < b ? std::pair{a, b} : std::pair{b, a};
}

/**
 * @brief Convert a blocked buffer of edges to a set of canonicalized edges
 */
template <typename EdgeBuffer>
auto edges_to_set(const EdgeBuffer& edges) {
    using Index = std::decay_t<decltype(edges[0][0])>;
    std::set<std::pair<Index, Index>> result;
    for (decltype(edges.size()) i = 0; i < edges.size(); ++i) {
        result.insert(canonicalize_edge(edges[i][0], edges[i][1]));
    }
    return result;
}

} // anonymous namespace

// =============================================================================
// make_boundary_edges - Open Mesh
// =============================================================================

TEMPLATE_TEST_CASE("make_boundary_edges_open_mesh", "[topology][boundary]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto mesh = tf::test::create_two_triangles_3d<index_t, real_t>();
    auto boundary = tf::make_boundary_edges(mesh.polygons());

    // Two triangles sharing edge (1,2):
    // Face 0: [0,1,2], Face 1: [1,3,2]
    // Shared edge: (1,2) - not boundary
    // Boundary edges: (0,1), (0,2), (1,3), (2,3)
    REQUIRE(boundary.size() == 4);

    auto edges_set = edges_to_set(boundary);
    REQUIRE(edges_set.count({0, 1}));
    REQUIRE(edges_set.count({0, 2}));
    REQUIRE(edges_set.count({1, 3}));
    REQUIRE(edges_set.count({2, 3}));

    // Shared edge should NOT be in boundary
    REQUIRE_FALSE(edges_set.count({1, 2}));
}

// =============================================================================
// make_boundary_edges - Closed Mesh
// =============================================================================

TEMPLATE_TEST_CASE("make_boundary_edges_closed_mesh", "[topology][boundary]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto mesh = tf::test::create_tetrahedron_3d<index_t, real_t>();
    auto boundary = tf::make_boundary_edges(mesh.polygons());

    // Tetrahedron is closed - no boundary edges
    REQUIRE(boundary.size() == 0);
}

// =============================================================================
// make_boundary_edges - Grid Mesh (has boundary)
// =============================================================================

TEMPLATE_TEST_CASE("make_boundary_edges_grid_mesh", "[topology][boundary]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    // 3x3 grid = 9 vertices, 8 triangles
    auto mesh = tf::test::create_grid_mesh_3d<index_t, real_t>(3, 3);
    auto boundary = tf::make_boundary_edges(mesh.polygons());

    // Grid boundary forms a square around the perimeter
    // 3x3 grid has 2*3 + 2*3 - 4 = 8 boundary edges
    // Actually: 4 edges on each side = 4*2 = 8 edges
    REQUIRE(boundary.size() == 8);

    // Verify boundary edges are on the perimeter
    auto edges_set = edges_to_set(boundary);

    // All boundary edges should connect adjacent perimeter vertices
    // Interior edges should not be in the boundary
    // Interior edge example: (4 is center of 3x3 grid - vertex index 4)
    // Vertex layout for 3x3:
    // 0 1 2
    // 3 4 5
    // 6 7 8
    // Edge (3,4) is interior (horizontal through center), should not be boundary
    REQUIRE_FALSE(edges_set.count({3, 4}));
    REQUIRE_FALSE(edges_set.count({4, 5}));
    REQUIRE_FALSE(edges_set.count({1, 4}));
    REQUIRE_FALSE(edges_set.count({4, 7}));
}

// =============================================================================
// make_boundary_edges - Triangle Strip
// =============================================================================

TEMPLATE_TEST_CASE("make_boundary_edges_triangle_strip", "[topology][boundary]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto mesh = tf::test::create_triangle_strip_3d<index_t, real_t>(5);
    auto boundary = tf::make_boundary_edges(mesh.polygons());

    // 5 triangles, 7 vertices
    // Boundary goes around the outside of the strip
    // Each end has 2 boundary edges + top and bottom edges
    REQUIRE(boundary.size() > 0);

    auto edges_set = edges_to_set(boundary);

    // First edge (0,1) should be boundary (start of strip)
    REQUIRE(edges_set.count({0, 1}));
}

// =============================================================================
// make_boundary_edges - Mesh with Hole
// =============================================================================

TEMPLATE_TEST_CASE("make_boundary_edges_mesh_with_hole", "[topology][boundary]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto mesh = tf::test::create_mesh_with_hole_3d<index_t, real_t>();
    auto boundary = tf::make_boundary_edges(mesh.polygons());

    // Mesh has two boundary loops:
    // - Outer square: 4 edges
    // - Inner square (hole): 4 edges
    // Total: 8 boundary edges
    REQUIRE(boundary.size() == 8);

    auto edges_set = edges_to_set(boundary);

    // Outer boundary edges (vertices 0-3)
    REQUIRE(edges_set.count({0, 1}));
    REQUIRE(edges_set.count({1, 2}));
    REQUIRE(edges_set.count({2, 3}));
    REQUIRE(edges_set.count({0, 3}));

    // Inner boundary edges (vertices 4-7)
    REQUIRE(edges_set.count({4, 5}));
    REQUIRE(edges_set.count({5, 6}));
    REQUIRE(edges_set.count({6, 7}));
    REQUIRE(edges_set.count({4, 7}));
}

// =============================================================================
// make_boundary_edges - Dynamic Mesh
// =============================================================================

TEMPLATE_TEST_CASE("make_boundary_edges_dynamic_mesh", "[topology][boundary]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto mesh = tf::test::create_dynamic_mesh_3d<index_t, real_t>();
    auto boundary = tf::make_boundary_edges(mesh.polygons());

    // Dynamic mesh has 1 triangle and 1 quad, sharing edge (0,2)
    // Boundary edges: (0,3), (2,3), (0,1), (1,4), (2,4)
    REQUIRE(boundary.size() == 5);

    auto edges_set = edges_to_set(boundary);

    // Shared edge (0,2) should NOT be boundary
    REQUIRE_FALSE(edges_set.count({0, 2}));
}

// =============================================================================
// make_boundary_paths - Single Loop
// =============================================================================

TEMPLATE_TEST_CASE("make_boundary_paths_single_loop", "[topology][boundary]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto mesh = tf::test::create_two_triangles_3d<index_t, real_t>();
    auto paths = tf::make_boundary_paths(mesh.polygons());
    // Two triangles sharing an edge form a single boundary loop
    REQUIRE(paths.size() == 1);

    // Closed path has first == last, so 4 unique vertices means 5 vertices in path
    REQUIRE(paths[0].size() == 5);
    REQUIRE(paths[0].front() == paths[0].back());

    // Verify the path contains all boundary vertices
    std::set<index_t> path_vertices(paths[0].begin(), paths[0].end());
    REQUIRE(path_vertices.count(0));
    REQUIRE(path_vertices.count(1));
    REQUIRE(path_vertices.count(2));
    REQUIRE(path_vertices.count(3));
}

// =============================================================================
// make_boundary_paths - Closed Mesh (no paths)
// =============================================================================

TEMPLATE_TEST_CASE("make_boundary_paths_closed_mesh", "[topology][boundary]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto mesh = tf::test::create_tetrahedron_3d<index_t, real_t>();
    auto paths = tf::make_boundary_paths(mesh.polygons());

    // Tetrahedron is closed - no boundary paths
    REQUIRE(paths.size() == 0);
}

// =============================================================================
// make_boundary_paths - Mesh with Hole (two loops)
// =============================================================================

TEMPLATE_TEST_CASE("make_boundary_paths_mesh_with_hole", "[topology][boundary]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto mesh = tf::test::create_mesh_with_hole_3d<index_t, real_t>();
    auto paths = tf::make_boundary_paths(mesh.polygons());

    // Mesh with hole has two boundary loops:
    // - Outer boundary
    // - Inner boundary (around hole)
    REQUIRE(paths.size() == 2);

    // Each closed boundary with 4 unique vertices has 5 vertices in path (first==last)
    std::size_t total_vertices = paths[0].size() + paths[1].size();
    REQUIRE(total_vertices == 10);
    REQUIRE(paths[0].front() == paths[0].back());
    REQUIRE(paths[1].front() == paths[1].back());

    // Collect vertices from both paths
    std::set<index_t> outer_vertices;
    std::set<index_t> inner_vertices;

    // Determine which path is outer (vertices 0-3) and which is inner (vertices 4-7)
    for (auto v : paths[0]) {
        if (v < 4) {
            outer_vertices.insert(v);
        } else {
            inner_vertices.insert(v);
        }
    }
    for (auto v : paths[1]) {
        if (v < 4) {
            outer_vertices.insert(v);
        } else {
            inner_vertices.insert(v);
        }
    }

    // All outer vertices should be present
    REQUIRE(outer_vertices.size() == 4);
    // All inner vertices should be present
    REQUIRE(inner_vertices.size() == 4);
}

// =============================================================================
// make_boundary_paths - Grid Mesh (single loop)
// =============================================================================

TEMPLATE_TEST_CASE("make_boundary_paths_grid_mesh", "[topology][boundary]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    // 4x4 grid
    auto mesh = tf::test::create_grid_mesh_3d<index_t, real_t>(4, 4);
    auto paths = tf::make_boundary_paths(mesh.polygons());

    // Grid has a single boundary loop around the perimeter
    REQUIRE(paths.size() == 1);

    // 4x4 grid perimeter has 12 vertices (4 per side minus corners counted once)
    // Actually: 4*4 - (4-2)*(4-2) = 16 - 4 = 12 perimeter vertices
    // Closed path has first == last, so 12 unique vertices means 13 in path
    REQUIRE(paths[0].size() == 13);
    REQUIRE(paths[0].front() == paths[0].back());
}

// =============================================================================
// Brute Force Verification
// =============================================================================

TEMPLATE_TEST_CASE("make_boundary_edges_brute_force_verification", "[topology][boundary]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto mesh = tf::test::create_grid_mesh_3d<index_t, real_t>(4, 4);

    // Compute boundary edges using the library
    auto boundary = tf::make_boundary_edges(mesh.polygons());
    auto boundary_set = edges_to_set(boundary);

    // Brute force: count edge occurrences
    std::map<std::pair<index_t, index_t>, int> edge_counts;
    for (decltype(mesh.faces().size()) f = 0; f < mesh.faces().size(); ++f) {
        const auto& face = mesh.faces()[f];
        for (decltype(face.size()) i = 0; i < face.size(); ++i) {
            index_t v0 = face[i];
            index_t v1 = face[(i + 1) % face.size()];
            auto edge = canonicalize_edge(v0, v1);
            edge_counts[edge]++;
        }
    }

    // Boundary edges are those with count == 1
    std::set<std::pair<index_t, index_t>> expected_boundary;
    for (const auto& [edge, count] : edge_counts) {
        if (count == 1) {
            expected_boundary.insert(edge);
        }
    }

    // Compare
    REQUIRE(boundary_set == expected_boundary);
}
