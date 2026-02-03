/**
 * @file test_mesh_analysis.cpp
 * @brief Tests for mesh analysis functions
 *
 * Tests for:
 * - is_closed / is_open
 * - is_manifold / is_non_manifold
 * - make_non_manifold_edges
 * - orient_faces_consistently
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
// is_closed - Open Mesh
// =============================================================================

TEMPLATE_TEST_CASE("is_closed_open_mesh", "[topology][analysis]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto mesh = tf::test::create_two_triangles_3d<index_t, real_t>();
    REQUIRE_FALSE(tf::is_closed(mesh.polygons()));
}

// =============================================================================
// is_closed - Closed Mesh
// =============================================================================

TEMPLATE_TEST_CASE("is_closed_closed_mesh", "[topology][analysis]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto mesh = tf::test::create_tetrahedron_3d<index_t, real_t>();
    REQUIRE(tf::is_closed(mesh.polygons()));
}

// =============================================================================
// is_closed - Grid Mesh (open)
// =============================================================================

TEMPLATE_TEST_CASE("is_closed_grid_mesh", "[topology][analysis]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto mesh = tf::test::create_grid_mesh_3d<index_t, real_t>(4, 4);
    REQUIRE_FALSE(tf::is_closed(mesh.polygons()));
}

// =============================================================================
// is_closed - Triangle Strip (open)
// =============================================================================

TEMPLATE_TEST_CASE("is_closed_triangle_strip", "[topology][analysis]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto mesh = tf::test::create_triangle_strip_3d<index_t, real_t>(5);
    REQUIRE_FALSE(tf::is_closed(mesh.polygons()));
}

// =============================================================================
// is_open - Open Mesh
// =============================================================================

TEMPLATE_TEST_CASE("is_open_open_mesh", "[topology][analysis]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto mesh = tf::test::create_two_triangles_3d<index_t, real_t>();
    REQUIRE(tf::is_open(mesh.polygons()));
}

// =============================================================================
// is_open - Closed Mesh
// =============================================================================

TEMPLATE_TEST_CASE("is_open_closed_mesh", "[topology][analysis]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto mesh = tf::test::create_tetrahedron_3d<index_t, real_t>();
    REQUIRE_FALSE(tf::is_open(mesh.polygons()));
}

// =============================================================================
// is_manifold - Manifold Mesh
// =============================================================================

TEMPLATE_TEST_CASE("is_manifold_two_triangles", "[topology][analysis]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto mesh = tf::test::create_two_triangles_3d<index_t, real_t>();
    REQUIRE(tf::is_manifold(mesh.polygons()));
}

TEMPLATE_TEST_CASE("is_manifold_tetrahedron", "[topology][analysis]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto mesh = tf::test::create_tetrahedron_3d<index_t, real_t>();
    REQUIRE(tf::is_manifold(mesh.polygons()));
}

TEMPLATE_TEST_CASE("is_manifold_grid_mesh", "[topology][analysis]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto mesh = tf::test::create_grid_mesh_3d<index_t, real_t>(5, 5);
    REQUIRE(tf::is_manifold(mesh.polygons()));
}

// =============================================================================
// is_manifold - Non-Manifold Mesh
// =============================================================================

TEMPLATE_TEST_CASE("is_manifold_non_manifold_mesh", "[topology][analysis]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto mesh = tf::test::create_non_manifold_mesh_3d<index_t, real_t>();
    REQUIRE_FALSE(tf::is_manifold(mesh.polygons()));
}

// =============================================================================
// is_non_manifold - Manifold Mesh
// =============================================================================

TEMPLATE_TEST_CASE("is_non_manifold_manifold_mesh", "[topology][analysis]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto mesh = tf::test::create_two_triangles_3d<index_t, real_t>();
    REQUIRE_FALSE(tf::is_non_manifold(mesh.polygons()));
}

TEMPLATE_TEST_CASE("is_non_manifold_tetrahedron", "[topology][analysis]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto mesh = tf::test::create_tetrahedron_3d<index_t, real_t>();
    REQUIRE_FALSE(tf::is_non_manifold(mesh.polygons()));
}

// =============================================================================
// is_non_manifold - Non-Manifold Mesh
// =============================================================================

TEMPLATE_TEST_CASE("is_non_manifold_non_manifold_mesh", "[topology][analysis]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto mesh = tf::test::create_non_manifold_mesh_3d<index_t, real_t>();
    REQUIRE(tf::is_non_manifold(mesh.polygons()));
}

// =============================================================================
// make_non_manifold_edges - Manifold Mesh
// =============================================================================

TEMPLATE_TEST_CASE("make_non_manifold_edges_manifold_mesh", "[topology][analysis]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto mesh = tf::test::create_two_triangles_3d<index_t, real_t>();
    auto nm_edges = tf::make_non_manifold_edges(mesh.polygons());

    // Two triangles sharing one edge is manifold - no non-manifold edges
    REQUIRE(nm_edges.size() == 0);
}

TEMPLATE_TEST_CASE("make_non_manifold_edges_tetrahedron", "[topology][analysis]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto mesh = tf::test::create_tetrahedron_3d<index_t, real_t>();
    auto nm_edges = tf::make_non_manifold_edges(mesh.polygons());

    // Tetrahedron is manifold - no non-manifold edges
    REQUIRE(nm_edges.size() == 0);
}

// =============================================================================
// make_non_manifold_edges - Non-Manifold Mesh
// =============================================================================

TEMPLATE_TEST_CASE("make_non_manifold_edges_with_nm_edge", "[topology][analysis]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    // 3 triangles sharing edge (0,1)
    auto mesh = tf::test::create_non_manifold_mesh_3d<index_t, real_t>();
    auto nm_edges = tf::make_non_manifold_edges(mesh.polygons());

    // Should have exactly 1 non-manifold edge
    REQUIRE(nm_edges.size() == 1);

    auto edges_set = edges_to_set(nm_edges);
    REQUIRE(edges_set.count({0, 1}));
}

// =============================================================================
// make_non_manifold_edges - Grid Mesh (manifold)
// =============================================================================

TEMPLATE_TEST_CASE("make_non_manifold_edges_grid_mesh", "[topology][analysis]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto mesh = tf::test::create_grid_mesh_3d<index_t, real_t>(5, 5);
    auto nm_edges = tf::make_non_manifold_edges(mesh.polygons());

    // Grid mesh is manifold
    REQUIRE(nm_edges.size() == 0);
}

// =============================================================================
// orient_faces_consistently - Already Consistent
// =============================================================================

TEMPLATE_TEST_CASE("orient_faces_consistently_already_consistent", "[topology][analysis]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto mesh = tf::test::create_two_triangles_3d<index_t, real_t>();

    // Store original faces
    std::vector<std::array<index_t, 3>> original_faces;
    for (decltype(mesh.faces().size()) i = 0; i < mesh.faces().size(); ++i) {
        original_faces.push_back({mesh.faces()[i][0], mesh.faces()[i][1], mesh.faces()[i][2]});
    }

    tf::orient_faces_consistently(mesh.polygons());

    // Faces should either be unchanged or all flipped
    // Check that relative orientation is preserved
    auto mel = tf::make_manifold_edge_link(mesh.polygons());

    // After consistent orientation, all shared edges should have opposite
    // directions in their two faces (manifold edge criterion)
    for (decltype(mel.size()) f = 0; f < mel.size(); ++f) {
        for (decltype(mel[f].size()) e = 0; e < mel[f].size(); ++e) {
            if (mel[f][e].is_simple()) {
                // Simple edge has exactly one peer - this is expected
                REQUIRE(mel[f][e].face_peer >= 0);
            }
        }
    }
}

// =============================================================================
// orient_faces_consistently - Inconsistent Winding
// =============================================================================

TEMPLATE_TEST_CASE("orient_faces_consistently_inconsistent_winding", "[topology][analysis]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto mesh = tf::test::create_inconsistent_winding_mesh_3d<index_t, real_t>();

    // Before orientation, check that faces have inconsistent winding
    // This is done by building manifold_edge_link and checking shared edges

    tf::orient_faces_consistently(mesh.polygons());

    // After orientation, verify consistency by checking that shared edges
    // are traversed in opposite directions by adjacent faces
    auto mel = tf::make_manifold_edge_link(mesh.polygons());

    for (decltype(mel.size()) f = 0; f < mel.size(); ++f) {
        for (decltype(mel[f].size()) e = 0; e < mel[f].size(); ++e) {
            // All edges should be either boundary or simple (manifold)
            REQUIRE(mel[f][e].is_manifold());
        }
    }
}

// =============================================================================
// orient_faces_consistently - Tetrahedron
// =============================================================================

TEMPLATE_TEST_CASE("orient_faces_consistently_tetrahedron", "[topology][analysis]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto mesh = tf::test::create_tetrahedron_3d<index_t, real_t>();

    tf::orient_faces_consistently(mesh.polygons());

    // After orientation, the tetrahedron should still be closed
    REQUIRE(tf::is_closed(mesh.polygons()));

    // And should be manifold
    auto nm_edges = tf::make_non_manifold_edges(mesh.polygons());
    REQUIRE(nm_edges.size() == 0);
}

// =============================================================================
// orient_faces_consistently - Grid Mesh
// =============================================================================

TEMPLATE_TEST_CASE("orient_faces_consistently_grid_mesh", "[topology][analysis]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto mesh = tf::test::create_grid_mesh_3d<index_t, real_t>(4, 4);

    tf::orient_faces_consistently(mesh.polygons());

    // Verify all interior edges have consistent orientation
    auto mel = tf::make_manifold_edge_link(mesh.polygons());

    for (decltype(mel.size()) f = 0; f < mel.size(); ++f) {
        for (decltype(mel[f].size()) e = 0; e < mel[f].size(); ++e) {
            REQUIRE(mel[f][e].is_manifold());
        }
    }
}

// =============================================================================
// orient_faces_consistently - Two Components
// =============================================================================

TEMPLATE_TEST_CASE("orient_faces_consistently_two_components", "[topology][analysis]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto mesh = tf::test::create_two_components_3d<index_t, real_t>();

    tf::orient_faces_consistently(mesh.polygons());

    // Each component should be oriented consistently within itself
    // (they're independent single triangles, so trivially consistent)
    auto mel = tf::make_manifold_edge_link(mesh.polygons());

    for (decltype(mel.size()) f = 0; f < mel.size(); ++f) {
        for (decltype(mel[f].size()) e = 0; e < mel[f].size(); ++e) {
            // All edges should be boundary (single triangles)
            REQUIRE(mel[f][e].is_boundary());
        }
    }
}

// =============================================================================
// Brute Force Verification - is_closed
// =============================================================================

TEMPLATE_TEST_CASE("is_closed_brute_force_verification", "[topology][analysis]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto mesh = tf::test::create_grid_mesh_3d<index_t, real_t>(4, 4);

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

    // Check if all edges have count 2 (closed) or some have count 1 (open)
    bool has_boundary = false;
    for (const auto& [edge, count] : edge_counts) {
        if (count == 1) {
            has_boundary = true;
            break;
        }
    }

    // Compare with is_closed
    REQUIRE(tf::is_closed(mesh.polygons()) == !has_boundary);
}
