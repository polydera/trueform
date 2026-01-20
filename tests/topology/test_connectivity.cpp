/**
 * @file test_connectivity.cpp
 * @brief Tests for mesh connectivity structures
 *
 * Tests for:
 * - make_face_membership
 * - make_manifold_edge_link
 * - make_face_link
 * - make_vertex_link
 *
 * Copyright (c) 2025 Ziga Sajovic, XLAB
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <trueform/trueform.hpp>
#include "type_traits.hpp"
#include "topology_generators.hpp"
#include <algorithm>
#include <set>

// =============================================================================
// make_face_membership
// =============================================================================

TEMPLATE_TEST_CASE("make_face_membership_two_triangles", "[topology][connectivity]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    // Two triangles: Face 0: [0,1,2], Face 1: [1,3,2]
    auto mesh = tf::test::create_two_triangles_3d<index_t, real_t>();
    auto fm = tf::make_face_membership(mesh.polygons());

    // Should have 4 entries (one per vertex)
    REQUIRE(fm.size() == 4);

    // Vertex 0 belongs to face 0 only
    REQUIRE(fm[0].size() == 1);
    REQUIRE(std::find(fm[0].begin(), fm[0].end(), index_t(0)) != fm[0].end());

    // Vertex 1 belongs to faces 0 and 1
    REQUIRE(fm[1].size() == 2);
    auto v1_faces = std::set<index_t>(fm[1].begin(), fm[1].end());
    REQUIRE(v1_faces.count(0));
    REQUIRE(v1_faces.count(1));

    // Vertex 2 belongs to faces 0 and 1
    REQUIRE(fm[2].size() == 2);
    auto v2_faces = std::set<index_t>(fm[2].begin(), fm[2].end());
    REQUIRE(v2_faces.count(0));
    REQUIRE(v2_faces.count(1));

    // Vertex 3 belongs to face 1 only
    REQUIRE(fm[3].size() == 1);
    REQUIRE(std::find(fm[3].begin(), fm[3].end(), index_t(1)) != fm[3].end());
}

TEMPLATE_TEST_CASE("make_face_membership_tetrahedron", "[topology][connectivity]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto mesh = tf::test::create_tetrahedron_3d<index_t, real_t>();
    auto fm = tf::make_face_membership(mesh.polygons());

    // 4 vertices
    REQUIRE(fm.size() == 4);

    // Each vertex in a tetrahedron belongs to exactly 3 faces
    for (decltype(fm.size()) i = 0; i < fm.size(); ++i) {
        REQUIRE(fm[i].size() == 3);
    }
}

TEMPLATE_TEST_CASE("make_face_membership_grid_mesh", "[topology][connectivity]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    // 3x3 grid = 9 vertices, 8 triangles
    auto mesh = tf::test::create_grid_mesh_3d<index_t, real_t>(3, 3);
    auto fm = tf::make_face_membership(mesh.polygons());

    REQUIRE(fm.size() == 9);

    // Corner vertex (0,0) belongs to 1-2 faces
    REQUIRE(fm[0].size() >= 1);

    // Center vertex belongs to 6 faces (fully surrounded)
    // Vertex 4 is the center in a 3x3 grid
    REQUIRE(fm[4].size() == 6);
}

TEMPLATE_TEST_CASE("make_face_membership_dynamic_mesh", "[topology][connectivity]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto mesh = tf::test::create_dynamic_mesh_3d<index_t, real_t>();
    auto fm = tf::make_face_membership(mesh.polygons());

    // 5 vertices
    REQUIRE(fm.size() == 5);

    // Vertex 0 belongs to both faces (triangle and quad)
    REQUIRE(fm[0].size() == 2);

    // Vertex 2 belongs to both faces (shared between triangle and quad)
    REQUIRE(fm[2].size() == 2);

    // Vertex 4 belongs only to the quad
    REQUIRE(fm[4].size() == 1);
}

// =============================================================================
// make_manifold_edge_link
// =============================================================================

TEMPLATE_TEST_CASE("make_manifold_edge_link_two_triangles", "[topology][connectivity]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto mesh = tf::test::create_two_triangles_3d<index_t, real_t>();
    auto mel = tf::make_manifold_edge_link(mesh.polygons());

    // 2 faces
    REQUIRE(mel.size() == 2);

    // Each face has 3 edges
    REQUIRE(mel[0].size() == 3);
    REQUIRE(mel[1].size() == 3);

    // Count boundary and simple edges in face 0
    int boundary_count = 0;
    int simple_count = 0;
    for (decltype(mel[0].size()) i = 0; i < mel[0].size(); ++i) {
        if (mel[0][i].is_boundary()) {
            boundary_count++;
        } else if (mel[0][i].is_simple()) {
            simple_count++;
        }
    }

    // Face 0 has 2 boundary edges and 1 shared edge
    REQUIRE(boundary_count == 2);
    REQUIRE(simple_count == 1);
}

TEMPLATE_TEST_CASE("make_manifold_edge_link_tetrahedron", "[topology][connectivity]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto mesh = tf::test::create_tetrahedron_3d<index_t, real_t>();
    auto mel = tf::make_manifold_edge_link(mesh.polygons());

    // 4 faces
    REQUIRE(mel.size() == 4);

    // Each face has 3 edges, all should be simple (no boundary)
    for (decltype(mel.size()) f = 0; f < mel.size(); ++f) {
        REQUIRE(mel[f].size() == 3);
        for (decltype(mel[f].size()) e = 0; e < mel[f].size(); ++e) {
            REQUIRE(mel[f][e].is_simple());
            REQUIRE_FALSE(mel[f][e].is_boundary());
        }
    }
}

TEMPLATE_TEST_CASE("make_manifold_edge_link_peer_consistency", "[topology][connectivity]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto mesh = tf::test::create_two_triangles_3d<index_t, real_t>();
    auto mel = tf::make_manifold_edge_link(mesh.polygons());

    // For each simple edge, verify that the peer relationship is symmetric
    for (decltype(mel.size()) f = 0; f < mel.size(); ++f) {
        for (decltype(mel[f].size()) e = 0; e < mel[f].size(); ++e) {
            if (mel[f][e].is_simple()) {
                index_t peer_face = mel[f][e].face_peer;
                // The peer face should have an edge that points back to face f
                bool found_back_reference = false;
                for (decltype(mel[peer_face].size()) pe = 0; pe < mel[peer_face].size(); ++pe) {
                    if (mel[peer_face][pe].is_simple() &&
                        mel[peer_face][pe].face_peer == static_cast<index_t>(f)) {
                        found_back_reference = true;
                        break;
                    }
                }
                REQUIRE(found_back_reference);
            }
        }
    }
}

// =============================================================================
// make_face_link
// =============================================================================

TEMPLATE_TEST_CASE("make_face_link_two_triangles", "[topology][connectivity]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto mesh = tf::test::create_two_triangles_3d<index_t, real_t>();
    auto fl = tf::make_face_link(mesh.polygons());

    // 2 faces
    REQUIRE(fl.size() == 2);

    // Face 0 is adjacent to face 1 (via shared edge)
    auto face0_neighbors = std::set<index_t>(fl[0].begin(), fl[0].end());
    REQUIRE(face0_neighbors.count(1));

    // Face 1 is adjacent to face 0
    auto face1_neighbors = std::set<index_t>(fl[1].begin(), fl[1].end());
    REQUIRE(face1_neighbors.count(0));
}

TEMPLATE_TEST_CASE("make_face_link_tetrahedron", "[topology][connectivity]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto mesh = tf::test::create_tetrahedron_3d<index_t, real_t>();
    auto fl = tf::make_face_link(mesh.polygons());

    // 4 faces
    REQUIRE(fl.size() == 4);

    // Each face in a tetrahedron is adjacent to exactly 3 other faces
    for (decltype(fl.size()) f = 0; f < fl.size(); ++f) {
        auto neighbors = std::set<index_t>(fl[f].begin(), fl[f].end());
        REQUIRE(neighbors.size() == 3);

        // Should not be adjacent to itself
        REQUIRE_FALSE(neighbors.count(static_cast<index_t>(f)));
    }
}

TEMPLATE_TEST_CASE("make_face_link_two_components", "[topology][connectivity]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto mesh = tf::test::create_two_components_3d<index_t, real_t>();
    auto fl = tf::make_face_link(mesh.polygons());

    // 2 faces (each is a single triangle in its own component)
    REQUIRE(fl.size() == 2);

    // Face 0 has no neighbors (isolated triangle)
    REQUIRE(fl[0].size() == 0);

    // Face 1 has no neighbors (isolated triangle)
    REQUIRE(fl[1].size() == 0);
}

TEMPLATE_TEST_CASE("make_face_link_grid_mesh", "[topology][connectivity]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    // 3x3 grid = 8 triangles (2 per cell, 4 cells)
    auto mesh = tf::test::create_grid_mesh_3d<index_t, real_t>(3, 3);
    auto fl = tf::make_face_link(mesh.polygons());

    // 8 faces
    REQUIRE(fl.size() == 8);

    // Each interior face should have neighbors
    // Verify all faces have at least 1 neighbor (no isolated faces in grid)
    for (decltype(fl.size()) f = 0; f < fl.size(); ++f) {
        REQUIRE(fl[f].size() >= 1);
    }
}

// =============================================================================
// make_vertex_link
// =============================================================================

TEMPLATE_TEST_CASE("make_vertex_link_two_triangles", "[topology][connectivity]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    // Two triangles: Face 0: [0,1,2], Face 1: [1,3,2]
    auto mesh = tf::test::create_two_triangles_3d<index_t, real_t>();
    auto vl = tf::make_vertex_link(mesh.polygons());

    // 4 vertices
    REQUIRE(vl.size() == 4);

    // Vertex 0's neighbors: {1, 2}
    auto v0_neighbors = std::set<index_t>(vl[0].begin(), vl[0].end());
    REQUIRE(v0_neighbors == std::set<index_t>{1, 2});

    // Vertex 1's neighbors: {0, 2, 3}
    auto v1_neighbors = std::set<index_t>(vl[1].begin(), vl[1].end());
    REQUIRE(v1_neighbors == std::set<index_t>{0, 2, 3});

    // Vertex 2's neighbors: {0, 1, 3}
    auto v2_neighbors = std::set<index_t>(vl[2].begin(), vl[2].end());
    REQUIRE(v2_neighbors == std::set<index_t>{0, 1, 3});

    // Vertex 3's neighbors: {1, 2}
    auto v3_neighbors = std::set<index_t>(vl[3].begin(), vl[3].end());
    REQUIRE(v3_neighbors == std::set<index_t>{1, 2});
}

TEMPLATE_TEST_CASE("make_vertex_link_tetrahedron", "[topology][connectivity]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto mesh = tf::test::create_tetrahedron_3d<index_t, real_t>();
    auto vl = tf::make_vertex_link(mesh.polygons());

    // 4 vertices
    REQUIRE(vl.size() == 4);

    // In a tetrahedron, each vertex is connected to all other 3 vertices
    for (decltype(vl.size()) v = 0; v < vl.size(); ++v) {
        auto neighbors = std::set<index_t>(vl[v].begin(), vl[v].end());
        REQUIRE(neighbors.size() == 3);

        // Should not be neighbor of itself
        REQUIRE_FALSE(neighbors.count(static_cast<index_t>(v)));

        // Should have all other vertices as neighbors
        for (decltype(vl.size()) other = 0; other < vl.size(); ++other) {
            if (other != v) {
                REQUIRE(neighbors.count(static_cast<index_t>(other)));
            }
        }
    }
}

TEMPLATE_TEST_CASE("make_vertex_link_grid_mesh", "[topology][connectivity]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    // 3x3 grid = 9 vertices
    auto mesh = tf::test::create_grid_mesh_3d<index_t, real_t>(3, 3);
    auto vl = tf::make_vertex_link(mesh.polygons());

    // 9 vertices
    REQUIRE(vl.size() == 9);

    // Center vertex (index 4) should have 6 neighbors in a triangulated grid
    // (4 adjacent + 2 diagonal due to triangulation)
    auto center_neighbors = std::set<index_t>(vl[4].begin(), vl[4].end());
    REQUIRE(center_neighbors.size() == 6);

    // Corner vertex (index 0) should have fewer neighbors
    auto corner_neighbors = std::set<index_t>(vl[0].begin(), vl[0].end());
    REQUIRE(corner_neighbors.size() >= 2);
    REQUIRE(corner_neighbors.size() <= 4);
}

TEMPLATE_TEST_CASE("make_vertex_link_two_components", "[topology][connectivity]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto mesh = tf::test::create_two_components_3d<index_t, real_t>();
    auto vl = tf::make_vertex_link(mesh.polygons());

    // 6 vertices
    REQUIRE(vl.size() == 6);

    // Vertices 0,1,2 should only be neighbors of each other (component 1)
    auto v0_neighbors = std::set<index_t>(vl[0].begin(), vl[0].end());
    REQUIRE(v0_neighbors == std::set<index_t>{1, 2});

    // Vertices 3,4,5 should only be neighbors of each other (component 2)
    auto v3_neighbors = std::set<index_t>(vl[3].begin(), vl[3].end());
    REQUIRE(v3_neighbors == std::set<index_t>{4, 5});

    // No cross-component neighbors
    REQUIRE_FALSE(v0_neighbors.count(3));
    REQUIRE_FALSE(v0_neighbors.count(4));
    REQUIRE_FALSE(v0_neighbors.count(5));
}

// =============================================================================
// Brute Force Verification
// =============================================================================

TEMPLATE_TEST_CASE("make_vertex_link_brute_force_verification", "[topology][connectivity]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto mesh = tf::test::create_grid_mesh_3d<index_t, real_t>(4, 4);
    auto vl = tf::make_vertex_link(mesh.polygons());

    // Brute force: compute neighbors by scanning all faces
    std::vector<std::set<index_t>> expected(mesh.points().size());
    for (decltype(mesh.faces().size()) f = 0; f < mesh.faces().size(); ++f) {
        const auto& face = mesh.faces()[f];
        for (decltype(face.size()) i = 0; i < face.size(); ++i) {
            index_t v = face[i];
            for (decltype(face.size()) j = 0; j < face.size(); ++j) {
                if (i != j) {
                    expected[v].insert(face[j]);
                }
            }
        }
    }

    // Compare
    REQUIRE(vl.size() == expected.size());
    for (decltype(vl.size()) v = 0; v < vl.size(); ++v) {
        auto actual = std::set<index_t>(vl[v].begin(), vl[v].end());
        REQUIRE(actual == expected[v]);
    }
}

TEMPLATE_TEST_CASE("make_face_membership_brute_force_verification", "[topology][connectivity]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto mesh = tf::test::create_grid_mesh_3d<index_t, real_t>(4, 4);
    auto fm = tf::make_face_membership(mesh.polygons());

    // Brute force: compute face membership by scanning all faces
    std::vector<std::set<index_t>> expected(mesh.points().size());
    for (decltype(mesh.faces().size()) f = 0; f < mesh.faces().size(); ++f) {
        const auto& face = mesh.faces()[f];
        for (decltype(face.size()) i = 0; i < face.size(); ++i) {
            expected[face[i]].insert(static_cast<index_t>(f));
        }
    }

    // Compare
    REQUIRE(fm.size() == expected.size());
    for (decltype(fm.size()) v = 0; v < fm.size(); ++v) {
        auto actual = std::set<index_t>(fm[v].begin(), fm[v].end());
        REQUIRE(actual == expected[v]);
    }
}
