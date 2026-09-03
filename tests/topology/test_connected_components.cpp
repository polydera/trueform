/**
 * @file test_connected_components.cpp
 * @brief Tests for connected component labeling functions
 *
 * Tests for:
 * - make_manifold_edge_connected_component_labels (per-face labels)
 * - make_edge_connected_component_labels (per-face labels)
 * - make_vertex_connected_component_labels (per-vertex labels)
 *
 * The three rules answer in different carriers and part ways only where a
 * mesh is non-manifold, so the fixtures that tell them apart are a fan on
 * one edge and a pinch at one vertex. The label partition is the semantics:
 * a count alone never states which faces joined.
 *
 * Copyright (c) 2025 Ziga Sajovic, XLAB
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <trueform/trueform.hpp>
#include "type_traits.hpp"
#include "topology_generators.hpp"
#include <cstdint>
#include <set>

namespace {

/**
 * @brief Three triangles on the one edge (0, 1) - a non-manifold edge.
 *
 *        2                   face 0: (0, 1, 2)   in +y
 *        |                   face 1: (1, 0, 3)   in -y
 *   0 ---+--- 1              face 2: (0, 1, 4)   in +z, out of the page
 *        |
 *        3
 */
template <typename Index, typename Real>
auto components_three_fins_3d() -> tf::polygons_buffer<Index, Real, 3, 3> {
    tf::polygons_buffer<Index, Real, 3, 3> mesh;
    mesh.points_buffer().emplace_back(Real(0), Real(0), Real(0));
    mesh.points_buffer().emplace_back(Real(1), Real(0), Real(0));
    mesh.points_buffer().emplace_back(Real(0.5), Real(1), Real(0));
    mesh.points_buffer().emplace_back(Real(0.5), Real(-1), Real(0));
    mesh.points_buffer().emplace_back(Real(0.5), Real(0), Real(1));
    mesh.faces_buffer().emplace_back(Index(0), Index(1), Index(2));
    mesh.faces_buffer().emplace_back(Index(1), Index(0), Index(3));
    mesh.faces_buffer().emplace_back(Index(0), Index(1), Index(4));
    return mesh;
}

/**
 * @brief Two triangles meeting at the one vertex 2 - a non-manifold vertex.
 *
 *   1           4            face 0: (0, 1, 2)
 *   | \       / |            face 1: (2, 3, 4)
 *   |   \   /   |
 *   |     \/    |            no edge is shared
 *   0-----2-----3
 */
template <typename Index, typename Real>
auto components_bowtie_3d() -> tf::polygons_buffer<Index, Real, 3, 3> {
    tf::polygons_buffer<Index, Real, 3, 3> mesh;
    mesh.points_buffer().emplace_back(Real(-1), Real(0), Real(0));
    mesh.points_buffer().emplace_back(Real(-1), Real(1), Real(0));
    mesh.points_buffer().emplace_back(Real(0), Real(0), Real(0));
    mesh.points_buffer().emplace_back(Real(1), Real(0), Real(0));
    mesh.points_buffer().emplace_back(Real(1), Real(1), Real(0));
    mesh.faces_buffer().emplace_back(Index(0), Index(1), Index(2));
    mesh.faces_buffer().emplace_back(Index(2), Index(3), Index(4));
    return mesh;
}

/**
 * @brief Two triangles on one ordinary edge - manifold everywhere.
 *
 *   2-----3                  face 0: (0, 1, 2)
 *   | \   |                  face 1: (1, 3, 2)
 *   |   \ |
 *   0-----1                  the diagonal (1, 2) is the one shared edge
 */
template <typename Index, typename Real>
auto components_manifold_strip_3d() -> tf::polygons_buffer<Index, Real, 3, 3> {
    tf::polygons_buffer<Index, Real, 3, 3> mesh;
    mesh.points_buffer().emplace_back(Real(0), Real(0), Real(0));
    mesh.points_buffer().emplace_back(Real(1), Real(0), Real(0));
    mesh.points_buffer().emplace_back(Real(0), Real(1), Real(0));
    mesh.points_buffer().emplace_back(Real(1), Real(1), Real(0));
    mesh.faces_buffer().emplace_back(Index(0), Index(1), Index(2));
    mesh.faces_buffer().emplace_back(Index(1), Index(3), Index(2));
    return mesh;
}

/**
 * @brief The manifold strip plus a triangle that touches nothing.
 *
 *   2-----3                  6                face 0: (0, 1, 2)
 *   | \   |                  | \              face 1: (1, 3, 2)
 *   |   \ |                  |   \            face 2: (4, 5, 6)
 *   0-----1                  4-----5
 */
template <typename Index, typename Real>
auto components_strip_and_far_triangle_3d()
    -> tf::polygons_buffer<Index, Real, 3, 3> {
    auto mesh = components_manifold_strip_3d<Index, Real>();
    mesh.points_buffer().emplace_back(Real(10), Real(0), Real(0));
    mesh.points_buffer().emplace_back(Real(11), Real(0), Real(0));
    mesh.points_buffer().emplace_back(Real(10), Real(1), Real(0));
    mesh.faces_buffer().emplace_back(Index(4), Index(5), Index(6));
    return mesh;
}

/**
 * @brief One triangle - every edge a boundary edge.
 *
 *   2
 *   | \                      face 0: (0, 1, 2)
 *   |   \
 *   0-----1
 */
template <typename Index, typename Real>
auto components_single_triangle_3d()
    -> tf::polygons_buffer<Index, Real, 3, 3> {
    tf::polygons_buffer<Index, Real, 3, 3> mesh;
    mesh.points_buffer().emplace_back(Real(0), Real(0), Real(0));
    mesh.points_buffer().emplace_back(Real(1), Real(0), Real(0));
    mesh.points_buffer().emplace_back(Real(0), Real(1), Real(0));
    mesh.faces_buffer().emplace_back(Index(0), Index(1), Index(2));
    return mesh;
}

template <typename Labels>
auto components_share_one_label(const Labels &components) -> bool {
    for (decltype(components.labels.size()) i = 1;
         i < components.labels.size(); ++i)
        if (components.labels[i] != components.labels[0])
            return false;
    return true;
}

template <typename Labels>
auto components_labels_all_distinct(const Labels &components) -> bool {
    std::set<typename Labels::label_type> seen;
    for (decltype(components.labels.size()) i = 0;
         i < components.labels.size(); ++i)
        seen.insert(components.labels[i]);
    return seen.size() == components.labels.size();
}

} // namespace

// =============================================================================
// make_vertex_connected_component_labels - Single Component
// =============================================================================

TEMPLATE_TEST_CASE("make_vertex_connected_component_labels_single", "[topology][components]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto mesh = tf::test::create_two_triangles_3d<index_t, real_t>();
    auto components = tf::make_vertex_connected_component_labels(mesh.polygons());

    REQUIRE(components.n_components == 1);

    // All vertices should have the same label
    for (decltype(components.labels.size()) i = 1; i < components.labels.size(); ++i) {
        REQUIRE(components.labels[i] == components.labels[0]);
    }
}

// =============================================================================
// make_vertex_connected_component_labels - Two Components
// =============================================================================

TEMPLATE_TEST_CASE("make_vertex_connected_component_labels_two", "[topology][components]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto mesh = tf::test::create_two_components_3d<index_t, real_t>();
    auto components = tf::make_vertex_connected_component_labels(mesh.polygons());

    REQUIRE(components.n_components == 2);

    // Vertices 0,1,2 should have one label
    REQUIRE(components.labels[0] == components.labels[1]);
    REQUIRE(components.labels[1] == components.labels[2]);

    // Vertices 3,4,5 should have another label
    REQUIRE(components.labels[3] == components.labels[4]);
    REQUIRE(components.labels[4] == components.labels[5]);

    // The two groups should have different labels
    REQUIRE(components.labels[0] != components.labels[3]);
}

// =============================================================================
// make_vertex_connected_component_labels - Tetrahedron
// =============================================================================

TEMPLATE_TEST_CASE("make_vertex_connected_component_labels_tetrahedron", "[topology][components]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto mesh = tf::test::create_tetrahedron_3d<index_t, real_t>();
    auto components = tf::make_vertex_connected_component_labels(mesh.polygons());

    REQUIRE(components.n_components == 1);
    REQUIRE(components.labels.size() == 4);

    // All vertices should have the same label
    for (decltype(components.labels.size()) i = 1; i < components.labels.size(); ++i) {
        REQUIRE(components.labels[i] == components.labels[0]);
    }
}

// =============================================================================
// make_vertex_connected_component_labels - Grid Mesh
// =============================================================================

TEMPLATE_TEST_CASE("make_vertex_connected_component_labels_grid", "[topology][components]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto mesh = tf::test::create_grid_mesh_3d<index_t, real_t>(5, 5);
    auto components = tf::make_vertex_connected_component_labels(mesh.polygons());

    // Grid is fully connected
    REQUIRE(components.n_components == 1);
    REQUIRE(components.labels.size() == 25);

    // All vertices should have the same label
    for (decltype(components.labels.size()) i = 1; i < components.labels.size(); ++i) {
        REQUIRE(components.labels[i] == components.labels[0]);
    }
}

// =============================================================================
// make_edge_connected_component_labels - Single Component
// =============================================================================

TEMPLATE_TEST_CASE("make_edge_connected_component_labels_single", "[topology][components]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto mesh = tf::test::create_two_triangles_3d<index_t, real_t>();
    auto components = tf::make_edge_connected_component_labels(mesh.polygons());

    REQUIRE(components.n_components == 1);
    REQUIRE(components.labels.size() == 2); // 2 faces

    // All faces should have the same label
    REQUIRE(components.labels[0] == components.labels[1]);
}

// =============================================================================
// make_edge_connected_component_labels - Two Components
// =============================================================================

TEMPLATE_TEST_CASE("make_edge_connected_component_labels_two", "[topology][components]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto mesh = tf::test::create_two_components_3d<index_t, real_t>();
    auto components = tf::make_edge_connected_component_labels(mesh.polygons());

    REQUIRE(components.n_components == 2);
    REQUIRE(components.labels.size() == 2); // 2 faces

    // The two faces should have different labels
    REQUIRE(components.labels[0] != components.labels[1]);
}

// =============================================================================
// make_edge_connected_component_labels - Tetrahedron
// =============================================================================

TEMPLATE_TEST_CASE("make_edge_connected_component_labels_tetrahedron", "[topology][components]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto mesh = tf::test::create_tetrahedron_3d<index_t, real_t>();
    auto components = tf::make_edge_connected_component_labels(mesh.polygons());

    REQUIRE(components.n_components == 1);
    REQUIRE(components.labels.size() == 4); // 4 faces

    // All faces should have the same label
    for (decltype(components.labels.size()) i = 1; i < components.labels.size(); ++i) {
        REQUIRE(components.labels[i] == components.labels[0]);
    }
}

// =============================================================================
// make_edge_connected_component_labels - Grid Mesh
// =============================================================================

TEMPLATE_TEST_CASE("make_edge_connected_component_labels_grid", "[topology][components]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    // 4x4 grid has 18 triangles (2 per cell, 9 cells)
    auto mesh = tf::test::create_grid_mesh_3d<index_t, real_t>(4, 4);
    auto components = tf::make_edge_connected_component_labels(mesh.polygons());

    // Grid is fully connected
    REQUIRE(components.n_components == 1);

    // All faces should have the same label
    for (decltype(components.labels.size()) i = 1; i < components.labels.size(); ++i) {
        REQUIRE(components.labels[i] == components.labels[0]);
    }
}

// =============================================================================
// make_edge_connected_component_labels - Triangle Strip
// =============================================================================

TEMPLATE_TEST_CASE("make_edge_connected_component_labels_strip", "[topology][components]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto mesh = tf::test::create_triangle_strip_3d<index_t, real_t>(5);
    auto components = tf::make_edge_connected_component_labels(mesh.polygons());

    // Strip is fully connected
    REQUIRE(components.n_components == 1);
    REQUIRE(components.labels.size() == 5); // 5 triangles

    // All faces should have the same label
    for (decltype(components.labels.size()) i = 1; i < components.labels.size(); ++i) {
        REQUIRE(components.labels[i] == components.labels[0]);
    }
}

// =============================================================================
// Component Labels are Valid
// =============================================================================

TEMPLATE_TEST_CASE("component_labels_are_valid", "[topology][components]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto mesh = tf::test::create_two_components_3d<index_t, real_t>();

    SECTION("vertex components") {
        auto components = tf::make_vertex_connected_component_labels(mesh.polygons());

        // Labels should be in range [0, n_components)
        std::set<index_t> unique_labels;
        for (decltype(components.labels.size()) i = 0; i < components.labels.size(); ++i) {
            REQUIRE(components.labels[i] >= 0);
            REQUIRE(components.labels[i] < static_cast<index_t>(components.n_components));
            unique_labels.insert(components.labels[i]);
        }

        // Number of unique labels should equal n_components
        REQUIRE(index_t(unique_labels.size()) == components.n_components);
    }

    SECTION("edge components") {
        auto components = tf::make_edge_connected_component_labels(mesh.polygons());

        // Labels should be in range [0, n_components)
        std::set<index_t> unique_labels;
        for (decltype(components.labels.size()) i = 0; i < components.labels.size(); ++i) {
            REQUIRE(components.labels[i] >= 0);
            REQUIRE(components.labels[i] < static_cast<index_t>(components.n_components));
            unique_labels.insert(components.labels[i]);
        }

        // Number of unique labels should equal n_components
        REQUIRE(index_t(unique_labels.size()) == components.n_components);
    }
}

// =============================================================================
// Brute Force Verification - Vertex Components
// =============================================================================

TEMPLATE_TEST_CASE("vertex_components_brute_force_verification", "[topology][components]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto mesh = tf::test::create_two_components_3d<index_t, real_t>();
    auto components = tf::make_vertex_connected_component_labels(mesh.polygons());
    auto vl = tf::make_vertex_link(mesh.polygons());

    // Verify: vertices with the same label are connected via edge path
    // Vertices with different labels are not connected

    // For each pair of vertices
    for (decltype(components.labels.size()) i = 0; i < components.labels.size(); ++i) {
        for (decltype(components.labels.size()) j = i + 1; j < components.labels.size(); ++j) {
            bool same_component = (components.labels[i] == components.labels[j]);

            // BFS to check if i and j are connected
            std::set<index_t> visited;
            std::vector<index_t> queue;
            queue.push_back(static_cast<index_t>(i));
            visited.insert(static_cast<index_t>(i));

            while (!queue.empty()) {
                index_t current = queue.back();
                queue.pop_back();

                for (auto neighbor : vl[current]) {
                    if (!visited.count(neighbor)) {
                        visited.insert(neighbor);
                        queue.push_back(neighbor);
                    }
                }
            }

            bool actually_connected = visited.count(static_cast<index_t>(j)) > 0;
            REQUIRE(same_component == actually_connected);
        }
    }
}

// =============================================================================
// Brute Force Verification - Edge Components
// =============================================================================

TEMPLATE_TEST_CASE("edge_components_brute_force_verification", "[topology][components]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto mesh = tf::test::create_two_components_3d<index_t, real_t>();
    auto components = tf::make_edge_connected_component_labels(mesh.polygons());
    auto fl = tf::make_face_link(mesh.polygons());

    // Verify: faces with the same label are connected via shared edges
    // Faces with different labels are not connected

    for (decltype(components.labels.size()) i = 0; i < components.labels.size(); ++i) {
        for (decltype(components.labels.size()) j = i + 1; j < components.labels.size(); ++j) {
            bool same_component = (components.labels[i] == components.labels[j]);

            // BFS to check if face i and face j are connected
            std::set<index_t> visited;
            std::vector<index_t> queue;
            queue.push_back(static_cast<index_t>(i));
            visited.insert(static_cast<index_t>(i));

            while (!queue.empty()) {
                index_t current = queue.back();
                queue.pop_back();

                for (auto neighbor : fl[current]) {
                    if (!visited.count(neighbor)) {
                        visited.insert(neighbor);
                        queue.push_back(neighbor);
                    }
                }
            }

            bool actually_connected = visited.count(static_cast<index_t>(j)) > 0;
            REQUIRE(same_component == actually_connected);
        }
    }
}

// =============================================================================
// Three fins on one edge - the edge is non-manifold
// =============================================================================

TEMPLATE_TEST_CASE("connected_components_three_fins_on_one_edge", "[topology][components]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto mesh = components_three_fins_3d<index_t, real_t>();

    auto manifold_edge =
        tf::make_manifold_edge_connected_component_labels(mesh.polygons());
    REQUIRE(manifold_edge.labels.size() == 3);
    CHECK(manifold_edge.n_components == 3);
    CHECK(components_labels_all_distinct(manifold_edge));

    auto edge = tf::make_edge_connected_component_labels(mesh.polygons());
    REQUIRE(edge.labels.size() == 3);
    CHECK(edge.n_components == 1);
    CHECK(edge.labels[0] == edge.labels[1]);
    CHECK(edge.labels[1] == edge.labels[2]);

    auto vertex = tf::make_vertex_connected_component_labels(mesh.polygons());
    REQUIRE(vertex.labels.size() == 5);
    CHECK(vertex.n_components == 1);
    CHECK(components_share_one_label(vertex));
}

// =============================================================================
// Bowtie - the vertex is non-manifold, no edge is shared
// =============================================================================

TEMPLATE_TEST_CASE("connected_components_bowtie_pinch", "[topology][components]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto mesh = components_bowtie_3d<index_t, real_t>();

    auto manifold_edge =
        tf::make_manifold_edge_connected_component_labels(mesh.polygons());
    REQUIRE(manifold_edge.labels.size() == 2);
    CHECK(manifold_edge.n_components == 2);
    CHECK(manifold_edge.labels[0] != manifold_edge.labels[1]);

    auto edge = tf::make_edge_connected_component_labels(mesh.polygons());
    REQUIRE(edge.labels.size() == 2);
    CHECK(edge.n_components == 2);
    CHECK(edge.labels[0] != edge.labels[1]);

    auto vertex = tf::make_vertex_connected_component_labels(mesh.polygons());
    REQUIRE(vertex.labels.size() == 5);
    CHECK(vertex.n_components == 1);
    CHECK(components_share_one_label(vertex));
}

// =============================================================================
// Manifold strip - permissiveness only ever joins, so all three agree
// =============================================================================

TEMPLATE_TEST_CASE("connected_components_manifold_strip_agrees", "[topology][components]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto mesh = components_manifold_strip_3d<index_t, real_t>();

    auto manifold_edge =
        tf::make_manifold_edge_connected_component_labels(mesh.polygons());
    REQUIRE(manifold_edge.labels.size() == 2);
    CHECK(manifold_edge.n_components == 1);
    CHECK(components_share_one_label(manifold_edge));

    auto edge = tf::make_edge_connected_component_labels(mesh.polygons());
    REQUIRE(edge.labels.size() == 2);
    CHECK(edge.n_components == 1);
    CHECK(components_share_one_label(edge));

    auto vertex = tf::make_vertex_connected_component_labels(mesh.polygons());
    REQUIRE(vertex.labels.size() == 4);
    CHECK(vertex.n_components == 1);
    CHECK(components_share_one_label(vertex));
}

// =============================================================================
// Disjoint pieces - two components, each producer counting its own carrier
// =============================================================================

TEMPLATE_TEST_CASE("connected_components_disjoint_pieces", "[topology][components]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto mesh = components_strip_and_far_triangle_3d<index_t, real_t>();

    auto manifold_edge =
        tf::make_manifold_edge_connected_component_labels(mesh.polygons());
    REQUIRE(manifold_edge.labels.size() == 3);
    CHECK(manifold_edge.n_components == 2);
    CHECK(manifold_edge.labels[0] == manifold_edge.labels[1]);
    CHECK(manifold_edge.labels[2] != manifold_edge.labels[0]);

    auto edge = tf::make_edge_connected_component_labels(mesh.polygons());
    REQUIRE(edge.labels.size() == 3);
    CHECK(edge.n_components == 2);
    CHECK(edge.labels[0] == edge.labels[1]);
    CHECK(edge.labels[2] != edge.labels[0]);

    auto vertex = tf::make_vertex_connected_component_labels(mesh.polygons());
    REQUIRE(vertex.labels.size() == 7);
    CHECK(vertex.n_components == 2);
    for (index_t i = 1; i < 4; ++i)
        CHECK(vertex.labels[i] == vertex.labels[0]);
    for (index_t i = 5; i < 7; ++i)
        CHECK(vertex.labels[i] == vertex.labels[4]);
    CHECK(vertex.labels[4] != vertex.labels[0]);
}

// =============================================================================
// One triangle, and no triangle at all
// =============================================================================

TEMPLATE_TEST_CASE("connected_components_boundary_and_empty", "[topology][components]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto triangle = components_single_triangle_3d<index_t, real_t>();

    auto manifold_edge =
        tf::make_manifold_edge_connected_component_labels(triangle.polygons());
    REQUIRE(manifold_edge.labels.size() == 1);
    CHECK(manifold_edge.n_components == 1);

    auto edge = tf::make_edge_connected_component_labels(triangle.polygons());
    REQUIRE(edge.labels.size() == 1);
    CHECK(edge.n_components == 1);

    auto vertex = tf::make_vertex_connected_component_labels(triangle.polygons());
    REQUIRE(vertex.labels.size() == 3);
    CHECK(vertex.n_components == 1);
    CHECK(components_share_one_label(vertex));

    tf::polygons_buffer<index_t, real_t, 3, 3> empty;

    auto empty_manifold_edge =
        tf::make_manifold_edge_connected_component_labels(empty.polygons());
    CHECK(empty_manifold_edge.labels.size() == 0);
    CHECK(empty_manifold_edge.n_components == 0);

    auto empty_edge = tf::make_edge_connected_component_labels(empty.polygons());
    CHECK(empty_edge.labels.size() == 0);
    CHECK(empty_edge.n_components == 0);

    auto empty_vertex =
        tf::make_vertex_connected_component_labels(empty.polygons());
    CHECK(empty_vertex.labels.size() == 0);
    CHECK(empty_vertex.n_components == 0);
}
