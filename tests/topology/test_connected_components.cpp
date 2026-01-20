/**
 * @file test_connected_components.cpp
 * @brief Tests for connected component labeling functions
 *
 * Tests for:
 * - make_vertex_connected_component_labels
 * - make_edge_connected_component_labels
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
