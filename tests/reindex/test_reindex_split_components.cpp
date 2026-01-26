/**
 * @file test_reindex_split_components.cpp
 * @brief Tests for tf::split_into_components(...)
 *
 * Copyright (c) 2025 Ziga Sajovic, XLAB
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <trueform/trueform.hpp>
#include "type_traits.hpp"
#include "mesh_generators.hpp"

namespace {

template <typename Index>
auto make_labels(std::initializer_list<Index> values) -> tf::buffer<Index> {
    tf::buffer<Index> labels;
    labels.allocate(values.size());
    std::size_t i = 0;
    for (auto v : values) {
        labels[i++] = v;
    }
    return labels;
}

} // anonymous namespace

// =============================================================================
// split_components_polygons_two
// =============================================================================

TEMPLATE_TEST_CASE("split_components_polygons_two", "[reindex][split_components][polygons]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    // Create mesh with 2 faces
    auto input = tf::test::create_triangle_polygons_3d<index_t, real_t>();
    // 2 faces

    // Label each face differently
    auto labels = make_labels<index_t>({0, 1});

    auto [components, comp_labels] = tf::split_into_components(input.polygons(), labels);

    REQUIRE(components.size() == 2);
    REQUIRE(comp_labels.size() == 2);

    // Each component should have 1 face
    REQUIRE(components[0].faces().size() == 1);
    REQUIRE(components[1].faces().size() == 1);
}

// =============================================================================
// split_components_polygons_multiple
// =============================================================================

TEMPLATE_TEST_CASE("split_components_polygons_multiple", "[reindex][split_components][polygons]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto input = tf::test::create_larger_triangle_polygons_3d<index_t, real_t>();
    // 4 faces

    // 3 different labels
    auto labels = make_labels<index_t>({0, 1, 2, 0});

    auto [components, comp_labels] = tf::split_into_components(input.polygons(), labels);

    REQUIRE(components.size() == 3);
    REQUIRE(comp_labels.size() == 3);

    // Component 0 should have 2 faces (labels 0 at indices 0 and 3)
    std::size_t total_faces = 0;
    for (const auto& comp : components) {
        total_faces += comp.faces().size();
    }
    REQUIRE(total_faces == 4);
}

// =============================================================================
// split_components_polygons_single
// =============================================================================

TEMPLATE_TEST_CASE("split_components_polygons_single", "[reindex][split_components][polygons]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto input = tf::test::create_larger_triangle_polygons_3d<index_t, real_t>();
    // 4 faces

    // All same label
    auto labels = make_labels<index_t>({0, 0, 0, 0});

    auto [components, comp_labels] = tf::split_into_components(input.polygons(), labels);

    REQUIRE(components.size() == 1);
    REQUIRE(components[0].faces().size() == 4);
}

// =============================================================================
// split_components_polygons_integrity
// =============================================================================

TEMPLATE_TEST_CASE("split_components_polygons_integrity", "[reindex][split_components][polygons]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto input = tf::test::create_cube_polygons<index_t, real_t>();
    // 12 faces

    // Split into 2 components (e.g., top and bottom)
    tf::buffer<index_t> labels;
    labels.allocate(input.faces().size());
    for (std::size_t i = 0; i < input.faces().size(); ++i) {
        labels[i] = static_cast<index_t>(i % 2);
    }

    auto [components, comp_labels] = tf::split_into_components(input.polygons(), labels);

    // Verify each component has valid mesh structure
    for (const auto& comp : components) {
        // All face indices should reference valid points
        for (std::size_t f = 0; f < comp.faces().size(); ++f) {
            for (auto idx : comp.faces()[f]) {
                REQUIRE(static_cast<std::size_t>(idx) < comp.points().size());
            }
        }
    }
}

// =============================================================================
// split_components_polygons_dynamic
// =============================================================================

TEMPLATE_TEST_CASE("split_components_polygons_dynamic", "[reindex][split_components][polygons][dynamic]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto input = tf::test::create_dynamic_polygons_3d<index_t, real_t>();
    // 2 faces

    auto labels = make_labels<index_t>({0, 1});

    auto [components, comp_labels] = tf::split_into_components(input.polygons(), labels);

    REQUIRE(components.size() == 2);
}

// =============================================================================
// split_components_polygons_dynamic_mixed
// =============================================================================

TEMPLATE_TEST_CASE("split_components_polygons_dynamic_mixed", "[reindex][split_components][polygons][dynamic]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto input = tf::test::create_mixed_polygons_3d<index_t, real_t>();
    // 1 triangle + 1 quad

    auto labels = make_labels<index_t>({0, 1});

    auto [components, comp_labels] = tf::split_into_components(input.polygons(), labels);

    REQUIRE(components.size() == 2);

    // One should have 3 verts (triangle), one should have 4 verts (quad)
    bool has_tri = false, has_quad = false;
    for (const auto& comp : components) {
        if (comp.faces().size() == 1) {
            if (comp.faces()[0].size() == 3) has_tri = true;
            if (comp.faces()[0].size() == 4) has_quad = true;
        }
    }
    REQUIRE(has_tri);
    REQUIRE(has_quad);
}

// =============================================================================
// split_components_segments_basic
// =============================================================================

TEMPLATE_TEST_CASE("split_components_segments_basic", "[reindex][split_components][segments]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    tf::segments_buffer<index_t, real_t, 3> input;
    input.edges_buffer().emplace_back(0, 1);
    input.edges_buffer().emplace_back(1, 2);
    input.edges_buffer().emplace_back(3, 4);

    input.points_buffer().emplace_back(real_t(0), real_t(0), real_t(0));
    input.points_buffer().emplace_back(real_t(1), real_t(0), real_t(0));
    input.points_buffer().emplace_back(real_t(2), real_t(0), real_t(0));
    input.points_buffer().emplace_back(real_t(10), real_t(0), real_t(0));
    input.points_buffer().emplace_back(real_t(11), real_t(0), real_t(0));

    // First two edges are component 0, third is component 1
    auto labels = make_labels<index_t>({0, 0, 1});

    auto [components, comp_labels] = tf::split_into_components(input.segments(), labels);

    REQUIRE(components.size() == 2);
    REQUIRE(components[0].edges().size() == 2);
    REQUIRE(components[1].edges().size() == 1);
}

// =============================================================================
// split_components_segments_multiple
// =============================================================================

TEMPLATE_TEST_CASE("split_components_segments_multiple", "[reindex][split_components][segments]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    tf::segments_buffer<index_t, real_t, 3> input;
    input.edges_buffer().emplace_back(0, 1);
    input.edges_buffer().emplace_back(2, 3);
    input.edges_buffer().emplace_back(4, 5);
    input.edges_buffer().emplace_back(6, 7);

    for (int i = 0; i < 8; ++i) {
        input.points_buffer().emplace_back(real_t(i), real_t(0), real_t(0));
    }

    // 4 different components
    auto labels = make_labels<index_t>({0, 1, 2, 3});

    auto [components, comp_labels] = tf::split_into_components(input.segments(), labels);

    REQUIRE(components.size() == 4);
    for (const auto& comp : components) {
        REQUIRE(comp.edges().size() == 1);
    }
}

// =============================================================================
// split_concatenate_roundtrip_polygons
// =============================================================================

TEMPLATE_TEST_CASE("split_concatenate_roundtrip_polygons", "[reindex][split_components][roundtrip]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto input = tf::test::create_larger_triangle_polygons_3d<index_t, real_t>();

    // Split into 2 components
    auto labels = make_labels<index_t>({0, 0, 1, 1});

    auto [components, comp_labels] = tf::split_into_components(input.polygons(), labels);

    REQUIRE(components.size() == 2);

    // Concatenate back
    auto result = tf::concatenated(components[0].polygons(), components[1].polygons());

    // Should have same number of faces and points
    REQUIRE(result.faces().size() == input.faces().size());
    // Points may be duplicated since split removes shared points
}

// =============================================================================
// split_concatenate_roundtrip_segments
// =============================================================================

TEMPLATE_TEST_CASE("split_concatenate_roundtrip_segments", "[reindex][split_components][roundtrip]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    tf::segments_buffer<index_t, real_t, 3> input;
    input.edges_buffer().emplace_back(0, 1);
    input.edges_buffer().emplace_back(2, 3);
    input.edges_buffer().emplace_back(4, 5);
    input.edges_buffer().emplace_back(6, 7);

    for (int i = 0; i < 8; ++i) {
        input.points_buffer().emplace_back(real_t(i), real_t(0), real_t(0));
    }

    // Split into 2 components
    auto labels = make_labels<index_t>({0, 0, 1, 1});

    auto [components, comp_labels] = tf::split_into_components(input.segments(), labels);

    REQUIRE(components.size() == 2);

    // Concatenate back
    auto result = tf::concatenated(components[0].segments(), components[1].segments());

    // Should have same number of edges
    REQUIRE(result.edges().size() == input.edges().size());
}

// =============================================================================
// split_components_labels_order
// =============================================================================

TEMPLATE_TEST_CASE("split_components_labels_order", "[reindex][split_components][polygons]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto input = tf::test::create_larger_triangle_polygons_3d<index_t, real_t>();
    // 4 faces

    // Non-consecutive labels
    auto labels = make_labels<index_t>({5, 10, 5, 10});

    auto [components, comp_labels] = tf::split_into_components(input.polygons(), labels);

    REQUIRE(components.size() == 2);

    // comp_labels should contain 5 and 10
    bool has_5 = false, has_10 = false;
    for (auto l : comp_labels) {
        if (l == 5) has_5 = true;
        if (l == 10) has_10 = true;
    }
    REQUIRE(has_5);
    REQUIRE(has_10);
}

// =============================================================================
// split_components_empty_input
// =============================================================================

TEMPLATE_TEST_CASE("split_components_empty_input", "[reindex][split_components][polygons]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    tf::polygons_buffer<index_t, real_t, 3, 3> input;
    tf::buffer<index_t> labels;

    auto [components, comp_labels] = tf::split_into_components(input.polygons(), labels);

    REQUIRE(components.size() == 0);
    REQUIRE(comp_labels.size() == 0);
}
