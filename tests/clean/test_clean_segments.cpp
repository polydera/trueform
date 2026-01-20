/**
 * @file test_clean_segments.cpp
 * @brief Tests for tf::cleaned(segments, ...)
 *
 * Tests for:
 * - clean_segments_no_duplicates
 * - clean_segments_duplicate_edges
 * - clean_segments_reversed_edges
 * - clean_segments_degenerate_edges
 * - clean_segments_tolerance
 * - clean_segments_unreferenced_points
 * - clean_segments_with_index_map
 * - clean_segments_empty
 *
 * Copyright (c) 2025 Ziga Sajovic, XLAB
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <trueform/trueform.hpp>
#include "type_traits.hpp"
#include "canonicalize_segments.hpp"

// =============================================================================
// clean_segments_no_duplicates
// =============================================================================

TEMPLATE_TEST_CASE("clean_segments_no_duplicates", "[clean][clean_segments]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    // Create segments with no duplicates
    tf::segments_buffer<index_t, real_t, 3> input;
    input.edges_buffer().emplace_back(0, 1);
    input.edges_buffer().emplace_back(1, 2);
    input.edges_buffer().emplace_back(2, 3);

    input.points_buffer().emplace_back(real_t(0), real_t(0), real_t(0));
    input.points_buffer().emplace_back(real_t(1), real_t(0), real_t(0));
    input.points_buffer().emplace_back(real_t(2), real_t(0), real_t(0));
    input.points_buffer().emplace_back(real_t(3), real_t(0), real_t(0));

    auto result = tf::cleaned(input.segments());

    // Segments should remain unchanged
    REQUIRE(result.edges().size() == 3);
    REQUIRE(result.points().size() == 4);

    // Verify canonical comparison
    auto canonical_result = tf::test::canonicalize_segments(result);
    auto canonical_expected = tf::test::canonicalize_segments(input);
    REQUIRE(tf::test::segments_equal(canonical_result, canonical_expected));
}

// =============================================================================
// clean_segments_duplicate_edges
// =============================================================================

TEMPLATE_TEST_CASE("clean_segments_duplicate_edges", "[clean][clean_segments]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    // Create segments with duplicate edges
    tf::segments_buffer<index_t, real_t, 3> input;
    input.edges_buffer().emplace_back(0, 1);
    input.edges_buffer().emplace_back(1, 2);
    input.edges_buffer().emplace_back(0, 1);  // duplicate
    input.edges_buffer().emplace_back(2, 3);

    input.points_buffer().emplace_back(real_t(0), real_t(0), real_t(0));
    input.points_buffer().emplace_back(real_t(1), real_t(0), real_t(0));
    input.points_buffer().emplace_back(real_t(2), real_t(0), real_t(0));
    input.points_buffer().emplace_back(real_t(3), real_t(0), real_t(0));

    auto result = tf::cleaned(input.segments());

    // Should have 3 unique edges
    REQUIRE(result.edges().size() == 3);
}

// =============================================================================
// clean_segments_reversed_edges
// =============================================================================

TEMPLATE_TEST_CASE("clean_segments_reversed_edges", "[clean][clean_segments]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    // Create segments with reversed duplicate edges
    tf::segments_buffer<index_t, real_t, 3> input;
    input.edges_buffer().emplace_back(0, 1);
    input.edges_buffer().emplace_back(1, 2);
    input.edges_buffer().emplace_back(1, 0);  // reversed duplicate of (0,1)
    input.edges_buffer().emplace_back(2, 3);

    input.points_buffer().emplace_back(real_t(0), real_t(0), real_t(0));
    input.points_buffer().emplace_back(real_t(1), real_t(0), real_t(0));
    input.points_buffer().emplace_back(real_t(2), real_t(0), real_t(0));
    input.points_buffer().emplace_back(real_t(3), real_t(0), real_t(0));

    auto result = tf::cleaned(input.segments());

    // Should have 3 unique edges (reversed edge is considered duplicate)
    REQUIRE(result.edges().size() == 3);
}

// =============================================================================
// clean_segments_degenerate_edges
// =============================================================================

TEMPLATE_TEST_CASE("clean_segments_degenerate_edges", "[clean][clean_segments]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    // Create segments with degenerate edges (zero-length)
    tf::segments_buffer<index_t, real_t, 3> input;
    input.edges_buffer().emplace_back(0, 1);
    input.edges_buffer().emplace_back(1, 1);  // degenerate: same start/end
    input.edges_buffer().emplace_back(1, 2);
    input.edges_buffer().emplace_back(2, 2);  // degenerate

    input.points_buffer().emplace_back(real_t(0), real_t(0), real_t(0));
    input.points_buffer().emplace_back(real_t(1), real_t(0), real_t(0));
    input.points_buffer().emplace_back(real_t(2), real_t(0), real_t(0));

    auto result = tf::cleaned(input.segments());

    // Should have 2 non-degenerate edges
    REQUIRE(result.edges().size() == 2);
}

// =============================================================================
// clean_segments_tolerance
// =============================================================================

TEMPLATE_TEST_CASE("clean_segments_tolerance", "[clean][clean_segments]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    // Create segments with points within tolerance
    tf::segments_buffer<index_t, real_t, 3> input;
    input.edges_buffer().emplace_back(0, 1);
    input.edges_buffer().emplace_back(2, 3);

    input.points_buffer().emplace_back(real_t(0), real_t(0), real_t(0));
    input.points_buffer().emplace_back(real_t(1), real_t(0), real_t(0));
    input.points_buffer().emplace_back(real_t(0.001), real_t(0), real_t(0));  // near point 0
    input.points_buffer().emplace_back(real_t(2), real_t(0), real_t(0));

    const real_t tolerance = real_t(0.01);
    auto result = tf::cleaned(input.segments(), tolerance);

    // Points 0 and 2 should merge, reducing point count
    REQUIRE(result.points().size() == 3);
}

// =============================================================================
// clean_segments_unreferenced_points
// =============================================================================

TEMPLATE_TEST_CASE("clean_segments_unreferenced_points", "[clean][clean_segments]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    // Create segments with unreferenced points
    tf::segments_buffer<index_t, real_t, 3> input;
    input.edges_buffer().emplace_back(0, 1);
    input.edges_buffer().emplace_back(1, 2);

    // Point 3 is not referenced by any edge
    input.points_buffer().emplace_back(real_t(0), real_t(0), real_t(0));
    input.points_buffer().emplace_back(real_t(1), real_t(0), real_t(0));
    input.points_buffer().emplace_back(real_t(2), real_t(0), real_t(0));
    input.points_buffer().emplace_back(real_t(99), real_t(99), real_t(99));  // unreferenced

    auto result = tf::cleaned(input.segments());

    // Should have only 3 points (unreferenced removed)
    REQUIRE(result.points().size() == 3);
    REQUIRE(result.edges().size() == 2);
}

// =============================================================================
// clean_segments_with_index_map
// =============================================================================

TEMPLATE_TEST_CASE("clean_segments_with_index_map", "[clean][clean_segments]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    // Create segments with duplicates
    tf::segments_buffer<index_t, real_t, 3> input;
    input.edges_buffer().emplace_back(0, 1);  // -> edge 0
    input.edges_buffer().emplace_back(1, 2);  // -> edge 1
    input.edges_buffer().emplace_back(0, 1);  // duplicate -> edge 0

    input.points_buffer().emplace_back(real_t(0), real_t(0), real_t(0));
    input.points_buffer().emplace_back(real_t(1), real_t(0), real_t(0));
    input.points_buffer().emplace_back(real_t(2), real_t(0), real_t(0));

    auto [result, edge_im, point_im] = tf::cleaned(input.segments(), tf::return_index_map);

    // Should have 2 unique edges
    REQUIRE(result.edges().size() == 2);

    // Edge index map should have 3 entries
    REQUIRE(edge_im.f().size() == 3);

    // Duplicate edge should map to same output index
    REQUIRE(edge_im.f()[0] == edge_im.f()[2]);

    // Point index map - all points referenced, so should be identity-like
    REQUIRE(point_im.f().size() == 3);
}

// =============================================================================
// clean_segments_empty
// =============================================================================

TEMPLATE_TEST_CASE("clean_segments_empty", "[clean][clean_segments]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    tf::segments_buffer<index_t, real_t, 3> input;

    auto result = tf::cleaned(input.segments());

    REQUIRE(result.edges().size() == 0);
    REQUIRE(result.points().size() == 0);
}

// =============================================================================
// clean_segments_2d
// =============================================================================

TEMPLATE_TEST_CASE("clean_segments_2d", "[clean][clean_segments]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    // Test with 2D segments
    tf::segments_buffer<index_t, real_t, 2> input;
    input.edges_buffer().emplace_back(0, 1);
    input.edges_buffer().emplace_back(1, 2);
    input.edges_buffer().emplace_back(0, 1);  // duplicate

    input.points_buffer().emplace_back(real_t(0), real_t(0));
    input.points_buffer().emplace_back(real_t(1), real_t(0));
    input.points_buffer().emplace_back(real_t(2), real_t(0));

    auto result = tf::cleaned(input.segments());

    REQUIRE(result.edges().size() == 2);
}

// =============================================================================
// clean_segments_single_edge
// =============================================================================

TEMPLATE_TEST_CASE("clean_segments_single_edge", "[clean][clean_segments]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    tf::segments_buffer<index_t, real_t, 3> input;
    input.edges_buffer().emplace_back(0, 1);

    input.points_buffer().emplace_back(real_t(0), real_t(0), real_t(0));
    input.points_buffer().emplace_back(real_t(1), real_t(0), real_t(0));

    auto result = tf::cleaned(input.segments());

    REQUIRE(result.edges().size() == 1);
    REQUIRE(result.points().size() == 2);
}

// =============================================================================
// clean_segments_all_degenerate
// =============================================================================

TEMPLATE_TEST_CASE("clean_segments_all_degenerate", "[clean][clean_segments]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    // All edges are degenerate
    tf::segments_buffer<index_t, real_t, 3> input;
    input.edges_buffer().emplace_back(0, 0);
    input.edges_buffer().emplace_back(1, 1);

    input.points_buffer().emplace_back(real_t(0), real_t(0), real_t(0));
    input.points_buffer().emplace_back(real_t(1), real_t(0), real_t(0));

    auto result = tf::cleaned(input.segments());

    // Should have no edges, and points may be removed as unreferenced
    REQUIRE(result.edges().size() == 0);
}

// =============================================================================
// clean_segments_tolerance_creates_degenerate
// =============================================================================

TEMPLATE_TEST_CASE("clean_segments_tolerance_creates_degenerate", "[clean][clean_segments]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    // Edge becomes degenerate after point merging
    tf::segments_buffer<index_t, real_t, 3> input;
    input.edges_buffer().emplace_back(0, 1);  // will become degenerate
    input.edges_buffer().emplace_back(2, 3);  // will remain valid

    input.points_buffer().emplace_back(real_t(0), real_t(0), real_t(0));
    input.points_buffer().emplace_back(real_t(0.001), real_t(0), real_t(0));  // merges with 0
    input.points_buffer().emplace_back(real_t(1), real_t(0), real_t(0));
    input.points_buffer().emplace_back(real_t(2), real_t(0), real_t(0));

    const real_t tolerance = real_t(0.01);
    auto result = tf::cleaned(input.segments(), tolerance);

    // First edge should become degenerate and be removed
    REQUIRE(result.edges().size() == 1);
}

// =============================================================================
// clean_segments_wireframe_cube
// =============================================================================

TEMPLATE_TEST_CASE("clean_segments_wireframe_cube", "[clean][clean_segments]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    // Create wireframe cube edges
    tf::segments_buffer<index_t, real_t, 3> input;

    // 8 vertices of unit cube
    input.points_buffer().emplace_back(real_t(0), real_t(0), real_t(0));
    input.points_buffer().emplace_back(real_t(1), real_t(0), real_t(0));
    input.points_buffer().emplace_back(real_t(1), real_t(1), real_t(0));
    input.points_buffer().emplace_back(real_t(0), real_t(1), real_t(0));
    input.points_buffer().emplace_back(real_t(0), real_t(0), real_t(1));
    input.points_buffer().emplace_back(real_t(1), real_t(0), real_t(1));
    input.points_buffer().emplace_back(real_t(1), real_t(1), real_t(1));
    input.points_buffer().emplace_back(real_t(0), real_t(1), real_t(1));

    // 12 edges of cube
    // Bottom face
    input.edges_buffer().emplace_back(0, 1);
    input.edges_buffer().emplace_back(1, 2);
    input.edges_buffer().emplace_back(2, 3);
    input.edges_buffer().emplace_back(3, 0);
    // Top face
    input.edges_buffer().emplace_back(4, 5);
    input.edges_buffer().emplace_back(5, 6);
    input.edges_buffer().emplace_back(6, 7);
    input.edges_buffer().emplace_back(7, 4);
    // Vertical edges
    input.edges_buffer().emplace_back(0, 4);
    input.edges_buffer().emplace_back(1, 5);
    input.edges_buffer().emplace_back(2, 6);
    input.edges_buffer().emplace_back(3, 7);

    auto result = tf::cleaned(input.segments());

    // Clean cube should be unchanged
    REQUIRE(result.edges().size() == 12);
    REQUIRE(result.points().size() == 8);

    // Verify canonical comparison
    auto canonical_result = tf::test::canonicalize_segments(result);
    auto canonical_expected = tf::test::canonicalize_segments(input);
    REQUIRE(tf::test::segments_equal(canonical_result, canonical_expected));
}

// =============================================================================
// clean_segments_mesh_edges_with_duplicates
// =============================================================================

TEMPLATE_TEST_CASE("clean_segments_mesh_edges_with_duplicates", "[clean][clean_segments]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    // Simulate mesh edges where interior edges appear twice (from adjacent faces)
    tf::segments_buffer<index_t, real_t, 3> input;

    // Triangle strip: two triangles sharing an edge
    input.points_buffer().emplace_back(real_t(0), real_t(0), real_t(0));
    input.points_buffer().emplace_back(real_t(1), real_t(0), real_t(0));
    input.points_buffer().emplace_back(real_t(0.5), real_t(1), real_t(0));
    input.points_buffer().emplace_back(real_t(1.5), real_t(1), real_t(0));

    // First triangle edges: 0-1, 1-2, 2-0
    input.edges_buffer().emplace_back(0, 1);
    input.edges_buffer().emplace_back(1, 2);
    input.edges_buffer().emplace_back(2, 0);

    // Second triangle edges: 1-3, 3-2, 2-1
    input.edges_buffer().emplace_back(1, 3);
    input.edges_buffer().emplace_back(3, 2);
    input.edges_buffer().emplace_back(2, 1);  // duplicate of 1-2 (reversed)

    auto result = tf::cleaned(input.segments());

    // Should have 5 unique edges (1-2 and 2-1 merge)
    REQUIRE(result.edges().size() == 5);
}

// =============================================================================
// clean_segments_chain_with_duplicates
// =============================================================================

TEMPLATE_TEST_CASE("clean_segments_chain_with_duplicates", "[clean][clean_segments]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    // Line segments forming a chain with some duplicates
    tf::segments_buffer<index_t, real_t, 3> input;

    for (int i = 0; i <= 10; ++i) {
        input.points_buffer().emplace_back(real_t(i), real_t(0), real_t(0));
    }

    // Forward chain
    for (int i = 0; i < 10; ++i) {
        input.edges_buffer().emplace_back(index_t(i), index_t(i + 1));
    }
    // Add some duplicate edges
    input.edges_buffer().emplace_back(0, 1);  // duplicate
    input.edges_buffer().emplace_back(5, 6);  // duplicate
    input.edges_buffer().emplace_back(6, 5);  // reversed duplicate

    auto result = tf::cleaned(input.segments());

    REQUIRE(result.edges().size() == 10);
    REQUIRE(result.points().size() == 11);
}

// =============================================================================
// clean_segments_many_degenerate
// =============================================================================

TEMPLATE_TEST_CASE("clean_segments_many_degenerate", "[clean][clean_segments]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    tf::segments_buffer<index_t, real_t, 3> input;

    // Points
    for (int i = 0; i < 5; ++i) {
        input.points_buffer().emplace_back(real_t(i), real_t(0), real_t(0));
    }

    // Valid edges
    input.edges_buffer().emplace_back(0, 1);
    input.edges_buffer().emplace_back(2, 3);

    // Many degenerate edges
    for (int i = 0; i < 5; ++i) {
        input.edges_buffer().emplace_back(index_t(i), index_t(i));
    }

    auto result = tf::cleaned(input.segments());

    REQUIRE(result.edges().size() == 2);
}

// =============================================================================
// clean_segments_tolerance_point_merge_chain
// =============================================================================

TEMPLATE_TEST_CASE("clean_segments_tolerance_point_merge_chain", "[clean][clean_segments]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    // Points that form a chain, each within tolerance of the next
    tf::segments_buffer<index_t, real_t, 3> input;

    input.points_buffer().emplace_back(real_t(0), real_t(0), real_t(0));
    input.points_buffer().emplace_back(real_t(0.005), real_t(0), real_t(0));
    input.points_buffer().emplace_back(real_t(1), real_t(0), real_t(0));
    input.points_buffer().emplace_back(real_t(1.005), real_t(0), real_t(0));

    // Edges connecting them
    input.edges_buffer().emplace_back(0, 2);  // valid edge
    input.edges_buffer().emplace_back(1, 3);  // after merge, same as 0-2

    const real_t tolerance = real_t(0.01);
    auto result = tf::cleaned(input.segments(), tolerance);

    // Points 0,1 merge and 2,3 merge -> 2 points, edges become duplicates -> 1 edge
    REQUIRE(result.points().size() == 2);
    REQUIRE(result.edges().size() == 1);
}

// =============================================================================
// clean_segments_star_pattern
// =============================================================================

TEMPLATE_TEST_CASE("clean_segments_star_pattern", "[clean][clean_segments]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    // Star pattern: center point connected to multiple outer points
    tf::segments_buffer<index_t, real_t, 3> input;

    // Center point
    input.points_buffer().emplace_back(real_t(0), real_t(0), real_t(0));

    // Outer points in a circle
    const int n_spokes = 8;
    for (int i = 0; i < n_spokes; ++i) {
        real_t angle = real_t(i) * real_t(2 * 3.14159265359 / n_spokes);
        input.points_buffer().emplace_back(std::cos(angle), std::sin(angle), real_t(0));
    }

    // Edges from center to each outer point
    for (int i = 0; i < n_spokes; ++i) {
        input.edges_buffer().emplace_back(0, index_t(i + 1));
    }

    // Add duplicate edges (some reversed)
    input.edges_buffer().emplace_back(0, 1);  // duplicate
    input.edges_buffer().emplace_back(3, 0);  // reversed duplicate

    auto result = tf::cleaned(input.segments());

    REQUIRE(result.edges().size() == n_spokes);
    REQUIRE(result.points().size() == n_spokes + 1);
}

// =============================================================================
// clean_segments_index_map_edge_tracking
// =============================================================================

TEMPLATE_TEST_CASE("clean_segments_index_map_edge_tracking", "[clean][clean_segments]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    tf::segments_buffer<index_t, real_t, 3> input;

    input.points_buffer().emplace_back(real_t(0), real_t(0), real_t(0));
    input.points_buffer().emplace_back(real_t(1), real_t(0), real_t(0));
    input.points_buffer().emplace_back(real_t(2), real_t(0), real_t(0));

    input.edges_buffer().emplace_back(0, 1);  // unique
    input.edges_buffer().emplace_back(1, 2);  // unique
    input.edges_buffer().emplace_back(0, 1);  // duplicate of edge 0
    input.edges_buffer().emplace_back(1, 0);  // reversed duplicate of edge 0

    auto [result, edge_im, point_im] = tf::cleaned(input.segments(), tf::return_index_map);

    REQUIRE(result.edges().size() == 2);
    REQUIRE(edge_im.f().size() == 4);

    // All duplicates should map to same edge
    REQUIRE(edge_im.f()[0] == edge_im.f()[2]);
    REQUIRE(edge_im.f()[0] == edge_im.f()[3]);

    // Verify all mapped indices are valid
    for (std::size_t i = 0; i < edge_im.f().size(); ++i) {
        REQUIRE(static_cast<std::size_t>(edge_im.f()[i]) < result.edges().size());
    }
}

// =============================================================================
// clean_segments_disconnected_components
// =============================================================================

TEMPLATE_TEST_CASE("clean_segments_disconnected_components", "[clean][clean_segments]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    // Multiple disconnected line segments
    tf::segments_buffer<index_t, real_t, 3> input;

    // Component 1
    input.points_buffer().emplace_back(real_t(0), real_t(0), real_t(0));
    input.points_buffer().emplace_back(real_t(1), real_t(0), real_t(0));

    // Component 2
    input.points_buffer().emplace_back(real_t(10), real_t(0), real_t(0));
    input.points_buffer().emplace_back(real_t(11), real_t(0), real_t(0));

    // Component 3
    input.points_buffer().emplace_back(real_t(20), real_t(0), real_t(0));
    input.points_buffer().emplace_back(real_t(21), real_t(0), real_t(0));

    input.edges_buffer().emplace_back(0, 1);
    input.edges_buffer().emplace_back(2, 3);
    input.edges_buffer().emplace_back(4, 5);

    auto result = tf::cleaned(input.segments());

    REQUIRE(result.edges().size() == 3);
    REQUIRE(result.points().size() == 6);
}
