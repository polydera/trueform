/**
 * @file test_reindex_by_mask.cpp
 * @brief Tests for tf::reindexed_by_mask(...)
 *
 * Tests for:
 * - reindex_by_mask_points_basic
 * - reindex_by_mask_points_all_false
 * - reindex_by_mask_points_all_true
 * - reindex_by_mask_points_index_map
 * - reindex_by_mask_points_single
 * - reindex_by_mask_vectors_basic
 * - reindex_by_mask_unit_vectors_basic
 * - reindex_by_mask_segments_basic
 * - reindex_by_mask_segments_index_map
 * - reindex_by_mask_segments_point_compaction
 * - reindex_by_mask_polygons_basic
 * - reindex_by_mask_polygons_index_map
 * - reindex_by_mask_polygons_point_compaction
 * - reindex_by_mask_polygons_all_removed
 * - reindex_by_mask_polygons_dynamic_basic
 * - reindex_by_mask_polygons_dynamic_mixed
 * - reindex_by_mask_polygons_dynamic_index_map
 * - reindex_by_mask_polygons_dynamic_point_compaction
 * - reindex_by_mask_range_basic
 *
 * Copyright (c) 2025 Ziga Sajovic, XLAB
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <trueform/trueform.hpp>
#include "type_traits.hpp"
#include "mesh_generators.hpp"

namespace {

// Helper to create a bool mask buffer
inline auto make_mask(std::initializer_list<bool> values) -> tf::buffer<bool> {
    tf::buffer<bool> mask;
    mask.allocate(values.size());
    std::size_t i = 0;
    for (bool v : values) {
        mask[i++] = v;
    }
    return mask;
}

} // anonymous namespace

// =============================================================================
// reindex_by_mask_points_basic
// =============================================================================

TEMPLATE_TEST_CASE("reindex_by_mask_points_basic", "[reindex][by_mask][points]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    // Create points
    tf::points_buffer<real_t, 3> input;
    input.emplace_back(real_t(0), real_t(0), real_t(0));
    input.emplace_back(real_t(1), real_t(0), real_t(0));
    input.emplace_back(real_t(2), real_t(0), real_t(0));
    input.emplace_back(real_t(3), real_t(0), real_t(0));

    // Mask to keep every other point
    auto mask = make_mask({true, false, true, false});

    auto result = tf::reindexed_by_mask<index_t>(input.points(), mask);

    // Should have 2 points
    REQUIRE(result.size() == 2);

    // Verify correct points kept (0 and 2)
    REQUIRE(std::abs(result[0][0] - real_t(0)) < real_t(1e-5));
    REQUIRE(std::abs(result[1][0] - real_t(2)) < real_t(1e-5));
}

// =============================================================================
// reindex_by_mask_points_all_false
// =============================================================================

TEMPLATE_TEST_CASE("reindex_by_mask_points_all_false", "[reindex][by_mask][points]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    tf::points_buffer<real_t, 3> input;
    input.emplace_back(real_t(0), real_t(0), real_t(0));
    input.emplace_back(real_t(1), real_t(0), real_t(0));
    input.emplace_back(real_t(2), real_t(0), real_t(0));

    // All false mask
    auto mask = make_mask({false, false, false});

    auto result = tf::reindexed_by_mask<index_t>(input.points(), mask);

    // Should be empty
    REQUIRE(result.size() == 0);
}

// =============================================================================
// reindex_by_mask_points_all_true
// =============================================================================

TEMPLATE_TEST_CASE("reindex_by_mask_points_all_true", "[reindex][by_mask][points]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    tf::points_buffer<real_t, 3> input;
    input.emplace_back(real_t(0), real_t(0), real_t(0));
    input.emplace_back(real_t(1), real_t(0), real_t(0));
    input.emplace_back(real_t(2), real_t(0), real_t(0));

    // All true mask
    auto mask = make_mask({true, true, true});

    auto result = tf::reindexed_by_mask<index_t>(input.points(), mask);

    // Should have all points
    REQUIRE(result.size() == 3);

    // Verify values preserved
    for (std::size_t i = 0; i < 3; ++i) {
        REQUIRE(std::abs(result[i][0] - real_t(i)) < real_t(1e-5));
    }
}

// =============================================================================
// reindex_by_mask_points_index_map
// =============================================================================

TEMPLATE_TEST_CASE("reindex_by_mask_points_index_map", "[reindex][by_mask][points]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    tf::points_buffer<real_t, 3> input;
    input.emplace_back(real_t(0), real_t(0), real_t(0));
    input.emplace_back(real_t(1), real_t(0), real_t(0));
    input.emplace_back(real_t(2), real_t(0), real_t(0));
    input.emplace_back(real_t(3), real_t(0), real_t(0));

    auto mask = make_mask({true, false, true, false});

    auto [result, index_map] = tf::reindexed_by_mask<index_t>(input.points(), mask, tf::return_index_map);

    REQUIRE(result.size() == 2);

    // Index map should have 4 entries
    REQUIRE(index_map.f().size() == 4);

    // Kept IDs should be 0 and 2
    REQUIRE(index_map.kept_ids().size() == 2);
    REQUIRE(index_map.kept_ids()[0] == 0);
    REQUIRE(index_map.kept_ids()[1] == 2);

    // Mapping verification: kept indices should map to valid outputs
    REQUIRE(index_map.f()[0] == 0);  // input 0 -> output 0
    REQUIRE(index_map.f()[2] == 1);  // input 2 -> output 1

    // Removed indices should map to sentinel
    auto sentinel = static_cast<index_t>(index_map.f().size());
    REQUIRE(index_map.f()[1] == sentinel);
    REQUIRE(index_map.f()[3] == sentinel);
}

// =============================================================================
// reindex_by_mask_points_single
// =============================================================================

TEMPLATE_TEST_CASE("reindex_by_mask_points_single", "[reindex][by_mask][points]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    tf::points_buffer<real_t, 3> input;
    input.emplace_back(real_t(5), real_t(6), real_t(7));

    auto mask = make_mask({true});

    auto result = tf::reindexed_by_mask<index_t>(input.points(), mask);

    REQUIRE(result.size() == 1);
    REQUIRE(std::abs(result[0][0] - real_t(5)) < real_t(1e-5));
    REQUIRE(std::abs(result[0][1] - real_t(6)) < real_t(1e-5));
    REQUIRE(std::abs(result[0][2] - real_t(7)) < real_t(1e-5));
}

// =============================================================================
// reindex_by_mask_vectors_basic
// =============================================================================

TEMPLATE_TEST_CASE("reindex_by_mask_vectors_basic", "[reindex][by_mask][vectors]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    tf::vectors_buffer<real_t, 3> input;
    input.emplace_back(real_t(1), real_t(0), real_t(0));
    input.emplace_back(real_t(0), real_t(1), real_t(0));
    input.emplace_back(real_t(0), real_t(0), real_t(1));
    input.emplace_back(real_t(1), real_t(1), real_t(1));

    auto mask = make_mask({true, false, true, false});

    auto result = tf::reindexed_by_mask<index_t>(input.vectors(), mask);

    REQUIRE(result.size() == 2);

    // Verify first kept vector (1, 0, 0)
    REQUIRE(std::abs(result[0][0] - real_t(1)) < real_t(1e-5));
    REQUIRE(std::abs(result[0][1] - real_t(0)) < real_t(1e-5));
    REQUIRE(std::abs(result[0][2] - real_t(0)) < real_t(1e-5));

    // Verify second kept vector (0, 0, 1)
    REQUIRE(std::abs(result[1][0] - real_t(0)) < real_t(1e-5));
    REQUIRE(std::abs(result[1][1] - real_t(0)) < real_t(1e-5));
    REQUIRE(std::abs(result[1][2] - real_t(1)) < real_t(1e-5));
}

// =============================================================================
// reindex_by_mask_unit_vectors_basic
// =============================================================================

TEMPLATE_TEST_CASE("reindex_by_mask_unit_vectors_basic", "[reindex][by_mask][unit_vectors]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    tf::unit_vectors_buffer<real_t, 3> input;
    input.emplace_back(real_t(1), real_t(0), real_t(0));
    input.emplace_back(real_t(0), real_t(1), real_t(0));
    input.emplace_back(real_t(0), real_t(0), real_t(1));

    auto mask = make_mask({false, true, true});

    auto result = tf::reindexed_by_mask<index_t>(input.unit_vectors(), mask);

    REQUIRE(result.size() == 2);

    // Verify (0, 1, 0) and (0, 0, 1) are kept
    REQUIRE(std::abs(result[0][1] - real_t(1)) < real_t(1e-5));
    REQUIRE(std::abs(result[1][2] - real_t(1)) < real_t(1e-5));
}

// =============================================================================
// reindex_by_mask_segments_basic
// =============================================================================

TEMPLATE_TEST_CASE("reindex_by_mask_segments_basic", "[reindex][by_mask][segments]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    // Create segments: 4 points, 3 edges
    tf::segments_buffer<index_t, real_t, 3> input;
    input.edges_buffer().emplace_back(0, 1);
    input.edges_buffer().emplace_back(1, 2);
    input.edges_buffer().emplace_back(2, 3);

    input.points_buffer().emplace_back(real_t(0), real_t(0), real_t(0));
    input.points_buffer().emplace_back(real_t(1), real_t(0), real_t(0));
    input.points_buffer().emplace_back(real_t(2), real_t(0), real_t(0));
    input.points_buffer().emplace_back(real_t(3), real_t(0), real_t(0));

    // Keep first and last edge
    auto mask = make_mask({true, false, true});

    auto result = tf::reindexed_by_mask(input.segments(), mask);

    // Should have 2 edges
    REQUIRE(result.edges().size() == 2);
}

// =============================================================================
// reindex_by_mask_segments_index_map
// =============================================================================

TEMPLATE_TEST_CASE("reindex_by_mask_segments_index_map", "[reindex][by_mask][segments]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    tf::segments_buffer<index_t, real_t, 3> input;
    input.edges_buffer().emplace_back(0, 1);
    input.edges_buffer().emplace_back(1, 2);
    input.edges_buffer().emplace_back(2, 3);

    input.points_buffer().emplace_back(real_t(0), real_t(0), real_t(0));
    input.points_buffer().emplace_back(real_t(1), real_t(0), real_t(0));
    input.points_buffer().emplace_back(real_t(2), real_t(0), real_t(0));
    input.points_buffer().emplace_back(real_t(3), real_t(0), real_t(0));

    auto mask = make_mask({true, false, true});

    auto [result, edge_im, point_im] = tf::reindexed_by_mask(input.segments(), mask, tf::return_index_map);

    // 2 edges kept
    REQUIRE(result.edges().size() == 2);

    // Edge index map should have 3 entries
    REQUIRE(edge_im.f().size() == 3);
    REQUIRE(edge_im.kept_ids().size() == 2);

    // Point index map should exist (points compacted)
    REQUIRE(point_im.f().size() == 4);
}

// =============================================================================
// reindex_by_mask_segments_point_compaction
// =============================================================================

TEMPLATE_TEST_CASE("reindex_by_mask_segments_point_compaction", "[reindex][by_mask][segments]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    // Create segments where removing an edge leaves a point unreferenced
    tf::segments_buffer<index_t, real_t, 3> input;
    input.edges_buffer().emplace_back(0, 1);  // edge 0
    input.edges_buffer().emplace_back(2, 3);  // edge 1 (uses different points)

    input.points_buffer().emplace_back(real_t(0), real_t(0), real_t(0));
    input.points_buffer().emplace_back(real_t(1), real_t(0), real_t(0));
    input.points_buffer().emplace_back(real_t(2), real_t(0), real_t(0));
    input.points_buffer().emplace_back(real_t(3), real_t(0), real_t(0));

    // Keep only edge 0 (uses points 0 and 1)
    auto mask = make_mask({true, false});

    auto result = tf::reindexed_by_mask(input.segments(), mask);

    // Should have 1 edge and only 2 points (points 2,3 should be removed)
    REQUIRE(result.edges().size() == 1);
    REQUIRE(result.points().size() == 2);

    // Edge indices should be remapped to new point indices
    REQUIRE(result.edges()[0][0] == 0);
    REQUIRE(result.edges()[0][1] == 1);
}

// =============================================================================
// reindex_by_mask_polygons_basic
// =============================================================================

TEMPLATE_TEST_CASE("reindex_by_mask_polygons_basic", "[reindex][by_mask][polygons]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto input = tf::test::create_larger_triangle_polygons_3d<index_t, real_t>();
    // 4 faces, 6 points

    // Keep faces 0 and 2
    auto mask = make_mask({true, false, true, false});

    auto result = tf::reindexed_by_mask(input.polygons(), mask);

    REQUIRE(result.faces().size() == 2);
}

// =============================================================================
// reindex_by_mask_polygons_index_map
// =============================================================================

TEMPLATE_TEST_CASE("reindex_by_mask_polygons_index_map", "[reindex][by_mask][polygons]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto input = tf::test::create_triangle_polygons_3d<index_t, real_t>();
    // 2 faces, 4 points

    auto mask = make_mask({true, false});

    auto [result, face_im, point_im] = tf::reindexed_by_mask(input.polygons(), mask, tf::return_index_map);

    REQUIRE(result.faces().size() == 1);

    // Face index map
    REQUIRE(face_im.f().size() == 2);
    REQUIRE(face_im.kept_ids().size() == 1);
    REQUIRE(face_im.kept_ids()[0] == 0);

    // Point index map should exist
    REQUIRE(point_im.f().size() == 4);
}

// =============================================================================
// reindex_by_mask_polygons_point_compaction
// =============================================================================

TEMPLATE_TEST_CASE("reindex_by_mask_polygons_point_compaction", "[reindex][by_mask][polygons]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    // Two separate triangles with no shared points
    tf::polygons_buffer<index_t, real_t, 3, 3> input;
    input.faces_buffer().emplace_back(0, 1, 2);  // first triangle
    input.faces_buffer().emplace_back(3, 4, 5);  // second triangle (separate points)

    input.points_buffer().emplace_back(real_t(0), real_t(0), real_t(0));
    input.points_buffer().emplace_back(real_t(1), real_t(0), real_t(0));
    input.points_buffer().emplace_back(real_t(0.5), real_t(1), real_t(0));
    input.points_buffer().emplace_back(real_t(10), real_t(0), real_t(0));
    input.points_buffer().emplace_back(real_t(11), real_t(0), real_t(0));
    input.points_buffer().emplace_back(real_t(10.5), real_t(1), real_t(0));

    // Keep only first triangle
    auto mask = make_mask({true, false});

    auto result = tf::reindexed_by_mask(input.polygons(), mask);

    // Should have 1 face and 3 points (points 3,4,5 removed)
    REQUIRE(result.faces().size() == 1);
    REQUIRE(result.points().size() == 3);

    // Face indices should reference valid points
    for (auto idx : result.faces()[0]) {
        REQUIRE(static_cast<std::size_t>(idx) < result.points().size());
    }
}

// =============================================================================
// reindex_by_mask_polygons_all_removed
// =============================================================================

TEMPLATE_TEST_CASE("reindex_by_mask_polygons_all_removed", "[reindex][by_mask][polygons]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto input = tf::test::create_triangle_polygons_3d<index_t, real_t>();

    // All false mask
    auto mask = make_mask({false, false});

    auto result = tf::reindexed_by_mask(input.polygons(), mask);

    REQUIRE(result.faces().size() == 0);
    REQUIRE(result.points().size() == 0);
}

// =============================================================================
// reindex_by_mask_polygons_dynamic_basic
// =============================================================================

TEMPLATE_TEST_CASE("reindex_by_mask_polygons_dynamic_basic", "[reindex][by_mask][polygons][dynamic]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto input = tf::test::create_dynamic_polygons_3d<index_t, real_t>();
    // 2 triangles, 4 points

    auto mask = make_mask({true, false});

    auto result = tf::reindexed_by_mask(input.polygons(), mask);

    REQUIRE(result.faces().size() == 1);
}

// =============================================================================
// reindex_by_mask_polygons_dynamic_mixed
// =============================================================================

TEMPLATE_TEST_CASE("reindex_by_mask_polygons_dynamic_mixed", "[reindex][by_mask][polygons][dynamic]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto input = tf::test::create_mixed_polygons_3d<index_t, real_t>();
    // 1 triangle + 1 quad = 2 faces, 5 points

    // Keep only the quad
    auto mask = make_mask({false, true});

    auto result = tf::reindexed_by_mask(input.polygons(), mask);

    REQUIRE(result.faces().size() == 1);
    // The quad has 4 vertices
    REQUIRE(result.faces()[0].size() == 4);
}

// =============================================================================
// reindex_by_mask_polygons_dynamic_index_map
// =============================================================================

TEMPLATE_TEST_CASE("reindex_by_mask_polygons_dynamic_index_map", "[reindex][by_mask][polygons][dynamic]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto input = tf::test::create_mixed_polygons_3d<index_t, real_t>();

    auto mask = make_mask({true, false});

    auto [result, face_im, point_im] = tf::reindexed_by_mask(input.polygons(), mask, tf::return_index_map);

    REQUIRE(result.faces().size() == 1);
    REQUIRE(face_im.f().size() == 2);
    REQUIRE(face_im.kept_ids().size() == 1);
    REQUIRE(face_im.kept_ids()[0] == 0);
}

// =============================================================================
// reindex_by_mask_polygons_dynamic_point_compaction
// =============================================================================

TEMPLATE_TEST_CASE("reindex_by_mask_polygons_dynamic_point_compaction", "[reindex][by_mask][polygons][dynamic]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    // Create dynamic polygons with separate point sets
    tf::polygons_buffer<index_t, real_t, 3, tf::dynamic_size> input;
    input.faces_buffer().push_back({0, 1, 2});       // triangle using points 0,1,2
    input.faces_buffer().push_back({3, 4, 5, 6});   // quad using points 3,4,5,6

    input.points_buffer().emplace_back(real_t(0), real_t(0), real_t(0));
    input.points_buffer().emplace_back(real_t(1), real_t(0), real_t(0));
    input.points_buffer().emplace_back(real_t(0.5), real_t(1), real_t(0));
    input.points_buffer().emplace_back(real_t(10), real_t(0), real_t(0));
    input.points_buffer().emplace_back(real_t(11), real_t(0), real_t(0));
    input.points_buffer().emplace_back(real_t(11), real_t(1), real_t(0));
    input.points_buffer().emplace_back(real_t(10), real_t(1), real_t(0));

    // Keep only the triangle
    auto mask = make_mask({true, false});

    auto result = tf::reindexed_by_mask(input.polygons(), mask);

    REQUIRE(result.faces().size() == 1);
    REQUIRE(result.points().size() == 3);  // Only points 0,1,2 should remain
}

// =============================================================================
// reindex_by_mask_range_basic
// =============================================================================

TEMPLATE_TEST_CASE("reindex_by_mask_range_basic", "[reindex][by_mask][range]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;

    // Create a simple range of integers
    tf::buffer<int> input;
    input.push_back(10);
    input.push_back(20);
    input.push_back(30);
    input.push_back(40);
    input.push_back(50);
    auto input_range = tf::make_range(input);

    auto mask = make_mask({true, false, true, false, true});

    auto result = tf::reindexed_by_mask<index_t>(input_range, mask);

    REQUIRE(result.size() == 3);
    REQUIRE(result[0] == 10);
    REQUIRE(result[1] == 30);
    REQUIRE(result[2] == 50);
}

// =============================================================================
// reindex_by_mask_points_empty
// =============================================================================

TEMPLATE_TEST_CASE("reindex_by_mask_points_empty", "[reindex][by_mask][points]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    tf::points_buffer<real_t, 3> input;
    tf::buffer<bool> mask;

    auto result = tf::reindexed_by_mask<index_t>(input.points(), mask);

    REQUIRE(result.size() == 0);
}

// =============================================================================
// reindex_by_mask_segments_empty
// =============================================================================

TEMPLATE_TEST_CASE("reindex_by_mask_segments_empty", "[reindex][by_mask][segments]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    tf::segments_buffer<index_t, real_t, 3> input;
    tf::buffer<bool> mask;

    auto result = tf::reindexed_by_mask(input.segments(), mask);

    REQUIRE(result.edges().size() == 0);
    REQUIRE(result.points().size() == 0);
}

// =============================================================================
// reindex_by_mask_polygons_single_face
// =============================================================================

TEMPLATE_TEST_CASE("reindex_by_mask_polygons_single_face", "[reindex][by_mask][polygons]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    tf::polygons_buffer<index_t, real_t, 3, 3> input;
    input.faces_buffer().emplace_back(0, 1, 2);
    input.points_buffer().emplace_back(real_t(0), real_t(0), real_t(0));
    input.points_buffer().emplace_back(real_t(1), real_t(0), real_t(0));
    input.points_buffer().emplace_back(real_t(0.5), real_t(1), real_t(0));

    auto mask = make_mask({true});

    auto result = tf::reindexed_by_mask(input.polygons(), mask);

    REQUIRE(result.faces().size() == 1);
    REQUIRE(result.points().size() == 3);
}

// =============================================================================
// reindex_by_mask_polygons_2d
// =============================================================================

TEMPLATE_TEST_CASE("reindex_by_mask_polygons_2d", "[reindex][by_mask][polygons]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto input = tf::test::create_triangle_polygons_2d<index_t, real_t>();

    auto mask = make_mask({true, false});

    auto result = tf::reindexed_by_mask(input.polygons(), mask);

    REQUIRE(result.faces().size() == 1);
}

// =============================================================================
// reindex_by_mask_cube_selective
// =============================================================================

TEMPLATE_TEST_CASE("reindex_by_mask_cube_selective", "[reindex][by_mask][polygons]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto input = tf::test::create_cube_polygons<index_t, real_t>();
    // 12 faces, 8 points

    // Keep every other face
    tf::buffer<bool> mask;
    mask.allocate(input.faces().size());
    for (std::size_t i = 0; i < input.faces().size(); ++i) {
        mask[i] = (i % 2 == 0);
    }

    auto result = tf::reindexed_by_mask(input.polygons(), mask);

    REQUIRE(result.faces().size() == 6);
}
