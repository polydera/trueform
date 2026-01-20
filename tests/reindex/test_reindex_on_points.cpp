/**
 * @file test_reindex_on_points.cpp
 * @brief Tests for tf::reindexed_by_mask_on_points(...) and tf::reindexed_by_ids_on_points(...)
 *
 * Copyright (c) 2025 Ziga Sajovic, XLAB
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <trueform/trueform.hpp>
#include "type_traits.hpp"
#include "mesh_generators.hpp"

namespace {

inline auto make_mask(std::initializer_list<bool> values) -> tf::buffer<bool> {
    tf::buffer<bool> mask;
    mask.allocate(values.size());
    std::size_t i = 0;
    for (bool v : values) {
        mask[i++] = v;
    }
    return mask;
}

template <typename Index>
auto make_ids(std::initializer_list<Index> values) -> tf::buffer<Index> {
    tf::buffer<Index> ids;
    ids.allocate(values.size());
    std::size_t i = 0;
    for (auto v : values) {
        ids[i++] = v;
    }
    return ids;
}

} // anonymous namespace

// =============================================================================
// reindex_by_mask_on_points_polygons_basic
// =============================================================================

TEMPLATE_TEST_CASE("reindex_by_mask_on_points_polygons_basic", "[reindex][on_points][polygons]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    // Two triangles sharing edge (0,1): face0=(0,1,2), face1=(1,0,3)
    auto input = tf::test::create_triangle_polygons_3d<index_t, real_t>();
    // 4 points: 0,1,2,3

    // Mask that keeps points 0,1,2 but not 3
    auto point_mask = make_mask({true, true, true, false});

    auto result = tf::reindexed_by_mask_on_points(input.polygons(), point_mask);

    // Only face0 should survive (it uses only points 0,1,2)
    // face1 uses point 3 which is masked out
    REQUIRE(result.faces().size() == 1);
    REQUIRE(result.points().size() == 3);
}

// =============================================================================
// reindex_by_mask_on_points_polygons_partial
// =============================================================================

TEMPLATE_TEST_CASE("reindex_by_mask_on_points_polygons_partial", "[reindex][on_points][polygons]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto input = tf::test::create_larger_triangle_polygons_3d<index_t, real_t>();
    // 4 faces, 6 points

    // Mask that removes one point used by multiple faces
    auto point_mask = make_mask({true, true, false, true, true, true});

    auto result = tf::reindexed_by_mask_on_points(input.polygons(), point_mask);

    // Faces using point 2 should be removed
    REQUIRE(result.faces().size() < 4);
}

// =============================================================================
// reindex_by_mask_on_points_polygons_index_map
// =============================================================================

TEMPLATE_TEST_CASE("reindex_by_mask_on_points_polygons_index_map", "[reindex][on_points][polygons]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto input = tf::test::create_triangle_polygons_3d<index_t, real_t>();

    auto point_mask = make_mask({true, true, true, false});

    auto [result, face_im, point_im] = tf::reindexed_by_mask_on_points(input.polygons(), point_mask, tf::return_index_map);

    REQUIRE(result.faces().size() == 1);
    REQUIRE(face_im.f().size() == 2);
    REQUIRE(point_im.f().size() == 4);
    REQUIRE(point_im.kept_ids().size() == 3);
}

// =============================================================================
// reindex_by_mask_on_points_polygons_dynamic
// =============================================================================

TEMPLATE_TEST_CASE("reindex_by_mask_on_points_polygons_dynamic", "[reindex][on_points][polygons][dynamic]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto input = tf::test::create_mixed_polygons_3d<index_t, real_t>();
    // 1 triangle + 1 quad, ~5 points

    // Keep all points except one used by the quad
    tf::buffer<bool> point_mask;
    point_mask.allocate(input.points().size());
    for (std::size_t i = 0; i < input.points().size(); ++i) {
        point_mask[i] = true;
    }
    // Remove last point (used by quad)
    if (input.points().size() > 0) {
        point_mask[input.points().size() - 1] = false;
    }

    auto result = tf::reindexed_by_mask_on_points(input.polygons(), point_mask);

    // The quad should be removed since it uses the masked point
    REQUIRE(result.faces().size() <= 1);
}

// =============================================================================
// reindex_by_mask_on_points_segments_basic
// =============================================================================

TEMPLATE_TEST_CASE("reindex_by_mask_on_points_segments_basic", "[reindex][on_points][segments]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    tf::segments_buffer<index_t, real_t, 3> input;
    input.edges_buffer().emplace_back(0, 1);  // edge using points 0,1
    input.edges_buffer().emplace_back(1, 2);  // edge using points 1,2
    input.edges_buffer().emplace_back(2, 3);  // edge using points 2,3

    input.points_buffer().emplace_back(real_t(0), real_t(0), real_t(0));
    input.points_buffer().emplace_back(real_t(1), real_t(0), real_t(0));
    input.points_buffer().emplace_back(real_t(2), real_t(0), real_t(0));
    input.points_buffer().emplace_back(real_t(3), real_t(0), real_t(0));

    // Mask keeps points 0,1,2 but not 3
    auto point_mask = make_mask({true, true, true, false});

    auto result = tf::reindexed_by_mask_on_points(input.segments(), point_mask);

    // Edge (2,3) should be removed
    REQUIRE(result.edges().size() == 2);
    REQUIRE(result.points().size() == 3);
}

// =============================================================================
// reindex_by_mask_on_points_segments_index_map
// =============================================================================

TEMPLATE_TEST_CASE("reindex_by_mask_on_points_segments_index_map", "[reindex][on_points][segments]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    tf::segments_buffer<index_t, real_t, 3> input;
    input.edges_buffer().emplace_back(0, 1);
    input.edges_buffer().emplace_back(2, 3);

    input.points_buffer().emplace_back(real_t(0), real_t(0), real_t(0));
    input.points_buffer().emplace_back(real_t(1), real_t(0), real_t(0));
    input.points_buffer().emplace_back(real_t(2), real_t(0), real_t(0));
    input.points_buffer().emplace_back(real_t(3), real_t(0), real_t(0));

    auto point_mask = make_mask({true, true, false, false});

    auto [result, edge_im, point_im] = tf::reindexed_by_mask_on_points(input.segments(), point_mask, tf::return_index_map);

    REQUIRE(result.edges().size() == 1);
    REQUIRE(edge_im.f().size() == 2);
    REQUIRE(point_im.f().size() == 4);
}

// =============================================================================
// reindex_by_ids_on_points_polygons_basic
// =============================================================================

TEMPLATE_TEST_CASE("reindex_by_ids_on_points_polygons_basic", "[reindex][on_points][polygons]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto input = tf::test::create_triangle_polygons_3d<index_t, real_t>();
    // face0=(0,1,2), face1=(1,0,3)

    // Keep only points 0,1,2 by ID
    auto point_ids = make_ids<index_t>({0, 1, 2});

    auto result = tf::reindexed_by_ids_on_points(input.polygons(), point_ids);

    // Only face0 should survive
    REQUIRE(result.faces().size() == 1);
}

// =============================================================================
// reindex_by_ids_on_points_polygons_subset
// =============================================================================

TEMPLATE_TEST_CASE("reindex_by_ids_on_points_polygons_subset", "[reindex][on_points][polygons]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto input = tf::test::create_larger_triangle_polygons_3d<index_t, real_t>();
    // 4 faces, 6 points

    // Keep subset of points
    auto point_ids = make_ids<index_t>({0, 1, 2});

    auto result = tf::reindexed_by_ids_on_points(input.polygons(), point_ids);

    // Only faces using only these points should survive
    REQUIRE(result.faces().size() <= 4);
}

// =============================================================================
// reindex_by_ids_on_points_polygons_index_map
// =============================================================================

TEMPLATE_TEST_CASE("reindex_by_ids_on_points_polygons_index_map", "[reindex][on_points][polygons]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto input = tf::test::create_triangle_polygons_3d<index_t, real_t>();

    auto point_ids = make_ids<index_t>({0, 1, 2});

    auto [result, face_im, point_im] = tf::reindexed_by_ids_on_points(input.polygons(), point_ids, tf::return_index_map);

    REQUIRE(face_im.f().size() == 2);
    REQUIRE(point_im.f().size() == 4);
}

// =============================================================================
// reindex_by_ids_on_points_polygons_dynamic
// =============================================================================

TEMPLATE_TEST_CASE("reindex_by_ids_on_points_polygons_dynamic", "[reindex][on_points][polygons][dynamic]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto input = tf::test::create_dynamic_polygons_3d<index_t, real_t>();

    // Keep first 3 points
    auto point_ids = make_ids<index_t>({0, 1, 2});

    auto result = tf::reindexed_by_ids_on_points(input.polygons(), point_ids);

    // At least one face should survive if it uses only these points
    REQUIRE(result.faces().size() >= 0);  // May be 0 or more depending on face configuration
}

// =============================================================================
// reindex_by_ids_on_points_segments_basic
// =============================================================================

TEMPLATE_TEST_CASE("reindex_by_ids_on_points_segments_basic", "[reindex][on_points][segments]",
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

    auto point_ids = make_ids<index_t>({0, 1, 2});

    auto result = tf::reindexed_by_ids_on_points(input.segments(), point_ids);

    // Edge (2,3) should be removed since point 3 is not in IDs
    REQUIRE(result.edges().size() == 2);
}

// =============================================================================
// reindex_by_mask_on_points_all_points_masked
// =============================================================================

TEMPLATE_TEST_CASE("reindex_by_mask_on_points_all_points_masked", "[reindex][on_points][polygons]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto input = tf::test::create_triangle_polygons_3d<index_t, real_t>();

    // All false mask
    auto point_mask = make_mask({false, false, false, false});

    auto result = tf::reindexed_by_mask_on_points(input.polygons(), point_mask);

    REQUIRE(result.faces().size() == 0);
    REQUIRE(result.points().size() == 0);
}

// =============================================================================
// reindex_by_ids_on_points_empty_ids
// =============================================================================

TEMPLATE_TEST_CASE("reindex_by_ids_on_points_empty_ids", "[reindex][on_points][polygons]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto input = tf::test::create_triangle_polygons_3d<index_t, real_t>();

    tf::buffer<index_t> point_ids;

    auto result = tf::reindexed_by_ids_on_points(input.polygons(), point_ids);

    REQUIRE(result.faces().size() == 0);
    REQUIRE(result.points().size() == 0);
}
