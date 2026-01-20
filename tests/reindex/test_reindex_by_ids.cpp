/**
 * @file test_reindex_by_ids.cpp
 * @brief Tests for tf::reindexed_by_ids(...)
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
// reindex_by_ids_points_basic
// =============================================================================

TEMPLATE_TEST_CASE("reindex_by_ids_points_basic", "[reindex][by_ids][points]",
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

    // Extract points 0 and 2
    auto ids = make_ids<index_t>({0, 2});

    auto result = tf::reindexed_by_ids<index_t>(input.points(), ids);

    REQUIRE(result.size() == 2);
    REQUIRE(std::abs(result[0][0] - real_t(0)) < real_t(1e-5));
    REQUIRE(std::abs(result[1][0] - real_t(2)) < real_t(1e-5));
}

// =============================================================================
// reindex_by_ids_points_reorder
// =============================================================================

TEMPLATE_TEST_CASE("reindex_by_ids_points_reorder", "[reindex][by_ids][points]",
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

    // Reorder: extract in reverse order
    auto ids = make_ids<index_t>({3, 2, 1, 0});

    auto result = tf::reindexed_by_ids<index_t>(input.points(), ids);

    REQUIRE(result.size() == 4);
    REQUIRE(std::abs(result[0][0] - real_t(3)) < real_t(1e-5));
    REQUIRE(std::abs(result[1][0] - real_t(2)) < real_t(1e-5));
    REQUIRE(std::abs(result[2][0] - real_t(1)) < real_t(1e-5));
    REQUIRE(std::abs(result[3][0] - real_t(0)) < real_t(1e-5));
}

// =============================================================================
// reindex_by_ids_points_index_map
// =============================================================================

TEMPLATE_TEST_CASE("reindex_by_ids_points_index_map", "[reindex][by_ids][points]",
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

    auto ids = make_ids<index_t>({1, 3});

    auto [result, index_map] = tf::reindexed_by_ids<index_t>(input.points(), ids, tf::return_index_map);

    REQUIRE(result.size() == 2);
    REQUIRE(index_map.f().size() == 4);
    REQUIRE(index_map.kept_ids().size() == 2);

    // Mapping verification
    REQUIRE(index_map.f()[1] == 0);  // input 1 -> output 0
    REQUIRE(index_map.f()[3] == 1);  // input 3 -> output 1
}

// =============================================================================
// reindex_by_ids_points_empty
// =============================================================================

TEMPLATE_TEST_CASE("reindex_by_ids_points_empty", "[reindex][by_ids][points]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    tf::points_buffer<real_t, 3> input;
    input.emplace_back(real_t(0), real_t(0), real_t(0));
    input.emplace_back(real_t(1), real_t(0), real_t(0));

    tf::buffer<index_t> ids;

    auto result = tf::reindexed_by_ids<index_t>(input.points(), ids);

    REQUIRE(result.size() == 0);
}

// =============================================================================
// reindex_by_ids_points_single
// =============================================================================

TEMPLATE_TEST_CASE("reindex_by_ids_points_single", "[reindex][by_ids][points]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    tf::points_buffer<real_t, 3> input;
    input.emplace_back(real_t(5), real_t(6), real_t(7));
    input.emplace_back(real_t(8), real_t(9), real_t(10));

    auto ids = make_ids<index_t>({1});

    auto result = tf::reindexed_by_ids<index_t>(input.points(), ids);

    REQUIRE(result.size() == 1);
    REQUIRE(std::abs(result[0][0] - real_t(8)) < real_t(1e-5));
    REQUIRE(std::abs(result[0][1] - real_t(9)) < real_t(1e-5));
    REQUIRE(std::abs(result[0][2] - real_t(10)) < real_t(1e-5));
}

// =============================================================================
// reindex_by_ids_vectors_basic
// =============================================================================

TEMPLATE_TEST_CASE("reindex_by_ids_vectors_basic", "[reindex][by_ids][vectors]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    tf::vectors_buffer<real_t, 3> input;
    input.emplace_back(real_t(1), real_t(0), real_t(0));
    input.emplace_back(real_t(0), real_t(1), real_t(0));
    input.emplace_back(real_t(0), real_t(0), real_t(1));

    auto ids = make_ids<index_t>({2, 0});

    auto result = tf::reindexed_by_ids<index_t>(input.vectors(), ids);

    REQUIRE(result.size() == 2);
    REQUIRE(std::abs(result[0][2] - real_t(1)) < real_t(1e-5));  // (0,0,1)
    REQUIRE(std::abs(result[1][0] - real_t(1)) < real_t(1e-5));  // (1,0,0)
}

// =============================================================================
// reindex_by_ids_segments_basic
// =============================================================================

TEMPLATE_TEST_CASE("reindex_by_ids_segments_basic", "[reindex][by_ids][segments]",
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

    auto ids = make_ids<index_t>({0, 2});

    auto result = tf::reindexed_by_ids(input.segments(), ids);

    REQUIRE(result.edges().size() == 2);
}

// =============================================================================
// reindex_by_ids_segments_index_map
// =============================================================================

TEMPLATE_TEST_CASE("reindex_by_ids_segments_index_map", "[reindex][by_ids][segments]",
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

    auto ids = make_ids<index_t>({1});

    auto [result, edge_im, point_im] = tf::reindexed_by_ids(input.segments(), ids, tf::return_index_map);

    REQUIRE(result.edges().size() == 1);
    REQUIRE(edge_im.f().size() == 3);
    REQUIRE(point_im.f().size() == 4);
}

// =============================================================================
// reindex_by_ids_polygons_basic
// =============================================================================

TEMPLATE_TEST_CASE("reindex_by_ids_polygons_basic", "[reindex][by_ids][polygons]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto input = tf::test::create_larger_triangle_polygons_3d<index_t, real_t>();
    // 4 faces

    auto ids = make_ids<index_t>({0, 2});

    auto result = tf::reindexed_by_ids(input.polygons(), ids);

    REQUIRE(result.faces().size() == 2);
}

// =============================================================================
// reindex_by_ids_polygons_index_map
// =============================================================================

TEMPLATE_TEST_CASE("reindex_by_ids_polygons_index_map", "[reindex][by_ids][polygons]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto input = tf::test::create_triangle_polygons_3d<index_t, real_t>();
    // 2 faces, 4 points

    auto ids = make_ids<index_t>({1});

    auto [result, face_im, point_im] = tf::reindexed_by_ids(input.polygons(), ids, tf::return_index_map);

    REQUIRE(result.faces().size() == 1);
    REQUIRE(face_im.f().size() == 2);
    REQUIRE(face_im.kept_ids().size() == 1);
    REQUIRE(face_im.kept_ids()[0] == 1);
}

// =============================================================================
// reindex_by_ids_polygons_point_compaction
// =============================================================================

TEMPLATE_TEST_CASE("reindex_by_ids_polygons_point_compaction", "[reindex][by_ids][polygons]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    // Two separate triangles
    tf::polygons_buffer<index_t, real_t, 3, 3> input;
    input.faces_buffer().emplace_back(0, 1, 2);
    input.faces_buffer().emplace_back(3, 4, 5);

    input.points_buffer().emplace_back(real_t(0), real_t(0), real_t(0));
    input.points_buffer().emplace_back(real_t(1), real_t(0), real_t(0));
    input.points_buffer().emplace_back(real_t(0.5), real_t(1), real_t(0));
    input.points_buffer().emplace_back(real_t(10), real_t(0), real_t(0));
    input.points_buffer().emplace_back(real_t(11), real_t(0), real_t(0));
    input.points_buffer().emplace_back(real_t(10.5), real_t(1), real_t(0));

    auto ids = make_ids<index_t>({1});  // Keep only second triangle

    auto result = tf::reindexed_by_ids(input.polygons(), ids);

    REQUIRE(result.faces().size() == 1);
    REQUIRE(result.points().size() == 3);  // Only points 3,4,5 kept

    // Face indices should be remapped to 0,1,2
    for (auto idx : result.faces()[0]) {
        REQUIRE(static_cast<std::size_t>(idx) < result.points().size());
    }
}

// =============================================================================
// reindex_by_ids_polygons_dynamic_basic
// =============================================================================

TEMPLATE_TEST_CASE("reindex_by_ids_polygons_dynamic_basic", "[reindex][by_ids][polygons][dynamic]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto input = tf::test::create_dynamic_polygons_3d<index_t, real_t>();

    auto ids = make_ids<index_t>({0});

    auto result = tf::reindexed_by_ids(input.polygons(), ids);

    REQUIRE(result.faces().size() == 1);
}

// =============================================================================
// reindex_by_ids_polygons_dynamic_mixed
// =============================================================================

TEMPLATE_TEST_CASE("reindex_by_ids_polygons_dynamic_mixed", "[reindex][by_ids][polygons][dynamic]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto input = tf::test::create_mixed_polygons_3d<index_t, real_t>();
    // 1 triangle (3 verts) + 1 quad (4 verts)

    auto ids = make_ids<index_t>({1});  // Keep only the quad

    auto result = tf::reindexed_by_ids(input.polygons(), ids);

    REQUIRE(result.faces().size() == 1);
    REQUIRE(result.faces()[0].size() == 4);
}

// =============================================================================
// reindex_by_ids_polygons_dynamic_index_map
// =============================================================================

TEMPLATE_TEST_CASE("reindex_by_ids_polygons_dynamic_index_map", "[reindex][by_ids][polygons][dynamic]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto input = tf::test::create_mixed_polygons_3d<index_t, real_t>();

    auto ids = make_ids<index_t>({0, 1});  // Keep both

    auto [result, face_im, point_im] = tf::reindexed_by_ids(input.polygons(), ids, tf::return_index_map);

    REQUIRE(result.faces().size() == 2);
    REQUIRE(face_im.f().size() == 2);
    REQUIRE(face_im.kept_ids().size() == 2);
}

// =============================================================================
// reindex_by_ids_range_basic
// =============================================================================

TEMPLATE_TEST_CASE("reindex_by_ids_range_basic", "[reindex][by_ids][range]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;

    tf::buffer<int> input;
    input.push_back(10);
    input.push_back(20);
    input.push_back(30);
    input.push_back(40);
    input.push_back(50);
    auto input_range = tf::make_range(input);

    auto ids = make_ids<index_t>({4, 2, 0});

    auto result = tf::reindexed_by_ids<index_t>(input_range, ids);

    REQUIRE(result.size() == 3);
    REQUIRE(result[0] == 50);
    REQUIRE(result[1] == 30);
    REQUIRE(result[2] == 10);
}

// =============================================================================
// reindex_by_ids_unit_vectors_basic
// =============================================================================

TEMPLATE_TEST_CASE("reindex_by_ids_unit_vectors_basic", "[reindex][by_ids][unit_vectors]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    tf::unit_vectors_buffer<real_t, 3> input;
    input.emplace_back(real_t(1), real_t(0), real_t(0));
    input.emplace_back(real_t(0), real_t(1), real_t(0));
    input.emplace_back(real_t(0), real_t(0), real_t(1));

    auto ids = make_ids<index_t>({1});

    auto result = tf::reindexed_by_ids<index_t>(input.unit_vectors(), ids);

    REQUIRE(result.size() == 1);
    REQUIRE(std::abs(result[0][1] - real_t(1)) < real_t(1e-5));
}
