/**
 * @file test_reindex_concatenated.cpp
 * @brief Tests for tf::concatenated(...)
 *
 * Copyright (c) 2025 Ziga Sajovic, XLAB
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <trueform/trueform.hpp>
#include "type_traits.hpp"
#include "mesh_generators.hpp"

// =============================================================================
// concatenated_points_two
// =============================================================================

TEMPLATE_TEST_CASE("concatenated_points_two", "[reindex][concatenated][points]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using real_t = typename TestType::real_type;

    tf::points_buffer<real_t, 3> input1;
    input1.emplace_back(real_t(0), real_t(0), real_t(0));
    input1.emplace_back(real_t(1), real_t(0), real_t(0));

    tf::points_buffer<real_t, 3> input2;
    input2.emplace_back(real_t(2), real_t(0), real_t(0));
    input2.emplace_back(real_t(3), real_t(0), real_t(0));

    auto result = tf::concatenated(input1.points(), input2.points());

    REQUIRE(result.size() == 4);
    REQUIRE(std::abs(result[0][0] - real_t(0)) < real_t(1e-5));
    REQUIRE(std::abs(result[1][0] - real_t(1)) < real_t(1e-5));
    REQUIRE(std::abs(result[2][0] - real_t(2)) < real_t(1e-5));
    REQUIRE(std::abs(result[3][0] - real_t(3)) < real_t(1e-5));
}

// =============================================================================
// concatenated_points_multiple
// =============================================================================

TEMPLATE_TEST_CASE("concatenated_points_multiple", "[reindex][concatenated][points]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using real_t = typename TestType::real_type;

    tf::points_buffer<real_t, 3> input1;
    input1.emplace_back(real_t(0), real_t(0), real_t(0));

    tf::points_buffer<real_t, 3> input2;
    input2.emplace_back(real_t(1), real_t(0), real_t(0));

    tf::points_buffer<real_t, 3> input3;
    input3.emplace_back(real_t(2), real_t(0), real_t(0));

    auto result = tf::concatenated(input1.points(), input2.points(), input3.points());

    REQUIRE(result.size() == 3);
}

// =============================================================================
// concatenated_points_empty
// =============================================================================

TEMPLATE_TEST_CASE("concatenated_points_empty", "[reindex][concatenated][points]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using real_t = typename TestType::real_type;

    tf::points_buffer<real_t, 3> input1;

    tf::points_buffer<real_t, 3> input2;
    input2.emplace_back(real_t(1), real_t(0), real_t(0));

    auto result = tf::concatenated(input1.points(), input2.points());

    REQUIRE(result.size() == 1);
}

// =============================================================================
// concatenated_vectors_basic
// =============================================================================

TEMPLATE_TEST_CASE("concatenated_vectors_basic", "[reindex][concatenated][vectors]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using real_t = typename TestType::real_type;

    tf::vectors_buffer<real_t, 3> input1;
    input1.emplace_back(real_t(1), real_t(0), real_t(0));

    tf::vectors_buffer<real_t, 3> input2;
    input2.emplace_back(real_t(0), real_t(1), real_t(0));

    auto result = tf::concatenated(input1.vectors(), input2.vectors());

    REQUIRE(result.size() == 2);
}

// =============================================================================
// concatenated_unit_vectors_basic
// =============================================================================

TEMPLATE_TEST_CASE("concatenated_unit_vectors_basic", "[reindex][concatenated][unit_vectors]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using real_t = typename TestType::real_type;

    tf::unit_vectors_buffer<real_t, 3> input1;
    input1.emplace_back(real_t(1), real_t(0), real_t(0));

    tf::unit_vectors_buffer<real_t, 3> input2;
    input2.emplace_back(real_t(0), real_t(1), real_t(0));

    auto result = tf::concatenated(input1.unit_vectors(), input2.unit_vectors());

    REQUIRE(result.size() == 2);
}

// =============================================================================
// concatenated_segments_basic
// =============================================================================

TEMPLATE_TEST_CASE("concatenated_segments_basic", "[reindex][concatenated][segments]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    tf::segments_buffer<index_t, real_t, 3> input1;
    input1.edges_buffer().emplace_back(0, 1);
    input1.points_buffer().emplace_back(real_t(0), real_t(0), real_t(0));
    input1.points_buffer().emplace_back(real_t(1), real_t(0), real_t(0));

    tf::segments_buffer<index_t, real_t, 3> input2;
    input2.edges_buffer().emplace_back(0, 1);
    input2.points_buffer().emplace_back(real_t(2), real_t(0), real_t(0));
    input2.points_buffer().emplace_back(real_t(3), real_t(0), real_t(0));

    auto result = tf::concatenated(input1.segments(), input2.segments());

    REQUIRE(result.edges().size() == 2);
    REQUIRE(result.points().size() == 4);

    // First edge indices should be in [0, 2)
    REQUIRE(result.edges()[0][0] >= 0);
    REQUIRE(result.edges()[0][0] < 2);
    REQUIRE(result.edges()[0][1] >= 0);
    REQUIRE(result.edges()[0][1] < 2);

    // Second edge indices should be in [2, 4)
    REQUIRE(result.edges()[1][0] >= 2);
    REQUIRE(result.edges()[1][0] < 4);
    REQUIRE(result.edges()[1][1] >= 2);
    REQUIRE(result.edges()[1][1] < 4);
}

// =============================================================================
// concatenated_segments_multiple
// =============================================================================

TEMPLATE_TEST_CASE("concatenated_segments_multiple", "[reindex][concatenated][segments]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    tf::segments_buffer<index_t, real_t, 3> input1;
    input1.edges_buffer().emplace_back(0, 1);
    input1.points_buffer().emplace_back(real_t(0), real_t(0), real_t(0));
    input1.points_buffer().emplace_back(real_t(1), real_t(0), real_t(0));

    tf::segments_buffer<index_t, real_t, 3> input2;
    input2.edges_buffer().emplace_back(0, 1);
    input2.points_buffer().emplace_back(real_t(2), real_t(0), real_t(0));
    input2.points_buffer().emplace_back(real_t(3), real_t(0), real_t(0));

    tf::segments_buffer<index_t, real_t, 3> input3;
    input3.edges_buffer().emplace_back(0, 1);
    input3.points_buffer().emplace_back(real_t(4), real_t(0), real_t(0));
    input3.points_buffer().emplace_back(real_t(5), real_t(0), real_t(0));

    auto result = tf::concatenated(input1.segments(), input2.segments(), input3.segments());

    REQUIRE(result.edges().size() == 3);
    REQUIRE(result.points().size() == 6);

    // First edge indices in [0, 2)
    REQUIRE(result.edges()[0][0] >= 0);
    REQUIRE(result.edges()[0][0] < 2);

    // Second edge indices in [2, 4)
    REQUIRE(result.edges()[1][0] >= 2);
    REQUIRE(result.edges()[1][0] < 4);

    // Third edge indices in [4, 6)
    REQUIRE(result.edges()[2][0] >= 4);
    REQUIRE(result.edges()[2][0] < 6);
}

// =============================================================================
// concatenated_segments_index_consistency
// =============================================================================

TEMPLATE_TEST_CASE("concatenated_segments_index_consistency", "[reindex][concatenated][segments]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    tf::segments_buffer<index_t, real_t, 3> input1;
    input1.edges_buffer().emplace_back(0, 1);
    input1.edges_buffer().emplace_back(1, 2);
    input1.points_buffer().emplace_back(real_t(0), real_t(0), real_t(0));
    input1.points_buffer().emplace_back(real_t(1), real_t(0), real_t(0));
    input1.points_buffer().emplace_back(real_t(2), real_t(0), real_t(0));

    tf::segments_buffer<index_t, real_t, 3> input2;
    input2.edges_buffer().emplace_back(0, 1);
    input2.edges_buffer().emplace_back(1, 2);
    input2.points_buffer().emplace_back(real_t(10), real_t(0), real_t(0));
    input2.points_buffer().emplace_back(real_t(11), real_t(0), real_t(0));
    input2.points_buffer().emplace_back(real_t(12), real_t(0), real_t(0));

    const auto n1 = input1.points().size();
    const auto n2 = input2.points().size();

    auto result = tf::concatenated(input1.segments(), input2.segments());

    // Edges from input1 should have indices in [0, n1)
    for (std::size_t e = 0; e < input1.edges().size(); ++e) {
        REQUIRE(static_cast<std::size_t>(result.edges()[e][0]) < n1);
        REQUIRE(static_cast<std::size_t>(result.edges()[e][1]) < n1);
    }

    // Edges from input2 should have indices in [n1, n1+n2)
    for (std::size_t e = 0; e < input2.edges().size(); ++e) {
        auto re = e + input1.edges().size();
        REQUIRE(static_cast<std::size_t>(result.edges()[re][0]) >= n1);
        REQUIRE(static_cast<std::size_t>(result.edges()[re][0]) < n1 + n2);
        REQUIRE(static_cast<std::size_t>(result.edges()[re][1]) >= n1);
        REQUIRE(static_cast<std::size_t>(result.edges()[re][1]) < n1 + n2);
    }
}

// =============================================================================
// concatenated_polygons_basic
// =============================================================================

TEMPLATE_TEST_CASE("concatenated_polygons_basic", "[reindex][concatenated][polygons]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    tf::polygons_buffer<index_t, real_t, 3, 3> input1;
    input1.faces_buffer().emplace_back(0, 1, 2);
    input1.points_buffer().emplace_back(real_t(0), real_t(0), real_t(0));
    input1.points_buffer().emplace_back(real_t(1), real_t(0), real_t(0));
    input1.points_buffer().emplace_back(real_t(0.5), real_t(1), real_t(0));

    tf::polygons_buffer<index_t, real_t, 3, 3> input2;
    input2.faces_buffer().emplace_back(0, 1, 2);
    input2.points_buffer().emplace_back(real_t(10), real_t(0), real_t(0));
    input2.points_buffer().emplace_back(real_t(11), real_t(0), real_t(0));
    input2.points_buffer().emplace_back(real_t(10.5), real_t(1), real_t(0));

    auto result = tf::concatenated(input1.polygons(), input2.polygons());

    REQUIRE(result.faces().size() == 2);
    REQUIRE(result.points().size() == 6);
}

// =============================================================================
// concatenated_polygons_index_offset
// =============================================================================

TEMPLATE_TEST_CASE("concatenated_polygons_index_offset", "[reindex][concatenated][polygons]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    tf::polygons_buffer<index_t, real_t, 3, 3> input1;
    input1.faces_buffer().emplace_back(0, 1, 2);
    input1.points_buffer().emplace_back(real_t(0), real_t(0), real_t(0));
    input1.points_buffer().emplace_back(real_t(1), real_t(0), real_t(0));
    input1.points_buffer().emplace_back(real_t(0.5), real_t(1), real_t(0));

    tf::polygons_buffer<index_t, real_t, 3, 3> input2;
    input2.faces_buffer().emplace_back(0, 1, 2);
    input2.points_buffer().emplace_back(real_t(10), real_t(0), real_t(0));
    input2.points_buffer().emplace_back(real_t(11), real_t(0), real_t(0));
    input2.points_buffer().emplace_back(real_t(10.5), real_t(1), real_t(0));

    const auto n1 = input1.points().size();

    auto result = tf::concatenated(input1.polygons(), input2.polygons());

    // First face indices should be in [0, n1)
    for (auto idx : result.faces()[0]) {
        REQUIRE(static_cast<std::size_t>(idx) < n1);
    }

    // Second face indices should be in [n1, n1+n2)
    for (auto idx : result.faces()[1]) {
        REQUIRE(static_cast<std::size_t>(idx) >= n1);
        REQUIRE(static_cast<std::size_t>(idx) < result.points().size());
    }
}

// =============================================================================
// concatenated_polygons_multiple
// =============================================================================

TEMPLATE_TEST_CASE("concatenated_polygons_multiple", "[reindex][concatenated][polygons]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    tf::polygons_buffer<index_t, real_t, 3, 3> input1;
    input1.faces_buffer().emplace_back(0, 1, 2);
    input1.points_buffer().emplace_back(real_t(0), real_t(0), real_t(0));
    input1.points_buffer().emplace_back(real_t(1), real_t(0), real_t(0));
    input1.points_buffer().emplace_back(real_t(0.5), real_t(1), real_t(0));

    tf::polygons_buffer<index_t, real_t, 3, 3> input2;
    input2.faces_buffer().emplace_back(0, 1, 2);
    input2.points_buffer().emplace_back(real_t(10), real_t(0), real_t(0));
    input2.points_buffer().emplace_back(real_t(11), real_t(0), real_t(0));
    input2.points_buffer().emplace_back(real_t(10.5), real_t(1), real_t(0));

    tf::polygons_buffer<index_t, real_t, 3, 3> input3;
    input3.faces_buffer().emplace_back(0, 1, 2);
    input3.points_buffer().emplace_back(real_t(20), real_t(0), real_t(0));
    input3.points_buffer().emplace_back(real_t(21), real_t(0), real_t(0));
    input3.points_buffer().emplace_back(real_t(20.5), real_t(1), real_t(0));

    const auto n1 = input1.points().size();
    const auto n2 = input2.points().size();

    auto result = tf::concatenated(input1.polygons(), input2.polygons(), input3.polygons());

    REQUIRE(result.faces().size() == 3);
    REQUIRE(result.points().size() == 9);

    // Face 0 indices in [0, n1)
    for (auto idx : result.faces()[0]) {
        REQUIRE(static_cast<std::size_t>(idx) < n1);
    }

    // Face 1 indices in [n1, n1+n2)
    for (auto idx : result.faces()[1]) {
        REQUIRE(static_cast<std::size_t>(idx) >= n1);
        REQUIRE(static_cast<std::size_t>(idx) < n1 + n2);
    }

    // Face 2 indices in [n1+n2, n1+n2+n3)
    for (auto idx : result.faces()[2]) {
        REQUIRE(static_cast<std::size_t>(idx) >= n1 + n2);
        REQUIRE(static_cast<std::size_t>(idx) < result.points().size());
    }
}

// =============================================================================
// concatenated_polygons_index_consistency
// =============================================================================

TEMPLATE_TEST_CASE("concatenated_polygons_index_consistency", "[reindex][concatenated][polygons]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto input1 = tf::test::create_cube_polygons<index_t, real_t>();
    auto input2 = tf::test::create_cube_polygons<index_t, real_t>(
        std::array<real_t, 3>{real_t(5), real_t(0), real_t(0)}, real_t(1));

    const auto n1 = input1.points().size();
    const auto f1 = input1.faces().size();

    auto result = tf::concatenated(input1.polygons(), input2.polygons());

    // Faces from input1 should have indices in [0, n1)
    for (std::size_t f = 0; f < f1; ++f) {
        for (auto idx : result.faces()[f]) {
            REQUIRE(static_cast<std::size_t>(idx) < n1);
        }
    }

    // Faces from input2 should have indices in [n1, total)
    for (std::size_t f = f1; f < result.faces().size(); ++f) {
        for (auto idx : result.faces()[f]) {
            REQUIRE(static_cast<std::size_t>(idx) >= n1);
            REQUIRE(static_cast<std::size_t>(idx) < result.points().size());
        }
    }
}

// =============================================================================
// concatenated_polygons_dynamic_same
// =============================================================================

TEMPLATE_TEST_CASE("concatenated_polygons_dynamic_same", "[reindex][concatenated][polygons][dynamic]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    tf::polygons_buffer<index_t, real_t, 3, 3> input1;
    input1.faces_buffer().emplace_back(0, 1, 2);
    input1.points_buffer().emplace_back(real_t(0), real_t(0), real_t(0));
    input1.points_buffer().emplace_back(real_t(1), real_t(0), real_t(0));
    input1.points_buffer().emplace_back(real_t(0.5), real_t(1), real_t(0));

    tf::polygons_buffer<index_t, real_t, 3, 3> input2;
    input2.faces_buffer().emplace_back(0, 1, 2);
    input2.points_buffer().emplace_back(real_t(10), real_t(0), real_t(0));
    input2.points_buffer().emplace_back(real_t(11), real_t(0), real_t(0));
    input2.points_buffer().emplace_back(real_t(10.5), real_t(1), real_t(0));

    auto result = tf::concatenated(input1.polygons(), input2.polygons());

    REQUIRE(result.faces().size() == 2);
    REQUIRE(result.faces()[0].size() == 3);
    REQUIRE(result.faces()[1].size() == 3);
}

// =============================================================================
// concatenated_polygons_dynamic_mixed
// =============================================================================

TEMPLATE_TEST_CASE("concatenated_polygons_dynamic_mixed", "[reindex][concatenated][polygons][dynamic]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    // Triangle mesh
    tf::polygons_buffer<index_t, real_t, 3, 3> input1;
    input1.faces_buffer().emplace_back(0, 1, 2);
    input1.points_buffer().emplace_back(real_t(0), real_t(0), real_t(0));
    input1.points_buffer().emplace_back(real_t(1), real_t(0), real_t(0));
    input1.points_buffer().emplace_back(real_t(0.5), real_t(1), real_t(0));

    // Quad mesh
    tf::polygons_buffer<index_t, real_t, 3, 4> input2;
    input2.faces_buffer().emplace_back(0, 1, 2, 3);
    input2.points_buffer().emplace_back(real_t(10), real_t(0), real_t(0));
    input2.points_buffer().emplace_back(real_t(11), real_t(0), real_t(0));
    input2.points_buffer().emplace_back(real_t(11), real_t(1), real_t(0));
    input2.points_buffer().emplace_back(real_t(10), real_t(1), real_t(0));

    const auto n1 = input1.points().size();

    auto result = tf::concatenated(input1.polygons(), input2.polygons());

    REQUIRE(result.faces().size() == 2);
    REQUIRE(result.faces()[0].size() == 3);
    REQUIRE(result.faces()[1].size() == 4);

    // Check index ranges
    for (auto idx : result.faces()[0]) {
        REQUIRE(static_cast<std::size_t>(idx) < n1);
    }
    for (auto idx : result.faces()[1]) {
        REQUIRE(static_cast<std::size_t>(idx) >= n1);
        REQUIRE(static_cast<std::size_t>(idx) < result.points().size());
    }
}

// =============================================================================
// concatenated_polygons_dynamic_to_dynamic
// =============================================================================

TEMPLATE_TEST_CASE("concatenated_polygons_dynamic_to_dynamic", "[reindex][concatenated][polygons][dynamic]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto input1 = tf::test::create_mixed_polygons_3d<index_t, real_t>();
    auto input2 = tf::test::create_dynamic_polygons_3d<index_t, real_t>();

    const auto n1 = input1.points().size();
    const auto f1 = input1.faces().size();

    auto result = tf::concatenated(input1.polygons(), input2.polygons());

    REQUIRE(result.faces().size() == f1 + input2.faces().size());

    // Check index ranges
    for (std::size_t f = 0; f < f1; ++f) {
        for (auto idx : result.faces()[f]) {
            REQUIRE(static_cast<std::size_t>(idx) < n1);
        }
    }
    for (std::size_t f = f1; f < result.faces().size(); ++f) {
        for (auto idx : result.faces()[f]) {
            REQUIRE(static_cast<std::size_t>(idx) >= n1);
            REQUIRE(static_cast<std::size_t>(idx) < result.points().size());
        }
    }
}

// =============================================================================
// concatenated_polygons_dynamic_index_verify
// =============================================================================

TEMPLATE_TEST_CASE("concatenated_polygons_dynamic_index_verify", "[reindex][concatenated][polygons][dynamic]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    tf::polygons_buffer<index_t, real_t, 3, tf::dynamic_size> input1;
    input1.faces_buffer().push_back({0, 1, 2});
    input1.faces_buffer().push_back({0, 2, 3});
    input1.points_buffer().emplace_back(real_t(0), real_t(0), real_t(0));
    input1.points_buffer().emplace_back(real_t(1), real_t(0), real_t(0));
    input1.points_buffer().emplace_back(real_t(1), real_t(1), real_t(0));
    input1.points_buffer().emplace_back(real_t(0), real_t(1), real_t(0));

    tf::polygons_buffer<index_t, real_t, 3, tf::dynamic_size> input2;
    input2.faces_buffer().push_back({0, 1, 2, 3});
    input2.points_buffer().emplace_back(real_t(10), real_t(0), real_t(0));
    input2.points_buffer().emplace_back(real_t(11), real_t(0), real_t(0));
    input2.points_buffer().emplace_back(real_t(11), real_t(1), real_t(0));
    input2.points_buffer().emplace_back(real_t(10), real_t(1), real_t(0));

    const auto n1 = input1.points().size();

    auto result = tf::concatenated(input1.polygons(), input2.polygons());

    REQUIRE(result.faces().size() == 3);
    REQUIRE(result.points().size() == 8);

    // Face 0: indices in [0, 4)
    for (auto idx : result.faces()[0]) {
        REQUIRE(static_cast<std::size_t>(idx) < n1);
    }

    // Face 1: indices in [0, 4)
    for (auto idx : result.faces()[1]) {
        REQUIRE(static_cast<std::size_t>(idx) < n1);
    }

    // Face 2: indices in [4, 8)
    for (auto idx : result.faces()[2]) {
        REQUIRE(static_cast<std::size_t>(idx) >= n1);
        REQUIRE(static_cast<std::size_t>(idx) < result.points().size());
    }
}

// =============================================================================
// concatenated_polygons_transformed
// =============================================================================

TEMPLATE_TEST_CASE("concatenated_polygons_transformed", "[reindex][concatenated][polygons][transformed]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    tf::polygons_buffer<index_t, real_t, 3, 3> input1;
    input1.faces_buffer().emplace_back(0, 1, 2);
    input1.points_buffer().emplace_back(real_t(0), real_t(0), real_t(0));
    input1.points_buffer().emplace_back(real_t(1), real_t(0), real_t(0));
    input1.points_buffer().emplace_back(real_t(0.5), real_t(1), real_t(0));

    tf::polygons_buffer<index_t, real_t, 3, 3> input2;
    input2.faces_buffer().emplace_back(0, 1, 2);
    input2.points_buffer().emplace_back(real_t(0), real_t(0), real_t(0));
    input2.points_buffer().emplace_back(real_t(1), real_t(0), real_t(0));
    input2.points_buffer().emplace_back(real_t(0.5), real_t(1), real_t(0));

    // Apply translation to second mesh
    auto translation = tf::make_transformation_from_translation(
        tf::vector<real_t, 3>{real_t(10), real_t(0), real_t(0)});
    auto transformed2 = input2.polygons() | tf::tag(translation);

    const auto n1 = input1.points().size();

    auto result = tf::concatenated(input1.polygons(), transformed2);

    REQUIRE(result.faces().size() == 2);
    REQUIRE(result.points().size() == 6);

    // First triangle points at original positions
    REQUIRE(std::abs(result.points()[0][0] - real_t(0)) < real_t(1e-5));
    REQUIRE(std::abs(result.points()[1][0] - real_t(1)) < real_t(1e-5));
    REQUIRE(std::abs(result.points()[2][0] - real_t(0.5)) < real_t(1e-5));

    // Second triangle points translated by 10
    REQUIRE(std::abs(result.points()[3][0] - real_t(10)) < real_t(1e-5));
    REQUIRE(std::abs(result.points()[4][0] - real_t(11)) < real_t(1e-5));
    REQUIRE(std::abs(result.points()[5][0] - real_t(10.5)) < real_t(1e-5));

    // Check index ranges
    for (auto idx : result.faces()[0]) {
        REQUIRE(static_cast<std::size_t>(idx) < n1);
    }
    for (auto idx : result.faces()[1]) {
        REQUIRE(static_cast<std::size_t>(idx) >= n1);
        REQUIRE(static_cast<std::size_t>(idx) < result.points().size());
    }
}

// =============================================================================
// concatenated_segments_transformed
// =============================================================================

TEMPLATE_TEST_CASE("concatenated_segments_transformed", "[reindex][concatenated][segments][transformed]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    tf::segments_buffer<index_t, real_t, 3> input1;
    input1.edges_buffer().emplace_back(0, 1);
    input1.points_buffer().emplace_back(real_t(0), real_t(0), real_t(0));
    input1.points_buffer().emplace_back(real_t(1), real_t(0), real_t(0));

    tf::segments_buffer<index_t, real_t, 3> input2;
    input2.edges_buffer().emplace_back(0, 1);
    input2.points_buffer().emplace_back(real_t(0), real_t(0), real_t(0));
    input2.points_buffer().emplace_back(real_t(1), real_t(0), real_t(0));

    auto translation = tf::make_transformation_from_translation(
        tf::vector<real_t, 3>{real_t(5), real_t(0), real_t(0)});
    auto transformed2 = input2.segments() | tf::tag(translation);

    const auto n1 = input1.points().size();

    auto result = tf::concatenated(input1.segments(), transformed2);

    REQUIRE(result.edges().size() == 2);
    REQUIRE(result.points().size() == 4);

    // Second segment points translated
    REQUIRE(std::abs(result.points()[2][0] - real_t(5)) < real_t(1e-5));
    REQUIRE(std::abs(result.points()[3][0] - real_t(6)) < real_t(1e-5));

    // Check index ranges
    REQUIRE(static_cast<std::size_t>(result.edges()[0][0]) < n1);
    REQUIRE(static_cast<std::size_t>(result.edges()[0][1]) < n1);
    REQUIRE(static_cast<std::size_t>(result.edges()[1][0]) >= n1);
    REQUIRE(static_cast<std::size_t>(result.edges()[1][1]) >= n1);
}

// =============================================================================
// concatenated_points_transformed
// =============================================================================

TEMPLATE_TEST_CASE("concatenated_points_transformed", "[reindex][concatenated][points][transformed]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using real_t = typename TestType::real_type;

    tf::points_buffer<real_t, 3> input1;
    input1.emplace_back(real_t(0), real_t(0), real_t(0));
    input1.emplace_back(real_t(1), real_t(0), real_t(0));

    tf::points_buffer<real_t, 3> input2;
    input2.emplace_back(real_t(0), real_t(0), real_t(0));
    input2.emplace_back(real_t(1), real_t(0), real_t(0));

    auto translation = tf::make_transformation_from_translation(
        tf::vector<real_t, 3>{real_t(10), real_t(20), real_t(30)});
    auto transformed2 = input2.points() | tf::tag(translation);

    auto result = tf::concatenated(input1.points(), transformed2);

    REQUIRE(result.size() == 4);

    // Second set of points translated
    REQUIRE(std::abs(result[2][0] - real_t(10)) < real_t(1e-5));
    REQUIRE(std::abs(result[2][1] - real_t(20)) < real_t(1e-5));
    REQUIRE(std::abs(result[2][2] - real_t(30)) < real_t(1e-5));

    REQUIRE(std::abs(result[3][0] - real_t(11)) < real_t(1e-5));
    REQUIRE(std::abs(result[3][1] - real_t(20)) < real_t(1e-5));
    REQUIRE(std::abs(result[3][2] - real_t(30)) < real_t(1e-5));
}

// =============================================================================
// concatenated_cube_meshes
// =============================================================================

TEMPLATE_TEST_CASE("concatenated_cube_meshes", "[reindex][concatenated][polygons]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto cube1 = tf::test::create_cube_polygons<index_t, real_t>();
    auto cube2 = tf::test::create_cube_polygons<index_t, real_t>(
        std::array<real_t, 3>{real_t(5), real_t(0), real_t(0)}, real_t(1));

    const auto n1 = cube1.points().size();
    const auto f1 = cube1.faces().size();

    auto result = tf::concatenated(cube1.polygons(), cube2.polygons());

    REQUIRE(result.faces().size() == f1 * 2);
    REQUIRE(result.points().size() == n1 * 2);

    // Check index ranges
    for (std::size_t f = 0; f < f1; ++f) {
        for (auto idx : result.faces()[f]) {
            REQUIRE(static_cast<std::size_t>(idx) < n1);
        }
    }
    for (std::size_t f = f1; f < result.faces().size(); ++f) {
        for (auto idx : result.faces()[f]) {
            REQUIRE(static_cast<std::size_t>(idx) >= n1);
            REQUIRE(static_cast<std::size_t>(idx) < result.points().size());
        }
    }
}

// =============================================================================
// concatenated_range_of_polygons
// =============================================================================

TEMPLATE_TEST_CASE("concatenated_range_of_polygons", "[reindex][concatenated][polygons]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    std::vector<tf::polygons_buffer<index_t, real_t, 3, 3>> inputs;

    for (int i = 0; i < 3; ++i) {
        tf::polygons_buffer<index_t, real_t, 3, 3> mesh;
        mesh.faces_buffer().emplace_back(0, 1, 2);
        mesh.points_buffer().emplace_back(real_t(i * 10), real_t(0), real_t(0));
        mesh.points_buffer().emplace_back(real_t(i * 10 + 1), real_t(0), real_t(0));
        mesh.points_buffer().emplace_back(real_t(i * 10 + 0.5), real_t(1), real_t(0));
        inputs.push_back(std::move(mesh));
    }

    auto views = tf::make_mapped_range(inputs, [](auto& x) { return x.polygons(); });

    auto result = tf::concatenated(views);

    REQUIRE(result.faces().size() == 3);
    REQUIRE(result.points().size() == 9);

    // Check index ranges for each face
    for (std::size_t f = 0; f < result.faces().size(); ++f) {
        std::size_t offset = f * 3;
        for (auto idx : result.faces()[f]) {
            REQUIRE(static_cast<std::size_t>(idx) >= offset);
            REQUIRE(static_cast<std::size_t>(idx) < offset + 3);
        }
    }
}

// =============================================================================
// concatenated_polygons_multiple_transforms
// =============================================================================

TEMPLATE_TEST_CASE("concatenated_polygons_multiple_transforms", "[reindex][concatenated][polygons][transformed]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    tf::polygons_buffer<index_t, real_t, 3, 3> base;
    base.faces_buffer().emplace_back(0, 1, 2);
    base.points_buffer().emplace_back(real_t(0), real_t(0), real_t(0));
    base.points_buffer().emplace_back(real_t(1), real_t(0), real_t(0));
    base.points_buffer().emplace_back(real_t(0.5), real_t(1), real_t(0));

    auto t1 = tf::make_transformation_from_translation(
        tf::vector<real_t, 3>{real_t(0), real_t(0), real_t(0)});
    auto t2 = tf::make_transformation_from_translation(
        tf::vector<real_t, 3>{real_t(5), real_t(0), real_t(0)});
    auto t3 = tf::make_transformation_from_translation(
        tf::vector<real_t, 3>{real_t(10), real_t(0), real_t(0)});

    auto trans1 = base.polygons() | tf::tag(t1);
    auto trans2 = base.polygons() | tf::tag(t2);
    auto trans3 = base.polygons() | tf::tag(t3);

    auto result = tf::concatenated(trans1, trans2, trans3);

    REQUIRE(result.faces().size() == 3);
    REQUIRE(result.points().size() == 9);

    // Verify point positions
    REQUIRE(std::abs(result.points()[0][0] - real_t(0)) < real_t(1e-5));
    REQUIRE(std::abs(result.points()[3][0] - real_t(5)) < real_t(1e-5));
    REQUIRE(std::abs(result.points()[6][0] - real_t(10)) < real_t(1e-5));

    // Check index ranges
    for (auto idx : result.faces()[0]) {
        REQUIRE(static_cast<std::size_t>(idx) < 3);
    }
    for (auto idx : result.faces()[1]) {
        REQUIRE(static_cast<std::size_t>(idx) >= 3);
        REQUIRE(static_cast<std::size_t>(idx) < 6);
    }
    for (auto idx : result.faces()[2]) {
        REQUIRE(static_cast<std::size_t>(idx) >= 6);
        REQUIRE(static_cast<std::size_t>(idx) < 9);
    }
}
