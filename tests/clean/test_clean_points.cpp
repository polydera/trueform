/**
 * @file test_clean_points.cpp
 * @brief Tests for tf::cleaned(points, ...)
 *
 * Tests for:
 * - clean_points_no_duplicates
 * - clean_points_exact_duplicates
 * - clean_points_tolerance_duplicates
 * - clean_points_with_index_map
 * - clean_points_empty
 * - clean_points_single
 *
 * Copyright (c) 2025 Ziga Sajovic, XLAB
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <trueform/trueform.hpp>
#include "type_traits.hpp"
#include "canonicalize_points.hpp"

// =============================================================================
// clean_points_no_duplicates
// =============================================================================

TEMPLATE_TEST_CASE("clean_points_no_duplicates", "[clean][clean_points]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using real_t = typename TestType::real_type;

    // Create points with no duplicates
    tf::points_buffer<real_t, 3> input;
    input.emplace_back(real_t(0), real_t(0), real_t(0));
    input.emplace_back(real_t(1), real_t(0), real_t(0));
    input.emplace_back(real_t(0), real_t(1), real_t(0));
    input.emplace_back(real_t(0), real_t(0), real_t(1));

    auto result = tf::cleaned(input.points());

    // Points should remain unchanged
    REQUIRE(result.size() == 4);

    // Verify canonical comparison
    auto canonical_result = tf::test::canonicalize_points(result);
    auto canonical_expected = tf::test::canonicalize_points(input);
    REQUIRE(tf::test::points_equal(canonical_result, canonical_expected));
}

// =============================================================================
// clean_points_exact_duplicates
// =============================================================================

TEMPLATE_TEST_CASE("clean_points_exact_duplicates", "[clean][clean_points]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using real_t = typename TestType::real_type;

    // Create points with exact duplicates
    tf::points_buffer<real_t, 3> input;
    input.emplace_back(real_t(0), real_t(0), real_t(0));
    input.emplace_back(real_t(1), real_t(0), real_t(0));
    input.emplace_back(real_t(0), real_t(0), real_t(0));  // duplicate of [0]
    input.emplace_back(real_t(1), real_t(0), real_t(0));  // duplicate of [1]
    input.emplace_back(real_t(2), real_t(0), real_t(0));

    auto result = tf::cleaned(input.points());

    // Should have 3 unique points
    REQUIRE(result.size() == 3);

    // Build expected output
    tf::points_buffer<real_t, 3> expected;
    expected.emplace_back(real_t(0), real_t(0), real_t(0));
    expected.emplace_back(real_t(1), real_t(0), real_t(0));
    expected.emplace_back(real_t(2), real_t(0), real_t(0));

    auto canonical_result = tf::test::canonicalize_points(result);
    auto canonical_expected = tf::test::canonicalize_points(expected);
    REQUIRE(tf::test::points_equal(canonical_result, canonical_expected));
}

// =============================================================================
// clean_points_tolerance_duplicates
// =============================================================================

TEMPLATE_TEST_CASE("clean_points_tolerance_duplicates", "[clean][clean_points]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using real_t = typename TestType::real_type;

    // Create points where some are within tolerance of each other
    tf::points_buffer<real_t, 3> input;
    input.emplace_back(real_t(0), real_t(0), real_t(0));
    input.emplace_back(real_t(0.001), real_t(0), real_t(0));  // within tolerance of [0]
    input.emplace_back(real_t(1), real_t(0), real_t(0));
    input.emplace_back(real_t(2), real_t(0), real_t(0));

    const real_t tolerance = real_t(0.01);
    auto result = tf::cleaned(input.points(), tolerance);

    // Points [0] and [1] should merge, leaving 3 points
    REQUIRE(result.size() == 3);
}

// =============================================================================
// clean_points_with_index_map
// =============================================================================

TEMPLATE_TEST_CASE("clean_points_with_index_map", "[clean][clean_points]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    // Create points with duplicates
    tf::points_buffer<real_t, 3> input;
    input.emplace_back(real_t(0), real_t(0), real_t(0));  // -> 0
    input.emplace_back(real_t(1), real_t(0), real_t(0));  // -> 1
    input.emplace_back(real_t(0), real_t(0), real_t(0));  // duplicate -> 0
    input.emplace_back(real_t(2), real_t(0), real_t(0));  // -> 2

    auto [result, index_map] = tf::cleaned<index_t>(input.points(), tf::return_index_map);

    // Should have 3 unique points
    REQUIRE(result.size() == 3);

    // Index map should have 4 entries (one per input point)
    REQUIRE(index_map.f().size() == 4);

    // Verify duplicate mapping: input[2] should map to same as input[0]
    REQUIRE(index_map.f()[0] == index_map.f()[2]);

    // All indices should point to valid output elements
    for (std::size_t i = 0; i < index_map.f().size(); ++i) {
        REQUIRE(index_map.f()[i] >= 0);
        REQUIRE(static_cast<std::size_t>(index_map.f()[i]) < result.size());
    }
}

// =============================================================================
// clean_points_with_index_map_tolerance
// =============================================================================

TEMPLATE_TEST_CASE("clean_points_with_index_map_tolerance", "[clean][clean_points]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    // Create points where some are within tolerance
    tf::points_buffer<real_t, 3> input;
    input.emplace_back(real_t(0), real_t(0), real_t(0));       // -> 0
    input.emplace_back(real_t(0.001), real_t(0), real_t(0));   // within tolerance -> 0
    input.emplace_back(real_t(1), real_t(0), real_t(0));       // -> 1

    const real_t tolerance = real_t(0.01);
    auto [result, index_map] = tf::cleaned<index_t>(input.points(), tolerance, tf::return_index_map);

    // Should have 2 unique points
    REQUIRE(result.size() == 2);

    // Index map should map first two points to same index
    REQUIRE(index_map.f()[0] == index_map.f()[1]);
}

// =============================================================================
// clean_points_empty
// =============================================================================

TEMPLATE_TEST_CASE("clean_points_empty", "[clean][clean_points]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using real_t = typename TestType::real_type;

    tf::points_buffer<real_t, 3> input;

    auto result = tf::cleaned(input.points());

    REQUIRE(result.size() == 0);
}

// =============================================================================
// clean_points_single
// =============================================================================

TEMPLATE_TEST_CASE("clean_points_single", "[clean][clean_points]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using real_t = typename TestType::real_type;

    tf::points_buffer<real_t, 3> input;
    input.emplace_back(real_t(1), real_t(2), real_t(3));

    auto result = tf::cleaned(input.points());

    REQUIRE(result.size() == 1);

    // Verify the point value
    REQUIRE(std::abs(result[0][0] - real_t(1)) < real_t(1e-5));
    REQUIRE(std::abs(result[0][1] - real_t(2)) < real_t(1e-5));
    REQUIRE(std::abs(result[0][2] - real_t(3)) < real_t(1e-5));
}

// =============================================================================
// clean_points_2d
// =============================================================================

TEMPLATE_TEST_CASE("clean_points_2d", "[clean][clean_points]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using real_t = typename TestType::real_type;

    // Test with 2D points
    tf::points_buffer<real_t, 2> input;
    input.emplace_back(real_t(0), real_t(0));
    input.emplace_back(real_t(1), real_t(0));
    input.emplace_back(real_t(0), real_t(0));  // duplicate
    input.emplace_back(real_t(0), real_t(1));

    auto result = tf::cleaned(input.points());

    REQUIRE(result.size() == 3);
}

// =============================================================================
// clean_points_all_duplicates
// =============================================================================

TEMPLATE_TEST_CASE("clean_points_all_duplicates", "[clean][clean_points]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using real_t = typename TestType::real_type;

    // All points are the same
    tf::points_buffer<real_t, 3> input;
    input.emplace_back(real_t(1), real_t(2), real_t(3));
    input.emplace_back(real_t(1), real_t(2), real_t(3));
    input.emplace_back(real_t(1), real_t(2), real_t(3));
    input.emplace_back(real_t(1), real_t(2), real_t(3));

    auto result = tf::cleaned(input.points());

    REQUIRE(result.size() == 1);
}

// =============================================================================
// clean_points_large_point_cloud
// =============================================================================

TEMPLATE_TEST_CASE("clean_points_large_point_cloud", "[clean][clean_points]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using real_t = typename TestType::real_type;

    // Create a grid of points with some duplicates
    tf::points_buffer<real_t, 3> input;

    // 10x10x10 grid
    for (int x = 0; x < 10; ++x) {
        for (int y = 0; y < 10; ++y) {
            for (int z = 0; z < 10; ++z) {
                input.emplace_back(real_t(x), real_t(y), real_t(z));
            }
        }
    }
    // Add duplicates of corner points
    input.emplace_back(real_t(0), real_t(0), real_t(0));
    input.emplace_back(real_t(9), real_t(9), real_t(9));
    input.emplace_back(real_t(0), real_t(0), real_t(0));

    auto result = tf::cleaned(input.points());

    // Should have exactly 1000 unique points
    REQUIRE(result.size() == 1000);
}

// =============================================================================
// clean_points_cluster_tolerance
// =============================================================================

TEMPLATE_TEST_CASE("clean_points_cluster_tolerance", "[clean][clean_points]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using real_t = typename TestType::real_type;

    // Create clusters of points that should merge
    tf::points_buffer<real_t, 3> input;

    // Cluster 1 around origin
    input.emplace_back(real_t(0), real_t(0), real_t(0));
    input.emplace_back(real_t(0.001), real_t(0.001), real_t(0));
    input.emplace_back(real_t(0), real_t(0.002), real_t(0.001));

    // Cluster 2 around (1, 0, 0)
    input.emplace_back(real_t(1), real_t(0), real_t(0));
    input.emplace_back(real_t(1.001), real_t(0), real_t(0.001));

    // Isolated point
    input.emplace_back(real_t(5), real_t(5), real_t(5));

    const real_t tolerance = real_t(0.01);
    auto result = tf::cleaned(input.points(), tolerance);

    // Should have 3 points: merged cluster 1, merged cluster 2, isolated point
    REQUIRE(result.size() == 3);
}

// =============================================================================
// clean_points_chain_merge_tolerance
// =============================================================================

TEMPLATE_TEST_CASE("clean_points_chain_merge_tolerance", "[clean][clean_points]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using real_t = typename TestType::real_type;

    // Points in a chain where each is within tolerance of the next
    // but first and last are beyond tolerance
    tf::points_buffer<real_t, 3> input;
    input.emplace_back(real_t(0), real_t(0), real_t(0));
    input.emplace_back(real_t(0.005), real_t(0), real_t(0));
    input.emplace_back(real_t(0.01), real_t(0), real_t(0));
    input.emplace_back(real_t(0.015), real_t(0), real_t(0));
    input.emplace_back(real_t(0.02), real_t(0), real_t(0));

    const real_t tolerance = real_t(0.006);  // Each adjacent pair is within tolerance
    auto result = tf::cleaned(input.points(), tolerance);

    // The merging behavior depends on implementation
    // At minimum, some points should merge
    REQUIRE(result.size() < 5);
}

// =============================================================================
// clean_points_negative_coordinates
// =============================================================================

TEMPLATE_TEST_CASE("clean_points_negative_coordinates", "[clean][clean_points]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using real_t = typename TestType::real_type;

    tf::points_buffer<real_t, 3> input;
    input.emplace_back(real_t(-1), real_t(-2), real_t(-3));
    input.emplace_back(real_t(1), real_t(2), real_t(3));
    input.emplace_back(real_t(-1), real_t(-2), real_t(-3));  // duplicate
    input.emplace_back(real_t(0), real_t(0), real_t(0));

    auto result = tf::cleaned(input.points());

    REQUIRE(result.size() == 3);
}

// =============================================================================
// clean_points_very_close_but_distinct
// =============================================================================

TEMPLATE_TEST_CASE("clean_points_very_close_but_distinct", "[clean][clean_points]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using real_t = typename TestType::real_type;

    // Points just outside tolerance should remain distinct
    tf::points_buffer<real_t, 3> input;
    input.emplace_back(real_t(0), real_t(0), real_t(0));
    input.emplace_back(real_t(0.02), real_t(0), real_t(0));  // just outside 0.01 tolerance
    input.emplace_back(real_t(0.04), real_t(0), real_t(0));

    const real_t tolerance = real_t(0.01);
    auto result = tf::cleaned(input.points(), tolerance);

    // All points should remain (none within tolerance)
    REQUIRE(result.size() == 3);
}

// =============================================================================
// clean_points_index_map_consistency
// =============================================================================

TEMPLATE_TEST_CASE("clean_points_index_map_consistency", "[clean][clean_points]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    // Verify index map allows reconstructing merged positions
    tf::points_buffer<real_t, 3> input;
    input.emplace_back(real_t(0), real_t(0), real_t(0));
    input.emplace_back(real_t(1), real_t(1), real_t(1));
    input.emplace_back(real_t(0), real_t(0), real_t(0));  // duplicate of 0
    input.emplace_back(real_t(2), real_t(2), real_t(2));
    input.emplace_back(real_t(1), real_t(1), real_t(1));  // duplicate of 1

    auto [result, index_map] = tf::cleaned<index_t>(input.points(), tf::return_index_map);

    REQUIRE(result.size() == 3);
    REQUIRE(index_map.f().size() == 5);

    // Verify duplicates map to same index
    REQUIRE(index_map.f()[0] == index_map.f()[2]);
    REQUIRE(index_map.f()[1] == index_map.f()[4]);

    // Verify all indices are valid
    for (std::size_t i = 0; i < index_map.f().size(); ++i) {
        REQUIRE(static_cast<std::size_t>(index_map.f()[i]) < result.size());
    }

    // Verify mapped points match original positions
    for (std::size_t i = 0; i < input.size(); ++i) {
        auto mapped_idx = index_map.f()[i];
        const auto& original = input[i];
        const auto& mapped = result[mapped_idx];
        for (std::size_t d = 0; d < 3; ++d) {
            REQUIRE(std::abs(original[d] - mapped[d]) < real_t(1e-5));
        }
    }
}
