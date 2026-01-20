/**
 * @file test_clean_curves.cpp
 * @brief Tests for tf::cleaned(curves, ...)
 *
 * Tests for:
 * - clean_curves_no_duplicates
 * - clean_curves_duplicate_points
 * - clean_curves_degenerate_edges
 * - clean_curves_tolerance
 * - clean_curves_reconnection
 * - clean_curves_with_index_map
 * - clean_curves_empty
 *
 * Copyright (c) 2025 Ziga Sajovic, XLAB
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <trueform/trueform.hpp>
#include "type_traits.hpp"

namespace {

/// Helper to create a curves_buffer with a single path
template <typename Index, typename Real, std::size_t Dims>
auto make_single_path_curves(const std::vector<std::array<Real, Dims>>& points,
                             const std::vector<Index>& path_indices)
    -> tf::curves_buffer<Index, Real, Dims>
{
    tf::curves_buffer<Index, Real, Dims> result;

    // Add points
    for (const auto& pt : points) {
        result.points_buffer().push_back(pt);
    }

    // Add single path
    result.paths_buffer().push_back(tf::make_range(path_indices));

    return result;
}

/// Helper to create a curves_buffer with multiple paths
template <typename Index, typename Real, std::size_t Dims>
auto make_multi_path_curves(const std::vector<std::array<Real, Dims>>& points,
                            const std::vector<std::vector<Index>>& paths)
    -> tf::curves_buffer<Index, Real, Dims>
{
    tf::curves_buffer<Index, Real, Dims> result;

    // Add points
    for (const auto& pt : points) {
        result.points_buffer().push_back(pt);
    }

    // Add paths
    for (const auto& path : paths) {
        result.paths_buffer().push_back(tf::make_range(path));
    }

    return result;
}

} // anonymous namespace

// =============================================================================
// clean_curves_no_duplicates
// =============================================================================

TEMPLATE_TEST_CASE("clean_curves_no_duplicates", "[clean][clean_curves]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    // Create curves with no duplicates
    std::vector<std::array<real_t, 3>> points = {
        {real_t(0), real_t(0), real_t(0)},
        {real_t(1), real_t(0), real_t(0)},
        {real_t(2), real_t(0), real_t(0)},
        {real_t(3), real_t(0), real_t(0)}
    };
    std::vector<index_t> path = {0, 1, 2, 3};

    auto input = make_single_path_curves<index_t, real_t, 3>(points, path);

    auto result = tf::cleaned(input.curves());

    // Should remain unchanged
    REQUIRE(result.points().size() == 4);
    REQUIRE(result.paths().size() == 1);
}

// =============================================================================
// clean_curves_duplicate_points
// =============================================================================

TEMPLATE_TEST_CASE("clean_curves_duplicate_points", "[clean][clean_curves]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    // Create curves with duplicate points
    std::vector<std::array<real_t, 3>> points = {
        {real_t(0), real_t(0), real_t(0)},
        {real_t(1), real_t(0), real_t(0)},
        {real_t(0), real_t(0), real_t(0)},  // duplicate of 0
        {real_t(2), real_t(0), real_t(0)}
    };
    // Path: 0 -> 1 -> 2 -> 3
    // After merging: 0 -> 1 -> 0 -> 2 (new indices)
    std::vector<index_t> path = {0, 1, 2, 3};

    auto input = make_single_path_curves<index_t, real_t, 3>(points, path);

    auto result = tf::cleaned(input.curves());

    // Points 0 and 2 should merge
    REQUIRE(result.points().size() == 3);
}

// =============================================================================
// clean_curves_degenerate_edges
// =============================================================================

TEMPLATE_TEST_CASE("clean_curves_degenerate_edges", "[clean][clean_curves]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    // Create curves with degenerate edges (same point repeated)
    std::vector<std::array<real_t, 3>> points = {
        {real_t(0), real_t(0), real_t(0)},
        {real_t(1), real_t(0), real_t(0)},
        {real_t(2), real_t(0), real_t(0)}
    };
    // Path with degenerate edge: 0 -> 1 -> 1 -> 2
    std::vector<index_t> path = {0, 1, 1, 2};

    auto input = make_single_path_curves<index_t, real_t, 3>(points, path);

    auto result = tf::cleaned(input.curves());

    // Degenerate edge should be removed
    // The path should be reconnected properly
    REQUIRE(result.points().size() >= 2);  // At least start and end
}

// =============================================================================
// clean_curves_tolerance
// =============================================================================

TEMPLATE_TEST_CASE("clean_curves_tolerance", "[clean][clean_curves]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    // Create curves with points within tolerance
    std::vector<std::array<real_t, 3>> points = {
        {real_t(0), real_t(0), real_t(0)},
        {real_t(1), real_t(0), real_t(0)},
        {real_t(0.001), real_t(0), real_t(0)},  // within tolerance of 0
        {real_t(2), real_t(0), real_t(0)}
    };
    std::vector<index_t> path = {0, 1, 2, 3};

    auto input = make_single_path_curves<index_t, real_t, 3>(points, path);

    const real_t tolerance = real_t(0.01);
    auto result = tf::cleaned(input.curves(), tolerance);

    // Points 0 and 2 should merge
    REQUIRE(result.points().size() == 3);
}

// =============================================================================
// clean_curves_reconnection
// =============================================================================

TEMPLATE_TEST_CASE("clean_curves_reconnection", "[clean][clean_curves]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    // Create multiple separate paths that should remain separate after cleaning
    std::vector<std::array<real_t, 3>> points = {
        {real_t(0), real_t(0), real_t(0)},
        {real_t(1), real_t(0), real_t(0)},
        {real_t(10), real_t(0), real_t(0)},
        {real_t(11), real_t(0), real_t(0)}
    };
    std::vector<std::vector<index_t>> paths = {
        {0, 1},  // first path
        {2, 3}   // second path (separate)
    };

    auto input = make_multi_path_curves<index_t, real_t, 3>(points, paths);

    auto result = tf::cleaned(input.curves());

    // Paths should remain separate (2 disconnected paths)
    REQUIRE(result.paths().size() == 2);
}

// =============================================================================
// clean_curves_with_index_map
// =============================================================================

TEMPLATE_TEST_CASE("clean_curves_with_index_map", "[clean][clean_curves]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    // Create curves with duplicate points
    std::vector<std::array<real_t, 3>> points = {
        {real_t(0), real_t(0), real_t(0)},
        {real_t(1), real_t(0), real_t(0)},
        {real_t(0), real_t(0), real_t(0)},  // duplicate of 0
        {real_t(2), real_t(0), real_t(0)}
    };
    std::vector<index_t> path = {0, 1, 2, 3};

    auto input = make_single_path_curves<index_t, real_t, 3>(points, path);

    auto [result, point_im] = tf::cleaned(input.curves(), tf::return_index_map);

    // Points 0 and 2 should merge
    REQUIRE(result.points().size() == 3);

    // Point index map should show duplicate mapping
    REQUIRE(point_im.f().size() == 4);
    REQUIRE(point_im.f()[0] == point_im.f()[2]);
}

// =============================================================================
// clean_curves_with_index_map_tolerance
// =============================================================================

TEMPLATE_TEST_CASE("clean_curves_with_index_map_tolerance", "[clean][clean_curves]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    std::vector<std::array<real_t, 3>> points = {
        {real_t(0), real_t(0), real_t(0)},
        {real_t(0.001), real_t(0), real_t(0)},  // within tolerance of 0
        {real_t(1), real_t(0), real_t(0)}
    };
    std::vector<index_t> path = {0, 1, 2};

    auto input = make_single_path_curves<index_t, real_t, 3>(points, path);

    const real_t tolerance = real_t(0.01);
    auto [result, point_im] = tf::cleaned(input.curves(), tolerance, tf::return_index_map);

    // Points 0 and 1 should merge
    REQUIRE(result.points().size() == 2);
    REQUIRE(point_im.f()[0] == point_im.f()[1]);
}

// =============================================================================
// clean_curves_empty
// =============================================================================

TEMPLATE_TEST_CASE("clean_curves_empty", "[clean][clean_curves]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    tf::curves_buffer<index_t, real_t, 3> input;

    auto result = tf::cleaned(input.curves());

    REQUIRE(result.points().size() == 0);
    REQUIRE(result.paths().size() == 0);
}

// =============================================================================
// clean_curves_single_point_path
// =============================================================================

TEMPLATE_TEST_CASE("clean_curves_single_point_path", "[clean][clean_curves]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    // Path with single point (no edges)
    std::vector<std::array<real_t, 3>> points = {
        {real_t(0), real_t(0), real_t(0)}
    };
    std::vector<index_t> path = {0};

    auto input = make_single_path_curves<index_t, real_t, 3>(points, path);

    auto result = tf::cleaned(input.curves());

    // Single point path has no edges, should result in empty curves
    REQUIRE(result.paths().size() == 0);
}

// =============================================================================
// clean_curves_two_point_path
// =============================================================================

TEMPLATE_TEST_CASE("clean_curves_two_point_path", "[clean][clean_curves]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    // Minimal valid path with two points
    std::vector<std::array<real_t, 3>> points = {
        {real_t(0), real_t(0), real_t(0)},
        {real_t(1), real_t(0), real_t(0)}
    };
    std::vector<index_t> path = {0, 1};

    auto input = make_single_path_curves<index_t, real_t, 3>(points, path);

    auto result = tf::cleaned(input.curves());

    // Should have one path with one edge
    REQUIRE(result.paths().size() == 1);
    REQUIRE(result.points().size() == 2);
}

// =============================================================================
// clean_curves_2d
// =============================================================================

TEMPLATE_TEST_CASE("clean_curves_2d", "[clean][clean_curves]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    // Test with 2D curves
    std::vector<std::array<real_t, 2>> points = {
        {real_t(0), real_t(0)},
        {real_t(1), real_t(0)},
        {real_t(0), real_t(0)},  // duplicate of 0
        {real_t(2), real_t(0)}
    };
    std::vector<index_t> path = {0, 1, 2, 3};

    auto input = make_single_path_curves<index_t, real_t, 2>(points, path);

    auto result = tf::cleaned(input.curves());

    // Points 0 and 2 should merge
    REQUIRE(result.points().size() == 3);
}

// =============================================================================
// clean_curves_closed_loop
// =============================================================================

TEMPLATE_TEST_CASE("clean_curves_closed_loop", "[clean][clean_curves]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    // Create a closed loop (returns to start)
    std::vector<std::array<real_t, 3>> points = {
        {real_t(0), real_t(0), real_t(0)},
        {real_t(1), real_t(0), real_t(0)},
        {real_t(1), real_t(1), real_t(0)},
        {real_t(0), real_t(1), real_t(0)}
    };
    // Closed loop: 0 -> 1 -> 2 -> 3 -> 0
    std::vector<index_t> path = {0, 1, 2, 3, 0};

    auto input = make_single_path_curves<index_t, real_t, 3>(points, path);

    auto result = tf::cleaned(input.curves());

    // Should preserve all points and the closed loop structure
    REQUIRE(result.points().size() == 4);
    REQUIRE(result.paths().size() >= 1);
}

// =============================================================================
// clean_curves_merge_creates_connection
// =============================================================================

TEMPLATE_TEST_CASE("clean_curves_merge_creates_connection", "[clean][clean_curves]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    // Two paths that become connected after point merging
    std::vector<std::array<real_t, 3>> points = {
        {real_t(0), real_t(0), real_t(0)},      // path 1 start
        {real_t(1), real_t(0), real_t(0)},      // path 1 end
        {real_t(1.001), real_t(0), real_t(0)},  // path 2 start (within tolerance of 1)
        {real_t(2), real_t(0), real_t(0)}       // path 2 end
    };
    std::vector<std::vector<index_t>> paths = {
        {0, 1},
        {2, 3}
    };

    auto input = make_multi_path_curves<index_t, real_t, 3>(points, paths);

    const real_t tolerance = real_t(0.01);
    auto result = tf::cleaned(input.curves(), tolerance);

    // Points 1 and 2 should merge, potentially connecting the paths
    REQUIRE(result.points().size() == 3);
}
