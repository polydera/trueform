/**
 * @file test_gather_ids.cpp
 * @brief Tests for gather_ids and gather_self_ids functionality
 *
 * Copyright (c) 2025 Ziga Sajovic, XLAB
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <trueform/trueform.hpp>
#include "type_traits.hpp"
#include "spatial_generators.hpp"
#include <algorithm>
#include <set>

// =============================================================================
// gather_ids - Single Form with AABB query
// =============================================================================

TEMPLATE_TEST_CASE("mesh_gather_ids_aabb_3d", "[spatial][gather_ids]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto mesh = tf::test::create_grid_mesh_3d<index_t, real_t>(5, 5);

    tf::aabb_tree<index_t, real_t, 3> tree(mesh.polygons(), tf::config_tree(4, 4));
    auto mesh_with_tree = mesh.polygons() | tf::tag(tree);

    SECTION("gather ids intersecting aabb - brute force verification") {
        auto query_aabb = tf::make_aabb(
            tf::make_point(real_t(1), real_t(1), real_t(-1)),
            tf::make_point(real_t(3), real_t(3), real_t(1))
        );

        std::vector<index_t> ids;
        tf::gather_ids(mesh_with_tree,
            [&](const auto &bv) { return tf::intersects(bv, query_aabb); },
            [&](const auto &prim) { return tf::intersects(prim, query_aabb); },
            std::back_inserter(ids));

        // Brute force
        std::vector<index_t> expected;
        for (std::size_t i = 0; i < mesh.faces().size(); ++i) {
            if (tf::intersects(mesh.polygons()[i], query_aabb)) {
                expected.push_back(static_cast<index_t>(i));
            }
        }

        std::sort(ids.begin(), ids.end());
        std::sort(expected.begin(), expected.end());
        REQUIRE(ids.size() == expected.size());
        REQUIRE(ids == expected);
    }

    SECTION("gather ids with no matches") {
        auto query_aabb = tf::make_aabb(
            tf::make_point(real_t(10), real_t(10), real_t(-1)),
            tf::make_point(real_t(12), real_t(12), real_t(1))
        );

        std::vector<index_t> ids;
        tf::gather_ids(mesh_with_tree,
            [&](const auto &bv) { return tf::intersects(bv, query_aabb); },
            [&](const auto &prim) { return tf::intersects(prim, query_aabb); },
            std::back_inserter(ids));

        // Brute force
        std::vector<index_t> expected;
        for (std::size_t i = 0; i < mesh.faces().size(); ++i) {
            if (tf::intersects(mesh.polygons()[i], query_aabb)) {
                expected.push_back(static_cast<index_t>(i));
            }
        }

        REQUIRE(ids.empty());
        REQUIRE(expected.empty());
    }
}

// =============================================================================
// gather_ids - Single Form with point query
// =============================================================================

TEMPLATE_TEST_CASE("mesh_gather_ids_point_3d", "[spatial][gather_ids]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto mesh = tf::test::create_grid_mesh_3d<index_t, real_t>(5, 5);

    tf::aabb_tree<index_t, real_t, 3> tree(mesh.polygons(), tf::config_tree(4, 4));
    auto mesh_with_tree = mesh.polygons() | tf::tag(tree);

    SECTION("gather ids containing point - brute force verification") {
        auto pt = tf::make_point(real_t(2), real_t(2), real_t(0));

        std::vector<index_t> ids;
        tf::gather_ids(mesh_with_tree,
            [&](const auto &bv) { return tf::intersects(bv, pt); },
            [&](const auto &prim) { return tf::intersects(prim, pt); },
            std::back_inserter(ids));

        // Brute force
        std::vector<index_t> expected;
        for (std::size_t i = 0; i < mesh.faces().size(); ++i) {
            if (tf::intersects(mesh.polygons()[i], pt)) {
                expected.push_back(static_cast<index_t>(i));
            }
        }

        std::sort(ids.begin(), ids.end());
        std::sort(expected.begin(), expected.end());
        REQUIRE(ids.size() == expected.size());
        REQUIRE(ids == expected);
    }
}

// =============================================================================
// gather_ids - Single Form with distance predicate
// =============================================================================

TEMPLATE_TEST_CASE("mesh_gather_ids_within_distance_3d", "[spatial][gather_ids]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto mesh = tf::test::create_grid_mesh_3d<index_t, real_t>(5, 5);

    tf::aabb_tree<index_t, real_t, 3> tree(mesh.polygons(), tf::config_tree(4, 4));
    auto mesh_with_tree = mesh.polygons() | tf::tag(tree);

    SECTION("gather ids within distance - brute force verification") {
        auto pt = tf::make_point(real_t(2), real_t(2), real_t(0.5));
        auto dist2 = real_t(1.0);

        std::vector<index_t> ids;
        tf::gather_ids(mesh_with_tree,
            [&](const auto &bv) { return tf::distance2(bv, pt) <= dist2; },
            [&](const auto &prim) { return tf::distance2(prim, pt) <= dist2; },
            std::back_inserter(ids));

        // Brute force
        std::vector<index_t> expected;
        for (std::size_t i = 0; i < mesh.faces().size(); ++i) {
            if (tf::distance2(mesh.polygons()[i], pt) <= dist2) {
                expected.push_back(static_cast<index_t>(i));
            }
        }

        std::sort(ids.begin(), ids.end());
        std::sort(expected.begin(), expected.end());
        REQUIRE(ids.size() == expected.size());
        REQUIRE(ids == expected);
    }
}

// =============================================================================
// gather_ids - Form vs Form (intersecting pairs)
// =============================================================================

TEMPLATE_TEST_CASE("mesh_gather_ids_form_form_3d", "[spatial][gather_ids]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto mesh0 = tf::test::create_grid_mesh_3d<index_t, real_t>(4, 4);
    auto mesh1 = tf::test::create_grid_mesh_3d<index_t, real_t>(
        4, 4, {real_t(1.5), real_t(1.5), real_t(0)});

    tf::aabb_tree<index_t, real_t, 3> tree0(mesh0.polygons(), tf::config_tree(4, 4));
    tf::aabb_tree<index_t, real_t, 3> tree1(mesh1.polygons(), tf::config_tree(4, 4));
    auto m0 = mesh0.polygons() | tf::tag(tree0);
    auto m1 = mesh1.polygons() | tf::tag(tree1);

    SECTION("gather intersecting pairs - brute force verification") {
        std::vector<std::pair<index_t, index_t>> pairs;
        tf::gather_ids(m0, m1, tf::intersects_f, tf::intersects_f, std::back_inserter(pairs));

        // Brute force
        std::set<std::pair<index_t, index_t>> expected;
        for (std::size_t i = 0; i < mesh0.faces().size(); ++i) {
            for (std::size_t j = 0; j < mesh1.faces().size(); ++j) {
                if (tf::intersects(mesh0.polygons()[i], mesh1.polygons()[j])) {
                    expected.insert({static_cast<index_t>(i), static_cast<index_t>(j)});
                }
            }
        }

        std::set<std::pair<index_t, index_t>> result_set(pairs.begin(), pairs.end());
        REQUIRE(result_set.size() == expected.size());
        REQUIRE(result_set == expected);
    }
}

// =============================================================================
// gather_ids - Segments
// =============================================================================

TEMPLATE_TEST_CASE("segments_gather_ids_3d", "[spatial][gather_ids]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto segments = tf::test::create_grid_segments_3d<index_t, real_t>(5, 5);

    tf::aabb_tree<index_t, real_t, 3> tree(segments.segments(), tf::config_tree(4, 4));
    auto segments_with_tree = segments.segments() | tf::tag(tree);

    SECTION("gather ids intersecting aabb - brute force verification") {
        auto query_aabb = tf::make_aabb(
            tf::make_point(real_t(0), real_t(0), real_t(-1)),
            tf::make_point(real_t(2), real_t(2), real_t(1))
        );

        std::vector<index_t> ids;
        tf::gather_ids(segments_with_tree,
            [&](const auto &bv) { return tf::intersects(bv, query_aabb); },
            [&](const auto &prim) { return tf::intersects(prim, query_aabb); },
            std::back_inserter(ids));

        // Brute force
        std::vector<index_t> expected;
        for (std::size_t i = 0; i < segments.edges().size(); ++i) {
            if (tf::intersects(segments.segments()[i], query_aabb)) {
                expected.push_back(static_cast<index_t>(i));
            }
        }

        std::sort(ids.begin(), ids.end());
        std::sort(expected.begin(), expected.end());
        REQUIRE(ids.size() == expected.size());
        REQUIRE(ids == expected);
    }
}

// =============================================================================
// gather_ids - Point Cloud within distance
// =============================================================================

TEMPLATE_TEST_CASE("point_cloud_gather_ids_3d", "[spatial][gather_ids]",
    float, double)
{
    using real_t = TestType;

    auto cloud = tf::test::create_grid_points_3d<real_t>(4, 4, 4);

    tf::aabb_tree<int, real_t, 3> tree(cloud.points(), tf::config_tree(4, 4));
    auto cloud_with_tree = cloud.points() | tf::tag(tree);

    SECTION("gather ids within distance - brute force verification") {
        auto pt = tf::make_point(real_t(1.5), real_t(1.5), real_t(1.5));
        auto dist2 = real_t(2.0);

        std::vector<int> ids;
        tf::gather_ids(cloud_with_tree,
            [&](const auto &bv) { return tf::distance2(bv, pt) <= dist2; },
            [&](const auto &prim) { return tf::distance2(prim, pt) <= dist2; },
            std::back_inserter(ids));

        // Brute force
        std::vector<int> expected;
        for (std::size_t i = 0; i < cloud.size(); ++i) {
            if (tf::distance2(cloud.points()[i], pt) <= dist2) {
                expected.push_back(static_cast<int>(i));
            }
        }

        std::sort(ids.begin(), ids.end());
        std::sort(expected.begin(), expected.end());
        REQUIRE(ids.size() == expected.size());
        REQUIRE(ids == expected);
    }
}

// =============================================================================
// gather_self_ids - Point Cloud (close points)
// =============================================================================

TEMPLATE_TEST_CASE("point_cloud_gather_self_ids_3d", "[spatial][gather_self_ids]",
    float, double)
{
    using real_t = TestType;

    auto cloud = tf::test::create_points_with_duplicates_3d<real_t>();

    tf::aabb_tree<int, real_t, 3> tree(cloud.points(), tf::config_tree(4, 4));
    auto cloud_with_tree = cloud.points() | tf::tag(tree);

    SECTION("find close pairs - brute force verification") {
        auto tolerance2 = real_t(0.01 * 0.01);

        std::vector<std::pair<int, int>> pairs;
        tf::gather_self_ids(cloud_with_tree,
            [=](const auto &a, const auto &b) { return tf::distance2(a, b) <= tolerance2; },
            std::back_inserter(pairs));

        // Brute force
        std::set<std::pair<int, int>> expected;
        for (std::size_t i = 0; i < cloud.size(); ++i) {
            for (std::size_t j = i + 1; j < cloud.size(); ++j) {
                if (tf::distance2(cloud.points()[i], cloud.points()[j]) <= tolerance2) {
                    expected.insert({static_cast<int>(i), static_cast<int>(j)});
                }
            }
        }

        // Normalize pairs (smaller id first)
        std::set<std::pair<int, int>> result_set;
        for (auto [i, j] : pairs) {
            if (i > j) std::swap(i, j);
            result_set.insert({i, j});
        }

        REQUIRE(result_set.size() == expected.size());
        REQUIRE(result_set == expected);
    }
}

// =============================================================================
// gather_self_ids - Mesh (self-intersections)
// =============================================================================

TEMPLATE_TEST_CASE("mesh_gather_self_ids_intersecting_3d", "[spatial][gather_self_ids]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto mesh = tf::test::create_self_intersecting_mesh_3d<index_t, real_t>();

    tf::aabb_tree<index_t, real_t, 3> tree(mesh.polygons(), tf::config_tree(4, 4));
    auto mesh_with_tree = mesh.polygons() | tf::tag(tree);

    SECTION("find self-intersections - brute force verification") {
        std::vector<std::pair<index_t, index_t>> pairs;
        tf::gather_self_ids(mesh_with_tree, tf::intersects_f, std::back_inserter(pairs));

        // Brute force
        std::set<std::pair<index_t, index_t>> expected;
        for (std::size_t i = 0; i < mesh.faces().size(); ++i) {
            for (std::size_t j = i + 1; j < mesh.faces().size(); ++j) {
                if (tf::intersects(mesh.polygons()[i], mesh.polygons()[j])) {
                    expected.insert({static_cast<index_t>(i), static_cast<index_t>(j)});
                }
            }
        }

        // Normalize pairs (smaller id first)
        std::set<std::pair<index_t, index_t>> result_set;
        for (auto [i, j] : pairs) {
            if (i > j) std::swap(i, j);
            result_set.insert({i, j});
        }

        REQUIRE(result_set.size() == expected.size());
        REQUIRE(result_set == expected);
    }
}

// =============================================================================
// gather_self_ids - Segments (crossing segments)
// =============================================================================

TEMPLATE_TEST_CASE("segments_gather_self_ids_3d", "[spatial][gather_self_ids]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto segments = tf::test::create_grid_segments_3d<index_t, real_t>(4, 4);

    tf::aabb_tree<index_t, real_t, 3> tree(segments.segments(), tf::config_tree(4, 4));
    auto segments_with_tree = segments.segments() | tf::tag(tree);

    SECTION("find intersecting segments - brute force verification") {
        std::vector<std::pair<index_t, index_t>> pairs;
        tf::gather_self_ids(segments_with_tree, tf::intersects_f, std::back_inserter(pairs));

        // Brute force
        std::set<std::pair<index_t, index_t>> expected;
        for (std::size_t i = 0; i < segments.edges().size(); ++i) {
            for (std::size_t j = i + 1; j < segments.edges().size(); ++j) {
                if (tf::intersects(segments.segments()[i], segments.segments()[j])) {
                    expected.insert({static_cast<index_t>(i), static_cast<index_t>(j)});
                }
            }
        }

        // Normalize pairs (smaller id first)
        std::set<std::pair<index_t, index_t>> result_set;
        for (auto [i, j] : pairs) {
            if (i > j) std::swap(i, j);
            result_set.insert({i, j});
        }

        REQUIRE(result_set.size() == expected.size());
        REQUIRE(result_set == expected);
    }
}

// =============================================================================
// gather_ids - 2D tests
// =============================================================================

TEMPLATE_TEST_CASE("mesh_gather_ids_2d", "[spatial][gather_ids]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto mesh = tf::test::create_grid_mesh_2d<index_t, real_t>(5, 5);

    tf::aabb_tree<index_t, real_t, 2> tree(mesh.polygons(), tf::config_tree(4, 4));
    auto mesh_with_tree = mesh.polygons() | tf::tag(tree);

    SECTION("gather ids intersecting aabb - brute force verification") {
        auto query_aabb = tf::make_aabb(
            tf::make_point(real_t(1), real_t(1)),
            tf::make_point(real_t(3), real_t(3))
        );

        std::vector<index_t> ids;
        tf::gather_ids(mesh_with_tree,
            [&](const auto &bv) { return tf::intersects(bv, query_aabb); },
            [&](const auto &prim) { return tf::intersects(prim, query_aabb); },
            std::back_inserter(ids));

        // Brute force
        std::vector<index_t> expected;
        for (std::size_t i = 0; i < mesh.faces().size(); ++i) {
            if (tf::intersects(mesh.polygons()[i], query_aabb)) {
                expected.push_back(static_cast<index_t>(i));
            }
        }

        std::sort(ids.begin(), ids.end());
        std::sort(expected.begin(), expected.end());
        REQUIRE(ids.size() == expected.size());
        REQUIRE(ids == expected);
    }
}

TEMPLATE_TEST_CASE("point_cloud_gather_self_ids_2d", "[spatial][gather_self_ids]",
    float, double)
{
    using real_t = TestType;

    auto cloud = tf::test::create_grid_points_2d<real_t>(5, 5);

    tf::aabb_tree<int, real_t, 2> tree(cloud.points(), tf::config_tree(4, 4));
    auto cloud_with_tree = cloud.points() | tf::tag(tree);

    SECTION("find close pairs - brute force verification") {
        // Points are at integer positions, so distance2 = 1 for adjacent points
        auto tolerance2 = real_t(1.01);

        std::vector<std::pair<int, int>> pairs;
        tf::gather_self_ids(cloud_with_tree,
            [=](const auto &a, const auto &b) { return tf::distance2(a, b) <= tolerance2; },
            std::back_inserter(pairs));

        // Brute force
        std::set<std::pair<int, int>> expected;
        for (std::size_t i = 0; i < cloud.size(); ++i) {
            for (std::size_t j = i + 1; j < cloud.size(); ++j) {
                if (tf::distance2(cloud.points()[i], cloud.points()[j]) <= tolerance2) {
                    expected.insert({static_cast<int>(i), static_cast<int>(j)});
                }
            }
        }

        // Normalize pairs (smaller id first)
        std::set<std::pair<int, int>> result_set;
        for (auto [i, j] : pairs) {
            if (i > j) std::swap(i, j);
            result_set.insert({i, j});
        }

        REQUIRE(result_set.size() == expected.size());
        REQUIRE(result_set == expected);
    }
}
