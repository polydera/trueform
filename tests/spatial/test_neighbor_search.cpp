/**
 * @file test_neighbor_search.cpp
 * @brief Tests for neighbor_search functionality on spatial forms
 *
 * Copyright (c) 2025 Ziga Sajovic, XLAB
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <trueform/trueform.hpp>
#include "type_traits.hpp"
#include "spatial_generators.hpp"
#include <limits>
#include <vector>
#include <algorithm>

// =============================================================================
// Point Cloud Neighbor Search (Form vs Point) - 3D
// =============================================================================

TEMPLATE_TEST_CASE("point_cloud_neighbor_search_3d", "[spatial][neighbor_search]",
    float, double)
{
    using real_t = TestType;

    // Create a 4x4x4 grid of points (64 points - large enough for tree traversal)
    auto cloud = tf::test::create_grid_points_3d<real_t>(4, 4, 4);

    tf::aabb_tree<int, real_t, 3> tree(cloud.points(), tf::config_tree(4, 4));
    auto cloud_with_tree = cloud.points() | tf::tag(tree);

    SECTION("nearest neighbor - brute force verification") {
        auto query = tf::make_point(real_t(0.1), real_t(0.2), real_t(0.3));
        auto result = tf::neighbor_search(cloud_with_tree, query);

        // Brute force
        real_t best = std::numeric_limits<real_t>::max();
        for (std::size_t i = 0; i < cloud.points().size(); ++i) {
            best = std::min(best, tf::distance2(cloud.points()[i], query));
        }
        REQUIRE(result);
        REQUIRE(result.metric() == Catch::Approx(best).margin(real_t(1e-5)));
    }

    SECTION("nearest neighbor - point at grid location") {
        auto query = tf::make_point(real_t(1), real_t(1), real_t(1));
        auto result = tf::neighbor_search(cloud_with_tree, query);

        REQUIRE(result);
        REQUIRE(result.metric() == Catch::Approx(real_t(0)).margin(real_t(1e-5)));
    }

    SECTION("kNN - results sorted by distance") {
        constexpr std::size_t k = 5;
        std::array<tf::nearest_neighbor<int, real_t, 3>, k> buffer;
        auto knn = tf::make_nearest_neighbors(buffer.begin(), k);
        auto query = tf::make_point(real_t(1.1), real_t(1.1), real_t(1.1));
        tf::neighbor_search(cloud_with_tree, query, knn);

        REQUIRE(knn.size() == k);
        for (std::size_t i = 0; i + 1 < knn.size(); ++i) {
            REQUIRE(knn.begin()[i].metric() <= knn.begin()[i + 1].metric());
        }
    }

    SECTION("kNN - brute force verification") {
        constexpr std::size_t k = 5;
        std::array<tf::nearest_neighbor<int, real_t, 3>, k> buffer;
        auto knn = tf::make_nearest_neighbors(buffer.begin(), k);
        auto query = tf::make_point(real_t(0.5), real_t(0.5), real_t(0.5));
        tf::neighbor_search(cloud_with_tree, query, knn);

        // Brute force: collect all distances and sort
        std::vector<real_t> all_dists;
        for (std::size_t i = 0; i < cloud.points().size(); ++i) {
            all_dists.push_back(tf::distance2(cloud.points()[i], query));
        }
        std::sort(all_dists.begin(), all_dists.end());

        REQUIRE(knn.size() == k);
        for (std::size_t i = 0; i < k; ++i) {
            REQUIRE(knn.begin()[i].metric() == Catch::Approx(all_dists[i]).margin(real_t(1e-5)));
        }
    }

    SECTION("with radius - point found within radius") {
        auto query = tf::make_point(real_t(0.1), real_t(0.1), real_t(0.1));
        auto result = tf::neighbor_search(cloud_with_tree, query, real_t(1.0));
        REQUIRE(result);
    }

    SECTION("with radius - nothing found outside radius") {
        auto query = tf::make_point(real_t(100), real_t(100), real_t(100));
        auto result = tf::neighbor_search(cloud_with_tree, query, real_t(1.0));
        REQUIRE_FALSE(result);
    }
}

// =============================================================================
// Point Cloud Neighbor Search - 2D
// =============================================================================

TEMPLATE_TEST_CASE("point_cloud_neighbor_search_2d", "[spatial][neighbor_search]",
    float, double)
{
    using real_t = TestType;

    // Create a 6x6 grid (36 points)
    auto cloud = tf::test::create_grid_points_2d<real_t>(6, 6);

    tf::aabb_tree<int, real_t, 2> tree(cloud.points(), tf::config_tree(4, 4));
    auto cloud_with_tree = cloud.points() | tf::tag(tree);

    SECTION("nearest neighbor - brute force verification") {
        auto query = tf::make_point(real_t(2.3), real_t(1.7));
        auto result = tf::neighbor_search(cloud_with_tree, query);

        // Brute force
        real_t best = std::numeric_limits<real_t>::max();
        for (std::size_t i = 0; i < cloud.points().size(); ++i) {
            best = std::min(best, tf::distance2(cloud.points()[i], query));
        }
        REQUIRE(result);
        REQUIRE(result.metric() == Catch::Approx(best).margin(real_t(1e-5)));
    }
}

// =============================================================================
// Mesh Neighbor Search (Form vs Point) - 3D
// =============================================================================

TEMPLATE_TEST_CASE("mesh_neighbor_search_point_3d", "[spatial][neighbor_search]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    // Create a 5x5 grid mesh (32 triangles)
    auto mesh = tf::test::create_grid_mesh_3d<index_t, real_t>(5, 5);

    tf::aabb_tree<index_t, real_t, 3> tree(mesh.polygons(), tf::config_tree(4, 4));
    auto mesh_with_tree = mesh.polygons() | tf::tag(tree);

    SECTION("nearest neighbor - brute force verification") {
        auto query = tf::make_point(real_t(2.3), real_t(1.7), real_t(1.0));
        auto result = tf::neighbor_search(mesh_with_tree, query);

        // Brute force
        real_t best = std::numeric_limits<real_t>::max();
        for (std::size_t i = 0; i < mesh.faces().size(); ++i) {
            best = std::min(best, tf::distance2(mesh.polygons()[i], query));
        }
        REQUIRE(result);
        REQUIRE(result.metric() == Catch::Approx(best).margin(real_t(1e-4)));
    }

    SECTION("nearest neighbor - point on mesh") {
        auto query = tf::make_point(real_t(2), real_t(2), real_t(0));
        auto result = tf::neighbor_search(mesh_with_tree, query);

        REQUIRE(result);
        REQUIRE(result.metric() == Catch::Approx(real_t(0)).margin(real_t(1e-5)));
    }

    SECTION("kNN from mesh") {
        constexpr std::size_t k = 5;
        std::array<tf::nearest_neighbor<index_t, real_t, 3>, k> buffer;
        auto knn = tf::make_nearest_neighbors(buffer.begin(), k);
        auto query = tf::make_point(real_t(2.5), real_t(2.5), real_t(0.5));
        tf::neighbor_search(mesh_with_tree, query, knn);

        REQUIRE(knn.size() == k);
        for (std::size_t i = 0; i + 1 < knn.size(); ++i) {
            REQUIRE(knn.begin()[i].metric() <= knn.begin()[i + 1].metric());
        }
    }
}

// =============================================================================
// Mesh Neighbor Search - 2D
// =============================================================================

TEMPLATE_TEST_CASE("mesh_neighbor_search_point_2d", "[spatial][neighbor_search]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    // Create a 5x5 grid mesh (32 triangles)
    auto mesh = tf::test::create_grid_mesh_2d<index_t, real_t>(5, 5);

    tf::aabb_tree<index_t, real_t, 2> tree(mesh.polygons(), tf::config_tree(4, 4));
    auto mesh_with_tree = mesh.polygons() | tf::tag(tree);

    SECTION("nearest neighbor - brute force verification") {
        auto query = tf::make_point(real_t(5), real_t(2.5));
        auto result = tf::neighbor_search(mesh_with_tree, query);

        // Brute force
        real_t best = std::numeric_limits<real_t>::max();
        for (std::size_t i = 0; i < mesh.faces().size(); ++i) {
            best = std::min(best, tf::distance2(mesh.polygons()[i], query));
        }
        REQUIRE(result);
        REQUIRE(result.metric() == Catch::Approx(best).margin(real_t(1e-5)));
    }
}

// =============================================================================
// Dynamic Mesh Neighbor Search - 3D
// =============================================================================

TEMPLATE_TEST_CASE("dynamic_mesh_neighbor_search_3d", "[spatial][neighbor_search]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    // Create a dynamic grid mesh (mixed quads and triangles)
    auto mesh = tf::test::create_dynamic_grid_mesh_3d<index_t, real_t>(5, 5);

    tf::aabb_tree<index_t, real_t, 3> tree(mesh.polygons(), tf::config_tree(4, 4));
    auto mesh_with_tree = mesh.polygons() | tf::tag(tree);

    SECTION("nearest neighbor - brute force verification") {
        auto query = tf::make_point(real_t(2.3), real_t(2.7), real_t(0.5));
        auto result = tf::neighbor_search(mesh_with_tree, query);

        // Brute force
        real_t best = std::numeric_limits<real_t>::max();
        for (std::size_t i = 0; i < mesh.faces().size(); ++i) {
            best = std::min(best, tf::distance2(mesh.polygons()[i], query));
        }
        REQUIRE(result);
        REQUIRE(result.metric() == Catch::Approx(best).margin(real_t(1e-4)));
    }
}

// =============================================================================
// Dynamic Mesh Neighbor Search - 2D
// =============================================================================

TEMPLATE_TEST_CASE("dynamic_mesh_neighbor_search_2d", "[spatial][neighbor_search]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto mesh = tf::test::create_dynamic_grid_mesh_2d<index_t, real_t>(5, 5);

    tf::aabb_tree<index_t, real_t, 2> tree(mesh.polygons(), tf::config_tree(4, 4));
    auto mesh_with_tree = mesh.polygons() | tf::tag(tree);

    SECTION("nearest neighbor - brute force verification") {
        auto query = tf::make_point(real_t(5), real_t(2.5));
        auto result = tf::neighbor_search(mesh_with_tree, query);

        // Brute force
        real_t best = std::numeric_limits<real_t>::max();
        for (std::size_t i = 0; i < mesh.faces().size(); ++i) {
            best = std::min(best, tf::distance2(mesh.polygons()[i], query));
        }
        REQUIRE(result);
        REQUIRE(result.metric() == Catch::Approx(best).margin(real_t(1e-5)));
    }
}

// =============================================================================
// Segments Neighbor Search - 3D
// =============================================================================

TEMPLATE_TEST_CASE("segments_neighbor_search_3d", "[spatial][neighbor_search]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    // Create a 5x5 grid of segments (40 segments)
    auto segments = tf::test::create_grid_segments_3d<index_t, real_t>(5, 5);

    tf::aabb_tree<index_t, real_t, 3> tree(segments.segments(), tf::config_tree(4, 4));
    auto segments_with_tree = segments.segments() | tf::tag(tree);

    SECTION("nearest neighbor - brute force verification") {
        auto query = tf::make_point(real_t(1.5), real_t(1.5), real_t(1.0));
        auto result = tf::neighbor_search(segments_with_tree, query);

        // Brute force
        real_t best = std::numeric_limits<real_t>::max();
        for (std::size_t i = 0; i < segments.edges().size(); ++i) {
            best = std::min(best, tf::distance2(segments.segments()[i], query));
        }
        REQUIRE(result);
        REQUIRE(result.metric() == Catch::Approx(best).margin(real_t(1e-5)));
    }
}

// =============================================================================
// Segments Neighbor Search - 2D
// =============================================================================

TEMPLATE_TEST_CASE("segments_neighbor_search_2d", "[spatial][neighbor_search]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto segments = tf::test::create_grid_segments_2d<index_t, real_t>(5, 5);

    tf::aabb_tree<index_t, real_t, 2> tree(segments.segments(), tf::config_tree(4, 4));
    auto segments_with_tree = segments.segments() | tf::tag(tree);

    SECTION("nearest neighbor - brute force verification") {
        auto query = tf::make_point(real_t(1.5), real_t(0.5));
        auto result = tf::neighbor_search(segments_with_tree, query);

        // Brute force
        real_t best = std::numeric_limits<real_t>::max();
        for (std::size_t i = 0; i < segments.edges().size(); ++i) {
            best = std::min(best, tf::distance2(segments.segments()[i], query));
        }
        REQUIRE(result);
        REQUIRE(result.metric() == Catch::Approx(best).margin(real_t(1e-5)));
    }
}

// =============================================================================
// Form vs Segment Query
// =============================================================================

TEMPLATE_TEST_CASE("mesh_neighbor_search_segment", "[spatial][neighbor_search]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto mesh = tf::test::create_grid_mesh_3d<index_t, real_t>(5, 5);

    tf::aabb_tree<index_t, real_t, 3> tree(mesh.polygons(), tf::config_tree(4, 4));
    auto mesh_with_tree = mesh.polygons() | tf::tag(tree);

    SECTION("segment above mesh") {
        auto seg = tf::make_segment_between_points(
            tf::make_point(real_t(1), real_t(1), real_t(2)),
            tf::make_point(real_t(3), real_t(3), real_t(2))
        );
        auto result = tf::neighbor_search(mesh_with_tree, seg);

        // Brute force
        real_t best = std::numeric_limits<real_t>::max();
        for (std::size_t i = 0; i < mesh.faces().size(); ++i) {
            best = std::min(best, tf::distance2(mesh.polygons()[i], seg));
        }
        REQUIRE(result);
        REQUIRE(result.metric() == Catch::Approx(best).margin(real_t(1e-4)));
    }
}

// =============================================================================
// Form vs Ray Query
// =============================================================================

TEMPLATE_TEST_CASE("mesh_neighbor_search_ray", "[spatial][neighbor_search]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto mesh = tf::test::create_grid_mesh_3d<index_t, real_t>(5, 5);

    tf::aabb_tree<index_t, real_t, 3> tree(mesh.polygons(), tf::config_tree(4, 4));
    auto mesh_with_tree = mesh.polygons() | tf::tag(tree);

    SECTION("ray passing beside mesh") {
        auto ray = tf::make_ray(
            tf::make_point(real_t(5), real_t(2), real_t(0)),
            tf::make_vector(real_t(0), real_t(1), real_t(0))
        );
        auto result = tf::neighbor_search(mesh_with_tree, ray);

        // Brute force
        real_t best = std::numeric_limits<real_t>::max();
        for (std::size_t i = 0; i < mesh.faces().size(); ++i) {
            best = std::min(best, tf::distance2(mesh.polygons()[i], ray));
        }
        REQUIRE(result);
        REQUIRE(result.metric() == Catch::Approx(best).margin(real_t(1e-4)));
    }
}

// =============================================================================
// Form vs Line Query
// =============================================================================

TEMPLATE_TEST_CASE("mesh_neighbor_search_line", "[spatial][neighbor_search]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto mesh = tf::test::create_grid_mesh_3d<index_t, real_t>(5, 5);

    tf::aabb_tree<index_t, real_t, 3> tree(mesh.polygons(), tf::config_tree(4, 4));
    auto mesh_with_tree = mesh.polygons() | tf::tag(tree);

    SECTION("line passing beside mesh") {
        auto line = tf::make_line_like(
            tf::make_point(real_t(5), real_t(2), real_t(0)),
            tf::make_vector(real_t(0), real_t(1), real_t(0))
        );
        auto result = tf::neighbor_search(mesh_with_tree, line);

        // Brute force
        real_t best = std::numeric_limits<real_t>::max();
        for (std::size_t i = 0; i < mesh.faces().size(); ++i) {
            best = std::min(best, tf::distance2(mesh.polygons()[i], line));
        }
        REQUIRE(result);
        REQUIRE(result.metric() == Catch::Approx(best).margin(real_t(1e-4)));
    }
}

// =============================================================================
// Form vs Polygon Query
// =============================================================================

TEMPLATE_TEST_CASE("mesh_neighbor_search_polygon", "[spatial][neighbor_search]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto mesh = tf::test::create_grid_mesh_3d<index_t, real_t>(5, 5);

    tf::aabb_tree<index_t, real_t, 3> tree(mesh.polygons(), tf::config_tree(4, 4));
    auto mesh_with_tree = mesh.polygons() | tf::tag(tree);

    SECTION("polygon above mesh") {
        std::array<tf::point<real_t, 3>, 3> pts = {{
            tf::make_point(real_t(1), real_t(1), real_t(2)),
            tf::make_point(real_t(3), real_t(1), real_t(2)),
            tf::make_point(real_t(2), real_t(3), real_t(2))
        }};
        auto poly = tf::make_polygon(pts);
        auto result = tf::neighbor_search(mesh_with_tree, poly);

        // Brute force
        real_t best = std::numeric_limits<real_t>::max();
        for (std::size_t i = 0; i < mesh.faces().size(); ++i) {
            best = std::min(best, tf::distance2(mesh.polygons()[i], poly));
        }
        REQUIRE(result);
        REQUIRE(result.metric() == Catch::Approx(best).margin(real_t(1e-4)));
    }
}

// =============================================================================
// Point Cloud vs Primitive Queries
// =============================================================================

TEMPLATE_TEST_CASE("point_cloud_neighbor_search_segment", "[spatial][neighbor_search]",
    float, double)
{
    using real_t = TestType;

    auto cloud = tf::test::create_grid_points_3d<real_t>(4, 4, 4);

    tf::aabb_tree<int, real_t, 3> tree(cloud.points(), tf::config_tree(4, 4));
    auto cloud_with_tree = cloud.points() | tf::tag(tree);

    SECTION("segment query") {
        auto seg = tf::make_segment_between_points(
            tf::make_point(real_t(5), real_t(1), real_t(1)),
            tf::make_point(real_t(5), real_t(2), real_t(2))
        );
        auto result = tf::neighbor_search(cloud_with_tree, seg);

        // Brute force
        real_t best = std::numeric_limits<real_t>::max();
        for (std::size_t i = 0; i < cloud.points().size(); ++i) {
            best = std::min(best, tf::distance2(cloud.points()[i], seg));
        }
        REQUIRE(result);
        REQUIRE(result.metric() == Catch::Approx(best).margin(real_t(1e-5)));
    }
}

TEMPLATE_TEST_CASE("point_cloud_neighbor_search_ray", "[spatial][neighbor_search]",
    float, double)
{
    using real_t = TestType;

    auto cloud = tf::test::create_grid_points_3d<real_t>(4, 4, 4);

    tf::aabb_tree<int, real_t, 3> tree(cloud.points(), tf::config_tree(4, 4));
    auto cloud_with_tree = cloud.points() | tf::tag(tree);

    SECTION("ray query") {
        auto ray = tf::make_ray(
            tf::make_point(real_t(5), real_t(1.5), real_t(1.5)),
            tf::make_vector(real_t(0), real_t(1), real_t(0))
        );
        auto result = tf::neighbor_search(cloud_with_tree, ray);

        // Brute force
        real_t best = std::numeric_limits<real_t>::max();
        for (std::size_t i = 0; i < cloud.points().size(); ++i) {
            best = std::min(best, tf::distance2(cloud.points()[i], ray));
        }
        REQUIRE(result);
        REQUIRE(result.metric() == Catch::Approx(best).margin(real_t(1e-5)));
    }
}

TEMPLATE_TEST_CASE("point_cloud_neighbor_search_line", "[spatial][neighbor_search]",
    float, double)
{
    using real_t = TestType;

    auto cloud = tf::test::create_grid_points_3d<real_t>(4, 4, 4);

    tf::aabb_tree<int, real_t, 3> tree(cloud.points(), tf::config_tree(4, 4));
    auto cloud_with_tree = cloud.points() | tf::tag(tree);

    SECTION("line query") {
        auto line = tf::make_line_like(
            tf::make_point(real_t(5), real_t(1.5), real_t(1.5)),
            tf::make_vector(real_t(0), real_t(1), real_t(0))
        );
        auto result = tf::neighbor_search(cloud_with_tree, line);

        // Brute force
        real_t best = std::numeric_limits<real_t>::max();
        for (std::size_t i = 0; i < cloud.points().size(); ++i) {
            best = std::min(best, tf::distance2(cloud.points()[i], line));
        }
        REQUIRE(result);
        REQUIRE(result.metric() == Catch::Approx(best).margin(real_t(1e-5)));
    }
}

// =============================================================================
// Segments vs Primitive Queries
// =============================================================================

TEMPLATE_TEST_CASE("segments_neighbor_search_segment_query", "[spatial][neighbor_search]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto segments = tf::test::create_grid_segments_3d<index_t, real_t>(5, 5);

    tf::aabb_tree<index_t, real_t, 3> tree(segments.segments(), tf::config_tree(4, 4));
    auto segments_with_tree = segments.segments() | tf::tag(tree);

    SECTION("segment query") {
        auto seg = tf::make_segment_between_points(
            tf::make_point(real_t(5), real_t(1), real_t(1)),
            tf::make_point(real_t(5), real_t(2), real_t(2))
        );
        auto result = tf::neighbor_search(segments_with_tree, seg);

        // Brute force
        real_t best = std::numeric_limits<real_t>::max();
        for (std::size_t i = 0; i < segments.edges().size(); ++i) {
            best = std::min(best, tf::distance2(segments.segments()[i], seg));
        }
        REQUIRE(result);
        REQUIRE(result.metric() == Catch::Approx(best).margin(real_t(1e-5)));
    }
}

TEMPLATE_TEST_CASE("segments_neighbor_search_ray_query", "[spatial][neighbor_search]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto segments = tf::test::create_grid_segments_3d<index_t, real_t>(5, 5);

    tf::aabb_tree<index_t, real_t, 3> tree(segments.segments(), tf::config_tree(4, 4));
    auto segments_with_tree = segments.segments() | tf::tag(tree);

    SECTION("ray query") {
        auto ray = tf::make_ray(
            tf::make_point(real_t(5), real_t(2), real_t(0)),
            tf::make_vector(real_t(0), real_t(1), real_t(0))
        );
        auto result = tf::neighbor_search(segments_with_tree, ray);

        // Brute force
        real_t best = std::numeric_limits<real_t>::max();
        for (std::size_t i = 0; i < segments.edges().size(); ++i) {
            best = std::min(best, tf::distance2(segments.segments()[i], ray));
        }
        REQUIRE(result);
        REQUIRE(result.metric() == Catch::Approx(best).margin(real_t(1e-5)));
    }
}

TEMPLATE_TEST_CASE("segments_neighbor_search_line_query", "[spatial][neighbor_search]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto segments = tf::test::create_grid_segments_3d<index_t, real_t>(5, 5);

    tf::aabb_tree<index_t, real_t, 3> tree(segments.segments(), tf::config_tree(4, 4));
    auto segments_with_tree = segments.segments() | tf::tag(tree);

    SECTION("line query") {
        auto line = tf::make_line_like(
            tf::make_point(real_t(5), real_t(2), real_t(0)),
            tf::make_vector(real_t(0), real_t(1), real_t(0))
        );
        auto result = tf::neighbor_search(segments_with_tree, line);

        // Brute force
        real_t best = std::numeric_limits<real_t>::max();
        for (std::size_t i = 0; i < segments.edges().size(); ++i) {
            best = std::min(best, tf::distance2(segments.segments()[i], line));
        }
        REQUIRE(result);
        REQUIRE(result.metric() == Catch::Approx(best).margin(real_t(1e-5)));
    }
}

// =============================================================================
// Form vs Form (Dual Tree) Neighbor Search
// =============================================================================

TEMPLATE_TEST_CASE("mesh_neighbor_search_mesh_3d", "[spatial][neighbor_search]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    // Create two grid meshes with some separation
    auto mesh0 = tf::test::create_grid_mesh_3d<index_t, real_t>(
        4, 4, {real_t(0), real_t(0), real_t(0)});
    auto mesh1 = tf::test::create_grid_mesh_3d<index_t, real_t>(
        4, 4, {real_t(5), real_t(0), real_t(0)});

    tf::aabb_tree<index_t, real_t, 3> tree0(mesh0.polygons(), tf::config_tree(4, 4));
    tf::aabb_tree<index_t, real_t, 3> tree1(mesh1.polygons(), tf::config_tree(4, 4));
    auto m0 = mesh0.polygons() | tf::tag(tree0);
    auto m1 = mesh1.polygons() | tf::tag(tree1);

    SECTION("form vs form nearest") {
        auto result = tf::neighbor_search(m0, m1);

        // Brute force
        real_t best = std::numeric_limits<real_t>::max();
        for (std::size_t i = 0; i < mesh0.faces().size(); ++i) {
            for (std::size_t j = 0; j < mesh1.faces().size(); ++j) {
                best = std::min(best, tf::distance2(mesh0.polygons()[i], mesh1.polygons()[j]));
            }
        }
        REQUIRE(result);
        REQUIRE(result.metric() == Catch::Approx(best).margin(real_t(1e-4)));
    }

    SECTION("form vs form with radius - found") {
        auto result = tf::neighbor_search(m0, m1, real_t(10.0));
        REQUIRE(result);
    }

    SECTION("form vs form with radius - not found") {
        auto result = tf::neighbor_search(m0, m1, real_t(0.5));
        REQUIRE_FALSE(result);
    }
}

// =============================================================================
// Point Cloud vs Point Cloud
// =============================================================================

TEMPLATE_TEST_CASE("point_cloud_neighbor_search_point_cloud", "[spatial][neighbor_search]",
    float, double)
{
    using real_t = TestType;

    // Create two point clouds with separation
    auto cloud0 = tf::test::create_grid_points_3d<real_t>(4, 4, 4);
    auto cloud1 = tf::test::create_grid_points_3d<real_t>(
        4, 4, 4, {real_t(5), real_t(0), real_t(0)});

    tf::aabb_tree<int, real_t, 3> tree0(cloud0.points(), tf::config_tree(4, 4));
    tf::aabb_tree<int, real_t, 3> tree1(cloud1.points(), tf::config_tree(4, 4));
    auto c0 = cloud0.points() | tf::tag(tree0);
    auto c1 = cloud1.points() | tf::tag(tree1);

    SECTION("form vs form nearest") {
        auto result = tf::neighbor_search(c0, c1);

        // Brute force
        real_t best = std::numeric_limits<real_t>::max();
        for (std::size_t i = 0; i < cloud0.points().size(); ++i) {
            for (std::size_t j = 0; j < cloud1.points().size(); ++j) {
                best = std::min(best, tf::distance2(cloud0.points()[i], cloud1.points()[j]));
            }
        }
        REQUIRE(result);
        REQUIRE(result.metric() == Catch::Approx(best).margin(real_t(1e-5)));
    }
}

// =============================================================================
// Segments vs Segments (Form vs Form)
// =============================================================================

TEMPLATE_TEST_CASE("segments_neighbor_search_segments", "[spatial][neighbor_search]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto seg0 = tf::test::create_grid_segments_3d<index_t, real_t>(4, 4);
    auto seg1 = tf::test::create_grid_segments_3d<index_t, real_t>(
        4, 4, {real_t(5), real_t(0), real_t(0)});

    tf::aabb_tree<index_t, real_t, 3> tree0(seg0.segments(), tf::config_tree(4, 4));
    tf::aabb_tree<index_t, real_t, 3> tree1(seg1.segments(), tf::config_tree(4, 4));
    auto s0 = seg0.segments() | tf::tag(tree0);
    auto s1 = seg1.segments() | tf::tag(tree1);

    SECTION("form vs form nearest") {
        auto result = tf::neighbor_search(s0, s1);

        // Brute force
        real_t best = std::numeric_limits<real_t>::max();
        for (std::size_t i = 0; i < seg0.edges().size(); ++i) {
            for (std::size_t j = 0; j < seg1.edges().size(); ++j) {
                best = std::min(best, tf::distance2(seg0.segments()[i], seg1.segments()[j]));
            }
        }
        REQUIRE(result);
        REQUIRE(result.metric() == Catch::Approx(best).margin(real_t(1e-5)));
    }
}

// =============================================================================
// Mixed Form vs Form: Mesh vs Segments
// =============================================================================

TEMPLATE_TEST_CASE("mesh_neighbor_search_segments", "[spatial][neighbor_search]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto mesh = tf::test::create_grid_mesh_3d<index_t, real_t>(4, 4);
    auto seg = tf::test::create_grid_segments_3d<index_t, real_t>(
        4, 4, {real_t(5), real_t(0), real_t(0)});

    tf::aabb_tree<index_t, real_t, 3> tree0(mesh.polygons(), tf::config_tree(4, 4));
    tf::aabb_tree<index_t, real_t, 3> tree1(seg.segments(), tf::config_tree(4, 4));
    auto m = mesh.polygons() | tf::tag(tree0);
    auto s = seg.segments() | tf::tag(tree1);

    SECTION("mesh vs segments nearest") {
        auto result = tf::neighbor_search(m, s);

        // Brute force
        real_t best = std::numeric_limits<real_t>::max();
        for (std::size_t i = 0; i < mesh.faces().size(); ++i) {
            for (std::size_t j = 0; j < seg.edges().size(); ++j) {
                best = std::min(best, tf::distance2(mesh.polygons()[i], seg.segments()[j]));
            }
        }
        REQUIRE(result);
        REQUIRE(result.metric() == Catch::Approx(best).margin(real_t(1e-4)));
    }
}

// =============================================================================
// Mixed Form vs Form: Mesh vs Point Cloud
// =============================================================================

TEMPLATE_TEST_CASE("mesh_neighbor_search_point_cloud", "[spatial][neighbor_search]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto mesh = tf::test::create_grid_mesh_3d<index_t, real_t>(4, 4);
    auto cloud = tf::test::create_grid_points_3d<real_t>(
        4, 4, 4, {real_t(5), real_t(0), real_t(0)});

    tf::aabb_tree<index_t, real_t, 3> tree0(mesh.polygons(), tf::config_tree(4, 4));
    tf::aabb_tree<int, real_t, 3> tree1(cloud.points(), tf::config_tree(4, 4));
    auto m = mesh.polygons() | tf::tag(tree0);
    auto c = cloud.points() | tf::tag(tree1);

    SECTION("mesh vs point cloud nearest") {
        auto result = tf::neighbor_search(m, c);

        // Brute force
        real_t best = std::numeric_limits<real_t>::max();
        for (std::size_t i = 0; i < mesh.faces().size(); ++i) {
            for (std::size_t j = 0; j < cloud.points().size(); ++j) {
                best = std::min(best, tf::distance2(mesh.polygons()[i], cloud.points()[j]));
            }
        }
        REQUIRE(result);
        REQUIRE(result.metric() == Catch::Approx(best).margin(real_t(1e-4)));
    }
}

// =============================================================================
// Mixed Form vs Form: Segments vs Point Cloud
// =============================================================================

TEMPLATE_TEST_CASE("segments_neighbor_search_point_cloud", "[spatial][neighbor_search]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto seg = tf::test::create_grid_segments_3d<index_t, real_t>(4, 4);
    auto cloud = tf::test::create_grid_points_3d<real_t>(
        4, 4, 4, {real_t(5), real_t(0), real_t(0)});

    tf::aabb_tree<index_t, real_t, 3> tree0(seg.segments(), tf::config_tree(4, 4));
    tf::aabb_tree<int, real_t, 3> tree1(cloud.points(), tf::config_tree(4, 4));
    auto s = seg.segments() | tf::tag(tree0);
    auto c = cloud.points() | tf::tag(tree1);

    SECTION("segments vs point cloud nearest") {
        auto result = tf::neighbor_search(s, c);

        // Brute force
        real_t best = std::numeric_limits<real_t>::max();
        for (std::size_t i = 0; i < seg.edges().size(); ++i) {
            for (std::size_t j = 0; j < cloud.points().size(); ++j) {
                best = std::min(best, tf::distance2(seg.segments()[i], cloud.points()[j]));
            }
        }
        REQUIRE(result);
        REQUIRE(result.metric() == Catch::Approx(best).margin(real_t(1e-5)));
    }
}

// =============================================================================
// kNN with Radius Limit
// =============================================================================

TEMPLATE_TEST_CASE("knn_with_radius_limit", "[spatial][neighbor_search]",
    float, double)
{
    using real_t = TestType;

    auto cloud = tf::test::create_grid_points_3d<real_t>(4, 4, 4);

    tf::aabb_tree<int, real_t, 3> tree(cloud.points(), tf::config_tree(4, 4));
    auto cloud_with_tree = cloud.points() | tf::tag(tree);

    SECTION("kNN within tight radius") {
        constexpr std::size_t k = 10;
        std::array<tf::nearest_neighbor<int, real_t, 3>, k> buffer;
        // Use radius of 1.5 - should only include nearby neighbors
        auto knn = tf::make_nearest_neighbors(buffer.begin(), k, real_t(1.5));
        auto query = tf::make_point(real_t(1), real_t(1), real_t(1));
        tf::neighbor_search(cloud_with_tree, query, knn);

        // Verify all results are within radius squared
        for (std::size_t i = 0; i < knn.size(); ++i) {
            REQUIRE(knn.begin()[i].metric() <= real_t(1.5 * 1.5));
        }
    }
}
