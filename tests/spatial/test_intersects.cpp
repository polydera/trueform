/**
 * @file test_intersects.cpp
 * @brief Tests for intersects functionality on spatial forms
 *
 * Copyright (c) 2025 Ziga Sajovic, XLAB
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <trueform/trueform.hpp>
#include "type_traits.hpp"
#include "spatial_generators.hpp"

// =============================================================================
// Mesh vs Point - 3D
// =============================================================================

TEMPLATE_TEST_CASE("mesh_intersects_point_3d", "[spatial][intersects]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto mesh = tf::test::create_grid_mesh_3d<index_t, real_t>(5, 5);

    tf::aabb_tree<index_t, real_t, 3> tree(mesh.polygons(), tf::config_tree(4, 4));
    auto mesh_with_tree = mesh.polygons() | tf::tag(tree);

    SECTION("point on mesh surface - hit") {
        auto pt = tf::make_point(real_t(2), real_t(2), real_t(0));
        REQUIRE(tf::intersects(mesh_with_tree, pt));
        REQUIRE(tf::intersects(pt, mesh_with_tree)); // symmetric
    }

    SECTION("point above mesh - miss") {
        auto pt = tf::make_point(real_t(2), real_t(2), real_t(1));
        REQUIRE_FALSE(tf::intersects(mesh_with_tree, pt));
    }

    SECTION("point outside mesh bounds - miss") {
        auto pt = tf::make_point(real_t(10), real_t(10), real_t(0));
        REQUIRE_FALSE(tf::intersects(mesh_with_tree, pt));
    }

    SECTION("brute force verification") {
        auto pt = tf::make_point(real_t(1.5), real_t(1.5), real_t(0));
        auto result = tf::intersects(mesh_with_tree, pt);

        // Brute force
        bool expected = false;
        for (std::size_t i = 0; i < mesh.faces().size(); ++i) {
            if (tf::intersects(mesh.polygons()[i], pt)) {
                expected = true;
                break;
            }
        }
        REQUIRE(result == expected);
    }
}

// =============================================================================
// Mesh vs Point - 2D
// =============================================================================

TEMPLATE_TEST_CASE("mesh_intersects_point_2d", "[spatial][intersects]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto mesh = tf::test::create_grid_mesh_2d<index_t, real_t>(5, 5);

    tf::aabb_tree<index_t, real_t, 2> tree(mesh.polygons(), tf::config_tree(4, 4));
    auto mesh_with_tree = mesh.polygons() | tf::tag(tree);

    SECTION("point inside mesh - hit") {
        auto pt = tf::make_point(real_t(1.5), real_t(1.5));
        REQUIRE(tf::intersects(mesh_with_tree, pt));
    }

    SECTION("point outside mesh - miss") {
        auto pt = tf::make_point(real_t(10), real_t(10));
        REQUIRE_FALSE(tf::intersects(mesh_with_tree, pt));
    }
}

// =============================================================================
// Mesh vs Segment - 3D
// =============================================================================

TEMPLATE_TEST_CASE("mesh_intersects_segment_3d", "[spatial][intersects]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto mesh = tf::test::create_grid_mesh_3d<index_t, real_t>(5, 5);

    tf::aabb_tree<index_t, real_t, 3> tree(mesh.polygons(), tf::config_tree(4, 4));
    auto mesh_with_tree = mesh.polygons() | tf::tag(tree);

    SECTION("segment through mesh - hit") {
        auto seg = tf::make_segment_between_points(
            tf::make_point(real_t(2), real_t(2), real_t(-1)),
            tf::make_point(real_t(2), real_t(2), real_t(1))
        );
        REQUIRE(tf::intersects(mesh_with_tree, seg));
        REQUIRE(tf::intersects(seg, mesh_with_tree)); // symmetric
    }

    SECTION("segment above mesh - miss") {
        auto seg = tf::make_segment_between_points(
            tf::make_point(real_t(2), real_t(2), real_t(1)),
            tf::make_point(real_t(2), real_t(2), real_t(2))
        );
        REQUIRE_FALSE(tf::intersects(mesh_with_tree, seg));
    }

    SECTION("segment on mesh surface - hit") {
        auto seg = tf::make_segment_between_points(
            tf::make_point(real_t(1), real_t(1), real_t(0)),
            tf::make_point(real_t(2), real_t(2), real_t(0))
        );
        REQUIRE(tf::intersects(mesh_with_tree, seg));
    }

    SECTION("brute force verification") {
        auto seg = tf::make_segment_between_points(
            tf::make_point(real_t(1.5), real_t(1.5), real_t(-0.5)),
            tf::make_point(real_t(1.5), real_t(1.5), real_t(0.5))
        );
        auto result = tf::intersects(mesh_with_tree, seg);

        // Brute force
        bool expected = false;
        for (std::size_t i = 0; i < mesh.faces().size(); ++i) {
            if (tf::intersects(mesh.polygons()[i], seg)) {
                expected = true;
                break;
            }
        }
        REQUIRE(result == expected);
    }
}

// =============================================================================
// Mesh vs Segment - 2D
// =============================================================================

TEMPLATE_TEST_CASE("mesh_intersects_segment_2d", "[spatial][intersects]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto mesh = tf::test::create_grid_mesh_2d<index_t, real_t>(5, 5);

    tf::aabb_tree<index_t, real_t, 2> tree(mesh.polygons(), tf::config_tree(4, 4));
    auto mesh_with_tree = mesh.polygons() | tf::tag(tree);

    SECTION("segment through mesh - hit") {
        auto seg = tf::make_segment_between_points(
            tf::make_point(real_t(-1), real_t(2)),
            tf::make_point(real_t(5), real_t(2))
        );
        REQUIRE(tf::intersects(mesh_with_tree, seg));
    }

    SECTION("segment outside mesh - miss") {
        auto seg = tf::make_segment_between_points(
            tf::make_point(real_t(10), real_t(10)),
            tf::make_point(real_t(11), real_t(11))
        );
        REQUIRE_FALSE(tf::intersects(mesh_with_tree, seg));
    }
}

// =============================================================================
// Mesh vs Ray - 3D
// =============================================================================

TEMPLATE_TEST_CASE("mesh_intersects_ray_3d", "[spatial][intersects]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto mesh = tf::test::create_grid_mesh_3d<index_t, real_t>(5, 5);

    tf::aabb_tree<index_t, real_t, 3> tree(mesh.polygons(), tf::config_tree(4, 4));
    auto mesh_with_tree = mesh.polygons() | tf::tag(tree);

    SECTION("ray hitting mesh - hit") {
        auto ray = tf::make_ray(
            tf::make_point(real_t(2), real_t(2), real_t(5)),
            tf::make_vector(real_t(0), real_t(0), real_t(-1))
        );
        REQUIRE(tf::intersects(mesh_with_tree, ray));
        REQUIRE(tf::intersects(ray, mesh_with_tree)); // symmetric
    }

    SECTION("ray pointing away - miss") {
        auto ray = tf::make_ray(
            tf::make_point(real_t(2), real_t(2), real_t(5)),
            tf::make_vector(real_t(0), real_t(0), real_t(1))
        );
        REQUIRE_FALSE(tf::intersects(mesh_with_tree, ray));
    }

    SECTION("ray missing mesh - miss") {
        auto ray = tf::make_ray(
            tf::make_point(real_t(10), real_t(10), real_t(5)),
            tf::make_vector(real_t(0), real_t(0), real_t(-1))
        );
        REQUIRE_FALSE(tf::intersects(mesh_with_tree, ray));
    }
}

// =============================================================================
// Mesh vs Ray - 2D
// =============================================================================

TEMPLATE_TEST_CASE("mesh_intersects_ray_2d", "[spatial][intersects]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto mesh = tf::test::create_grid_mesh_2d<index_t, real_t>(5, 5);

    tf::aabb_tree<index_t, real_t, 2> tree(mesh.polygons(), tf::config_tree(4, 4));
    auto mesh_with_tree = mesh.polygons() | tf::tag(tree);

    SECTION("ray hitting mesh") {
        auto ray = tf::make_ray(
            tf::make_point(real_t(-1), real_t(2)),
            tf::make_vector(real_t(1), real_t(0))
        );
        REQUIRE(tf::intersects(mesh_with_tree, ray));
    }

    SECTION("ray missing mesh") {
        auto ray = tf::make_ray(
            tf::make_point(real_t(-1), real_t(10)),
            tf::make_vector(real_t(1), real_t(0))
        );
        REQUIRE_FALSE(tf::intersects(mesh_with_tree, ray));
    }
}

// =============================================================================
// Mesh vs Line - 3D
// =============================================================================

TEMPLATE_TEST_CASE("mesh_intersects_line_3d", "[spatial][intersects]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto mesh = tf::test::create_grid_mesh_3d<index_t, real_t>(5, 5);

    tf::aabb_tree<index_t, real_t, 3> tree(mesh.polygons(), tf::config_tree(4, 4));
    auto mesh_with_tree = mesh.polygons() | tf::tag(tree);

    SECTION("line through mesh - brute force verification") {
        auto line = tf::make_line_like(
            tf::make_point(real_t(2), real_t(2), real_t(5)),
            tf::make_vector(real_t(0), real_t(0), real_t(1))
        );
        auto result = tf::intersects(mesh_with_tree, line);

        // Brute force
        bool expected = false;
        for (std::size_t i = 0; i < mesh.faces().size(); ++i) {
            if (tf::intersects(mesh.polygons()[i], line)) {
                expected = true;
                break;
            }
        }
        REQUIRE(result == expected);
    }

    SECTION("line missing mesh - brute force verification") {
        auto line = tf::make_line_like(
            tf::make_point(real_t(10), real_t(10), real_t(5)),
            tf::make_vector(real_t(0), real_t(0), real_t(1))
        );
        auto result = tf::intersects(mesh_with_tree, line);

        // Brute force
        bool expected = false;
        for (std::size_t i = 0; i < mesh.faces().size(); ++i) {
            if (tf::intersects(mesh.polygons()[i], line)) {
                expected = true;
                break;
            }
        }
        REQUIRE(result == expected);
    }
}

// =============================================================================
// Mesh vs Polygon - 3D
// =============================================================================

TEMPLATE_TEST_CASE("mesh_intersects_polygon_3d", "[spatial][intersects]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto mesh = tf::test::create_grid_mesh_3d<index_t, real_t>(5, 5);

    tf::aabb_tree<index_t, real_t, 3> tree(mesh.polygons(), tf::config_tree(4, 4));
    auto mesh_with_tree = mesh.polygons() | tf::tag(tree);

    SECTION("polygon crossing mesh - hit") {
        std::array<tf::point<real_t, 3>, 3> pts = {{
            tf::make_point(real_t(2), real_t(2), real_t(-1)),
            tf::make_point(real_t(3), real_t(2), real_t(1)),
            tf::make_point(real_t(2), real_t(3), real_t(1))
        }};
        auto poly = tf::make_polygon(pts);
        REQUIRE(tf::intersects(mesh_with_tree, poly));
        REQUIRE(tf::intersects(poly, mesh_with_tree)); // symmetric
    }

    SECTION("polygon above mesh - miss") {
        std::array<tf::point<real_t, 3>, 3> pts = {{
            tf::make_point(real_t(2), real_t(2), real_t(1)),
            tf::make_point(real_t(3), real_t(2), real_t(1)),
            tf::make_point(real_t(2), real_t(3), real_t(1))
        }};
        auto poly = tf::make_polygon(pts);
        REQUIRE_FALSE(tf::intersects(mesh_with_tree, poly));
    }

    SECTION("brute force verification") {
        std::array<tf::point<real_t, 3>, 3> pts = {{
            tf::make_point(real_t(1.5), real_t(1.5), real_t(-0.5)),
            tf::make_point(real_t(2.5), real_t(1.5), real_t(0.5)),
            tf::make_point(real_t(2), real_t(2.5), real_t(0))
        }};
        auto poly = tf::make_polygon(pts);
        auto result = tf::intersects(mesh_with_tree, poly);

        // Brute force
        bool expected = false;
        for (std::size_t i = 0; i < mesh.faces().size(); ++i) {
            if (tf::intersects(mesh.polygons()[i], poly)) {
                expected = true;
                break;
            }
        }
        REQUIRE(result == expected);
    }
}

// =============================================================================
// Mesh vs AABB - 3D
// =============================================================================

TEMPLATE_TEST_CASE("mesh_intersects_aabb_3d", "[spatial][intersects]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto mesh = tf::test::create_grid_mesh_3d<index_t, real_t>(5, 5);

    tf::aabb_tree<index_t, real_t, 3> tree(mesh.polygons(), tf::config_tree(4, 4));
    auto mesh_with_tree = mesh.polygons() | tf::tag(tree);

    SECTION("aabb crossing mesh surface") {
        auto aabb = tf::make_aabb(
            tf::make_point(real_t(1), real_t(1), real_t(-1)),
            tf::make_point(real_t(3), real_t(3), real_t(1))
        );
        auto result = tf::intersects(mesh_with_tree, aabb);

        // Brute force
        bool expected = false;
        for (std::size_t i = 0; i < mesh.faces().size(); ++i) {
            if (tf::intersects(mesh.polygons()[i], aabb)) {
                expected = true;
                break;
            }
        }
        REQUIRE(result == expected);
        REQUIRE(tf::intersects(aabb, mesh_with_tree) == result); // symmetric
    }

    SECTION("aabb above mesh surface") {
        auto aabb = tf::make_aabb(
            tf::make_point(real_t(1), real_t(1), real_t(1)),
            tf::make_point(real_t(3), real_t(3), real_t(2))
        );
        auto result = tf::intersects(mesh_with_tree, aabb);

        // Brute force
        bool expected = false;
        for (std::size_t i = 0; i < mesh.faces().size(); ++i) {
            if (tf::intersects(mesh.polygons()[i], aabb)) {
                expected = true;
                break;
            }
        }
        REQUIRE(result == expected);
    }

    SECTION("aabb outside mesh bounds") {
        auto aabb = tf::make_aabb(
            tf::make_point(real_t(10), real_t(10), real_t(-1)),
            tf::make_point(real_t(12), real_t(12), real_t(1))
        );
        auto result = tf::intersects(mesh_with_tree, aabb);

        // Brute force
        bool expected = false;
        for (std::size_t i = 0; i < mesh.faces().size(); ++i) {
            if (tf::intersects(mesh.polygons()[i], aabb)) {
                expected = true;
                break;
            }
        }
        REQUIRE(result == expected);
    }
}

// =============================================================================
// Dynamic Mesh Intersects - 3D
// =============================================================================

TEMPLATE_TEST_CASE("dynamic_mesh_intersects_3d", "[spatial][intersects]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto mesh = tf::test::create_dynamic_grid_mesh_3d<index_t, real_t>(5, 5);

    tf::aabb_tree<index_t, real_t, 3> tree(mesh.polygons(), tf::config_tree(4, 4));
    auto mesh_with_tree = mesh.polygons() | tf::tag(tree);

    SECTION("point on mesh - hit") {
        auto pt = tf::make_point(real_t(2), real_t(2), real_t(0));
        REQUIRE(tf::intersects(mesh_with_tree, pt));
    }

    SECTION("segment through mesh - hit") {
        auto seg = tf::make_segment_between_points(
            tf::make_point(real_t(2), real_t(2), real_t(-1)),
            tf::make_point(real_t(2), real_t(2), real_t(1))
        );
        REQUIRE(tf::intersects(mesh_with_tree, seg));
    }

    SECTION("ray hitting mesh - hit") {
        auto ray = tf::make_ray(
            tf::make_point(real_t(2), real_t(2), real_t(5)),
            tf::make_vector(real_t(0), real_t(0), real_t(-1))
        );
        REQUIRE(tf::intersects(mesh_with_tree, ray));
    }
}

// =============================================================================
// Dynamic Mesh Intersects - 2D
// =============================================================================

TEMPLATE_TEST_CASE("dynamic_mesh_intersects_2d", "[spatial][intersects]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto mesh = tf::test::create_dynamic_grid_mesh_2d<index_t, real_t>(5, 5);

    tf::aabb_tree<index_t, real_t, 2> tree(mesh.polygons(), tf::config_tree(4, 4));
    auto mesh_with_tree = mesh.polygons() | tf::tag(tree);

    SECTION("point inside mesh - hit") {
        auto pt = tf::make_point(real_t(2.5), real_t(2.5));
        REQUIRE(tf::intersects(mesh_with_tree, pt));
    }

    SECTION("segment through mesh - hit") {
        auto seg = tf::make_segment_between_points(
            tf::make_point(real_t(-1), real_t(2)),
            tf::make_point(real_t(5), real_t(2))
        );
        REQUIRE(tf::intersects(mesh_with_tree, seg));
    }
}

// =============================================================================
// Static Quad Mesh Intersects - 3D
// =============================================================================

TEMPLATE_TEST_CASE("quad_mesh_intersects_3d", "[spatial][intersects]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    // Create a simple quad mesh at z=0
    tf::polygons_buffer<index_t, real_t, 3, 4> mesh;
    for (int j = 0; j < 3; ++j) {
        for (int i = 0; i < 3; ++i) {
            mesh.points_buffer().emplace_back(real_t(i), real_t(j), real_t(0));
        }
    }
    mesh.faces_buffer().emplace_back(index_t(0), index_t(1), index_t(4), index_t(3));
    mesh.faces_buffer().emplace_back(index_t(1), index_t(2), index_t(5), index_t(4));
    mesh.faces_buffer().emplace_back(index_t(3), index_t(4), index_t(7), index_t(6));
    mesh.faces_buffer().emplace_back(index_t(4), index_t(5), index_t(8), index_t(7));

    tf::aabb_tree<index_t, real_t, 3> tree(mesh.polygons(), tf::config_tree(4, 4));
    auto mesh_with_tree = mesh.polygons() | tf::tag(tree);

    SECTION("point on mesh - hit") {
        auto pt = tf::make_point(real_t(0.5), real_t(0.5), real_t(0));
        REQUIRE(tf::intersects(mesh_with_tree, pt));
    }

    SECTION("segment through mesh - hit") {
        auto seg = tf::make_segment_between_points(
            tf::make_point(real_t(0.5), real_t(0.5), real_t(-1)),
            tf::make_point(real_t(0.5), real_t(0.5), real_t(1))
        );
        REQUIRE(tf::intersects(mesh_with_tree, seg));
    }

    SECTION("ray hitting mesh - hit") {
        auto ray = tf::make_ray(
            tf::make_point(real_t(1), real_t(1), real_t(5)),
            tf::make_vector(real_t(0), real_t(0), real_t(-1))
        );
        REQUIRE(tf::intersects(mesh_with_tree, ray));
    }
}

// =============================================================================
// Segments Intersects - 3D
// =============================================================================

TEMPLATE_TEST_CASE("segments_intersects_3d", "[spatial][intersects]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto segments = tf::test::create_grid_segments_3d<index_t, real_t>(5, 5);

    tf::aabb_tree<index_t, real_t, 3> tree(segments.segments(), tf::config_tree(4, 4));
    auto segments_with_tree = segments.segments() | tf::tag(tree);

    SECTION("point on segment - hit") {
        auto pt = tf::make_point(real_t(0.5), real_t(0), real_t(0));
        REQUIRE(tf::intersects(segments_with_tree, pt));
    }

    SECTION("point off segments - miss") {
        auto pt = tf::make_point(real_t(0.5), real_t(0.5), real_t(1));
        REQUIRE_FALSE(tf::intersects(segments_with_tree, pt));
    }

    SECTION("segment crossing - hit") {
        auto seg = tf::make_segment_between_points(
            tf::make_point(real_t(0.5), real_t(-1), real_t(0)),
            tf::make_point(real_t(0.5), real_t(1), real_t(0))
        );
        REQUIRE(tf::intersects(segments_with_tree, seg));
    }

    SECTION("ray hitting segment - hit") {
        auto ray = tf::make_ray(
            tf::make_point(real_t(0.5), real_t(0), real_t(-1)),
            tf::make_vector(real_t(0), real_t(0), real_t(1))
        );
        REQUIRE(tf::intersects(segments_with_tree, ray));
    }
}

// =============================================================================
// Segments Intersects - 2D
// =============================================================================

TEMPLATE_TEST_CASE("segments_intersects_2d", "[spatial][intersects]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto segments = tf::test::create_grid_segments_2d<index_t, real_t>(5, 5);

    tf::aabb_tree<index_t, real_t, 2> tree(segments.segments(), tf::config_tree(4, 4));
    auto segments_with_tree = segments.segments() | tf::tag(tree);

    SECTION("point on segment - hit") {
        auto pt = tf::make_point(real_t(0.5), real_t(0));
        REQUIRE(tf::intersects(segments_with_tree, pt));
    }

    SECTION("segment crossing - hit") {
        auto seg = tf::make_segment_between_points(
            tf::make_point(real_t(0.5), real_t(-1)),
            tf::make_point(real_t(0.5), real_t(1))
        );
        REQUIRE(tf::intersects(segments_with_tree, seg));
    }

    SECTION("ray hitting segments - hit") {
        auto ray = tf::make_ray(
            tf::make_point(real_t(-1), real_t(0)),
            tf::make_vector(real_t(1), real_t(0))
        );
        REQUIRE(tf::intersects(segments_with_tree, ray));
    }
}

// =============================================================================
// Point Cloud Intersects - 3D
// =============================================================================

TEMPLATE_TEST_CASE("point_cloud_intersects_3d", "[spatial][intersects]",
    float, double)
{
    using real_t = TestType;

    auto cloud = tf::test::create_grid_points_3d<real_t>(4, 4, 4);

    tf::aabb_tree<int, real_t, 3> tree(cloud.points(), tf::config_tree(4, 4));
    auto cloud_with_tree = cloud.points() | tf::tag(tree);

    SECTION("point at grid location - hit") {
        auto pt = tf::make_point(real_t(1), real_t(1), real_t(1));
        REQUIRE(tf::intersects(cloud_with_tree, pt));
    }

    SECTION("point off grid - miss") {
        auto pt = tf::make_point(real_t(0.5), real_t(0.5), real_t(0.5));
        REQUIRE_FALSE(tf::intersects(cloud_with_tree, pt));
    }

    SECTION("segment through point - hit") {
        auto seg = tf::make_segment_between_points(
            tf::make_point(real_t(1), real_t(1), real_t(-1)),
            tf::make_point(real_t(1), real_t(1), real_t(5))
        );
        REQUIRE(tf::intersects(cloud_with_tree, seg));
    }

    SECTION("ray through point - hit") {
        auto ray = tf::make_ray(
            tf::make_point(real_t(1), real_t(1), real_t(-1)),
            tf::make_vector(real_t(0), real_t(0), real_t(1))
        );
        REQUIRE(tf::intersects(cloud_with_tree, ray));
    }
}

// =============================================================================
// Point Cloud Intersects - 2D
// =============================================================================

TEMPLATE_TEST_CASE("point_cloud_intersects_2d", "[spatial][intersects]",
    float, double)
{
    using real_t = TestType;

    auto cloud = tf::test::create_grid_points_2d<real_t>(5, 5);

    tf::aabb_tree<int, real_t, 2> tree(cloud.points(), tf::config_tree(4, 4));
    auto cloud_with_tree = cloud.points() | tf::tag(tree);

    SECTION("point at grid location - hit") {
        auto pt = tf::make_point(real_t(2), real_t(2));
        REQUIRE(tf::intersects(cloud_with_tree, pt));
    }

    SECTION("point off grid - miss") {
        auto pt = tf::make_point(real_t(0.5), real_t(0.5));
        REQUIRE_FALSE(tf::intersects(cloud_with_tree, pt));
    }

    SECTION("segment through points - hit") {
        auto seg = tf::make_segment_between_points(
            tf::make_point(real_t(-1), real_t(2)),
            tf::make_point(real_t(5), real_t(2))
        );
        REQUIRE(tf::intersects(cloud_with_tree, seg));
    }
}

// =============================================================================
// Form vs Form Intersects - Mesh vs Mesh
// =============================================================================

TEMPLATE_TEST_CASE("mesh_intersects_mesh_3d", "[spatial][intersects]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    SECTION("overlapping meshes - hit") {
        auto mesh0 = tf::test::create_grid_mesh_3d<index_t, real_t>(4, 4);
        auto mesh1 = tf::test::create_grid_mesh_3d<index_t, real_t>(
            4, 4, {real_t(1), real_t(1), real_t(0)});

        tf::aabb_tree<index_t, real_t, 3> tree0(mesh0.polygons(), tf::config_tree(4, 4));
        tf::aabb_tree<index_t, real_t, 3> tree1(mesh1.polygons(), tf::config_tree(4, 4));
        auto m0 = mesh0.polygons() | tf::tag(tree0);
        auto m1 = mesh1.polygons() | tf::tag(tree1);

        REQUIRE(tf::intersects(m0, m1));
        REQUIRE(tf::intersects(m1, m0)); // symmetric
    }

    SECTION("separated meshes - miss") {
        auto mesh0 = tf::test::create_grid_mesh_3d<index_t, real_t>(4, 4);
        auto mesh1 = tf::test::create_grid_mesh_3d<index_t, real_t>(
            4, 4, {real_t(10), real_t(0), real_t(0)});

        tf::aabb_tree<index_t, real_t, 3> tree0(mesh0.polygons(), tf::config_tree(4, 4));
        tf::aabb_tree<index_t, real_t, 3> tree1(mesh1.polygons(), tf::config_tree(4, 4));
        auto m0 = mesh0.polygons() | tf::tag(tree0);
        auto m1 = mesh1.polygons() | tf::tag(tree1);

        REQUIRE_FALSE(tf::intersects(m0, m1));
    }

    SECTION("self-intersecting mesh - brute force") {
        auto mesh0 = tf::test::create_grid_mesh_3d<index_t, real_t>(4, 4);
        // Mesh at different z to cross through mesh0
        auto mesh1 = tf::test::create_grid_mesh_3d<index_t, real_t>(
            4, 4, {real_t(1.5), real_t(1.5), real_t(-0.5)});
        // Tilt mesh1 by modifying z coordinates
        for (std::size_t i = 0; i < mesh1.points().size(); ++i) {
            mesh1.points()[i][2] += mesh1.points()[i][0] * real_t(0.5);
        }

        tf::aabb_tree<index_t, real_t, 3> tree0(mesh0.polygons(), tf::config_tree(4, 4));
        tf::aabb_tree<index_t, real_t, 3> tree1(mesh1.polygons(), tf::config_tree(4, 4));
        auto m0 = mesh0.polygons() | tf::tag(tree0);
        auto m1 = mesh1.polygons() | tf::tag(tree1);

        auto result = tf::intersects(m0, m1);

        // Brute force
        bool expected = false;
        for (std::size_t i = 0; i < mesh0.faces().size() && !expected; ++i) {
            for (std::size_t j = 0; j < mesh1.faces().size() && !expected; ++j) {
                if (tf::intersects(mesh0.polygons()[i], mesh1.polygons()[j])) {
                    expected = true;
                }
            }
        }
        REQUIRE(result == expected);
    }
}

// =============================================================================
// Form vs Form Intersects - Segments vs Segments
// =============================================================================

TEMPLATE_TEST_CASE("segments_intersects_segments_3d", "[spatial][intersects]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    SECTION("overlapping segment grids - hit") {
        auto seg0 = tf::test::create_grid_segments_3d<index_t, real_t>(4, 4);
        auto seg1 = tf::test::create_grid_segments_3d<index_t, real_t>(
            4, 4, {real_t(0.5), real_t(0.5), real_t(0)});

        tf::aabb_tree<index_t, real_t, 3> tree0(seg0.segments(), tf::config_tree(4, 4));
        tf::aabb_tree<index_t, real_t, 3> tree1(seg1.segments(), tf::config_tree(4, 4));
        auto s0 = seg0.segments() | tf::tag(tree0);
        auto s1 = seg1.segments() | tf::tag(tree1);

        REQUIRE(tf::intersects(s0, s1));
    }

    SECTION("separated segment grids - miss") {
        auto seg0 = tf::test::create_grid_segments_3d<index_t, real_t>(4, 4);
        auto seg1 = tf::test::create_grid_segments_3d<index_t, real_t>(
            4, 4, {real_t(10), real_t(0), real_t(0)});

        tf::aabb_tree<index_t, real_t, 3> tree0(seg0.segments(), tf::config_tree(4, 4));
        tf::aabb_tree<index_t, real_t, 3> tree1(seg1.segments(), tf::config_tree(4, 4));
        auto s0 = seg0.segments() | tf::tag(tree0);
        auto s1 = seg1.segments() | tf::tag(tree1);

        REQUIRE_FALSE(tf::intersects(s0, s1));
    }
}

// =============================================================================
// Form vs Form Intersects - Mesh vs Segments
// =============================================================================

TEMPLATE_TEST_CASE("mesh_intersects_segments_3d", "[spatial][intersects]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto mesh = tf::test::create_grid_mesh_3d<index_t, real_t>(4, 4);

    SECTION("segments on mesh - hit") {
        auto seg = tf::test::create_grid_segments_3d<index_t, real_t>(
            4, 4, {real_t(0), real_t(0), real_t(0)});

        tf::aabb_tree<index_t, real_t, 3> tree0(mesh.polygons(), tf::config_tree(4, 4));
        tf::aabb_tree<index_t, real_t, 3> tree1(seg.segments(), tf::config_tree(4, 4));
        auto m = mesh.polygons() | tf::tag(tree0);
        auto s = seg.segments() | tf::tag(tree1);

        REQUIRE(tf::intersects(m, s));
        REQUIRE(tf::intersects(s, m)); // symmetric
    }

    SECTION("segments off mesh - miss") {
        auto seg = tf::test::create_grid_segments_3d<index_t, real_t>(
            4, 4, {real_t(10), real_t(0), real_t(0)});

        tf::aabb_tree<index_t, real_t, 3> tree0(mesh.polygons(), tf::config_tree(4, 4));
        tf::aabb_tree<index_t, real_t, 3> tree1(seg.segments(), tf::config_tree(4, 4));
        auto m = mesh.polygons() | tf::tag(tree0);
        auto s = seg.segments() | tf::tag(tree1);

        REQUIRE_FALSE(tf::intersects(m, s));
    }
}

// =============================================================================
// Form vs Form Intersects - Point Cloud vs Point Cloud
// =============================================================================

TEMPLATE_TEST_CASE("point_cloud_intersects_point_cloud_3d", "[spatial][intersects]",
    float, double)
{
    using real_t = TestType;

    SECTION("overlapping point clouds - hit") {
        auto cloud0 = tf::test::create_grid_points_3d<real_t>(4, 4, 4);
        auto cloud1 = tf::test::create_grid_points_3d<real_t>(
            4, 4, 4, {real_t(1), real_t(1), real_t(1)});

        tf::aabb_tree<int, real_t, 3> tree0(cloud0.points(), tf::config_tree(4, 4));
        tf::aabb_tree<int, real_t, 3> tree1(cloud1.points(), tf::config_tree(4, 4));
        auto c0 = cloud0.points() | tf::tag(tree0);
        auto c1 = cloud1.points() | tf::tag(tree1);

        REQUIRE(tf::intersects(c0, c1));
    }

    SECTION("separated point clouds - miss") {
        auto cloud0 = tf::test::create_grid_points_3d<real_t>(4, 4, 4);
        auto cloud1 = tf::test::create_grid_points_3d<real_t>(
            4, 4, 4, {real_t(10), real_t(0), real_t(0)});

        tf::aabb_tree<int, real_t, 3> tree0(cloud0.points(), tf::config_tree(4, 4));
        tf::aabb_tree<int, real_t, 3> tree1(cloud1.points(), tf::config_tree(4, 4));
        auto c0 = cloud0.points() | tf::tag(tree0);
        auto c1 = cloud1.points() | tf::tag(tree1);

        REQUIRE_FALSE(tf::intersects(c0, c1));
    }
}
