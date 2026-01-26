/**
 * @file test_ray_cast.cpp
 * @brief Tests for ray_cast functionality on spatial forms
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

// =============================================================================
// Mesh Ray Cast - 3D
// =============================================================================

TEMPLATE_TEST_CASE("mesh_ray_cast_3d", "[spatial][ray_cast]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    // Create a 5x5 grid mesh at z=0 (32 triangles)
    auto mesh = tf::test::create_grid_mesh_3d<index_t, real_t>(5, 5);

    tf::aabb_tree<index_t, real_t, 3> tree(mesh.polygons(), tf::config_tree(4, 4));
    auto mesh_with_tree = mesh.polygons() | tf::tag(tree);

    SECTION("ray hit - from above") {
        auto ray = tf::make_ray(
            tf::make_point(real_t(2), real_t(2), real_t(5)),
            tf::make_vector(real_t(0), real_t(0), real_t(-1))
        );
        auto result = tf::ray_cast(ray, mesh_with_tree);

        REQUIRE(result);
        REQUIRE(result.info.t == Catch::Approx(real_t(5)).margin(real_t(1e-5)));

        // Brute force verification
        real_t best_t = std::numeric_limits<real_t>::max();
        for (std::size_t i = 0; i < mesh.faces().size(); ++i) {
            auto hit = tf::ray_cast(ray, mesh.polygons()[i]);
            if (hit && hit.t < best_t) {
                best_t = hit.t;
            }
        }
        REQUIRE(result.info.t == Catch::Approx(best_t).margin(real_t(1e-5)));
    }

    SECTION("ray hit - from below") {
        auto ray = tf::make_ray(
            tf::make_point(real_t(2), real_t(2), real_t(-5)),
            tf::make_vector(real_t(0), real_t(0), real_t(1))
        );
        auto result = tf::ray_cast(ray, mesh_with_tree);

        REQUIRE(result);
        REQUIRE(result.info.t == Catch::Approx(real_t(5)).margin(real_t(1e-5)));
    }

    SECTION("ray miss - outside mesh bounds") {
        auto ray = tf::make_ray(
            tf::make_point(real_t(10), real_t(10), real_t(5)),
            tf::make_vector(real_t(0), real_t(0), real_t(-1))
        );
        auto result = tf::ray_cast(ray, mesh_with_tree);

        REQUIRE_FALSE(result);
    }

    SECTION("ray miss - pointing away") {
        auto ray = tf::make_ray(
            tf::make_point(real_t(2), real_t(2), real_t(5)),
            tf::make_vector(real_t(0), real_t(0), real_t(1))
        );
        auto result = tf::ray_cast(ray, mesh_with_tree);

        REQUIRE_FALSE(result);
    }

    SECTION("ray hit with config - min_t") {
        auto ray = tf::make_ray(
            tf::make_point(real_t(2), real_t(2), real_t(5)),
            tf::make_vector(real_t(0), real_t(0), real_t(-1))
        );
        // Hit would be at t=5, but min_t=6
        auto config = tf::make_ray_config(real_t(6), std::numeric_limits<real_t>::max());
        auto result = tf::ray_cast(ray, mesh_with_tree, config);

        REQUIRE_FALSE(result);
    }

    SECTION("ray hit with config - max_t") {
        auto ray = tf::make_ray(
            tf::make_point(real_t(2), real_t(2), real_t(5)),
            tf::make_vector(real_t(0), real_t(0), real_t(-1))
        );
        // Hit would be at t=5, but max_t=4
        auto config = tf::make_ray_config(real_t(0), real_t(4));
        auto result = tf::ray_cast(ray, mesh_with_tree, config);

        REQUIRE_FALSE(result);
    }
}

// =============================================================================
// Mesh Ray Cast - 2D
// =============================================================================

TEMPLATE_TEST_CASE("mesh_ray_cast_2d", "[spatial][ray_cast]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto mesh = tf::test::create_grid_mesh_2d<index_t, real_t>(5, 5);

    tf::aabb_tree<index_t, real_t, 2> tree(mesh.polygons(), tf::config_tree(4, 4));
    auto mesh_with_tree = mesh.polygons() | tf::tag(tree);

    SECTION("ray hit - horizontal") {
        auto ray = tf::make_ray(
            tf::make_point(real_t(-1), real_t(2)),
            tf::make_vector(real_t(1), real_t(0))
        );
        auto result = tf::ray_cast(ray, mesh_with_tree);

        REQUIRE(result);

        // Brute force verification
        real_t best_t = std::numeric_limits<real_t>::max();
        for (std::size_t i = 0; i < mesh.faces().size(); ++i) {
            auto hit = tf::ray_cast(ray, mesh.polygons()[i]);
            if (hit && hit.t < best_t) {
                best_t = hit.t;
            }
        }
        REQUIRE(result.info.t == Catch::Approx(best_t).margin(real_t(1e-5)));
    }

    SECTION("ray miss") {
        auto ray = tf::make_ray(
            tf::make_point(real_t(-1), real_t(10)),
            tf::make_vector(real_t(1), real_t(0))
        );
        auto result = tf::ray_cast(ray, mesh_with_tree);

        REQUIRE_FALSE(result);
    }
}

// =============================================================================
// Static Quad Mesh Ray Cast - 3D
// =============================================================================

TEMPLATE_TEST_CASE("quad_mesh_ray_cast_3d", "[spatial][ray_cast]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    // Create a simple quad mesh at z=0
    tf::polygons_buffer<index_t, real_t, 3, 4> mesh;
    // 3x3 grid of quads = 4 quads
    for (int j = 0; j < 3; ++j) {
        for (int i = 0; i < 3; ++i) {
            mesh.points_buffer().emplace_back(real_t(i), real_t(j), real_t(0));
        }
    }
    // Quads (CCW winding)
    mesh.faces_buffer().emplace_back(index_t(0), index_t(1), index_t(4), index_t(3));
    mesh.faces_buffer().emplace_back(index_t(1), index_t(2), index_t(5), index_t(4));
    mesh.faces_buffer().emplace_back(index_t(3), index_t(4), index_t(7), index_t(6));
    mesh.faces_buffer().emplace_back(index_t(4), index_t(5), index_t(8), index_t(7));

    tf::aabb_tree<index_t, real_t, 3> tree(mesh.polygons(), tf::config_tree(4, 4));
    auto mesh_with_tree = mesh.polygons() | tf::tag(tree);

    SECTION("ray hit - from above") {
        auto ray = tf::make_ray(
            tf::make_point(real_t(0.5), real_t(0.5), real_t(5)),
            tf::make_vector(real_t(0), real_t(0), real_t(-1))
        );
        auto result = tf::ray_cast(ray, mesh_with_tree);

        REQUIRE(result);
        REQUIRE(result.info.t == Catch::Approx(real_t(5)).margin(real_t(1e-5)));

        // Brute force verification
        real_t best_t = std::numeric_limits<real_t>::max();
        for (std::size_t i = 0; i < mesh.faces().size(); ++i) {
            auto hit = tf::ray_cast(ray, mesh.polygons()[i]);
            if (hit && hit.t < best_t) {
                best_t = hit.t;
            }
        }
        REQUIRE(result.info.t == Catch::Approx(best_t).margin(real_t(1e-5)));
    }

    SECTION("ray miss") {
        auto ray = tf::make_ray(
            tf::make_point(real_t(10), real_t(10), real_t(5)),
            tf::make_vector(real_t(0), real_t(0), real_t(-1))
        );
        auto result = tf::ray_cast(ray, mesh_with_tree);

        REQUIRE_FALSE(result);
    }
}

// =============================================================================
// Static Quad Mesh Ray Cast - 2D
// =============================================================================

TEMPLATE_TEST_CASE("quad_mesh_ray_cast_2d", "[spatial][ray_cast]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    // Create a simple quad mesh
    tf::polygons_buffer<index_t, real_t, 2, 4> mesh;
    for (int j = 0; j < 3; ++j) {
        for (int i = 0; i < 3; ++i) {
            mesh.points_buffer().emplace_back(real_t(i), real_t(j));
        }
    }
    mesh.faces_buffer().emplace_back(index_t(0), index_t(1), index_t(4), index_t(3));
    mesh.faces_buffer().emplace_back(index_t(1), index_t(2), index_t(5), index_t(4));
    mesh.faces_buffer().emplace_back(index_t(3), index_t(4), index_t(7), index_t(6));
    mesh.faces_buffer().emplace_back(index_t(4), index_t(5), index_t(8), index_t(7));

    tf::aabb_tree<index_t, real_t, 2> tree(mesh.polygons(), tf::config_tree(4, 4));
    auto mesh_with_tree = mesh.polygons() | tf::tag(tree);

    SECTION("ray hit") {
        auto ray = tf::make_ray(
            tf::make_point(real_t(-1), real_t(0.5)),
            tf::make_vector(real_t(1), real_t(0))
        );
        auto result = tf::ray_cast(ray, mesh_with_tree);

        REQUIRE(result);

        // Brute force verification
        real_t best_t = std::numeric_limits<real_t>::max();
        for (std::size_t i = 0; i < mesh.faces().size(); ++i) {
            auto hit = tf::ray_cast(ray, mesh.polygons()[i]);
            if (hit && hit.t < best_t) {
                best_t = hit.t;
            }
        }
        REQUIRE(result.info.t == Catch::Approx(best_t).margin(real_t(1e-5)));
    }

    SECTION("ray miss") {
        auto ray = tf::make_ray(
            tf::make_point(real_t(-1), real_t(10)),
            tf::make_vector(real_t(1), real_t(0))
        );
        auto result = tf::ray_cast(ray, mesh_with_tree);

        REQUIRE_FALSE(result);
    }
}

// =============================================================================
// Dynamic Mesh Ray Cast - 3D
// =============================================================================

TEMPLATE_TEST_CASE("dynamic_mesh_ray_cast_3d", "[spatial][ray_cast]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto mesh = tf::test::create_dynamic_grid_mesh_3d<index_t, real_t>(5, 5);

    tf::aabb_tree<index_t, real_t, 3> tree(mesh.polygons(), tf::config_tree(4, 4));
    auto mesh_with_tree = mesh.polygons() | tf::tag(tree);

    SECTION("ray hit") {
        auto ray = tf::make_ray(
            tf::make_point(real_t(2), real_t(2), real_t(5)),
            tf::make_vector(real_t(0), real_t(0), real_t(-1))
        );
        auto result = tf::ray_cast(ray, mesh_with_tree);

        REQUIRE(result);

        // Brute force verification
        real_t best_t = std::numeric_limits<real_t>::max();
        for (std::size_t i = 0; i < mesh.faces().size(); ++i) {
            auto hit = tf::ray_cast(ray, mesh.polygons()[i]);
            if (hit && hit.t < best_t) {
                best_t = hit.t;
            }
        }
        REQUIRE(result.info.t == Catch::Approx(best_t).margin(real_t(1e-5)));
    }

    SECTION("ray miss") {
        auto ray = tf::make_ray(
            tf::make_point(real_t(10), real_t(10), real_t(5)),
            tf::make_vector(real_t(0), real_t(0), real_t(-1))
        );
        auto result = tf::ray_cast(ray, mesh_with_tree);

        REQUIRE_FALSE(result);
    }
}

// =============================================================================
// Dynamic Mesh Ray Cast - 2D
// =============================================================================

TEMPLATE_TEST_CASE("dynamic_mesh_ray_cast_2d", "[spatial][ray_cast]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto mesh = tf::test::create_dynamic_grid_mesh_2d<index_t, real_t>(5, 5);

    tf::aabb_tree<index_t, real_t, 2> tree(mesh.polygons(), tf::config_tree(4, 4));
    auto mesh_with_tree = mesh.polygons() | tf::tag(tree);

    SECTION("ray hit") {
        auto ray = tf::make_ray(
            tf::make_point(real_t(-1), real_t(2)),
            tf::make_vector(real_t(1), real_t(0))
        );
        auto result = tf::ray_cast(ray, mesh_with_tree);

        REQUIRE(result);

        // Brute force verification
        real_t best_t = std::numeric_limits<real_t>::max();
        for (std::size_t i = 0; i < mesh.faces().size(); ++i) {
            auto hit = tf::ray_cast(ray, mesh.polygons()[i]);
            if (hit && hit.t < best_t) {
                best_t = hit.t;
            }
        }
        REQUIRE(result.info.t == Catch::Approx(best_t).margin(real_t(1e-5)));
    }
}

// =============================================================================
// Segments Ray Cast - 3D
// =============================================================================

TEMPLATE_TEST_CASE("segments_ray_cast_3d", "[spatial][ray_cast]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto segments = tf::test::create_grid_segments_3d<index_t, real_t>(5, 5);

    tf::aabb_tree<index_t, real_t, 3> tree(segments.segments(), tf::config_tree(4, 4));
    auto segments_with_tree = segments.segments() | tf::tag(tree);

    SECTION("ray hit - perpendicular to segment") {
        // Ray pointing at segment at y=2 from below
        auto ray = tf::make_ray(
            tf::make_point(real_t(0.5), real_t(2), real_t(-1)),
            tf::make_vector(real_t(0), real_t(0), real_t(1))
        );
        auto result = tf::ray_cast(ray, segments_with_tree);

        REQUIRE(result);

        // Brute force verification
        real_t best_t = std::numeric_limits<real_t>::max();
        for (std::size_t i = 0; i < segments.edges().size(); ++i) {
            auto hit = tf::ray_cast(ray, segments.segments()[i]);
            if (hit && hit.t < best_t) {
                best_t = hit.t;
            }
        }
        REQUIRE(result.info.t == Catch::Approx(best_t).margin(real_t(1e-5)));
    }

    SECTION("ray miss") {
        auto ray = tf::make_ray(
            tf::make_point(real_t(10), real_t(10), real_t(-1)),
            tf::make_vector(real_t(0), real_t(0), real_t(1))
        );
        auto result = tf::ray_cast(ray, segments_with_tree);

        REQUIRE_FALSE(result);
    }
}

// =============================================================================
// Segments Ray Cast - 2D
// =============================================================================

TEMPLATE_TEST_CASE("segments_ray_cast_2d", "[spatial][ray_cast]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto segments = tf::test::create_grid_segments_2d<index_t, real_t>(5, 5);

    tf::aabb_tree<index_t, real_t, 2> tree(segments.segments(), tf::config_tree(4, 4));
    auto segments_with_tree = segments.segments() | tf::tag(tree);

    SECTION("ray hit") {
        auto ray = tf::make_ray(
            tf::make_point(real_t(-1), real_t(2)),
            tf::make_vector(real_t(1), real_t(0))
        );
        auto result = tf::ray_cast(ray, segments_with_tree);

        REQUIRE(result);

        // Brute force verification
        real_t best_t = std::numeric_limits<real_t>::max();
        for (std::size_t i = 0; i < segments.edges().size(); ++i) {
            auto hit = tf::ray_cast(ray, segments.segments()[i]);
            if (hit && hit.t < best_t) {
                best_t = hit.t;
            }
        }
        REQUIRE(result.info.t == Catch::Approx(best_t).margin(real_t(1e-5)));
    }

    SECTION("ray miss") {
        auto ray = tf::make_ray(
            tf::make_point(real_t(-1), real_t(10)),
            tf::make_vector(real_t(1), real_t(0))
        );
        auto result = tf::ray_cast(ray, segments_with_tree);

        REQUIRE_FALSE(result);
    }
}

// =============================================================================
// Point Cloud Ray Cast - 3D
// =============================================================================

TEMPLATE_TEST_CASE("point_cloud_ray_cast_3d", "[spatial][ray_cast]",
    float, double)
{
    using real_t = TestType;

    auto cloud = tf::test::create_grid_points_3d<real_t>(4, 4, 4);

    tf::aabb_tree<int, real_t, 3> tree(cloud.points(), tf::config_tree(4, 4));
    auto cloud_with_tree = cloud.points() | tf::tag(tree);

    SECTION("ray hit - through point") {
        // Ray going through point at (1,1,1)
        auto ray = tf::make_ray(
            tf::make_point(real_t(1), real_t(1), real_t(-1)),
            tf::make_vector(real_t(0), real_t(0), real_t(1))
        );
        auto result = tf::ray_cast(ray, cloud_with_tree);

        REQUIRE(result);

        // Brute force verification
        real_t best_t = std::numeric_limits<real_t>::max();
        for (std::size_t i = 0; i < cloud.points().size(); ++i) {
            auto hit = tf::ray_cast(ray, cloud.points()[i]);
            if (hit && hit.t < best_t) {
                best_t = hit.t;
            }
        }
        REQUIRE(result.info.t == Catch::Approx(best_t).margin(real_t(1e-5)));
    }

    SECTION("ray miss") {
        auto ray = tf::make_ray(
            tf::make_point(real_t(0.5), real_t(0.5), real_t(-1)),
            tf::make_vector(real_t(0), real_t(0), real_t(1))
        );
        auto result = tf::ray_cast(ray, cloud_with_tree);

        REQUIRE_FALSE(result);
    }
}

// =============================================================================
// Point Cloud Ray Cast - 2D
// =============================================================================

TEMPLATE_TEST_CASE("point_cloud_ray_cast_2d", "[spatial][ray_cast]",
    float, double)
{
    using real_t = TestType;

    auto cloud = tf::test::create_grid_points_2d<real_t>(5, 5);

    tf::aabb_tree<int, real_t, 2> tree(cloud.points(), tf::config_tree(4, 4));
    auto cloud_with_tree = cloud.points() | tf::tag(tree);

    SECTION("ray hit - through point") {
        // Ray going through point at (2,2)
        auto ray = tf::make_ray(
            tf::make_point(real_t(-1), real_t(2)),
            tf::make_vector(real_t(1), real_t(0))
        );
        auto result = tf::ray_cast(ray, cloud_with_tree);

        REQUIRE(result);

        // Brute force verification
        real_t best_t = std::numeric_limits<real_t>::max();
        for (std::size_t i = 0; i < cloud.points().size(); ++i) {
            auto hit = tf::ray_cast(ray, cloud.points()[i]);
            if (hit && hit.t < best_t) {
                best_t = hit.t;
            }
        }
        REQUIRE(result.info.t == Catch::Approx(best_t).margin(real_t(1e-5)));
    }

    SECTION("ray miss") {
        auto ray = tf::make_ray(
            tf::make_point(real_t(-1), real_t(0.5)),
            tf::make_vector(real_t(1), real_t(0))
        );
        auto result = tf::ray_cast(ray, cloud_with_tree);

        REQUIRE_FALSE(result);
    }
}

// =============================================================================
// Multiple Hits - Verify First Hit Returned
// =============================================================================

TEMPLATE_TEST_CASE("ray_cast_first_hit", "[spatial][ray_cast]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    // Create two separate meshes at different z levels
    auto mesh0 = tf::test::create_grid_mesh_3d<index_t, real_t>(
        3, 3, {real_t(0), real_t(0), real_t(0)});
    auto mesh1 = tf::test::create_grid_mesh_3d<index_t, real_t>(
        3, 3, {real_t(0), real_t(0), real_t(2)});

    // Combine points and faces
    tf::polygons_buffer<index_t, real_t, 3, 3> combined;
    for (std::size_t i = 0; i < mesh0.points().size(); ++i) {
        combined.points_buffer().push_back(mesh0.points()[i]);
    }
    auto offset = static_cast<index_t>(mesh0.points().size());
    for (std::size_t i = 0; i < mesh1.points().size(); ++i) {
        combined.points_buffer().push_back(mesh1.points()[i]);
    }
    for (std::size_t i = 0; i < mesh0.faces().size(); ++i) {
        combined.faces_buffer().push_back(mesh0.faces()[i]);
    }
    for (std::size_t i = 0; i < mesh1.faces().size(); ++i) {
        auto f = mesh1.faces()[i];
        combined.faces_buffer().emplace_back(f[0] + offset, f[1] + offset, f[2] + offset);
    }

    tf::aabb_tree<index_t, real_t, 3> tree(combined.polygons(), tf::config_tree(4, 4));
    auto combined_with_tree = combined.polygons() | tf::tag(tree);

    SECTION("ray hits closer mesh first") {
        auto ray = tf::make_ray(
            tf::make_point(real_t(1), real_t(1), real_t(5)),
            tf::make_vector(real_t(0), real_t(0), real_t(-1))
        );
        auto result = tf::ray_cast(ray, combined_with_tree);

        REQUIRE(result);
        // Should hit mesh at z=2 first (t=3), not z=0 (t=5)
        REQUIRE(result.info.t == Catch::Approx(real_t(3)).margin(real_t(1e-5)));
    }
}
