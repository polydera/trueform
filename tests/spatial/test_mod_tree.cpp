/**
 * @file test_mod_tree.cpp
 * @brief Tests for mod_tree (modifiable AABB tree) operations
 *
 * Tests for:
 * - mod_tree with raycast after boolean operations
 * - mod_tree with neighbor_search after boolean operations
 * - Comparison of stitched mod_tree vs fresh tree
 *
 * Copyright (c) 2025 Ziga Sajovic, XLAB
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <trueform/trueform.hpp>
#include "type_traits.hpp"
#include <cmath>

// =============================================================================
// Helper functions
// =============================================================================

template <typename Index, typename Real>
auto create_test_box_for_boolean() {
    return tf::make_box_mesh<Index>(Real(2), Real(2), Real(2));
}

template <typename Index, typename Real>
auto create_test_sphere_for_boolean() {
    return tf::make_sphere_mesh<Index>(Real(0.5), 20, 20);
}

// =============================================================================
// Test 1: mod_tree basic build and raycast
// =============================================================================

TEMPLATE_TEST_CASE("mod_tree_basic_raycast", "[mod_tree]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto box = tf::make_box_mesh<index_t>(real_t(2), real_t(2), real_t(2));

    // Build mod_tree
    tf::mod_tree<index_t, tf::aabb<real_t, 3>> tree;
    tree.build(box.polygons(), tf::config_tree(4, 4));

    // Test raycast to each face
    for (std::size_t poly_id = 0; poly_id < box.size(); ++poly_id) {
        auto poly = box.polygons()[poly_id];
        auto centroid = tf::centroid(poly);
        auto normal = tf::make_normal(poly);

        real_t offset = real_t(0.01);
        auto ray = tf::make_ray(centroid + offset * normal, -normal);

        auto form = box.polygons() | tf::tag(tree);
        auto info = tf::ray_cast(ray, form);

        REQUIRE(info);
        REQUIRE(info.element == static_cast<index_t>(poly_id));
    }
}

// =============================================================================
// Test 2: mod_tree basic build and neighbor_search
// =============================================================================

TEMPLATE_TEST_CASE("mod_tree_basic_neighbor_search", "[mod_tree]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto box = tf::make_box_mesh<index_t>(real_t(2), real_t(2), real_t(2));

    // Build mod_tree
    tf::mod_tree<index_t, tf::aabb<real_t, 3>> tree;
    tree.build(box.polygons(), tf::config_tree(4, 4));

    // Test neighbor_search from each face centroid
    for (std::size_t poly_id = 0; poly_id < box.size(); ++poly_id) {
        auto poly = box.polygons()[poly_id];
        auto centroid = tf::centroid(poly);

        auto form = box.polygons() | tf::tag(tree);
        auto nearest = tf::neighbor_search(form, centroid);

        REQUIRE(nearest);
        // Should find either this polygon or one with same centroid
        REQUIRE(nearest.metric() < tf::epsilon<real_t>);
    }
}

// =============================================================================
// Test 3: mod_tree vs regular aabb_tree consistency
// =============================================================================

TEMPLATE_TEST_CASE("mod_tree_vs_aabb_tree_raycast", "[mod_tree]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto sphere = tf::make_sphere_mesh<index_t>(real_t(1), 30, 30);

    // Build both tree types
    tf::mod_tree<index_t, tf::aabb<real_t, 3>> mod_tree;
    mod_tree.build(sphere.polygons(), tf::config_tree(4, 4));

    tf::aabb_tree<index_t, real_t, 3> regular_tree(sphere.polygons(), tf::config_tree(4, 4));

    // Sample some polygons and verify both trees give same results
    std::array<std::size_t, 5> sample_ids = {0, sphere.size() / 4, sphere.size() / 2,
                                              3 * sphere.size() / 4, sphere.size() - 1};

    for (auto poly_id : sample_ids) {
        auto poly = sphere.polygons()[poly_id];
        auto centroid = tf::centroid(poly);
        auto normal = tf::make_normal(poly);

        real_t offset = real_t(0.01);
        auto ray = tf::make_ray(centroid + offset * normal, -normal);

        auto form_mod = sphere.polygons() | tf::tag(mod_tree);
        auto form_reg = sphere.polygons() | tf::tag(regular_tree);

        auto info_mod = tf::ray_cast(ray, form_mod);
        auto info_reg = tf::ray_cast(ray, form_reg);

        REQUIRE(info_mod);
        REQUIRE(info_reg);
        REQUIRE(info_mod.element == info_reg.element);
    }
}

// =============================================================================
// Test 4: mod_tree vs regular aabb_tree neighbor_search consistency
// =============================================================================

TEMPLATE_TEST_CASE("mod_tree_vs_aabb_tree_neighbor_search", "[mod_tree]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto sphere = tf::make_sphere_mesh<index_t>(real_t(1), 30, 30);

    // Build both tree types
    tf::mod_tree<index_t, tf::aabb<real_t, 3>> mod_tree;
    mod_tree.build(sphere.polygons(), tf::config_tree(4, 4));

    tf::aabb_tree<index_t, real_t, 3> regular_tree(sphere.polygons(), tf::config_tree(4, 4));

    // Test from some query points
    std::array<tf::point<real_t, 3>, 4> query_points = {{
        {real_t(0), real_t(0), real_t(1.5)},
        {real_t(1.5), real_t(0), real_t(0)},
        {real_t(0), real_t(-1.5), real_t(0)},
        {real_t(0.5), real_t(0.5), real_t(0.5)}
    }};

    for (const auto& query : query_points) {
        auto form_mod = sphere.polygons() | tf::tag(mod_tree);
        auto form_reg = sphere.polygons() | tf::tag(regular_tree);

        auto nearest_mod = tf::neighbor_search(form_mod, query);
        auto nearest_reg = tf::neighbor_search(form_reg, query);

        REQUIRE(nearest_mod);
        REQUIRE(nearest_reg);
        // Distances should match (even if different polygons at same distance)
        REQUIRE(std::abs(nearest_mod.metric() - nearest_reg.metric()) < tf::epsilon<real_t>);
    }
}

// =============================================================================
// Test 5: mod_tree with stitched boolean - raycast
// =============================================================================

TEMPLATE_TEST_CASE("mod_tree_stitched_boolean_raycast", "[mod_tree]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    // Create box and sphere
    auto box = tf::make_box_mesh<index_t>(real_t(2), real_t(2), real_t(2));
    auto sphere = tf::make_sphere_mesh<index_t>(real_t(0.5), 20, 20);

    // Build topology for box
    tf::face_membership<index_t> fm0;
    fm0.build(box.polygons());
    tf::manifold_edge_link<index_t, 3> mel0;
    mel0.build(box.faces(), fm0);

    // Build mod_tree for box
    tf::mod_tree<index_t, tf::aabb<real_t, 3>> mod_tree_input;
    mod_tree_input.build(box.polygons(), tf::config_tree(4, 4));

    // Build topology for sphere
    tf::face_membership<index_t> fm1;
    fm1.build(sphere.polygons());
    tf::manifold_edge_link<index_t, 3> mel1;
    mel1.build(sphere.faces(), fm1);
    tf::aabb_tree<index_t, real_t, 3> tree1(sphere.polygons(), tf::config_tree(4, 4));

    // Position sphere at corner of box
    auto frame = tf::make_frame(tf::make_transformation_from_translation(
        tf::vector<real_t, 3>{real_t(0.5), real_t(0.5), real_t(0.5)}));

    // Do the boolean (left difference)
    auto [result, labels, index_maps] = tf::make_boolean(
        box.polygons() | tf::tag(fm0) | tf::tag(mel0) | tf::tag(mod_tree_input),
        sphere.polygons() | tf::tag(fm1) | tf::tag(mel1) | tf::tag(tree1) | tf::tag(frame),
        tf::boolean_op::left_difference, tf::return_index_map);

    // Stitch the mod_tree
    tf::stitch_mod_tree(result.polygons(), mod_tree_input, tf::none, index_maps, tf::config_tree(4, 4));

    // Build fresh tree for comparison
    tf::aabb_tree<index_t, real_t, 3> fresh_tree(result.polygons(), tf::config_tree(4, 4));

    // Test raycast on some result polygons
    if (result.size() > 0) {
        std::size_t test_count = std::min(std::size_t(10), result.size());
        std::size_t step = result.size() / test_count;

        for (std::size_t i = 0; i < test_count; ++i) {
            std::size_t poly_id = i * step;
            auto poly = result.polygons()[poly_id];
            auto centroid = tf::centroid(poly);
            auto normal = tf::make_normal(poly);

            real_t offset = real_t(0.01);
            auto ray = tf::make_ray(centroid + offset * normal, -normal);

            auto form_stitched = result.polygons() | tf::tag(mod_tree_input);
            auto form_fresh = result.polygons() | tf::tag(fresh_tree);

            auto info_stitched = tf::ray_cast(ray, form_stitched);
            auto info_fresh = tf::ray_cast(ray, form_fresh);

            REQUIRE(info_stitched);
            REQUIRE(info_fresh);
            REQUIRE(info_stitched.element == info_fresh.element);
        }
    }
}

// =============================================================================
// Test 6: mod_tree with stitched boolean - neighbor_search
// =============================================================================

TEMPLATE_TEST_CASE("mod_tree_stitched_boolean_neighbor_search", "[mod_tree]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    // Create box and sphere
    auto box = tf::make_box_mesh<index_t>(real_t(2), real_t(2), real_t(2));
    auto sphere = tf::make_sphere_mesh<index_t>(real_t(0.5), 20, 20);

    // Build topology for box
    tf::face_membership<index_t> fm0;
    fm0.build(box.polygons());
    tf::manifold_edge_link<index_t, 3> mel0;
    mel0.build(box.faces(), fm0);

    // Build mod_tree for box
    tf::mod_tree<index_t, tf::aabb<real_t, 3>> mod_tree_input;
    mod_tree_input.build(box.polygons(), tf::config_tree(4, 4));

    // Build topology for sphere
    tf::face_membership<index_t> fm1;
    fm1.build(sphere.polygons());
    tf::manifold_edge_link<index_t, 3> mel1;
    mel1.build(sphere.faces(), fm1);
    tf::aabb_tree<index_t, real_t, 3> tree1(sphere.polygons(), tf::config_tree(4, 4));

    // Position sphere at corner of box
    auto frame = tf::make_frame(tf::make_transformation_from_translation(
        tf::vector<real_t, 3>{real_t(0.5), real_t(0.5), real_t(0.5)}));

    // Do the boolean
    auto [result, labels, index_maps] = tf::make_boolean(
        box.polygons() | tf::tag(fm0) | tf::tag(mel0) | tf::tag(mod_tree_input),
        sphere.polygons() | tf::tag(fm1) | tf::tag(mel1) | tf::tag(tree1) | tf::tag(frame),
        tf::boolean_op::left_difference, tf::return_index_map);

    // Stitch the mod_tree
    tf::stitch_mod_tree(result.polygons(), mod_tree_input, tf::none, index_maps, tf::config_tree(4, 4));

    // Build fresh tree for comparison
    tf::aabb_tree<index_t, real_t, 3> fresh_tree(result.polygons(), tf::config_tree(4, 4));

    // Test neighbor_search on some result polygon centroids
    if (result.size() > 0) {
        std::size_t test_count = std::min(std::size_t(10), result.size());
        std::size_t step = result.size() / test_count;

        for (std::size_t i = 0; i < test_count; ++i) {
            std::size_t poly_id = i * step;
            auto poly = result.polygons()[poly_id];
            auto centroid = tf::centroid(poly);

            auto form_stitched = result.polygons() | tf::tag(mod_tree_input);
            auto form_fresh = result.polygons() | tf::tag(fresh_tree);

            auto nearest_stitched = tf::neighbor_search(form_stitched, centroid);
            auto nearest_fresh = tf::neighbor_search(form_fresh, centroid);

            REQUIRE(nearest_stitched);
            REQUIRE(nearest_fresh);
            REQUIRE(std::abs(nearest_stitched.metric() - nearest_fresh.metric()) < tf::epsilon<real_t>);
        }
    }
}

// =============================================================================
// Test 7: mod_tree main and delta tree access
// =============================================================================

TEMPLATE_TEST_CASE("mod_tree_main_and_delta", "[mod_tree]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    // Create large sphere as main mesh (like original uses dragon mesh)
    auto sphere0 = tf::make_sphere_mesh<index_t>(real_t(1), 40, 40);
    // Create smaller sphere to cut into it
    auto sphere1 = tf::make_sphere_mesh<index_t>(real_t(0.3), 20, 20);

    // Build topology for sphere0
    tf::face_membership<index_t> fm0;
    fm0.build(sphere0.polygons());
    tf::manifold_edge_link<index_t, 3> mel0;
    mel0.build(sphere0.faces(), fm0);

    // Build mod_tree for sphere0
    tf::mod_tree<index_t, tf::aabb<real_t, 3>> mod_tree_input;
    mod_tree_input.build(sphere0.polygons(), tf::config_tree(4, 4));

    // Build topology for sphere1
    tf::face_membership<index_t> fm1;
    fm1.build(sphere1.polygons());
    tf::manifold_edge_link<index_t, 3> mel1;
    mel1.build(sphere1.faces(), fm1);
    tf::aabb_tree<index_t, real_t, 3> tree1(sphere1.polygons(), tf::config_tree(4, 4));

    // Position sphere1 at north pole of sphere0 (z=1)
    auto frame = tf::make_frame(tf::make_transformation_from_translation(
        tf::vector<real_t, 3>{real_t(0), real_t(0), real_t(1)}));

    // Do the boolean
    auto [result, labels, index_maps] = tf::make_boolean(
        sphere0.polygons() | tf::tag(fm0) | tf::tag(mel0) | tf::tag(mod_tree_input),
        sphere1.polygons() | tf::tag(fm1) | tf::tag(mel1) | tf::tag(tree1) | tf::tag(frame),
        tf::boolean_op::left_difference, tf::return_index_map);

    // Stitch the mod_tree
    tf::stitch_mod_tree(result.polygons(), mod_tree_input, tf::none, index_maps, tf::config_tree(4, 4));

    // Access main and delta trees
    const auto& main_tree = mod_tree_input.main_tree();
    const auto& delta_tree = mod_tree_input.delta_tree();

    // Main tree should have preserved polygons
    REQUIRE(main_tree.ids().size() > 0);

    // Delta tree should have new polygons from boolean
    REQUIRE(delta_tree.ids().size() > 0);

    // Test raycast to polygon from main tree
    if (main_tree.ids().size() > 0) {
        auto main_id = main_tree.ids()[0];
        auto poly = result.polygons()[main_id];
        auto centroid = tf::centroid(poly);
        auto normal = tf::make_normal(poly);

        real_t offset = real_t(0.01);
        auto ray = tf::make_ray(centroid + offset * normal, -normal);

        auto form = result.polygons() | tf::tag(mod_tree_input);
        auto info = tf::ray_cast(ray, form);

        REQUIRE(info);
        REQUIRE(info.element == main_id);
    }

    // Test raycast to polygon from delta tree
    if (delta_tree.ids().size() > 0) {
        auto delta_id = delta_tree.ids()[0];
        auto poly = result.polygons()[delta_id];
        auto centroid = tf::centroid(poly);
        auto normal = tf::make_normal(poly);

        real_t offset = real_t(0.01);
        auto ray = tf::make_ray(centroid + offset * normal, -normal);

        auto form = result.polygons() | tf::tag(mod_tree_input);
        auto info = tf::ray_cast(ray, form);

        REQUIRE(info);
        REQUIRE(info.element == delta_id);
    }
}

// =============================================================================
// Test 8: mod_tree with union boolean operation
// =============================================================================

TEMPLATE_TEST_CASE("mod_tree_union_boolean", "[mod_tree]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    // Create two boxes
    auto box1 = tf::make_box_mesh<index_t>(real_t(1), real_t(1), real_t(1));
    auto box2 = tf::make_box_mesh<index_t>(real_t(1), real_t(1), real_t(1));

    // Build topology for box1
    tf::face_membership<index_t> fm0;
    fm0.build(box1.polygons());
    tf::manifold_edge_link<index_t, 3> mel0;
    mel0.build(box1.faces(), fm0);

    // Build mod_tree for box1
    tf::mod_tree<index_t, tf::aabb<real_t, 3>> mod_tree_input;
    mod_tree_input.build(box1.polygons(), tf::config_tree(4, 4));

    // Build topology for box2
    tf::face_membership<index_t> fm1;
    fm1.build(box2.polygons());
    tf::manifold_edge_link<index_t, 3> mel1;
    mel1.build(box2.faces(), fm1);
    tf::aabb_tree<index_t, real_t, 3> tree1(box2.polygons(), tf::config_tree(4, 4));

    // Position box2 offset from box1
    auto frame = tf::make_frame(tf::make_transformation_from_translation(
        tf::vector<real_t, 3>{real_t(0.5), real_t(0), real_t(0)}));

    // Do union boolean
    auto [result, labels, index_maps] = tf::make_boolean(
        box1.polygons() | tf::tag(fm0) | tf::tag(mel0) | tf::tag(mod_tree_input),
        box2.polygons() | tf::tag(fm1) | tf::tag(mel1) | tf::tag(tree1) | tf::tag(frame),
        tf::boolean_op::merge, tf::return_index_map);

    // Stitch the mod_tree
    tf::stitch_mod_tree(result.polygons(), mod_tree_input, tf::none, index_maps, tf::config_tree(4, 4));

    // Build fresh tree for comparison
    tf::aabb_tree<index_t, real_t, 3> fresh_tree(result.polygons(), tf::config_tree(4, 4));

    // Verify stitched and fresh trees give consistent results
    if (result.size() > 0) {
        for (std::size_t poly_id = 0; poly_id < std::min(std::size_t(5), result.size()); ++poly_id) {
            auto poly = result.polygons()[poly_id];
            auto centroid = tf::centroid(poly);
            auto normal = tf::make_normal(poly);

            real_t offset = real_t(0.01);
            auto ray = tf::make_ray(centroid + offset * normal, -normal);

            auto form_stitched = result.polygons() | tf::tag(mod_tree_input);
            auto form_fresh = result.polygons() | tf::tag(fresh_tree);

            auto info_stitched = tf::ray_cast(ray, form_stitched);
            auto info_fresh = tf::ray_cast(ray, form_fresh);

            REQUIRE(info_stitched);
            REQUIRE(info_fresh);
            REQUIRE(info_stitched.element == info_fresh.element);
        }
    }
}
