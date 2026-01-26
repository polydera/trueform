/**
 * @file test_stitching.cpp
 * @brief Tests for stitched topology structures after boolean operations
 *
 * Tests for:
 * - tf::stitched_face_membership - compare stitched vs fresh build
 * - tf::stitched_manifold_edge_link - compare stitched vs fresh build
 *
 * Copyright (c) 2025 Ziga Sajovic, XLAB
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <trueform/trueform.hpp>
#include "type_traits.hpp"

// =============================================================================
// Helper: Compare face_membership structures
// =============================================================================

template <typename FaceMembership>
bool face_memberships_equal(const FaceMembership& a, const FaceMembership& b) {
    if (a.offsets_buffer().size() != b.offsets_buffer().size()) {
        return false;
    }
    if (a.data_buffer().size() != b.data_buffer().size()) {
        return false;
    }
    for (std::size_t i = 0; i < a.offsets_buffer().size(); ++i) {
        if (a.offsets_buffer()[i] != b.offsets_buffer()[i]) {
            return false;
        }
    }
    for (std::size_t pt = 0; pt < a.size(); ++pt) {
        auto r1 = a[pt];
        auto r2 = b[pt];
        if (r1.size() != r2.size()) {
            return false;
        }
        for (std::size_t j = 0; j < r1.size(); ++j) {
            if (r1[j] != r2[j]) {
                return false;
            }
        }
    }
    return true;
}

// =============================================================================
// Helper: Compare manifold_edge_link structures
// =============================================================================

template <typename ManifoldEdgeLink>
bool manifold_edge_links_equal(const ManifoldEdgeLink& a, const ManifoldEdgeLink& b) {
    if (a.data_buffer().size() != b.data_buffer().size()) {
        return false;
    }
    for (std::size_t face = 0; face < std::size_t(a.size()); ++face) {
        auto r1 = a[face];
        auto r2 = b[face];
        if (r1.size() != r2.size()) {
            return false;
        }
        for (std::size_t edge = 0; edge < r1.size(); ++edge) {
            if (r1[edge].face_peer != r2[edge].face_peer) {
                return false;
            }
        }
    }
    return true;
}

// =============================================================================
// Test 1: stitched_face_membership basic - box minus sphere
// =============================================================================

TEMPLATE_TEST_CASE("stitched_face_membership_box_minus_sphere", "[topology][stitching]",
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
    tf::aabb_tree<index_t, real_t, 3> tree0(box.polygons(), tf::config_tree(4, 4));

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
        box.polygons() | tf::tag(fm0) | tf::tag(mel0) | tf::tag(tree0),
        sphere.polygons() | tf::tag(fm1) | tf::tag(mel1) | tf::tag(tree1) | tf::tag(frame),
        tf::boolean_op::left_difference, tf::return_index_map);

    // Build stitched face_membership
    auto fm_stitched = tf::stitched_face_membership(
        result.faces(), index_t(result.points().size()), fm0, fm1, index_maps);

    // Build fresh face_membership for comparison
    tf::face_membership<index_t> fm_fresh;
    fm_fresh.build(result.polygons());

    // Compare
    REQUIRE(fm_stitched.size() == fm_fresh.size());
    REQUIRE(fm_stitched.offsets_buffer().size() == fm_fresh.offsets_buffer().size());
    REQUIRE(fm_stitched.data_buffer().size() == fm_fresh.data_buffer().size());
    REQUIRE(face_memberships_equal(fm_stitched, fm_fresh));
}

// =============================================================================
// Test 2: stitched_manifold_edge_link basic - box minus sphere
// =============================================================================

TEMPLATE_TEST_CASE("stitched_manifold_edge_link_box_minus_sphere", "[topology][stitching]",
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
    tf::aabb_tree<index_t, real_t, 3> tree0(box.polygons(), tf::config_tree(4, 4));

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
        box.polygons() | tf::tag(fm0) | tf::tag(mel0) | tf::tag(tree0),
        sphere.polygons() | tf::tag(fm1) | tf::tag(mel1) | tf::tag(tree1) | tf::tag(frame),
        tf::boolean_op::left_difference, tf::return_index_map);

    // Build stitched structures
    auto fm_stitched = tf::stitched_face_membership(
        result.faces(), index_t(result.points().size()), fm0, fm1, index_maps);
    auto mel_stitched = tf::stitched_manifold_edge_link(
        result.faces(), mel0, mel1, fm_stitched, index_maps);

    // Build fresh structures for comparison
    tf::face_membership<index_t> fm_fresh;
    fm_fresh.build(result.polygons());
    tf::manifold_edge_link<index_t, 3> mel_fresh;
    mel_fresh.build(result.faces(), fm_fresh);

    // Compare
    REQUIRE(mel_stitched.size() == mel_fresh.size());
    REQUIRE(mel_stitched.data_buffer().size() == mel_fresh.data_buffer().size());
    REQUIRE(manifold_edge_links_equal(mel_stitched, mel_fresh));
}

// =============================================================================
// Test 3: stitched_face_membership - sphere minus sphere
// =============================================================================

TEMPLATE_TEST_CASE("stitched_face_membership_sphere_minus_sphere", "[topology][stitching]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    // Create two spheres
    auto sphere0 = tf::make_sphere_mesh<index_t>(real_t(1), 30, 30);
    auto sphere1 = tf::make_sphere_mesh<index_t>(real_t(0.4), 20, 20);

    // Build topology for sphere0
    tf::face_membership<index_t> fm0;
    fm0.build(sphere0.polygons());
    tf::manifold_edge_link<index_t, 3> mel0;
    mel0.build(sphere0.faces(), fm0);
    tf::aabb_tree<index_t, real_t, 3> tree0(sphere0.polygons(), tf::config_tree(4, 4));

    // Build topology for sphere1
    tf::face_membership<index_t> fm1;
    fm1.build(sphere1.polygons());
    tf::manifold_edge_link<index_t, 3> mel1;
    mel1.build(sphere1.faces(), fm1);
    tf::aabb_tree<index_t, real_t, 3> tree1(sphere1.polygons(), tf::config_tree(4, 4));

    // Position sphere1 at north pole of sphere0
    auto frame = tf::make_frame(tf::make_transformation_from_translation(
        tf::vector<real_t, 3>{real_t(0), real_t(0), real_t(1)}));

    // Do the boolean (left difference)
    auto [result, labels, index_maps] = tf::make_boolean(
        sphere0.polygons() | tf::tag(fm0) | tf::tag(mel0) | tf::tag(tree0),
        sphere1.polygons() | tf::tag(fm1) | tf::tag(mel1) | tf::tag(tree1) | tf::tag(frame),
        tf::boolean_op::left_difference, tf::return_index_map);

    // Build stitched face_membership
    auto fm_stitched = tf::stitched_face_membership(
        result.faces(), index_t(result.points().size()), fm0, fm1, index_maps);

    // Build fresh face_membership for comparison
    tf::face_membership<index_t> fm_fresh;
    fm_fresh.build(result.polygons());

    // Compare
    REQUIRE(fm_stitched.size() == fm_fresh.size());
    REQUIRE(face_memberships_equal(fm_stitched, fm_fresh));
}

// =============================================================================
// Test 4: stitched_manifold_edge_link - sphere minus sphere
// =============================================================================

TEMPLATE_TEST_CASE("stitched_manifold_edge_link_sphere_minus_sphere", "[topology][stitching]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    // Create two spheres
    auto sphere0 = tf::make_sphere_mesh<index_t>(real_t(1), 30, 30);
    auto sphere1 = tf::make_sphere_mesh<index_t>(real_t(0.4), 20, 20);

    // Build topology for sphere0
    tf::face_membership<index_t> fm0;
    fm0.build(sphere0.polygons());
    tf::manifold_edge_link<index_t, 3> mel0;
    mel0.build(sphere0.faces(), fm0);
    tf::aabb_tree<index_t, real_t, 3> tree0(sphere0.polygons(), tf::config_tree(4, 4));

    // Build topology for sphere1
    tf::face_membership<index_t> fm1;
    fm1.build(sphere1.polygons());
    tf::manifold_edge_link<index_t, 3> mel1;
    mel1.build(sphere1.faces(), fm1);
    tf::aabb_tree<index_t, real_t, 3> tree1(sphere1.polygons(), tf::config_tree(4, 4));

    // Position sphere1 at north pole of sphere0
    auto frame = tf::make_frame(tf::make_transformation_from_translation(
        tf::vector<real_t, 3>{real_t(0), real_t(0), real_t(1)}));

    // Do the boolean (left difference)
    auto [result, labels, index_maps] = tf::make_boolean(
        sphere0.polygons() | tf::tag(fm0) | tf::tag(mel0) | tf::tag(tree0),
        sphere1.polygons() | tf::tag(fm1) | tf::tag(mel1) | tf::tag(tree1) | tf::tag(frame),
        tf::boolean_op::left_difference, tf::return_index_map);

    // Build stitched structures
    auto fm_stitched = tf::stitched_face_membership(
        result.faces(), index_t(result.points().size()), fm0, fm1, index_maps);
    auto mel_stitched = tf::stitched_manifold_edge_link(
        result.faces(), mel0, mel1, fm_stitched, index_maps);

    // Build fresh structures for comparison
    tf::face_membership<index_t> fm_fresh;
    fm_fresh.build(result.polygons());
    tf::manifold_edge_link<index_t, 3> mel_fresh;
    mel_fresh.build(result.faces(), fm_fresh);

    // Compare
    REQUIRE(mel_stitched.size() == mel_fresh.size());
    REQUIRE(manifold_edge_links_equal(mel_stitched, mel_fresh));
}

// =============================================================================
// Test 5: stitched_face_membership - union operation
// =============================================================================

TEMPLATE_TEST_CASE("stitched_face_membership_union", "[topology][stitching]",
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
    tf::aabb_tree<index_t, real_t, 3> tree0(box1.polygons(), tf::config_tree(4, 4));

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
        box1.polygons() | tf::tag(fm0) | tf::tag(mel0) | tf::tag(tree0),
        box2.polygons() | tf::tag(fm1) | tf::tag(mel1) | tf::tag(tree1) | tf::tag(frame),
        tf::boolean_op::merge, tf::return_index_map);

    // Build stitched face_membership
    auto fm_stitched = tf::stitched_face_membership(
        result.faces(), index_t(result.points().size()), fm0, fm1, index_maps);

    // Build fresh face_membership for comparison
    tf::face_membership<index_t> fm_fresh;
    fm_fresh.build(result.polygons());

    // Compare
    REQUIRE(fm_stitched.size() == fm_fresh.size());
    REQUIRE(face_memberships_equal(fm_stitched, fm_fresh));
}

// =============================================================================
// Test 6: stitched_manifold_edge_link - union operation
// =============================================================================

TEMPLATE_TEST_CASE("stitched_manifold_edge_link_union", "[topology][stitching]",
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
    tf::aabb_tree<index_t, real_t, 3> tree0(box1.polygons(), tf::config_tree(4, 4));

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
        box1.polygons() | tf::tag(fm0) | tf::tag(mel0) | tf::tag(tree0),
        box2.polygons() | tf::tag(fm1) | tf::tag(mel1) | tf::tag(tree1) | tf::tag(frame),
        tf::boolean_op::merge, tf::return_index_map);

    // Build stitched structures
    auto fm_stitched = tf::stitched_face_membership(
        result.faces(), index_t(result.points().size()), fm0, fm1, index_maps);
    auto mel_stitched = tf::stitched_manifold_edge_link(
        result.faces(), mel0, mel1, fm_stitched, index_maps);

    // Build fresh structures for comparison
    tf::face_membership<index_t> fm_fresh;
    fm_fresh.build(result.polygons());
    tf::manifold_edge_link<index_t, 3> mel_fresh;
    mel_fresh.build(result.faces(), fm_fresh);

    // Compare
    REQUIRE(mel_stitched.size() == mel_fresh.size());
    REQUIRE(manifold_edge_links_equal(mel_stitched, mel_fresh));
}

// =============================================================================
// Test 7: stitched structures - cylinder mesh
// =============================================================================

TEMPLATE_TEST_CASE("stitched_structures_cylinder", "[topology][stitching]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    // Create cylinder and sphere
    auto cylinder = tf::make_cylinder_mesh<index_t>(real_t(1), real_t(2), 30);
    auto sphere = tf::make_sphere_mesh<index_t>(real_t(0.3), 15, 15);

    // Build topology for cylinder
    tf::face_membership<index_t> fm0;
    fm0.build(cylinder.polygons());
    tf::manifold_edge_link<index_t, 3> mel0;
    mel0.build(cylinder.faces(), fm0);
    tf::aabb_tree<index_t, real_t, 3> tree0(cylinder.polygons(), tf::config_tree(4, 4));

    // Build topology for sphere
    tf::face_membership<index_t> fm1;
    fm1.build(sphere.polygons());
    tf::manifold_edge_link<index_t, 3> mel1;
    mel1.build(sphere.faces(), fm1);
    tf::aabb_tree<index_t, real_t, 3> tree1(sphere.polygons(), tf::config_tree(4, 4));

    // Position sphere on side of cylinder
    auto frame = tf::make_frame(tf::make_transformation_from_translation(
        tf::vector<real_t, 3>{real_t(1), real_t(0), real_t(0)}));

    // Do the boolean (left difference)
    auto [result, labels, index_maps] = tf::make_boolean(
        cylinder.polygons() | tf::tag(fm0) | tf::tag(mel0) | tf::tag(tree0),
        sphere.polygons() | tf::tag(fm1) | tf::tag(mel1) | tf::tag(tree1) | tf::tag(frame),
        tf::boolean_op::left_difference, tf::return_index_map);

    // Build stitched structures
    auto fm_stitched = tf::stitched_face_membership(
        result.faces(), index_t(result.points().size()), fm0, fm1, index_maps);
    auto mel_stitched = tf::stitched_manifold_edge_link(
        result.faces(), mel0, mel1, fm_stitched, index_maps);

    // Build fresh structures for comparison
    tf::face_membership<index_t> fm_fresh;
    fm_fresh.build(result.polygons());
    tf::manifold_edge_link<index_t, 3> mel_fresh;
    mel_fresh.build(result.faces(), fm_fresh);

    // Compare face_membership
    REQUIRE(fm_stitched.size() == fm_fresh.size());
    REQUIRE(face_memberships_equal(fm_stitched, fm_fresh));

    // Compare manifold_edge_link
    REQUIRE(mel_stitched.size() == mel_fresh.size());
    REQUIRE(manifold_edge_links_equal(mel_stitched, mel_fresh));
}

// =============================================================================
// Test 8: stitched structures - chained booleans
// =============================================================================

TEMPLATE_TEST_CASE("stitched_structures_chained_booleans", "[topology][stitching]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    // Create base box and two spheres
    auto box = tf::make_box_mesh<index_t>(real_t(2), real_t(2), real_t(2));
    auto sphere1 = tf::make_sphere_mesh<index_t>(real_t(0.4), 15, 15);
    auto sphere2 = tf::make_sphere_mesh<index_t>(real_t(0.4), 15, 15);

    // Build topology for box
    tf::face_membership<index_t> fm0;
    fm0.build(box.polygons());
    tf::manifold_edge_link<index_t, 3> mel0;
    mel0.build(box.faces(), fm0);
    tf::aabb_tree<index_t, real_t, 3> tree0(box.polygons(), tf::config_tree(4, 4));

    // Build topology for sphere1
    tf::face_membership<index_t> fm1;
    fm1.build(sphere1.polygons());
    tf::manifold_edge_link<index_t, 3> mel1;
    mel1.build(sphere1.faces(), fm1);
    tf::aabb_tree<index_t, real_t, 3> tree1(sphere1.polygons(), tf::config_tree(4, 4));

    // Position sphere1 at one corner
    auto frame1 = tf::make_frame(tf::make_transformation_from_translation(
        tf::vector<real_t, 3>{real_t(0.5), real_t(0.5), real_t(0.5)}));

    // First boolean
    auto [result1, labels1, index_maps1] = tf::make_boolean(
        box.polygons() | tf::tag(fm0) | tf::tag(mel0) | tf::tag(tree0),
        sphere1.polygons() | tf::tag(fm1) | tf::tag(mel1) | tf::tag(tree1) | tf::tag(frame1),
        tf::boolean_op::left_difference, tf::return_index_map);

    // Build stitched structures for first boolean
    auto fm_res1 = tf::stitched_face_membership(
        result1.faces(), index_t(result1.points().size()), fm0, fm1, index_maps1);
    auto mel_res1 = tf::stitched_manifold_edge_link(
        result1.faces(), mel0, mel1, fm_res1, index_maps1);

    // Build topology for sphere2
    tf::face_membership<index_t> fm2;
    fm2.build(sphere2.polygons());
    tf::manifold_edge_link<index_t, 3> mel2;
    mel2.build(sphere2.faces(), fm2);
    tf::aabb_tree<index_t, real_t, 3> tree2(sphere2.polygons(), tf::config_tree(4, 4));

    // Build tree for result1
    tf::aabb_tree<index_t, real_t, 3> tree_res1(result1.polygons(), tf::config_tree(4, 4));

    // Position sphere2 at opposite corner
    auto frame2 = tf::make_frame(tf::make_transformation_from_translation(
        tf::vector<real_t, 3>{real_t(-0.5), real_t(-0.5), real_t(-0.5)}));

    // Second boolean
    auto [result2, labels2, index_maps2] = tf::make_boolean(
        result1.polygons() | tf::tag(fm_res1) | tf::tag(mel_res1) | tf::tag(tree_res1),
        sphere2.polygons() | tf::tag(fm2) | tf::tag(mel2) | tf::tag(tree2) | tf::tag(frame2),
        tf::boolean_op::left_difference, tf::return_index_map);

    // Build stitched structures for second boolean
    auto fm_stitched = tf::stitched_face_membership(
        result2.faces(), index_t(result2.points().size()), fm_res1, fm2, index_maps2);
    auto mel_stitched = tf::stitched_manifold_edge_link(
        result2.faces(), mel_res1, mel2, fm_stitched, index_maps2);

    // Build fresh structures for comparison
    tf::face_membership<index_t> fm_fresh;
    fm_fresh.build(result2.polygons());
    tf::manifold_edge_link<index_t, 3> mel_fresh;
    mel_fresh.build(result2.faces(), fm_fresh);

    // Compare
    REQUIRE(fm_stitched.size() == fm_fresh.size());
    REQUIRE(face_memberships_equal(fm_stitched, fm_fresh));

    REQUIRE(mel_stitched.size() == mel_fresh.size());
    REQUIRE(manifold_edge_links_equal(mel_stitched, mel_fresh));
}
