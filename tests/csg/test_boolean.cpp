/**
 * @file test_boolean.cpp
 * @brief The public pairwise wrapper tf::make_boolean.
 *
 * One end-to-end scenario over two overlapping unit boxes, exercising
 * all four boolean_op values and the shape of the tuple the wrapper
 * returns. The solid scenarios themselves — bicylinder, nested and
 * overlapping spheres, non-overlapping and multi-component bodies — are
 * answered against the engine in tests/csg/test_csg_solids.cpp, which is
 * the one path that owns those facts.
 *
 * Copyright (c) 2025 Ziga Sajovic, XLAB
 */

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <trueform/trueform.hpp>

#include "csg_readers.hpp"
#include "mesh_generators.hpp"
#include "tagged_operand.hpp"
#include "type_traits.hpp"
#include <algorithm>
#include <cmath>

TEMPLATE_TEST_CASE("boolean_overlapping_boxes", "[boolean]",
    (tf::test::type_pair_dyn2<std::int32_t, float, false, false>),
    (tf::test::type_pair_dyn2<std::int32_t, float, true, false>),
    (tf::test::type_pair_dyn2<std::int32_t, float, false, true>),
    (tf::test::type_pair_dyn2<std::int32_t, float, true, true>),
    (tf::test::type_pair_dyn2<std::int64_t, double, false, false>),
    (tf::test::type_pair_dyn2<std::int64_t, double, true, false>),
    (tf::test::type_pair_dyn2<std::int64_t, double, false, true>),
    (tf::test::type_pair_dyn2<std::int64_t, double, true, true>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;
    constexpr bool dyn1 = TestType::is_dynamic1;
    constexpr bool dyn2 = TestType::is_dynamic2;

    // Two unit boxes, second translated by (0.5, 0, 0)
    auto box1 = tf::test::maybe_as_dynamic<dyn1>(
        tf::make_box_mesh<index_t>(real_t(1), real_t(1), real_t(1)));
    tf::ensure_positive_orientation(box1.polygons());

    auto box2 = tf::test::maybe_as_dynamic<dyn2>(
        tf::make_box_mesh<index_t>(real_t(1), real_t(1), real_t(1)));
    tf::ensure_positive_orientation(box2.polygons());

    auto box2_transform = tf::make_transformation_from_translation(
        tf::make_vector(real_t(0.5), real_t(0), real_t(0)));

    auto op1 = tf::test::make_tagged_operand(box1);
    auto op2 = tf::test::make_tagged_operand(
        box2, tf::transformation<real_t, 3>(box2_transform));

    // Box volumes: each is 1 cubic unit
    real_t box_volume = real_t(1);
    // Overlap volume: 0.5 * 1 * 1 = 0.5 cubic units
    real_t overlap_volume = real_t(0.5);

    // Test merge (union)
    {
      auto [merged, labels, fl_] =
          tf::test::boolean_of(op1.form(), op2.form(), tf::boolean_op::merge);

      auto boundaries = tf::make_boundary_paths(merged.polygons());
      auto non_manifold = tf::make_non_manifold_edges(merged.polygons());

      REQUIRE(boundaries.size() == 0);
      REQUIRE(non_manifold.size() == 0);

      // Union volume = box1 + box2 - overlap
      real_t merged_volume = tf::signed_volume(merged.polygons());
      real_t expected = box_volume + box_volume - overlap_volume;

      REQUIRE(std::abs(merged_volume - expected) <
              std::max(tf::epsilon<real_t>, real_t(0.01)));
    }

    // Test intersection
    {
      auto [intersection, labels, fl_] = tf::test::boolean_of(
          op1.form(), op2.form(), tf::boolean_op::intersection);

      auto boundaries = tf::make_boundary_paths(intersection.polygons());
      auto non_manifold = tf::make_non_manifold_edges(intersection.polygons());

      REQUIRE(boundaries.size() == 0);
      REQUIRE(non_manifold.size() == 0);

      // Intersection volume = overlap
      real_t intersection_volume = tf::signed_volume(intersection.polygons());

      REQUIRE(std::abs(intersection_volume - overlap_volume) <
              std::max(tf::epsilon<real_t>, real_t(0.01)));
    }

    // Test left difference (box1 - box2)
    {
      auto [diff, labels, fl_] = tf::test::boolean_of(
          op1.form(), op2.form(), tf::boolean_op::left_difference);

      auto boundaries = tf::make_boundary_paths(diff.polygons());
      auto non_manifold = tf::make_non_manifold_edges(diff.polygons());

      REQUIRE(boundaries.size() == 0);
      REQUIRE(non_manifold.size() == 0);

      // Left difference volume = box1 - overlap
      real_t diff_volume = tf::signed_volume(diff.polygons());
      real_t expected = box_volume - overlap_volume;

      REQUIRE(std::abs(diff_volume - expected) <
              std::max(tf::epsilon<real_t>, real_t(0.01)));
    }

    // Test right difference (box2 - box1)
    {
      auto [diff, labels, fl_] = tf::test::boolean_of(
          op1.form(), op2.form(), tf::boolean_op::right_difference);

      auto boundaries = tf::make_boundary_paths(diff.polygons());
      auto non_manifold = tf::make_non_manifold_edges(diff.polygons());

      REQUIRE(boundaries.size() == 0);
      REQUIRE(non_manifold.size() == 0);

      // Right difference volume = box2 - overlap
      real_t diff_volume = tf::signed_volume(diff.polygons());
      real_t expected = box_volume - overlap_volume;

      REQUIRE(std::abs(diff_volume - expected) <
              std::max(tf::epsilon<real_t>, real_t(0.01)));
    }
}
