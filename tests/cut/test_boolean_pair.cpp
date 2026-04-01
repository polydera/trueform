/**
 * @file test_boolean_pair.cpp
 * @brief Tests for make_boolean_pair
 *
 * Verifies that boolean pair produces correct halves by checking:
 * - Face count sum matches combined boolean
 * - No degenerate triangles
 * - Boundary curves match intersection curves structurally
 * - Curves overload returns valid curves
 *
 * Copyright (c) 2025 Ziga Sajovic, XLAB
 */

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <trueform/trueform.hpp>
#include "type_traits.hpp"

template <typename index_t, typename real_t>
struct tagged_mesh {
  tf::face_membership<index_t> fm;
  tf::manifold_edge_link<index_t, 3> mel;
  tf::aabb_tree<index_t, real_t, 3> tree;
};

template <typename index_t, typename real_t, typename Mesh>
auto build_tags(const Mesh &mesh) -> tagged_mesh<index_t, real_t> {
  tagged_mesh<index_t, real_t> t;
  t.fm.build(mesh.polygons());
  t.mel.build(mesh.faces(), t.fm);
  t.tree = tf::aabb_tree<index_t, real_t, 3>(mesh.polygons(),
                                              tf::config_tree(4, 4));
  return t;
}

template <typename Mesh>
auto count_degenerate(const Mesh &mesh) -> int {
  int n = 0;
  for (auto face : mesh.faces())
    if (face[0] == face[1] || face[1] == face[2] || face[0] == face[2])
      ++n;
  return n;
}

template <typename index_t, typename Mesh>
auto boundary_curve_sizes(const Mesh &mesh) -> std::vector<std::size_t> {
  auto be = tf::make_boundary_edges(mesh.polygons());
  auto paths = tf::connect_edges_to_paths(tf::make_edges(be));
  std::vector<std::size_t> sizes;
  for (auto path : paths)
    sizes.push_back(path.size());
  std::sort(sizes.begin(), sizes.end());
  return sizes;
}

// ============================================================================
// Overlapping spheres — faces sum, boundaries match intersection curves
// ============================================================================

TEMPLATE_TEST_CASE("boolean_pair_spheres_diff", "[boolean_pair]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
  using index_t = typename TestType::index_type;
  using real_t = typename TestType::real_type;

  auto s0 = tf::make_sphere_mesh<index_t>(real_t(1), 30, 30);
  auto s1 = tf::make_sphere_mesh<index_t>(real_t(0.8), 25, 25);
  auto t0 = build_tags<index_t, real_t>(s0);
  auto t1 = build_tags<index_t, real_t>(s1);

  auto frame = tf::make_frame(tf::make_transformation_from_translation(
      tf::vector<real_t, 3>{real_t(0.5), real_t(0), real_t(0)}));

  auto tagged0 = s0.polygons() | tf::tag(t0.fm) | tf::tag(t0.mel) |
                 tf::tag(t0.tree);
  auto tagged1 = s1.polygons() | tf::tag(t1.fm) | tf::tag(t1.mel) |
                 tf::tag(t1.tree) | tf::tag(frame);

  auto [combined, labels, fl_, curves] =
      tf::make_boolean(tagged0, tagged1,
                       tf::boolean_op::left_difference, tf::return_curves);

  auto [left, right, fl_left, fl_right] =
      tf::make_boolean_pair(tagged0, tagged1,
                            tf::boolean_op::left_difference);

  REQUIRE(count_degenerate(left) == 0);
  REQUIRE(count_degenerate(right) == 0);
  REQUIRE(fl_left.size() == left.faces().size());
  REQUIRE(fl_right.size() == right.faces().size());
  REQUIRE(left.faces().size() + right.faces().size() ==
          combined.faces().size());

  auto left_b = boundary_curve_sizes<index_t>(left);
  auto right_b = boundary_curve_sizes<index_t>(right);

  std::vector<std::size_t> ig_sizes;
  for (auto path : curves.paths())
    ig_sizes.push_back(path.size());
  std::sort(ig_sizes.begin(), ig_sizes.end());

  REQUIRE(left_b.size() == ig_sizes.size());
  REQUIRE(right_b.size() == ig_sizes.size());
  REQUIRE(left_b == ig_sizes);
  REQUIRE(right_b == ig_sizes);
}

// ============================================================================
// Overlapping spheres — union
// ============================================================================

TEMPLATE_TEST_CASE("boolean_pair_spheres_union", "[boolean_pair]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
  using index_t = typename TestType::index_type;
  using real_t = typename TestType::real_type;

  auto s0 = tf::make_sphere_mesh<index_t>(real_t(1), 25, 25);
  auto s1 = tf::make_sphere_mesh<index_t>(real_t(0.7), 20, 20);
  auto t0 = build_tags<index_t, real_t>(s0);
  auto t1 = build_tags<index_t, real_t>(s1);

  auto frame = tf::make_frame(tf::make_transformation_from_translation(
      tf::vector<real_t, 3>{real_t(0.5), real_t(0), real_t(0)}));

  auto tagged0 = s0.polygons() | tf::tag(t0.fm) | tf::tag(t0.mel) |
                 tf::tag(t0.tree);
  auto tagged1 = s1.polygons() | tf::tag(t1.fm) | tf::tag(t1.mel) |
                 tf::tag(t1.tree) | tf::tag(frame);

  auto [combined, labels, fl_] =
      tf::make_boolean(tagged0, tagged1, tf::boolean_op::merge);

  auto [left, right, fl_left, fl_right] =
      tf::make_boolean_pair(tagged0, tagged1, tf::boolean_op::merge);

  REQUIRE(count_degenerate(left) == 0);
  REQUIRE(count_degenerate(right) == 0);
  REQUIRE(fl_left.size() == left.faces().size());
  REQUIRE(fl_right.size() == right.faces().size());
  REQUIRE(left.faces().size() + right.faces().size() ==
          combined.faces().size());
}

// ============================================================================
// Curves overload returns valid curves
// ============================================================================

TEMPLATE_TEST_CASE("boolean_pair_with_curves", "[boolean_pair]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
  using index_t = typename TestType::index_type;
  using real_t = typename TestType::real_type;

  auto s0 = tf::make_sphere_mesh<index_t>(real_t(1), 25, 25);
  auto s1 = tf::make_sphere_mesh<index_t>(real_t(0.7), 20, 20);
  auto t0 = build_tags<index_t, real_t>(s0);
  auto t1 = build_tags<index_t, real_t>(s1);

  auto frame = tf::make_frame(tf::make_transformation_from_translation(
      tf::vector<real_t, 3>{real_t(0.5), real_t(0), real_t(0)}));

  auto tagged0 = s0.polygons() | tf::tag(t0.fm) | tf::tag(t0.mel) |
                 tf::tag(t0.tree);
  auto tagged1 = s1.polygons() | tf::tag(t1.fm) | tf::tag(t1.mel) |
                 tf::tag(t1.tree) | tf::tag(frame);

  auto [left, right, fl_left, fl_right, curves] =
      tf::make_boolean_pair(tagged0, tagged1,
                            tf::boolean_op::left_difference, tf::return_curves);

  REQUIRE(count_degenerate(left) == 0);
  REQUIRE(count_degenerate(right) == 0);
  REQUIRE(curves.paths().size() > 0);
  REQUIRE(curves.points().size() > 0);

  std::vector<std::size_t> curve_sizes;
  for (auto path : curves.paths())
    curve_sizes.push_back(path.size());
  std::sort(curve_sizes.begin(), curve_sizes.end());

  auto left_b = boundary_curve_sizes<index_t>(left);
  REQUIRE(left_b == curve_sizes);
}

// ============================================================================
// Box + sphere at corner — actual intersection
// ============================================================================

TEMPLATE_TEST_CASE("boolean_pair_box_sphere_corner", "[boolean_pair]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
  using index_t = typename TestType::index_type;
  using real_t = typename TestType::real_type;

  auto box = tf::make_box_mesh<index_t>(real_t(2), real_t(2), real_t(2));
  auto sphere = tf::make_sphere_mesh<index_t>(real_t(0.5), 20, 20);
  auto t0 = build_tags<index_t, real_t>(box);
  auto t1 = build_tags<index_t, real_t>(sphere);

  auto frame = tf::make_frame(tf::make_transformation_from_translation(
      tf::vector<real_t, 3>{real_t(1), real_t(1), real_t(1)}));

  auto tagged0 = box.polygons() | tf::tag(t0.fm) | tf::tag(t0.mel) |
                 tf::tag(t0.tree);
  auto tagged1 = sphere.polygons() | tf::tag(t1.fm) | tf::tag(t1.mel) |
                 tf::tag(t1.tree) | tf::tag(frame);

  auto [combined, labels, fl_, curves] =
      tf::make_boolean(tagged0, tagged1,
                       tf::boolean_op::left_difference, tf::return_curves);

  auto [left, right, fl_left, fl_right] =
      tf::make_boolean_pair(tagged0, tagged1,
                            tf::boolean_op::left_difference);

  REQUIRE(count_degenerate(left) == 0);
  REQUIRE(count_degenerate(right) == 0);
  REQUIRE(fl_left.size() == left.faces().size());
  REQUIRE(fl_right.size() == right.faces().size());
  REQUIRE(left.faces().size() + right.faces().size() ==
          combined.faces().size());

  REQUIRE(curves.paths().size() > 0);

  auto left_b = boundary_curve_sizes<index_t>(left);
  auto right_b = boundary_curve_sizes<index_t>(right);

  std::vector<std::size_t> ig_sizes;
  for (auto path : curves.paths())
    ig_sizes.push_back(path.size());
  std::sort(ig_sizes.begin(), ig_sizes.end());

  REQUIRE(left_b == ig_sizes);
  REQUIRE(right_b == ig_sizes);
}
