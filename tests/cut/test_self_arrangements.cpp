/**
 * @file test_self_arrangements.cpp
 * @brief Tests for self-intersection: embedded_self_intersection_curves
 *        and make_polygon_arrangements produce same results as
 *        make_mesh_arrangements on equivalent geometry.
 *
 * Copyright (c) 2025 Ziga Sajovic, XLAB
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <set>
#include <trueform/trueform.hpp>
#include "type_traits.hpp"

template <typename Paths, typename index_t>
auto count_endpoints(const Paths &paths) -> int {
  std::set<index_t> eps;
  for (auto path : paths) {
    if (path.size() < 2)
      continue;
    if (path[0] != path[path.size() - 1]) {
      eps.insert(path[0]);
      eps.insert(path[path.size() - 1]);
    }
  }
  return static_cast<int>(eps.size());
}

template <typename Mesh>
auto count_degenerate(const Mesh &mesh) -> int {
  int n = 0;
  for (auto face : mesh.faces())
    if (face[0] == face[1] || face[1] == face[2] || face[0] == face[2])
      ++n;
  return n;
}

TEMPLATE_TEST_CASE("Self arrangements: 3 spheres match mesh arrangements",
                   "[self_arrangements]",
                   (tf::test::type_pair<std::int32_t, float>),
                   (tf::test::type_pair<std::int64_t, double>)) {
  using index_t = typename TestType::index_type;
  using real_t = typename TestType::real_type;

  auto s0 = tf::make_sphere_mesh<index_t>(real_t(1), 30, 30);
  auto s1 = tf::make_sphere_mesh<index_t>(real_t(1), 30, 30);
  auto s2 = tf::make_sphere_mesh<index_t>(real_t(1), 30, 30);

  auto f0 = tf::make_frame(tf::make_transformation_from_translation(
      tf::vector<real_t, 3>{real_t(0.5), real_t(0), real_t(0)}));
  auto f1 = tf::make_frame(tf::make_transformation_from_translation(
      tf::vector<real_t, 3>{real_t(-0.5), real_t(0), real_t(0)}));
  auto f2 = tf::make_frame(tf::make_transformation_from_translation(
      tf::vector<real_t, 3>{real_t(0), real_t(0.5), real_t(0)}));

  auto p0 = s0.polygons() | tf::tag(f0);
  auto p1 = s1.polygons() | tf::tag(f1);
  auto p2 = s2.polygons() | tf::tag(f2);
  decltype(p0) forms[] = {p0, p1, p2};
  auto merged = tf::concatenated(tf::make_range(forms, forms + 3));

  // Reference: make_mesh_arrangements
  auto [ref_mesh, ref_tags, ref_faces, ref_curves] =
      tf::make_mesh_arrangements(tf::make_range(forms, forms + 3),
                                 tf::return_curves);

  SECTION("embedded_self_intersection_curves") {
    auto [mesh, fl_, curves] = tf::embedded_self_intersection_curves(
        merged.polygons(), tf::return_curves);

    REQUIRE(count_degenerate(mesh) == 0);
    REQUIRE(mesh.faces().size() == ref_mesh.faces().size());
    REQUIRE(mesh.points().size() == ref_mesh.points().size());
    REQUIRE(curves.paths().size() == ref_curves.paths().size());
    REQUIRE(count_endpoints<decltype(curves.paths()), index_t>(curves.paths()) ==
            count_endpoints<decltype(ref_curves.paths()), index_t>(
                ref_curves.paths()));
  }

  SECTION("make_polygon_arrangements") {
    auto [mesh, face_labels, curves] =
        tf::make_polygon_arrangements(merged.polygons(), tf::return_curves);

    REQUIRE(count_degenerate(mesh) == 0);
    REQUIRE(mesh.faces().size() == ref_mesh.faces().size());
    REQUIRE(mesh.points().size() == ref_mesh.points().size());
    REQUIRE(face_labels.size() == mesh.faces().size());
    REQUIRE(curves.paths().size() == ref_curves.paths().size());
    REQUIRE(count_endpoints<decltype(curves.paths()), index_t>(curves.paths()) ==
            count_endpoints<decltype(ref_curves.paths()), index_t>(
                ref_curves.paths()));
  }
}

TEMPLATE_TEST_CASE("Self arrangements: 3 spheres - 6 curves, 2 endpoints",
                   "[self_arrangements]",
                   (tf::test::type_pair<std::int32_t, float>),
                   (tf::test::type_pair<std::int64_t, double>)) {
  using index_t = typename TestType::index_type;
  using real_t = typename TestType::real_type;

  auto s0 = tf::make_sphere_mesh<index_t>(real_t(1), 30, 30);
  auto s1 = tf::make_sphere_mesh<index_t>(real_t(1), 30, 30);
  auto s2 = tf::make_sphere_mesh<index_t>(real_t(1), 30, 30);

  auto f0 = tf::make_frame(tf::make_transformation_from_translation(
      tf::vector<real_t, 3>{real_t(0.5), real_t(0), real_t(0)}));
  auto f1 = tf::make_frame(tf::make_transformation_from_translation(
      tf::vector<real_t, 3>{real_t(-0.5), real_t(0), real_t(0)}));
  auto f2 = tf::make_frame(tf::make_transformation_from_translation(
      tf::vector<real_t, 3>{real_t(0), real_t(0.5), real_t(0)}));

  auto p0 = s0.polygons() | tf::tag(f0);
  auto p1 = s1.polygons() | tf::tag(f1);
  auto p2 = s2.polygons() | tf::tag(f2);
  decltype(p0) forms[] = {p0, p1, p2};
  auto merged = tf::concatenated(tf::make_range(forms, forms + 3));

  auto [mesh, face_labels, curves] =
      tf::make_polygon_arrangements(merged.polygons(), tf::return_curves);

  REQUIRE(count_degenerate(mesh) == 0);
  REQUIRE(curves.paths().size() == 6);
  REQUIRE(count_endpoints<decltype(curves.paths()), index_t>(curves.paths()) ==
          2);
}
