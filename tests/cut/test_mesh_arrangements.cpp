/**
 * @file test_mesh_arrangements.cpp
 * @brief Tests for make_mesh_arrangements with N meshes
 *
 * Verifies structural properties of mesh arrangements:
 * - No degenerate triangles
 * - All tags present
 * - Intersection curves match non-manifold paths
 * - Curve count and endpoint count for known configurations
 *
 * Copyright (c) 2025 Ziga Sajovic, XLAB
 */

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <set>
#include <trueform/trueform.hpp>
#include "type_traits.hpp"

template <typename index_t, typename Paths>
auto get_sorted_sizes(const Paths &paths) -> std::vector<std::size_t> {
  std::vector<std::size_t> sizes;
  for (auto path : paths)
    sizes.push_back(path.size());
  std::sort(sizes.begin(), sizes.end());
  return sizes;
}

template <typename index_t, typename Paths>
auto count_open_endpoints(const Paths &paths) -> int {
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

TEMPLATE_TEST_CASE("mesh_arrangements_3_spheres", "[arrangements]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
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

  auto [mesh, tag_labels, face_labels, curves] =
      tf::make_mesh_arrangements(tf::make_range(forms, forms + 3),
                                 tf::return_curves);

  REQUIRE(count_degenerate(mesh) == 0);

  int tag_counts[3] = {};
  for (auto t : tag_labels)
    tag_counts[t]++;
  REQUIRE(tag_counts[0] > 0);
  REQUIRE(tag_counts[1] > 0);
  REQUIRE(tag_counts[2] > 0);

  auto nmedges = tf::make_non_manifold_edges(mesh.polygons());
  auto nm_paths = tf::connect_edges_to_paths(tf::make_edges(nmedges));
  auto ig_sizes = get_sorted_sizes<index_t>(curves.paths());
  auto nm_sizes = get_sorted_sizes<index_t>(nm_paths);
  REQUIRE(ig_sizes.size() == nm_sizes.size());
  REQUIRE(ig_sizes == nm_sizes);

  // 3 pairwise intersections split at 2 triple points → 6 curves, 2 endpoints
  REQUIRE(curves.paths().size() == 6);
  REQUIRE(count_open_endpoints<index_t>(curves.paths()) == 2);
}

TEMPLATE_TEST_CASE("mesh_arrangements_2_spheres", "[arrangements]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
  using index_t = typename TestType::index_type;
  using real_t = typename TestType::real_type;

  auto s0 = tf::make_sphere_mesh<index_t>(real_t(1), 25, 25);
  auto s1 = tf::make_sphere_mesh<index_t>(real_t(0.8), 20, 20);

  auto frame = tf::make_frame(tf::make_transformation_from_translation(
      tf::vector<real_t, 3>{real_t(0.5), real_t(0), real_t(0)}));

  auto [mesh, tag_labels, face_labels, curves] = tf::make_mesh_arrangements(
      s0.polygons(), s1.polygons() | tf::tag(frame), tf::return_curves);

  REQUIRE(count_degenerate(mesh) == 0);

  int tag_counts[2] = {};
  for (auto t : tag_labels)
    tag_counts[t]++;
  REQUIRE(tag_counts[0] > 0);
  REQUIRE(tag_counts[1] > 0);

  auto nmedges = tf::make_non_manifold_edges(mesh.polygons());
  auto nm_paths = tf::connect_edges_to_paths(tf::make_edges(nmedges));
  auto ig_sizes = get_sorted_sizes<index_t>(curves.paths());
  auto nm_sizes = get_sorted_sizes<index_t>(nm_paths);
  REQUIRE(ig_sizes.size() == nm_sizes.size());
  REQUIRE(ig_sizes == nm_sizes);

  // 2 overlapping spheres → 1 closed intersection curve, 0 endpoints
  REQUIRE(curves.paths().size() == 1);
  REQUIRE(count_open_endpoints<index_t>(curves.paths()) == 0);
}
