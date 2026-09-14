/*
 * Copyright (c) 2025 XLAB
 * All rights reserved.
 *
 * This file is part of trueform (trueform.polydera.com)
 *
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0.
 * Commercial licensing available via info@polydera.com.
 *
 * Author: Žiga Sajovic
 */
#include "carriers.hpp"
#include "fixtures.hpp"
#include "trueform/core/polygons_buffer.hpp"
#include "trueform/cpp/core/mesh.hpp"
#include "trueform/cpp/geometry/make_box_mesh.hpp"
#include "trueform/cpp/topology.hpp"
#include "trueform/cpp/topology/cdt.hpp"
#include "trueform/cpp/topology/domain_labels.hpp"

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <future>
#include <initializer_list>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace {

struct counting_resolver {
  std::shared_ptr<std::atomic<int>> submissions;

  template <typename T>
  using state_type = tf::cpp::async::detail::future_state<T>;

  template <typename T>
  auto make_state() const -> std::shared_ptr<state_type<T>> {
    submissions->fetch_add(1, std::memory_order_relaxed);
    return std::make_shared<state_type<T>>();
  }
};

template <typename T>
auto make_array(std::initializer_list<T> values, tf::small_vector<int, 3> shape)
    -> tf::cpp::nd_array<T> {
  tf::buffer<T> buffer;
  buffer.allocate(values.size());
  auto output = buffer.begin();
  for (const auto value : values)
    *output++ = value;
  return tf::cpp::nd_array<T>::from_buffer(std::move(buffer), std::move(shape));
}

template <typename Real> auto cdt_points() -> tf::cpp::nd_array<Real> {
  return make_array<Real>({Real{0}, Real{0}, Real{2}, Real{0}, Real{2}, Real{2},
                           Real{0}, Real{2}, Real{1}, Real{1}},
                          {5, 2});
}

auto cdt_edges() -> tf::cpp::nd_array<std::int32_t> {
  return make_array<std::int32_t>({0, 1, 1, 2, 2, 3, 3, 0}, {4, 2});
}

template <typename T>
auto same_array(const tf::cpp::nd_array<T> &a, const tf::cpp::nd_array<T> &b)
    -> bool {
  if (a.raw_shape() != b.raw_shape())
    return false;
  for (std::size_t index = 0; index < a.length(); ++index)
    if (a[index] != b[index])
      return false;
  return true;
}

} // namespace

TEMPLATE_TEST_CASE("topology queries and extraction preserve native shapes",
                   "[cpp][topology][queries]", float, double) {
  auto closed = tf::cpp::test::tetrahedron_mesh<TestType>();
  auto open = tf::cpp::test::two_triangle_mesh<TestType>();

  CHECK(tf::cpp::is_closed(closed.mesh()));
  CHECK_FALSE(tf::cpp::is_open(closed.mesh()));
  CHECK(tf::cpp::is_manifold(closed.mesh()));
  CHECK_FALSE(tf::cpp::is_non_manifold(closed.mesh()));
  CHECK(tf::cpp::euler_characteristic(closed.mesh()) == 2);

  CHECK_FALSE(tf::cpp::is_closed(open.mesh()));
  CHECK(tf::cpp::is_open(open.mesh()));
  CHECK(tf::cpp::euler_characteristic(open.mesh()) == 1);
  CHECK(tf::cpp::boundary_edges(closed.mesh()).raw_shape() ==
        tf::small_vector<int, 3>{0, 2});
  CHECK(tf::cpp::boundary_edges(open.mesh()).raw_shape() ==
        tf::small_vector<int, 3>{4, 2});
  CHECK(tf::cpp::non_manifold_edges(open.mesh()).raw_shape() ==
        tf::small_vector<int, 3>{0, 2});

  auto paths = tf::cpp::boundary_paths(open.mesh());
  REQUIRE(paths.size() == 1);
  CHECK(paths.get(0).length() == 5);
  CHECK(paths.get(0)[0] == paths.get(0)[4]);

  auto rings = tf::cpp::k_rings(open.mesh(), 1);
  auto inclusive = tf::cpp::k_rings(open.mesh(), 1, true);
  REQUIRE(rings.size() == 4);
  REQUIRE(inclusive.size() == 4);
  CHECK(inclusive.get(2).length() == rings.get(2).length() + 1);
  CHECK(std::find(inclusive.get(2).begin(), inclusive.get(2).end(), 2) !=
        inclusive.get(2).end());

  auto neighbors = tf::cpp::neighborhoods(open.mesh(), TestType{1.01});
  REQUIRE(neighbors.size() == 4);
  CHECK(neighbors.get(0).length() >= 2);
}

TEST_CASE("non-manifold edges and connected component overloads are typed",
          "[cpp][topology][components]") {
  tf::cpp::test::owned_mesh<tf::cpp::default_index_t, float> non_manifold{
      tf::cpp::test::polygons_of<tf::cpp::default_index_t, float>(
          {0, 1, 2, 1, 0, 3, 0, 1, 4},
          {0, 0, 0, 1, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 1})};
  CHECK(tf::cpp::is_non_manifold(non_manifold.mesh()));
  CHECK_FALSE(tf::cpp::is_manifold(non_manifold.mesh()));
  CHECK(tf::cpp::non_manifold_edges(non_manifold.mesh()).raw_shape() ==
        tf::small_vector<int, 3>{1, 2});

  auto fixed = make_array<std::int32_t>({1, -1, 0, -1, 3, -1, 2, -1}, {4, 2});
  auto fixed_result = tf::cpp::label_connected_components(fixed);
  CHECK(fixed_result.n_components == 2);
  CHECK(fixed_result.labels.raw_shape() == tf::small_vector<int, 3>{4});
  CHECK(fixed_result.labels[0] == fixed_result.labels[1]);
  CHECK(fixed_result.labels[0] != fixed_result.labels[2]);

  auto offsets = make_array<std::int32_t>({0, 1, 2, 3, 4}, {5});
  auto data = make_array<std::int32_t>({1, 0, 3, 2}, {4});
  auto variable =
      tf::cpp::offset_blocked_buffer<std::int32_t, std::int32_t>::create(
          offsets, data);
  auto variable_result = tf::cpp::label_connected_components(variable);
  CHECK(variable_result.n_components == 2);
  CHECK(same_array(fixed_result.labels, variable_result.labels));
}

TEST_CASE("edge paths cover loops, chains, and valid empty input",
          "[cpp][topology][paths][empty]") {
  auto edges = make_array<std::int32_t>({0, 1, 1, 2, 2, 0, 7, 8}, {4, 2});
  auto paths = tf::cpp::connect_edges_to_paths(edges);
  CHECK(paths.size() == 2);

  auto empty_edges = make_array<std::int32_t>({}, {0, 2});
  auto empty_paths = tf::cpp::connect_edges_to_paths(empty_edges);
  CHECK(empty_paths.is_valid());
  CHECK(empty_paths.size() == 0);

  auto empty_connectivity = make_array<std::int32_t>({}, {0, 2});
  auto components = tf::cpp::label_connected_components(empty_connectivity);
  CHECK(components.n_components == 0);
  CHECK(components.labels.is_valid());
  CHECK(components.labels.raw_shape() == tf::small_vector<int, 3>{0});
}

TEMPLATE_TEST_CASE("consistent orientation states storage of its own and the "
                   "frame stays with the reading",
                   "[cpp][topology][orientation][transform]", float, double) {
  auto source = tf::cpp::test::two_triangle_mesh<TestType>();
  auto &indices = source.polygons.faces_buffer().data_buffer();
  indices[4] = 3;
  indices[5] = 2;
  source.cache.faces_changed();
  source.place({1, 0, 0, 20, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1});

  auto result = tf::cpp::orient_faces_consistently(source.mesh());
  CHECK(result.faces_buffer().size() == source.polygons.faces_buffer().size());
  CHECK(result.points_buffer().size() ==
        source.polygons.points_buffer().size());
  CHECK(result.faces_buffer().data_buffer().begin() != indices.begin());

  // the neighbourhood is metric in the reading's own coordinates, so the
  // placement it is read at does not change it
  auto before = tf::cpp::neighborhoods(source.mesh(), TestType{1.01});
  auto after =
      tf::cpp::neighborhoods(source.mesh().at_identity(), TestType{1.01});
  CHECK(same_array(before.offsets(), after.offsets()));
  CHECK(same_array(before.data(), after.data()));
}

TEMPLATE_TEST_CASE("CDT preserves precision, maps, masks, and determinism",
                   "[cpp][topology][cdt]", float, double) {
  auto points = cdt_points<TestType>();
  auto edges = cdt_edges();
  auto mask = make_array<std::int8_t>({1, 1, 1, 1}, {4});

  auto hull = tf::cpp::make_cdt(points);
  auto constrained = tf::cpp::make_cdt(points, edges);
  auto mapped = tf::cpp::make_cdt_with_maps(points, edges);
  auto masked = tf::cpp::make_cdt(points, edges, mask);
  auto mapped_masked = tf::cpp::make_cdt_with_maps(points, edges, mask);

  CHECK(hull.faces.shape_at(1) == 3);
  CHECK(hull.points.raw_shape() == tf::small_vector<int, 3>{5, 2});
  CHECK(constrained.faces.shape_at(1) == 3);
  CHECK(same_array(constrained.faces, masked.faces));
  CHECK(same_array(constrained.points, masked.points));
  CHECK(mapped.point_index_map.f.shape_at(0) == points.shape_at(0));
  CHECK(mapped.point_index_map.kept_ids.shape_at(0) ==
        mapped.points.shape_at(0));
  CHECK(same_array(mapped.faces, mapped_masked.faces));

  auto repeated = tf::cpp::make_cdt(points, edges);
  CHECK(same_array(constrained.faces, repeated.faces));
  CHECK(same_array(constrained.points, repeated.points));

  auto false_mask = make_array<std::int8_t>({0, 0, 0, 0}, {4});
  auto no_interior = tf::cpp::make_cdt(points, edges, false_mask);
  CHECK(no_interior.faces.raw_shape() == tf::small_vector<int, 3>{0, 3});
  CHECK(no_interior.points.raw_shape() == tf::small_vector<int, 3>{0, 2});
}

TEMPLATE_TEST_CASE("CDT owns valid empty and degenerate results",
                   "[cpp][topology][cdt][empty][degenerate]", float, double) {
  auto empty = make_array<TestType>({}, {0, 2});
  auto empty_result = tf::cpp::make_cdt(empty);
  auto empty_mapped = tf::cpp::make_cdt_with_maps(empty);
  CHECK(empty_result.faces.raw_shape() == tf::small_vector<int, 3>{0, 3});
  CHECK(empty_result.points.raw_shape() == tf::small_vector<int, 3>{0, 2});
  CHECK(empty_mapped.point_index_map.f.raw_shape() ==
        tf::small_vector<int, 3>{0});

  auto collinear = make_array<TestType>({0, 0, 1, 0, 2, 0}, {3, 2});
  auto degenerate = tf::cpp::make_cdt(collinear);
  CHECK(degenerate.faces.raw_shape() == tf::small_vector<int, 3>{0, 3});
  CHECK(degenerate.points.shape_at(1) == 2);

  auto crossing_points = make_array<TestType>({0, 0, 1, 1, 0, 1, 1, 0}, {4, 2});
  auto crossing_edges = make_array<std::int32_t>({0, 1, 2, 3}, {2, 2});
  auto crossing_mask = make_array<std::int8_t>({1, 1}, {2});
  auto unsplit = tf::cpp::make_cdt(crossing_points, crossing_edges, false);
  auto unsplit_mapped =
      tf::cpp::make_cdt_with_maps(crossing_points, crossing_edges, false);
  auto masked_unsplit =
      tf::cpp::make_cdt(crossing_points, crossing_edges, crossing_mask, false);
  auto masked_unsplit_mapped = tf::cpp::make_cdt_with_maps(
      crossing_points, crossing_edges, crossing_mask, false);
  CHECK(unsplit.faces.empty());
  CHECK(unsplit.points.empty());
  CHECK(unsplit_mapped.faces.empty());
  CHECK(unsplit_mapped.points.empty());
  CHECK(unsplit_mapped.point_index_map.f.raw_shape() ==
        tf::small_vector<int, 3>{4});
  CHECK(unsplit_mapped.point_index_map.kept_ids.empty());
  for (const auto mapped : unsplit_mapped.point_index_map.f)
    CHECK(mapped == 4);
  CHECK(masked_unsplit.faces.empty());
  CHECK(masked_unsplit.points.empty());
  CHECK(masked_unsplit_mapped.faces.empty());
  CHECK(masked_unsplit_mapped.points.empty());
  CHECK(masked_unsplit_mapped.point_index_map.f.raw_shape() ==
        tf::small_vector<int, 3>{4});
  CHECK(masked_unsplit_mapped.point_index_map.kept_ids.empty());
  for (const auto mapped : masked_unsplit_mapped.point_index_map.f)
    CHECK(mapped == 4);
}

TEMPLATE_TEST_CASE("domain labels expose range, ownership, and empty semantics",
                   "[cpp][topology][domains]", float, double) {
  tf::cpp::test::owned_mesh<tf::cpp::default_index_t, TestType> box{
      tf::cpp::make_box_mesh(TestType{2}, TestType{2}, TestType{2})};
  const auto face_count = static_cast<int>(box.polygons.faces_buffer().size());
  auto excluded = tf::cpp::make_domain_labels(
      box.mesh(), tf::domain_config::exclude_outer_shell);
  CHECK(excluded.number_of_faces() == face_count);
  CHECK(excluded.labels().raw_shape() ==
        tf::small_vector<int, 3>{face_count, 2});
  CHECK(excluded.number_of_domains() == 1);
  CHECK(excluded.valid_label_begin() == 0);
  CHECK(excluded.valid_label_end() == 1);
  CHECK(excluded.sentinel_label() == 1);
  CHECK(excluded.outer_shell_label() == excluded.sentinel_label());
  CHECK_FALSE(excluded.has_outer_shell_domain());
  CHECK_FALSE(excluded.empty());
  for (std::int32_t face = 0; face < excluded.number_of_faces(); ++face)
    for (std::int32_t side = 0; side < 2; ++side)
      CHECK(excluded.label(face, side) >= excluded.valid_label_begin());

  auto included = tf::cpp::make_domain_labels(box.mesh());
  CHECK(included.has_outer_shell_domain());
  CHECK(included.outer_shell_label() >= included.valid_label_begin());
  CHECK(included.outer_shell_label() < included.valid_label_end());

  box.place({2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1});
  auto transformed = tf::cpp::make_domain_labels(
      box.mesh(), tf::domain_config::exclude_outer_shell);
  CHECK(same_array(excluded.labels(), transformed.labels()));

  auto no_faces = tf::cpp::test::empty_mesh<TestType>();
  auto empty = tf::cpp::make_domain_labels(no_faces.mesh());
  CHECK(empty.empty());
  CHECK(empty.number_of_faces() == 0);
  CHECK(empty.number_of_domains() == 0);
  CHECK(empty.sentinel_label() == 0);
  CHECK(empty.outer_shell_label() == -1);
  CHECK(empty.labels().raw_shape() == tf::small_vector<int, 3>{0, 2});
}

TEST_CASE("topology validates malformed arrays, indices, and options",
          "[cpp][topology][validation]") {
  auto open = tf::cpp::test::two_triangle_mesh<float>();
  CHECK_THROWS_AS(tf::cpp::k_rings(open.mesh(), 0), std::invalid_argument);
  CHECK_THROWS_AS(tf::cpp::k_rings(open.mesh(), -1), std::invalid_argument);
  CHECK_THROWS_AS(tf::cpp::neighborhoods(open.mesh(), 0.0F),
                  std::invalid_argument);
  CHECK_THROWS_AS(tf::cpp::neighborhoods(open.mesh(), -1.0F),
                  std::invalid_argument);

  auto bad_edges = make_array<std::int32_t>({0, 1, 2}, {3});
  CHECK_THROWS_AS(tf::cpp::connect_edges_to_paths(bad_edges),
                  std::invalid_argument);

  auto bad_connectivity = make_array<std::int32_t>({1, 4}, {1, 2});
  CHECK_THROWS_AS(tf::cpp::label_connected_components(bad_connectivity),
                  std::out_of_range);

  auto points = cdt_points<float>();
  auto wrong_points = points.reshape({10});
  CHECK_THROWS_AS(tf::cpp::make_cdt(wrong_points), std::invalid_argument);
  auto out_of_range_edges = make_array<std::int32_t>({0, 5}, {1, 2});
  CHECK_THROWS_AS(tf::cpp::make_cdt(points, out_of_range_edges),
                  std::out_of_range);
  auto edges = cdt_edges();
  auto bad_mask = make_array<std::int8_t>({1, 1}, {2});
  CHECK_THROWS_AS(tf::cpp::make_cdt(points, edges, bad_mask),
                  std::invalid_argument);

  CHECK_THROWS_AS(tf::cpp::make_domain_labels(
                      open.mesh(), static_cast<tf::domain_config>(4)),
                  std::invalid_argument);
  CHECK_THROWS_AS(tf::cpp::domain_labels_result(
                      make_array<std::int32_t>({2, 0}, {1, 2}), 1, -1),
                  std::out_of_range);
}

TEMPLATE_TEST_CASE("native edge paths reject negative point IDs",
                   "[cpp][topology][paths][validation][native-contract]",
                   std::int32_t, std::int64_t) {
  auto negative_edges = make_array<TestType>({-1, 2}, {1, 2});
  CHECK_THROWS_AS(tf::cpp::connect_edges_to_paths(negative_edges),
                  std::out_of_range);
}

TEST_CASE("all topology async APIs have exact futures, and an array-taking one "
          "owns its input",
          "[cpp][topology][async][ownership]") {
  auto owned = tf::cpp::test::two_triangle_mesh<float>();
  const auto mesh = owned.mesh();
  auto edges = cdt_edges();
  auto wide_edges = make_array<std::int64_t>({0, 1, 1, 2, 2, 3, 3, 0}, {4, 2});
  auto points = cdt_points<float>();
  auto mask = make_array<std::int8_t>({1, 1, 1, 1}, {4});
  auto connectivity = make_array<std::int32_t>({1, 0}, {2, 1});
  auto wide_connectivity = make_array<std::int64_t>({1, 0}, {2, 1});
  auto paths = tf::cpp::connect_edges_to_paths(edges);
  auto variable_connectivity = owned.cache.vertex_link_handle(mesh.geometry());
  auto wide_variable_connectivity =
      tf::cpp::offset_blocked_buffer<std::int64_t, std::int64_t>::create(
          make_array<std::int64_t>({0, 1, 2}, {3}),
          make_array<std::int64_t>({1, 0}, {2}));

  auto f0 = tf::cpp::async::is_closed(mesh);
  auto f1 = tf::cpp::async::is_open(mesh);
  auto f2 = tf::cpp::async::is_manifold(mesh);
  auto f3 = tf::cpp::async::is_non_manifold(mesh);
  auto f4 = tf::cpp::async::euler_characteristic(mesh);
  auto f5 = tf::cpp::async::boundary_edges(mesh);
  auto f6 = tf::cpp::async::non_manifold_edges(mesh);
  auto f7 = tf::cpp::async::boundary_paths(mesh);
  auto boundary_curve_future = tf::cpp::async::boundary_curves(mesh);
  auto f8 = tf::cpp::async::k_rings(mesh, 1);
  auto f9 = tf::cpp::async::neighborhoods(mesh, 2.0F);
  auto f10 = tf::cpp::async::connect_edges_to_paths(edges);
  auto wide_paths = tf::cpp::async::connect_edges_to_paths(wide_edges);
  auto f11 = tf::cpp::async::label_connected_components(connectivity);
  auto f12 = tf::cpp::async::label_connected_components(variable_connectivity);
  auto dense_hint =
      tf::cpp::async::label_connected_components(connectivity, 1000);
  auto variable_hint =
      tf::cpp::async::label_connected_components(variable_connectivity, 1000);
  auto wide_dense =
      tf::cpp::async::label_connected_components(wide_connectivity);
  auto wide_dense_hint =
      tf::cpp::async::label_connected_components(wide_connectivity, 1000);
  auto wide_variable =
      tf::cpp::async::label_connected_components(wide_variable_connectivity);
  auto wide_variable_hint = tf::cpp::async::label_connected_components(
      wide_variable_connectivity, 1000);
  auto f13 = tf::cpp::async::orient_faces_consistently(mesh);
  auto f14 = tf::cpp::async::make_cdt(points);
  auto f15 = tf::cpp::async::make_cdt_with_maps(points);
  auto f16 = tf::cpp::async::make_cdt(points, edges);
  auto f17 = tf::cpp::async::make_cdt_with_maps(points, edges);
  auto f18 = tf::cpp::async::make_cdt(points, edges, mask);
  auto f19 = tf::cpp::async::make_cdt_with_maps(points, edges, mask);
  tf::cpp::test::owned_mesh<tf::cpp::default_index_t, float> box{
      tf::cpp::make_box_mesh(1.0F, 1.0F, 1.0F)};
  auto f20 = tf::cpp::async::make_domain_labels(box.mesh());

  static_assert(std::is_same_v<decltype(f0), std::future<bool>>);
  static_assert(std::is_same_v<decltype(f4), std::future<std::int32_t>>);
  static_assert(std::is_same_v<decltype(f5),
                               std::future<tf::cpp::nd_array<std::int32_t>>>);
  static_assert(
      std::is_same_v<decltype(f7), std::future<tf::cpp::offset_blocked_buffer<
                                       std::int32_t, std::int32_t>>>);
  static_assert(
      std::is_same_v<decltype(boundary_curve_future),
                     std::future<tf::cpp::boundary_curves_result<std::int32_t,
                                                                 float, 3>>>);
  static_assert(
      std::is_same_v<decltype(f10), std::future<tf::cpp::offset_blocked_buffer<
                                        std::int32_t, std::int32_t>>>);
  static_assert(std::is_same_v<decltype(wide_paths),
                               std::future<tf::cpp::offset_blocked_buffer<
                                   std::int64_t, std::int64_t>>>);
  static_assert(
      std::is_same_v<decltype(f11),
                     std::future<tf::cpp::connected_components_result<>>>);
  static_assert(
      std::is_same_v<
          decltype(wide_dense),
          std::future<tf::cpp::connected_components_result<std::int64_t>>>);
  static_assert(
      std::is_same_v<decltype(wide_variable_hint), decltype(wide_dense)>);
  static_assert(
      std::is_same_v<decltype(f13),
                     std::future<tf::polygons_buffer<tf::cpp::default_index_t,
                                                     float, 3, 3>>>);
  static_assert(
      std::is_same_v<decltype(f14), std::future<tf::cpp::cdt_result<float>>>);
  static_assert(
      std::is_same_v<decltype(f15),
                     std::future<tf::cpp::cdt_result_with_map<float>>>);
  static_assert(std::is_same_v<decltype(f20),
                               std::future<tf::cpp::domain_labels_result<>>>);

  // an array is a handle, so the entry that takes one retains it and the
  // caller may drop its own; a mesh is the caller's own memory and stays alive
  // for as long as the call is pending
  edges.destroy();
  wide_edges.destroy();
  points.destroy();
  mask.destroy();
  connectivity.destroy();
  wide_connectivity.destroy();
  paths.destroy();
  variable_connectivity.destroy();
  wide_variable_connectivity.destroy();

  CHECK_FALSE(f0.get());
  CHECK(f1.get());
  CHECK(f2.get());
  CHECK_FALSE(f3.get());
  CHECK(f4.get() == 1);
  CHECK(f5.get().shape_at(0) == 4);
  CHECK(f6.get().empty());
  CHECK(f7.get().size() == 1);
  auto boundary_curve_result = boundary_curve_future.get();
  CHECK(boundary_curve_result.paths.size() == 1);
  CHECK(boundary_curve_result.points.raw_shape() ==
        tf::small_vector<int, 3>{4, 3});
  CHECK(f8.get().size() == 4);
  CHECK(f9.get().size() == 4);
  CHECK(f10.get().size() == 1);
  auto wide_path_result = wide_paths.get();
  CHECK(wide_path_result.size() == 1);
  CHECK(wide_path_result.data()[0] == std::int64_t{0});
  CHECK(wide_path_result.data()[4] == std::int64_t{0});
  CHECK(f11.get().n_components == 1);
  CHECK(f12.get().n_components == 1);
  CHECK(dense_hint.get().n_components == 1);
  CHECK(variable_hint.get().n_components == 1);
  CHECK(wide_dense.get().n_components == 1);
  CHECK(wide_dense_hint.get().n_components == 1);
  CHECK(wide_variable.get().n_components == 1);
  CHECK(wide_variable_hint.get().n_components == 1);
  CHECK(f13.get().faces_buffer().size() == 2);
  CHECK(f14.get().faces.shape_at(1) == 3);
  CHECK(f15.get().point_index_map.f.shape_at(0) == 5);
  CHECK(f16.get().faces.shape_at(1) == 3);
  CHECK(f17.get().point_index_map.f.shape_at(0) == 5);
  CHECK(f18.get().faces.shape_at(1) == 3);
  CHECK(f19.get().point_index_map.f.shape_at(0) == 5);
  CHECK(f20.get().number_of_faces() == 12);
}

TEST_CASE("custom resolvers submit once and async exceptions propagate",
          "[cpp][topology][async][resolver][exception]") {
  auto submissions = std::make_shared<std::atomic<int>>(0);
  auto points = cdt_points<double>();
  auto result =
      tf::cpp::async::make_cdt<double>(counting_resolver{submissions}, points);
  CHECK(submissions->load(std::memory_order_relaxed) == 1);
  CHECK(result.get().points.raw_shape() == tf::small_vector<int, 3>{5, 2});

  auto path_submissions = std::make_shared<std::atomic<int>>(0);
  auto path_resolver = counting_resolver{path_submissions};
  auto path_edges32 = make_array<std::int32_t>({0, 1, 1, 2}, {2, 2});
  auto path_edges64 = make_array<std::int64_t>({0, 1, 1, 2}, {2, 2});
  auto paths32 =
      tf::cpp::async::connect_edges_to_paths(path_resolver, path_edges32);
  auto paths64 =
      tf::cpp::async::connect_edges_to_paths(path_resolver, path_edges64);
  CHECK(path_submissions->load(std::memory_order_relaxed) == 2);
  path_edges32.destroy();
  path_edges64.destroy();
  CHECK(paths32.get().data().length() == 3);
  CHECK(paths64.get().data().length() == 3);

  auto boundary_submissions = std::make_shared<std::atomic<int>>(0);
  auto boundary_resolver = counting_resolver{boundary_submissions};
  tf::cpp::test::owned_mesh<std::int64_t, double, 2> typed_owner{
      tf::cpp::test::polygons_of<std::int64_t, double, 2>({0, 1, 2},
                                                          {0, 0, 1, 0, 0, 1})};
  const auto typed_mesh = typed_owner.mesh();
  auto typed_edges =
      tf::cpp::async::boundary_edges(boundary_resolver, typed_mesh);
  auto typed_paths =
      tf::cpp::async::boundary_paths(boundary_resolver, typed_mesh);
  auto typed_curves =
      tf::cpp::async::boundary_curves(boundary_resolver, typed_mesh);
  CHECK(boundary_submissions->load(std::memory_order_relaxed) == 3);
  static_assert(std::is_same_v<decltype(typed_edges),
                               std::future<tf::cpp::nd_array<std::int64_t>>>);
  static_assert(std::is_same_v<decltype(typed_paths),
                               std::future<tf::cpp::offset_blocked_buffer<
                                   std::int64_t, std::int64_t>>>);
  static_assert(
      std::is_same_v<decltype(typed_curves),
                     std::future<tf::cpp::boundary_curves_result<std::int64_t,
                                                                 double, 2>>>);
  CHECK(typed_edges.get().raw_shape() == tf::small_vector<int, 3>{3, 2});
  CHECK(typed_paths.get().size() == 1);
  auto owned_curves = typed_curves.get();
  CHECK(owned_curves.paths.size() == 1);
  CHECK(owned_curves.points.raw_shape() == tf::small_vector<int, 3>{3, 2});

  auto component_submissions = std::make_shared<std::atomic<int>>(0);
  auto resolver = counting_resolver{component_submissions};
  auto dense32 = make_array<std::int32_t>({1, 0}, {2, 1});
  auto dense64 = make_array<std::int64_t>({1, 0}, {2, 1});
  auto variable32 =
      tf::cpp::offset_blocked_buffer<std::int32_t, std::int32_t>::create(
          make_array<std::int32_t>({0, 1, 2}, {3}),
          make_array<std::int32_t>({1, 0}, {2}));
  auto variable64 =
      tf::cpp::offset_blocked_buffer<std::int64_t, std::int64_t>::create(
          make_array<std::int64_t>({0, 1, 2}, {3}),
          make_array<std::int64_t>({1, 0}, {2}));
  auto dense32_default =
      tf::cpp::async::label_connected_components(resolver, dense32);
  auto dense32_hint =
      tf::cpp::async::label_connected_components(resolver, dense32, 1000);
  auto variable32_default =
      tf::cpp::async::label_connected_components(resolver, variable32);
  auto variable32_hint =
      tf::cpp::async::label_connected_components(resolver, variable32, 1000);
  auto dense64_default =
      tf::cpp::async::label_connected_components(resolver, dense64);
  auto dense64_hint =
      tf::cpp::async::label_connected_components(resolver, dense64, 1000);
  auto variable64_default =
      tf::cpp::async::label_connected_components(resolver, variable64);
  auto variable64_hint =
      tf::cpp::async::label_connected_components(resolver, variable64, 1000);
  CHECK(component_submissions->load(std::memory_order_relaxed) == 8);
  dense32.destroy();
  dense64.destroy();
  variable32.destroy();
  variable64.destroy();
  CHECK(dense32_default.get().n_components == 1);
  CHECK(dense32_hint.get().n_components == 1);
  CHECK(variable32_default.get().n_components == 1);
  CHECK(variable32_hint.get().n_components == 1);
  CHECK(dense64_default.get().n_components == 1);
  CHECK(dense64_hint.get().n_components == 1);
  CHECK(variable64_default.get().n_components == 1);
  CHECK(variable64_hint.get().n_components == 1);

  auto invalid_hint_connectivity = make_array<std::int64_t>({-1}, {1, 1});
  auto invalid_hint =
      tf::cpp::async::label_connected_components(invalid_hint_connectivity, 0);
  CHECK_THROWS_AS(invalid_hint.get(), std::invalid_argument);

  auto bad_points = points.reshape({10});
  auto failed = tf::cpp::async::make_cdt(bad_points);
  CHECK_THROWS_AS(failed.get(), std::invalid_argument);

  // a caller holding nothing holds the EMPTY mesh, which is closed
  tf::cpp::test::owned_mesh<tf::cpp::default_index_t, double> nothing;
  CHECK(tf::cpp::async::is_closed(nothing.mesh()).get());
}

TEST_CASE("typed edge path overload addresses link from separate ABI lanes",
          "[cpp][topology][paths][archive-link][abi]") {
  using paths32 = tf::cpp::offset_blocked_buffer<std::int32_t, std::int32_t>;
  using paths64 = tf::cpp::offset_blocked_buffer<std::int64_t, std::int64_t>;

  auto (*legacy)(const tf::cpp::nd_array<std::int32_t> &)->paths32 =
      &tf::cpp::connect_edges_to_paths;
  auto (*wide)(const tf::cpp::nd_array<std::int64_t> &)->paths64 =
      &tf::cpp::connect_edges_to_paths;

  auto legacy_edges = make_array<std::int32_t>({0, 1, 1, 2}, {2, 2});
  auto wide_edges = make_array<std::int64_t>({0, 1, 1, 2}, {2, 2});
  CHECK(legacy(legacy_edges).data().length() == 3);
  CHECK(wide(wide_edges).data().length() == 3);
}

TEMPLATE_TEST_CASE("topology symbols link from the native archive",
                   "[cpp][topology][archive-link]", float, double) {
  using index_type = tf::cpp::default_index_t;
  using mesh_type = tf::cpp::mesh<index_type, TestType, 3>;
  using paths_type = tf::cpp::offset_blocked_buffer<index_type, index_type>;
  using curves_type = tf::cpp::boundary_curves_result<index_type, TestType, 3>;
  using polygons_type = tf::polygons_buffer<index_type, TestType, 3, 3>;

  auto (*closed)(const mesh_type &)->bool =
      &tf::cpp::is_closed<index_type, TestType, 3>;
  auto (*open)(const mesh_type &)->bool =
      &tf::cpp::is_open<index_type, TestType, 3>;
  auto (*manifold)(const mesh_type &)->bool =
      &tf::cpp::is_manifold<index_type, TestType, 3>;
  auto (*non_manifold)(const mesh_type &)->bool =
      &tf::cpp::is_non_manifold<index_type, TestType, 3>;
  auto (*euler)(const mesh_type &)->std::int32_t =
      &tf::cpp::euler_characteristic<index_type, TestType, 3>;
  auto (*boundary)(const mesh_type &)->tf::cpp::nd_array<index_type> =
      &tf::cpp::boundary_edges<index_type, TestType, 3>;
  auto (*non_manifold_edge_fn)(const mesh_type &)
      ->tf::cpp::nd_array<index_type> =
      &tf::cpp::non_manifold_edges<index_type, TestType, 3>;
  auto (*boundary_path_fn)(const mesh_type &)->paths_type =
      &tf::cpp::boundary_paths<index_type, TestType, 3>;
  auto (*boundary_curve_fn)(const mesh_type &)->curves_type =
      &tf::cpp::boundary_curves<index_type, TestType, 3>;
  auto (*rings)(const mesh_type &, std::int32_t, bool)->paths_type =
      &tf::cpp::k_rings<index_type, TestType, 3>;
  auto (*neighbors)(const mesh_type &, TestType, bool)->paths_type =
      &tf::cpp::neighborhoods<index_type, TestType, 3>;
  auto (*oriented)(const mesh_type &)->polygons_type =
      &tf::cpp::orient_faces_consistently<index_type, TestType, 3, 3>;
  auto (*cdt)(const tf::cpp::nd_array<TestType> &)
      ->tf::cpp::cdt_result<TestType> = &tf::cpp::make_cdt<TestType>;
  auto (*cdt_map)(const tf::cpp::nd_array<TestType> &)
      ->tf::cpp::cdt_result_with_map<TestType> =
      &tf::cpp::make_cdt_with_maps<TestType>;
  auto (*domains)(const mesh_type &, tf::domain_config)
      ->tf::cpp::domain_labels_result<> =
      &tf::cpp::make_domain_labels<std::int32_t, TestType, 3>;

  auto owned = tf::cpp::test::tetrahedron_mesh<TestType>();
  const auto mesh = owned.mesh();
  CHECK(closed(mesh));
  CHECK_FALSE(open(mesh));
  CHECK(manifold(mesh));
  CHECK_FALSE(non_manifold(mesh));
  CHECK(euler(mesh) == 2);
  CHECK(boundary(mesh).empty());
  CHECK(non_manifold_edge_fn(mesh).empty());
  CHECK(boundary_path_fn(mesh).size() == 0);
  CHECK(boundary_curve_fn(mesh).paths.size() == 0);
  CHECK(rings(mesh, 1, false).size() == 4);
  CHECK(neighbors(mesh, TestType{2}, false).size() == 4);
  CHECK(oriented(mesh).faces_buffer().size() == 4);
  CHECK(cdt(cdt_points<TestType>()).faces.shape_at(1) == 3);
  CHECK(cdt_map(cdt_points<TestType>()).point_index_map.f.shape_at(0) == 5);
  CHECK(domains(mesh, tf::domain_config::none).number_of_faces() == 4);
}

TEST_CASE("every boundary async lane submits exactly once",
          "[cpp][topology][boundary][async][resolver]") {
  const auto submissions = std::make_shared<std::atomic<int>>(0);
  const auto resolver = counting_resolver{submissions};

  auto square = tf::cpp::test::two_triangle_mesh<float>();
  const auto square_mesh = square.mesh();
  auto square_edges = tf::cpp::async::boundary_edges(resolver, square_mesh);
  auto square_paths = tf::cpp::async::boundary_paths(resolver, square_mesh);
  CHECK(square_edges.get().shape_at(0) == 4);
  CHECK(square_paths.get().size() == 1);

  tf::cpp::test::owned_mesh<std::int64_t, double, 2> typed{
      tf::cpp::test::polygons_of<std::int64_t, double, 2>(
          {0, 1, 2, 0, 2, 3}, {0, 0, 1, 0, 1, 1, 0, 1})};
  const auto typed_mesh = typed.mesh();
  auto typed_edges = tf::cpp::async::boundary_edges(resolver, typed_mesh);
  auto typed_paths = tf::cpp::async::boundary_paths(resolver, typed_mesh);
  auto typed_curves = tf::cpp::async::boundary_curves(resolver, typed_mesh);
  CHECK(typed_edges.get().shape_at(0) == 4);
  CHECK(typed_paths.get().size() == 1);
  const auto curves = typed_curves.get();
  CHECK(curves.paths.size() == 1);
  CHECK(curves.points.shape_at(0) == 4);
  CHECK(submissions->load(std::memory_order_relaxed) == 5);
}

TEST_CASE("typed mesh boundary symbol addresses link from independent lanes",
          "[cpp][topology][boundary][archive-link][abi]") {
  using typed_mesh = tf::cpp::mesh<std::int64_t, float, 2>;
  using typed_paths =
      tf::cpp::offset_blocked_buffer<std::int64_t, std::int64_t>;
  using typed_curves = tf::cpp::boundary_curves_result<std::int64_t, float, 2>;

  auto (*edges)(const typed_mesh &)->tf::cpp::nd_array<std::int64_t> =
      &tf::cpp::boundary_edges<std::int64_t, float, 2>;
  auto (*paths)(const typed_mesh &)->typed_paths =
      &tf::cpp::boundary_paths<std::int64_t, float, 2>;
  auto (*curves)(const typed_mesh &)->typed_curves =
      &tf::cpp::boundary_curves<std::int64_t, float, 2>;

  tf::cpp::test::owned_mesh<std::int64_t, float, 2> owned{
      tf::cpp::test::polygons_of<std::int64_t, float, 2>({0, 1, 2},
                                                         {0, 0, 1, 0, 0, 1})};
  const auto mesh = owned.mesh();
  CHECK(edges(mesh).raw_shape() == tf::small_vector<int, 3>{3, 2});
  CHECK(paths(mesh).size() == 1);
  const auto curve_result = curves(mesh);
  CHECK(curve_result.paths.size() == 1);
  CHECK(curve_result.points.raw_shape() == tf::small_vector<int, 3>{3, 2});
}
