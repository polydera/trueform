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

#include "trueform/cpp/core/build_tree.hpp"
#include "trueform/cpp/core/edge_mesh.hpp"
#include "trueform/cpp/core/mesh.hpp"
#include "trueform/cpp/core/point_cloud.hpp"
#include "trueform/cpp/core/point_cloud_cache.hpp"
#include "trueform/cpp/geometry/make_box_mesh.hpp"
#include "trueform/cpp/spatial/async/neighbor_search.hpp"
#include "trueform/cpp/spatial/async/neighbor_search_knn.hpp"
#include "trueform/cpp/spatial/neighbor_search.hpp"
#include "trueform/cpp/spatial/neighbor_search_knn.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <future>
#include <limits>
#include <memory>
#include <stdexcept>
#include <thread>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

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

template <typename T>
auto make_empty(tf::small_vector<int, 3> shape) -> tf::cpp::nd_array<T> {
  tf::buffer<T> buffer;
  buffer.allocate(0);
  return tf::cpp::nd_array<T>::from_buffer(std::move(buffer), std::move(shape));
}

template <typename Real>
auto triangle_mesh(Real z = Real{0})
    -> tf::cpp::test::owned_mesh<tf::cpp::default_index_t, Real> {
  return {tf::cpp::test::polygons_of<tf::cpp::default_index_t, Real>(
      {0, 1, 2},
      {Real{-1}, Real{-1}, z, Real{1}, Real{-1}, z, Real{0}, Real{1}, z})};
}

/// What a caller assembles when no handle of its own outlives the call: the
/// storage lives in a shared block and the carrier retains it.
template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto held_carrier_of(tf::cpp::test::owned_mesh<Index, Real, Dims, Ngon> owner)
    -> tf::cpp::mesh<Index, Real, Dims, Ngon> {
  const auto held = std::make_shared<
      const tf::cpp::test::owned_mesh<Index, Real, Dims, Ngon>>(
      std::move(owner));
  return {held->polygons.faces(), held->polygons.points(), held->cache,
          held->frame(), held};
}

template <typename Index, typename Real, std::size_t Dims>
auto held_carrier_of(tf::cpp::test::owned_edge_mesh<Index, Real, Dims> owner)
    -> tf::cpp::edge_mesh<Index, Real, Dims> {
  const auto held =
      std::make_shared<const tf::cpp::test::owned_edge_mesh<Index, Real, Dims>>(
          std::move(owner));
  return {held->segments.edges(), held->segments.points(), held->cache,
          held->frame(), held};
}

template <typename Real, std::size_t Dims>
auto held_carrier_of(tf::cpp::test::owned_point_cloud<Real, Dims> owner)
    -> tf::cpp::point_cloud<Real, Dims> {
  const auto held =
      std::make_shared<const tf::cpp::test::owned_point_cloud<Real, Dims>>(
          std::move(owner));
  return {held->points.points(), held->normals.unit_vectors(), held->cache,
          held->frame(), held};
}

template <typename Real>
auto line_cloud(Real offset = Real{0})
    -> tf::cpp::test::owned_point_cloud<Real> {
  return {tf::cpp::test::points_of<Real>({offset, Real{0}, Real{0},
                                          offset + Real{2}, Real{0}, Real{0},
                                          offset + Real{5}, Real{0}, Real{0}})};
}

template <typename Real>
auto point_query(Real x, Real y = Real{0}, Real z = Real{0}) {
  return tf::cpp::primitive<Real>(tf::cpp::primitive_kind::point,
                                  make_array<Real>({x, y, z}, {3}));
}

template <typename Real>
auto point_batch(std::initializer_list<Real> values, int count) {
  return tf::cpp::primitive<Real>(tf::cpp::primitive_kind::point,
                                  make_array<Real>(values, {count, 3}));
}

template <typename Real>
auto origin_query(tf::cpp::primitive_kind kind) -> tf::cpp::primitive<Real> {
  using tf::cpp::primitive;
  switch (kind) {
  case tf::cpp::primitive_kind::point:
    return primitive<Real>(kind, make_array<Real>({0, 0, 0}, {3}));
  case tf::cpp::primitive_kind::segment:
    return primitive<Real>(kind, make_array<Real>({-1, 0, 0, 1, 0, 0}, {2, 3}));
  case tf::cpp::primitive_kind::triangle:
    return primitive<Real>(
        kind, make_array<Real>({-1, -1, 0, 1, -1, 0, 0, 1, 0}, {3, 3}));
  case tf::cpp::primitive_kind::ray:
  case tf::cpp::primitive_kind::line:
    return primitive<Real>(kind, make_array<Real>({0, 0, 0, 1, 0, 0}, {2, 3}));
  case tf::cpp::primitive_kind::plane:
    return primitive<Real>(kind, make_array<Real>({0, 0, 1, 0}, {4}));
  case tf::cpp::primitive_kind::aabb:
    return primitive<Real>(kind,
                           make_array<Real>({-1, -1, -1, 1, 1, 1}, {2, 3}));
  case tf::cpp::primitive_kind::polygon:
    return primitive<Real>(
        kind,
        make_array<Real>({-1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 1, 0}, {4, 3}));
  case tf::cpp::primitive_kind::vector:
    return primitive<Real>(kind, make_array<Real>({1, 0, 0}, {3}));
  }
  throw std::logic_error("unknown primitive kind");
}

/// The placement a carrier is assembled at, in the shape `place` states it.
template <typename Real>
auto translation(Real x, Real y, Real z) -> std::array<Real, 16> {
  return {1, 0, 0, x, 0, 1, 0, y, 0, 0, 1, z, 0, 0, 0, 1};
}

constexpr std::array<tf::cpp::primitive_kind, 8> spatial_kinds{
    tf::cpp::primitive_kind::point,    tf::cpp::primitive_kind::segment,
    tf::cpp::primitive_kind::triangle, tf::cpp::primitive_kind::ray,
    tf::cpp::primitive_kind::line,     tf::cpp::primitive_kind::plane,
    tf::cpp::primitive_kind::aabb,     tf::cpp::primitive_kind::polygon};

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

template <typename Real> struct tree_observing_resolver {
  const tf::cpp::point_cloud_cache<Real> *cache;
  bool *tree_was_built;

  template <typename T>
  using state_type = tf::cpp::async::detail::future_state<T>;

  template <typename T>
  auto make_state() const -> std::shared_ptr<state_type<T>> {
    *tree_was_built = cache->is_tree_built();
    return std::make_shared<state_type<T>>();
  }
};

} // namespace

TEMPLATE_TEST_CASE(
    "async neighbor-search fronts preserve exact types and sync parity",
    "[cpp][spatial][neighbor-search][async]", float, double) {
  const auto mesh0_storage = triangle_mesh<TestType>();
  const auto mesh1_storage = triangle_mesh<TestType>(TestType{2});
  const auto cloud0_storage = line_cloud<TestType>();
  const auto cloud1_storage = tf::cpp::test::owned_point_cloud<TestType>{
      tf::cpp::test::points_of<TestType>({0, 0, 2})};
  const auto mesh0 = mesh0_storage.mesh();
  const auto mesh1 = mesh1_storage.mesh();
  const auto cloud0 = cloud0_storage.point_cloud();
  const auto cloud1 = cloud1_storage.point_cloud();
  const auto query = point_query<TestType>(TestType{0});
  const auto queries = point_batch<TestType>({0, 0, 0, 100, 0, 0}, 2);

  // a cache shared by concurrent jobs is filled before it is shared
  tf::cpp::build_tree(mesh0);
  tf::cpp::build_tree(mesh1);
  tf::cpp::build_tree(cloud0);
  tf::cpp::build_tree(cloud1);

  auto mesh_single = tf::cpp::async::neighbor_search(mesh0, query);
  auto cloud_single = tf::cpp::async::neighbor_search(cloud0, query);
  auto mesh_batch =
      tf::cpp::async::neighbor_search_batch(mesh0, queries, TestType{3});
  auto cloud_batch =
      tf::cpp::async::neighbor_search_batch(cloud0, queries, TestType{3});
  auto mesh_knn = tf::cpp::async::neighbor_search_knn(mesh0, query, 2);
  auto cloud_knn = tf::cpp::async::neighbor_search_knn(cloud0, query, 2);
  auto mesh_knn_batch =
      tf::cpp::async::neighbor_search_knn_batch(mesh0, queries, 2, TestType{3});
  auto cloud_knn_batch = tf::cpp::async::neighbor_search_knn_batch(
      cloud0, queries, 2, TestType{3});
  auto mesh_mesh = tf::cpp::async::neighbor_search(mesh0, mesh1);
  auto mesh_cloud = tf::cpp::async::neighbor_search(mesh0, cloud1);
  auto cloud_mesh = tf::cpp::async::neighbor_search(cloud1, mesh0);
  auto cloud_cloud = tf::cpp::async::neighbor_search(cloud0, cloud1);

  static_assert(std::is_same_v<decltype(mesh_single),
                               std::future<tf::cpp::neighbor_result<
                                   tf::cpp::default_index_t, TestType>>>);
  static_assert(std::is_same_v<decltype(mesh_batch),
                               std::future<tf::cpp::neighbor_batch_result<
                                   tf::cpp::default_index_t, TestType>>>);
  static_assert(std::is_same_v<decltype(mesh_knn),
                               std::future<tf::cpp::neighbor_knn_result<
                                   tf::cpp::default_index_t, TestType>>>);
  static_assert(std::is_same_v<decltype(mesh_knn_batch),
                               std::future<tf::cpp::neighbor_knn_batch_result<
                                   tf::cpp::default_index_t, TestType>>>);
  static_assert(
      std::is_same_v<
          decltype(mesh_mesh),
          std::future<tf::cpp::neighbor_pair_result<
              tf::cpp::default_index_t, tf::cpp::default_index_t, TestType>>>);

  const auto actual_mesh_single = mesh_single.get();
  const auto actual_cloud_single = cloud_single.get();
  const auto actual_mesh_batch = mesh_batch.get();
  const auto actual_cloud_batch = cloud_batch.get();
  const auto actual_mesh_knn = mesh_knn.get();
  const auto actual_cloud_knn = cloud_knn.get();
  const auto actual_mesh_knn_batch = mesh_knn_batch.get();
  const auto actual_cloud_knn_batch = cloud_knn_batch.get();
  const auto actual_mesh_mesh = mesh_mesh.get();
  const auto actual_mesh_cloud = mesh_cloud.get();
  const auto actual_cloud_mesh = cloud_mesh.get();
  const auto actual_cloud_cloud = cloud_cloud.get();

  CHECK(actual_mesh_single.element_id ==
        tf::cpp::neighbor_search(mesh0, query).element_id);
  CHECK(actual_cloud_single.element_id ==
        tf::cpp::neighbor_search(cloud0, query).element_id);
  CHECK(actual_mesh_batch.element_ids[1] ==
        tf::cpp::neighbor_search_batch(mesh0, queries, TestType{3})
            .element_ids[1]);
  CHECK(actual_cloud_batch.element_ids[1] ==
        tf::cpp::neighbor_search_batch(cloud0, queries, TestType{3})
            .element_ids[1]);
  CHECK(actual_mesh_knn.element_ids[0] ==
        tf::cpp::neighbor_search_knn(mesh0, query, 2).element_ids[0]);
  CHECK(actual_cloud_knn.element_ids[1] ==
        tf::cpp::neighbor_search_knn(cloud0, query, 2).element_ids[1]);
  CHECK(actual_mesh_knn_batch.counts[1] ==
        tf::cpp::neighbor_search_knn_batch(mesh0, queries, 2, TestType{3})
            .counts[1]);
  CHECK(actual_cloud_knn_batch.counts[0] ==
        tf::cpp::neighbor_search_knn_batch(cloud0, queries, 2, TestType{3})
            .counts[0]);
  CHECK(actual_mesh_mesh.distance2 == Catch::Approx(4));
  CHECK(actual_mesh_cloud.distance2 == Catch::Approx(4));
  CHECK(actual_cloud_mesh.distance2 == Catch::Approx(4));
  CHECK(actual_cloud_cloud.distance2 ==
        tf::cpp::neighbor_search(cloud0, cloud1).distance2);

  const auto submissions = std::make_shared<std::atomic<int>>(0);
  auto custom = tf::cpp::async::neighbor_search(counting_resolver{submissions},
                                                cloud0, query, TestType{0});
  static_assert(std::is_same_v<decltype(custom),
                               std::future<tf::cpp::neighbor_result<
                                   tf::cpp::default_index_t, TestType>>>);
  CHECK(submissions->load(std::memory_order_relaxed) == 1);
  CHECK(custom.get().element_id ==
        tf::cpp::neighbor_search(cloud0, query, TestType{0}).element_id);
}

TEMPLATE_TEST_CASE(
    "async neighbor search owns inputs caches sentinels and exceptions",
    "[cpp][spatial][neighbor-search][async][ownership]", float, double) {
  const auto cloud_storage = line_cloud<TestType>();
  const auto cloud = cloud_storage.point_cloud();
  auto query = point_query<TestType>(TestType{0});
  bool tree_was_built = true;
  auto retained = tf::cpp::async::neighbor_search(
      tree_observing_resolver<TestType>{&cloud_storage.cache, &tree_was_built},
      cloud, query);
  // assembling and submitting build nothing: the job asks when it reads
  CHECK_FALSE(tree_was_built);
  query = point_query<TestType>(TestType{100});
  const auto retained_result = retained.get();
  CHECK(retained_result.element_id == 0);
  CHECK(retained_result.distance2 == Catch::Approx(0));
  CHECK(retained_result.point[0] == TestType{0});

  // the carriers are borrowed, so the storage they name is kept alive by the
  // keepalive they were assembled with and no handle of the caller's survives
  auto retained_pair = tf::cpp::async::neighbor_search(
      held_carrier_of(line_cloud<TestType>()),
      held_carrier_of(line_cloud<TestType>(TestType{2})));
  const auto pair_result = retained_pair.get();
  CHECK(pair_result.element_id0 == 1);
  CHECK(pair_result.element_id1 == 0);
  CHECK(pair_result.distance2 == Catch::Approx(0));

  const auto empty_cloud_storage = tf::cpp::test::owned_point_cloud<TestType>{};
  const auto empty_mesh_storage =
      tf::cpp::test::owned_mesh<tf::cpp::default_index_t, TestType>{};
  const auto empty_cloud = empty_cloud_storage.point_cloud();
  const auto empty_mesh = empty_mesh_storage.mesh();
  const auto single = tf::cpp::async::neighbor_search(empty_cloud, query).get();
  CHECK(single.element_id == -1);
  CHECK(std::isinf(single.distance2));

  const auto empty_queries = tf::cpp::primitive<TestType>(
      tf::cpp::primitive_kind::point, make_empty<TestType>({0, 3}));
  const auto batch =
      tf::cpp::async::neighbor_search_batch(empty_cloud, empty_queries).get();
  CHECK(batch.element_ids.raw_shape() == tf::small_vector<int, 3>{0});
  CHECK(batch.points.raw_shape() == tf::small_vector<int, 3>{0, 3});
  const auto knn_batch =
      tf::cpp::async::neighbor_search_knn_batch(empty_cloud, empty_queries, 3)
          .get();
  CHECK(knn_batch.element_ids.raw_shape() == tf::small_vector<int, 3>{0, 3});
  CHECK(knn_batch.counts.raw_shape() == tf::small_vector<int, 3>{0});

  const auto full_cloud_storage = line_cloud<TestType>();
  const auto full_cloud = full_cloud_storage.point_cloud();
  const auto empty_pair =
      tf::cpp::async::neighbor_search(empty_mesh, full_cloud).get();
  CHECK(empty_pair.element_id0 == -1);
  CHECK(empty_pair.element_id1 == -1);
  CHECK(std::isinf(empty_pair.distance2));

  auto failure = tf::cpp::async::neighbor_search_knn(full_cloud, query, 0);
  CHECK_THROWS_AS(failure.get(), std::invalid_argument);
}

TEST_CASE("concurrent async jobs share one filled cache",
          "[cpp][spatial][neighbor-search][async][concurrency]") {
  // the caller fills before it shares, so sixty-four jobs read the one tree
  const auto cloud_storage = tf::cpp::test::owned_point_cloud<float>{
      tf::cpp::test::points_of<float>({0, 0, 0})};
  const auto cloud = cloud_storage.point_cloud();
  const auto query = point_query<float>(0);
  tf::cpp::build_tree(cloud);

  std::vector<
      std::future<tf::cpp::neighbor_result<tf::cpp::default_index_t, float>>>
      results;
  for (int index = 0; index < 64; ++index)
    results.push_back(tf::cpp::async::neighbor_search(cloud, query));

  for (auto &future : results) {
    const auto result = future.get();
    CHECK(result.element_id == 0);
    CHECK(result.distance2 == Catch::Approx(0));
    CHECK(result.point[0] == Catch::Approx(0));
  }
  CHECK(cloud_storage.cache.tree_build_count() == 1);
}

TEMPLATE_TEST_CASE(
    "mesh and point cloud neighbor search support every query kind",
    "[cpp][spatial][neighbor-search][dispatch]", float, double) {
  const auto mesh_storage = triangle_mesh<TestType>();
  const auto cloud_storage = line_cloud<TestType>();
  const auto mesh = mesh_storage.mesh();
  const auto cloud = cloud_storage.point_cloud();
  for (const auto kind : spatial_kinds) {
    INFO("primitive kind " << static_cast<int>(kind));
    const auto query = origin_query<TestType>(kind);
    const auto mesh_result = tf::cpp::neighbor_search(mesh, query);
    const auto cloud_result = tf::cpp::neighbor_search(cloud, query);
    CHECK(mesh_result.element_id == 0);
    CHECK(cloud_result.element_id == 0);
    CHECK(mesh_result.distance2 == Catch::Approx(0));
    CHECK(cloud_result.distance2 == Catch::Approx(0));
    CHECK(mesh_result.point.raw_shape() == tf::small_vector<int, 3>{3});
    CHECK(cloud_result.point.raw_shape() == tf::small_vector<int, 3>{3});
  }
  CHECK(mesh_storage.cache.tree_build_count() == 1);
  CHECK(cloud_storage.cache.tree_build_count() == 1);
}

TEMPLATE_TEST_CASE(
    "neighbor search preserves strict radius and sentinel misses",
    "[cpp][spatial][neighbor-search][radius]", float, double) {
  const auto cloud_storage = line_cloud<TestType>();
  const auto cloud = cloud_storage.point_cloud();
  const auto boundary = point_query<TestType>(TestType{1});
  const auto boundary_result =
      tf::cpp::neighbor_search(cloud, boundary, TestType{1});
  CHECK(boundary_result.element_id == -1);
  CHECK(boundary_result.distance2 == Catch::Approx(1));
  CHECK(boundary_result.point[0] == TestType{0});

  const auto exact = point_query<TestType>(TestType{0});
  CHECK(tf::cpp::neighbor_search(cloud, exact, TestType{0}).element_id == -1);
  CHECK(tf::cpp::neighbor_search(cloud, exact).element_id == 0);
}

TEMPLATE_TEST_CASE(
    "batch nearest results preserve shapes misses and one tree warm",
    "[cpp][spatial][neighbor-search][batch]", float, double) {
  const auto cloud_storage = line_cloud<TestType>();
  const auto cloud = cloud_storage.point_cloud();
  const auto queries = point_batch<TestType>({0, 0, 0, 100, 0, 0}, 2);
  const auto result =
      tf::cpp::neighbor_search_batch(cloud, queries, TestType{3});
  CHECK(result.element_ids.raw_shape() == tf::small_vector<int, 3>{2});
  CHECK(result.points.raw_shape() == tf::small_vector<int, 3>{2, 3});
  CHECK(result.distances.raw_shape() == tf::small_vector<int, 3>{2});
  CHECK(result.element_ids[0] == 0);
  CHECK(result.element_ids[1] == -1);
  CHECK(result.distances[0] == Catch::Approx(0));
  CHECK(result.distances[1] == Catch::Approx(9));
  CHECK(cloud_storage.cache.tree_build_count() == 1);
}

TEMPLATE_TEST_CASE("single and batched kNN preserve sorting padding and counts",
                   "[cpp][spatial][neighbor-search][knn]", float, double) {
  const auto cloud_storage = line_cloud<TestType>();
  const auto cloud = cloud_storage.point_cloud();
  const auto query = point_query<TestType>(TestType{0});
  const auto nearest = tf::cpp::neighbor_search_knn(cloud, query, 5);
  CHECK(nearest.element_ids.raw_shape() == tf::small_vector<int, 3>{3});
  CHECK(nearest.points.raw_shape() == tf::small_vector<int, 3>{3, 3});
  CHECK(nearest.distances.raw_shape() == tf::small_vector<int, 3>{3});
  CHECK(nearest.element_ids[0] == 0);
  CHECK(nearest.element_ids[1] == 1);
  CHECK(nearest.element_ids[2] == 2);
  CHECK(nearest.distances[0] == Catch::Approx(0));
  CHECK(nearest.distances[1] == Catch::Approx(4));
  CHECK(nearest.distances[2] == Catch::Approx(25));

  const auto limited =
      tf::cpp::neighbor_search_knn(cloud, query, 5, TestType{2});
  REQUIRE(limited.element_ids.size() == 1);
  CHECK(limited.element_ids[0] == 0);

  const auto queries = point_batch<TestType>({0, 0, 0, 100, 0, 0}, 2);
  const auto batch =
      tf::cpp::neighbor_search_knn_batch(cloud, queries, 4, TestType{3});
  CHECK(batch.element_ids.raw_shape() == tf::small_vector<int, 3>{2, 4});
  CHECK(batch.points.raw_shape() == tf::small_vector<int, 3>{2, 4, 3});
  CHECK(batch.distances.raw_shape() == tf::small_vector<int, 3>{2, 4});
  CHECK(batch.counts.raw_shape() == tf::small_vector<int, 3>{2});
  CHECK(batch.counts[0] == 2);
  CHECK(batch.counts[1] == 0);
  CHECK(batch.element_ids[0] == 0);
  CHECK(batch.element_ids[1] == 1);
  CHECK(batch.element_ids[2] == -1);
  CHECK(batch.element_ids[4] == -1);
  CHECK(batch.distances[2] == Catch::Approx(9));

  const auto mesh_storage = triangle_mesh<TestType>();
  const auto mesh = mesh_storage.mesh();
  const auto mesh_queries = point_batch<TestType>({0, 0, 0, 0, 0, 50}, 2);
  const auto mesh_batch =
      tf::cpp::neighbor_search_knn_batch(mesh, mesh_queries, 5, TestType{2});
  CHECK(mesh_batch.counts[0] == 1);
  CHECK(mesh_batch.counts[1] == 0);
  CHECK(mesh_batch.element_ids[0] == 0);
  CHECK(mesh_batch.element_ids[5] == -1);

  const auto box_storage =
      tf::cpp::test::owned_mesh<tf::cpp::default_index_t, TestType>{
          tf::cpp::make_box_mesh(TestType{2}, TestType{2}, TestType{2})};
  const auto box = box_storage.mesh();
  const auto box_queries = point_batch<TestType>({1, 0, 0, 0, 0, 50}, 2);
  const auto box_batch =
      tf::cpp::neighbor_search_knn_batch(box, box_queries, 5, TestType{2});
  CHECK(box_batch.counts[0] > 0);
  CHECK(box_batch.counts[1] == 0);
}

TEMPLATE_TEST_CASE("nearest-pair search covers the mesh point-cloud matrix",
                   "[cpp][spatial][neighbor-search][pair]", float, double) {
  const auto mesh0_storage = triangle_mesh<TestType>();
  const auto mesh1_storage = triangle_mesh<TestType>(TestType{2});
  const auto cloud0_storage = tf::cpp::test::owned_point_cloud<TestType>{
      tf::cpp::test::points_of<TestType>({0, 0, 0})};
  const auto cloud1_storage = tf::cpp::test::owned_point_cloud<TestType>{
      tf::cpp::test::points_of<TestType>({0, 0, 2})};
  const auto mesh0 = mesh0_storage.mesh();
  const auto mesh1 = mesh1_storage.mesh();
  const auto cloud0 = cloud0_storage.point_cloud();
  const auto cloud1 = cloud1_storage.point_cloud();

  const auto mm = tf::cpp::neighbor_search(mesh0, mesh1);
  const auto mp = tf::cpp::neighbor_search(mesh0, cloud1);
  const auto pm = tf::cpp::neighbor_search(cloud1, mesh0);
  const auto pp = tf::cpp::neighbor_search(cloud0, cloud1);
  for (const auto distance :
       {mm.distance2, mp.distance2, pm.distance2, pp.distance2})
    CHECK(distance == Catch::Approx(4));
  CHECK(mm.element_id0 == 0);
  CHECK(mm.element_id1 == 0);
  CHECK(mp.element_id0 == 0);
  CHECK(mp.element_id1 == 0);
  CHECK(pm.element_id0 == 0);
  CHECK(pm.element_id1 == 0);
  CHECK(pp.element_id0 == 0);
  CHECK(pp.element_id1 == 0);
  CHECK(mp.point0.raw_shape() == tf::small_vector<int, 3>{3});
  CHECK(mp.point1.raw_shape() == tf::small_vector<int, 3>{3});

  const auto miss = tf::cpp::neighbor_search(mesh0, cloud1, TestType{2});
  CHECK(miss.element_id0 == -1);
  CHECK(miss.element_id1 == -1);
  CHECK(miss.distance2 == Catch::Approx(4));
}

TEMPLATE_TEST_CASE(
    "neighbor search applies transformations and reuses coherent caches",
    "[cpp][spatial][neighbor-search][tree]", float, double) {
  auto cloud_storage = tf::cpp::test::owned_point_cloud<TestType>{
      tf::cpp::test::points_of<TestType>({0, 0, 0})};
  const auto at_ten = point_query<TestType>(TestType{10});
  CHECK(
      tf::cpp::neighbor_search(cloud_storage.point_cloud(), at_ten).distance2 ==
      Catch::Approx(100));
  CHECK(cloud_storage.cache.tree_build_count() == 1);

  // a cache is built in local coordinates, so a placement is not a change
  cloud_storage.place(translation<TestType>(10, 0, 0));
  CHECK(
      tf::cpp::neighbor_search(cloud_storage.point_cloud(), at_ten).distance2 ==
      Catch::Approx(0));
  CHECK(cloud_storage.cache.tree_build_count() == 1);
  CHECK(tf::cpp::neighbor_search(cloud_storage.point_cloud(), at_ten)
            .element_id == 0);
  CHECK(cloud_storage.cache.tree_build_count() == 1);

  cloud_storage.place(translation<TestType>(0, 0, 0));
  cloud_storage.points.data_buffer()[0] = TestType{20};
  cloud_storage.cache.points_changed();
  const auto moved = cloud_storage.point_cloud();
  CHECK_FALSE(cloud_storage.cache.is_tree_fresh(moved.geometry()));
  const auto at_twenty = point_query<TestType>(TestType{20});
  CHECK(tf::cpp::neighbor_search(moved, at_twenty).distance2 ==
        Catch::Approx(0));
  CHECK(cloud_storage.cache.is_tree_fresh(moved.geometry()));
  CHECK(cloud_storage.cache.tree_build_count() == 2);

  auto mesh_storage = triangle_mesh<TestType>();
  mesh_storage.place(translation<TestType>(0, 0, 5));
  const auto above = point_query<TestType>(0, 0, 5);
  CHECK(tf::cpp::neighbor_search(mesh_storage.mesh(), above).distance2 ==
        Catch::Approx(0));
  CHECK(mesh_storage.cache.tree_build_count() == 1);
}

TEMPLATE_TEST_CASE(
    "neighbor search validates handles cardinality options and vectors",
    "[cpp][spatial][neighbor-search][validation]", float, double) {
  // an empty cloud has nothing to be near, so the search answers a miss
  const auto nothing_storage = tf::cpp::test::owned_point_cloud<TestType>{};
  const auto query = point_query<TestType>(0);
  CHECK(tf::cpp::neighbor_search(nothing_storage.point_cloud(), query)
            .element_id == -1);

  const auto cloud_storage = line_cloud<TestType>();
  const auto cloud = cloud_storage.point_cloud();
  const auto batch = point_batch<TestType>({0, 0, 0}, 1);
  CHECK_THROWS_AS(tf::cpp::neighbor_search(cloud, batch),
                  std::invalid_argument);
  CHECK_THROWS_AS(tf::cpp::neighbor_search_batch(cloud, query),
                  std::invalid_argument);
  CHECK_THROWS_AS(tf::cpp::neighbor_search_knn(cloud, query, 0),
                  std::invalid_argument);
  CHECK_THROWS_AS(tf::cpp::neighbor_search_knn_batch(cloud, batch, -1),
                  std::invalid_argument);
  CHECK_THROWS_AS(tf::cpp::neighbor_search(cloud, query, TestType{-1}),
                  std::invalid_argument);
  CHECK_THROWS_AS(tf::cpp::neighbor_search(
                      cloud, query, std::numeric_limits<TestType>::quiet_NaN()),
                  std::invalid_argument);
  CHECK_THROWS_AS(tf::cpp::neighbor_search(
                      cloud, query, -std::numeric_limits<TestType>::infinity()),
                  std::invalid_argument);
  CHECK_NOTHROW(tf::cpp::neighbor_search(
      cloud, query, std::numeric_limits<TestType>::infinity()));

  const auto vector = origin_query<TestType>(tf::cpp::primitive_kind::vector);
  CHECK_THROWS_AS(tf::cpp::neighbor_search(cloud, vector),
                  std::invalid_argument);
  const auto vector_batch = tf::cpp::primitive<TestType>(
      tf::cpp::primitive_kind::vector, make_array<TestType>({1, 0, 0}, {1, 3}));
  CHECK_THROWS_AS(tf::cpp::neighbor_search_batch(cloud, vector_batch),
                  std::invalid_argument);
}

TEMPLATE_TEST_CASE("empty forms and query batches return safe owned results",
                   "[cpp][spatial][neighbor-search][empty]", float, double) {
  const auto empty_cloud_storage = tf::cpp::test::owned_point_cloud<TestType>{};
  const auto empty_mesh_storage =
      tf::cpp::test::owned_mesh<tf::cpp::default_index_t, TestType>{};
  const auto empty_cloud = empty_cloud_storage.point_cloud();
  const auto empty_mesh = empty_mesh_storage.mesh();
  auto query = point_query<TestType>(0);
  const auto cloud_result = tf::cpp::neighbor_search(empty_cloud, query);
  const auto mesh_result = tf::cpp::neighbor_search(empty_mesh, query);
  CHECK(cloud_result.element_id == -1);
  CHECK(mesh_result.element_id == -1);
  CHECK_FALSE(empty_cloud_storage.cache.is_tree_built());
  CHECK_FALSE(empty_mesh_storage.cache.is_tree_built());

  const auto empty_queries = tf::cpp::primitive<TestType>(
      tf::cpp::primitive_kind::point, make_empty<TestType>({0, 3}));
  const auto batch = tf::cpp::neighbor_search_batch(empty_cloud, empty_queries);
  CHECK(batch.element_ids.raw_shape() == tf::small_vector<int, 3>{0});
  CHECK(batch.points.raw_shape() == tf::small_vector<int, 3>{0, 3});
  const auto knn =
      tf::cpp::neighbor_search_knn_batch(empty_cloud, empty_queries, 3);
  CHECK(knn.element_ids.raw_shape() == tf::small_vector<int, 3>{0, 3});
  CHECK(knn.points.raw_shape() == tf::small_vector<int, 3>{0, 3, 3});
  CHECK(knn.counts.raw_shape() == tf::small_vector<int, 3>{0});

  auto full_cloud_storage = line_cloud<TestType>();
  const auto pair =
      tf::cpp::neighbor_search(empty_mesh, full_cloud_storage.point_cloud());
  CHECK(pair.element_id0 == -1);
  CHECK(pair.element_id1 == -1);

  // the result owns its own arrays, whatever becomes of the form it read
  auto owned =
      tf::cpp::neighbor_search(full_cloud_storage.point_cloud(), query);
  query = point_query<TestType>(100);
  full_cloud_storage = {};
  REQUIRE(owned.point.is_valid());
  CHECK(owned.point[0] == TestType{0});
}

TEST_CASE("the neighbor result types are one per axis set",
          "[cpp][spatial][neighbor-search][abi]") {
  static_assert(std::is_same_v<
                tf::cpp::neighbor_result<std::int32_t, float, 3>,
                tf::cpp::neighbor_result<tf::cpp::default_index_t, float>>);
  static_assert(
      std::is_same_v<
          tf::cpp::neighbor_batch_result<std::int32_t, float, 3>,
          tf::cpp::neighbor_batch_result<tf::cpp::default_index_t, float>>);
  static_assert(
      std::is_same_v<
          tf::cpp::neighbor_pair_result<std::int32_t, std::int32_t, float, 3>,
          tf::cpp::neighbor_pair_result<tf::cpp::default_index_t,
                                        tf::cpp::default_index_t, float>>);
  static_assert(std::is_same_v<
                tf::cpp::neighbor_knn_result<std::int32_t, float, 3>,
                tf::cpp::neighbor_knn_result<tf::cpp::default_index_t, float>>);
  static_assert(
      std::is_same_v<
          tf::cpp::neighbor_knn_batch_result<std::int32_t, float, 3>,
          tf::cpp::neighbor_knn_batch_result<tf::cpp::default_index_t, float>>);
  static_assert(
      std::is_same_v<tf::cpp::neighbor_result<std::int64_t, float, 2>,
                     tf::cpp::neighbor_result<std::int64_t, float, 2>>);
  static_assert(
      std::is_same_v<
          tf::cpp::neighbor_pair_result<std::int32_t, std::int64_t, float, 2>,
          tf::cpp::neighbor_pair_result<std::int32_t, std::int64_t, float, 2>>);
}

TEMPLATE_TEST_CASE("neighbor-search overloads link from the native archive",
                   "[cpp][spatial][neighbor-search][archive-link][abi]", float,
                   double) {
  // the compiled symbol takes the carrier the operation is about, so the
  // address that links is the view's
  using form_type = tf::cpp::point_cloud<TestType>;
  using single_function =
      tf::cpp::neighbor_result<tf::cpp::default_index_t, TestType> (*)(
          const form_type &, const tf::cpp::primitive<TestType> &, TestType);
  using batch_function =
      tf::cpp::neighbor_batch_result<tf::cpp::default_index_t, TestType> (*)(
          const form_type &, const tf::cpp::primitive<TestType> &, TestType);
  using knn_function =
      tf::cpp::neighbor_knn_result<tf::cpp::default_index_t, TestType> (*)(
          const form_type &, const tf::cpp::primitive<TestType> &, int,
          TestType);
  using knn_batch_function =
      tf::cpp::neighbor_knn_batch_result<tf::cpp::default_index_t, TestType> (
              *)(const form_type &, const tf::cpp::primitive<TestType> &, int,
                 TestType);
  using pair_function =
      tf::cpp::neighbor_pair_result<tf::cpp::default_index_t,
                                    tf::cpp::default_index_t, TestType> (*)(
          const form_type &, const form_type &, TestType);
  const auto single =
      static_cast<single_function>(&tf::cpp::neighbor_search<TestType>);
  const auto batch =
      static_cast<batch_function>(&tf::cpp::neighbor_search_batch<TestType>);
  const auto knn =
      static_cast<knn_function>(&tf::cpp::neighbor_search_knn<TestType>);
  const auto knn_batch = static_cast<knn_batch_function>(
      &tf::cpp::neighbor_search_knn_batch<TestType>);
  const auto pair =
      static_cast<pair_function>(&tf::cpp::neighbor_search<TestType>);

  auto owner = line_cloud<TestType>();
  auto second_owner = line_cloud<TestType>();
  const auto cloud = owner.point_cloud();
  const auto second_cloud = second_owner.point_cloud();
  const auto query = point_query<TestType>(0);
  const auto queries = point_batch<TestType>({0, 0, 0}, 1);
  CHECK(single(cloud, query, std::numeric_limits<TestType>::infinity())
            .element_id == 0);
  CHECK(batch(cloud, queries, std::numeric_limits<TestType>::infinity())
            .element_ids[0] == 0);
  CHECK(knn(cloud, query, 1, std::numeric_limits<TestType>::infinity())
            .element_ids[0] == 0);
  CHECK(knn_batch(cloud, queries, 1, std::numeric_limits<TestType>::infinity())
            .element_ids[0] == 0);
  CHECK(pair(cloud, second_cloud, std::numeric_limits<TestType>::infinity())
            .element_id0 == 0);
}

// The storage and the cache are the CALLER'S, so a thread that holds its own
// moves its own points and builds its own tree, and the caller's cloud answers
// its own throughout.
TEST_CASE("a cloud of one's own is another thread's own cloud",
          "[cpp][spatial][neighbor-search][concurrency]") {
  const auto cloud_storage = tf::cpp::test::owned_point_cloud<float>{
      tf::cpp::test::points_of<float>({0, 0, 0})};
  const auto cloud = cloud_storage.point_cloud();
  const auto query = point_query<float>(0);
  std::atomic<bool> start{false};
  std::thread writer([&start] {
    auto own = tf::cpp::test::owned_point_cloud<float>{
        tf::cpp::test::points_of<float>({0, 0, 0})};
    while (!start.load(std::memory_order_acquire))
      std::this_thread::yield();
    for (int index = 0; index < 100; ++index) {
      own.points.data_buffer()[0] = index % 2 == 0 ? 0.0F : 10.0F;
      own.cache.points_changed();
      static_cast<void>(
          tf::cpp::neighbor_search(own.point_cloud(), point_query<float>(0)));
    }
  });
  start.store(true, std::memory_order_release);
  for (int index = 0; index < 100; ++index) {
    const auto result = tf::cpp::neighbor_search(cloud, query);
    CHECK(result.element_id == 0);
    CHECK(result.distance2 == Catch::Approx(0));
  }
  writer.join();
  const auto final_result = tf::cpp::neighbor_search(cloud, query);
  CHECK(final_result.element_id == 0);
  CHECK(final_result.distance2 == Catch::Approx(0));
  CHECK(cloud_storage.cache.is_tree_fresh(cloud.geometry()));
  CHECK(cloud_storage.cache.tree_build_count() == 1);
}

namespace {

template <typename Index, typename Real, std::size_t Dims>
struct neighbor_matrix_row {
  using real_type = Real;
  using index_type = Index;
  static constexpr std::size_t dims = Dims;
};

using neighbor_matrix_rows =
    std::tuple<neighbor_matrix_row<std::int32_t, float, 2>,
               neighbor_matrix_row<std::int32_t, float, 3>,
               neighbor_matrix_row<std::int64_t, float, 2>,
               neighbor_matrix_row<std::int64_t, float, 3>,
               neighbor_matrix_row<std::int32_t, double, 2>,
               neighbor_matrix_row<std::int32_t, double, 3>,
               neighbor_matrix_row<std::int64_t, double, 2>,
               neighbor_matrix_row<std::int64_t, double, 3>>;

template <typename Row> auto matrix_triangle_mesh() {
  using Real = typename Row::real_type;
  using Index = typename Row::index_type;
  constexpr auto Dims = Row::dims;
  if constexpr (Dims == 2)
    return tf::cpp::test::owned_mesh<Index, Real, 2>{
        tf::cpp::test::polygons_of<Index, Real, 2>({0, 1, 2},
                                                   {0, 0, 2, 0, 0, 2})};
  else
    return tf::cpp::test::owned_mesh<Index, Real, 3>{
        tf::cpp::test::polygons_of<Index, Real, 3>(
            {0, 1, 2}, {0, 0, 0, 2, 0, 0, 0, 2, 0})};
}

template <typename Row> auto matrix_dynamic_quad_mesh() {
  using Real = typename Row::real_type;
  using Index = typename Row::index_type;
  constexpr auto Dims = Row::dims;
  if constexpr (Dims == 2)
    return tf::cpp::test::owned_mesh<Index, Real, 2, tf::dynamic_size>{
        tf::cpp::test::polygons_of<Index, Real, 2>({0, 4}, {0, 1, 2, 3},
                                                   {0, 0, 1, 0, 1, 1, 0, 1})};
  else
    return tf::cpp::test::owned_mesh<Index, Real, 3, tf::dynamic_size>{
        tf::cpp::test::polygons_of<Index, Real, 3>(
            {0, 4}, {0, 1, 2, 3}, {0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0})};
}

template <typename Row> auto matrix_edge_mesh() {
  using Real = typename Row::real_type;
  using Index = typename Row::index_type;
  constexpr auto Dims = Row::dims;
  if constexpr (Dims == 2)
    return tf::cpp::test::owned_edge_mesh<Index, Real, 2>{
        tf::cpp::test::segments_of<Index, Real, 2>({0, 1, 1, 2, 2, 3},
                                                   {0, 0, 1, 0, 2, 0, 3, 0})};
  else
    return tf::cpp::test::owned_edge_mesh<Index, Real, 3>{
        tf::cpp::test::segments_of<Index, Real, 3>(
            {0, 1, 1, 2, 2, 3}, {0, 0, 0, 1, 0, 0, 2, 0, 0, 3, 0, 0})};
}

template <typename Row> auto matrix_cloud() {
  using Real = typename Row::real_type;
  constexpr auto Dims = Row::dims;
  if constexpr (Dims == 2)
    return tf::cpp::test::owned_point_cloud<Real, 2>{
        tf::cpp::test::points_of<Real, 2>({0, 0, 1, 0, 2, 0, 3, 0})};
  else
    return tf::cpp::test::owned_point_cloud<Real, 3>{
        tf::cpp::test::points_of<Real, 3>(
            {0, 0, 0, 1, 0, 0, 2, 0, 0, 3, 0, 0})};
}

template <typename Row>
auto matrix_point(typename Row::real_type x, typename Row::real_type y,
                  typename Row::real_type z = typename Row::real_type{}) {
  using Real = typename Row::real_type;
  if constexpr (Row::dims == 2)
    return tf::cpp::primitive<Real, 2>(tf::cpp::primitive_kind::point,
                                       make_array<Real>({x, y}, {2}));
  else
    return tf::cpp::primitive<Real, 3>(tf::cpp::primitive_kind::point,
                                       make_array<Real>({x, y, z}, {3}));
}

template <typename Row> auto matrix_point_batch() {
  using Real = typename Row::real_type;
  if constexpr (Row::dims == 2)
    return tf::cpp::primitive<Real, 2>(
        tf::cpp::primitive_kind::point,
        make_array<Real>({0, 0, 20, 20}, {2, 2}));
  else
    return tf::cpp::primitive<Real, 3>(
        tf::cpp::primitive_kind::point,
        make_array<Real>({0, 0, 0, 20, 20, 20}, {2, 3}));
}

template <typename Row> auto matrix_origin_query(tf::cpp::primitive_kind kind) {
  using Real = typename Row::real_type;
  constexpr auto Dims = Row::dims;
  if (kind == tf::cpp::primitive_kind::point)
    return matrix_point<Row>(Real{0}, Real{0});
  if (kind == tf::cpp::primitive_kind::segment ||
      kind == tf::cpp::primitive_kind::ray ||
      kind == tf::cpp::primitive_kind::line) {
    if constexpr (Dims == 2)
      return tf::cpp::primitive<Real, 2>(
          kind, make_array<Real>({0, 0, 1, 0}, {2, 2}));
    else
      return tf::cpp::primitive<Real, 3>(
          kind, make_array<Real>({0, 0, 0, 1, 0, 0}, {2, 3}));
  }
  if (kind == tf::cpp::primitive_kind::triangle) {
    if constexpr (Dims == 2)
      return tf::cpp::primitive<Real, 2>(
          kind, make_array<Real>({0, 0, 1, 0, 0, 1}, {3, 2}));
    else
      return tf::cpp::primitive<Real, 3>(
          kind, make_array<Real>({0, 0, 0, 1, 0, 0, 0, 1, 0}, {3, 3}));
  }
  if (kind == tf::cpp::primitive_kind::polygon) {
    if constexpr (Dims == 2)
      return tf::cpp::primitive<Real, 2>(
          kind, make_array<Real>({0, 0, 1, 0, 1, 1, 0, 1}, {4, 2}));
    else
      return tf::cpp::primitive<Real, 3>(
          kind, make_array<Real>({0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0}, {4, 3}));
  }
  if (kind == tf::cpp::primitive_kind::aabb) {
    if constexpr (Dims == 2)
      return tf::cpp::primitive<Real, 2>(
          kind, make_array<Real>({0, 0, 1, 1}, {2, 2}));
    else
      return tf::cpp::primitive<Real, 3>(
          kind, make_array<Real>({0, 0, 0, 1, 1, 1}, {2, 3}));
  }
  if constexpr (Dims == 3)
    return tf::cpp::primitive<Real, 3>(tf::cpp::primitive_kind::plane,
                                       make_array<Real>({0, 1, 0, 0}, {4}));
  else
    throw std::logic_error("2D planes are not runtime spatial primitives");
}

/// The placements a carrier is assembled at, in the shape `place` states them.
template <typename Row>
auto matrix_translation(typename Row::real_type x, typename Row::real_type y,
                        typename Row::real_type z = typename Row::real_type{})
    -> std::array<typename Row::real_type, (Row::dims + 1) * (Row::dims + 1)> {
  if constexpr (Row::dims == 2)
    return {1, 0, x, 0, 1, y, 0, 0, 1};
  else
    return {1, 0, 0, x, 0, 1, 0, y, 0, 0, 1, z, 0, 0, 0, 1};
}

template <typename Row>
auto matrix_quarter_turn()
    -> std::array<typename Row::real_type, (Row::dims + 1) * (Row::dims + 1)> {
  if constexpr (Row::dims == 2)
    return {0, -1, 0, 1, 0, 0, 0, 0, 1};
  else
    return {0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
}

template <typename Row>
auto matrix_double_scale()
    -> std::array<typename Row::real_type, (Row::dims + 1) * (Row::dims + 1)> {
  if constexpr (Row::dims == 2)
    return {2, 0, 0, 0, 2, 0, 0, 0, 1};
  else
    return {2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1};
}

template <typename Left, typename Right, typename = void>
struct is_neighbor_invocable : std::false_type {};

template <typename Left, typename Right>
struct is_neighbor_invocable<
    Left, Right,
    std::void_t<decltype(tf::cpp::neighbor_search(
        std::declval<const Left &>(), std::declval<const Right &>()))>>
    : std::true_type {};

template <typename T>
auto require_matrix_shape(const tf::cpp::nd_array<T> &value,
                          std::initializer_list<int> shape) -> void {
  REQUIRE(value.ndim() == static_cast<int>(shape.size()));
  auto dimension = 0;
  for (const auto extent : shape)
    CHECK(value.shape_at(dimension++) == extent);
}

template <typename Real> struct query_releasing_resolver {
  tf::cpp::nd_array<Real> *values;

  template <typename T>
  using state_type = tf::cpp::async::detail::future_state<T>;

  template <typename T>
  auto make_state() const -> std::shared_ptr<state_type<T>> {
    values->destroy();
    return std::make_shared<state_type<T>>();
  }
};

} // namespace

TEMPLATE_LIST_TEST_CASE(
    "Python neighbor primitive facade covers every scalar index dimension and "
    "carrier",
    "[cpp][spatial][neighbor-search][python-parity][matrix][primitive]",
    neighbor_matrix_rows) {
  using Real = typename TestType::real_type;
  using Index = typename TestType::index_type;
  constexpr auto Dims = TestType::dims;

  const auto mesh_storage = matrix_triangle_mesh<TestType>();
  const auto dynamic_storage = matrix_dynamic_quad_mesh<TestType>();
  const auto edge_storage = matrix_edge_mesh<TestType>();
  const auto cloud_storage = matrix_cloud<TestType>();
  const auto mesh = mesh_storage.mesh();
  const auto dynamic = dynamic_storage.mesh();
  const auto edges = edge_storage.edge_mesh();
  const auto cloud = cloud_storage.point_cloud();
  const auto query = matrix_point<TestType>(Real{0.5}, Real{1});

  const auto mesh_result = tf::cpp::neighbor_search(mesh, query);
  const auto dynamic_result = tf::cpp::neighbor_search(dynamic, query);
  const auto edge_result = tf::cpp::neighbor_search(edges, query);
  const auto cloud_result = tf::cpp::neighbor_search(cloud, query);
  static_assert(std::is_same_v<decltype(mesh_result.element_id), Index>);
  static_assert(std::is_same_v<decltype(dynamic_result.element_id), Index>);
  static_assert(std::is_same_v<decltype(edge_result.element_id), Index>);
  static_assert(
      std::is_same_v<decltype(cloud_result.element_id), std::int32_t>);
  require_matrix_shape(mesh_result.point, {static_cast<int>(Dims)});
  require_matrix_shape(dynamic_result.point, {static_cast<int>(Dims)});
  require_matrix_shape(edge_result.point, {static_cast<int>(Dims)});
  require_matrix_shape(cloud_result.point, {static_cast<int>(Dims)});
  CHECK(mesh_result.element_id == Index{0});
  CHECK(dynamic_result.element_id == Index{0});
  CHECK(edge_result.element_id == Index{0});
  CHECK(cloud_result.element_id == 0);
  CHECK(edge_result.distance2 == Catch::Approx(1));
  CHECK(edge_result.point[0] == Catch::Approx(Real{0.5}));
  CHECK(edge_result.point[1] == Catch::Approx(Real{0}));

  const std::array<tf::cpp::primitive_kind, 6> kinds{
      tf::cpp::primitive_kind::point,    tf::cpp::primitive_kind::segment,
      tf::cpp::primitive_kind::triangle, tf::cpp::primitive_kind::ray,
      tf::cpp::primitive_kind::line,     tf::cpp::primitive_kind::polygon};
  for (const auto kind : kinds) {
    const auto actual =
        tf::cpp::neighbor_search(edges, matrix_origin_query<TestType>(kind));
    CHECK(actual.element_id >= Index{0});
    CHECK(actual.distance2 == Catch::Approx(0));
    require_matrix_shape(actual.point, {static_cast<int>(Dims)});
  }
  const auto box = tf::cpp::neighbor_search(
      edges, matrix_origin_query<TestType>(tf::cpp::primitive_kind::aabb));
  CHECK(box.distance2 == Catch::Approx(0));
  if constexpr (Dims == 3) {
    const auto plane = tf::cpp::neighbor_search(
        edges, matrix_origin_query<TestType>(tf::cpp::primitive_kind::plane));
    CHECK(plane.distance2 == Catch::Approx(0));
  }

  const auto mesh_knn = tf::cpp::neighbor_search_knn(mesh, query, 4);
  const auto dynamic_knn = tf::cpp::neighbor_search_knn(dynamic, query, 4);
  const auto edge_knn = tf::cpp::neighbor_search_knn(edges, query, 4);
  const auto cloud_knn = tf::cpp::neighbor_search_knn(cloud, query, 4);
  CHECK(mesh_knn.element_ids.size() == 1);
  CHECK(dynamic_knn.element_ids.size() == 1);
  CHECK(edge_knn.element_ids.size() == 3);
  CHECK(cloud_knn.element_ids.size() == 4);
  require_matrix_shape(edge_knn.points, {3, static_cast<int>(Dims)});

  const auto batch_queries = matrix_point_batch<TestType>();
  const auto batch =
      tf::cpp::neighbor_search_batch(dynamic, batch_queries, Real{2});
  require_matrix_shape(batch.element_ids, {2});
  require_matrix_shape(batch.points, {2, static_cast<int>(Dims)});
  CHECK(batch.element_ids[0] == Index{0});
  CHECK(batch.element_ids[1] == Index{-1});
  CHECK(batch.distances[1] == Catch::Approx(4));

  const auto padded =
      tf::cpp::neighbor_search_knn_batch(edges, batch_queries, 5, Real{2});
  require_matrix_shape(padded.element_ids, {2, 5});
  require_matrix_shape(padded.points, {2, 5, static_cast<int>(Dims)});
  require_matrix_shape(padded.distances, {2, 5});
  require_matrix_shape(padded.counts, {2});
  CHECK(padded.counts[0] == 2);
  CHECK(padded.counts[1] == 0);
  CHECK(padded.element_ids[2] == Index{-1});
  CHECK(padded.element_ids[5] == Index{-1});
  CHECK(padded.distances[2] == Catch::Approx(4));

  const auto one =
      tf::cpp::primitive<Real, Dims>(tf::cpp::primitive_kind::point, [&] {
        if constexpr (Dims == 2)
          return make_array<Real>({0, 0}, {1, 2});
        else
          return make_array<Real>({0, 0, 0}, {1, 3});
      }());
  const auto one_batch = tf::cpp::neighbor_search_batch(cloud, one);
  require_matrix_shape(one_batch.element_ids, {1});
  require_matrix_shape(one_batch.points, {1, static_cast<int>(Dims)});

  CHECK(tf::cpp::neighbor_search(edges, query, Real{1}).element_id ==
        Index{-1});
  CHECK(mesh_storage.cache.tree_build_count() == 1);
  CHECK(dynamic_storage.cache.tree_build_count() == 1);
  CHECK(edge_storage.cache.tree_build_count() == 1);
  CHECK(cloud_storage.cache.tree_build_count() == 1);
}

TEMPLATE_LIST_TEST_CASE(
    "Python neighbor form pairs cover mesh point cloud and edge mesh order",
    "[cpp][spatial][neighbor-search][python-parity][matrix][pair]",
    neighbor_matrix_rows) {
  using Real = typename TestType::real_type;
  using Index = typename TestType::index_type;
  constexpr auto Dims = TestType::dims;
  const auto mesh_storage = matrix_triangle_mesh<TestType>();
  const auto dynamic_storage = matrix_dynamic_quad_mesh<TestType>();
  const auto edge_storage = matrix_edge_mesh<TestType>();
  const auto cloud_storage = matrix_cloud<TestType>();
  const auto mesh = mesh_storage.mesh();
  const auto dynamic = dynamic_storage.mesh();
  const auto edges = edge_storage.edge_mesh();
  const auto cloud = cloud_storage.point_cloud();

  const auto mm = tf::cpp::neighbor_search(mesh, dynamic);
  const auto mp = tf::cpp::neighbor_search(mesh, cloud);
  const auto me = tf::cpp::neighbor_search(mesh, edges);
  const auto pm = tf::cpp::neighbor_search(cloud, mesh);
  const auto pp = tf::cpp::neighbor_search(cloud, cloud);
  const auto pe = tf::cpp::neighbor_search(cloud, edges);
  const auto em = tf::cpp::neighbor_search(edges, mesh);
  const auto ep = tf::cpp::neighbor_search(edges, cloud);
  const auto ee = tf::cpp::neighbor_search(edges, edges);
  static_assert(std::is_same_v<decltype(me.element_id0), Index>);
  static_assert(std::is_same_v<decltype(me.element_id1), Index>);
  static_assert(std::is_same_v<decltype(pe.element_id0), std::int32_t>);
  static_assert(std::is_same_v<decltype(pe.element_id1), Index>);

  for (const auto distance :
       {mm.distance2, mp.distance2, me.distance2, pm.distance2, pp.distance2,
        pe.distance2, em.distance2, ep.distance2, ee.distance2})
    CHECK(distance == Catch::Approx(0));
  CHECK(mm.element_id0 == Index{0});
  CHECK(mm.element_id1 == Index{0});
  CHECK(me.element_id0 == Index{0});
  CHECK(me.element_id1 == Index{0});
  CHECK(pe.element_id0 == 0);
  CHECK(pe.element_id1 == Index{0});
  require_matrix_shape(me.point0, {static_cast<int>(Dims)});
  require_matrix_shape(me.point1, {static_cast<int>(Dims)});

  auto translated_storage = matrix_edge_mesh<TestType>();
  translated_storage.place(matrix_translation<TestType>(0, 3));
  const auto translated_edges = translated_storage.edge_mesh();
  const auto transformed = tf::cpp::neighbor_search(cloud, translated_edges);
  CHECK(transformed.element_id0 == 0);
  CHECK(transformed.element_id1 == Index{0});
  CHECK(transformed.distance2 == Catch::Approx(9));
  CHECK(transformed.point0[1] == Catch::Approx(0));
  CHECK(transformed.point1[1] == Catch::Approx(3));
  const auto miss = tf::cpp::neighbor_search(cloud, translated_edges, Real{3});
  CHECK(miss.element_id0 == -1);
  CHECK(miss.element_id1 == Index{-1});
  CHECK(miss.distance2 == Catch::Approx(9));
}

TEMPLATE_LIST_TEST_CASE(
    "Python neighbor transforms ties validation and dynamic OBA queries are "
    "exact",
    "[cpp][spatial][neighbor-search][python-parity][matrix][validation]",
    neighbor_matrix_rows) {
  using Real = typename TestType::real_type;
  using Index = typename TestType::index_type;
  constexpr auto Dims = TestType::dims;
  auto mesh_storage = matrix_dynamic_quad_mesh<TestType>();
  auto edge_storage = matrix_edge_mesh<TestType>();
  auto cloud_storage = matrix_cloud<TestType>();

  // Exact Python OBA quad fixture: one dynamic face contains the center point.
  const auto center = matrix_point<TestType>(Real{0.5}, Real{0.5});
  const auto dynamic_result =
      tf::cpp::neighbor_search(mesh_storage.mesh(), center);
  CHECK(dynamic_result.element_id == Index{0});
  CHECK(dynamic_result.distance2 == Catch::Approx(0));

  // the placement is this instance's, so a moved carrier is a new assembly
  mesh_storage.place(matrix_translation<TestType>(10, 5));
  edge_storage.place(matrix_translation<TestType>(10, 5));
  cloud_storage.place(matrix_translation<TestType>(10, 5));
  const auto transformed_query = matrix_point<TestType>(Real{10.5}, Real{6});
  CHECK(tf::cpp::neighbor_search(mesh_storage.mesh(), transformed_query)
            .distance2 == Catch::Approx(0));
  CHECK(tf::cpp::neighbor_search(edge_storage.edge_mesh(), transformed_query)
            .distance2 == Catch::Approx(1));
  CHECK(tf::cpp::neighbor_search(cloud_storage.point_cloud(), transformed_query)
            .distance2 == Catch::Approx(1.25));

  cloud_storage.place(matrix_quarter_turn<TestType>());
  CHECK(tf::cpp::neighbor_search(cloud_storage.point_cloud(),
                                 matrix_point<TestType>(Real{0}, Real{1}))
            .distance2 == Catch::Approx(0));
  cloud_storage.place(matrix_double_scale<TestType>());
  CHECK(tf::cpp::neighbor_search(cloud_storage.point_cloud(),
                                 matrix_point<TestType>(Real{6}, Real{0}))
            .distance2 == Catch::Approx(0));
  cloud_storage.place(matrix_translation<TestType>(0, 0));
  CHECK(tf::cpp::neighbor_search(cloud_storage.point_cloud(),
                                 matrix_point<TestType>(Real{6}, Real{0}))
            .distance2 == Catch::Approx(9));

  const auto tie_storage = [&] {
    if constexpr (Dims == 2)
      return tf::cpp::test::owned_point_cloud<Real, 2>{
          tf::cpp::test::points_of<Real, 2>({-1, 0, 1, 0})};
    else
      return tf::cpp::test::owned_point_cloud<Real, 3>{
          tf::cpp::test::points_of<Real, 3>({-1, 0, 0, 1, 0, 0})};
  }();
  const auto tie = tie_storage.point_cloud();
  const auto tied =
      tf::cpp::neighbor_search(tie, matrix_point<TestType>(Real{0}, Real{0}));
  CHECK((tied.element_id == 0 || tied.element_id == 1));
  CHECK(tied.distance2 == Catch::Approx(1));
  const auto tied_knn = tf::cpp::neighbor_search_knn(
      tie, matrix_point<TestType>(Real{0}, Real{0}), 2);
  CHECK(tied_knn.element_ids.size() == 2);
  CHECK(tied_knn.distances[0] == Catch::Approx(1));
  CHECK(tied_knn.distances[1] == Catch::Approx(1));

  // an empty edge mesh has nothing to be near, so the search answers a miss
  const auto nothing_storage =
      tf::cpp::test::owned_edge_mesh<Index, Real, Dims>{};
  const auto edges = edge_storage.edge_mesh();
  const auto query = matrix_point<TestType>(Real{0}, Real{0});
  CHECK(
      tf::cpp::neighbor_search(nothing_storage.edge_mesh(), query).element_id ==
      -1);
  CHECK_THROWS_AS(tf::cpp::neighbor_search(edges, query, Real{-1}),
                  std::invalid_argument);
  CHECK_THROWS_AS(tf::cpp::neighbor_search(
                      edges, query, std::numeric_limits<Real>::quiet_NaN()),
                  std::invalid_argument);
  CHECK_THROWS_AS(tf::cpp::neighbor_search_knn(edges, query, 0),
                  std::invalid_argument);
  CHECK_THROWS_AS(tf::cpp::neighbor_search_batch(edges, query),
                  std::invalid_argument);
  const auto batch = matrix_point_batch<TestType>();
  CHECK_THROWS_AS(tf::cpp::neighbor_search(edges, batch),
                  std::invalid_argument);
  const auto vector = [&] {
    if constexpr (Dims == 2)
      return tf::cpp::primitive<Real, 2>(tf::cpp::primitive_kind::vector,
                                         make_array<Real>({1, 0}, {2}));
    else
      return tf::cpp::primitive<Real, 3>(tf::cpp::primitive_kind::vector,
                                         make_array<Real>({1, 0, 0}, {3}));
  }();
  CHECK_THROWS_AS(tf::cpp::neighbor_search(edges, vector),
                  std::invalid_argument);
}

TEST_CASE("neighbor pair IDs follow mixed source index types and dimensions",
          "[cpp][spatial][neighbor-search][python-parity][matrix][pair]") {
  using Row64 = neighbor_matrix_row<std::int64_t, double, 2>;
  using Row32 = neighbor_matrix_row<std::int32_t, double, 2>;
  const auto mesh64_storage = matrix_dynamic_quad_mesh<Row64>();
  const auto mesh32_storage = matrix_triangle_mesh<Row32>();
  const auto edge64_storage = matrix_edge_mesh<Row64>();
  const auto edge32_storage = matrix_edge_mesh<Row32>();
  const auto mesh64 = mesh64_storage.mesh();
  const auto mesh32 = mesh32_storage.mesh();
  const auto edge64 = edge64_storage.edge_mesh();
  const auto edge32 = edge32_storage.edge_mesh();

  const auto mm = tf::cpp::neighbor_search(mesh64, mesh32);
  const auto me = tf::cpp::neighbor_search(mesh64, edge32);
  const auto em = tf::cpp::neighbor_search(edge32, mesh64);
  const auto ee = tf::cpp::neighbor_search(edge64, edge32);
  static_assert(std::is_same_v<decltype(mm.element_id0), std::int64_t>);
  static_assert(std::is_same_v<decltype(mm.element_id1), std::int32_t>);
  static_assert(std::is_same_v<decltype(me.element_id0), std::int64_t>);
  static_assert(std::is_same_v<decltype(me.element_id1), std::int32_t>);
  static_assert(std::is_same_v<decltype(em.element_id0), std::int32_t>);
  static_assert(std::is_same_v<decltype(em.element_id1), std::int64_t>);
  static_assert(std::is_same_v<decltype(ee.element_id0), std::int64_t>);
  static_assert(std::is_same_v<decltype(ee.element_id1), std::int32_t>);
  CHECK(mm.distance2 == Catch::Approx(0));
  CHECK(me.distance2 == Catch::Approx(0));
  CHECK(em.distance2 == Catch::Approx(0));
  CHECK(ee.distance2 == Catch::Approx(0));

  using mesh2 = tf::cpp::mesh<std::int32_t, float, 2>;
  using mesh3 = tf::cpp::mesh<std::int32_t, float, 3>;
  using edge2 = tf::cpp::edge_mesh<std::int32_t, float, 2>;
  using edge3 = tf::cpp::edge_mesh<std::int32_t, float, 3>;
  using point2 = tf::cpp::primitive<float, 2>;
  using point3 = tf::cpp::primitive<float, 3>;
  static_assert(!is_neighbor_invocable<mesh2, point3>::value);
  static_assert(!is_neighbor_invocable<mesh3, point2>::value);
  static_assert(!is_neighbor_invocable<mesh2, edge3>::value);
  static_assert(!is_neighbor_invocable<edge2, mesh3>::value);
}

TEST_CASE("generalized neighbor async APIs retain the query and answer from "
          "the reading they were handed",
          "[cpp][spatial][neighbor-search][async][matrix]") {
  using Row = neighbor_matrix_row<std::int64_t, double, 2>;
  const auto query = matrix_point<Row>(0.5, 1);

  // the carriers are borrowed, so the storage they name is kept alive by the
  // keepalive they were assembled with and no handle of the caller's survives
  auto mesh_pending = tf::cpp::async::neighbor_search(
      held_carrier_of(matrix_dynamic_quad_mesh<Row>()), query);
  static_assert(
      std::is_same_v<
          decltype(mesh_pending),
          std::future<tf::cpp::neighbor_result<std::int64_t, double, 2>>>);
  CHECK(mesh_pending.get().distance2 == Catch::Approx(0));

  auto edge_pending = tf::cpp::async::neighbor_search(
      held_carrier_of(matrix_edge_mesh<Row>()), query);
  CHECK(edge_pending.get().distance2 == Catch::Approx(1));

  auto cloud_pending = tf::cpp::async::neighbor_search(
      held_carrier_of(matrix_cloud<Row>()), query);
  CHECK(cloud_pending.get().distance2 == Catch::Approx(1.25));

  // the query is carried as the handle it is, so the caller may release its
  // own before the job runs
  const auto query_edge_storage = matrix_edge_mesh<Row>();
  auto released_query = matrix_point<Row>(0.5, 1);
  auto released_values = released_query.data();
  auto query_pending = tf::cpp::async::neighbor_search(
      query_releasing_resolver<double>{&released_values},
      query_edge_storage.edge_mesh(), released_query);
  CHECK_FALSE(released_values.is_valid());
  CHECK(query_pending.get().distance2 == Catch::Approx(1));

  auto pair =
      tf::cpp::async::neighbor_search(held_carrier_of(matrix_cloud<Row>()),
                                      held_carrier_of(matrix_edge_mesh<Row>()));
  const auto pair_result = pair.get();
  CHECK(pair_result.distance2 == Catch::Approx(0));
  CHECK(pair_result.element_id0 == 0);
  CHECK(pair_result.element_id1 == std::int64_t{0});
}

TEST_CASE("generalized neighbor symbols retain typed shard extraction",
          "[cpp][spatial][neighbor-search][archive-link][matrix]") {
  using form_type = tf::cpp::edge_mesh<std::int64_t, double, 2>;
  using query_type = tf::cpp::primitive<double, 2>;
  using single_function = tf::cpp::neighbor_result<std::int64_t, double, 2> (*)(
      const form_type &, const query_type &, double);
  using batch_function =
      tf::cpp::neighbor_knn_batch_result<std::int64_t, double, 2> (*)(
          const form_type &, const query_type &, int, double);
  const auto single = static_cast<single_function>(
      &tf::cpp::neighbor_search<std::int64_t, double, 2>);
  const auto batch = static_cast<batch_function>(
      &tf::cpp::neighbor_search_knn_batch<std::int64_t, double, 2>);

  using Row = neighbor_matrix_row<std::int64_t, double, 2>;
  const auto owner = matrix_edge_mesh<Row>();
  const auto form = owner.edge_mesh();
  const auto query = matrix_point<Row>(0.5, 1);
  const auto queries = matrix_point_batch<Row>();
  CHECK(
      single(form, query, std::numeric_limits<double>::infinity()).element_id ==
      std::int64_t{0});
  CHECK(batch(form, queries, 2, std::numeric_limits<double>::infinity())
            .counts[0] == 2);
}
