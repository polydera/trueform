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

#include "trueform/cpp/core/mesh.hpp"
#include "trueform/cpp/spatial/async/gather_ids.hpp"
#include "trueform/cpp/spatial/async/gather_ids_within_distance.hpp"
#include "trueform/cpp/spatial/gather_ids.hpp"
#include "trueform/cpp/spatial/gather_ids_within_distance.hpp"

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <future>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

template <typename T>
auto make_array(std::initializer_list<T> values, tf::small_vector<int, 3> shape)
    -> tf::cpp::nd_array<T> {
  tf::buffer<T> output;
  output.allocate(values.size());
  std::copy(values.begin(), values.end(), output.begin());
  return tf::cpp::nd_array<T>::from_buffer(std::move(output), std::move(shape));
}

/// The placement a carrier is assembled at, in the shape `place` states it.
template <typename Real, std::size_t Dims>
auto identity_with_translation(std::size_t axis, Real offset)
    -> std::array<Real, (Dims + 1) * (Dims + 1)> {
  constexpr auto side = Dims + 1;
  std::array<Real, side * side> values{};
  for (std::size_t index = 0; index < side; ++index)
    values[index * side + index] = Real{1};
  values[axis * side + Dims] = offset;
  return values;
}

template <typename Index, typename Real, std::size_t Dims>
auto fixed_mesh() -> tf::cpp::test::owned_mesh<Index, Real, Dims> {
  if constexpr (Dims == 2) {
    return {tf::cpp::test::polygons_of<Index, Real, Dims>(
        {0, 1, 2, 1, 3, 2}, {0, 0, 1, 0, 0, 1, 1, 1})};
  } else {
    return {tf::cpp::test::polygons_of<Index, Real, Dims>(
        {0, 1, 2, 1, 3, 2}, {0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0})};
  }
}

template <typename Real, std::size_t Dims>
auto origin_cloud(bool duplicate = false)
    -> tf::cpp::test::owned_point_cloud<Real, Dims> {
  if constexpr (Dims == 2) {
    if (duplicate)
      return {tf::cpp::test::points_of<Real, Dims>({0, 0, 0, 0})};
    return {tf::cpp::test::points_of<Real, Dims>({0, 0})};
  } else {
    if (duplicate)
      return {tf::cpp::test::points_of<Real, Dims>({0, 0, 0, 0, 0, 0})};
    return {tf::cpp::test::points_of<Real, Dims>({0, 0, 0})};
  }
}

template <typename Real, std::size_t Dims>
auto canonical_points() -> std::vector<Real> {
  if constexpr (Dims == 2)
    return {Real(-0.25), Real(-0.25), Real(0.25), Real(-0.25), 0, Real(0.25),
            Real(3.75),  Real(3.75),  Real(4.25), Real(3.75),  4, Real(4.25)};
  else
    return {Real(-0.25), Real(-0.25), 0, Real(0.25), Real(-0.25), 0,
            0,           Real(0.25),  0, Real(3.75), Real(3.75),  4,
            Real(4.25),  Real(3.75),  4, 4,          Real(4.25),  4};
}

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon = 3>
auto canonical_mesh() -> tf::cpp::test::owned_mesh<Index, Real, Dims, Ngon> {
  if constexpr (Ngon == 3)
    return {tf::cpp::test::polygons_of<Index, Real, Dims>(
        std::initializer_list<Index>{0, 1, 2, 3, 4, 5},
        canonical_points<Real, Dims>())};
  else
    return {tf::cpp::test::polygons_of<Index, Real, Dims>(
        std::initializer_list<Index>{0, 3, 6},
        std::initializer_list<Index>{0, 1, 2, 3, 4, 5},
        canonical_points<Real, Dims>())};
}

template <typename Index, typename Real, std::size_t Dims>
auto canonical_edges() -> tf::cpp::test::owned_edge_mesh<Index, Real, Dims> {
  if constexpr (Dims == 2) {
    return {tf::cpp::test::segments_of<Index, Real, Dims>(
        {0, 1, 2, 3, 4, 5}, {Real(-0.2), 0, Real(0.2), 0, Real(3.8), 4,
                             Real(4.2), 4, Real(7.8), 8, Real(8.2), 8})};
  } else {
    return {tf::cpp::test::segments_of<Index, Real, Dims>(
        {0, 1, 2, 3, 4, 5},
        {Real(-0.2), 0, 0, Real(0.2), 0, 0, Real(3.8), 4, 4, Real(4.2), 4, 4,
         Real(7.8), 8, 8, Real(8.2), 8, 8})};
  }
}

template <typename Real, std::size_t Dims>
auto canonical_cloud() -> tf::cpp::test::owned_point_cloud<Real, Dims> {
  if constexpr (Dims == 2)
    return {tf::cpp::test::points_of<Real, Dims>({0, 0, 4, 4, 8, 8})};
  else
    return {tf::cpp::test::points_of<Real, Dims>({0, 0, 0, 4, 4, 4, 8, 8, 8})};
}

template <typename Real, std::size_t Dims>
auto origin_primitive(tf::cpp::primitive_kind kind, Real offset = Real{})
    -> tf::cpp::primitive<Real, Dims> {
  const auto lo = offset - Real{1};
  const auto hi = offset + Real{1};
  if constexpr (Dims == 2) {
    switch (kind) {
    case tf::cpp::primitive_kind::point:
      return {kind, make_array<Real>({offset, offset}, {2})};
    case tf::cpp::primitive_kind::segment:
      return {kind, make_array<Real>({lo, offset, hi, offset}, {2, 2})};
    case tf::cpp::primitive_kind::triangle:
      return {kind, make_array<Real>({lo, lo, hi, lo, offset, hi}, {3, 2})};
    case tf::cpp::primitive_kind::ray:
    case tf::cpp::primitive_kind::line:
      return {kind, make_array<Real>({offset, offset, 1, 0}, {2, 2})};
    case tf::cpp::primitive_kind::aabb:
      return {kind, make_array<Real>({lo, lo, hi, hi}, {2, 2})};
    case tf::cpp::primitive_kind::polygon:
      return {kind, make_array<Real>({lo, lo, hi, lo, hi, hi, lo, hi}, {4, 2})};
    case tf::cpp::primitive_kind::vector:
      return {kind, make_array<Real>({1, 0}, {2})};
    case tf::cpp::primitive_kind::plane:
      break;
    }
  } else {
    switch (kind) {
    case tf::cpp::primitive_kind::point:
      return {kind, make_array<Real>({offset, offset, offset}, {3})};
    case tf::cpp::primitive_kind::segment:
      return {kind, make_array<Real>({lo, offset, offset, hi, offset, offset},
                                     {2, 3})};
    case tf::cpp::primitive_kind::triangle:
      return {kind, make_array<Real>(
                        {lo, lo, offset, hi, lo, offset, offset, hi, offset},
                        {3, 3})};
    case tf::cpp::primitive_kind::ray:
    case tf::cpp::primitive_kind::line:
      return {kind,
              make_array<Real>({offset, offset, offset, 1, 0, 0}, {2, 3})};
    case tf::cpp::primitive_kind::plane:
      return {kind, make_array<Real>({0, 0, 1, -offset}, {4})};
    case tf::cpp::primitive_kind::aabb:
      return {kind, make_array<Real>({lo, lo, lo, hi, hi, hi}, {2, 3})};
    case tf::cpp::primitive_kind::polygon:
      return {kind, make_array<Real>({lo, lo, offset, hi, lo, offset, hi, hi,
                                      offset, lo, hi, offset},
                                     {4, 3})};
    case tf::cpp::primitive_kind::vector:
      return {kind, make_array<Real>({1, 0, 0}, {3})};
    }
  }
  throw std::logic_error("unsupported primitive fixture");
}

template <typename Index>
auto canonical_ids(const tf::cpp::nd_array<Index> &values)
    -> std::vector<Index> {
  REQUIRE(values.ndim() == 1);
  std::vector<Index> result(values.begin(), values.end());
  std::sort(result.begin(), result.end());
  return result;
}

template <typename Index>
auto canonical_pairs(const tf::cpp::nd_array<Index> &values)
    -> std::vector<std::pair<Index, Index>> {
  REQUIRE(values.ndim() == 2);
  REQUIRE(values.shape_at(1) == 2);
  std::vector<std::pair<Index, Index>> result;
  result.reserve(values.shape_at(0));
  for (int row = 0; row < values.shape_at(0); ++row)
    result.emplace_back(values[row * 2], values[row * 2 + 1]);
  std::sort(result.begin(), result.end());
  return result;
}

template <typename Index>
auto swapped(std::vector<std::pair<Index, Index>> values)
    -> std::vector<std::pair<Index, Index>> {
  for (auto &value : values)
    std::swap(value.first, value.second);
  std::sort(values.begin(), values.end());
  return values;
}

template <typename Left, typename Right, typename = void>
struct has_gather_ids : std::false_type {};
template <typename Left, typename Right>
struct has_gather_ids<
    Left, Right,
    std::void_t<decltype(tf::cpp::gather_ids(std::declval<const Left &>(),
                                             std::declval<const Right &>()))>>
    : std::true_type {};

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

struct mutating_resolver {
  std::function<void()> mutate;
  std::shared_ptr<std::atomic<int>> submissions;

  template <typename T>
  using state_type = tf::cpp::async::detail::future_state<T>;

  template <typename T> auto make_state() -> std::shared_ptr<state_type<T>> {
    submissions->fetch_add(1, std::memory_order_relaxed);
    mutate();
    return std::make_shared<state_type<T>>();
  }
};

template <typename Index, typename Real, std::size_t Dims>
auto check_transformed_pair_ids() -> void {
  const auto mesh_storage = canonical_mesh<Index, Real, Dims>();
  auto moved_mesh_storage =
      canonical_mesh<Index, Real, Dims, tf::dynamic_size>();
  const auto edge_storage = canonical_edges<Index, Real, Dims>();
  auto moved_edge_storage = canonical_edges<Index, Real, Dims>();
  const auto cloud_storage = canonical_cloud<Real, Dims>();
  auto moved_cloud_storage = canonical_cloud<Real, Dims>();

  const auto mesh = mesh_storage.mesh();
  const auto edges = edge_storage.edge_mesh();
  const auto cloud = cloud_storage.point_cloud();
  const std::vector<std::pair<Index, Index>> mesh_pairs{{0, 0}, {1, 1}};
  const std::vector<std::pair<Index, Index>> edge_pairs{{0, 0}, {1, 1}, {2, 2}};
  const std::vector<std::pair<std::int32_t, std::int32_t>> cloud_pairs{
      {0, 0}, {1, 1}, {2, 2}};

  CHECK(canonical_pairs(tf::cpp::gather_ids(mesh, moved_mesh_storage.mesh())) ==
        mesh_pairs);
  CHECK(canonical_pairs(tf::cpp::gather_ids(
            edges, moved_edge_storage.edge_mesh())) == edge_pairs);
  CHECK(canonical_pairs(tf::cpp::gather_ids(
            cloud, moved_cloud_storage.point_cloud())) == cloud_pairs);

  // the placement is this instance's, so a moved carrier is a new assembly
  moved_mesh_storage.place(identity_with_translation<Real, Dims>(1, Real{1}));
  moved_edge_storage.place(identity_with_translation<Real, Dims>(1, Real{1}));
  moved_cloud_storage.place(identity_with_translation<Real, Dims>(1, Real{1}));
  const auto transformed_mesh = moved_mesh_storage.mesh();
  const auto transformed_edges = moved_edge_storage.edge_mesh();
  const auto transformed_cloud = moved_cloud_storage.point_cloud();
  CHECK(canonical_pairs(tf::cpp::gather_ids(mesh, transformed_mesh)).empty());
  CHECK(canonical_pairs(tf::cpp::gather_ids(mesh, transformed_edges)).empty());
  CHECK(canonical_pairs(tf::cpp::gather_ids(mesh, transformed_cloud)).empty());
  CHECK(canonical_pairs(tf::cpp::gather_ids(edges, transformed_mesh)).empty());
  CHECK(canonical_pairs(tf::cpp::gather_ids(edges, transformed_edges)).empty());
  CHECK(canonical_pairs(tf::cpp::gather_ids(edges, transformed_cloud)).empty());
  CHECK(canonical_pairs(tf::cpp::gather_ids(cloud, transformed_mesh)).empty());
  CHECK(canonical_pairs(tf::cpp::gather_ids(cloud, transformed_edges)).empty());
  CHECK(canonical_pairs(tf::cpp::gather_ids(cloud, transformed_cloud)).empty());
  CHECK(canonical_pairs(tf::cpp::gather_ids_within_distance(
            mesh, transformed_mesh, Real{1})) == mesh_pairs);
  CHECK(canonical_pairs(tf::cpp::gather_ids_within_distance(
            mesh, transformed_edges, Real{1})) == mesh_pairs);
  CHECK(canonical_pairs(tf::cpp::gather_ids_within_distance(
            mesh, transformed_cloud, Real{1})) == mesh_pairs);
  CHECK(canonical_pairs(tf::cpp::gather_ids_within_distance(
            edges, transformed_mesh, Real{1})) == mesh_pairs);
  CHECK(canonical_pairs(tf::cpp::gather_ids_within_distance(
            edges, transformed_edges, Real{1})) == edge_pairs);
  CHECK(canonical_pairs(tf::cpp::gather_ids_within_distance(
            edges, transformed_cloud, Real{1})) == edge_pairs);
  CHECK(canonical_pairs(tf::cpp::gather_ids_within_distance(
            cloud, transformed_mesh, Real{1})) == mesh_pairs);
  CHECK(canonical_pairs(tf::cpp::gather_ids_within_distance(
            cloud, transformed_edges, Real{1})) == edge_pairs);
  CHECK(canonical_pairs(tf::cpp::gather_ids_within_distance(
            cloud, transformed_cloud, Real{1})) == cloud_pairs);
}

template <typename Index, typename Real, std::size_t Dims>
auto check_archive_matrix() -> void {
  const auto mesh_storage = canonical_mesh<Index, Real, Dims>();
  const auto dynamic_storage =
      canonical_mesh<Index, Real, Dims, tf::dynamic_size>();
  const auto edge_storage = canonical_edges<Index, Real, Dims>();
  const auto cloud_storage = canonical_cloud<Real, Dims>();
  const auto mesh = mesh_storage.mesh();
  const auto dynamic = dynamic_storage.mesh();
  const auto edges = edge_storage.edge_mesh();
  const auto cloud = cloud_storage.point_cloud();
  auto query = origin_primitive<Real, Dims>(tf::cpp::primitive_kind::point);
  const std::vector<Index> first{0};
  const std::vector<std::pair<Index, Index>> mesh_pairs{{0, 0}, {1, 1}};
  const std::vector<std::pair<Index, Index>> edge_pairs{{0, 0}, {1, 1}, {2, 2}};
  const std::vector<std::pair<std::int32_t, std::int32_t>> cloud_pairs{
      {0, 0}, {1, 1}, {2, 2}};

  CHECK(canonical_ids(tf::cpp::gather_ids(mesh, query)) == first);
  CHECK(canonical_ids(tf::cpp::gather_ids(dynamic, query)) == first);
  CHECK(canonical_ids(tf::cpp::gather_ids(edges, query)) == first);
  CHECK(canonical_ids(tf::cpp::gather_ids(cloud, query)) ==
        std::vector<std::int32_t>{0});

  CHECK(canonical_pairs(tf::cpp::gather_ids(mesh, dynamic)) == mesh_pairs);
  CHECK(canonical_pairs(tf::cpp::gather_ids(mesh, edges)) == mesh_pairs);
  CHECK(canonical_pairs(tf::cpp::gather_ids(edges, mesh)) == mesh_pairs);
  CHECK(canonical_pairs(tf::cpp::gather_ids(edges, edges)) == edge_pairs);
  CHECK(canonical_pairs(tf::cpp::gather_ids(mesh, cloud)) == mesh_pairs);
  CHECK(canonical_pairs(tf::cpp::gather_ids(cloud, mesh)) == mesh_pairs);
  CHECK(canonical_pairs(tf::cpp::gather_ids(edges, cloud)) == edge_pairs);
  CHECK(canonical_pairs(tf::cpp::gather_ids(cloud, edges)) == edge_pairs);
  CHECK(canonical_pairs(tf::cpp::gather_ids(cloud, cloud)) == cloud_pairs);

  CHECK(canonical_ids(tf::cpp::gather_ids_within_distance(mesh, query,
                                                          Real{0})) == first);
  CHECK(canonical_pairs(tf::cpp::gather_ids_within_distance(
            dynamic, edges, Real(0.01))) == mesh_pairs);
  CHECK(canonical_pairs(
            tf::cpp::async::gather_ids_within_distance(cloud, edges, Real{0})
                .get()) == edge_pairs);
}

} // namespace

TEST_CASE("gather archive shards cover every Real Index and Dims matrix",
          "[cpp][spatial][gather_ids][matrix][archive]") {
  check_archive_matrix<std::int32_t, float, 2>();
  check_archive_matrix<std::int64_t, float, 2>();
  check_archive_matrix<std::int32_t, double, 2>();
  check_archive_matrix<std::int64_t, double, 2>();
  check_archive_matrix<std::int32_t, float, 3>();
  check_archive_matrix<std::int64_t, float, 3>();
  check_archive_matrix<std::int32_t, double, 3>();
  check_archive_matrix<std::int64_t, double, 3>();
}

TEMPLATE_TEST_CASE("gather primitive targets match Python exact predicates",
                   "[cpp][spatial][gather_ids][primitive][python-parity]",
                   float, double) {
  constexpr std::array kinds2{
      tf::cpp::primitive_kind::point,    tf::cpp::primitive_kind::segment,
      tf::cpp::primitive_kind::triangle, tf::cpp::primitive_kind::ray,
      tf::cpp::primitive_kind::line,     tf::cpp::primitive_kind::aabb,
      tf::cpp::primitive_kind::polygon};
  constexpr std::array kinds3{
      tf::cpp::primitive_kind::point,    tf::cpp::primitive_kind::segment,
      tf::cpp::primitive_kind::triangle, tf::cpp::primitive_kind::ray,
      tf::cpp::primitive_kind::line,     tf::cpp::primitive_kind::plane,
      tf::cpp::primitive_kind::aabb,     tf::cpp::primitive_kind::polygon};

  const auto mesh2_storage = canonical_mesh<std::int64_t, TestType, 2>();
  const auto edges2_storage = canonical_edges<std::int64_t, TestType, 2>();
  const auto cloud2_storage = canonical_cloud<TestType, 2>();
  const auto mesh2 = mesh2_storage.mesh();
  const auto edges2 = edges2_storage.edge_mesh();
  const auto cloud2 = cloud2_storage.point_cloud();
  for (const auto kind : kinds2) {
    INFO("2D kind " << static_cast<int>(kind));
    auto hit = origin_primitive<TestType, 2>(kind);
    auto miss = origin_primitive<TestType, 2>(kind, TestType{10});
    CHECK(canonical_ids(tf::cpp::gather_ids(mesh2, hit)) ==
          std::vector<std::int64_t>{0});
    CHECK(canonical_ids(tf::cpp::gather_ids(edges2, hit)) ==
          std::vector<std::int64_t>{0});
    CHECK(canonical_ids(tf::cpp::gather_ids(cloud2, hit)) ==
          std::vector<std::int32_t>{0});
    CHECK(canonical_ids(tf::cpp::gather_ids(mesh2, miss)).empty());
    CHECK(canonical_ids(tf::cpp::gather_ids(edges2, miss)).empty());
    CHECK(canonical_ids(tf::cpp::gather_ids(cloud2, miss)).empty());
  }

  const auto mesh3_storage = canonical_mesh<std::int64_t, TestType, 3>();
  const auto edges3_storage = canonical_edges<std::int64_t, TestType, 3>();
  const auto cloud3_storage = canonical_cloud<TestType, 3>();
  const auto mesh3 = mesh3_storage.mesh();
  const auto edges3 = edges3_storage.edge_mesh();
  const auto cloud3 = cloud3_storage.point_cloud();
  for (const auto kind : kinds3) {
    INFO("3D kind " << static_cast<int>(kind));
    auto hit = origin_primitive<TestType, 3>(kind);
    auto miss = origin_primitive<TestType, 3>(kind, TestType{10});
    CHECK(canonical_ids(tf::cpp::gather_ids(mesh3, hit)) ==
          std::vector<std::int64_t>{0});
    CHECK(canonical_ids(tf::cpp::gather_ids(edges3, hit)) ==
          std::vector<std::int64_t>{0});
    CHECK(canonical_ids(tf::cpp::gather_ids(cloud3, hit)) ==
          std::vector<std::int32_t>{0});
    CHECK(canonical_ids(tf::cpp::gather_ids(mesh3, miss)).empty());
    CHECK(canonical_ids(tf::cpp::gather_ids(edges3, miss)).empty());
    CHECK(canonical_ids(tf::cpp::gather_ids(cloud3, miss)).empty());
  }

  auto near2 = tf::cpp::primitive<TestType, 2>(
      tf::cpp::primitive_kind::point,
      make_array<TestType>({0, TestType{0.5}}, {2}));
  CHECK(canonical_ids(tf::cpp::gather_ids(mesh2, near2)).empty());
  CHECK(canonical_ids(tf::cpp::gather_ids(edges2, near2)).empty());
  CHECK(canonical_ids(tf::cpp::gather_ids(cloud2, near2)).empty());
  CHECK(canonical_ids(tf::cpp::gather_ids_within_distance(
            mesh2, near2, TestType{0.5})) == std::vector<std::int64_t>{0});
  CHECK(canonical_ids(tf::cpp::gather_ids_within_distance(
            edges2, near2, TestType{0.5})) == std::vector<std::int64_t>{0});
  CHECK(canonical_ids(tf::cpp::gather_ids_within_distance(
            cloud2, near2, TestType{0.5})) == std::vector<std::int32_t>{0});

  auto near3 = tf::cpp::primitive<TestType, 3>(
      tf::cpp::primitive_kind::point,
      make_array<TestType>({0, TestType{0.5}, 0}, {3}));
  CHECK(canonical_ids(tf::cpp::gather_ids(mesh3, near3)).empty());
  CHECK(canonical_ids(tf::cpp::gather_ids(edges3, near3)).empty());
  CHECK(canonical_ids(tf::cpp::gather_ids(cloud3, near3)).empty());
  CHECK(canonical_ids(tf::cpp::gather_ids_within_distance(
            mesh3, near3, TestType{0.5})) == std::vector<std::int64_t>{0});
  CHECK(canonical_ids(tf::cpp::gather_ids_within_distance(
            edges3, near3, TestType{0.5})) == std::vector<std::int64_t>{0});
  CHECK(canonical_ids(tf::cpp::gather_ids_within_distance(
            cloud3, near3, TestType{0.5})) == std::vector<std::int32_t>{0});
}

TEST_CASE("gather pair results preserve mixed index width and operand order",
          "[cpp][spatial][gather_ids][index][symmetry]") {
  const auto mesh32_storage = canonical_mesh<std::int32_t, float, 2>();
  const auto edge64_storage = canonical_edges<std::int64_t, float, 2>();
  const auto mesh32 = mesh32_storage.mesh();
  const auto edge64 = edge64_storage.edge_mesh();
  const auto forward = tf::cpp::gather_ids(mesh32, edge64);
  const auto reverse = tf::cpp::gather_ids(edge64, mesh32);
  const std::vector<std::pair<std::int64_t, std::int64_t>> expected{{0, 0},
                                                                    {1, 1}};
  STATIC_REQUIRE(std::is_same_v<std::decay_t<decltype(forward)>,
                                tf::cpp::nd_array<std::int64_t>>);
  STATIC_REQUIRE(std::is_same_v<std::decay_t<decltype(reverse)>,
                                tf::cpp::nd_array<std::int64_t>>);
  CHECK(canonical_pairs(forward) == expected);
  CHECK(canonical_pairs(reverse) == swapped(expected));

  const auto cloud_storage = canonical_cloud<float, 2>();
  const auto cloud_edge =
      tf::cpp::gather_ids(cloud_storage.point_cloud(), edge64);
  STATIC_REQUIRE(std::is_same_v<std::decay_t<decltype(cloud_edge)>,
                                tf::cpp::nd_array<std::int64_t>>);
  CHECK(canonical_pairs(cloud_edge) ==
        std::vector<std::pair<std::int64_t, std::int64_t>>{
            {0, 0}, {1, 1}, {2, 2}});
}

TEST_CASE("gather preserves self multiplicity and inclusive boundaries",
          "[cpp][spatial][gather_ids][self][boundary][multiplicity]") {
  const auto mesh_storage = fixed_mesh<std::int32_t, double, 2>();
  const auto mesh = mesh_storage.mesh();
  auto boundary = tf::cpp::primitive<double, 2>(
      tf::cpp::primitive_kind::point, make_array<double>({0.5, 0.5}, {2}));
  CHECK(canonical_ids(tf::cpp::gather_ids(mesh, boundary)) ==
        std::vector<std::int32_t>{0, 1});

  const auto separated_mesh_storage = canonical_mesh<std::int32_t, double, 2>();
  const auto separated_edge_storage =
      canonical_edges<std::int32_t, double, 2>();
  const auto separated_cloud_storage = canonical_cloud<double, 2>();
  const auto separated_mesh = separated_mesh_storage.mesh();
  const auto separated_edges = separated_edge_storage.edge_mesh();
  const auto separated_cloud = separated_cloud_storage.point_cloud();
  CHECK(canonical_pairs(tf::cpp::gather_ids(separated_mesh, separated_mesh)) ==
        std::vector<std::pair<std::int32_t, std::int32_t>>{{0, 0}, {1, 1}});
  CHECK(
      canonical_pairs(tf::cpp::gather_ids(separated_edges, separated_edges)) ==
      std::vector<std::pair<std::int32_t, std::int32_t>>{
          {0, 0}, {1, 1}, {2, 2}});
  CHECK(
      canonical_pairs(tf::cpp::gather_ids(separated_cloud, separated_cloud)) ==
      std::vector<std::pair<std::int32_t, std::int32_t>>{
          {0, 0}, {1, 1}, {2, 2}});

  const auto duplicate_storage = origin_cloud<double, 2>(true);
  auto origin = origin_primitive<double, 2>(tf::cpp::primitive_kind::point);
  CHECK(canonical_ids(
            tf::cpp::gather_ids(duplicate_storage.point_cloud(), origin)) ==
        std::vector<std::int32_t>{0, 1});

  const auto one_storage = origin_cloud<double, 2>();
  auto half = tf::cpp::primitive<double, 2>(tf::cpp::primitive_kind::point,
                                            make_array<double>({0.5, 0}, {2}));
  CHECK(canonical_ids(tf::cpp::gather_ids_within_distance(
            one_storage.point_cloud(), half, 0.5)) ==
        std::vector<std::int32_t>{0});

  const auto boundary_edge_storage =
      tf::cpp::test::owned_edge_mesh<std::int32_t, double, 2>{
          tf::cpp::test::segments_of<std::int32_t, double, 2>({0, 1},
                                                              {0, 0, 1, 0})};
  CHECK(canonical_pairs(
            tf::cpp::gather_ids(mesh, boundary_edge_storage.edge_mesh())) ==
        std::vector<std::pair<std::int32_t, std::int32_t>>{{0, 0}, {1, 0}});
}

TEST_CASE("gather honors fixed dynamic and transformed form snapshots",
          "[cpp][spatial][gather_ids][dynamic][transform]") {
  check_transformed_pair_ids<std::int32_t, float, 2>();
  check_transformed_pair_ids<std::int64_t, double, 3>();
}

TEST_CASE("gather validates thresholds primitives handles and dimensions",
          "[cpp][spatial][gather_ids][validation]") {
  const auto mesh_storage = fixed_mesh<std::int32_t, float, 2>();
  const auto mesh = mesh_storage.mesh();
  auto point = origin_primitive<float, 2>(tf::cpp::primitive_kind::point);
  auto vector = origin_primitive<float, 2>(tf::cpp::primitive_kind::vector);
  auto batch = tf::cpp::primitive<float, 2>(
      tf::cpp::primitive_kind::point, make_array<float>({0, 0, 1, 1}, {2, 2}));
  const auto empty_storage =
      tf::cpp::test::owned_mesh<std::int32_t, float, 2>{};

  CHECK_THROWS_AS(tf::cpp::gather_ids(mesh, vector), std::invalid_argument);
  CHECK_THROWS_AS(tf::cpp::gather_ids(mesh, batch), std::invalid_argument);
  // an empty mesh has nothing to gather
  CHECK(tf::cpp::gather_ids(empty_storage.mesh(), point).empty());
  CHECK_THROWS_AS(tf::cpp::gather_ids_within_distance(mesh, point, -0.1F),
                  std::invalid_argument);
  CHECK_THROWS_AS(tf::cpp::gather_ids_within_distance(
                      mesh, point, std::numeric_limits<float>::quiet_NaN()),
                  std::invalid_argument);

  STATIC_REQUIRE_FALSE(has_gather_ids<tf::cpp::mesh<std::int32_t, float, 2>,
                                      tf::cpp::primitive<float, 3>>::value);
  STATIC_REQUIRE_FALSE(
      has_gather_ids<tf::cpp::mesh<std::int32_t, float, 2>,
                     tf::cpp::mesh<std::int32_t, double, 2>>::value);
}

TEST_CASE("async gather retains the query and answers from the reading it was "
          "handed",
          "[cpp][spatial][gather_ids][async][ownership]") {
  // the query is carried as the handle it is, so the caller may release its
  // own; the carrier is borrowed and the storage it names is kept alive by the
  // keepalive it was assembled with
  const auto cloud_storage = origin_cloud<float, 2>();
  auto query = origin_primitive<float, 2>(tf::cpp::primitive_kind::point);
  const auto submissions = std::make_shared<std::atomic<int>>(0);
  auto result = tf::cpp::async::gather_ids(
      mutating_resolver{[&] {
                          query = origin_primitive<float, 2>(
                              tf::cpp::primitive_kind::point, 20.0F);
                        },
                        submissions},
      cloud_storage.point_cloud(), query);
  CHECK(submissions->load(std::memory_order_relaxed) == 1);
  CHECK(canonical_ids(result.get()) == std::vector<std::int32_t>{0});

  auto point = origin_primitive<double, 2>(tf::cpp::primitive_kind::point);
  auto held = tf::cpp::async::gather_ids(
      held_carrier_of(fixed_mesh<std::int64_t, double, 2>()), point);
  STATIC_REQUIRE(std::is_same_v<decltype(held),
                                std::future<tf::cpp::nd_array<std::int64_t>>>);
  point = tf::cpp::primitive<double, 2>(tf::cpp::primitive_kind::point,
                                        make_array<double>({100, 100}, {2}));
  CHECK(canonical_ids(held.get()) == std::vector<std::int64_t>{0});

  const auto failure_storage = fixed_mesh<std::int32_t, float, 2>();
  auto failure_point =
      origin_primitive<float, 2>(tf::cpp::primitive_kind::point);
  auto failure = tf::cpp::async::gather_ids_within_distance(
      failure_storage.mesh(), failure_point, -1.0F);
  CHECK_THROWS_AS(failure.get(), std::invalid_argument);
}
