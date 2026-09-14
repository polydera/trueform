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
#pragma once

#include "carriers.hpp"

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace intersects_matrix {

template <typename Index, typename Real, std::size_t Dims> struct row {
  using real_type = Real;
  using index_type = Index;
  static constexpr auto dims = Dims;
};

using f32_i32_2 = row<std::int32_t, float, 2>;
using f32_i64_2 = row<std::int64_t, float, 2>;
using f64_i32_2 = row<std::int32_t, double, 2>;
using f64_i64_2 = row<std::int64_t, double, 2>;
using f32_i32_3 = row<std::int32_t, float, 3>;
using f32_i64_3 = row<std::int64_t, float, 3>;
using f64_i32_3 = row<std::int32_t, double, 3>;
using f64_i64_3 = row<std::int64_t, double, 3>;

using form_rows = std::tuple<f32_i32_2, f32_i64_2, f64_i32_2, f64_i64_2,
                             f32_i32_3, f32_i64_3, f64_i32_3, f64_i64_3>;

template <typename Real, std::size_t Dims>
auto point_values(Real x, Real y, Real z = Real{0}) -> tf::cpp::nd_array<Real> {
  if constexpr (Dims == 2)
    return make_array<Real>({x, y}, {2});
  else
    return make_array<Real>({x, y, z}, {3});
}

template <typename Real, std::size_t Dims>
auto points_values(std::initializer_list<Real> xy,
                   std::initializer_list<Real> xyz) -> tf::cpp::nd_array<Real> {
  if constexpr (Dims == 2)
    return make_array<Real>(xy, {static_cast<int>(xy.size() / 2), 2});
  else
    return make_array<Real>(xyz, {static_cast<int>(xyz.size() / 3), 3});
}

template <typename Real, std::size_t Dims>
auto coordinates(std::initializer_list<Real> xy,
                 std::initializer_list<Real> xyz) -> std::vector<Real> {
  if constexpr (Dims == 2)
    return std::vector<Real>(xy);
  else
    return std::vector<Real>(xyz);
}

template <typename Index, typename Real, std::size_t Dims>
auto fixed_mesh(Real x = Real{0}, Real z = Real{0})
    -> tf::cpp::test::owned_mesh<Index, Real, Dims> {
  return {tf::cpp::test::polygons_of<Index, Real, Dims>(
      std::initializer_list<Index>{0, 1, 4, 0, 4, 3, 1, 2, 5, 1, 5, 4},
      coordinates<Real, Dims>(
          {x, 0, x + 1, 0, x + 2, 0, x, 1, x + 1, 1, x + 2, 1},
          {x, 0, z, x + 1, 0, z, x + 2, 0, z, x, 1, z, x + 1, 1, z, x + 2, 1,
           z}))};
}

template <typename Index, typename Real, std::size_t Dims>
auto dynamic_mesh(Real x = Real{0}, Real z = Real{0})
    -> tf::cpp::test::owned_mesh<Index, Real, Dims, tf::dynamic_size> {
  return {tf::cpp::test::polygons_of<Index, Real, Dims>(
      std::initializer_list<Index>{0, 4, 7, 10},
      std::initializer_list<Index>{0, 1, 4, 3, 1, 2, 5, 1, 5, 4},
      coordinates<Real, Dims>(
          {x, 0, x + 1, 0, x + 2, 0, x, 1, x + 1, 1, x + 2, 1},
          {x, 0, z, x + 1, 0, z, x + 2, 0, z, x, 1, z, x + 1, 1, z, x + 2, 1,
           z}))};
}

template <typename Index, typename Real, std::size_t Dims>
auto edges(Real x = Real{0}, Real z = Real{0})
    -> tf::cpp::test::owned_edge_mesh<Index, Real, Dims> {
  return {tf::cpp::test::segments_of<Index, Real, Dims>(
      std::initializer_list<Index>{0, 1, 1, 2, 2, 3},
      coordinates<Real, Dims>(
          {x, 0, x + 1, 0, x + 2, 0, x + 3, 0},
          {x, 0, z, x + 1, 0, z, x + 2, 0, z, x + 3, 0, z}))};
}

template <typename Real, std::size_t Dims>
auto cloud(Real x = Real{0}) -> tf::cpp::test::owned_point_cloud<Real, Dims> {
  return {tf::cpp::test::points_of<Real, Dims>(coordinates<Real, Dims>(
      {x, 0, x + 1, 0, x + 2, 0, x, 1, x + 1, 1, x + 2, 1, x, 2, x + 1, 2,
       x + 2, 2},
      {x, 0,     0, x + 1, 0, 0, x + 2, 0,     0, x, 1,     0, x + 1, 1,
       0, x + 2, 1, 0,     x, 2, 0,     x + 1, 2, 0, x + 2, 2, 0}))};
}

/// The placement a carrier is assembled at, in the shape `place` states it.
template <typename Real, std::size_t Dims>
auto translation(Real x) -> std::array<Real, (Dims + 1) * (Dims + 1)> {
  if constexpr (Dims == 2)
    return {1, 0, x, 0, 1, 0, 0, 0, 1};
  else
    return {1, 0, 0, x, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
}

template <typename Real, std::size_t Dims>
auto point(Real x, Real y = Real{0}, Real z = Real{0})
    -> tf::cpp::primitive<Real, Dims> {
  return tf::cpp::primitive<Real, Dims>(tf::cpp::primitive_kind::point,
                                        point_values<Real, Dims>(x, y, z));
}

template <typename Real, std::size_t Dims>
auto point_batch() -> tf::cpp::primitive<Real, Dims> {
  return tf::cpp::primitive<Real, Dims>(
      tf::cpp::primitive_kind::point,
      points_values<Real, Dims>({0, 0, 10, 10}, {0, 0, 0, 10, 10, 10}));
}

/// The carrier an owner is read through, whichever of the three it holds.
template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto carrier_of(
    const tf::cpp::test::owned_mesh<Index, Real, Dims, Ngon> &value) {
  return value.mesh();
}

template <typename Index, typename Real, std::size_t Dims>
auto carrier_of(
    const tf::cpp::test::owned_edge_mesh<Index, Real, Dims> &value) {
  return value.edge_mesh();
}

template <typename Real, std::size_t Dims>
auto carrier_of(const tf::cpp::test::owned_point_cloud<Real, Dims> &value) {
  return value.point_cloud();
}

/// The carriers a tuple of owners is read through, in its own order.
template <typename Owners> auto carriers_of(const Owners &owners) {
  return std::apply(
      [](const auto &...owner) {
        return std::make_tuple(carrier_of(owner)...);
      },
      owners);
}

template <typename Form, typename Real, std::size_t Dims>
auto check_point_batch(const Form &form) -> void {
  auto query = point_batch<Real, Dims>();
  const auto result = tf::cpp::intersects(form, query);
  const auto reversed = tf::cpp::intersects(query, form);
  REQUIRE(result.is_batch());
  REQUIRE(reversed.is_batch());
  REQUIRE(result.batch().length() == 2);
  REQUIRE(reversed.batch().length() == 2);
  CHECK(result.batch()[0] == 1);
  CHECK(result.batch()[1] == 0);
  CHECK(reversed.batch()[0] == 1);
  CHECK(reversed.batch()[1] == 0);
}

template <typename Real, std::size_t Dims> auto origin_primitives() {
  using primitive = tf::cpp::primitive<Real, Dims>;
  auto point_value = primitive(tf::cpp::primitive_kind::point,
                               point_values<Real, Dims>(0, 0, 0));
  auto segment_value =
      primitive(tf::cpp::primitive_kind::segment,
                points_values<Real, Dims>({-1, 0, 1, 0}, {-1, 0, 0, 1, 0, 0}));
  auto triangle_value =
      primitive(tf::cpp::primitive_kind::triangle,
                points_values<Real, Dims>({-1, -1, 1, -1, 0, 1},
                                          {-1, -1, 0, 1, -1, 0, 0, 1, 0}));
  auto ray_value =
      primitive(tf::cpp::primitive_kind::ray,
                points_values<Real, Dims>({0, 0, 1, 0}, {0, 0, 1, 0, 0, -1}));
  auto line_value =
      primitive(tf::cpp::primitive_kind::line,
                points_values<Real, Dims>({0, 0, 1, 0}, {0, 0, 0, 1, 0, 0}));
  auto aabb_value = primitive(
      tf::cpp::primitive_kind::aabb,
      points_values<Real, Dims>({-1, -1, 1, 1}, {-1, -1, -1, 1, 1, 1}));
  auto polygon_value = primitive(
      tf::cpp::primitive_kind::polygon,
      points_values<Real, Dims>({-1, -1, 1, -1, 1, 1, -1, 1},
                                {-1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 1, 0}));

  if constexpr (Dims == 2) {
    return std::make_tuple(std::move(point_value), std::move(segment_value),
                           std::move(triangle_value), std::move(ray_value),
                           std::move(line_value), std::move(aabb_value),
                           std::move(polygon_value));
  } else {
    auto plane_value = primitive(tf::cpp::primitive_kind::plane,
                                 make_array<Real>({0, 0, 1, 0}, {4}));
    return std::make_tuple(std::move(point_value), std::move(segment_value),
                           std::move(triangle_value), std::move(ray_value),
                           std::move(line_value), std::move(plane_value),
                           std::move(aabb_value), std::move(polygon_value));
  }
}

template <typename Form, typename Real, std::size_t Dims>
auto check_all_primitives(const Form &form) -> void {
  auto values = origin_primitives<Real, Dims>();
  std::apply(
      [&](auto &...query) {
        const auto check = [&](const auto &value) {
          CHECK(tf::cpp::intersects(form, value).scalar());
          CHECK(tf::cpp::intersects(value, form).scalar());
        };
        (check(query), ...);
      },
      values);
}

template <typename A, typename Tuple>
auto check_pairs(const A &a, const Tuple &forms) -> void {
  std::apply(
      [&](auto &...b) {
        const auto check = [&](const auto &value) {
          CHECK(tf::cpp::intersects(a, value).scalar());
        };
        (check(b), ...);
      },
      forms);
}

template <typename A, typename Tuple>
auto check_mixed_pairs(const A &a, const Tuple &forms) -> void {
  std::apply(
      [&](auto &...b) {
        const auto check = [&](const auto &value) {
          CHECK_THROWS_AS(tf::cpp::intersects(a, value), std::invalid_argument);
          CHECK_THROWS_AS(tf::cpp::intersects(value, a), std::invalid_argument);
        };
        (check(b), ...);
      },
      forms);
}

template <typename A, typename B, typename = void>
struct has_form_intersects : std::false_type {};

template <typename A, typename B>
struct has_form_intersects<
    A, B,
    std::void_t<decltype(tf::cpp::intersects(std::declval<const A &>(),
                                             std::declval<const B &>()))>>
    : std::true_type {};

struct mutating_resolver {
  std::shared_ptr<std::atomic<int>> submissions;
  std::function<void()> mutation;

  template <typename T>
  using state_type = tf::cpp::async::detail::future_state<T>;

  template <typename T>
  auto make_state() const -> std::shared_ptr<state_type<T>> {
    submissions->fetch_add(1, std::memory_order_relaxed);
    mutation();
    return std::make_shared<state_type<T>>();
  }
};

} // namespace intersects_matrix

TEMPLATE_LIST_TEST_CASE(
    "form intersects covers every Python carrier scalar index dimension and "
    "layout",
    "[cpp][spatial][intersects][form][matrix][python-parity][archive-link]",
    intersects_matrix::form_rows) {
  using Real = typename TestType::real_type;
  using Index = typename TestType::index_type;
  constexpr auto Dims = TestType::dims;
  using OtherReal =
      std::conditional_t<std::is_same_v<Real, float>, double, float>;

  auto fixed_storage = intersects_matrix::fixed_mesh<Index, Real, Dims>();
  auto dynamic_storage = intersects_matrix::dynamic_mesh<Index, Real, Dims>();
  auto edge_storage = intersects_matrix::edges<Index, Real, Dims>();
  auto cloud_storage = intersects_matrix::cloud<Real, Dims>();

  const auto fixed = fixed_storage.mesh();
  const auto dynamic = dynamic_storage.mesh();
  const auto edge = edge_storage.edge_mesh();
  const auto points = cloud_storage.point_cloud();

  intersects_matrix::check_all_primitives<decltype(fixed), Real, Dims>(fixed);
  intersects_matrix::check_all_primitives<decltype(dynamic), Real, Dims>(
      dynamic);
  intersects_matrix::check_all_primitives<decltype(edge), Real, Dims>(edge);
  intersects_matrix::check_all_primitives<decltype(points), Real, Dims>(points);
  intersects_matrix::check_point_batch<decltype(fixed), OtherReal, Dims>(fixed);
  intersects_matrix::check_point_batch<decltype(dynamic), OtherReal, Dims>(
      dynamic);
  intersects_matrix::check_point_batch<decltype(edge), OtherReal, Dims>(edge);
  intersects_matrix::check_point_batch<decltype(points), OtherReal, Dims>(
      points);

  auto mixed_query = intersects_matrix::point<OtherReal, Dims>(0);
  CHECK(tf::cpp::intersects(fixed, mixed_query).scalar());
  CHECK(tf::cpp::intersects(mixed_query, dynamic).scalar());
  CHECK(tf::cpp::intersects(edge, mixed_query).scalar());
  CHECK(tf::cpp::intersects(mixed_query, points).scalar());

  auto miss = intersects_matrix::point<Real, Dims>(10, 10, 10);
  CHECK_FALSE(tf::cpp::intersects(fixed, miss).scalar());
  CHECK_FALSE(tf::cpp::intersects(miss, dynamic).scalar());
  CHECK_FALSE(tf::cpp::intersects(edge, miss).scalar());
  CHECK_FALSE(tf::cpp::intersects(miss, points).scalar());

  // the placement is this instance's, so a moved carrier is a new assembly
  const auto frame = intersects_matrix::translation<Real, Dims>(Real{5});
  fixed_storage.place(frame);
  dynamic_storage.place(frame);
  edge_storage.place(frame);
  cloud_storage.place(frame);
  auto transformed_point = intersects_matrix::point<OtherReal, Dims>(Real{5});
  CHECK(tf::cpp::intersects(fixed_storage.mesh(), transformed_point).scalar());
  CHECK(
      tf::cpp::intersects(transformed_point, dynamic_storage.mesh()).scalar());
  CHECK(tf::cpp::intersects(edge_storage.edge_mesh(), transformed_point)
            .scalar());
  CHECK(tf::cpp::intersects(transformed_point, cloud_storage.point_cloud())
            .scalar());
}

TEMPLATE_TEST_CASE(
    "form pairs cover fixed dynamic edge point identity transforms self touch "
    "and boundary",
    "[cpp][spatial][intersects][form][pair][matrix][python-parity]", float,
    double) {
  using Real = TestType;
  for (const auto dims : {2, 3}) {
    if (dims == 2) {
      const auto storage = std::make_tuple(
          intersects_matrix::fixed_mesh<std::int32_t, Real, 2>(),
          intersects_matrix::dynamic_mesh<std::int32_t, Real, 2>(),
          intersects_matrix::fixed_mesh<std::int64_t, Real, 2>(),
          intersects_matrix::dynamic_mesh<std::int64_t, Real, 2>(),
          intersects_matrix::edges<std::int32_t, Real, 2>(),
          intersects_matrix::edges<std::int64_t, Real, 2>(),
          intersects_matrix::cloud<Real, 2>());
      const auto forms = intersects_matrix::carriers_of(storage);
      std::apply(
          [&](const auto &...form) {
            (intersects_matrix::check_pairs(form, forms), ...);
          },
          forms);

      auto edge0 = intersects_matrix::edges<std::int64_t, Real, 2>();
      auto edge_overlap =
          intersects_matrix::edges<std::int32_t, Real, 2>(Real{1.5});
      auto edge_far = intersects_matrix::edges<std::int32_t, Real, 2>(Real{5});
      auto edge_touch =
          intersects_matrix::edges<std::int32_t, Real, 2>(Real{3});
      CHECK(tf::cpp::intersects(edge0.edge_mesh(), edge_overlap.edge_mesh())
                .scalar());
      CHECK_FALSE(tf::cpp::intersects(edge0.edge_mesh(), edge_far.edge_mesh())
                      .scalar());
      CHECK(tf::cpp::intersects(edge0.edge_mesh(), edge_touch.edge_mesh())
                .scalar());

      auto mesh = intersects_matrix::fixed_mesh<std::int64_t, Real, 2>();
      auto dynamic_overlap =
          intersects_matrix::dynamic_mesh<std::int32_t, Real, 2>(Real{1});
      auto dynamic_far =
          intersects_matrix::dynamic_mesh<std::int32_t, Real, 2>(Real{5});
      auto mesh_touch =
          intersects_matrix::fixed_mesh<std::int32_t, Real, 2>(Real{2});
      auto edge = intersects_matrix::edges<std::int32_t, Real, 2>(Real{0.5});
      CHECK(tf::cpp::intersects(mesh.mesh(), dynamic_overlap.mesh()).scalar());
      CHECK_FALSE(
          tf::cpp::intersects(mesh.mesh(), dynamic_far.mesh()).scalar());
      CHECK(tf::cpp::intersects(mesh.mesh(), mesh_touch.mesh()).scalar());
      CHECK(tf::cpp::intersects(mesh.mesh(), edge.edge_mesh()).scalar());
      edge.place(intersects_matrix::translation<Real, 2>(100));
      CHECK_FALSE(tf::cpp::intersects(mesh.mesh(), edge.edge_mesh()).scalar());
      edge.place(intersects_matrix::translation<Real, 2>(0));
      CHECK(tf::cpp::intersects(mesh.mesh(), edge.edge_mesh()).scalar());
      mesh.place(intersects_matrix::translation<Real, 2>(5));
      edge.place(intersects_matrix::translation<Real, 2>(5));
      CHECK(tf::cpp::intersects(mesh.mesh(), edge.edge_mesh()).scalar());
    } else {
      const auto storage = std::make_tuple(
          intersects_matrix::fixed_mesh<std::int32_t, Real, 3>(),
          intersects_matrix::dynamic_mesh<std::int32_t, Real, 3>(),
          intersects_matrix::fixed_mesh<std::int64_t, Real, 3>(),
          intersects_matrix::dynamic_mesh<std::int64_t, Real, 3>(),
          intersects_matrix::edges<std::int32_t, Real, 3>(),
          intersects_matrix::edges<std::int64_t, Real, 3>(),
          intersects_matrix::cloud<Real, 3>());
      const auto forms = intersects_matrix::carriers_of(storage);
      std::apply(
          [&](const auto &...form) {
            (intersects_matrix::check_pairs(form, forms), ...);
          },
          forms);

      auto mesh0 = intersects_matrix::dynamic_mesh<std::int64_t, Real, 3>();
      auto mesh_far = intersects_matrix::fixed_mesh<std::int32_t, Real, 3>(
          Real{0}, Real{5});
      auto edge0 = intersects_matrix::edges<std::int64_t, Real, 3>();
      auto edge_far =
          intersects_matrix::edges<std::int32_t, Real, 3>(Real{0}, Real{5});
      CHECK_FALSE(tf::cpp::intersects(mesh0.mesh(), mesh_far.mesh()).scalar());
      CHECK(tf::cpp::intersects(mesh0.mesh(), edge0.edge_mesh()).scalar());
      CHECK_FALSE(
          tf::cpp::intersects(mesh0.mesh(), edge_far.edge_mesh()).scalar());
      mesh0.place(intersects_matrix::translation<Real, 3>(10));
      edge0.place(intersects_matrix::translation<Real, 3>(10));
      CHECK(tf::cpp::intersects(mesh0.mesh(), edge0.edge_mesh()).scalar());
    }
  }
}

TEST_CASE(
    "form intersects precision and dimension mismatch contracts are explicit",
    "[cpp][spatial][intersects][form][matrix][validation]") {
  using mesh2 = tf::cpp::mesh<std::int64_t, float, 2>;
  using mesh3 = tf::cpp::mesh<std::int64_t, float, 3>;
  using edge2 = tf::cpp::edge_mesh<std::int32_t, float, 2>;
  using edge3 = tf::cpp::edge_mesh<std::int32_t, float, 3>;
  using cloud2 = tf::cpp::point_cloud<float, 2>;
  using point3 = tf::cpp::primitive<float, 3>;

  STATIC_REQUIRE_FALSE(
      intersects_matrix::has_form_intersects<mesh2, mesh3>::value);
  STATIC_REQUIRE_FALSE(
      intersects_matrix::has_form_intersects<edge2, edge3>::value);
  STATIC_REQUIRE_FALSE(
      intersects_matrix::has_form_intersects<cloud2, point3>::value);
  STATIC_REQUIRE_FALSE(
      intersects_matrix::has_form_intersects<point3, edge2>::value);

  const auto float2_storage =
      std::make_tuple(intersects_matrix::fixed_mesh<std::int32_t, float, 2>(),
                      intersects_matrix::dynamic_mesh<std::int64_t, float, 2>(),
                      intersects_matrix::edges<std::int32_t, float, 2>(),
                      intersects_matrix::edges<std::int64_t, float, 2>(),
                      intersects_matrix::cloud<float, 2>());
  const auto double2_storage = std::make_tuple(
      intersects_matrix::fixed_mesh<std::int32_t, double, 2>(),
      intersects_matrix::dynamic_mesh<std::int64_t, double, 2>(),
      intersects_matrix::edges<std::int32_t, double, 2>(),
      intersects_matrix::edges<std::int64_t, double, 2>(),
      intersects_matrix::cloud<double, 2>());
  const auto float2 = intersects_matrix::carriers_of(float2_storage);
  const auto double2 = intersects_matrix::carriers_of(double2_storage);
  std::apply(
      [&](const auto &...form) {
        (intersects_matrix::check_mixed_pairs(form, double2), ...);
      },
      float2);

  const auto float3_storage =
      std::make_tuple(intersects_matrix::fixed_mesh<std::int32_t, float, 3>(),
                      intersects_matrix::dynamic_mesh<std::int64_t, float, 3>(),
                      intersects_matrix::edges<std::int32_t, float, 3>(),
                      intersects_matrix::edges<std::int64_t, float, 3>(),
                      intersects_matrix::cloud<float, 3>());
  const auto double3_storage = std::make_tuple(
      intersects_matrix::fixed_mesh<std::int32_t, double, 3>(),
      intersects_matrix::dynamic_mesh<std::int64_t, double, 3>(),
      intersects_matrix::edges<std::int32_t, double, 3>(),
      intersects_matrix::edges<std::int64_t, double, 3>(),
      intersects_matrix::cloud<double, 3>());
  const auto float3 = intersects_matrix::carriers_of(float3_storage);
  const auto double3 = intersects_matrix::carriers_of(double3_storage);
  std::apply(
      [&](const auto &...form) {
        (intersects_matrix::check_mixed_pairs(form, double3), ...);
      },
      float3);
}

TEST_CASE("async form intersects copies the query and answers from the reading "
          "it was handed",
          "[cpp][spatial][intersects][form][matrix][async]") {
  const auto submissions = std::make_shared<std::atomic<int>>(0);

  // the query is carried as the handle it is, so the caller may release its
  // own; the carrier is borrowed and the storage it names is kept alive by the
  // keepalive it was assembled with
  auto query = intersects_matrix::point<float, 3>(0);
  auto pending = tf::cpp::async::intersects(
      intersects_matrix::mutating_resolver{
          submissions,
          [&] { query = intersects_matrix::point<float, 3>(100); }},
      held_carrier_of(
          intersects_matrix::dynamic_mesh<std::int64_t, double, 3>()),
      query);
  CHECK(submissions->load(std::memory_order_relaxed) == 1);
  CHECK(pending.get().scalar());

  auto pair_query = intersects_matrix::point<float, 2>(0);
  auto pair_pending = tf::cpp::async::intersects(
      intersects_matrix::mutating_resolver{
          submissions,
          [&] { pair_query = intersects_matrix::point<float, 2>(100); }},
      held_carrier_of(intersects_matrix::edges<std::int64_t, float, 2>()),
      pair_query);
  CHECK(submissions->load(std::memory_order_relaxed) == 2);
  CHECK(pair_pending.get().scalar());

  auto reversed_query = intersects_matrix::point<float, 3>(0);
  auto reversed = tf::cpp::async::intersects(
      intersects_matrix::mutating_resolver{
          submissions,
          [&] { reversed_query = intersects_matrix::point<float, 3>(100); }},
      reversed_query,
      held_carrier_of(intersects_matrix::edges<std::int32_t, double, 3>()));
  CHECK(submissions->load(std::memory_order_relaxed) == 3);
  CHECK(reversed.get().scalar());
}
