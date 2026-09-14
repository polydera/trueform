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
#include "trueform/cpp/core/mesh.hpp"
#include "trueform/cpp/spatial/async/distance.hpp"
#include "trueform/cpp/spatial/async/neighbor_search.hpp"
#include "trueform/cpp/spatial/async/ray_cast.hpp"
#include "trueform/cpp/spatial/distance.hpp"
#include "trueform/cpp/spatial/neighbor_search.hpp"
#include "trueform/cpp/spatial/ray_cast.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <future>
#include <memory>
#include <stdexcept>
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

template <typename T>
auto make_array(const std::vector<T> &values, tf::small_vector<int, 3> shape)
    -> tf::cpp::nd_array<T> {
  tf::buffer<T> buffer;
  buffer.allocate(values.size());
  std::copy(values.begin(), values.end(), buffer.begin());
  return tf::cpp::nd_array<T>::from_buffer(std::move(buffer), std::move(shape));
}

template <typename Real, std::size_t Dims>
auto primitive(tf::cpp::primitive_kind kind, std::initializer_list<Real> values,
               tf::small_vector<int, 3> shape)
    -> tf::cpp::primitive<Real, Dims> {
  return tf::cpp::primitive<Real, Dims>(
      kind, make_array<Real>(values, std::move(shape)));
}

template <typename Real, std::size_t Dims>
auto point_d(std::initializer_list<Real> values)
    -> tf::cpp::primitive<Real, Dims> {
  return primitive<Real, Dims>(tf::cpp::primitive_kind::point, values,
                               {static_cast<int>(Dims)});
}

template <typename Real, std::size_t Dims>
auto point_batch_d(std::initializer_list<Real> values, int count)
    -> tf::cpp::primitive<Real, Dims> {
  return primitive<Real, Dims>(tf::cpp::primitive_kind::point, values,
                               {count, static_cast<int>(Dims)});
}

template <typename Real, std::size_t Dims>
auto point_batch_d(const std::vector<Real> &values, int count)
    -> tf::cpp::primitive<Real, Dims> {
  return tf::cpp::primitive<Real, Dims>(
      tf::cpp::primitive_kind::point,
      make_array<Real>(values, {count, static_cast<int>(Dims)}));
}

template <typename Real>
auto point(Real x, Real y, Real z) -> tf::cpp::primitive<Real> {
  return tf::cpp::primitive<Real>(tf::cpp::primitive_kind::point,
                                  make_array<Real>({x, y, z}, {3}));
}

template <typename Real>
auto point_batch(std::initializer_list<Real> values, int count)
    -> tf::cpp::primitive<Real> {
  return tf::cpp::primitive<Real>(tf::cpp::primitive_kind::point,
                                  make_array<Real>(values, {count, 3}));
}

template <typename Real>
auto origin_primitive(tf::cpp::primitive_kind kind)
    -> tf::cpp::primitive<Real> {
  using tf::cpp::primitive;
  switch (kind) {
  case tf::cpp::primitive_kind::point:
    return point<Real>(0, 0, 0);
  case tf::cpp::primitive_kind::vector:
    return primitive<Real>(kind, make_array<Real>({1, 0, 0}, {3}));
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
  }
  throw std::logic_error("unknown primitive kind");
}

template <typename Real>
auto triangle_mesh(Real z = Real{})
    -> tf::cpp::test::owned_mesh<tf::cpp::default_index_t, Real> {
  return {tf::cpp::test::polygons_of<tf::cpp::default_index_t, Real>(
      {0, 1, 2}, {-1, -1, z, 1, -1, z, 0, 1, z})};
}

/// What a caller assembles when no handle of its own outlives the call: the
/// storage lives in a shared block and the mesh retains it.
template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto held_carrier_of(tf::cpp::test::owned_mesh<Index, Real, Dims, Ngon> owner)
    -> tf::cpp::mesh<Index, Real, Dims, Ngon> {
  const auto held = std::make_shared<
      const tf::cpp::test::owned_mesh<Index, Real, Dims, Ngon>>(
      std::move(owner));
  return {held->polygons.faces(), held->polygons.points(), held->cache,
          held->frame(), held};
}

template <typename Real>
auto origin_cloud(Real z = Real{}) -> tf::cpp::test::owned_point_cloud<Real> {
  return {tf::cpp::test::points_of<Real>({0, 0, z})};
}

template <typename Real>
auto empty_mesh() -> tf::cpp::test::owned_mesh<tf::cpp::default_index_t, Real> {
  return {};
}

template <typename Real>
auto empty_cloud() -> tf::cpp::test::owned_point_cloud<Real> {
  return {};
}

/// The placement a carrier is assembled at, in the shape `place` states it.
template <typename Real>
auto translation(Real x, Real y, Real z) -> std::array<Real, 16> {
  return {1, 0, 0, x, 0, 1, 0, y, 0, 0, 1, z, 0, 0, 0, 1};
}

template <typename Real, std::size_t Dims>
auto translation_d(std::size_t axis, Real offset)
    -> std::array<Real, (Dims + 1) * (Dims + 1)> {
  constexpr auto side = Dims + 1;
  std::array<Real, side * side> values{};
  for (std::size_t diagonal = 0; diagonal < side; ++diagonal)
    values[diagonal * side + diagonal] = Real{1};
  values[axis * side + Dims] = offset;
  return values;
}

template <typename Index, typename Real, std::size_t Dims>
auto triangle_mesh_d(Real offset = Real{})
    -> tf::cpp::test::owned_mesh<Index, Real, Dims> {
  auto points = [&]() -> std::vector<Real> {
    if constexpr (Dims == 2)
      return {-1, Real(-1 + offset), 1, Real(-1 + offset), 0, Real(1 + offset)};
    else
      return {-1, -1, offset, 1, -1, offset, 0, 1, offset};
  }();
  return {tf::cpp::test::polygons_of<Index, Real, Dims>(
      std::initializer_list<Index>{0, 1, 2}, points)};
}

template <typename Real, std::size_t Dims>
auto origin_cloud_d(Real offset = Real{})
    -> tf::cpp::test::owned_point_cloud<Real, Dims> {
  if constexpr (Dims == 2)
    return {tf::cpp::test::points_of<Real, Dims>({0, offset})};
  else
    return {tf::cpp::test::points_of<Real, Dims>({0, 0, offset})};
}

template <typename Index, typename Real, std::size_t Dims>
auto dynamic_triangle_quad_mesh()
    -> tf::cpp::test::owned_mesh<Index, Real, Dims, tf::dynamic_size> {
  auto points = [&]() -> std::vector<Real> {
    if constexpr (Dims == 2)
      return {0, 0, 1, 0, 0, 1, 2, 0, 3, 0, 3, 1, 2, 1};
    else
      return {0, 0, 0, 1, 0, 0, 0, 1, 0, 2, 0, 0, 3, 0, 0, 3, 1, 0, 2, 1, 0};
  }();
  return {tf::cpp::test::polygons_of<Index, Real, Dims>(
      std::initializer_list<Index>{0, 3, 7},
      std::initializer_list<Index>{0, 1, 2, 3, 4, 5, 6}, points)};
}

template <typename Real> constexpr auto tolerance() -> double {
  return std::is_same<Real, float>::value ? 1e-5 : 1e-11;
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

template <typename Real>
auto owned_batch_result() -> tf::cpp::distance_result<Real> {
  auto a = point<Real>(0, 0, 0);
  auto b = point_batch<Real>({0, 0, 3, 0, 4, 0}, 2);
  return tf::cpp::distance(a, b);
}

template <typename Form0, typename Form1>
auto check_empty_form_pair(const Form0 &a, const Form1 &b) -> void {
  CHECK_THROWS_AS(tf::cpp::distance(a, b), std::invalid_argument);
  CHECK_THROWS_AS(tf::cpp::distance2(a, b), std::invalid_argument);
}

template <typename Left, typename Right, typename = void>
struct is_distance_invocable : std::false_type {};

template <typename Left, typename Right>
struct is_distance_invocable<
    Left, Right,
    std::void_t<decltype(tf::cpp::distance(std::declval<const Left &>(),
                                           std::declval<const Right &>()))>>
    : std::true_type {};

template <typename Left, typename Right, typename = void>
struct is_distance2_invocable : std::false_type {};

template <typename Left, typename Right>
struct is_distance2_invocable<
    Left, Right,
    std::void_t<decltype(tf::cpp::distance2(std::declval<const Left &>(),
                                            std::declval<const Right &>()))>>
    : std::true_type {};

template <typename Left, typename Right>
auto check_primitive_pair(const Left &left, const Right &right, double expected,
                          double expected2, double margin,
                          bool check_swap = false) -> void {
  CHECK(tf::cpp::distance(left, right).scalar() ==
        Catch::Approx(expected).margin(margin));
  CHECK(tf::cpp::distance2(left, right).scalar() ==
        Catch::Approx(expected2).margin(margin));
  if (check_swap) {
    CHECK(tf::cpp::distance(right, left).scalar() ==
          Catch::Approx(expected).margin(margin));
    CHECK(tf::cpp::distance2(right, left).scalar() ==
          Catch::Approx(expected2).margin(margin));
  }
}

template <typename Real>
auto check_batch(const tf::cpp::distance_result<Real> &result,
                 std::initializer_list<double> expected, double margin = 1e-5)
    -> void {
  REQUIRE(result.is_batch());
  const auto values = result.batch();
  REQUIRE(values.length() == expected.size());
  std::size_t index = 0;
  for (const auto expected_value : expected)
    CHECK(values[index++] == Catch::Approx(expected_value).margin(margin));
}

template <typename Index, typename Real, std::size_t Dims>
auto check_distance_operation_matrix() -> void {
  auto primitive0 = [&] {
    if constexpr (Dims == 2)
      return point_d<Real, Dims>({0, 0});
    else
      return point_d<Real, Dims>({0, 0, 0});
  }();
  auto primitive4 = [&] {
    if constexpr (Dims == 2)
      return point_d<Real, Dims>({0, 4});
    else
      return point_d<Real, Dims>({0, 0, 4});
  }();
  const auto mesh0_storage = triangle_mesh_d<Index, Real, Dims>();
  const auto mesh4_storage = triangle_mesh_d<Index, Real, Dims>(4);
  const auto cloud0_storage = origin_cloud_d<Real, Dims>();
  const auto cloud4_storage = origin_cloud_d<Real, Dims>(4);
  const auto mesh0 = mesh0_storage.mesh();
  const auto mesh4 = mesh4_storage.mesh();
  const auto cloud0 = cloud0_storage.point_cloud();
  const auto cloud4 = cloud4_storage.point_cloud();
  constexpr auto mesh_distance = Dims == 2 ? 3.0 : 4.0;
  constexpr auto mesh_mesh_distance = Dims == 2 ? 2.0 : 4.0;
  constexpr auto margin = std::is_same_v<Real, float> ? 1e-5 : 1e-11;

  CHECK(tf::cpp::distance(primitive0, primitive4).scalar() ==
        Catch::Approx(4).margin(margin));
  CHECK(tf::cpp::distance2(primitive0, primitive4).scalar() ==
        Catch::Approx(16).margin(margin));
  CHECK(tf::cpp::async::distance(primitive0, primitive4).get().scalar() ==
        Catch::Approx(4).margin(margin));
  CHECK(tf::cpp::async::distance2(primitive0, primitive4).get().scalar() ==
        Catch::Approx(16).margin(margin));

  CHECK(tf::cpp::distance(mesh0, primitive4).scalar() ==
        Catch::Approx(mesh_distance).margin(margin));
  CHECK(tf::cpp::distance2(mesh0, primitive4).scalar() ==
        Catch::Approx(mesh_distance * mesh_distance).margin(margin));
  CHECK(tf::cpp::async::distance(mesh0, primitive4).get().scalar() ==
        Catch::Approx(mesh_distance).margin(margin));
  CHECK(tf::cpp::async::distance2(mesh0, primitive4).get().scalar() ==
        Catch::Approx(mesh_distance * mesh_distance).margin(margin));

  CHECK(tf::cpp::distance(cloud0, primitive4).scalar() ==
        Catch::Approx(4).margin(margin));
  CHECK(tf::cpp::distance2(cloud0, primitive4).scalar() ==
        Catch::Approx(16).margin(margin));
  CHECK(tf::cpp::async::distance(cloud0, primitive4).get().scalar() ==
        Catch::Approx(4).margin(margin));
  CHECK(tf::cpp::async::distance2(cloud0, primitive4).get().scalar() ==
        Catch::Approx(16).margin(margin));

  CHECK(tf::cpp::distance(mesh0, mesh4) ==
        Catch::Approx(mesh_mesh_distance).margin(margin));
  CHECK(tf::cpp::distance2(mesh0, mesh4) ==
        Catch::Approx(mesh_mesh_distance * mesh_mesh_distance).margin(margin));
  CHECK(tf::cpp::async::distance(mesh0, mesh4).get() ==
        Catch::Approx(mesh_mesh_distance).margin(margin));
  CHECK(tf::cpp::async::distance2(mesh0, mesh4).get() ==
        Catch::Approx(mesh_mesh_distance * mesh_mesh_distance).margin(margin));

  CHECK(tf::cpp::distance(mesh0, cloud4) ==
        Catch::Approx(mesh_distance).margin(margin));
  CHECK(tf::cpp::distance2(mesh0, cloud4) ==
        Catch::Approx(mesh_distance * mesh_distance).margin(margin));
  CHECK(tf::cpp::async::distance(mesh0, cloud4).get() ==
        Catch::Approx(mesh_distance).margin(margin));
  CHECK(tf::cpp::async::distance2(mesh0, cloud4).get() ==
        Catch::Approx(mesh_distance * mesh_distance).margin(margin));

  CHECK(tf::cpp::distance(cloud4, mesh0) ==
        Catch::Approx(mesh_distance).margin(margin));
  CHECK(tf::cpp::distance2(cloud4, mesh0) ==
        Catch::Approx(mesh_distance * mesh_distance).margin(margin));
  CHECK(tf::cpp::async::distance(cloud4, mesh0).get() ==
        Catch::Approx(mesh_distance).margin(margin));
  CHECK(tf::cpp::async::distance2(cloud4, mesh0).get() ==
        Catch::Approx(mesh_distance * mesh_distance).margin(margin));

  CHECK(tf::cpp::distance(cloud0, cloud4) == Catch::Approx(4).margin(margin));
  CHECK(tf::cpp::distance2(cloud0, cloud4) == Catch::Approx(16).margin(margin));
  CHECK(tf::cpp::async::distance(cloud0, cloud4).get() ==
        Catch::Approx(4).margin(margin));
  CHECK(tf::cpp::async::distance2(cloud0, cloud4).get() ==
        Catch::Approx(16).margin(margin));
}

template <typename Left, typename Right>
auto check_mixed_forms_rejected(const Left &left, const Right &right) -> void {
  CHECK_THROWS_AS(tf::cpp::distance(left, right), std::invalid_argument);
  CHECK_THROWS_AS(tf::cpp::distance2(left, right), std::invalid_argument);
  CHECK_THROWS_AS(tf::cpp::async::distance(left, right).get(),
                  std::invalid_argument);
  CHECK_THROWS_AS(tf::cpp::async::distance2(left, right).get(),
                  std::invalid_argument);
}

template <std::size_t Dims> auto check_ordered_mixed_real_rejection() -> void {
  const auto mesh32_storage = triangle_mesh_d<std::int32_t, float, Dims>();
  const auto mesh64_storage = triangle_mesh_d<std::int64_t, double, Dims>(4);
  const auto cloud32_storage = origin_cloud_d<float, Dims>();
  const auto cloud64_storage = origin_cloud_d<double, Dims>(4);
  const auto mesh32 = mesh32_storage.mesh();
  const auto mesh64 = mesh64_storage.mesh();
  const auto cloud32 = cloud32_storage.point_cloud();
  const auto cloud64 = cloud64_storage.point_cloud();

  check_mixed_forms_rejected(mesh32, mesh64);
  check_mixed_forms_rejected(mesh64, mesh32);
  check_mixed_forms_rejected(mesh32, cloud64);
  check_mixed_forms_rejected(cloud64, mesh32);
  check_mixed_forms_rejected(mesh64, cloud32);
  check_mixed_forms_rejected(cloud32, mesh64);
  check_mixed_forms_rejected(cloud32, cloud64);
  check_mixed_forms_rejected(cloud64, cloud32);
}

template <typename Index, typename Real, std::size_t Dims>
auto check_dynamic_transformed_distance() -> void {
  const auto first_storage = dynamic_triangle_quad_mesh<Index, Real, Dims>();
  auto second_storage = dynamic_triangle_quad_mesh<Index, Real, Dims>();
  second_storage.place(translation_d<Real, Dims>(Dims - 1, Real{5}));
  const auto first = first_storage.mesh();
  const auto second = second_storage.mesh();
  constexpr auto expected = Dims == 2 ? 4.0 : 5.0;
  const auto margin = tolerance<Real>();
  CHECK(tf::cpp::distance(first, second) ==
        Catch::Approx(expected).margin(margin));
  CHECK(tf::cpp::distance2(first, second) ==
        Catch::Approx(expected * expected).margin(margin));
  CHECK(tf::cpp::async::distance(first, second).get() ==
        Catch::Approx(expected).margin(margin));
  CHECK(tf::cpp::async::distance2(first, second).get() ==
        Catch::Approx(expected * expected).margin(margin));
}

} // namespace

TEMPLATE_TEST_CASE(
    "primitive distances preserve scalar broadcast and pairwise modes",
    "[cpp][spatial][distance][primitive]", float, double) {
  auto origin = point<TestType>(0, 0, 0);
  auto offset = point<TestType>(0, 3, 4);
  const auto scalar2 = tf::cpp::distance2(origin, offset);
  const auto scalar = tf::cpp::distance(origin, offset);
  STATIC_REQUIRE(std::is_same<decltype(scalar.scalar()), TestType>::value);
  REQUIRE(scalar2.is_scalar());
  CHECK_FALSE(scalar2.is_batch());
  CHECK(scalar2.scalar() == Catch::Approx(25).margin(tolerance<TestType>()));
  CHECK(scalar.scalar() == Catch::Approx(5).margin(tolerance<TestType>()));
  CHECK_THROWS_AS(scalar.batch(), std::logic_error);

  auto batch = point_batch<TestType>({0, 0, 0, 0, 3, 4, 0, 0, 8}, 3);
  const auto broadcast_right = tf::cpp::distance(origin, batch);
  const auto broadcast_left = tf::cpp::distance(batch, origin);
  REQUIRE(broadcast_right.is_batch());
  REQUIRE(broadcast_left.is_batch());
  CHECK(broadcast_right.batch().shape_at(0) == 3);
  CHECK(broadcast_right.batch()[0] == Catch::Approx(0));
  CHECK(broadcast_right.batch()[1] == Catch::Approx(5));
  CHECK(broadcast_right.batch()[2] == Catch::Approx(8));
  CHECK(broadcast_left.batch()[1] == Catch::Approx(5));
  CHECK_THROWS_AS(broadcast_right.scalar(), std::logic_error);

  auto pair = point_batch<TestType>({0, 0, 1, 0, 3, 0, 0, 0, 10}, 3);
  const auto pairwise = tf::cpp::distance2(batch, pair);
  REQUIRE(pairwise.is_batch());
  CHECK(pairwise.batch()[0] == Catch::Approx(1));
  CHECK(pairwise.batch()[1] == Catch::Approx(16));
  CHECK(pairwise.batch()[2] == Catch::Approx(4));
}

TEMPLATE_TEST_CASE("Python scalar primitive distance values have native parity",
                   "[cpp][spatial][distance][python-parity]", float, double) {
  constexpr auto margin = 1e-5;

  const auto same0 = point_d<TestType, 3>({1, 2, 3});
  const auto same1 = point_d<TestType, 3>({1, 2, 3});
  const auto diagonal = point_d<TestType, 3>({2, 3, 4});
  check_primitive_pair(same0, same1, 0, 0, margin);
  check_primitive_pair(same0, diagonal, std::sqrt(3.0), 3, margin);

  const auto box2 = primitive<TestType, 2>(tf::cpp::primitive_kind::aabb,
                                           {0, 0, 1, 1}, {2, 2});
  const auto inside = point_d<TestType, 2>({0.5, 0.5});
  const auto outside = point_d<TestType, 2>({2, 2});
  check_primitive_pair(inside, box2, 0, 0, margin);
  check_primitive_pair(outside, box2, std::sqrt(2.0), 2, margin, true);

  const auto box3a = primitive<TestType, 3>(tf::cpp::primitive_kind::aabb,
                                            {0, 0, 0, 2, 2, 2}, {2, 3});
  const auto box3b = primitive<TestType, 3>(tf::cpp::primitive_kind::aabb,
                                            {1, 1, 1, 3, 3, 3}, {2, 3});
  const auto box3c = primitive<TestType, 3>(tf::cpp::primitive_kind::aabb,
                                            {5, 5, 5, 6, 6, 6}, {2, 3});
  check_primitive_pair(box3a, box3b, 0, 0, margin);
  check_primitive_pair(box3a, box3c, std::sqrt(27.0), 27, margin);

  const auto diagonal_segment = primitive<TestType, 2>(
      tf::cpp::primitive_kind::segment, {0, 0, 1, 1}, {2, 2});
  const auto point_on_segment = point_d<TestType, 2>({0.5, 0.5});
  const auto point_off_segment = point_d<TestType, 2>({0.5, 0});
  check_primitive_pair(point_on_segment, diagonal_segment, 0, 0, margin, true);
  check_primitive_pair(point_off_segment, diagonal_segment, std::sqrt(0.125),
                       0.125, 1e-4);

  const auto segment2a = primitive<TestType, 2>(
      tf::cpp::primitive_kind::segment, {0, 0, 1, 0}, {2, 2});
  const auto segment2b = primitive<TestType, 2>(
      tf::cpp::primitive_kind::segment, {0, 2, 1, 2}, {2, 2});
  const auto segment3a = primitive<TestType, 3>(
      tf::cpp::primitive_kind::segment, {0, 0, 0, 1, 0, 0}, {2, 3});
  const auto segment3b = primitive<TestType, 3>(
      tf::cpp::primitive_kind::segment, {0, 1, 1, 1, 1, 1}, {2, 3});
  check_primitive_pair(segment2a, segment2b, 2, 4, margin);
  check_primitive_pair(segment3a, segment3b, std::sqrt(2.0), 2, margin);

  const auto square = primitive<TestType, 2>(tf::cpp::primitive_kind::polygon,
                                             {0, 0, 1, 0, 1, 1, 0, 1}, {4, 2});
  check_primitive_pair(inside, square, 0, 0, margin, true);
  check_primitive_pair(outside, square, std::sqrt(2.0), 2, margin);

  const auto line = primitive<TestType, 2>(tf::cpp::primitive_kind::line,
                                           {0, 0, 1, 0}, {2, 2});
  const auto point_on_line = point_d<TestType, 2>({5, 0});
  const auto point_off_line = point_d<TestType, 2>({0, 3});
  check_primitive_pair(point_on_line, line, 0, 0, margin);
  check_primitive_pair(point_off_line, line, 3, 9, margin, true);

  const auto ray = primitive<TestType, 2>(tf::cpp::primitive_kind::ray,
                                          {0, 0, 1, 0}, {2, 2});
  const auto point_behind_ray = point_d<TestType, 2>({-2, 0});
  check_primitive_pair(point_on_line, ray, 0, 0, margin, true);
  check_primitive_pair(point_behind_ray, ray, 2, 4, margin);

  const auto plane =
      primitive<TestType, 3>(tf::cpp::primitive_kind::plane, {0, 0, 1, 0}, {4});
  const auto point_on_plane = point_d<TestType, 3>({1, 1, 0});
  const auto point_above_plane = point_d<TestType, 3>({1, 1, 5});
  check_primitive_pair(point_on_plane, plane, 0, 0, margin);
  check_primitive_pair(point_above_plane, plane, 5, 25, margin, true);

  const auto line2 = primitive<TestType, 2>(tf::cpp::primitive_kind::line,
                                            {0, 3, 1, 0}, {2, 2});
  const auto crossing_line = primitive<TestType, 2>(
      tf::cpp::primitive_kind::line, {0, 0, 0, 1}, {2, 2});
  check_primitive_pair(line, line2, 3, 9, margin);
  check_primitive_pair(line, crossing_line, 0, 0, margin);

  const auto ray2 = primitive<TestType, 2>(tf::cpp::primitive_kind::ray,
                                           {0, 2, 1, 0}, {2, 2});
  check_primitive_pair(ray, ray2, 2, 4, margin);
}

TEST_CASE("primitive dimension mismatch is compile-time non-invocable",
          "[cpp][spatial][distance][python-parity][dimension]") {
  STATIC_REQUIRE_FALSE(
      (is_distance_invocable<tf::cpp::primitive<float, 2>,
                             tf::cpp::primitive<float, 3>>::value));
  STATIC_REQUIRE_FALSE(
      (is_distance2_invocable<tf::cpp::primitive<float, 2>,
                              tf::cpp::primitive<float, 3>>::value));
  STATIC_REQUIRE_FALSE(
      (is_distance_invocable<tf::cpp::primitive<double, 3>,
                             tf::cpp::primitive<double, 2>>::value));
  STATIC_REQUIRE_FALSE(
      (is_distance2_invocable<tf::cpp::primitive<double, 3>,
                              tf::cpp::primitive<double, 2>>::value));
}

TEMPLATE_TEST_CASE("Python signed plane distance batches have native parity",
                   "[cpp][spatial][distance][python-parity][batch]", float,
                   double) {
  const auto plane =
      primitive<TestType, 3>(tf::cpp::primitive_kind::plane, {0, 0, 1, 0}, {4});
  const auto points =
      point_batch_d<TestType, 3>({0, 0, 2, 1, 1, 0, 0.5, 0.5, -1, 2, 2, 3}, 4);
  check_batch(tf::cpp::distance(points, plane), {2, 0, -1, 3});

  std::vector<TestType> values;
  values.reserve(3000);
  for (int index = 0; index < 1000; ++index) {
    const auto x = static_cast<TestType>((index % 17) / 17.0);
    const auto y = static_cast<TestType>((index % 29) / 29.0);
    const auto z = static_cast<TestType>((index % 37) / 37.0);
    values.insert(values.end(), {x, y, z});
  }
  const auto large_points = point_batch_d<TestType, 3>(values, 1000);
  const auto distances = tf::cpp::distance(large_points, plane).batch();
  REQUIRE(distances.length() == 1000);
  for (std::size_t index = 0; index < distances.length(); ++index)
    CHECK(distances[index] ==
          Catch::Approx(values[index * 3 + 2]).margin(1e-5));
}

TEMPLATE_TEST_CASE("Python segment distance batches have native parity",
                   "[cpp][spatial][distance][python-parity][batch]", float,
                   double) {
  const auto segment2 = primitive<TestType, 2>(tf::cpp::primitive_kind::segment,
                                               {0, 0, 2, 0}, {2, 2});
  const auto points2 =
      point_batch_d<TestType, 2>({1, 0, 1, 1, 3, 0, -1, 0, 1, 2}, 5);
  check_batch(tf::cpp::distance(points2, segment2), {0, 1, 1, 1, 2});

  const auto segment3 = primitive<TestType, 3>(tf::cpp::primitive_kind::segment,
                                               {0, 0, 0, 2, 0, 0}, {2, 3});
  const auto points3 = point_batch_d<TestType, 3>(
      {1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 3, 0, 0}, 5);
  check_batch(tf::cpp::distance(points3, segment3),
              {0, 1, 1, std::sqrt(2.0), 1});
}

TEMPLATE_TEST_CASE("Python polygon distance batches have native parity",
                   "[cpp][spatial][distance][python-parity][batch]", float,
                   double) {
  const auto square = primitive<TestType, 2>(tf::cpp::primitive_kind::polygon,
                                             {0, 0, 1, 0, 1, 1, 0, 1}, {4, 2});
  const auto points2 = point_batch_d<TestType, 2>(
      {0.5, 0.5, 0, 0.5, 2, 0.5, -1, 0.5, 0.5, 2}, 5);
  const auto distances2 = tf::cpp::distance(points2, square).batch();
  REQUIRE(distances2.length() == 5);
  CHECK(distances2[0] >= 0);
  CHECK(distances2[0] <= 0.5);
  CHECK(distances2[1] == Catch::Approx(0).margin(1e-5));
  CHECK(distances2[2] == Catch::Approx(1).margin(1e-5));
  CHECK(distances2[3] == Catch::Approx(1).margin(1e-5));
  CHECK(distances2[4] == Catch::Approx(1).margin(1e-5));

  const auto triangle = primitive<TestType, 3>(
      tf::cpp::primitive_kind::polygon, {0, 0, 0, 1, 0, 0, 0.5, 1, 0}, {3, 3});
  const auto points3 = point_batch_d<TestType, 3>(
      {0.5, 0.3, 0, 0.5, 0.3, 1, 0.5, 0.3, -1, 2, 0, 0}, 4);
  const auto distances3 = tf::cpp::distance(points3, triangle).batch();
  REQUIRE(distances3.length() == 4);
  CHECK(distances3[0] == Catch::Approx(0).margin(1e-5));
  CHECK(distances3[1] > 0);
  CHECK(distances3[2] > 0);
  CHECK(distances3[3] > 0);
}

TEMPLATE_TEST_CASE("Python line and AABB distance batches have native parity",
                   "[cpp][spatial][distance][python-parity][batch]", float,
                   double) {
  const auto line2 = primitive<TestType, 2>(tf::cpp::primitive_kind::line,
                                            {1, 0, 0, 1}, {2, 2});
  const auto line_points2 =
      point_batch_d<TestType, 2>({1, 0, 1, 5, 0, 0, 2, 0, 3, 0}, 5);
  check_batch(tf::cpp::distance(line_points2, line2), {0, 0, 1, 1, 2});

  const auto line3 = primitive<TestType, 3>(tf::cpp::primitive_kind::line,
                                            {0, 0, 0, 0, 0, 1}, {2, 3});
  const auto line_points3 = point_batch_d<TestType, 3>(
      {0, 0, 0, 0, 0, 5, 1, 0, 0, 0, 1, 0, 1, 1, 0}, 5);
  check_batch(tf::cpp::distance(line_points3, line3),
              {0, 0, 1, 1, std::sqrt(2.0)});

  const auto box2 = primitive<TestType, 2>(tf::cpp::primitive_kind::aabb,
                                           {0, 0, 1, 1}, {2, 2});
  const auto box_points2 =
      point_batch_d<TestType, 2>({0.5, 0.5, 0, 0.5, 2, 0.5, -1, 0.5, 2, 2}, 5);
  check_batch(tf::cpp::distance(box_points2, box2),
              {0, 0, 1, 1, std::sqrt(2.0)});

  const auto box3 = primitive<TestType, 3>(tf::cpp::primitive_kind::aabb,
                                           {0, 0, 0, 1, 1, 1}, {2, 3});
  const auto box_points3 = point_batch_d<TestType, 3>(
      {0.5, 0.5, 0.5, 0, 0.5, 0.5, 2, 0.5, 0.5, 0.5, 0.5, 2, 2, 2, 2}, 5);
  check_batch(tf::cpp::distance(box_points3, box3),
              {0, 0, 1, 1, std::sqrt(3.0)});
}

TEST_CASE("mixed primitive distances promote to double",
          "[cpp][spatial][distance][precision]") {
  auto a32 = point<float>(0, 0, 0);
  auto b64 = point<double>(0, 0, 2.5);
  const auto forward = tf::cpp::distance(a32, b64);
  const auto reverse = tf::cpp::distance2(b64, a32);
  STATIC_REQUIRE(std::is_same<decltype(forward.scalar()), double>::value);
  CHECK(forward.scalar() == Catch::Approx(2.5));
  CHECK(reverse.scalar() == Catch::Approx(6.25));

  auto batch32 = point_batch<float>({0, 0, 1, 0, 0, 3}, 2);
  const auto promoted_batch = tf::cpp::distance2(batch32, b64);
  STATIC_REQUIRE(std::is_same<decltype(promoted_batch.batch()),
                              tf::cpp::nd_array<double>>::value);
  CHECK(promoted_batch.batch()[0] == Catch::Approx(2.25));
  CHECK(promoted_batch.batch()[1] == Catch::Approx(0.25));
}

TEST_CASE("primitive dispatch covers every ordered spatial pair",
          "[cpp][spatial][distance][dispatch]") {
  for (const auto kind_a : spatial_kinds) {
    for (const auto kind_b : spatial_kinds) {
      INFO("ordered primitive pair " << static_cast<int>(kind_a) << " x "
                                     << static_cast<int>(kind_b));
      const auto a = origin_primitive<double>(kind_a);
      const auto b = origin_primitive<double>(kind_b);
      CHECK(tf::cpp::distance2(a, b).is_scalar());
      CHECK(tf::cpp::distance(a, b).is_scalar());
    }
  }
}

TEMPLATE_TEST_CASE("form primitive distances support mesh cloud and conversion",
                   "[cpp][spatial][distance][form]", float, double) {
  const auto mesh_storage = triangle_mesh<TestType>();
  const auto cloud_storage = origin_cloud<TestType>();
  const auto mesh = mesh_storage.mesh();
  const auto cloud = cloud_storage.point_cloud();
  using Other =
      std::conditional_t<std::is_same<TestType, float>::value, double, float>;
  auto query = point<Other>(0, 0, 2);

  const auto mesh_d2 = tf::cpp::distance2(mesh, query);
  const auto mesh_d = tf::cpp::distance(mesh, query);
  const auto cloud_d2 = tf::cpp::distance2(cloud, query);
  const auto cloud_d = tf::cpp::distance(cloud, query);
  STATIC_REQUIRE(std::is_same<decltype(mesh_d.scalar()), TestType>::value);
  CHECK(mesh_d2.scalar() == Catch::Approx(4).margin(tolerance<TestType>()));
  CHECK(mesh_d.scalar() == Catch::Approx(2).margin(tolerance<TestType>()));
  CHECK(cloud_d2.scalar() == Catch::Approx(4).margin(tolerance<TestType>()));
  CHECK(cloud_d.scalar() == Catch::Approx(2).margin(tolerance<TestType>()));

  auto batch = point_batch<Other>({0, 0, 2, 0, 0, 3}, 2);
  const auto mesh_batch = tf::cpp::distance2(mesh, batch);
  const auto cloud_batch = tf::cpp::distance(cloud, batch);
  REQUIRE(mesh_batch.is_batch());
  CHECK(mesh_batch.batch()[0] == Catch::Approx(4));
  CHECK(mesh_batch.batch()[1] == Catch::Approx(9));
  CHECK(cloud_batch.batch()[0] == Catch::Approx(2));
  CHECK(cloud_batch.batch()[1] == Catch::Approx(3));
}

TEST_CASE("full generalized distance operation matrix links and runs",
          "[cpp][spatial][distance][matrix][archive]") {
  check_distance_operation_matrix<std::int32_t, float, 2>();
  check_distance_operation_matrix<std::int32_t, double, 2>();
  check_distance_operation_matrix<std::int64_t, float, 2>();
  check_distance_operation_matrix<std::int64_t, double, 2>();
  check_distance_operation_matrix<std::int32_t, float, 3>();
  check_distance_operation_matrix<std::int32_t, double, 3>();
  check_distance_operation_matrix<std::int64_t, float, 3>();
  check_distance_operation_matrix<std::int64_t, double, 3>();

  const auto point32 = point_d<float, 2>({0, 0});
  const auto point64 = point_d<double, 2>({3, 4});
  CHECK(tf::cpp::distance(point32, point64).scalar() == Catch::Approx(5));
  CHECK(tf::cpp::distance2(point64, point32).scalar() == Catch::Approx(25));
  CHECK(tf::cpp::async::distance(point64, point32).get().scalar() ==
        Catch::Approx(5));
  CHECK(tf::cpp::async::distance2(point32, point64).get().scalar() ==
        Catch::Approx(25));

  check_ordered_mixed_real_rejection<2>();
  check_ordered_mixed_real_rejection<3>();
}

TEST_CASE("dynamic triangle quad meshes honor transformations",
          "[cpp][spatial][distance][matrix][dynamic][transform]") {
  check_dynamic_transformed_distance<std::int32_t, float, 2>();
  check_dynamic_transformed_distance<std::int64_t, double, 2>();
  check_dynamic_transformed_distance<std::int32_t, float, 3>();
  check_dynamic_transformed_distance<std::int64_t, double, 3>();
}

TEMPLATE_TEST_CASE("all form pair matrices support distance and distance2",
                   "[cpp][spatial][distance][form]", float, double) {
  const auto mesh0_storage = triangle_mesh<TestType>(0);
  const auto mesh2_storage = triangle_mesh<TestType>(2);
  const auto cloud0_storage = origin_cloud<TestType>(0);
  const auto cloud2_storage = origin_cloud<TestType>(2);
  const auto mesh0 = mesh0_storage.mesh();
  const auto mesh2 = mesh2_storage.mesh();
  const auto cloud0 = cloud0_storage.point_cloud();
  const auto cloud2 = cloud2_storage.point_cloud();

  CHECK(tf::cpp::distance2(mesh0, mesh2) ==
        Catch::Approx(4).margin(tolerance<TestType>()));
  CHECK(tf::cpp::distance(mesh0, mesh2) ==
        Catch::Approx(2).margin(tolerance<TestType>()));
  CHECK(tf::cpp::distance2(mesh0, cloud2) ==
        Catch::Approx(4).margin(tolerance<TestType>()));
  CHECK(tf::cpp::distance(mesh0, cloud2) ==
        Catch::Approx(2).margin(tolerance<TestType>()));
  CHECK(tf::cpp::distance2(cloud2, mesh0) ==
        Catch::Approx(4).margin(tolerance<TestType>()));
  CHECK(tf::cpp::distance(cloud2, mesh0) ==
        Catch::Approx(2).margin(tolerance<TestType>()));
  CHECK(tf::cpp::distance2(cloud0, cloud2) ==
        Catch::Approx(4).margin(tolerance<TestType>()));
  CHECK(tf::cpp::distance(cloud0, cloud2) ==
        Catch::Approx(2).margin(tolerance<TestType>()));
}

TEMPLATE_TEST_CASE(
    "empty primitive query batches bypass form tree construction",
    "[cpp][spatial][distance][empty]", float, double) {
  const auto full_mesh_storage = triangle_mesh<TestType>();
  const auto full_cloud_storage = origin_cloud<TestType>();
  const auto no_mesh_storage = empty_mesh<TestType>();
  const auto no_cloud_storage = empty_cloud<TestType>();
  const auto full_mesh = full_mesh_storage.mesh();
  const auto full_cloud = full_cloud_storage.point_cloud();
  const auto no_mesh = no_mesh_storage.mesh();
  const auto no_cloud = no_cloud_storage.point_cloud();
  using Other =
      std::conditional_t<std::is_same<TestType, float>::value, double, float>;
  auto empty_query = tf::cpp::primitive<Other>(tf::cpp::primitive_kind::point,
                                               make_empty<Other>({0, 3}));

  const auto full_mesh_result = tf::cpp::distance(full_mesh, empty_query);
  const auto full_cloud_result = tf::cpp::distance2(full_cloud, empty_query);
  const auto empty_mesh_result = tf::cpp::distance2(no_mesh, empty_query);
  const auto empty_cloud_result = tf::cpp::distance(no_cloud, empty_query);
  REQUIRE(full_mesh_result.is_batch());
  REQUIRE(full_cloud_result.is_batch());
  REQUIRE(empty_mesh_result.is_batch());
  REQUIRE(empty_cloud_result.is_batch());
  CHECK(full_mesh_result.batch().empty());
  CHECK(full_cloud_result.batch().empty());
  CHECK(empty_mesh_result.batch().empty());
  CHECK(empty_cloud_result.batch().empty());
  CHECK_FALSE(full_mesh_storage.cache.is_tree_built());
  CHECK_FALSE(full_cloud_storage.cache.is_tree_built());
  CHECK_FALSE(no_mesh_storage.cache.is_tree_built());
  CHECK_FALSE(no_cloud_storage.cache.is_tree_built());
  CHECK(full_mesh_storage.cache.tree_build_count() == 0);
  CHECK(full_cloud_storage.cache.tree_build_count() == 0);
  CHECK(no_mesh_storage.cache.tree_build_count() == 0);
  CHECK(no_cloud_storage.cache.tree_build_count() == 0);
}

TEMPLATE_TEST_CASE("empty forms reject scalar and nonempty primitive queries",
                   "[cpp][spatial][distance][empty]", float, double) {
  const auto mesh_storage = empty_mesh<TestType>();
  const auto cloud_storage = empty_cloud<TestType>();
  const auto mesh = mesh_storage.mesh();
  const auto cloud = cloud_storage.point_cloud();
  auto scalar = point<TestType>(0, 0, 0);
  auto batch = point_batch<TestType>({0, 0, 0, 1, 1, 1}, 2);
  CHECK_FALSE(mesh_storage.cache.is_tree_built());
  CHECK_FALSE(cloud_storage.cache.is_tree_built());

  CHECK_THROWS_AS(tf::cpp::distance(mesh, scalar), std::invalid_argument);
  CHECK(mesh_storage.cache.is_tree_fresh(mesh.geometry()));
  CHECK(mesh_storage.cache.tree_build_count() == 1);
  CHECK_THROWS_AS(tf::cpp::distance2(mesh, batch), std::invalid_argument);
  CHECK(mesh_storage.cache.tree_build_count() == 1);

  CHECK_THROWS_AS(tf::cpp::distance2(cloud, scalar), std::invalid_argument);
  CHECK(cloud_storage.cache.is_tree_fresh(cloud.geometry()));
  CHECK(cloud_storage.cache.tree_build_count() == 1);
  CHECK_THROWS_AS(tf::cpp::distance(cloud, batch), std::invalid_argument);
  CHECK(cloud_storage.cache.tree_build_count() == 1);
}

TEMPLATE_TEST_CASE("empty forms reject every form pair before distance kernels",
                   "[cpp][spatial][distance][empty]", float, double) {
  const auto no_mesh_a_storage = empty_mesh<TestType>();
  const auto no_mesh_b_storage = empty_mesh<TestType>();
  const auto no_cloud_a_storage = empty_cloud<TestType>();
  const auto no_cloud_b_storage = empty_cloud<TestType>();
  const auto full_mesh_a_storage = triangle_mesh<TestType>();
  const auto full_cloud_a_storage = origin_cloud<TestType>();
  const auto no_mesh_a = no_mesh_a_storage.mesh();
  const auto no_mesh_b = no_mesh_b_storage.mesh();
  const auto no_cloud_a = no_cloud_a_storage.point_cloud();
  const auto no_cloud_b = no_cloud_b_storage.point_cloud();
  const auto full_mesh_a = full_mesh_a_storage.mesh();
  const auto full_cloud_a = full_cloud_a_storage.point_cloud();

  check_empty_form_pair(no_mesh_a, no_mesh_b);
  check_empty_form_pair(no_mesh_a, no_cloud_a);
  check_empty_form_pair(no_cloud_a, no_mesh_a);
  check_empty_form_pair(no_cloud_a, no_cloud_b);
  check_empty_form_pair(no_mesh_a, full_mesh_a);
  check_empty_form_pair(full_mesh_a, no_mesh_a);
  check_empty_form_pair(no_mesh_a, full_cloud_a);
  check_empty_form_pair(full_mesh_a, no_cloud_a);
  check_empty_form_pair(no_cloud_a, full_mesh_a);
  check_empty_form_pair(full_cloud_a, no_mesh_a);
  check_empty_form_pair(no_cloud_a, full_cloud_a);
  check_empty_form_pair(full_cloud_a, no_cloud_a);

  const auto first_empty_storage = empty_mesh<TestType>();
  const auto skipped_second_storage = origin_cloud<TestType>();
  const auto first_empty = first_empty_storage.mesh();
  const auto skipped_second = skipped_second_storage.point_cloud();
  check_empty_form_pair(first_empty, skipped_second);
  CHECK(first_empty_storage.cache.is_tree_fresh(first_empty.geometry()));
  CHECK(first_empty_storage.cache.tree_build_count() == 1);
  CHECK_FALSE(skipped_second_storage.cache.is_tree_built());
  CHECK(skipped_second_storage.cache.tree_build_count() == 0);

  const auto visited_first_storage = triangle_mesh<TestType>();
  const auto second_empty_storage = empty_cloud<TestType>();
  const auto visited_first = visited_first_storage.mesh();
  const auto second_empty = second_empty_storage.point_cloud();
  check_empty_form_pair(visited_first, second_empty);
  CHECK(visited_first_storage.cache.is_tree_fresh(visited_first.geometry()));
  CHECK(second_empty_storage.cache.is_tree_fresh(second_empty.geometry()));
  CHECK(visited_first_storage.cache.tree_build_count() == 1);
  CHECK(second_empty_storage.cache.tree_build_count() == 1);
}

TEST_CASE("form distances reject mixed form precision",
          "[cpp][spatial][distance][precision]") {
  const auto mesh32_storage = triangle_mesh<float>();
  const auto mesh64_storage = triangle_mesh<double>();
  const auto cloud32_storage = origin_cloud<float>();
  const auto cloud64_storage = origin_cloud<double>();
  const auto mesh32 = mesh32_storage.mesh();
  const auto mesh64 = mesh64_storage.mesh();
  const auto cloud32 = cloud32_storage.point_cloud();
  const auto cloud64 = cloud64_storage.point_cloud();
  CHECK_THROWS_AS(tf::cpp::distance(mesh32, mesh64), std::invalid_argument);
  CHECK_THROWS_AS(tf::cpp::distance2(mesh32, cloud64), std::invalid_argument);
  CHECK_THROWS_AS(tf::cpp::distance(cloud32, mesh64), std::invalid_argument);
  CHECK_THROWS_AS(tf::cpp::distance2(cloud32, cloud64), std::invalid_argument);
}

TEMPLATE_TEST_CASE("form distances honor transformations",
                   "[cpp][spatial][distance][transform]", float, double) {
  auto mesh_storage = triangle_mesh<TestType>();
  auto cloud_storage = origin_cloud<TestType>();
  mesh_storage.place(translation<TestType>(0, 0, 5));
  cloud_storage.place(translation<TestType>(0, 0, 5));
  const auto mesh = mesh_storage.mesh();
  const auto cloud = cloud_storage.point_cloud();
  auto query = point<TestType>(0, 0, 0);

  CHECK(tf::cpp::distance(mesh, query).scalar() ==
        Catch::Approx(5).margin(tolerance<TestType>()));
  CHECK(tf::cpp::distance2(cloud, query).scalar() ==
        Catch::Approx(25).margin(tolerance<TestType>()));

  const auto origin_mesh_storage = triangle_mesh<TestType>();
  const auto origin_cloud_storage = origin_cloud<TestType>();
  CHECK(tf::cpp::distance(origin_mesh_storage.mesh(), mesh) ==
        Catch::Approx(5).margin(tolerance<TestType>()));
  CHECK(tf::cpp::distance2(origin_cloud_storage.point_cloud(), cloud) ==
        Catch::Approx(25).margin(tolerance<TestType>()));
}

TEST_CASE("distance queries reuse fresh trees and rebuild stale trees",
          "[cpp][spatial][distance][tree]") {
  auto mesh_storage = triangle_mesh<float>();
  auto query = point<float>(0, 0, 2);
  const auto mesh = mesh_storage.mesh();
  CHECK_FALSE(mesh_storage.cache.is_tree_built());
  CHECK(tf::cpp::distance(mesh, query).scalar() == Catch::Approx(2));
  CHECK(mesh_storage.cache.is_tree_fresh(mesh.geometry()));
  CHECK(mesh_storage.cache.tree_build_count() == 1);
  static_cast<void>(tf::cpp::distance2(mesh, query));
  CHECK(mesh_storage.cache.tree_build_count() == 1);

  // the caller states the change, and assembles again for the reading it made
  auto &coordinates = mesh_storage.polygons.points_buffer().data_buffer();
  coordinates[2] = 1;
  coordinates[5] = 1;
  coordinates[8] = 1;
  mesh_storage.cache.points_changed();
  const auto moved_mesh = mesh_storage.mesh();
  CHECK_FALSE(mesh_storage.cache.is_tree_fresh(moved_mesh.geometry()));
  CHECK(tf::cpp::distance(moved_mesh, query).scalar() == Catch::Approx(1));
  CHECK(mesh_storage.cache.is_tree_fresh(moved_mesh.geometry()));
  CHECK(mesh_storage.cache.tree_build_count() == 2);

  auto cloud_storage = origin_cloud<float>();
  const auto cloud = cloud_storage.point_cloud();
  CHECK_FALSE(cloud_storage.cache.is_tree_built());
  CHECK(tf::cpp::distance(cloud, query).scalar() == Catch::Approx(2));
  CHECK(cloud_storage.cache.tree_build_count() == 1);
  static_cast<void>(tf::cpp::distance2(cloud, query));
  CHECK(cloud_storage.cache.tree_build_count() == 1);
  cloud_storage.points.data_buffer()[2] = 1;
  cloud_storage.cache.points_changed();
  const auto moved_cloud = cloud_storage.point_cloud();
  CHECK_FALSE(cloud_storage.cache.is_tree_fresh(moved_cloud.geometry()));
  CHECK(tf::cpp::distance(moved_cloud, query).scalar() == Catch::Approx(1));
  CHECK(cloud_storage.cache.tree_build_count() == 2);
}

TEST_CASE(
    "distance validates malformed vector unequal empty and invalid inputs",
    "[cpp][spatial][distance][validation]") {
  CHECK_THROWS_AS((tf::cpp::primitive<float>(
                      tf::cpp::primitive_kind::point,
                      make_array<float>({0, 0, 0, 1, 1, 1}, {2, 3, 1}))),
                  std::invalid_argument);
  CHECK_THROWS_AS((tf::cpp::primitive<float>(
                      tf::cpp::primitive_kind::polygon,
                      make_array<float>({0, 0, 0, 1, 0, 0}, {1, 2, 3}))),
                  std::invalid_argument);

  auto vector = origin_primitive<float>(tf::cpp::primitive_kind::vector);
  auto scalar = point<float>(0, 0, 0);
  const auto mesh_storage = triangle_mesh<float>();
  const auto mesh = mesh_storage.mesh();
  CHECK_THROWS_AS(tf::cpp::distance(vector, scalar), std::invalid_argument);
  CHECK_THROWS_AS(tf::cpp::distance2(scalar, vector), std::invalid_argument);
  CHECK_THROWS_AS(tf::cpp::distance(mesh, vector), std::invalid_argument);

  auto batch2 = point_batch<float>({0, 0, 0, 1, 1, 1}, 2);
  auto batch3 = point_batch<float>({0, 0, 0, 1, 1, 1, 2, 2, 2}, 3);
  CHECK_THROWS_AS(tf::cpp::distance(batch2, batch3), std::invalid_argument);
  CHECK_THROWS_AS(tf::cpp::distance2(batch3, batch2), std::invalid_argument);

  auto empty = tf::cpp::primitive<float>(tf::cpp::primitive_kind::point,
                                         make_empty<float>({0, 3}));
  REQUIRE(tf::cpp::distance(empty, scalar).is_batch());
  CHECK(tf::cpp::distance(empty, scalar).batch().empty());
  CHECK(tf::cpp::distance2(scalar, empty).batch().empty());
  CHECK(tf::cpp::distance(empty, empty).batch().empty());
  CHECK(tf::cpp::distance(mesh, empty).batch().empty());

  const auto no_mesh_storage = empty_mesh<float>();
  const auto no_cloud_storage = empty_cloud<float>();
  const auto no_mesh = no_mesh_storage.mesh();
  const auto no_cloud = no_cloud_storage.point_cloud();
  CHECK_THROWS_AS(tf::cpp::distance(no_mesh, scalar), std::invalid_argument);
  CHECK_THROWS_AS(tf::cpp::distance(no_cloud, scalar), std::invalid_argument);
  CHECK_THROWS_AS(tf::cpp::distance(no_mesh, mesh), std::invalid_argument);
}

TEST_CASE(
    "mesh spatial queries validate face point indices before tree kernels",
    "[cpp][spatial][validation][face-indices]") {
  const auto query = point<float>(0, 0, 2);
  const auto ray =
      tf::cpp::primitive<float>(tf::cpp::primitive_kind::ray,
                                make_array<float>({0, 0, -1, 0, 0, 1}, {2, 3}));
  const auto check_refusals = [&](const auto &storage) {
    const auto mesh = storage.mesh();
    CHECK_THROWS_AS(tf::cpp::distance(mesh, query), std::out_of_range);
    CHECK_THROWS_AS(tf::cpp::neighbor_search(mesh, query), std::out_of_range);
    CHECK_THROWS_AS(tf::cpp::ray_cast(ray, mesh), std::out_of_range);
    // the refusal is the cache's, per reading, and no tree kernel ran
    CHECK(storage.cache.tree_build_count() == 0);
  };

  const auto sound_storage = triangle_mesh<float>();
  CHECK_NOTHROW(tf::cpp::distance(sound_storage.mesh(), query));
  REQUIRE(sound_storage.cache.tree_build_count() == 1);

  using storage_type =
      tf::cpp::test::owned_mesh<tf::cpp::default_index_t, float>;
  const auto negative_storage =
      storage_type{tf::cpp::test::polygons_of<tf::cpp::default_index_t, float>(
          {-1, 1, 2}, {-1, -1, 0, 1, -1, 0, 0, 1, 0})};
  const auto out_of_reach_storage =
      storage_type{tf::cpp::test::polygons_of<tf::cpp::default_index_t, float>(
          {0, 1, 2}, {-1, -1, 0, 1, -1, 0})};
  check_refusals(negative_storage);
  check_refusals(out_of_reach_storage);
}

TEST_CASE("distance batch results own their output storage",
          "[cpp][spatial][distance][ownership]") {
  auto result = owned_batch_result<double>();
  REQUIRE(result.is_batch());
  auto values = result.batch();
  CHECK(values[0] == Catch::Approx(3));
  CHECK(values[1] == Catch::Approx(4));
  result = tf::cpp::distance_result<double>(1.0);
  CHECK(values[0] == Catch::Approx(3));
  CHECK(values[1] == Catch::Approx(4));
}

TEST_CASE("a filled cache is read concurrently without ceremony",
          "[cpp][spatial][distance][concurrency]") {
  // filling is the caller's to sequence; reads of filled state are free and
  // unlimited, so the warm carrier is shared as it stands
  const auto cloud_storage = origin_cloud<float>();
  const auto cloud = cloud_storage.point_cloud();
  auto query = point<float>(0, 0, 5);
  tf::cpp::build_tree(cloud);
  REQUIRE(cloud_storage.cache.tree_build_count() == 1);

  std::atomic<bool> start{false};
  auto reader = std::async(std::launch::async, [&] {
    while (!start.load(std::memory_order_acquire)) {
    }
    for (int iteration = 0; iteration < 100; ++iteration)
      static_cast<void>(tf::cpp::distance(cloud, query));
  });

  start.store(true, std::memory_order_release);
  for (int iteration = 0; iteration < 100; ++iteration)
    CHECK(tf::cpp::distance(cloud, query).scalar() ==
          Catch::Approx(5).margin(1e-5));
  reader.get();
  CHECK(cloud_storage.cache.is_tree_fresh(cloud.geometry()));
  CHECK(cloud_storage.cache.tree_build_count() == 1);
}

TEST_CASE("async distances preserve PP scalar batch and mixed precision",
          "[cpp][spatial][distance][async][primitive]") {
  auto origin32 = point<float>(0, 0, 0);
  auto offset64 = point<double>(0, 3, 4);
  auto batch32 = point_batch<float>({0, 0, 0, 0, 0, 2}, 2);
  const auto expected_scalar = tf::cpp::distance(origin32, offset64).scalar();
  const auto expected_batch = tf::cpp::distance2(batch32, offset64).batch();

  auto scalar = tf::cpp::async::distance(origin32, offset64);
  auto batch = tf::cpp::async::distance2(batch32, offset64);
  STATIC_REQUIRE(std::is_same_v<decltype(scalar),
                                std::future<tf::cpp::distance_result<double>>>);
  STATIC_REQUIRE(std::is_same_v<decltype(batch),
                                std::future<tf::cpp::distance_result<double>>>);
  CHECK(scalar.get().scalar() == Catch::Approx(expected_scalar));
  const auto actual_batch = batch.get().batch();
  REQUIRE(actual_batch.length() == expected_batch.length());
  CHECK(actual_batch[0] == Catch::Approx(expected_batch[0]));
  CHECK(actual_batch[1] == Catch::Approx(expected_batch[1]));

  const auto submissions = std::make_shared<std::atomic<int>>(0);
  auto custom = tf::cpp::async::distance(counting_resolver{submissions},
                                         origin32, offset64);
  CHECK(submissions->load(std::memory_order_relaxed) == 1);
  CHECK(custom.get().scalar() == Catch::Approx(expected_scalar));

  auto owned_a = point<float>(0, 0, 0);
  auto owned_b = point<double>(0, 0, 7);
  auto owned = tf::cpp::async::distance(owned_a, owned_b);
  owned_a = point<float>(100, 0, 0);
  owned_b = point<double>(100, 0, 0);
  CHECK(owned.get().scalar() == Catch::Approx(7));
}

TEST_CASE("async distances preserve FP FF cache empty and error behavior",
          "[cpp][spatial][distance][async][form]") {
  const auto mesh_storage = triangle_mesh<float>();
  const auto cloud_storage = origin_cloud<float>(2);
  const auto mesh = mesh_storage.mesh();
  const auto cloud = cloud_storage.point_cloud();
  auto scalar64 = point<double>(0, 0, 3);
  auto batch64 = point_batch<double>({0, 0, 3, 0, 0, 4}, 2);

  const auto expected_fp = tf::cpp::distance(mesh, scalar64).scalar();
  const auto expected_ff = tf::cpp::distance2(mesh, cloud);

  // a cache shared by concurrent jobs is filled before it is shared
  tf::cpp::build_tree(mesh);
  tf::cpp::build_tree(cloud);
  const auto mesh_builds = mesh_storage.cache.tree_build_count();
  const auto cloud_builds = cloud_storage.cache.tree_build_count();

  auto fp = tf::cpp::async::distance(mesh, scalar64);
  auto fp_batch = tf::cpp::async::distance2(cloud, batch64);
  auto ff = tf::cpp::async::distance2(mesh, cloud);
  STATIC_REQUIRE(std::is_same_v<decltype(fp),
                                std::future<tf::cpp::distance_result<float>>>);
  STATIC_REQUIRE(std::is_same_v<decltype(ff), std::future<float>>);
  CHECK(fp.get().scalar() == Catch::Approx(expected_fp));
  const auto values = fp_batch.get().batch();
  CHECK(values[0] == Catch::Approx(1));
  CHECK(values[1] == Catch::Approx(4));
  CHECK(ff.get() == Catch::Approx(expected_ff));
  CHECK(tf::cpp::async::distance(mesh, cloud).get() ==
        Catch::Approx(tf::cpp::distance(mesh, cloud)));
  CHECK(mesh_storage.cache.tree_build_count() == mesh_builds);
  CHECK(cloud_storage.cache.tree_build_count() == cloud_builds);

  // the carrier is borrowed, so the storage it names is kept alive by the
  // keepalive it was assembled with and no handle of the caller's has to
  // outlive the call
  auto held_query = point<double>(0, 0, 2);
  auto held = tf::cpp::async::distance(held_carrier_of(triangle_mesh<float>()),
                                       held_query);
  held_query = point<double>(0, 0, 20);
  CHECK(held.get().scalar() == Catch::Approx(2));

  const auto no_mesh_storage = empty_mesh<float>();
  const auto no_mesh = no_mesh_storage.mesh();
  auto empty_query = tf::cpp::primitive<double>(tf::cpp::primitive_kind::point,
                                                make_empty<double>({0, 3}));
  const auto empty = tf::cpp::async::distance2(no_mesh, empty_query).get();
  REQUIRE(empty.is_batch());
  CHECK(empty.batch().empty());
  CHECK_FALSE(no_mesh_storage.cache.is_tree_built());

  auto invalid_query = origin_primitive<float>(tf::cpp::primitive_kind::vector);
  auto failure = tf::cpp::async::distance(no_mesh, invalid_query);
  CHECK_THROWS_AS(failure.get(), std::invalid_argument);

  const auto mesh64_storage = triangle_mesh<double>();
  auto mixed_failure = tf::cpp::async::distance(mesh, mesh64_storage.mesh());
  STATIC_REQUIRE(std::is_same_v<decltype(mixed_failure), std::future<double>>);
  CHECK_THROWS_AS(mixed_failure.get(), std::invalid_argument);
}
