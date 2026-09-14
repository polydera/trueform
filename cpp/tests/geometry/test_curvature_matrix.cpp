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
#include "mixed_mesh.hpp"

#include "trueform/cpp/core/mesh.hpp"
#include "trueform/cpp/geometry/async/principal_curvatures.hpp"
#include "trueform/cpp/geometry/async/principal_directions.hpp"
#include "trueform/cpp/geometry/async/shape_index.hpp"
#include "trueform/cpp/geometry/point_normals.hpp"
#include "trueform/cpp/geometry/principal_curvatures.hpp"
#include "trueform/cpp/geometry/principal_directions.hpp"
#include "trueform/cpp/geometry/shape_index.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <future>
#include <initializer_list>
#include <map>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

template <typename Index, typename Real> struct matrix_row {
  using real_type = Real;
  using index_type = Index;
  using owned_type = tf::cpp::test::owned_mesh<Index, Real, 3>;
};

using float_int32 = matrix_row<std::int32_t, float>;
using float_int64 = matrix_row<std::int64_t, float>;
using double_int32 = matrix_row<std::int32_t, double>;
using double_int64 = matrix_row<std::int64_t, double>;

template <typename Mesh, typename = void>
struct has_principal_curvatures : std::false_type {};
template <typename Mesh>
struct has_principal_curvatures<
    Mesh, std::void_t<decltype(tf::cpp::principal_curvatures(
              std::declval<const Mesh &>()))>> : std::true_type {};

template <typename Mesh, typename = void>
struct has_principal_directions : std::false_type {};
template <typename Mesh>
struct has_principal_directions<
    Mesh, std::void_t<decltype(tf::cpp::principal_directions(
              std::declval<const Mesh &>()))>> : std::true_type {};

template <typename Mesh, typename = void>
struct has_shape_index : std::false_type {};
template <typename Mesh>
struct has_shape_index<Mesh, std::void_t<decltype(tf::cpp::shape_index(
                                 std::declval<const Mesh &>()))>>
    : std::true_type {};

template <typename Mesh, typename = void>
struct has_async_principal_curvatures : std::false_type {};
template <typename Mesh>
struct has_async_principal_curvatures<
    Mesh, std::void_t<decltype(tf::cpp::async::principal_curvatures(
              std::declval<const Mesh &>()))>> : std::true_type {};

template <typename Mesh, typename = void>
struct has_async_principal_directions : std::false_type {};
template <typename Mesh>
struct has_async_principal_directions<
    Mesh, std::void_t<decltype(tf::cpp::async::principal_directions(
              std::declval<const Mesh &>()))>> : std::true_type {};

template <typename Mesh, typename = void>
struct has_async_shape_index : std::false_type {};
template <typename Mesh>
struct has_async_shape_index<Mesh,
                             std::void_t<decltype(tf::cpp::async::shape_index(
                                 std::declval<const Mesh &>()))>>
    : std::true_type {};

static_assert(
    has_principal_curvatures<tf::cpp::mesh<std::int64_t, float, 3>>::value);
static_assert(
    has_principal_directions<tf::cpp::mesh<std::int32_t, double, 3>>::value);
static_assert(has_shape_index<tf::cpp::mesh<std::int64_t, double, 3>>::value);
static_assert(has_async_principal_curvatures<
              tf::cpp::mesh<std::int32_t, float, 3>>::value);
static_assert(has_async_principal_directions<
              tf::cpp::mesh<std::int64_t, double, 3>>::value);
static_assert(
    has_async_shape_index<tf::cpp::mesh<std::int64_t, float, 3>>::value);
static_assert(
    !has_principal_curvatures<tf::cpp::mesh<std::int32_t, float, 2>>::value);
static_assert(
    !has_principal_directions<tf::cpp::mesh<std::int64_t, double, 2>>::value);
static_assert(!has_shape_index<tf::cpp::mesh<std::int64_t, float, 2>>::value);
static_assert(!has_async_principal_curvatures<
              tf::cpp::mesh<std::int32_t, double, 2>>::value);
static_assert(!has_async_principal_directions<
              tf::cpp::mesh<std::int64_t, float, 2>>::value);
static_assert(
    !has_async_shape_index<tf::cpp::mesh<std::int64_t, double, 2>>::value);

template <typename Row, std::size_t Ngon>
auto python_planar_fixture()
    -> tf::cpp::test::mesh_at<typename Row::owned_type, Ngon> {
  using Real = typename Row::real_type;
  using Index = typename Row::index_type;
  if constexpr (Ngon == 3) {
    // Exact python/tests/test_geometry_compute.py triangle fixture.
    return {tf::cpp::test::polygons_of<Index, Real>(
        {0, 1, 2, 1, 3, 2},
        {0, 0, 0, 1, 0, 0, Real{0.5}, 1, 0, Real{1.5}, 1, 0})};
  } else {
    // Exact mixed triangle/quad Python dynamic fixture.
    return {tf::cpp::test::polygons_of<Index, Real>(
        {0, 3, 7}, {0, 1, 2, 1, 3, 4, 2},
        {0, 0, 0, 1, 0, 0, Real{0.5}, 1, 0, 2, 0, 0, Real{1.5}, 1, 0})};
  }
}

template <typename Real>
auto normalized(std::array<Real, 3> value) -> std::array<Real, 3> {
  const auto length = std::sqrt(value[0] * value[0] + value[1] * value[1] +
                                value[2] * value[2]);
  for (auto &coordinate : value)
    coordinate /= length;
  return value;
}

template <typename Row, std::size_t Ngon>
auto python_sphere_fixture()
    -> tf::cpp::test::mesh_at<typename Row::owned_type, Ngon> {
  using Real = typename Row::real_type;
  using Index = typename Row::index_type;
  // Exact icosphere seed and two subdivisions from the Python curvature tests.
  const auto phi = static_cast<Real>((1.0 + std::sqrt(5.0)) / 2.0);
  std::vector<std::array<Real, 3>> vertices = {
      {-1, phi, 0}, {1, phi, 0}, {-1, -phi, 0}, {1, -phi, 0},
      {0, -1, phi}, {0, 1, phi}, {0, -1, -phi}, {0, 1, -phi},
      {phi, 0, -1}, {phi, 0, 1}, {-phi, 0, -1}, {-phi, 0, 1},
  };
  for (auto &vertex : vertices)
    vertex = normalized(vertex);

  std::vector<std::array<Index, 3>> triangles = {
      {0, 11, 5}, {0, 5, 1},  {0, 1, 7},   {0, 7, 10}, {0, 10, 11},
      {1, 5, 9},  {5, 11, 4}, {11, 10, 2}, {10, 7, 6}, {7, 1, 8},
      {3, 9, 4},  {3, 4, 2},  {3, 2, 6},   {3, 6, 8},  {3, 8, 9},
      {4, 9, 5},  {2, 4, 11}, {6, 2, 10},  {8, 6, 7},  {9, 8, 1},
  };

  for (int subdivision = 0; subdivision < 2; ++subdivision) {
    std::map<std::pair<Index, Index>, Index> midpoint_ids;
    std::vector<std::array<Index, 3>> subdivided;
    subdivided.reserve(triangles.size() * 4);
    auto midpoint = [&](Index first, Index second) {
      const auto key = std::minmax(first, second);
      const auto found = midpoint_ids.find(key);
      if (found != midpoint_ids.end())
        return found->second;
      const auto &a = vertices[static_cast<std::size_t>(first)];
      const auto &b = vertices[static_cast<std::size_t>(second)];
      const auto id = static_cast<Index>(vertices.size());
      vertices.push_back(normalized(
          std::array<Real, 3>{(a[0] + b[0]) / Real{2}, (a[1] + b[1]) / Real{2},
                              (a[2] + b[2]) / Real{2}}));
      midpoint_ids.emplace(key, id);
      return id;
    };
    for (const auto &triangle : triangles) {
      const auto m01 = midpoint(triangle[0], triangle[1]);
      const auto m12 = midpoint(triangle[1], triangle[2]);
      const auto m20 = midpoint(triangle[2], triangle[0]);
      subdivided.push_back({triangle[0], m01, m20});
      subdivided.push_back({triangle[1], m12, m01});
      subdivided.push_back({triangle[2], m20, m12});
      subdivided.push_back({m01, m12, m20});
    }
    triangles = std::move(subdivided);
  }

  std::vector<Real> point_data;
  point_data.reserve(vertices.size() * 3);
  for (const auto &vertex : vertices)
    point_data.insert(point_data.end(), vertex.begin(), vertex.end());

  std::vector<Index> index_data;
  index_data.reserve(triangles.size() * 3);
  for (const auto &triangle : triangles)
    index_data.insert(index_data.end(), triangle.begin(), triangle.end());
  if constexpr (Ngon == 3) {
    return {
        tf::cpp::test::polygons_of<Index, Real, 3, std::vector<Index>,
                                   std::vector<Real>>(index_data, point_data)};
  } else {
    std::vector<Index> offsets(triangles.size() + 1);
    for (std::size_t index = 0; index < offsets.size(); ++index)
      offsets[index] = static_cast<Index>(3 * index);
    return {tf::cpp::test::polygons_of<Index, Real, 3, std::vector<Index>,
                                       std::vector<Index>, std::vector<Real>>(
        offsets, index_data, point_data)};
  }
}

template <typename Row> auto saddle_mesh() -> typename Row::owned_type {
  using Real = typename Row::real_type;
  using Index = typename Row::index_type;
  return {tf::cpp::test::polygons_of<Index, Real>(
      {0, 1, 4, 0, 4, 3, 1, 2, 5, 1, 5, 4, 3, 4, 7, 3, 7, 6, 4, 5, 8, 4, 8, 7},
      {-1, -1, 1, 0, -1, 0, 1,  -1, -1, -1, 0, 0, 0, 0,
       0,  1,  0, 0, -1, 1, -1, 0,  1,  0,  1, 1, 1})};
}

template <typename Row, std::size_t Ngon>
auto empty_mesh() -> tf::cpp::test::mesh_at<typename Row::owned_type, Ngon> {
  return {};
}

template <typename Real> auto tolerance() -> double {
  return std::is_same_v<Real, float> ? 2e-4 : 2e-10;
}

template <typename Real>
auto arrays_match(const tf::cpp::nd_array<Real> &first,
                  const tf::cpp::nd_array<Real> &second) -> bool {
  if (first.raw_shape() != second.raw_shape())
    return false;
  for (std::size_t index = 0; index < first.length(); ++index) {
    const auto scale =
        std::max(1.0, std::abs(static_cast<double>(second[index])));
    if (std::abs(static_cast<double>(first[index] - second[index])) >
        tolerance<Real>() * scale)
      return false;
  }
  return true;
}

struct curvature_matrix_counting_resolver {
  std::shared_ptr<std::atomic<int>> submissions;

  template <typename T>
  using state_type = tf::cpp::async::detail::future_state<T>;

  template <typename T>
  auto make_state() const -> std::shared_ptr<state_type<T>> {
    submissions->fetch_add(1, std::memory_order_relaxed);
    return std::make_shared<state_type<T>>();
  }
};

template <typename TestType, std::size_t Ngon>
auto check_curvature_case_1() -> void {
  using Real = typename TestType::real_type;
  using Index = typename TestType::index_type;
  static_cast<void>(sizeof(Index));
  const auto owned = python_planar_fixture<TestType, Ngon>();
  const auto values =
      tf::cpp::principal_curvatures<Index, Real, 3, Ngon>(owned.mesh(), 2);
  const auto full =
      tf::cpp::principal_directions<Index, Real, 3, Ngon>(owned.mesh(), 2);
  const auto indices =
      tf::cpp::shape_index<Index, Real, 3, Ngon>(owned.mesh(), 2);
  static_assert(std::is_same_v<decltype(values.k0), tf::cpp::nd_array<Real>>);
  static_assert(std::is_same_v<decltype(full.d0), tf::cpp::nd_array<Real>>);
  static_assert(
      std::is_same_v<decltype(indices), const tf::cpp::nd_array<Real>>);
  const auto count = static_cast<int>(owned.mesh().number_of_points());
  REQUIRE(values.k0.raw_shape() == tf::small_vector<int, 3>{count});
  REQUIRE(values.k1.raw_shape() == tf::small_vector<int, 3>{count});
  REQUIRE(full.d0.raw_shape() == tf::small_vector<int, 3>{count, 3});
  REQUIRE(full.d1.raw_shape() == tf::small_vector<int, 3>{count, 3});
  REQUIRE(indices.raw_shape() == tf::small_vector<int, 3>{count});
  for (int point = 0; point < count; ++point) {
    CHECK(values.k0[point] == Real{0});
    CHECK(values.k1[point] == Real{0});
    CHECK(full.k0[point] == Real{0});
    CHECK(full.k1[point] == Real{0});
    CHECK(indices[point] == Real{0});
    const auto offset = static_cast<std::size_t>(3 * point);
    CHECK(full.d0[offset] == Real{1});
    CHECK(full.d0[offset + 1] == Real{0});
    CHECK(full.d0[offset + 2] == Real{0});
    CHECK(full.d1[offset] == Real{0});
    CHECK(full.d1[offset + 1] == Real{1});
    CHECK(full.d1[offset + 2] == Real{0});
  }
  CHECK(owned.cache.is_face_membership_fresh(owned.mesh().geometry()));
  CHECK(owned.cache.is_vertex_link_fresh(owned.mesh().geometry()));
  CHECK(owned.cache.face_membership_build_count() == 1);
}

template <typename TestType, std::size_t Ngon>
auto check_curvature_case_2() -> void {
  using Real = typename TestType::real_type;
  using Index = typename TestType::index_type;
  static_cast<void>(sizeof(Index));
  const auto owned = python_sphere_fixture<TestType, Ngon>();
  const auto mesh = owned.mesh();
  const auto zero = tf::cpp::principal_curvatures<Index, Real, 3>(mesh, 0);
  const auto k1 = tf::cpp::principal_curvatures<Index, Real, 3>(mesh, 1);
  const auto k2 = tf::cpp::principal_directions<Index, Real, 3>(mesh, 2);
  const auto k3 = tf::cpp::principal_curvatures<Index, Real, 3>(mesh, 3);
  const auto shape = tf::cpp::shape_index<Index, Real, 3>(mesh, 2);
  const auto normals = tf::cpp::point_normals(mesh);
  const auto count = static_cast<int>(mesh.number_of_points());
  REQUIRE(count == 162);
  REQUIRE(k1.k0.raw_shape() == tf::small_vector<int, 3>{count});
  REQUIRE(k2.d0.raw_shape() == tf::small_vector<int, 3>{count, 3});
  REQUIRE(k3.k1.raw_shape() == tf::small_vector<int, 3>{count});

  auto mean_k0 = 0.0;
  auto mean_k1 = 0.0;
  auto mean_shape = 0.0;
  for (int point = 0; point < count; ++point) {
    CHECK(zero.k0[point] == Real{0});
    CHECK(zero.k1[point] == Real{0});
    CHECK(std::isfinite(static_cast<double>(k1.k0[point])));
    CHECK(std::isfinite(static_cast<double>(k3.k1[point])));
    CHECK(k2.k0[point] >= k2.k1[point]);
    CHECK(shape[point] >= Real{-1} - Real{1e-2});
    CHECK(shape[point] <= Real{1} + Real{1e-2});
    const auto sum = static_cast<double>(k2.k0[point] + k2.k1[point]);
    const auto difference = static_cast<double>(k2.k0[point] - k2.k1[point]);
    const auto expected_shape =
        difference == 0.0
            ? (k2.k0[point] > Real{0} ? 1.0
                                      : (k2.k0[point] < Real{0} ? -1.0 : 0.0))
            : (2.0 / 3.14159265358979323846) * std::atan(sum / difference);
    CHECK(static_cast<double>(shape[point]) ==
          Catch::Approx(expected_shape).margin(tolerance<Real>()));

    const auto offset = static_cast<std::size_t>(3 * point);
    auto d0_length2 = 0.0;
    auto d1_length2 = 0.0;
    auto directions_dot = 0.0;
    auto d0_normal_dot = 0.0;
    auto d1_normal_dot = 0.0;
    for (int coordinate = 0; coordinate < 3; ++coordinate) {
      const auto d0 = static_cast<double>(k2.d0[offset + coordinate]);
      const auto d1 = static_cast<double>(k2.d1[offset + coordinate]);
      const auto normal = static_cast<double>(normals[offset + coordinate]);
      d0_length2 += d0 * d0;
      d1_length2 += d1 * d1;
      directions_dot += d0 * d1;
      d0_normal_dot += d0 * normal;
      d1_normal_dot += d1 * normal;
    }
    CHECK(std::sqrt(d0_length2) ==
          Catch::Approx(1.0).margin(tolerance<Real>()));
    CHECK(std::sqrt(d1_length2) ==
          Catch::Approx(1.0).margin(tolerance<Real>()));
    CHECK(directions_dot == Catch::Approx(0.0).margin(tolerance<Real>()));
    CHECK(d0_normal_dot == Catch::Approx(0.0).margin(tolerance<Real>()));
    CHECK(d1_normal_dot == Catch::Approx(0.0).margin(tolerance<Real>()));
    mean_k0 += static_cast<double>(k2.k0[point]);
    mean_k1 += static_cast<double>(k2.k1[point]);
    mean_shape += static_cast<double>(shape[point]);
  }
  mean_k0 /= count;
  mean_k1 /= count;
  mean_shape /= count;
  CHECK(mean_k0 == Catch::Approx(1.0).margin(0.35));
  CHECK(mean_k1 == Catch::Approx(1.0).margin(0.35));
  CHECK(mean_k0 == Catch::Approx(mean_k1).margin(0.3));
  CHECK(mean_shape > 0.0);
}

template <typename TestType, std::size_t Ngon>
auto check_curvature_case_3() -> void {
  using Real = typename TestType::real_type;
  using Index = typename TestType::index_type;
  static_cast<void>(sizeof(Index));
  const auto empty = empty_mesh<TestType, Ngon>();
  const auto values =
      tf::cpp::principal_curvatures<Index, Real, 3>(empty.mesh(), 2);
  const auto directions =
      tf::cpp::principal_directions<Index, Real, 3>(empty.mesh(), 2);
  const auto shape = tf::cpp::shape_index<Index, Real, 3>(empty.mesh(), 2);
  CHECK(values.k0.raw_shape() == tf::small_vector<int, 3>{0});
  CHECK(values.k1.raw_shape() == tf::small_vector<int, 3>{0});
  CHECK(directions.d0.raw_shape() == tf::small_vector<int, 3>{0, 3});
  CHECK(directions.d1.raw_shape() == tf::small_vector<int, 3>{0, 3});
  CHECK(shape.raw_shape() == tf::small_vector<int, 3>{0});

  const auto negative = python_planar_fixture<TestType, Ngon>();
  CHECK_THROWS_AS(
      (tf::cpp::principal_curvatures<Index, Real, 3>(negative.mesh(), -1)),
      std::invalid_argument);
  CHECK_THROWS_AS(
      (tf::cpp::principal_directions<Index, Real, 3>(negative.mesh(), -1)),
      std::invalid_argument);
  CHECK_THROWS_AS((tf::cpp::shape_index<Index, Real, 3>(negative.mesh(), -1)),
                  std::invalid_argument);
  CHECK_FALSE(negative.cache.is_face_membership_built());
  CHECK_FALSE(negative.cache.is_vertex_link_built());

  const auto async_negative = python_planar_fixture<TestType, Ngon>();
  auto failure =
      tf::cpp::async::shape_index<Index, Real, 3>(async_negative.mesh(), -1);
  CHECK_THROWS_AS(failure.get(), std::invalid_argument);
  CHECK_FALSE(async_negative.cache.is_face_membership_built());
  CHECK_FALSE(async_negative.cache.is_vertex_link_built());
}

template <typename TestType, std::size_t Ngon>
auto check_curvature_case_4() -> void {
  using Real = typename TestType::real_type;
  using Index = typename TestType::index_type;
  const auto owned = python_sphere_fixture<TestType, Ngon>();
  const auto expected =
      tf::cpp::principal_curvatures<Index, Real, 3>(owned.mesh(), 2);
  const auto expected_directions =
      tf::cpp::principal_directions<Index, Real, 3>(owned.mesh(), 2);
  const auto expected_shape =
      tf::cpp::shape_index<Index, Real, 3>(owned.mesh(), 2);

  const auto submissions = std::make_shared<std::atomic<int>>(0);
  auto future = tf::cpp::async::principal_curvatures(
      curvature_matrix_counting_resolver{submissions}, owned.mesh(), 2);
  static_assert(
      std::is_same_v<decltype(future),
                     std::future<tf::cpp::principal_curvatures_result<Real>>>);
  const auto actual = future.get();
  CHECK(arrays_match(actual.k0, expected.k0));
  CHECK(arrays_match(actual.k1, expected.k1));

  auto directions = tf::cpp::async::principal_directions<Index, Real, 3, Ngon>(
      owned.mesh(), 2);
  auto shape =
      tf::cpp::async::shape_index<Index, Real, 3, Ngon>(owned.mesh(), 2);
  static_assert(
      std::is_same_v<decltype(directions),
                     std::future<tf::cpp::principal_directions_result<Real>>>);
  static_assert(
      std::is_same_v<decltype(shape), std::future<tf::cpp::nd_array<Real>>>);
  const auto actual_directions = directions.get();
  const auto actual_shape = shape.get();
  CHECK(arrays_match(actual_directions.k0, expected_directions.k0));
  CHECK(arrays_match(actual_directions.k1, expected_directions.k1));
  CHECK(arrays_match(actual_directions.d0, expected_directions.d0));
  CHECK(arrays_match(actual_directions.d1, expected_directions.d1));
  CHECK(arrays_match(actual_shape, expected_shape));
  CHECK(submissions->load(std::memory_order_relaxed) == 1);

  // every array a result states is its own storage, so a caller may write one
  // without the others hearing of it
  CHECK(actual_directions.k0.raw_owner() != actual_directions.k1.raw_owner());
  CHECK(actual_directions.d0.raw_owner() != actual_directions.d1.raw_owner());
}

} // namespace

TEMPLATE_TEST_CASE("curvature matrix matches Python planar fixtures and typed "
                   "result contracts",
                   "[cpp][geometry][curvature][matrix][python-parity][planar]",
                   float_int32, float_int64, double_int32, double_int64) {
  using Real = typename TestType::real_type;
  using Index = typename TestType::index_type;
  using Mesh = tf::cpp::mesh<Index, Real, 3, 3>;
  using curvature_fn =
      tf::cpp::principal_curvatures_result<Real> (*)(const Mesh &, int);
  using directions_fn =
      tf::cpp::principal_directions_result<Real> (*)(const Mesh &, int);
  using shape_fn = tf::cpp::nd_array<Real> (*)(const Mesh &, int);
  const curvature_fn curvatures =
      &tf::cpp::principal_curvatures<Index, Real, 3, 3>;
  const directions_fn directions =
      &tf::cpp::principal_directions<Index, Real, 3, 3>;
  const shape_fn shape = &tf::cpp::shape_index<Index, Real, 3, 3>;
  REQUIRE(curvatures != nullptr);
  REQUIRE(directions != nullptr);
  REQUIRE(shape != nullptr);

  check_curvature_case_1<TestType, 3>();
  check_curvature_case_1<TestType, tf::dynamic_size>();
}

TEMPLATE_TEST_CASE(
    "shape index keeps the hyperbolic saddle on the authoritative branch",
    "[cpp][geometry][curvature][shape-index][saddle]", float_int32, float_int64,
    double_int32, double_int64) {
  using Real = typename TestType::real_type;
  using Index = typename TestType::index_type;
  const auto owned = saddle_mesh<TestType>();
  const auto values = tf::cpp::shape_index<Index, Real, 3>(owned.mesh(), 1);
  REQUIRE(values.length() == 9);
  CHECK(std::abs(values[4]) < Real{0.25});
  for (const auto value : values) {
    CHECK(value >= Real{-1});
    CHECK(value <= Real{1});
  }
}

TEMPLATE_TEST_CASE(
    "curvature matrix matches Python sphere k values directions and ranges",
    "[cpp][geometry][curvature][matrix][python-parity][sphere]", float_int32,
    float_int64, double_int32, double_int64) {
  check_curvature_case_2<TestType, 3>();
  check_curvature_case_2<TestType, tf::dynamic_size>();
}

TEMPLATE_TEST_CASE("curvature matrix handles empty invalid and negative inputs",
                   "[cpp][geometry][curvature][matrix][validation]",
                   float_int32, float_int64, double_int32, double_int64) {
  check_curvature_case_3<TestType, 3>();
  check_curvature_case_3<TestType, tf::dynamic_size>();
}

TEMPLATE_TEST_CASE("async curvature answers as the synchronous entry does",
                   "[cpp][geometry][curvature][matrix][async][ownership]",
                   float_int32, float_int64, double_int32, double_int64) {
  check_curvature_case_4<TestType, 3>();
  check_curvature_case_4<TestType, tf::dynamic_size>();
}
