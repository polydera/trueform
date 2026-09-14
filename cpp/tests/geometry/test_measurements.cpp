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

#include "trueform/cpp/geometry/area.hpp"
#include "trueform/cpp/geometry/async/area.hpp"
#include "trueform/cpp/geometry/async/max_edge_length.hpp"
#include "trueform/cpp/geometry/async/mean_edge_length.hpp"
#include "trueform/cpp/geometry/async/min_edge_length.hpp"
#include "trueform/cpp/geometry/async/normals.hpp"
#include "trueform/cpp/geometry/async/signed_volume.hpp"
#include "trueform/cpp/geometry/async/volume.hpp"
#include "trueform/cpp/geometry/max_edge_length.hpp"
#include "trueform/cpp/geometry/mean_edge_length.hpp"
#include "trueform/cpp/geometry/min_edge_length.hpp"
#include "trueform/cpp/geometry/normals.hpp"
#include "trueform/cpp/geometry/signed_volume.hpp"
#include "trueform/cpp/geometry/volume.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <future>
#include <initializer_list>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>

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

template <typename Real>
using measured_owned =
    tf::cpp::test::owned_mesh<tf::cpp::default_index_t, Real>;

template <typename Real> auto tetrahedron() -> measured_owned<Real> {
  return {tf::cpp::test::polygons_of<tf::cpp::default_index_t, Real>(
      {0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3},
      {Real{0}, Real{0}, Real{0}, Real{1}, Real{0}, Real{0}, Real{0}, Real{2},
       Real{0}, Real{0}, Real{0}, Real{3}})};
}

template <typename Real> auto scaled_and_translated() -> std::array<Real, 16> {
  return {Real{2}, Real{0},  Real{0}, Real{5}, Real{0}, Real{3},
          Real{0}, Real{-7}, Real{0}, Real{0}, Real{4}, Real{11},
          Real{0}, Real{0},  Real{0}, Real{1}};
}

template <typename T>
auto make_empty(tf::small_vector<int, 3> shape) -> tf::cpp::nd_array<T> {
  tf::buffer<T> buffer;
  buffer.allocate(0);
  return tf::cpp::nd_array<T>::from_buffer(std::move(buffer), std::move(shape));
}

template <typename Real, std::size_t Dims>
auto make_primitive(tf::cpp::primitive_kind kind,
                    std::initializer_list<Real> values,
                    tf::small_vector<int, 3> shape)
    -> tf::cpp::primitive<Real, Dims> {
  return tf::cpp::primitive<Real, Dims>(
      kind, make_array<Real>(values, std::move(shape)));
}

template <typename Real>
auto check_array(const tf::cpp::nd_array<Real> &actual,
                 tf::small_vector<int, 3> expected_shape,
                 std::initializer_list<double> expected_values, double margin)
    -> void {
  REQUIRE(actual.raw_shape() == expected_shape);
  REQUIRE(actual.length() == expected_values.size());
  std::size_t index = 0;
  for (const auto expected : expected_values) {
    CHECK(static_cast<double>(actual[index]) ==
          Catch::Approx(expected).margin(margin));
    ++index;
  }
}

template <typename Real> auto tolerance() -> double {
  return std::is_same_v<Real, float> ? 1e-5 : 1e-12;
}

template <typename Primitive, typename = void>
struct has_primitive_normals : std::false_type {};

template <typename Primitive>
struct has_primitive_normals<
    Primitive,
    std::void_t<decltype(tf::cpp::normals(std::declval<const Primitive &>()))>>
    : std::true_type {};

template <typename Primitive, typename = void>
struct has_async_primitive_normals : std::false_type {};

template <typename Primitive>
struct has_async_primitive_normals<Primitive,
                                   std::void_t<decltype(tf::cpp::async::normals(
                                       std::declval<const Primitive &>()))>>
    : std::true_type {};

static_assert(has_primitive_normals<tf::cpp::primitive<float, 3>>::value);
static_assert(has_primitive_normals<tf::cpp::primitive<double, 3>>::value);
static_assert(!has_primitive_normals<tf::cpp::primitive<float, 2>>::value,
              "runtime primitive normals must be unavailable in 2D");
static_assert(!has_primitive_normals<tf::cpp::primitive<double, 2>>::value,
              "runtime primitive normals must be unavailable in 2D");
static_assert(has_async_primitive_normals<tf::cpp::primitive<float, 3>>::value);
static_assert(
    has_async_primitive_normals<tf::cpp::primitive<double, 3>>::value);
static_assert(!has_async_primitive_normals<tf::cpp::primitive<float, 2>>::value,
              "async runtime primitive normals must be unavailable in 2D");
static_assert(
    !has_async_primitive_normals<tf::cpp::primitive<double, 2>>::value,
    "async runtime primitive normals must be unavailable in 2D");

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

} // namespace

TEMPLATE_TEST_CASE("mesh measurements preserve dtype and geometric values",
                   "[cpp][geometry][measurements]", float, double) {
  const auto owned = tetrahedron<TestType>();
  const auto value = owned.mesh();

  static_assert(std::is_same_v<decltype(tf::cpp::area(value)), TestType>);
  static_assert(
      std::is_same_v<decltype(tf::cpp::signed_volume(value)), TestType>);
  static_assert(
      std::is_same_v<decltype(tf::cpp::mean_edge_length(value)), TestType>);
  static_assert(
      std::is_same_v<decltype(tf::cpp::min_edge_length(value)), TestType>);
  static_assert(
      std::is_same_v<decltype(tf::cpp::max_edge_length(value)), TestType>);

  const auto expected_mean =
      (1.0 + 2.0 + 3.0 + std::sqrt(5.0) + std::sqrt(10.0) + std::sqrt(13.0)) /
      6.0;
  CHECK(static_cast<double>(tf::cpp::area(value)) ==
        Catch::Approx(9.0).margin(tolerance<TestType>()));
  CHECK(static_cast<double>(tf::cpp::signed_volume(value)) ==
        Catch::Approx(1.0).margin(tolerance<TestType>()));
  CHECK(static_cast<double>(tf::cpp::mean_edge_length(value)) ==
        Catch::Approx(expected_mean).margin(tolerance<TestType>()));
  CHECK(static_cast<double>(tf::cpp::min_edge_length(value)) ==
        Catch::Approx(1.0).margin(tolerance<TestType>()));
  CHECK(static_cast<double>(tf::cpp::max_edge_length(value)) ==
        Catch::Approx(std::sqrt(13.0)).margin(tolerance<TestType>()));
}

TEMPLATE_TEST_CASE("mesh measurements apply stored affine transformations",
                   "[cpp][geometry][measurements]", float, double) {
  auto owned = tetrahedron<TestType>();
  owned.place(scaled_and_translated<TestType>());
  const auto value = owned.mesh();

  const auto expected_mean = (2.0 + 6.0 + 12.0 + std::sqrt(40.0) +
                              std::sqrt(148.0) + std::sqrt(180.0)) /
                             6.0;
  CHECK(static_cast<double>(tf::cpp::area(value)) ==
        Catch::Approx(54.0 + 6.0 * std::sqrt(41.0))
            .margin(tolerance<TestType>()));
  CHECK(static_cast<double>(tf::cpp::signed_volume(value)) ==
        Catch::Approx(24.0).margin(tolerance<TestType>()));
  CHECK(static_cast<double>(tf::cpp::mean_edge_length(value)) ==
        Catch::Approx(expected_mean).margin(tolerance<TestType>()));
  CHECK(static_cast<double>(tf::cpp::min_edge_length(value)) ==
        Catch::Approx(2.0).margin(tolerance<TestType>()));
  CHECK(static_cast<double>(tf::cpp::max_edge_length(value)) ==
        Catch::Approx(std::sqrt(180.0)).margin(tolerance<TestType>()));
}

TEMPLATE_TEST_CASE("measurements link through native archive instantiations",
                   "[cpp][geometry][measurements][archive-link]", float,
                   double) {
  const auto owned = tetrahedron<TestType>();
  const auto value = owned.mesh();
  using carrier = tf::cpp::mesh<tf::cpp::default_index_t, TestType, 3>;
  TestType (*const area_function)(const carrier &) =
      &tf::cpp::area<tf::cpp::default_index_t, TestType, 3, 3>;
  TestType (*const signed_volume_function)(const carrier &) =
      &tf::cpp::signed_volume<tf::cpp::default_index_t, TestType, 3, 3>;
  TestType (*const mean_edge_length_function)(const carrier &) =
      &tf::cpp::mean_edge_length<tf::cpp::default_index_t, TestType, 3, 3>;
  TestType (*const min_edge_length_function)(const carrier &) =
      &tf::cpp::min_edge_length<tf::cpp::default_index_t, TestType, 3, 3>;
  TestType (*const max_edge_length_function)(const carrier &) =
      &tf::cpp::max_edge_length<tf::cpp::default_index_t, TestType, 3, 3>;

  CHECK(area_function(value) > TestType{0});
  CHECK(signed_volume_function(value) > TestType{0});
  CHECK(mean_edge_length_function(value) > TestType{0});
  CHECK(min_edge_length_function(value) > TestType{0});
  CHECK(max_edge_length_function(value) > TestType{0});
}

// A default-assembled carrier is the EMPTY mesh, and every measurement
// answers it rather than refusing it.
TEMPLATE_TEST_CASE("measurements answer the empty mesh",
                   "[cpp][geometry][measurements][empty]", float, double) {
  const measured_owned<TestType> owned;
  const auto value = owned.mesh();
  CHECK(tf::cpp::area(value) == TestType{0});
  CHECK(tf::cpp::signed_volume(value) == TestType{0});
  CHECK(tf::cpp::mean_edge_length(value) == TestType{0});
  CHECK(tf::cpp::min_edge_length(value) == TestType{0});
  CHECK(tf::cpp::max_edge_length(value) == TestType{0});
}

TEMPLATE_TEST_CASE("async measurements answer as the synchronous entries do",
                   "[cpp][geometry][measurements][async]", float, double) {
  const auto owned = tetrahedron<TestType>();
  const auto value = owned.mesh();
  const auto expected_area = tf::cpp::area(value);
  const auto expected_volume = tf::cpp::signed_volume(value);
  const auto expected_mean = tf::cpp::mean_edge_length(value);
  const auto expected_min = tf::cpp::min_edge_length(value);
  const auto expected_max = tf::cpp::max_edge_length(value);

  auto area = tf::cpp::async::area(value);
  auto volume = tf::cpp::async::signed_volume(value);
  auto mean = tf::cpp::async::mean_edge_length(value);
  auto minimum = tf::cpp::async::min_edge_length(value);
  auto maximum = tf::cpp::async::max_edge_length(value);
  static_assert(std::is_same_v<decltype(area), std::future<TestType>>);
  static_assert(std::is_same_v<decltype(volume), std::future<TestType>>);
  static_assert(std::is_same_v<decltype(mean), std::future<TestType>>);
  static_assert(std::is_same_v<decltype(minimum), std::future<TestType>>);
  static_assert(std::is_same_v<decltype(maximum), std::future<TestType>>);
  CHECK(area.get() == Catch::Approx(expected_area));
  CHECK(volume.get() == Catch::Approx(expected_volume));
  CHECK(mean.get() == Catch::Approx(expected_mean));
  CHECK(minimum.get() == Catch::Approx(expected_min));
  CHECK(maximum.get() == Catch::Approx(expected_max));

  const auto submissions = std::make_shared<std::atomic<int>>(0);
  auto custom =
      tf::cpp::async::signed_volume(counting_resolver{submissions}, value);
  CHECK(submissions->load(std::memory_order_relaxed) == 1);
  CHECK(custom.get() == Catch::Approx(expected_volume));

  const measured_owned<TestType> empty;
  CHECK(tf::cpp::async::max_edge_length(empty.mesh()).get() == TestType{0});
}

TEMPLATE_TEST_CASE(
    "runtime primitive area matches Python triangle and polygon fixtures",
    "[cpp][geometry][measurements][primitive][python-parity][area]", float,
    double) {
  using result_type = tf::cpp::area_result<TestType>;
  static_assert(std::is_same_v<
                decltype(tf::cpp::area(
                    std::declval<const tf::cpp::primitive<TestType, 2> &>())),
                result_type>);
  static_assert(std::is_same_v<
                decltype(tf::cpp::area(
                    std::declval<const tf::cpp::primitive<TestType, 3> &>())),
                result_type>);

  const auto triangle_3d = make_primitive<TestType, 3>(
      tf::cpp::primitive_kind::triangle,
      {TestType{0}, TestType{0}, TestType{0}, TestType{2}, TestType{0},
       TestType{0}, TestType{0}, TestType{2}, TestType{0}},
      {3, 3});
  const auto triangle_3d_area = tf::cpp::area(triangle_3d);
  REQUIRE(triangle_3d_area.is_scalar());
  CHECK(triangle_3d_area.scalar() ==
        Catch::Approx(2.0).margin(tolerance<TestType>()));

  const auto unit_right_3d = make_primitive<TestType, 3>(
      tf::cpp::primitive_kind::triangle,
      {TestType{0}, TestType{0}, TestType{0}, TestType{1}, TestType{0},
       TestType{0}, TestType{0}, TestType{1}, TestType{0}},
      {3, 3});
  CHECK(tf::cpp::area(unit_right_3d).scalar() ==
        Catch::Approx(0.5).margin(tolerance<TestType>()));

  const auto triangle_2d =
      make_primitive<TestType, 2>(tf::cpp::primitive_kind::triangle,
                                  {TestType{0}, TestType{0}, TestType{1},
                                   TestType{0}, TestType{0}, TestType{1}},
                                  {3, 2});
  const auto triangle_2d_area = tf::cpp::area(triangle_2d);
  REQUIRE(triangle_2d_area.is_scalar());
  CHECK(triangle_2d_area.scalar() ==
        Catch::Approx(0.5).margin(tolerance<TestType>()));

  const auto triangles_3d = make_primitive<TestType, 3>(
      tf::cpp::primitive_kind::triangle,
      {TestType{0}, TestType{0}, TestType{0}, TestType{1}, TestType{0},
       TestType{0}, TestType{0}, TestType{1}, TestType{0}, TestType{0},
       TestType{0}, TestType{0}, TestType{2}, TestType{0}, TestType{0},
       TestType{0}, TestType{2}, TestType{0}, TestType{0}, TestType{0},
       TestType{0}, TestType{3}, TestType{0}, TestType{0}, TestType{0},
       TestType{3}, TestType{0}},
      {3, 3, 3});
  const auto triangle_areas_3d = tf::cpp::area(triangles_3d);
  REQUIRE(triangle_areas_3d.is_batch());
  check_array(triangle_areas_3d.batch(), {3}, {0.5, 2.0, 4.5},
              tolerance<TestType>());

  const auto triangles_2d = make_primitive<TestType, 2>(
      tf::cpp::primitive_kind::triangle,
      {TestType{0}, TestType{0}, TestType{1}, TestType{0}, TestType{0},
       TestType{1}, TestType{0}, TestType{0}, TestType{2}, TestType{0},
       TestType{0}, TestType{2}},
      {2, 3, 2});
  const auto triangle_areas_2d = tf::cpp::area(triangles_2d);
  REQUIRE(triangle_areas_2d.is_batch());
  check_array(triangle_areas_2d.batch(), {2}, {0.5, 2.0},
              tolerance<TestType>());

  const auto polygon_3d = make_primitive<TestType, 3>(
      tf::cpp::primitive_kind::polygon,
      {TestType{0}, TestType{0}, TestType{0}, TestType{1}, TestType{0},
       TestType{0}, TestType{1}, TestType{1}, TestType{0}, TestType{0},
       TestType{1}, TestType{0}},
      {4, 3});
  const auto polygon_3d_area = tf::cpp::area(polygon_3d);
  REQUIRE(polygon_3d_area.is_scalar());
  CHECK(polygon_3d_area.scalar() ==
        Catch::Approx(1.0).margin(tolerance<TestType>()));

  const auto polygon_2d = make_primitive<TestType, 2>(
      tf::cpp::primitive_kind::polygon,
      {TestType{0}, TestType{0}, TestType{1}, TestType{0}, TestType{1},
       TestType{1}, TestType{0}, TestType{1}},
      {4, 2});
  const auto polygon_2d_area = tf::cpp::area(polygon_2d);
  REQUIRE(polygon_2d_area.is_scalar());
  CHECK(polygon_2d_area.scalar() ==
        Catch::Approx(1.0).margin(tolerance<TestType>()));

  const auto rectangle_2d = make_primitive<TestType, 2>(
      tf::cpp::primitive_kind::polygon,
      {TestType{0}, TestType{0}, TestType{2}, TestType{0}, TestType{2},
       TestType{3}, TestType{0}, TestType{3}},
      {4, 2});
  const auto rectangle_2d_area = tf::cpp::area(rectangle_2d);
  REQUIRE(rectangle_2d_area.is_scalar());
  CHECK(rectangle_2d_area.scalar() ==
        Catch::Approx(6.0).margin(tolerance<TestType>()));

  const auto polygons_3d = make_primitive<TestType, 3>(
      tf::cpp::primitive_kind::polygon,
      {TestType{0}, TestType{0}, TestType{0}, TestType{1}, TestType{0},
       TestType{0}, TestType{1}, TestType{1}, TestType{0}, TestType{0},
       TestType{1}, TestType{0}, TestType{0}, TestType{0}, TestType{0},
       TestType{2}, TestType{0}, TestType{0}, TestType{2}, TestType{2},
       TestType{0}, TestType{0}, TestType{2}, TestType{0}},
      {2, 4, 3});
  const auto polygon_areas_3d = tf::cpp::area(polygons_3d);
  REQUIRE(polygon_areas_3d.is_batch());
  check_array(polygon_areas_3d.batch(), {2}, {1.0, 4.0}, tolerance<TestType>());

  const auto polygons_2d = make_primitive<TestType, 2>(
      tf::cpp::primitive_kind::polygon,
      {TestType{0}, TestType{0}, TestType{1}, TestType{0}, TestType{1},
       TestType{1}, TestType{0}, TestType{1}, TestType{0}, TestType{0},
       TestType{3}, TestType{0}, TestType{3}, TestType{2}, TestType{0},
       TestType{2}},
      {2, 4, 2});
  const auto polygon_areas_2d = tf::cpp::area(polygons_2d);
  REQUIRE(polygon_areas_2d.is_batch());
  check_array(polygon_areas_2d.batch(), {2}, {1.0, 6.0}, tolerance<TestType>());

  const auto root_three = static_cast<TestType>(std::sqrt(3.0));
  const auto regular_hexagon_3d = make_primitive<TestType, 3>(
      tf::cpp::primitive_kind::polygon,
      {TestType{2}, TestType{0}, TestType{0}, TestType{1}, root_three,
       TestType{0}, TestType{-1}, root_three, TestType{0}, TestType{-2},
       TestType{0}, TestType{0}, TestType{-1}, -root_three, TestType{0},
       TestType{1}, -root_three, TestType{0}},
      {6, 3});
  const auto regular_hexagon_area = tf::cpp::area(regular_hexagon_3d);
  REQUIRE(regular_hexagon_area.is_scalar());
  CHECK(regular_hexagon_area.scalar() ==
        Catch::Approx(6.0 * std::sqrt(3.0)).margin(tolerance<TestType>()));

  const auto concave_l_3d = make_primitive<TestType, 3>(
      tf::cpp::primitive_kind::polygon,
      {TestType{0}, TestType{0}, TestType{0}, TestType{2}, TestType{0},
       TestType{0}, TestType{2}, TestType{1}, TestType{0}, TestType{1},
       TestType{1}, TestType{0}, TestType{1}, TestType{2}, TestType{0},
       TestType{0}, TestType{2}, TestType{0}},
      {6, 3});
  constexpr double manual_fan_area = 1.0 + 0.5 + 0.5 + 1.0;
  static_assert(manual_fan_area == 3.0);
  const auto concave_l_area = tf::cpp::area(concave_l_3d);
  REQUIRE(concave_l_area.is_scalar());
  CHECK(concave_l_area.scalar() ==
        Catch::Approx(manual_fan_area).margin(tolerance<TestType>()));
}

TEMPLATE_TEST_CASE(
    "runtime primitive mean edge length matches Python fixtures in 2D and 3D",
    "[cpp][geometry][measurements][primitive][python-parity][mean-edge]", float,
    double) {
  static_assert(std::is_same_v<
                decltype(tf::cpp::mean_edge_length(
                    std::declval<const tf::cpp::primitive<TestType, 2> &>())),
                TestType>);
  static_assert(std::is_same_v<
                decltype(tf::cpp::mean_edge_length(
                    std::declval<const tf::cpp::primitive<TestType, 3> &>())),
                TestType>);

  const auto expected_triangle = (2.0 + std::sqrt(2.0)) / 3.0;
  const auto expected_triangle_batch = (6.0 + 3.0 * std::sqrt(2.0)) / 6.0;

  const auto triangle_3d = make_primitive<TestType, 3>(
      tf::cpp::primitive_kind::triangle,
      {TestType{0}, TestType{0}, TestType{0}, TestType{1}, TestType{0},
       TestType{0}, TestType{0}, TestType{1}, TestType{0}},
      {3, 3});
  CHECK(tf::cpp::mean_edge_length(triangle_3d) ==
        Catch::Approx(expected_triangle).margin(tolerance<TestType>()));

  const auto equilateral = make_primitive<TestType, 3>(
      tf::cpp::primitive_kind::triangle,
      {TestType{0}, TestType{0}, TestType{0}, TestType{2}, TestType{0},
       TestType{0}, TestType{1}, static_cast<TestType>(std::sqrt(3.0)),
       TestType{0}},
      {3, 3});
  CHECK(tf::cpp::mean_edge_length(equilateral) ==
        Catch::Approx(2.0).margin(tolerance<TestType>()));

  const auto triangle_2d =
      make_primitive<TestType, 2>(tf::cpp::primitive_kind::triangle,
                                  {TestType{0}, TestType{0}, TestType{1},
                                   TestType{0}, TestType{0}, TestType{1}},
                                  {3, 2});
  CHECK(tf::cpp::mean_edge_length(triangle_2d) ==
        Catch::Approx(expected_triangle).margin(tolerance<TestType>()));

  const auto triangles_3d = make_primitive<TestType, 3>(
      tf::cpp::primitive_kind::triangle,
      {TestType{0}, TestType{0}, TestType{0}, TestType{1}, TestType{0},
       TestType{0}, TestType{0}, TestType{1}, TestType{0}, TestType{0},
       TestType{0}, TestType{0}, TestType{2}, TestType{0}, TestType{0},
       TestType{0}, TestType{2}, TestType{0}},
      {2, 3, 3});
  CHECK(tf::cpp::mean_edge_length(triangles_3d) ==
        Catch::Approx(expected_triangle_batch).margin(tolerance<TestType>()));

  const auto triangles_2d = make_primitive<TestType, 2>(
      tf::cpp::primitive_kind::triangle,
      {TestType{0}, TestType{0}, TestType{1}, TestType{0}, TestType{0},
       TestType{1}, TestType{0}, TestType{0}, TestType{2}, TestType{0},
       TestType{0}, TestType{2}},
      {2, 3, 2});
  CHECK(tf::cpp::mean_edge_length(triangles_2d) ==
        Catch::Approx(expected_triangle_batch).margin(tolerance<TestType>()));

  const auto polygon_3d = make_primitive<TestType, 3>(
      tf::cpp::primitive_kind::polygon,
      {TestType{0}, TestType{0}, TestType{0}, TestType{1}, TestType{0},
       TestType{0}, TestType{1}, TestType{1}, TestType{0}, TestType{0},
       TestType{1}, TestType{0}},
      {4, 3});
  CHECK(tf::cpp::mean_edge_length(polygon_3d) ==
        Catch::Approx(1.0).margin(tolerance<TestType>()));

  const auto polygon_2d = make_primitive<TestType, 2>(
      tf::cpp::primitive_kind::polygon,
      {TestType{0}, TestType{0}, TestType{1}, TestType{0}, TestType{1},
       TestType{1}, TestType{0}, TestType{1}},
      {4, 2});
  CHECK(tf::cpp::mean_edge_length(polygon_2d) ==
        Catch::Approx(1.0).margin(tolerance<TestType>()));

  const auto polygons_3d = make_primitive<TestType, 3>(
      tf::cpp::primitive_kind::polygon,
      {TestType{0}, TestType{0}, TestType{0}, TestType{1}, TestType{0},
       TestType{0}, TestType{1}, TestType{1}, TestType{0}, TestType{0},
       TestType{1}, TestType{0}, TestType{0}, TestType{0}, TestType{0},
       TestType{2}, TestType{0}, TestType{0}, TestType{2}, TestType{2},
       TestType{0}, TestType{0}, TestType{2}, TestType{0}},
      {2, 4, 3});
  CHECK(tf::cpp::mean_edge_length(polygons_3d) ==
        Catch::Approx(1.5).margin(tolerance<TestType>()));

  const auto polygons_2d = make_primitive<TestType, 2>(
      tf::cpp::primitive_kind::polygon,
      {TestType{0}, TestType{0}, TestType{1}, TestType{0}, TestType{1},
       TestType{1}, TestType{0}, TestType{1}, TestType{0}, TestType{0},
       TestType{2}, TestType{0}, TestType{2}, TestType{2}, TestType{0},
       TestType{2}},
      {2, 4, 2});
  CHECK(tf::cpp::mean_edge_length(polygons_2d) ==
        Catch::Approx(1.5).margin(tolerance<TestType>()));
}

TEMPLATE_TEST_CASE(
    "runtime primitive normals match Python shapes values and directions",
    "[cpp][geometry][normals][primitive][python-parity]", float, double) {
  static_assert(std::is_same_v<
                decltype(tf::cpp::normals(
                    std::declval<const tf::cpp::primitive<TestType, 3> &>())),
                tf::cpp::nd_array<TestType>>);

  const auto triangle = make_primitive<TestType, 3>(
      tf::cpp::primitive_kind::triangle,
      {TestType{0}, TestType{0}, TestType{0}, TestType{1}, TestType{0},
       TestType{0}, TestType{0}, TestType{1}, TestType{0}},
      {3, 3});
  check_array(tf::cpp::normals(triangle), {3}, {0.0, 0.0, 1.0},
              tolerance<TestType>());

  const auto translated_triangle = make_primitive<TestType, 3>(
      tf::cpp::primitive_kind::triangle,
      {TestType{1}, TestType{2}, TestType{3}, TestType{4}, TestType{5},
       TestType{3}, TestType{2}, TestType{7}, TestType{3}},
      {3, 3});
  const auto translated_normal = tf::cpp::normals(translated_triangle);
  check_array(translated_normal, {3}, {0.0, 0.0, 1.0}, tolerance<TestType>());
  const auto translated_length = std::sqrt(
      static_cast<double>(translated_normal[0] * translated_normal[0] +
                          translated_normal[1] * translated_normal[1] +
                          translated_normal[2] * translated_normal[2]));
  CHECK(translated_length == Catch::Approx(1.0).margin(tolerance<TestType>()));

  const auto triangles = make_primitive<TestType, 3>(
      tf::cpp::primitive_kind::triangle,
      {TestType{0}, TestType{0}, TestType{0}, TestType{1}, TestType{0},
       TestType{0}, TestType{0}, TestType{1}, TestType{0}, TestType{0},
       TestType{0}, TestType{0}, TestType{0}, TestType{1}, TestType{0},
       TestType{0}, TestType{0}, TestType{1}, TestType{0}, TestType{0},
       TestType{0}, TestType{0}, TestType{0}, TestType{1}, TestType{1},
       TestType{0}, TestType{0}},
      {3, 3, 3});
  const auto triangle_normals = tf::cpp::normals(triangles);
  check_array(triangle_normals, {3, 3},
              {0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0},
              tolerance<TestType>());
  const auto one_triangle_batch = make_primitive<TestType, 3>(
      tf::cpp::primitive_kind::triangle,
      {TestType{0}, TestType{0}, TestType{0}, TestType{1}, TestType{0},
       TestType{0}, TestType{0}, TestType{1}, TestType{0}},
      {1, 3, 3});
  check_array(tf::cpp::normals(one_triangle_batch), {1, 3}, {0.0, 0.0, 1.0},
              tolerance<TestType>());
  for (int normal = 0; normal < triangle_normals.shape_at(0); ++normal) {
    const auto offset = static_cast<std::size_t>(3 * normal);
    const auto length = std::sqrt(static_cast<double>(
        triangle_normals[offset] * triangle_normals[offset] +
        triangle_normals[offset + 1] * triangle_normals[offset + 1] +
        triangle_normals[offset + 2] * triangle_normals[offset + 2]));
    CHECK(length == Catch::Approx(1.0).margin(tolerance<TestType>()));
  }

  const auto polygon = make_primitive<TestType, 3>(
      tf::cpp::primitive_kind::polygon,
      {TestType{0}, TestType{0}, TestType{0}, TestType{1}, TestType{0},
       TestType{0}, TestType{1}, TestType{1}, TestType{0}, TestType{0},
       TestType{1}, TestType{0}},
      {4, 3});
  check_array(tf::cpp::normals(polygon), {3}, {0.0, 0.0, 1.0},
              tolerance<TestType>());

  const auto translated_polygon = make_primitive<TestType, 3>(
      tf::cpp::primitive_kind::polygon,
      {TestType{1}, TestType{2}, TestType{3}, TestType{4}, TestType{5},
       TestType{3}, TestType{2}, TestType{7}, TestType{3}, TestType{0},
       TestType{4}, TestType{3}},
      {4, 3});
  const auto translated_polygon_normal = tf::cpp::normals(translated_polygon);
  check_array(translated_polygon_normal, {3}, {0.0, 0.0, 1.0},
              tolerance<TestType>());

  const auto polygons = make_primitive<TestType, 3>(
      tf::cpp::primitive_kind::polygon,
      {TestType{0}, TestType{0}, TestType{0}, TestType{1}, TestType{0},
       TestType{0}, TestType{1}, TestType{1}, TestType{0}, TestType{0},
       TestType{1}, TestType{0}, TestType{0}, TestType{0}, TestType{0},
       TestType{0}, TestType{1}, TestType{0}, TestType{0}, TestType{1},
       TestType{1}, TestType{0}, TestType{0}, TestType{1}},
      {2, 4, 3});
  const auto polygon_normals = tf::cpp::normals(polygons);
  check_array(polygon_normals, {2, 3}, {0.0, 0.0, 1.0, 1.0, 0.0, 0.0},
              tolerance<TestType>());
  for (int normal = 0; normal < polygon_normals.shape_at(0); ++normal) {
    const auto offset = static_cast<std::size_t>(3 * normal);
    const auto length = std::sqrt(static_cast<double>(
        polygon_normals[offset] * polygon_normals[offset] +
        polygon_normals[offset + 1] * polygon_normals[offset + 1] +
        polygon_normals[offset + 2] * polygon_normals[offset + 2]));
    CHECK(length == Catch::Approx(1.0).margin(tolerance<TestType>()));
  }
}

TEMPLATE_TEST_CASE(
    "runtime primitive geometry preserves empty batches and degeneracy",
    "[cpp][geometry][primitive][empty][degenerate]", float, double) {
  const auto empty_triangles_2d = tf::cpp::primitive<TestType, 2>(
      tf::cpp::primitive_kind::triangle, make_empty<TestType>({0, 3, 2}));
  const auto empty_triangle_areas = tf::cpp::area(empty_triangles_2d);
  REQUIRE(empty_triangle_areas.is_batch());
  check_array(empty_triangle_areas.batch(), {0}, {}, tolerance<TestType>());
  CHECK(std::isnan(
      static_cast<double>(tf::cpp::mean_edge_length(empty_triangles_2d))));

  const auto empty_triangles_3d = tf::cpp::primitive<TestType, 3>(
      tf::cpp::primitive_kind::triangle, make_empty<TestType>({0, 3, 3}));
  check_array(tf::cpp::normals(empty_triangles_3d), {0, 3}, {},
              tolerance<TestType>());

  const auto empty_polygons_3d = tf::cpp::primitive<TestType, 3>(
      tf::cpp::primitive_kind::polygon, make_empty<TestType>({0, 4, 3}));
  const auto empty_polygon_areas = tf::cpp::area(empty_polygons_3d);
  REQUIRE(empty_polygon_areas.is_batch());
  check_array(empty_polygon_areas.batch(), {0}, {}, tolerance<TestType>());
  CHECK(std::isnan(
      static_cast<double>(tf::cpp::mean_edge_length(empty_polygons_3d))));
  check_array(tf::cpp::normals(empty_polygons_3d), {0, 3}, {},
              tolerance<TestType>());

  const auto degenerate_triangle_2d =
      make_primitive<TestType, 2>(tf::cpp::primitive_kind::triangle,
                                  {TestType{0}, TestType{0}, TestType{1},
                                   TestType{0}, TestType{2}, TestType{0}},
                                  {3, 2});
  CHECK(tf::cpp::area(degenerate_triangle_2d).scalar() == TestType{0});

  const auto degenerate_triangle = make_primitive<TestType, 3>(
      tf::cpp::primitive_kind::triangle,
      {TestType{0}, TestType{0}, TestType{0}, TestType{1}, TestType{0},
       TestType{0}, TestType{2}, TestType{0}, TestType{0}},
      {3, 3});
  CHECK(tf::cpp::area(degenerate_triangle).scalar() == TestType{0});
  check_array(tf::cpp::normals(degenerate_triangle), {3}, {0.0, 0.0, 0.0},
              tolerance<TestType>());

  const auto degenerate_polygons = make_primitive<TestType, 3>(
      tf::cpp::primitive_kind::polygon,
      {TestType{0}, TestType{0}, TestType{0}, TestType{1}, TestType{0},
       TestType{0}, TestType{2}, TestType{0}, TestType{0}, TestType{3},
       TestType{0}, TestType{0}, TestType{0}, TestType{0}, TestType{0},
       TestType{0}, TestType{0}, TestType{0}, TestType{0}, TestType{0},
       TestType{0}, TestType{0}, TestType{0}, TestType{0}},
      {2, 4, 3});
  const auto degenerate_areas = tf::cpp::area(degenerate_polygons);
  REQUIRE(degenerate_areas.is_batch());
  check_array(degenerate_areas.batch(), {2}, {0.0, 0.0}, tolerance<TestType>());
  check_array(tf::cpp::normals(degenerate_polygons), {2, 3},
              {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, tolerance<TestType>());
}

TEMPLATE_TEST_CASE(
    "area result validates state and access",
    "[cpp][geometry][measurements][primitive][result][validation]", float,
    double) {
  tf::cpp::area_result<TestType> scalar(TestType{2});
  REQUIRE(scalar.is_scalar());
  CHECK_FALSE(scalar.is_batch());
  CHECK(scalar.scalar() == TestType{2});
  CHECK_THROWS_AS(scalar.batch(), std::logic_error);

  auto source = make_array<TestType>({TestType{1}, TestType{4}}, {2});
  const auto *storage = source.raw_owner().get();
  tf::cpp::area_result<TestType> batch(source);
  source.destroy();
  REQUIRE(batch.is_batch());
  CHECK_FALSE(batch.is_scalar());
  CHECK_THROWS_AS(batch.scalar(), std::logic_error);
  const auto values = batch.batch();
  REQUIRE(values.is_valid());
  CHECK(values.raw_owner().get() == storage);
  check_array(values, {2}, {1.0, 4.0}, tolerance<TestType>());

  CHECK_THROWS_AS(tf::cpp::area_result<TestType>(tf::cpp::nd_array<TestType>{}),
                  std::invalid_argument);
  CHECK_THROWS_AS(
      tf::cpp::area_result<TestType>(make_array<TestType>(
          {TestType{1}, TestType{2}, TestType{3}, TestType{4}}, {2, 2})),
      std::invalid_argument);
}

TEMPLATE_TEST_CASE(
    "runtime primitive geometry async APIs preserve exact types and ownership",
    "[cpp][geometry][primitive][async][ownership]", float, double) {
  using area_type = tf::cpp::area_result<TestType>;
  using array_type = tf::cpp::nd_array<TestType>;

  const auto triangle = make_primitive<TestType, 3>(
      tf::cpp::primitive_kind::triangle,
      {TestType{0}, TestType{0}, TestType{0}, TestType{1}, TestType{0},
       TestType{0}, TestType{0}, TestType{1}, TestType{0}},
      {3, 3});
  auto area_future = tf::cpp::async::area(triangle);
  auto mean_future = tf::cpp::async::mean_edge_length(triangle);
  auto normals_future = tf::cpp::async::normals(triangle);
  static_assert(std::is_same_v<decltype(area_future), std::future<area_type>>);
  static_assert(std::is_same_v<decltype(mean_future), std::future<TestType>>);
  static_assert(
      std::is_same_v<decltype(normals_future), std::future<array_type>>);
  static_assert(std::is_same_v<
                decltype(tf::cpp::async::area(
                    std::declval<const tf::cpp::primitive<TestType, 2> &>())),
                std::future<area_type>>);
  static_assert(std::is_same_v<
                decltype(tf::cpp::async::mean_edge_length(
                    std::declval<const tf::cpp::primitive<TestType, 2> &>())),
                std::future<TestType>>);

  const auto async_area = area_future.get();
  REQUIRE(async_area.is_scalar());
  CHECK(async_area.scalar() ==
        Catch::Approx(0.5).margin(tolerance<TestType>()));
  CHECK(mean_future.get() == Catch::Approx((2.0 + std::sqrt(2.0)) / 3.0)
                                 .margin(tolerance<TestType>()));
  check_array(normals_future.get(), {3}, {0.0, 0.0, 1.0},
              tolerance<TestType>());

  const auto triangle_batch = make_primitive<TestType, 3>(
      tf::cpp::primitive_kind::triangle,
      {TestType{0}, TestType{0}, TestType{0}, TestType{1}, TestType{0},
       TestType{0}, TestType{0}, TestType{1}, TestType{0}, TestType{0},
       TestType{0}, TestType{0}, TestType{2}, TestType{0}, TestType{0},
       TestType{0}, TestType{2}, TestType{0}},
      {2, 3, 3});
  auto batch_area_future = tf::cpp::async::area(triangle_batch);
  auto batch_mean_future = tf::cpp::async::mean_edge_length(triangle_batch);
  auto batch_normals_future = tf::cpp::async::normals(triangle_batch);
  const auto batch_area = batch_area_future.get();
  REQUIRE(batch_area.is_batch());
  check_array(batch_area.batch(), {2}, {0.5, 2.0}, tolerance<TestType>());
  CHECK(batch_mean_future.get() ==
        Catch::Approx((6.0 + 3.0 * std::sqrt(2.0)) / 6.0)
            .margin(tolerance<TestType>()));
  check_array(batch_normals_future.get(), {2, 3},
              {0.0, 0.0, 1.0, 0.0, 0.0, 1.0}, tolerance<TestType>());

  const auto submissions = std::make_shared<std::atomic<int>>(0);
  auto custom_area =
      tf::cpp::async::area(counting_resolver{submissions}, triangle);
  auto custom_mean = tf::cpp::async::mean_edge_length(
      counting_resolver{submissions}, triangle);
  auto custom_normals =
      tf::cpp::async::normals(counting_resolver{submissions}, triangle);
  static_assert(std::is_same_v<decltype(custom_area), std::future<area_type>>);
  static_assert(std::is_same_v<decltype(custom_mean), std::future<TestType>>);
  static_assert(
      std::is_same_v<decltype(custom_normals), std::future<array_type>>);
  CHECK(submissions->load(std::memory_order_relaxed) == 3);
  CHECK(custom_area.get().scalar() ==
        Catch::Approx(0.5).margin(tolerance<TestType>()));
  CHECK(custom_mean.get() == Catch::Approx((2.0 + std::sqrt(2.0)) / 3.0)
                                 .margin(tolerance<TestType>()));
  check_array(custom_normals.get(), {3}, {0.0, 0.0, 1.0},
              tolerance<TestType>());

  auto [owned_area, owned_mean, owned_normals] = [] {
    auto source = make_array<TestType>({TestType{0}, TestType{0}, TestType{0},
                                        TestType{2}, TestType{0}, TestType{0},
                                        TestType{0}, TestType{2}, TestType{0}},
                                       {3, 3});
    const auto value = tf::cpp::primitive<TestType, 3>(
        tf::cpp::primitive_kind::triangle, source);
    auto area = tf::cpp::async::area(value);
    auto mean = tf::cpp::async::mean_edge_length(value);
    auto normals = tf::cpp::async::normals(value);
    source.destroy();
    return std::make_tuple(std::move(area), std::move(mean),
                           std::move(normals));
  }();
  CHECK(owned_area.get().scalar() ==
        Catch::Approx(2.0).margin(tolerance<TestType>()));
  CHECK(owned_mean.get() == Catch::Approx(2.0 * (2.0 + std::sqrt(2.0)) / 3.0)
                                .margin(tolerance<TestType>()));
  check_array(owned_normals.get(), {3}, {0.0, 0.0, 1.0}, tolerance<TestType>());
}

TEMPLATE_TEST_CASE("runtime primitive geometry reports unsupported kinds "
                   "synchronously and asynchronously",
                   "[cpp][geometry][primitive][validation][async]", float,
                   double) {
  const auto unsupported_2d = make_primitive<TestType, 2>(
      tf::cpp::primitive_kind::point, {TestType{0}, TestType{0}}, {2});
  const auto unsupported_3d =
      make_primitive<TestType, 3>(tf::cpp::primitive_kind::point,
                                  {TestType{0}, TestType{0}, TestType{0}}, {3});

  CHECK_THROWS_AS(tf::cpp::area(unsupported_2d), std::invalid_argument);
  CHECK_THROWS_AS(tf::cpp::mean_edge_length(unsupported_2d),
                  std::invalid_argument);
  CHECK_THROWS_AS(tf::cpp::area(unsupported_3d), std::invalid_argument);
  CHECK_THROWS_AS(tf::cpp::mean_edge_length(unsupported_3d),
                  std::invalid_argument);
  CHECK_THROWS_AS(tf::cpp::normals(unsupported_3d), std::invalid_argument);

  auto area_failure_2d = tf::cpp::async::area(unsupported_2d);
  auto mean_failure_2d = tf::cpp::async::mean_edge_length(unsupported_2d);
  auto area_failure_3d = tf::cpp::async::area(unsupported_3d);
  auto mean_failure_3d = tf::cpp::async::mean_edge_length(unsupported_3d);
  auto normals_failure = tf::cpp::async::normals(unsupported_3d);
  CHECK_THROWS_AS(area_failure_2d.get(), std::invalid_argument);
  CHECK_THROWS_AS(mean_failure_2d.get(), std::invalid_argument);
  CHECK_THROWS_AS(area_failure_3d.get(), std::invalid_argument);
  CHECK_THROWS_AS(mean_failure_3d.get(), std::invalid_argument);
  CHECK_THROWS_AS(normals_failure.get(), std::invalid_argument);
}

TEMPLATE_TEST_CASE(
    "runtime primitive geometry links through native archive instantiations",
    "[cpp][geometry][primitive][archive-link]", float, double) {
  using primitive_2d = tf::cpp::primitive<TestType, 2>;
  using primitive_3d = tf::cpp::primitive<TestType, 3>;
  using area_function_2d =
      tf::cpp::area_result<TestType> (*)(const primitive_2d &);
  using area_function_3d =
      tf::cpp::area_result<TestType> (*)(const primitive_3d &);
  using mean_function_2d = TestType (*)(const primitive_2d &);
  using mean_function_3d = TestType (*)(const primitive_3d &);
  using normals_function =
      tf::cpp::nd_array<TestType> (*)(const primitive_3d &);

  const area_function_2d area_2d = &tf::cpp::area<TestType, 2>;
  const area_function_3d area_3d = &tf::cpp::area<TestType, 3>;
  const mean_function_2d mean_2d = &tf::cpp::mean_edge_length<TestType, 2>;
  const mean_function_3d mean_3d = &tf::cpp::mean_edge_length<TestType, 3>;
  const normals_function primitive_normals = &tf::cpp::normals<TestType>;

  const auto triangle_2d =
      make_primitive<TestType, 2>(tf::cpp::primitive_kind::triangle,
                                  {TestType{0}, TestType{0}, TestType{1},
                                   TestType{0}, TestType{0}, TestType{1}},
                                  {3, 2});
  const auto triangle_3d = make_primitive<TestType, 3>(
      tf::cpp::primitive_kind::triangle,
      {TestType{0}, TestType{0}, TestType{0}, TestType{1}, TestType{0},
       TestType{0}, TestType{0}, TestType{1}, TestType{0}},
      {3, 3});

  CHECK(area_2d(triangle_2d).scalar() ==
        Catch::Approx(0.5).margin(tolerance<TestType>()));
  CHECK(area_3d(triangle_3d).scalar() ==
        Catch::Approx(0.5).margin(tolerance<TestType>()));
  CHECK(mean_2d(triangle_2d) == Catch::Approx((2.0 + std::sqrt(2.0)) / 3.0)
                                    .margin(tolerance<TestType>()));
  CHECK(mean_3d(triangle_3d) == Catch::Approx((2.0 + std::sqrt(2.0)) / 3.0)
                                    .margin(tolerance<TestType>()));
  check_array(primitive_normals(triangle_3d), {3}, {0.0, 0.0, 1.0},
              tolerance<TestType>());
}
