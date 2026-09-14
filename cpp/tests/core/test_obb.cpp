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
#include "trueform/cpp/core/async/obb.hpp"
#include "trueform/cpp/core/obb.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <future>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace {

template <typename Real> auto make_box_points() -> tf::cpp::nd_array<Real> {
  constexpr auto inverse_sqrt_two = 0.7071067811865475244;
  const std::array<double, 3> origin{1.0, 2.0, 3.0};
  const std::array<std::array<double, 3>, 3> edges{{
      {{6.0 * inverse_sqrt_two, 6.0 * inverse_sqrt_two, 0.0}},
      {{-4.0 * inverse_sqrt_two, 4.0 * inverse_sqrt_two, 0.0}},
      {{0.0, 0.0, 2.0}},
  }};

  tf::buffer<Real> buffer;
  buffer.allocate(8 * 3);
  for (std::size_t corner = 0; corner < 8; ++corner) {
    for (std::size_t coordinate = 0; coordinate < 3; ++coordinate) {
      auto value = origin[coordinate];
      for (std::size_t axis = 0; axis < 3; ++axis)
        if ((corner & (std::size_t{1} << axis)) != 0)
          value += edges[axis][coordinate];
      buffer[corner * 3 + coordinate] = static_cast<Real>(value);
    }
  }
  return tf::cpp::nd_array<Real>::from_buffer(std::move(buffer), {8, 3});
}

template <typename Real>
auto dot_axis(const tf::cpp::obb_result<Real> &result, int first, int second)
    -> Real {
  auto value = Real{0};
  for (int coordinate = 0; coordinate < 3; ++coordinate)
    value += result.axes[static_cast<std::size_t>(first * 3 + coordinate)] *
             result.axes[static_cast<std::size_t>(second * 3 + coordinate)];
  return value;
}

} // namespace

TEMPLATE_TEST_CASE("OBB returns typed arrays with canonical shapes",
                   "[cpp][core][obb]", float, double) {
  const auto points = make_box_points<TestType>();
  const auto result =
      tf::cpp::obb_from(tf::cpp::obb_options<TestType>{points.shallow_copy()});

  static_assert(
      std::is_same_v<decltype(result.origin), tf::cpp::nd_array<TestType>>);
  static_assert(
      std::is_same_v<decltype(result.axes), tf::cpp::nd_array<TestType>>);
  static_assert(
      std::is_same_v<decltype(result.extent), tf::cpp::nd_array<TestType>>);

  CHECK((result.origin.raw_shape() == tf::small_vector<int, 3>{3}));
  CHECK((result.axes.raw_shape() == tf::small_vector<int, 3>{3, 3}));
  CHECK((result.extent.raw_shape() == tf::small_vector<int, 3>{3}));

  const auto tolerance = std::is_same_v<TestType, float> ? 1e-4 : 1e-10;
  for (int first = 0; first < 3; ++first)
    for (int second = 0; second < 3; ++second)
      CHECK(static_cast<double>(dot_axis(result, first, second)) ==
            Catch::Approx(first == second ? 1.0 : 0.0).margin(tolerance));

  std::array<double, 3> extents{
      static_cast<double>(result.extent[0]),
      static_cast<double>(result.extent[1]),
      static_cast<double>(result.extent[2]),
  };
  std::sort(extents.begin(), extents.end());
  CHECK(extents[0] == Catch::Approx(2.0).margin(tolerance));
  CHECK(extents[1] == Catch::Approx(4.0).margin(tolerance));
  CHECK(extents[2] == Catch::Approx(6.0).margin(tolerance));

  for (int axis = 0; axis < 3; ++axis) {
    auto minimum = std::numeric_limits<double>::infinity();
    auto maximum = -std::numeric_limits<double>::infinity();
    for (int point = 0; point < points.shape_at(0); ++point) {
      auto projection = 0.0;
      for (int coordinate = 0; coordinate < 3; ++coordinate) {
        const auto delta =
            static_cast<double>(
                points[static_cast<std::size_t>(point * 3 + coordinate)]) -
            static_cast<double>(
                result.origin[static_cast<std::size_t>(coordinate)]);
        projection +=
            delta *
            static_cast<double>(
                result.axes[static_cast<std::size_t>(axis * 3 + coordinate)]);
      }
      minimum = std::min(minimum, projection);
      maximum = std::max(maximum, projection);
    }
    CHECK(minimum == Catch::Approx(0.0).margin(tolerance));
    CHECK(maximum ==
          Catch::Approx(static_cast<double>(
                            result.extent[static_cast<std::size_t>(axis)]))
              .margin(tolerance));
  }
}

TEMPLATE_TEST_CASE("OBB preserves empty input behavior", "[cpp][core][obb]",
                   float, double) {
  tf::buffer<TestType> buffer;
  buffer.allocate(0);
  const auto result = tf::cpp::obb_from(tf::cpp::obb_options<TestType>{
      tf::cpp::nd_array<TestType>::from_buffer(std::move(buffer), {0, 3})});

  for (int coordinate = 0; coordinate < 3; ++coordinate) {
    CHECK(result.origin[static_cast<std::size_t>(coordinate)] == TestType{0});
    CHECK(result.extent[static_cast<std::size_t>(coordinate)] == TestType{0});
    for (int axis = 0; axis < 3; ++axis)
      CHECK(result.axes[static_cast<std::size_t>(axis * 3 + coordinate)] ==
            (axis == coordinate ? TestType{1} : TestType{0}));
  }
}

TEST_CASE("OBB rejects invalid point layouts", "[cpp][core][obb]") {
  tf::buffer<float> buffer;
  buffer.allocate(4);
  auto points =
      tf::cpp::nd_array<float>::from_buffer(std::move(buffer), {2, 2});
  CHECK_THROWS_AS(
      tf::cpp::obb_from(tf::cpp::obb_options<float>{std::move(points)}),
      std::invalid_argument);
}

TEST_CASE("async OBB retains options and preserves typed native results",
          "[cpp][core][async][obb]") {
  auto points = make_box_points<double>();
  auto result = tf::cpp::async::obb_from(
      tf::cpp::obb_options<double>{points.shallow_copy()});
  static_assert(std::is_same_v<decltype(result),
                               std::future<tf::cpp::obb_result<double>>>);
  points.destroy();
  const auto box = result.get();
  CHECK((box.origin.raw_shape() == tf::small_vector<int, 3>{3}));
  CHECK((box.axes.raw_shape() == tf::small_vector<int, 3>{3, 3}));

  const auto custom_points = make_box_points<float>();
  auto custom_pending = tf::cpp::async::obb_from(
      tf::cpp::async::future_resolver{},
      tf::cpp::obb_options<float>{custom_points.shallow_copy()});
  static_assert(std::is_same_v<decltype(custom_pending),
                               std::future<tf::cpp::obb_result<float>>>);
  const auto custom = custom_pending.get();
  CHECK((custom.extent.raw_shape() == tf::small_vector<int, 3>{3}));

  tf::buffer<float> buffer;
  buffer.allocate(4);
  const auto invalid =
      tf::cpp::nd_array<float>::from_buffer(std::move(buffer), {2, 2});
  auto failure = tf::cpp::async::obb_from(
      tf::cpp::obb_options<float>{invalid.shallow_copy()});
  CHECK_THROWS_AS(failure.get(), std::invalid_argument);
}
