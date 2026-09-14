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

#include "trueform/cpp/geometry/async/chamfer_error.hpp"
#include "trueform/cpp/geometry/async/fit_icp.hpp"
#include "trueform/cpp/geometry/async/fit_knn.hpp"
#include "trueform/cpp/geometry/async/fit_obb.hpp"
#include "trueform/cpp/geometry/async/fit_rigid.hpp"
#include "trueform/cpp/geometry/async/symmetric_chamfer_error.hpp"
#include "trueform/cpp/geometry/chamfer_error.hpp"
#include "trueform/cpp/geometry/chamfer_error_options.hpp"
#include "trueform/cpp/geometry/fit_icp.hpp"
#include "trueform/cpp/geometry/fit_knn.hpp"
#include "trueform/cpp/geometry/fit_obb.hpp"
#include "trueform/cpp/geometry/fit_rigid.hpp"
#include "trueform/cpp/geometry/symmetric_chamfer_error.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <future>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

struct counting_resolver {
  int *count;

  template <typename T>
  using state_type = tf::cpp::async::detail::future_state<T>;

  template <typename T>
  auto make_state() const -> std::shared_ptr<state_type<T>> {
    ++*count;
    return std::make_shared<state_type<T>>();
  }
};

template <typename Real> struct tree_observing_resolver {
  const tf::cpp::point_cloud_cache<Real, 3> *target;
  bool *tree_was_built;

  template <typename T>
  using state_type = tf::cpp::async::detail::future_state<T>;

  template <typename T>
  auto make_state() const -> std::shared_ptr<state_type<T>> {
    *tree_was_built = target->is_tree_built();
    return std::make_shared<state_type<T>>();
  }
};

template <typename Real>
auto make_points(const std::vector<std::array<double, 3>> &values)
    -> tf::cpp::nd_array<Real> {
  tf::buffer<Real> buffer;
  buffer.allocate(values.size() * 3);
  for (std::size_t point = 0; point < values.size(); ++point)
    for (std::size_t coordinate = 0; coordinate < 3; ++coordinate)
      buffer[point * 3 + coordinate] =
          static_cast<Real>(values[point][coordinate]);
  return tf::cpp::nd_array<Real>::from_buffer(
      std::move(buffer), {static_cast<int>(values.size()), 3});
}

template <typename Real> auto sample_points() -> tf::cpp::nd_array<Real> {
  std::vector<std::array<double, 3>> values;
  values.reserve(32);
  for (int index = 0; index < 32; ++index) {
    const auto x = static_cast<double>((index * 7) % 17) / 17.0;
    const auto y = static_cast<double>((index * 11 + 3) % 19) / 19.0;
    const auto z = static_cast<double>((index * 13 + 5) % 23) / 23.0;
    values.push_back({x + 0.07 * y, y + 0.03 * z, z + 0.05 * x});
  }
  return make_points<Real>(values);
}

template <typename Real> auto box_points() -> tf::cpp::nd_array<Real> {
  std::vector<std::array<double, 3>> values;
  values.reserve(8);
  for (int x = 0; x < 2; ++x)
    for (int y = 0; y < 2; ++y)
      for (int z = 0; z < 2; ++z)
        values.push_back(
            {x == 0 ? -2.0 : 2.0, y == 0 ? -1.0 : 1.0, z == 0 ? -0.5 : 0.5});
  return make_points<Real>(values);
}

template <typename Real>
auto repeated_points(int count, double x) -> tf::cpp::nd_array<Real> {
  return make_points<Real>(std::vector<std::array<double, 3>>(
      static_cast<std::size_t>(count), {x, 0.0, 0.0}));
}

template <typename Real>
auto translated(const tf::cpp::nd_array<Real> &points, double x, double y,
                double z) -> tf::cpp::nd_array<Real> {
  auto result = points.deep_copy();
  for (int point = 0; point < result.shape_at(0); ++point) {
    const auto offset = static_cast<std::size_t>(point * 3);
    result[offset] += static_cast<Real>(x);
    result[offset + 1] += static_cast<Real>(y);
    result[offset + 2] += static_cast<Real>(z);
  }
  return result;
}

template <typename Real>
auto radial_normals(const tf::cpp::nd_array<Real> &points)
    -> tf::cpp::nd_array<Real> {
  auto result = points.deep_copy();
  for (int point = 0; point < result.shape_at(0); ++point) {
    const auto offset = static_cast<std::size_t>(point * 3);
    const auto x = static_cast<double>(result[offset]);
    const auto y = static_cast<double>(result[offset + 1]);
    const auto z = static_cast<double>(result[offset + 2]);
    const auto length = std::sqrt(x * x + y * y + z * z);
    result[offset] = static_cast<Real>(x / length);
    result[offset + 1] = static_cast<Real>(y / length);
    result[offset + 2] = static_cast<Real>(z / length);
  }
  return result;
}

template <typename Real>
auto translation(double x, double y = 0.0, double z = 0.0)
    -> tf::cpp::nd_array<Real> {
  tf::buffer<Real> buffer;
  buffer.allocate(16);
  std::fill(buffer.begin(), buffer.end(), Real{0});
  buffer[0] = Real{1};
  buffer[5] = Real{1};
  buffer[10] = Real{1};
  buffer[15] = Real{1};
  buffer[3] = static_cast<Real>(x);
  buffer[7] = static_cast<Real>(y);
  buffer[11] = static_cast<Real>(z);
  return tf::cpp::nd_array<Real>::from_buffer(std::move(buffer), {4, 4});
}

template <typename Real, std::size_t Dims>
using owned_cloud = tf::cpp::test::owned_point_cloud<Real, Dims>;

/// A test holds its points, its cache and its placement, and `point_cloud()`
/// is the assembly every entry takes.
template <typename Real, std::size_t Dims>
auto dimensional_cloud(const tf::cpp::nd_array<Real> &points)
    -> owned_cloud<Real, Dims> {
  owned_cloud<Real, Dims> owned;
  auto &storage = owned.points.data_buffer();
  storage.allocate(points.length());
  std::copy(points.begin(), points.end(), storage.begin());
  return owned;
}

template <typename Real>
auto cloud(const tf::cpp::nd_array<Real> &points) -> owned_cloud<Real, 3> {
  return dimensional_cloud<Real, 3>(points);
}

template <typename Real, std::size_t Dims>
auto points_array(const owned_cloud<Real, Dims> &owned)
    -> tf::cpp::nd_array<Real> {
  const auto &storage = owned.points.data_buffer();
  return tf::cpp::nd_array<Real>::from_borrowed(
      {}, const_cast<Real *>(storage.data()), storage.size(),
      {static_cast<int>(storage.size() / Dims), static_cast<int>(Dims)});
}

/// A test writes through its own storage and then says what moved.
template <typename Real, std::size_t Dims>
auto restate_points(owned_cloud<Real, Dims> &owned,
                    const tf::cpp::nd_array<Real> &values) -> void {
  auto &storage = owned.points.data_buffer();
  storage.allocate(values.length());
  std::copy(values.begin(), values.end(), storage.begin());
  owned.cache.points_changed();
}

template <typename Real, std::size_t Dims>
auto state_normals(owned_cloud<Real, Dims> &owned,
                   const tf::cpp::nd_array<Real> &values) -> void {
  auto &storage = owned.normals.data_buffer();
  storage.allocate(values.length());
  std::copy(values.begin(), values.end(), storage.begin());
}

template <typename Real, std::size_t Dims>
auto place_from(owned_cloud<Real, Dims> &owned,
                const tf::cpp::nd_array<Real> &matrix) -> void {
  std::array<Real, (Dims + 1) * (Dims + 1)> placement{};
  std::copy(matrix.begin(), matrix.end(), placement.begin());
  owned.place(placement);
}

template <typename Real>
auto has_matrix_shape(const tf::cpp::nd_array<Real> &matrix) -> bool {
  return matrix.raw_shape() == tf::small_vector<int, 3>{4, 4};
}

template <typename Real>
auto matrix_is_finite(const tf::cpp::nd_array<Real> &matrix) -> bool {
  return std::all_of(matrix.begin(), matrix.end(),
                     [](Real value) { return std::isfinite(value); });
}

template <typename Real> auto tolerance() -> double {
  return std::is_same_v<Real, float> ? 2e-3 : 1e-8;
}

template <typename Real>
auto check_translation(const tf::cpp::nd_array<Real> &matrix, double x,
                       double y, double z) -> void {
  REQUIRE(has_matrix_shape(matrix));
  CHECK(matrix[3] == Catch::Approx(x).margin(tolerance<Real>()));
  CHECK(matrix[7] == Catch::Approx(y).margin(tolerance<Real>()));
  CHECK(matrix[11] == Catch::Approx(z).margin(tolerance<Real>()));
  CHECK(matrix[12] == Real{0});
  CHECK(matrix[13] == Real{0});
  CHECK(matrix[14] == Real{0});
  CHECK(matrix[15] == Real{1});
}

template <typename Real, std::size_t Dims>
auto make_dimensional_points(
    const std::vector<std::array<double, Dims>> &values)
    -> tf::cpp::nd_array<Real> {
  tf::buffer<Real> buffer;
  buffer.allocate(values.size() * Dims);
  for (std::size_t point = 0; point < values.size(); ++point)
    for (std::size_t coordinate = 0; coordinate < Dims; ++coordinate)
      buffer[point * Dims + coordinate] =
          static_cast<Real>(values[point][coordinate]);
  return tf::cpp::nd_array<Real>::from_buffer(
      std::move(buffer),
      {static_cast<int>(values.size()), static_cast<int>(Dims)});
}

template <typename Real, std::size_t Dims>
auto deterministic_points(std::size_t count) -> tf::cpp::nd_array<Real> {
  std::vector<std::array<double, Dims>> values(count);
  for (std::size_t index = 0; index < count; ++index) {
    const auto i = static_cast<int>(index);
    values[index][0] = (static_cast<double>((i * 37 + 11) % 101) / 50.0 - 1.0) +
                       0.03 * static_cast<double>((i * 19) % 17) / 17.0;
    values[index][1] = (static_cast<double>((i * 53 + 17) % 103) / 51.0 - 1.0) +
                       0.02 * static_cast<double>((i * 23) % 13) / 13.0;
    if constexpr (Dims == 3)
      values[index][2] =
          (static_cast<double>((i * 61 + 29) % 107) / 53.0 - 1.0) +
          0.01 * static_cast<double>((i * 31) % 11) / 11.0;
  }
  return make_dimensional_points<Real, Dims>(values);
}

template <typename Real, std::size_t Dims>
auto repeat_each_point(const tf::cpp::nd_array<Real> &points, int repeats)
    -> tf::cpp::nd_array<Real> {
  tf::buffer<Real> buffer;
  buffer.allocate(static_cast<std::size_t>(points.shape_at(0)) *
                  static_cast<std::size_t>(repeats) * Dims);
  auto output = buffer.begin();
  for (int point = 0; point < points.shape_at(0); ++point)
    for (int repeat = 0; repeat < repeats; ++repeat)
      for (std::size_t coordinate = 0; coordinate < Dims; ++coordinate)
        *output++ = points[static_cast<std::size_t>(point) * Dims + coordinate];
  return tf::cpp::nd_array<Real>::from_buffer(
      std::move(buffer),
      {points.shape_at(0) * repeats, static_cast<int>(Dims)});
}

template <typename Real, std::size_t Dims>
auto translate_dimensional(const tf::cpp::nd_array<Real> &points,
                           const std::array<double, Dims> &offset)
    -> tf::cpp::nd_array<Real> {
  auto result = points.deep_copy();
  for (int point = 0; point < result.shape_at(0); ++point)
    for (std::size_t coordinate = 0; coordinate < Dims; ++coordinate)
      result[static_cast<std::size_t>(point) * Dims + coordinate] +=
          static_cast<Real>(offset[coordinate]);
  return result;
}

template <typename Real, std::size_t Dims>
auto rigid_transform(double angle, const std::array<double, Dims> &offset)
    -> tf::cpp::nd_array<Real> {
  constexpr auto side = Dims + 1;
  tf::buffer<Real> buffer;
  buffer.allocate(side * side);
  std::fill(buffer.begin(), buffer.end(), Real{0});
  for (std::size_t coordinate = 0; coordinate < side; ++coordinate)
    buffer[coordinate * side + coordinate] = Real{1};
  const auto cosine = static_cast<Real>(std::cos(angle));
  const auto sine = static_cast<Real>(std::sin(angle));
  buffer[0] = cosine;
  buffer[1] = -sine;
  buffer[side] = sine;
  buffer[side + 1] = cosine;
  for (std::size_t coordinate = 0; coordinate < Dims; ++coordinate)
    buffer[coordinate * side + Dims] = static_cast<Real>(offset[coordinate]);
  return tf::cpp::nd_array<Real>::from_buffer(
      std::move(buffer), {static_cast<int>(side), static_cast<int>(side)});
}

template <typename Real, std::size_t Dims>
auto apply_transform(const tf::cpp::nd_array<Real> &points,
                     const tf::cpp::nd_array<Real> &matrix)
    -> tf::cpp::nd_array<Real> {
  constexpr auto side = Dims + 1;
  tf::buffer<Real> buffer;
  buffer.allocate(static_cast<std::size_t>(points.shape_at(0)) * Dims);
  for (int point = 0; point < points.shape_at(0); ++point) {
    for (std::size_t row = 0; row < Dims; ++row) {
      auto value = matrix[row * side + Dims];
      for (std::size_t column = 0; column < Dims; ++column)
        value += matrix[row * side + column] *
                 points[static_cast<std::size_t>(point) * Dims + column];
      buffer[static_cast<std::size_t>(point) * Dims + row] = value;
    }
  }
  return tf::cpp::nd_array<Real>::from_buffer(
      std::move(buffer), {points.shape_at(0), static_cast<int>(Dims)});
}

template <typename Real, std::size_t Dims>
auto has_transform_shape(const tf::cpp::nd_array<Real> &matrix) -> bool {
  const auto side = static_cast<int>(Dims + 1);
  return matrix.raw_shape() == tf::small_vector<int, 3>{side, side};
}

template <typename Real, std::size_t Dims>
auto check_identity(const tf::cpp::nd_array<Real> &matrix, double margin = 1e-4)
    -> void {
  constexpr auto side = Dims + 1;
  REQUIRE(has_transform_shape<Real, Dims>(matrix));
  for (std::size_t row = 0; row < side; ++row)
    for (std::size_t column = 0; column < side; ++column)
      CHECK(matrix[row * side + column] ==
            Catch::Approx(row == column ? 1.0 : 0.0).margin(margin));
}

template <typename Real, std::size_t Dims>
auto check_rigid_transform(const tf::cpp::nd_array<Real> &matrix) -> void {
  constexpr auto side = Dims + 1;
  REQUIRE(has_transform_shape<Real, Dims>(matrix));
  REQUIRE(matrix_is_finite(matrix));
  for (std::size_t row = 0; row < Dims; ++row) {
    for (std::size_t other = 0; other < Dims; ++other) {
      double dot = 0;
      for (std::size_t column = 0; column < Dims; ++column)
        dot += static_cast<double>(matrix[row * side + column]) *
               static_cast<double>(matrix[other * side + column]);
      CHECK(dot == Catch::Approx(row == other ? 1.0 : 0.0).margin(2e-4));
    }
  }
  double determinant;
  if constexpr (Dims == 2) {
    determinant = static_cast<double>(matrix[0] * matrix[side + 1] -
                                      matrix[1] * matrix[side]);
  } else {
    determinant =
        static_cast<double>(matrix[0]) *
            (static_cast<double>(matrix[side + 1] * matrix[2 * side + 2] -
                                 matrix[side + 2] * matrix[2 * side + 1])) -
        static_cast<double>(matrix[1]) *
            (static_cast<double>(matrix[side] * matrix[2 * side + 2] -
                                 matrix[side + 2] * matrix[2 * side])) +
        static_cast<double>(matrix[2]) *
            (static_cast<double>(matrix[side] * matrix[2 * side + 1] -
                                 matrix[side + 1] * matrix[2 * side]));
  }
  CHECK(determinant == Catch::Approx(1.0).margin(2e-4));
  for (std::size_t column = 0; column < Dims; ++column)
    CHECK(matrix[Dims * side + column] == Real{0});
  CHECK(matrix[side * side - 1] == Real{1});
}

template <typename Real, std::size_t Dims>
auto unit_normals(std::size_t count) -> tf::cpp::nd_array<Real> {
  std::vector<std::array<double, Dims>> values(count);
  for (auto &value : values) {
    value.fill(0.0);
    value[Dims - 1] = 1.0;
  }
  return make_dimensional_points<Real, Dims>(values);
}

} // namespace

TEMPLATE_TEST_CASE(
    "fit_knn matches Python identity translation rotation and option behavior",
    "[cpp][geometry][registration][knn][python-parity]", float, double) {
  const auto check_dimension = [&](auto dimension) {
    constexpr std::size_t Dims = decltype(dimension)::value;
    INFO("Dims = " << Dims);
    auto points = deterministic_points<TestType, Dims>(100);

    auto identity_source =
        dimensional_cloud<TestType, Dims>(points.deep_copy());
    auto identity_target =
        dimensional_cloud<TestType, Dims>(points.deep_copy());
    auto identity = tf::cpp::fit_knn(identity_source.point_cloud(),
                                     identity_target.point_cloud());
    static_assert(
        std::is_same_v<decltype(identity), tf::cpp::nd_array<TestType>>);
    check_identity<TestType, Dims>(identity);

    std::array<double, Dims> offset{};
    offset[0] = 0.05;
    offset[1] = 0.04;
    if constexpr (Dims == 3)
      offset[2] = -0.03;
    auto translated_points = translate_dimensional(points, offset);
    auto translation_source =
        dimensional_cloud<TestType, Dims>(points.deep_copy());
    auto translation_target =
        dimensional_cloud<TestType, Dims>(translated_points.deep_copy());
    const auto error_before = tf::cpp::chamfer_error(
        translation_source.point_cloud(), translation_target.point_cloud());
    const auto translation_step = tf::cpp::fit_knn(
        translation_source.point_cloud(), translation_target.point_cloud());
    auto moved = dimensional_cloud<TestType, Dims>(
        apply_transform<TestType, Dims>(points, translation_step));
    const auto error_after = tf::cpp::chamfer_error(
        moved.point_cloud(), translation_target.point_cloud());
    CHECK(error_after < error_before);
    check_rigid_transform<TestType, Dims>(translation_step);

    std::array<double, Dims> no_offset{};
    const auto rotation = rigid_transform<TestType, Dims>(0.08, no_offset);
    auto rotated_points = apply_transform<TestType, Dims>(points, rotation);
    auto rotation_source =
        dimensional_cloud<TestType, Dims>(points.deep_copy());
    auto rotation_target =
        dimensional_cloud<TestType, Dims>(rotated_points.deep_copy());
    const auto rotation_error_before = tf::cpp::chamfer_error(
        rotation_source.point_cloud(), rotation_target.point_cloud());
    const auto rotation_step = tf::cpp::fit_knn(rotation_source.point_cloud(),
                                                rotation_target.point_cloud());
    auto rotated = dimensional_cloud<TestType, Dims>(
        apply_transform<TestType, Dims>(points, rotation_step));
    CHECK(tf::cpp::chamfer_error(rotated.point_cloud(),
                                 rotation_target.point_cloud()) <
          rotation_error_before);
    check_rigid_transform<TestType, Dims>(rotation_step);

    for (const int k : std::array<int, 4>{1, 3, 5, 10}) {
      tf::cpp::fit_knn_options<TestType> options;
      options.k = k;
      const auto result =
          tf::cpp::fit_knn(translation_source.point_cloud(),
                           translation_target.point_cloud(), options);
      check_rigid_transform<TestType, Dims>(result);
    }

    tf::cpp::fit_knn_options<TestType> bounded_ten;
    bounded_ten.k = 10;
    auto bounded_above_ten = bounded_ten;
    bounded_above_ten.k = 11;
    const auto ten_result =
        tf::cpp::fit_knn(translation_source.point_cloud(),
                         translation_target.point_cloud(), bounded_ten);
    const auto eleven_result =
        tf::cpp::fit_knn(translation_source.point_cloud(),
                         translation_target.point_cloud(), bounded_above_ten);
    REQUIRE(ten_result.length() == eleven_result.length());
    for (std::size_t index = 0; index < ten_result.length(); ++index)
      CHECK(eleven_result[index] ==
            Catch::Approx(ten_result[index]).margin(tolerance<TestType>()));

    tf::cpp::fit_knn_options<TestType> adaptive;
    adaptive.k = 5;
    adaptive.sigma = TestType{-1};
    const auto adaptive_result =
        tf::cpp::fit_knn(translation_source.point_cloud(),
                         translation_target.point_cloud(), adaptive);
    check_rigid_transform<TestType, Dims>(adaptive_result);

    auto explicit_sigma = adaptive;
    explicit_sigma.sigma = TestType{0.2};
    const auto explicit_result =
        tf::cpp::fit_knn(translation_source.point_cloud(),
                         translation_target.point_cloud(), explicit_sigma);
    check_rigid_transform<TestType, Dims>(explicit_result);
    CHECK_FALSE(std::equal(adaptive_result.begin(), adaptive_result.end(),
                           explicit_result.begin()));

    auto tiny_sigma = adaptive;
    tiny_sigma.sigma = std::sqrt(std::numeric_limits<TestType>::min());
    const auto tiny_sigma_result =
        tf::cpp::fit_knn(translation_source.point_cloud(),
                         translation_target.point_cloud(), tiny_sigma);
    check_rigid_transform<TestType, Dims>(tiny_sigma_result);
    tiny_sigma.outlier_proportion = TestType{0.2};
    const auto tiny_sigma_trimmed =
        tf::cpp::fit_knn(translation_source.point_cloud(),
                         translation_target.point_cloud(), tiny_sigma);
    check_rigid_transform<TestType, Dims>(tiny_sigma_trimmed);

    const auto duplicate_points = deterministic_points<TestType, Dims>(20);
    auto duplicate_source =
        dimensional_cloud<TestType, Dims>(duplicate_points.deep_copy());
    auto duplicate_target = dimensional_cloud<TestType, Dims>(
        repeat_each_point<TestType, Dims>(duplicate_points, 5));
    tf::cpp::fit_knn_options<TestType> zero_width;
    zero_width.k = 5;
    zero_width.sigma = TestType{0};
    const auto zero_width_result =
        tf::cpp::fit_knn(duplicate_source.point_cloud(),
                         duplicate_target.point_cloud(), zero_width);
    const auto adaptive_zero_result = tf::cpp::fit_knn(
        duplicate_source.point_cloud(), duplicate_target.point_cloud(),
        tf::cpp::fit_knn_options<TestType>{5, TestType{-1}, TestType{0}});
    check_rigid_transform<TestType, Dims>(zero_width_result);
    check_rigid_transform<TestType, Dims>(adaptive_zero_result);

    auto trimmed = adaptive;
    trimmed.outlier_proportion = TestType{0.2};
    const auto trimmed_result =
        tf::cpp::fit_knn(translation_source.point_cloud(),
                         translation_target.point_cloud(), trimmed);
    check_rigid_transform<TestType, Dims>(trimmed_result);
  };

  check_dimension(std::integral_constant<std::size_t, 2>{});
  check_dimension(std::integral_constant<std::size_t, 3>{});
}

TEMPLATE_TEST_CASE(
    "rigid ICP and OBB match Python across dimensional carrier axes",
    "[cpp][geometry][registration][python-parity][rigid][icp][obb]", float,
    double) {
  const auto check_dimension = [&](auto dimension) {
    constexpr std::size_t Dims = decltype(dimension)::value;
    INFO("Dims = " << Dims);
    auto points = deterministic_points<TestType, Dims>(160);
    std::array<double, Dims> offset{};
    offset[0] = 0.04;
    offset[1] = -0.03;
    if constexpr (Dims == 3)
      offset[2] = 0.02;
    const auto expected_transform =
        rigid_transform<TestType, Dims>(0.08, offset);
    auto target_points =
        apply_transform<TestType, Dims>(points, expected_transform);

    auto identity_source =
        dimensional_cloud<TestType, Dims>(points.deep_copy());
    auto identity_target =
        dimensional_cloud<TestType, Dims>(points.deep_copy());
    check_identity<TestType, Dims>(tf::cpp::fit_rigid(
        identity_source.point_cloud(), identity_target.point_cloud()));
    check_identity<TestType, Dims>(tf::cpp::fit_icp(
        identity_source.point_cloud(), identity_target.point_cloud()));
    check_identity<TestType, Dims>(tf::cpp::fit_obb(
        identity_source.point_cloud(), identity_target.point_cloud()));

    auto rigid_source = dimensional_cloud<TestType, Dims>(points.deep_copy());
    auto rigid_target =
        dimensional_cloud<TestType, Dims>(target_points.deep_copy());
    const auto rigid = tf::cpp::fit_rigid(rigid_source.point_cloud(),
                                          rigid_target.point_cloud());
    check_rigid_transform<TestType, Dims>(rigid);
    const auto rigid_points = apply_transform<TestType, Dims>(points, rigid);
    REQUIRE(rigid_points.length() == target_points.length());
    for (std::size_t index = 0; index < rigid_points.length(); ++index)
      CHECK(rigid_points[index] ==
            Catch::Approx(target_points[index])
                .margin(TestType{20} * tolerance<TestType>()));

    auto unequal_target = dimensional_cloud<TestType, Dims>(
        deterministic_points<TestType, Dims>(159));
    CHECK_THROWS_AS(tf::cpp::fit_rigid(rigid_source.point_cloud(),
                                       unequal_target.point_cloud()),
                    std::invalid_argument);

    std::vector<std::array<double, Dims>> minimum_values(Dims + 1);
    for (auto &value : minimum_values)
      value.fill(0.0);
    for (std::size_t axis = 0; axis < Dims; ++axis)
      minimum_values[axis + 1][axis] = 1.0;
    auto minimum_points =
        make_dimensional_points<TestType, Dims>(minimum_values);
    auto minimum_target_points = translate_dimensional(minimum_points, offset);
    auto minimum_source =
        dimensional_cloud<TestType, Dims>(minimum_points.deep_copy());
    auto minimum_target =
        dimensional_cloud<TestType, Dims>(minimum_target_points.deep_copy());
    const auto minimum_rigid = tf::cpp::fit_rigid(minimum_source.point_cloud(),
                                                  minimum_target.point_cloud());
    const auto minimum_aligned =
        apply_transform<TestType, Dims>(minimum_points, minimum_rigid);
    for (std::size_t index = 0; index < minimum_aligned.length(); ++index)
      CHECK(minimum_aligned[index] ==
            Catch::Approx(minimum_target_points[index])
                .margin(TestType{20} * tolerance<TestType>()));

    auto icp_source = dimensional_cloud<TestType, Dims>(points.deep_copy());
    auto icp_target =
        dimensional_cloud<TestType, Dims>(target_points.deep_copy());
    const auto icp_error_before = tf::cpp::chamfer_error(
        icp_source.point_cloud(), icp_target.point_cloud());
    tf::cpp::fit_icp_options<TestType> icp_options;
    icp_options.max_iterations = 30;
    icp_options.n_samples = 0;
    icp_options.k = 1;
    icp_options.min_relative_improvement = TestType{0};
    const auto icp = tf::cpp::fit_icp(icp_source.point_cloud(),
                                      icp_target.point_cloud(), icp_options);
    CHECK(icp.raw_shape() ==
          tf::small_vector<int, 3>{static_cast<int>(Dims + 1),
                                   static_cast<int>(Dims + 1)});
    CHECK(matrix_is_finite(icp));
    auto icp_aligned = dimensional_cloud<TestType, Dims>(
        apply_transform<TestType, Dims>(points, icp));
    CHECK(tf::cpp::chamfer_error(icp_aligned.point_cloud(),
                                 icp_target.point_cloud()) < icp_error_before);

    auto fewer_icp_source = dimensional_cloud<TestType, Dims>(
        deterministic_points<TestType, Dims>(100));
    const auto unequal_icp = tf::cpp::fit_icp(
        fewer_icp_source.point_cloud(), icp_target.point_cloud(), icp_options);
    CHECK(unequal_icp.raw_shape() ==
          tf::small_vector<int, 3>{static_cast<int>(Dims + 1),
                                   static_cast<int>(Dims + 1)});
    CHECK(matrix_is_finite(unequal_icp));

    std::vector<std::array<double, Dims>> box_values;
    if constexpr (Dims == 2) {
      box_values = {{{-3.0, -1.0}}, {{3.0, -1.0}}, {{3.0, 1.0}}, {{-3.0, 1.0}}};
    } else {
      for (const double x : {-3.0, 3.0})
        for (const double y : {-1.0, 1.0})
          for (const double z : {-0.5, 0.5})
            box_values.push_back({x, y, z});
    }
    auto obb_points = make_dimensional_points<TestType, Dims>(box_values);
    auto obb_target_points =
        apply_transform<TestType, Dims>(obb_points, expected_transform);
    auto obb_source = dimensional_cloud<TestType, Dims>(obb_points.deep_copy());
    auto obb_target =
        dimensional_cloud<TestType, Dims>(obb_target_points.deep_copy());
    const auto obb_error_before = tf::cpp::chamfer_error(
        obb_source.point_cloud(), obb_target.point_cloud());
    for (const int sample_size : std::array<int, 3>{100, 0, 3}) {
      const auto obb =
          tf::cpp::fit_obb(obb_source.point_cloud(), obb_target.point_cloud(),
                           tf::cpp::fit_obb_options{sample_size});
      check_rigid_transform<TestType, Dims>(obb);
      auto obb_aligned = dimensional_cloud<TestType, Dims>(
          apply_transform<TestType, Dims>(obb_points, obb));
      CHECK(tf::cpp::chamfer_error(obb_aligned.point_cloud(),
                                   obb_target.point_cloud()) <
            obb_error_before);
    }
    auto unequal_obb_points =
        deterministic_points<TestType, Dims>(Dims == 2 ? 41 : 81);
    auto unequal_obb_target_points =
        apply_transform<TestType, Dims>(unequal_obb_points, expected_transform);
    auto unequal_obb_source =
        dimensional_cloud<TestType, Dims>(points.deep_copy());
    auto unequal_obb_target = dimensional_cloud<TestType, Dims>(
        unequal_obb_target_points.deep_copy());
    check_rigid_transform<TestType, Dims>(tf::cpp::fit_obb(
        unequal_obb_source.point_cloud(), unequal_obb_target.point_cloud()));

    auto framed_source = dimensional_cloud<TestType, Dims>(points.deep_copy());
    auto framed_target = dimensional_cloud<TestType, Dims>(points.deep_copy());
    std::array<double, Dims> source_frame{};
    std::array<double, Dims> target_frame{};
    source_frame[0] = 1.0;
    target_frame[0] = 3.0;
    place_from(framed_source,
               rigid_transform<TestType, Dims>(0.0, source_frame));
    place_from(framed_target,
               rigid_transform<TestType, Dims>(0.0, target_frame));
    const auto framed_rigid = tf::cpp::fit_rigid(framed_source.point_cloud(),
                                                 framed_target.point_cloud());
    const auto framed_obb = tf::cpp::fit_obb(framed_source.point_cloud(),
                                             framed_target.point_cloud());
    CHECK(framed_rigid[Dims] ==
          Catch::Approx(2.0).margin(TestType{20} * tolerance<TestType>()));
    CHECK(framed_obb[Dims] ==
          Catch::Approx(2.0).margin(TestType{20} * tolerance<TestType>()));

    if constexpr (Dims == 2) {
      auto normal_source =
          dimensional_cloud<TestType, Dims>(points.deep_copy());
      auto normal_target =
          dimensional_cloud<TestType, Dims>(points.deep_copy());
      state_normals(normal_target,
                    unit_normals<TestType, Dims>(
                        static_cast<std::size_t>(points.shape_at(0))));
      CHECK_THROWS_AS(tf::cpp::fit_rigid(normal_source.point_cloud(),
                                         normal_target.point_cloud()),
                      std::invalid_argument);
      CHECK_THROWS_AS(tf::cpp::fit_icp(normal_source.point_cloud(),
                                       normal_target.point_cloud()),
                      std::invalid_argument);
      check_rigid_transform<TestType, Dims>(tf::cpp::fit_obb(
          normal_source.point_cloud(), normal_target.point_cloud()));
    }
  };

  check_dimension(std::integral_constant<std::size_t, 2>{});
  check_dimension(std::integral_constant<std::size_t, 3>{});
}

TEMPLATE_TEST_CASE(
    "fit_knn handles repeated steps unequal sizes and world frames",
    "[cpp][geometry][registration][knn][python-parity][frames]", float,
    double) {
  const auto check_dimension = [&](auto dimension) {
    constexpr std::size_t Dims = decltype(dimension)::value;
    INFO("Dims = " << Dims);
    auto points = deterministic_points<TestType, Dims>(160);
    std::array<double, Dims> offset{};
    offset[0] = 0.2;
    offset[1] = 0.12;
    if constexpr (Dims == 3)
      offset[2] = 0.08;
    const auto true_transform = rigid_transform<TestType, Dims>(0.18, offset);
    auto target_points =
        apply_transform<TestType, Dims>(points, true_transform);
    auto target = dimensional_cloud<TestType, Dims>(target_points.deep_copy());
    auto current_points = points.deep_copy();
    auto current =
        dimensional_cloud<TestType, Dims>(current_points.deep_copy());
    const auto initial_error =
        tf::cpp::chamfer_error(current.point_cloud(), target.point_cloud());
    for (int iteration = 0; iteration < 10; ++iteration) {
      const auto step =
          tf::cpp::fit_knn(current.point_cloud(), target.point_cloud());
      current_points = apply_transform<TestType, Dims>(current_points, step);
      current = dimensional_cloud<TestType, Dims>(current_points.deep_copy());
    }
    const auto final_error =
        tf::cpp::chamfer_error(current.point_cloud(), target.point_cloud());
    CHECK(final_error < initial_error);
    CHECK(final_error < TestType{0.1});

    auto fewer = dimensional_cloud<TestType, Dims>(
        deterministic_points<TestType, Dims>(50));
    auto more_points = deterministic_points<TestType, Dims>(100);
    std::array<double, Dims> small_offset{};
    small_offset.fill(0.03);
    auto more = dimensional_cloud<TestType, Dims>(
        translate_dimensional(more_points, small_offset));
    const auto unequal =
        tf::cpp::fit_knn(fewer.point_cloud(), more.point_cloud());
    check_rigid_transform<TestType, Dims>(unequal);

    auto framed_source = dimensional_cloud<TestType, Dims>(points.deep_copy());
    auto framed_target = dimensional_cloud<TestType, Dims>(points.deep_copy());
    std::array<double, Dims> source_frame{};
    std::array<double, Dims> target_frame{};
    source_frame[0] = 0.01;
    source_frame[1] = -0.02;
    target_frame[0] = 0.04;
    target_frame[1] = 0.03;
    if constexpr (Dims == 3) {
      source_frame[2] = 0.02;
      target_frame[2] = -0.01;
    }
    place_from(framed_source,
               rigid_transform<TestType, Dims>(0.0, source_frame));
    place_from(framed_target,
               rigid_transform<TestType, Dims>(0.0, target_frame));
    const auto framed = tf::cpp::fit_knn(framed_source.point_cloud(),
                                         framed_target.point_cloud());
    check_rigid_transform<TestType, Dims>(framed);
    auto source_world_points = translate_dimensional(points, source_frame);
    auto target_world_points = translate_dimensional(points, target_frame);
    auto source_world =
        dimensional_cloud<TestType, Dims>(source_world_points.deep_copy());
    auto target_world =
        dimensional_cloud<TestType, Dims>(target_world_points.deep_copy());
    const auto framed_error_before = tf::cpp::chamfer_error(
        source_world.point_cloud(), target_world.point_cloud());
    auto aligned_world = dimensional_cloud<TestType, Dims>(
        apply_transform<TestType, Dims>(source_world_points, framed));
    CHECK(tf::cpp::chamfer_error(aligned_world.point_cloud(),
                                 target_world.point_cloud()) <
          framed_error_before);
  };

  check_dimension(std::integral_constant<std::size_t, 2>{});
  check_dimension(std::integral_constant<std::size_t, 3>{});
}

TEMPLATE_TEST_CASE("fit_knn supports 3D normal modes and rejects 2D normals",
                   "[cpp][geometry][registration][knn][normals]", float,
                   double) {
  auto points = deterministic_points<TestType, 3>(100);
  std::array<double, 3> offset{0.02, -0.01, 0.015};
  auto shifted = translate_dimensional(points, offset);

  auto plain_source_3d = dimensional_cloud<TestType, 3>(shifted.deep_copy());
  auto plain_target_3d = dimensional_cloud<TestType, 3>(points.deep_copy());
  const auto point_to_point = tf::cpp::fit_knn(plain_source_3d.point_cloud(),
                                               plain_target_3d.point_cloud());

  auto plane_source = dimensional_cloud<TestType, 3>(shifted.deep_copy());
  auto plane_target = dimensional_cloud<TestType, 3>(points.deep_copy());
  state_normals(plane_target, radial_normals(points_array(plane_target)));
  const auto point_to_plane_before = tf::cpp::chamfer_error(
      plane_source.point_cloud(), plane_target.point_cloud());
  const auto point_to_plane =
      tf::cpp::fit_knn(plane_source.point_cloud(), plane_target.point_cloud());
  check_rigid_transform<TestType, 3>(point_to_plane);

  tf::cpp::fit_knn_options<TestType> tiny_sigma;
  tiny_sigma.k = 5;
  tiny_sigma.sigma = std::sqrt(std::numeric_limits<TestType>::min());
  const auto tiny_sigma_plane = tf::cpp::fit_knn(
      plane_source.point_cloud(), plane_target.point_cloud(), tiny_sigma);
  check_rigid_transform<TestType, 3>(tiny_sigma_plane);
  tiny_sigma.outlier_proportion = TestType{0.2};
  const auto tiny_sigma_trimmed = tf::cpp::fit_knn(
      plane_source.point_cloud(), plane_target.point_cloud(), tiny_sigma);
  check_rigid_transform<TestType, 3>(tiny_sigma_trimmed);

  CHECK_FALSE(std::equal(point_to_point.begin(), point_to_point.end(),
                         point_to_plane.begin()));
  auto plane_aligned = dimensional_cloud<TestType, 3>(
      apply_transform<TestType, 3>(shifted, point_to_plane));
  CHECK(tf::cpp::chamfer_error(plane_aligned.point_cloud(),
                               plane_target.point_cloud()) <
        point_to_plane_before);

  auto weighted_source = dimensional_cloud<TestType, 3>(shifted.deep_copy());
  auto weighted_target = dimensional_cloud<TestType, 3>(points.deep_copy());
  state_normals(weighted_source, radial_normals(points));
  state_normals(weighted_target, radial_normals(points));
  const auto normal_weighted_before = tf::cpp::chamfer_error(
      weighted_source.point_cloud(), weighted_target.point_cloud());
  const auto normal_weighted = tf::cpp::fit_knn(weighted_source.point_cloud(),
                                                weighted_target.point_cloud());
  check_rigid_transform<TestType, 3>(normal_weighted);
  auto weighted_aligned = dimensional_cloud<TestType, 3>(
      apply_transform<TestType, 3>(shifted, normal_weighted));
  CHECK(tf::cpp::chamfer_error(weighted_aligned.point_cloud(),
                               weighted_target.point_cloud()) <
        normal_weighted_before);

  auto points_2d = deterministic_points<TestType, 2>(20);
  auto source_normals = dimensional_cloud<TestType, 2>(points_2d.deep_copy());
  auto plain_target = dimensional_cloud<TestType, 2>(points_2d.deep_copy());
  state_normals(source_normals, unit_normals<TestType, 2>(20));
  CHECK_THROWS_AS(tf::cpp::fit_knn(source_normals.point_cloud(),
                                   plain_target.point_cloud()),
                  std::invalid_argument);

  auto plain_source = dimensional_cloud<TestType, 2>(points_2d.deep_copy());
  auto target_normals = dimensional_cloud<TestType, 2>(points_2d.deep_copy());
  state_normals(target_normals, unit_normals<TestType, 2>(20));
  CHECK_THROWS_AS(tf::cpp::fit_knn(plain_source.point_cloud(),
                                   target_normals.point_cloud()),
                  std::invalid_argument);
}

TEMPLATE_TEST_CASE("fit_knn validates controls and empty point clouds",
                   "[cpp][geometry][registration][knn][validation]", float,
                   double) {
  const auto check_dimension = [&](auto dimension) {
    constexpr std::size_t Dims = decltype(dimension)::value;
    INFO("Dims = " << Dims);
    auto points = deterministic_points<TestType, Dims>(20);
    auto source = dimensional_cloud<TestType, Dims>(points.deep_copy());
    auto target = dimensional_cloud<TestType, Dims>(points.deep_copy());
    const auto nan = std::numeric_limits<TestType>::quiet_NaN();
    const auto infinity = std::numeric_limits<TestType>::infinity();

    tf::cpp::fit_knn_options<TestType> options;
    options.k = 0;
    CHECK_THROWS_AS(
        tf::cpp::fit_knn(source.point_cloud(), target.point_cloud(), options),
        std::invalid_argument);
    options.k = -1;
    CHECK_THROWS_AS(
        tf::cpp::fit_knn(source.point_cloud(), target.point_cloud(), options),
        std::invalid_argument);
    options = {};
    options.sigma = nan;
    CHECK_THROWS_AS(
        tf::cpp::fit_knn(source.point_cloud(), target.point_cloud(), options),
        std::invalid_argument);
    options.sigma = infinity;
    CHECK_THROWS_AS(
        tf::cpp::fit_knn(source.point_cloud(), target.point_cloud(), options),
        std::invalid_argument);
    options = {};
    options.outlier_proportion = nan;
    CHECK_THROWS_AS(
        tf::cpp::fit_knn(source.point_cloud(), target.point_cloud(), options),
        std::invalid_argument);
    options.outlier_proportion = infinity;
    CHECK_THROWS_AS(
        tf::cpp::fit_knn(source.point_cloud(), target.point_cloud(), options),
        std::invalid_argument);
    options.outlier_proportion = TestType{-0.1};
    CHECK_THROWS_AS(
        tf::cpp::fit_knn(source.point_cloud(), target.point_cloud(), options),
        std::invalid_argument);
    options.outlier_proportion = TestType{1};
    CHECK_THROWS_AS(
        tf::cpp::fit_knn(source.point_cloud(), target.point_cloud(), options),
        std::invalid_argument);
    CHECK_FALSE(target.cache.is_tree_built());

    auto empty_source = dimensional_cloud<TestType, Dims>(
        deterministic_points<TestType, Dims>(0));
    auto empty_target = dimensional_cloud<TestType, Dims>(
        deterministic_points<TestType, Dims>(0));
    CHECK_THROWS_AS(
        tf::cpp::fit_knn(empty_source.point_cloud(), target.point_cloud()),
        std::invalid_argument);
    CHECK_THROWS_AS(
        tf::cpp::fit_knn(source.point_cloud(), empty_target.point_cloud()),
        std::invalid_argument);

    tf::cpp::chamfer_error_options<TestType> chamfer_options;
    chamfer_options.outlier_proportion = nan;
    CHECK_THROWS_AS(tf::cpp::chamfer_error(source.point_cloud(),
                                           target.point_cloud(),
                                           chamfer_options),
                    std::invalid_argument);
    CHECK_THROWS_AS(tf::cpp::symmetric_chamfer_error(source.point_cloud(),
                                                     target.point_cloud(),
                                                     chamfer_options),
                    std::invalid_argument);
    chamfer_options.outlier_proportion = TestType{1};
    CHECK_THROWS_AS(tf::cpp::chamfer_error(source.point_cloud(),
                                           target.point_cloud(),
                                           chamfer_options),
                    std::invalid_argument);
    CHECK_THROWS_AS(tf::cpp::symmetric_chamfer_error(source.point_cloud(),
                                                     target.point_cloud(),
                                                     chamfer_options),
                    std::invalid_argument);
    CHECK_THROWS_AS(tf::cpp::chamfer_error(empty_source.point_cloud(),
                                           target.point_cloud()),
                    std::invalid_argument);
    CHECK_THROWS_AS(tf::cpp::chamfer_error(source.point_cloud(),
                                           empty_target.point_cloud()),
                    std::invalid_argument);
    CHECK_THROWS_AS(tf::cpp::symmetric_chamfer_error(empty_source.point_cloud(),
                                                     target.point_cloud()),
                    std::invalid_argument);
    CHECK_THROWS_AS(tf::cpp::symmetric_chamfer_error(
                        source.point_cloud(), empty_target.point_cloud()),
                    std::invalid_argument);
    CHECK_FALSE(target.cache.is_tree_built());
    CHECK_FALSE(empty_target.cache.is_tree_built());
  };

  check_dimension(std::integral_constant<std::size_t, 2>{});
  check_dimension(std::integral_constant<std::size_t, 3>{});
}

TEMPLATE_TEST_CASE("directed Chamfer matches 2D and 3D Python fixtures",
                   "[cpp][geometry][registration][chamfer][python-parity]",
                   float, double) {
  const auto check_dimension = [&](auto dimension) {
    constexpr std::size_t Dims = decltype(dimension)::value;
    INFO("Dims = " << Dims);
    auto points = deterministic_points<TestType, Dims>(30);
    auto identical_source =
        dimensional_cloud<TestType, Dims>(points.deep_copy());
    auto identical_target =
        dimensional_cloud<TestType, Dims>(points.deep_copy());
    const auto identical = tf::cpp::chamfer_error(
        identical_source.point_cloud(), identical_target.point_cloud());
    static_assert(std::is_same_v<decltype(identical), const TestType>);
    CHECK(identical == Catch::Approx(0.0).margin(tolerance<TestType>()));

    std::vector<std::array<double, Dims>> triangle(3);
    triangle[0].fill(0.0);
    triangle[1].fill(0.0);
    triangle[2].fill(0.0);
    triangle[1][0] = 1.0;
    triangle[2][1] = 1.0;
    auto base_points = make_dimensional_points<TestType, Dims>(triangle);
    std::array<double, Dims> half_offset{};
    half_offset[0] = 0.5;
    auto offset_points = translate_dimensional(base_points, half_offset);
    auto offset_source =
        dimensional_cloud<TestType, Dims>(base_points.deep_copy());
    auto offset_target =
        dimensional_cloud<TestType, Dims>(offset_points.deep_copy());
    CHECK(tf::cpp::chamfer_error(offset_source.point_cloud(),
                                 offset_target.point_cloud()) ==
          Catch::Approx(0.5).margin(tolerance<TestType>()));

    std::vector<std::array<double, Dims>> first_two(2);
    std::vector<std::array<double, Dims>> first_four(4);
    for (std::size_t index = 0; index < first_four.size(); ++index) {
      first_four[index].fill(0.0);
      first_four[index][0] = static_cast<double>(index);
      if (index < first_two.size())
        first_two[index] = first_four[index];
    }
    auto fewer = dimensional_cloud<TestType, Dims>(
        make_dimensional_points<TestType, Dims>(first_two));
    auto more = dimensional_cloud<TestType, Dims>(
        make_dimensional_points<TestType, Dims>(first_four));
    CHECK(tf::cpp::chamfer_error(fewer.point_cloud(), more.point_cloud()) ==
          Catch::Approx(0.0).margin(tolerance<TestType>()));

    std::vector<std::array<double, Dims>> origin_values(1);
    origin_values[0].fill(0.0);
    std::vector<std::array<double, Dims>> axes(Dims);
    for (std::size_t axis = 0; axis < Dims; ++axis) {
      axes[axis].fill(0.0);
      axes[axis][axis] = 1.0;
    }
    auto origin = dimensional_cloud<TestType, Dims>(
        make_dimensional_points<TestType, Dims>(origin_values));
    auto unit_axes = dimensional_cloud<TestType, Dims>(
        make_dimensional_points<TestType, Dims>(axes));
    CHECK(
        tf::cpp::chamfer_error(origin.point_cloud(), unit_axes.point_cloud()) ==
        Catch::Approx(1.0).margin(tolerance<TestType>()));
  };

  check_dimension(std::integral_constant<std::size_t, 2>{});
  check_dimension(std::integral_constant<std::size_t, 3>{});

  auto triangle_2d = make_dimensional_points<TestType, 2>(
      {{{0.0, 1.0}}, {{-0.866025403784, -0.5}}, {{0.866025403784, -0.5}}});
  auto shifted_triangle_2d =
      translate_dimensional(triangle_2d, std::array<double, 2>{2.0, 0.0});
  auto triangle_source =
      dimensional_cloud<TestType, 2>(triangle_2d.deep_copy());
  auto triangle_target =
      dimensional_cloud<TestType, 2>(shifted_triangle_2d.deep_copy());
  const auto triangle_error = tf::cpp::chamfer_error(
      triangle_source.point_cloud(), triangle_target.point_cloud());
  CHECK(triangle_error > TestType{0});
  CHECK(triangle_error < TestType{2.5});

  std::vector<std::array<double, 3>> cube_values;
  for (int x = 0; x < 2; ++x)
    for (int y = 0; y < 2; ++y)
      for (int z = 0; z < 2; ++z)
        cube_values.push_back({static_cast<double>(x), static_cast<double>(y),
                               static_cast<double>(z)});
  auto cube_points = make_dimensional_points<TestType, 3>(cube_values);
  auto shifted_cube =
      translate_dimensional(cube_points, std::array<double, 3>{1.0, 0.0, 0.0});
  auto cube_source = dimensional_cloud<TestType, 3>(cube_points.deep_copy());
  auto cube_target = dimensional_cloud<TestType, 3>(shifted_cube.deep_copy());
  CHECK(tf::cpp::chamfer_error(cube_source.point_cloud(),
                               cube_target.point_cloud()) ==
        Catch::Approx(0.5).margin(tolerance<TestType>()));
}

TEMPLATE_TEST_CASE(
    "symmetric Chamfer is exact swap invariant trimmed framed and reuses trees",
    "[cpp][geometry][registration][chamfer][symmetric]", float, double) {
  const auto check_dimension = [&](auto dimension) {
    constexpr std::size_t Dims = decltype(dimension)::value;
    INFO("Dims = " << Dims);
    std::vector<std::array<double, Dims>> source_values(1);
    source_values[0].fill(0.0);
    std::vector<std::array<double, Dims>> target_values(2);
    target_values[0].fill(0.0);
    target_values[1].fill(0.0);
    target_values[0][0] = 1.0;
    target_values[1][0] = 2.0;
    auto source = dimensional_cloud<TestType, Dims>(
        make_dimensional_points<TestType, Dims>(source_values));
    auto target = dimensional_cloud<TestType, Dims>(
        make_dimensional_points<TestType, Dims>(target_values));
    CHECK_FALSE(source.cache.is_tree_built());
    CHECK_FALSE(target.cache.is_tree_built());
    const auto forward = tf::cpp::symmetric_chamfer_error(source.point_cloud(),
                                                          target.point_cloud());
    static_assert(std::is_same_v<decltype(forward), const TestType>);
    CHECK(forward == Catch::Approx(1.25).margin(tolerance<TestType>()));
    CHECK(source.cache.tree_build_count() == 1);
    CHECK(target.cache.tree_build_count() == 1);
    CHECK(tf::cpp::symmetric_chamfer_error(target.point_cloud(),
                                           source.point_cloud()) ==
          Catch::Approx(forward).margin(tolerance<TestType>()));
    CHECK(tf::cpp::chamfer_error(source.point_cloud(), target.point_cloud()) ==
          Catch::Approx(1.0).margin(tolerance<TestType>()));
    CHECK(tf::cpp::chamfer_error(target.point_cloud(), source.point_cloud()) ==
          Catch::Approx(1.5).margin(tolerance<TestType>()));
    static_cast<void>(tf::cpp::symmetric_chamfer_error(source.point_cloud(),
                                                       target.point_cloud()));
    CHECK(source.cache.tree_build_count() == 1);
    CHECK(target.cache.tree_build_count() == 1);

    std::vector<std::array<double, Dims>> outlier_values(4);
    for (std::size_t index = 0; index < outlier_values.size(); ++index) {
      outlier_values[index].fill(0.0);
      outlier_values[index][0] =
          std::array<double, 4>{1.0, 2.0, 3.0, 100.0}[index];
    }
    auto outliers = dimensional_cloud<TestType, Dims>(
        make_dimensional_points<TestType, Dims>(outlier_values));
    auto trim_target = dimensional_cloud<TestType, Dims>(
        make_dimensional_points<TestType, Dims>(source_values));
    const auto untrimmed = tf::cpp::symmetric_chamfer_error(
        outliers.point_cloud(), trim_target.point_cloud());
    const auto trimmed = tf::cpp::symmetric_chamfer_error(
        outliers.point_cloud(), trim_target.point_cloud(),
        tf::cpp::chamfer_error_options<TestType>{TestType{0.25}});
    CHECK(trimmed == Catch::Approx(1.5).margin(tolerance<TestType>()));
    CHECK(trimmed < untrimmed);

    auto framed_source = dimensional_cloud<TestType, Dims>(
        make_dimensional_points<TestType, Dims>(source_values));
    auto framed_target = dimensional_cloud<TestType, Dims>(
        make_dimensional_points<TestType, Dims>(source_values));
    std::array<double, Dims> source_frame{};
    std::array<double, Dims> target_frame{};
    source_frame[0] = 1.0;
    target_frame[0] = 3.0;
    place_from(framed_source,
               rigid_transform<TestType, Dims>(0.0, source_frame));
    place_from(framed_target,
               rigid_transform<TestType, Dims>(0.0, target_frame));
    CHECK(tf::cpp::symmetric_chamfer_error(framed_source.point_cloud(),
                                           framed_target.point_cloud()) ==
          Catch::Approx(2.0).margin(tolerance<TestType>()));
  };

  check_dimension(std::integral_constant<std::size_t, 2>{});
  check_dimension(std::integral_constant<std::size_t, 3>{});
}

TEMPLATE_TEST_CASE(
    "new registration fronts support sync future resolver lifetime and caches",
    "[cpp][geometry][registration][async][knn][chamfer][ownership]", float,
    double) {
  const auto check_dimension = [&](auto dimension) {
    constexpr std::size_t Dims = decltype(dimension)::value;
    INFO("Dims = " << Dims);
    auto points = deterministic_points<TestType, Dims>(40);
    std::array<double, Dims> offset{};
    offset.fill(0.02);
    auto shifted = translate_dimensional(points, offset);
    auto source = dimensional_cloud<TestType, Dims>(points.deep_copy());
    auto target = dimensional_cloud<TestType, Dims>(shifted.deep_copy());
    int submissions = 0;

    tf::cpp::fit_knn_options<TestType> async_options;
    async_options.k = 5;
    async_options.sigma = TestType{0.2};
    async_options.outlier_proportion = TestType{0.1};
    check_rigid_transform<TestType, Dims>(tf::cpp::fit_knn(
        source.point_cloud(), target.point_cloud(), async_options));
    check_rigid_transform<TestType, Dims>(
        tf::cpp::async::fit_knn(source.point_cloud(), target.point_cloud(),
                                async_options)
            .get());
    auto resolved_fit = tf::cpp::async::fit_knn(
        counting_resolver{&submissions}, source.point_cloud(),
        target.point_cloud(), async_options);
    static_assert(std::is_same_v<decltype(resolved_fit),
                                 std::future<tf::cpp::nd_array<TestType>>>);
    check_rigid_transform<TestType, Dims>(resolved_fit.get());
    CHECK_FALSE(source.cache.is_tree_built());
    CHECK(target.cache.tree_build_count() == 1);

    tf::cpp::fit_icp_options<TestType> zero_icp;
    zero_icp.max_iterations = 0;
    zero_icp.n_samples = 0;
    zero_icp.k = 0;
    check_rigid_transform<TestType, Dims>(
        tf::cpp::async::fit_icp(source.point_cloud(), target.point_cloud(),
                                zero_icp)
            .get());
    check_rigid_transform<TestType, Dims>(
        tf::cpp::async::fit_rigid(source.point_cloud(), target.point_cloud())
            .get());
    check_rigid_transform<TestType, Dims>(
        tf::cpp::async::fit_obb(source.point_cloud(), target.point_cloud(),
                                tf::cpp::fit_obb_options{0})
            .get());
    check_rigid_transform<TestType, Dims>(
        tf::cpp::async::fit_icp(counting_resolver{&submissions},
                                source.point_cloud(), target.point_cloud(),
                                zero_icp)
            .get());
    check_rigid_transform<TestType, Dims>(
        tf::cpp::async::fit_rigid(counting_resolver{&submissions},
                                  source.point_cloud(), target.point_cloud())
            .get());
    check_rigid_transform<TestType, Dims>(
        tf::cpp::async::fit_obb(counting_resolver{&submissions},
                                source.point_cloud(), target.point_cloud(),
                                tf::cpp::fit_obb_options{0})
            .get());

    const auto directed =
        tf::cpp::chamfer_error(source.point_cloud(), target.point_cloud());
    const auto future_directed = tf::cpp::async::chamfer_error(
                                     source.point_cloud(), target.point_cloud())
                                     .get();
    auto resolved_directed = tf::cpp::async::chamfer_error(
        counting_resolver{&submissions}, source.point_cloud(),
        target.point_cloud());
    static_assert(
        std::is_same_v<decltype(resolved_directed), std::future<TestType>>);
    CHECK(future_directed ==
          Catch::Approx(directed).margin(tolerance<TestType>()));
    CHECK(resolved_directed.get() ==
          Catch::Approx(directed).margin(tolerance<TestType>()));
    CHECK_FALSE(source.cache.is_tree_built());
    CHECK(target.cache.tree_build_count() == 1);

    const auto symmetric = tf::cpp::symmetric_chamfer_error(
        source.point_cloud(), target.point_cloud());
    const auto future_symmetric =
        tf::cpp::async::symmetric_chamfer_error(source.point_cloud(),
                                                target.point_cloud())
            .get();
    auto resolved_symmetric = tf::cpp::async::symmetric_chamfer_error(
        counting_resolver{&submissions}, source.point_cloud(),
        target.point_cloud());
    static_assert(
        std::is_same_v<decltype(resolved_symmetric), std::future<TestType>>);
    CHECK(future_symmetric ==
          Catch::Approx(symmetric).margin(tolerance<TestType>()));
    CHECK(resolved_symmetric.get() ==
          Catch::Approx(symmetric).margin(tolerance<TestType>()));
    CHECK(submissions == 6);
    CHECK(source.cache.tree_build_count() == 1);
    CHECK(target.cache.tree_build_count() == 1);

    auto future_source = dimensional_cloud<TestType, Dims>(points.deep_copy());
    auto future_target = dimensional_cloud<TestType, Dims>(shifted.deep_copy());
    auto owned_fit = tf::cpp::async::fit_knn(future_source.point_cloud(),
                                             future_target.point_cloud());
    auto owned_icp = tf::cpp::async::fit_icp(
        future_source.point_cloud(), future_target.point_cloud(), zero_icp);
    auto owned_rigid = tf::cpp::async::fit_rigid(future_source.point_cloud(),
                                                 future_target.point_cloud());
    auto owned_obb = tf::cpp::async::fit_obb(future_source.point_cloud(),
                                             future_target.point_cloud(),
                                             tf::cpp::fit_obb_options{0});
    auto owned_directed = tf::cpp::async::chamfer_error(
        future_source.point_cloud(), future_target.point_cloud());
    auto owned_symmetric = tf::cpp::async::symmetric_chamfer_error(
        future_source.point_cloud(), future_target.point_cloud());
    check_rigid_transform<TestType, Dims>(owned_fit.get());
    check_rigid_transform<TestType, Dims>(owned_icp.get());
    check_rigid_transform<TestType, Dims>(owned_rigid.get());
    check_rigid_transform<TestType, Dims>(owned_obb.get());
    CHECK(std::isfinite(owned_directed.get()));
    CHECK(std::isfinite(owned_symmetric.get()));

    auto resolver_source =
        dimensional_cloud<TestType, Dims>(points.deep_copy());
    auto resolver_target =
        dimensional_cloud<TestType, Dims>(shifted.deep_copy());
    auto resolver_fit = tf::cpp::async::fit_knn(counting_resolver{&submissions},
                                                resolver_source.point_cloud(),
                                                resolver_target.point_cloud());
    auto resolver_directed = tf::cpp::async::chamfer_error(
        counting_resolver{&submissions}, resolver_source.point_cloud(),
        resolver_target.point_cloud());
    auto resolver_symmetric = tf::cpp::async::symmetric_chamfer_error(
        counting_resolver{&submissions}, resolver_source.point_cloud(),
        resolver_target.point_cloud());
    check_rigid_transform<TestType, Dims>(resolver_fit.get());
    CHECK(std::isfinite(resolver_directed.get()));
    CHECK(std::isfinite(resolver_symmetric.get()));
    CHECK(submissions == 9);
  };

  check_dimension(std::integral_constant<std::size_t, 2>{});
  check_dimension(std::integral_constant<std::size_t, 3>{});
}

TEMPLATE_TEST_CASE("async registration preserves result and option semantics",
                   "[cpp][geometry][registration][async]", float, double) {
  auto points = sample_points<TestType>();
  auto shifted = translated(points, 0.01, -0.005, 0.008);
  auto source = cloud(points);
  auto target = cloud(shifted);

  tf::cpp::fit_icp_options<TestType> icp_options;
  icp_options.max_iterations = 0;
  icp_options.n_samples = 0;
  icp_options.k = 0;
  auto icp = tf::cpp::async::fit_icp(source.point_cloud(), target.point_cloud(),
                                     icp_options);
  auto rigid =
      tf::cpp::async::fit_rigid(source.point_cloud(), target.point_cloud());
  auto obb = tf::cpp::async::fit_obb(source.point_cloud(), target.point_cloud(),
                                     tf::cpp::fit_obb_options{0});
  int submissions = 0;
  const tf::cpp::chamfer_error_options<TestType> chamfer_options{
      TestType{0.25}};
  auto chamfer = tf::cpp::async::chamfer_error(
      counting_resolver{&submissions}, source.point_cloud(),
      target.point_cloud(), chamfer_options);

  static_assert(
      std::is_same_v<decltype(icp), std::future<tf::cpp::nd_array<TestType>>>);
  static_assert(std::is_same_v<decltype(rigid),
                               std::future<tf::cpp::nd_array<TestType>>>);
  static_assert(
      std::is_same_v<decltype(obb), std::future<tf::cpp::nd_array<TestType>>>);
  static_assert(std::is_same_v<decltype(chamfer), std::future<TestType>>);
  CHECK(submissions == 1);

  const auto expected_icp =
      tf::cpp::fit_icp(source.point_cloud(), target.point_cloud(), icp_options);
  const auto expected_rigid =
      tf::cpp::fit_rigid(source.point_cloud(), target.point_cloud());
  const auto expected_obb = tf::cpp::fit_obb(
      source.point_cloud(), target.point_cloud(), tf::cpp::fit_obb_options{0});
  const auto expected_chamfer = tf::cpp::chamfer_error(
      source.point_cloud(), target.point_cloud(), chamfer_options);
  const auto actual_icp = icp.get();
  const auto actual_rigid = rigid.get();
  const auto actual_obb = obb.get();
  REQUIRE(actual_icp.length() == expected_icp.length());
  REQUIRE(actual_rigid.length() == expected_rigid.length());
  REQUIRE(actual_obb.length() == expected_obb.length());
  for (std::size_t index = 0; index < actual_icp.length(); ++index) {
    CHECK(actual_icp[index] == expected_icp[index]);
    CHECK(actual_rigid[index] ==
          Catch::Approx(expected_rigid[index]).margin(tolerance<TestType>()));
    CHECK(actual_obb[index] ==
          Catch::Approx(expected_obb[index]).margin(tolerance<TestType>()));
  }
  CHECK(chamfer.get() ==
        Catch::Approx(expected_chamfer).margin(tolerance<TestType>()));
}

TEMPLATE_TEST_CASE("async registration fills the cache it was handed",
                   "[cpp][geometry][registration][async][cache]", float,
                   double) {
  auto points = sample_points<TestType>();
  auto source = cloud(points);
  auto target = cloud(points);
  bool tree_was_built = true;

  auto error = tf::cpp::async::chamfer_error(
      tree_observing_resolver<TestType>{&target.cache, &tree_was_built},
      source.point_cloud(), target.point_cloud());
  CHECK_FALSE(tree_was_built);
  CHECK(error.get() == Catch::Approx(0.0));
  // one job is one filler, and the cloud it carries is the caller's, so the
  // tree the worker built is there when the future completes
  CHECK(target.cache.is_tree_built());
  CHECK(target.cache.tree_build_count() == 1);

  auto retained =
      tf::cpp::async::fit_rigid(source.point_cloud(), target.point_cloud());
  CHECK(has_matrix_shape(retained.get()));

  auto invalid_source = cloud(points);
  auto invalid_target = cloud(points);
  auto failure = tf::cpp::async::fit_obb(invalid_source.point_cloud(),
                                         invalid_target.point_cloud(),
                                         tf::cpp::fit_obb_options{-1});
  CHECK_THROWS_AS(failure.get(), std::invalid_argument);
  CHECK_FALSE(invalid_target.cache.is_tree_built());
}

TEMPLATE_TEST_CASE("registration operations return typed 4x4 results",
                   "[cpp][geometry][registration]", float, double) {
  auto points = sample_points<TestType>();
  auto shifted = translated(points, 0.01, -0.005, 0.008);
  auto source = cloud(points);
  auto target = cloud(shifted);

  const auto rigid =
      tf::cpp::fit_rigid(source.point_cloud(), target.point_cloud());
  check_translation(rigid, 0.01, -0.005, 0.008);

  tf::cpp::fit_icp_options<TestType> icp_options;
  icp_options.max_iterations = 30;
  icp_options.n_samples = 0;
  icp_options.k = 1;
  icp_options.min_relative_improvement = TestType{0};
  const auto icp =
      tf::cpp::fit_icp(source.point_cloud(), target.point_cloud(), icp_options);
  REQUIRE(has_matrix_shape(icp));
  CHECK(matrix_is_finite(icp));
  CHECK(icp[3] == Catch::Approx(0.01).margin(0.005));
  CHECK(icp[7] == Catch::Approx(-0.005).margin(0.005));
  CHECK(icp[11] == Catch::Approx(0.008).margin(0.005));

  const auto error =
      tf::cpp::chamfer_error(source.point_cloud(), target.point_cloud());
  static_assert(std::is_same_v<decltype(error), const TestType>);
  CHECK(error > TestType{0});
  CHECK(std::isfinite(error));

  auto obb_source = cloud(box_points<TestType>());
  auto obb_target = cloud(translated(points_array(obb_source), 3.0, -2.0, 1.0));
  const auto obb =
      tf::cpp::fit_obb(obb_source.point_cloud(), obb_target.point_cloud(),
                       tf::cpp::fit_obb_options{0});
  REQUIRE(has_matrix_shape(obb));
  CHECK(matrix_is_finite(obb));
  check_translation(obb, 3.0, -2.0, 1.0);
}

TEMPLATE_TEST_CASE("registration preserves cloud normals and frames",
                   "[cpp][geometry][registration][policy]", float, double) {
  auto points = box_points<TestType>();
  auto source = cloud(points);
  auto target = cloud(points);
  state_normals(source, radial_normals(points_array(source)));
  state_normals(target, radial_normals(points_array(target)));

  const auto rigid_with_normals =
      tf::cpp::fit_rigid(source.point_cloud(), target.point_cloud());
  REQUIRE(has_matrix_shape(rigid_with_normals));
  CHECK(matrix_is_finite(rigid_with_normals));

  tf::cpp::fit_icp_options<TestType> options;
  options.max_iterations = 1;
  options.n_samples = 0;
  options.min_relative_improvement = TestType{0};
  const auto icp_with_normals =
      tf::cpp::fit_icp(source.point_cloud(), target.point_cloud(), options);
  REQUIRE(has_matrix_shape(icp_with_normals));
  CHECK(matrix_is_finite(icp_with_normals));

  auto framed_source = cloud(points);
  auto framed_target = cloud(points);
  place_from(framed_source, translation<TestType>(1.0, -2.0, 0.5));
  place_from(framed_target, translation<TestType>(3.0, 1.0, -1.5));

  const auto framed_rigid = tf::cpp::fit_rigid(framed_source.point_cloud(),
                                               framed_target.point_cloud());
  check_translation(framed_rigid, 2.0, 3.0, -2.0);

  auto framed_error_source = cloud(make_points<TestType>({{{0.0, 0.0, 0.0}}}));
  auto framed_error_target = cloud(make_points<TestType>({{{0.0, 0.0, 0.0}}}));
  place_from(framed_error_source, translation<TestType>(1.0, -2.0, 0.5));
  place_from(framed_error_target, translation<TestType>(3.0, 1.0, -1.5));
  CHECK(tf::cpp::chamfer_error(framed_error_source.point_cloud(),
                               framed_error_target.point_cloud()) ==
        Catch::Approx(std::sqrt(17.0)).margin(tolerance<TestType>()));

  const auto framed_obb = tf::cpp::fit_obb(framed_source.point_cloud(),
                                           framed_target.point_cloud());
  check_translation(framed_obb, 2.0, 3.0, -2.0);
}

TEMPLATE_TEST_CASE("registration options preserve zero and trimming semantics",
                   "[cpp][geometry][registration][options]", float, double) {
  auto points = sample_points<TestType>();
  auto source = cloud(points);
  auto target = cloud(translated(points, 0.1, 0.0, 0.0));

  tf::cpp::fit_icp_options<TestType> zero_iterations;
  zero_iterations.max_iterations = 0;
  zero_iterations.n_samples = 0;
  zero_iterations.k = 0;
  const auto identity = tf::cpp::fit_icp(source.point_cloud(),
                                         target.point_cloud(), zero_iterations);
  REQUIRE(has_matrix_shape(identity));
  for (int row = 0; row < 4; ++row)
    for (int column = 0; column < 4; ++column)
      CHECK(identity[static_cast<std::size_t>(row * 4 + column)] ==
            (row == column ? TestType{1} : TestType{0}));

  auto error_source = cloud(make_points<TestType>(
      {{1.0, 0.0, 0.0}, {2.0, 0.0, 0.0}, {3.0, 0.0, 0.0}, {100.0, 0.0, 0.0}}));
  auto error_target = cloud(make_points<TestType>({{{0.0, 0.0, 0.0}}}));
  const auto trimmed = tf::cpp::chamfer_error(
      error_source.point_cloud(), error_target.point_cloud(),
      tf::cpp::chamfer_error_options<TestType>{TestType{0.25}});
  CHECK(trimmed == Catch::Approx(2.0).margin(tolerance<TestType>()));
}

TEMPLATE_TEST_CASE("registration rejects empty point clouds",
                   "[cpp][geometry][registration][validation]", float, double) {
  auto empty_source = cloud(make_points<TestType>({}));
  auto empty_target = cloud(make_points<TestType>({}));
  auto source = cloud(sample_points<TestType>());
  auto target = cloud(sample_points<TestType>());

  CHECK_THROWS_AS(
      tf::cpp::fit_icp(empty_source.point_cloud(), target.point_cloud()),
      std::invalid_argument);
  CHECK_THROWS_AS(
      tf::cpp::fit_rigid(empty_source.point_cloud(), target.point_cloud()),
      std::invalid_argument);
  CHECK_THROWS_AS(
      tf::cpp::fit_obb(empty_source.point_cloud(), target.point_cloud()),
      std::invalid_argument);
  CHECK_THROWS_AS(
      tf::cpp::chamfer_error(empty_source.point_cloud(), target.point_cloud()),
      std::invalid_argument);
  CHECK_FALSE(target.cache.is_tree_built());

  CHECK_THROWS_AS(
      tf::cpp::fit_icp(source.point_cloud(), empty_target.point_cloud()),
      std::invalid_argument);
  CHECK_THROWS_AS(
      tf::cpp::fit_rigid(source.point_cloud(), empty_target.point_cloud()),
      std::invalid_argument);
  CHECK_THROWS_AS(
      tf::cpp::fit_obb(source.point_cloud(), empty_target.point_cloud()),
      std::invalid_argument);
  CHECK_THROWS_AS(
      tf::cpp::chamfer_error(source.point_cloud(), empty_target.point_cloud()),
      std::invalid_argument);
  CHECK_FALSE(empty_target.cache.is_tree_built());
}

TEMPLATE_TEST_CASE("registration validates floating controls",
                   "[cpp][geometry][registration][validation][options]", float,
                   double) {
  auto source = cloud(sample_points<TestType>());
  auto target = cloud(sample_points<TestType>());
  const auto nan = std::numeric_limits<TestType>::quiet_NaN();
  const auto infinity = std::numeric_limits<TestType>::infinity();

  tf::cpp::fit_icp_options<TestType> icp_options;
  icp_options.min_relative_improvement = nan;
  CHECK_THROWS_AS(
      tf::cpp::fit_icp(source.point_cloud(), target.point_cloud(), icp_options),
      std::invalid_argument);
  icp_options = {};
  icp_options.min_relative_improvement = infinity;
  CHECK_THROWS_AS(
      tf::cpp::fit_icp(source.point_cloud(), target.point_cloud(), icp_options),
      std::invalid_argument);
  icp_options = {};
  icp_options.ema_alpha = nan;
  CHECK_THROWS_AS(
      tf::cpp::fit_icp(source.point_cloud(), target.point_cloud(), icp_options),
      std::invalid_argument);
  icp_options = {};
  icp_options.ema_alpha = infinity;
  CHECK_THROWS_AS(
      tf::cpp::fit_icp(source.point_cloud(), target.point_cloud(), icp_options),
      std::invalid_argument);
  icp_options = {};
  icp_options.ema_alpha = static_cast<TestType>(-0.1);
  CHECK_THROWS_AS(
      tf::cpp::fit_icp(source.point_cloud(), target.point_cloud(), icp_options),
      std::invalid_argument);
  icp_options = {};
  icp_options.ema_alpha = static_cast<TestType>(1.1);
  CHECK_THROWS_AS(
      tf::cpp::fit_icp(source.point_cloud(), target.point_cloud(), icp_options),
      std::invalid_argument);
  icp_options = {};
  icp_options.sigma = nan;
  CHECK_THROWS_AS(
      tf::cpp::fit_icp(source.point_cloud(), target.point_cloud(), icp_options),
      std::invalid_argument);
  icp_options = {};
  icp_options.sigma = infinity;
  CHECK_THROWS_AS(
      tf::cpp::fit_icp(source.point_cloud(), target.point_cloud(), icp_options),
      std::invalid_argument);
  icp_options = {};
  icp_options.outlier_proportion = nan;
  CHECK_THROWS_AS(
      tf::cpp::fit_icp(source.point_cloud(), target.point_cloud(), icp_options),
      std::invalid_argument);
  icp_options = {};
  icp_options.outlier_proportion = infinity;
  CHECK_THROWS_AS(
      tf::cpp::fit_icp(source.point_cloud(), target.point_cloud(), icp_options),
      std::invalid_argument);
  icp_options = {};
  icp_options.outlier_proportion = static_cast<TestType>(-0.1);
  CHECK_THROWS_AS(
      tf::cpp::fit_icp(source.point_cloud(), target.point_cloud(), icp_options),
      std::invalid_argument);
  icp_options = {};
  icp_options.outlier_proportion = TestType{1};
  CHECK_THROWS_AS(
      tf::cpp::fit_icp(source.point_cloud(), target.point_cloud(), icp_options),
      std::invalid_argument);

  const tf::cpp::chamfer_error_options<TestType> nan_chamfer{nan};
  CHECK_THROWS_AS(tf::cpp::chamfer_error(source.point_cloud(),
                                         target.point_cloud(), nan_chamfer),
                  std::invalid_argument);
  const tf::cpp::chamfer_error_options<TestType> infinite_chamfer{infinity};
  CHECK_THROWS_AS(tf::cpp::chamfer_error(source.point_cloud(),
                                         target.point_cloud(),
                                         infinite_chamfer),
                  std::invalid_argument);
  const tf::cpp::chamfer_error_options<TestType> negative_chamfer{
      static_cast<TestType>(-0.1)};
  CHECK_THROWS_AS(tf::cpp::chamfer_error(source.point_cloud(),
                                         target.point_cloud(),
                                         negative_chamfer),
                  std::invalid_argument);
  const tf::cpp::chamfer_error_options<TestType> full_chamfer{TestType{1}};
  CHECK_THROWS_AS(tf::cpp::chamfer_error(source.point_cloud(),
                                         target.point_cloud(), full_chamfer),
                  std::invalid_argument);
  CHECK_FALSE(target.cache.is_tree_built());
}

// A cloud's normals RIDE ITS READING: they are a per-point attribute the
// caller holds, so a caller that restates its points states the normals that
// name them, and one that states none is read point to point.
TEMPLATE_TEST_CASE("a cloud's normals name the points it holds",
                   "[cpp][geometry][registration][validation][normals]", float,
                   double) {
  auto original_points = sample_points<TestType>();
  auto source = cloud(original_points);
  auto target = cloud(original_points);
  state_normals(source, radial_normals(points_array(source)));
  state_normals(target, radial_normals(points_array(target)));
  REQUIRE(source.point_cloud().has_normals());

  auto updated_points = box_points<TestType>();
  restate_points(source, updated_points);
  state_normals(source, radial_normals(points_array(source)));
  restate_points(target, translated(updated_points, 3.0, -2.0, 1.0));
  state_normals(target, radial_normals(points_array(target)));
  CHECK(source.point_cloud().number_of_points() ==
        static_cast<std::size_t>(updated_points.shape_at(0)));

  const auto obb = tf::cpp::fit_obb(source.point_cloud(), target.point_cloud(),
                                    tf::cpp::fit_obb_options{0});
  check_translation(obb, 3.0, -2.0, 1.0);
  CHECK(tf::cpp::chamfer_error(source.point_cloud(), target.point_cloud()) >
        TestType{0});

  auto plain_source = cloud(updated_points);
  auto plain_target = cloud(translated(updated_points, 3.0, -2.0, 1.0));
  CHECK_FALSE(plain_source.point_cloud().has_normals());
  CHECK_NOTHROW(
      tf::cpp::fit_icp(plain_source.point_cloud(), plain_target.point_cloud()));
  CHECK_NOTHROW(tf::cpp::fit_rigid(plain_source.point_cloud(),
                                   plain_target.point_cloud()));
}

TEMPLATE_TEST_CASE("registration warms and reuses target trees",
                   "[cpp][geometry][registration][cache]", float, double) {
  auto points = sample_points<TestType>();
  auto source = cloud(points);
  auto target = cloud(points);
  CHECK_FALSE(source.cache.is_tree_built());
  CHECK_FALSE(target.cache.is_tree_built());

  static_cast<void>(
      tf::cpp::fit_rigid(source.point_cloud(), target.point_cloud()));
  CHECK_FALSE(source.cache.is_tree_built());
  CHECK_FALSE(target.cache.is_tree_built());

  static_cast<void>(
      tf::cpp::chamfer_error(source.point_cloud(), target.point_cloud()));
  REQUIRE(target.cache.is_tree_fresh(target.point_cloud().geometry()));
  CHECK(target.cache.tree_build_count() == 1);
  CHECK_FALSE(source.cache.is_tree_built());

  tf::cpp::fit_icp_options<TestType> options;
  options.max_iterations = 0;
  static_cast<void>(
      tf::cpp::fit_icp(source.point_cloud(), target.point_cloud(), options));
  static_cast<void>(tf::cpp::fit_obb(source.point_cloud(), target.point_cloud(),
                                     tf::cpp::fit_obb_options{1}));
  CHECK(target.cache.tree_build_count() == 1);
  CHECK_FALSE(source.cache.is_tree_built());

  restate_points(target, translated(points, 0.01, 0.0, 0.0));
  REQUIRE_FALSE(target.cache.is_tree_fresh(target.point_cloud().geometry()));
  static_cast<void>(
      tf::cpp::chamfer_error(source.point_cloud(), target.point_cloud()));
  CHECK(target.cache.is_tree_fresh(target.point_cloud().geometry()));
  CHECK(target.cache.tree_build_count() == 2);
}

TEMPLATE_TEST_CASE("registration validates handles, layouts, and signed counts",
                   "[cpp][geometry][registration][validation]", float, double) {
  auto points = sample_points<TestType>();
  auto source = cloud(points);
  auto target = cloud(points);

  tf::cpp::fit_icp_options<TestType> icp_options;
  icp_options.max_iterations = -1;
  CHECK_THROWS_AS(
      tf::cpp::fit_icp(source.point_cloud(), target.point_cloud(), icp_options),
      std::invalid_argument);
  icp_options = {};
  icp_options.n_samples = -1;
  CHECK_THROWS_AS(
      tf::cpp::fit_icp(source.point_cloud(), target.point_cloud(), icp_options),
      std::invalid_argument);
  icp_options = {};
  icp_options.k = -1;
  CHECK_THROWS_AS(
      tf::cpp::fit_icp(source.point_cloud(), target.point_cloud(), icp_options),
      std::invalid_argument);
  icp_options = {};
  icp_options.k = 0;
  CHECK_THROWS_AS(
      tf::cpp::fit_icp(source.point_cloud(), target.point_cloud(), icp_options),
      std::invalid_argument);
  CHECK_THROWS_AS(tf::cpp::fit_obb(source.point_cloud(), target.point_cloud(),
                                   tf::cpp::fit_obb_options{-1}),
                  std::invalid_argument);
  CHECK_FALSE(target.cache.is_tree_built());

  auto fewer = cloud(repeated_points<TestType>(3, 0.0));
  CHECK_THROWS_AS(tf::cpp::fit_rigid(source.point_cloud(), fewer.point_cloud()),
                  std::invalid_argument);

  restate_points(source, repeated_points<TestType>(7, 1.0));
  CHECK_THROWS_AS(
      tf::cpp::fit_rigid(source.point_cloud(), target.point_cloud()),
      std::invalid_argument);

  auto destroyed = cloud(points);
  destroyed = {};
  CHECK_THROWS_AS(
      tf::cpp::fit_rigid(destroyed.point_cloud(), target.point_cloud()),
      std::invalid_argument);
  CHECK_THROWS_AS(
      tf::cpp::fit_icp(destroyed.point_cloud(), target.point_cloud()),
      std::invalid_argument);
  CHECK_THROWS_AS(
      tf::cpp::fit_obb(destroyed.point_cloud(), target.point_cloud()),
      std::invalid_argument);
  CHECK_THROWS_AS(
      tf::cpp::chamfer_error(destroyed.point_cloud(), target.point_cloud()),
      std::invalid_argument);
}

namespace {

template <typename Real, std::size_t Dims> struct registration_entries {
  using cloud_type = tf::cpp::point_cloud<Real, Dims>;
  using matrix_type = tf::cpp::nd_array<Real>;

  static auto fit_icp() {
    return static_cast<matrix_type (*)(const cloud_type &, const cloud_type &,
                                       const tf::cpp::fit_icp_options<Real> &)>(
        &tf::cpp::fit_icp<Real, Dims>);
  }
  static auto fit_rigid() {
    return static_cast<matrix_type (*)(const cloud_type &, const cloud_type &)>(
        &tf::cpp::fit_rigid<Real, Dims>);
  }
  static auto fit_knn() {
    return static_cast<matrix_type (*)(const cloud_type &, const cloud_type &,
                                       const tf::cpp::fit_knn_options<Real> &)>(
        &tf::cpp::fit_knn<Real, Dims>);
  }
  static auto fit_obb() {
    return static_cast<matrix_type (*)(const cloud_type &, const cloud_type &,
                                       const tf::cpp::fit_obb_options &)>(
        &tf::cpp::fit_obb<Real, Dims>);
  }
  static auto chamfer_error() {
    return static_cast<Real (*)(const cloud_type &, const cloud_type &,
                                const tf::cpp::chamfer_error_options<Real> &)>(
        &tf::cpp::chamfer_error<Real, Dims>);
  }
  static auto symmetric_chamfer_error() {
    return static_cast<Real (*)(const cloud_type &, const cloud_type &,
                                const tf::cpp::chamfer_error_options<Real> &)>(
        &tf::cpp::symmetric_chamfer_error<Real, Dims>);
  }
};

template <typename Real, std::size_t Dims>
auto check_registration_entries() -> void {
  using entries = registration_entries<Real, Dims>;
  CHECK(entries::fit_icp() != nullptr);
  CHECK(entries::fit_rigid() != nullptr);
  CHECK(entries::fit_knn() != nullptr);
  CHECK(entries::fit_obb() != nullptr);
  CHECK(entries::chamfer_error() != nullptr);
  CHECK(entries::symmetric_chamfer_error() != nullptr);
}

} // namespace

TEST_CASE(
    "registration float and double overloads link from the native archive",
    "[cpp][geometry][registration][archive-link]") {
  check_registration_entries<float, 2>();
  check_registration_entries<float, 3>();
  check_registration_entries<double, 2>();
  check_registration_entries<double, 3>();
}
