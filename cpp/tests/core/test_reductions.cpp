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
#include "trueform/cpp/core/async/reductions.hpp"
#include "trueform/cpp/core/parallel_config.hpp"
#include "trueform/cpp/core/reductions.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <tbb/global_control.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <future>
#include <initializer_list>
#include <limits>
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

template <typename T>
auto check_values(const tf::cpp::nd_array<T> &array,
                  std::initializer_list<T> expected) -> void {
  REQUIRE(array.length() == expected.size());
  auto index = std::size_t{0};
  for (const auto value : expected)
    CHECK(array[index++] == value);
}

template <typename T, typename Generator>
auto make_generated_array(tf::small_vector<int, 3> shape, Generator generator)
    -> tf::cpp::nd_array<T> {
  auto length = std::size_t{1};
  for (const auto dimension : shape)
    length *= static_cast<std::size_t>(dimension);
  tf::buffer<T> buffer;
  buffer.allocate(length);
  for (std::size_t index = 0; index < length; ++index)
    buffer[index] = generator(index);
  return tf::cpp::nd_array<T>::from_buffer(std::move(buffer), std::move(shape));
}

template <typename T>
auto check_same_storage(const tf::cpp::nd_array<T> &a,
                        const tf::cpp::nd_array<T> &b) -> void {
  REQUIRE(a.raw_shape() == b.raw_shape());
  REQUIRE(a.length() == b.length());
  CHECK(std::memcmp(a.raw_data(), b.raw_data(), a.length() * sizeof(T)) == 0);
}

template <typename Operation>
auto run_with_thread_cap(std::size_t thread_cap, Operation operation)
    -> decltype(operation()) {
  tbb::global_control control(tbb::global_control::max_allowed_parallelism,
                              thread_cap);
  return operation();
}

template <typename T> auto check_reduction_dtype() -> void {
  const auto array =
      make_array<T>({T{1}, T{2}, T{3}, T{4}, T{5}, T{6}}, {2, 3});
  CHECK(tf::cpp::sum(array) == 21);
  CHECK(tf::cpp::min(array) == T{1});
  CHECK(tf::cpp::max(array) == T{6});
  CHECK(tf::cpp::mean(array) == Catch::Approx(3.5));
  CHECK(tf::cpp::norm(array) == Catch::Approx(std::sqrt(91.0)));
  CHECK(tf::cpp::argmin(array) == 0);
  CHECK(tf::cpp::argmax(array) == 5);

  const auto summed_rows = tf::cpp::sum(array, 1);
  CHECK(summed_rows.length() == 2);
  CHECK(summed_rows[0] == 6);
  CHECK(summed_rows[1] == 15);
  check_values(tf::cpp::min(array, 0), {T{1}, T{2}, T{3}});
  check_values(tf::cpp::max(array, 0), {T{4}, T{5}, T{6}});

  const auto means = tf::cpp::mean(array, -1);
  CHECK(means[0] == Catch::Approx(2.0));
  CHECK(means[1] == Catch::Approx(5.0));
  const auto norms = tf::cpp::norm(array, 1);
  CHECK(norms[0] == Catch::Approx(std::sqrt(14.0)));
  CHECK(norms[1] == Catch::Approx(std::sqrt(77.0)));
  check_values(tf::cpp::argmin(array, 0),
               {std::int32_t{0}, std::int32_t{0}, std::int32_t{0}});
  check_values(tf::cpp::argmax(array, 1), {std::int32_t{2}, std::int32_t{2}});
}

} // namespace

TEST_CASE("reductions preserve the complete dtype and result matrix",
          "[cpp][core][reductions]") {
  check_reduction_dtype<std::int8_t>();
  check_reduction_dtype<std::int32_t>();
  check_reduction_dtype<std::int64_t>();
  check_reduction_dtype<float>();
  check_reduction_dtype<double>();

  const auto bytes = make_array<std::int8_t>({100, 100, 100}, {3});
  static_assert(std::is_same_v<decltype(tf::cpp::sum(bytes)), std::int32_t>);
  static_assert(std::is_same_v<decltype(tf::cpp::sum(bytes, 0)),
                               tf::cpp::nd_array<std::int32_t>>);
  CHECK(tf::cpp::sum(bytes) == 300);
  check_values(tf::cpp::sum(bytes, 0), {std::int32_t{300}});

  const auto wide = std::int64_t{1} << 40;
  const auto wide_values = make_array<std::int64_t>({wide, wide, wide}, {3});
  static_assert(
      std::is_same_v<decltype(tf::cpp::sum(wide_values)), std::int64_t>);
  CHECK(tf::cpp::sum(wide_values) == 3 * wide);
  check_values(tf::cpp::sum(wide_values, 0), {3 * wide});
  CHECK(tf::cpp::max(wide_values) == wide);

  const auto integers = make_array<std::int32_t>({1, 2}, {2});
  const auto doubles = make_array<double>({1, 2}, {2});
  static_assert(std::is_same_v<decltype(tf::cpp::mean(integers, 0)),
                               tf::cpp::nd_array<float>>);
  static_assert(std::is_same_v<decltype(tf::cpp::norm(integers)), float>);
  static_assert(std::is_same_v<decltype(tf::cpp::mean(doubles, 0)),
                               tf::cpp::nd_array<double>>);
  static_assert(std::is_same_v<decltype(tf::cpp::norm(doubles)), double>);
}

TEST_CASE("boolean reductions return scalar and axis results",
          "[cpp][core][reductions]") {
  const auto values = make_array<std::int8_t>({1, 0, 1, 1, 1, 1}, {2, 3});
  CHECK(tf::cpp::any(values) == 1);
  CHECK(tf::cpp::all(values) == 0);
  check_values(tf::cpp::any(values, 1), {std::int8_t{1}, std::int8_t{1}});
  check_values(tf::cpp::all(values, 1), {std::int8_t{0}, std::int8_t{1}});
  check_values(tf::cpp::any(values, 0),
               {std::int8_t{1}, std::int8_t{1}, std::int8_t{1}});
}

TEST_CASE("reductions define empty and singleton behavior",
          "[cpp][core][reductions]") {
  const auto empty = make_array<float>({}, {0});
  CHECK(tf::cpp::sum(empty) == 0.0F);
  CHECK(tf::cpp::min(empty) == std::numeric_limits<float>::max());
  CHECK(tf::cpp::max(empty) == std::numeric_limits<float>::lowest());
  CHECK(std::isnan(tf::cpp::mean(empty)));
  CHECK(tf::cpp::norm(empty) == 0.0F);
  CHECK_THROWS_AS(tf::cpp::argmin(empty), std::invalid_argument);
  CHECK_THROWS_AS(tf::cpp::argmax(empty), std::invalid_argument);

  const auto empty_bools = make_array<std::int8_t>({}, {0});
  CHECK(tf::cpp::any(empty_bools) == 0);
  CHECK(tf::cpp::all(empty_bools) == 1);

  const auto empty_axis = make_array<float>({}, {2, 0});
  check_values(tf::cpp::sum(empty_axis, 1), {0.0F, 0.0F});
  const auto empty_means = tf::cpp::mean(empty_axis, 1);
  REQUIRE(empty_means.length() == 2);
  CHECK(std::isnan(empty_means[0]));
  CHECK(std::isnan(empty_means[1]));
  check_values(tf::cpp::norm(empty_axis, 1), {0.0F, 0.0F});
  CHECK_THROWS_AS(tf::cpp::argmin(empty_axis, 1), std::invalid_argument);

  const auto singleton = make_array<double>({-3.0}, {1});
  CHECK(tf::cpp::sum(singleton) == -3.0);
  CHECK(tf::cpp::min(singleton) == -3.0);
  CHECK(tf::cpp::max(singleton) == -3.0);
  CHECK(tf::cpp::mean(singleton) == -3.0);
  CHECK(tf::cpp::norm(singleton) == 3.0);
  CHECK(tf::cpp::argmin(singleton) == 0);
  CHECK(tf::cpp::argmax(singleton) == 0);
}

TEST_CASE("axis reductions reject out-of-range axes",
          "[cpp][core][reductions]") {
  const auto values = make_array<float>({1, 2, 3, 4}, {2, 2});
  CHECK_THROWS_AS(tf::cpp::sum(values, 2), std::out_of_range);
  CHECK_THROWS_AS(tf::cpp::min(values, -3), std::out_of_range);
  CHECK_THROWS_AS(tf::cpp::mean(values, 3), std::out_of_range);
  CHECK_THROWS_AS(tf::cpp::norm(values, 3), std::out_of_range);
  CHECK_THROWS_AS(tf::cpp::argmax(values, 3), std::out_of_range);
}

TEST_CASE("work-aware axis reductions preserve every carrier's serial order",
          "[cpp][core][reductions][parallel]") {
  constexpr int outer = 4;
  constexpr int reduced = 4096;
  constexpr int inner = 16;
  const auto input = make_generated_array<float>(
      {outer, reduced, inner}, [](std::size_t index) {
        const auto outer_index = index / (reduced * inner);
        const auto reduced_index = (index / inner) % reduced;
        const auto inner_index = index % inner;
        const auto value =
            static_cast<int>(
                (reduced_index * 37 + inner_index * 11 + outer_index * 3) %
                257) -
            128;
        return static_cast<float>(value) * 0.125F;
      });

  const auto compare_caps = [&](auto operation) {
    const auto single = run_with_thread_cap(1, operation);
    const auto multi = run_with_thread_cap(8, operation);
    check_same_storage(single, multi);
    return multi;
  };

  const auto sums = compare_caps([&] { return tf::cpp::sum(input, 1); });
  const auto minima = compare_caps([&] { return tf::cpp::min(input, 1); });
  const auto maxima = compare_caps([&] { return tf::cpp::max(input, 1); });
  const auto means = compare_caps([&] { return tf::cpp::mean(input, 1); });
  const auto norms = compare_caps([&] { return tf::cpp::norm(input, 1); });
  const auto argmin = compare_caps([&] { return tf::cpp::argmin(input, 1); });
  const auto argmax = compare_caps([&] { return tf::cpp::argmax(input, 1); });

  REQUIRE((sums.raw_shape() == tf::small_vector<int, 3>{outer, inner}));
  const auto outer_index = 2;
  const auto inner_index = 7;
  const auto carrier =
      static_cast<std::size_t>(outer_index * inner + inner_index);
  auto expected_sum = 0.0F;
  auto expected_min = std::numeric_limits<float>::max();
  auto expected_max = std::numeric_limits<float>::lowest();
  auto expected_argmin = 0;
  auto expected_argmax = 0;
  auto expected_mean_total = 0.0;
  auto expected_squared_norm = 0.0;
  for (auto reduced_index = 0; reduced_index < reduced; ++reduced_index) {
    const auto index = static_cast<std::size_t>(
        (outer_index * reduced + reduced_index) * inner + inner_index);
    const auto value = input[index];
    expected_sum += value;
    expected_mean_total += static_cast<double>(value);
    expected_squared_norm += static_cast<double>(value) * value;
    if (value < expected_min) {
      expected_min = value;
      expected_argmin = reduced_index;
    }
    if (value > expected_max) {
      expected_max = value;
      expected_argmax = reduced_index;
    }
  }
  CHECK(sums[carrier] == expected_sum);
  CHECK(minima[carrier] == expected_min);
  CHECK(maxima[carrier] == expected_max);
  CHECK(means[carrier] == static_cast<float>(expected_mean_total / reduced));
  CHECK(norms[carrier] == static_cast<float>(std::sqrt(expected_squared_norm)));
  CHECK(argmin[carrier] == expected_argmin);
  CHECK(argmax[carrier] == expected_argmax);
}

TEST_CASE("above-threshold single-carrier reduction remains valid",
          "[cpp][core][reductions][parallel]") {
  const auto reduced =
      static_cast<int>(tf::cpp::parallel_threshold + std::size_t{1});
  REQUIRE(static_cast<std::size_t>(reduced) > tf::cpp::parallel_threshold);
  const auto input =
      make_generated_array<float>({1, reduced}, [](std::size_t index) {
        return static_cast<float>(static_cast<int>(index % 17) - 8) * 0.25F;
      });

  auto expected = 0.0F;
  for (auto reduced_index = 0; reduced_index < reduced; ++reduced_index)
    expected += input[static_cast<std::size_t>(reduced_index)];

  const auto result = tf::cpp::sum(input, 1);
  REQUIRE((result.raw_shape() == tf::small_vector<int, 3>{1}));
  REQUIRE(result.length() == 1);
  CHECK(result[0] == expected);
}

TEST_CASE("async reductions preserve scalar, axis, and boolean result types",
          "[cpp][core][async][reductions]") {
  auto values = make_array<std::int8_t>({1, 2, 3, 4, 5, 6}, {2, 3});
  auto scalar = tf::cpp::async::sum(values);
  static_assert(std::is_same_v<decltype(scalar), std::future<std::int32_t>>);
  values.destroy();
  CHECK(scalar.get() == 21);

  const auto axis_values = make_array<std::int32_t>({1, 2, 3, 4, 5, 6}, {2, 3});
  auto axis = tf::cpp::async::mean(axis_values, 1);
  static_assert(
      std::is_same_v<decltype(axis), std::future<tf::cpp::nd_array<float>>>);
  const auto means = axis.get();
  CHECK(means[0] == Catch::Approx(2.0));
  CHECK(means[1] == Catch::Approx(5.0));

  const auto bools = make_array<std::int8_t>({1, 0, 1, 1}, {2, 2});
  auto any_scalar = tf::cpp::async::any(bools);
  auto all_axis = tf::cpp::async::all(bools, 1);
  static_assert(std::is_same_v<decltype(any_scalar), std::future<int>>);
  static_assert(std::is_same_v<decltype(all_axis),
                               std::future<tf::cpp::nd_array<std::int8_t>>>);
  CHECK(any_scalar.get() == 1);
  check_values(all_axis.get(), {std::int8_t{0}, std::int8_t{1}});

  auto custom_pending =
      tf::cpp::async::argmax(tf::cpp::async::future_resolver{}, axis_values, 1);
  static_assert(std::is_same_v<decltype(custom_pending),
                               std::future<tf::cpp::nd_array<std::int32_t>>>);
  check_values(custom_pending.get(), {std::int32_t{2}, std::int32_t{2}});

  auto failure = tf::cpp::async::norm(axis_values, 3);
  CHECK_THROWS_AS(failure.get(), std::out_of_range);
}
