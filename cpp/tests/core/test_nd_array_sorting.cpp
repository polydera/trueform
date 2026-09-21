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
#include "trueform/cpp/core/async/nd_array_sorting.hpp"
#include "trueform/cpp/core/nd_array_sorting.hpp"

#include <catch2/catch_test_macros.hpp>
#include <tbb/global_control.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
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

template <typename Operation>
auto run_with_thread_cap(std::size_t thread_cap, Operation operation)
    -> decltype(operation()) {
  tbb::global_control control(tbb::global_control::max_allowed_parallelism,
                              thread_cap);
  return operation();
}

template <std::size_t Width> auto check_row_sort_oracle() -> void {
  constexpr auto rows = std::size_t{4};
  constexpr std::int32_t first[] = {3, 1, 2, 1};
  constexpr std::int32_t second[] = {0, 4, 1, 2};
  constexpr std::int32_t expected_rows[] = {3, 1, 2, 0};
  tf::buffer<std::int32_t> buffer;
  buffer.allocate(rows * Width);
  for (std::size_t row = 0; row < rows; ++row) {
    buffer[row * Width] = first[row];
    buffer[row * Width + 1] = second[row];
    for (std::size_t column = 2; column < Width; ++column)
      buffer[row * Width + column] =
          static_cast<std::int32_t>(row * Width + column);
  }
  const auto input = tf::cpp::nd_array<std::int32_t>::from_buffer(
      std::move(buffer), {static_cast<int>(rows), static_cast<int>(Width)});

  check_values(tf::cpp::argsort(input), {std::int32_t{3}, std::int32_t{1},
                                         std::int32_t{2}, std::int32_t{0}});
  const auto sorted = tf::cpp::sort(input);
  REQUIRE(sorted.length() == input.length());
  for (std::size_t output_row = 0; output_row < rows; ++output_row)
    for (std::size_t column = 0; column < Width; ++column)
      CHECK(sorted[output_row * Width + column] ==
            input[static_cast<std::size_t>(expected_rows[output_row]) * Width +
                  column]);
}

auto row_sort_digest(const tf::cpp::nd_array<std::int32_t> &array)
    -> std::uint64_t {
  auto digest = std::uint64_t{1469598103934665603ULL};
  for (std::size_t index = 0; index < array.length(); ++index) {
    digest ^= static_cast<std::uint32_t>(array[index]);
    digest *= std::uint64_t{1099511628211ULL};
  }
  return digest;
}

template <typename T> auto check_sorting_dtype() -> void {
  const auto input = make_array<T>({T{3}, T{1}, T{2}}, {3});
  const auto sorted = tf::cpp::sort(input);
  check_values(sorted, {T{1}, T{2}, T{3}});
  check_values(input, {T{3}, T{1}, T{2}});
  check_values(tf::cpp::argsort(input),
               {std::int32_t{1}, std::int32_t{2}, std::int32_t{0}});

  auto inplace = input.deep_copy();
  tf::cpp::sort_inplace(inplace);
  check_values(inplace, {T{1}, T{2}, T{3}});

  const auto rows = make_array<T>({T{3}, T{1}, T{1}, T{2}, T{1}, T{0}}, {3, 2});
  const auto sorted_rows = tf::cpp::sort(rows);
  check_values(sorted_rows, {T{1}, T{0}, T{1}, T{2}, T{3}, T{1}});
  check_values(tf::cpp::argsort(rows),
               {std::int32_t{2}, std::int32_t{1}, std::int32_t{0}});

  const auto duplicates =
      make_array<T>({T{1}, T{0}, T{1}, T{0}, T{2}, T{0}, T{2}, T{0}}, {4, 2});
  check_values(tf::cpp::unique(duplicates), {T{1}, T{0}, T{2}, T{0}});

  const auto a = make_array<T>({T{1}, T{0}, T{3}, T{0}, T{5}, T{0}}, {3, 2});
  const auto b = make_array<T>({T{2}, T{0}, T{3}, T{0}, T{6}, T{0}}, {3, 2});
  check_values(tf::cpp::set_union(a, b),
               {T{1}, T{0}, T{2}, T{0}, T{3}, T{0}, T{5}, T{0}, T{6}, T{0}});
  check_values(tf::cpp::set_intersection(a, b), {T{3}, T{0}});
  check_values(tf::cpp::set_difference(a, b), {T{1}, T{0}, T{5}, T{0}});
}

} // namespace

TEST_CASE("sorting and ordered sets support every storage dtype",
          "[cpp][core][ndarray-sorting]") {
  check_sorting_dtype<std::int8_t>();
  check_sorting_dtype<std::int32_t>();
  check_sorting_dtype<std::int64_t>();
  check_sorting_dtype<float>();
  check_sorting_dtype<double>();

  const auto wide = std::int64_t{1} << 40;
  const auto wide_values =
      make_array<std::int64_t>({wide + 2, wide, wide + 1}, {3});
  check_values(tf::cpp::sort(wide_values), {wide, wide + 1, wide + 2});
  check_values(tf::cpp::argsort(wide_values),
               {std::int32_t{1}, std::int32_t{2}, std::int32_t{0}});
  const auto wide_duplicates =
      make_array<std::int64_t>({wide, wide, wide + 1}, {3});
  check_values(tf::cpp::unique(wide_duplicates), {wide, wide + 1});
}

TEST_CASE("row sorting dispatch matches lexicographic oracles",
          "[cpp][core][ndarray-sorting]") {
  SECTION("fixed widths") {
    check_row_sort_oracle<2>();
    check_row_sort_oracle<3>();
    check_row_sort_oracle<4>();
    check_row_sort_oracle<8>();
  }
  SECTION("generic width") { check_row_sort_oracle<5>(); }
}

TEST_CASE("row sorting preserves duplicates without requiring stable ties",
          "[cpp][core][ndarray-sorting]") {
  const auto input = make_array<std::int32_t>(
      {2, 0, 0, 1, 4, 0, 2, 0, 0, 1, 4, -1, 1, 4, 0}, {5, 3});
  check_values(tf::cpp::sort(input),
               {1, 4, -1, 1, 4, 0, 1, 4, 0, 2, 0, 0, 2, 0, 0});
}

TEST_CASE("floating row sorting retains partial NaN and infinity comparisons",
          "[cpp][core][ndarray-sorting]") {
  const auto infinity = std::numeric_limits<float>::infinity();
  const auto nan = std::numeric_limits<float>::quiet_NaN();

  const auto nan_rows =
      make_array<float>({nan, 7.0F, -infinity, 42.0F, 7.0F, infinity}, {2, 3});
  const auto nan_permutation = tf::cpp::argsort(nan_rows);
  check_values(nan_permutation, {std::int32_t{0}, std::int32_t{1}});
  const auto nan_sorted = tf::cpp::sort(nan_rows);
  CHECK(std::isnan(nan_sorted[0]));
  CHECK(nan_sorted[1] == 7.0F);
  CHECK(nan_sorted[2] == -infinity);
  CHECK(nan_sorted[3] == 42.0F);
  CHECK(nan_sorted[4] == 7.0F);
  CHECK(nan_sorted[5] == infinity);

  const auto infinity_rows =
      make_array<float>({infinity, 0.0F, -infinity, 0.0F}, {2, 2});
  check_values(tf::cpp::argsort(infinity_rows),
               {std::int32_t{1}, std::int32_t{0}});
}

TEST_CASE("row sorting is byte-equivalent across thread caps",
          "[cpp][core][ndarray-sorting]") {
  constexpr auto rows = std::size_t{4096};
  constexpr auto width = std::size_t{8};
  tf::buffer<std::int32_t> buffer;
  buffer.allocate(rows * width);
  for (std::size_t row = 0; row < rows; ++row) {
    buffer[row * width] = static_cast<std::int32_t>((row * 4051) % rows);
    for (std::size_t column = 1; column < width; ++column)
      buffer[row * width + column] =
          static_cast<std::int32_t>(row * 17 + column);
  }
  const auto input = tf::cpp::nd_array<std::int32_t>::from_buffer(
      std::move(buffer), {static_cast<int>(rows), static_cast<int>(width)});

  const auto cap_one =
      run_with_thread_cap(1, [&] { return tf::cpp::sort(input); });
  const auto cap_eight =
      run_with_thread_cap(8, [&] { return tf::cpp::sort(input); });
  REQUIRE(cap_one.length() == cap_eight.length());
  for (std::size_t index = 0; index < cap_one.length(); ++index)
    CHECK(cap_one[index] == cap_eight[index]);
  CHECK(row_sort_digest(cap_one) == row_sort_digest(cap_eight));
}

TEST_CASE("sorting and ordered sets preserve empty and singleton arrays",
          "[cpp][core][ndarray-sorting]") {
  const auto empty = make_array<float>({}, {0, 2});
  CHECK(tf::cpp::sort(empty).empty());
  CHECK(tf::cpp::argsort(empty).empty());
  CHECK((tf::cpp::unique(empty).raw_shape() == tf::small_vector<int, 3>{0, 2}));
  CHECK(tf::cpp::set_union(empty, empty).empty());
  CHECK(tf::cpp::set_intersection(empty, empty).empty());
  CHECK(tf::cpp::set_difference(empty, empty).empty());

  auto zero_width = make_array<std::int32_t>({}, {4, 0});
  const auto sorted_zero_width = tf::cpp::sort(zero_width);
  CHECK((sorted_zero_width.raw_shape() == tf::small_vector<int, 3>{4, 0}));
  CHECK(sorted_zero_width.empty());
  check_values(
      tf::cpp::argsort(zero_width),
      {std::int32_t{0}, std::int32_t{1}, std::int32_t{2}, std::int32_t{3}});
  tf::cpp::sort_inplace(zero_width);
  CHECK((zero_width.raw_shape() == tf::small_vector<int, 3>{4, 0}));
  CHECK((tf::cpp::unique(zero_width).raw_shape() ==
         tf::small_vector<int, 3>{1, 0}));

  const auto two_zero_width_rows = make_array<std::int32_t>({}, {2, 0});
  CHECK((tf::cpp::set_union(zero_width, two_zero_width_rows).raw_shape() ==
         tf::small_vector<int, 3>{4, 0}));
  CHECK(
      (tf::cpp::set_intersection(zero_width, two_zero_width_rows).raw_shape() ==
       tf::small_vector<int, 3>{2, 0}));
  CHECK((tf::cpp::set_difference(zero_width, two_zero_width_rows).raw_shape() ==
         tf::small_vector<int, 3>{2, 0}));

  const auto no_zero_width_rows = make_array<std::int32_t>({}, {0, 0});
  CHECK((tf::cpp::sort(no_zero_width_rows).raw_shape() ==
         tf::small_vector<int, 3>{0, 0}));
  CHECK(tf::cpp::argsort(no_zero_width_rows).empty());

  auto singleton = make_array<float>({7}, {1});
  tf::cpp::sort_inplace(singleton);
  check_values(singleton, {7.0F});
  check_values(tf::cpp::unique(singleton), {7.0F});
}

TEST_CASE("ordered sets reject incompatible row shapes",
          "[cpp][core][ndarray-sorting]") {
  const auto rows_two = make_array<float>({1, 2}, {1, 2});
  const auto rows_three = make_array<float>({1, 2, 3}, {1, 3});
  const auto flat = make_array<float>({1, 2}, {2});
  CHECK_THROWS_AS(tf::cpp::set_union(rows_two, rows_three),
                  std::invalid_argument);
  CHECK_THROWS_AS(tf::cpp::set_intersection(rows_two, flat),
                  std::invalid_argument);
  CHECK_THROWS_AS(tf::cpp::set_difference(rows_two, rows_three),
                  std::invalid_argument);
}

TEST_CASE("async argsort matches sync and resolver-selected results",
          "[cpp][core][async][ndarray-sorting]") {
  auto input = make_array<float>({3, 1, 4, 2}, {4});
  const auto expected = tf::cpp::argsort(input);

  auto default_result = [&input] {
    tbb::global_control block_workers(
        tbb::global_control::max_allowed_parallelism, 1);
    auto result = tf::cpp::async::argsort(input);
    input.destroy();
    return result;
  }();
  static_assert(
      std::is_same<decltype(default_result),
                   std::future<tf::cpp::nd_array<std::int32_t>>>::value,
      "default async argsort returns its exact native future type");
  const auto actual = default_result.get();
  REQUIRE(actual.length() == expected.length());
  for (std::size_t index = 0; index < actual.length(); ++index)
    CHECK(actual[index] == expected[index]);

  const auto custom_input = make_array<float>({3, 1, 4, 2}, {4});
  auto custom_pending =
      tf::cpp::async::argsort(tf::cpp::async::future_resolver{}, custom_input);
  static_assert(std::is_same_v<decltype(custom_pending),
                               std::future<tf::cpp::nd_array<std::int32_t>>>);
  const auto custom = custom_pending.get();
  REQUIRE(custom.length() == expected.length());
  for (std::size_t index = 0; index < custom.length(); ++index)
    CHECK(custom[index] == expected[index]);
}

TEST_CASE("async sort_inplace resolves void and propagates validation errors",
          "[cpp][core][async][ndarray-sorting]") {
  auto input = make_array<float>({3, 1, 2}, {3});
  auto result = tf::cpp::async::sort_inplace(input);
  static_assert(std::is_same<decltype(result), std::future<void>>::value,
                "default async sort_inplace returns future<void>");
  CHECK_NOTHROW(result.get());
  check_values(input, {1.0F, 2.0F, 3.0F});

  tf::cpp::nd_array<float> invalid;
  auto failure = tf::cpp::async::sort_inplace(invalid);
  CHECK_THROWS_AS(failure.get(), std::invalid_argument);

  auto custom = make_array<float>({2, 1}, {2});
  auto custom_pending =
      tf::cpp::async::sort_inplace(tf::cpp::async::future_resolver{}, custom);
  static_assert(std::is_same_v<decltype(custom_pending), std::future<void>>);
  custom_pending.get();
  check_values(custom, {1.0F, 2.0F});
}

TEST_CASE("async sorting and ordered sets retain inputs and preserve overloads",
          "[cpp][core][async][ndarray-sorting]") {
  auto input = make_array<std::int32_t>({3, 1, 2}, {3});
  auto sorted_future = tf::cpp::async::sort(input);
  static_assert(std::is_same_v<decltype(sorted_future),
                               std::future<tf::cpp::nd_array<std::int32_t>>>);
  input.destroy();
  check_values(sorted_future.get(),
               {std::int32_t{1}, std::int32_t{2}, std::int32_t{3}});

  const auto duplicates = make_array<std::int32_t>({1, 1, 2, 2, 3, 3}, {6});
  auto unique_future = tf::cpp::async::unique(duplicates);
  static_assert(std::is_same_v<decltype(unique_future),
                               std::future<tf::cpp::nd_array<std::int32_t>>>);
  check_values(unique_future.get(),
               {std::int32_t{1}, std::int32_t{2}, std::int32_t{3}});

  const auto a = make_array<std::int32_t>({1, 3, 5}, {3});
  const auto b = make_array<std::int32_t>({2, 3, 6}, {3});
  check_values(tf::cpp::async::set_union(a, b).get(),
               {std::int32_t{1}, std::int32_t{2}, std::int32_t{3},
                std::int32_t{5}, std::int32_t{6}});
  check_values(tf::cpp::async::set_intersection(a, b).get(), {std::int32_t{3}});
  check_values(tf::cpp::async::set_difference(a, b).get(),
               {std::int32_t{1}, std::int32_t{5}});

  const auto rows_two = make_array<float>({1, 2}, {1, 2});
  const auto rows_three = make_array<float>({1, 2, 3}, {1, 3});
  auto failure = tf::cpp::async::set_union(rows_two, rows_three);
  CHECK_THROWS_AS(failure.get(), std::invalid_argument);
}
