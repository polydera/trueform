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
#include "trueform/cpp/core/async/nd_array_creation.hpp"
#include "trueform/cpp/core/async/nd_array_structure.hpp"
#include "trueform/cpp/core/nd_array_creation.hpp"
#include "trueform/cpp/core/nd_array_structure.hpp"
#include "trueform/cpp/core/parallel_config.hpp"

#include <catch2/catch_test_macros.hpp>
#include <tbb/global_control.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <future>
#include <initializer_list>
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

template <typename T>
auto check_same(const tf::cpp::nd_array<T> &actual,
                const tf::cpp::nd_array<T> &expected) -> void {
  REQUIRE(actual.raw_shape() == expected.raw_shape());
  REQUIRE(actual.length() == expected.length());
  for (std::size_t index = 0; index < actual.length(); ++index)
    CHECK(actual[index] == expected[index]);
}

template <typename Invoke>
auto resolve_once(const std::shared_ptr<std::atomic<int>> &submissions,
                  std::function<void()> mutation, Invoke &&invoke) {
  const auto before = submissions->load(std::memory_order_relaxed);
  auto result = std::forward<Invoke>(invoke)(
      mutating_resolver{submissions, std::move(mutation)});
  CHECK(submissions->load(std::memory_order_relaxed) == before + 1);
  return result.get();
}

template <typename T> auto check_basic_factories() -> void {
  const auto zero = tf::cpp::zeros<T>({2, 2});
  CHECK((zero.raw_shape() == tf::small_vector<int, 3>{2, 2}));
  check_values(zero, {T{0}, T{0}, T{0}, T{0}});

  const auto one = tf::cpp::ones<T>({2, 2});
  check_values(one, {T{1}, T{1}, T{1}, T{1}});

  const auto filled = tf::cpp::full<T>({2, 2}, T{7});
  check_values(filled, {T{7}, T{7}, T{7}, T{7}});
}

template <typename T> auto check_range_factories() -> void {
  const auto identity = tf::cpp::eye<T>(3);
  CHECK((identity.raw_shape() == tf::small_vector<int, 3>{3, 3}));
  check_values(identity,
               {T{1}, T{0}, T{0}, T{0}, T{1}, T{0}, T{0}, T{0}, T{1}});

  const auto increasing = tf::cpp::arange<T>(T{1}, T{7}, T{2});
  check_values(increasing, {T{1}, T{3}, T{5}});
  const auto decreasing = tf::cpp::arange<T>(T{5}, T{-1}, T{-2});
  check_values(decreasing, {T{5}, T{3}, T{1}});
}

template <typename T> auto check_structural_operations() -> void {
  const auto a = make_array<T>({T{1}, T{2}, T{3}, T{4}}, {2, 2});
  const auto b = make_array<T>({T{5}, T{6}, T{7}, T{8}}, {2, 2});

  const auto stacked = tf::cpp::stack<T>({a, b}, 1);
  CHECK((stacked.raw_shape() == tf::small_vector<int, 3>{2, 2, 2}));
  check_values(stacked, {T{1}, T{2}, T{5}, T{6}, T{3}, T{4}, T{7}, T{8}});

  const auto concatenated = tf::cpp::concatenate<T>({a, b}, 1);
  CHECK((concatenated.raw_shape() == tf::small_vector<int, 3>{2, 4}));
  check_values(concatenated, {T{1}, T{2}, T{5}, T{6}, T{3}, T{4}, T{7}, T{8}});

  const auto tiled = tf::cpp::tile(a, {2, 1});
  CHECK((tiled.raw_shape() == tf::small_vector<int, 3>{4, 2}));
  check_values(tiled, {T{1}, T{2}, T{3}, T{4}, T{1}, T{2}, T{3}, T{4}});

  const auto transposed = tf::cpp::transpose(a);
  CHECK((transposed.raw_shape() == tf::small_vector<int, 3>{2, 2}));
  check_values(transposed, {T{1}, T{3}, T{2}, T{4}});

  const auto condition = make_array<std::int8_t>(
      {std::int8_t{1}, std::int8_t{0}, std::int8_t{0}, std::int8_t{1}}, {2, 2});
  const auto selected = tf::cpp::where(condition, a, b);
  check_values(selected, {T{1}, T{6}, T{7}, T{4}});

  auto copied = tf::cpp::clone(a);
  copied[0] = T{9};
  CHECK(a[0] == T{1});
  CHECK(copied[0] == T{9});
}

} // namespace

TEST_CASE("nd_array factories preserve the supported dtype matrices",
          "[cpp][core][ndarray-operations]") {
  check_basic_factories<std::int8_t>();
  check_basic_factories<std::int32_t>();
  check_basic_factories<std::int64_t>();
  check_basic_factories<float>();
  check_basic_factories<double>();

  check_range_factories<std::int32_t>();
  check_range_factories<std::int64_t>();
  check_range_factories<float>();
  check_range_factories<double>();

  const auto wide = std::int64_t{1} << 40;
  check_values(tf::cpp::full<std::int64_t>({2}, wide), {wide, wide});
  check_values(tf::cpp::arange<std::int64_t>(wide, wide + 3, 1),
               {wide, wide + 1, wide + 2});

  check_values(tf::cpp::linspace<float>(0, 1, 5),
               {0.0F, 0.25F, 0.5F, 0.75F, 1.0F});
  check_values(tf::cpp::linspace<double>(0, 1, 3), {0.0, 0.5, 1.0});
}

TEST_CASE("random arrays use native shapes and supported dtypes",
          "[cpp][core][ndarray-operations]") {
  const auto wide = std::int64_t{1} << 40;
  const auto integers = tf::cpp::random<std::int32_t>({4, 3}, 2, 8);
  const auto wide_integers =
      tf::cpp::random<std::int64_t>({4, 3}, wide, wide + 6);
  const auto floats = tf::cpp::random<float>({4, 3}, -1, 1);
  const auto doubles = tf::cpp::random<double>({4, 3}, -2, 2);

  CHECK((integers.raw_shape() == tf::small_vector<int, 3>{4, 3}));
  CHECK((wide_integers.raw_shape() == tf::small_vector<int, 3>{4, 3}));
  CHECK((floats.raw_shape() == tf::small_vector<int, 3>{4, 3}));
  CHECK((doubles.raw_shape() == tf::small_vector<int, 3>{4, 3}));
  for (const auto value : integers) {
    CHECK(value >= 2);
    CHECK(value <= 8);
  }
  for (const auto value : wide_integers) {
    CHECK(value >= wide);
    CHECK(value <= wide + 6);
  }
  for (const auto value : floats) {
    CHECK(value >= -1);
    CHECK(value <= 1);
  }
  for (const auto value : doubles) {
    CHECK(value >= -2);
    CHECK(value <= 2);
  }
}

TEST_CASE("nd_array structural operations support every storage dtype",
          "[cpp][core][ndarray-operations]") {
  check_structural_operations<std::int8_t>();
  check_structural_operations<std::int32_t>();
  check_structural_operations<std::int64_t>();
  check_structural_operations<float>();
  check_structural_operations<double>();
}

TEST_CASE("stack and concatenate preserve zero-width shapes without storage "
          "access",
          "[cpp][core][ndarray-operations][empty]") {
  const auto a = make_array<std::int32_t>({}, {2, 0});
  const auto b = make_array<std::int32_t>({}, {2, 0});

  CHECK((tf::cpp::stack<std::int32_t>({a, b}, 0).raw_shape() ==
         tf::small_vector<int, 3>{2, 2, 0}));
  CHECK((tf::cpp::stack<std::int32_t>({a, b}, 1).raw_shape() ==
         tf::small_vector<int, 3>{2, 2, 0}));
  CHECK((tf::cpp::stack<std::int32_t>({a, b}, 2).raw_shape() ==
         tf::small_vector<int, 3>{2, 0, 2}));
  CHECK((tf::cpp::concatenate<std::int32_t>({a, b}, 0).raw_shape() ==
         tf::small_vector<int, 3>{4, 0}));
  CHECK((tf::cpp::concatenate<std::int32_t>({a, b}, 1).raw_shape() ==
         tf::small_vector<int, 3>{2, 0}));

  const auto values = make_array<std::int32_t>({1, 2, 3, 4}, {2, 2});
  const auto mixed = tf::cpp::concatenate<std::int32_t>({a, values, b}, 1);
  CHECK((mixed.raw_shape() == tf::small_vector<int, 3>{2, 2}));
  check_values(mixed, {1, 2, 3, 4});
}

TEST_CASE("transpose accepts an explicit axis permutation",
          "[cpp][core][ndarray-operations]") {
  const auto array = make_array<std::int32_t>({1, 2, 3, 4, 5, 6}, {1, 2, 3});
  const auto result = tf::cpp::transpose(array, {2, 0, 1});
  CHECK((result.raw_shape() == tf::small_vector<int, 3>{3, 1, 2}));
  check_values(result, {1, 4, 2, 5, 3, 6});
}

TEST_CASE("large structural kernels are byte-identical across thread caps",
          "[cpp][core][ndarray-operations][parallel]") {
  SECTION("stack") {
    const auto rows =
        static_cast<int>(tf::cpp::parallel_threshold / std::size_t{6} + 1);
    const auto a =
        make_generated_array<std::int32_t>({rows, 2}, [](std::size_t index) {
          return static_cast<std::int32_t>(1'000'000 + index);
        });
    const auto b =
        make_generated_array<std::int32_t>({rows, 2}, [](std::size_t index) {
          return static_cast<std::int32_t>(2'000'000 + index);
        });
    const auto c =
        make_generated_array<std::int32_t>({rows, 2}, [](std::size_t index) {
          return static_cast<std::int32_t>(3'000'000 + index);
        });
    const auto run = [&] { return tf::cpp::stack<std::int32_t>({a, b, c}, 2); };
    const auto cap_one = run_with_thread_cap(1, run);
    const auto cap_eight = run_with_thread_cap(8, run);

    REQUIRE(cap_one.length() > tf::cpp::parallel_threshold);
    CHECK((cap_one.raw_shape() == tf::small_vector<int, 3>{rows, 2, 3}));
    check_same_storage(cap_one, cap_eight);
    const auto check_coordinate = [&](int row, int column, int array_index) {
      const auto output_index =
          (static_cast<std::size_t>(row) * 2 + column) * 3 + array_index;
      const auto source_index = static_cast<std::size_t>(row) * 2 + column;
      CHECK(cap_one[output_index] ==
            static_cast<std::int32_t>((array_index + 1) * 1'000'000 +
                                      source_index));
    };
    check_coordinate(0, 0, 0);
    check_coordinate(rows / 2, 1, 2);
    check_coordinate(rows - 1, 1, 1);
  }

  SECTION("concatenate") {
    const auto rows =
        static_cast<int>(tf::cpp::parallel_threshold / std::size_t{3} + 1);
    const auto a =
        make_generated_array<std::int32_t>({rows, 1}, [](std::size_t index) {
          return static_cast<std::int32_t>(1'000'000 + index);
        });
    const auto b =
        make_generated_array<std::int32_t>({rows, 2}, [](std::size_t index) {
          return static_cast<std::int32_t>(2'000'000 + index);
        });
    const auto run = [&] {
      return tf::cpp::concatenate<std::int32_t>({a, b}, 1);
    };
    const auto cap_one = run_with_thread_cap(1, run);
    const auto cap_eight = run_with_thread_cap(8, run);

    REQUIRE(cap_one.length() > tf::cpp::parallel_threshold);
    CHECK((cap_one.raw_shape() == tf::small_vector<int, 3>{rows, 3}));
    check_same_storage(cap_one, cap_eight);
    const auto check_row = [&](int row) {
      const auto output_base = static_cast<std::size_t>(row) * 3;
      const auto b_base = static_cast<std::size_t>(row) * 2;
      CHECK(cap_one[output_base] == 1'000'000 + row);
      CHECK(cap_one[output_base + 1] ==
            static_cast<std::int32_t>(2'000'000 + b_base));
      CHECK(cap_one[output_base + 2] ==
            static_cast<std::int32_t>(2'000'001 + b_base));
    };
    check_row(0);
    check_row(rows / 2);
    check_row(rows - 1);
  }

  SECTION("where") {
    const auto length = static_cast<int>(tf::cpp::parallel_threshold + 1);
    const auto condition =
        make_generated_array<std::int8_t>({length}, [](std::size_t index) {
          return static_cast<std::int8_t>((index % 4) != 1);
        });
    const auto x =
        make_generated_array<std::int32_t>({length}, [](std::size_t index) {
          return static_cast<std::int32_t>(3 * index - 7);
        });
    const auto y =
        make_generated_array<std::int32_t>({length}, [](std::size_t index) {
          return static_cast<std::int32_t>(11 - 5 * index);
        });
    const auto run = [&] { return tf::cpp::where(condition, x, y); };
    const auto cap_one = run_with_thread_cap(1, run);
    const auto cap_eight = run_with_thread_cap(8, run);

    REQUIRE(cap_one.length() > tf::cpp::parallel_threshold);
    check_same_storage(cap_one, cap_eight);
    CHECK(cap_one[0] == -7);
    CHECK(cap_one[12'345] == 11 - 5 * 12'345);
    CHECK(cap_one[12'346] == 3 * 12'346 - 7);
    CHECK(cap_one[cap_one.length() - 1] ==
          3 * static_cast<std::int32_t>(cap_one.length() - 1) - 7);
  }

  SECTION("explicit-axis 3D transpose") {
    constexpr int first = 53;
    constexpr int second = 47;
    constexpr int third = 61;
    const auto input = make_generated_array<std::int32_t>(
        {first, second, third}, [](std::size_t index) {
          return static_cast<std::int32_t>(3 * index + 7);
        });
    const auto run = [&] { return tf::cpp::transpose(input, {2, 0, 1}); };
    const auto cap_one = run_with_thread_cap(1, run);
    const auto cap_eight = run_with_thread_cap(8, run);

    REQUIRE(cap_one.length() > tf::cpp::parallel_threshold);
    CHECK((cap_one.raw_shape() ==
           tf::small_vector<int, 3>{third, first, second}));
    check_same_storage(cap_one, cap_eight);
    const auto check_coordinate = [&](int output_first, int output_second,
                                      int output_third) {
      const auto output_index =
          (static_cast<std::size_t>(output_first) * first + output_second) *
              second +
          output_third;
      const auto source_index =
          (static_cast<std::size_t>(output_second) * second + output_third) *
              third +
          output_first;
      CHECK(cap_one[output_index] ==
            static_cast<std::int32_t>(3 * source_index + 7));
    };
    check_coordinate(0, 0, 0);
    check_coordinate(17, 23, 31);
    check_coordinate(third - 1, first - 1, second - 1);
  }
}

TEST_CASE("transpose rejects incomplete or invalid axis permutations",
          "[cpp][core][ndarray-operations]") {
  const auto array = make_array<std::int32_t>({1, 2, 3, 4, 5, 6}, {1, 2, 3});
  CHECK_THROWS_AS(tf::cpp::transpose(array, {0, 1}), std::runtime_error);
  CHECK_THROWS_AS(tf::cpp::transpose(array, {0, 1, 3}), std::out_of_range);
  CHECK_THROWS_AS(tf::cpp::transpose(array, {-1, 0, 1}), std::out_of_range);
  CHECK_THROWS_AS(tf::cpp::transpose(array, {0, 1, 1}), std::invalid_argument);

  const tf::cpp::nd_array<std::int32_t> invalid;
  CHECK_THROWS_AS(tf::cpp::transpose(invalid, {}), std::invalid_argument);
}

TEST_CASE("where validates valid identically shaped inputs",
          "[cpp][core][ndarray-operations]") {
  const auto condition = make_array<std::int8_t>(
      {std::int8_t{1}, std::int8_t{0}, std::int8_t{1}, std::int8_t{0}}, {2, 2});
  const auto x = make_array<float>({1, 2, 3, 4}, {2, 2});
  const auto y = make_array<float>({10, 20, 30, 40}, {2, 2});
  check_values(tf::cpp::where(condition, x, y), {1.0F, 20.0F, 3.0F, 40.0F});

  const auto short_condition =
      make_array<std::int8_t>({std::int8_t{1}, std::int8_t{0}}, {2});
  CHECK_THROWS_AS(tf::cpp::where(short_condition, x, y), std::invalid_argument);

  const auto reshaped_y = make_array<float>({10, 20, 30, 40}, {4});
  CHECK_THROWS_AS(tf::cpp::where(condition, x, reshaped_y),
                  std::invalid_argument);

  const tf::cpp::nd_array<float> invalid;
  CHECK_THROWS_AS(tf::cpp::where(condition, invalid, y), std::invalid_argument);
}

TEST_CASE("async nd_array operations match sync overloads and factories",
          "[cpp][core][async][ndarray-operations]") {
  const auto a = make_array<std::int32_t>({1, 2, 3, 4}, {2, 2});
  const auto b = make_array<std::int32_t>({5, 6, 7, 8}, {2, 2});
  const auto condition = make_array<std::int8_t>({1, 0, 0, 1}, {2, 2});

  auto stacked = tf::cpp::async::stack(std::vector{a, b}, 1);
  auto concatenated = tf::cpp::async::concatenate(std::vector{a, b}, 1);
  auto tiled = tf::cpp::async::tile(a, {2, 1});
  auto transposed = tf::cpp::async::transpose(a);
  auto permuted = tf::cpp::async::transpose(a, {1, 0});
  auto selected = tf::cpp::async::where(condition, a, b);
  auto cloned = tf::cpp::async::clone(a);
  static_assert(std::is_same_v<decltype(stacked),
                               std::future<tf::cpp::nd_array<std::int32_t>>>);
  static_assert(std::is_same_v<decltype(concatenated), decltype(stacked)>);
  static_assert(std::is_same_v<decltype(tiled), decltype(stacked)>);
  static_assert(std::is_same_v<decltype(transposed), decltype(stacked)>);
  static_assert(std::is_same_v<decltype(permuted), decltype(stacked)>);
  static_assert(std::is_same_v<decltype(selected), decltype(stacked)>);
  static_assert(std::is_same_v<decltype(cloned), decltype(stacked)>);

  check_same(stacked.get(), tf::cpp::stack<std::int32_t>({a, b}, 1));
  check_same(concatenated.get(), tf::cpp::concatenate<std::int32_t>({a, b}, 1));
  check_same(tiled.get(), tf::cpp::tile(a, {2, 1}));
  check_same(transposed.get(), tf::cpp::transpose(a));
  check_same(permuted.get(), tf::cpp::transpose(a, {1, 0}));
  check_same(selected.get(), tf::cpp::where(condition, a, b));
  check_same(cloned.get(), tf::cpp::clone(a));

  auto zero = tf::cpp::async::zeros<float>({2, 2});
  auto one = tf::cpp::async::ones<float>({2, 2});
  auto filled = tf::cpp::async::full<float>({2, 2}, 3.0F);
  auto identity = tf::cpp::async::eye<float>(3);
  auto increasing = tf::cpp::async::arange<float>(1, 7, 2);
  auto spaced = tf::cpp::async::linspace<float>(0, 1, 5);
  auto generated = tf::cpp::async::random<float>({3, 2}, -1, 1);
  static_assert(
      std::is_same_v<decltype(zero), std::future<tf::cpp::nd_array<float>>>);
  static_assert(std::is_same_v<decltype(one), decltype(zero)>);
  static_assert(std::is_same_v<decltype(filled), decltype(zero)>);
  static_assert(std::is_same_v<decltype(identity), decltype(zero)>);
  static_assert(std::is_same_v<decltype(increasing), decltype(zero)>);
  static_assert(std::is_same_v<decltype(spaced), decltype(zero)>);
  static_assert(std::is_same_v<decltype(generated), decltype(zero)>);

  check_same(zero.get(), tf::cpp::zeros<float>({2, 2}));
  check_same(one.get(), tf::cpp::ones<float>({2, 2}));
  check_same(filled.get(), tf::cpp::full<float>({2, 2}, 3.0F));
  check_same(identity.get(), tf::cpp::eye<float>(3));
  check_same(increasing.get(), tf::cpp::arange<float>(1, 7, 2));
  check_same(spaced.get(), tf::cpp::linspace<float>(0, 1, 5));
  const auto random = generated.get();
  CHECK((random.raw_shape() == tf::small_vector<int, 3>{3, 2}));
  for (const auto value : random) {
    CHECK(value >= -1);
    CHECK(value <= 1);
  }
}

TEST_CASE("every resolver-first nd_array operation owns inputs before one "
          "submission",
          "[cpp][core][async][ndarray-operations][ownership]") {
  auto submissions = std::make_shared<std::atomic<int>>(0);
  const auto make_arrays = [] {
    return std::vector<tf::cpp::nd_array<std::int32_t>>{
        make_array<std::int32_t>({1, 2}, {2}),
        make_array<std::int32_t>({3, 4}, {2})};
  };

  auto arrays = make_arrays();
  check_values(resolve_once(
                   submissions,
                   [&] {
                     for (auto &array : arrays)
                       array.destroy();
                     arrays.clear();
                   },
                   [&](auto resolver) {
                     return tf::cpp::async::stack(resolver, arrays, 0);
                   }),
               {1, 2, 3, 4});
  CHECK(arrays.empty());

  arrays = make_arrays();
  check_values(resolve_once(
                   submissions,
                   [&] {
                     for (auto &array : arrays)
                       array.destroy();
                     arrays.clear();
                   },
                   [&](auto resolver) {
                     return tf::cpp::async::concatenate(resolver, arrays, 0);
                   }),
               {1, 2, 3, 4});
  CHECK(arrays.empty());

  auto array = make_array<std::int32_t>({1, 2, 3, 4}, {2, 2});
  auto repetitions = std::vector<int>{2, 1};
  check_values(resolve_once(
                   submissions,
                   [&] {
                     array.destroy();
                     repetitions.clear();
                   },
                   [&](auto resolver) {
                     return tf::cpp::async::tile(resolver, array, repetitions);
                   }),
               {1, 2, 3, 4, 1, 2, 3, 4});
  CHECK_FALSE(array.is_valid());
  CHECK(repetitions.empty());

  array = make_array<std::int32_t>({1, 2, 3, 4}, {2, 2});
  check_values(resolve_once(
                   submissions, [&] { array.destroy(); },
                   [&](auto resolver) {
                     return tf::cpp::async::transpose(resolver, array);
                   }),
               {1, 3, 2, 4});
  CHECK_FALSE(array.is_valid());

  array = make_array<std::int32_t>({1, 2, 3, 4}, {2, 2});
  auto axes = std::vector<int>{1, 0};
  check_values(resolve_once(
                   submissions,
                   [&] {
                     array.destroy();
                     axes.clear();
                   },
                   [&](auto resolver) {
                     return tf::cpp::async::transpose(resolver, array, axes);
                   }),
               {1, 3, 2, 4});
  CHECK_FALSE(array.is_valid());
  CHECK(axes.empty());

  auto condition = make_array<std::int8_t>({1, 0}, {2});
  auto x = make_array<std::int32_t>({5, 6}, {2});
  auto y = make_array<std::int32_t>({7, 8}, {2});
  check_values(resolve_once(
                   submissions,
                   [&] {
                     condition.destroy();
                     x.destroy();
                     y.destroy();
                   },
                   [&](auto resolver) {
                     return tf::cpp::async::where(resolver, condition, x, y);
                   }),
               {5, 8});
  CHECK_FALSE(condition.is_valid());
  CHECK_FALSE(x.is_valid());
  CHECK_FALSE(y.is_valid());

  array = make_array<std::int32_t>({1, 2}, {2});
  check_values(resolve_once(
                   submissions, [&] { array.destroy(); },
                   [&](auto resolver) {
                     return tf::cpp::async::clone(resolver, array);
                   }),
               {1, 2});
  CHECK_FALSE(array.is_valid());

  auto shape = tf::small_vector<int, 3>{2, 2};
  check_values(resolve_once(
                   submissions, [&] { shape.clear(); },
                   [&](auto resolver) {
                     return tf::cpp::async::zeros<float>(resolver, shape);
                   }),
               {0.0F, 0.0F, 0.0F, 0.0F});
  CHECK(shape.empty());

  shape = {2, 2};
  check_values(resolve_once(
                   submissions, [&] { shape.clear(); },
                   [&](auto resolver) {
                     return tf::cpp::async::ones<float>(resolver, shape);
                   }),
               {1.0F, 1.0F, 1.0F, 1.0F});
  CHECK(shape.empty());

  shape = {2};
  check_values(resolve_once(
                   submissions, [&] { shape.clear(); },
                   [&](auto resolver) {
                     return tf::cpp::async::full<float>(resolver, shape, 4.0F);
                   }),
               {4.0F, 4.0F});
  CHECK(shape.empty());

  check_values(resolve_once(
                   submissions, [] {},
                   [](auto resolver) {
                     return tf::cpp::async::eye<float>(resolver, 2);
                   }),
               {1.0F, 0.0F, 0.0F, 1.0F});
  check_values(resolve_once(
                   submissions, [] {},
                   [](auto resolver) {
                     return tf::cpp::async::arange<float>(resolver, 1, 6, 2);
                   }),
               {1.0F, 3.0F, 5.0F});
  check_values(resolve_once(
                   submissions, [] {},
                   [](auto resolver) {
                     return tf::cpp::async::linspace<float>(resolver, 0, 1, 3);
                   }),
               {0.0F, 0.5F, 1.0F});

  shape = {2, 2};
  const auto random = resolve_once(
      submissions, [&] { shape.clear(); },
      [&](auto resolver) {
        return tf::cpp::async::random<float>(resolver, shape, -1, 1);
      });
  CHECK(shape.empty());
  CHECK((random.raw_shape() == tf::small_vector<int, 3>{2, 2}));
  for (const auto value : random) {
    CHECK(value >= -1);
    CHECK(value <= 1);
  }
  CHECK(submissions->load(std::memory_order_relaxed) == 14);
}

TEST_CASE("async nd_array operations propagate synchronous validation errors",
          "[cpp][core][async][ndarray-operations]") {
  const auto array = make_array<std::int32_t>({1, 2, 3, 4}, {2, 2});
  auto bad_transpose = tf::cpp::async::transpose(array, {0});
  CHECK_THROWS_AS(bad_transpose.get(), std::runtime_error);
  auto bad_factory = tf::cpp::async::zeros<float>({2, -1});
  CHECK_THROWS_AS(bad_factory.get(), std::invalid_argument);
}

TEST_CASE("nd_array factories reject negative dimensions before allocation",
          "[cpp][core][ndarray-operations]") {
  CHECK_THROWS_AS(tf::cpp::zeros<float>({2, -1}), std::invalid_argument);
  CHECK_THROWS_AS(tf::cpp::ones<float>({-1}), std::invalid_argument);
  CHECK_THROWS_AS(tf::cpp::full<double>({3, -2}, 1.0), std::invalid_argument);
  CHECK_THROWS_AS(tf::cpp::random<float>({-4, 3}, 0.0F, 1.0F),
                  std::invalid_argument);
  CHECK_THROWS_AS(tf::cpp::eye<std::int32_t>(-1), std::invalid_argument);
  CHECK_THROWS_AS(tf::cpp::linspace<double>(0.0, 1.0, -1),
                  std::invalid_argument);

  const auto array = make_array<std::int32_t>({1, 2}, {2});
  CHECK_THROWS_AS(tf::cpp::tile(array, {-1}), std::invalid_argument);
}
