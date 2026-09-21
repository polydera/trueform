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
#include "trueform/cpp/core/async/nd_array_indexing.hpp"
#include "trueform/cpp/core/nd_array_indexing.hpp"
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
  if (a.length() != 0)
    CHECK(std::memcmp(a.raw_data(), b.raw_data(), a.length() * sizeof(T)) == 0);
}

template <typename Operation>
auto run_with_thread_cap(std::size_t thread_cap, Operation operation)
    -> decltype(operation()) {
  tbb::global_control control(tbb::global_control::max_allowed_parallelism,
                              thread_cap);
  return operation();
}

template <typename T> struct multi_take_reference_result {
  tf::small_vector<int, 3> shape;
  std::vector<T> values;
};

template <typename T>
auto old_mixed_radix_multi_take_reference(
    const tf::cpp::nd_array<T> &array,
    const std::vector<tf::cpp::multi_take_index> &indices)
    -> multi_take_reference_result<T> {
  const auto &input_shape = array.raw_shape();
  const auto dimensions = static_cast<int>(input_shape.size());
  REQUIRE(dimensions > 0);
  REQUIRE(indices.size() <= input_shape.size());

  std::vector<std::vector<int>> selections(
      static_cast<std::size_t>(dimensions));
  std::vector<int> counts(static_cast<std::size_t>(dimensions));
  tf::small_vector<int, 3> output_shape;
  const auto normalize = [](int index, int size) {
    if (index < 0)
      index += size;
    REQUIRE(index >= 0);
    REQUIRE(index < size);
    return index;
  };

  for (int dimension = 0; dimension < dimensions; ++dimension) {
    const auto slot = static_cast<std::size_t>(dimension);
    const auto axis_size = input_shape[dimension];
    if (slot >= indices.size() || indices[slot].selection_mode() ==
                                      tf::cpp::multi_take_index::mode::all) {
      counts[slot] = axis_size;
      output_shape.push_back(axis_size);
      continue;
    }
    if (indices[slot].selection_mode() ==
        tf::cpp::multi_take_index::mode::single) {
      selections[slot].push_back(
          normalize(indices[slot].single_index(), axis_size));
      counts[slot] = 1;
      continue;
    }

    for (const auto index : indices[slot].index_array())
      selections[slot].push_back(normalize(index, axis_size));
    counts[slot] = static_cast<int>(selections[slot].size());
    output_shape.push_back(counts[slot]);
  }
  if (output_shape.empty())
    output_shape.push_back(1);

  std::vector<std::size_t> input_strides(static_cast<std::size_t>(dimensions),
                                         1);
  for (int dimension = dimensions - 2; dimension >= 0; --dimension)
    input_strides[static_cast<std::size_t>(dimension)] =
        input_strides[static_cast<std::size_t>(dimension + 1)] *
        static_cast<std::size_t>(input_shape[dimension + 1]);

  auto total = std::size_t{1};
  for (const auto size : output_shape)
    total *= static_cast<std::size_t>(size);
  std::vector<T> values(total);
  for (std::size_t flat_index = 0; flat_index < total; ++flat_index) {
    auto remainder = flat_index;
    auto source_index = std::size_t{0};
    for (int dimension = dimensions - 1; dimension >= 0; --dimension) {
      const auto slot = static_cast<std::size_t>(dimension);
      const auto count = static_cast<std::size_t>(counts[slot]);
      const auto coordinate = remainder % count;
      remainder /= count;
      auto selected = static_cast<int>(coordinate);
      if (slot < indices.size() && indices[slot].selection_mode() !=
                                       tf::cpp::multi_take_index::mode::all)
        selected = selections[slot][coordinate];
      source_index += static_cast<std::size_t>(selected) * input_strides[slot];
    }
    values[flat_index] = array.raw_data()[source_index];
  }
  return {std::move(output_shape), std::move(values)};
}

template <typename T>
auto check_multi_take_reference(
    const tf::cpp::nd_array<T> &array,
    const std::vector<tf::cpp::multi_take_index> &indices) -> void {
  const auto expected = old_mixed_radix_multi_take_reference(array, indices);
  const auto actual = tf::cpp::multi_take(array, indices);
  REQUIRE(actual.raw_shape() == expected.shape);
  REQUIRE(actual.length() == expected.values.size());
  if (actual.length() != 0)
    CHECK(std::memcmp(actual.raw_data(), expected.values.data(),
                      actual.length() * sizeof(T)) == 0);
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

template <typename T> auto check_indexing_dtype() -> void {
  const auto array =
      make_array<T>({T{0}, T{1}, T{2}, T{3}, T{4}, T{5}}, {2, 3});
  const auto columns = make_array<std::int32_t>({-1, 0}, {2});
  const auto taken = tf::cpp::take(array, columns, 1);
  CHECK((taken.raw_shape() == tf::small_vector<int, 3>{2, 2}));
  check_values(taken, {T{2}, T{0}, T{5}, T{3}});

  const auto cartesian = tf::cpp::multi_take(
      array, {tf::cpp::multi_take_index{},
              tf::cpp::multi_take_index(std::vector<std::int32_t>{2, 0})});
  CHECK((cartesian.raw_shape() == tf::small_vector<int, 3>{2, 2}));
  check_values(cartesian, {T{2}, T{0}, T{5}, T{3}});

  const auto row = tf::cpp::multi_take(array, {tf::cpp::multi_take_index{-1}});
  CHECK((row.raw_shape() == tf::small_vector<int, 3>{3}));
  check_values(row, {T{3}, T{4}, T{5}});

  const auto element = tf::cpp::multi_take(
      array, {tf::cpp::multi_take_index{1}, tf::cpp::multi_take_index{-1}});
  CHECK((element.raw_shape() == tf::small_vector<int, 3>{1}));
  check_values(element, {T{5}});

  const auto per_element_indices =
      make_array<std::int32_t>({2, 0, 1, -1}, {2, 2});
  const auto along = tf::cpp::take_along_axis(array, per_element_indices, -1);
  CHECK((along.raw_shape() == tf::small_vector<int, 3>{2, 2}));
  check_values(along, {T{2}, T{0}, T{4}, T{5}});

  const auto mask = make_array<std::int8_t>({1, 0}, {2});
  const auto selected = tf::cpp::boolean_index(array, mask);
  CHECK((selected.raw_shape() == tf::small_vector<int, 3>{1, 3}));
  check_values(selected, {T{0}, T{1}, T{2}});
}

} // namespace

TEST_CASE("nd_array indexing supports every storage dtype",
          "[cpp][core][ndarray-indexing]") {
  check_indexing_dtype<std::int8_t>();
  check_indexing_dtype<std::int32_t>();
  check_indexing_dtype<std::int64_t>();
  check_indexing_dtype<float>();
  check_indexing_dtype<double>();

  const auto wide = std::int64_t{1} << 40;
  const auto wide_rows =
      make_array<std::int64_t>({wide, wide + 1, wide + 2, wide + 3}, {2, 2});
  const auto reversed = make_array<std::int32_t>({1, 0}, {2});
  check_values(tf::cpp::take(wide_rows, reversed, 0),
               {wide + 2, wide + 3, wide, wide + 1});
  const auto second = make_array<std::int8_t>({0, 1}, {2});
  check_values(tf::cpp::boolean_index(wide_rows, second), {wide + 2, wide + 3});
}

TEST_CASE("multi_take matches the old mixed-radix oracle across generic "
          "selection shapes",
          "[cpp][core][ndarray-indexing][multi-take]") {
  SECTION("rank one repeated, reordered, and negative indices") {
    const auto input =
        make_generated_array<std::int32_t>({5}, [](std::size_t index) {
          return static_cast<std::int32_t>(index);
        });
    check_multi_take_reference(
        input, {tf::cpp::multi_take_index(
                   std::vector<std::int32_t>{-1, 0, 2, 2, -5})});
  }

  SECTION("rank two all and squeezed single modes") {
    const auto input =
        make_generated_array<std::int32_t>({3, 4}, [](std::size_t index) {
          return static_cast<std::int32_t>(100 + index);
        });
    check_multi_take_reference(
        input, {tf::cpp::multi_take_index{-2}, tf::cpp::multi_take_index{}});
  }

  SECTION("rank four mixes every mode") {
    const auto input =
        make_generated_array<std::int32_t>({2, 3, 2, 4}, [](std::size_t index) {
          return static_cast<std::int32_t>(7 * index + 3);
        });
    check_multi_take_reference(
        input,
        {tf::cpp::multi_take_index(std::vector<std::int32_t>{-1, 0, -1}),
         tf::cpp::multi_take_index{-2}, tf::cpp::multi_take_index{},
         tf::cpp::multi_take_index(std::vector<std::int32_t>{3, 0, 3, -2})});
  }

  SECTION("all axes squeezed") {
    const auto input =
        make_generated_array<std::int32_t>({2, 3, 4}, [](std::size_t index) {
          return static_cast<std::int32_t>(11 * index - 9);
        });
    check_multi_take_reference(input, {tf::cpp::multi_take_index{-1},
                                       tf::cpp::multi_take_index{1},
                                       tf::cpp::multi_take_index{-3}});
  }

  SECTION("empty input axis") {
    const auto input = make_generated_array<std::int32_t>(
        {2, 0, 3}, [](std::size_t) { return std::int32_t{0}; });
    check_multi_take_reference(input, {});
  }

  SECTION("empty explicit selection") {
    const auto input =
        make_generated_array<std::int32_t>({2, 3}, [](std::size_t index) {
          return static_cast<std::int32_t>(index + 1);
        });
    check_multi_take_reference(
        input, {tf::cpp::multi_take_index(std::vector<std::int32_t>{}),
                tf::cpp::multi_take_index{}});
  }
}

TEST_CASE("large multi_take is oracle-correct and byte-identical across thread "
          "caps",
          "[cpp][core][ndarray-indexing][multi-take][parallel]") {
  const auto input =
      make_generated_array<std::int32_t>({96, 80, 48}, [](std::size_t index) {
        return static_cast<std::int32_t>(3 * index + 17);
      });
  std::vector<std::int32_t> first(80);
  std::vector<std::int32_t> second(70);
  std::vector<std::int32_t> third(40);
  for (std::size_t index = 0; index < first.size(); ++index)
    first[index] = index % 2 == 0
                       ? static_cast<std::int32_t>((index * 37 + 11) % 96)
                       : -static_cast<std::int32_t>((index * 13) % 96) - 1;
  for (std::size_t index = 0; index < second.size(); ++index)
    second[index] = static_cast<std::int32_t>((index * 29 + 7) % 80);
  for (std::size_t index = 0; index < third.size(); ++index)
    third[index] = static_cast<std::int32_t>((index * 17 + 5) % 48);
  const auto selections = std::vector<tf::cpp::multi_take_index>{
      tf::cpp::multi_take_index(std::move(first)),
      tf::cpp::multi_take_index(std::move(second)),
      tf::cpp::multi_take_index(std::move(third))};
  const auto expected = old_mixed_radix_multi_take_reference(input, selections);
  const auto run = [&] { return tf::cpp::multi_take(input, selections); };
  const auto cap_one = run_with_thread_cap(1, run);
  const auto cap_eight = run_with_thread_cap(8, run);

  REQUIRE(cap_one.length() > tf::cpp::parallel_threshold);
  REQUIRE(cap_one.raw_shape() == expected.shape);
  REQUIRE(cap_one.length() == expected.values.size());
  CHECK(std::memcmp(cap_one.raw_data(), expected.values.data(),
                    cap_one.length() * sizeof(std::int32_t)) == 0);
  check_same_storage(cap_one, cap_eight);
}

TEST_CASE("boolean_index keeps exact stable row order for sparse and dense "
          "large masks",
          "[cpp][core][ndarray-indexing][boolean-index][parallel]") {
  const auto rows =
      static_cast<int>(tf::cpp::parallel_threshold + std::size_t{37});
  constexpr int width = 2;
  const auto input =
      make_generated_array<std::int32_t>({rows, width}, [](std::size_t index) {
        const auto row = index / width;
        const auto column = index % width;
        return static_cast<std::int32_t>(row * 10 + column);
      });

  const auto check_mask = [&](const tf::cpp::nd_array<std::int8_t> &mask) {
    std::vector<std::int32_t> expected;
    for (int row = 0; row < rows; ++row) {
      if (!mask.raw_data()[static_cast<std::size_t>(row)])
        continue;
      expected.push_back(row * 10);
      expected.push_back(row * 10 + 1);
    }
    const auto run = [&] { return tf::cpp::boolean_index(input, mask); };
    const auto cap_one = run_with_thread_cap(1, run);
    const auto cap_eight = run_with_thread_cap(8, run);
    REQUIRE(cap_one.length() == expected.size());
    CHECK(cap_one.raw_shape().front() ==
          static_cast<int>(expected.size() / width));
    if (!expected.empty())
      CHECK(std::memcmp(cap_one.raw_data(), expected.data(),
                        expected.size() * sizeof(std::int32_t)) == 0);
    check_same_storage(cap_one, cap_eight);
  };

  SECTION("sparse") {
    const auto mask =
        make_generated_array<std::int8_t>({rows}, [](std::size_t row) {
          return static_cast<std::int8_t>(
              row % 997 == 3 || row + 1 == static_cast<std::size_t>(rows));
        });
    check_mask(mask);
  }

  SECTION("dense") {
    const auto mask =
        make_generated_array<std::int8_t>({rows}, [](std::size_t row) {
          return static_cast<std::int8_t>(row % 10 != 4);
        });
    check_mask(mask);
  }

  SECTION("all false") {
    const auto mask = make_generated_array<std::int8_t>(
        {rows}, [](std::size_t) { return std::int8_t{0}; });
    check_mask(mask);
  }

  SECTION("all true") {
    const auto mask = make_generated_array<std::int8_t>(
        {rows}, [](std::size_t) { return std::int8_t{1}; });
    check_mask(mask);
  }
}

TEST_CASE("take handles few wide rows and many narrow blocks identically "
          "across thread caps",
          "[cpp][core][ndarray-indexing][take][parallel]") {
  SECTION("few very wide rows") {
    const auto width = static_cast<int>(
        tf::cpp::parallel_threshold / std::size_t{2} + std::size_t{17});
    const auto input =
        make_generated_array<std::int32_t>({4, width}, [](std::size_t index) {
          return static_cast<std::int32_t>(index);
        });
    const auto indices = make_array<std::int32_t>({-1, 1}, {2});
    const auto run = [&] { return tf::cpp::take(input, indices, 0); };
    const auto cap_one = run_with_thread_cap(1, run);
    const auto cap_eight = run_with_thread_cap(8, run);
    REQUIRE(cap_one.length() > tf::cpp::parallel_threshold);
    check_same_storage(cap_one, cap_eight);
    CHECK(std::memcmp(cap_one.raw_data(),
                      input.raw_data() + static_cast<std::size_t>(3 * width),
                      static_cast<std::size_t>(width) * sizeof(std::int32_t)) ==
          0);
    CHECK(std::memcmp(cap_one.raw_data() + static_cast<std::size_t>(width),
                      input.raw_data() + static_cast<std::size_t>(width),
                      static_cast<std::size_t>(width) * sizeof(std::int32_t)) ==
          0);
  }

  SECTION("many narrow blocks") {
    const auto rows = static_cast<int>(
        tf::cpp::parallel_threshold / std::size_t{4} + std::size_t{1});
    const auto input =
        make_generated_array<std::int32_t>({rows, 4}, [](std::size_t index) {
          return static_cast<std::int32_t>(index + 23);
        });
    const auto indices = make_array<std::int32_t>({3, -3, 3, 0}, {4});
    const auto run = [&] { return tf::cpp::take(input, indices, 1); };
    const auto cap_one = run_with_thread_cap(1, run);
    const auto cap_eight = run_with_thread_cap(8, run);
    REQUIRE(cap_one.length() > tf::cpp::parallel_threshold);
    check_same_storage(cap_one, cap_eight);
    std::vector<std::int32_t> expected(cap_one.length());
    for (int row = 0; row < rows; ++row) {
      const auto input_base = static_cast<std::size_t>(row) * 4;
      expected[input_base] = input[input_base + 3];
      expected[input_base + 1] = input[input_base + 1];
      expected[input_base + 2] = input[input_base + 3];
      expected[input_base + 3] = input[input_base];
    }
    CHECK(std::memcmp(cap_one.raw_data(), expected.data(),
                      expected.size() * sizeof(std::int32_t)) == 0);
  }
}

TEST_CASE("indexing rejects oversized take outputs before allocation",
          "[cpp][core][ndarray-indexing][take]") {
  const auto input =
      make_generated_array<std::int8_t>({65535, 1, 2}, [](std::size_t index) {
        return static_cast<std::int8_t>(index);
      });
  const auto indices = make_generated_array<std::int32_t>(
      {32769}, [](std::size_t) { return std::int32_t{0}; });
  CHECK_THROWS_AS(tf::cpp::take(input, indices, 1), std::length_error);
}

TEST_CASE("zero-width indexing avoids zero-byte source and destination access",
          "[cpp][core][ndarray-indexing][empty]") {
  const auto input = make_generated_array<std::int32_t>(
      {4, 3, 0}, [](std::size_t) { return std::int32_t{0}; });
  const auto indices = make_array<std::int32_t>({2, 0}, {2});
  const auto cap_one =
      run_with_thread_cap(1, [&] { return tf::cpp::take(input, indices, 1); });
  const auto cap_eight =
      run_with_thread_cap(8, [&] { return tf::cpp::take(input, indices, 1); });
  CHECK((cap_one.raw_shape() == tf::small_vector<int, 3>{4, 2, 0}));
  CHECK(cap_one.empty());
  check_same_storage(cap_one, cap_eight);

  const auto rows =
      static_cast<int>(tf::cpp::parallel_threshold + std::size_t{37});
  const auto zero_width = make_generated_array<std::int32_t>(
      {rows, 0}, [](std::size_t) { return std::int32_t{0}; });
  const auto mask =
      make_generated_array<std::int8_t>({rows}, [](std::size_t row) {
        return static_cast<std::int8_t>(row % 3 == 1);
      });
  const auto selected_count = rows / 3 + (rows % 3 > 1);
  const auto selected_one = run_with_thread_cap(
      1, [&] { return tf::cpp::boolean_index(zero_width, mask); });
  const auto selected_eight = run_with_thread_cap(
      8, [&] { return tf::cpp::boolean_index(zero_width, mask); });
  CHECK((selected_one.raw_shape() ==
         tf::small_vector<int, 3>{selected_count, 0}));
  CHECK(selected_one.empty());
  check_same_storage(selected_one, selected_eight);
}

TEST_CASE("nd_array indexing preserves empty and singleton shapes",
          "[cpp][core][ndarray-indexing]") {
  const auto empty = make_array<float>({}, {0, 3});
  const auto no_rows = make_array<std::int32_t>({}, {0});
  const auto empty_take = tf::cpp::take(empty, no_rows, 0);
  CHECK((empty_take.raw_shape() == tf::small_vector<int, 3>{0, 3}));
  CHECK(empty_take.empty());

  const auto nonempty = make_array<float>({0, 1, 2, 3, 4, 5}, {2, 3});
  const auto no_columns = tf::cpp::take(nonempty, no_rows, 1);
  CHECK((no_columns.raw_shape() == tf::small_vector<int, 3>{2, 0}));
  CHECK(no_columns.empty());

  const auto empty_mask = make_array<std::int8_t>({}, {0});
  const auto empty_selection = tf::cpp::boolean_index(empty, empty_mask);
  CHECK((empty_selection.raw_shape() == tf::small_vector<int, 3>{0, 3}));

  const auto singleton = make_array<float>({4.0F}, {1});
  const auto last = make_array<std::int32_t>({-1}, {1});
  check_values(tf::cpp::take(singleton, last), {4.0F});
}

TEST_CASE("async nd_array indexing matches every synchronous operation",
          "[cpp][core][async][ndarray-indexing]") {
  const auto array = make_array<float>({0, 1, 2, 3, 4, 5}, {2, 3});
  const auto columns = make_array<std::int32_t>({2, 0}, {2});
  const auto rows = make_array<std::int32_t>({1, 0}, {2});
  const auto per_element = make_array<std::int32_t>({2, 0, 1, -1}, {2, 2});
  const auto mask = make_array<std::int8_t>({1, 0}, {2});
  const auto cartesian = std::vector<tf::cpp::multi_take_index>{
      tf::cpp::multi_take_index{},
      tf::cpp::multi_take_index(std::vector<std::int32_t>{2, 0})};

  auto taken = tf::cpp::async::take(array, columns, 1);
  auto default_axis = tf::cpp::async::take(array, rows);
  auto multi = tf::cpp::async::multi_take(array, cartesian);
  auto along = tf::cpp::async::take_along_axis(array, per_element, -1);
  auto selected = tf::cpp::async::boolean_index(array, mask);
  static_assert(
      std::is_same_v<decltype(taken), std::future<tf::cpp::nd_array<float>>>);
  static_assert(std::is_same_v<decltype(default_axis), decltype(taken)>);
  static_assert(std::is_same_v<decltype(multi), decltype(taken)>);
  static_assert(std::is_same_v<decltype(along), decltype(taken)>);
  static_assert(std::is_same_v<decltype(selected), decltype(taken)>);

  check_same(taken.get(), tf::cpp::take(array, columns, 1));
  check_same(default_axis.get(), tf::cpp::take(array, rows));
  check_same(multi.get(), tf::cpp::multi_take(array, cartesian));
  check_same(along.get(), tf::cpp::take_along_axis(array, per_element, -1));
  check_same(selected.get(), tf::cpp::boolean_index(array, mask));
}

TEST_CASE("every resolver-first nd_array indexing operation owns inputs "
          "before one submission",
          "[cpp][core][async][ndarray-indexing][ownership]") {
  auto submissions = std::make_shared<std::atomic<int>>(0);
  const auto make_source = [] {
    return make_array<std::int32_t>({0, 1, 2, 3, 4, 5}, {2, 3});
  };

  auto array = make_source();
  auto indices = make_array<std::int32_t>({2, 0}, {2});
  check_values(resolve_once(
                   submissions,
                   [&] {
                     array.destroy();
                     indices.destroy();
                   },
                   [&](auto resolver) {
                     return tf::cpp::async::take(resolver, array, indices, 1);
                   }),
               {2, 0, 5, 3});
  CHECK_FALSE(array.is_valid());
  CHECK_FALSE(indices.is_valid());

  array = make_source();
  auto options = std::vector<tf::cpp::multi_take_index>{
      tf::cpp::multi_take_index{},
      tf::cpp::multi_take_index(std::vector<std::int32_t>{2, 0})};
  check_values(resolve_once(
                   submissions,
                   [&] {
                     array.destroy();
                     options.clear();
                   },
                   [&](auto resolver) {
                     return tf::cpp::async::multi_take(resolver, array,
                                                       options);
                   }),
               {2, 0, 5, 3});
  CHECK_FALSE(array.is_valid());
  CHECK(options.empty());

  array = make_source();
  indices = make_array<std::int32_t>({2, 0, 1, -1}, {2, 2});
  check_values(resolve_once(
                   submissions,
                   [&] {
                     array.destroy();
                     indices.destroy();
                   },
                   [&](auto resolver) {
                     return tf::cpp::async::take_along_axis(resolver, array,
                                                            indices, -1);
                   }),
               {2, 0, 4, 5});
  CHECK_FALSE(array.is_valid());
  CHECK_FALSE(indices.is_valid());

  array = make_source();
  auto mask = make_array<std::int8_t>({1, 0}, {2});
  check_values(resolve_once(
                   submissions,
                   [&] {
                     array.destroy();
                     mask.destroy();
                   },
                   [&](auto resolver) {
                     return tf::cpp::async::boolean_index(resolver, array,
                                                          mask);
                   }),
               {0, 1, 2});
  CHECK_FALSE(array.is_valid());
  CHECK_FALSE(mask.is_valid());
  CHECK(submissions->load(std::memory_order_relaxed) == 4);
}

TEST_CASE("async nd_array indexing propagates synchronous validation errors",
          "[cpp][core][async][ndarray-indexing]") {
  const auto array = make_array<float>({0, 1, 2, 3, 4, 5}, {2, 3});
  const auto indices = make_array<std::int32_t>({0}, {1});
  auto failure = tf::cpp::async::take(array, indices, 2);
  CHECK_THROWS_AS(failure.get(), std::out_of_range);
}

TEST_CASE("nd_array indexing rejects invalid axes, shapes, and indices",
          "[cpp][core][ndarray-indexing]") {
  const auto array = make_array<float>({0, 1, 2, 3, 4, 5}, {2, 3});
  const auto valid = make_array<std::int32_t>({0}, {1});
  const auto invalid = make_array<std::int32_t>({3}, {1});
  CHECK_THROWS_AS(tf::cpp::take(array, valid, 2), std::out_of_range);
  CHECK_THROWS_AS(tf::cpp::take(array, invalid, 1), std::out_of_range);
  CHECK_THROWS_AS(tf::cpp::multi_take(array, {tf::cpp::multi_take_index{},
                                              tf::cpp::multi_take_index{},
                                              tf::cpp::multi_take_index{}}),
                  std::invalid_argument);
  CHECK_THROWS_AS(tf::cpp::multi_take(array, {tf::cpp::multi_take_index{-3}}),
                  std::out_of_range);

  const auto bad_rank = make_array<std::int32_t>({0, 1}, {2});
  const auto bad_shape = make_array<std::int32_t>({0, 1, 0}, {1, 3});
  CHECK_THROWS_AS(tf::cpp::take_along_axis(array, bad_rank, 1),
                  std::invalid_argument);
  CHECK_THROWS_AS(tf::cpp::take_along_axis(array, bad_shape, 1),
                  std::invalid_argument);

  const auto bad_mask = make_array<std::int8_t>({1}, {1});
  const auto rank_two_mask = make_array<std::int8_t>({1, 0}, {2, 1});
  CHECK_THROWS_AS(tf::cpp::boolean_index(array, bad_mask),
                  std::invalid_argument);
  CHECK_THROWS_AS(tf::cpp::boolean_index(array, rank_two_mask),
                  std::invalid_argument);
}
