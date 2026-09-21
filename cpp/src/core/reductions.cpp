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
#include "trueform/cpp/core/reductions.hpp"

#include "checked_work.hpp"

#include "trueform/core/algorithm/parallel_for_each.hpp"
#include "trueform/core/algorithm/reduce.hpp"
#include "trueform/core/buffer.hpp"
#include "trueform/core/views/mapped_range.hpp"
#include "trueform/core/views/sequence_range.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace tf::cpp {
namespace {

struct axis_layout {
  int outer;
  int reduced;
  int inner;
  tf::small_vector<int, 3> output_shape;
};

auto make_axis_layout(const tf::small_vector<int, 3> &shape, int axis,
                      const char *operation) -> axis_layout {
  const auto dimensions = static_cast<int>(shape.size());
  if (dimensions == 0)
    throw std::invalid_argument(std::string(operation) +
                                ": array must have at least one axis");
  if (axis < 0)
    axis += dimensions;
  if (axis < 0 || axis >= dimensions)
    throw std::out_of_range(std::string(operation) + ": axis out of range");

  axis_layout result{1, shape[axis], 1, {}};
  for (int dimension = 0; dimension < axis; ++dimension)
    result.outer *= shape[dimension];
  for (int dimension = axis + 1; dimension < dimensions; ++dimension)
    result.inner *= shape[dimension];
  for (int dimension = 0; dimension < dimensions; ++dimension)
    if (dimension != axis)
      result.output_shape.push_back(shape[dimension]);
  if (result.output_shape.empty())
    result.output_shape.push_back(1);
  return result;
}

template <typename Input, typename Output, typename Operation>
auto axis_reduce(const nd_array<Input> &array, int axis, Operation operation,
                 Output initial, const char *operation_name)
    -> nd_array<Output> {
  const auto layout = make_axis_layout(array.raw_shape(), axis, operation_name);
  const auto result_length =
      static_cast<std::size_t>(layout.outer) * layout.inner;
  tf::buffer<Output> buffer;
  buffer.allocate(result_length);
  auto *output = buffer.data();
  const auto *data = array.raw_data();
  const auto reduce_one = [=](int flat_index) {
    const auto outer = flat_index / layout.inner;
    const auto inner = flat_index % layout.inner;
    auto accumulator = initial;
    for (int reduced = 0; reduced < layout.reduced; ++reduced)
      accumulator =
          operation(accumulator, data[outer * layout.reduced * layout.inner +
                                      reduced * layout.inner + inner]);
    output[static_cast<std::size_t>(flat_index)] = accumulator;
  };
  tf::parallel_for_each(
      tf::make_sequence_range(static_cast<int>(result_length)), reduce_one,
      checked_work(layout.reduced));
  return nd_array<Output>::from_buffer(std::move(buffer), layout.output_shape);
}

template <typename T>
auto require_arg_values(const nd_array<T> &array, const char *operation)
    -> void {
  if (array.empty())
    throw std::invalid_argument(std::string(operation) +
                                ": cannot reduce an empty array");
}

template <typename T>
auto axis_arg_reduce(const nd_array<T> &array, int axis, bool find_minimum,
                     const char *operation) -> nd_array<std::int32_t> {
  const auto layout = make_axis_layout(array.raw_shape(), axis, operation);
  if (layout.reduced == 0)
    throw std::invalid_argument(std::string(operation) +
                                ": cannot reduce an empty axis");
  const auto result_length =
      static_cast<std::size_t>(layout.outer) * layout.inner;
  tf::buffer<std::int32_t> buffer;
  buffer.allocate(result_length);
  auto *output = buffer.data();
  const auto *data = array.raw_data();
  const auto reduce_one = [=](int flat_index) {
    const auto outer = flat_index / layout.inner;
    const auto inner = flat_index % layout.inner;
    auto best_index = 0;
    auto best_value = data[outer * layout.reduced * layout.inner + inner];
    for (int reduced = 1; reduced < layout.reduced; ++reduced) {
      const auto value = data[outer * layout.reduced * layout.inner +
                              reduced * layout.inner + inner];
      if ((find_minimum && value < best_value) ||
          (!find_minimum && value > best_value)) {
        best_value = value;
        best_index = reduced;
      }
    }
    output[static_cast<std::size_t>(flat_index)] = best_index;
  };
  tf::parallel_for_each(
      tf::make_sequence_range(static_cast<int>(result_length)), reduce_one,
      checked_work(layout.reduced));
  return nd_array<std::int32_t>::from_buffer(std::move(buffer),
                                             layout.output_shape);
}

} // namespace

template <typename T>
auto sum(const nd_array<T> &array)
    -> std::conditional_t<std::is_same_v<T, std::int8_t>, std::int32_t, T> {
  if constexpr (std::is_same_v<T, std::int8_t>) {
    std::int32_t result = 0;
    for (const auto value : array)
      result += static_cast<std::int32_t>(value);
    return result;
  } else {
    return tf::reduce(array.make_range(), std::plus<T>{}, T{0},
                      checked_work(1));
  }
}

template <typename T>
auto sum(const nd_array<T> &array, int axis) -> nd_array<
    std::conditional_t<std::is_same_v<T, std::int8_t>, std::int32_t, T>> {
  if constexpr (std::is_same_v<T, std::int8_t>) {
    return axis_reduce<T, std::int32_t>(
        array, axis,
        [](std::int32_t a, std::int8_t b) {
          return a + static_cast<std::int32_t>(b);
        },
        std::int32_t{0}, "sum");
  } else {
    return axis_reduce<T, T>(array, axis, std::plus<T>{}, T{0}, "sum");
  }
}

template <typename T> auto min(const nd_array<T> &array) -> T {
  return tf::reduce(
      array.make_range(), [](T a, T b) { return a < b ? a : b; },
      std::numeric_limits<T>::max(), checked_work(1));
}

template <typename T>
auto min(const nd_array<T> &array, int axis) -> nd_array<T> {
  return axis_reduce<T, T>(
      array, axis, [](T a, T b) { return a < b ? a : b; },
      std::numeric_limits<T>::max(), "min");
}

template <typename T> auto max(const nd_array<T> &array) -> T {
  return tf::reduce(
      array.make_range(), [](T a, T b) { return a > b ? a : b; },
      std::numeric_limits<T>::lowest(), checked_work(1));
}

template <typename T>
auto max(const nd_array<T> &array, int axis) -> nd_array<T> {
  return axis_reduce<T, T>(
      array, axis, [](T a, T b) { return a > b ? a : b; },
      std::numeric_limits<T>::lowest(), "max");
}

template <typename T> auto mean(const nd_array<T> &array) -> double {
  auto converted = tf::make_mapped_range(
      array.make_range(), [](T value) { return static_cast<double>(value); });
  const auto total =
      tf::reduce(converted, std::plus<double>{}, 0.0, checked_work(1));
  return total / static_cast<double>(array.length());
}

template <typename T>
auto mean(const nd_array<T> &array, int axis)
    -> nd_array<std::conditional_t<std::is_same_v<T, double>, double, float>> {
  const auto layout = make_axis_layout(array.raw_shape(), axis, "mean");
  const auto result_length =
      static_cast<std::size_t>(layout.outer) * layout.inner;
  tf::buffer<std::conditional_t<std::is_same_v<T, double>, double, float>>
      buffer;
  buffer.allocate(result_length);
  auto *output = buffer.data();
  const auto *data = array.raw_data();
  const auto inverse = 1.0 / static_cast<double>(layout.reduced);
  const auto reduce_one = [=](int flat_index) {
    const auto outer = flat_index / layout.inner;
    const auto inner = flat_index % layout.inner;
    double accumulator = 0.0;
    for (int reduced = 0; reduced < layout.reduced; ++reduced)
      accumulator +=
          static_cast<double>(data[outer * layout.reduced * layout.inner +
                                   reduced * layout.inner + inner]);
    output[static_cast<std::size_t>(flat_index)] = static_cast<
        std::conditional_t<std::is_same_v<T, double>, double, float>>(
        accumulator * inverse);
  };
  tf::parallel_for_each(
      tf::make_sequence_range(static_cast<int>(result_length)), reduce_one,
      checked_work(layout.reduced));
  return nd_array<std::conditional_t<std::is_same_v<T, double>, double,
                                     float>>::from_buffer(std::move(buffer),
                                                          layout.output_shape);
}

template <typename T>
auto norm(const nd_array<T> &array)
    -> std::conditional_t<std::is_same_v<T, double>, double, float> {
  auto squared = tf::make_mapped_range(array.make_range(), [](T value) {
    const auto converted = static_cast<double>(value);
    return converted * converted;
  });
  const auto total =
      tf::reduce(squared, std::plus<double>{}, 0.0, checked_work(1));
  return static_cast<
      std::conditional_t<std::is_same_v<T, double>, double, float>>(
      std::sqrt(total));
}

template <typename T>
auto norm(const nd_array<T> &array, int axis)
    -> nd_array<std::conditional_t<std::is_same_v<T, double>, double, float>> {
  const auto layout = make_axis_layout(array.raw_shape(), axis, "norm");
  const auto result_length =
      static_cast<std::size_t>(layout.outer) * layout.inner;
  tf::buffer<std::conditional_t<std::is_same_v<T, double>, double, float>>
      buffer;
  buffer.allocate(result_length);
  auto *output = buffer.data();
  const auto *data = array.raw_data();
  const auto reduce_one = [=](int flat_index) {
    const auto outer = flat_index / layout.inner;
    const auto inner = flat_index % layout.inner;
    double accumulator = 0.0;
    for (int reduced = 0; reduced < layout.reduced; ++reduced) {
      const auto value =
          static_cast<double>(data[outer * layout.reduced * layout.inner +
                                   reduced * layout.inner + inner]);
      accumulator += value * value;
    }
    output[static_cast<std::size_t>(flat_index)] = static_cast<
        std::conditional_t<std::is_same_v<T, double>, double, float>>(
        std::sqrt(accumulator));
  };
  tf::parallel_for_each(
      tf::make_sequence_range(static_cast<int>(result_length)), reduce_one,
      checked_work(layout.reduced));
  return nd_array<std::conditional_t<std::is_same_v<T, double>, double,
                                     float>>::from_buffer(std::move(buffer),
                                                          layout.output_shape);
}

template <typename T> auto argmin(const nd_array<T> &array) -> std::int32_t {
  require_arg_values(array, "argmin");
  std::int32_t best = 0;
  auto best_value = array[0];
  for (std::size_t index = 1; index < array.length(); ++index)
    if (array[index] < best_value) {
      best_value = array[index];
      best = static_cast<std::int32_t>(index);
    }
  return best;
}

template <typename T>
auto argmin(const nd_array<T> &array, int axis) -> nd_array<std::int32_t> {
  return axis_arg_reduce(array, axis, true, "argmin");
}

template <typename T> auto argmax(const nd_array<T> &array) -> std::int32_t {
  require_arg_values(array, "argmax");
  std::int32_t best = 0;
  auto best_value = array[0];
  for (std::size_t index = 1; index < array.length(); ++index)
    if (array[index] > best_value) {
      best_value = array[index];
      best = static_cast<std::int32_t>(index);
    }
  return best;
}

template <typename T>
auto argmax(const nd_array<T> &array, int axis) -> nd_array<std::int32_t> {
  return axis_arg_reduce(array, axis, false, "argmax");
}

auto any(const nd_array<std::int8_t> &array) -> int {
  for (const auto value : array)
    if (value)
      return 1;
  return 0;
}

auto any(const nd_array<std::int8_t> &array, int axis)
    -> nd_array<std::int8_t> {
  return axis_reduce<std::int8_t, std::int8_t>(
      array, axis,
      [](std::int8_t a, std::int8_t b) {
        return static_cast<std::int8_t>(a || b);
      },
      std::int8_t{0}, "any");
}

auto all(const nd_array<std::int8_t> &array) -> int {
  for (const auto value : array)
    if (!value)
      return 0;
  return 1;
}

auto all(const nd_array<std::int8_t> &array, int axis)
    -> nd_array<std::int8_t> {
  return axis_reduce<std::int8_t, std::int8_t>(
      array, axis,
      [](std::int8_t a, std::int8_t b) {
        return static_cast<std::int8_t>(a && b);
      },
      std::int8_t{1}, "all");
}

#define TF_CPP_INSTANTIATE_REDUCTIONS(T)                                       \
  template auto sum<T>(const nd_array<T> &)                                    \
      -> std::conditional_t<std::is_same_v<T, std::int8_t>, std::int32_t, T>;  \
  template auto sum<T>(const nd_array<T> &, int) -> nd_array<                  \
      std::conditional_t<std::is_same_v<T, std::int8_t>, std::int32_t, T>>;    \
  template auto min<T>(const nd_array<T> &) -> T;                              \
  template auto min<T>(const nd_array<T> &, int) -> nd_array<T>;               \
  template auto max<T>(const nd_array<T> &) -> T;                              \
  template auto max<T>(const nd_array<T> &, int) -> nd_array<T>;               \
  template auto mean<T>(const nd_array<T> &) -> double;                        \
  template auto mean<T>(const nd_array<T> &, int) -> nd_array<                 \
      std::conditional_t<std::is_same_v<T, double>, double, float>>;           \
  template auto norm<T>(const nd_array<T> &)                                   \
      -> std::conditional_t<std::is_same_v<T, double>, double, float>;         \
  template auto norm<T>(const nd_array<T> &, int) -> nd_array<                 \
      std::conditional_t<std::is_same_v<T, double>, double, float>>;           \
  template auto argmin<T>(const nd_array<T> &) -> std::int32_t;                \
  template auto argmin<T>(const nd_array<T> &, int) -> nd_array<std::int32_t>; \
  template auto argmax<T>(const nd_array<T> &) -> std::int32_t;                \
  template auto argmax<T>(const nd_array<T> &, int) -> nd_array<std::int32_t>

TF_CPP_INSTANTIATE_REDUCTIONS(std::int8_t);
TF_CPP_INSTANTIATE_REDUCTIONS(std::int32_t);
TF_CPP_INSTANTIATE_REDUCTIONS(std::int64_t);
TF_CPP_INSTANTIATE_REDUCTIONS(float);
TF_CPP_INSTANTIATE_REDUCTIONS(double);

#undef TF_CPP_INSTANTIATE_REDUCTIONS

} // namespace tf::cpp
