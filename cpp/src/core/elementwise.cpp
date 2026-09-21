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
#include "trueform/cpp/core/elementwise/arithmetic.hpp"
#include "trueform/cpp/core/elementwise/assign.hpp"
#include "trueform/cpp/core/elementwise/detail/broadcast.hpp"
#include "trueform/cpp/core/elementwise/cast.hpp"
#include "trueform/cpp/core/elementwise/comparison.hpp"
#include "trueform/cpp/core/elementwise/logical.hpp"
#include "trueform/cpp/core/elementwise/unary.hpp"
#include "trueform/cpp/core/elementwise/vector.hpp"

#include "checked_work.hpp"
#include "shape_size.hpp"

#include "trueform/core/algorithm/parallel_fill.hpp"

#include "trueform/core/algorithm/parallel_for.hpp"
#include "trueform/core/algorithm/parallel_for_each.hpp"
#include "trueform/core/algorithm/parallel_transform.hpp"
#include "trueform/core/buffer.hpp"
#include "trueform/core/checked.hpp"
#include "trueform/core/inverted.hpp"
#include "trueform/core/range.hpp"
#include "trueform/core/sqrt.hpp"
#include "trueform/core/transformation_view.hpp"
#include "trueform/core/views/sequence_range.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace tf::cpp {
namespace detail {

auto broadcast_shape(const tf::small_vector<int, 3> &a,
                     const tf::small_vector<int, 3> &b)
    -> tf::small_vector<int, 3> {
  const auto a_dimensions = static_cast<int>(a.size());
  const auto b_dimensions = static_cast<int>(b.size());
  const auto dimensions = std::max(a_dimensions, b_dimensions);
  tf::small_vector<int, 3> output;
  for (int dimension = 0; dimension < dimensions; ++dimension) {
    const auto a_size = dimension < dimensions - a_dimensions
                            ? 1
                            : a[dimension - (dimensions - a_dimensions)];
    const auto b_size = dimension < dimensions - b_dimensions
                            ? 1
                            : b[dimension - (dimensions - b_dimensions)];
    if (a_size == b_size)
      output.push_back(a_size);
    else if (a_size == 1)
      output.push_back(b_size);
    else if (b_size == 1)
      output.push_back(a_size);
    else
      throw std::runtime_error("shapes not broadcastable");
  }
  return output;
}

auto compute_strides(const tf::small_vector<int, 3> &shape)
    -> tf::small_vector<int, 3> {
  const auto dimensions = static_cast<int>(shape.size());
  tf::small_vector<int, 3> strides;
  strides.resize(static_cast<std::size_t>(dimensions));
  int stride = 1;
  for (int dimension = dimensions - 1; dimension >= 0; --dimension) {
    strides[dimension] = stride;
    stride *= shape[dimension];
  }
  return strides;
}

auto broadcast_strides(const tf::small_vector<int, 3> &source_shape,
                       const tf::small_vector<int, 3> &output_shape)
    -> tf::small_vector<int, 3> {
  const auto dimensions = static_cast<int>(output_shape.size());
  const auto source_dimensions = static_cast<int>(source_shape.size());
  const auto source_strides = compute_strides(source_shape);
  tf::small_vector<int, 3> result;
  result.resize(static_cast<std::size_t>(dimensions));
  for (int dimension = 0; dimension < dimensions; ++dimension) {
    const auto source_dimension = dimension - (dimensions - source_dimensions);
    result[dimension] =
        source_dimension < 0 || source_shape[source_dimension] == 1
            ? 0
            : source_strides[source_dimension];
  }
  return result;
}

auto broadcast_index(int flat_index,
                     const tf::small_vector<int, 3> &output_strides,
                     const tf::small_vector<int, 3> &broadcasted_strides,
                     int dimensions) -> int {
  int source_index = 0;
  for (int dimension = 0; dimension < dimensions; ++dimension) {
    const auto coordinate = flat_index / output_strides[dimension];
    flat_index %= output_strides[dimension];
    source_index += coordinate * broadcasted_strides[dimension];
  }
  return source_index;
}

auto shapes_equal(const tf::small_vector<int, 3> &a,
                  const tf::small_vector<int, 3> &b) -> bool {
  if (a.size() != b.size())
    return false;
  for (std::size_t dimension = 0; dimension < a.size(); ++dimension)
    if (a[dimension] != b[dimension])
      return false;
  return true;
}

auto total_size(const tf::small_vector<int, 3> &shape) -> int {
  int total = 1;
  for (const auto dimension : shape)
    total *= dimension;
  return total;
}

} // namespace detail

namespace {

enum class broadcast_mapping_kind {
  flat_linear,
  repeated_trailing_tile,
  generic
};

struct broadcast_mapping_plan {
  tf::small_vector<int, 3> strides;
  broadcast_mapping_kind kind;
  int period;
};

struct broadcast_traversal_plan {
  tf::small_vector<int, 3> output_shape;
  tf::small_vector<int, 3> output_strides;
  broadcast_mapping_plan a;
  broadcast_mapping_plan b;
};

auto make_broadcast_mapping_plan(const tf::small_vector<int, 3> &output_shape,
                                 const tf::small_vector<int, 3> &output_strides,
                                 const tf::small_vector<int, 3> &source_shape)
    -> broadcast_mapping_plan {
  auto strides = detail::broadcast_strides(source_shape, output_shape);
  auto flat_linear = true;
  for (std::size_t dimension = 0; dimension < output_shape.size();
       ++dimension) {
    if (output_shape[dimension] > 1 &&
        strides[dimension] != output_strides[dimension]) {
      flat_linear = false;
      break;
    }
  }
  if (flat_linear)
    return {std::move(strides), broadcast_mapping_kind::flat_linear, 0};

  const auto period = detail::total_size(source_shape);
  auto repeated_trailing_tile = period > 0;
  for (std::size_t dimension = 0;
       repeated_trailing_tile && dimension < output_shape.size(); ++dimension) {
    if (output_shape[dimension] > 1 &&
        strides[dimension] != output_strides[dimension] % period)
      repeated_trailing_tile = false;
  }
  if (repeated_trailing_tile)
    return {std::move(strides), broadcast_mapping_kind::repeated_trailing_tile,
            period};

  return {std::move(strides), broadcast_mapping_kind::generic, 0};
}

auto make_broadcast_traversal_plan(const tf::small_vector<int, 3> &output_shape,
                                   const tf::small_vector<int, 3> &a_shape,
                                   const tf::small_vector<int, 3> &b_shape)
    -> broadcast_traversal_plan {
  auto output_strides = detail::compute_strides(output_shape);
  auto a = make_broadcast_mapping_plan(output_shape, output_strides, a_shape);
  auto b = make_broadcast_mapping_plan(output_shape, output_strides, b_shape);
  return {output_shape, std::move(output_strides), std::move(a), std::move(b)};
}

class broadcast_cursor {
public:
  explicit broadcast_cursor(const broadcast_traversal_plan &plan)
      : _plan(plan) {
    _coordinates.resize(plan.output_shape.size());
  }

  auto initialize(int flat_index) -> void {
    _a_offset = 0;
    _b_offset = 0;
    auto remaining = flat_index;
    for (std::size_t dimension = 0; dimension < _plan.output_shape.size();
         ++dimension) {
      const auto coordinate = remaining / _plan.output_strides[dimension];
      remaining %= _plan.output_strides[dimension];
      _coordinates[dimension] = coordinate;
      if (!a_is_flat())
        _a_offset += coordinate * _plan.a.strides[dimension];
      if (!b_is_flat())
        _b_offset += coordinate * _plan.b.strides[dimension];
    }
  }

  auto a_index(int flat_index) const -> int {
    return a_is_flat() ? flat_index : _a_offset;
  }

  auto b_index(int flat_index) const -> int {
    return b_is_flat() ? flat_index : _b_offset;
  }

  auto a_innermost_stride() const -> int {
    return a_is_flat() ? 1 : _plan.a.strides.back();
  }

  auto b_innermost_stride() const -> int {
    return b_is_flat() ? 1 : _plan.b.strides.back();
  }

  auto innermost_remaining() const -> int {
    return _plan.output_shape.back() - _coordinates.back();
  }

  auto advance(int count) -> void {
    const auto innermost = static_cast<int>(_coordinates.size()) - 1;
    _coordinates[innermost] += count;
    if (!a_is_flat())
      _a_offset += count * _plan.a.strides[innermost];
    if (!b_is_flat())
      _b_offset += count * _plan.b.strides[innermost];
    if (_coordinates[innermost] < _plan.output_shape[innermost])
      return;

    _coordinates[innermost] = 0;
    if (!a_is_flat())
      _a_offset -=
          _plan.a.strides[innermost] * _plan.output_shape[innermost];
    if (!b_is_flat())
      _b_offset -=
          _plan.b.strides[innermost] * _plan.output_shape[innermost];

    for (auto dimension = innermost - 1; dimension >= 0; --dimension) {
      ++_coordinates[dimension];
      if (!a_is_flat())
        _a_offset += _plan.a.strides[dimension];
      if (!b_is_flat())
        _b_offset += _plan.b.strides[dimension];
      if (_coordinates[dimension] < _plan.output_shape[dimension])
        return;

      _coordinates[dimension] = 0;
      if (!a_is_flat())
        _a_offset -=
            _plan.a.strides[dimension] * _plan.output_shape[dimension];
      if (!b_is_flat())
        _b_offset -=
            _plan.b.strides[dimension] * _plan.output_shape[dimension];
    }
  }

private:
  auto a_is_flat() const -> bool {
    return _plan.a.kind == broadcast_mapping_kind::flat_linear;
  }

  auto b_is_flat() const -> bool {
    return _plan.b.kind == broadcast_mapping_kind::flat_linear;
  }

  const broadcast_traversal_plan &_plan;
  tf::small_vector<int, 3> _coordinates;
  int _a_offset = 0;
  int _b_offset = 0;
};

/// A carrier that costs nothing keeps the whole range serial, which is what a
/// walk that must stay sequential asks for.
constexpr auto sequential = checked_work(0);

template <bool AFlat, typename Function, std::size_t... Offsets>
auto apply_static_broadcast_tile(int base, const Function &function,
                                 std::index_sequence<Offsets...>) -> void {
  (function(
       base + static_cast<int>(Offsets),
       AFlat ? base + static_cast<int>(Offsets) : static_cast<int>(Offsets),
       AFlat ? static_cast<int>(Offsets) : base + static_cast<int>(Offsets)),
   ...);
}

/// The tile walk carries `period` outputs per block, so the schedule stated in
/// outputs becomes a schedule in blocks.
auto tiled_schedule(tf::checked_t schedule, int period) -> tf::checked_t {
  const auto blocks = static_cast<unsigned long>(period);
  return {schedule.serial_below / blocks +
          (schedule.serial_below % blocks != 0)};
}

template <int Period, bool AFlat, typename Function>
auto for_each_static_broadcast_tile(int total, tf::checked_t schedule,
                                    Function function) -> void {
  const auto process_blocks = [function](int begin, int end) {
    for (auto block = begin; block < end; ++block)
      apply_static_broadcast_tile<AFlat>(block * Period, function,
                                         std::make_index_sequence<Period>{});
  };
  tf::parallel_for(0, total / Period, process_blocks,
                   tiled_schedule(schedule, Period));
}

template <bool AFlat, typename Function>
auto for_each_runtime_broadcast_tile(int total, int period,
                                     tf::checked_t schedule, Function function)
    -> void {
  const auto process_blocks = [function, period](int begin, int end) {
    for (auto block = begin; block < end; ++block) {
      const auto base = block * period;
      for (auto offset = 0; offset < period; ++offset) {
        const auto flat_index = base + offset;
        function(flat_index, AFlat ? flat_index : offset,
                 AFlat ? offset : flat_index);
      }
    }
  };
  tf::parallel_for(0, total / period, process_blocks,
                   tiled_schedule(schedule, period));
}

template <bool AFlat, typename Function>
auto for_each_broadcast_tile(int total, int period, tf::checked_t schedule,
                             Function function) -> void {
  switch (period) {
  case 1:
    return for_each_static_broadcast_tile<1, AFlat>(total, schedule, function);
  case 2:
    return for_each_static_broadcast_tile<2, AFlat>(total, schedule, function);
  case 3:
    return for_each_static_broadcast_tile<3, AFlat>(total, schedule, function);
  case 4:
    return for_each_static_broadcast_tile<4, AFlat>(total, schedule, function);
  default:
    return for_each_runtime_broadcast_tile<AFlat>(total, period, schedule,
                                                  function);
  }
}

template <typename Function>
auto for_each_broadcast_index(int total, const broadcast_traversal_plan &plan,
                              tf::checked_t schedule, Function function)
    -> void {
  if (total == 0)
    return;

  if (plan.a.kind == broadcast_mapping_kind::flat_linear &&
      plan.b.kind == broadcast_mapping_kind::flat_linear) {
    const auto process_range = [function](int begin, int end) {
      for (auto index = begin; index < end; ++index)
        function(index, index, index);
    };
    return tf::parallel_for(0, total, process_range, schedule);
  }

  if (plan.a.kind == broadcast_mapping_kind::flat_linear &&
      plan.b.kind == broadcast_mapping_kind::repeated_trailing_tile)
    return for_each_broadcast_tile<true>(total, plan.b.period, schedule,
                                         function);

  if (plan.b.kind == broadcast_mapping_kind::flat_linear &&
      plan.a.kind == broadcast_mapping_kind::repeated_trailing_tile)
    return for_each_broadcast_tile<false>(total, plan.a.period, schedule,
                                          function);

  const auto process_range = [plan, function](int begin, int end) {
    broadcast_cursor cursor(plan);
    cursor.initialize(begin);
    auto index = begin;
    while (index < end) {
      const auto run = std::min(cursor.innermost_remaining(), end - index);
      auto a_index = cursor.a_index(index);
      auto b_index = cursor.b_index(index);
      const auto a_stride = cursor.a_innermost_stride();
      const auto b_stride = cursor.b_innermost_stride();
      for (int offset = 0; offset < run; ++offset) {
        function(index + offset, a_index, b_index);
        a_index += a_stride;
        b_index += b_stride;
      }
      cursor.advance(run);
      index += run;
    }
  };
  tf::parallel_for(0, total, process_range, schedule);
}

template <typename Output, typename T, typename BinaryOperation>
auto binary_transform(const nd_array<T> &a, const nd_array<T> &b,
                      BinaryOperation operation) -> nd_array<Output> {
  auto output_shape = detail::broadcast_shape(a.raw_shape(), b.raw_shape());
  const auto total = detail::total_size(output_shape);
  tf::buffer<Output> buffer;
  buffer.allocate(static_cast<std::size_t>(total));
  auto *output = buffer.data();
  const auto *a_data = a.raw_data();
  const auto *b_data = b.raw_data();

  if (detail::shapes_equal(a.raw_shape(), b.raw_shape())) {
    tf::parallel_for_each(
        tf::make_sequence_range(total),
        [=](int index) {
          output[index] = operation(a_data[index], b_data[index]);
        },
        checked_work(1));
  } else {
    const auto plan = make_broadcast_traversal_plan(output_shape, a.raw_shape(),
                                                    b.raw_shape());
    for_each_broadcast_index(
        total, plan, checked_work(1), [=](int index, int a_index, int b_index) {
          output[index] = operation(a_data[a_index], b_data[b_index]);
        });
  }
  return nd_array<Output>::from_buffer(std::move(buffer),
                                       std::move(output_shape));
}

template <typename T, typename BinaryOperation>
auto binary_transform_inplace(nd_array<T> &a, const nd_array<T> &b,
                              BinaryOperation operation) -> void {
  const auto output_shape = detail::broadcast_shape(a.raw_shape(), b.raw_shape());
  if (!detail::shapes_equal(a.raw_shape(), output_shape))
    throw std::runtime_error("in-place op requires lhs shape == output shape");

  const auto total = a.size();
  auto *a_data = a.raw_data();
  const auto *b_data = b.raw_data();
  if (detail::shapes_equal(a.raw_shape(), b.raw_shape())) {
    tf::parallel_for_each(
        tf::make_sequence_range(total),
        [=](int index) {
          a_data[index] = operation(a_data[index], b_data[index]);
        },
        checked_work(1));
  } else {
    const auto plan = make_broadcast_traversal_plan(output_shape, a.raw_shape(),
                                                    b.raw_shape());
    for_each_broadcast_index(
        total, plan, checked_work(1), [=](int index, int, int b_index) {
          a_data[index] = operation(a_data[index], b_data[b_index]);
        });
  }
}

template <typename Output, typename T, typename UnaryOperation>
auto unary_transform(const nd_array<T> &array, UnaryOperation operation)
    -> nd_array<Output> {
  tf::buffer<Output> buffer;
  buffer.allocate(array.length());
  auto *output = buffer.data();
  const auto *input = array.raw_data();
  tf::parallel_transform(tf::make_range(input, array.length()),
                         tf::make_range(output, array.length()), operation,
                         checked_work(1));
  return nd_array<Output>::from_buffer(std::move(buffer), array.raw_shape());
}

template <typename T, typename UnaryOperation>
auto unary_transform_inplace(nd_array<T> &array, UnaryOperation operation)
    -> void {
  auto *data = array.raw_data();
  tf::parallel_for_each(
      tf::make_sequence_range(array.size()),
      [=](int index) { data[index] = operation(data[index]); },
      checked_work(1));
}

template <typename T, typename BinaryOperation>
auto scalar_transform(const nd_array<T> &array, T scalar,
                      BinaryOperation operation) -> nd_array<T> {
  return unary_transform<T>(array,
                            [=](T value) { return operation(value, scalar); });
}

template <typename T, typename BinaryOperation>
auto scalar_transform_inplace(nd_array<T> &array, T scalar,
                              BinaryOperation operation) -> void {
  unary_transform_inplace(array,
                          [=](T value) { return operation(value, scalar); });
}

template <typename T, typename Comparison>
auto compare(const nd_array<T> &a, const nd_array<T> &b, Comparison comparison)
    -> nd_array<std::int8_t> {
  return binary_transform<std::int8_t>(
      a, b, [=](T x, T y) -> std::int8_t { return comparison(x, y) ? 1 : 0; });
}

template <typename T, typename Comparison>
auto compare_scalar(const nd_array<T> &a, T scalar, Comparison comparison)
    -> nd_array<std::int8_t> {
  return unary_transform<std::int8_t>(a, [=](T value) -> std::int8_t {
    return comparison(value, scalar) ? 1 : 0;
  });
}

template <typename T>
auto assignment_row_count(const nd_array<T> &target) -> int {
  if (!target.is_valid() || target.ndim() < 1)
    throw std::invalid_argument("assign: target must be a valid array");
  return target.shape_at(0);
}

auto normalized_assignment_indices(const nd_array<std::int32_t> &indices,
                                   int row_count) -> std::vector<std::size_t> {
  if (!indices.is_valid())
    throw std::invalid_argument("assign: indices must be a valid array");

  // The count is exact, so the rows are written into place rather than
  // appended one at a time.
  std::vector<std::size_t> normalized(indices.length());
  for (std::size_t index = 0; index < indices.length(); ++index) {
    auto value = static_cast<std::int64_t>(indices[index]);
    if (value < 0)
      value += row_count;
    if (value < 0 || value >= row_count)
      throw std::out_of_range("assign: index out of range");
    normalized[index] = static_cast<std::size_t>(value);
  }
  return normalized;
}

template <typename T>
auto require_assignment_mask(const nd_array<T> &target,
                             const nd_array<std::int8_t> &mask) -> int {
  const auto row_count = assignment_row_count(target);
  if (!mask.is_valid() || mask.ndim() != 1 || mask.shape_at(0) != row_count)
    throw std::invalid_argument(
        "assign: mask must be 1D and match the target row count");
  return row_count;
}

} // namespace

#define TF_CPP_DEFINE_BINARY(NAME, OPERATION)                                  \
  template <typename T>                                                        \
  auto NAME(const nd_array<T> &a, const nd_array<T> &b) -> nd_array<T> {       \
    return binary_transform<T>(a, b, OPERATION);                               \
  }                                                                            \
  template <typename T>                                                        \
  auto NAME##_inplace(nd_array<T> &a, const nd_array<T> &b) -> void {          \
    binary_transform_inplace(a, b, OPERATION);                                 \
  }                                                                            \
  template <typename T>                                                        \
  auto NAME##_scalar(const nd_array<T> &a, T scalar) -> nd_array<T> {          \
    return scalar_transform(a, scalar, OPERATION);                             \
  }                                                                            \
  template <typename T>                                                        \
  auto NAME##_scalar_inplace(nd_array<T> &a, T scalar) -> void {               \
    scalar_transform_inplace(a, scalar, OPERATION);                            \
  }

TF_CPP_DEFINE_BINARY(add, std::plus<T>{})
TF_CPP_DEFINE_BINARY(sub, std::minus<T>{})
TF_CPP_DEFINE_BINARY(mul, std::multiplies<T>{})
TF_CPP_DEFINE_BINARY(div, std::divides<T>{})
TF_CPP_DEFINE_BINARY(mod, [](T x, T y) -> T {
  if constexpr (std::is_floating_point_v<T>)
    return std::fmod(x, y);
  else
    return x % y;
})

#undef TF_CPP_DEFINE_BINARY

template <typename T>
auto mat_mul(const nd_array<T> &a, const nd_array<T> &b) -> nd_array<T> {
  const auto &a_shape = a.raw_shape();
  const auto &b_shape = b.raw_shape();
  const auto a_dimensions = static_cast<int>(a_shape.size());
  const auto b_dimensions = static_cast<int>(b_shape.size());
  if (a_dimensions < 2 || b_dimensions < 2)
    throw std::runtime_error("mat_mul requires at least 2D arrays");

  const auto rows = a_shape[a_dimensions - 2];
  const auto shared = a_shape[a_dimensions - 1];
  const auto b_shared = b_shape[b_dimensions - 2];
  const auto columns = b_shape[b_dimensions - 1];
  if (shared != b_shared)
    throw std::runtime_error("mat_mul inner dimensions mismatch");

  const auto a_batch = a_dimensions == 2 ? 1 : a_shape[0];
  const auto b_batch = b_dimensions == 2 ? 1 : b_shape[0];
  if (a_batch != b_batch && a_batch != 1 && b_batch != 1)
    throw std::runtime_error("mat_mul batch dimensions not broadcastable");
  const auto batch = std::max(a_batch, b_batch);
  const auto a_matrix_stride = rows * shared;
  const auto b_matrix_stride = shared * columns;
  const auto output_matrix_stride = rows * columns;
  const auto a_stride = a_dimensions == 2 || a_batch == 1 ? 0 : a_matrix_stride;
  const auto b_stride = b_dimensions == 2 || b_batch == 1 ? 0 : b_matrix_stride;
  const auto total = batch * rows * columns;

  tf::buffer<T> buffer;
  buffer.allocate(static_cast<std::size_t>(total));
  auto *output = buffer.data();
  const auto *a_data = a.raw_data();
  const auto *b_data = b.raw_data();
  const auto total_rows = batch * rows;
  const auto process_row = [=](int batch_row) {
    const auto batch_index = batch_row / rows;
    const auto row = batch_row % rows;
    const auto *a_row = a_data + batch_index * a_stride + row * shared;
    const auto *b_matrix = b_data + batch_index * b_stride;
    auto *output_row =
        output + batch_index * output_matrix_stride + row * columns;
    for (int column = 0; column < columns; ++column)
      output_row[column] = T{0};
    for (int inner = 0; inner < shared; ++inner) {
      const auto value = a_row[inner];
      for (int column = 0; column < columns; ++column)
        output_row[column] += value * b_matrix[inner * columns + column];
    }
  };
  tf::parallel_for_each(tf::make_sequence_range(total_rows), process_row,
                        checked_work(columns, shared));

  tf::small_vector<int, 3> output_shape;
  if (a_dimensions == 3 || b_dimensions == 3)
    output_shape.push_back(batch);
  output_shape.push_back(rows);
  output_shape.push_back(columns);
  return nd_array<T>::from_buffer(std::move(buffer), std::move(output_shape));
}

template <typename T> auto inverted(const nd_array<T> &matrix) -> nd_array<T> {
  if (!matrix.is_valid())
    throw std::invalid_argument("inverted: matrix must be a valid array");
  if (matrix.ndim() != 2 || matrix.shape_at(0) != matrix.shape_at(1) ||
      (matrix.shape_at(0) != 3 && matrix.shape_at(0) != 4))
    throw std::invalid_argument("inverted: matrix must have shape 3x3 or 4x4");

  const auto size = matrix.shape_at(0);
  tf::buffer<T> buffer;
  buffer.allocate(static_cast<std::size_t>(size * size));
  if (size == 4) {
    const auto result =
        tf::inverted(tf::make_transformation_view<3>(matrix.raw_data()));
    for (std::size_t row = 0; row < 3; ++row)
      for (std::size_t column = 0; column < 4; ++column)
        buffer[row * 4 + column] = result(row, column);
    buffer[12] = T{0};
    buffer[13] = T{0};
    buffer[14] = T{0};
    buffer[15] = T{1};
  } else {
    const auto result =
        tf::inverted(tf::make_transformation_view<2>(matrix.raw_data()));
    for (std::size_t row = 0; row < 2; ++row)
      for (std::size_t column = 0; column < 3; ++column)
        buffer[row * 3 + column] = result(row, column);
    buffer[6] = T{0};
    buffer[7] = T{0};
    buffer[8] = T{1};
  }
  return nd_array<T>::from_buffer(std::move(buffer), {size, size});
}

template <typename T>
auto assign_scalar(nd_array<T> &target, T scalar) -> void {
  tf::parallel_fill(target.make_range(), scalar);
}

template <typename T>
auto assign_array(nd_array<T> &target, const nd_array<T> &source) -> void {
  const auto output_shape =
      detail::broadcast_shape(target.raw_shape(), source.raw_shape());
  if (!detail::shapes_equal(target.raw_shape(), output_shape))
    throw std::runtime_error(
        "assign: source shape not broadcastable to target shape");

  auto *output = target.raw_data();
  const auto *input = source.raw_data();
  if (detail::shapes_equal(target.raw_shape(), source.raw_shape())) {
    std::memcpy(output, input, target.length() * sizeof(T));
    return;
  }

  const auto plan = make_broadcast_traversal_plan(
      output_shape, target.raw_shape(), source.raw_shape());
  for_each_broadcast_index(target.size(), plan, sequential,
                           [=](int index, int, int source_index) {
                             output[index] = input[source_index];
                           });
}

template <typename T>
auto assign_indexed_scalar(nd_array<T> &target,
                           const nd_array<std::int32_t> &indices, T scalar)
    -> void {
  const auto normalized =
      normalized_assignment_indices(indices, assignment_row_count(target));
  const auto &shape = target.raw_shape();
  std::size_t row_size = 1;
  for (std::size_t dimension = 1; dimension < shape.size(); ++dimension)
    row_size *= static_cast<std::size_t>(shape[dimension]);
  auto *output = target.raw_data();
  for (const auto index : normalized) {
    auto *row = output + index * row_size;
    for (std::size_t column = 0; column < row_size; ++column)
      row[column] = scalar;
  }
}

template <typename T>
auto assign_indexed_array(nd_array<T> &target,
                          const nd_array<std::int32_t> &indices,
                          const nd_array<T> &values) -> void {
  const auto normalized =
      normalized_assignment_indices(indices, assignment_row_count(target));
  if (!values.is_valid())
    throw std::invalid_argument("assign: values must be a valid array");
  const auto &shape = target.raw_shape();
  std::size_t row_size = 1;
  for (std::size_t dimension = 1; dimension < shape.size(); ++dimension)
    row_size *= static_cast<std::size_t>(shape[dimension]);

  tf::small_vector<int, 3> selection_shape;
  selection_shape.push_back(static_cast<int>(indices.length()));
  for (std::size_t dimension = 1; dimension < shape.size(); ++dimension)
    selection_shape.push_back(shape[dimension]);
  const auto output_shape =
      detail::broadcast_shape(selection_shape, values.raw_shape());
  if (!detail::shapes_equal(selection_shape, output_shape))
    throw std::runtime_error(
        "assign: values shape not broadcastable to selection shape");

  auto *output = target.raw_data();
  const auto *input = values.raw_data();
  if (detail::shapes_equal(selection_shape, values.raw_shape())) {
    for (std::size_t index = 0; index < normalized.size(); ++index)
      std::memcpy(output + normalized[index] * row_size,
                  input + index * row_size, row_size * sizeof(T));
    return;
  }

  const auto plan = make_broadcast_traversal_plan(
      selection_shape, selection_shape, values.raw_shape());
  const auto total = detail::total_size(selection_shape);
  // Repeated indices make the write order part of the result.
  for_each_broadcast_index(
      total, plan, sequential,
      [=, &normalized](int flat_index, int, int source_index) {
        const auto row = static_cast<std::size_t>(flat_index) / row_size;
        const auto column = static_cast<std::size_t>(flat_index) % row_size;
        output[normalized[row] * row_size + column] = input[source_index];
      });
}

template <typename T>
auto assign_masked_scalar(nd_array<T> &target,
                          const nd_array<std::int8_t> &mask, T scalar) -> void {
  const auto row_count = require_assignment_mask(target, mask);
  const auto &shape = target.raw_shape();
  std::size_t row_size = 1;
  for (std::size_t dimension = 1; dimension < shape.size(); ++dimension)
    row_size *= static_cast<std::size_t>(shape[dimension]);
  auto *output = target.raw_data();
  for (int row = 0; row < row_count; ++row) {
    if (mask[static_cast<std::size_t>(row)]) {
      for (std::size_t column = 0; column < row_size; ++column)
        output[static_cast<std::size_t>(row) * row_size + column] = scalar;
    }
  }
}

template <typename T>
auto assign_masked_array(nd_array<T> &target, const nd_array<std::int8_t> &mask,
                         const nd_array<T> &values) -> void {
  const auto row_count = require_assignment_mask(target, mask);
  if (!values.is_valid())
    throw std::invalid_argument("assign: values must be a valid array");
  const auto &shape = target.raw_shape();
  std::size_t row_size = 1;
  for (std::size_t dimension = 1; dimension < shape.size(); ++dimension)
    row_size *= static_cast<std::size_t>(shape[dimension]);
  int selected_count = 0;
  for (int row = 0; row < row_count; ++row)
    if (mask[static_cast<std::size_t>(row)])
      ++selected_count;

  tf::small_vector<int, 3> selection_shape;
  selection_shape.push_back(selected_count);
  for (std::size_t dimension = 1; dimension < shape.size(); ++dimension)
    selection_shape.push_back(shape[dimension]);
  const auto output_shape =
      detail::broadcast_shape(selection_shape, values.raw_shape());
  if (!detail::shapes_equal(selection_shape, output_shape))
    throw std::runtime_error(
        "assign: values shape not broadcastable to selection shape");

  auto *output = target.raw_data();
  const auto *input = values.raw_data();
  if (detail::shapes_equal(selection_shape, values.raw_shape())) {
    std::size_t source_offset = 0;
    for (int row = 0; row < row_count; ++row) {
      if (mask[static_cast<std::size_t>(row)]) {
        std::memcpy(output + static_cast<std::size_t>(row) * row_size,
                    input + source_offset, row_size * sizeof(T));
        source_offset += row_size;
      }
    }
    return;
  }

  const auto plan = make_broadcast_traversal_plan(
      selection_shape, selection_shape, values.raw_shape());
  const auto total = detail::total_size(selection_shape);
  // The selected row advances with the walk, so the walk is sequential.
  auto selection_row = 0;
  for_each_broadcast_index(
      total, plan, sequential,
      [=, &selection_row](int flat_index, int, int source_index) {
        while (!mask[static_cast<std::size_t>(selection_row)])
          ++selection_row;
        const auto column = static_cast<std::size_t>(flat_index) % row_size;
        output[static_cast<std::size_t>(selection_row) * row_size + column] =
            input[source_index];
        if (column + 1 == row_size)
          ++selection_row;
      });
}

#define TF_CPP_DEFINE_COMPARISON(NAME, COMPARISON)                             \
  template <typename T>                                                        \
  auto NAME(const nd_array<T> &a, const nd_array<T> &b)                        \
      -> nd_array<std::int8_t> {                                               \
    return compare(a, b, COMPARISON);                                          \
  }                                                                            \
  template <typename T>                                                        \
  auto NAME##_scalar(const nd_array<T> &a, T scalar)                           \
      -> nd_array<std::int8_t> {                                               \
    return compare_scalar(a, scalar, COMPARISON);                              \
  }

TF_CPP_DEFINE_COMPARISON(eq, std::equal_to<T>{})
TF_CPP_DEFINE_COMPARISON(neq, std::not_equal_to<T>{})
TF_CPP_DEFINE_COMPARISON(lt, std::less<T>{})
TF_CPP_DEFINE_COMPARISON(gt, std::greater<T>{})
TF_CPP_DEFINE_COMPARISON(lte, std::less_equal<T>{})
TF_CPP_DEFINE_COMPARISON(gte, std::greater_equal<T>{})

#undef TF_CPP_DEFINE_COMPARISON

auto logical_not(const nd_array<std::int8_t> &a) -> nd_array<std::int8_t> {
  return unary_transform<std::int8_t>(
      a, [](std::int8_t value) -> std::int8_t { return value ? 0 : 1; });
}

auto logical_not_inplace(nd_array<std::int8_t> &a) -> void {
  unary_transform_inplace(
      a, [](std::int8_t value) -> std::int8_t { return value ? 0 : 1; });
}

auto logical_and(const nd_array<std::int8_t> &a, const nd_array<std::int8_t> &b)
    -> nd_array<std::int8_t> {
  return binary_transform<std::int8_t>(
      a, b, [](std::int8_t x, std::int8_t y) -> std::int8_t {
        return x && y ? 1 : 0;
      });
}

auto logical_or(const nd_array<std::int8_t> &a, const nd_array<std::int8_t> &b)
    -> nd_array<std::int8_t> {
  return binary_transform<std::int8_t>(
      a, b, [](std::int8_t x, std::int8_t y) -> std::int8_t {
        return x || y ? 1 : 0;
      });
}

#define TF_CPP_DEFINE_UNARY(NAME, EXPRESSION)                                  \
  template <typename T> auto NAME(const nd_array<T> &a) -> nd_array<T> {       \
    return unary_transform<T>(a, [](T value) -> T { return EXPRESSION; });     \
  }                                                                            \
  template <typename T> auto NAME##_inplace(nd_array<T> &a) -> void {          \
    unary_transform_inplace(a, [](T value) -> T { return EXPRESSION; });       \
  }

TF_CPP_DEFINE_UNARY(sqrt, tf::sqrt(value))
TF_CPP_DEFINE_UNARY(sin, std::sin(value))
TF_CPP_DEFINE_UNARY(cos, std::cos(value))
TF_CPP_DEFINE_UNARY(tan, std::tan(value))
TF_CPP_DEFINE_UNARY(asin, std::asin(value))
TF_CPP_DEFINE_UNARY(acos, std::acos(value))
TF_CPP_DEFINE_UNARY(atan, std::atan(value))
TF_CPP_DEFINE_UNARY(exp, std::exp(value))
TF_CPP_DEFINE_UNARY(log, std::log(value))
TF_CPP_DEFINE_UNARY(log2, std::log2(value))
TF_CPP_DEFINE_UNARY(log10, std::log10(value))
TF_CPP_DEFINE_UNARY(floor, std::floor(value))
TF_CPP_DEFINE_UNARY(ceil, std::ceil(value))
TF_CPP_DEFINE_UNARY(round, std::round(value))
TF_CPP_DEFINE_UNARY(abs, value < 0 ? -value : value)
TF_CPP_DEFINE_UNARY(neg, -value)

#undef TF_CPP_DEFINE_UNARY

template <typename T>
auto pow(const nd_array<T> &a, T exponent) -> nd_array<T> {
  return unary_transform<T>(
      a, [=](T value) -> T { return std::pow(value, exponent); });
}

template <typename T> auto pow_inplace(nd_array<T> &a, T exponent) -> void {
  unary_transform_inplace(
      a, [=](T value) -> T { return std::pow(value, exponent); });
}

template <typename T>
auto atan2(const nd_array<T> &y, const nd_array<T> &x) -> nd_array<T> {
  return binary_transform<T>(y, x, [](T y_value, T x_value) -> T {
    return std::atan2(y_value, x_value);
  });
}

template <typename T>
auto clip(const nd_array<T> &a, T lower, T upper) -> nd_array<T> {
  return unary_transform<T>(a, [=](T value) -> T {
    return value < lower ? lower : (value > upper ? upper : value);
  });
}

template <typename T>
auto clip_inplace(nd_array<T> &a, T lower, T upper) -> void {
  unary_transform_inplace(a, [=](T value) -> T {
    return value < lower ? lower : (value > upper ? upper : value);
  });
}

template <typename T>
auto dot(const nd_array<T> &a, const nd_array<T> &b) -> nd_array<T> {
  const auto output_shape = detail::broadcast_shape(a.raw_shape(), b.raw_shape());
  const auto dimensions = static_cast<int>(output_shape.size());
  if (dimensions < 1)
    throw std::runtime_error("dot requires at least 1D arrays");
  const auto inner = output_shape[dimensions - 1];
  int batch = 1;
  for (int dimension = 0; dimension < dimensions - 1; ++dimension)
    batch *= output_shape[dimension];

  tf::buffer<T> buffer;
  buffer.allocate(static_cast<std::size_t>(batch));
  auto *output = buffer.data();
  const auto *a_data = a.raw_data();
  const auto *b_data = b.raw_data();
  const auto schedule = checked_work(inner);
  if (detail::shapes_equal(a.raw_shape(), b.raw_shape())) {
    const auto process = [=](int index) {
      T value = T{0};
      const auto base = index * inner;
      for (int inner_index = 0; inner_index < inner; ++inner_index)
        value += a_data[base + inner_index] * b_data[base + inner_index];
      output[index] = value;
    };
    tf::parallel_for_each(tf::make_sequence_range(batch), process, schedule);
  } else {
    const auto output_strides = detail::compute_strides(output_shape);
    const auto a_strides = detail::broadcast_strides(a.raw_shape(), output_shape);
    const auto b_strides = detail::broadcast_strides(b.raw_shape(), output_shape);
    const auto process = [=](int index) {
      T value = T{0};
      const auto base = index * inner;
      for (int inner_index = 0; inner_index < inner; ++inner_index) {
        const auto flat_index = base + inner_index;
        const auto a_index =
            detail::broadcast_index(flat_index, output_strides, a_strides, dimensions);
        const auto b_index =
            detail::broadcast_index(flat_index, output_strides, b_strides, dimensions);
        value += a_data[a_index] * b_data[b_index];
      }
      output[index] = value;
    };
    tf::parallel_for_each(tf::make_sequence_range(batch), process, schedule);
  }

  tf::small_vector<int, 3> result_shape;
  for (int dimension = 0; dimension < dimensions - 1; ++dimension)
    result_shape.push_back(output_shape[dimension]);
  if (result_shape.empty())
    result_shape.push_back(1);
  return nd_array<T>::from_buffer(std::move(buffer), std::move(result_shape));
}

template <typename T>
auto cross(const nd_array<T> &a, const nd_array<T> &b) -> nd_array<T> {
  auto output_shape = detail::broadcast_shape(a.raw_shape(), b.raw_shape());
  const auto dimensions = static_cast<int>(output_shape.size());
  if (dimensions < 1 || output_shape[dimensions - 1] != 3)
    throw std::runtime_error("cross requires last axis to be 3");
  int batch = 1;
  for (int dimension = 0; dimension < dimensions - 1; ++dimension)
    batch *= output_shape[dimension];

  tf::buffer<T> buffer;
  buffer.allocate(static_cast<std::size_t>(batch * 3));
  auto *output = buffer.data();
  const auto *a_data = a.raw_data();
  const auto *b_data = b.raw_data();
  if (detail::shapes_equal(a.raw_shape(), b.raw_shape())) {
    const auto process = [=](int index) {
      const auto base = index * 3;
      const auto a0 = a_data[base];
      const auto a1 = a_data[base + 1];
      const auto a2 = a_data[base + 2];
      const auto b0 = b_data[base];
      const auto b1 = b_data[base + 1];
      const auto b2 = b_data[base + 2];
      output[base] = a1 * b2 - a2 * b1;
      output[base + 1] = a2 * b0 - a0 * b2;
      output[base + 2] = a0 * b1 - a1 * b0;
    };
    tf::parallel_for_each(tf::make_sequence_range(batch), process,
                          checked_work(3));
  } else {
    const auto output_strides = detail::compute_strides(output_shape);
    const auto a_strides = detail::broadcast_strides(a.raw_shape(), output_shape);
    const auto b_strides = detail::broadcast_strides(b.raw_shape(), output_shape);
    const auto process = [=](int index) {
      const auto base = index * 3;
      const auto a_index =
          detail::broadcast_index(base, output_strides, a_strides, dimensions);
      const auto b_index =
          detail::broadcast_index(base, output_strides, b_strides, dimensions);
      const auto a0 = a_data[a_index];
      const auto a1 = a_data[a_index + 1];
      const auto a2 = a_data[a_index + 2];
      const auto b0 = b_data[b_index];
      const auto b1 = b_data[b_index + 1];
      const auto b2 = b_data[b_index + 2];
      output[base] = a1 * b2 - a2 * b1;
      output[base + 1] = a2 * b0 - a0 * b2;
      output[base + 2] = a0 * b1 - a1 * b0;
    };
    tf::parallel_for_each(tf::make_sequence_range(batch), process,
                          checked_work(3));
  }
  return nd_array<T>::from_buffer(std::move(buffer), std::move(output_shape));
}

template <typename T> auto normalize(const nd_array<T> &a) -> nd_array<T> {
  auto result = a.deep_copy();
  normalize_inplace(result);
  return result;
}

template <typename T>
auto normalize(const nd_array<T> &a, int axis) -> nd_array<T> {
  auto result = a.deep_copy();
  normalize_inplace(result, axis);
  return result;
}

template <typename T> auto normalize_inplace(nd_array<T> &a) -> void {
  T squared_norm = T{0};
  for (const auto value : a)
    squared_norm += value * value;
  const auto norm = tf::sqrt(squared_norm);
  scalar_transform_inplace(a, norm, std::divides<T>{});
}

template <typename T> auto normalize_inplace(nd_array<T> &a, int axis) -> void {
  axis = detail::normalized_axis(a.raw_shape(), axis, "normalize");
  int outer = 1;
  for (int dimension = 0; dimension < axis; ++dimension)
    outer *= a.raw_shape()[dimension];
  const auto axis_size = a.raw_shape()[axis];
  int inner = 1;
  for (int dimension = axis + 1; dimension < a.ndim(); ++dimension)
    inner *= a.raw_shape()[dimension];

  auto *data = a.raw_data();
  const auto carrier_count = outer * inner;
  const auto normalize_one = [=](int carrier) {
    const auto outer_index = carrier / inner;
    const auto inner_index = carrier % inner;
    T squared_norm = T{0};
    for (int axis_index = 0; axis_index < axis_size; ++axis_index) {
      const auto index =
          (outer_index * axis_size + axis_index) * inner + inner_index;
      squared_norm += data[index] * data[index];
    }
    const auto norm = tf::sqrt(squared_norm);
    for (int axis_index = 0; axis_index < axis_size; ++axis_index) {
      const auto index =
          (outer_index * axis_size + axis_index) * inner + inner_index;
      data[index] /= norm;
    }
  };
  tf::parallel_for_each(tf::make_sequence_range(carrier_count), normalize_one,
                        checked_work(axis_size));
}

template <typename From, typename To>
auto cast(const nd_array<From> &a) -> nd_array<To> {
  return unary_transform<To>(
      a, [](From value) -> To { return static_cast<To>(value); });
}

#define TF_CPP_INSTANTIATE_BINARY(NAME, T)                                     \
  template auto NAME<T>(const nd_array<T> &, const nd_array<T> &)              \
      -> nd_array<T>;                                                          \
  template auto NAME##_inplace<T>(nd_array<T> &, const nd_array<T> &) -> void; \
  template auto NAME##_scalar<T>(const nd_array<T> &, T) -> nd_array<T>;       \
  template auto NAME##_scalar_inplace<T>(nd_array<T> &, T) -> void

#define TF_CPP_INSTANTIATE_ARITHMETIC(T)                                       \
  TF_CPP_INSTANTIATE_BINARY(add, T);                                           \
  TF_CPP_INSTANTIATE_BINARY(sub, T);                                           \
  TF_CPP_INSTANTIATE_BINARY(mul, T);                                           \
  TF_CPP_INSTANTIATE_BINARY(div, T);                                           \
  TF_CPP_INSTANTIATE_BINARY(mod, T)

TF_CPP_INSTANTIATE_ARITHMETIC(std::int8_t);
TF_CPP_INSTANTIATE_ARITHMETIC(std::int32_t);
TF_CPP_INSTANTIATE_ARITHMETIC(std::int64_t);
TF_CPP_INSTANTIATE_ARITHMETIC(float);
TF_CPP_INSTANTIATE_ARITHMETIC(double);

#undef TF_CPP_INSTANTIATE_ARITHMETIC
#undef TF_CPP_INSTANTIATE_BINARY

#define TF_CPP_INSTANTIATE_ASSIGN(T)                                           \
  template auto assign_scalar<T>(nd_array<T> &, T) -> void;                    \
  template auto assign_array<T>(nd_array<T> &, const nd_array<T> &) -> void;   \
  template auto assign_indexed_scalar<T>(                                      \
      nd_array<T> &, const nd_array<std::int32_t> &, T) -> void;               \
  template auto assign_indexed_array<T>(nd_array<T> &,                         \
                                        const nd_array<std::int32_t> &,        \
                                        const nd_array<T> &) -> void;          \
  template auto assign_masked_scalar<T>(                                       \
      nd_array<T> &, const nd_array<std::int8_t> &, T) -> void;                \
  template auto assign_masked_array<T>(nd_array<T> &,                          \
                                       const nd_array<std::int8_t> &,          \
                                       const nd_array<T> &) -> void

TF_CPP_INSTANTIATE_ASSIGN(std::int8_t);
TF_CPP_INSTANTIATE_ASSIGN(std::int32_t);
TF_CPP_INSTANTIATE_ASSIGN(std::int64_t);
TF_CPP_INSTANTIATE_ASSIGN(float);
TF_CPP_INSTANTIATE_ASSIGN(double);

#undef TF_CPP_INSTANTIATE_ASSIGN

#define TF_CPP_INSTANTIATE_COMPARISON(NAME, T)                                 \
  template auto NAME<T>(const nd_array<T> &, const nd_array<T> &)              \
      -> nd_array<std::int8_t>;                                                \
  template auto NAME##_scalar<T>(const nd_array<T> &, T)                       \
      -> nd_array<std::int8_t>

#define TF_CPP_INSTANTIATE_COMPARISONS(T)                                      \
  TF_CPP_INSTANTIATE_COMPARISON(eq, T);                                        \
  TF_CPP_INSTANTIATE_COMPARISON(neq, T);                                       \
  TF_CPP_INSTANTIATE_COMPARISON(lt, T);                                        \
  TF_CPP_INSTANTIATE_COMPARISON(gt, T);                                        \
  TF_CPP_INSTANTIATE_COMPARISON(lte, T);                                       \
  TF_CPP_INSTANTIATE_COMPARISON(gte, T)

TF_CPP_INSTANTIATE_COMPARISONS(std::int8_t);
TF_CPP_INSTANTIATE_COMPARISONS(std::int32_t);
TF_CPP_INSTANTIATE_COMPARISONS(std::int64_t);
TF_CPP_INSTANTIATE_COMPARISONS(float);
TF_CPP_INSTANTIATE_COMPARISONS(double);

#undef TF_CPP_INSTANTIATE_COMPARISONS
#undef TF_CPP_INSTANTIATE_COMPARISON

#define TF_CPP_INSTANTIATE_UNARY(NAME, T)                                      \
  template auto NAME<T>(const nd_array<T> &) -> nd_array<T>;                   \
  template auto NAME##_inplace<T>(nd_array<T> &) -> void

#define TF_CPP_INSTANTIATE_FLOAT_UNARIES(T)                                    \
  TF_CPP_INSTANTIATE_UNARY(sqrt, T);                                           \
  TF_CPP_INSTANTIATE_UNARY(sin, T);                                            \
  TF_CPP_INSTANTIATE_UNARY(cos, T);                                            \
  TF_CPP_INSTANTIATE_UNARY(tan, T);                                            \
  TF_CPP_INSTANTIATE_UNARY(asin, T);                                           \
  TF_CPP_INSTANTIATE_UNARY(acos, T);                                           \
  TF_CPP_INSTANTIATE_UNARY(atan, T);                                           \
  TF_CPP_INSTANTIATE_UNARY(exp, T);                                            \
  TF_CPP_INSTANTIATE_UNARY(log, T);                                            \
  TF_CPP_INSTANTIATE_UNARY(log2, T);                                           \
  TF_CPP_INSTANTIATE_UNARY(log10, T);                                          \
  TF_CPP_INSTANTIATE_UNARY(floor, T);                                          \
  TF_CPP_INSTANTIATE_UNARY(ceil, T);                                           \
  TF_CPP_INSTANTIATE_UNARY(round, T);                                          \
  template auto pow<T>(const nd_array<T> &, T) -> nd_array<T>;                 \
  template auto pow_inplace<T>(nd_array<T> &, T) -> void;                      \
  template auto atan2<T>(const nd_array<T> &, const nd_array<T> &)             \
      -> nd_array<T>;                                                          \
  template auto normalize<T>(const nd_array<T> &) -> nd_array<T>;              \
  template auto normalize<T>(const nd_array<T> &, int) -> nd_array<T>;         \
  template auto normalize_inplace<T>(nd_array<T> &) -> void;                   \
  template auto normalize_inplace<T>(nd_array<T> &, int) -> void

TF_CPP_INSTANTIATE_FLOAT_UNARIES(float);
TF_CPP_INSTANTIATE_FLOAT_UNARIES(double);

#undef TF_CPP_INSTANTIATE_FLOAT_UNARIES

#define TF_CPP_INSTANTIATE_GENERAL(T)                                          \
  TF_CPP_INSTANTIATE_UNARY(abs, T);                                            \
  TF_CPP_INSTANTIATE_UNARY(neg, T);                                            \
  template auto clip<T>(const nd_array<T> &, T, T) -> nd_array<T>;             \
  template auto clip_inplace<T>(nd_array<T> &, T, T) -> void

TF_CPP_INSTANTIATE_GENERAL(std::int8_t);
TF_CPP_INSTANTIATE_GENERAL(std::int32_t);
TF_CPP_INSTANTIATE_GENERAL(std::int64_t);
TF_CPP_INSTANTIATE_GENERAL(float);
TF_CPP_INSTANTIATE_GENERAL(double);

#undef TF_CPP_INSTANTIATE_GENERAL
#undef TF_CPP_INSTANTIATE_UNARY

#define TF_CPP_INSTANTIATE_VECTOR(T)                                           \
  template auto mat_mul<T>(const nd_array<T> &, const nd_array<T> &)           \
      -> nd_array<T>;                                                          \
  template auto dot<T>(const nd_array<T> &, const nd_array<T> &)               \
      -> nd_array<T>;                                                          \
  template auto cross<T>(const nd_array<T> &, const nd_array<T> &)             \
      -> nd_array<T>

TF_CPP_INSTANTIATE_VECTOR(std::int32_t);
TF_CPP_INSTANTIATE_VECTOR(std::int64_t);
TF_CPP_INSTANTIATE_VECTOR(float);
TF_CPP_INSTANTIATE_VECTOR(double);

template auto inverted<float>(const nd_array<float> &) -> nd_array<float>;
template auto inverted<double>(const nd_array<double> &) -> nd_array<double>;

#undef TF_CPP_INSTANTIATE_VECTOR

#define TF_CPP_INSTANTIATE_CAST(FROM, TO)                                      \
  template auto cast<FROM, TO>(const nd_array<FROM> &) -> nd_array<TO>

TF_CPP_INSTANTIATE_CAST(std::int8_t, std::int32_t);
TF_CPP_INSTANTIATE_CAST(std::int8_t, std::int64_t);
TF_CPP_INSTANTIATE_CAST(std::int8_t, float);
TF_CPP_INSTANTIATE_CAST(std::int8_t, double);
TF_CPP_INSTANTIATE_CAST(std::int32_t, std::int8_t);
TF_CPP_INSTANTIATE_CAST(std::int32_t, std::int64_t);
TF_CPP_INSTANTIATE_CAST(std::int32_t, float);
TF_CPP_INSTANTIATE_CAST(std::int32_t, double);
TF_CPP_INSTANTIATE_CAST(std::int64_t, std::int8_t);
TF_CPP_INSTANTIATE_CAST(std::int64_t, std::int32_t);
TF_CPP_INSTANTIATE_CAST(std::int64_t, float);
TF_CPP_INSTANTIATE_CAST(std::int64_t, double);
TF_CPP_INSTANTIATE_CAST(float, std::int8_t);
TF_CPP_INSTANTIATE_CAST(float, std::int32_t);
TF_CPP_INSTANTIATE_CAST(float, std::int64_t);
TF_CPP_INSTANTIATE_CAST(float, double);
TF_CPP_INSTANTIATE_CAST(double, std::int8_t);
TF_CPP_INSTANTIATE_CAST(double, std::int32_t);
TF_CPP_INSTANTIATE_CAST(double, std::int64_t);
TF_CPP_INSTANTIATE_CAST(double, float);

#undef TF_CPP_INSTANTIATE_CAST

} // namespace tf::cpp
