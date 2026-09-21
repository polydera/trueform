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
#include "trueform/cpp/core/nd_array_indexing.hpp"

#include "checked_work.hpp"
#include "shape_size.hpp"

#include "trueform/core/algorithm/block_reduce_sequenced_aggregate.hpp"
#include "trueform/core/algorithm/parallel_for.hpp"
#include "trueform/core/algorithm/parallel_for_each.hpp"
#include "trueform/core/buffer.hpp"
#include "trueform/core/checked.hpp"
#include "trueform/core/range.hpp"
#include "trueform/core/views/sequence_range.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace tf::cpp {
namespace {

auto normalized_index(std::int32_t index, int size, const char *operation)
    -> int {
  auto result = static_cast<int>(index);
  if (result < 0)
    result += size;
  if (result < 0 || result >= size)
    throw std::out_of_range(std::string(operation) + ": index out of range");
  return result;
}

auto row_stride(const tf::small_vector<int, 3> &shape) -> std::size_t {
  std::size_t stride = 1;
  for (std::size_t dimension = 1; dimension < shape.size(); ++dimension)
    stride *= static_cast<std::size_t>(shape[dimension]);
  return stride;
}

struct mask_chunk {
  std::size_t first_row = 0;
  std::size_t row_count = 0;
  std::size_t selected = 0;
  std::size_t output_row = 0;
};

} // namespace

multi_take_index::multi_take_index(int index)
    : _mode(mode::single), _single_index(index) {}

multi_take_index::multi_take_index(std::vector<std::int32_t> indices)
    : _mode(mode::indices), _indices(std::move(indices)) {}

auto multi_take_index::selection_mode() const -> mode { return _mode; }

auto multi_take_index::single_index() const -> int { return _single_index; }

auto multi_take_index::index_array() const
    -> const std::vector<std::int32_t> & {
  return _indices;
}

template <typename T>
auto take(const nd_array<T> &array, const nd_array<std::int32_t> &indices,
          int axis) -> nd_array<T> {
  const auto &input_shape = array.raw_shape();
  axis = detail::normalized_axis(input_shape, axis, "take");
  const auto axis_size = input_shape[axis];

  tf::buffer<int> normalized_indices;
  normalized_indices.allocate(indices.length());
  for (std::size_t index = 0; index < indices.length(); ++index)
    normalized_indices[index] =
        normalized_index(indices[index], axis_size, "take");

  std::size_t outer = 1;
  for (int dimension = 0; dimension < axis; ++dimension)
    outer *= static_cast<std::size_t>(input_shape[dimension]);
  std::size_t inner = 1;
  for (int dimension = axis + 1;
       dimension < static_cast<int>(input_shape.size()); ++dimension)
    inner *= static_cast<std::size_t>(input_shape[dimension]);

  auto output_shape = input_shape;
  output_shape[axis] = static_cast<int>(indices.length());
  tf::buffer<T> buffer;
  if (outer == 0 || indices.empty() || inner == 0)
    return nd_array<T>::from_buffer(std::move(buffer), std::move(output_shape));

  constexpr auto max_length =
      static_cast<std::size_t>(std::numeric_limits<int>::max());
  if (outer > max_length / indices.length())
    throw std::length_error("take: output is too large");
  const auto block_count = outer * indices.length();
  if (block_count > max_length / inner)
    throw std::length_error("take: output is too large");
  const auto output_length = block_count * inner;
  buffer.allocate(output_length);
  auto *output = buffer.data();
  const auto *source = array.raw_data();
  const auto process_blocks = [=, &normalized_indices](std::size_t begin,
                                                       std::size_t end) {
    for (auto block_index = begin; block_index < end; ++block_index) {
      const auto outer_index = block_index / indices.length();
      const auto index_index = block_index % indices.length();
      std::memcpy(
          output + block_index * inner,
          source + (outer_index * static_cast<std::size_t>(axis_size) +
                    static_cast<std::size_t>(normalized_indices[index_index])) *
                       inner,
          inner * sizeof(T));
    }
  };
  tf::parallel_for(std::size_t{0}, block_count, process_blocks,
                   checked_work(inner));
  return nd_array<T>::from_buffer(std::move(buffer), std::move(output_shape));
}

template <typename T>
auto multi_take(const nd_array<T> &array,
                const std::vector<multi_take_index> &indices) -> nd_array<T> {
  const auto &input_shape = array.raw_shape();
  const auto dimensions = static_cast<int>(input_shape.size());
  if (dimensions == 0)
    throw std::invalid_argument(
        "multi_take: array must have at least one axis");
  if (indices.size() > input_shape.size())
    throw std::invalid_argument("multi_take: too many index specifications");

  std::vector<std::size_t> input_strides(static_cast<std::size_t>(dimensions),
                                         1);
  for (int dimension = dimensions - 2; dimension >= 0; --dimension)
    input_strides[static_cast<std::size_t>(dimension)] =
        input_strides[static_cast<std::size_t>(dimension + 1)] *
        static_cast<std::size_t>(input_shape[dimension + 1]);

  std::vector<multi_take_index::mode> modes(
      static_cast<std::size_t>(dimensions), multi_take_index::mode::all);
  std::vector<std::size_t> singles(static_cast<std::size_t>(dimensions), 0);
  std::vector<std::vector<std::size_t>> arrays(
      static_cast<std::size_t>(dimensions));
  std::vector<int> counts(static_cast<std::size_t>(dimensions));
  tf::small_vector<int, 3> output_shape;

  for (int dimension = 0; dimension < dimensions; ++dimension) {
    const auto axis_size = input_shape[dimension];
    if (static_cast<std::size_t>(dimension) >= indices.size()) {
      counts[static_cast<std::size_t>(dimension)] = axis_size;
      output_shape.push_back(axis_size);
      continue;
    }

    const auto &specification = indices[static_cast<std::size_t>(dimension)];
    const auto mode = specification.selection_mode();
    modes[static_cast<std::size_t>(dimension)] = mode;
    if (mode == multi_take_index::mode::all) {
      counts[static_cast<std::size_t>(dimension)] = axis_size;
      output_shape.push_back(axis_size);
    } else if (mode == multi_take_index::mode::single) {
      singles[static_cast<std::size_t>(dimension)] =
          static_cast<std::size_t>(normalized_index(
              specification.single_index(), axis_size, "multi_take")) *
          input_strides[static_cast<std::size_t>(dimension)];
      counts[static_cast<std::size_t>(dimension)] = 1;
    } else {
      const auto &source_indices = specification.index_array();
      auto &axis_offsets = arrays[static_cast<std::size_t>(dimension)];
      axis_offsets.resize(source_indices.size());
      for (std::size_t index = 0; index < source_indices.size(); ++index)
        axis_offsets[index] =
            static_cast<std::size_t>(normalized_index(
                source_indices[index], axis_size, "multi_take")) *
            input_strides[static_cast<std::size_t>(dimension)];
      counts[static_cast<std::size_t>(dimension)] =
          static_cast<int>(source_indices.size());
      output_shape.push_back(static_cast<int>(source_indices.size()));
    }
  }
  if (output_shape.empty())
    output_shape.push_back(1);

  std::size_t total = 1;
  for (const auto size : output_shape)
    total *= static_cast<std::size_t>(size);
  tf::buffer<T> buffer;
  buffer.allocate(total);
  auto *output = buffer.data();
  const auto *source = array.raw_data();
  const auto process_range = [=, &modes, &singles, &arrays, &counts,
                              &input_strides](std::size_t begin,
                                              std::size_t end) {
    if (begin == end)
      return;

    std::vector<std::size_t> coordinates(static_cast<std::size_t>(dimensions),
                                         0);
    auto remainder = begin;
    std::size_t source_index = 0;
    const auto selected_offset = [&](int dimension, std::size_t coordinate) {
      const auto slot = static_cast<std::size_t>(dimension);
      const auto mode = modes[slot];
      if (mode == multi_take_index::mode::all)
        return coordinate * input_strides[slot];
      if (mode == multi_take_index::mode::single)
        return singles[slot];
      return arrays[slot][coordinate];
    };

    for (int dimension = dimensions - 1; dimension >= 0; --dimension) {
      const auto slot = static_cast<std::size_t>(dimension);
      const auto count = static_cast<std::size_t>(counts[slot]);
      const auto coordinate = remainder % count;
      remainder /= count;
      coordinates[slot] = coordinate;
      source_index += selected_offset(dimension, coordinate);
    }

    for (auto output_index = begin; output_index < end; ++output_index) {
      output[output_index] = source[source_index];
      if (output_index + 1 == end)
        break;

      for (int dimension = dimensions - 1; dimension >= 0; --dimension) {
        const auto slot = static_cast<std::size_t>(dimension);
        const auto old_coordinate = coordinates[slot];
        source_index -= selected_offset(dimension, old_coordinate);
        const auto next_coordinate = old_coordinate + 1;
        if (next_coordinate < static_cast<std::size_t>(counts[slot])) {
          coordinates[slot] = next_coordinate;
          source_index += selected_offset(dimension, next_coordinate);
          break;
        }
        coordinates[slot] = 0;
        source_index += selected_offset(dimension, 0);
      }
    }
  };
  tf::parallel_for(std::size_t{0}, total, process_range, tf::checked);
  return nd_array<T>::from_buffer(std::move(buffer), std::move(output_shape));
}

template <typename T>
auto take_along_axis(const nd_array<T> &array,
                     const nd_array<std::int32_t> &indices, int axis)
    -> nd_array<T> {
  const auto &input_shape = array.raw_shape();
  const auto &index_shape = indices.raw_shape();
  axis = detail::normalized_axis(input_shape, axis, "take_along_axis");
  if (index_shape.size() != input_shape.size())
    throw std::invalid_argument("take_along_axis: ndim mismatch");
  for (int dimension = 0; dimension < static_cast<int>(input_shape.size());
       ++dimension)
    if (dimension != axis && input_shape[dimension] != index_shape[dimension])
      throw std::invalid_argument(
          "take_along_axis: non-axis dimensions must match");

  std::size_t inner = 1;
  for (int dimension = axis + 1;
       dimension < static_cast<int>(input_shape.size()); ++dimension)
    inner *= static_cast<std::size_t>(input_shape[dimension]);
  const auto input_axis_size = input_shape[axis];
  const auto index_axis_size = index_shape[axis];

  tf::buffer<T> buffer;
  buffer.allocate(indices.length());
  auto *output = buffer.data();
  const auto *source = array.raw_data();
  tf::parallel_for_each(
      tf::make_sequence_range(static_cast<int>(indices.length())),
      [=, &indices](int flat_index) {
        const auto flat = static_cast<std::size_t>(flat_index);
        const auto inner_index = flat % inner;
        const auto grouped = flat / inner;
        const auto outer_index =
            grouped / static_cast<std::size_t>(index_axis_size);
        const auto selected =
            normalized_index(indices[flat], input_axis_size, "take_along_axis");
        output[flat] =
            source[(outer_index * static_cast<std::size_t>(input_axis_size) +
                    static_cast<std::size_t>(selected)) *
                       inner +
                   inner_index];
      },
      tf::checked);
  return nd_array<T>::from_buffer(std::move(buffer), index_shape);
}

template <typename T>
auto boolean_index(const nd_array<T> &array, const nd_array<std::int8_t> &mask)
    -> nd_array<T> {
  const auto &input_shape = array.raw_shape();
  if (input_shape.empty())
    throw std::invalid_argument(
        "boolean_index: array must have at least one axis");
  if (mask.ndim() != 1 ||
      mask.length() != static_cast<std::size_t>(input_shape.front()))
    throw std::invalid_argument(
        "boolean_index: mask must be 1D with one value per row");

  const auto row_count = mask.length();
  const auto stride = row_stride(input_shape);
  const auto *mask_data = mask.raw_data();
  const auto *source = array.raw_data();

  // A chunk's selected rows land contiguously, so counting them in parallel
  // and appending the counts in input order gives the output base each chunk
  // writes from.
  tf::buffer<mask_chunk> chunks;
  tf::blocked_reduce_sequenced_aggregate(
      tf::make_sequence_range(row_count), chunks, mask_chunk{},
      [mask_data](const auto &rows, mask_chunk &chunk) {
        chunk = mask_chunk{*rows.begin(), rows.size(), 0, 0};
        for (const auto row : rows)
          chunk.selected += mask_data[row] != 0;
      },
      [](const mask_chunk &chunk, tf::buffer<mask_chunk> &out) {
        out.push_back(chunk);
      },
      checked_work(stride));

  auto selected_rows = std::size_t{0};
  for (auto &chunk : chunks) {
    chunk.output_row = selected_rows;
    selected_rows += chunk.selected;
  }

  auto output_shape = input_shape;
  output_shape[0] = static_cast<int>(selected_rows);
  tf::buffer<T> buffer;
  buffer.allocate(selected_rows * stride);
  if (stride == 0)
    return nd_array<T>::from_buffer(std::move(buffer), std::move(output_shape));

  auto *output = buffer.data();
  tf::parallel_for_each(
      tf::make_range(chunks),
      [=](const mask_chunk &chunk) {
        auto output_row = chunk.output_row;
        const auto end = chunk.first_row + chunk.row_count;
        for (auto row = chunk.first_row; row < end; ++row) {
          if (!mask_data[row])
            continue;
          std::memcpy(output + output_row * stride, source + row * stride,
                      stride * sizeof(T));
          ++output_row;
        }
      },
      tf::checked(2));
  return nd_array<T>::from_buffer(std::move(buffer), std::move(output_shape));
}

#define TF_CPP_INSTANTIATE_INDEXING(T)                                         \
  template auto take<T>(const nd_array<T> &, const nd_array<std::int32_t> &,   \
                        int) -> nd_array<T>;                                   \
  template auto multi_take<T>(const nd_array<T> &,                             \
                              const std::vector<multi_take_index> &)           \
      -> nd_array<T>;                                                          \
  template auto take_along_axis<T>(const nd_array<T> &,                        \
                                   const nd_array<std::int32_t> &, int)        \
      -> nd_array<T>;                                                          \
  template auto boolean_index<T>(const nd_array<T> &,                          \
                                 const nd_array<std::int8_t> &) -> nd_array<T>

TF_CPP_INSTANTIATE_INDEXING(std::int8_t);
TF_CPP_INSTANTIATE_INDEXING(std::int32_t);
TF_CPP_INSTANTIATE_INDEXING(std::int64_t);
TF_CPP_INSTANTIATE_INDEXING(float);
TF_CPP_INSTANTIATE_INDEXING(double);

#undef TF_CPP_INSTANTIATE_INDEXING

} // namespace tf::cpp
