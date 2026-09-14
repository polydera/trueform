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
#include "trueform/cpp/core/nd_array_structure.hpp"

#include "checked_work.hpp"
#include "shape_size.hpp"

#include "trueform/core/algorithm/parallel_for.hpp"
#include "trueform/core/buffer.hpp"
#include "trueform/core/small_vector.hpp"
#include "trueform/cpp/core/elementwise/detail/broadcast.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace tf::cpp {

template <typename T>
auto stack(const std::vector<nd_array<T>> &arrays, int axis) -> nd_array<T> {
  if (arrays.empty())
    throw std::runtime_error("stack: empty array list");

  const auto &reference_shape = arrays.front().raw_shape();
  const auto dimensions = static_cast<int>(reference_shape.size());
  if (axis < 0)
    axis += dimensions + 1;
  if (axis < 0 || axis > dimensions)
    throw std::runtime_error("stack: axis out of range");

  for (std::size_t index = 1; index < arrays.size(); ++index) {
    const auto &shape = arrays[index].raw_shape();
    if (static_cast<int>(shape.size()) != dimensions)
      throw std::runtime_error("stack: ndim mismatch");
    for (int dimension = 0; dimension < dimensions; ++dimension)
      if (shape[dimension] != reference_shape[dimension])
        throw std::runtime_error("stack: shape mismatch");
  }

  if (arrays.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    throw std::length_error("stack: too many arrays");
  tf::small_vector<int, 3> output_shape;
  for (int dimension = 0; dimension < axis; ++dimension)
    output_shape.push_back(reference_shape[dimension]);
  output_shape.push_back(static_cast<int>(arrays.size()));
  for (int dimension = axis; dimension < dimensions; ++dimension)
    output_shape.push_back(reference_shape[dimension]);

  std::size_t outer = 1;
  for (int dimension = 0; dimension < axis; ++dimension)
    outer *= static_cast<std::size_t>(reference_shape[dimension]);
  std::size_t inner = 1;
  for (int dimension = axis; dimension < dimensions; ++dimension)
    inner *= static_cast<std::size_t>(reference_shape[dimension]);

  tf::buffer<T> buffer;
  buffer.allocate(detail::shape_size(output_shape));
  if (buffer.size() == 0)
    return nd_array<T>::from_buffer(std::move(buffer), std::move(output_shape));
  auto *output = buffer.data();
  std::vector<const T *> input_data;
  input_data.reserve(arrays.size());
  for (const auto &array : arrays)
    input_data.push_back(array.raw_data());

  const auto array_count = arrays.size();
  const auto output_block_size = array_count * inner;
  const auto copy_blocks = [&](std::size_t begin, std::size_t end) {
    for (auto block = begin; block < end; ++block) {
      const auto outer_index = block / array_count;
      const auto array_index = block % array_count;
      std::memcpy(output + outer_index * output_block_size +
                      array_index * inner,
                  input_data[array_index] + outer_index * inner,
                  inner * sizeof(T));
    }
  };
  tf::parallel_for(std::size_t{0}, outer * array_count, copy_blocks,
                   checked_work(inner));
  return nd_array<T>::from_buffer(std::move(buffer), std::move(output_shape));
}

template <typename T>
auto concatenate(const std::vector<nd_array<T>> &arrays, int axis)
    -> nd_array<T> {
  if (arrays.empty())
    throw std::runtime_error("concatenate: empty array list");

  const auto &reference_shape = arrays.front().raw_shape();
  const auto dimensions = static_cast<int>(reference_shape.size());
  if (axis < 0)
    axis += dimensions;
  if (axis < 0 || axis >= dimensions)
    throw std::runtime_error("concatenate: axis out of range");

  for (std::size_t index = 1; index < arrays.size(); ++index) {
    const auto &shape = arrays[index].raw_shape();
    if (static_cast<int>(shape.size()) != dimensions)
      throw std::runtime_error("concatenate: ndim mismatch");
    for (int dimension = 0; dimension < dimensions; ++dimension)
      if (dimension != axis && shape[dimension] != reference_shape[dimension])
        throw std::runtime_error(
            "concatenate: shape mismatch on non-concat axis");
  }

  auto concatenated_size = std::size_t{0};
  for (const auto &array : arrays) {
    const auto axis_size = static_cast<std::size_t>(array.raw_shape()[axis]);
    if (axis_size > static_cast<std::size_t>(std::numeric_limits<int>::max()) -
                        concatenated_size)
      throw std::length_error("concatenate: axis size exceeds int range");
    concatenated_size += axis_size;
  }
  auto output_shape = reference_shape;
  output_shape[axis] = static_cast<int>(concatenated_size);

  std::size_t outer = 1;
  for (int dimension = 0; dimension < axis; ++dimension)
    outer *= static_cast<std::size_t>(reference_shape[dimension]);
  std::size_t inner = 1;
  for (int dimension = axis + 1; dimension < dimensions; ++dimension)
    inner *= static_cast<std::size_t>(reference_shape[dimension]);

  tf::buffer<T> buffer;
  buffer.allocate(detail::shape_size(output_shape));
  if (buffer.size() == 0)
    return nd_array<T>::from_buffer(std::move(buffer), std::move(output_shape));
  auto *output = buffer.data();
  std::vector<const T *> input_data;
  std::vector<std::size_t> block_sizes;
  std::vector<std::size_t> block_offsets;
  input_data.reserve(arrays.size());
  block_sizes.reserve(arrays.size());
  block_offsets.reserve(arrays.size());
  auto output_block_size = std::size_t{0};
  for (const auto &array : arrays) {
    input_data.push_back(array.raw_data());
    block_offsets.push_back(output_block_size);
    const auto axis_size = static_cast<std::size_t>(array.raw_shape()[axis]);
    block_sizes.push_back(axis_size * inner);
    output_block_size += block_sizes.back();
  }

  const auto array_count = arrays.size();
  const auto copy_blocks = [&](std::size_t begin, std::size_t end) {
    for (auto block = begin; block < end; ++block) {
      const auto outer_index = block / array_count;
      const auto array_index = block % array_count;
      const auto block_size = block_sizes[array_index];
      if (block_size == 0)
        continue;
      std::memcpy(output + outer_index * output_block_size +
                      block_offsets[array_index],
                  input_data[array_index] + outer_index * block_size,
                  block_size * sizeof(T));
    }
  };
  tf::parallel_for(std::size_t{0}, outer * array_count, copy_blocks,
                   checked_work(inner));
  return nd_array<T>::from_buffer(std::move(buffer), std::move(output_shape));
}

template <typename T>
auto tile(const nd_array<T> &array, const std::vector<int> &repetitions)
    -> nd_array<T> {
  if (!array.is_valid())
    throw std::invalid_argument("tile: array must be valid");
  const auto &input_shape = array.raw_shape();
  const auto input_dimensions = static_cast<int>(input_shape.size());
  const auto repetition_dimensions = static_cast<int>(repetitions.size());
  const auto output_dimensions =
      std::max(input_dimensions, repetition_dimensions);

  std::vector<int> padded_shape(static_cast<std::size_t>(output_dimensions), 1);
  for (int dimension = 0; dimension < input_dimensions; ++dimension)
    padded_shape[static_cast<std::size_t>(output_dimensions - input_dimensions +
                                          dimension)] = input_shape[dimension];
  std::vector<int> padded_repetitions(
      static_cast<std::size_t>(output_dimensions), 1);
  for (int dimension = 0; dimension < repetition_dimensions; ++dimension)
    padded_repetitions[static_cast<std::size_t>(
        output_dimensions - repetition_dimensions + dimension)] =
        repetitions[static_cast<std::size_t>(dimension)];

  tf::small_vector<int, 3> output_shape;
  for (int dimension = 0; dimension < output_dimensions; ++dimension) {
    const auto shape_dimension =
        padded_shape[static_cast<std::size_t>(dimension)];
    const auto repetition =
        padded_repetitions[static_cast<std::size_t>(dimension)];
    if (repetition < 0)
      throw std::invalid_argument("tile: repetitions must be nonnegative");
    if (shape_dimension != 0 &&
        repetition > std::numeric_limits<int>::max() / shape_dimension)
      throw std::overflow_error("tile: output shape overflows int");
    const auto size = shape_dimension * repetition;
    output_shape.push_back(size);
  }
  const auto total = detail::shape_size(output_shape);

  tf::buffer<T> buffer;
  buffer.allocate(total);
  if (total == 0)
    return nd_array<T>::from_buffer(std::move(buffer), std::move(output_shape));
  auto *output = buffer.data();
  std::memcpy(output, array.raw_data(), array.length() * sizeof(T));
  auto current_length = array.length();

  for (int dimension = output_dimensions - 1; dimension >= 0; --dimension) {
    const auto repetition =
        padded_repetitions[static_cast<std::size_t>(dimension)];
    if (repetition == 1)
      continue;

    auto block = static_cast<std::size_t>(
        padded_shape[static_cast<std::size_t>(dimension)]);
    for (int inner_dimension = dimension + 1;
         inner_dimension < output_dimensions; ++inner_dimension)
      block *= static_cast<std::size_t>(output_shape[inner_dimension]);

    const auto block_count = current_length / block;
    const auto new_block = block * static_cast<std::size_t>(repetition);
    for (auto block_index = block_count; block_index-- > 0;) {
      auto *source = output + block_index * block;
      auto *destination = output + block_index * new_block;
      if (destination != source)
        std::memmove(destination, source, block * sizeof(T));
      for (int index = 1; index < repetition; ++index)
        std::memcpy(destination + static_cast<std::size_t>(index) * block,
                    destination, block * sizeof(T));
    }
    current_length = block_count * new_block;
  }
  return nd_array<T>::from_buffer(std::move(buffer), std::move(output_shape));
}

template <typename T> auto transpose(const nd_array<T> &array) -> nd_array<T> {
  std::vector<int> axes(static_cast<std::size_t>(array.ndim()));
  for (int dimension = 0; dimension < array.ndim(); ++dimension)
    axes[static_cast<std::size_t>(dimension)] = array.ndim() - 1 - dimension;
  return transpose(array, axes);
}

template <typename T>
auto transpose(const nd_array<T> &array, const std::vector<int> &axes)
    -> nd_array<T> {
  if (!array.is_valid())
    throw std::invalid_argument("transpose: array must be valid");
  const auto &input_shape = array.raw_shape();
  const auto dimensions = static_cast<int>(input_shape.size());
  if (static_cast<int>(axes.size()) != dimensions)
    throw std::runtime_error("transpose: axes size mismatch");

  std::vector<bool> seen(static_cast<std::size_t>(dimensions), false);
  for (const auto axis : axes) {
    if (axis < 0 || axis >= dimensions)
      throw std::out_of_range("transpose: axis out of range");
    if (seen[static_cast<std::size_t>(axis)])
      throw std::invalid_argument("transpose: axes must be unique");
    seen[static_cast<std::size_t>(axis)] = true;
  }

  tf::small_vector<int, 3> output_shape;
  for (int dimension = 0; dimension < dimensions; ++dimension)
    output_shape.push_back(
        input_shape[axes[static_cast<std::size_t>(dimension)]]);

  std::vector<std::size_t> input_strides(static_cast<std::size_t>(dimensions));
  input_strides.back() = 1;
  for (int dimension = dimensions - 2; dimension >= 0; --dimension)
    input_strides[static_cast<std::size_t>(dimension)] =
        input_strides[static_cast<std::size_t>(dimension + 1)] *
        static_cast<std::size_t>(input_shape[dimension + 1]);

  std::vector<std::size_t> output_strides(static_cast<std::size_t>(dimensions));
  output_strides.back() = 1;
  for (int dimension = dimensions - 2; dimension >= 0; --dimension)
    output_strides[static_cast<std::size_t>(dimension)] =
        output_strides[static_cast<std::size_t>(dimension + 1)] *
        static_cast<std::size_t>(output_shape[dimension + 1]);

  tf::buffer<T> buffer;
  buffer.allocate(array.length());
  auto *output = buffer.data();
  const auto *input = array.raw_data();
  const auto transpose_range = [&](std::size_t begin, std::size_t end) {
    for (auto flat_index = begin; flat_index < end; ++flat_index) {
      auto remainder = flat_index;
      std::size_t source_index = 0;
      for (int dimension = 0; dimension < dimensions; ++dimension) {
        const auto coordinate =
            remainder / output_strides[static_cast<std::size_t>(dimension)];
        remainder %= output_strides[static_cast<std::size_t>(dimension)];
        source_index +=
            coordinate * input_strides[static_cast<std::size_t>(
                             axes[static_cast<std::size_t>(dimension)])];
      }
      output[flat_index] = input[source_index];
    }
  };
  tf::parallel_for(std::size_t{0}, array.length(), transpose_range,
                   checked_work(1));
  return nd_array<T>::from_buffer(std::move(buffer), std::move(output_shape));
}

template <typename T>
auto where(const nd_array<std::int8_t> &condition, const nd_array<T> &x,
           const nd_array<T> &y) -> nd_array<T> {
  if (!condition.is_valid() || !x.is_valid() || !y.is_valid())
    throw std::invalid_argument("where: inputs must be valid arrays");
  if (!detail::shapes_equal(condition.raw_shape(), x.raw_shape()) ||
      !detail::shapes_equal(x.raw_shape(), y.raw_shape()))
    throw std::invalid_argument("where: input shapes must match");
  tf::buffer<T> buffer;
  buffer.allocate(x.length());
  auto *output = buffer.data();
  const auto *condition_data = condition.raw_data();
  const auto *x_data = x.raw_data();
  const auto *y_data = y.raw_data();
  const auto select_range = [=](std::size_t begin, std::size_t end) {
    for (auto index = begin; index < end; ++index)
      output[index] = condition_data[index] ? x_data[index] : y_data[index];
  };
  tf::parallel_for(std::size_t{0}, x.length(), select_range, checked_work(1));
  return nd_array<T>::from_buffer(std::move(buffer), x.raw_shape());
}

template <typename T> auto clone(const nd_array<T> &array) -> nd_array<T> {
  return array.deep_copy();
}

#define TF_CPP_INSTANTIATE_ND_ARRAY_STRUCTURE(T)                               \
  template auto stack<T>(const std::vector<nd_array<T>> &, int)                \
      -> nd_array<T>;                                                          \
  template auto concatenate<T>(const std::vector<nd_array<T>> &, int)          \
      -> nd_array<T>;                                                          \
  template auto tile<T>(const nd_array<T> &, const std::vector<int> &)         \
      -> nd_array<T>;                                                          \
  template auto transpose<T>(const nd_array<T> &) -> nd_array<T>;              \
  template auto transpose<T>(const nd_array<T> &, const std::vector<int> &)    \
      -> nd_array<T>;                                                          \
  template auto where<T>(const nd_array<std::int8_t> &, const nd_array<T> &,   \
                         const nd_array<T> &) -> nd_array<T>;                  \
  template auto clone<T>(const nd_array<T> &) -> nd_array<T>

TF_CPP_INSTANTIATE_ND_ARRAY_STRUCTURE(std::int8_t);
TF_CPP_INSTANTIATE_ND_ARRAY_STRUCTURE(std::int32_t);
TF_CPP_INSTANTIATE_ND_ARRAY_STRUCTURE(float);
TF_CPP_INSTANTIATE_ND_ARRAY_STRUCTURE(double);

#undef TF_CPP_INSTANTIATE_ND_ARRAY_STRUCTURE

} // namespace tf::cpp
