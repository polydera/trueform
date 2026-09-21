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
#include "trueform/cpp/core/nd_array_sorting.hpp"

#include "trueform/core/algorithm/parallel_iota.hpp"
#include "trueform/core/buffer.hpp"
#include "trueform/cpp/core/nd_array_indexing.hpp"

#include <tbb/parallel_sort.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>

namespace tf::cpp {
namespace {

auto require_rows(const tf::small_vector<int, 3> &shape, const char *operation)
    -> void {
  if (shape.empty())
    throw std::invalid_argument(std::string(operation) +
                                ": array must have at least one axis");
}

auto row_stride(const tf::small_vector<int, 3> &shape) -> std::size_t {
  std::size_t stride = 1;
  for (std::size_t dimension = 1; dimension < shape.size(); ++dimension)
    stride *= static_cast<std::size_t>(shape[dimension]);
  return stride;
}

template <std::size_t Stride, typename T>
auto fixed_row_compare(const T *a, const T *b) -> int {
  for (std::size_t index = 0; index < Stride; ++index) {
    if (a[index] < b[index])
      return -1;
    if (a[index] > b[index])
      return 1;
  }
  return 0;
}

template <typename T>
auto row_compare(const T *a, const T *b, std::size_t stride) -> int {
  for (std::size_t index = 0; index < stride; ++index) {
    if (a[index] < b[index])
      return -1;
    if (a[index] > b[index])
      return 1;
  }
  return 0;
}

template <std::size_t Stride, typename T>
auto sort_row_indices(tf::buffer<std::int32_t> &indices, const T *source)
    -> void {
  tbb::parallel_sort(
      indices.begin(), indices.end(), [source](std::int32_t a, std::int32_t b) {
        return fixed_row_compare<Stride>(
                   source + static_cast<std::size_t>(a) * Stride,
                   source + static_cast<std::size_t>(b) * Stride) < 0;
      });
}

template <typename T>
auto sort_row_indices(tf::buffer<std::int32_t> &indices, const T *source,
                      std::size_t stride) -> void {
  switch (stride) {
  case 1:
    return sort_row_indices<1>(indices, source);
  case 2:
    return sort_row_indices<2>(indices, source);
  case 3:
    return sort_row_indices<3>(indices, source);
  case 4:
    return sort_row_indices<4>(indices, source);
  case 8:
    return sort_row_indices<8>(indices, source);
  default:
    tbb::parallel_sort(
        indices.begin(), indices.end(),
        [source, stride](std::int32_t a, std::int32_t b) {
          return row_compare(source + static_cast<std::size_t>(a) * stride,
                             source + static_cast<std::size_t>(b) * stride,
                             stride) < 0;
        });
  }
}

template <typename T>
auto require_compatible_rows(const nd_array<T> &a, const nd_array<T> &b,
                             const char *operation) -> std::size_t {
  require_rows(a.raw_shape(), operation);
  require_rows(b.raw_shape(), operation);
  if (a.ndim() != b.ndim())
    throw std::invalid_argument(std::string(operation) + ": ndim mismatch");
  for (int dimension = 1; dimension < a.ndim(); ++dimension)
    if (a.shape_at(dimension) != b.shape_at(dimension))
      throw std::invalid_argument(std::string(operation) +
                                  ": row shape mismatch");
  return row_stride(a.raw_shape());
}

template <typename T>
auto output_shape(const tf::small_vector<int, 3> &input_shape, std::size_t rows)
    -> tf::small_vector<int, 3> {
  auto shape = input_shape;
  shape[0] = static_cast<int>(rows);
  return shape;
}

} // namespace

template <typename T>
auto argsort(const nd_array<T> &array) -> nd_array<std::int32_t> {
  const auto &shape = array.raw_shape();
  require_rows(shape, "argsort");
  const auto rows = shape.front();
  const auto stride = row_stride(shape);
  tf::buffer<std::int32_t> buffer;
  buffer.allocate(static_cast<std::size_t>(rows));
  if (rows > 0)
    tf::parallel_iota(buffer, std::int32_t{0});
  if (rows > 1 && stride > 0)
    sort_row_indices(buffer, array.raw_data(), stride);
  return nd_array<std::int32_t>::from_buffer(std::move(buffer), {rows});
}

template <typename T> auto sort(const nd_array<T> &array) -> nd_array<T> {
  require_rows(array.raw_shape(), "sort");
  if (array.ndim() == 1) {
    auto result = array.deep_copy();
    if (result.length() > 1)
      tbb::parallel_sort(result.begin(), result.end());
    return result;
  }
  if (array.length() == 0) {
    tf::buffer<T> empty;
    return nd_array<T>::from_buffer(std::move(empty), array.raw_shape());
  }
  const auto permutation = argsort(array);
  return take(array, permutation, 0);
}

template <typename T> auto sort_inplace(nd_array<T> &array) -> void {
  require_rows(array.raw_shape(), "sort_inplace");
  if (array.ndim() == 1) {
    if (array.length() > 1)
      tbb::parallel_sort(array.begin(), array.end());
    return;
  }
  if (array.length() == 0)
    return;
  const auto sorted = sort(array);
  std::memcpy(array.raw_data(), sorted.raw_data(), array.length() * sizeof(T));
}

template <typename T> auto unique(const nd_array<T> &array) -> nd_array<T> {
  const auto &shape = array.raw_shape();
  require_rows(shape, "unique");
  const auto rows = shape.front();
  const auto stride = row_stride(shape);
  if (rows == 0) {
    tf::buffer<T> empty;
    return nd_array<T>::from_buffer(std::move(empty), shape);
  }
  if (stride == 0) {
    tf::buffer<T> empty;
    return nd_array<T>::from_buffer(std::move(empty),
                                    output_shape<T>(shape, 1));
  }
  const auto *source = array.raw_data();

  std::size_t count = 1;
  for (int row = 1; row < rows; ++row)
    if (row_compare(source + static_cast<std::size_t>(row - 1) * stride,
                    source + static_cast<std::size_t>(row) * stride,
                    stride) != 0)
      ++count;

  tf::buffer<T> buffer;
  buffer.allocate(count * stride);
  auto *output = buffer.data();
  std::memcpy(output, source, stride * sizeof(T));
  output += stride;
  for (int row = 1; row < rows; ++row) {
    if (row_compare(source + static_cast<std::size_t>(row - 1) * stride,
                    source + static_cast<std::size_t>(row) * stride,
                    stride) != 0) {
      std::memcpy(output, source + static_cast<std::size_t>(row) * stride,
                  stride * sizeof(T));
      output += stride;
    }
  }
  return nd_array<T>::from_buffer(std::move(buffer),
                                  output_shape<T>(shape, count));
}

template <typename T>
auto set_union(const nd_array<T> &a, const nd_array<T> &b) -> nd_array<T> {
  const auto stride = require_compatible_rows(a, b, "set_union");
  const auto a_rows = a.shape_at(0);
  const auto b_rows = b.shape_at(0);
  if (stride == 0) {
    tf::buffer<T> empty;
    return nd_array<T>::from_buffer(
        std::move(empty),
        output_shape<T>(a.raw_shape(), std::max(a_rows, b_rows)));
  }
  const auto *a_data = a.raw_data();
  const auto *b_data = b.raw_data();
  tf::buffer<T> buffer;
  buffer.allocate(static_cast<std::size_t>(a_rows + b_rows) * stride);
  auto *output = buffer.data();
  int a_row = 0;
  int b_row = 0;
  std::size_t count = 0;
  while (a_row < a_rows && b_row < b_rows) {
    const auto comparison =
        row_compare(a_data + static_cast<std::size_t>(a_row) * stride,
                    b_data + static_cast<std::size_t>(b_row) * stride, stride);
    if (comparison < 0) {
      std::memcpy(output + count * stride,
                  a_data + static_cast<std::size_t>(a_row++) * stride,
                  stride * sizeof(T));
    } else if (comparison > 0) {
      std::memcpy(output + count * stride,
                  b_data + static_cast<std::size_t>(b_row++) * stride,
                  stride * sizeof(T));
    } else {
      std::memcpy(output + count * stride,
                  a_data + static_cast<std::size_t>(a_row++) * stride,
                  stride * sizeof(T));
      ++b_row;
    }
    ++count;
  }
  while (a_row < a_rows) {
    std::memcpy(output + count++ * stride,
                a_data + static_cast<std::size_t>(a_row++) * stride,
                stride * sizeof(T));
  }
  while (b_row < b_rows) {
    std::memcpy(output + count++ * stride,
                b_data + static_cast<std::size_t>(b_row++) * stride,
                stride * sizeof(T));
  }
  buffer.reallocate(count * stride);
  return nd_array<T>::from_buffer(std::move(buffer),
                                  output_shape<T>(a.raw_shape(), count));
}

template <typename T>
auto set_intersection(const nd_array<T> &a, const nd_array<T> &b)
    -> nd_array<T> {
  const auto stride = require_compatible_rows(a, b, "set_intersection");
  const auto a_rows = a.shape_at(0);
  const auto b_rows = b.shape_at(0);
  if (stride == 0) {
    tf::buffer<T> empty;
    return nd_array<T>::from_buffer(
        std::move(empty),
        output_shape<T>(a.raw_shape(), std::min(a_rows, b_rows)));
  }
  const auto *a_data = a.raw_data();
  const auto *b_data = b.raw_data();
  tf::buffer<T> buffer;
  buffer.allocate(static_cast<std::size_t>(std::min(a_rows, b_rows)) * stride);
  auto *output = buffer.data();
  int a_row = 0;
  int b_row = 0;
  std::size_t count = 0;
  while (a_row < a_rows && b_row < b_rows) {
    const auto comparison =
        row_compare(a_data + static_cast<std::size_t>(a_row) * stride,
                    b_data + static_cast<std::size_t>(b_row) * stride, stride);
    if (comparison < 0) {
      ++a_row;
    } else if (comparison > 0) {
      ++b_row;
    } else {
      std::memcpy(output + count++ * stride,
                  a_data + static_cast<std::size_t>(a_row++) * stride,
                  stride * sizeof(T));
      ++b_row;
    }
  }
  buffer.reallocate(count * stride);
  return nd_array<T>::from_buffer(std::move(buffer),
                                  output_shape<T>(a.raw_shape(), count));
}

template <typename T>
auto set_difference(const nd_array<T> &a, const nd_array<T> &b) -> nd_array<T> {
  const auto stride = require_compatible_rows(a, b, "set_difference");
  const auto a_rows = a.shape_at(0);
  const auto b_rows = b.shape_at(0);
  if (stride == 0) {
    tf::buffer<T> empty;
    return nd_array<T>::from_buffer(
        std::move(empty),
        output_shape<T>(a.raw_shape(), std::max(a_rows - b_rows, 0)));
  }
  const auto *a_data = a.raw_data();
  const auto *b_data = b.raw_data();
  tf::buffer<T> buffer;
  buffer.allocate(static_cast<std::size_t>(a_rows) * stride);
  auto *output = buffer.data();
  int a_row = 0;
  int b_row = 0;
  std::size_t count = 0;
  while (a_row < a_rows && b_row < b_rows) {
    const auto comparison =
        row_compare(a_data + static_cast<std::size_t>(a_row) * stride,
                    b_data + static_cast<std::size_t>(b_row) * stride, stride);
    if (comparison < 0) {
      std::memcpy(output + count++ * stride,
                  a_data + static_cast<std::size_t>(a_row++) * stride,
                  stride * sizeof(T));
    } else if (comparison > 0) {
      ++b_row;
    } else {
      ++a_row;
      ++b_row;
    }
  }
  while (a_row < a_rows) {
    std::memcpy(output + count++ * stride,
                a_data + static_cast<std::size_t>(a_row++) * stride,
                stride * sizeof(T));
  }
  buffer.reallocate(count * stride);
  return nd_array<T>::from_buffer(std::move(buffer),
                                  output_shape<T>(a.raw_shape(), count));
}

#define TF_CPP_INSTANTIATE_SORTING(T)                                          \
  template auto sort<T>(const nd_array<T> &) -> nd_array<T>;                   \
  template auto sort_inplace<T>(nd_array<T> &) -> void;                        \
  template auto argsort<T>(const nd_array<T> &) -> nd_array<std::int32_t>;     \
  template auto unique<T>(const nd_array<T> &) -> nd_array<T>;                 \
  template auto set_union<T>(const nd_array<T> &, const nd_array<T> &)         \
      -> nd_array<T>;                                                          \
  template auto set_intersection<T>(const nd_array<T> &, const nd_array<T> &)  \
      -> nd_array<T>;                                                          \
  template auto set_difference<T>(const nd_array<T> &, const nd_array<T> &)    \
      -> nd_array<T>

TF_CPP_INSTANTIATE_SORTING(std::int8_t);
TF_CPP_INSTANTIATE_SORTING(std::int32_t);
TF_CPP_INSTANTIATE_SORTING(std::int64_t);
TF_CPP_INSTANTIATE_SORTING(float);
TF_CPP_INSTANTIATE_SORTING(double);

#undef TF_CPP_INSTANTIATE_SORTING

} // namespace tf::cpp
