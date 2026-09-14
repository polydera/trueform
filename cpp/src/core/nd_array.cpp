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
#include "trueform/cpp/core/nd_array.hpp"

#include "trueform/core/algorithm/parallel_copy.hpp"
#include "trueform/core/buffer.hpp"
#include "trueform/core/small_vector.hpp"

#include <tbb/global_control.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>

namespace tf::cpp {
namespace {

constexpr std::size_t serial_copy_max_bytes = 1U * 1024U * 1024U;

template <typename T> constexpr auto serial_copy_max_elements() -> std::size_t {
  // Convert the byte crossover to elements instead of multiplying _length by
  // sizeof(T), which could overflow size_t on 32-bit Wasm.
  return serial_copy_max_bytes / sizeof(T);
}

} // namespace

template <typename T>
nd_array<T>::nd_array(std::shared_ptr<void> owner, T *data, std::size_t length,
                      tf::small_vector<int, 3> shape)
    : _owner(std::move(owner)), _data(data), _length(length),
      _shape(std::move(shape)) {
  if (checked_shape_size(_shape) != _length)
    throw std::invalid_argument("nd_array: shape does not match data size");
  if (_length > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    throw std::length_error("nd_array: size exceeds int range");
}

template <typename T>
auto nd_array<T>::from_buffer(tf::buffer<T> &&buffer,
                              tf::small_vector<int, 3> shape) -> nd_array {
  auto storage = std::make_shared<tf::buffer<T>>(std::move(buffer));
  auto *const data = storage->data();
  const auto length = storage->size();
  return nd_array(std::shared_ptr<void>(std::move(storage)), data, length,
                  std::move(shape));
}

template <typename T>
auto nd_array<T>::from_buffer(tf::buffer<T> &&buffer) -> nd_array {
  const auto length = buffer.size();
  return from_buffer(std::move(buffer), {static_cast<int>(length)});
}

template <typename T>
auto nd_array<T>::from_borrowed(std::shared_ptr<void> owner, T *data,
                                std::size_t length,
                                tf::small_vector<int, 3> shape) -> nd_array {
  return nd_array(std::move(owner), data, length, std::move(shape));
}

template <typename T> auto nd_array<T>::size() const -> int {
  return static_cast<int>(_length);
}

template <typename T> auto nd_array<T>::length() const -> std::size_t {
  return _length;
}

template <typename T> auto nd_array<T>::empty() const -> bool {
  return _length == 0;
}

template <typename T> auto nd_array<T>::ndim() const -> int {
  return static_cast<int>(_shape.size());
}

template <typename T> auto nd_array<T>::shape_at(int index) const -> int {
  if (index < 0 || index >= ndim())
    throw std::out_of_range("nd_array: shape index out of range");
  return _shape[static_cast<std::size_t>(index)];
}

template <typename T>
auto nd_array<T>::raw_shape() const -> const tf::small_vector<int, 3> & {
  return _shape;
}

template <typename T>
auto nd_array<T>::set_shape(tf::small_vector<int, 3> shape) -> void {
  if (checked_shape_size(shape) != _length)
    throw std::invalid_argument("reshape: total size mismatch");
  _shape = std::move(shape);
}

template <typename T>
auto nd_array<T>::reshape(tf::small_vector<int, 3> shape) const -> nd_array {
  auto result = shallow_copy();
  result.set_shape(std::move(shape));
  return result;
}

template <typename T> auto nd_array<T>::row(int index) const -> nd_array {
  if (!is_valid())
    throw std::logic_error("nd_array: array is not valid");
  if (index < 0 || index >= _shape[0])
    throw std::out_of_range("nd_array: row index out of range");

  const auto stride = row_stride();
  tf::small_vector<int, 3> shape;
  for (std::size_t dimension = 1; dimension < _shape.size(); ++dimension)
    shape.push_back(_shape[dimension]);
  if (shape.empty())
    shape.push_back(1);
  return nd_array(_owner, _data + static_cast<std::size_t>(index) * stride,
                  stride, std::move(shape));
}

template <typename T>
auto nd_array<T>::slice(int start, int end) const -> nd_array {
  if (!is_valid())
    throw std::logic_error("nd_array: array is not valid");
  if (start < 0 || end < start || end > _shape[0])
    throw std::out_of_range("nd_array: slice bounds out of range");

  const auto stride = row_stride();
  auto shape = _shape;
  shape[0] = end - start;
  const auto rows = static_cast<std::size_t>(end - start);
  return nd_array(_owner, _data + static_cast<std::size_t>(start) * stride,
                  rows * stride, std::move(shape));
}

template <typename T> auto nd_array<T>::shallow_copy() const -> nd_array {
  return *this;
}

template <typename T> auto nd_array<T>::deep_copy() const -> nd_array {
  if (!is_valid())
    return {};
  tf::buffer<T> buffer;
  buffer.allocate(_length);
  if (_length != 0) {
    const auto effective_parallelism = tbb::global_control::active_value(
        tbb::global_control::max_allowed_parallelism);
    if (effective_parallelism <= 1 || _length <= serial_copy_max_elements<T>())
      std::copy(raw_data(), raw_data() + _length, buffer.begin());
    else
      tf::parallel_copy(make_range(), buffer);
  }
  return from_buffer(std::move(buffer), _shape);
}

template <typename T> auto nd_array<T>::destroy() -> void {
  _owner.reset();
  _data = nullptr;
  _length = 0;
  _shape.clear();
}

template <typename T> auto nd_array<T>::is_valid() const -> bool {
  return !_shape.empty();
}

template <typename T> auto nd_array<T>::raw_data() -> T * { return _data; }

template <typename T> auto nd_array<T>::raw_data() const -> const T * {
  return _data;
}

template <typename T>
auto nd_array<T>::raw_owner() const -> const std::shared_ptr<void> & {
  return _owner;
}

template <typename T> auto nd_array<T>::operator[](std::size_t index) -> T & {
  return _data[index];
}

template <typename T>
auto nd_array<T>::operator[](std::size_t index) const -> const T & {
  return _data[index];
}

template <typename T> auto nd_array<T>::at(std::size_t index) -> T & {
  if (index >= _length)
    throw std::out_of_range("nd_array: element index out of range");
  return (*this)[index];
}

template <typename T>
auto nd_array<T>::at(std::size_t index) const -> const T & {
  if (index >= _length)
    throw std::out_of_range("nd_array: element index out of range");
  return (*this)[index];
}

template <typename T>
auto nd_array<T>::checked_shape_size(const tf::small_vector<int, 3> &shape)
    -> std::size_t {
  if (shape.empty())
    throw std::invalid_argument("nd_array: shape must not be empty");

  std::size_t total = 1;
  for (const auto dimension : shape) {
    if (dimension < 0)
      throw std::invalid_argument(
          "nd_array: shape dimensions must be nonnegative");
    const auto value = static_cast<std::size_t>(dimension);
    if (value != 0 && total > std::numeric_limits<std::size_t>::max() / value)
      throw std::overflow_error("nd_array: shape size overflows size_t");
    total *= value;
  }
  return total;
}

template <typename T> auto nd_array<T>::row_stride() const -> std::size_t {
  std::size_t stride = 1;
  for (std::size_t dimension = 1; dimension < _shape.size(); ++dimension)
    stride *= static_cast<std::size_t>(_shape[dimension]);
  return stride;
}

template class nd_array<std::int8_t>;
template class nd_array<std::int32_t>;
template class nd_array<std::int64_t>;
template class nd_array<float>;
template class nd_array<double>;

} // namespace tf::cpp
