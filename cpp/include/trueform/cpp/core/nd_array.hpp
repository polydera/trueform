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
#pragma once

#include "trueform/core/buffer.hpp"
#include "trueform/core/range.hpp"
#include "trueform/core/small_vector.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>

namespace tf::cpp {

/// @brief Contiguous n-dimensional array over storage it owns or borrows.
///
/// The array is its first element, its length, its shape, and the keepalive
/// that must outlive every copy of it: from_buffer() keeps the buffer it was
/// handed, from_borrowed() keeps whatever the caller hands over and an empty
/// keepalive means the caller guarantees the memory instead. Copies and views
/// share that storage and carry their own shape; deep_copy() makes an owning
/// array out of any of them.
template <typename T> class nd_array {
  std::shared_ptr<void> _owner;
  T *_data = nullptr;
  std::size_t _length = 0;
  tf::small_vector<int, 3> _shape;

public:
  nd_array() = default;

  static auto from_buffer(tf::buffer<T> &&buffer,
                          tf::small_vector<int, 3> shape) -> nd_array;
  static auto from_buffer(tf::buffer<T> &&buffer) -> nd_array;
  /// @brief An array over memory the caller owns.
  ///
  /// `owner` is retained for the lifetime of the array and of every copy and
  /// view taken from it; an empty `owner` transfers that guarantee to the
  /// caller. Writes through the array land in the caller's memory, so borrow
  /// read-only storage only to deep_copy() it.
  static auto from_borrowed(std::shared_ptr<void> owner, T *data,
                            std::size_t length, tf::small_vector<int, 3> shape)
      -> nd_array;

  auto size() const -> int;
  auto length() const -> std::size_t;
  auto empty() const -> bool;
  auto ndim() const -> int;
  auto shape_at(int index) const -> int;
  auto raw_shape() const -> const tf::small_vector<int, 3> &;

  auto set_shape(tf::small_vector<int, 3> shape) -> void;
  auto reshape(tf::small_vector<int, 3> shape) const -> nd_array;

  auto row(int index) const -> nd_array;
  auto slice(int start, int end) const -> nd_array;

  auto shallow_copy() const -> nd_array;
  auto deep_copy() const -> nd_array;

  auto destroy() -> void;
  auto is_valid() const -> bool;

  auto raw_data() -> T *;
  auto raw_data() const -> const T *;
  /// @brief The keepalive this array stands on, empty when the caller owns it.
  auto raw_owner() const -> const std::shared_ptr<void> &;

  auto operator[](std::size_t index) -> T &;
  auto operator[](std::size_t index) const -> const T &;
  auto at(std::size_t index) -> T &;
  auto at(std::size_t index) const -> const T &;

  auto begin() -> T * { return _data; }
  auto end() -> T * { return _data + _length; }
  auto begin() const -> const T * { return _data; }
  auto end() const -> const T * { return _data + _length; }

  auto make_range() { return tf::make_range(_data, _length); }
  auto make_range() const {
    return tf::make_range(static_cast<const T *>(_data), _length);
  }

private:
  nd_array(std::shared_ptr<void> owner, T *data, std::size_t length,
           tf::small_vector<int, 3> shape);

  static auto checked_shape_size(const tf::small_vector<int, 3> &shape)
      -> std::size_t;
  auto row_stride() const -> std::size_t;
};

extern template class nd_array<std::int8_t>;
extern template class nd_array<std::int32_t>;
extern template class nd_array<std::int64_t>;
extern template class nd_array<float>;
extern template class nd_array<double>;

} // namespace tf::cpp
