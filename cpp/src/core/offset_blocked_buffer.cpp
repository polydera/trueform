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
#include "trueform/cpp/core/offset_blocked_buffer.hpp"

#include "trueform/core/algorithm/parallel_contains.hpp"
#include "trueform/core/algorithm/parallel_transform.hpp"
#include "trueform/core/checked.hpp"
#include "trueform/core/views/sequence_range.hpp"
#include "trueform/core/views/slide_range.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>

namespace tf::cpp {

template <typename IndexT, typename ValueT>
offset_blocked_buffer<IndexT, ValueT>::offset_blocked_buffer(
    nd_array<IndexT> offsets, nd_array<ValueT> data)
    : _offsets(std::move(offsets)), _data(std::move(data)) {
  validate();
}

template <typename IndexT, typename ValueT>
auto offset_blocked_buffer<IndexT, ValueT>::create(nd_array<IndexT> offsets,
                                                   nd_array<ValueT> data)
    -> offset_blocked_buffer {
  return offset_blocked_buffer(std::move(offsets), std::move(data));
}

template <typename IndexT, typename ValueT>
auto offset_blocked_buffer<IndexT, ValueT>::from_buffer(
    tf::offset_block_buffer<IndexT, ValueT> &&buffer) -> offset_blocked_buffer {
  auto offsets =
      nd_array<IndexT>::from_buffer(std::move(buffer.offsets_buffer()));
  auto data = nd_array<ValueT>::from_buffer(std::move(buffer.data_buffer()));
  return create(std::move(offsets), std::move(data));
}

template <typename IndexT, typename ValueT>
auto offset_blocked_buffer<IndexT, ValueT>::from_uniform(
    const nd_array<ValueT> &array) -> offset_blocked_buffer {
  if (!array.is_valid())
    throw std::invalid_argument(
        "offset_blocked_buffer::from_uniform: input must be valid");
  if (array.ndim() != 2)
    throw std::invalid_argument(
        "offset_blocked_buffer::from_uniform: input must be two-dimensional");

  const auto rows = static_cast<std::size_t>(array.shape_at(0));
  const auto width = static_cast<std::size_t>(array.shape_at(1));
  if (rows == static_cast<std::size_t>(std::numeric_limits<int>::max()))
    throw std::length_error(
        "offset_blocked_buffer::from_uniform: offset count exceeds int range");
  if (array.length() >
      static_cast<std::size_t>(std::numeric_limits<IndexT>::max()))
    throw std::overflow_error(
        "offset_blocked_buffer::from_uniform: data size exceeds offset range");

  tf::buffer<IndexT> offsets_buffer;
  offsets_buffer.allocate(rows + 1);
  tf::parallel_transform(
      tf::make_sequence_range(rows + 1), offsets_buffer,
      [width](std::size_t row) { return static_cast<IndexT>(row * width); },
      tf::checked);

  auto offsets = nd_array<IndexT>::from_buffer(std::move(offsets_buffer));
  auto data = array.deep_copy().reshape({array.size()});
  return create(std::move(offsets), std::move(data));
}

template <typename IndexT, typename ValueT>
auto offset_blocked_buffer<IndexT, ValueT>::offsets() const
    -> nd_array<IndexT> {
  return _offsets;
}

template <typename IndexT, typename ValueT>
auto offset_blocked_buffer<IndexT, ValueT>::data() const -> nd_array<ValueT> {
  return _data;
}

template <typename IndexT, typename ValueT>
auto offset_blocked_buffer<IndexT, ValueT>::size() const -> int {
  return is_valid() && !_offsets.empty() ? _offsets.size() - 1 : 0;
}

template <typename IndexT, typename ValueT>
auto offset_blocked_buffer<IndexT, ValueT>::get(int index) const
    -> nd_array<ValueT> {
  if (!is_valid())
    throw std::logic_error("offset_blocked_buffer: buffer is not valid");
  if (index < 0 || index >= size())
    throw std::out_of_range("offset_blocked_buffer: block index out of range");
  return _data.slice(static_cast<int>(_offsets[index]),
                     static_cast<int>(_offsets[index + 1]));
}

template <typename IndexT, typename ValueT>
auto offset_blocked_buffer<IndexT, ValueT>::shallow_copy() const
    -> offset_blocked_buffer {
  return *this;
}

template <typename IndexT, typename ValueT>
auto offset_blocked_buffer<IndexT, ValueT>::deep_copy() const
    -> offset_blocked_buffer {
  if (!is_valid())
    return {};
  return create(_offsets.deep_copy(), _data.deep_copy());
}

template <typename IndexT, typename ValueT>
auto offset_blocked_buffer<IndexT, ValueT>::destroy() -> void {
  _offsets.destroy();
  _data.destroy();
}

template <typename IndexT, typename ValueT>
auto offset_blocked_buffer<IndexT, ValueT>::is_valid() const -> bool {
  return _offsets.is_valid() && _data.is_valid();
}

template <typename IndexT, typename ValueT>
auto offset_blocked_buffer<IndexT, ValueT>::validate() const -> void {
  if (!_offsets.is_valid() || !_data.is_valid())
    throw std::invalid_argument(
        "offset_blocked_buffer: offsets and data must be valid");
  if (_offsets.ndim() != 1 || _data.ndim() != 1)
    throw std::invalid_argument(
        "offset_blocked_buffer: offsets and data must be one-dimensional");
  if (_offsets.length() == 0) {
    if (!_data.empty())
      throw std::invalid_argument(
          "offset_blocked_buffer: empty offsets require empty data");
    return;
  }
  if (_offsets[0] != IndexT{0})
    throw std::invalid_argument(
        "offset_blocked_buffer: first offset must be zero");

  const auto disordered = tf::parallel_contains(
      tf::make_slide_range<2>(_offsets.make_range()),
      [](const auto &block) { return block[1] < block[0]; }, tf::checked);
  if (disordered)
    throw std::invalid_argument(
        "offset_blocked_buffer: offsets must be nondecreasing");
  if (_offsets[_offsets.length() - 1] != static_cast<IndexT>(_data.length()))
    throw std::invalid_argument(
        "offset_blocked_buffer: final offset must equal data size");
}

auto as_offset_blocked(const nd_array<std::int32_t> &array)
    -> offset_blocked_buffer<std::int32_t, std::int32_t> {
  return offset_blocked_buffer<std::int32_t, std::int32_t>::from_uniform(array);
}

auto as_offset_blocked(const nd_array<std::int64_t> &array)
    -> offset_blocked_buffer<std::int64_t, std::int64_t> {
  return offset_blocked_buffer<std::int64_t, std::int64_t>::from_uniform(array);
}

template class offset_blocked_buffer<std::int32_t, std::int32_t>;
template class offset_blocked_buffer<std::int64_t, std::int64_t>;

} // namespace tf::cpp
