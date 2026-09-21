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

#include "trueform/core/offset_block_buffer.hpp"
#include "trueform/core/views/offset_block_range.hpp"
#include "trueform/cpp/core/nd_array.hpp"

#include <cstdint>
#include <memory>

namespace tf::cpp {

/// @brief Shared-storage offset-blocked buffer.
template <typename IndexT, typename ValueT> class offset_blocked_buffer {
  nd_array<IndexT> _offsets;
  nd_array<ValueT> _data;

public:
  offset_blocked_buffer() = default;
  offset_blocked_buffer(nd_array<IndexT> offsets, nd_array<ValueT> data);

  static auto create(nd_array<IndexT> offsets, nd_array<ValueT> data)
      -> offset_blocked_buffer;
  static auto from_buffer(tf::offset_block_buffer<IndexT, ValueT> &&buffer)
      -> offset_blocked_buffer;
  /// @brief Deep-copy a two-dimensional uniform array into offset blocks.
  static auto from_uniform(const nd_array<ValueT> &array)
      -> offset_blocked_buffer;

  auto offsets() const -> nd_array<IndexT>;
  auto data() const -> nd_array<ValueT>;
  auto size() const -> int;
  auto get(int index) const -> nd_array<ValueT>;

  auto shallow_copy() const -> offset_blocked_buffer;
  auto deep_copy() const -> offset_blocked_buffer;
  auto destroy() -> void;
  auto is_valid() const -> bool;

  auto make_range() {
    return tf::make_offset_block_range(_offsets.make_range(),
                                       _data.make_range());
  }
  auto make_range() const {
    return tf::make_offset_block_range(_offsets.make_range(),
                                       _data.make_range());
  }

  auto raw_offsets() const -> const std::shared_ptr<void> & {
    return _offsets.raw_owner();
  }
  auto raw_data() const -> const std::shared_ptr<void> & {
    return _data.raw_owner();
  }

private:
  auto validate() const -> void;
};

/// @brief Convert a uniform integer array to owning offset-blocked storage.
auto as_offset_blocked(const nd_array<std::int32_t> &array)
    -> offset_blocked_buffer<std::int32_t, std::int32_t>;
auto as_offset_blocked(const nd_array<std::int64_t> &array)
    -> offset_blocked_buffer<std::int64_t, std::int64_t>;

extern template class offset_blocked_buffer<std::int32_t, std::int32_t>;
extern template class offset_blocked_buffer<std::int64_t, std::int64_t>;

} // namespace tf::cpp
