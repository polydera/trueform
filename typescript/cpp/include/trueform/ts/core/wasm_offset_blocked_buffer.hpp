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
#include "trueform/core/offset_block_buffer.hpp"
#include "trueform/core/range.hpp"
#include "trueform/core/views/offset_block_range.hpp"
#include "trueform/ts/core/wasm_ndarray.hpp"
#include <memory>

namespace tf {
namespace ts {

/// @brief WASM-resident offset-blocked array with shared ownership.
///
/// Wraps two shared buffers (offsets + data) and exposes block-level access
/// as wasm_ndarray views. Each view shares the underlying data buffer —
/// no copies or allocations on access.
///
/// Used for topology structures (face_membership, vertex_link, face_link).
///
/// @tparam IndexT  Offset index type (typically int).
/// @tparam ValueT  Data element type (typically int).
template <typename IndexT, typename ValueT>
class wasm_offset_blocked_buffer {
  std::shared_ptr<tf::buffer<IndexT>> _offsets; // [N+1]
  std::shared_ptr<tf::buffer<ValueT>> _data;    // [total]
  int _size; // number of blocks = offsets.size() - 1

public:
  wasm_offset_blocked_buffer() : _size(0) {}

  wasm_offset_blocked_buffer(std::shared_ptr<tf::buffer<IndexT>> offsets,
                            std::shared_ptr<tf::buffer<ValueT>> data, int size)
      : _offsets(std::move(offsets)), _data(std::move(data)), _size(size) {}

  /// Construct from two ndarrays (shares storage).
  static auto create(wasm_ndarray<IndexT> offsets, wasm_ndarray<ValueT> data)
      -> wasm_offset_blocked_buffer {
    auto size = static_cast<int>(offsets.size()) - 1;
    return wasm_offset_blocked_buffer(offsets.raw_storage(), data.raw_storage(),
                                      size);
  }

  /// Construct from an offset_block_buffer (moves data, wraps in shared_ptrs).
  static auto from_buffer(tf::offset_block_buffer<IndexT, ValueT> &&ob)
      -> wasm_offset_blocked_buffer {
    auto size = static_cast<int>(ob.size());
    auto offsets =
        std::make_shared<tf::buffer<IndexT>>(std::move(ob.offsets_buffer()));
    auto data =
        std::make_shared<tf::buffer<ValueT>>(std::move(ob.data_buffer()));
    return wasm_offset_blocked_buffer(std::move(offsets), std::move(data), size);
  }

  /// Returns offsets as a wasm_ndarray view [N+1]. Shares storage.
  auto offsets() const -> wasm_ndarray<IndexT> {
    if (!_offsets)
      return {};
    auto len = _offsets->size();
    return wasm_ndarray<IndexT>(_offsets, 0, len,
                              {static_cast<int>(len)});
  }

  /// Returns data as a wasm_ndarray view [total]. Shares storage.
  auto data() const -> wasm_ndarray<ValueT> {
    if (!_data)
      return {};
    auto len = _data->size();
    return wasm_ndarray<ValueT>(_data, 0, len,
                              {static_cast<int>(len)});
  }

  /// Number of blocks.
  auto size() const -> int { return _size; }

  /// Returns a view of block i as wasm_ndarray. Shares _data, no allocation.
  auto get(int i) const -> wasm_ndarray<ValueT> {
    if (!_offsets || !_data || i < 0 || i >= _size)
      return {};
    auto start = static_cast<std::size_t>((*_offsets)[i]);
    auto end = static_cast<std::size_t>((*_offsets)[i + 1]);
    auto len = end - start;
    return wasm_ndarray<ValueT>(_data, start, len,
                              {static_cast<int>(len)});
  }

  auto destroy() -> void {
    _offsets.reset();
    _data.reset();
    _size = 0;
  }

  auto is_valid() const -> bool { return _offsets != nullptr; }

  // -- Internal: C++ view construction (like Python make_range) --

  auto make_range() {
    auto offsets_range =
        tf::make_range(_offsets->data(), _offsets->size());
    auto data_range = tf::make_range(_data->data(), _data->size());
    return tf::make_offset_block_range(offsets_range, data_range);
  }

  auto make_range() const {
    auto offsets_range =
        tf::make_range(const_cast<const IndexT *>(_offsets->data()),
                       _offsets->size());
    auto data_range =
        tf::make_range(const_cast<const ValueT *>(_data->data()),
                       _data->size());
    return tf::make_offset_block_range(offsets_range, data_range);
  }

  // -- Internal: raw buffer access --

  auto raw_offsets() -> std::shared_ptr<tf::buffer<IndexT>> & {
    return _offsets;
  }

  auto raw_data() -> std::shared_ptr<tf::buffer<ValueT>> & { return _data; }

  auto raw_offsets() const -> const std::shared_ptr<tf::buffer<IndexT>> & {
    return _offsets;
  }

  auto raw_data() const -> const std::shared_ptr<tf::buffer<ValueT>> & {
    return _data;
  }
};

} // namespace ts
} // namespace tf
