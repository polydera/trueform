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
#include <emscripten/bind.h>
#include <emscripten/val.h>
#include <memory>

namespace tf {
namespace ts {

/// @brief WASM-resident array with shared ownership and shape metadata.
///
/// Foundation type for all WASM-resident data in TypeScript bindings.
/// Instantiated for int (NativeInt32NDArray) and float (NativeFloat32NDArray).
///
/// Ownership model:
/// - Each wasm_ndarray holds a shared_ptr to a tf::buffer<T>.
/// - Views (row, slice) share the same shared_ptr.
/// - Aliasing shared_ptr can keep a parent (e.g. mesh) alive.
/// - destroy() calls shared_ptr::reset(), which is idempotent.
///
/// @tparam T Element type (int or float).
template <typename T> class wasm_ndarray {
  std::shared_ptr<tf::buffer<T>> _storage;
  std::size_t _offset = 0;
  std::size_t _length = 0;
  tf::small_vector<int, 3> _shape;

public:
  wasm_ndarray() = default;

  wasm_ndarray(std::shared_ptr<tf::buffer<T>> storage, std::size_t offset,
             std::size_t length, tf::small_vector<int, 3> shape)
      : _storage(std::move(storage)), _offset(offset), _length(length),
        _shape(std::move(shape)) {}

  // -- Data access --

  /// Returns a typed_memory_view into the WASM heap.
  auto data() const -> emscripten::val {
    if (!_storage)
      return emscripten::val::undefined();
    return emscripten::val(
        emscripten::typed_memory_view(_length, _storage->data() + _offset));
  }

  // -- Metadata --

  auto size() const -> int { return static_cast<int>(_length); }

  auto ndim() const -> int { return static_cast<int>(_shape.size()); }

  auto shape() const -> emscripten::val {
    auto result = emscripten::val::array();
    for (int i = 0; i < static_cast<int>(_shape.size()); ++i) {
      result.call<void>("push", _shape[i]);
    }
    return result;
  }

  auto shape_at(int i) const -> int { return _shape[i]; }

  auto set_shape(emscripten::val js_shape) -> void {
    tf::small_vector<int, 3> new_shape;
    auto ndim = js_shape["length"].as<int>();
    std::size_t total = 1;
    for (int i = 0; i < ndim; ++i) {
      int d = js_shape[i].as<int>();
      new_shape.push_back(d);
      total *= d;
    }
    if (total != _length)
      throw std::runtime_error("reshape: total size mismatch");
    _shape = std::move(new_shape);
  }

  // -- Views --

  /// Returns a view of the i-th element along the first axis.
  /// Shares _storage. Shape is _shape[1:].
  auto row(int i) const -> wasm_ndarray<T> {
    auto stride = _compute_row_stride();
    tf::small_vector<int, 3> new_shape;
    for (int d = 1; d < static_cast<int>(_shape.size()); ++d) {
      new_shape.push_back(_shape[d]);
    }
    if (new_shape.empty())
      new_shape.push_back(1);
    return wasm_ndarray<T>(_storage, _offset + i * stride, stride,
                         std::move(new_shape));
  }

  /// Returns a view of rows [start, end) along the first axis.
  /// Shares _storage. Shape is {end-start, ...rest}.
  auto slice(int start, int end) const -> wasm_ndarray<T> {
    auto stride = _compute_row_stride();
    auto new_rows = end - start;
    tf::small_vector<int, 3> new_shape = _shape;
    new_shape[0] = new_rows;
    return wasm_ndarray<T>(_storage, _offset + start * stride, new_rows * stride,
                         std::move(new_shape));
  }

  // -- Lifecycle --

  /// Releases the shared_ptr to the buffer and resets all metadata. Idempotent.
  auto destroy() -> void {
    _storage.reset();
    _offset = 0;
    _length = 0;
    _shape.clear();
  }

  auto is_valid() const -> bool { return _storage != nullptr; }

  auto shallow_copy() const -> wasm_ndarray<T> {
    return wasm_ndarray<T>(_storage, _offset, _length, _shape);
  }

  // -- Construction helpers --

  /// Create from a JS TypedArray + shape array.
  static auto from_js(emscripten::val js_array, emscripten::val js_shape)
      -> wasm_ndarray<T> {
    auto length = js_array["length"].as<std::size_t>();
    auto buf = std::make_shared<tf::buffer<T>>();
    buf->allocate(length);

    // Copy from JS into WASM heap
    auto mem_view = emscripten::val(
        emscripten::typed_memory_view(length, buf->data()));
    mem_view.call<void>("set", js_array);

    tf::small_vector<int, 3> shape;
    auto shape_length = js_shape["length"].as<int>();
    for (int i = 0; i < shape_length; ++i) {
      shape.push_back(js_shape[i].as<int>());
    }

    return wasm_ndarray<T>(std::move(buf), 0, length, std::move(shape));
  }

  /// Create from a C++ buffer (takes ownership via move).
  static auto from_buffer(tf::buffer<T> &&buf, tf::small_vector<int, 3> shape)
      -> wasm_ndarray<T> {
    auto length = buf.size();
    auto shared =
        std::make_shared<tf::buffer<T>>(std::move(buf));
    return wasm_ndarray<T>(std::move(shared), 0, length, std::move(shape));
  }

  /// Create from a C++ buffer with flat shape (takes ownership via move).
  static auto from_buffer(tf::buffer<T> &&buf) -> wasm_ndarray<T> {
    auto length = buf.size();
    return from_buffer(std::move(buf), {static_cast<int>(length)});
  }

  // -- Internal: C++ view construction --

  auto make_range() {
    return tf::make_range(_storage->data() + _offset, _length);
  }

  auto make_range() const {
    return tf::make_range(
        const_cast<const T *>(_storage->data() + _offset), _length);
  }

  // -- Internal: raw access --

  auto raw_data() -> T * {
    return _storage ? _storage->data() + _offset : nullptr;
  }

  auto raw_data() const -> const T * {
    return _storage ? _storage->data() + _offset : nullptr;
  }

  auto raw_storage() -> std::shared_ptr<tf::buffer<T>> & { return _storage; }

  auto raw_storage() const -> const std::shared_ptr<tf::buffer<T>> & {
    return _storage;
  }

  auto offset() const -> std::size_t { return _offset; }

  auto length() const -> std::size_t { return _length; }

  auto raw_shape() const -> const tf::small_vector<int, 3> & { return _shape; }

private:
  auto _compute_row_stride() const -> std::size_t {
    std::size_t stride = 1;
    for (int d = 1; d < static_cast<int>(_shape.size()); ++d) {
      stride *= _shape[d];
    }
    return stride;
  }
};

} // namespace ts
} // namespace tf
