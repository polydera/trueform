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

#include "trueform/ts/core/promise.hpp"
#include "trueform/ts/core/reductions.hpp"
#include <cstdint>
#include <emscripten/bind.h>

// ============================================================================
// Sync — returns emscripten::val (number or NativeFloat32NDArray/NativeInt32NDArray)
// ============================================================================

template <typename T>
static auto sync_sum(tf::ts::wasm_ndarray<T> &arr, int axis) -> emscripten::val {
  if (axis == -1)
    return emscripten::val(tf::ts::sum(arr));
  return emscripten::val(tf::ts::sum(arr, axis));
}

template <typename T>
static auto sync_min(tf::ts::wasm_ndarray<T> &arr, int axis) -> emscripten::val {
  if (axis == -1)
    return emscripten::val(tf::ts::min(arr));
  return emscripten::val(tf::ts::min(arr, axis));
}

template <typename T>
static auto sync_max(tf::ts::wasm_ndarray<T> &arr, int axis) -> emscripten::val {
  if (axis == -1)
    return emscripten::val(tf::ts::max(arr));
  return emscripten::val(tf::ts::max(arr, axis));
}

template <typename T>
static auto sync_mean(tf::ts::wasm_ndarray<T> &arr, int axis)
    -> emscripten::val {
  if (axis == -1)
    return emscripten::val(tf::ts::mean(arr));
  return emscripten::val(tf::ts::mean(arr, axis));
}

// ============================================================================
// Sync — int8 specializations (sum accumulates to int to avoid overflow)
// ============================================================================

static auto sync_sum_int8(tf::ts::wasm_ndarray<std::int8_t> &arr,
                          int axis) -> emscripten::val {
  if (axis == -1) {
    auto *data = arr.raw_data();
    auto len = arr.length();
    int acc = 0;
    for (std::size_t i = 0; i < len; ++i)
      acc += static_cast<int>(data[i]);
    return emscripten::val(acc);
  }
  // axis reduce: int8 input → int output
  auto [outer, reduce, inner] =
      tf::ts::detail::axis_strides(arr.raw_shape(), axis);
  auto rshape = tf::ts::detail::remove_axis(arr.raw_shape(), axis);
  auto result_len = static_cast<std::size_t>(outer) * inner;
  auto *data = arr.raw_data();

  tf::buffer<int> buf;
  buf.allocate(result_len);
  auto *out = buf.data();

  for (std::size_t idx = 0; idx < result_len; ++idx) {
    int o = static_cast<int>(idx) / inner;
    int i = static_cast<int>(idx) % inner;
    int acc = 0;
    for (int r = 0; r < reduce; ++r)
      acc += static_cast<int>(data[o * reduce * inner + r * inner + i]);
    out[idx] = acc;
  }

  return emscripten::val(
      tf::ts::wasm_ndarray<int>::from_buffer(std::move(buf), std::move(rshape)));
}

static auto sync_min_int8(tf::ts::wasm_ndarray<std::int8_t> &arr,
                          int axis) -> emscripten::val {
  if (axis == -1)
    return emscripten::val(tf::ts::min(arr));
  return emscripten::val(tf::ts::min(arr, axis));
}

static auto sync_max_int8(tf::ts::wasm_ndarray<std::int8_t> &arr,
                          int axis) -> emscripten::val {
  if (axis == -1)
    return emscripten::val(tf::ts::max(arr));
  return emscripten::val(tf::ts::max(arr, axis));
}

static auto sync_mean_int8(tf::ts::wasm_ndarray<std::int8_t> &arr,
                           int axis) -> emscripten::val {
  if (axis == -1)
    return emscripten::val(tf::ts::mean(arr));
  return emscripten::val(tf::ts::mean(arr, axis));
}

// ============================================================================
// Async — dispatches to TBB, returns slot for Atomics.waitAsync
// ============================================================================

template <typename T>
static auto async_sum(tf::ts::wasm_ndarray<T> &arr, int axis)
    -> tf::ts::promise_t {
  if (axis == -1)
    return tf::ts::promise([a = arr]() -> T { return tf::ts::sum(a); });
  return tf::ts::promise([a = arr, axis]() -> tf::ts::wasm_ndarray<T> {
    return tf::ts::sum(a, axis);
  });
}

template <typename T>
static auto async_min(tf::ts::wasm_ndarray<T> &arr, int axis)
    -> tf::ts::promise_t {
  if (axis == -1)
    return tf::ts::promise([a = arr]() -> T { return tf::ts::min(a); });
  return tf::ts::promise([a = arr, axis]() -> tf::ts::wasm_ndarray<T> {
    return tf::ts::min(a, axis);
  });
}

template <typename T>
static auto async_max(tf::ts::wasm_ndarray<T> &arr, int axis)
    -> tf::ts::promise_t {
  if (axis == -1)
    return tf::ts::promise([a = arr]() -> T { return tf::ts::max(a); });
  return tf::ts::promise([a = arr, axis]() -> tf::ts::wasm_ndarray<T> {
    return tf::ts::max(a, axis);
  });
}

template <typename T>
static auto async_mean(tf::ts::wasm_ndarray<T> &arr, int axis)
    -> tf::ts::promise_t {
  if (axis == -1)
    return tf::ts::promise([a = arr]() -> double { return tf::ts::mean(a); });
  return tf::ts::promise([a = arr, axis]() -> tf::ts::wasm_ndarray<float> {
    return tf::ts::mean(a, axis);
  });
}

// ============================================================================
// Sync — norm
// ============================================================================

template <typename T>
static auto sync_norm(tf::ts::wasm_ndarray<T> &arr, int axis)
    -> emscripten::val {
  if (axis == -1)
    return emscripten::val(tf::ts::norm(arr));
  return emscripten::val(tf::ts::norm(arr, axis));
}

// ============================================================================
// Async — norm
// ============================================================================

template <typename T>
static auto async_norm(tf::ts::wasm_ndarray<T> &arr, int axis)
    -> tf::ts::promise_t {
  if (axis == -1)
    return tf::ts::promise([a = arr]() -> float { return tf::ts::norm(a); });
  return tf::ts::promise([a = arr, axis]() -> tf::ts::wasm_ndarray<float> {
    return tf::ts::norm(a, axis);
  });
}

// ============================================================================
// Sync — argmin / argmax
// ============================================================================

template <typename T>
static auto sync_argmin(tf::ts::wasm_ndarray<T> &arr, int axis)
    -> emscripten::val {
  if (axis == -1)
    return emscripten::val(tf::ts::argmin(arr));
  return emscripten::val(tf::ts::argmin(arr, axis));
}

template <typename T>
static auto sync_argmax(tf::ts::wasm_ndarray<T> &arr, int axis)
    -> emscripten::val {
  if (axis == -1)
    return emscripten::val(tf::ts::argmax(arr));
  return emscripten::val(tf::ts::argmax(arr, axis));
}

// ============================================================================
// Sync — any / all (int8 only)
// ============================================================================

static auto sync_any(tf::ts::wasm_ndarray<std::int8_t> &arr, int axis)
    -> emscripten::val {
  if (axis == -1)
    return emscripten::val(tf::ts::any(arr));
  return emscripten::val(tf::ts::any(arr, axis));
}

static auto sync_all(tf::ts::wasm_ndarray<std::int8_t> &arr, int axis)
    -> emscripten::val {
  if (axis == -1)
    return emscripten::val(tf::ts::all(arr));
  return emscripten::val(tf::ts::all(arr, axis));
}

// ============================================================================
// Async — int8 specializations
// ============================================================================

static auto async_sum_int8(tf::ts::wasm_ndarray<std::int8_t> &arr,
                           int axis) -> tf::ts::promise_t {
  if (axis == -1) {
    return tf::ts::promise([a = arr]() -> int {
      auto *data = a.raw_data();
      auto len = a.length();
      int acc = 0;
      for (std::size_t i = 0; i < len; ++i)
        acc += static_cast<int>(data[i]);
      return acc;
    });
  }
  return tf::ts::promise(
      [a = arr, axis]() -> tf::ts::wasm_ndarray<int> {
        auto [outer, reduce, inner] =
            tf::ts::detail::axis_strides(a.raw_shape(), axis);
        auto rshape = tf::ts::detail::remove_axis(a.raw_shape(), axis);
        auto result_len = static_cast<std::size_t>(outer) * inner;
        auto *data = a.raw_data();

        tf::buffer<int> buf;
        buf.allocate(result_len);
        auto *out = buf.data();

        for (std::size_t idx = 0; idx < result_len; ++idx) {
          int o = static_cast<int>(idx) / inner;
          int i = static_cast<int>(idx) % inner;
          int acc = 0;
          for (int r = 0; r < reduce; ++r)
            acc += static_cast<int>(data[o * reduce * inner + r * inner + i]);
          out[idx] = acc;
        }

        return tf::ts::wasm_ndarray<int>::from_buffer(std::move(buf),
                                                      std::move(rshape));
      });
}

static auto async_min_int8(tf::ts::wasm_ndarray<std::int8_t> &arr,
                           int axis) -> tf::ts::promise_t {
  if (axis == -1)
    return tf::ts::promise(
        [a = arr]() -> std::int8_t { return tf::ts::min(a); });
  return tf::ts::promise(
      [a = arr, axis]() -> tf::ts::wasm_ndarray<std::int8_t> {
        return tf::ts::min(a, axis);
      });
}

static auto async_max_int8(tf::ts::wasm_ndarray<std::int8_t> &arr,
                           int axis) -> tf::ts::promise_t {
  if (axis == -1)
    return tf::ts::promise(
        [a = arr]() -> std::int8_t { return tf::ts::max(a); });
  return tf::ts::promise(
      [a = arr, axis]() -> tf::ts::wasm_ndarray<std::int8_t> {
        return tf::ts::max(a, axis);
      });
}

static auto async_mean_int8(tf::ts::wasm_ndarray<std::int8_t> &arr,
                            int axis) -> tf::ts::promise_t {
  if (axis == -1)
    return tf::ts::promise(
        [a = arr]() -> double { return tf::ts::mean(a); });
  return tf::ts::promise(
      [a = arr, axis]() -> tf::ts::wasm_ndarray<float> {
        return tf::ts::mean(a, axis);
      });
}

// ============================================================================
// Async — argmin / argmax
// ============================================================================

template <typename T>
static auto async_argmin(tf::ts::wasm_ndarray<T> &arr, int axis)
    -> tf::ts::promise_t {
  if (axis == -1)
    return tf::ts::promise(
        [a = arr]() -> int { return tf::ts::argmin(a); });
  return tf::ts::promise(
      [a = arr, axis]() -> tf::ts::wasm_ndarray<int> {
        return tf::ts::argmin(a, axis);
      });
}

template <typename T>
static auto async_argmax(tf::ts::wasm_ndarray<T> &arr, int axis)
    -> tf::ts::promise_t {
  if (axis == -1)
    return tf::ts::promise(
        [a = arr]() -> int { return tf::ts::argmax(a); });
  return tf::ts::promise(
      [a = arr, axis]() -> tf::ts::wasm_ndarray<int> {
        return tf::ts::argmax(a, axis);
      });
}

// ============================================================================
// Async — any / all (int8 only)
// ============================================================================

static auto async_any(tf::ts::wasm_ndarray<std::int8_t> &arr, int axis)
    -> tf::ts::promise_t {
  if (axis == -1)
    return tf::ts::promise(
        [a = arr]() -> int { return tf::ts::any(a); });
  return tf::ts::promise(
      [a = arr, axis]() -> tf::ts::wasm_ndarray<std::int8_t> {
        return tf::ts::any(a, axis);
      });
}

static auto async_all(tf::ts::wasm_ndarray<std::int8_t> &arr, int axis)
    -> tf::ts::promise_t {
  if (axis == -1)
    return tf::ts::promise(
        [a = arr]() -> int { return tf::ts::all(a); });
  return tf::ts::promise(
      [a = arr, axis]() -> tf::ts::wasm_ndarray<std::int8_t> {
        return tf::ts::all(a, axis);
      });
}

// ============================================================================
// Embind
// ============================================================================

EMSCRIPTEN_BINDINGS(trueform_reductions) {
  // Sync
  emscripten::function("sum_float32", &sync_sum<float>);
  emscripten::function("sum_int32", &sync_sum<int>);
  emscripten::function("min_float32", &sync_min<float>);
  emscripten::function("min_int32", &sync_min<int>);
  emscripten::function("max_float32", &sync_max<float>);
  emscripten::function("max_int32", &sync_max<int>);
  emscripten::function("mean_float32", &sync_mean<float>);
  emscripten::function("mean_int32", &sync_mean<int>);

  // Sync — norm
  emscripten::function("norm_float32", &sync_norm<float>);
  emscripten::function("norm_int32", &sync_norm<int>);

  // Sync — int8
  emscripten::function("sum_int8", &sync_sum_int8);
  emscripten::function("min_int8", &sync_min_int8);
  emscripten::function("max_int8", &sync_max_int8);
  emscripten::function("mean_int8", &sync_mean_int8);
  emscripten::function("norm_int8", &sync_norm<std::int8_t>);

  // Async dispatch
  emscripten::function("dispatch_sum_float32", &async_sum<float>);
  emscripten::function("dispatch_sum_int32", &async_sum<int>);
  emscripten::function("dispatch_min_float32", &async_min<float>);
  emscripten::function("dispatch_min_int32", &async_min<int>);
  emscripten::function("dispatch_max_float32", &async_max<float>);
  emscripten::function("dispatch_max_int32", &async_max<int>);
  emscripten::function("dispatch_mean_float32", &async_mean<float>);
  emscripten::function("dispatch_mean_int32", &async_mean<int>);

  // Async dispatch — norm
  emscripten::function("dispatch_norm_float32", &async_norm<float>);
  emscripten::function("dispatch_norm_int32", &async_norm<int>);

  // Async dispatch — int8
  emscripten::function("dispatch_sum_int8", &async_sum_int8);
  emscripten::function("dispatch_min_int8", &async_min_int8);
  emscripten::function("dispatch_max_int8", &async_max_int8);
  emscripten::function("dispatch_mean_int8", &async_mean_int8);
  emscripten::function("dispatch_norm_int8", &async_norm<std::int8_t>);

  // Sync — argmin / argmax
  emscripten::function("argmin_float32", &sync_argmin<float>);
  emscripten::function("argmin_int32", &sync_argmin<int>);
  emscripten::function("argmin_int8", &sync_argmin<std::int8_t>);
  emscripten::function("argmax_float32", &sync_argmax<float>);
  emscripten::function("argmax_int32", &sync_argmax<int>);
  emscripten::function("argmax_int8", &sync_argmax<std::int8_t>);

  // Sync — any / all (int8 only)
  emscripten::function("any_int8", &sync_any);
  emscripten::function("all_int8", &sync_all);

  // Async dispatch — argmin / argmax
  emscripten::function("dispatch_argmin_float32", &async_argmin<float>);
  emscripten::function("dispatch_argmin_int32", &async_argmin<int>);
  emscripten::function("dispatch_argmin_int8", &async_argmin<std::int8_t>);
  emscripten::function("dispatch_argmax_float32", &async_argmax<float>);
  emscripten::function("dispatch_argmax_int32", &async_argmax<int>);
  emscripten::function("dispatch_argmax_int8", &async_argmax<std::int8_t>);

  // Async dispatch — any / all (int8 only)
  emscripten::function("dispatch_any_int8", &async_any);
  emscripten::function("dispatch_all_int8", &async_all);
}
