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

#include "trueform/core/algorithm/parallel_for_each.hpp"
#include "trueform/core/algorithm/parallel_transform.hpp"
#include "trueform/core/algorithm/reduce.hpp"
#include "trueform/core/range.hpp"
#include "trueform/core/views/blocked_range.hpp"
#include "trueform/core/views/sequence_range.hpp"
#include "trueform/ts/core/parallel_config.hpp"
#include "trueform/ts/core/wasm_ndarray.hpp"
#include <cmath>
#include <limits>

namespace tf {
namespace ts {

namespace detail {

inline auto axis_strides(const tf::small_vector<int, 3> &shape, int axis)
    -> std::tuple<int, int, int> {
  int outer = 1;
  for (int i = 0; i < axis; ++i)
    outer *= shape[i];
  int reduce = shape[axis];
  int inner = 1;
  for (int i = axis + 1; i < static_cast<int>(shape.size()); ++i)
    inner *= shape[i];
  return {outer, reduce, inner};
}

inline auto remove_axis(const tf::small_vector<int, 3> &shape, int axis)
    -> tf::small_vector<int, 3> {
  tf::small_vector<int, 3> out;
  for (int i = 0; i < static_cast<int>(shape.size()); ++i) {
    if (i != axis)
      out.push_back(shape[i]);
  }
  if (out.empty())
    out.push_back(1);
  return out;
}

template <typename T, typename BinaryOp>
auto axis_reduce_impl(const wasm_ndarray<T> &arr, int axis, BinaryOp op,
                      T initial) -> wasm_ndarray<T> {
  auto [outer, reduce, inner] = axis_strides(arr.raw_shape(), axis);
  auto rshape = remove_axis(arr.raw_shape(), axis);
  auto result_len = static_cast<std::size_t>(outer) * inner;
  auto *data = arr.raw_data();

  tf::buffer<T> buf;
  buf.allocate(result_len);
  auto *out = buf.data();

  if (arr.length() < parallel_threshold) {
    // Sequential path
    for (std::size_t idx = 0; idx < result_len; ++idx) {
      int o = static_cast<int>(idx) / inner;
      int i = static_cast<int>(idx) % inner;
      T acc = initial;
      for (int r = 0; r < reduce; ++r)
        acc = op(acc, data[o * reduce * inner + r * inner + i]);
      out[idx] = acc;
    }
  } else if (inner == 1) {
    auto flat =
        tf::make_range(data, static_cast<std::size_t>(outer) * reduce);
    auto rows =
        tf::make_blocked_range(flat, static_cast<std::size_t>(reduce));
    auto output = tf::make_range(out, static_cast<std::size_t>(outer));
    tf::parallel_transform(rows, output, [op, initial](auto row) {
      T acc = initial;
      for (auto e : row)
        acc = op(acc, e);
      return acc;
    }, tf::checked);
  } else {
    auto seq = tf::make_sequence_range(static_cast<int>(result_len));
    tf::parallel_for_each(seq, [inner = inner, reduce = reduce, op,
                                initial, data, out](int idx) {
      int o = idx / inner;
      int i = idx % inner;
      T acc = initial;
      for (int r = 0; r < reduce; ++r)
        acc = op(acc, data[o * reduce * inner + r * inner + i]);
      out[idx] = acc;
    }, tf::checked);
  }

  return wasm_ndarray<T>::from_buffer(std::move(buf), std::move(rshape));
}

} // namespace detail

// ============================================================================
// sum
// ============================================================================

template <typename T> auto sum(const wasm_ndarray<T> &arr) -> T {
  auto r = tf::make_range(arr.raw_data(), arr.length());
  if (arr.length() < parallel_threshold) {
    T acc{0};
    for (auto e : r) acc += e;
    return acc;
  }
  return tf::reduce(r, std::plus<T>{}, T{0}, tf::checked);
}

template <typename T>
auto sum(const wasm_ndarray<T> &arr, int axis) -> wasm_ndarray<T> {
  return detail::axis_reduce_impl(arr, axis, std::plus<T>{}, T{0});
}

// ============================================================================
// min
// ============================================================================

template <typename T> auto min(const wasm_ndarray<T> &arr) -> T {
  auto r = tf::make_range(arr.raw_data(), arr.length());
  if (arr.length() < parallel_threshold) {
    T acc = std::numeric_limits<T>::max();
    for (auto e : r) acc = acc < e ? acc : e;
    return acc;
  }
  return tf::reduce(
      r, [](T a, T b) { return a < b ? a : b; },
      std::numeric_limits<T>::max(), tf::checked);
}

template <typename T>
auto min(const wasm_ndarray<T> &arr, int axis) -> wasm_ndarray<T> {
  return detail::axis_reduce_impl(
      arr, axis, [](T a, T b) { return a < b ? a : b; },
      std::numeric_limits<T>::max());
}

// ============================================================================
// max
// ============================================================================

template <typename T> auto max(const wasm_ndarray<T> &arr) -> T {
  auto r = tf::make_range(arr.raw_data(), arr.length());
  if (arr.length() < parallel_threshold) {
    T acc = std::numeric_limits<T>::lowest();
    for (auto e : r) acc = acc > e ? acc : e;
    return acc;
  }
  return tf::reduce(
      r, [](T a, T b) { return a > b ? a : b; },
      std::numeric_limits<T>::lowest(), tf::checked);
}

template <typename T>
auto max(const wasm_ndarray<T> &arr, int axis) -> wasm_ndarray<T> {
  return detail::axis_reduce_impl(
      arr, axis, [](T a, T b) { return a > b ? a : b; },
      std::numeric_limits<T>::lowest());
}

// ============================================================================
// mean
// ============================================================================

template <typename T> auto mean(const wasm_ndarray<T> &arr) -> double {
  auto r = tf::make_range(arr.raw_data(), arr.length());
  double s;
  if (arr.length() < parallel_threshold) {
    s = 0.0;
    for (auto e : r) s += static_cast<double>(e);
  } else {
    s = tf::reduce(
        r, [](double a, T b) { return a + static_cast<double>(b); }, 0.0,
        tf::checked);
  }
  return s / static_cast<double>(arr.length());
}

template <typename T>
auto mean(const wasm_ndarray<T> &arr, int axis) -> wasm_ndarray<float> {
  auto [outer, reduce, inner] =
      detail::axis_strides(arr.raw_shape(), axis);
  auto rshape = detail::remove_axis(arr.raw_shape(), axis);
  auto result_len = static_cast<std::size_t>(outer) * inner;
  auto *data = arr.raw_data();
  auto inv = 1.0 / static_cast<double>(reduce);

  tf::buffer<float> buf;
  buf.allocate(result_len);
  auto *out = buf.data();

  if (arr.length() < parallel_threshold) {
    for (std::size_t idx = 0; idx < result_len; ++idx) {
      int o = static_cast<int>(idx) / inner;
      int i = static_cast<int>(idx) % inner;
      double acc = 0.0;
      for (int r = 0; r < reduce; ++r)
        acc += static_cast<double>(data[o * reduce * inner + r * inner + i]);
      out[idx] = static_cast<float>(acc * inv);
    }
  } else if (inner == 1) {
    auto flat =
        tf::make_range(data, static_cast<std::size_t>(outer) * reduce);
    auto rows =
        tf::make_blocked_range(flat, static_cast<std::size_t>(reduce));
    auto output = tf::make_range(out, static_cast<std::size_t>(outer));
    tf::parallel_transform(rows, output, [inv](auto row) {
      double acc = 0.0;
      for (auto e : row)
        acc += static_cast<double>(e);
      return static_cast<float>(acc * inv);
    }, tf::checked);
  } else {
    auto seq = tf::make_sequence_range(static_cast<int>(result_len));
    tf::parallel_for_each(seq, [inner = inner, reduce = reduce, inv, data,
                                out](int idx) {
      int o = idx / inner;
      int i = idx % inner;
      double acc = 0.0;
      for (int r = 0; r < reduce; ++r)
        acc += static_cast<double>(data[o * reduce * inner + r * inner + i]);
      out[idx] = static_cast<float>(acc * inv);
    }, tf::checked);
  }

  return wasm_ndarray<float>::from_buffer(std::move(buf), std::move(rshape));
}

// ============================================================================
// norm (L2)
// ============================================================================

template <typename T> auto norm(const wasm_ndarray<T> &arr) -> float {
  auto *data = arr.raw_data();
  auto len = arr.length();
  double s;
  if (len < parallel_threshold) {
    s = 0.0;
    for (std::size_t i = 0; i < len; ++i)
      s += static_cast<double>(data[i]) * static_cast<double>(data[i]);
  } else {
    auto r = tf::make_range(data, len);
    s = tf::reduce(
        r, [](double a, T b) { return a + static_cast<double>(b) * static_cast<double>(b); },
        0.0, tf::checked);
  }
  return static_cast<float>(std::sqrt(s));
}

template <typename T>
auto norm(const wasm_ndarray<T> &arr, int axis) -> wasm_ndarray<float> {
  auto [outer, reduce, inner] =
      detail::axis_strides(arr.raw_shape(), axis);
  auto rshape = detail::remove_axis(arr.raw_shape(), axis);
  auto result_len = static_cast<std::size_t>(outer) * inner;
  auto *data = arr.raw_data();

  tf::buffer<float> buf;
  buf.allocate(result_len);
  auto *out = buf.data();

  if (arr.length() < parallel_threshold) {
    for (std::size_t idx = 0; idx < result_len; ++idx) {
      int o = static_cast<int>(idx) / inner;
      int i = static_cast<int>(idx) % inner;
      double acc = 0.0;
      for (int r = 0; r < reduce; ++r) {
        double v = static_cast<double>(data[o * reduce * inner + r * inner + i]);
        acc += v * v;
      }
      out[idx] = static_cast<float>(std::sqrt(acc));
    }
  } else if (inner == 1) {
    auto flat =
        tf::make_range(data, static_cast<std::size_t>(outer) * reduce);
    auto rows =
        tf::make_blocked_range(flat, static_cast<std::size_t>(reduce));
    auto output = tf::make_range(out, static_cast<std::size_t>(outer));
    tf::parallel_transform(rows, output, [](auto row) {
      double acc = 0.0;
      for (auto e : row)
        acc += static_cast<double>(e) * static_cast<double>(e);
      return static_cast<float>(std::sqrt(acc));
    }, tf::checked);
  } else {
    auto seq = tf::make_sequence_range(static_cast<int>(result_len));
    tf::parallel_for_each(seq, [inner = inner, reduce = reduce, data,
                                out](int idx) {
      int o = idx / inner;
      int i = idx % inner;
      double acc = 0.0;
      for (int r = 0; r < reduce; ++r) {
        double v = static_cast<double>(data[o * reduce * inner + r * inner + i]);
        acc += v * v;
      }
      out[idx] = static_cast<float>(std::sqrt(acc));
    }, tf::checked);
  }

  return wasm_ndarray<float>::from_buffer(std::move(buf), std::move(rshape));
}

// ============================================================================
// argmin
// ============================================================================

template <typename T> auto argmin(const wasm_ndarray<T> &arr) -> int {
  auto *data = arr.raw_data();
  auto len = arr.length();
  int best = 0;
  T best_val = data[0];
  for (std::size_t i = 1; i < len; ++i) {
    if (data[i] < best_val) {
      best_val = data[i];
      best = static_cast<int>(i);
    }
  }
  return best;
}

template <typename T>
auto argmin(const wasm_ndarray<T> &arr, int axis) -> wasm_ndarray<int> {
  auto [outer, reduce, inner] = detail::axis_strides(arr.raw_shape(), axis);
  auto rshape = detail::remove_axis(arr.raw_shape(), axis);
  auto result_len = static_cast<std::size_t>(outer) * inner;
  auto *data = arr.raw_data();

  tf::buffer<int> buf;
  buf.allocate(result_len);
  auto *out = buf.data();

  for (std::size_t idx = 0; idx < result_len; ++idx) {
    int o = static_cast<int>(idx) / inner;
    int i = static_cast<int>(idx) % inner;
    int best = 0;
    T best_val = data[o * reduce * inner + i];
    for (int r = 1; r < reduce; ++r) {
      T val = data[o * reduce * inner + r * inner + i];
      if (val < best_val) {
        best_val = val;
        best = r;
      }
    }
    out[idx] = best;
  }

  return wasm_ndarray<int>::from_buffer(std::move(buf), std::move(rshape));
}

// ============================================================================
// argmax
// ============================================================================

template <typename T> auto argmax(const wasm_ndarray<T> &arr) -> int {
  auto *data = arr.raw_data();
  auto len = arr.length();
  int best = 0;
  T best_val = data[0];
  for (std::size_t i = 1; i < len; ++i) {
    if (data[i] > best_val) {
      best_val = data[i];
      best = static_cast<int>(i);
    }
  }
  return best;
}

template <typename T>
auto argmax(const wasm_ndarray<T> &arr, int axis) -> wasm_ndarray<int> {
  auto [outer, reduce, inner] = detail::axis_strides(arr.raw_shape(), axis);
  auto rshape = detail::remove_axis(arr.raw_shape(), axis);
  auto result_len = static_cast<std::size_t>(outer) * inner;
  auto *data = arr.raw_data();

  tf::buffer<int> buf;
  buf.allocate(result_len);
  auto *out = buf.data();

  for (std::size_t idx = 0; idx < result_len; ++idx) {
    int o = static_cast<int>(idx) / inner;
    int i = static_cast<int>(idx) % inner;
    int best = 0;
    T best_val = data[o * reduce * inner + i];
    for (int r = 1; r < reduce; ++r) {
      T val = data[o * reduce * inner + r * inner + i];
      if (val > best_val) {
        best_val = val;
        best = r;
      }
    }
    out[idx] = best;
  }

  return wasm_ndarray<int>::from_buffer(std::move(buf), std::move(rshape));
}

// ============================================================================
// any (int8/bool only)
// ============================================================================

inline auto any(const wasm_ndarray<std::int8_t> &arr) -> int {
  auto *data = arr.raw_data();
  for (std::size_t i = 0; i < arr.length(); ++i)
    if (data[i])
      return 1;
  return 0;
}

inline auto any(const wasm_ndarray<std::int8_t> &arr, int axis)
    -> wasm_ndarray<std::int8_t> {
  auto [outer, reduce, inner] = detail::axis_strides(arr.raw_shape(), axis);
  auto rshape = detail::remove_axis(arr.raw_shape(), axis);
  auto result_len = static_cast<std::size_t>(outer) * inner;
  auto *data = arr.raw_data();

  tf::buffer<std::int8_t> buf;
  buf.allocate(result_len);
  auto *out = buf.data();

  for (std::size_t idx = 0; idx < result_len; ++idx) {
    int o = static_cast<int>(idx) / inner;
    int i = static_cast<int>(idx) % inner;
    std::int8_t found = 0;
    for (int r = 0; r < reduce; ++r) {
      if (data[o * reduce * inner + r * inner + i]) {
        found = 1;
        break;
      }
    }
    out[idx] = found;
  }

  return wasm_ndarray<std::int8_t>::from_buffer(std::move(buf),
                                                 std::move(rshape));
}

// ============================================================================
// all (int8/bool only)
// ============================================================================

inline auto all(const wasm_ndarray<std::int8_t> &arr) -> int {
  auto *data = arr.raw_data();
  for (std::size_t i = 0; i < arr.length(); ++i)
    if (!data[i])
      return 0;
  return 1;
}

inline auto all(const wasm_ndarray<std::int8_t> &arr, int axis)
    -> wasm_ndarray<std::int8_t> {
  auto [outer, reduce, inner] = detail::axis_strides(arr.raw_shape(), axis);
  auto rshape = detail::remove_axis(arr.raw_shape(), axis);
  auto result_len = static_cast<std::size_t>(outer) * inner;
  auto *data = arr.raw_data();

  tf::buffer<std::int8_t> buf;
  buf.allocate(result_len);
  auto *out = buf.data();

  for (std::size_t idx = 0; idx < result_len; ++idx) {
    int o = static_cast<int>(idx) / inner;
    int i = static_cast<int>(idx) % inner;
    std::int8_t ok = 1;
    for (int r = 0; r < reduce; ++r) {
      if (!data[o * reduce * inner + r * inner + i]) {
        ok = 0;
        break;
      }
    }
    out[idx] = ok;
  }

  return wasm_ndarray<std::int8_t>::from_buffer(std::move(buf),
                                                 std::move(rshape));
}

} // namespace ts
} // namespace tf
