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
#include "trueform/core/range.hpp"
#include "trueform/core/sqrt.hpp"
#include "trueform/core/views/sequence_range.hpp"
#include "trueform/ts/core/parallel_config.hpp"
#include "trueform/ts/core/wasm_ndarray.hpp"
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <stdexcept>

namespace tf {
namespace ts {

namespace detail {

// ============================================================================
// Broadcasting utilities
// ============================================================================

inline auto broadcast_shape(const tf::small_vector<int, 3> &a,
                            const tf::small_vector<int, 3> &b)
    -> tf::small_vector<int, 3> {
  int a_ndim = static_cast<int>(a.size());
  int b_ndim = static_cast<int>(b.size());
  int ndim = a_ndim > b_ndim ? a_ndim : b_ndim;
  tf::small_vector<int, 3> out;
  for (int i = 0; i < ndim; ++i) {
    int ai = i < ndim - a_ndim ? 1 : a[i - (ndim - a_ndim)];
    int bi = i < ndim - b_ndim ? 1 : b[i - (ndim - b_ndim)];
    if (ai == bi)
      out.push_back(ai);
    else if (ai == 1)
      out.push_back(bi);
    else if (bi == 1)
      out.push_back(ai);
    else
      throw std::runtime_error("shapes not broadcastable");
  }
  return out;
}

inline auto compute_strides(const tf::small_vector<int, 3> &shape)
    -> tf::small_vector<int, 3> {
  int ndim = static_cast<int>(shape.size());
  tf::small_vector<int, 3> strides;
  strides.resize(ndim);
  int stride = 1;
  for (int i = ndim - 1; i >= 0; --i) {
    strides[i] = stride;
    stride *= shape[i];
  }
  return strides;
}

inline auto broadcast_strides(const tf::small_vector<int, 3> &src_shape,
                              const tf::small_vector<int, 3> &out_shape)
    -> tf::small_vector<int, 3> {
  int ndim = static_cast<int>(out_shape.size());
  int src_ndim = static_cast<int>(src_shape.size());
  auto src_strides = compute_strides(src_shape);
  tf::small_vector<int, 3> result;
  result.resize(ndim);
  for (int i = 0; i < ndim; ++i) {
    int si = i - (ndim - src_ndim);
    if (si < 0 || src_shape[si] == 1)
      result[i] = 0;
    else
      result[i] = src_strides[si];
  }
  return result;
}

inline auto broadcast_index(int flat_idx,
                            const tf::small_vector<int, 3> &out_strides,
                            const tf::small_vector<int, 3> &bcast_strides,
                            int ndim) -> int {
  int src_idx = 0;
  for (int d = 0; d < ndim; ++d) {
    int coord = flat_idx / out_strides[d];
    flat_idx %= out_strides[d];
    src_idx += coord * bcast_strides[d];
  }
  return src_idx;
}

inline auto shapes_equal(const tf::small_vector<int, 3> &a,
                         const tf::small_vector<int, 3> &b) -> bool {
  if (a.size() != b.size())
    return false;
  for (int i = 0; i < static_cast<int>(a.size()); ++i)
    if (a[i] != b[i])
      return false;
  return true;
}

inline auto total_size(const tf::small_vector<int, 3> &shape) -> int {
  int n = 1;
  for (int i = 0; i < static_cast<int>(shape.size()); ++i)
    n *= shape[i];
  return n;
}

} // namespace detail

// ============================================================================
// Generic element-wise: copy
// ============================================================================

template <typename T, typename BinaryOp>
auto elementwise(const wasm_ndarray<T> &a, const wasm_ndarray<T> &b,
                 BinaryOp op) -> wasm_ndarray<T> {
  auto out_shape = detail::broadcast_shape(a.raw_shape(), b.raw_shape());
  int total = detail::total_size(out_shape);

  tf::buffer<T> buf;
  buf.allocate(static_cast<std::size_t>(total));
  auto *out = buf.data();
  auto *ad = a.raw_data();
  auto *bd = b.raw_data();

  if (detail::shapes_equal(a.raw_shape(), b.raw_shape())) {
    // Fast path: same shape, no broadcast
    if (static_cast<std::size_t>(total) < parallel_threshold) {
      for (int i = 0; i < total; ++i)
        out[i] = op(ad[i], bd[i]);
    } else {
      auto seq = tf::make_sequence_range(total);
      tf::parallel_for_each(
          seq, [ad, bd, out, op](int i) { out[i] = op(ad[i], bd[i]); },
          tf::checked);
    }
  } else {
    // General broadcast path
    auto out_strides = detail::compute_strides(out_shape);
    auto a_bstrides =
        detail::broadcast_strides(a.raw_shape(), out_shape);
    auto b_bstrides =
        detail::broadcast_strides(b.raw_shape(), out_shape);
    int ndim = static_cast<int>(out_shape.size());

    if (static_cast<std::size_t>(total) < parallel_threshold) {
      for (int i = 0; i < total; ++i) {
        int ai =
            detail::broadcast_index(i, out_strides, a_bstrides, ndim);
        int bi =
            detail::broadcast_index(i, out_strides, b_bstrides, ndim);
        out[i] = op(ad[ai], bd[bi]);
      }
    } else {
      auto seq = tf::make_sequence_range(total);
      tf::parallel_for_each(
          seq,
          [ad, bd, out, op, out_strides, a_bstrides, b_bstrides,
           ndim](int i) {
            int ai = detail::broadcast_index(i, out_strides, a_bstrides,
                                             ndim);
            int bi = detail::broadcast_index(i, out_strides, b_bstrides,
                                             ndim);
            out[i] = op(ad[ai], bd[bi]);
          },
          tf::checked);
    }
  }

  return wasm_ndarray<T>::from_buffer(std::move(buf), std::move(out_shape));
}

// ============================================================================
// Generic element-wise: in-place on a
// ============================================================================

template <typename T, typename BinaryOp>
auto elementwise_inplace(wasm_ndarray<T> &a, const wasm_ndarray<T> &b,
                         BinaryOp op) -> void {
  auto out_shape = detail::broadcast_shape(a.raw_shape(), b.raw_shape());
  if (!detail::shapes_equal(a.raw_shape(), out_shape))
    throw std::runtime_error(
        "in-place op requires lhs shape == output shape");

  int total = static_cast<int>(a.length());
  auto *ad = a.raw_data();
  auto *bd = b.raw_data();

  if (detail::shapes_equal(a.raw_shape(), b.raw_shape())) {
    if (static_cast<std::size_t>(total) < parallel_threshold) {
      for (int i = 0; i < total; ++i)
        ad[i] = op(ad[i], bd[i]);
    } else {
      auto seq = tf::make_sequence_range(total);
      tf::parallel_for_each(
          seq, [ad, bd, op](int i) { ad[i] = op(ad[i], bd[i]); },
          tf::checked);
    }
  } else {
    auto out_strides = detail::compute_strides(out_shape);
    auto b_bstrides =
        detail::broadcast_strides(b.raw_shape(), out_shape);
    int ndim = static_cast<int>(out_shape.size());

    if (static_cast<std::size_t>(total) < parallel_threshold) {
      for (int i = 0; i < total; ++i) {
        int bi =
            detail::broadcast_index(i, out_strides, b_bstrides, ndim);
        ad[i] = op(ad[i], bd[bi]);
      }
    } else {
      auto seq = tf::make_sequence_range(total);
      tf::parallel_for_each(
          seq,
          [ad, bd, op, out_strides, b_bstrides, ndim](int i) {
            int bi = detail::broadcast_index(i, out_strides, b_bstrides,
                                             ndim);
            ad[i] = op(ad[i], bd[bi]);
          },
          tf::checked);
    }
  }
}

// ============================================================================
// Concrete binary ops
// ============================================================================

template <typename T>
auto add(const wasm_ndarray<T> &a, const wasm_ndarray<T> &b)
    -> wasm_ndarray<T> {
  return elementwise(a, b, std::plus<T>{});
}

template <typename T>
auto add_inplace(wasm_ndarray<T> &a, const wasm_ndarray<T> &b) -> void {
  elementwise_inplace(a, b, std::plus<T>{});
}

template <typename T>
auto sub(const wasm_ndarray<T> &a, const wasm_ndarray<T> &b)
    -> wasm_ndarray<T> {
  return elementwise(a, b, std::minus<T>{});
}

template <typename T>
auto sub_inplace(wasm_ndarray<T> &a, const wasm_ndarray<T> &b) -> void {
  elementwise_inplace(a, b, std::minus<T>{});
}

template <typename T>
auto mul(const wasm_ndarray<T> &a, const wasm_ndarray<T> &b)
    -> wasm_ndarray<T> {
  return elementwise(a, b, std::multiplies<T>{});
}

template <typename T>
auto mul_inplace(wasm_ndarray<T> &a, const wasm_ndarray<T> &b) -> void {
  elementwise_inplace(a, b, std::multiplies<T>{});
}

// ============================================================================
// Scalar ops
// ============================================================================

namespace detail {

template <typename T, typename BinaryOp>
auto scalar_op(const wasm_ndarray<T> &a, T scalar, BinaryOp op)
    -> wasm_ndarray<T> {
  auto total = a.length();
  tf::buffer<T> buf;
  buf.allocate(total);
  auto *out = buf.data();
  auto *ad = a.raw_data();

  if (total < parallel_threshold) {
    for (std::size_t i = 0; i < total; ++i)
      out[i] = op(ad[i], scalar);
  } else {
    auto in_range = tf::make_range(ad, total);
    auto out_range = tf::make_range(out, total);
    tf::parallel_transform(
        in_range, out_range,
        [scalar, op](T v) { return op(v, scalar); }, tf::checked);
  }

  return wasm_ndarray<T>::from_buffer(std::move(buf),
                                     tf::small_vector<int, 3>(a.raw_shape()));
}

template <typename T, typename BinaryOp>
auto scalar_op_inplace(wasm_ndarray<T> &a, T scalar, BinaryOp op) -> void {
  auto total = a.length();
  auto *ad = a.raw_data();

  if (total < parallel_threshold) {
    for (std::size_t i = 0; i < total; ++i)
      ad[i] = op(ad[i], scalar);
  } else {
    auto seq = tf::make_sequence_range(static_cast<int>(total));
    tf::parallel_for_each(
        seq, [ad, scalar, op](int i) { ad[i] = op(ad[i], scalar); },
        tf::checked);
  }
}

} // namespace detail

template <typename T>
auto add_scalar(const wasm_ndarray<T> &a, T scalar) -> wasm_ndarray<T> {
  return detail::scalar_op(a, scalar, std::plus<T>{});
}

template <typename T>
auto add_scalar_inplace(wasm_ndarray<T> &a, T scalar) -> void {
  detail::scalar_op_inplace(a, scalar, std::plus<T>{});
}

template <typename T>
auto sub_scalar(const wasm_ndarray<T> &a, T scalar) -> wasm_ndarray<T> {
  return detail::scalar_op(a, scalar, std::minus<T>{});
}

template <typename T>
auto sub_scalar_inplace(wasm_ndarray<T> &a, T scalar) -> void {
  detail::scalar_op_inplace(a, scalar, std::minus<T>{});
}

template <typename T>
auto mul_scalar(const wasm_ndarray<T> &a, T scalar) -> wasm_ndarray<T> {
  return detail::scalar_op(a, scalar, std::multiplies<T>{});
}

template <typename T>
auto mul_scalar_inplace(wasm_ndarray<T> &a, T scalar) -> void {
  detail::scalar_op_inplace(a, scalar, std::multiplies<T>{});
}

// ============================================================================
// div
// ============================================================================

template <typename T>
auto div(const wasm_ndarray<T> &a, const wasm_ndarray<T> &b)
    -> wasm_ndarray<T> {
  return elementwise(a, b, std::divides<T>{});
}

template <typename T>
auto div_inplace(wasm_ndarray<T> &a, const wasm_ndarray<T> &b) -> void {
  elementwise_inplace(a, b, std::divides<T>{});
}

template <typename T>
auto div_scalar(const wasm_ndarray<T> &a, T scalar) -> wasm_ndarray<T> {
  return detail::scalar_op(a, scalar, std::divides<T>{});
}

template <typename T>
auto div_scalar_inplace(wasm_ndarray<T> &a, T scalar) -> void {
  detail::scalar_op_inplace(a, scalar, std::divides<T>{});
}

// ============================================================================
// Generic unary op
// ============================================================================

namespace detail {

template <typename T, typename UnaryOp>
auto unary_op(const wasm_ndarray<T> &a, UnaryOp op) -> wasm_ndarray<T> {
  auto total = a.length();
  tf::buffer<T> buf;
  buf.allocate(total);
  auto *out = buf.data();
  auto *ad = a.raw_data();

  if (total < parallel_threshold) {
    for (std::size_t i = 0; i < total; ++i)
      out[i] = op(ad[i]);
  } else {
    auto in_range = tf::make_range(ad, total);
    auto out_range = tf::make_range(out, total);
    tf::parallel_transform(in_range, out_range, op, tf::checked);
  }

  return wasm_ndarray<T>::from_buffer(std::move(buf),
                                      tf::small_vector<int, 3>(a.raw_shape()));
}

template <typename T, typename UnaryOp>
auto unary_op_inplace(wasm_ndarray<T> &a, UnaryOp op) -> void {
  auto total = a.length();
  auto *ad = a.raw_data();

  if (total < parallel_threshold) {
    for (std::size_t i = 0; i < total; ++i)
      ad[i] = op(ad[i]);
  } else {
    auto seq = tf::make_sequence_range(static_cast<int>(total));
    tf::parallel_for_each(
        seq, [ad, op](int i) { ad[i] = op(ad[i]); }, tf::checked);
  }
}

} // namespace detail

// ============================================================================
// Concrete unary ops
// ============================================================================

template <typename T>
auto neg(const wasm_ndarray<T> &a) -> wasm_ndarray<T> {
  return detail::unary_op(a, std::negate<T>{});
}

template <typename T>
auto neg_inplace(wasm_ndarray<T> &a) -> void {
  detail::unary_op_inplace(a, std::negate<T>{});
}

template <typename T>
auto abs(const wasm_ndarray<T> &a) -> wasm_ndarray<T> {
  return detail::unary_op(a, [](T v) -> T { return v < 0 ? -v : v; });
}

template <typename T>
auto abs_inplace(wasm_ndarray<T> &a) -> void {
  detail::unary_op_inplace(a, [](T v) -> T { return v < 0 ? -v : v; });
}

inline auto sqrt(const wasm_ndarray<float> &a) -> wasm_ndarray<float> {
  return detail::unary_op(a, [](float v) -> float { return tf::sqrt(v); });
}

inline auto sqrt_inplace(wasm_ndarray<float> &a) -> void {
  detail::unary_op_inplace(a, [](float v) -> float { return tf::sqrt(v); });
}

// ============================================================================
// clip
// ============================================================================

template <typename T>
auto clip(const wasm_ndarray<T> &a, T lo, T hi) -> wasm_ndarray<T> {
  return detail::unary_op(a, [lo, hi](T v) -> T {
    return v < lo ? lo : (v > hi ? hi : v);
  });
}

template <typename T>
auto clip_inplace(wasm_ndarray<T> &a, T lo, T hi) -> void {
  detail::unary_op_inplace(a, [lo, hi](T v) -> T {
    return v < lo ? lo : (v > hi ? hi : v);
  });
}

// ============================================================================
// Assign — in-place write with broadcasting
// ============================================================================

template <typename T>
auto assign_scalar(wasm_ndarray<T> &target, T scalar) -> void {
  auto total = target.length();
  auto *dst = target.raw_data();
  for (std::size_t i = 0; i < total; ++i)
    dst[i] = scalar;
}

template <typename T>
auto assign_array(wasm_ndarray<T> &target, const wasm_ndarray<T> &source)
    -> void {
  auto out_shape =
      detail::broadcast_shape(target.raw_shape(), source.raw_shape());
  if (!detail::shapes_equal(target.raw_shape(), out_shape))
    throw std::runtime_error(
        "assign: source shape not broadcastable to target shape");

  auto total = target.length();
  auto *dst = target.raw_data();
  auto *src = source.raw_data();

  if (detail::shapes_equal(target.raw_shape(), source.raw_shape())) {
    std::memcpy(dst, src, total * sizeof(T));
  } else {
    auto out_strides = detail::compute_strides(out_shape);
    auto src_bstrides =
        detail::broadcast_strides(source.raw_shape(), out_shape);
    int ndim = static_cast<int>(out_shape.size());
    for (std::size_t i = 0; i < total; ++i) {
      int si = detail::broadcast_index(static_cast<int>(i), out_strides,
                                        src_bstrides, ndim);
      dst[i] = src[si];
    }
  }
}

template <typename T>
auto assign_indexed_scalar(wasm_ndarray<T> &target,
                            const wasm_ndarray<int> &indices, T scalar)
    -> void {
  auto &shape = target.raw_shape();
  auto ndim = static_cast<int>(shape.size());
  auto num_indices = indices.length();
  auto *idx_data = indices.raw_data();
  auto *dst = target.raw_data();

  std::size_t row_stride = 1;
  for (int d = 1; d < ndim; ++d)
    row_stride *= shape[d];

  for (std::size_t i = 0; i < num_indices; ++i) {
    auto idx = idx_data[i];
    auto *row = dst + idx * row_stride;
    for (std::size_t j = 0; j < row_stride; ++j)
      row[j] = scalar;
  }
}

template <typename T>
auto assign_indexed_array(wasm_ndarray<T> &target,
                           const wasm_ndarray<int> &indices,
                           const wasm_ndarray<T> &values) -> void {
  auto &shape = target.raw_shape();
  auto ndim = static_cast<int>(shape.size());
  auto num_indices = indices.length();
  auto *idx_data = indices.raw_data();
  auto *dst = target.raw_data();
  auto *src = values.raw_data();

  std::size_t row_stride = 1;
  for (int d = 1; d < ndim; ++d)
    row_stride *= shape[d];

  // Selection shape: [num_indices, shape[1], shape[2], ...]
  tf::small_vector<int, 3> sel_shape;
  sel_shape.push_back(static_cast<int>(num_indices));
  for (int d = 1; d < ndim; ++d)
    sel_shape.push_back(shape[d]);

  auto bcast_shape =
      detail::broadcast_shape(sel_shape, values.raw_shape());
  if (!detail::shapes_equal(sel_shape, bcast_shape))
    throw std::runtime_error(
        "assign: values shape not broadcastable to selection shape");

  if (detail::shapes_equal(sel_shape, values.raw_shape())) {
    // Fast path: exact match — memcpy row by row
    for (std::size_t i = 0; i < num_indices; ++i) {
      auto idx = idx_data[i];
      std::memcpy(dst + idx * row_stride, src + i * row_stride,
                  row_stride * sizeof(T));
    }
  } else {
    // Broadcast path
    auto sel_strides = detail::compute_strides(sel_shape);
    auto val_bstrides =
        detail::broadcast_strides(values.raw_shape(), sel_shape);
    int sel_ndim = static_cast<int>(sel_shape.size());
    auto sel_total = static_cast<std::size_t>(detail::total_size(sel_shape));

    for (std::size_t flat = 0; flat < sel_total; ++flat) {
      auto row_idx = flat / row_stride;
      auto col_idx = flat % row_stride;
      auto target_row = idx_data[row_idx];
      int si = detail::broadcast_index(static_cast<int>(flat), sel_strides,
                                        val_bstrides, sel_ndim);
      dst[target_row * row_stride + col_idx] = src[si];
    }
  }
}

template <typename T>
auto assign_masked_scalar(wasm_ndarray<T> &target,
                           const wasm_ndarray<std::int8_t> &mask, T scalar)
    -> void {
  auto &shape = target.raw_shape();
  auto ndim = static_cast<int>(shape.size());
  auto *mask_data = mask.raw_data();
  auto *dst = target.raw_data();
  auto num_rows = shape[0];

  std::size_t row_stride = 1;
  for (int d = 1; d < ndim; ++d)
    row_stride *= shape[d];

  for (int i = 0; i < num_rows; ++i) {
    if (mask_data[i]) {
      auto *row = dst + i * row_stride;
      for (std::size_t j = 0; j < row_stride; ++j)
        row[j] = scalar;
    }
  }
}

template <typename T>
auto assign_masked_array(wasm_ndarray<T> &target,
                          const wasm_ndarray<std::int8_t> &mask,
                          const wasm_ndarray<T> &values) -> void {
  auto &shape = target.raw_shape();
  auto ndim = static_cast<int>(shape.size());
  auto *mask_data = mask.raw_data();
  auto *dst = target.raw_data();
  auto *src = values.raw_data();
  auto num_rows = shape[0];

  std::size_t row_stride = 1;
  for (int d = 1; d < ndim; ++d)
    row_stride *= shape[d];

  // Count selected rows for broadcast validation
  int count = 0;
  for (int i = 0; i < num_rows; ++i)
    if (mask_data[i])
      ++count;

  // Selection shape: [count, shape[1], shape[2], ...]
  tf::small_vector<int, 3> sel_shape;
  sel_shape.push_back(count);
  for (int d = 1; d < ndim; ++d)
    sel_shape.push_back(shape[d]);

  auto bcast_shape =
      detail::broadcast_shape(sel_shape, values.raw_shape());
  if (!detail::shapes_equal(sel_shape, bcast_shape))
    throw std::runtime_error(
        "assign: values shape not broadcastable to selection shape");

  if (detail::shapes_equal(sel_shape, values.raw_shape())) {
    // Fast path: exact match — memcpy row by row
    std::size_t src_offset = 0;
    for (int i = 0; i < num_rows; ++i) {
      if (mask_data[i]) {
        std::memcpy(dst + i * row_stride, src + src_offset,
                    row_stride * sizeof(T));
        src_offset += row_stride;
      }
    }
  } else {
    // Broadcast path
    auto sel_strides = detail::compute_strides(sel_shape);
    auto val_bstrides =
        detail::broadcast_strides(values.raw_shape(), sel_shape);
    int sel_ndim = static_cast<int>(sel_shape.size());

    int sel_row = 0;
    for (int i = 0; i < num_rows; ++i) {
      if (mask_data[i]) {
        for (std::size_t j = 0; j < row_stride; ++j) {
          int flat = sel_row * static_cast<int>(row_stride) + static_cast<int>(j);
          int si = detail::broadcast_index(flat, sel_strides,
                                            val_bstrides, sel_ndim);
          dst[i * row_stride + j] = src[si];
        }
        ++sel_row;
      }
    }
  }
}

// ============================================================================
// Math functions (float32 only)
// ============================================================================

inline auto sin(const wasm_ndarray<float> &a) -> wasm_ndarray<float> {
  return detail::unary_op(a, [](float v) -> float { return std::sin(v); });
}

inline auto sin_inplace(wasm_ndarray<float> &a) -> void {
  detail::unary_op_inplace(a, [](float v) -> float { return std::sin(v); });
}

inline auto cos(const wasm_ndarray<float> &a) -> wasm_ndarray<float> {
  return detail::unary_op(a, [](float v) -> float { return std::cos(v); });
}

inline auto cos_inplace(wasm_ndarray<float> &a) -> void {
  detail::unary_op_inplace(a, [](float v) -> float { return std::cos(v); });
}

inline auto tan(const wasm_ndarray<float> &a) -> wasm_ndarray<float> {
  return detail::unary_op(a, [](float v) -> float { return std::tan(v); });
}

inline auto tan_inplace(wasm_ndarray<float> &a) -> void {
  detail::unary_op_inplace(a, [](float v) -> float { return std::tan(v); });
}

inline auto asin(const wasm_ndarray<float> &a) -> wasm_ndarray<float> {
  return detail::unary_op(a, [](float v) -> float { return std::asin(v); });
}

inline auto asin_inplace(wasm_ndarray<float> &a) -> void {
  detail::unary_op_inplace(a, [](float v) -> float { return std::asin(v); });
}

inline auto acos(const wasm_ndarray<float> &a) -> wasm_ndarray<float> {
  return detail::unary_op(a, [](float v) -> float { return std::acos(v); });
}

inline auto acos_inplace(wasm_ndarray<float> &a) -> void {
  detail::unary_op_inplace(a, [](float v) -> float { return std::acos(v); });
}

inline auto atan(const wasm_ndarray<float> &a) -> wasm_ndarray<float> {
  return detail::unary_op(a, [](float v) -> float { return std::atan(v); });
}

inline auto atan_inplace(wasm_ndarray<float> &a) -> void {
  detail::unary_op_inplace(a, [](float v) -> float { return std::atan(v); });
}

inline auto exp(const wasm_ndarray<float> &a) -> wasm_ndarray<float> {
  return detail::unary_op(a, [](float v) -> float { return std::exp(v); });
}

inline auto exp_inplace(wasm_ndarray<float> &a) -> void {
  detail::unary_op_inplace(a, [](float v) -> float { return std::exp(v); });
}

inline auto log(const wasm_ndarray<float> &a) -> wasm_ndarray<float> {
  return detail::unary_op(a, [](float v) -> float { return std::log(v); });
}

inline auto log_inplace(wasm_ndarray<float> &a) -> void {
  detail::unary_op_inplace(a, [](float v) -> float { return std::log(v); });
}

inline auto log2(const wasm_ndarray<float> &a) -> wasm_ndarray<float> {
  return detail::unary_op(a, [](float v) -> float { return std::log2(v); });
}

inline auto log2_inplace(wasm_ndarray<float> &a) -> void {
  detail::unary_op_inplace(a, [](float v) -> float { return std::log2(v); });
}

inline auto log10(const wasm_ndarray<float> &a) -> wasm_ndarray<float> {
  return detail::unary_op(a, [](float v) -> float { return std::log10(v); });
}

inline auto log10_inplace(wasm_ndarray<float> &a) -> void {
  detail::unary_op_inplace(a, [](float v) -> float { return std::log10(v); });
}

inline auto floor(const wasm_ndarray<float> &a) -> wasm_ndarray<float> {
  return detail::unary_op(a, [](float v) -> float { return std::floor(v); });
}

inline auto floor_inplace(wasm_ndarray<float> &a) -> void {
  detail::unary_op_inplace(a, [](float v) -> float { return std::floor(v); });
}

inline auto ceil(const wasm_ndarray<float> &a) -> wasm_ndarray<float> {
  return detail::unary_op(a, [](float v) -> float { return std::ceil(v); });
}

inline auto ceil_inplace(wasm_ndarray<float> &a) -> void {
  detail::unary_op_inplace(a, [](float v) -> float { return std::ceil(v); });
}

inline auto round(const wasm_ndarray<float> &a) -> wasm_ndarray<float> {
  return detail::unary_op(a, [](float v) -> float { return std::round(v); });
}

inline auto round_inplace(wasm_ndarray<float> &a) -> void {
  detail::unary_op_inplace(a, [](float v) -> float { return std::round(v); });
}

inline auto pow(const wasm_ndarray<float> &a, float exponent)
    -> wasm_ndarray<float> {
  return detail::unary_op(
      a, [exponent](float v) -> float { return std::pow(v, exponent); });
}

inline auto pow_inplace(wasm_ndarray<float> &a, float exponent) -> void {
  detail::unary_op_inplace(
      a, [exponent](float v) -> float { return std::pow(v, exponent); });
}

// ============================================================================
// Logical ops (int8/bool only)
// ============================================================================

inline auto logical_not(const wasm_ndarray<std::int8_t> &a)
    -> wasm_ndarray<std::int8_t> {
  return detail::unary_op(a,
      [](std::int8_t v) -> std::int8_t { return v ? 0 : 1; });
}

inline auto logical_not_inplace(wasm_ndarray<std::int8_t> &a) -> void {
  detail::unary_op_inplace(a,
      [](std::int8_t v) -> std::int8_t { return v ? 0 : 1; });
}

inline auto logical_and(const wasm_ndarray<std::int8_t> &a,
                         const wasm_ndarray<std::int8_t> &b)
    -> wasm_ndarray<std::int8_t> {
  return elementwise(a, b,
      [](std::int8_t x, std::int8_t y) -> std::int8_t {
        return (x && y) ? 1 : 0;
      });
}

inline auto logical_or(const wasm_ndarray<std::int8_t> &a,
                        const wasm_ndarray<std::int8_t> &b)
    -> wasm_ndarray<std::int8_t> {
  return elementwise(a, b,
      [](std::int8_t x, std::int8_t y) -> std::int8_t {
        return (x || y) ? 1 : 0;
      });
}

// ============================================================================
// Matrix multiply (with batch broadcasting)
// ============================================================================

template <typename T>
auto mat_mul(const wasm_ndarray<T> &a, const wasm_ndarray<T> &b)
    -> wasm_ndarray<T> {
  auto &as = a.raw_shape();
  auto &bs = b.raw_shape();
  int a_ndim = static_cast<int>(as.size());
  int b_ndim = static_cast<int>(bs.size());

  if (a_ndim < 2 || b_ndim < 2)
    throw std::runtime_error("matMul requires at least 2D arrays");

  int M = as[a_ndim - 2];
  int K = as[a_ndim - 1];
  int K2 = bs[b_ndim - 2];
  int N = bs[b_ndim - 1];

  if (K != K2)
    throw std::runtime_error("matMul inner dimensions mismatch");

  // Batch dimensions
  int a_batch = (a_ndim == 2) ? 1 : as[0];
  int b_batch = (b_ndim == 2) ? 1 : bs[0];

  if (a_batch != b_batch && a_batch != 1 && b_batch != 1)
    throw std::runtime_error("matMul batch dimensions not broadcastable");

  int batch = a_batch > b_batch ? a_batch : b_batch;
  int a_mat_stride = M * K;
  int b_mat_stride = K * N;
  int out_mat_stride = M * N;

  // 2D input: stride 0 (reuse same matrix for all batch elements)
  int a_stride = (a_ndim == 2) ? 0 : a_mat_stride;
  int b_stride = (b_ndim == 2) ? 0 : b_mat_stride;

  int total = batch * M * N;
  tf::buffer<T> buf;
  buf.allocate(static_cast<std::size_t>(total));
  auto *out = buf.data();
  auto *ad = a.raw_data();
  auto *bd = b.raw_data();

  int total_rows = batch * M;
  if (static_cast<std::size_t>(total) < parallel_threshold) {
    for (int br = 0; br < total_rows; ++br) {
      int bi = br / M;
      int i = br % M;
      auto *a_ptr = ad + bi * a_stride + i * K;
      auto *b_ptr = bd + bi * b_stride;
      auto *o_ptr = out + bi * out_mat_stride + i * N;
      for (int j = 0; j < N; ++j)
        o_ptr[j] = T{0};
      for (int k = 0; k < K; ++k) {
        T a_ik = a_ptr[k];
        for (int j = 0; j < N; ++j)
          o_ptr[j] += a_ik * b_ptr[k * N + j];
      }
    }
  } else {
    auto rows = tf::make_sequence_range(total_rows);
    tf::parallel_for_each(
        rows,
        [ad, bd, out, K, N, M, a_stride, b_stride, out_mat_stride](int br) {
          int bi = br / M;
          int i = br % M;
          auto *a_ptr = ad + bi * a_stride + i * K;
          auto *b_ptr = bd + bi * b_stride;
          auto *o_ptr = out + bi * out_mat_stride + i * N;
          for (int j = 0; j < N; ++j)
            o_ptr[j] = T{0};
          for (int k = 0; k < K; ++k) {
            T a_ik = a_ptr[k];
            for (int j = 0; j < N; ++j)
              o_ptr[j] += a_ik * b_ptr[k * N + j];
          }
        },
        tf::checked);
  }

  // Output shape: 3D if either input is 3D, else 2D
  tf::small_vector<int, 3> out_shape;
  if (a_ndim == 3 || b_ndim == 3)
    out_shape.push_back(batch);
  out_shape.push_back(M);
  out_shape.push_back(N);

  return wasm_ndarray<T>::from_buffer(std::move(buf), std::move(out_shape));
}

// ============================================================================
// Cross-type element-wise comparison: T input → int8_t output
// ============================================================================

template <typename T, typename BinaryOp>
auto elementwise_cmp(const wasm_ndarray<T> &a, const wasm_ndarray<T> &b,
                     BinaryOp op) -> wasm_ndarray<std::int8_t> {
  auto out_shape = detail::broadcast_shape(a.raw_shape(), b.raw_shape());
  int total = detail::total_size(out_shape);

  tf::buffer<std::int8_t> buf;
  buf.allocate(static_cast<std::size_t>(total));
  auto *out = buf.data();
  auto *ad = a.raw_data();
  auto *bd = b.raw_data();

  if (detail::shapes_equal(a.raw_shape(), b.raw_shape())) {
    if (static_cast<std::size_t>(total) < parallel_threshold) {
      for (int i = 0; i < total; ++i)
        out[i] = op(ad[i], bd[i]) ? 1 : 0;
    } else {
      auto seq = tf::make_sequence_range(total);
      tf::parallel_for_each(
          seq,
          [ad, bd, out, op](int i) { out[i] = op(ad[i], bd[i]) ? 1 : 0; },
          tf::checked);
    }
  } else {
    auto out_strides = detail::compute_strides(out_shape);
    auto a_bstrides = detail::broadcast_strides(a.raw_shape(), out_shape);
    auto b_bstrides = detail::broadcast_strides(b.raw_shape(), out_shape);
    int ndim = static_cast<int>(out_shape.size());

    if (static_cast<std::size_t>(total) < parallel_threshold) {
      for (int i = 0; i < total; ++i) {
        int ai = detail::broadcast_index(i, out_strides, a_bstrides, ndim);
        int bi = detail::broadcast_index(i, out_strides, b_bstrides, ndim);
        out[i] = op(ad[ai], bd[bi]) ? 1 : 0;
      }
    } else {
      auto seq = tf::make_sequence_range(total);
      tf::parallel_for_each(
          seq,
          [ad, bd, out, op, out_strides, a_bstrides, b_bstrides, ndim](int i) {
            int ai =
                detail::broadcast_index(i, out_strides, a_bstrides, ndim);
            int bi =
                detail::broadcast_index(i, out_strides, b_bstrides, ndim);
            out[i] = op(ad[ai], bd[bi]) ? 1 : 0;
          },
          tf::checked);
    }
  }

  return wasm_ndarray<std::int8_t>::from_buffer(std::move(buf),
                                                std::move(out_shape));
}

template <typename T, typename BinaryOp>
auto scalar_cmp(const wasm_ndarray<T> &a, T scalar,
                BinaryOp op) -> wasm_ndarray<std::int8_t> {
  auto total = a.length();
  tf::buffer<std::int8_t> buf;
  buf.allocate(total);
  auto *out = buf.data();
  auto *ad = a.raw_data();

  if (total < parallel_threshold) {
    for (std::size_t i = 0; i < total; ++i)
      out[i] = op(ad[i], scalar) ? 1 : 0;
  } else {
    auto in_range = tf::make_range(ad, total);
    auto out_range = tf::make_range(out, total);
    tf::parallel_transform(
        in_range, out_range,
        [scalar, op](T v) -> std::int8_t { return op(v, scalar) ? 1 : 0; },
        tf::checked);
  }

  return wasm_ndarray<std::int8_t>::from_buffer(
      std::move(buf), tf::small_vector<int, 3>(a.raw_shape()));
}

// Concrete comparison ops

template <typename T>
auto eq(const wasm_ndarray<T> &a, const wasm_ndarray<T> &b)
    -> wasm_ndarray<std::int8_t> {
  return elementwise_cmp(a, b, std::equal_to<T>{});
}

template <typename T>
auto eq_scalar(const wasm_ndarray<T> &a, T scalar)
    -> wasm_ndarray<std::int8_t> {
  return scalar_cmp(a, scalar, std::equal_to<T>{});
}

template <typename T>
auto neq(const wasm_ndarray<T> &a, const wasm_ndarray<T> &b)
    -> wasm_ndarray<std::int8_t> {
  return elementwise_cmp(a, b, std::not_equal_to<T>{});
}

template <typename T>
auto neq_scalar(const wasm_ndarray<T> &a, T scalar)
    -> wasm_ndarray<std::int8_t> {
  return scalar_cmp(a, scalar, std::not_equal_to<T>{});
}

template <typename T>
auto lt(const wasm_ndarray<T> &a, const wasm_ndarray<T> &b)
    -> wasm_ndarray<std::int8_t> {
  return elementwise_cmp(a, b, std::less<T>{});
}

template <typename T>
auto lt_scalar(const wasm_ndarray<T> &a, T scalar)
    -> wasm_ndarray<std::int8_t> {
  return scalar_cmp(a, scalar, std::less<T>{});
}

template <typename T>
auto gt(const wasm_ndarray<T> &a, const wasm_ndarray<T> &b)
    -> wasm_ndarray<std::int8_t> {
  return elementwise_cmp(a, b, std::greater<T>{});
}

template <typename T>
auto gt_scalar(const wasm_ndarray<T> &a, T scalar)
    -> wasm_ndarray<std::int8_t> {
  return scalar_cmp(a, scalar, std::greater<T>{});
}

template <typename T>
auto lte(const wasm_ndarray<T> &a, const wasm_ndarray<T> &b)
    -> wasm_ndarray<std::int8_t> {
  return elementwise_cmp(a, b, std::less_equal<T>{});
}

template <typename T>
auto lte_scalar(const wasm_ndarray<T> &a, T scalar)
    -> wasm_ndarray<std::int8_t> {
  return scalar_cmp(a, scalar, std::less_equal<T>{});
}

template <typename T>
auto gte(const wasm_ndarray<T> &a, const wasm_ndarray<T> &b)
    -> wasm_ndarray<std::int8_t> {
  return elementwise_cmp(a, b, std::greater_equal<T>{});
}

template <typename T>
auto gte_scalar(const wasm_ndarray<T> &a, T scalar)
    -> wasm_ndarray<std::int8_t> {
  return scalar_cmp(a, scalar, std::greater_equal<T>{});
}

// ============================================================================
// dot — sum of element-wise product along last axis
// ============================================================================

template <typename T>
auto dot(const wasm_ndarray<T> &a, const wasm_ndarray<T> &b)
    -> wasm_ndarray<T> {
  auto out_shape = detail::broadcast_shape(a.raw_shape(), b.raw_shape());
  int ndim = static_cast<int>(out_shape.size());
  if (ndim < 1)
    throw std::runtime_error("dot requires at least 1D arrays");

  int inner = out_shape[ndim - 1];

  // Compute batch size (product of all dims except last)
  int batch = 1;
  for (int d = 0; d < ndim - 1; ++d)
    batch *= out_shape[d];

  tf::buffer<T> buf;
  buf.allocate(static_cast<std::size_t>(batch));
  auto *out = buf.data();
  auto *ad = a.raw_data();
  auto *bd = b.raw_data();

  if (detail::shapes_equal(a.raw_shape(), b.raw_shape())) {
    // Fast path: same shape
    if (static_cast<std::size_t>(batch) < parallel_threshold) {
      for (int i = 0; i < batch; ++i) {
        T acc = T{0};
        int base = i * inner;
        for (int j = 0; j < inner; ++j)
          acc += ad[base + j] * bd[base + j];
        out[i] = acc;
      }
    } else {
      auto seq = tf::make_sequence_range(batch);
      tf::parallel_for_each(
          seq,
          [ad, bd, out, inner](int i) {
            T acc = T{0};
            int base = i * inner;
            for (int j = 0; j < inner; ++j)
              acc += ad[base + j] * bd[base + j];
            out[i] = acc;
          },
          tf::checked);
    }
  } else {
    // Broadcast path
    auto out_strides = detail::compute_strides(out_shape);
    auto a_bstrides =
        detail::broadcast_strides(a.raw_shape(), out_shape);
    auto b_bstrides =
        detail::broadcast_strides(b.raw_shape(), out_shape);

    if (static_cast<std::size_t>(batch) < parallel_threshold) {
      for (int i = 0; i < batch; ++i) {
        T acc = T{0};
        int flat_base = i * inner;
        for (int j = 0; j < inner; ++j) {
          int flat = flat_base + j;
          int ai = detail::broadcast_index(flat, out_strides, a_bstrides,
                                            ndim);
          int bi = detail::broadcast_index(flat, out_strides, b_bstrides,
                                            ndim);
          acc += ad[ai] * bd[bi];
        }
        out[i] = acc;
      }
    } else {
      auto seq = tf::make_sequence_range(batch);
      tf::parallel_for_each(
          seq,
          [ad, bd, out, inner, out_strides, a_bstrides, b_bstrides,
           ndim](int i) {
            T acc = T{0};
            int flat_base = i * inner;
            for (int j = 0; j < inner; ++j) {
              int flat = flat_base + j;
              int ai = detail::broadcast_index(flat, out_strides, a_bstrides,
                                                ndim);
              int bi = detail::broadcast_index(flat, out_strides, b_bstrides,
                                                ndim);
              acc += ad[ai] * bd[bi];
            }
            out[i] = acc;
          },
          tf::checked);
    }
  }

  // Output shape is broadcast shape minus last dim
  tf::small_vector<int, 3> result_shape;
  for (int d = 0; d < ndim - 1; ++d)
    result_shape.push_back(out_shape[d]);
  if (result_shape.empty())
    result_shape.push_back(1);

  return wasm_ndarray<T>::from_buffer(std::move(buf), std::move(result_shape));
}

// ============================================================================
// cross — cross product along last axis (must be 3)
// ============================================================================

template <typename T>
auto cross(const wasm_ndarray<T> &a, const wasm_ndarray<T> &b)
    -> wasm_ndarray<T> {
  auto out_shape = detail::broadcast_shape(a.raw_shape(), b.raw_shape());
  int ndim = static_cast<int>(out_shape.size());
  if (ndim < 1 || out_shape[ndim - 1] != 3)
    throw std::runtime_error("cross requires last axis to be 3");

  int batch = 1;
  for (int d = 0; d < ndim - 1; ++d)
    batch *= out_shape[d];

  int total = batch * 3;
  tf::buffer<T> buf;
  buf.allocate(static_cast<std::size_t>(total));
  auto *out = buf.data();
  auto *ad = a.raw_data();
  auto *bd = b.raw_data();

  if (detail::shapes_equal(a.raw_shape(), b.raw_shape())) {
    // Fast path: same shape
    if (static_cast<std::size_t>(batch) < parallel_threshold) {
      for (int i = 0; i < batch; ++i) {
        int base = i * 3;
        T a0 = ad[base], a1 = ad[base + 1], a2 = ad[base + 2];
        T b0 = bd[base], b1 = bd[base + 1], b2 = bd[base + 2];
        out[base] = a1 * b2 - a2 * b1;
        out[base + 1] = a2 * b0 - a0 * b2;
        out[base + 2] = a0 * b1 - a1 * b0;
      }
    } else {
      auto seq = tf::make_sequence_range(batch);
      tf::parallel_for_each(
          seq,
          [ad, bd, out](int i) {
            int base = i * 3;
            T a0 = ad[base], a1 = ad[base + 1], a2 = ad[base + 2];
            T b0 = bd[base], b1 = bd[base + 1], b2 = bd[base + 2];
            out[base] = a1 * b2 - a2 * b1;
            out[base + 1] = a2 * b0 - a0 * b2;
            out[base + 2] = a0 * b1 - a1 * b0;
          },
          tf::checked);
    }
  } else {
    // Broadcast path
    auto out_strides = detail::compute_strides(out_shape);
    auto a_bstrides =
        detail::broadcast_strides(a.raw_shape(), out_shape);
    auto b_bstrides =
        detail::broadcast_strides(b.raw_shape(), out_shape);

    if (static_cast<std::size_t>(batch) < parallel_threshold) {
      for (int i = 0; i < batch; ++i) {
        int flat_base = i * 3;
        int ai0 = detail::broadcast_index(flat_base, out_strides,
                                           a_bstrides, ndim);
        int bi0 = detail::broadcast_index(flat_base, out_strides,
                                           b_bstrides, ndim);
        T a0 = ad[ai0], a1 = ad[ai0 + 1], a2 = ad[ai0 + 2];
        T b0 = bd[bi0], b1 = bd[bi0 + 1], b2 = bd[bi0 + 2];
        out[flat_base] = a1 * b2 - a2 * b1;
        out[flat_base + 1] = a2 * b0 - a0 * b2;
        out[flat_base + 2] = a0 * b1 - a1 * b0;
      }
    } else {
      auto seq = tf::make_sequence_range(batch);
      tf::parallel_for_each(
          seq,
          [ad, bd, out, out_strides, a_bstrides, b_bstrides, ndim](int i) {
            int flat_base = i * 3;
            int ai0 = detail::broadcast_index(flat_base, out_strides,
                                               a_bstrides, ndim);
            int bi0 = detail::broadcast_index(flat_base, out_strides,
                                               b_bstrides, ndim);
            T a0 = ad[ai0], a1 = ad[ai0 + 1], a2 = ad[ai0 + 2];
            T b0 = bd[bi0], b1 = bd[bi0 + 1], b2 = bd[bi0 + 2];
            out[flat_base] = a1 * b2 - a2 * b1;
            out[flat_base + 1] = a2 * b0 - a0 * b2;
            out[flat_base + 2] = a0 * b1 - a1 * b0;
          },
          tf::checked);
    }
  }

  return wasm_ndarray<T>::from_buffer(std::move(buf), std::move(out_shape));
}

// ============================================================================
// atan2 (float32 only, broadcasts)
// ============================================================================

inline auto atan2(const wasm_ndarray<float> &y, const wasm_ndarray<float> &x)
    -> wasm_ndarray<float> {
  return elementwise(y, x,
      [](float a, float b) -> float { return std::atan2(a, b); });
}

// ============================================================================
// Type casting
// ============================================================================

template <typename From, typename To>
auto cast(const wasm_ndarray<From> &a) -> wasm_ndarray<To> {
  auto total = a.length();
  tf::buffer<To> buf;
  buf.allocate(total);
  auto *out = buf.data();
  auto *ad = a.raw_data();

  if (total < parallel_threshold) {
    for (std::size_t i = 0; i < total; ++i)
      out[i] = static_cast<To>(ad[i]);
  } else {
    auto in_range = tf::make_range(ad, total);
    auto out_range = tf::make_range(out, total);
    tf::parallel_transform(
        in_range, out_range,
        [](From v) -> To { return static_cast<To>(v); }, tf::checked);
  }

  return wasm_ndarray<To>::from_buffer(std::move(buf),
                                       tf::small_vector<int, 3>(a.raw_shape()));
}

} // namespace ts
} // namespace tf
