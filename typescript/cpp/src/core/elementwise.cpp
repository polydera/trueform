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

#include "trueform/core/inverted.hpp"
#include "trueform/core/transformation_view.hpp"
#include "trueform/ts/core/elementwise.hpp"
#include "trueform/ts/core/promise.hpp"
#include <cstdint>
#include <emscripten/bind.h>

// ============================================================================
// Sync — buffer × buffer (copy)
// ============================================================================

template <typename T>
static auto sync_add(tf::ts::wasm_ndarray<T> &a, tf::ts::wasm_ndarray<T> &b)
    -> tf::ts::wasm_ndarray<T> {
  return tf::ts::add(a, b);
}

template <typename T>
static auto sync_sub(tf::ts::wasm_ndarray<T> &a, tf::ts::wasm_ndarray<T> &b)
    -> tf::ts::wasm_ndarray<T> {
  return tf::ts::sub(a, b);
}

template <typename T>
static auto sync_mul(tf::ts::wasm_ndarray<T> &a, tf::ts::wasm_ndarray<T> &b)
    -> tf::ts::wasm_ndarray<T> {
  return tf::ts::mul(a, b);
}

// ============================================================================
// Sync — buffer × scalar (copy)
// ============================================================================

template <typename T>
static auto sync_add_scalar(tf::ts::wasm_ndarray<T> &a, T scalar)
    -> tf::ts::wasm_ndarray<T> {
  return tf::ts::add_scalar(a, scalar);
}

template <typename T>
static auto sync_sub_scalar(tf::ts::wasm_ndarray<T> &a, T scalar)
    -> tf::ts::wasm_ndarray<T> {
  return tf::ts::sub_scalar(a, scalar);
}

template <typename T>
static auto sync_mul_scalar(tf::ts::wasm_ndarray<T> &a, T scalar)
    -> tf::ts::wasm_ndarray<T> {
  return tf::ts::mul_scalar(a, scalar);
}

// ============================================================================
// Sync — buffer × buffer (in-place)
// ============================================================================

template <typename T>
static auto sync_add_inplace(tf::ts::wasm_ndarray<T> &a,
                             tf::ts::wasm_ndarray<T> &b) -> void {
  tf::ts::add_inplace(a, b);
}

template <typename T>
static auto sync_sub_inplace(tf::ts::wasm_ndarray<T> &a,
                             tf::ts::wasm_ndarray<T> &b) -> void {
  tf::ts::sub_inplace(a, b);
}

template <typename T>
static auto sync_mul_inplace(tf::ts::wasm_ndarray<T> &a,
                             tf::ts::wasm_ndarray<T> &b) -> void {
  tf::ts::mul_inplace(a, b);
}

// ============================================================================
// Sync — buffer × scalar (in-place)
// ============================================================================

template <typename T>
static auto sync_add_scalar_inplace(tf::ts::wasm_ndarray<T> &a, T scalar)
    -> void {
  tf::ts::add_scalar_inplace(a, scalar);
}

template <typename T>
static auto sync_sub_scalar_inplace(tf::ts::wasm_ndarray<T> &a, T scalar)
    -> void {
  tf::ts::sub_scalar_inplace(a, scalar);
}

template <typename T>
static auto sync_mul_scalar_inplace(tf::ts::wasm_ndarray<T> &a, T scalar)
    -> void {
  tf::ts::mul_scalar_inplace(a, scalar);
}

// ============================================================================
// Sync — matMul
// ============================================================================

template <typename T>
static auto sync_mat_mul(tf::ts::wasm_ndarray<T> &a,
                         tf::ts::wasm_ndarray<T> &b)
    -> tf::ts::wasm_ndarray<T> {
  return tf::ts::mat_mul(a, b);
}

// ============================================================================
// Sync — inverted (4x4 or 3x3 affine matrix)
// ============================================================================

template <typename T>
static auto sync_inverted(tf::ts::wasm_ndarray<T> &m)
    -> tf::ts::wasm_ndarray<T> {
  auto rows = m.shape_at(0);
  if (rows == 4) {
    auto input_view = tf::make_transformation_view<3>(
        const_cast<T *>(m.raw_data()));
    auto out = tf::inverted(input_view);
    tf::buffer<T> buf;
    buf.allocate(16);
    for (std::size_t i = 0; i < 3; ++i)
      for (std::size_t j = 0; j < 4; ++j)
        buf[i * 4 + j] = out(i, j);
    buf[12] = T{0}; buf[13] = T{0}; buf[14] = T{0}; buf[15] = T{1};
    return tf::ts::wasm_ndarray<T>::from_buffer(std::move(buf), {4, 4});
  }
  if (rows == 3) {
    auto input_view = tf::make_transformation_view<2>(
        const_cast<T *>(m.raw_data()));
    auto out = tf::inverted(input_view);
    tf::buffer<T> buf;
    buf.allocate(9);
    for (std::size_t i = 0; i < 2; ++i)
      for (std::size_t j = 0; j < 3; ++j)
        buf[i * 3 + j] = out(i, j);
    buf[6] = T{0}; buf[7] = T{0}; buf[8] = T{1};
    return tf::ts::wasm_ndarray<T>::from_buffer(std::move(buf), {3, 3});
  }
  return {};
}

// ============================================================================
// Async — buffer × buffer (copy)
// ============================================================================

template <typename T>
static auto async_add(tf::ts::wasm_ndarray<T> &a, tf::ts::wasm_ndarray<T> &b)
    -> tf::ts::promise_t {
  return tf::ts::promise(
      [a, b]() { return tf::ts::add(a, b); });
}

template <typename T>
static auto async_sub(tf::ts::wasm_ndarray<T> &a, tf::ts::wasm_ndarray<T> &b)
    -> tf::ts::promise_t {
  return tf::ts::promise(
      [a, b]() { return tf::ts::sub(a, b); });
}

template <typename T>
static auto async_mul(tf::ts::wasm_ndarray<T> &a, tf::ts::wasm_ndarray<T> &b)
    -> tf::ts::promise_t {
  return tf::ts::promise(
      [a, b]() { return tf::ts::mul(a, b); });
}

// ============================================================================
// Async — buffer × scalar (copy)
// ============================================================================

template <typename T>
static auto async_add_scalar(tf::ts::wasm_ndarray<T> &a, T scalar)
    -> tf::ts::promise_t {
  return tf::ts::promise(
      [a, scalar]() { return tf::ts::add_scalar(a, scalar); });
}

template <typename T>
static auto async_sub_scalar(tf::ts::wasm_ndarray<T> &a, T scalar)
    -> tf::ts::promise_t {
  return tf::ts::promise(
      [a, scalar]() { return tf::ts::sub_scalar(a, scalar); });
}

template <typename T>
static auto async_mul_scalar(tf::ts::wasm_ndarray<T> &a, T scalar)
    -> tf::ts::promise_t {
  return tf::ts::promise(
      [a, scalar]() { return tf::ts::mul_scalar(a, scalar); });
}

// ============================================================================
// Async — matMul
// ============================================================================

template <typename T>
static auto async_mat_mul(tf::ts::wasm_ndarray<T> &a,
                          tf::ts::wasm_ndarray<T> &b) -> tf::ts::promise_t {
  return tf::ts::promise(
      [a, b]() { return tf::ts::mat_mul(a, b); });
}

// ============================================================================
// Async — buffer × buffer (in-place)
// ============================================================================

template <typename T>
static auto async_add_inplace(tf::ts::wasm_ndarray<T> &a,
                              tf::ts::wasm_ndarray<T> &b)
    -> tf::ts::promise_t {
  return tf::ts::promise([a, b]() {
    tf::ts::add_inplace(const_cast<tf::ts::wasm_ndarray<T> &>(a), b);
    return a;
  });
}

template <typename T>
static auto async_sub_inplace(tf::ts::wasm_ndarray<T> &a,
                              tf::ts::wasm_ndarray<T> &b)
    -> tf::ts::promise_t {
  return tf::ts::promise([a, b]() {
    tf::ts::sub_inplace(const_cast<tf::ts::wasm_ndarray<T> &>(a), b);
    return a;
  });
}

template <typename T>
static auto async_mul_inplace(tf::ts::wasm_ndarray<T> &a,
                              tf::ts::wasm_ndarray<T> &b)
    -> tf::ts::promise_t {
  return tf::ts::promise([a, b]() {
    tf::ts::mul_inplace(const_cast<tf::ts::wasm_ndarray<T> &>(a), b);
    return a;
  });
}

// ============================================================================
// Async — buffer × scalar (in-place)
// ============================================================================

template <typename T>
static auto async_add_scalar_inplace(tf::ts::wasm_ndarray<T> &a, T scalar)
    -> tf::ts::promise_t {
  return tf::ts::promise([a, scalar]() {
    tf::ts::add_scalar_inplace(const_cast<tf::ts::wasm_ndarray<T> &>(a),
                               scalar);
    return a;
  });
}

template <typename T>
static auto async_sub_scalar_inplace(tf::ts::wasm_ndarray<T> &a, T scalar)
    -> tf::ts::promise_t {
  return tf::ts::promise([a, scalar]() {
    tf::ts::sub_scalar_inplace(const_cast<tf::ts::wasm_ndarray<T> &>(a),
                               scalar);
    return a;
  });
}

template <typename T>
static auto async_mul_scalar_inplace(tf::ts::wasm_ndarray<T> &a, T scalar)
    -> tf::ts::promise_t {
  return tf::ts::promise([a, scalar]() {
    tf::ts::mul_scalar_inplace(const_cast<tf::ts::wasm_ndarray<T> &>(a),
                               scalar);
    return a;
  });
}

// ============================================================================
// Sync — div
// ============================================================================

template <typename T>
static auto sync_div(tf::ts::wasm_ndarray<T> &a, tf::ts::wasm_ndarray<T> &b)
    -> tf::ts::wasm_ndarray<T> {
  return tf::ts::div(a, b);
}

template <typename T>
static auto sync_div_scalar(tf::ts::wasm_ndarray<T> &a, T scalar)
    -> tf::ts::wasm_ndarray<T> {
  return tf::ts::div_scalar(a, scalar);
}

template <typename T>
static auto sync_div_inplace(tf::ts::wasm_ndarray<T> &a,
                              tf::ts::wasm_ndarray<T> &b) -> void {
  tf::ts::div_inplace(a, b);
}

template <typename T>
static auto sync_div_scalar_inplace(tf::ts::wasm_ndarray<T> &a, T scalar)
    -> void {
  tf::ts::div_scalar_inplace(a, scalar);
}

// ============================================================================
// Async — div
// ============================================================================

template <typename T>
static auto async_div(tf::ts::wasm_ndarray<T> &a, tf::ts::wasm_ndarray<T> &b)
    -> tf::ts::promise_t {
  return tf::ts::promise([a, b]() { return tf::ts::div(a, b); });
}

template <typename T>
static auto async_div_scalar(tf::ts::wasm_ndarray<T> &a, T scalar)
    -> tf::ts::promise_t {
  return tf::ts::promise(
      [a, scalar]() { return tf::ts::div_scalar(a, scalar); });
}

template <typename T>
static auto async_div_inplace(tf::ts::wasm_ndarray<T> &a,
                               tf::ts::wasm_ndarray<T> &b)
    -> tf::ts::promise_t {
  return tf::ts::promise([a, b]() {
    tf::ts::div_inplace(const_cast<tf::ts::wasm_ndarray<T> &>(a), b);
    return a;
  });
}

template <typename T>
static auto async_div_scalar_inplace(tf::ts::wasm_ndarray<T> &a, T scalar)
    -> tf::ts::promise_t {
  return tf::ts::promise([a, scalar]() {
    tf::ts::div_scalar_inplace(const_cast<tf::ts::wasm_ndarray<T> &>(a),
                                scalar);
    return a;
  });
}

// ============================================================================
// Sync — mod
// ============================================================================

template <typename T>
static auto sync_mod(tf::ts::wasm_ndarray<T> &a, tf::ts::wasm_ndarray<T> &b)
    -> tf::ts::wasm_ndarray<T> {
  return tf::ts::mod(a, b);
}

template <typename T>
static auto sync_mod_scalar(tf::ts::wasm_ndarray<T> &a, T scalar)
    -> tf::ts::wasm_ndarray<T> {
  return tf::ts::mod_scalar(a, scalar);
}

template <typename T>
static auto sync_mod_inplace(tf::ts::wasm_ndarray<T> &a,
                              tf::ts::wasm_ndarray<T> &b) -> void {
  tf::ts::mod_inplace(a, b);
}

template <typename T>
static auto sync_mod_scalar_inplace(tf::ts::wasm_ndarray<T> &a, T scalar)
    -> void {
  tf::ts::mod_scalar_inplace(a, scalar);
}

// ============================================================================
// Async — mod
// ============================================================================

template <typename T>
static auto async_mod(tf::ts::wasm_ndarray<T> &a, tf::ts::wasm_ndarray<T> &b)
    -> tf::ts::promise_t {
  return tf::ts::promise([a, b]() { return tf::ts::mod(a, b); });
}

template <typename T>
static auto async_mod_scalar(tf::ts::wasm_ndarray<T> &a, T scalar)
    -> tf::ts::promise_t {
  return tf::ts::promise(
      [a, scalar]() { return tf::ts::mod_scalar(a, scalar); });
}

template <typename T>
static auto async_mod_inplace(tf::ts::wasm_ndarray<T> &a,
                               tf::ts::wasm_ndarray<T> &b)
    -> tf::ts::promise_t {
  return tf::ts::promise([a, b]() {
    tf::ts::mod_inplace(const_cast<tf::ts::wasm_ndarray<T> &>(a), b);
    return a;
  });
}

template <typename T>
static auto async_mod_scalar_inplace(tf::ts::wasm_ndarray<T> &a, T scalar)
    -> tf::ts::promise_t {
  return tf::ts::promise([a, scalar]() {
    tf::ts::mod_scalar_inplace(const_cast<tf::ts::wasm_ndarray<T> &>(a),
                                scalar);
    return a;
  });
}

// ============================================================================
// Sync — unary ops
// ============================================================================

template <typename T>
static auto sync_neg(tf::ts::wasm_ndarray<T> &a) -> tf::ts::wasm_ndarray<T> {
  return tf::ts::neg(a);
}

template <typename T>
static auto sync_neg_inplace(tf::ts::wasm_ndarray<T> &a) -> void {
  tf::ts::neg_inplace(a);
}

template <typename T>
static auto sync_abs(tf::ts::wasm_ndarray<T> &a) -> tf::ts::wasm_ndarray<T> {
  return tf::ts::abs(a);
}

template <typename T>
static auto sync_abs_inplace(tf::ts::wasm_ndarray<T> &a) -> void {
  tf::ts::abs_inplace(a);
}

template <typename T>
static auto sync_sqrt(tf::ts::wasm_ndarray<T> &a)
    -> tf::ts::wasm_ndarray<T> {
  return tf::ts::sqrt(a);
}

template <typename T>
static auto sync_sqrt_inplace(tf::ts::wasm_ndarray<T> &a) -> void {
  tf::ts::sqrt_inplace(a);
}

// ============================================================================
// Async — unary ops
// ============================================================================

template <typename T>
static auto async_neg(tf::ts::wasm_ndarray<T> &a) -> tf::ts::promise_t {
  return tf::ts::promise([a]() { return tf::ts::neg(a); });
}

template <typename T>
static auto async_abs(tf::ts::wasm_ndarray<T> &a) -> tf::ts::promise_t {
  return tf::ts::promise([a]() { return tf::ts::abs(a); });
}

template <typename T>
static auto async_sqrt(tf::ts::wasm_ndarray<T> &a) -> tf::ts::promise_t {
  return tf::ts::promise([a]() { return tf::ts::sqrt(a); });
}

// ============================================================================
// Sync — clip
// ============================================================================

template <typename T>
static auto sync_clip(tf::ts::wasm_ndarray<T> &a, T lo, T hi)
    -> tf::ts::wasm_ndarray<T> {
  return tf::ts::clip(a, lo, hi);
}

template <typename T>
static auto sync_clip_inplace(tf::ts::wasm_ndarray<T> &a, T lo, T hi)
    -> void {
  tf::ts::clip_inplace(a, lo, hi);
}

// ============================================================================
// Async — clip
// ============================================================================

template <typename T>
static auto async_clip(tf::ts::wasm_ndarray<T> &a, T lo, T hi)
    -> tf::ts::promise_t {
  return tf::ts::promise(
      [a, lo, hi]() { return tf::ts::clip(a, lo, hi); });
}

template <typename T>
static auto async_clip_inplace(tf::ts::wasm_ndarray<T> &a, T lo, T hi)
    -> tf::ts::promise_t {
  return tf::ts::promise([a, lo, hi]() {
    tf::ts::clip_inplace(const_cast<tf::ts::wasm_ndarray<T> &>(a), lo, hi);
    return a;
  });
}

// ============================================================================
// Sync — assign
// ============================================================================

template <typename T>
static auto sync_assign_scalar(tf::ts::wasm_ndarray<T> &target, T scalar)
    -> void {
  tf::ts::assign_scalar(target, scalar);
}

template <typename T>
static auto sync_assign_array(tf::ts::wasm_ndarray<T> &target,
                               tf::ts::wasm_ndarray<T> &source) -> void {
  tf::ts::assign_array(target, source);
}

template <typename T>
static auto sync_assign_indexed_scalar(tf::ts::wasm_ndarray<T> &target,
                                        tf::ts::wasm_ndarray<int> &indices,
                                        T scalar) -> void {
  tf::ts::assign_indexed_scalar(target, indices, scalar);
}

template <typename T>
static auto sync_assign_indexed_array(tf::ts::wasm_ndarray<T> &target,
                                       tf::ts::wasm_ndarray<int> &indices,
                                       tf::ts::wasm_ndarray<T> &values)
    -> void {
  tf::ts::assign_indexed_array(target, indices, values);
}

template <typename T>
static auto sync_assign_masked_scalar(tf::ts::wasm_ndarray<T> &target,
                                       tf::ts::wasm_ndarray<std::int8_t> &mask,
                                       T scalar) -> void {
  tf::ts::assign_masked_scalar(target, mask, scalar);
}

template <typename T>
static auto sync_assign_masked_array(tf::ts::wasm_ndarray<T> &target,
                                      tf::ts::wasm_ndarray<std::int8_t> &mask,
                                      tf::ts::wasm_ndarray<T> &values)
    -> void {
  tf::ts::assign_masked_array(target, mask, values);
}

// ============================================================================
// Sync — math functions
// ============================================================================

template <typename T>
static auto sync_sin(tf::ts::wasm_ndarray<T> &a) -> tf::ts::wasm_ndarray<T> {
  return tf::ts::sin(a);
}
template <typename T>
static auto sync_sin_inplace(tf::ts::wasm_ndarray<T> &a) -> void {
  tf::ts::sin_inplace(a);
}
template <typename T>
static auto sync_cos(tf::ts::wasm_ndarray<T> &a) -> tf::ts::wasm_ndarray<T> {
  return tf::ts::cos(a);
}
template <typename T>
static auto sync_cos_inplace(tf::ts::wasm_ndarray<T> &a) -> void {
  tf::ts::cos_inplace(a);
}
template <typename T>
static auto sync_tan(tf::ts::wasm_ndarray<T> &a) -> tf::ts::wasm_ndarray<T> {
  return tf::ts::tan(a);
}
template <typename T>
static auto sync_tan_inplace(tf::ts::wasm_ndarray<T> &a) -> void {
  tf::ts::tan_inplace(a);
}
template <typename T>
static auto sync_asin(tf::ts::wasm_ndarray<T> &a) -> tf::ts::wasm_ndarray<T> {
  return tf::ts::asin(a);
}
template <typename T>
static auto sync_asin_inplace(tf::ts::wasm_ndarray<T> &a) -> void {
  tf::ts::asin_inplace(a);
}
template <typename T>
static auto sync_acos(tf::ts::wasm_ndarray<T> &a) -> tf::ts::wasm_ndarray<T> {
  return tf::ts::acos(a);
}
template <typename T>
static auto sync_acos_inplace(tf::ts::wasm_ndarray<T> &a) -> void {
  tf::ts::acos_inplace(a);
}
template <typename T>
static auto sync_atan(tf::ts::wasm_ndarray<T> &a) -> tf::ts::wasm_ndarray<T> {
  return tf::ts::atan(a);
}
template <typename T>
static auto sync_atan_inplace(tf::ts::wasm_ndarray<T> &a) -> void {
  tf::ts::atan_inplace(a);
}
template <typename T>
static auto sync_exp(tf::ts::wasm_ndarray<T> &a) -> tf::ts::wasm_ndarray<T> {
  return tf::ts::exp(a);
}
template <typename T>
static auto sync_exp_inplace(tf::ts::wasm_ndarray<T> &a) -> void {
  tf::ts::exp_inplace(a);
}
template <typename T>
static auto sync_log(tf::ts::wasm_ndarray<T> &a) -> tf::ts::wasm_ndarray<T> {
  return tf::ts::log(a);
}
template <typename T>
static auto sync_log_inplace(tf::ts::wasm_ndarray<T> &a) -> void {
  tf::ts::log_inplace(a);
}
template <typename T>
static auto sync_log2(tf::ts::wasm_ndarray<T> &a) -> tf::ts::wasm_ndarray<T> {
  return tf::ts::log2(a);
}
template <typename T>
static auto sync_log2_inplace(tf::ts::wasm_ndarray<T> &a) -> void {
  tf::ts::log2_inplace(a);
}
template <typename T>
static auto sync_log10(tf::ts::wasm_ndarray<T> &a) -> tf::ts::wasm_ndarray<T> {
  return tf::ts::log10(a);
}
template <typename T>
static auto sync_log10_inplace(tf::ts::wasm_ndarray<T> &a) -> void {
  tf::ts::log10_inplace(a);
}
template <typename T>
static auto sync_floor(tf::ts::wasm_ndarray<T> &a) -> tf::ts::wasm_ndarray<T> {
  return tf::ts::floor(a);
}
template <typename T>
static auto sync_floor_inplace(tf::ts::wasm_ndarray<T> &a) -> void {
  tf::ts::floor_inplace(a);
}
template <typename T>
static auto sync_ceil(tf::ts::wasm_ndarray<T> &a) -> tf::ts::wasm_ndarray<T> {
  return tf::ts::ceil(a);
}
template <typename T>
static auto sync_ceil_inplace(tf::ts::wasm_ndarray<T> &a) -> void {
  tf::ts::ceil_inplace(a);
}
template <typename T>
static auto sync_round(tf::ts::wasm_ndarray<T> &a) -> tf::ts::wasm_ndarray<T> {
  return tf::ts::round(a);
}
template <typename T>
static auto sync_round_inplace(tf::ts::wasm_ndarray<T> &a) -> void {
  tf::ts::round_inplace(a);
}
template <typename T>
static auto sync_pow(tf::ts::wasm_ndarray<T> &a, T exponent)
    -> tf::ts::wasm_ndarray<T> {
  return tf::ts::pow(a, exponent);
}
template <typename T>
static auto sync_pow_inplace(tf::ts::wasm_ndarray<T> &a, T exponent) -> void {
  tf::ts::pow_inplace(a, exponent);
}

// ============================================================================
// Async — math functions
// ============================================================================

template <typename T>
static auto async_sin(tf::ts::wasm_ndarray<T> &a) -> tf::ts::promise_t {
  return tf::ts::promise([a]() { return tf::ts::sin(a); });
}
template <typename T>
static auto async_sin_inplace(tf::ts::wasm_ndarray<T> &a)
    -> tf::ts::promise_t {
  return tf::ts::promise([a]() {
    tf::ts::sin_inplace(const_cast<tf::ts::wasm_ndarray<T> &>(a));
    return a;
  });
}
template <typename T>
static auto async_cos(tf::ts::wasm_ndarray<T> &a) -> tf::ts::promise_t {
  return tf::ts::promise([a]() { return tf::ts::cos(a); });
}
template <typename T>
static auto async_cos_inplace(tf::ts::wasm_ndarray<T> &a)
    -> tf::ts::promise_t {
  return tf::ts::promise([a]() {
    tf::ts::cos_inplace(const_cast<tf::ts::wasm_ndarray<T> &>(a));
    return a;
  });
}
template <typename T>
static auto async_tan(tf::ts::wasm_ndarray<T> &a) -> tf::ts::promise_t {
  return tf::ts::promise([a]() { return tf::ts::tan(a); });
}
template <typename T>
static auto async_tan_inplace(tf::ts::wasm_ndarray<T> &a)
    -> tf::ts::promise_t {
  return tf::ts::promise([a]() {
    tf::ts::tan_inplace(const_cast<tf::ts::wasm_ndarray<T> &>(a));
    return a;
  });
}
template <typename T>
static auto async_asin(tf::ts::wasm_ndarray<T> &a) -> tf::ts::promise_t {
  return tf::ts::promise([a]() { return tf::ts::asin(a); });
}
template <typename T>
static auto async_asin_inplace(tf::ts::wasm_ndarray<T> &a)
    -> tf::ts::promise_t {
  return tf::ts::promise([a]() {
    tf::ts::asin_inplace(const_cast<tf::ts::wasm_ndarray<T> &>(a));
    return a;
  });
}
template <typename T>
static auto async_acos(tf::ts::wasm_ndarray<T> &a) -> tf::ts::promise_t {
  return tf::ts::promise([a]() { return tf::ts::acos(a); });
}
template <typename T>
static auto async_acos_inplace(tf::ts::wasm_ndarray<T> &a)
    -> tf::ts::promise_t {
  return tf::ts::promise([a]() {
    tf::ts::acos_inplace(const_cast<tf::ts::wasm_ndarray<T> &>(a));
    return a;
  });
}
template <typename T>
static auto async_atan(tf::ts::wasm_ndarray<T> &a) -> tf::ts::promise_t {
  return tf::ts::promise([a]() { return tf::ts::atan(a); });
}
template <typename T>
static auto async_atan_inplace(tf::ts::wasm_ndarray<T> &a)
    -> tf::ts::promise_t {
  return tf::ts::promise([a]() {
    tf::ts::atan_inplace(const_cast<tf::ts::wasm_ndarray<T> &>(a));
    return a;
  });
}
template <typename T>
static auto async_exp(tf::ts::wasm_ndarray<T> &a) -> tf::ts::promise_t {
  return tf::ts::promise([a]() { return tf::ts::exp(a); });
}
template <typename T>
static auto async_exp_inplace(tf::ts::wasm_ndarray<T> &a)
    -> tf::ts::promise_t {
  return tf::ts::promise([a]() {
    tf::ts::exp_inplace(const_cast<tf::ts::wasm_ndarray<T> &>(a));
    return a;
  });
}
template <typename T>
static auto async_log(tf::ts::wasm_ndarray<T> &a) -> tf::ts::promise_t {
  return tf::ts::promise([a]() { return tf::ts::log(a); });
}
template <typename T>
static auto async_log_inplace(tf::ts::wasm_ndarray<T> &a)
    -> tf::ts::promise_t {
  return tf::ts::promise([a]() {
    tf::ts::log_inplace(const_cast<tf::ts::wasm_ndarray<T> &>(a));
    return a;
  });
}
template <typename T>
static auto async_log2(tf::ts::wasm_ndarray<T> &a) -> tf::ts::promise_t {
  return tf::ts::promise([a]() { return tf::ts::log2(a); });
}
template <typename T>
static auto async_log2_inplace(tf::ts::wasm_ndarray<T> &a)
    -> tf::ts::promise_t {
  return tf::ts::promise([a]() {
    tf::ts::log2_inplace(const_cast<tf::ts::wasm_ndarray<T> &>(a));
    return a;
  });
}
template <typename T>
static auto async_log10(tf::ts::wasm_ndarray<T> &a) -> tf::ts::promise_t {
  return tf::ts::promise([a]() { return tf::ts::log10(a); });
}
template <typename T>
static auto async_log10_inplace(tf::ts::wasm_ndarray<T> &a)
    -> tf::ts::promise_t {
  return tf::ts::promise([a]() {
    tf::ts::log10_inplace(const_cast<tf::ts::wasm_ndarray<T> &>(a));
    return a;
  });
}
template <typename T>
static auto async_floor(tf::ts::wasm_ndarray<T> &a) -> tf::ts::promise_t {
  return tf::ts::promise([a]() { return tf::ts::floor(a); });
}
template <typename T>
static auto async_floor_inplace(tf::ts::wasm_ndarray<T> &a)
    -> tf::ts::promise_t {
  return tf::ts::promise([a]() {
    tf::ts::floor_inplace(const_cast<tf::ts::wasm_ndarray<T> &>(a));
    return a;
  });
}
template <typename T>
static auto async_ceil(tf::ts::wasm_ndarray<T> &a) -> tf::ts::promise_t {
  return tf::ts::promise([a]() { return tf::ts::ceil(a); });
}
template <typename T>
static auto async_ceil_inplace(tf::ts::wasm_ndarray<T> &a)
    -> tf::ts::promise_t {
  return tf::ts::promise([a]() {
    tf::ts::ceil_inplace(const_cast<tf::ts::wasm_ndarray<T> &>(a));
    return a;
  });
}
template <typename T>
static auto async_round(tf::ts::wasm_ndarray<T> &a) -> tf::ts::promise_t {
  return tf::ts::promise([a]() { return tf::ts::round(a); });
}
template <typename T>
static auto async_round_inplace(tf::ts::wasm_ndarray<T> &a)
    -> tf::ts::promise_t {
  return tf::ts::promise([a]() {
    tf::ts::round_inplace(const_cast<tf::ts::wasm_ndarray<T> &>(a));
    return a;
  });
}
template <typename T>
static auto async_pow(tf::ts::wasm_ndarray<T> &a, T exponent)
    -> tf::ts::promise_t {
  return tf::ts::promise(
      [a, exponent]() { return tf::ts::pow(a, exponent); });
}
template <typename T>
static auto async_pow_inplace(tf::ts::wasm_ndarray<T> &a, T exponent)
    -> tf::ts::promise_t {
  return tf::ts::promise([a, exponent]() {
    tf::ts::pow_inplace(const_cast<tf::ts::wasm_ndarray<T> &>(a), exponent);
    return a;
  });
}

// ============================================================================
// Sync — atan2 (broadcasts)
// ============================================================================

template <typename T>
static auto sync_atan2(tf::ts::wasm_ndarray<T> &y, tf::ts::wasm_ndarray<T> &x)
    -> tf::ts::wasm_ndarray<T> {
  return tf::ts::atan2(y, x);
}

// ============================================================================
// Async — atan2
// ============================================================================

template <typename T>
static auto async_atan2(tf::ts::wasm_ndarray<T> &y, tf::ts::wasm_ndarray<T> &x)
    -> tf::ts::promise_t {
  return tf::ts::promise([y, x]() { return tf::ts::atan2(y, x); });
}

// ============================================================================
// Sync — dot / cross
// ============================================================================

template <typename T>
static auto sync_dot(tf::ts::wasm_ndarray<T> &a, tf::ts::wasm_ndarray<T> &b)
    -> tf::ts::wasm_ndarray<T> {
  return tf::ts::dot(a, b);
}

template <typename T>
static auto sync_cross(tf::ts::wasm_ndarray<T> &a, tf::ts::wasm_ndarray<T> &b)
    -> tf::ts::wasm_ndarray<T> {
  return tf::ts::cross(a, b);
}

// ============================================================================
// Async — dot / cross
// ============================================================================

template <typename T>
static auto async_dot(tf::ts::wasm_ndarray<T> &a, tf::ts::wasm_ndarray<T> &b)
    -> tf::ts::promise_t {
  return tf::ts::promise(
      [a, b]() { return tf::ts::dot(a, b); });
}

template <typename T>
static auto async_cross(tf::ts::wasm_ndarray<T> &a, tf::ts::wasm_ndarray<T> &b)
    -> tf::ts::promise_t {
  return tf::ts::promise(
      [a, b]() { return tf::ts::cross(a, b); });
}

// ============================================================================
// Sync — logical ops (int8/bool)
// ============================================================================

static auto sync_logical_not(tf::ts::wasm_ndarray<std::int8_t> &a)
    -> tf::ts::wasm_ndarray<std::int8_t> {
  return tf::ts::logical_not(a);
}

static auto sync_logical_and(tf::ts::wasm_ndarray<std::int8_t> &a,
                              tf::ts::wasm_ndarray<std::int8_t> &b)
    -> tf::ts::wasm_ndarray<std::int8_t> {
  return tf::ts::logical_and(a, b);
}

static auto sync_logical_or(tf::ts::wasm_ndarray<std::int8_t> &a,
                             tf::ts::wasm_ndarray<std::int8_t> &b)
    -> tf::ts::wasm_ndarray<std::int8_t> {
  return tf::ts::logical_or(a, b);
}

// ============================================================================
// Async — logical ops
// ============================================================================

static auto async_logical_not(tf::ts::wasm_ndarray<std::int8_t> &a)
    -> tf::ts::promise_t {
  return tf::ts::promise([a]() { return tf::ts::logical_not(a); });
}

static auto async_logical_and(tf::ts::wasm_ndarray<std::int8_t> &a,
                               tf::ts::wasm_ndarray<std::int8_t> &b)
    -> tf::ts::promise_t {
  return tf::ts::promise([a, b]() { return tf::ts::logical_and(a, b); });
}

static auto async_logical_or(tf::ts::wasm_ndarray<std::int8_t> &a,
                              tf::ts::wasm_ndarray<std::int8_t> &b)
    -> tf::ts::promise_t {
  return tf::ts::promise([a, b]() { return tf::ts::logical_or(a, b); });
}

// ============================================================================
// Sync — relational (buffer × buffer)
// ============================================================================

template <typename T>
static auto sync_eq(tf::ts::wasm_ndarray<T> &a, tf::ts::wasm_ndarray<T> &b)
    -> tf::ts::wasm_ndarray<std::int8_t> {
  return tf::ts::eq(a, b);
}

template <typename T>
static auto sync_neq(tf::ts::wasm_ndarray<T> &a, tf::ts::wasm_ndarray<T> &b)
    -> tf::ts::wasm_ndarray<std::int8_t> {
  return tf::ts::neq(a, b);
}

template <typename T>
static auto sync_lt(tf::ts::wasm_ndarray<T> &a, tf::ts::wasm_ndarray<T> &b)
    -> tf::ts::wasm_ndarray<std::int8_t> {
  return tf::ts::lt(a, b);
}

template <typename T>
static auto sync_gt(tf::ts::wasm_ndarray<T> &a, tf::ts::wasm_ndarray<T> &b)
    -> tf::ts::wasm_ndarray<std::int8_t> {
  return tf::ts::gt(a, b);
}

template <typename T>
static auto sync_lte(tf::ts::wasm_ndarray<T> &a, tf::ts::wasm_ndarray<T> &b)
    -> tf::ts::wasm_ndarray<std::int8_t> {
  return tf::ts::lte(a, b);
}

template <typename T>
static auto sync_gte(tf::ts::wasm_ndarray<T> &a, tf::ts::wasm_ndarray<T> &b)
    -> tf::ts::wasm_ndarray<std::int8_t> {
  return tf::ts::gte(a, b);
}

// ============================================================================
// Sync — relational (buffer × scalar)
// ============================================================================

template <typename T>
static auto sync_eq_scalar(tf::ts::wasm_ndarray<T> &a, T scalar)
    -> tf::ts::wasm_ndarray<std::int8_t> {
  return tf::ts::eq_scalar(a, scalar);
}

template <typename T>
static auto sync_neq_scalar(tf::ts::wasm_ndarray<T> &a, T scalar)
    -> tf::ts::wasm_ndarray<std::int8_t> {
  return tf::ts::neq_scalar(a, scalar);
}

template <typename T>
static auto sync_lt_scalar(tf::ts::wasm_ndarray<T> &a, T scalar)
    -> tf::ts::wasm_ndarray<std::int8_t> {
  return tf::ts::lt_scalar(a, scalar);
}

template <typename T>
static auto sync_gt_scalar(tf::ts::wasm_ndarray<T> &a, T scalar)
    -> tf::ts::wasm_ndarray<std::int8_t> {
  return tf::ts::gt_scalar(a, scalar);
}

template <typename T>
static auto sync_lte_scalar(tf::ts::wasm_ndarray<T> &a, T scalar)
    -> tf::ts::wasm_ndarray<std::int8_t> {
  return tf::ts::lte_scalar(a, scalar);
}

template <typename T>
static auto sync_gte_scalar(tf::ts::wasm_ndarray<T> &a, T scalar)
    -> tf::ts::wasm_ndarray<std::int8_t> {
  return tf::ts::gte_scalar(a, scalar);
}

// ============================================================================
// Async — relational (buffer × buffer)
// ============================================================================

template <typename T>
static auto async_eq(tf::ts::wasm_ndarray<T> &a, tf::ts::wasm_ndarray<T> &b)
    -> tf::ts::promise_t {
  return tf::ts::promise(
      [a, b]() { return tf::ts::eq(a, b); });
}

template <typename T>
static auto async_neq(tf::ts::wasm_ndarray<T> &a, tf::ts::wasm_ndarray<T> &b)
    -> tf::ts::promise_t {
  return tf::ts::promise(
      [a, b]() { return tf::ts::neq(a, b); });
}

template <typename T>
static auto async_lt(tf::ts::wasm_ndarray<T> &a, tf::ts::wasm_ndarray<T> &b)
    -> tf::ts::promise_t {
  return tf::ts::promise(
      [a, b]() { return tf::ts::lt(a, b); });
}

template <typename T>
static auto async_gt(tf::ts::wasm_ndarray<T> &a, tf::ts::wasm_ndarray<T> &b)
    -> tf::ts::promise_t {
  return tf::ts::promise(
      [a, b]() { return tf::ts::gt(a, b); });
}

template <typename T>
static auto async_lte(tf::ts::wasm_ndarray<T> &a, tf::ts::wasm_ndarray<T> &b)
    -> tf::ts::promise_t {
  return tf::ts::promise(
      [a, b]() { return tf::ts::lte(a, b); });
}

template <typename T>
static auto async_gte(tf::ts::wasm_ndarray<T> &a, tf::ts::wasm_ndarray<T> &b)
    -> tf::ts::promise_t {
  return tf::ts::promise(
      [a, b]() { return tf::ts::gte(a, b); });
}

// ============================================================================
// Async — relational (buffer × scalar)
// ============================================================================

template <typename T>
static auto async_eq_scalar(tf::ts::wasm_ndarray<T> &a, T scalar)
    -> tf::ts::promise_t {
  return tf::ts::promise(
      [a, scalar]() { return tf::ts::eq_scalar(a, scalar); });
}

template <typename T>
static auto async_neq_scalar(tf::ts::wasm_ndarray<T> &a, T scalar)
    -> tf::ts::promise_t {
  return tf::ts::promise(
      [a, scalar]() { return tf::ts::neq_scalar(a, scalar); });
}

template <typename T>
static auto async_lt_scalar(tf::ts::wasm_ndarray<T> &a, T scalar)
    -> tf::ts::promise_t {
  return tf::ts::promise(
      [a, scalar]() { return tf::ts::lt_scalar(a, scalar); });
}

template <typename T>
static auto async_gt_scalar(tf::ts::wasm_ndarray<T> &a, T scalar)
    -> tf::ts::promise_t {
  return tf::ts::promise(
      [a, scalar]() { return tf::ts::gt_scalar(a, scalar); });
}

template <typename T>
static auto async_lte_scalar(tf::ts::wasm_ndarray<T> &a, T scalar)
    -> tf::ts::promise_t {
  return tf::ts::promise(
      [a, scalar]() { return tf::ts::lte_scalar(a, scalar); });
}

template <typename T>
static auto async_gte_scalar(tf::ts::wasm_ndarray<T> &a, T scalar)
    -> tf::ts::promise_t {
  return tf::ts::promise(
      [a, scalar]() { return tf::ts::gte_scalar(a, scalar); });
}

// ============================================================================
// Sync — type casting
// ============================================================================

template <typename From, typename To>
static auto sync_cast(tf::ts::wasm_ndarray<From> &a)
    -> tf::ts::wasm_ndarray<To> {
  return tf::ts::cast<From, To>(a);
}

// ============================================================================
// Embind
// ============================================================================

EMSCRIPTEN_BINDINGS(trueform_elementwise) {
  // --- add: buffer ---
  emscripten::function("add_float32", &sync_add<float>);
  emscripten::function("add_float64", &sync_add<double>);
  emscripten::function("add_int32", &sync_add<int>);
  emscripten::function("add_inplace_float32", &sync_add_inplace<float>);
  emscripten::function("add_inplace_float64", &sync_add_inplace<double>);
  emscripten::function("add_inplace_int32", &sync_add_inplace<int>);
  // --- add: scalar ---
  emscripten::function("add_scalar_float32", &sync_add_scalar<float>);
  emscripten::function("add_scalar_float64", &sync_add_scalar<double>);
  emscripten::function("add_scalar_int32", &sync_add_scalar<int>);
  emscripten::function("add_scalar_inplace_float32",
                       &sync_add_scalar_inplace<float>);
  emscripten::function("add_scalar_inplace_float64",
                       &sync_add_scalar_inplace<double>);
  emscripten::function("add_scalar_inplace_int32",
                       &sync_add_scalar_inplace<int>);

  // --- sub: buffer ---
  emscripten::function("sub_float32", &sync_sub<float>);
  emscripten::function("sub_float64", &sync_sub<double>);
  emscripten::function("sub_int32", &sync_sub<int>);
  emscripten::function("sub_inplace_float32", &sync_sub_inplace<float>);
  emscripten::function("sub_inplace_float64", &sync_sub_inplace<double>);
  emscripten::function("sub_inplace_int32", &sync_sub_inplace<int>);
  // --- sub: scalar ---
  emscripten::function("sub_scalar_float32", &sync_sub_scalar<float>);
  emscripten::function("sub_scalar_float64", &sync_sub_scalar<double>);
  emscripten::function("sub_scalar_int32", &sync_sub_scalar<int>);
  emscripten::function("sub_scalar_inplace_float32",
                       &sync_sub_scalar_inplace<float>);
  emscripten::function("sub_scalar_inplace_float64",
                       &sync_sub_scalar_inplace<double>);
  emscripten::function("sub_scalar_inplace_int32",
                       &sync_sub_scalar_inplace<int>);

  // --- mul: buffer ---
  emscripten::function("mul_float32", &sync_mul<float>);
  emscripten::function("mul_float64", &sync_mul<double>);
  emscripten::function("mul_int32", &sync_mul<int>);
  emscripten::function("mul_inplace_float32", &sync_mul_inplace<float>);
  emscripten::function("mul_inplace_float64", &sync_mul_inplace<double>);
  emscripten::function("mul_inplace_int32", &sync_mul_inplace<int>);
  // --- mul: scalar ---
  emscripten::function("mul_scalar_float32", &sync_mul_scalar<float>);
  emscripten::function("mul_scalar_float64", &sync_mul_scalar<double>);
  emscripten::function("mul_scalar_int32", &sync_mul_scalar<int>);
  emscripten::function("mul_scalar_inplace_float32",
                       &sync_mul_scalar_inplace<float>);
  emscripten::function("mul_scalar_inplace_float64",
                       &sync_mul_scalar_inplace<double>);
  emscripten::function("mul_scalar_inplace_int32",
                       &sync_mul_scalar_inplace<int>);

  // --- matMul ---
  emscripten::function("mat_mul_float32", &sync_mat_mul<float>);
  emscripten::function("mat_mul_float64", &sync_mat_mul<double>);
  emscripten::function("mat_mul_int32", &sync_mat_mul<int>);

  // --- inverted ---
  emscripten::function("inverted_float32", &sync_inverted<float>);
  emscripten::function("inverted_float64", &sync_inverted<double>);

  // --- Async: buffer × buffer (copy) ---
  emscripten::function("dispatch_add_float32", &async_add<float>);
  emscripten::function("dispatch_add_float64", &async_add<double>);
  emscripten::function("dispatch_add_int32", &async_add<int>);
  emscripten::function("dispatch_sub_float32", &async_sub<float>);
  emscripten::function("dispatch_sub_float64", &async_sub<double>);
  emscripten::function("dispatch_sub_int32", &async_sub<int>);
  emscripten::function("dispatch_mul_float32", &async_mul<float>);
  emscripten::function("dispatch_mul_float64", &async_mul<double>);
  emscripten::function("dispatch_mul_int32", &async_mul<int>);

  // --- Async: buffer × scalar (copy) ---
  emscripten::function("dispatch_add_scalar_float32",
                       &async_add_scalar<float>);
  emscripten::function("dispatch_add_scalar_float64",
                       &async_add_scalar<double>);
  emscripten::function("dispatch_add_scalar_int32", &async_add_scalar<int>);
  emscripten::function("dispatch_sub_scalar_float32",
                       &async_sub_scalar<float>);
  emscripten::function("dispatch_sub_scalar_float64",
                       &async_sub_scalar<double>);
  emscripten::function("dispatch_sub_scalar_int32", &async_sub_scalar<int>);
  emscripten::function("dispatch_mul_scalar_float32",
                       &async_mul_scalar<float>);
  emscripten::function("dispatch_mul_scalar_float64",
                       &async_mul_scalar<double>);
  emscripten::function("dispatch_mul_scalar_int32", &async_mul_scalar<int>);

  // --- Async: matMul ---
  emscripten::function("dispatch_mat_mul_float32", &async_mat_mul<float>);
  emscripten::function("dispatch_mat_mul_float64", &async_mat_mul<double>);
  emscripten::function("dispatch_mat_mul_int32", &async_mat_mul<int>);

  // --- Async: buffer × buffer (in-place) ---
  emscripten::function("dispatch_add_inplace_float32",
                       &async_add_inplace<float>);
  emscripten::function("dispatch_add_inplace_float64",
                       &async_add_inplace<double>);
  emscripten::function("dispatch_add_inplace_int32",
                       &async_add_inplace<int>);
  emscripten::function("dispatch_sub_inplace_float32",
                       &async_sub_inplace<float>);
  emscripten::function("dispatch_sub_inplace_float64",
                       &async_sub_inplace<double>);
  emscripten::function("dispatch_sub_inplace_int32",
                       &async_sub_inplace<int>);
  emscripten::function("dispatch_mul_inplace_float32",
                       &async_mul_inplace<float>);
  emscripten::function("dispatch_mul_inplace_float64",
                       &async_mul_inplace<double>);
  emscripten::function("dispatch_mul_inplace_int32",
                       &async_mul_inplace<int>);

  // --- Async: buffer × scalar (in-place) ---
  emscripten::function("dispatch_add_scalar_inplace_float32",
                       &async_add_scalar_inplace<float>);
  emscripten::function("dispatch_add_scalar_inplace_float64",
                       &async_add_scalar_inplace<double>);
  emscripten::function("dispatch_add_scalar_inplace_int32",
                       &async_add_scalar_inplace<int>);
  emscripten::function("dispatch_sub_scalar_inplace_float32",
                       &async_sub_scalar_inplace<float>);
  emscripten::function("dispatch_sub_scalar_inplace_float64",
                       &async_sub_scalar_inplace<double>);
  emscripten::function("dispatch_sub_scalar_inplace_int32",
                       &async_sub_scalar_inplace<int>);
  emscripten::function("dispatch_mul_scalar_inplace_float32",
                       &async_mul_scalar_inplace<float>);
  emscripten::function("dispatch_mul_scalar_inplace_float64",
                       &async_mul_scalar_inplace<double>);
  emscripten::function("dispatch_mul_scalar_inplace_int32",
                       &async_mul_scalar_inplace<int>);

  // --- div: buffer ---
  emscripten::function("div_float32", &sync_div<float>);
  emscripten::function("div_float64", &sync_div<double>);
  emscripten::function("div_int32", &sync_div<int>);
  emscripten::function("div_inplace_float32", &sync_div_inplace<float>);
  emscripten::function("div_inplace_float64", &sync_div_inplace<double>);
  emscripten::function("div_inplace_int32", &sync_div_inplace<int>);
  // --- div: scalar ---
  emscripten::function("div_scalar_float32", &sync_div_scalar<float>);
  emscripten::function("div_scalar_float64", &sync_div_scalar<double>);
  emscripten::function("div_scalar_int32", &sync_div_scalar<int>);
  emscripten::function("div_scalar_inplace_float32",
                       &sync_div_scalar_inplace<float>);
  emscripten::function("div_scalar_inplace_float64",
                       &sync_div_scalar_inplace<double>);
  emscripten::function("div_scalar_inplace_int32",
                       &sync_div_scalar_inplace<int>);

  // --- Async: div ---
  emscripten::function("dispatch_div_float32", &async_div<float>);
  emscripten::function("dispatch_div_float64", &async_div<double>);
  emscripten::function("dispatch_div_int32", &async_div<int>);
  emscripten::function("dispatch_div_scalar_float32",
                       &async_div_scalar<float>);
  emscripten::function("dispatch_div_scalar_float64",
                       &async_div_scalar<double>);
  emscripten::function("dispatch_div_scalar_int32", &async_div_scalar<int>);
  emscripten::function("dispatch_div_inplace_float32",
                       &async_div_inplace<float>);
  emscripten::function("dispatch_div_inplace_float64",
                       &async_div_inplace<double>);
  emscripten::function("dispatch_div_inplace_int32", &async_div_inplace<int>);
  emscripten::function("dispatch_div_scalar_inplace_float32",
                       &async_div_scalar_inplace<float>);
  emscripten::function("dispatch_div_scalar_inplace_float64",
                       &async_div_scalar_inplace<double>);
  emscripten::function("dispatch_div_scalar_inplace_int32",
                       &async_div_scalar_inplace<int>);

  // --- mod: buffer ---
  emscripten::function("mod_float32", &sync_mod<float>);
  emscripten::function("mod_float64", &sync_mod<double>);
  emscripten::function("mod_int32", &sync_mod<int>);
  emscripten::function("mod_inplace_float32", &sync_mod_inplace<float>);
  emscripten::function("mod_inplace_float64", &sync_mod_inplace<double>);
  emscripten::function("mod_inplace_int32", &sync_mod_inplace<int>);
  // --- mod: scalar ---
  emscripten::function("mod_scalar_float32", &sync_mod_scalar<float>);
  emscripten::function("mod_scalar_float64", &sync_mod_scalar<double>);
  emscripten::function("mod_scalar_int32", &sync_mod_scalar<int>);
  emscripten::function("mod_scalar_inplace_float32",
                       &sync_mod_scalar_inplace<float>);
  emscripten::function("mod_scalar_inplace_float64",
                       &sync_mod_scalar_inplace<double>);
  emscripten::function("mod_scalar_inplace_int32",
                       &sync_mod_scalar_inplace<int>);

  // --- Async: mod ---
  emscripten::function("dispatch_mod_float32", &async_mod<float>);
  emscripten::function("dispatch_mod_float64", &async_mod<double>);
  emscripten::function("dispatch_mod_int32", &async_mod<int>);
  emscripten::function("dispatch_mod_scalar_float32",
                       &async_mod_scalar<float>);
  emscripten::function("dispatch_mod_scalar_float64",
                       &async_mod_scalar<double>);
  emscripten::function("dispatch_mod_scalar_int32", &async_mod_scalar<int>);
  emscripten::function("dispatch_mod_inplace_float32",
                       &async_mod_inplace<float>);
  emscripten::function("dispatch_mod_inplace_float64",
                       &async_mod_inplace<double>);
  emscripten::function("dispatch_mod_inplace_int32", &async_mod_inplace<int>);
  emscripten::function("dispatch_mod_scalar_inplace_float32",
                       &async_mod_scalar_inplace<float>);
  emscripten::function("dispatch_mod_scalar_inplace_float64",
                       &async_mod_scalar_inplace<double>);
  emscripten::function("dispatch_mod_scalar_inplace_int32",
                       &async_mod_scalar_inplace<int>);

  // ==========================================================================
  // Unary ops
  // ==========================================================================

  // --- neg ---
  emscripten::function("neg_float32", &sync_neg<float>);
  emscripten::function("neg_float64", &sync_neg<double>);
  emscripten::function("neg_int32", &sync_neg<int>);
  emscripten::function("neg_int8", &sync_neg<std::int8_t>);
  emscripten::function("neg_inplace_float32", &sync_neg_inplace<float>);
  emscripten::function("neg_inplace_float64", &sync_neg_inplace<double>);
  emscripten::function("neg_inplace_int32", &sync_neg_inplace<int>);
  emscripten::function("neg_inplace_int8", &sync_neg_inplace<std::int8_t>);
  emscripten::function("dispatch_neg_float32", &async_neg<float>);
  emscripten::function("dispatch_neg_float64", &async_neg<double>);
  emscripten::function("dispatch_neg_int32", &async_neg<int>);
  emscripten::function("dispatch_neg_int8", &async_neg<std::int8_t>);

  // --- abs ---
  emscripten::function("abs_float32", &sync_abs<float>);
  emscripten::function("abs_float64", &sync_abs<double>);
  emscripten::function("abs_int32", &sync_abs<int>);
  emscripten::function("abs_int8", &sync_abs<std::int8_t>);
  emscripten::function("abs_inplace_float32", &sync_abs_inplace<float>);
  emscripten::function("abs_inplace_float64", &sync_abs_inplace<double>);
  emscripten::function("abs_inplace_int32", &sync_abs_inplace<int>);
  emscripten::function("abs_inplace_int8", &sync_abs_inplace<std::int8_t>);
  emscripten::function("dispatch_abs_float32", &async_abs<float>);
  emscripten::function("dispatch_abs_float64", &async_abs<double>);
  emscripten::function("dispatch_abs_int32", &async_abs<int>);
  emscripten::function("dispatch_abs_int8", &async_abs<std::int8_t>);

  // --- sqrt ---
  emscripten::function("sqrt_float32", &sync_sqrt<float>);
  emscripten::function("sqrt_float64", &sync_sqrt<double>);
  emscripten::function("sqrt_inplace_float32", &sync_sqrt_inplace<float>);
  emscripten::function("sqrt_inplace_float64", &sync_sqrt_inplace<double>);
  emscripten::function("dispatch_sqrt_float32", &async_sqrt<float>);
  emscripten::function("dispatch_sqrt_float64", &async_sqrt<double>);

  // --- clip ---
  emscripten::function("clip_float32", &sync_clip<float>);
  emscripten::function("clip_float64", &sync_clip<double>);
  emscripten::function("clip_int32", &sync_clip<int>);
  emscripten::function("clip_int8", &sync_clip<std::int8_t>);
  emscripten::function("clip_inplace_float32", &sync_clip_inplace<float>);
  emscripten::function("clip_inplace_float64", &sync_clip_inplace<double>);
  emscripten::function("clip_inplace_int32", &sync_clip_inplace<int>);
  emscripten::function("clip_inplace_int8", &sync_clip_inplace<std::int8_t>);
  emscripten::function("dispatch_clip_float32", &async_clip<float>);
  emscripten::function("dispatch_clip_float64", &async_clip<double>);
  emscripten::function("dispatch_clip_int32", &async_clip<int>);
  emscripten::function("dispatch_clip_int8", &async_clip<std::int8_t>);
  emscripten::function("dispatch_clip_inplace_float32",
                       &async_clip_inplace<float>);
  emscripten::function("dispatch_clip_inplace_float64",
                       &async_clip_inplace<double>);
  emscripten::function("dispatch_clip_inplace_int32",
                       &async_clip_inplace<int>);
  emscripten::function("dispatch_clip_inplace_int8",
                       &async_clip_inplace<std::int8_t>);

  // --- math functions ---
  emscripten::function("sin_float32", &sync_sin<float>);
  emscripten::function("sin_float64", &sync_sin<double>);
  emscripten::function("sin_inplace_float32", &sync_sin_inplace<float>);
  emscripten::function("sin_inplace_float64", &sync_sin_inplace<double>);
  emscripten::function("dispatch_sin_float32", &async_sin<float>);
  emscripten::function("dispatch_sin_float64", &async_sin<double>);
  emscripten::function("dispatch_sin_inplace_float32",
                       &async_sin_inplace<float>);
  emscripten::function("dispatch_sin_inplace_float64",
                       &async_sin_inplace<double>);

  emscripten::function("cos_float32", &sync_cos<float>);
  emscripten::function("cos_float64", &sync_cos<double>);
  emscripten::function("cos_inplace_float32", &sync_cos_inplace<float>);
  emscripten::function("cos_inplace_float64", &sync_cos_inplace<double>);
  emscripten::function("dispatch_cos_float32", &async_cos<float>);
  emscripten::function("dispatch_cos_float64", &async_cos<double>);
  emscripten::function("dispatch_cos_inplace_float32",
                       &async_cos_inplace<float>);
  emscripten::function("dispatch_cos_inplace_float64",
                       &async_cos_inplace<double>);

  emscripten::function("tan_float32", &sync_tan<float>);
  emscripten::function("tan_float64", &sync_tan<double>);
  emscripten::function("tan_inplace_float32", &sync_tan_inplace<float>);
  emscripten::function("tan_inplace_float64", &sync_tan_inplace<double>);
  emscripten::function("dispatch_tan_float32", &async_tan<float>);
  emscripten::function("dispatch_tan_float64", &async_tan<double>);
  emscripten::function("dispatch_tan_inplace_float32",
                       &async_tan_inplace<float>);
  emscripten::function("dispatch_tan_inplace_float64",
                       &async_tan_inplace<double>);

  emscripten::function("asin_float32", &sync_asin<float>);
  emscripten::function("asin_float64", &sync_asin<double>);
  emscripten::function("asin_inplace_float32", &sync_asin_inplace<float>);
  emscripten::function("asin_inplace_float64", &sync_asin_inplace<double>);
  emscripten::function("dispatch_asin_float32", &async_asin<float>);
  emscripten::function("dispatch_asin_float64", &async_asin<double>);
  emscripten::function("dispatch_asin_inplace_float32",
                       &async_asin_inplace<float>);
  emscripten::function("dispatch_asin_inplace_float64",
                       &async_asin_inplace<double>);

  emscripten::function("acos_float32", &sync_acos<float>);
  emscripten::function("acos_float64", &sync_acos<double>);
  emscripten::function("acos_inplace_float32", &sync_acos_inplace<float>);
  emscripten::function("acos_inplace_float64", &sync_acos_inplace<double>);
  emscripten::function("dispatch_acos_float32", &async_acos<float>);
  emscripten::function("dispatch_acos_float64", &async_acos<double>);
  emscripten::function("dispatch_acos_inplace_float32",
                       &async_acos_inplace<float>);
  emscripten::function("dispatch_acos_inplace_float64",
                       &async_acos_inplace<double>);

  emscripten::function("atan_float32", &sync_atan<float>);
  emscripten::function("atan_float64", &sync_atan<double>);
  emscripten::function("atan_inplace_float32", &sync_atan_inplace<float>);
  emscripten::function("atan_inplace_float64", &sync_atan_inplace<double>);
  emscripten::function("dispatch_atan_float32", &async_atan<float>);
  emscripten::function("dispatch_atan_float64", &async_atan<double>);
  emscripten::function("dispatch_atan_inplace_float32",
                       &async_atan_inplace<float>);
  emscripten::function("dispatch_atan_inplace_float64",
                       &async_atan_inplace<double>);

  emscripten::function("atan2_float32", &sync_atan2<float>);
  emscripten::function("atan2_float64", &sync_atan2<double>);
  emscripten::function("dispatch_atan2_float32", &async_atan2<float>);
  emscripten::function("dispatch_atan2_float64", &async_atan2<double>);

  emscripten::function("exp_float32", &sync_exp<float>);
  emscripten::function("exp_float64", &sync_exp<double>);
  emscripten::function("exp_inplace_float32", &sync_exp_inplace<float>);
  emscripten::function("exp_inplace_float64", &sync_exp_inplace<double>);
  emscripten::function("dispatch_exp_float32", &async_exp<float>);
  emscripten::function("dispatch_exp_float64", &async_exp<double>);
  emscripten::function("dispatch_exp_inplace_float32",
                       &async_exp_inplace<float>);
  emscripten::function("dispatch_exp_inplace_float64",
                       &async_exp_inplace<double>);

  emscripten::function("log_float32", &sync_log<float>);
  emscripten::function("log_float64", &sync_log<double>);
  emscripten::function("log_inplace_float32", &sync_log_inplace<float>);
  emscripten::function("log_inplace_float64", &sync_log_inplace<double>);
  emscripten::function("dispatch_log_float32", &async_log<float>);
  emscripten::function("dispatch_log_float64", &async_log<double>);
  emscripten::function("dispatch_log_inplace_float32",
                       &async_log_inplace<float>);
  emscripten::function("dispatch_log_inplace_float64",
                       &async_log_inplace<double>);

  emscripten::function("log2_float32", &sync_log2<float>);
  emscripten::function("log2_float64", &sync_log2<double>);
  emscripten::function("log2_inplace_float32", &sync_log2_inplace<float>);
  emscripten::function("log2_inplace_float64", &sync_log2_inplace<double>);
  emscripten::function("dispatch_log2_float32", &async_log2<float>);
  emscripten::function("dispatch_log2_float64", &async_log2<double>);
  emscripten::function("dispatch_log2_inplace_float32",
                       &async_log2_inplace<float>);
  emscripten::function("dispatch_log2_inplace_float64",
                       &async_log2_inplace<double>);

  emscripten::function("log10_float32", &sync_log10<float>);
  emscripten::function("log10_float64", &sync_log10<double>);
  emscripten::function("log10_inplace_float32", &sync_log10_inplace<float>);
  emscripten::function("log10_inplace_float64", &sync_log10_inplace<double>);
  emscripten::function("dispatch_log10_float32", &async_log10<float>);
  emscripten::function("dispatch_log10_float64", &async_log10<double>);
  emscripten::function("dispatch_log10_inplace_float32",
                       &async_log10_inplace<float>);
  emscripten::function("dispatch_log10_inplace_float64",
                       &async_log10_inplace<double>);

  emscripten::function("floor_float32", &sync_floor<float>);
  emscripten::function("floor_float64", &sync_floor<double>);
  emscripten::function("floor_inplace_float32", &sync_floor_inplace<float>);
  emscripten::function("floor_inplace_float64", &sync_floor_inplace<double>);
  emscripten::function("dispatch_floor_float32", &async_floor<float>);
  emscripten::function("dispatch_floor_float64", &async_floor<double>);
  emscripten::function("dispatch_floor_inplace_float32",
                       &async_floor_inplace<float>);
  emscripten::function("dispatch_floor_inplace_float64",
                       &async_floor_inplace<double>);

  emscripten::function("ceil_float32", &sync_ceil<float>);
  emscripten::function("ceil_float64", &sync_ceil<double>);
  emscripten::function("ceil_inplace_float32", &sync_ceil_inplace<float>);
  emscripten::function("ceil_inplace_float64", &sync_ceil_inplace<double>);
  emscripten::function("dispatch_ceil_float32", &async_ceil<float>);
  emscripten::function("dispatch_ceil_float64", &async_ceil<double>);
  emscripten::function("dispatch_ceil_inplace_float32",
                       &async_ceil_inplace<float>);
  emscripten::function("dispatch_ceil_inplace_float64",
                       &async_ceil_inplace<double>);

  emscripten::function("round_float32", &sync_round<float>);
  emscripten::function("round_float64", &sync_round<double>);
  emscripten::function("round_inplace_float32", &sync_round_inplace<float>);
  emscripten::function("round_inplace_float64", &sync_round_inplace<double>);
  emscripten::function("dispatch_round_float32", &async_round<float>);
  emscripten::function("dispatch_round_float64", &async_round<double>);
  emscripten::function("dispatch_round_inplace_float32",
                       &async_round_inplace<float>);
  emscripten::function("dispatch_round_inplace_float64",
                       &async_round_inplace<double>);

  emscripten::function("pow_float32", &sync_pow<float>);
  emscripten::function("pow_float64", &sync_pow<double>);
  emscripten::function("pow_inplace_float32", &sync_pow_inplace<float>);
  emscripten::function("pow_inplace_float64", &sync_pow_inplace<double>);
  emscripten::function("dispatch_pow_float32", &async_pow<float>);
  emscripten::function("dispatch_pow_float64", &async_pow<double>);
  emscripten::function("dispatch_pow_inplace_float32",
                       &async_pow_inplace<float>);
  emscripten::function("dispatch_pow_inplace_float64",
                       &async_pow_inplace<double>);

  // --- dot ---
  emscripten::function("dot_float32", &sync_dot<float>);
  emscripten::function("dot_float64", &sync_dot<double>);
  emscripten::function("dot_int32", &sync_dot<int>);
  emscripten::function("dispatch_dot_float32", &async_dot<float>);
  emscripten::function("dispatch_dot_float64", &async_dot<double>);
  emscripten::function("dispatch_dot_int32", &async_dot<int>);

  // --- cross ---
  emscripten::function("cross_float32", &sync_cross<float>);
  emscripten::function("cross_float64", &sync_cross<double>);
  emscripten::function("cross_int32", &sync_cross<int>);
  emscripten::function("dispatch_cross_float32", &async_cross<float>);
  emscripten::function("dispatch_cross_float64", &async_cross<double>);
  emscripten::function("dispatch_cross_int32", &async_cross<int>);

  // ==========================================================================
  // Logical ops (int8/bool)
  // ==========================================================================

  emscripten::function("not_int8", &sync_logical_not);
  emscripten::function("and_int8", &sync_logical_and);
  emscripten::function("or_int8", &sync_logical_or);
  emscripten::function("dispatch_not_int8", &async_logical_not);
  emscripten::function("dispatch_and_int8", &async_logical_and);
  emscripten::function("dispatch_or_int8", &async_logical_or);

  // ==========================================================================
  // int8 arithmetic
  // ==========================================================================

  // --- Sync: buffer ---
  emscripten::function("add_int8", &sync_add<std::int8_t>);
  emscripten::function("sub_int8", &sync_sub<std::int8_t>);
  emscripten::function("mul_int8", &sync_mul<std::int8_t>);
  emscripten::function("div_int8", &sync_div<std::int8_t>);
  emscripten::function("mod_int8", &sync_mod<std::int8_t>);
  emscripten::function("add_inplace_int8", &sync_add_inplace<std::int8_t>);
  emscripten::function("sub_inplace_int8", &sync_sub_inplace<std::int8_t>);
  emscripten::function("mul_inplace_int8", &sync_mul_inplace<std::int8_t>);
  emscripten::function("div_inplace_int8", &sync_div_inplace<std::int8_t>);
  emscripten::function("mod_inplace_int8", &sync_mod_inplace<std::int8_t>);
  // --- Sync: scalar ---
  emscripten::function("add_scalar_int8", &sync_add_scalar<std::int8_t>);
  emscripten::function("sub_scalar_int8", &sync_sub_scalar<std::int8_t>);
  emscripten::function("mul_scalar_int8", &sync_mul_scalar<std::int8_t>);
  emscripten::function("div_scalar_int8", &sync_div_scalar<std::int8_t>);
  emscripten::function("mod_scalar_int8", &sync_mod_scalar<std::int8_t>);
  emscripten::function("add_scalar_inplace_int8",
                       &sync_add_scalar_inplace<std::int8_t>);
  emscripten::function("sub_scalar_inplace_int8",
                       &sync_sub_scalar_inplace<std::int8_t>);
  emscripten::function("mul_scalar_inplace_int8",
                       &sync_mul_scalar_inplace<std::int8_t>);
  emscripten::function("div_scalar_inplace_int8",
                       &sync_div_scalar_inplace<std::int8_t>);
  emscripten::function("mod_scalar_inplace_int8",
                       &sync_mod_scalar_inplace<std::int8_t>);
  // --- Async: buffer ---
  emscripten::function("dispatch_add_int8", &async_add<std::int8_t>);
  emscripten::function("dispatch_sub_int8", &async_sub<std::int8_t>);
  emscripten::function("dispatch_mul_int8", &async_mul<std::int8_t>);
  emscripten::function("dispatch_div_int8", &async_div<std::int8_t>);
  emscripten::function("dispatch_mod_int8", &async_mod<std::int8_t>);
  emscripten::function("dispatch_add_inplace_int8",
                       &async_add_inplace<std::int8_t>);
  emscripten::function("dispatch_sub_inplace_int8",
                       &async_sub_inplace<std::int8_t>);
  emscripten::function("dispatch_mul_inplace_int8",
                       &async_mul_inplace<std::int8_t>);
  emscripten::function("dispatch_div_inplace_int8",
                       &async_div_inplace<std::int8_t>);
  emscripten::function("dispatch_mod_inplace_int8",
                       &async_mod_inplace<std::int8_t>);
  // --- Async: scalar ---
  emscripten::function("dispatch_add_scalar_int8",
                       &async_add_scalar<std::int8_t>);
  emscripten::function("dispatch_sub_scalar_int8",
                       &async_sub_scalar<std::int8_t>);
  emscripten::function("dispatch_mul_scalar_int8",
                       &async_mul_scalar<std::int8_t>);
  emscripten::function("dispatch_div_scalar_int8",
                       &async_div_scalar<std::int8_t>);
  emscripten::function("dispatch_mod_scalar_int8",
                       &async_mod_scalar<std::int8_t>);
  emscripten::function("dispatch_add_scalar_inplace_int8",
                       &async_add_scalar_inplace<std::int8_t>);
  emscripten::function("dispatch_sub_scalar_inplace_int8",
                       &async_sub_scalar_inplace<std::int8_t>);
  emscripten::function("dispatch_mul_scalar_inplace_int8",
                       &async_mul_scalar_inplace<std::int8_t>);
  emscripten::function("dispatch_div_scalar_inplace_int8",
                       &async_div_scalar_inplace<std::int8_t>);
  emscripten::function("dispatch_mod_scalar_inplace_int8",
                       &async_mod_scalar_inplace<std::int8_t>);

  // ==========================================================================
  // Relational ops — sync
  // ==========================================================================

  // --- eq ---
  emscripten::function("eq_float32", &sync_eq<float>);
  emscripten::function("eq_float64", &sync_eq<double>);
  emscripten::function("eq_int32", &sync_eq<int>);
  emscripten::function("eq_int8", &sync_eq<std::int8_t>);
  emscripten::function("eq_scalar_float32", &sync_eq_scalar<float>);
  emscripten::function("eq_scalar_float64", &sync_eq_scalar<double>);
  emscripten::function("eq_scalar_int32", &sync_eq_scalar<int>);
  emscripten::function("eq_scalar_int8", &sync_eq_scalar<std::int8_t>);
  // --- neq ---
  emscripten::function("neq_float32", &sync_neq<float>);
  emscripten::function("neq_float64", &sync_neq<double>);
  emscripten::function("neq_int32", &sync_neq<int>);
  emscripten::function("neq_int8", &sync_neq<std::int8_t>);
  emscripten::function("neq_scalar_float32", &sync_neq_scalar<float>);
  emscripten::function("neq_scalar_float64", &sync_neq_scalar<double>);
  emscripten::function("neq_scalar_int32", &sync_neq_scalar<int>);
  emscripten::function("neq_scalar_int8", &sync_neq_scalar<std::int8_t>);
  // --- lt ---
  emscripten::function("lt_float32", &sync_lt<float>);
  emscripten::function("lt_float64", &sync_lt<double>);
  emscripten::function("lt_int32", &sync_lt<int>);
  emscripten::function("lt_int8", &sync_lt<std::int8_t>);
  emscripten::function("lt_scalar_float32", &sync_lt_scalar<float>);
  emscripten::function("lt_scalar_float64", &sync_lt_scalar<double>);
  emscripten::function("lt_scalar_int32", &sync_lt_scalar<int>);
  emscripten::function("lt_scalar_int8", &sync_lt_scalar<std::int8_t>);
  // --- gt ---
  emscripten::function("gt_float32", &sync_gt<float>);
  emscripten::function("gt_float64", &sync_gt<double>);
  emscripten::function("gt_int32", &sync_gt<int>);
  emscripten::function("gt_int8", &sync_gt<std::int8_t>);
  emscripten::function("gt_scalar_float32", &sync_gt_scalar<float>);
  emscripten::function("gt_scalar_float64", &sync_gt_scalar<double>);
  emscripten::function("gt_scalar_int32", &sync_gt_scalar<int>);
  emscripten::function("gt_scalar_int8", &sync_gt_scalar<std::int8_t>);
  // --- lte ---
  emscripten::function("lte_float32", &sync_lte<float>);
  emscripten::function("lte_float64", &sync_lte<double>);
  emscripten::function("lte_int32", &sync_lte<int>);
  emscripten::function("lte_int8", &sync_lte<std::int8_t>);
  emscripten::function("lte_scalar_float32", &sync_lte_scalar<float>);
  emscripten::function("lte_scalar_float64", &sync_lte_scalar<double>);
  emscripten::function("lte_scalar_int32", &sync_lte_scalar<int>);
  emscripten::function("lte_scalar_int8", &sync_lte_scalar<std::int8_t>);
  // --- gte ---
  emscripten::function("gte_float32", &sync_gte<float>);
  emscripten::function("gte_float64", &sync_gte<double>);
  emscripten::function("gte_int32", &sync_gte<int>);
  emscripten::function("gte_int8", &sync_gte<std::int8_t>);
  emscripten::function("gte_scalar_float32", &sync_gte_scalar<float>);
  emscripten::function("gte_scalar_float64", &sync_gte_scalar<double>);
  emscripten::function("gte_scalar_int32", &sync_gte_scalar<int>);
  emscripten::function("gte_scalar_int8", &sync_gte_scalar<std::int8_t>);

  // ==========================================================================
  // Relational ops — async
  // ==========================================================================

  // --- eq ---
  emscripten::function("dispatch_eq_float32", &async_eq<float>);
  emscripten::function("dispatch_eq_float64", &async_eq<double>);
  emscripten::function("dispatch_eq_int32", &async_eq<int>);
  emscripten::function("dispatch_eq_int8", &async_eq<std::int8_t>);
  emscripten::function("dispatch_eq_scalar_float32", &async_eq_scalar<float>);
  emscripten::function("dispatch_eq_scalar_float64",
                       &async_eq_scalar<double>);
  emscripten::function("dispatch_eq_scalar_int32", &async_eq_scalar<int>);
  emscripten::function("dispatch_eq_scalar_int8",
                       &async_eq_scalar<std::int8_t>);
  // --- neq ---
  emscripten::function("dispatch_neq_float32", &async_neq<float>);
  emscripten::function("dispatch_neq_float64", &async_neq<double>);
  emscripten::function("dispatch_neq_int32", &async_neq<int>);
  emscripten::function("dispatch_neq_int8", &async_neq<std::int8_t>);
  emscripten::function("dispatch_neq_scalar_float32",
                       &async_neq_scalar<float>);
  emscripten::function("dispatch_neq_scalar_float64",
                       &async_neq_scalar<double>);
  emscripten::function("dispatch_neq_scalar_int32", &async_neq_scalar<int>);
  emscripten::function("dispatch_neq_scalar_int8",
                       &async_neq_scalar<std::int8_t>);
  // --- lt ---
  emscripten::function("dispatch_lt_float32", &async_lt<float>);
  emscripten::function("dispatch_lt_float64", &async_lt<double>);
  emscripten::function("dispatch_lt_int32", &async_lt<int>);
  emscripten::function("dispatch_lt_int8", &async_lt<std::int8_t>);
  emscripten::function("dispatch_lt_scalar_float32", &async_lt_scalar<float>);
  emscripten::function("dispatch_lt_scalar_float64",
                       &async_lt_scalar<double>);
  emscripten::function("dispatch_lt_scalar_int32", &async_lt_scalar<int>);
  emscripten::function("dispatch_lt_scalar_int8",
                       &async_lt_scalar<std::int8_t>);
  // --- gt ---
  emscripten::function("dispatch_gt_float32", &async_gt<float>);
  emscripten::function("dispatch_gt_float64", &async_gt<double>);
  emscripten::function("dispatch_gt_int32", &async_gt<int>);
  emscripten::function("dispatch_gt_int8", &async_gt<std::int8_t>);
  emscripten::function("dispatch_gt_scalar_float32", &async_gt_scalar<float>);
  emscripten::function("dispatch_gt_scalar_float64",
                       &async_gt_scalar<double>);
  emscripten::function("dispatch_gt_scalar_int32", &async_gt_scalar<int>);
  emscripten::function("dispatch_gt_scalar_int8",
                       &async_gt_scalar<std::int8_t>);
  // --- lte ---
  emscripten::function("dispatch_lte_float32", &async_lte<float>);
  emscripten::function("dispatch_lte_float64", &async_lte<double>);
  emscripten::function("dispatch_lte_int32", &async_lte<int>);
  emscripten::function("dispatch_lte_int8", &async_lte<std::int8_t>);
  emscripten::function("dispatch_lte_scalar_float32",
                       &async_lte_scalar<float>);
  emscripten::function("dispatch_lte_scalar_float64",
                       &async_lte_scalar<double>);
  emscripten::function("dispatch_lte_scalar_int32", &async_lte_scalar<int>);
  emscripten::function("dispatch_lte_scalar_int8",
                       &async_lte_scalar<std::int8_t>);
  // --- gte ---
  emscripten::function("dispatch_gte_float32", &async_gte<float>);
  emscripten::function("dispatch_gte_float64", &async_gte<double>);
  emscripten::function("dispatch_gte_int32", &async_gte<int>);
  emscripten::function("dispatch_gte_int8", &async_gte<std::int8_t>);
  emscripten::function("dispatch_gte_scalar_float32",
                       &async_gte_scalar<float>);
  emscripten::function("dispatch_gte_scalar_float64",
                       &async_gte_scalar<double>);
  emscripten::function("dispatch_gte_scalar_int32", &async_gte_scalar<int>);
  emscripten::function("dispatch_gte_scalar_int8",
                       &async_gte_scalar<std::int8_t>);

  // ==========================================================================
  // Type casting
  // ==========================================================================

  emscripten::function("cast_int8_to_int32",
                       &sync_cast<std::int8_t, int>);
  emscripten::function("cast_int8_to_float32",
                       &sync_cast<std::int8_t, float>);
  emscripten::function("cast_int8_to_float64",
                       &sync_cast<std::int8_t, double>);
  emscripten::function("cast_int32_to_float32",
                       &sync_cast<int, float>);
  emscripten::function("cast_int32_to_int8",
                       &sync_cast<int, std::int8_t>);
  emscripten::function("cast_int32_to_float64",
                       &sync_cast<int, double>);
  emscripten::function("cast_float32_to_int32",
                       &sync_cast<float, int>);
  emscripten::function("cast_float32_to_int8",
                       &sync_cast<float, std::int8_t>);
  emscripten::function("cast_float32_to_float64",
                       &sync_cast<float, double>);
  emscripten::function("cast_float64_to_int8",
                       &sync_cast<double, std::int8_t>);
  emscripten::function("cast_float64_to_int32",
                       &sync_cast<double, int>);
  emscripten::function("cast_float64_to_float32",
                       &sync_cast<double, float>);

  // ==========================================================================
  // Assign
  // ==========================================================================

  // --- assign_scalar ---
  emscripten::function("assign_scalar_float32", &sync_assign_scalar<float>);
  emscripten::function("assign_scalar_float64", &sync_assign_scalar<double>);
  emscripten::function("assign_scalar_int32", &sync_assign_scalar<int>);
  emscripten::function("assign_scalar_int8",
                       &sync_assign_scalar<std::int8_t>);

  // --- assign_array ---
  emscripten::function("assign_array_float32", &sync_assign_array<float>);
  emscripten::function("assign_array_float64", &sync_assign_array<double>);
  emscripten::function("assign_array_int32", &sync_assign_array<int>);
  emscripten::function("assign_array_int8",
                       &sync_assign_array<std::int8_t>);

  // --- assign_indexed_scalar ---
  emscripten::function("assign_indexed_scalar_float32",
                       &sync_assign_indexed_scalar<float>);
  emscripten::function("assign_indexed_scalar_float64",
                       &sync_assign_indexed_scalar<double>);
  emscripten::function("assign_indexed_scalar_int32",
                       &sync_assign_indexed_scalar<int>);
  emscripten::function("assign_indexed_scalar_int8",
                       &sync_assign_indexed_scalar<std::int8_t>);

  // --- assign_indexed_array ---
  emscripten::function("assign_indexed_array_float32",
                       &sync_assign_indexed_array<float>);
  emscripten::function("assign_indexed_array_float64",
                       &sync_assign_indexed_array<double>);
  emscripten::function("assign_indexed_array_int32",
                       &sync_assign_indexed_array<int>);
  emscripten::function("assign_indexed_array_int8",
                       &sync_assign_indexed_array<std::int8_t>);

  // --- assign_masked_scalar ---
  emscripten::function("assign_masked_scalar_float32",
                       &sync_assign_masked_scalar<float>);
  emscripten::function("assign_masked_scalar_float64",
                       &sync_assign_masked_scalar<double>);
  emscripten::function("assign_masked_scalar_int32",
                       &sync_assign_masked_scalar<int>);
  emscripten::function("assign_masked_scalar_int8",
                       &sync_assign_masked_scalar<std::int8_t>);

  // --- assign_masked_array ---
  emscripten::function("assign_masked_array_float32",
                       &sync_assign_masked_array<float>);
  emscripten::function("assign_masked_array_float64",
                       &sync_assign_masked_array<double>);
  emscripten::function("assign_masked_array_int32",
                       &sync_assign_masked_array<int>);
  emscripten::function("assign_masked_array_int8",
                       &sync_assign_masked_array<std::int8_t>);
}
