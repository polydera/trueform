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

#include "trueform/ts/core/histogram_ops.hpp"
#include "trueform/ts/core/histogram_result.hpp"
#include "trueform/ts/core/promise.hpp"
#include "trueform/ts/core/wasm_ndarray.hpp"
#include <algorithm>
#include <cmath>
#include <emscripten/bind.h>
#include <emscripten/val.h>
#include <optional>
#include <stdexcept>

namespace {

using namespace tf::ts;

// ============================================================================
// Binding-layer helpers
// ============================================================================

// Resolve (lo, hi): NaN inputs mean "derive from data min/max".
auto resolve_range(wasm_ndarray<float> &x, float lo, float hi)
    -> std::pair<float, float> {
  bool auto_lo = std::isnan(lo);
  bool auto_hi = std::isnan(hi);
  if (auto_lo || auto_hi) {
    auto [mn, mx] = finite_min_max(x.make_range());
    if (!std::isfinite(mn) || !std::isfinite(mx)) {
      if (auto_lo) lo = 0.f;
      if (auto_hi) hi = 1.f;
    } else {
      if (auto_lo) lo = mn;
      if (auto_hi) hi = mx;
    }
  }
  if (!(hi > lo)) {
    lo -= 0.5f;
    hi += 0.5f;
  }
  return {lo, hi};
}

auto build_edges_ndarray(float lo, float hi, int num_bins)
    -> wasm_ndarray<float> {
  auto buf = make_linear_edges(lo, hi, num_bins);
  return wasm_ndarray<float>::from_buffer(std::move(buf), {num_bins + 1});
}

// In-place density normalization: counts[i] /= total * (edges[i+1] - edges[i]).
// Integration computed in double to reduce FP drift for many bins.
auto apply_density(tf::range<float *, tf::dynamic_size> counts,
                   tf::range<float *, tf::dynamic_size> edges) -> void {
  auto n = counts.size();
  double total = 0.0;
  for (std::size_t i = 0; i < n; ++i)
    total += counts[i];
  if (total <= 0.0)
    return;
  for (std::size_t i = 0; i < n; ++i) {
    double width = double{edges[i + 1]} - double{edges[i]};
    counts[i] = width > 0.0
                    ? static_cast<float>(double{counts[i]} / (total * width))
                    : 0.f;
  }
}

auto optional_weights(const emscripten::val &v)
    -> std::optional<wasm_ndarray<float>> {
  if (v.isNull() || v.isUndefined())
    return std::nullopt;
  return v.as<wasm_ndarray<float>>(emscripten::allow_raw_pointers());
}

// ============================================================================
// Core work (no emscripten::val — safe for sync and async).
// Each *_impl takes already-resolved inputs and does the actual counting.
// ============================================================================

auto bincount_int_impl(wasm_ndarray<int> &x, int min_length)
    -> wasm_ndarray<int> {
  if (min_length < 0)
    throw std::invalid_argument("bincount: minLength must be >= 0");
  int max_val = max_nonneg_or_throw(x.make_range());
  int num_bins = std::max(max_val + 1, min_length);
  tf::buffer<int> counts;
  counts.allocate_and_initialize(static_cast<std::size_t>(num_bins), 0);
  bincount(x.make_range(),
           tf::make_range(counts.data(), counts.size()));
  return wasm_ndarray<int>::from_buffer(std::move(counts), {num_bins});
}

auto bincount_weighted_impl(wasm_ndarray<int> &x, wasm_ndarray<float> &w,
                            int min_length) -> wasm_ndarray<float> {
  if (min_length < 0)
    throw std::invalid_argument("bincount: minLength must be >= 0");
  if (x.length() != w.length())
    throw std::invalid_argument(
        "bincount: weights must have the same length as x");
  int max_val = max_nonneg_or_throw(x.make_range());
  int num_bins = std::max(max_val + 1, min_length);
  tf::buffer<float> counts;
  counts.allocate_and_initialize(static_cast<std::size_t>(num_bins), 0.f);
  bincount(x.make_range(), w.make_range(),
           tf::make_range(counts.data(), counts.size()));
  return wasm_ndarray<float>::from_buffer(std::move(counts), {num_bins});
}

auto histogram_eq_int_impl(wasm_ndarray<float> &x, int num_bins, float lo,
                           float hi) -> histogram_result_int {
  if (num_bins <= 0)
    throw std::invalid_argument("histogram: bins must be > 0");
  auto [rlo, rhi] = resolve_range(x, lo, hi);
  tf::buffer<int> counts;
  counts.allocate_and_initialize(static_cast<std::size_t>(num_bins), 0);
  histogram_equal_width(x.make_range(), rlo, rhi,
                        tf::make_range(counts.data(), counts.size()));
  return {wasm_ndarray<int>::from_buffer(std::move(counts), {num_bins}),
          build_edges_ndarray(rlo, rhi, num_bins)};
}

auto histogram_eq_weighted_impl(wasm_ndarray<float> &x, wasm_ndarray<float> &w,
                                int num_bins, float lo, float hi)
    -> histogram_result_float {
  if (num_bins <= 0)
    throw std::invalid_argument("histogram: bins must be > 0");
  if (x.length() != w.length())
    throw std::invalid_argument(
        "histogram: weights must have the same length as x");
  auto [rlo, rhi] = resolve_range(x, lo, hi);
  tf::buffer<float> counts;
  counts.allocate_and_initialize(static_cast<std::size_t>(num_bins), 0.f);
  histogram_equal_width(x.make_range(), w.make_range(), rlo, rhi,
                        tf::make_range(counts.data(), counts.size()));
  return {wasm_ndarray<float>::from_buffer(std::move(counts), {num_bins}),
          build_edges_ndarray(rlo, rhi, num_bins)};
}

auto histogram_edges_int_impl(wasm_ndarray<float> &x,
                              wasm_ndarray<float> &edges)
    -> histogram_result_int {
  int num_bins = static_cast<int>(edges.length()) - 1;
  if (num_bins <= 0)
    throw std::invalid_argument("histogram: edges must have length >= 2");
  tf::buffer<int> counts;
  counts.allocate_and_initialize(static_cast<std::size_t>(num_bins), 0);
  histogram_edges(x.make_range(), edges.make_range(),
                  tf::make_range(counts.data(), counts.size()));
  return {wasm_ndarray<int>::from_buffer(std::move(counts), {num_bins}),
          edges.shallow_copy()};
}

auto histogram_edges_weighted_impl(wasm_ndarray<float> &x,
                                   wasm_ndarray<float> &w,
                                   wasm_ndarray<float> &edges)
    -> histogram_result_float {
  int num_bins = static_cast<int>(edges.length()) - 1;
  if (num_bins <= 0)
    throw std::invalid_argument("histogram: edges must have length >= 2");
  if (x.length() != w.length())
    throw std::invalid_argument(
        "histogram: weights must have the same length as x");
  tf::buffer<float> counts;
  counts.allocate_and_initialize(static_cast<std::size_t>(num_bins), 0.f);
  histogram_edges(x.make_range(), w.make_range(), edges.make_range(),
                  tf::make_range(counts.data(), counts.size()));
  return {wasm_ndarray<float>::from_buffer(std::move(counts), {num_bins}),
          edges.shallow_copy()};
}

// Density — weights already resolved by the caller (nullopt = no weights).
auto histogram_density_eq_impl(wasm_ndarray<float> &x,
                               std::optional<wasm_ndarray<float>> w_opt,
                               int num_bins, float lo, float hi)
    -> histogram_result_float {
  if (num_bins <= 0)
    throw std::invalid_argument("histogram: bins must be > 0");
  if (w_opt && x.length() != w_opt->length())
    throw std::invalid_argument(
        "histogram: weights must have the same length as x");
  auto [rlo, rhi] = resolve_range(x, lo, hi);
  tf::buffer<float> counts;
  counts.allocate_and_initialize(static_cast<std::size_t>(num_bins), 0.f);
  auto c_range = tf::make_range(counts.data(), counts.size());
  if (w_opt)
    histogram_equal_width(x.make_range(), w_opt->make_range(), rlo, rhi,
                          c_range);
  else
    histogram_equal_width(x.make_range(), rlo, rhi, c_range);
  auto edges_arr = build_edges_ndarray(rlo, rhi, num_bins);
  apply_density(c_range,
                tf::make_range(edges_arr.raw_data(), edges_arr.length()));
  return {wasm_ndarray<float>::from_buffer(std::move(counts), {num_bins}),
          std::move(edges_arr)};
}

auto histogram_density_edges_impl(wasm_ndarray<float> &x,
                                  std::optional<wasm_ndarray<float>> w_opt,
                                  wasm_ndarray<float> &edges)
    -> histogram_result_float {
  int num_bins = static_cast<int>(edges.length()) - 1;
  if (num_bins <= 0)
    throw std::invalid_argument("histogram: edges must have length >= 2");
  if (w_opt && x.length() != w_opt->length())
    throw std::invalid_argument(
        "histogram: weights must have the same length as x");
  tf::buffer<float> counts;
  counts.allocate_and_initialize(static_cast<std::size_t>(num_bins), 0.f);
  auto c_range = tf::make_range(counts.data(), counts.size());
  if (w_opt)
    histogram_edges(x.make_range(), w_opt->make_range(), edges.make_range(),
                    c_range);
  else
    histogram_edges(x.make_range(), edges.make_range(), c_range);
  apply_density(c_range, edges.make_range());
  return {wasm_ndarray<float>::from_buffer(std::move(counts), {num_bins}),
          edges.shallow_copy()};
}

// ============================================================================
// Sync entry points — thin wrappers around *_impl
// ============================================================================

auto sync_bincount_int32(wasm_ndarray<int> &x, int min_length)
    -> wasm_ndarray<int> {
  return bincount_int_impl(x, min_length);
}

auto sync_bincount_weighted_int32(wasm_ndarray<int> &x, wasm_ndarray<float> &w,
                                  int min_length) -> wasm_ndarray<float> {
  return bincount_weighted_impl(x, w, min_length);
}

auto sync_histogram_equal_width_float32(wasm_ndarray<float> &x, int num_bins,
                                        float lo, float hi)
    -> histogram_result_int {
  return histogram_eq_int_impl(x, num_bins, lo, hi);
}

auto sync_histogram_equal_width_weighted_float32(wasm_ndarray<float> &x,
                                                 wasm_ndarray<float> &w,
                                                 int num_bins, float lo,
                                                 float hi)
    -> histogram_result_float {
  return histogram_eq_weighted_impl(x, w, num_bins, lo, hi);
}

auto sync_histogram_edges_float32(wasm_ndarray<float> &x,
                                  wasm_ndarray<float> &edges)
    -> histogram_result_int {
  return histogram_edges_int_impl(x, edges);
}

auto sync_histogram_edges_weighted_float32(wasm_ndarray<float> &x,
                                           wasm_ndarray<float> &w,
                                           wasm_ndarray<float> &edges)
    -> histogram_result_float {
  return histogram_edges_weighted_impl(x, w, edges);
}

auto sync_histogram_density_equal_width_float32(wasm_ndarray<float> &x,
                                                int num_bins, float lo,
                                                float hi,
                                                emscripten::val weights_opt)
    -> histogram_result_float {
  return histogram_density_eq_impl(x, optional_weights(weights_opt), num_bins,
                                   lo, hi);
}

auto sync_histogram_density_edges_float32(wasm_ndarray<float> &x,
                                          wasm_ndarray<float> &edges,
                                          emscripten::val weights_opt)
    -> histogram_result_float {
  return histogram_density_edges_impl(x, optional_weights(weights_opt), edges);
}

// ============================================================================
// Async entry points — resolve emscripten::val on main thread, dispatch *_impl
// ============================================================================

auto async_bincount_int32(wasm_ndarray<int> &x, int min_length) -> promise_t {
  return promise([a = x, min_length]() -> wasm_ndarray<int> {
    return bincount_int_impl(const_cast<wasm_ndarray<int> &>(a), min_length);
  });
}

auto async_bincount_weighted_int32(wasm_ndarray<int> &x, wasm_ndarray<float> &w,
                                   int min_length) -> promise_t {
  return promise([a = x, b = w, min_length]() -> wasm_ndarray<float> {
    return bincount_weighted_impl(const_cast<wasm_ndarray<int> &>(a),
                                  const_cast<wasm_ndarray<float> &>(b),
                                  min_length);
  });
}

auto async_histogram_equal_width_float32(wasm_ndarray<float> &x, int num_bins,
                                         float lo, float hi) -> promise_t {
  return promise([a = x, num_bins, lo, hi]() -> histogram_result_int {
    return histogram_eq_int_impl(const_cast<wasm_ndarray<float> &>(a),
                                 num_bins, lo, hi);
  });
}

auto async_histogram_equal_width_weighted_float32(wasm_ndarray<float> &x,
                                                  wasm_ndarray<float> &w,
                                                  int num_bins, float lo,
                                                  float hi) -> promise_t {
  return promise(
      [a = x, b = w, num_bins, lo, hi]() -> histogram_result_float {
        return histogram_eq_weighted_impl(
            const_cast<wasm_ndarray<float> &>(a),
            const_cast<wasm_ndarray<float> &>(b), num_bins, lo, hi);
      });
}

auto async_histogram_edges_float32(wasm_ndarray<float> &x,
                                   wasm_ndarray<float> &edges) -> promise_t {
  return promise([a = x, e = edges]() -> histogram_result_int {
    return histogram_edges_int_impl(const_cast<wasm_ndarray<float> &>(a),
                                    const_cast<wasm_ndarray<float> &>(e));
  });
}

auto async_histogram_edges_weighted_float32(wasm_ndarray<float> &x,
                                            wasm_ndarray<float> &w,
                                            wasm_ndarray<float> &edges)
    -> promise_t {
  return promise([a = x, b = w, e = edges]() -> histogram_result_float {
    return histogram_edges_weighted_impl(
        const_cast<wasm_ndarray<float> &>(a),
        const_cast<wasm_ndarray<float> &>(b),
        const_cast<wasm_ndarray<float> &>(e));
  });
}

auto async_histogram_density_equal_width_float32(
    wasm_ndarray<float> &x, int num_bins, float lo, float hi,
    emscripten::val weights_opt) -> promise_t {
  auto w_opt = optional_weights(weights_opt);   // main thread
  return promise(
      [a = x, w_opt, num_bins, lo, hi]() -> histogram_result_float {
        return histogram_density_eq_impl(
            const_cast<wasm_ndarray<float> &>(a), w_opt, num_bins, lo, hi);
      });
}

auto async_histogram_density_edges_float32(wasm_ndarray<float> &x,
                                           wasm_ndarray<float> &edges,
                                           emscripten::val weights_opt)
    -> promise_t {
  auto w_opt = optional_weights(weights_opt);   // main thread
  return promise([a = x, w_opt, e = edges]() -> histogram_result_float {
    return histogram_density_edges_impl(
        const_cast<wasm_ndarray<float> &>(a), w_opt,
        const_cast<wasm_ndarray<float> &>(e));
  });
}

} // namespace

EMSCRIPTEN_BINDINGS(trueform_histogram) {
  emscripten::value_object<tf::ts::histogram_result_int>("HistogramResultInt")
      .field("counts", &tf::ts::histogram_result_int::counts)
      .field("edges", &tf::ts::histogram_result_int::edges);

  emscripten::value_object<tf::ts::histogram_result_float>(
      "HistogramResultFloat")
      .field("counts", &tf::ts::histogram_result_float::counts)
      .field("edges", &tf::ts::histogram_result_float::edges);

  // Bincount
  emscripten::function("bincount_int32", &sync_bincount_int32);
  emscripten::function("bincount_weighted_int32",
                       &sync_bincount_weighted_int32);
  emscripten::function("dispatch_bincount_int32", &async_bincount_int32);
  emscripten::function("dispatch_bincount_weighted_int32",
                       &async_bincount_weighted_int32);

  // Histogram — equal width
  emscripten::function("histogram_equal_width_float32",
                       &sync_histogram_equal_width_float32);
  emscripten::function("histogram_equal_width_weighted_float32",
                       &sync_histogram_equal_width_weighted_float32);
  emscripten::function("dispatch_histogram_equal_width_float32",
                       &async_histogram_equal_width_float32);
  emscripten::function("dispatch_histogram_equal_width_weighted_float32",
                       &async_histogram_equal_width_weighted_float32);

  // Histogram — explicit edges
  emscripten::function("histogram_edges_float32",
                       &sync_histogram_edges_float32);
  emscripten::function("histogram_edges_weighted_float32",
                       &sync_histogram_edges_weighted_float32);
  emscripten::function("dispatch_histogram_edges_float32",
                       &async_histogram_edges_float32);
  emscripten::function("dispatch_histogram_edges_weighted_float32",
                       &async_histogram_edges_weighted_float32);

  // Histogram — density
  emscripten::function("histogram_density_equal_width_float32",
                       &sync_histogram_density_equal_width_float32);
  emscripten::function("histogram_density_edges_float32",
                       &sync_histogram_density_edges_float32);
  emscripten::function("dispatch_histogram_density_equal_width_float32",
                       &async_histogram_density_equal_width_float32);
  emscripten::function("dispatch_histogram_density_edges_float32",
                       &async_histogram_density_edges_float32);
}
