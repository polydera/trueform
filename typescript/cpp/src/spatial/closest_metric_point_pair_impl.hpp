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
#include "trueform/core/closest_metric_point_pair.hpp"
#include "trueform/core/views/sequence_range.hpp"
#include "trueform/ts/core/promise.hpp"
#include "trueform/ts/core/wasm_ndarray.hpp"
#include "trueform/ts/spatial/prim_dispatch.hpp"
#include <emscripten/bind.h>

namespace tf {
namespace ts {

// ============================================================================
// Templated result types
// ============================================================================

template <typename Real> struct metric_point_pair_result_t {
  wasm_ndarray<Real> point0; // [3]
  wasm_ndarray<Real> point1; // [3]
  Real distance2;
};

template <typename Real> struct metric_point_pair_batch_result_t {
  wasm_ndarray<Real> points0;   // [N, 3]
  wasm_ndarray<Real> points1;   // [N, 3]
  wasm_ndarray<Real> distances; // [N]
};

// ============================================================================
// Helper: copy a tf::point<Real,3> into a fresh wasm_ndarray<Real> [3]
// ============================================================================

template <typename Real, typename PointLike>
inline auto cmpp_point_to_ndarray(const PointLike &pt) -> wasm_ndarray<Real> {
  tf::buffer<Real> buf;
  buf.allocate(3);
  auto *p = buf.data();
  p[0] = pt[0];
  p[1] = pt[1];
  p[2] = pt[2];
  return wasm_ndarray<Real>::from_buffer(std::move(buf), {3});
}

// ============================================================================
// PP helpers (prim × prim)
// ============================================================================

template <typename Real>
inline auto pp_single(const Real *a, prim_type ta, const Real *b, prim_type tb,
                      int va, int vb) -> metric_point_pair_result_t<Real> {
  return dispatch_pair(
      [](const auto &pa, const auto &pb) -> metric_point_pair_result_t<Real> {
        auto r = tf::closest_metric_point_pair(pa, pb);
        return {cmpp_point_to_ndarray<Real>(r.first),
                cmpp_point_to_ndarray<Real>(r.second),
                static_cast<Real>(r.metric)};
      },
      a, ta, b, tb, va, vb);
}

template <typename Real>
inline auto pp_batch(wasm_ndarray<Real> &a, prim_type ta, wasm_ndarray<Real> &b,
                     prim_type tb) -> metric_point_pair_batch_result_t<Real> {
  int va = poly_verts(a, ta);
  int vb = poly_verts(b, tb);
  int sa = stride_of(a, ta);
  int sb = stride_of(b, tb);
  bool ba = is_batch(a, ta);
  bool bb = is_batch(b, tb);
  int n = ba ? batch_count(a, ta) : batch_count(b, tb);
  int stride_a = ba ? sa : 0;
  int stride_b = bb ? sb : 0;

  const Real *pa = a.raw_data();
  const Real *pb = b.raw_data();

  tf::buffer<Real> pts0;
  pts0.allocate(n * 3);
  tf::buffer<Real> pts1;
  pts1.allocate(n * 3);
  tf::buffer<Real> dists;
  dists.allocate(n);
  Real *pts0_dst = pts0.data();
  Real *pts1_dst = pts1.data();
  Real *dist_dst = dists.data();

  auto compute = [=](int i) {
    auto r = dispatch_pair(
        [](const auto &a, const auto &b) {
          return tf::closest_metric_point_pair(a, b);
        },
        pa + i * stride_a, ta, pb + i * stride_b, tb, va, vb);
    dist_dst[i] = static_cast<Real>(r.metric);
    auto *p0 = pts0_dst + i * 3;
    p0[0] = r.first[0];
    p0[1] = r.first[1];
    p0[2] = r.first[2];
    auto *p1 = pts1_dst + i * 3;
    p1[0] = r.second[0];
    p1[1] = r.second[1];
    p1[2] = r.second[2];
  };

  if (n >= 1000)
    tf::parallel_for_each(tf::make_sequence_range(n), compute);
  else
    for (int i = 0; i < n; ++i)
      compute(i);

  return {wasm_ndarray<Real>::from_buffer(std::move(pts0), {n, 3}),
          wasm_ndarray<Real>::from_buffer(std::move(pts1), {n, 3}),
          wasm_ndarray<Real>::from_buffer(std::move(dists), {n})};
}

// ============================================================================
// Sync entry point
// ============================================================================

template <typename Real>
inline auto sync_closest_metric_point_pair(wasm_ndarray<Real> &a, int ta_int,
                                           wasm_ndarray<Real> &b, int tb_int)
    -> emscripten::val {
  auto ta = static_cast<prim_type>(ta_int);
  auto tb = static_cast<prim_type>(tb_int);
  if (!is_batch(a, ta) && !is_batch(b, tb))
    return emscripten::val(pp_single<Real>(a.raw_data(), ta, b.raw_data(), tb,
                                           poly_verts(a, ta),
                                           poly_verts(b, tb)));
  return emscripten::val(pp_batch<Real>(a, ta, b, tb));
}

// ============================================================================
// Async entry point
// ============================================================================

template <typename Real>
inline auto async_closest_metric_point_pair(wasm_ndarray<Real> &a, int ta_int,
                                            wasm_ndarray<Real> &b, int tb_int)
    -> promise_t {
  auto ta = static_cast<prim_type>(ta_int);
  auto tb = static_cast<prim_type>(tb_int);

  if (!is_batch(a, ta) && !is_batch(b, tb)) {
    int va = poly_verts(a, ta);
    int vb = poly_verts(b, tb);
    return promise(
        [a = a, b = b, ta, tb, va, vb]() -> metric_point_pair_result_t<Real> {
          return pp_single<Real>(
              const_cast<wasm_ndarray<Real> &>(a).raw_data(), ta,
              const_cast<wasm_ndarray<Real> &>(b).raw_data(), tb, va, vb);
        });
  }

  return promise(
      [a = a, b = b, ta, tb]() -> metric_point_pair_batch_result_t<Real> {
        return pp_batch<Real>(const_cast<wasm_ndarray<Real> &>(a), ta,
                              const_cast<wasm_ndarray<Real> &>(b), tb);
      });
}

} // namespace ts
} // namespace tf
