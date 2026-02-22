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

#include "trueform/core/algorithm/parallel_for_each.hpp"
#include "trueform/core/closest_metric_point_pair.hpp"
#include "trueform/core/views/sequence_range.hpp"
#include "trueform/ts/core/promise.hpp"
#include "trueform/ts/core/wasm_ndarray.hpp"
#include "trueform/ts/spatial/prim_dispatch.hpp"
#include "trueform/ts/spatial/result_types.hpp"
#include <emscripten/bind.h>

namespace {

using namespace tf::ts;

// ============================================================================
// PP helpers (prim × prim)
// ============================================================================

auto pp_single(const float *a, prim_type ta, const float *b, prim_type tb,
               int va, int vb) -> metric_point_pair_result {
  return dispatch_pair(
      [](const auto &pa, const auto &pb) -> metric_point_pair_result {
        auto r = tf::closest_metric_point_pair(pa, pb);
        return {point_to_ndarray(r.first), point_to_ndarray(r.second),
                r.metric};
      },
      a, ta, b, tb, va, vb);
}

auto pp_batch(wasm_ndarray<float> &a, prim_type ta, wasm_ndarray<float> &b,
              prim_type tb) -> metric_point_pair_batch_result {
  int va = poly_verts(a, ta);
  int vb = poly_verts(b, tb);
  int sa = stride_of(a, ta);
  int sb = stride_of(b, tb);
  bool ba = is_batch(a, ta);
  bool bb = is_batch(b, tb);
  int n = ba ? batch_count(a, ta) : batch_count(b, tb);
  int stride_a = ba ? sa : 0;
  int stride_b = bb ? sb : 0;

  const float *pa = a.raw_data();
  const float *pb = b.raw_data();

  tf::buffer<float> pts0;
  pts0.allocate(n * 3);
  tf::buffer<float> pts1;
  pts1.allocate(n * 3);
  tf::buffer<float> dists;
  dists.allocate(n);
  float *pts0_dst = pts0.data();
  float *pts1_dst = pts1.data();
  float *dist_dst = dists.data();

  auto compute = [=](int i) {
    auto r = dispatch_pair(
        [](const auto &a, const auto &b) {
          return tf::closest_metric_point_pair(a, b);
        },
        pa + i * stride_a, ta, pb + i * stride_b, tb, va, vb);
    dist_dst[i] = r.metric;
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

  return {wasm_ndarray<float>::from_buffer(std::move(pts0), {n, 3}),
          wasm_ndarray<float>::from_buffer(std::move(pts1), {n, 3}),
          wasm_ndarray<float>::from_buffer(std::move(dists), {n})};
}

// ============================================================================
// Sync entry point
// ============================================================================

auto sync_closest_metric_point_pair(wasm_ndarray<float> &a, int ta_int,
                                    wasm_ndarray<float> &b,
                                    int tb_int) -> emscripten::val {
  auto ta = static_cast<prim_type>(ta_int);
  auto tb = static_cast<prim_type>(tb_int);
  if (!is_batch(a, ta) && !is_batch(b, tb))
    return emscripten::val(pp_single(a.raw_data(), ta, b.raw_data(), tb,
                                     poly_verts(a, ta), poly_verts(b, tb)));
  return emscripten::val(pp_batch(a, ta, b, tb));
}

// ============================================================================
// Async entry point
// ============================================================================

auto async_closest_metric_point_pair(wasm_ndarray<float> &a, int ta_int,
                                     wasm_ndarray<float> &b,
                                     int tb_int) -> promise_t {
  auto ta = static_cast<prim_type>(ta_int);
  auto tb = static_cast<prim_type>(tb_int);

  if (!is_batch(a, ta) && !is_batch(b, tb)) {
    int va = poly_verts(a, ta);
    int vb = poly_verts(b, tb);
    return promise(
        [a = a, b = b, ta, tb, va, vb]() -> metric_point_pair_result {
          return pp_single(
              const_cast<wasm_ndarray<float> &>(a).raw_data(), ta,
              const_cast<wasm_ndarray<float> &>(b).raw_data(), tb, va, vb);
        });
  }

  return promise(
      [a = a, b = b, ta, tb]() -> metric_point_pair_batch_result {
        return pp_batch(const_cast<wasm_ndarray<float> &>(a), ta,
                        const_cast<wasm_ndarray<float> &>(b), tb);
      });
}

} // namespace

EMSCRIPTEN_BINDINGS(trueform_closest_metric_point_pair) {
  emscripten::value_object<tf::ts::metric_point_pair_result>(
      "MetricPointPairResult")
      .field("point0", &tf::ts::metric_point_pair_result::point0)
      .field("point1", &tf::ts::metric_point_pair_result::point1)
      .field("distance2", &tf::ts::metric_point_pair_result::distance2);

  emscripten::value_object<tf::ts::metric_point_pair_batch_result>(
      "MetricPointPairBatchResult")
      .field("points0", &tf::ts::metric_point_pair_batch_result::points0)
      .field("points1", &tf::ts::metric_point_pair_batch_result::points1)
      .field("distances", &tf::ts::metric_point_pair_batch_result::distances);

  emscripten::function("closest_metric_point_pair",
                       &sync_closest_metric_point_pair);
  emscripten::function("dispatch_closest_metric_point_pair",
                       &async_closest_metric_point_pair);
}
