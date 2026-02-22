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
#include "trueform/core/views/sequence_range.hpp"
#include "trueform/spatial/neighbor_search.hpp"
#include "trueform/ts/core/promise.hpp"
#include "trueform/ts/core/wasm_mesh.hpp"
#include "trueform/ts/core/wasm_ndarray.hpp"
#include "trueform/ts/spatial/prim_dispatch.hpp"
#include "trueform/ts/spatial/result_types.hpp"
#include <emscripten/bind.h>

namespace {

using namespace tf::ts;

// ============================================================================
// FP helpers (form × prim)
// ============================================================================

auto fp_single(wasm_mesh &m, const float *b, prim_type tb,
               int vb) -> neighbor_result {
  return m.with_form([&](const auto &form) -> neighbor_result {
    return dispatch_single(
        [&](const auto &pb) -> neighbor_result {
          auto r = tf::neighbor_search(form, pb);
          return {r.element, r.info.metric, point_to_ndarray(r.info.point)};
        },
        b, tb, vb);
  });
}

auto fp_batch(wasm_mesh &m, wasm_ndarray<float> &b,
              prim_type tb) -> neighbor_batch_result {
  int vb = poly_verts(b, tb);
  int sb = stride_of(b, tb);
  int n = batch_count(b, tb);
  const float *pb = b.raw_data();

  tf::buffer<int> ids;
  ids.allocate(n);
  tf::buffer<float> pts;
  pts.allocate(n * 3);
  tf::buffer<float> dists;
  dists.allocate(n);
  int *id_dst = ids.data();
  float *pts_dst = pts.data();
  float *dist_dst = dists.data();

  m.with_form([&](const auto &form) {
    auto compute = [&](int i) {
      auto r = dispatch_single(
          [&](const auto &prim) { return tf::neighbor_search(form, prim); },
          pb + i * sb, tb, vb);
      id_dst[i] = r.element;
      dist_dst[i] = r.info.metric;
      auto *p = pts_dst + i * 3;
      p[0] = r.info.point[0];
      p[1] = r.info.point[1];
      p[2] = r.info.point[2];
    };
    if (n >= 100)
      tf::parallel_for_each(tf::make_sequence_range(n), compute);
    else
      for (int i = 0; i < n; ++i)
        compute(i);
  });

  return {wasm_ndarray<int>::from_buffer(std::move(ids), {n}),
          wasm_ndarray<float>::from_buffer(std::move(pts), {n, 3}),
          wasm_ndarray<float>::from_buffer(std::move(dists), {n})};
}

// ============================================================================
// FF helper (form × form)
// ============================================================================

auto ff_compute(wasm_mesh &m0, wasm_mesh &m1) -> neighbor_result_pair {
  return m0.with_form([&](const auto &form0) -> neighbor_result_pair {
    return m1.with_form([&](const auto &form1) -> neighbor_result_pair {
      auto r = tf::neighbor_search(form0, form1);
      return {r.elements.first, r.elements.second, r.info.metric,
              point_to_ndarray(r.info.first),
              point_to_ndarray(r.info.second)};
    });
  });
}

// ============================================================================
// Sync entry points
// ============================================================================

auto sync_neighbor_search_fp(wasm_mesh &m, wasm_ndarray<float> &b,
                             int tb_int) -> emscripten::val {
  auto tb = static_cast<prim_type>(tb_int);
  if (!is_batch(b, tb))
    return emscripten::val(
        fp_single(m, b.raw_data(), tb, poly_verts(b, tb)));
  return emscripten::val(fp_batch(m, b, tb));
}

auto sync_neighbor_search_ff(wasm_mesh &m0,
                             wasm_mesh &m1) -> neighbor_result_pair {
  return ff_compute(m0, m1);
}

// ============================================================================
// Async entry points
// ============================================================================

auto async_neighbor_search_fp(wasm_mesh &m, wasm_ndarray<float> &b,
                              int tb_int) -> promise_t {
  auto tb = static_cast<prim_type>(tb_int);

  if (!is_batch(b, tb)) {
    int vb = poly_verts(b, tb);
    return promise([m = m, b = b, tb, vb]() -> neighbor_result {
      return fp_single(const_cast<wasm_mesh &>(m),
                       const_cast<wasm_ndarray<float> &>(b).raw_data(), tb, vb);
    });
  }

  return promise(
      [m = m, b = b, tb]() -> neighbor_batch_result {
        return fp_batch(const_cast<wasm_mesh &>(m),
                        const_cast<wasm_ndarray<float> &>(b), tb);
      });
}

auto async_neighbor_search_ff(wasm_mesh &m0, wasm_mesh &m1) -> promise_t {
  return promise([a = m0, b = m1]() -> neighbor_result_pair {
    return ff_compute(const_cast<wasm_mesh &>(a), const_cast<wasm_mesh &>(b));
  });
}

} // namespace

EMSCRIPTEN_BINDINGS(trueform_neighbor_search) {
  emscripten::value_object<tf::ts::neighbor_result>("NeighborResult")
      .field("elementId", &tf::ts::neighbor_result::element_id)
      .field("distance2", &tf::ts::neighbor_result::distance2)
      .field("point", &tf::ts::neighbor_result::point);

  emscripten::value_object<tf::ts::neighbor_batch_result>(
      "NeighborBatchResult")
      .field("elementIds", &tf::ts::neighbor_batch_result::element_ids)
      .field("points", &tf::ts::neighbor_batch_result::points)
      .field("distances", &tf::ts::neighbor_batch_result::distances);

  emscripten::value_object<tf::ts::neighbor_result_pair>("NeighborResultPair")
      .field("elementId0", &tf::ts::neighbor_result_pair::element_id0)
      .field("elementId1", &tf::ts::neighbor_result_pair::element_id1)
      .field("distance2", &tf::ts::neighbor_result_pair::distance2)
      .field("point0", &tf::ts::neighbor_result_pair::point0)
      .field("point1", &tf::ts::neighbor_result_pair::point1);

  emscripten::function("neighbor_search_fp", &sync_neighbor_search_fp);
  emscripten::function("neighbor_search_ff", &sync_neighbor_search_ff);
  emscripten::function("dispatch_neighbor_search_fp",
                       &async_neighbor_search_fp);
  emscripten::function("dispatch_neighbor_search_ff",
                       &async_neighbor_search_ff);
}
