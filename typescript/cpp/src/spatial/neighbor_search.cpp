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

#include "trueform/core/algorithm/parallel_fill.hpp"
#include "trueform/core/algorithm/parallel_for_each.hpp"
#include "trueform/core/metric_point.hpp"
#include "trueform/core/views/sequence_range.hpp"
#include "trueform/spatial/nearest_neighbors.hpp"
#include "trueform/spatial/neighbor_search.hpp"
#include "trueform/spatial/tree_metric_info.hpp"
#include "trueform/ts/core/promise.hpp"
#include "trueform/ts/core/wasm_mesh.hpp"
#include "trueform/ts/core/wasm_ndarray.hpp"
#include "trueform/ts/core/wasm_point_cloud.hpp"
#include "trueform/ts/spatial/prim_dispatch.hpp"
#include "trueform/ts/spatial/result_types.hpp"
#include <emscripten/bind.h>
#include <limits>

namespace {

using namespace tf::ts;

constexpr float no_radius = std::numeric_limits<float>::max();

// ============================================================================
// FP helpers (form × prim) — always take radius
// ============================================================================

template <typename FormT>
auto fp_single(FormT &m, const float *b, prim_type tb, int vb,
               float radius) -> neighbor_result {
  return m.with_form([&](const auto &form) -> neighbor_result {
    return dispatch_single(
        [&](const auto &pb) -> neighbor_result {
          auto r = tf::neighbor_search(form, pb, radius);
          return {r.element, r.info.metric, point_to_ndarray(r.info.point)};
        },
        b, tb, vb);
  });
}

template <typename FormT>
auto fp_batch(FormT &m, wasm_ndarray<float> &b, prim_type tb,
              float radius) -> neighbor_batch_result {
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
          [&](const auto &prim) {
            return tf::neighbor_search(form, prim, radius);
          },
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
// FP k-NN helpers (form × prim only)
// ============================================================================

template <typename FormT>
auto fp_single_knn(FormT &m, const float *b, prim_type tb, int vb, int k,
                   float radius) -> neighbor_knn_result {
  using result_t = tf::tree_metric_info<int, tf::metric_point<float, 3>>;
  std::vector<result_t> buf;

  int count = 0;
  m.with_form([&](const auto &form) {
    buf.resize(k);
    auto knn = (radius < no_radius)
        ? tf::make_nearest_neighbors(buf.data(), k, radius)
        : tf::make_nearest_neighbors(buf.data(), k);
    dispatch_single(
        [&](const auto &prim) { tf::neighbor_search(form, prim, knn); }, b, tb,
        vb);
    count = static_cast<int>(knn.size());
  });

  tf::buffer<int> ids;
  ids.allocate(count);
  tf::buffer<float> pts;
  pts.allocate(count * 3);
  tf::buffer<float> dists;
  dists.allocate(count);
  for (int j = 0; j < count; ++j) {
    ids.data()[j] = buf[j].element;
    dists.data()[j] = buf[j].info.metric;
    pts.data()[j * 3 + 0] = buf[j].info.point[0];
    pts.data()[j * 3 + 1] = buf[j].info.point[1];
    pts.data()[j * 3 + 2] = buf[j].info.point[2];
  }
  return {wasm_ndarray<int>::from_buffer(std::move(ids), {count}),
          wasm_ndarray<float>::from_buffer(std::move(pts), {count, 3}),
          wasm_ndarray<float>::from_buffer(std::move(dists), {count})};
}

template <typename FormT>
auto fp_batch_knn(FormT &m, wasm_ndarray<float> &b, prim_type tb, int k,
                  float radius) -> neighbor_knn_batch_result {
  int n = batch_count(b, tb);
  int sb = stride_of(b, tb);
  int vb = poly_verts(b, tb);
  const float *pb = b.raw_data();

  tf::buffer<int> ids;
  ids.allocate(n * k);
  tf::buffer<float> pts;
  pts.allocate(n * k * 3);
  tf::buffer<float> dists;
  dists.allocate(n * k);
  tf::buffer<int> counts;
  counts.allocate(n);
  tf::parallel_fill(ids, -1);

  int *id_dst = ids.data();
  float *pts_dst = pts.data();
  float *dist_dst = dists.data();
  int *count_dst = counts.data();

  m.with_form([&](const auto &form) {
    using result_t = tf::tree_metric_info<int, tf::metric_point<float, 3>>;
    std::vector<result_t> state_buf;

    auto compute = [&, pb, sb, tb, vb, k,
                    radius](int i, std::vector<result_t> &buf) {
      buf.resize(k);
      auto knn = (radius < no_radius)
          ? tf::make_nearest_neighbors(buf.data(), k, radius)
          : tf::make_nearest_neighbors(buf.data(), k);
      dispatch_single(
          [&](const auto &prim) { tf::neighbor_search(form, prim, knn); },
          pb + i * sb, tb, vb);

      int count = static_cast<int>(knn.size());
      count_dst[i] = count;
      int base = i * k;
      for (int j = 0; j < count; ++j) {
        id_dst[base + j] = buf[j].element;
        dist_dst[base + j] = buf[j].info.metric;
        pts_dst[(base + j) * 3 + 0] = buf[j].info.point[0];
        pts_dst[(base + j) * 3 + 1] = buf[j].info.point[1];
        pts_dst[(base + j) * 3 + 2] = buf[j].info.point[2];
      }
    };

    if (n >= 100)
      tf::parallel_for_each(tf::make_sequence_range(n), compute,
                            std::move(state_buf));
    else {
      state_buf.resize(k);
      for (int i = 0; i < n; ++i)
        compute(i, state_buf);
    }
  });

  return {wasm_ndarray<int>::from_buffer(std::move(ids), {n, k}),
          wasm_ndarray<float>::from_buffer(std::move(pts), {n, k, 3}),
          wasm_ndarray<float>::from_buffer(std::move(dists), {n, k}),
          wasm_ndarray<int>::from_buffer(std::move(counts), {n})};
}

// ============================================================================
// FF helper (form × form) — always take radius
// ============================================================================

template <typename F0, typename F1>
auto ff_compute(F0 &m0, F1 &m1, float radius) -> neighbor_result_pair {
  return m0.with_form([&](const auto &form0) -> neighbor_result_pair {
    return m1.with_form([&](const auto &form1) -> neighbor_result_pair {
      auto r = tf::neighbor_search(form0, form1, radius);
      return {r.elements.first, r.elements.second, r.info.metric,
              point_to_ndarray(r.info.first),
              point_to_ndarray(r.info.second)};
    });
  });
}

// ============================================================================
// Sync FP dispatch — template on FormT
// ============================================================================

template <typename FormT>
auto sync_fp(FormT &m, wasm_ndarray<float> &b, int tb_int,
             float radius) -> emscripten::val {
  auto tb = static_cast<prim_type>(tb_int);
  if (!is_batch(b, tb))
    return emscripten::val(
        fp_single(m, b.raw_data(), tb, poly_verts(b, tb), radius));
  return emscripten::val(fp_batch(m, b, tb, radius));
}

template <typename FormT>
auto sync_fp_knn(FormT &m, wasm_ndarray<float> &b, int tb_int, int k,
                 float radius) -> emscripten::val {
  auto tb = static_cast<prim_type>(tb_int);
  if (!is_batch(b, tb))
    return emscripten::val(
        fp_single_knn(m, b.raw_data(), tb, poly_verts(b, tb), k, radius));
  return emscripten::val(fp_batch_knn(m, b, tb, k, radius));
}

// ============================================================================
// Sync entry points — thin typed wrappers
// ============================================================================

// FP mesh
auto sync_neighbor_search_fp(wasm_mesh &m, wasm_ndarray<float> &b,
                              int tb, float r) -> emscripten::val {
  return sync_fp(m, b, tb, r);
}

// FP point cloud
auto sync_neighbor_search_fp_pc(wasm_point_cloud &m, wasm_ndarray<float> &b,
                                 int tb, float r) -> emscripten::val {
  return sync_fp(m, b, tb, r);
}

// FP k-NN mesh
auto sync_neighbor_search_fp_knn(wasm_mesh &m, wasm_ndarray<float> &b,
                                  int tb, int k, float r) -> emscripten::val {
  return sync_fp_knn(m, b, tb, k, r);
}

// FP k-NN point cloud
auto sync_neighbor_search_fp_knn_pc(wasm_point_cloud &m,
                                     wasm_ndarray<float> &b, int tb, int k,
                                     float r) -> emscripten::val {
  return sync_fp_knn(m, b, tb, k, r);
}

// FF — all 4 combos
auto sync_neighbor_search_ff(wasm_mesh &a, wasm_mesh &b,
                              float r) -> neighbor_result_pair {
  return ff_compute(a, b, r);
}
auto sync_neighbor_search_ff_mp(wasm_mesh &a, wasm_point_cloud &b,
                                 float r) -> neighbor_result_pair {
  return ff_compute(a, b, r);
}
auto sync_neighbor_search_ff_pm(wasm_point_cloud &a, wasm_mesh &b,
                                 float r) -> neighbor_result_pair {
  return ff_compute(a, b, r);
}
auto sync_neighbor_search_ff_pc(wasm_point_cloud &a, wasm_point_cloud &b,
                                 float r) -> neighbor_result_pair {
  return ff_compute(a, b, r);
}

// ============================================================================
// Async FP dispatch — template on FormT
// ============================================================================

template <typename FormT>
auto async_fp(FormT &m, wasm_ndarray<float> &b, int tb_int,
              float radius) -> promise_t {
  auto tb = static_cast<prim_type>(tb_int);
  if (!is_batch(b, tb)) {
    int vb = poly_verts(b, tb);
    return promise([m = m, b = b, tb, vb, radius]() -> neighbor_result {
      return fp_single(const_cast<FormT &>(m),
                       const_cast<wasm_ndarray<float> &>(b).raw_data(), tb, vb,
                       radius);
    });
  }
  return promise([m = m, b = b, tb, radius]() -> neighbor_batch_result {
    return fp_batch(const_cast<FormT &>(m),
                    const_cast<wasm_ndarray<float> &>(b), tb, radius);
  });
}

template <typename FormT>
auto async_fp_knn(FormT &m, wasm_ndarray<float> &b, int tb_int, int k,
                  float radius) -> promise_t {
  auto tb = static_cast<prim_type>(tb_int);
  if (!is_batch(b, tb)) {
    int vb = poly_verts(b, tb);
    return promise(
        [m = m, b = b, tb, vb, k, radius]() -> neighbor_knn_result {
          return fp_single_knn(const_cast<FormT &>(m),
                               const_cast<wasm_ndarray<float> &>(b).raw_data(),
                               tb, vb, k, radius);
        });
  }
  return promise(
      [m = m, b = b, tb, k, radius]() -> neighbor_knn_batch_result {
        return fp_batch_knn(const_cast<FormT &>(m),
                            const_cast<wasm_ndarray<float> &>(b), tb, k,
                            radius);
      });
}

// ============================================================================
// Async entry points — thin typed wrappers
// ============================================================================

// FP mesh / point cloud
auto async_neighbor_search_fp(wasm_mesh &m, wasm_ndarray<float> &b, int tb,
                               float r) -> promise_t {
  return async_fp(m, b, tb, r);
}
auto async_neighbor_search_fp_pc(wasm_point_cloud &m, wasm_ndarray<float> &b,
                                  int tb, float r) -> promise_t {
  return async_fp(m, b, tb, r);
}

// FP k-NN mesh / point cloud
auto async_neighbor_search_fp_knn(wasm_mesh &m, wasm_ndarray<float> &b,
                                   int tb, int k, float r) -> promise_t {
  return async_fp_knn(m, b, tb, k, r);
}
auto async_neighbor_search_fp_knn_pc(wasm_point_cloud &m,
                                      wasm_ndarray<float> &b, int tb, int k,
                                      float r) -> promise_t {
  return async_fp_knn(m, b, tb, k, r);
}

// FF — all 4 combos
auto async_neighbor_search_ff(wasm_mesh &a, wasm_mesh &b,
                               float r) -> promise_t {
  return promise([a = a, b = b, r]() -> neighbor_result_pair {
    return ff_compute(const_cast<wasm_mesh &>(a), const_cast<wasm_mesh &>(b),
                      r);
  });
}
auto async_neighbor_search_ff_mp(wasm_mesh &a, wasm_point_cloud &b,
                                  float r) -> promise_t {
  return promise([a = a, b = b, r]() -> neighbor_result_pair {
    return ff_compute(const_cast<wasm_mesh &>(a),
                      const_cast<wasm_point_cloud &>(b), r);
  });
}
auto async_neighbor_search_ff_pm(wasm_point_cloud &a, wasm_mesh &b,
                                  float r) -> promise_t {
  return promise([a = a, b = b, r]() -> neighbor_result_pair {
    return ff_compute(const_cast<wasm_point_cloud &>(a),
                      const_cast<wasm_mesh &>(b), r);
  });
}
auto async_neighbor_search_ff_pc(wasm_point_cloud &a, wasm_point_cloud &b,
                                  float r) -> promise_t {
  return promise([a = a, b = b, r]() -> neighbor_result_pair {
    return ff_compute(const_cast<wasm_point_cloud &>(a),
                      const_cast<wasm_point_cloud &>(b), r);
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

  emscripten::value_object<tf::ts::neighbor_knn_result>("NeighborKnnResult")
      .field("elementIds", &tf::ts::neighbor_knn_result::element_ids)
      .field("points", &tf::ts::neighbor_knn_result::points)
      .field("distances", &tf::ts::neighbor_knn_result::distances);

  emscripten::value_object<tf::ts::neighbor_knn_batch_result>(
      "NeighborKnnBatchResult")
      .field("elementIds", &tf::ts::neighbor_knn_batch_result::element_ids)
      .field("points", &tf::ts::neighbor_knn_batch_result::points)
      .field("distances", &tf::ts::neighbor_knn_batch_result::distances)
      .field("counts", &tf::ts::neighbor_knn_batch_result::counts);

  // FP — mesh & point cloud (all take radius, TS sends Infinity for "no limit")
  emscripten::function("neighbor_search_fp", &sync_neighbor_search_fp);
  emscripten::function("neighbor_search_fp_pc", &sync_neighbor_search_fp_pc);
  emscripten::function("dispatch_neighbor_search_fp",
                       &async_neighbor_search_fp);
  emscripten::function("dispatch_neighbor_search_fp_pc",
                       &async_neighbor_search_fp_pc);

  // FP k-NN — mesh & point cloud
  emscripten::function("neighbor_search_fp_knn", &sync_neighbor_search_fp_knn);
  emscripten::function("neighbor_search_fp_knn_pc",
                       &sync_neighbor_search_fp_knn_pc);
  emscripten::function("dispatch_neighbor_search_fp_knn",
                       &async_neighbor_search_fp_knn);
  emscripten::function("dispatch_neighbor_search_fp_knn_pc",
                       &async_neighbor_search_fp_knn_pc);

  // FF — all 4 combos (all take radius)
  emscripten::function("neighbor_search_ff", &sync_neighbor_search_ff);
  emscripten::function("neighbor_search_ff_mp", &sync_neighbor_search_ff_mp);
  emscripten::function("neighbor_search_ff_pm", &sync_neighbor_search_ff_pm);
  emscripten::function("neighbor_search_ff_pc", &sync_neighbor_search_ff_pc);
  emscripten::function("dispatch_neighbor_search_ff",
                       &async_neighbor_search_ff);
  emscripten::function("dispatch_neighbor_search_ff_mp",
                       &async_neighbor_search_ff_mp);
  emscripten::function("dispatch_neighbor_search_ff_pm",
                       &async_neighbor_search_ff_pm);
  emscripten::function("dispatch_neighbor_search_ff_pc",
                       &async_neighbor_search_ff_pc);
}
