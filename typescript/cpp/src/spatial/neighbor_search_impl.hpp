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
#include <emscripten/bind.h>
#include <limits>
#include <vector>

namespace tf {
namespace ts {

// ============================================================================
// Templated result types
// ============================================================================

template <typename Real> struct neighbor_result_t {
  int element_id; // -1 if no hit
  Real distance2;
  wasm_ndarray<Real> point; // [3]
};

template <typename Real> struct neighbor_batch_result_t {
  wasm_ndarray<int> element_ids;  // [N]
  wasm_ndarray<Real> points;      // [N, 3]
  wasm_ndarray<Real> distances;   // [N]
};

template <typename Real> struct neighbor_result_pair_t {
  int element_id0;
  int element_id1;
  Real distance2;
  wasm_ndarray<Real> point0; // [3]
  wasm_ndarray<Real> point1; // [3]
};

template <typename Real> struct neighbor_knn_result_t {
  wasm_ndarray<int> element_ids; // [count]
  wasm_ndarray<Real> points;     // [count, 3]
  wasm_ndarray<Real> distances;  // [count]
};

template <typename Real> struct neighbor_knn_batch_result_t {
  wasm_ndarray<int> element_ids; // [N, k] padded with -1
  wasm_ndarray<Real> points;     // [N, k, 3]
  wasm_ndarray<Real> distances;  // [N, k]
  wasm_ndarray<int> counts;      // [N]
};

// ============================================================================
// Helper: copy a tf::point<Real,3> into a fresh wasm_ndarray<Real> [3]
// ============================================================================

template <typename Real, typename PointLike>
inline auto point_to_ndarray_real(const PointLike &pt) -> wasm_ndarray<Real> {
  tf::buffer<Real> buf;
  buf.allocate(3);
  auto *p = buf.data();
  p[0] = pt[0];
  p[1] = pt[1];
  p[2] = pt[2];
  return wasm_ndarray<Real>::from_buffer(std::move(buf), {3});
}

// ============================================================================
// FP helpers (form × prim) — always take radius
// ============================================================================

template <typename Real, typename FormT>
inline auto fp_single(FormT &m, const Real *b, prim_type tb, int vb,
                      Real radius) -> neighbor_result_t<Real> {
  return m.with_form([&](const auto &form) -> neighbor_result_t<Real> {
    return dispatch_single(
        [&](const auto &pb) -> neighbor_result_t<Real> {
          auto r = tf::neighbor_search(form, pb, radius);
          return {r.element, r.info.metric,
                  point_to_ndarray_real<Real>(r.info.point)};
        },
        b, tb, vb);
  });
}

template <typename Real, typename FormT>
inline auto fp_batch(FormT &m, wasm_ndarray<Real> &b, prim_type tb,
                     Real radius) -> neighbor_batch_result_t<Real> {
  int vb = poly_verts(b, tb);
  int sb = stride_of(b, tb);
  int n = batch_count(b, tb);
  const Real *pb = b.raw_data();

  tf::buffer<int> ids;
  ids.allocate(n);
  tf::buffer<Real> pts;
  pts.allocate(n * 3);
  tf::buffer<Real> dists;
  dists.allocate(n);
  int *id_dst = ids.data();
  Real *pts_dst = pts.data();
  Real *dist_dst = dists.data();

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
          wasm_ndarray<Real>::from_buffer(std::move(pts), {n, 3}),
          wasm_ndarray<Real>::from_buffer(std::move(dists), {n})};
}

// ============================================================================
// FP k-NN helpers (form × prim only)
// ============================================================================

template <typename Real, typename FormT>
inline auto fp_single_knn(FormT &m, const Real *b, prim_type tb, int vb, int k,
                          Real radius) -> neighbor_knn_result_t<Real> {
  using result_t = tf::tree_metric_info<int, tf::metric_point<Real, 3>>;
  std::vector<result_t> buf;

  constexpr Real no_radius = std::numeric_limits<Real>::max();
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
  tf::buffer<Real> pts;
  pts.allocate(count * 3);
  tf::buffer<Real> dists;
  dists.allocate(count);
  for (int j = 0; j < count; ++j) {
    ids.data()[j] = buf[j].element;
    dists.data()[j] = buf[j].info.metric;
    pts.data()[j * 3 + 0] = buf[j].info.point[0];
    pts.data()[j * 3 + 1] = buf[j].info.point[1];
    pts.data()[j * 3 + 2] = buf[j].info.point[2];
  }
  return {wasm_ndarray<int>::from_buffer(std::move(ids), {count}),
          wasm_ndarray<Real>::from_buffer(std::move(pts), {count, 3}),
          wasm_ndarray<Real>::from_buffer(std::move(dists), {count})};
}

template <typename Real, typename FormT>
inline auto fp_batch_knn(FormT &m, wasm_ndarray<Real> &b, prim_type tb, int k,
                         Real radius) -> neighbor_knn_batch_result_t<Real> {
  int n = batch_count(b, tb);
  int sb = stride_of(b, tb);
  int vb = poly_verts(b, tb);
  const Real *pb = b.raw_data();

  tf::buffer<int> ids;
  ids.allocate(n * k);
  tf::buffer<Real> pts;
  pts.allocate(n * k * 3);
  tf::buffer<Real> dists;
  dists.allocate(n * k);
  tf::buffer<int> counts;
  counts.allocate(n);
  tf::parallel_fill(ids, -1);

  int *id_dst = ids.data();
  Real *pts_dst = pts.data();
  Real *dist_dst = dists.data();
  int *count_dst = counts.data();

  constexpr Real no_radius = std::numeric_limits<Real>::max();

  m.with_form([&](const auto &form) {
    using result_t = tf::tree_metric_info<int, tf::metric_point<Real, 3>>;
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
          wasm_ndarray<Real>::from_buffer(std::move(pts), {n, k, 3}),
          wasm_ndarray<Real>::from_buffer(std::move(dists), {n, k}),
          wasm_ndarray<int>::from_buffer(std::move(counts), {n})};
}

// ============================================================================
// FF helper (form × form) — always take radius
// ============================================================================

template <typename Real, typename F0, typename F1>
inline auto ff_compute(F0 &m0, F1 &m1, Real radius)
    -> neighbor_result_pair_t<Real> {
  return m0.with_form([&](const auto &form0) -> neighbor_result_pair_t<Real> {
    return m1.with_form([&](const auto &form1) -> neighbor_result_pair_t<Real> {
      auto r = tf::neighbor_search(form0, form1, radius);
      return {r.elements.first, r.elements.second, r.info.metric,
              point_to_ndarray_real<Real>(r.info.first),
              point_to_ndarray_real<Real>(r.info.second)};
    });
  });
}

// ============================================================================
// Sync FP dispatch — template on Real and FormT
// ============================================================================

template <typename Real, typename FormT>
inline auto sync_fp(FormT &m, wasm_ndarray<Real> &b, int tb_int, Real radius)
    -> emscripten::val {
  auto tb = static_cast<prim_type>(tb_int);
  if (!is_batch(b, tb))
    return emscripten::val(
        fp_single<Real>(m, b.raw_data(), tb, poly_verts(b, tb), radius));
  return emscripten::val(fp_batch<Real>(m, b, tb, radius));
}

template <typename Real, typename FormT>
inline auto sync_fp_knn(FormT &m, wasm_ndarray<Real> &b, int tb_int, int k,
                        Real radius) -> emscripten::val {
  auto tb = static_cast<prim_type>(tb_int);
  if (!is_batch(b, tb))
    return emscripten::val(
        fp_single_knn<Real>(m, b.raw_data(), tb, poly_verts(b, tb), k, radius));
  return emscripten::val(fp_batch_knn<Real>(m, b, tb, k, radius));
}

// ============================================================================
// Sync entry points
// ============================================================================

template <typename Real>
inline auto sync_neighbor_search_fp(wasm_mesh<Real> &m, wasm_ndarray<Real> &b,
                                    int tb, Real r) -> emscripten::val {
  return sync_fp<Real>(m, b, tb, r);
}

template <typename Real>
inline auto sync_neighbor_search_fp_pc(wasm_point_cloud<Real> &m,
                                       wasm_ndarray<Real> &b, int tb, Real r)
    -> emscripten::val {
  return sync_fp<Real>(m, b, tb, r);
}

template <typename Real>
inline auto sync_neighbor_search_fp_knn(wasm_mesh<Real> &m,
                                        wasm_ndarray<Real> &b, int tb, int k,
                                        Real r) -> emscripten::val {
  return sync_fp_knn<Real>(m, b, tb, k, r);
}

template <typename Real>
inline auto sync_neighbor_search_fp_knn_pc(wasm_point_cloud<Real> &m,
                                           wasm_ndarray<Real> &b, int tb,
                                           int k, Real r) -> emscripten::val {
  return sync_fp_knn<Real>(m, b, tb, k, r);
}

template <typename Real>
inline auto sync_neighbor_search_ff(wasm_mesh<Real> &a, wasm_mesh<Real> &b,
                                    Real r) -> neighbor_result_pair_t<Real> {
  return ff_compute<Real>(a, b, r);
}

template <typename Real>
inline auto sync_neighbor_search_ff_mp(wasm_mesh<Real> &a,
                                       wasm_point_cloud<Real> &b, Real r)
    -> neighbor_result_pair_t<Real> {
  return ff_compute<Real>(a, b, r);
}

template <typename Real>
inline auto sync_neighbor_search_ff_pm(wasm_point_cloud<Real> &a,
                                       wasm_mesh<Real> &b, Real r)
    -> neighbor_result_pair_t<Real> {
  return ff_compute<Real>(a, b, r);
}

template <typename Real>
inline auto sync_neighbor_search_ff_pc(wasm_point_cloud<Real> &a,
                                       wasm_point_cloud<Real> &b, Real r)
    -> neighbor_result_pair_t<Real> {
  return ff_compute<Real>(a, b, r);
}

// ============================================================================
// Async FP dispatch — template on Real and FormT
// ============================================================================

template <typename Real, typename FormT>
inline auto async_fp(FormT &m, wasm_ndarray<Real> &b, int tb_int, Real radius)
    -> promise_t {
  auto tb = static_cast<prim_type>(tb_int);
  if (!is_batch(b, tb)) {
    int vb = poly_verts(b, tb);
    return promise(
        [m = m, b = b, tb, vb, radius]() -> neighbor_result_t<Real> {
          return fp_single<Real>(
              const_cast<FormT &>(m),
              const_cast<wasm_ndarray<Real> &>(b).raw_data(), tb, vb, radius);
        });
  }
  return promise([m = m, b = b, tb, radius]() -> neighbor_batch_result_t<Real> {
    return fp_batch<Real>(const_cast<FormT &>(m),
                          const_cast<wasm_ndarray<Real> &>(b), tb, radius);
  });
}

template <typename Real, typename FormT>
inline auto async_fp_knn(FormT &m, wasm_ndarray<Real> &b, int tb_int, int k,
                         Real radius) -> promise_t {
  auto tb = static_cast<prim_type>(tb_int);
  if (!is_batch(b, tb)) {
    int vb = poly_verts(b, tb);
    return promise(
        [m = m, b = b, tb, vb, k, radius]() -> neighbor_knn_result_t<Real> {
          return fp_single_knn<Real>(
              const_cast<FormT &>(m),
              const_cast<wasm_ndarray<Real> &>(b).raw_data(), tb, vb, k,
              radius);
        });
  }
  return promise(
      [m = m, b = b, tb, k, radius]() -> neighbor_knn_batch_result_t<Real> {
        return fp_batch_knn<Real>(const_cast<FormT &>(m),
                                  const_cast<wasm_ndarray<Real> &>(b), tb, k,
                                  radius);
      });
}

// ============================================================================
// Async entry points
// ============================================================================

template <typename Real>
inline auto async_neighbor_search_fp(wasm_mesh<Real> &m, wasm_ndarray<Real> &b,
                                     int tb, Real r) -> promise_t {
  return async_fp<Real>(m, b, tb, r);
}

template <typename Real>
inline auto async_neighbor_search_fp_pc(wasm_point_cloud<Real> &m,
                                        wasm_ndarray<Real> &b, int tb, Real r)
    -> promise_t {
  return async_fp<Real>(m, b, tb, r);
}

template <typename Real>
inline auto async_neighbor_search_fp_knn(wasm_mesh<Real> &m,
                                         wasm_ndarray<Real> &b, int tb, int k,
                                         Real r) -> promise_t {
  return async_fp_knn<Real>(m, b, tb, k, r);
}

template <typename Real>
inline auto async_neighbor_search_fp_knn_pc(wasm_point_cloud<Real> &m,
                                            wasm_ndarray<Real> &b, int tb,
                                            int k, Real r) -> promise_t {
  return async_fp_knn<Real>(m, b, tb, k, r);
}

template <typename Real>
inline auto async_neighbor_search_ff(wasm_mesh<Real> &a, wasm_mesh<Real> &b,
                                     Real r) -> promise_t {
  return promise([a = a, b = b, r]() -> neighbor_result_pair_t<Real> {
    return ff_compute<Real>(const_cast<wasm_mesh<Real> &>(a),
                            const_cast<wasm_mesh<Real> &>(b), r);
  });
}

template <typename Real>
inline auto async_neighbor_search_ff_mp(wasm_mesh<Real> &a,
                                        wasm_point_cloud<Real> &b, Real r)
    -> promise_t {
  return promise([a = a, b = b, r]() -> neighbor_result_pair_t<Real> {
    return ff_compute<Real>(const_cast<wasm_mesh<Real> &>(a),
                            const_cast<wasm_point_cloud<Real> &>(b), r);
  });
}

template <typename Real>
inline auto async_neighbor_search_ff_pm(wasm_point_cloud<Real> &a,
                                        wasm_mesh<Real> &b, Real r)
    -> promise_t {
  return promise([a = a, b = b, r]() -> neighbor_result_pair_t<Real> {
    return ff_compute<Real>(const_cast<wasm_point_cloud<Real> &>(a),
                            const_cast<wasm_mesh<Real> &>(b), r);
  });
}

template <typename Real>
inline auto async_neighbor_search_ff_pc(wasm_point_cloud<Real> &a,
                                        wasm_point_cloud<Real> &b, Real r)
    -> promise_t {
  return promise([a = a, b = b, r]() -> neighbor_result_pair_t<Real> {
    return ff_compute<Real>(const_cast<wasm_point_cloud<Real> &>(a),
                            const_cast<wasm_point_cloud<Real> &>(b), r);
  });
}

} // namespace ts
} // namespace tf
