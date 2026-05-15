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
#include "trueform/core/distance.hpp"
#include "trueform/core/views/sequence_range.hpp"
#include "trueform/spatial/distance.hpp"
#include "trueform/ts/core/promise.hpp"
#include "trueform/ts/core/wasm_mesh.hpp"
#include "trueform/ts/core/wasm_ndarray.hpp"
#include "trueform/ts/core/wasm_point_cloud.hpp"
#include "trueform/ts/spatial/prim_dispatch.hpp"
#include <emscripten/bind.h>

namespace tf {
namespace ts {

// ============================================================================
// PP helpers — distance2 (prim × prim)
// ============================================================================

template <typename Real>
inline auto pp_single_d2(const Real *a, prim_type ta, const Real *b,
                         prim_type tb, int va, int vb) -> Real {
  return dispatch_pair(
      [](const auto &pa, const auto &pb) -> Real {
        return tf::distance2(pa, pb);
      },
      a, ta, b, tb, va, vb);
}

template <typename Real>
inline auto pp_batch_d2(wasm_ndarray<Real> &a, prim_type ta,
                        wasm_ndarray<Real> &b, prim_type tb)
    -> wasm_ndarray<Real> {
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

  tf::buffer<Real> out;
  out.allocate(n);
  Real *dst = out.data();

  auto compute = [=](int i) {
    dst[i] =
        pp_single_d2<Real>(pa + i * stride_a, ta, pb + i * stride_b, tb, va, vb);
  };

  if (n >= 5000)
    tf::parallel_for_each(tf::make_sequence_range(n), compute);
  else
    for (int i = 0; i < n; ++i)
      compute(i);

  return wasm_ndarray<Real>::from_buffer(std::move(out), {n});
}

// ============================================================================
// PP helpers — distance (prim × prim)
// ============================================================================

template <typename Real>
inline auto pp_single_d(const Real *a, prim_type ta, const Real *b,
                        prim_type tb, int va, int vb) -> Real {
  return dispatch_pair(
      [](const auto &pa, const auto &pb) -> Real {
        return tf::distance(pa, pb);
      },
      a, ta, b, tb, va, vb);
}

template <typename Real>
inline auto pp_batch_d(wasm_ndarray<Real> &a, prim_type ta,
                       wasm_ndarray<Real> &b, prim_type tb)
    -> wasm_ndarray<Real> {
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

  tf::buffer<Real> out;
  out.allocate(n);
  Real *dst = out.data();

  auto compute = [=](int i) {
    dst[i] =
        pp_single_d<Real>(pa + i * stride_a, ta, pb + i * stride_b, tb, va, vb);
  };

  if (n >= 5000)
    tf::parallel_for_each(tf::make_sequence_range(n), compute);
  else
    for (int i = 0; i < n; ++i)
      compute(i);

  return wasm_ndarray<Real>::from_buffer(std::move(out), {n});
}

// ============================================================================
// FP helpers — distance2 (form × prim)
// ============================================================================

template <typename Real, typename FormT>
inline auto fp_single_d2(FormT &m, const Real *b, prim_type tb, int vb)
    -> Real {
  return m.with_form([&](const auto &form) -> Real {
    return dispatch_single(
        [&](const auto &pb) -> Real { return tf::distance2(form, pb); }, b, tb,
        vb);
  });
}

template <typename Real, typename FormT>
inline auto fp_batch_d2(FormT &m, wasm_ndarray<Real> &b, prim_type tb)
    -> wasm_ndarray<Real> {
  int vb = poly_verts(b, tb);
  int sb = stride_of(b, tb);
  int n = batch_count(b, tb);
  const Real *pb = b.raw_data();

  tf::buffer<Real> out;
  out.allocate(n);
  Real *dst = out.data();

  m.with_form([&](const auto &form) {
    auto compute = [&](int i) {
      dst[i] = dispatch_single(
          [&](const auto &prim) -> Real { return tf::distance2(form, prim); },
          pb + i * sb, tb, vb);
    };
    if (n >= 100)
      tf::parallel_for_each(tf::make_sequence_range(n), compute);
    else
      for (int i = 0; i < n; ++i)
        compute(i);
  });

  return wasm_ndarray<Real>::from_buffer(std::move(out), {n});
}

// ============================================================================
// FP helpers — distance (form × prim)
// ============================================================================

template <typename Real, typename FormT>
inline auto fp_single_d(FormT &m, const Real *b, prim_type tb, int vb) -> Real {
  return m.with_form([&](const auto &form) -> Real {
    return dispatch_single(
        [&](const auto &pb) -> Real { return tf::distance(form, pb); }, b, tb,
        vb);
  });
}

template <typename Real, typename FormT>
inline auto fp_batch_d(FormT &m, wasm_ndarray<Real> &b, prim_type tb)
    -> wasm_ndarray<Real> {
  int vb = poly_verts(b, tb);
  int sb = stride_of(b, tb);
  int n = batch_count(b, tb);
  const Real *pb = b.raw_data();

  tf::buffer<Real> out;
  out.allocate(n);
  Real *dst = out.data();

  m.with_form([&](const auto &form) {
    auto compute = [&](int i) {
      dst[i] = dispatch_single(
          [&](const auto &prim) -> Real { return tf::distance(form, prim); },
          pb + i * sb, tb, vb);
    };
    if (n >= 100)
      tf::parallel_for_each(tf::make_sequence_range(n), compute);
    else
      for (int i = 0; i < n; ++i)
        compute(i);
  });

  return wasm_ndarray<Real>::from_buffer(std::move(out), {n});
}

// ============================================================================
// FF helpers — distance2 (form × form)
// ============================================================================

template <typename Real, typename F0, typename F1>
inline auto ff_compute_d2(F0 &m0, F1 &m1) -> Real {
  return m0.with_form([&](const auto &form0) -> Real {
    return m1.with_form([&](const auto &form1) -> Real {
      return tf::distance2(form0, form1);
    });
  });
}

// ============================================================================
// FF helpers — distance (form × form)
// ============================================================================

template <typename Real, typename F0, typename F1>
inline auto ff_compute_d(F0 &m0, F1 &m1) -> Real {
  return m0.with_form([&](const auto &form0) -> Real {
    return m1.with_form([&](const auto &form1) -> Real {
      return tf::distance(form0, form1);
    });
  });
}

// ============================================================================
// Sync FP dispatch — template on FormT
// ============================================================================

template <typename Real, typename FormT>
inline auto sync_fp_d2(FormT &m, wasm_ndarray<Real> &b, int tb_int)
    -> emscripten::val {
  auto tb = static_cast<prim_type>(tb_int);
  if (!is_batch(b, tb))
    return emscripten::val(
        fp_single_d2<Real>(m, b.raw_data(), tb, poly_verts(b, tb)));
  return emscripten::val(fp_batch_d2<Real>(m, b, tb));
}

template <typename Real, typename FormT>
inline auto sync_fp_d(FormT &m, wasm_ndarray<Real> &b, int tb_int)
    -> emscripten::val {
  auto tb = static_cast<prim_type>(tb_int);
  if (!is_batch(b, tb))
    return emscripten::val(
        fp_single_d<Real>(m, b.raw_data(), tb, poly_verts(b, tb)));
  return emscripten::val(fp_batch_d<Real>(m, b, tb));
}

// ============================================================================
// Sync entry points — distance2
// ============================================================================

template <typename Real>
inline auto sync_distance2_pp(wasm_ndarray<Real> &a, int ta_int,
                              wasm_ndarray<Real> &b, int tb_int)
    -> emscripten::val {
  auto ta = static_cast<prim_type>(ta_int);
  auto tb = static_cast<prim_type>(tb_int);
  if (!is_batch(a, ta) && !is_batch(b, tb))
    return emscripten::val(
        pp_single_d2<Real>(a.raw_data(), ta, b.raw_data(), tb, poly_verts(a, ta),
                           poly_verts(b, tb)));
  return emscripten::val(pp_batch_d2<Real>(a, ta, b, tb));
}

template <typename Real>
inline auto sync_distance2_fp(wasm_mesh<Real> &m, wasm_ndarray<Real> &b,
                              int tb_int) -> emscripten::val {
  return sync_fp_d2<Real>(m, b, tb_int);
}

template <typename Real>
inline auto sync_distance2_fp_pc(wasm_point_cloud<Real> &m,
                                 wasm_ndarray<Real> &b, int tb_int)
    -> emscripten::val {
  return sync_fp_d2<Real>(m, b, tb_int);
}

template <typename Real>
inline auto sync_distance2_ff(wasm_mesh<Real> &m0, wasm_mesh<Real> &m1)
    -> Real {
  return ff_compute_d2<Real>(m0, m1);
}

template <typename Real>
inline auto sync_distance2_ff_mp(wasm_mesh<Real> &m0,
                                 wasm_point_cloud<Real> &m1) -> Real {
  return ff_compute_d2<Real>(m0, m1);
}

template <typename Real>
inline auto sync_distance2_ff_pm(wasm_point_cloud<Real> &m0,
                                 wasm_mesh<Real> &m1) -> Real {
  return ff_compute_d2<Real>(m0, m1);
}

template <typename Real>
inline auto sync_distance2_ff_pc(wasm_point_cloud<Real> &m0,
                                 wasm_point_cloud<Real> &m1) -> Real {
  return ff_compute_d2<Real>(m0, m1);
}

// ============================================================================
// Sync entry points — distance
// ============================================================================

template <typename Real>
inline auto sync_distance_pp(wasm_ndarray<Real> &a, int ta_int,
                             wasm_ndarray<Real> &b, int tb_int)
    -> emscripten::val {
  auto ta = static_cast<prim_type>(ta_int);
  auto tb = static_cast<prim_type>(tb_int);
  if (!is_batch(a, ta) && !is_batch(b, tb))
    return emscripten::val(
        pp_single_d<Real>(a.raw_data(), ta, b.raw_data(), tb, poly_verts(a, ta),
                          poly_verts(b, tb)));
  return emscripten::val(pp_batch_d<Real>(a, ta, b, tb));
}

template <typename Real>
inline auto sync_distance_fp(wasm_mesh<Real> &m, wasm_ndarray<Real> &b,
                             int tb_int) -> emscripten::val {
  return sync_fp_d<Real>(m, b, tb_int);
}

template <typename Real>
inline auto sync_distance_fp_pc(wasm_point_cloud<Real> &m,
                                wasm_ndarray<Real> &b, int tb_int)
    -> emscripten::val {
  return sync_fp_d<Real>(m, b, tb_int);
}

template <typename Real>
inline auto sync_distance_ff(wasm_mesh<Real> &m0, wasm_mesh<Real> &m1) -> Real {
  return ff_compute_d<Real>(m0, m1);
}

template <typename Real>
inline auto sync_distance_ff_mp(wasm_mesh<Real> &m0,
                                wasm_point_cloud<Real> &m1) -> Real {
  return ff_compute_d<Real>(m0, m1);
}

template <typename Real>
inline auto sync_distance_ff_pm(wasm_point_cloud<Real> &m0,
                                wasm_mesh<Real> &m1) -> Real {
  return ff_compute_d<Real>(m0, m1);
}

template <typename Real>
inline auto sync_distance_ff_pc(wasm_point_cloud<Real> &m0,
                                wasm_point_cloud<Real> &m1) -> Real {
  return ff_compute_d<Real>(m0, m1);
}

// ============================================================================
// Async FP dispatch — template on FormT
// ============================================================================

template <typename Real, typename FormT>
inline auto async_fp_d2(FormT &m, wasm_ndarray<Real> &b, int tb_int)
    -> promise_t {
  auto tb = static_cast<prim_type>(tb_int);

  if (!is_batch(b, tb)) {
    int vb = poly_verts(b, tb);
    return promise([m = m, b = b, tb, vb]() -> Real {
      return fp_single_d2<Real>(const_cast<FormT &>(m),
                                const_cast<wasm_ndarray<Real> &>(b).raw_data(),
                                tb, vb);
    });
  }

  return promise([m = m, b = b, tb]() -> wasm_ndarray<Real> {
    return fp_batch_d2<Real>(const_cast<FormT &>(m),
                             const_cast<wasm_ndarray<Real> &>(b), tb);
  });
}

template <typename Real, typename FormT>
inline auto async_fp_d(FormT &m, wasm_ndarray<Real> &b, int tb_int)
    -> promise_t {
  auto tb = static_cast<prim_type>(tb_int);

  if (!is_batch(b, tb)) {
    int vb = poly_verts(b, tb);
    return promise([m = m, b = b, tb, vb]() -> Real {
      return fp_single_d<Real>(const_cast<FormT &>(m),
                               const_cast<wasm_ndarray<Real> &>(b).raw_data(),
                               tb, vb);
    });
  }

  return promise([m = m, b = b, tb]() -> wasm_ndarray<Real> {
    return fp_batch_d<Real>(const_cast<FormT &>(m),
                            const_cast<wasm_ndarray<Real> &>(b), tb);
  });
}

// ============================================================================
// Async entry points — distance2
// ============================================================================

template <typename Real>
inline auto async_distance2_pp(wasm_ndarray<Real> &a, int ta_int,
                               wasm_ndarray<Real> &b, int tb_int) -> promise_t {
  auto ta = static_cast<prim_type>(ta_int);
  auto tb = static_cast<prim_type>(tb_int);

  if (!is_batch(a, ta) && !is_batch(b, tb)) {
    int va = poly_verts(a, ta);
    int vb = poly_verts(b, tb);
    return promise([a = a, b = b, ta, tb, va, vb]() -> Real {
      return pp_single_d2<Real>(
          const_cast<wasm_ndarray<Real> &>(a).raw_data(), ta,
          const_cast<wasm_ndarray<Real> &>(b).raw_data(), tb, va, vb);
    });
  }

  return promise([a = a, b = b, ta, tb]() -> wasm_ndarray<Real> {
    return pp_batch_d2<Real>(const_cast<wasm_ndarray<Real> &>(a), ta,
                             const_cast<wasm_ndarray<Real> &>(b), tb);
  });
}

template <typename Real>
inline auto async_distance2_fp(wasm_mesh<Real> &m, wasm_ndarray<Real> &b,
                               int tb_int) -> promise_t {
  return async_fp_d2<Real>(m, b, tb_int);
}

template <typename Real>
inline auto async_distance2_fp_pc(wasm_point_cloud<Real> &m,
                                  wasm_ndarray<Real> &b, int tb_int)
    -> promise_t {
  return async_fp_d2<Real>(m, b, tb_int);
}

template <typename Real>
inline auto async_distance2_ff(wasm_mesh<Real> &m0, wasm_mesh<Real> &m1)
    -> promise_t {
  return promise([a = m0, b = m1]() -> Real {
    return ff_compute_d2<Real>(const_cast<wasm_mesh<Real> &>(a),
                               const_cast<wasm_mesh<Real> &>(b));
  });
}

template <typename Real>
inline auto async_distance2_ff_mp(wasm_mesh<Real> &m0,
                                  wasm_point_cloud<Real> &m1) -> promise_t {
  return promise([a = m0, b = m1]() -> Real {
    return ff_compute_d2<Real>(const_cast<wasm_mesh<Real> &>(a),
                               const_cast<wasm_point_cloud<Real> &>(b));
  });
}

template <typename Real>
inline auto async_distance2_ff_pm(wasm_point_cloud<Real> &m0,
                                  wasm_mesh<Real> &m1) -> promise_t {
  return promise([a = m0, b = m1]() -> Real {
    return ff_compute_d2<Real>(const_cast<wasm_point_cloud<Real> &>(a),
                               const_cast<wasm_mesh<Real> &>(b));
  });
}

template <typename Real>
inline auto async_distance2_ff_pc(wasm_point_cloud<Real> &m0,
                                  wasm_point_cloud<Real> &m1) -> promise_t {
  return promise([a = m0, b = m1]() -> Real {
    return ff_compute_d2<Real>(const_cast<wasm_point_cloud<Real> &>(a),
                               const_cast<wasm_point_cloud<Real> &>(b));
  });
}

// ============================================================================
// Async entry points — distance
// ============================================================================

template <typename Real>
inline auto async_distance_pp(wasm_ndarray<Real> &a, int ta_int,
                              wasm_ndarray<Real> &b, int tb_int) -> promise_t {
  auto ta = static_cast<prim_type>(ta_int);
  auto tb = static_cast<prim_type>(tb_int);

  if (!is_batch(a, ta) && !is_batch(b, tb)) {
    int va = poly_verts(a, ta);
    int vb = poly_verts(b, tb);
    return promise([a = a, b = b, ta, tb, va, vb]() -> Real {
      return pp_single_d<Real>(
          const_cast<wasm_ndarray<Real> &>(a).raw_data(), ta,
          const_cast<wasm_ndarray<Real> &>(b).raw_data(), tb, va, vb);
    });
  }

  return promise([a = a, b = b, ta, tb]() -> wasm_ndarray<Real> {
    return pp_batch_d<Real>(const_cast<wasm_ndarray<Real> &>(a), ta,
                            const_cast<wasm_ndarray<Real> &>(b), tb);
  });
}

template <typename Real>
inline auto async_distance_fp(wasm_mesh<Real> &m, wasm_ndarray<Real> &b,
                              int tb_int) -> promise_t {
  return async_fp_d<Real>(m, b, tb_int);
}

template <typename Real>
inline auto async_distance_fp_pc(wasm_point_cloud<Real> &m,
                                 wasm_ndarray<Real> &b, int tb_int)
    -> promise_t {
  return async_fp_d<Real>(m, b, tb_int);
}

template <typename Real>
inline auto async_distance_ff(wasm_mesh<Real> &m0, wasm_mesh<Real> &m1)
    -> promise_t {
  return promise([a = m0, b = m1]() -> Real {
    return ff_compute_d<Real>(const_cast<wasm_mesh<Real> &>(a),
                              const_cast<wasm_mesh<Real> &>(b));
  });
}

template <typename Real>
inline auto async_distance_ff_mp(wasm_mesh<Real> &m0,
                                 wasm_point_cloud<Real> &m1) -> promise_t {
  return promise([a = m0, b = m1]() -> Real {
    return ff_compute_d<Real>(const_cast<wasm_mesh<Real> &>(a),
                              const_cast<wasm_point_cloud<Real> &>(b));
  });
}

template <typename Real>
inline auto async_distance_ff_pm(wasm_point_cloud<Real> &m0,
                                 wasm_mesh<Real> &m1) -> promise_t {
  return promise([a = m0, b = m1]() -> Real {
    return ff_compute_d<Real>(const_cast<wasm_point_cloud<Real> &>(a),
                              const_cast<wasm_mesh<Real> &>(b));
  });
}

template <typename Real>
inline auto async_distance_ff_pc(wasm_point_cloud<Real> &m0,
                                 wasm_point_cloud<Real> &m1) -> promise_t {
  return promise([a = m0, b = m1]() -> Real {
    return ff_compute_d<Real>(const_cast<wasm_point_cloud<Real> &>(a),
                              const_cast<wasm_point_cloud<Real> &>(b));
  });
}

} // namespace ts
} // namespace tf
