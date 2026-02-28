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
#include "trueform/core/intersects.hpp"
#include "trueform/core/views/sequence_range.hpp"
#include "trueform/spatial/intersects.hpp"
#include "trueform/ts/core/promise.hpp"
#include "trueform/ts/core/wasm_mesh.hpp"
#include "trueform/ts/core/wasm_ndarray.hpp"
#include "trueform/ts/core/wasm_point_cloud.hpp"
#include "trueform/ts/spatial/prim_dispatch.hpp"
#include <emscripten/bind.h>

namespace {

using namespace tf::ts;

// ============================================================================
// PP helpers (prim × prim)
// ============================================================================

auto pp_single(const float *a, prim_type ta, const float *b, prim_type tb,
               int va, int vb) -> bool {
  return dispatch_pair(
      [](const auto &pa, const auto &pb) -> bool {
        return tf::intersects(pa, pb);
      },
      a, ta, b, tb, va, vb);
}

auto pp_batch(wasm_ndarray<float> &a, prim_type ta, wasm_ndarray<float> &b,
              prim_type tb) -> wasm_ndarray<std::int8_t> {
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

  tf::buffer<std::int8_t> out;
  out.allocate(n);
  std::int8_t *dst = out.data();

  auto compute = [=](int i) {
    dst[i] = pp_single(pa + i * stride_a, ta, pb + i * stride_b, tb, va, vb)
                 ? 1
                 : 0;
  };

  if (n >= 1000)
    tf::parallel_for_each(tf::make_sequence_range(n), compute);
  else
    for (int i = 0; i < n; ++i)
      compute(i);

  return wasm_ndarray<std::int8_t>::from_buffer(std::move(out), {n});
}

// ============================================================================
// FP helpers (form × prim) — templated on FormT
// ============================================================================

template <typename FormT>
auto fp_single(FormT &m, const float *b, prim_type tb, int vb) -> bool {
  return m.with_form([&](const auto &form) -> bool {
    return dispatch_single(
        [&](const auto &pb) -> bool { return tf::intersects(form, pb); }, b, tb,
        vb);
  });
}

template <typename FormT>
auto fp_batch(FormT &m, wasm_ndarray<float> &b,
              prim_type tb) -> wasm_ndarray<std::int8_t> {
  int vb = poly_verts(b, tb);
  int sb = stride_of(b, tb);
  int n = batch_count(b, tb);
  const float *pb = b.raw_data();

  tf::buffer<std::int8_t> out;
  out.allocate(n);
  std::int8_t *dst = out.data();

  m.with_form([&](const auto &form) {
    auto compute = [&](int i) {
      dst[i] = dispatch_single(
                   [&](const auto &prim) -> bool {
                     return tf::intersects(form, prim);
                   },
                   pb + i * sb, tb, vb)
                   ? 1
                   : 0;
    };
    if (n >= 100)
      tf::parallel_for_each(tf::make_sequence_range(n), compute);
    else
      for (int i = 0; i < n; ++i)
        compute(i);
  });

  return wasm_ndarray<std::int8_t>::from_buffer(std::move(out), {n});
}

// ============================================================================
// FF helper (form × form) — templated on F0, F1
// ============================================================================

template <typename F0, typename F1>
auto ff_compute(F0 &m0, F1 &m1) -> bool {
  return m0.with_form([&](const auto &form0) -> bool {
    return m1.with_form([&](const auto &form1) -> bool {
      return tf::intersects(form0, form1);
    });
  });
}

// ============================================================================
// Sync FP dispatch — template on FormT
// ============================================================================

template <typename FormT>
auto sync_fp(FormT &m, wasm_ndarray<float> &b, int tb_int) -> emscripten::val {
  auto tb = static_cast<prim_type>(tb_int);
  if (!is_batch(b, tb))
    return emscripten::val(fp_single(m, b.raw_data(), tb, poly_verts(b, tb)));
  return emscripten::val(fp_batch(m, b, tb));
}

// ============================================================================
// Sync entry points
// ============================================================================

// PP
auto sync_intersects_pp(wasm_ndarray<float> &a, int ta_int,
                        wasm_ndarray<float> &b,
                        int tb_int) -> emscripten::val {
  auto ta = static_cast<prim_type>(ta_int);
  auto tb = static_cast<prim_type>(tb_int);
  if (!is_batch(a, ta) && !is_batch(b, tb))
    return emscripten::val(
        pp_single(a.raw_data(), ta, b.raw_data(), tb, poly_verts(a, ta),
                  poly_verts(b, tb)));
  return emscripten::val(pp_batch(a, ta, b, tb));
}

// FP thin wrappers
auto sync_intersects_fp(wasm_mesh &m, wasm_ndarray<float> &b,
                        int tb) -> emscripten::val {
  return sync_fp(m, b, tb);
}
auto sync_intersects_fp_pc(wasm_point_cloud &m, wasm_ndarray<float> &b,
                           int tb) -> emscripten::val {
  return sync_fp(m, b, tb);
}

// FF thin wrappers
auto sync_intersects_ff(wasm_mesh &m0, wasm_mesh &m1) -> bool {
  return ff_compute(m0, m1);
}
auto sync_intersects_ff_mp(wasm_mesh &m0, wasm_point_cloud &m1) -> bool {
  return ff_compute(m0, m1);
}
auto sync_intersects_ff_pm(wasm_point_cloud &m0, wasm_mesh &m1) -> bool {
  return ff_compute(m0, m1);
}
auto sync_intersects_ff_pc(wasm_point_cloud &m0,
                           wasm_point_cloud &m1) -> bool {
  return ff_compute(m0, m1);
}

// ============================================================================
// Async FP dispatch — template on FormT
// ============================================================================

template <typename FormT>
auto async_fp(FormT &m, wasm_ndarray<float> &b, int tb_int) -> promise_t {
  auto tb = static_cast<prim_type>(tb_int);

  if (!is_batch(b, tb)) {
    int vb = poly_verts(b, tb);
    return promise([m = m, b = b, tb, vb]() -> bool {
      return fp_single(const_cast<FormT &>(m),
                       const_cast<wasm_ndarray<float> &>(b).raw_data(), tb, vb);
    });
  }

  return promise(
      [m = m, b = b, tb]() -> wasm_ndarray<std::int8_t> {
        return fp_batch(const_cast<FormT &>(m),
                        const_cast<wasm_ndarray<float> &>(b), tb);
      });
}

// ============================================================================
// Async entry points
// ============================================================================

// PP
auto async_intersects_pp(wasm_ndarray<float> &a, int ta_int,
                         wasm_ndarray<float> &b,
                         int tb_int) -> promise_t {
  auto ta = static_cast<prim_type>(ta_int);
  auto tb = static_cast<prim_type>(tb_int);

  if (!is_batch(a, ta) && !is_batch(b, tb)) {
    int va = poly_verts(a, ta);
    int vb = poly_verts(b, tb);
    return promise([a = a, b = b, ta, tb, va, vb]() -> bool {
      return pp_single(const_cast<wasm_ndarray<float> &>(a).raw_data(), ta,
                       const_cast<wasm_ndarray<float> &>(b).raw_data(), tb, va,
                       vb);
    });
  }

  return promise(
      [a = a, b = b, ta, tb]() -> wasm_ndarray<std::int8_t> {
        return pp_batch(const_cast<wasm_ndarray<float> &>(a), ta,
                        const_cast<wasm_ndarray<float> &>(b), tb);
      });
}

// FP thin wrappers
auto async_intersects_fp(wasm_mesh &m, wasm_ndarray<float> &b,
                         int tb) -> promise_t {
  return async_fp(m, b, tb);
}
auto async_intersects_fp_pc(wasm_point_cloud &m, wasm_ndarray<float> &b,
                            int tb) -> promise_t {
  return async_fp(m, b, tb);
}

// FF — template dispatch
template <typename F0, typename F1>
auto async_ff(F0 &m0, F1 &m1) -> promise_t {
  return promise([a = m0, b = m1]() -> bool {
    return ff_compute(const_cast<F0 &>(a), const_cast<F1 &>(b));
  });
}

// FF thin wrappers
auto async_intersects_ff(wasm_mesh &m0, wasm_mesh &m1) -> promise_t {
  return async_ff(m0, m1);
}
auto async_intersects_ff_mp(wasm_mesh &m0, wasm_point_cloud &m1) -> promise_t {
  return async_ff(m0, m1);
}
auto async_intersects_ff_pm(wasm_point_cloud &m0, wasm_mesh &m1) -> promise_t {
  return async_ff(m0, m1);
}
auto async_intersects_ff_pc(wasm_point_cloud &m0,
                            wasm_point_cloud &m1) -> promise_t {
  return async_ff(m0, m1);
}

} // namespace

EMSCRIPTEN_BINDINGS(trueform_intersects) {
  // PP
  emscripten::function("intersects_pp", &sync_intersects_pp);
  emscripten::function("dispatch_intersects_pp", &async_intersects_pp);

  // FP — mesh & point cloud
  emscripten::function("intersects_fp", &sync_intersects_fp);
  emscripten::function("intersects_fp_pc", &sync_intersects_fp_pc);
  emscripten::function("dispatch_intersects_fp", &async_intersects_fp);
  emscripten::function("dispatch_intersects_fp_pc", &async_intersects_fp_pc);

  // FF — all 4 combos
  emscripten::function("intersects_ff", &sync_intersects_ff);
  emscripten::function("intersects_ff_mp", &sync_intersects_ff_mp);
  emscripten::function("intersects_ff_pm", &sync_intersects_ff_pm);
  emscripten::function("intersects_ff_pc", &sync_intersects_ff_pc);
  emscripten::function("dispatch_intersects_ff", &async_intersects_ff);
  emscripten::function("dispatch_intersects_ff_mp", &async_intersects_ff_mp);
  emscripten::function("dispatch_intersects_ff_pm", &async_intersects_ff_pm);
  emscripten::function("dispatch_intersects_ff_pc", &async_intersects_ff_pc);
}
