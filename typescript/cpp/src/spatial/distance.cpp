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
#include "trueform/core/distance.hpp"
#include "trueform/core/views/sequence_range.hpp"
#include "trueform/spatial/distance.hpp"
#include "trueform/ts/core/promise.hpp"
#include "trueform/ts/core/wasm_mesh.hpp"
#include "trueform/ts/core/wasm_ndarray.hpp"
#include "trueform/ts/core/wasm_point_cloud.hpp"
#include "trueform/ts/spatial/prim_dispatch.hpp"
#include <emscripten/bind.h>

namespace {

using namespace tf::ts;

// ============================================================================
// PP helpers — distance2 (prim × prim)
// ============================================================================

auto pp_single_d2(const float *a, prim_type ta, const float *b, prim_type tb,
                  int va, int vb) -> float {
  return dispatch_pair(
      [](const auto &pa, const auto &pb) -> float {
        return tf::distance2(pa, pb);
      },
      a, ta, b, tb, va, vb);
}

auto pp_batch_d2(wasm_ndarray<float> &a, prim_type ta, wasm_ndarray<float> &b,
                 prim_type tb) -> wasm_ndarray<float> {
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

  tf::buffer<float> out;
  out.allocate(n);
  float *dst = out.data();

  auto compute = [=](int i) {
    dst[i] =
        pp_single_d2(pa + i * stride_a, ta, pb + i * stride_b, tb, va, vb);
  };

  if (n >= 5000)
    tf::parallel_for_each(tf::make_sequence_range(n), compute);
  else
    for (int i = 0; i < n; ++i)
      compute(i);

  return wasm_ndarray<float>::from_buffer(std::move(out), {n});
}

// ============================================================================
// PP helpers — distance (prim × prim)
// ============================================================================

auto pp_single_d(const float *a, prim_type ta, const float *b, prim_type tb,
                 int va, int vb) -> float {
  return dispatch_pair(
      [](const auto &pa, const auto &pb) -> float {
        return tf::distance(pa, pb);
      },
      a, ta, b, tb, va, vb);
}

auto pp_batch_d(wasm_ndarray<float> &a, prim_type ta, wasm_ndarray<float> &b,
                prim_type tb) -> wasm_ndarray<float> {
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

  tf::buffer<float> out;
  out.allocate(n);
  float *dst = out.data();

  auto compute = [=](int i) {
    dst[i] =
        pp_single_d(pa + i * stride_a, ta, pb + i * stride_b, tb, va, vb);
  };

  if (n >= 5000)
    tf::parallel_for_each(tf::make_sequence_range(n), compute);
  else
    for (int i = 0; i < n; ++i)
      compute(i);

  return wasm_ndarray<float>::from_buffer(std::move(out), {n});
}

// ============================================================================
// FP helpers — distance2 (form × prim)
// ============================================================================

template <typename FormT>
auto fp_single_d2(FormT &m, const float *b, prim_type tb,
                  int vb) -> float {
  return m.with_form([&](const auto &form) -> float {
    return dispatch_single(
        [&](const auto &pb) -> float { return tf::distance2(form, pb); }, b,
        tb, vb);
  });
}

template <typename FormT>
auto fp_batch_d2(FormT &m, wasm_ndarray<float> &b,
                 prim_type tb) -> wasm_ndarray<float> {
  int vb = poly_verts(b, tb);
  int sb = stride_of(b, tb);
  int n = batch_count(b, tb);
  const float *pb = b.raw_data();

  tf::buffer<float> out;
  out.allocate(n);
  float *dst = out.data();

  m.with_form([&](const auto &form) {
    auto compute = [&](int i) {
      dst[i] = dispatch_single(
          [&](const auto &prim) -> float { return tf::distance2(form, prim); },
          pb + i * sb, tb, vb);
    };
    if (n >= 100)
      tf::parallel_for_each(tf::make_sequence_range(n), compute);
    else
      for (int i = 0; i < n; ++i)
        compute(i);
  });

  return wasm_ndarray<float>::from_buffer(std::move(out), {n});
}

// ============================================================================
// FP helpers — distance (form × prim)
// ============================================================================

template <typename FormT>
auto fp_single_d(FormT &m, const float *b, prim_type tb,
                 int vb) -> float {
  return m.with_form([&](const auto &form) -> float {
    return dispatch_single(
        [&](const auto &pb) -> float { return tf::distance(form, pb); }, b, tb,
        vb);
  });
}

template <typename FormT>
auto fp_batch_d(FormT &m, wasm_ndarray<float> &b,
                prim_type tb) -> wasm_ndarray<float> {
  int vb = poly_verts(b, tb);
  int sb = stride_of(b, tb);
  int n = batch_count(b, tb);
  const float *pb = b.raw_data();

  tf::buffer<float> out;
  out.allocate(n);
  float *dst = out.data();

  m.with_form([&](const auto &form) {
    auto compute = [&](int i) {
      dst[i] = dispatch_single(
          [&](const auto &prim) -> float { return tf::distance(form, prim); },
          pb + i * sb, tb, vb);
    };
    if (n >= 100)
      tf::parallel_for_each(tf::make_sequence_range(n), compute);
    else
      for (int i = 0; i < n; ++i)
        compute(i);
  });

  return wasm_ndarray<float>::from_buffer(std::move(out), {n});
}

// ============================================================================
// FF helpers — distance2 (form × form)
// ============================================================================

template <typename F0, typename F1>
auto ff_compute_d2(F0 &m0, F1 &m1) -> float {
  return m0.with_form([&](const auto &form0) -> float {
    return m1.with_form([&](const auto &form1) -> float {
      return tf::distance2(form0, form1);
    });
  });
}

// ============================================================================
// FF helpers — distance (form × form)
// ============================================================================

template <typename F0, typename F1>
auto ff_compute_d(F0 &m0, F1 &m1) -> float {
  return m0.with_form([&](const auto &form0) -> float {
    return m1.with_form([&](const auto &form1) -> float {
      return tf::distance(form0, form1);
    });
  });
}

// ============================================================================
// Sync FP dispatch — template on FormT
// ============================================================================

template <typename FormT>
auto sync_fp_d2(FormT &m, wasm_ndarray<float> &b,
                int tb_int) -> emscripten::val {
  auto tb = static_cast<prim_type>(tb_int);
  if (!is_batch(b, tb))
    return emscripten::val(
        fp_single_d2(m, b.raw_data(), tb, poly_verts(b, tb)));
  return emscripten::val(fp_batch_d2(m, b, tb));
}

template <typename FormT>
auto sync_fp_d(FormT &m, wasm_ndarray<float> &b,
               int tb_int) -> emscripten::val {
  auto tb = static_cast<prim_type>(tb_int);
  if (!is_batch(b, tb))
    return emscripten::val(
        fp_single_d(m, b.raw_data(), tb, poly_verts(b, tb)));
  return emscripten::val(fp_batch_d(m, b, tb));
}

// ============================================================================
// Sync entry points — distance2
// ============================================================================

auto sync_distance2_pp(wasm_ndarray<float> &a, int ta_int,
                       wasm_ndarray<float> &b,
                       int tb_int) -> emscripten::val {
  auto ta = static_cast<prim_type>(ta_int);
  auto tb = static_cast<prim_type>(tb_int);
  if (!is_batch(a, ta) && !is_batch(b, tb))
    return emscripten::val(
        pp_single_d2(a.raw_data(), ta, b.raw_data(), tb, poly_verts(a, ta),
                     poly_verts(b, tb)));
  return emscripten::val(pp_batch_d2(a, ta, b, tb));
}

// FP mesh
auto sync_distance2_fp(wasm_mesh &m, wasm_ndarray<float> &b,
                       int tb_int) -> emscripten::val {
  return sync_fp_d2(m, b, tb_int);
}

// FP point cloud
auto sync_distance2_fp_pc(wasm_point_cloud &m, wasm_ndarray<float> &b,
                          int tb_int) -> emscripten::val {
  return sync_fp_d2(m, b, tb_int);
}

// FF — all 4 combos
auto sync_distance2_ff(wasm_mesh &m0, wasm_mesh &m1) -> float {
  return ff_compute_d2(m0, m1);
}
auto sync_distance2_ff_mp(wasm_mesh &m0, wasm_point_cloud &m1) -> float {
  return ff_compute_d2(m0, m1);
}
auto sync_distance2_ff_pm(wasm_point_cloud &m0, wasm_mesh &m1) -> float {
  return ff_compute_d2(m0, m1);
}
auto sync_distance2_ff_pc(wasm_point_cloud &m0,
                          wasm_point_cloud &m1) -> float {
  return ff_compute_d2(m0, m1);
}

// ============================================================================
// Sync entry points — distance
// ============================================================================

auto sync_distance_pp(wasm_ndarray<float> &a, int ta_int,
                      wasm_ndarray<float> &b,
                      int tb_int) -> emscripten::val {
  auto ta = static_cast<prim_type>(ta_int);
  auto tb = static_cast<prim_type>(tb_int);
  if (!is_batch(a, ta) && !is_batch(b, tb))
    return emscripten::val(
        pp_single_d(a.raw_data(), ta, b.raw_data(), tb, poly_verts(a, ta),
                    poly_verts(b, tb)));
  return emscripten::val(pp_batch_d(a, ta, b, tb));
}

// FP mesh
auto sync_distance_fp(wasm_mesh &m, wasm_ndarray<float> &b,
                      int tb_int) -> emscripten::val {
  return sync_fp_d(m, b, tb_int);
}

// FP point cloud
auto sync_distance_fp_pc(wasm_point_cloud &m, wasm_ndarray<float> &b,
                         int tb_int) -> emscripten::val {
  return sync_fp_d(m, b, tb_int);
}

// FF — all 4 combos
auto sync_distance_ff(wasm_mesh &m0, wasm_mesh &m1) -> float {
  return ff_compute_d(m0, m1);
}
auto sync_distance_ff_mp(wasm_mesh &m0, wasm_point_cloud &m1) -> float {
  return ff_compute_d(m0, m1);
}
auto sync_distance_ff_pm(wasm_point_cloud &m0, wasm_mesh &m1) -> float {
  return ff_compute_d(m0, m1);
}
auto sync_distance_ff_pc(wasm_point_cloud &m0,
                         wasm_point_cloud &m1) -> float {
  return ff_compute_d(m0, m1);
}

// ============================================================================
// Async FP dispatch — template on FormT
// ============================================================================

template <typename FormT>
auto async_fp_d2(FormT &m, wasm_ndarray<float> &b,
                 int tb_int) -> promise_t {
  auto tb = static_cast<prim_type>(tb_int);

  if (!is_batch(b, tb)) {
    int vb = poly_verts(b, tb);
    return promise([m = m, b = b, tb, vb]() -> float {
      return fp_single_d2(const_cast<FormT &>(m),
                          const_cast<wasm_ndarray<float> &>(b).raw_data(), tb,
                          vb);
    });
  }

  return promise(
      [m = m, b = b, tb]() -> wasm_ndarray<float> {
        return fp_batch_d2(const_cast<FormT &>(m),
                           const_cast<wasm_ndarray<float> &>(b), tb);
      });
}

template <typename FormT>
auto async_fp_d(FormT &m, wasm_ndarray<float> &b,
                int tb_int) -> promise_t {
  auto tb = static_cast<prim_type>(tb_int);

  if (!is_batch(b, tb)) {
    int vb = poly_verts(b, tb);
    return promise([m = m, b = b, tb, vb]() -> float {
      return fp_single_d(const_cast<FormT &>(m),
                         const_cast<wasm_ndarray<float> &>(b).raw_data(), tb,
                         vb);
    });
  }

  return promise(
      [m = m, b = b, tb]() -> wasm_ndarray<float> {
        return fp_batch_d(const_cast<FormT &>(m),
                          const_cast<wasm_ndarray<float> &>(b), tb);
      });
}

// ============================================================================
// Async entry points — distance2
// ============================================================================

auto async_distance2_pp(wasm_ndarray<float> &a, int ta_int,
                        wasm_ndarray<float> &b,
                        int tb_int) -> promise_t {
  auto ta = static_cast<prim_type>(ta_int);
  auto tb = static_cast<prim_type>(tb_int);

  if (!is_batch(a, ta) && !is_batch(b, tb)) {
    int va = poly_verts(a, ta);
    int vb = poly_verts(b, tb);
    return promise([a = a, b = b, ta, tb, va, vb]() -> float {
      return pp_single_d2(const_cast<wasm_ndarray<float> &>(a).raw_data(), ta,
                          const_cast<wasm_ndarray<float> &>(b).raw_data(), tb,
                          va, vb);
    });
  }

  return promise(
      [a = a, b = b, ta, tb]() -> wasm_ndarray<float> {
        return pp_batch_d2(const_cast<wasm_ndarray<float> &>(a), ta,
                           const_cast<wasm_ndarray<float> &>(b), tb);
      });
}

// FP mesh
auto async_distance2_fp(wasm_mesh &m, wasm_ndarray<float> &b,
                        int tb_int) -> promise_t {
  return async_fp_d2(m, b, tb_int);
}

// FP point cloud
auto async_distance2_fp_pc(wasm_point_cloud &m, wasm_ndarray<float> &b,
                           int tb_int) -> promise_t {
  return async_fp_d2(m, b, tb_int);
}

// FF — all 4 combos
auto async_distance2_ff(wasm_mesh &m0, wasm_mesh &m1) -> promise_t {
  return promise([a = m0, b = m1]() -> float {
    return ff_compute_d2(const_cast<wasm_mesh &>(a),
                         const_cast<wasm_mesh &>(b));
  });
}
auto async_distance2_ff_mp(wasm_mesh &m0, wasm_point_cloud &m1) -> promise_t {
  return promise([a = m0, b = m1]() -> float {
    return ff_compute_d2(const_cast<wasm_mesh &>(a),
                         const_cast<wasm_point_cloud &>(b));
  });
}
auto async_distance2_ff_pm(wasm_point_cloud &m0, wasm_mesh &m1) -> promise_t {
  return promise([a = m0, b = m1]() -> float {
    return ff_compute_d2(const_cast<wasm_point_cloud &>(a),
                         const_cast<wasm_mesh &>(b));
  });
}
auto async_distance2_ff_pc(wasm_point_cloud &m0,
                           wasm_point_cloud &m1) -> promise_t {
  return promise([a = m0, b = m1]() -> float {
    return ff_compute_d2(const_cast<wasm_point_cloud &>(a),
                         const_cast<wasm_point_cloud &>(b));
  });
}

// ============================================================================
// Async entry points — distance
// ============================================================================

auto async_distance_pp(wasm_ndarray<float> &a, int ta_int,
                       wasm_ndarray<float> &b,
                       int tb_int) -> promise_t {
  auto ta = static_cast<prim_type>(ta_int);
  auto tb = static_cast<prim_type>(tb_int);

  if (!is_batch(a, ta) && !is_batch(b, tb)) {
    int va = poly_verts(a, ta);
    int vb = poly_verts(b, tb);
    return promise([a = a, b = b, ta, tb, va, vb]() -> float {
      return pp_single_d(const_cast<wasm_ndarray<float> &>(a).raw_data(), ta,
                         const_cast<wasm_ndarray<float> &>(b).raw_data(), tb,
                         va, vb);
    });
  }

  return promise(
      [a = a, b = b, ta, tb]() -> wasm_ndarray<float> {
        return pp_batch_d(const_cast<wasm_ndarray<float> &>(a), ta,
                          const_cast<wasm_ndarray<float> &>(b), tb);
      });
}

// FP mesh
auto async_distance_fp(wasm_mesh &m, wasm_ndarray<float> &b,
                       int tb_int) -> promise_t {
  return async_fp_d(m, b, tb_int);
}

// FP point cloud
auto async_distance_fp_pc(wasm_point_cloud &m, wasm_ndarray<float> &b,
                          int tb_int) -> promise_t {
  return async_fp_d(m, b, tb_int);
}

// FF — all 4 combos
auto async_distance_ff(wasm_mesh &m0, wasm_mesh &m1) -> promise_t {
  return promise([a = m0, b = m1]() -> float {
    return ff_compute_d(const_cast<wasm_mesh &>(a),
                        const_cast<wasm_mesh &>(b));
  });
}
auto async_distance_ff_mp(wasm_mesh &m0, wasm_point_cloud &m1) -> promise_t {
  return promise([a = m0, b = m1]() -> float {
    return ff_compute_d(const_cast<wasm_mesh &>(a),
                        const_cast<wasm_point_cloud &>(b));
  });
}
auto async_distance_ff_pm(wasm_point_cloud &m0, wasm_mesh &m1) -> promise_t {
  return promise([a = m0, b = m1]() -> float {
    return ff_compute_d(const_cast<wasm_point_cloud &>(a),
                        const_cast<wasm_mesh &>(b));
  });
}
auto async_distance_ff_pc(wasm_point_cloud &m0,
                          wasm_point_cloud &m1) -> promise_t {
  return promise([a = m0, b = m1]() -> float {
    return ff_compute_d(const_cast<wasm_point_cloud &>(a),
                        const_cast<wasm_point_cloud &>(b));
  });
}

} // namespace

EMSCRIPTEN_BINDINGS(trueform_distance) {
  // PP
  emscripten::function("distance2_pp", &sync_distance2_pp);
  emscripten::function("dispatch_distance2_pp", &async_distance2_pp);
  emscripten::function("distance_pp", &sync_distance_pp);
  emscripten::function("dispatch_distance_pp", &async_distance_pp);

  // FP — mesh & point cloud
  emscripten::function("distance2_fp", &sync_distance2_fp);
  emscripten::function("distance2_fp_pc", &sync_distance2_fp_pc);
  emscripten::function("dispatch_distance2_fp", &async_distance2_fp);
  emscripten::function("dispatch_distance2_fp_pc", &async_distance2_fp_pc);

  emscripten::function("distance_fp", &sync_distance_fp);
  emscripten::function("distance_fp_pc", &sync_distance_fp_pc);
  emscripten::function("dispatch_distance_fp", &async_distance_fp);
  emscripten::function("dispatch_distance_fp_pc", &async_distance_fp_pc);

  // FF — all 4 combos
  emscripten::function("distance2_ff", &sync_distance2_ff);
  emscripten::function("distance2_ff_mp", &sync_distance2_ff_mp);
  emscripten::function("distance2_ff_pm", &sync_distance2_ff_pm);
  emscripten::function("distance2_ff_pc", &sync_distance2_ff_pc);
  emscripten::function("dispatch_distance2_ff", &async_distance2_ff);
  emscripten::function("dispatch_distance2_ff_mp", &async_distance2_ff_mp);
  emscripten::function("dispatch_distance2_ff_pm", &async_distance2_ff_pm);
  emscripten::function("dispatch_distance2_ff_pc", &async_distance2_ff_pc);

  emscripten::function("distance_ff", &sync_distance_ff);
  emscripten::function("distance_ff_mp", &sync_distance_ff_mp);
  emscripten::function("distance_ff_pm", &sync_distance_ff_pm);
  emscripten::function("distance_ff_pc", &sync_distance_ff_pc);
  emscripten::function("dispatch_distance_ff", &async_distance_ff);
  emscripten::function("dispatch_distance_ff_mp", &async_distance_ff_mp);
  emscripten::function("dispatch_distance_ff_pm", &async_distance_ff_pm);
  emscripten::function("dispatch_distance_ff_pc", &async_distance_ff_pc);
}
