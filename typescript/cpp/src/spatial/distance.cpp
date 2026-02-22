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
#include "trueform/ts/spatial/prim_dispatch.hpp"
#include <emscripten/bind.h>

namespace {

using namespace tf::ts;

// ============================================================================
// PP helpers (prim × prim)
// ============================================================================

auto pp_single(const float *a, prim_type ta, const float *b, prim_type tb,
               int va, int vb) -> float {
  return dispatch_pair(
      [](const auto &pa, const auto &pb) -> float {
        return tf::distance2(pa, pb);
      },
      a, ta, b, tb, va, vb);
}

auto pp_batch(wasm_ndarray<float> &a, prim_type ta, wasm_ndarray<float> &b,
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
        pp_single(pa + i * stride_a, ta, pb + i * stride_b, tb, va, vb);
  };

  if (n >= 5000)
    tf::parallel_for_each(tf::make_sequence_range(n), compute);
  else
    for (int i = 0; i < n; ++i)
      compute(i);

  return wasm_ndarray<float>::from_buffer(std::move(out), {n});
}

// ============================================================================
// FP helpers (form × prim)
// ============================================================================

auto fp_single(wasm_mesh &m, const float *b, prim_type tb, int vb) -> float {
  return m.with_form([&](const auto &form) -> float {
    return dispatch_single(
        [&](const auto &pb) -> float { return tf::distance2(form, pb); }, b,
        tb, vb);
  });
}

auto fp_batch(wasm_mesh &m, wasm_ndarray<float> &b,
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
// FF helper (form × form)
// ============================================================================

auto ff_compute(wasm_mesh &m0, wasm_mesh &m1) -> float {
  return m0.with_form([&](const auto &form0) -> float {
    return m1.with_form([&](const auto &form1) -> float {
      return tf::distance2(form0, form1);
    });
  });
}

// ============================================================================
// Sync entry points
// ============================================================================

auto sync_distance2_pp(wasm_ndarray<float> &a, int ta_int,
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

auto sync_distance2_fp(wasm_mesh &m, wasm_ndarray<float> &b,
                       int tb_int) -> emscripten::val {
  auto tb = static_cast<prim_type>(tb_int);
  if (!is_batch(b, tb))
    return emscripten::val(
        fp_single(m, b.raw_data(), tb, poly_verts(b, tb)));
  return emscripten::val(fp_batch(m, b, tb));
}

auto sync_distance2_ff(wasm_mesh &m0, wasm_mesh &m1) -> float {
  return ff_compute(m0, m1);
}

// ============================================================================
// Async entry points
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
      return pp_single(const_cast<wasm_ndarray<float> &>(a).raw_data(), ta,
                       const_cast<wasm_ndarray<float> &>(b).raw_data(), tb, va,
                       vb);
    });
  }

  return promise(
      [a = a, b = b, ta, tb]() -> wasm_ndarray<float> {
        return pp_batch(const_cast<wasm_ndarray<float> &>(a), ta,
                        const_cast<wasm_ndarray<float> &>(b), tb);
      });
}

auto async_distance2_fp(wasm_mesh &m, wasm_ndarray<float> &b,
                        int tb_int) -> promise_t {
  auto tb = static_cast<prim_type>(tb_int);

  if (!is_batch(b, tb)) {
    int vb = poly_verts(b, tb);
    return promise([m = m, b = b, tb, vb]() -> float {
      return fp_single(const_cast<wasm_mesh &>(m),
                       const_cast<wasm_ndarray<float> &>(b).raw_data(), tb, vb);
    });
  }

  return promise(
      [m = m, b = b, tb]() -> wasm_ndarray<float> {
        return fp_batch(const_cast<wasm_mesh &>(m),
                        const_cast<wasm_ndarray<float> &>(b), tb);
      });
}

auto async_distance2_ff(wasm_mesh &m0, wasm_mesh &m1) -> promise_t {
  return promise([a = m0, b = m1]() -> float {
    return ff_compute(const_cast<wasm_mesh &>(a), const_cast<wasm_mesh &>(b));
  });
}

} // namespace

EMSCRIPTEN_BINDINGS(trueform_distance) {
  emscripten::function("distance2_pp", &sync_distance2_pp);
  emscripten::function("distance2_fp", &sync_distance2_fp);
  emscripten::function("distance2_ff", &sync_distance2_ff);
  emscripten::function("dispatch_distance2_pp", &async_distance2_pp);
  emscripten::function("dispatch_distance2_fp", &async_distance2_fp);
  emscripten::function("dispatch_distance2_ff", &async_distance2_ff);
}
