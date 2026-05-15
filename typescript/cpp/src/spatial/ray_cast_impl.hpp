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
#include "trueform/core/ray_cast.hpp"
#include "trueform/core/views/constant.hpp"
#include "trueform/core/views/sequence_range.hpp"
#include "trueform/spatial/ray_cast.hpp"
#include "trueform/ts/core/promise.hpp"
#include "trueform/ts/core/wasm_mesh.hpp"
#include "trueform/ts/core/wasm_ndarray.hpp"
#include "trueform/ts/core/wasm_point_cloud.hpp"
#include "trueform/ts/spatial/prim_dispatch.hpp"
#include <cstdint>
#include <emscripten/bind.h>

namespace tf {
namespace ts {

// ============================================================================
// Templated result types
// ============================================================================

template <typename Real> struct ray_cast_result_t {
  bool hit;
  Real t;
  int element_id; // -1 for primitive targets
};

template <typename Real> struct ray_cast_prim_batch_result_t {
  wasm_ndarray<std::int8_t> hits; // [N]
  wasm_ndarray<Real> ts;          // [N]
};

template <typename Real> struct ray_cast_form_batch_result_t {
  wasm_ndarray<std::int8_t> hits; // [N]
  wasm_ndarray<Real> ts;          // [N]
  wasm_ndarray<int> element_ids;  // [N]
};

// ============================================================================
// Prim target helpers (manual dispatch — not all types support ray_cast)
// ============================================================================

template <typename Real>
inline auto rc_prim_single(const Real *ray_ptr, const Real *b, prim_type tb,
                           int vb, Real min_t, Real max_t)
    -> ray_cast_result_t<Real> {
  auto ray = as_ray(ray_ptr);
  auto config = tf::make_ray_config(min_t, max_t);
  switch (tb) {
  case prim_type::triangle: {
    auto r = tf::ray_cast(ray, as_triangle(b), config);
    return {bool(r), r.t, -1};
  }
  case prim_type::polygon: {
    auto r = tf::ray_cast(ray, as_polygon(b, vb), config);
    return {bool(r), r.t, -1};
  }
  case prim_type::plane: {
    auto r = tf::ray_cast(ray, as_plane(b), config);
    return {bool(r), r.t, -1};
  }
  case prim_type::segment: {
    auto r = tf::ray_cast(ray, as_segment(b), config);
    return {bool(r), r.t, -1};
  }
  case prim_type::line: {
    auto r = tf::ray_cast(ray, as_line(b), config);
    return {bool(r), r.t, -1};
  }
  case prim_type::ray: {
    auto r = tf::ray_cast(ray, as_ray(b), config);
    return {bool(r), r.t, -1};
  }
  case prim_type::point: {
    auto r = tf::ray_cast(ray, as_point(b), config);
    return {bool(r), r.t, -1};
  }
  case prim_type::aabb: {
    auto r = tf::ray_cast(ray, as_aabb(b), config);
    return {bool(r), r.t, -1};
  }
  default:
    return {false, Real(0), -1};
  }
}

template <typename Real, typename MinR, typename MaxR>
inline auto rc_prim_batch_impl(wasm_ndarray<Real> &rays, wasm_ndarray<Real> &b,
                               prim_type tb, MinR min_r, MaxR max_r)
    -> ray_cast_prim_batch_result_t<Real> {
  int vb = poly_verts(b, tb);
  int sb = stride_of(b, tb);
  bool b_rays = is_batch(rays, prim_type::ray);
  bool b_tgt = is_batch(b, tb);
  int n = b_rays ? batch_count(rays, prim_type::ray) : batch_count(b, tb);
  int stride_ray = b_rays ? 6 : 0;
  int stride_b = b_tgt ? sb : 0;

  const Real *pr = rays.raw_data();
  const Real *pb = b.raw_data();

  tf::buffer<std::int8_t> hits;
  tf::buffer<Real> ts;
  hits.allocate(n);
  ts.allocate(n);
  auto *h = hits.data();
  auto *t = ts.data();

  auto compute = [=](int i) {
    auto r = rc_prim_single<Real>(pr + i * stride_ray, pb + i * stride_b, tb,
                                  vb, min_r[i], max_r[i]);
    h[i] = r.hit ? 1 : 0;
    t[i] = r.t;
  };

  if (n >= 1000)
    tf::parallel_for_each(tf::make_sequence_range(n), compute);
  else
    for (int i = 0; i < n; ++i)
      compute(i);

  return {wasm_ndarray<std::int8_t>::from_buffer(std::move(hits), {n}),
          wasm_ndarray<Real>::from_buffer(std::move(ts), {n})};
}

template <typename Real>
inline auto rc_prim_batch(wasm_ndarray<Real> &rays, wasm_ndarray<Real> &b,
                          prim_type tb, Real min_t, Real max_t)
    -> ray_cast_prim_batch_result_t<Real> {
  bool b_rays = is_batch(rays, prim_type::ray);
  int n = b_rays ? batch_count(rays, prim_type::ray) : batch_count(b, tb);
  return rc_prim_batch_impl<Real>(rays, b, tb,
                                  tf::make_constant_range(min_t, n),
                                  tf::make_constant_range(max_t, n));
}

template <typename Real>
inline auto rc_prim_batch_rc(wasm_ndarray<Real> &rays, wasm_ndarray<Real> &b,
                             prim_type tb, wasm_ndarray<Real> &min_ts,
                             wasm_ndarray<Real> &max_ts)
    -> ray_cast_prim_batch_result_t<Real> {
  return rc_prim_batch_impl<Real>(rays, b, tb, min_ts.make_range(),
                                  max_ts.make_range());
}

// ============================================================================
// Form target helpers — templated on Real and FormT
// ============================================================================

template <typename Real, typename FormT>
inline auto rc_form_single(FormT &m, const Real *ray_ptr, Real min_t,
                           Real max_t) -> ray_cast_result_t<Real> {
  auto ray = as_ray(ray_ptr);
  auto config = tf::make_ray_config(min_t, max_t);
  return m.with_form([&](const auto &form) -> ray_cast_result_t<Real> {
    auto r = tf::ray_cast(ray, form, config);
    return {bool(r), r.info.t, r.element};
  });
}

template <typename Real, typename FormT, typename MinR, typename MaxR>
inline auto rc_form_batch_impl(FormT &m, wasm_ndarray<Real> &rays, MinR min_r,
                               MaxR max_r) -> ray_cast_form_batch_result_t<Real> {
  int n = batch_count(rays, prim_type::ray);
  const Real *pr = rays.raw_data();

  tf::buffer<std::int8_t> hits;
  tf::buffer<Real> ts;
  tf::buffer<int> ids;
  hits.allocate(n);
  ts.allocate(n);
  ids.allocate(n);
  auto *h = hits.data();
  auto *t = ts.data();
  auto *d = ids.data();

  m.with_form([&](const auto &form) {
    auto compute = [&](int i) {
      auto ray = as_ray(pr + i * 6);
      auto config = tf::make_ray_config(min_r[i], max_r[i]);
      auto r = tf::ray_cast(ray, form, config);
      h[i] = bool(r) ? 1 : 0;
      t[i] = r.info.t;
      d[i] = r.element;
    };
    if (n >= 100)
      tf::parallel_for_each(tf::make_sequence_range(n), compute);
    else
      for (int i = 0; i < n; ++i)
        compute(i);
  });

  return {wasm_ndarray<std::int8_t>::from_buffer(std::move(hits), {n}),
          wasm_ndarray<Real>::from_buffer(std::move(ts), {n}),
          wasm_ndarray<int>::from_buffer(std::move(ids), {n})};
}

template <typename Real, typename FormT>
inline auto rc_form_batch(FormT &m, wasm_ndarray<Real> &rays, Real min_t,
                          Real max_t) -> ray_cast_form_batch_result_t<Real> {
  int n = batch_count(rays, prim_type::ray);
  return rc_form_batch_impl<Real>(m, rays, tf::make_constant_range(min_t, n),
                                  tf::make_constant_range(max_t, n));
}

template <typename Real, typename FormT>
inline auto rc_form_batch_rc(FormT &m, wasm_ndarray<Real> &rays,
                             wasm_ndarray<Real> &min_ts,
                             wasm_ndarray<Real> &max_ts)
    -> ray_cast_form_batch_result_t<Real> {
  return rc_form_batch_impl<Real>(m, rays, min_ts.make_range(),
                                  max_ts.make_range());
}

// ============================================================================
// Sync FP dispatch helpers — templated on Real and FormT
// ============================================================================

template <typename Real, typename FormT>
inline auto sync_rc_f(wasm_ndarray<Real> &rays, FormT &m, Real min_t,
                      Real max_t) -> emscripten::val {
  if (!is_batch(rays, prim_type::ray))
    return emscripten::val(
        rc_form_single<Real>(m, rays.raw_data(), min_t, max_t));
  return emscripten::val(rc_form_batch<Real>(m, rays, min_t, max_t));
}

template <typename Real, typename FormT>
inline auto sync_rc_f_rc(wasm_ndarray<Real> &rays, FormT &m,
                         wasm_ndarray<Real> &min_ts, wasm_ndarray<Real> &max_ts)
    -> emscripten::val {
  if (!is_batch(rays, prim_type::ray))
    return emscripten::val(rc_form_single<Real>(m, rays.raw_data(),
                                                min_ts.raw_data()[0],
                                                max_ts.raw_data()[0]));
  return emscripten::val(rc_form_batch_rc<Real>(m, rays, min_ts, max_ts));
}

// ============================================================================
// Sync entry points
// ============================================================================

template <typename Real>
inline auto sync_ray_cast_p(wasm_ndarray<Real> &rays, wasm_ndarray<Real> &b,
                            int tb_int, Real min_t, Real max_t)
    -> emscripten::val {
  auto tb = static_cast<prim_type>(tb_int);
  bool br = is_batch(rays, prim_type::ray);
  bool bb = is_batch(b, tb);
  if (!br && !bb)
    return emscripten::val(rc_prim_single<Real>(rays.raw_data(), b.raw_data(),
                                                tb, poly_verts(b, tb), min_t,
                                                max_t));
  return emscripten::val(rc_prim_batch<Real>(rays, b, tb, min_t, max_t));
}

template <typename Real>
inline auto sync_ray_cast_p_rc(wasm_ndarray<Real> &rays, wasm_ndarray<Real> &b,
                               int tb_int, wasm_ndarray<Real> &min_ts,
                               wasm_ndarray<Real> &max_ts) -> emscripten::val {
  auto tb = static_cast<prim_type>(tb_int);
  bool br = is_batch(rays, prim_type::ray);
  bool bb = is_batch(b, tb);
  if (!br && !bb)
    return emscripten::val(rc_prim_single<Real>(
        rays.raw_data(), b.raw_data(), tb, poly_verts(b, tb),
        min_ts.raw_data()[0], max_ts.raw_data()[0]));
  return emscripten::val(rc_prim_batch_rc<Real>(rays, b, tb, min_ts, max_ts));
}

template <typename Real>
inline auto sync_ray_cast_f(wasm_ndarray<Real> &rays, wasm_mesh<Real> &m,
                            Real min_t, Real max_t) -> emscripten::val {
  return sync_rc_f<Real>(rays, m, min_t, max_t);
}

template <typename Real>
inline auto sync_ray_cast_f_rc(wasm_ndarray<Real> &rays, wasm_mesh<Real> &m,
                               wasm_ndarray<Real> &min_ts,
                               wasm_ndarray<Real> &max_ts) -> emscripten::val {
  return sync_rc_f_rc<Real>(rays, m, min_ts, max_ts);
}

template <typename Real>
inline auto sync_ray_cast_f_pc(wasm_ndarray<Real> &rays,
                               wasm_point_cloud<Real> &m, Real min_t,
                               Real max_t) -> emscripten::val {
  return sync_rc_f<Real>(rays, m, min_t, max_t);
}

template <typename Real>
inline auto sync_ray_cast_f_pc_rc(wasm_ndarray<Real> &rays,
                                  wasm_point_cloud<Real> &m,
                                  wasm_ndarray<Real> &min_ts,
                                  wasm_ndarray<Real> &max_ts)
    -> emscripten::val {
  return sync_rc_f_rc<Real>(rays, m, min_ts, max_ts);
}

// ============================================================================
// Async FP dispatch — templated on Real and FormT
// ============================================================================

template <typename Real, typename FormT>
inline auto async_rc_f(wasm_ndarray<Real> &rays, FormT &m, Real min_t,
                       Real max_t) -> promise_t {
  if (!is_batch(rays, prim_type::ray)) {
    return promise(
        [rays = rays, m = m, min_t, max_t]() -> ray_cast_result_t<Real> {
          return rc_form_single<Real>(
              const_cast<FormT &>(m),
              const_cast<wasm_ndarray<Real> &>(rays).raw_data(), min_t, max_t);
        });
  }

  return promise([rays = rays, m = m, min_t,
                  max_t]() -> ray_cast_form_batch_result_t<Real> {
    return rc_form_batch<Real>(const_cast<FormT &>(m),
                               const_cast<wasm_ndarray<Real> &>(rays), min_t,
                               max_t);
  });
}

template <typename Real, typename FormT>
inline auto async_rc_f_rc(wasm_ndarray<Real> &rays, FormT &m,
                          wasm_ndarray<Real> &min_ts,
                          wasm_ndarray<Real> &max_ts) -> promise_t {
  if (!is_batch(rays, prim_type::ray)) {
    Real mn = min_ts.raw_data()[0];
    Real mx = max_ts.raw_data()[0];
    return promise(
        [rays = rays, m = m, mn, mx]() -> ray_cast_result_t<Real> {
          return rc_form_single<Real>(
              const_cast<FormT &>(m),
              const_cast<wasm_ndarray<Real> &>(rays).raw_data(), mn, mx);
        });
  }

  return promise([rays = rays, m = m, min_ts = min_ts,
                  max_ts = max_ts]() -> ray_cast_form_batch_result_t<Real> {
    return rc_form_batch_rc<Real>(
        const_cast<FormT &>(m), const_cast<wasm_ndarray<Real> &>(rays),
        const_cast<wasm_ndarray<Real> &>(min_ts),
        const_cast<wasm_ndarray<Real> &>(max_ts));
  });
}

// ============================================================================
// Async entry points
// ============================================================================

template <typename Real>
inline auto async_ray_cast_p(wasm_ndarray<Real> &rays, wasm_ndarray<Real> &b,
                             int tb_int, Real min_t, Real max_t) -> promise_t {
  auto tb = static_cast<prim_type>(tb_int);
  bool br = is_batch(rays, prim_type::ray);
  bool bb = is_batch(b, tb);

  if (!br && !bb) {
    int vb = poly_verts(b, tb);
    return promise([rays = rays, b = b, tb, vb, min_t,
                    max_t]() -> ray_cast_result_t<Real> {
      return rc_prim_single<Real>(
          const_cast<wasm_ndarray<Real> &>(rays).raw_data(),
          const_cast<wasm_ndarray<Real> &>(b).raw_data(), tb, vb, min_t, max_t);
    });
  }

  return promise([rays = rays, b = b, tb, min_t,
                  max_t]() -> ray_cast_prim_batch_result_t<Real> {
    return rc_prim_batch<Real>(const_cast<wasm_ndarray<Real> &>(rays),
                               const_cast<wasm_ndarray<Real> &>(b), tb, min_t,
                               max_t);
  });
}

template <typename Real>
inline auto async_ray_cast_p_rc(wasm_ndarray<Real> &rays, wasm_ndarray<Real> &b,
                                int tb_int, wasm_ndarray<Real> &min_ts,
                                wasm_ndarray<Real> &max_ts) -> promise_t {
  auto tb = static_cast<prim_type>(tb_int);
  bool br = is_batch(rays, prim_type::ray);
  bool bb = is_batch(b, tb);

  if (!br && !bb) {
    int vb = poly_verts(b, tb);
    Real mn = min_ts.raw_data()[0];
    Real mx = max_ts.raw_data()[0];
    return promise(
        [rays = rays, b = b, tb, vb, mn, mx]() -> ray_cast_result_t<Real> {
          return rc_prim_single<Real>(
              const_cast<wasm_ndarray<Real> &>(rays).raw_data(),
              const_cast<wasm_ndarray<Real> &>(b).raw_data(), tb, vb, mn, mx);
        });
  }

  return promise([rays = rays, b = b, tb, min_ts = min_ts,
                  max_ts = max_ts]() -> ray_cast_prim_batch_result_t<Real> {
    return rc_prim_batch_rc<Real>(
        const_cast<wasm_ndarray<Real> &>(rays),
        const_cast<wasm_ndarray<Real> &>(b), tb,
        const_cast<wasm_ndarray<Real> &>(min_ts),
        const_cast<wasm_ndarray<Real> &>(max_ts));
  });
}

template <typename Real>
inline auto async_ray_cast_f(wasm_ndarray<Real> &rays, wasm_mesh<Real> &m,
                             Real min_t, Real max_t) -> promise_t {
  return async_rc_f<Real>(rays, m, min_t, max_t);
}

template <typename Real>
inline auto async_ray_cast_f_rc(wasm_ndarray<Real> &rays, wasm_mesh<Real> &m,
                                wasm_ndarray<Real> &min_ts,
                                wasm_ndarray<Real> &max_ts) -> promise_t {
  return async_rc_f_rc<Real>(rays, m, min_ts, max_ts);
}

template <typename Real>
inline auto async_ray_cast_f_pc(wasm_ndarray<Real> &rays,
                                wasm_point_cloud<Real> &m, Real min_t,
                                Real max_t) -> promise_t {
  return async_rc_f<Real>(rays, m, min_t, max_t);
}

template <typename Real>
inline auto async_ray_cast_f_pc_rc(wasm_ndarray<Real> &rays,
                                   wasm_point_cloud<Real> &m,
                                   wasm_ndarray<Real> &min_ts,
                                   wasm_ndarray<Real> &max_ts) -> promise_t {
  return async_rc_f_rc<Real>(rays, m, min_ts, max_ts);
}

} // namespace ts
} // namespace tf
