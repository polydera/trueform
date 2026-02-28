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
#include "trueform/core/ray_cast.hpp"
#include "trueform/core/views/constant.hpp"
#include "trueform/core/views/sequence_range.hpp"
#include "trueform/spatial/ray_cast.hpp"
#include "trueform/ts/core/promise.hpp"
#include "trueform/ts/core/wasm_mesh.hpp"
#include "trueform/ts/core/wasm_ndarray.hpp"
#include "trueform/ts/core/wasm_point_cloud.hpp"
#include "trueform/ts/spatial/prim_dispatch.hpp"
#include "trueform/ts/spatial/result_types.hpp"
#include <emscripten/bind.h>

namespace {

using namespace tf::ts;

// ============================================================================
// Prim target helpers (manual dispatch — not all types support ray_cast)
// ============================================================================

auto rc_prim_single(const float *ray_ptr, const float *b, prim_type tb, int vb,
                    float min_t, float max_t) -> ray_cast_result {
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
    return {false, 0.f, -1}; // unreachable — TS validates prim_type
  }
}

template <typename MinR, typename MaxR>
auto rc_prim_batch_impl(wasm_ndarray<float> &rays, wasm_ndarray<float> &b,
                        prim_type tb, MinR min_r,
                        MaxR max_r) -> ray_cast_prim_batch_result {
  int vb = poly_verts(b, tb);
  int sb = stride_of(b, tb);
  bool b_rays = is_batch(rays, prim_type::ray);
  bool b_tgt = is_batch(b, tb);
  int n = b_rays ? batch_count(rays, prim_type::ray) : batch_count(b, tb);
  int stride_ray = b_rays ? 6 : 0;
  int stride_b = b_tgt ? sb : 0;

  const float *pr = rays.raw_data();
  const float *pb = b.raw_data();

  tf::buffer<std::int8_t> hits;
  tf::buffer<float> ts;
  hits.allocate(n);
  ts.allocate(n);
  auto *h = hits.data();
  auto *t = ts.data();

  auto compute = [=](int i) {
    auto r = rc_prim_single(pr + i * stride_ray, pb + i * stride_b, tb, vb,
                            min_r[i], max_r[i]);
    h[i] = r.hit ? 1 : 0;
    t[i] = r.t;
  };

  if (n >= 1000)
    tf::parallel_for_each(tf::make_sequence_range(n), compute);
  else
    for (int i = 0; i < n; ++i)
      compute(i);

  return {wasm_ndarray<std::int8_t>::from_buffer(std::move(hits), {n}),
          wasm_ndarray<float>::from_buffer(std::move(ts), {n})};
}

auto rc_prim_batch(wasm_ndarray<float> &rays, wasm_ndarray<float> &b,
                   prim_type tb, float min_t,
                   float max_t) -> ray_cast_prim_batch_result {
  bool b_rays = is_batch(rays, prim_type::ray);
  int n = b_rays ? batch_count(rays, prim_type::ray) : batch_count(b, tb);
  return rc_prim_batch_impl(rays, b, tb, tf::make_constant_range(min_t, n),
                            tf::make_constant_range(max_t, n));
}

auto rc_prim_batch_rc(wasm_ndarray<float> &rays, wasm_ndarray<float> &b,
                      prim_type tb, wasm_ndarray<float> &min_ts,
                      wasm_ndarray<float> &max_ts)
    -> ray_cast_prim_batch_result {
  return rc_prim_batch_impl(rays, b, tb, min_ts.make_range(),
                            max_ts.make_range());
}

// ============================================================================
// Form target helpers — templated on FormT
// ============================================================================

template <typename FormT>
auto rc_form_single(FormT &m, const float *ray_ptr, float min_t,
                    float max_t) -> ray_cast_result {
  auto ray = as_ray(ray_ptr);
  auto config = tf::make_ray_config(min_t, max_t);
  return m.with_form([&](const auto &form) -> ray_cast_result {
    auto r = tf::ray_cast(ray, form, config);
    return {bool(r), r.info.t, r.element};
  });
}

template <typename FormT, typename MinR, typename MaxR>
auto rc_form_batch_impl(FormT &m, wasm_ndarray<float> &rays, MinR min_r,
                        MaxR max_r) -> ray_cast_form_batch_result {
  int n = batch_count(rays, prim_type::ray);
  const float *pr = rays.raw_data();

  tf::buffer<std::int8_t> hits;
  tf::buffer<float> ts;
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
          wasm_ndarray<float>::from_buffer(std::move(ts), {n}),
          wasm_ndarray<int>::from_buffer(std::move(ids), {n})};
}

template <typename FormT>
auto rc_form_batch(FormT &m, wasm_ndarray<float> &rays, float min_t,
                   float max_t) -> ray_cast_form_batch_result {
  int n = batch_count(rays, prim_type::ray);
  return rc_form_batch_impl(m, rays, tf::make_constant_range(min_t, n),
                            tf::make_constant_range(max_t, n));
}

template <typename FormT>
auto rc_form_batch_rc(FormT &m, wasm_ndarray<float> &rays,
                      wasm_ndarray<float> &min_ts,
                      wasm_ndarray<float> &max_ts)
    -> ray_cast_form_batch_result {
  return rc_form_batch_impl(m, rays, min_ts.make_range(),
                            max_ts.make_range());
}

// ============================================================================
// Sync entry points
// ============================================================================

auto sync_ray_cast_p(wasm_ndarray<float> &rays, wasm_ndarray<float> &b,
                     int tb_int, float min_t,
                     float max_t) -> emscripten::val {
  auto tb = static_cast<prim_type>(tb_int);
  bool br = is_batch(rays, prim_type::ray);
  bool bb = is_batch(b, tb);
  if (!br && !bb)
    return emscripten::val(rc_prim_single(rays.raw_data(), b.raw_data(), tb,
                                          poly_verts(b, tb), min_t, max_t));
  return emscripten::val(rc_prim_batch(rays, b, tb, min_t, max_t));
}

auto sync_ray_cast_p_rc(wasm_ndarray<float> &rays, wasm_ndarray<float> &b,
                        int tb_int, wasm_ndarray<float> &min_ts,
                        wasm_ndarray<float> &max_ts) -> emscripten::val {
  auto tb = static_cast<prim_type>(tb_int);
  bool br = is_batch(rays, prim_type::ray);
  bool bb = is_batch(b, tb);
  if (!br && !bb)
    return emscripten::val(rc_prim_single(
        rays.raw_data(), b.raw_data(), tb, poly_verts(b, tb),
        min_ts.raw_data()[0], max_ts.raw_data()[0]));
  return emscripten::val(rc_prim_batch_rc(rays, b, tb, min_ts, max_ts));
}

template <typename FormT>
auto sync_rc_f(wasm_ndarray<float> &rays, FormT &m, float min_t,
               float max_t) -> emscripten::val {
  if (!is_batch(rays, prim_type::ray))
    return emscripten::val(rc_form_single(m, rays.raw_data(), min_t, max_t));
  return emscripten::val(rc_form_batch(m, rays, min_t, max_t));
}

template <typename FormT>
auto sync_rc_f_rc(wasm_ndarray<float> &rays, FormT &m,
                  wasm_ndarray<float> &min_ts,
                  wasm_ndarray<float> &max_ts) -> emscripten::val {
  if (!is_batch(rays, prim_type::ray))
    return emscripten::val(rc_form_single(m, rays.raw_data(),
                                          min_ts.raw_data()[0],
                                          max_ts.raw_data()[0]));
  return emscripten::val(rc_form_batch_rc(m, rays, min_ts, max_ts));
}

auto sync_ray_cast_f(wasm_ndarray<float> &rays, wasm_mesh &m, float min_t,
                     float max_t) -> emscripten::val {
  return sync_rc_f(rays, m, min_t, max_t);
}

auto sync_ray_cast_f_rc(wasm_ndarray<float> &rays, wasm_mesh &m,
                        wasm_ndarray<float> &min_ts,
                        wasm_ndarray<float> &max_ts) -> emscripten::val {
  return sync_rc_f_rc(rays, m, min_ts, max_ts);
}

auto sync_ray_cast_f_pc(wasm_ndarray<float> &rays, wasm_point_cloud &m,
                        float min_t, float max_t) -> emscripten::val {
  return sync_rc_f(rays, m, min_t, max_t);
}

auto sync_ray_cast_f_pc_rc(wasm_ndarray<float> &rays, wasm_point_cloud &m,
                           wasm_ndarray<float> &min_ts,
                           wasm_ndarray<float> &max_ts) -> emscripten::val {
  return sync_rc_f_rc(rays, m, min_ts, max_ts);
}

// ============================================================================
// Async entry points
// ============================================================================

auto async_ray_cast_p(wasm_ndarray<float> &rays, wasm_ndarray<float> &b,
                      int tb_int, float min_t,
                      float max_t) -> promise_t {
  auto tb = static_cast<prim_type>(tb_int);
  bool br = is_batch(rays, prim_type::ray);
  bool bb = is_batch(b, tb);

  if (!br && !bb) {
    int vb = poly_verts(b, tb);
    return promise([rays = rays, b = b, tb, vb, min_t,
                    max_t]() -> ray_cast_result {
      return rc_prim_single(
          const_cast<wasm_ndarray<float> &>(rays).raw_data(),
          const_cast<wasm_ndarray<float> &>(b).raw_data(), tb, vb, min_t,
          max_t);
    });
  }

  return promise([rays = rays, b = b, tb, min_t,
                  max_t]() -> ray_cast_prim_batch_result {
    return rc_prim_batch(const_cast<wasm_ndarray<float> &>(rays),
                         const_cast<wasm_ndarray<float> &>(b), tb, min_t,
                         max_t);
  });
}

auto async_ray_cast_p_rc(wasm_ndarray<float> &rays, wasm_ndarray<float> &b,
                         int tb_int, wasm_ndarray<float> &min_ts,
                         wasm_ndarray<float> &max_ts) -> promise_t {
  auto tb = static_cast<prim_type>(tb_int);
  bool br = is_batch(rays, prim_type::ray);
  bool bb = is_batch(b, tb);

  if (!br && !bb) {
    int vb = poly_verts(b, tb);
    float mn = min_ts.raw_data()[0];
    float mx = max_ts.raw_data()[0];
    return promise(
        [rays = rays, b = b, tb, vb, mn, mx]() -> ray_cast_result {
          return rc_prim_single(
              const_cast<wasm_ndarray<float> &>(rays).raw_data(),
              const_cast<wasm_ndarray<float> &>(b).raw_data(), tb, vb, mn, mx);
        });
  }

  return promise([rays = rays, b = b, tb, min_ts = min_ts,
                  max_ts = max_ts]() -> ray_cast_prim_batch_result {
    return rc_prim_batch_rc(const_cast<wasm_ndarray<float> &>(rays),
                            const_cast<wasm_ndarray<float> &>(b), tb,
                            const_cast<wasm_ndarray<float> &>(min_ts),
                            const_cast<wasm_ndarray<float> &>(max_ts));
  });
}

template <typename FormT>
auto async_rc_f(wasm_ndarray<float> &rays, FormT &m, float min_t,
                float max_t) -> promise_t {
  if (!is_batch(rays, prim_type::ray)) {
    return promise(
        [rays = rays, m = m, min_t, max_t]() -> ray_cast_result {
          return rc_form_single(
              const_cast<FormT &>(m),
              const_cast<wasm_ndarray<float> &>(rays).raw_data(), min_t, max_t);
        });
  }

  return promise([rays = rays, m = m, min_t,
                  max_t]() -> ray_cast_form_batch_result {
    return rc_form_batch(const_cast<FormT &>(m),
                         const_cast<wasm_ndarray<float> &>(rays), min_t,
                         max_t);
  });
}

template <typename FormT>
auto async_rc_f_rc(wasm_ndarray<float> &rays, FormT &m,
                   wasm_ndarray<float> &min_ts,
                   wasm_ndarray<float> &max_ts) -> promise_t {
  if (!is_batch(rays, prim_type::ray)) {
    float mn = min_ts.raw_data()[0];
    float mx = max_ts.raw_data()[0];
    return promise(
        [rays = rays, m = m, mn, mx]() -> ray_cast_result {
          return rc_form_single(
              const_cast<FormT &>(m),
              const_cast<wasm_ndarray<float> &>(rays).raw_data(), mn, mx);
        });
  }

  return promise([rays = rays, m = m, min_ts = min_ts,
                  max_ts = max_ts]() -> ray_cast_form_batch_result {
    return rc_form_batch_rc(const_cast<FormT &>(m),
                            const_cast<wasm_ndarray<float> &>(rays),
                            const_cast<wasm_ndarray<float> &>(min_ts),
                            const_cast<wasm_ndarray<float> &>(max_ts));
  });
}

auto async_ray_cast_f(wasm_ndarray<float> &rays, wasm_mesh &m, float min_t,
                      float max_t) -> promise_t {
  return async_rc_f(rays, m, min_t, max_t);
}

auto async_ray_cast_f_rc(wasm_ndarray<float> &rays, wasm_mesh &m,
                         wasm_ndarray<float> &min_ts,
                         wasm_ndarray<float> &max_ts) -> promise_t {
  return async_rc_f_rc(rays, m, min_ts, max_ts);
}

auto async_ray_cast_f_pc(wasm_ndarray<float> &rays, wasm_point_cloud &m,
                         float min_t, float max_t) -> promise_t {
  return async_rc_f(rays, m, min_t, max_t);
}

auto async_ray_cast_f_pc_rc(wasm_ndarray<float> &rays, wasm_point_cloud &m,
                            wasm_ndarray<float> &min_ts,
                            wasm_ndarray<float> &max_ts) -> promise_t {
  return async_rc_f_rc(rays, m, min_ts, max_ts);
}

} // namespace

EMSCRIPTEN_BINDINGS(trueform_ray_cast) {
  emscripten::value_object<tf::ts::ray_cast_result>("RayCastResult")
      .field("hit", &tf::ts::ray_cast_result::hit)
      .field("t", &tf::ts::ray_cast_result::t)
      .field("elementId", &tf::ts::ray_cast_result::element_id);

  emscripten::value_object<tf::ts::ray_cast_prim_batch_result>(
      "RayCastPrimBatchResult")
      .field("hits", &tf::ts::ray_cast_prim_batch_result::hits)
      .field("ts", &tf::ts::ray_cast_prim_batch_result::ts);

  emscripten::value_object<tf::ts::ray_cast_form_batch_result>(
      "RayCastFormBatchResult")
      .field("hits", &tf::ts::ray_cast_form_batch_result::hits)
      .field("ts", &tf::ts::ray_cast_form_batch_result::ts)
      .field("elementIds", &tf::ts::ray_cast_form_batch_result::element_ids);

  emscripten::function("ray_cast_p", &sync_ray_cast_p);
  emscripten::function("ray_cast_p_rc", &sync_ray_cast_p_rc);
  emscripten::function("ray_cast_f", &sync_ray_cast_f);
  emscripten::function("ray_cast_f_rc", &sync_ray_cast_f_rc);
  emscripten::function("ray_cast_f_pc", &sync_ray_cast_f_pc);
  emscripten::function("ray_cast_f_pc_rc", &sync_ray_cast_f_pc_rc);
  emscripten::function("dispatch_ray_cast_p", &async_ray_cast_p);
  emscripten::function("dispatch_ray_cast_p_rc", &async_ray_cast_p_rc);
  emscripten::function("dispatch_ray_cast_f", &async_ray_cast_f);
  emscripten::function("dispatch_ray_cast_f_rc", &async_ray_cast_f_rc);
  emscripten::function("dispatch_ray_cast_f_pc", &async_ray_cast_f_pc);
  emscripten::function("dispatch_ray_cast_f_pc_rc", &async_ray_cast_f_pc_rc);
}
