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

#include "trueform/core/aabb_like.hpp"
#include "trueform/core/line_like.hpp"
#include "trueform/core/plane_like.hpp"
#include "trueform/core/point_view.hpp"
#include "trueform/core/points.hpp"
#include "trueform/core/polygon.hpp"
#include "trueform/core/range.hpp"
#include "trueform/core/ray_like.hpp"
#include "trueform/core/segment.hpp"
#include "trueform/core/unit_vector_view.hpp"
#include "trueform/core/vector_view.hpp"
#include "trueform/ts/core/wasm_ndarray.hpp"

namespace tf {
namespace ts {

// ============================================================================
// Primitive type enum (mirrors TS PrimitiveType, excluding "vector")
// ============================================================================

enum class prim_type : int {
  point = 0,
  segment = 1,
  triangle = 2,
  ray = 3,
  line = 4,
  plane = 5,
  aabb = 6,
  polygon = 7
};

// ============================================================================
// Single-element adapters: raw float* → C++ primitive view
// ============================================================================

inline auto as_point(const float *ptr) { return tf::make_point_view<3>(ptr); }

inline auto as_segment(const float *ptr) {
  return tf::make_segment_between_points(tf::make_point_view<3>(ptr),
                                         tf::make_point_view<3>(ptr + 3));
}

inline auto as_triangle(const float *ptr) {
  return tf::make_polygon<3>(tf::make_points<3>(tf::make_range(ptr, 9)));
}

inline auto as_ray(const float *ptr) {
  return tf::make_ray_like(tf::make_point_view<3>(ptr),
                           tf::make_vector_view<3>(ptr + 3));
}

inline auto as_line(const float *ptr) {
  return tf::make_line_like(tf::make_point_view<3>(ptr),
                            tf::make_vector_view<3>(ptr + 3));
}

inline auto as_plane(const float *ptr) {
  return tf::make_plane_like(tf::make_unit_vector_view(
                                 tf::unsafe, tf::make_vector_view<3>(ptr)),
                             *(ptr + 3));
}

inline auto as_aabb(const float *ptr) {
  return tf::make_aabb_like(tf::make_point_view<3>(ptr),
                            tf::make_point_view<3>(ptr + 3));
}

inline auto as_polygon(const float *ptr, int n_verts) {
  return tf::make_polygon(
      tf::make_points<3>(tf::make_range(ptr, n_verts * 3)));
}

// ============================================================================
// Stride / batch helpers
// ============================================================================

/// Floats per single element for a given primitive type.
inline int prim_stride(prim_type t, int poly_verts = 0) {
  switch (t) {
  case prim_type::point:
    return 3;
  case prim_type::segment:
    return 6;
  case prim_type::triangle:
    return 9;
  case prim_type::ray:
    return 6;
  case prim_type::line:
    return 6;
  case prim_type::plane:
    return 4;
  case prim_type::aabb:
    return 6;
  case prim_type::polygon:
    return poly_verts * 3;
  }
  return 0;
}

/// Expected ndim for a single (non-batched) element.
inline int single_ndim(prim_type t) {
  switch (t) {
  case prim_type::point:
  case prim_type::plane:
    return 1;
  default:
    return 2;
  }
}

/// Number of vertices for polygon-type primitives from shape.
inline int poly_verts(const wasm_ndarray<float> &arr, prim_type t) {
  if (t == prim_type::polygon) {
    // shape is [V, 3] for single or [N, V, 3] for batch
    auto &s = arr.raw_shape();
    return s[static_cast<int>(s.size()) - 2];
  }
  if (t == prim_type::triangle)
    return 3;
  return 0;
}

inline bool is_batch(const wasm_ndarray<float> &arr, prim_type t) {
  return arr.ndim() > single_ndim(t);
}

inline int batch_count(const wasm_ndarray<float> &arr, prim_type t) {
  return is_batch(arr, t) ? arr.raw_shape()[0] : 1;
}

inline int stride_of(const wasm_ndarray<float> &arr, prim_type t) {
  return prim_stride(t, poly_verts(arr, t));
}

// ============================================================================
// Single dispatch: call fn(prim) for one wasm_ndarray element
// ============================================================================

template <typename Fn>
auto dispatch_single(Fn &&fn, const float *ptr, prim_type t, int n_verts = 0)
    -> decltype(fn(as_point(ptr))) {
  switch (t) {
  case prim_type::point:
    return fn(as_point(ptr));
  case prim_type::segment:
    return fn(as_segment(ptr));
  case prim_type::triangle:
    return fn(as_triangle(ptr));
  case prim_type::ray:
    return fn(as_ray(ptr));
  case prim_type::line:
    return fn(as_line(ptr));
  case prim_type::plane:
    return fn(as_plane(ptr));
  case prim_type::aabb:
    return fn(as_aabb(ptr));
  case prim_type::polygon:
    return fn(as_polygon(ptr, n_verts));
  }
}

// ============================================================================
// Pair dispatch: call fn(primA, primB)
// ============================================================================

template <typename Fn>
auto dispatch_pair(Fn &&fn, const float *a, prim_type ta, const float *b,
                   prim_type tb, int verts_a = 0, int verts_b = 0)
    -> decltype(fn(as_point(a), as_point(b))) {
  return dispatch_single(
      [&](auto &&pa) {
        return dispatch_single(
            [&](auto &&pb) { return fn(pa, pb); }, b, tb, verts_b);
      },
      a, ta, verts_a);
}

} // namespace ts
} // namespace tf
