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
 * Author: Ziga Sajovic
 */

#pragma once

#include "primitive_wrapper.hpp"
#include <trueform/core/aabb_like.hpp>
#include <trueform/core/line_like.hpp>
#include <trueform/core/plane.hpp>
#include <trueform/core/point_view.hpp>
#include <trueform/core/points.hpp>
#include <trueform/core/polygon.hpp>
#include <trueform/core/polygons.hpp>
#include <trueform/core/range.hpp>
#include <trueform/core/ray_like.hpp>
#include <trueform/core/segment.hpp>
#include <trueform/core/segments.hpp>
#include <trueform/core/unit_vector_view.hpp>
#include <trueform/core/unsafe.hpp>
#include <trueform/core/vector_view.hpp>
#include <trueform/core/views/blocked_range.hpp>
#include <trueform/core/views/mapped_range.hpp>

namespace tf::py {

// ============================================================================
// Single-element factories: primitive_wrapper → tf primitive view
// ============================================================================

template <std::size_t Dims, typename RealT>
auto make_point(const primitive_wrapper<Dims, RealT> &pw)
    -> decltype(tf::make_point_view<Dims>(pw.data_ptr())) {
  return tf::make_point_view<Dims>(pw.data_ptr());
}

template <std::size_t Dims, typename RealT>
auto make_segment(const primitive_wrapper<Dims, RealT> &pw)
    -> decltype(tf::make_segment(
        tf::make_points<Dims>(tf::make_range(pw.data_ptr(), 2 * Dims)))) {
  auto pts = tf::make_points<Dims>(tf::make_range(pw.data_ptr(), 2 * Dims));
  return tf::make_segment(pts);
}

template <std::size_t Dims, typename RealT>
auto make_triangle(const primitive_wrapper<Dims, RealT> &pw)
    -> decltype(tf::make_polygon<3>(
        tf::make_points<Dims>(tf::make_range(pw.data_ptr(), 3 * Dims)))) {
  auto pts = tf::make_points<Dims>(tf::make_range(pw.data_ptr(), 3 * Dims));
  return tf::make_polygon<3>(pts);
}

template <std::size_t Dims, typename RealT>
auto make_ray(const primitive_wrapper<Dims, RealT> &pw)
    -> decltype(tf::make_ray_like(tf::make_point_view<Dims>(pw.data_ptr()),
                                  tf::make_vector_view<Dims>(pw.data_ptr()))) {
  auto pts = tf::make_points<Dims>(tf::make_range(pw.data_ptr(), 2 * Dims));
  return tf::make_ray_like(pts[0], pts[1].as_vector_view());
}

template <std::size_t Dims, typename RealT>
auto make_line(const primitive_wrapper<Dims, RealT> &pw)
    -> decltype(tf::make_line_like(tf::make_point_view<Dims>(pw.data_ptr()),
                                   tf::make_vector_view<Dims>(pw.data_ptr()))) {
  auto pts = tf::make_points<Dims>(tf::make_range(pw.data_ptr(), 2 * Dims));
  return tf::make_line_like(pts[0], pts[1].as_vector_view());
}

template <std::size_t Dims, typename RealT>
auto make_plane(const primitive_wrapper<Dims, RealT> &pw)
    -> decltype(tf::make_plane(
        tf::make_unit_vector_view(tf::unsafe,
                                  tf::make_vector_view<Dims>(pw.data_ptr())),
        pw.data_ptr()[Dims])) {
  auto ptr = pw.data_ptr();
  return tf::make_plane(
      tf::make_unit_vector_view(tf::unsafe, tf::make_vector_view<Dims>(ptr)),
      ptr[Dims]);
}

template <std::size_t Dims, typename RealT>
auto make_aabb(const primitive_wrapper<Dims, RealT> &pw)
    -> decltype(tf::make_aabb_like(tf::make_point_view<Dims>(pw.data_ptr()),
                                   tf::make_point_view<Dims>(pw.data_ptr()))) {
  auto pts = tf::make_points<Dims>(tf::make_range(pw.data_ptr(), 2 * Dims));
  return tf::make_aabb_like(pts[0], pts[1]);
}

template <std::size_t Dims, typename RealT>
auto make_polygon(const primitive_wrapper<Dims, RealT> &pw)
    -> decltype(tf::make_polygon(tf::make_points<Dims>(tf::make_range(
        pw.data_ptr(),
        static_cast<std::size_t>(pw.poly_verts()) * Dims)))) {
  auto pts = tf::make_points<Dims>(tf::make_range(
      pw.data_ptr(), static_cast<std::size_t>(pw.poly_verts()) * Dims));
  return tf::make_polygon(pts);
}

// ============================================================================
// Batch factories: primitive_wrapper → tf batch type / mapped range
// ============================================================================

template <std::size_t Dims, typename RealT>
auto make_points(const primitive_wrapper<Dims, RealT> &pw) -> decltype(auto) {
  return tf::make_points<Dims>(tf::make_range(
      pw.data_ptr(), static_cast<std::size_t>(pw.count()) * Dims));
}

template <std::size_t Dims, typename RealT>
auto make_segments(const primitive_wrapper<Dims, RealT> &pw)
    -> decltype(auto) {
  auto pts = tf::make_points<Dims>(tf::make_range(
      pw.data_ptr(), static_cast<std::size_t>(pw.count()) * 2 * Dims));
  auto blocks = tf::make_blocked_range<2>(pts);
  return tf::make_segments(tf::make_mapped_range(
      blocks, [](auto block) { return tf::make_segment(block); }));
}

template <std::size_t Dims, typename RealT>
auto make_triangles(const primitive_wrapper<Dims, RealT> &pw)
    -> decltype(auto) {
  auto pts = tf::make_points<Dims>(tf::make_range(
      pw.data_ptr(), static_cast<std::size_t>(pw.count()) * 3 * Dims));
  auto blocks = tf::make_blocked_range<3>(pts);
  return tf::make_polygons(tf::make_mapped_range(
      blocks, [](auto block) { return tf::make_polygon<3>(block); }));
}

template <std::size_t Dims, typename RealT>
auto make_rays(const primitive_wrapper<Dims, RealT> &pw) -> decltype(auto) {
  auto pts = tf::make_points<Dims>(tf::make_range(
      pw.data_ptr(), static_cast<std::size_t>(pw.count()) * 2 * Dims));
  auto blocks = tf::make_blocked_range<2>(pts);
  return tf::make_mapped_range(blocks, [](auto block) {
    return tf::make_ray_like(block[0], block[1].as_vector_view());
  });
}

template <std::size_t Dims, typename RealT>
auto make_lines(const primitive_wrapper<Dims, RealT> &pw) -> decltype(auto) {
  auto pts = tf::make_points<Dims>(tf::make_range(
      pw.data_ptr(), static_cast<std::size_t>(pw.count()) * 2 * Dims));
  auto blocks = tf::make_blocked_range<2>(pts);
  return tf::make_mapped_range(blocks, [](auto block) {
    return tf::make_line_like(block[0], block[1].as_vector_view());
  });
}

template <std::size_t Dims, typename RealT>
auto make_planes(const primitive_wrapper<Dims, RealT> &pw) -> decltype(auto) {
  auto flat = tf::make_range(pw.data_ptr(),
                             static_cast<std::size_t>(pw.count()) * (Dims + 1));
  auto blocks = tf::make_blocked_range<Dims + 1>(flat);
  return tf::make_mapped_range(blocks, [](auto block) {
    return tf::make_plane(
        tf::make_unit_vector_view(tf::unsafe,
                                  tf::make_vector_view<Dims>(&block[0])),
        block[Dims]);
  });
}

template <std::size_t Dims, typename RealT>
auto make_aabbs(const primitive_wrapper<Dims, RealT> &pw) -> decltype(auto) {
  auto pts = tf::make_points<Dims>(tf::make_range(
      pw.data_ptr(), static_cast<std::size_t>(pw.count()) * 2 * Dims));
  auto blocks = tf::make_blocked_range<2>(pts);
  return tf::make_mapped_range(blocks, [](auto block) {
    return tf::make_aabb_like(block[0], block[1]);
  });
}

template <std::size_t Dims, typename RealT>
auto make_polygons(const primitive_wrapper<Dims, RealT> &pw)
    -> decltype(auto) {
  auto nv = pw.poly_verts();
  auto pts = tf::make_points<Dims>(tf::make_range(
      pw.data_ptr(),
      static_cast<std::size_t>(pw.count()) * static_cast<std::size_t>(nv) *
          Dims));
  auto blocks = tf::make_blocked_range(pts, static_cast<std::size_t>(nv));
  return tf::make_polygons(tf::make_mapped_range(
      blocks, [](auto block) { return tf::make_polygon(block); }));
}

// ============================================================================
// dispatch_single: non-batch primitive_wrapper → fn(tf_view)
// ============================================================================

template <std::size_t Dims, typename RealT, typename Fn>
auto dispatch_single(Fn &&fn, const primitive_wrapper<Dims, RealT> &pw)
    -> decltype(fn(make_point(pw))) {
  switch (pw.type()) {
  case prim_type::point:
    return fn(make_point(pw));
  case prim_type::segment:
    return fn(make_segment(pw));
  case prim_type::triangle:
    return fn(make_triangle(pw));
  case prim_type::ray:
    return fn(make_ray(pw));
  case prim_type::line:
    return fn(make_line(pw));
  case prim_type::plane:
    if constexpr (Dims >= 3) {
      return fn(make_plane(pw));
    } else {
      throw std::runtime_error("plane primitive requires 3D");
    }
  case prim_type::aabb:
    return fn(make_aabb(pw));
  case prim_type::polygon:
    return fn(make_polygon(pw));
  default: // silence MSVC C4715: not all control paths return a value
    throw std::runtime_error("unknown primitive type");
  }
}

// ============================================================================
// dispatch_at: per-element dispatch for batch loops
// ============================================================================

template <std::size_t Dims, typename RealT, typename Fn>
auto dispatch_at(Fn &&fn, const primitive_wrapper<Dims, RealT> &pw, int i)
    -> decltype(fn(tf::make_point_view<Dims>(pw.element_ptr(i)))) {
  auto ptr = pw.element_ptr(i);
  auto nv = pw.poly_verts();
  switch (pw.type()) {
  case prim_type::point:
    return fn(tf::make_point_view<Dims>(ptr));
  case prim_type::segment: {
    auto pts = tf::make_points<Dims>(tf::make_range(ptr, 2 * Dims));
    return fn(tf::make_segment(pts));
  }
  case prim_type::triangle: {
    auto pts = tf::make_points<Dims>(tf::make_range(ptr, 3 * Dims));
    return fn(tf::make_polygon<3>(pts));
  }
  case prim_type::ray: {
    auto pts = tf::make_points<Dims>(tf::make_range(ptr, 2 * Dims));
    return fn(tf::make_ray_like(pts[0], pts[1].as_vector_view()));
  }
  case prim_type::line: {
    auto pts = tf::make_points<Dims>(tf::make_range(ptr, 2 * Dims));
    return fn(tf::make_line_like(pts[0], pts[1].as_vector_view()));
  }
  case prim_type::plane:
    if constexpr (Dims >= 3) {
      return fn(tf::make_plane(
          tf::make_unit_vector_view(tf::unsafe,
                                    tf::make_vector_view<Dims>(ptr)),
          ptr[Dims]));
    } else {
      throw std::runtime_error("plane primitive requires 3D");
    }
  case prim_type::aabb: {
    auto pts = tf::make_points<Dims>(tf::make_range(ptr, 2 * Dims));
    return fn(tf::make_aabb_like(pts[0], pts[1]));
  }
  case prim_type::polygon: {
    auto pts = tf::make_points<Dims>(
        tf::make_range(ptr, static_cast<std::size_t>(nv) * Dims));
    return fn(tf::make_polygon(pts));
  }
  default: // silence MSVC C4715: not all control paths return a value
    throw std::runtime_error("unknown primitive type");
  }
}

// ============================================================================
// dispatch_pair: non-batch pair dispatch via nested dispatch_single
// ============================================================================

template <std::size_t Dims, typename RealT, typename Fn>
auto dispatch_pair(Fn &&fn, const primitive_wrapper<Dims, RealT> &a,
                   const primitive_wrapper<Dims, RealT> &b)
    -> decltype(fn(make_point(a), make_point(b))) {
  return dispatch_single(
      [&](auto &&pa) {
        return dispatch_single(
            [&](auto &&pb) { return fn(pa, pb); }, b);
      },
      a);
}

// ============================================================================
// dispatch_pair_at: per-element pair dispatch via nested dispatch_at
// ============================================================================

template <std::size_t Dims, typename RealT, typename Fn>
auto dispatch_pair_at(Fn &&fn, const primitive_wrapper<Dims, RealT> &a, int ia,
                      const primitive_wrapper<Dims, RealT> &b, int ib)
    -> decltype(fn(tf::make_point_view<Dims>(a.data_ptr()),
                   tf::make_point_view<Dims>(b.data_ptr()))) {
  return dispatch_at(
      [&](auto &&pa) {
        return dispatch_at([&](auto &&pb) { return fn(pa, pb); }, b, ib);
      },
      a, ia);
}

} // namespace tf::py
