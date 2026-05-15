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

#include "trueform/clean/polygons.hpp"
#include "trueform/geometry/triangulated.hpp"
#include "trueform/geometry/triangulated_faces.hpp"
#include "trueform/ts/core/promise.hpp"
#include "trueform/ts/core/wasm_mesh.hpp"
#include "trueform/ts/core/wasm_ndarray.hpp"
#include "trueform/ts/core/wasm_offset_blocked_buffer.hpp"

namespace tf {
namespace ts {

// Helper: wrap triangulated face buffer + shared points into wasm_mesh<Real>
template <typename Real>
auto make_tri_mesh(tf::blocked_buffer<int, 3> &&faces,
                   wasm_ndarray<Real> points) -> wasm_mesh<Real> {
  auto &buf = faces.data_buffer();
  auto len = buf.size();
  auto nd = wasm_ndarray<int>::from_buffer(std::move(buf),
                                           {static_cast<int>(len / 3), 3});
  return wasm_mesh<Real>::create(std::move(nd), std::move(points));
}

// -- Sync --

template <typename Real>
auto sync_triangulate_mesh(wasm_mesh<Real> &m) -> wasm_mesh<Real> {
  auto run = [&](auto &&polys) -> wasm_mesh<Real> {
    auto faces = tf::triangulated_faces(polys);
    return make_tri_mesh<Real>(std::move(faces), m.points());
  };
  if (m.has_transformation())
    return run(m.polygons_range() | tf::tag(m.transformation_view()));
  else
    return run(m.polygons_range());
}

template <typename Real>
auto sync_triangulate_dynamic(wasm_offset_blocked_buffer<int, int> &faces,
                              wasm_ndarray<Real> &points) -> wasm_mesh<Real> {
  auto f = tf::make_faces(faces.make_range());
  auto p = tf::make_points<3>(points.make_range());
  auto polys = tf::make_polygons(f, p);
  auto tri_faces = tf::triangulated_faces(polys);
  return make_tri_mesh<Real>(std::move(tri_faces), points);
}

template <typename Real>
auto sync_triangulate_fixed(wasm_ndarray<int> &faces,
                            wasm_ndarray<Real> &points) -> wasm_mesh<Real> {
  int N = faces.raw_shape()[1];
  auto f = tf::make_faces(tf::make_blocked_range(faces.make_range(), N));
  auto p = tf::make_points<3>(points.make_range());
  auto polys = tf::make_polygons(f, p);
  auto tri_faces = tf::triangulated_faces(polys);
  return make_tri_mesh<Real>(std::move(tri_faces), points);
}

template <typename Real>
auto sync_triangulate_polygon(wasm_ndarray<Real> &poly) -> wasm_mesh<Real> {
  auto ndim = poly.raw_shape().size();
  if (ndim == 2) {
    // Single polygon [V, 3]
    auto pts = tf::make_points<3>(poly.make_range());
    auto result = tf::triangulated(tf::make_polygon(pts));
    return wasm_mesh<Real>::from_polygons_buffer(std::move(result));
  } else {
    // Batch [N, V, 3]
    int V = poly.raw_shape()[1];
    auto pts = tf::make_points<3>(poly.make_range());
    auto polys = tf::make_polygons(tf::make_mapped_range(
        tf::make_blocked_range(pts, V),
        [](auto &&block) { return tf::make_polygon(block); }));
    auto buffer = tf::cleaned(polys);
    auto tri_faces = tf::triangulated_faces(buffer.polygons());
    auto n_pts = static_cast<int>(buffer.points_buffer().size());
    return make_tri_mesh<Real>(std::move(tri_faces),
                               wasm_ndarray<Real>::from_buffer(
                                   std::move(buffer.points_buffer().data_buffer()),
                                   {n_pts, 3}));
  }
}

// -- Async --

template <typename Real>
auto async_triangulate_mesh(wasm_mesh<Real> &m) -> promise_t {
  return promise([a = m]() -> wasm_mesh<Real> {
    return sync_triangulate_mesh<Real>(const_cast<wasm_mesh<Real> &>(a));
  });
}

template <typename Real>
auto async_triangulate_dynamic(wasm_offset_blocked_buffer<int, int> &faces,
                               wasm_ndarray<Real> &points) -> promise_t {
  return promise([f = faces, p = points]() -> wasm_mesh<Real> {
    return sync_triangulate_dynamic<Real>(
        const_cast<wasm_offset_blocked_buffer<int, int> &>(f),
        const_cast<wasm_ndarray<Real> &>(p));
  });
}

template <typename Real>
auto async_triangulate_fixed(wasm_ndarray<int> &faces,
                             wasm_ndarray<Real> &points) -> promise_t {
  return promise([f = faces, p = points]() -> wasm_mesh<Real> {
    return sync_triangulate_fixed<Real>(const_cast<wasm_ndarray<int> &>(f),
                                        const_cast<wasm_ndarray<Real> &>(p));
  });
}

template <typename Real>
auto async_triangulate_polygon(wasm_ndarray<Real> &poly) -> promise_t {
  return promise([a = poly]() -> wasm_mesh<Real> {
    return sync_triangulate_polygon<Real>(const_cast<wasm_ndarray<Real> &>(a));
  });
}

} // namespace ts
} // namespace tf
