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

#include "trueform/arrangement/mesh/materialize_mesh_triangulation.hpp"
#include "trueform/arrangement/mesh/mesh_triangulation.hpp"
#include "trueform/clean/polygons.hpp"
#include "trueform/geometry/triangulated.hpp"
#include "trueform/geometry/triangulation/make_triangulation_faces.hpp"
#include "trueform/ts/core/promise.hpp"
#include "trueform/ts/core/wasm_mesh.hpp"
#include "trueform/ts/core/wasm_ndarray.hpp"
#include "trueform/ts/core/wasm_offset_blocked_buffer.hpp"

#include <utility>

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

/// THE POINT TABLE IS THE INPUT'S UNTIL A FACE IS RESOLVED. A face whose loop
/// crosses itself is resolved rather than dropped, and it mints the identity
/// its crossing stands on — a corner past the input's own extent has no row in
/// the input's array, so a build that minted one must own the product's whole
/// table. A build that minted none names the input's own ids and nothing else,
/// and there the points stay shared, uncopied. That is the mesh in practice,
/// so `shared_points` is asked for only on the path that shares.
template <typename Real, typename Polygons, typename SharedPoints>
auto triangulated_wasm_mesh(const Polygons &polys,
                            const SharedPoints &shared_points)
    -> wasm_mesh<Real> {
  const auto triangulation = tf::arrangement::make_mesh_triangulation(polys);
  if (triangulation.created_points().size() != 0)
    return wasm_mesh<Real>::from_polygons_buffer(
        tf::arrangement::materialize_mesh_triangulation(triangulation, polys));
  auto faces = tf::geometry::make_triangulation_faces(triangulation);
  return make_tri_mesh<Real>(std::move(faces), shared_points());
}

// -- Sync --

template <typename Real>
auto sync_triangulate_mesh(wasm_mesh<Real> &m) -> wasm_mesh<Real> {
  // THE TWO BRANCHES MUST ANSWER IN ONE SPACE, and the sharing branch hands
  // back the mesh's own points — so the tier reads them as stated. A frame
  // here would move the minted branch's table and not the shared one.
  return triangulated_wasm_mesh<Real>(m.polygons_range(),
                                      [&m] { return m.points(); });
}

template <typename Real>
auto sync_triangulate_dynamic(wasm_offset_blocked_buffer<int, int> &faces,
                              wasm_ndarray<Real> &points) -> wasm_mesh<Real> {
  auto f = tf::make_faces(faces.make_range());
  auto p = tf::make_points<3>(points.make_range());
  auto polys = tf::make_polygons(f, p);
  return triangulated_wasm_mesh<Real>(polys, [&points] { return points; });
}

template <typename Real>
auto sync_triangulate_fixed(wasm_ndarray<int> &faces,
                            wasm_ndarray<Real> &points) -> wasm_mesh<Real> {
  int N = faces.raw_shape()[1];
  auto f = tf::make_faces(tf::make_blocked_range(faces.make_range(), N));
  auto p = tf::make_points<3>(points.make_range());
  auto polys = tf::make_polygons(f, p);
  return triangulated_wasm_mesh<Real>(polys, [&points] { return points; });
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
    // `tf::triangulated` cleans a soup itself, but it must return a
    // self-contained table; this call OWNS the cleaned points and donates
    // them, so it states the clean here and the sharing path below moves them
    // into the result rather than copying — and only that path asks
    auto buffer = tf::cleaned(polys);
    auto n_pts = static_cast<int>(buffer.points_buffer().size());
    return triangulated_wasm_mesh<Real>(buffer.polygons(), [&buffer, n_pts] {
      return wasm_ndarray<Real>::from_buffer(
          std::move(buffer.points_buffer().data_buffer()), {n_pts, 3});
    });
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
