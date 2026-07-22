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

#include "trueform/cut/make_polygon_arrangements.hpp"
#include "trueform/topology/triangulation_type.hpp"
#include "trueform/ts/core/build_intersect_structures.hpp"
#include "trueform/ts/core/promise.hpp"
#include "trueform/ts/core/wasm_curves.hpp"
#include "trueform/ts/core/wasm_mesh.hpp"
#include "trueform/ts/core/wasm_ndarray.hpp"
#include <cstdint>

namespace tf {
namespace ts {

// ============================================================================
// Result types — the split mesh + per-face origin labels carry the input's
// coordinate dtype; face labels stay int32. Curves match the mesh dtype.
// ============================================================================

template <typename Real> struct polygon_arrangement_result_t {
  wasm_mesh<Real> mesh;
  wasm_ndarray<std::int32_t> face_labels;
};

template <typename Real> struct polygon_arrangement_result_with_curves_t {
  wasm_mesh<Real> mesh;
  wasm_ndarray<std::int32_t> face_labels;
  wasm_curves<Real> curves;
};

// ============================================================================
// Sync entry points — tag the polygons with the cached tree / topology and
// dispatch to the trueform arrangement pipeline.
// ============================================================================

template <typename Real>
auto sync_polygon_arrangements(wasm_mesh<Real> &m, int mode, double tolerance,
                              int triangulation)
    -> polygon_arrangement_result_t<Real> {
  build_intersect_structures(m);
  auto fm = m.face_membership_range();
  auto mel = m.manifold_edge_link_range();
  auto [mesh, face_labels] = tf::make_polygon_arrangements(
      m.polygons_range() | tf::tag(m.tree()) | tf::tag(fm) | tf::tag(mel),
      tf::arrangement_config{
          tf::intersect_config{static_cast<tf::intersect_mode>(mode),
                               tolerance},
          static_cast<tf::triangulation_type>(triangulation)});
  return {wasm_mesh<Real>::from_polygons_buffer(std::move(mesh)),
          wasm_ndarray<std::int32_t>::from_buffer(std::move(face_labels))};
}

template <typename Real>
auto sync_polygon_arrangements_with_curves(wasm_mesh<Real> &m, int mode,
                                           double tolerance, int triangulation)
    -> polygon_arrangement_result_with_curves_t<Real> {
  build_intersect_structures(m);
  auto fm = m.face_membership_range();
  auto mel = m.manifold_edge_link_range();
  auto [mesh, face_labels, curves] = tf::make_polygon_arrangements(
      m.polygons_range() | tf::tag(m.tree()) | tf::tag(fm) | tf::tag(mel),
      tf::arrangement_config{
          tf::intersect_config{static_cast<tf::intersect_mode>(mode),
                               tolerance},
          static_cast<tf::triangulation_type>(triangulation)},
      tf::return_curves);
  return {wasm_mesh<Real>::from_polygons_buffer(std::move(mesh)),
          wasm_ndarray<std::int32_t>::from_buffer(std::move(face_labels)),
          wasm_curves<Real>::from_curves_buffer(std::move(curves))};
}

// ============================================================================
// Async — capture the handle by copy (shared_ptr refcount++) and run the
// sync path on the worker thread.
// ============================================================================

template <typename Real>
auto async_polygon_arrangements(wasm_mesh<Real> &m, int mode, double tolerance,
                               int triangulation) -> promise_t {
  return promise([a = m, mode, tolerance,
                  triangulation]() -> polygon_arrangement_result_t<Real> {
    return sync_polygon_arrangements<Real>(const_cast<wasm_mesh<Real> &>(a),
                                           mode, tolerance, triangulation);
  });
}

template <typename Real>
auto async_polygon_arrangements_with_curves(wasm_mesh<Real> &m, int mode,
                                            double tolerance, int triangulation)
    -> promise_t {
  return promise([a = m, mode, tolerance, triangulation]()
                     -> polygon_arrangement_result_with_curves_t<Real> {
    return sync_polygon_arrangements_with_curves<Real>(
        const_cast<wasm_mesh<Real> &>(a), mode, tolerance, triangulation);
  });
}

} // namespace ts
} // namespace tf
