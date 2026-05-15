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

#include "trueform/cut/embedded_intersection_curves.hpp"
#include "trueform/cut/embedded_self_intersection_curves.hpp"
#include "trueform/ts/core/promise.hpp"
#include "trueform/ts/core/wasm_curves.hpp"
#include "trueform/ts/core/wasm_mesh.hpp"
#include "trueform/ts/core/wasm_ndarray.hpp"
#include <cstdint>

namespace tf {
namespace ts {

// ============================================================================
// Result types — the embedded-curve mesh + (optional) intersection curves
// carry the inputs' coordinate dtype; face labels stay int32.
// ============================================================================

template <typename Real> struct cut_result_t {
  wasm_mesh<Real> mesh;
  wasm_ndarray<std::int32_t> face_labels;
};

template <typename Real> struct cut_result_with_curves_t {
  wasm_mesh<Real> mesh;
  wasm_ndarray<std::int32_t> face_labels;
  wasm_curves<Real> curves;
};

// ============================================================================
// Embedded intersection curves (mesh0 × mesh1 → split mesh0)
// Per-mesh transformations stitch into the tagged form-range; the trueform
// pipeline is templated end-to-end on Real.
// ============================================================================

template <typename Real>
auto sync_embedded_intersection_curves(wasm_mesh<Real> &m0,
                                       wasm_mesh<Real> &m1, int mode)
    -> cut_result_t<Real> {
  bool has0 = m0.has_transformation();
  bool has1 = m1.has_transformation();
  auto fm0 = m0.face_membership_range();
  auto mel0 = m0.manifold_edge_link_range();
  auto fm1 = m1.face_membership_range();
  auto mel1 = m1.manifold_edge_link_range();
  auto m = static_cast<tf::intersect_mode>(mode);
  auto make_return = [&](auto &&poly0,
                         auto &&poly1) -> cut_result_t<Real> {
    auto [poly, face_labels] = tf::embedded_intersection_curves(
        poly0 | tf::tag(m0.tree()) | tf::tag(fm0) | tf::tag(mel0),
        poly1 | tf::tag(m1.tree()) | tf::tag(fm1) | tf::tag(mel1), m);
    return {wasm_mesh<Real>::from_polygons_buffer(std::move(poly)),
            wasm_ndarray<std::int32_t>::from_buffer(std::move(face_labels))};
  };
  if (has0 && has1)
    return make_return(
        m0.polygons_range() | tf::tag(m0.transformation_view()),
        m1.polygons_range() | tf::tag(m1.transformation_view()));
  else if (has0)
    return make_return(
        m0.polygons_range() | tf::tag(m0.transformation_view()),
        m1.polygons_range());
  else if (has1)
    return make_return(m0.polygons_range(),
                       m1.polygons_range() |
                           tf::tag(m1.transformation_view()));
  else
    return make_return(m0.polygons_range(), m1.polygons_range());
}

template <typename Real>
auto sync_embedded_intersection_curves_with_curves(wasm_mesh<Real> &m0,
                                                   wasm_mesh<Real> &m1,
                                                   int mode)
    -> cut_result_with_curves_t<Real> {
  bool has0 = m0.has_transformation();
  bool has1 = m1.has_transformation();
  auto fm0 = m0.face_membership_range();
  auto mel0 = m0.manifold_edge_link_range();
  auto fm1 = m1.face_membership_range();
  auto mel1 = m1.manifold_edge_link_range();
  auto m = static_cast<tf::intersect_mode>(mode);
  auto make_return =
      [&](auto &&poly0, auto &&poly1) -> cut_result_with_curves_t<Real> {
    auto [poly, face_labels, curves] = tf::embedded_intersection_curves(
        poly0 | tf::tag(m0.tree()) | tf::tag(fm0) | tf::tag(mel0),
        poly1 | tf::tag(m1.tree()) | tf::tag(fm1) | tf::tag(mel1), m,
        tf::return_curves);
    return {wasm_mesh<Real>::from_polygons_buffer(std::move(poly)),
            wasm_ndarray<std::int32_t>::from_buffer(std::move(face_labels)),
            wasm_curves<Real>::from_curves_buffer(std::move(curves))};
  };
  if (has0 && has1)
    return make_return(
        m0.polygons_range() | tf::tag(m0.transformation_view()),
        m1.polygons_range() | tf::tag(m1.transformation_view()));
  else if (has0)
    return make_return(
        m0.polygons_range() | tf::tag(m0.transformation_view()),
        m1.polygons_range());
  else if (has1)
    return make_return(m0.polygons_range(),
                       m1.polygons_range() |
                           tf::tag(m1.transformation_view()));
  else
    return make_return(m0.polygons_range(), m1.polygons_range());
}

// ============================================================================
// Embedded self-intersection curves (mesh → split mesh)
// ============================================================================

template <typename Real>
auto sync_embedded_self_intersection_curves(wasm_mesh<Real> &m, int mode)
    -> cut_result_t<Real> {
  auto fm = m.face_membership_range();
  auto mel = m.manifold_edge_link_range();
  auto [poly, face_labels] = tf::embedded_self_intersection_curves(
      m.polygons_range() | tf::tag(m.tree()) | tf::tag(fm) | tf::tag(mel),
      static_cast<tf::intersect_mode>(mode));
  return {wasm_mesh<Real>::from_polygons_buffer(std::move(poly)),
          wasm_ndarray<std::int32_t>::from_buffer(std::move(face_labels))};
}

template <typename Real>
auto sync_embedded_self_intersection_curves_with_curves(wasm_mesh<Real> &m,
                                                        int mode)
    -> cut_result_with_curves_t<Real> {
  auto fm = m.face_membership_range();
  auto mel = m.manifold_edge_link_range();
  auto [poly, face_labels, curves] = tf::embedded_self_intersection_curves(
      m.polygons_range() | tf::tag(m.tree()) | tf::tag(fm) | tf::tag(mel),
      static_cast<tf::intersect_mode>(mode), tf::return_curves);
  return {wasm_mesh<Real>::from_polygons_buffer(std::move(poly)),
          wasm_ndarray<std::int32_t>::from_buffer(std::move(face_labels)),
          wasm_curves<Real>::from_curves_buffer(std::move(curves))};
}

// ============================================================================
// Async — capture by copy (shared_ptr refcount bump) and const_cast through
// to the sync entry point on the worker thread.
// ============================================================================

template <typename Real>
auto async_embedded_intersection_curves(wasm_mesh<Real> &m0,
                                        wasm_mesh<Real> &m1, int mode)
    -> promise_t {
  return promise([a = m0, b = m1, mode]() -> cut_result_t<Real> {
    return sync_embedded_intersection_curves<Real>(
        const_cast<wasm_mesh<Real> &>(a),
        const_cast<wasm_mesh<Real> &>(b), mode);
  });
}

template <typename Real>
auto async_embedded_intersection_curves_with_curves(wasm_mesh<Real> &m0,
                                                    wasm_mesh<Real> &m1,
                                                    int mode) -> promise_t {
  return promise(
      [a = m0, b = m1, mode]() -> cut_result_with_curves_t<Real> {
        return sync_embedded_intersection_curves_with_curves<Real>(
            const_cast<wasm_mesh<Real> &>(a),
            const_cast<wasm_mesh<Real> &>(b), mode);
      });
}

template <typename Real>
auto async_embedded_self_intersection_curves(wasm_mesh<Real> &m, int mode)
    -> promise_t {
  return promise([a = m, mode]() -> cut_result_t<Real> {
    return sync_embedded_self_intersection_curves<Real>(
        const_cast<wasm_mesh<Real> &>(a), mode);
  });
}

template <typename Real>
auto async_embedded_self_intersection_curves_with_curves(wasm_mesh<Real> &m,
                                                         int mode)
    -> promise_t {
  return promise([a = m, mode]() -> cut_result_with_curves_t<Real> {
    return sync_embedded_self_intersection_curves_with_curves<Real>(
        const_cast<wasm_mesh<Real> &>(a), mode);
  });
}

} // namespace ts
} // namespace tf
