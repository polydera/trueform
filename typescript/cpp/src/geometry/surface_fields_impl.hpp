/*
 * Copyright (c) 2026 XLAB
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

#include "trueform/core/none.hpp"
#include "trueform/geometry/dirichlet_order.hpp"
#include "trueform/geometry/make_dirichlet_field.hpp"
#include "trueform/geometry/make_geodesic_distances.hpp"
#include "trueform/ts/core/promise.hpp"
#include "trueform/ts/core/wasm_mesh.hpp"

namespace tf {
namespace ts {

// The field is in the mesh's dtype; `status` is the tf::dirichlet_field_status
// value.
template <typename Real> struct dirichlet_field_t {
  wasm_ndarray<Real> field;
  int status;
};

// The ids, values and weights arrive validated by the facade: ids in range
// (and distinct for the Dirichlet field), values finite, three weights per
// face, and `order` a tf::dirichlet_order value.

// -- Sync --

template <typename Real>
auto sync_geodesic_distances(wasm_mesh<Real> &m, wasm_ndarray<int> &seeds,
                             double radius) -> wasm_ndarray<Real> {
  auto polys = m.polygons_range() | tf::tag(m.face_membership_range()) |
               tf::tag(m.vertex_link_range());
  return wasm_ndarray<Real>::from_buffer(
      tf::make_geodesic_distances(polys, seeds.make_range(), Real(radius)),
      {m.number_of_points()});
}

template <typename Real, typename Weights>
auto dirichlet_field_of(wasm_mesh<Real> &m, const Weights &corner_weights,
                        wasm_ndarray<int> &fixed_ids,
                        wasm_ndarray<double> &fixed_values, int order)
    -> dirichlet_field_t<Real> {
  auto result = tf::make_dirichlet_field(
      m.polygons_range() | tf::tag(m.face_membership_range()), corner_weights,
      fixed_ids.make_range(), fixed_values.make_range(),
      static_cast<tf::dirichlet_order>(order));
  return {wasm_ndarray<Real>::from_buffer(std::move(result.field),
                                          {m.number_of_points()}),
          static_cast<int>(result.status)};
}

template <typename Real>
auto sync_dirichlet_field(wasm_mesh<Real> &m,
                          wasm_ndarray<double> &corner_weights,
                          wasm_ndarray<int> &fixed_ids,
                          wasm_ndarray<double> &fixed_values, int order)
    -> dirichlet_field_t<Real> {
  return dirichlet_field_of(m, corner_weights.make_range(), fixed_ids,
                            fixed_values, order);
}

template <typename Real>
auto sync_dirichlet_field_cotangent(wasm_mesh<Real> &m,
                                    wasm_ndarray<int> &fixed_ids,
                                    wasm_ndarray<double> &fixed_values,
                                    int order) -> dirichlet_field_t<Real> {
  return dirichlet_field_of(m, tf::none, fixed_ids, fixed_values, order);
}

// -- Async --

template <typename Real>
auto async_geodesic_distances(wasm_mesh<Real> &m, wasm_ndarray<int> &seeds,
                              double radius) -> promise_t {
  return promise([a = m, s = seeds, radius]() -> wasm_ndarray<Real> {
    return sync_geodesic_distances<Real>(const_cast<wasm_mesh<Real> &>(a),
                                         const_cast<wasm_ndarray<int> &>(s),
                                         radius);
  });
}

template <typename Real>
auto async_dirichlet_field(wasm_mesh<Real> &m,
                           wasm_ndarray<double> &corner_weights,
                           wasm_ndarray<int> &fixed_ids,
                           wasm_ndarray<double> &fixed_values, int order)
    -> promise_t {
  return promise([a = m, w = corner_weights, i = fixed_ids, v = fixed_values,
                  order]() -> dirichlet_field_t<Real> {
    return sync_dirichlet_field<Real>(const_cast<wasm_mesh<Real> &>(a),
                                      const_cast<wasm_ndarray<double> &>(w),
                                      const_cast<wasm_ndarray<int> &>(i),
                                      const_cast<wasm_ndarray<double> &>(v),
                                      order);
  });
}

template <typename Real>
auto async_dirichlet_field_cotangent(wasm_mesh<Real> &m,
                                     wasm_ndarray<int> &fixed_ids,
                                     wasm_ndarray<double> &fixed_values,
                                     int order) -> promise_t {
  return promise([a = m, i = fixed_ids, v = fixed_values,
                  order]() -> dirichlet_field_t<Real> {
    return sync_dirichlet_field_cotangent<Real>(
        const_cast<wasm_mesh<Real> &>(a), const_cast<wasm_ndarray<int> &>(i),
        const_cast<wasm_ndarray<double> &>(v), order);
  });
}

} // namespace ts
} // namespace tf
