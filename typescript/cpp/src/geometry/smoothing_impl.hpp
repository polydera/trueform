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

#include "trueform/geometry/laplacian_smoothed.hpp"
#include "trueform/geometry/taubin_smoothed.hpp"
#include "trueform/ts/core/promise.hpp"
#include "trueform/ts/core/wasm_mesh.hpp"

namespace tf {
namespace ts {

template <typename Real>
auto sync_laplacian_smoothed(wasm_mesh<Real> &m, int iterations, Real lambda)
    -> wasm_mesh<Real> {
  auto vl = m.vertex_link_range();
  auto pts = m.points_range() | tf::tag(vl);

  auto smoothed =
      tf::laplacian_smoothed(pts, static_cast<std::size_t>(iterations), lambda);

  auto result_points = wasm_ndarray<Real>::from_buffer(
      std::move(smoothed.data_buffer()), {m.number_of_points(), 3});
  return wasm_mesh<Real>::create(m.faces(), std::move(result_points));
}

template <typename Real>
auto sync_taubin_smoothed(wasm_mesh<Real> &m, int iterations, Real lambda,
                          Real kpb) -> wasm_mesh<Real> {
  auto vl = m.vertex_link_range();
  auto pts = m.points_range() | tf::tag(vl);

  auto smoothed = tf::taubin_smoothed(
      pts, static_cast<std::size_t>(iterations), lambda, kpb);

  auto result_points = wasm_ndarray<Real>::from_buffer(
      std::move(smoothed.data_buffer()), {m.number_of_points(), 3});
  return wasm_mesh<Real>::create(m.faces(), std::move(result_points));
}

template <typename Real>
auto async_laplacian_smoothed(wasm_mesh<Real> &m, int iterations, Real lambda)
    -> promise_t {
  return promise([a = m, iterations, lambda]() -> wasm_mesh<Real> {
    return sync_laplacian_smoothed<Real>(const_cast<wasm_mesh<Real> &>(a),
                                         iterations, lambda);
  });
}

template <typename Real>
auto async_taubin_smoothed(wasm_mesh<Real> &m, int iterations, Real lambda,
                           Real kpb) -> promise_t {
  return promise([a = m, iterations, lambda, kpb]() -> wasm_mesh<Real> {
    return sync_taubin_smoothed<Real>(const_cast<wasm_mesh<Real> &>(a),
                                      iterations, lambda, kpb);
  });
}

} // namespace ts
} // namespace tf
