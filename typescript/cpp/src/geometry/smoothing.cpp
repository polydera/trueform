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

#include "trueform/geometry/taubin_smoothed.hpp"
#include "trueform/ts/core/promise.hpp"
#include "trueform/ts/core/wasm_mesh.hpp"
#include <emscripten/bind.h>

namespace {

using namespace tf::ts;

// -- Sync --

auto sync_laplacian_smoothed(wasm_mesh &m, int iterations, float lambda)
    -> wasm_mesh {
  auto vl = m.vertex_link_range();
  auto pts = m.points_range() | tf::tag(vl);

  auto smoothed =
      tf::laplacian_smoothed(pts, static_cast<std::size_t>(iterations), lambda);

  auto result_points = wasm_ndarray<float>::from_buffer(
      std::move(smoothed.data_buffer()),
      {m.number_of_points(), 3});
  return wasm_mesh::create(m.faces(), std::move(result_points));
}

auto sync_taubin_smoothed(wasm_mesh &m, int iterations, float lambda,
                          float kpb) -> wasm_mesh {
  auto vl = m.vertex_link_range();
  auto pts = m.points_range() | tf::tag(vl);

  auto smoothed = tf::taubin_smoothed(
      pts, static_cast<std::size_t>(iterations), lambda, kpb);

  auto result_points = wasm_ndarray<float>::from_buffer(
      std::move(smoothed.data_buffer()),
      {m.number_of_points(), 3});
  return wasm_mesh::create(m.faces(), std::move(result_points));
}

// -- Async --

auto async_laplacian_smoothed(wasm_mesh &m, int iterations, float lambda)
    -> promise_t {
  return promise([a = m, iterations, lambda]() -> wasm_mesh {
    return sync_laplacian_smoothed(const_cast<wasm_mesh &>(a), iterations,
                                   lambda);
  });
}

auto async_taubin_smoothed(wasm_mesh &m, int iterations, float lambda,
                           float kpb) -> promise_t {
  return promise([a = m, iterations, lambda, kpb]() -> wasm_mesh {
    return sync_taubin_smoothed(const_cast<wasm_mesh &>(a), iterations, lambda,
                                kpb);
  });
}

} // namespace

EMSCRIPTEN_BINDINGS(trueform_geometry_smoothing) {
  emscripten::function("laplacian_smoothed", &sync_laplacian_smoothed);
  emscripten::function("taubin_smoothed", &sync_taubin_smoothed);

  emscripten::function("dispatch_laplacian_smoothed",
                       &async_laplacian_smoothed);
  emscripten::function("dispatch_taubin_smoothed", &async_taubin_smoothed);
}
