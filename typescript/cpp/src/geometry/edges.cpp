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

#include "trueform/geometry/make_sharp_edges.hpp"
#include "trueform/ts/core/promise.hpp"
#include "trueform/ts/core/wasm_mesh.hpp"
#include <emscripten/bind.h>

namespace {

using namespace tf::ts;

auto sync_sharp_edges(wasm_mesh &m, float angle_deg) -> wasm_ndarray<int> {
  auto poly = m.polygons_range() | tf::tag(m.manifold_edge_link_range());
  auto edges = tf::make_sharp_edges(poly, tf::deg{angle_deg});
  auto n = static_cast<int>(edges.size());
  return wasm_ndarray<int>::from_buffer(std::move(edges.data_buffer()), {n, 2});
}

auto async_sharp_edges(wasm_mesh &m, float angle_deg) -> promise_t {
  return promise([a = m, angle_deg]() -> wasm_ndarray<int> {
    return sync_sharp_edges(const_cast<wasm_mesh &>(a), angle_deg);
  });
}

} // namespace

EMSCRIPTEN_BINDINGS(trueform_geometry_edges) {
  emscripten::function("sharp_edges", &sync_sharp_edges);
  emscripten::function("dispatch_sharp_edges", &async_sharp_edges);
}
