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

#include "trueform/geometry/make_sharp_edges.hpp"
#include "trueform/ts/core/promise.hpp"
#include "trueform/ts/core/wasm_mesh.hpp"

namespace tf {
namespace ts {

template <typename Real>
auto sync_sharp_edges(wasm_mesh<Real> &m, Real angle_deg) -> wasm_ndarray<int> {
  auto poly = m.polygons_range() | tf::tag(m.manifold_edge_link_range());
  auto edges = tf::make_sharp_edges(poly, tf::deg{angle_deg});
  auto n = static_cast<int>(edges.size());
  return wasm_ndarray<int>::from_buffer(std::move(edges.data_buffer()), {n, 2});
}

template <typename Real>
auto async_sharp_edges(wasm_mesh<Real> &m, Real angle_deg) -> promise_t {
  return promise([a = m, angle_deg]() -> wasm_ndarray<int> {
    return sync_sharp_edges<Real>(const_cast<wasm_mesh<Real> &>(a), angle_deg);
  });
}

} // namespace ts
} // namespace tf
