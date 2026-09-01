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

#include "trueform/csg/make_outer_shell.hpp"
#include "trueform/ts/core/build_intersect_structures.hpp"
#include "trueform/ts/core/promise.hpp"
#include "trueform/ts/core/wasm_mesh.hpp"

namespace tf {
namespace ts {

// ============================================================================
// Sync entry point — tag the polygons with the cached tree / topology and
// dispatch to the trueform outer-shell repair pipeline. The intersect
// config stays at the library default.
// ============================================================================

template <typename Real>
auto sync_outer_shell(wasm_mesh<Real> &m) -> wasm_mesh<Real> {
  build_intersect_structures(m);
  auto fm = m.face_membership_range();
  auto mel = m.manifold_edge_link_range();
  auto shell = tf::make_outer_shell(
      m.polygons_range() | tf::tag(m.tree()) | tf::tag(fm) | tf::tag(mel),
      tf::intersect_config{tf::intersect_mode::primitives |
                           tf::intersect_mode::resolve_contours});
  return wasm_mesh<Real>::from_polygons_buffer(std::move(shell));
}

// ============================================================================
// Async — capture the handle by copy (shared_ptr refcount++) and run the
// sync path on the worker thread.
// ============================================================================

template <typename Real>
auto async_outer_shell(wasm_mesh<Real> &m) -> promise_t {
  return promise([a = m]() -> wasm_mesh<Real> {
    return sync_outer_shell<Real>(const_cast<wasm_mesh<Real> &>(a));
  });
}

} // namespace ts
} // namespace tf
