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

#include "trueform/core/area.hpp"
#include "trueform/core/max_edge_length.hpp"
#include "trueform/core/mean_edge_length.hpp"
#include "trueform/core/min_edge_length.hpp"
#include "trueform/core/signed_volume.hpp"
#include "trueform/ts/core/promise.hpp"
#include "trueform/ts/core/wasm_mesh.hpp"

namespace tf {
namespace ts {

// Helper: dispatch with/without transformation
template <typename Real, typename Fn>
auto with_polys(wasm_mesh<Real> &m, Fn &&fn) -> decltype(auto) {
  if (m.has_transformation()) {
    return fn(m.polygons_range() | tf::tag(m.transformation_view()));
  } else {
    return fn(m.polygons_range());
  }
}

// -- Sync --

template <typename Real>
auto sync_area(wasm_mesh<Real> &m) -> Real {
  return with_polys(m, [](auto &&polys) -> Real { return tf::area(polys); });
}

template <typename Real>
auto sync_signed_volume(wasm_mesh<Real> &m) -> Real {
  return with_polys(
      m, [](auto &&polys) -> Real { return tf::signed_volume(polys); });
}

template <typename Real>
auto sync_mean_edge_length(wasm_mesh<Real> &m) -> Real {
  return with_polys(
      m, [](auto &&polys) -> Real { return tf::mean_edge_length(polys); });
}

template <typename Real>
auto sync_min_edge_length(wasm_mesh<Real> &m) -> Real {
  return with_polys(
      m, [](auto &&polys) -> Real { return tf::min_edge_length(polys); });
}

template <typename Real>
auto sync_max_edge_length(wasm_mesh<Real> &m) -> Real {
  return with_polys(
      m, [](auto &&polys) -> Real { return tf::max_edge_length(polys); });
}

// -- Async --

template <typename Real>
auto async_area(wasm_mesh<Real> &m) -> promise_t {
  return promise([a = m]() -> Real {
    return sync_area<Real>(const_cast<wasm_mesh<Real> &>(a));
  });
}

template <typename Real>
auto async_signed_volume(wasm_mesh<Real> &m) -> promise_t {
  return promise([a = m]() -> Real {
    return sync_signed_volume<Real>(const_cast<wasm_mesh<Real> &>(a));
  });
}

template <typename Real>
auto async_mean_edge_length(wasm_mesh<Real> &m) -> promise_t {
  return promise([a = m]() -> Real {
    return sync_mean_edge_length<Real>(const_cast<wasm_mesh<Real> &>(a));
  });
}

template <typename Real>
auto async_min_edge_length(wasm_mesh<Real> &m) -> promise_t {
  return promise([a = m]() -> Real {
    return sync_min_edge_length<Real>(const_cast<wasm_mesh<Real> &>(a));
  });
}

template <typename Real>
auto async_max_edge_length(wasm_mesh<Real> &m) -> promise_t {
  return promise([a = m]() -> Real {
    return sync_max_edge_length<Real>(const_cast<wasm_mesh<Real> &>(a));
  });
}

} // namespace ts
} // namespace tf
