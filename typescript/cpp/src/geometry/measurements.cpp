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

#include "trueform/core/area.hpp"
#include "trueform/core/signed_volume.hpp"
#include "trueform/core/mean_edge_length.hpp"
#include "trueform/core/min_edge_length.hpp"
#include "trueform/core/max_edge_length.hpp"
#include "trueform/ts/core/promise.hpp"
#include "trueform/ts/core/wasm_mesh.hpp"
#include <emscripten/bind.h>

namespace {

using namespace tf::ts;

// Helper: dispatch with/without transformation
template <typename Fn>
auto with_polys(wasm_mesh &m, Fn &&fn) -> decltype(auto) {
  if (m.has_transformation()) {
    return fn(m.polygons_range() | tf::tag(m.transformation_view()));
  } else {
    return fn(m.polygons_range());
  }
}

// -- Sync --

auto sync_area(wasm_mesh &m) -> float {
  return with_polys(m, [](auto &&polys) -> float { return tf::area(polys); });
}

auto sync_signed_volume(wasm_mesh &m) -> float {
  return with_polys(
      m, [](auto &&polys) -> float { return tf::signed_volume(polys); });
}

auto sync_mean_edge_length(wasm_mesh &m) -> float {
  return with_polys(
      m, [](auto &&polys) -> float { return tf::mean_edge_length(polys); });
}

auto sync_min_edge_length(wasm_mesh &m) -> float {
  return with_polys(
      m, [](auto &&polys) -> float { return tf::min_edge_length(polys); });
}

auto sync_max_edge_length(wasm_mesh &m) -> float {
  return with_polys(
      m, [](auto &&polys) -> float { return tf::max_edge_length(polys); });
}

// -- Async --

auto async_area(wasm_mesh &m) -> promise_t {
  return promise(
      [a = m]() -> float { return sync_area(const_cast<wasm_mesh &>(a)); });
}

auto async_signed_volume(wasm_mesh &m) -> promise_t {
  return promise([a = m]() -> float {
    return sync_signed_volume(const_cast<wasm_mesh &>(a));
  });
}

auto async_mean_edge_length(wasm_mesh &m) -> promise_t {
  return promise([a = m]() -> float {
    return sync_mean_edge_length(const_cast<wasm_mesh &>(a));
  });
}

auto async_min_edge_length(wasm_mesh &m) -> promise_t {
  return promise([a = m]() -> float {
    return sync_min_edge_length(const_cast<wasm_mesh &>(a));
  });
}

auto async_max_edge_length(wasm_mesh &m) -> promise_t {
  return promise([a = m]() -> float {
    return sync_max_edge_length(const_cast<wasm_mesh &>(a));
  });
}

} // namespace

EMSCRIPTEN_BINDINGS(trueform_geometry_measurements) {
  // Sync
  emscripten::function("mesh_area", &sync_area);
  emscripten::function("mesh_signed_volume", &sync_signed_volume);
  emscripten::function("mesh_mean_edge_length", &sync_mean_edge_length);
  emscripten::function("mesh_min_edge_length", &sync_min_edge_length);
  emscripten::function("mesh_max_edge_length", &sync_max_edge_length);

  // Async
  emscripten::function("dispatch_mesh_area", &async_area);
  emscripten::function("dispatch_mesh_signed_volume", &async_signed_volume);
  emscripten::function("dispatch_mesh_mean_edge_length",
                       &async_mean_edge_length);
  emscripten::function("dispatch_mesh_min_edge_length",
                       &async_min_edge_length);
  emscripten::function("dispatch_mesh_max_edge_length",
                       &async_max_edge_length);
}
