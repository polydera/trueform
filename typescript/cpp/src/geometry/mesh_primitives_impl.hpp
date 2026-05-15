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

#include "trueform/geometry/make_box_mesh.hpp"
#include "trueform/geometry/make_cylinder_mesh.hpp"
#include "trueform/geometry/make_plane_mesh.hpp"
#include "trueform/geometry/make_sphere_mesh.hpp"
#include "trueform/geometry/make_tube_mesh.hpp"
#include "trueform/ts/core/promise.hpp"
#include "trueform/ts/core/wasm_curves.hpp"
#include "trueform/ts/core/wasm_mesh.hpp"

namespace tf {
namespace ts {

// -- Sync --

template <typename Real>
auto sync_make_sphere_mesh(Real radius, int stacks, int segments)
    -> wasm_mesh<Real> {
  auto result = tf::make_sphere_mesh(radius, stacks, segments);
  return wasm_mesh<Real>::from_polygons_buffer(std::move(result));
}

template <typename Real>
auto sync_make_cylinder_mesh(Real radius, Real height, int segments)
    -> wasm_mesh<Real> {
  auto result = tf::make_cylinder_mesh(radius, height, segments);
  return wasm_mesh<Real>::from_polygons_buffer(std::move(result));
}

template <typename Real>
auto sync_make_box_mesh(Real width, Real height, Real depth)
    -> wasm_mesh<Real> {
  auto result = tf::make_box_mesh(width, height, depth);
  return wasm_mesh<Real>::from_polygons_buffer(std::move(result));
}

template <typename Real>
auto sync_make_box_mesh_subdivided(Real width, Real height, Real depth,
                                   int width_ticks, int height_ticks,
                                   int depth_ticks) -> wasm_mesh<Real> {
  auto result = tf::make_box_mesh(width, height, depth, width_ticks,
                                  height_ticks, depth_ticks);
  return wasm_mesh<Real>::from_polygons_buffer(std::move(result));
}

template <typename Real>
auto sync_make_plane_mesh(Real width, Real height, int width_ticks,
                          int height_ticks) -> wasm_mesh<Real> {
  auto result = tf::make_plane_mesh(width, height, width_ticks, height_ticks);
  return wasm_mesh<Real>::from_polygons_buffer(std::move(result));
}

template <typename Real>
auto sync_make_tube_mesh(wasm_curves<Real> &curves, Real radius,
                         int radial_segments) -> wasm_mesh<Real> {
  auto result =
      tf::make_tube_mesh(curves.curves_range(), radius, radial_segments);
  return wasm_mesh<Real>::from_polygons_buffer(std::move(result));
}

// -- Async --

template <typename Real>
auto async_make_sphere_mesh(Real radius, int stacks, int segments)
    -> promise_t {
  return promise([=]() -> wasm_mesh<Real> {
    return sync_make_sphere_mesh<Real>(radius, stacks, segments);
  });
}

template <typename Real>
auto async_make_cylinder_mesh(Real radius, Real height, int segments)
    -> promise_t {
  return promise([=]() -> wasm_mesh<Real> {
    return sync_make_cylinder_mesh<Real>(radius, height, segments);
  });
}

template <typename Real>
auto async_make_box_mesh(Real width, Real height, Real depth) -> promise_t {
  return promise([=]() -> wasm_mesh<Real> {
    return sync_make_box_mesh<Real>(width, height, depth);
  });
}

template <typename Real>
auto async_make_box_mesh_subdivided(Real width, Real height, Real depth,
                                    int width_ticks, int height_ticks,
                                    int depth_ticks) -> promise_t {
  return promise([=]() -> wasm_mesh<Real> {
    return sync_make_box_mesh_subdivided<Real>(
        width, height, depth, width_ticks, height_ticks, depth_ticks);
  });
}

template <typename Real>
auto async_make_plane_mesh(Real width, Real height, int width_ticks,
                           int height_ticks) -> promise_t {
  return promise([=]() -> wasm_mesh<Real> {
    return sync_make_plane_mesh<Real>(width, height, width_ticks, height_ticks);
  });
}

template <typename Real>
auto async_make_tube_mesh(wasm_curves<Real> &curves, Real radius,
                          int radial_segments) -> promise_t {
  return promise([c = curves, radius, radial_segments]() -> wasm_mesh<Real> {
    return sync_make_tube_mesh<Real>(const_cast<wasm_curves<Real> &>(c), radius,
                                     radial_segments);
  });
}

} // namespace ts
} // namespace tf
