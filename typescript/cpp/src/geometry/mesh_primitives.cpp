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

#include "trueform/geometry/make_sphere_mesh.hpp"
#include "trueform/geometry/make_cylinder_mesh.hpp"
#include "trueform/geometry/make_box_mesh.hpp"
#include "trueform/geometry/make_plane_mesh.hpp"
#include "trueform/ts/core/promise.hpp"
#include "trueform/ts/core/wasm_mesh.hpp"
#include <emscripten/bind.h>

namespace {

using namespace tf::ts;

// -- Sync --

auto sync_make_sphere_mesh(float radius, int stacks, int segments)
    -> wasm_mesh {
  auto result = tf::make_sphere_mesh(radius, stacks, segments);
  return wasm_mesh::from_polygons_buffer(std::move(result));
}

auto sync_make_cylinder_mesh(float radius, float height, int segments)
    -> wasm_mesh {
  auto result = tf::make_cylinder_mesh(radius, height, segments);
  return wasm_mesh::from_polygons_buffer(std::move(result));
}

auto sync_make_box_mesh(float width, float height, float depth) -> wasm_mesh {
  auto result = tf::make_box_mesh(width, height, depth);
  return wasm_mesh::from_polygons_buffer(std::move(result));
}

auto sync_make_box_mesh_subdivided(float width, float height, float depth,
                                   int width_ticks, int height_ticks,
                                   int depth_ticks) -> wasm_mesh {
  auto result = tf::make_box_mesh(width, height, depth, width_ticks,
                                  height_ticks, depth_ticks);
  return wasm_mesh::from_polygons_buffer(std::move(result));
}

auto sync_make_plane_mesh(float width, float height, int width_ticks,
                          int height_ticks) -> wasm_mesh {
  auto result = tf::make_plane_mesh(width, height, width_ticks, height_ticks);
  return wasm_mesh::from_polygons_buffer(std::move(result));
}

// -- Async --

auto async_make_sphere_mesh(float radius, int stacks, int segments)
    -> promise_t {
  return promise([=]() -> wasm_mesh {
    return sync_make_sphere_mesh(radius, stacks, segments);
  });
}

auto async_make_cylinder_mesh(float radius, float height, int segments)
    -> promise_t {
  return promise([=]() -> wasm_mesh {
    return sync_make_cylinder_mesh(radius, height, segments);
  });
}

auto async_make_box_mesh(float width, float height, float depth) -> promise_t {
  return promise([=]() -> wasm_mesh {
    return sync_make_box_mesh(width, height, depth);
  });
}

auto async_make_box_mesh_subdivided(float width, float height, float depth,
                                    int width_ticks, int height_ticks,
                                    int depth_ticks) -> promise_t {
  return promise([=]() -> wasm_mesh {
    return sync_make_box_mesh_subdivided(width, height, depth, width_ticks,
                                        height_ticks, depth_ticks);
  });
}

auto async_make_plane_mesh(float width, float height, int width_ticks,
                           int height_ticks) -> promise_t {
  return promise([=]() -> wasm_mesh {
    return sync_make_plane_mesh(width, height, width_ticks, height_ticks);
  });
}

} // namespace

EMSCRIPTEN_BINDINGS(trueform_geometry_mesh_primitives) {
  // Sync
  emscripten::function("make_sphere_mesh", &sync_make_sphere_mesh);
  emscripten::function("make_cylinder_mesh", &sync_make_cylinder_mesh);
  emscripten::function("make_box_mesh", &sync_make_box_mesh);
  emscripten::function("make_box_mesh_subdivided",
                       &sync_make_box_mesh_subdivided);
  emscripten::function("make_plane_mesh", &sync_make_plane_mesh);

  // Async
  emscripten::function("dispatch_make_sphere_mesh", &async_make_sphere_mesh);
  emscripten::function("dispatch_make_cylinder_mesh",
                       &async_make_cylinder_mesh);
  emscripten::function("dispatch_make_box_mesh", &async_make_box_mesh);
  emscripten::function("dispatch_make_box_mesh_subdivided",
                       &async_make_box_mesh_subdivided);
  emscripten::function("dispatch_make_plane_mesh", &async_make_plane_mesh);
}
