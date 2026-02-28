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

#include "trueform/clean/points.hpp"
#include "trueform/clean/polygons.hpp"
#include "trueform/ts/clean/result_types.hpp"
#include "trueform/ts/core/promise.hpp"
#include <emscripten/bind.h>

namespace {

using namespace tf::ts;

// ============================================================================
// Polygon soup → Mesh
// ============================================================================

auto sync_cleaned_polygon_soup(wasm_ndarray<float> &poly, float tolerance)
    -> wasm_mesh {
  auto pts = tf::make_points<3>(poly.make_range());
  auto polys = tf::make_polygons(tf::make_blocked_range<3>(pts));
  if (tolerance > 0)
    return wasm_mesh::from_polygons_buffer(tf::cleaned(polys, tolerance));
  return wasm_mesh::from_polygons_buffer(tf::cleaned(polys));
}

auto async_cleaned_polygon_soup(wasm_ndarray<float> &poly, float tolerance)
    -> promise_t {
  return promise([a = poly, tolerance]() -> wasm_mesh {
    return sync_cleaned_polygon_soup(const_cast<wasm_ndarray<float> &>(a),
                                     tolerance);
  });
}

// ============================================================================
// Mesh → Mesh
// ============================================================================

auto sync_cleaned_mesh(wasm_mesh &m, float tolerance) -> wasm_mesh {
  auto poly = m.polygons_range();
  if (tolerance > 0) {
    auto result = tf::cleaned(poly, tolerance);
    return wasm_mesh::from_polygons_buffer(std::move(result));
  }
  auto result = tf::cleaned(poly);
  return wasm_mesh::from_polygons_buffer(std::move(result));
}

auto async_cleaned_mesh(wasm_mesh &m, float tolerance) -> promise_t {
  return promise([a = m, tolerance]() -> wasm_mesh {
    return sync_cleaned_mesh(const_cast<wasm_mesh &>(a), tolerance);
  });
}

// ============================================================================
// Mesh → Mesh + IndexMaps
// ============================================================================

auto sync_cleaned_mesh_with_maps(wasm_mesh &m, float tolerance)
    -> cleaned_mesh_result {
  auto poly = m.polygons_range();
  if (tolerance > 0) {
    auto [result, face_im, point_im] =
        tf::cleaned(poly, tolerance, tf::return_index_map);
    return {
        wasm_mesh::from_polygons_buffer(std::move(result)),
        wasm_index_map::from_index_map_buffer(std::move(face_im)),
        wasm_index_map::from_index_map_buffer(std::move(point_im)),
    };
  }
  auto [result, face_im, point_im] = tf::cleaned(poly, tf::return_index_map);
  return {
      wasm_mesh::from_polygons_buffer(std::move(result)),
      wasm_index_map::from_index_map_buffer(std::move(face_im)),
      wasm_index_map::from_index_map_buffer(std::move(point_im)),
  };
}

auto async_cleaned_mesh_with_maps(wasm_mesh &m, float tolerance) -> promise_t {
  return promise([a = m, tolerance]() -> cleaned_mesh_result {
    return sync_cleaned_mesh_with_maps(const_cast<wasm_mesh &>(a), tolerance);
  });
}

// ============================================================================
// Points → Points
// ============================================================================

auto sync_cleaned_points(wasm_ndarray<float> &arr, float tolerance)
    -> wasm_ndarray<float> {
  auto points = tf::make_points<3>(arr.make_range());
  if (tolerance > 0) {
    auto result = tf::cleaned(points, tolerance);
    auto n = static_cast<int>(result.size());
    return wasm_ndarray<float>::from_buffer(std::move(result.data_buffer()),
                                            {n, 3});
  }
  auto result = tf::cleaned(points);
  auto n = static_cast<int>(result.size());
  return wasm_ndarray<float>::from_buffer(std::move(result.data_buffer()),
                                          {n, 3});
}

auto async_cleaned_points(wasm_ndarray<float> &arr, float tolerance)
    -> promise_t {
  return promise([a = arr, tolerance]() -> wasm_ndarray<float> {
    return sync_cleaned_points(const_cast<wasm_ndarray<float> &>(a), tolerance);
  });
}

// ============================================================================
// Points → Points + IndexMap
// ============================================================================

auto sync_cleaned_points_with_map(wasm_ndarray<float> &arr, float tolerance)
    -> cleaned_points_result {
  auto points = tf::make_points<3>(arr.make_range());
  if (tolerance > 0) {
    auto [result, point_im] =
        tf::cleaned(points, tolerance, tf::return_index_map);
    auto n = static_cast<int>(result.size());
    return {
        wasm_ndarray<float>::from_buffer(std::move(result.data_buffer()),
                                         {n, 3}),
        wasm_index_map::from_index_map_buffer(std::move(point_im)),
    };
  }
  auto [result, point_im] = tf::cleaned(points, tf::return_index_map);
  auto n = static_cast<int>(result.size());
  return {
      wasm_ndarray<float>::from_buffer(std::move(result.data_buffer()),
                                       {n, 3}),
      wasm_index_map::from_index_map_buffer(std::move(point_im)),
  };
}

auto async_cleaned_points_with_map(wasm_ndarray<float> &arr, float tolerance)
    -> promise_t {
  return promise([a = arr, tolerance]() -> cleaned_points_result {
    return sync_cleaned_points_with_map(const_cast<wasm_ndarray<float> &>(a),
                                        tolerance);
  });
}

} // namespace

EMSCRIPTEN_BINDINGS(trueform_clean) {
  // Result types
  emscripten::value_object<tf::ts::cleaned_mesh_result>("CleanedMeshResult")
      .field("mesh", &tf::ts::cleaned_mesh_result::mesh)
      .field("faceMap", &tf::ts::cleaned_mesh_result::face_map)
      .field("pointMap", &tf::ts::cleaned_mesh_result::point_map);

  emscripten::value_object<tf::ts::cleaned_points_result>(
      "CleanedPointsResult")
      .field("points", &tf::ts::cleaned_points_result::points)
      .field("pointMap", &tf::ts::cleaned_points_result::point_map);

  // Sync
  emscripten::function("cleaned_polygon_soup", &sync_cleaned_polygon_soup);
  emscripten::function("cleaned_mesh", &sync_cleaned_mesh);
  emscripten::function("cleaned_mesh_with_maps", &sync_cleaned_mesh_with_maps);
  emscripten::function("cleaned_points", &sync_cleaned_points);
  emscripten::function("cleaned_points_with_map",
                       &sync_cleaned_points_with_map);

  // Async
  emscripten::function("dispatch_cleaned_polygon_soup",
                       &async_cleaned_polygon_soup);
  emscripten::function("dispatch_cleaned_mesh", &async_cleaned_mesh);
  emscripten::function("dispatch_cleaned_mesh_with_maps",
                       &async_cleaned_mesh_with_maps);
  emscripten::function("dispatch_cleaned_points", &async_cleaned_points);
  emscripten::function("dispatch_cleaned_points_with_map",
                       &async_cleaned_points_with_map);
}
