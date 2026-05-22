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

#include "trueform/clean/config.hpp"
#include "trueform/clean/points.hpp"
#include "trueform/clean/polygons.hpp"
#include "trueform/ts/core/promise.hpp"
#include "trueform/ts/core/wasm_index_map.hpp"
#include "trueform/ts/core/wasm_mesh.hpp"
#include "trueform/ts/core/wasm_ndarray.hpp"
#include <emscripten/bind.h>

namespace tf {
namespace ts {

// ============================================================================
// Result types — the cleaned mesh and points carry the input's coordinate
// dtype; index maps stay int32.
// ============================================================================

template <typename Real> struct cleaned_mesh_result_t {
  wasm_mesh<Real> mesh;
  wasm_index_map face_map;
  wasm_index_map point_map;
};

template <typename Real> struct cleaned_points_result_t {
  wasm_ndarray<Real> points; // [N, 3]
  wasm_index_map point_map;
};

// ============================================================================
// Polygon soup → Mesh
// `clean_config_t` does not apply to soups. Bools accepted for API uniformity
// but ignored.
// ============================================================================

template <typename Real>
auto sync_cleaned_polygon_soup(wasm_ndarray<Real> &poly, Real tolerance,
                               bool /*remove_duplicate_primitives*/,
                               bool /*remove_unreferenced_points*/)
    -> wasm_mesh<Real> {
  auto pts = tf::make_points<3>(poly.make_range());
  auto polys = tf::make_polygons(tf::make_blocked_range<3>(pts));
  if (tolerance > 0)
    return wasm_mesh<Real>::from_polygons_buffer(tf::cleaned(polys, tolerance));
  return wasm_mesh<Real>::from_polygons_buffer(tf::cleaned(polys));
}

template <typename Real>
auto async_cleaned_polygon_soup(wasm_ndarray<Real> &poly, Real tolerance,
                                bool remove_duplicate_primitives,
                                bool remove_unreferenced_points) -> promise_t {
  return promise([a = poly, tolerance, remove_duplicate_primitives,
                  remove_unreferenced_points]() -> wasm_mesh<Real> {
    return sync_cleaned_polygon_soup<Real>(
        const_cast<wasm_ndarray<Real> &>(a), tolerance,
        remove_duplicate_primitives, remove_unreferenced_points);
  });
}

// ============================================================================
// Mesh → Mesh
// ============================================================================

template <typename Real>
auto sync_cleaned_mesh(wasm_mesh<Real> &m, Real tolerance,
                       bool remove_duplicate_primitives,
                       bool remove_unreferenced_points) -> wasm_mesh<Real> {
  auto poly = m.polygons_range();
  tf::clean_config_t<Real> cfg{tolerance, remove_duplicate_primitives,
                               remove_unreferenced_points};
  return wasm_mesh<Real>::from_polygons_buffer(tf::cleaned(poly, cfg));
}

template <typename Real>
auto async_cleaned_mesh(wasm_mesh<Real> &m, Real tolerance,
                        bool remove_duplicate_primitives,
                        bool remove_unreferenced_points) -> promise_t {
  return promise([a = m, tolerance, remove_duplicate_primitives,
                  remove_unreferenced_points]() -> wasm_mesh<Real> {
    return sync_cleaned_mesh<Real>(const_cast<wasm_mesh<Real> &>(a), tolerance,
                                   remove_duplicate_primitives,
                                   remove_unreferenced_points);
  });
}

// ============================================================================
// Mesh → Mesh + IndexMaps
// ============================================================================

template <typename Real>
auto sync_cleaned_mesh_with_maps(wasm_mesh<Real> &m, Real tolerance,
                                 bool remove_duplicate_primitives,
                                 bool remove_unreferenced_points)
    -> cleaned_mesh_result_t<Real> {
  auto poly = m.polygons_range();
  tf::clean_config_t<Real> cfg{tolerance, remove_duplicate_primitives,
                               remove_unreferenced_points};
  auto [result, face_im, point_im] =
      tf::cleaned(poly, cfg, tf::return_index_map);
  return {
      wasm_mesh<Real>::from_polygons_buffer(std::move(result)),
      wasm_index_map::from_index_map_buffer(std::move(face_im)),
      wasm_index_map::from_index_map_buffer(std::move(point_im)),
  };
}

template <typename Real>
auto async_cleaned_mesh_with_maps(wasm_mesh<Real> &m, Real tolerance,
                                  bool remove_duplicate_primitives,
                                  bool remove_unreferenced_points)
    -> promise_t {
  return promise([a = m, tolerance, remove_duplicate_primitives,
                  remove_unreferenced_points]() -> cleaned_mesh_result_t<Real> {
    return sync_cleaned_mesh_with_maps<Real>(const_cast<wasm_mesh<Real> &>(a),
                                             tolerance,
                                             remove_duplicate_primitives,
                                             remove_unreferenced_points);
  });
}

// ============================================================================
// Points → Points
// `clean_config_t` does not apply to point cleaning. Bools accepted for
// API uniformity but ignored.
// ============================================================================

template <typename Real>
auto sync_cleaned_points(wasm_ndarray<Real> &arr, Real tolerance,
                         bool /*remove_duplicate_primitives*/,
                         bool /*remove_unreferenced_points*/)
    -> wasm_ndarray<Real> {
  auto points = tf::make_points<3>(arr.make_range());
  if (tolerance > 0) {
    auto result = tf::cleaned(points, tolerance);
    auto n = static_cast<int>(result.size());
    return wasm_ndarray<Real>::from_buffer(std::move(result.data_buffer()),
                                           {n, 3});
  }
  auto result = tf::cleaned(points);
  auto n = static_cast<int>(result.size());
  return wasm_ndarray<Real>::from_buffer(std::move(result.data_buffer()),
                                         {n, 3});
}

template <typename Real>
auto async_cleaned_points(wasm_ndarray<Real> &arr, Real tolerance,
                          bool remove_duplicate_primitives,
                          bool remove_unreferenced_points) -> promise_t {
  return promise([a = arr, tolerance, remove_duplicate_primitives,
                  remove_unreferenced_points]() -> wasm_ndarray<Real> {
    return sync_cleaned_points<Real>(const_cast<wasm_ndarray<Real> &>(a),
                                     tolerance, remove_duplicate_primitives,
                                     remove_unreferenced_points);
  });
}

// ============================================================================
// Points → Points + IndexMap
// ============================================================================

template <typename Real>
auto sync_cleaned_points_with_map(wasm_ndarray<Real> &arr, Real tolerance,
                                  bool /*remove_duplicate_primitives*/,
                                  bool /*remove_unreferenced_points*/)
    -> cleaned_points_result_t<Real> {
  auto points = tf::make_points<3>(arr.make_range());
  if (tolerance > 0) {
    auto [result, point_im] =
        tf::cleaned(points, tolerance, tf::return_index_map);
    auto n = static_cast<int>(result.size());
    return {
        wasm_ndarray<Real>::from_buffer(std::move(result.data_buffer()),
                                        {n, 3}),
        wasm_index_map::from_index_map_buffer(std::move(point_im)),
    };
  }
  auto [result, point_im] = tf::cleaned(points, tf::return_index_map);
  auto n = static_cast<int>(result.size());
  return {
      wasm_ndarray<Real>::from_buffer(std::move(result.data_buffer()), {n, 3}),
      wasm_index_map::from_index_map_buffer(std::move(point_im)),
  };
}

template <typename Real>
auto async_cleaned_points_with_map(wasm_ndarray<Real> &arr, Real tolerance,
                                   bool remove_duplicate_primitives,
                                   bool remove_unreferenced_points)
    -> promise_t {
  return promise([a = arr, tolerance, remove_duplicate_primitives,
                  remove_unreferenced_points]()
                     -> cleaned_points_result_t<Real> {
    return sync_cleaned_points_with_map<Real>(
        const_cast<wasm_ndarray<Real> &>(a), tolerance,
        remove_duplicate_primitives, remove_unreferenced_points);
  });
}

} // namespace ts
} // namespace tf
