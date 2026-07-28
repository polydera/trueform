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

#include "trueform/topology/make_cdt.hpp"
#include "trueform/ts/core/promise.hpp"
#include "trueform/ts/core/wasm_index_map.hpp"
#include "trueform/ts/core/wasm_ndarray.hpp"
#include <cstdint>

namespace tf {
namespace ts {

// ============================================================================
// Result types — coordinate NDArrays follow the input point dtype, face
// indices stay int32.
// ============================================================================

template <typename Real> struct cdt_result_t {
  wasm_ndarray<int> faces;      // [K, 3] int32 triangle indices
  wasm_ndarray<Real> points;    // [M, 2] vertex coordinates
};

template <typename Real> struct cdt_result_with_map_t {
  wasm_ndarray<int> faces;
  wasm_ndarray<Real> points;
  wasm_index_map index_map;
};

// ============================================================================
// Helpers — package the make_cdt return into wasm-side result types.
// ============================================================================

template <typename Real, typename Polys>
auto pack_cdt_result(Polys &&polys) -> cdt_result_t<Real> {
  auto n_faces = static_cast<int>(polys.faces().size());
  auto n_points = static_cast<int>(polys.points().size());
  return {
      wasm_ndarray<int>::from_buffer(
          std::move(polys.faces_buffer().data_buffer()), {n_faces, 3}),
      wasm_ndarray<Real>::from_buffer(
          std::move(polys.points_buffer().data_buffer()), {n_points, 2}),
  };
}

template <typename Real, typename Polys>
auto pack_cdt_result_with_map(Polys &&polys, tf::index_map_buffer<int> &&im)
    -> cdt_result_with_map_t<Real> {
  auto n_faces = static_cast<int>(polys.faces().size());
  auto n_points = static_cast<int>(polys.points().size());
  return {
      wasm_ndarray<int>::from_buffer(
          std::move(polys.faces_buffer().data_buffer()), {n_faces, 3}),
      wasm_ndarray<Real>::from_buffer(
          std::move(polys.points_buffer().data_buffer()), {n_points, 2}),
      wasm_index_map::from_index_map_buffer(std::move(im)),
  };
}

// ============================================================================
// No edges — convex-hull Delaunay only.
// ============================================================================

template <typename Real>
auto sync_make_cdt(wasm_ndarray<Real> &points) -> cdt_result_t<Real> {
  auto pts = tf::make_points<2>(points.make_range());
  return pack_cdt_result<Real>(tf::make_cdt(pts));
}

template <typename Real>
auto async_make_cdt(wasm_ndarray<Real> &points) -> promise_t {
  return promise([a = points]() -> cdt_result_t<Real> {
    return sync_make_cdt<Real>(const_cast<wasm_ndarray<Real> &>(a));
  });
}

template <typename Real>
auto sync_make_cdt_with_maps(wasm_ndarray<Real> &points)
    -> cdt_result_with_map_t<Real> {
  auto pts = tf::make_points<2>(points.make_range());
  auto [polys, im] = tf::make_cdt(pts, tf::return_index_map);
  return pack_cdt_result_with_map<Real>(std::move(polys), std::move(im));
}

template <typename Real>
auto async_make_cdt_with_maps(wasm_ndarray<Real> &points) -> promise_t {
  return promise([a = points]() -> cdt_result_with_map_t<Real> {
    return sync_make_cdt_with_maps<Real>(const_cast<wasm_ndarray<Real> &>(a));
  });
}

// ============================================================================
// With constraint edges, no mask (every edge is a region boundary).
// ============================================================================

template <typename Real>
auto sync_make_cdt_edges(wasm_ndarray<Real> &points, wasm_ndarray<int> &edges,
                         bool split_constraints) -> cdt_result_t<Real> {
  auto pts = tf::make_points<2>(points.make_range());
  auto eds = tf::make_edges(tf::make_blocked_range<2>(edges.make_range()));
  return pack_cdt_result<Real>(tf::make_cdt(pts, eds, split_constraints));
}

template <typename Real>
auto async_make_cdt_edges(wasm_ndarray<Real> &points, wasm_ndarray<int> &edges,
                          bool split_constraints) -> promise_t {
  return promise([a = points, b = edges,
                  split_constraints]() -> cdt_result_t<Real> {
    return sync_make_cdt_edges<Real>(const_cast<wasm_ndarray<Real> &>(a),
                                     const_cast<wasm_ndarray<int> &>(b),
                                     split_constraints);
  });
}

template <typename Real>
auto sync_make_cdt_edges_with_maps(wasm_ndarray<Real> &points,
                                   wasm_ndarray<int> &edges,
                                   bool split_constraints)
    -> cdt_result_with_map_t<Real> {
  auto pts = tf::make_points<2>(points.make_range());
  auto eds = tf::make_edges(tf::make_blocked_range<2>(edges.make_range()));
  auto [polys, im] =
      tf::make_cdt(pts, eds, tf::return_index_map, split_constraints);
  return pack_cdt_result_with_map<Real>(std::move(polys), std::move(im));
}

template <typename Real>
auto async_make_cdt_edges_with_maps(wasm_ndarray<Real> &points,
                                    wasm_ndarray<int> &edges,
                                    bool split_constraints) -> promise_t {
  return promise([a = points, b = edges,
                  split_constraints]() -> cdt_result_with_map_t<Real> {
    return sync_make_cdt_edges_with_maps<Real>(
        const_cast<wasm_ndarray<Real> &>(a), const_cast<wasm_ndarray<int> &>(b),
        split_constraints);
  });
}

// ============================================================================
// With constraint edges + per-edge boundary mask (int8: 0 = non-boundary,
// non-zero = boundary). Non-boundary constrained edges are preserved in
// the triangulation but do not flip region parity.
// ============================================================================

template <typename Real>
auto sync_make_cdt_edges_masked(wasm_ndarray<Real> &points,
                                wasm_ndarray<int> &edges,
                                wasm_ndarray<std::int8_t> &edge_mask,
                                bool split_constraints) -> cdt_result_t<Real> {
  auto pts = tf::make_points<2>(points.make_range());
  auto eds = tf::make_edges(tf::make_blocked_range<2>(edges.make_range()));
  auto mask = edge_mask.make_range();
  return pack_cdt_result<Real>(tf::make_cdt(pts, eds, mask, split_constraints));
}

template <typename Real>
auto async_make_cdt_edges_masked(wasm_ndarray<Real> &points,
                                 wasm_ndarray<int> &edges,
                                 wasm_ndarray<std::int8_t> &edge_mask,
                                 bool split_constraints) -> promise_t {
  return promise([a = points, b = edges, c = edge_mask,
                  split_constraints]() -> cdt_result_t<Real> {
    return sync_make_cdt_edges_masked<Real>(
        const_cast<wasm_ndarray<Real> &>(a), const_cast<wasm_ndarray<int> &>(b),
        const_cast<wasm_ndarray<std::int8_t> &>(c), split_constraints);
  });
}

template <typename Real>
auto sync_make_cdt_edges_masked_with_maps(wasm_ndarray<Real> &points,
                                          wasm_ndarray<int> &edges,
                                          wasm_ndarray<std::int8_t> &edge_mask,
                                          bool split_constraints)
    -> cdt_result_with_map_t<Real> {
  auto pts = tf::make_points<2>(points.make_range());
  auto eds = tf::make_edges(tf::make_blocked_range<2>(edges.make_range()));
  auto mask = edge_mask.make_range();
  auto [polys, im] =
      tf::make_cdt(pts, eds, mask, tf::return_index_map, split_constraints);
  return pack_cdt_result_with_map<Real>(std::move(polys), std::move(im));
}

template <typename Real>
auto async_make_cdt_edges_masked_with_maps(wasm_ndarray<Real> &points,
                                           wasm_ndarray<int> &edges,
                                           wasm_ndarray<std::int8_t> &edge_mask,
                                           bool split_constraints)
    -> promise_t {
  return promise([a = points, b = edges, c = edge_mask,
                  split_constraints]() -> cdt_result_with_map_t<Real> {
    return sync_make_cdt_edges_masked_with_maps<Real>(
        const_cast<wasm_ndarray<Real> &>(a), const_cast<wasm_ndarray<int> &>(b),
        const_cast<wasm_ndarray<std::int8_t> &>(c), split_constraints);
  });
}

} // namespace ts
} // namespace tf
