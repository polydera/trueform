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

#include "trueform/topology/make_cdt.hpp"
#include "trueform/ts/core/promise.hpp"
#include "trueform/ts/core/wasm_index_map.hpp"
#include "trueform/ts/core/wasm_ndarray.hpp"
#include "trueform/ts/topology/cdt_result.hpp"
#include <cstdint>
#include <emscripten/bind.h>

namespace {

using namespace tf::ts;

// ============================================================================
// Helpers — package the make_cdt return into wasm-side result types.
// ============================================================================

template <typename Polys>
auto pack_cdt_result(Polys &&polys) -> cdt_result {
  auto n_faces = static_cast<int>(polys.faces().size());
  auto n_points = static_cast<int>(polys.points().size());
  return {
      wasm_ndarray<int>::from_buffer(
          std::move(polys.faces_buffer().data_buffer()), {n_faces, 3}),
      wasm_ndarray<float>::from_buffer(
          std::move(polys.points_buffer().data_buffer()), {n_points, 2}),
  };
}

template <typename Polys>
auto pack_cdt_result_with_map(Polys &&polys, tf::index_map_buffer<int> &&im)
    -> cdt_result_with_map {
  auto n_faces = static_cast<int>(polys.faces().size());
  auto n_points = static_cast<int>(polys.points().size());
  return {
      wasm_ndarray<int>::from_buffer(
          std::move(polys.faces_buffer().data_buffer()), {n_faces, 3}),
      wasm_ndarray<float>::from_buffer(
          std::move(polys.points_buffer().data_buffer()), {n_points, 2}),
      wasm_index_map::from_index_map_buffer(std::move(im)),
  };
}

// ============================================================================
// No edges — convex-hull Delaunay only.
// ============================================================================

auto sync_make_cdt(wasm_ndarray<float> &points) -> cdt_result {
  auto pts = tf::make_points<2>(points.make_range());
  return pack_cdt_result(tf::make_cdt(pts));
}

auto async_make_cdt(wasm_ndarray<float> &points) -> promise_t {
  return promise([a = points]() -> cdt_result {
    return sync_make_cdt(const_cast<wasm_ndarray<float> &>(a));
  });
}

auto sync_make_cdt_with_maps(wasm_ndarray<float> &points)
    -> cdt_result_with_map {
  auto pts = tf::make_points<2>(points.make_range());
  auto [polys, im] = tf::make_cdt(pts, tf::return_index_map);
  return pack_cdt_result_with_map(std::move(polys), std::move(im));
}

auto async_make_cdt_with_maps(wasm_ndarray<float> &points) -> promise_t {
  return promise([a = points]() -> cdt_result_with_map {
    return sync_make_cdt_with_maps(const_cast<wasm_ndarray<float> &>(a));
  });
}

// ============================================================================
// With constraint edges, no mask (every edge is a region boundary).
// ============================================================================

auto sync_make_cdt_edges(wasm_ndarray<float> &points,
                         wasm_ndarray<int> &edges) -> cdt_result {
  auto pts = tf::make_points<2>(points.make_range());
  auto eds = tf::make_edges(tf::make_blocked_range<2>(edges.make_range()));
  return pack_cdt_result(tf::make_cdt(pts, eds));
}

auto async_make_cdt_edges(wasm_ndarray<float> &points,
                          wasm_ndarray<int> &edges) -> promise_t {
  return promise([a = points, b = edges]() -> cdt_result {
    return sync_make_cdt_edges(const_cast<wasm_ndarray<float> &>(a),
                               const_cast<wasm_ndarray<int> &>(b));
  });
}

auto sync_make_cdt_edges_with_maps(wasm_ndarray<float> &points,
                                   wasm_ndarray<int> &edges)
    -> cdt_result_with_map {
  auto pts = tf::make_points<2>(points.make_range());
  auto eds = tf::make_edges(tf::make_blocked_range<2>(edges.make_range()));
  auto [polys, im] = tf::make_cdt(pts, eds, tf::return_index_map);
  return pack_cdt_result_with_map(std::move(polys), std::move(im));
}

auto async_make_cdt_edges_with_maps(wasm_ndarray<float> &points,
                                    wasm_ndarray<int> &edges) -> promise_t {
  return promise([a = points, b = edges]() -> cdt_result_with_map {
    return sync_make_cdt_edges_with_maps(
        const_cast<wasm_ndarray<float> &>(a),
        const_cast<wasm_ndarray<int> &>(b));
  });
}

// ============================================================================
// With constraint edges + per-edge boundary mask (int8: 0 = non-boundary,
// non-zero = boundary). Non-boundary constrained edges are preserved in
// the triangulation but do not flip region parity.
// ============================================================================

auto sync_make_cdt_edges_masked(wasm_ndarray<float> &points,
                                wasm_ndarray<int> &edges,
                                wasm_ndarray<std::int8_t> &edge_mask)
    -> cdt_result {
  auto pts = tf::make_points<2>(points.make_range());
  auto eds = tf::make_edges(tf::make_blocked_range<2>(edges.make_range()));
  auto mask = edge_mask.make_range();
  return pack_cdt_result(tf::make_cdt(pts, eds, mask));
}

auto async_make_cdt_edges_masked(wasm_ndarray<float> &points,
                                 wasm_ndarray<int> &edges,
                                 wasm_ndarray<std::int8_t> &edge_mask)
    -> promise_t {
  return promise([a = points, b = edges, c = edge_mask]() -> cdt_result {
    return sync_make_cdt_edges_masked(
        const_cast<wasm_ndarray<float> &>(a),
        const_cast<wasm_ndarray<int> &>(b),
        const_cast<wasm_ndarray<std::int8_t> &>(c));
  });
}

auto sync_make_cdt_edges_masked_with_maps(wasm_ndarray<float> &points,
                                          wasm_ndarray<int> &edges,
                                          wasm_ndarray<std::int8_t> &edge_mask)
    -> cdt_result_with_map {
  auto pts = tf::make_points<2>(points.make_range());
  auto eds = tf::make_edges(tf::make_blocked_range<2>(edges.make_range()));
  auto mask = edge_mask.make_range();
  auto [polys, im] = tf::make_cdt(pts, eds, mask, tf::return_index_map);
  return pack_cdt_result_with_map(std::move(polys), std::move(im));
}

auto async_make_cdt_edges_masked_with_maps(wasm_ndarray<float> &points,
                                           wasm_ndarray<int> &edges,
                                           wasm_ndarray<std::int8_t> &edge_mask)
    -> promise_t {
  return promise(
      [a = points, b = edges, c = edge_mask]() -> cdt_result_with_map {
        return sync_make_cdt_edges_masked_with_maps(
            const_cast<wasm_ndarray<float> &>(a),
            const_cast<wasm_ndarray<int> &>(b),
            const_cast<wasm_ndarray<std::int8_t> &>(c));
      });
}

} // namespace

EMSCRIPTEN_BINDINGS(trueform_cdt) {
  // Result types
  emscripten::value_object<tf::ts::cdt_result>("CdtResult")
      .field("faces", &tf::ts::cdt_result::faces)
      .field("points", &tf::ts::cdt_result::points);

  emscripten::value_object<tf::ts::cdt_result_with_map>("CdtResultWithMap")
      .field("faces", &tf::ts::cdt_result_with_map::faces)
      .field("points", &tf::ts::cdt_result_with_map::points)
      .field("indexMap", &tf::ts::cdt_result_with_map::index_map);

  // Sync
  emscripten::function("make_cdt", &sync_make_cdt);
  emscripten::function("make_cdt_with_maps", &sync_make_cdt_with_maps);
  emscripten::function("make_cdt_edges", &sync_make_cdt_edges);
  emscripten::function("make_cdt_edges_with_maps",
                       &sync_make_cdt_edges_with_maps);
  emscripten::function("make_cdt_edges_masked", &sync_make_cdt_edges_masked);
  emscripten::function("make_cdt_edges_masked_with_maps",
                       &sync_make_cdt_edges_masked_with_maps);

  // Async
  emscripten::function("dispatch_make_cdt", &async_make_cdt);
  emscripten::function("dispatch_make_cdt_with_maps",
                       &async_make_cdt_with_maps);
  emscripten::function("dispatch_make_cdt_edges", &async_make_cdt_edges);
  emscripten::function("dispatch_make_cdt_edges_with_maps",
                       &async_make_cdt_edges_with_maps);
  emscripten::function("dispatch_make_cdt_edges_masked",
                       &async_make_cdt_edges_masked);
  emscripten::function("dispatch_make_cdt_edges_masked_with_maps",
                       &async_make_cdt_edges_masked_with_maps);
}
