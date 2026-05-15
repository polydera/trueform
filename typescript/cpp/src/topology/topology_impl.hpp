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

#include "trueform/topology.hpp"
#include "trueform/core/algorithm/parallel_copy.hpp"
#include "trueform/ts/core/promise.hpp"
#include "trueform/ts/core/wasm_mesh.hpp"
#include "trueform/ts/core/wasm_ndarray.hpp"
#include "trueform/ts/core/wasm_offset_blocked_buffer.hpp"
#include "trueform/ts/topology/result_types.hpp"

namespace tf {
namespace ts {

// ============================================================================
// Boolean queries
// ============================================================================

template <typename Real>
auto sync_is_closed(wasm_mesh<Real> &m) -> bool {
  auto poly = m.polygons_range() | tf::tag(m.face_membership_range());
  return tf::is_closed(poly);
}

template <typename Real>
auto sync_is_open(wasm_mesh<Real> &m) -> bool {
  auto poly = m.polygons_range() | tf::tag(m.face_membership_range());
  return tf::is_open(poly);
}

template <typename Real>
auto sync_is_manifold(wasm_mesh<Real> &m) -> bool {
  auto poly = m.polygons_range() | tf::tag(m.face_membership_range());
  return tf::is_manifold(poly);
}

template <typename Real>
auto sync_is_non_manifold(wasm_mesh<Real> &m) -> bool {
  auto poly = m.polygons_range() | tf::tag(m.face_membership_range());
  return tf::is_non_manifold(poly);
}

// ============================================================================
// Scalar queries
// ============================================================================

template <typename Real>
auto sync_euler_characteristic(wasm_mesh<Real> &m) -> int {
  return tf::euler_characteristic(m.polygons_range());
}

// ============================================================================
// Edge results [N, 2]
// ============================================================================

template <typename Real>
auto sync_boundary_edges(wasm_mesh<Real> &m) -> wasm_ndarray<int> {
  auto poly = m.polygons_range() | tf::tag(m.face_membership_range());
  auto edges = tf::make_boundary_edges(poly);
  auto n = static_cast<int>(edges.size());
  return wasm_ndarray<int>::from_buffer(std::move(edges.data_buffer()), {n, 2});
}

template <typename Real>
auto sync_non_manifold_edges(wasm_mesh<Real> &m) -> wasm_ndarray<int> {
  auto poly = m.polygons_range() | tf::tag(m.face_membership_range());
  auto edges = tf::make_non_manifold_edges(poly);
  auto n = static_cast<int>(edges.size());
  return wasm_ndarray<int>::from_buffer(std::move(edges.data_buffer()), {n, 2});
}

// ============================================================================
// OffsetBlockedBuffer results
// ============================================================================

template <typename Real>
auto sync_boundary_paths(wasm_mesh<Real> &m)
    -> wasm_offset_blocked_buffer<int, int> {
  auto poly = m.polygons_range() | tf::tag(m.face_membership_range());
  auto paths = tf::make_boundary_paths(poly);
  return wasm_offset_blocked_buffer<int, int>::from_buffer(std::move(paths));
}

template <typename Real>
auto sync_k_rings(wasm_mesh<Real> &m, int k, bool inclusive)
    -> wasm_offset_blocked_buffer<int, int> {
  auto result = tf::make_k_rings(m.vertex_link_range(),
                                 static_cast<std::size_t>(k), inclusive);
  return wasm_offset_blocked_buffer<int, int>::from_buffer(std::move(result));
}

template <typename Real>
auto sync_neighborhoods(wasm_mesh<Real> &m, Real radius, bool inclusive)
    -> wasm_offset_blocked_buffer<int, int> {
  auto pts = m.points_range() | tf::tag(m.vertex_link_range());
  auto result = tf::make_neighborhoods(pts, radius, inclusive);
  return wasm_offset_blocked_buffer<int, int>::from_buffer(std::move(result));
}

// Non-templated: only operates on int32 ndarray of edges.
inline auto sync_connect_edges_to_paths(wasm_ndarray<int> &edges_arr)
    -> wasm_offset_blocked_buffer<int, int> {
  auto edges = tf::make_edges(edges_arr.make_range());
  auto result = tf::connect_edges_to_paths(edges);
  return wasm_offset_blocked_buffer<int, int>::from_buffer(std::move(result));
}

// ============================================================================
// Connected components — Layer 1: generic on OffsetBlockedBuffer
// ============================================================================

inline auto sync_label_connected_components_obb(
    wasm_offset_blocked_buffer<int, int> &conn) -> connected_components_result {
  auto range = conn.make_range();
  auto n = conn.size();
  tf::connected_component_labels<int> cl;
  cl.labels.allocate(n);
  auto applier = [&range](auto id, const auto &f) {
    for (auto n_id : range[id])
      f(n_id);
  };
  cl.n_components = tf::label_connected_components(cl.labels, applier);
  return {wasm_ndarray<int>::from_buffer(std::move(cl.labels), {n}),
          cl.n_components};
}

// ============================================================================
// Connected components — Layer 2: generic on NDArrayInt32 [N, K], -1 = none
// ============================================================================

inline auto sync_label_connected_components_ndarray(wasm_ndarray<int> &conn)
    -> connected_components_result {
  auto flat = conn.make_range();
  auto n = conn.shape_at(0);
  auto k = static_cast<std::size_t>(conn.shape_at(1));
  auto range = tf::make_blocked_range(flat, k);
  tf::connected_component_labels<int> cl;
  cl.labels.allocate(n);
  auto applier = [&range](auto id, const auto &f) {
    for (auto peer : range[id]) {
      if (peer >= 0)
        f(peer);
    }
  };
  cl.n_components = tf::label_connected_components(cl.labels, applier);
  return {wasm_ndarray<int>::from_buffer(std::move(cl.labels), {n}),
          cl.n_components};
}

// ============================================================================
// Mesh mutation: consistently oriented faces
// ============================================================================

template <typename Real>
auto sync_consistently_oriented(wasm_mesh<Real> &m) -> wasm_mesh<Real> {
  const auto num_faces = m.number_of_faces();
  const auto num_ints = static_cast<std::size_t>(num_faces) * 3;

  tf::buffer<int> face_buf;
  face_buf.allocate(num_ints);
  tf::parallel_copy(m.faces().make_range(),
                    tf::make_range(face_buf.data(), num_ints));

  auto faces = tf::make_faces<3>(tf::make_range(face_buf.data(), num_ints));
  auto polys = tf::make_polygons(faces, m.points_range());
  auto tagged = polys | tf::tag(m.manifold_edge_link_range());
  tf::orient_faces_consistently(tagged);

  auto result_faces =
      wasm_ndarray<int>::from_buffer(std::move(face_buf), {num_faces, 3});
  return wasm_mesh<Real>::create(std::move(result_faces), m.points());
}

// ============================================================================
// Async wrappers
// ============================================================================

template <typename Real>
auto async_is_closed(wasm_mesh<Real> &m) -> promise_t {
  return promise([a = m]() -> bool {
    return sync_is_closed<Real>(const_cast<wasm_mesh<Real> &>(a));
  });
}

template <typename Real>
auto async_is_open(wasm_mesh<Real> &m) -> promise_t {
  return promise([a = m]() -> bool {
    return sync_is_open<Real>(const_cast<wasm_mesh<Real> &>(a));
  });
}

template <typename Real>
auto async_is_manifold(wasm_mesh<Real> &m) -> promise_t {
  return promise([a = m]() -> bool {
    return sync_is_manifold<Real>(const_cast<wasm_mesh<Real> &>(a));
  });
}

template <typename Real>
auto async_is_non_manifold(wasm_mesh<Real> &m) -> promise_t {
  return promise([a = m]() -> bool {
    return sync_is_non_manifold<Real>(const_cast<wasm_mesh<Real> &>(a));
  });
}

template <typename Real>
auto async_euler_characteristic(wasm_mesh<Real> &m) -> promise_t {
  return promise([a = m]() -> int {
    return sync_euler_characteristic<Real>(const_cast<wasm_mesh<Real> &>(a));
  });
}

template <typename Real>
auto async_boundary_edges(wasm_mesh<Real> &m) -> promise_t {
  return promise([a = m]() -> wasm_ndarray<int> {
    return sync_boundary_edges<Real>(const_cast<wasm_mesh<Real> &>(a));
  });
}

template <typename Real>
auto async_non_manifold_edges(wasm_mesh<Real> &m) -> promise_t {
  return promise([a = m]() -> wasm_ndarray<int> {
    return sync_non_manifold_edges<Real>(const_cast<wasm_mesh<Real> &>(a));
  });
}

template <typename Real>
auto async_boundary_paths(wasm_mesh<Real> &m) -> promise_t {
  return promise([a = m]() -> wasm_offset_blocked_buffer<int, int> {
    return sync_boundary_paths<Real>(const_cast<wasm_mesh<Real> &>(a));
  });
}

template <typename Real>
auto async_k_rings(wasm_mesh<Real> &m, int k, bool inclusive) -> promise_t {
  return promise(
      [a = m, k, inclusive]() -> wasm_offset_blocked_buffer<int, int> {
        return sync_k_rings<Real>(const_cast<wasm_mesh<Real> &>(a), k,
                                  inclusive);
      });
}

template <typename Real>
auto async_neighborhoods(wasm_mesh<Real> &m, Real radius, bool inclusive)
    -> promise_t {
  return promise([a = m, radius,
                  inclusive]() -> wasm_offset_blocked_buffer<int, int> {
    return sync_neighborhoods<Real>(const_cast<wasm_mesh<Real> &>(a), radius,
                                    inclusive);
  });
}

inline auto async_connect_edges_to_paths(wasm_ndarray<int> &edges_arr)
    -> promise_t {
  return promise([a = edges_arr]() -> wasm_offset_blocked_buffer<int, int> {
    return sync_connect_edges_to_paths(const_cast<wasm_ndarray<int> &>(a));
  });
}

inline auto async_label_connected_components_obb(
    wasm_offset_blocked_buffer<int, int> &conn) -> promise_t {
  return promise([a = conn]() -> connected_components_result {
    return sync_label_connected_components_obb(
        const_cast<wasm_offset_blocked_buffer<int, int> &>(a));
  });
}

inline auto async_label_connected_components_ndarray(wasm_ndarray<int> &conn)
    -> promise_t {
  return promise([a = conn]() -> connected_components_result {
    return sync_label_connected_components_ndarray(
        const_cast<wasm_ndarray<int> &>(a));
  });
}

template <typename Real>
auto async_consistently_oriented(wasm_mesh<Real> &m) -> promise_t {
  return promise([a = m]() -> wasm_mesh<Real> {
    return sync_consistently_oriented<Real>(const_cast<wasm_mesh<Real> &>(a));
  });
}

} // namespace ts
} // namespace tf
