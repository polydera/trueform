/*
 * Copyright (c) 2026 XLAB
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
#include "../../core/unsafe.hpp"
#include "../../topology/half_edge.hpp"
#include "../../topology/half_edge_handle.hpp"
#include "../../topology/half_edges.hpp"
#include <cstddef>

namespace tf::fill {

/// Split one interior edge of a patch at a vertex standing on it.
///
/// The edge's two triangles `(u, v, w)` and `(v, u, x)` become the four the
/// insertion makes — `(u, m, w)`, `(m, v, w)`, `(v, m, x)` and `(m, u, x)` —
/// so the split edge keeps its own slot as `u m`, three edges and two faces
/// are appended, and nothing is removed. The caller has already named `m`:
/// it is the next vertex of the patch, and its position lane is the caller's.
///
/// The edge must be interior and manifold, which every chord of a patch is.
/// The structure's cached face and vertex counts are its build's; the
/// appended buffers state the current ones until `rebuild_handles` restates
/// them.
template <typename Index>
auto split_hole_patch_edge(tf::half_edges<Index> &he,
                           const tf::edge_handle<Index> &eh, Index m) -> void {
  const auto a0 = he.half_edge_handle(tf::unsafe, eh, false);
  const auto a1 = he.next(tf::unsafe, a0);
  const auto a2 = he.next(tf::unsafe, a1);
  const auto b0 = he.half_edge_handle(tf::unsafe, eh, true);
  const auto b1 = he.next(tf::unsafe, b0);
  const auto b2 = he.next(tf::unsafe, b1);

  const Index u = he.half_edge(a0).vertex;
  const Index v = he.half_edge(b0).vertex;
  const Index w = he.half_edge(a2).vertex;
  const Index x = he.half_edge(b2).vertex;
  const Index f0 = he.half_edge(a0).face;
  const Index f3 = he.half_edge(b0).face;

  auto &edges = he.half_edges_buffer();
  const Index base = Index(edges.size());
  const Index m_v = base, v_m = Index(base + 1);
  const Index m_w = Index(base + 2), w_m = Index(base + 3);
  const Index m_x = Index(base + 4), x_m = Index(base + 5);
  edges.reallocate(std::size_t(base) + 6);

  auto &faces = he.face_half_edge_handles_buffer();
  const Index f1 = Index(faces.size());
  const Index f2 = Index(f1 + 1);

  edges[std::size_t(a0.id())] = {f0, u, m_w, a2.id()};
  edges[std::size_t(m_w)] = {f0, m, a2.id(), a0.id()};
  edges[std::size_t(a2.id())] = {f0, w, a0.id(), m_w};

  edges[std::size_t(m_v)] = {f1, m, a1.id(), w_m};
  edges[std::size_t(a1.id())] = {f1, v, w_m, m_v};
  edges[std::size_t(w_m)] = {f1, w, m_v, a1.id()};

  edges[std::size_t(v_m)] = {f2, v, m_x, b2.id()};
  edges[std::size_t(m_x)] = {f2, m, b2.id(), v_m};
  edges[std::size_t(b2.id())] = {f2, x, v_m, m_x};

  edges[std::size_t(b0.id())] = {f3, m, b1.id(), x_m};
  edges[std::size_t(b1.id())] = {f3, u, x_m, b0.id()};
  edges[std::size_t(x_m)] = {f3, x, b0.id(), b1.id()};

  faces[std::size_t(f0)] = a0;
  faces[std::size_t(f3)] = b0;
  faces.push_back(tf::half_edge_handle<Index>{m_v});
  faces.push_back(tf::half_edge_handle<Index>{v_m});

  auto &vertices = he.vertex_half_edge_handles_buffer();
  // b0 left v for m, so a vertex whose representative it was needs another.
  if (vertices[std::size_t(v)] == b0)
    vertices[std::size_t(v)] = a1;
  vertices.push_back(tf::half_edge_handle<Index>{m_v});
  he.boundary_vertex_data_buffer().push_back(char(0));
  he.non_manifold_vertex_data_buffer().push_back(char(0));
}

} // namespace tf::fill
