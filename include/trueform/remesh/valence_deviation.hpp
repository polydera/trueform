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

#include <cstdint>

#include "../core/algorithm/parallel_for_each.hpp"
#include "../core/buffer.hpp"
#include "../core/views/zip.hpp"
#include "../topology/half_edges.hpp"

namespace tf::remesh {

/// @ingroup remesh
/// @brief Compute per-vertex valence deviation from ideal.
///
/// Returns a buffer where each entry is `actual_valence - ideal_valence`.
/// The ideal valence is 6 for interior vertices and 4 for boundary vertices.
///
/// @param he The half-edge structure.
/// @return Per-vertex valence deviation buffer.
template <typename Index>
auto compute_valence_deviations(const tf::half_edges<Index> &he)
    -> tf::buffer<std::int8_t> {
  Index n_verts = Index(he.vertex_half_edge_handles().size());

  tf::buffer<std::int8_t> deviation;
  deviation.allocate(n_verts);

  tf::parallel_for_each(tf::zip(deviation, he.boundary_vertices()),
                        [](auto pair) {
                          auto &[dev, bnd] = pair;
                          dev = bnd ? std::int8_t(-4) : std::int8_t(-6);
                        });

  for (auto fhe : he.face_half_edge_handles()) {
    if (!fhe.is_valid())
      continue;
    auto h0 = fhe;
    auto h1 = he.next(tf::unsafe, h0);
    auto h2 = he.next(tf::unsafe, h1);
    deviation[he.start_vertex_handle(tf::unsafe, h0).id()]++;
    deviation[he.start_vertex_handle(tf::unsafe, h1).id()]++;
    deviation[he.start_vertex_handle(tf::unsafe, h2).id()]++;
  }

  return deviation;
}

} // namespace tf::remesh
