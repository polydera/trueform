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

#include <cmath>
#include <cstdint>

#include "../core/buffer.hpp"
#include "../topology/half_edges.hpp"
#include "./valence_deviation.hpp"

namespace tf::remesh {

/// @ingroup remesh
/// @brief Optimize vertex valence by flipping edges.
///
/// Flips edges to bring vertex valences closer to the ideal (6 for
/// interior vertices, 4 for boundary vertices). Each flip is performed
/// only when it strictly reduces the total valence deviation of the
/// four involved vertices. The deviation buffer is modified in place
/// and remains valid for subsequent calls.
///
/// @param he The half-edge structure (modified in place).
/// @param deviation Per-vertex valence deviation (modified in place).
/// @param iterations The number of passes over all edges.
/// @return The number of edges flipped.
template <typename Index>
auto optimize_valence(tf::half_edges<Index> &he,
                      tf::buffer<std::int8_t> &deviation,
                      int iterations = 1) -> Index {
  Index n_flipped = 0;
  for (int iter = 0; iter < iterations; ++iter) {
    Index flipped_this_iter = 0;
    for (auto eh : he.edge_handles()) {
      if (!he.is_flip_ok(eh))
        continue;
      auto a0 = he.half_edge_handle(tf::unsafe, eh, false);
      auto a1 = he.next(tf::unsafe, a0);
      auto a2 = he.next(tf::unsafe, a1);
      auto b0 = he.half_edge_handle(tf::unsafe, eh, true);
      auto b1 = he.next(tf::unsafe, b0);
      auto b2 = he.next(tf::unsafe, b1);

      auto va0 = he.start_vertex_handle(tf::unsafe, a0).id();
      auto va1 = he.start_vertex_handle(tf::unsafe, a1).id();
      auto va2 = he.start_vertex_handle(tf::unsafe, a2).id();
      auto vb2 = he.start_vertex_handle(tf::unsafe, b2).id();

      auto before = std::abs(deviation[va0]) + std::abs(deviation[va1]) +
                    std::abs(deviation[va2]) + std::abs(deviation[vb2]);
      auto after =
          std::abs(deviation[va0] - 1) + std::abs(deviation[va1] - 1) +
          std::abs(deviation[va2] + 1) + std::abs(deviation[vb2] + 1);

      if (after >= before)
        continue;

      deviation[va0]--;
      deviation[va1]--;
      deviation[va2]++;
      deviation[vb2]++;
      he.flip(eh);
      ++flipped_this_iter;
    }
    n_flipped += flipped_this_iter;
    if (flipped_this_iter == 0)
      break;
  }
  return n_flipped;
}

} // namespace tf::remesh

namespace tf {

/// @ingroup remesh
/// @brief Optimize vertex valence by flipping edges.
///
/// Convenience overload that computes valence deviations internally.
///
/// @param he The half-edge structure (modified in place).
/// @param iterations The number of passes over all edges.
/// @return The number of edges flipped.
template <typename Index>
auto optimize_valence(tf::half_edges<Index> &he, int iterations = 1)
    -> Index {
  auto deviation = tf::remesh::compute_valence_deviations(he);
  return tf::remesh::optimize_valence(he, deviation, iterations);
}

} // namespace tf
