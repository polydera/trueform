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
#include "../../core/algorithm/circular_increment.hpp"
#include "../../core/algorithm/generic_generate.hpp"
#include "../../core/array_hash.hpp"
#include "../../core/buffer.hpp"
#include "../../core/hash_set.hpp"
#include "../arrangement_graph.hpp"
#include "../face_cuts.hpp"
#include "./make_non_manifold_edge_fans.hpp"
#include "tbb/parallel_sort.h"
#include <algorithm>
#include <array>

namespace tf::cut {

/// @ingroup cut
/// @brief For each representative NM edge, walk its incident loop fan
///        cyclically and emit:
///   1. one component-side merge pair per cyclic-adjacent loop pair to
///      `merges` (drives the domain partition);
///   2. one component-only merge pair per cyclic-adjacent loop pair to
///      `bundle_merges` (drives the 3D-connected-component partition).
///
/// Graph-native counterpart to
/// @ref tf::topology::domains::emit_domain_merges. Sides come from the
/// fan's per-occurrence direction bits; component labels come from
/// @ref tf::arrangement_graph.
template <typename Index, typename Index1>
void emit_domain_merges(
    const tf::arrangement_graph<Index> &ag,
    const tf::face_cuts<Index, Index1> &fc,
    const tf::cut::non_manifold_edge_fans<Index> &fans,
    const tf::buffer<Index> &reps,
    tf::buffer<std::array<Index, 2>> &merges,
    tf::buffer<std::array<Index, 2>> &bundle_merges) {
  using vertex_t = typename tf::cut::non_manifold_edge_fans<Index>::vertex_t;

  auto loops = fc.loops();
  auto loop_labels = ag.loop_labels();

  struct local_state_t {
    tf::hash_set<std::array<Index, 2>, tf::array_hash<Index, 2>> seen;
    tf::hash_set<std::array<Index, 2>, tf::array_hash<Index, 2>> bundle_seen;
  };

  auto dirs_view = tf::make_offset_block_range(fans.faces.offsets_buffer(),
                                               tf::make_range(fans.dirs));

  tf::generic_generate(
      reps, std::tie(merges, bundle_merges), local_state_t{},
      [&](Index rep, auto &out_buffers, local_state_t &state) {
        auto face_block = fans.faces[rep];
        auto dir_block = dirs_view[rep];
        Index K = Index(face_block.size());

        auto &out_merges = std::get<0>(out_buffers);
        auto &out_bundles = std::get<1>(out_buffers);

        for (Index r = 0; r < K; ++r) {
          Index rn = tf::circular_increment(r, K);
          Index Fa = face_block[r];
          Index Fb = face_block[rn];

          // Sides come from the occurrence's own traversal direction —
          // a slit loop's two fan entries take opposite sides, which a
          // per-loop containment test cannot express.
          Index sa = dir_block[r] ? Index(0) : Index(1);
          Index sb = (dir_block[rn] ? Index(0) : Index(1)) ^ Index(1);

          Index frag_a = loop_labels[Fa];
          Index frag_b = loop_labels[Fb];
          Index node_a = 2 * frag_a + sa;
          Index node_b = 2 * frag_b + sb;
          std::array<Index, 2> p = {std::min(node_a, node_b),
                                    std::max(node_a, node_b)};
          if (state.seen.insert(p).second)
            out_merges.push_back(p);

          if (frag_a != frag_b) {
            std::array<Index, 2> bp = {std::min(frag_a, frag_b),
                                       std::max(frag_a, frag_b)};
            if (state.bundle_seen.insert(bp).second)
              out_bundles.push_back(bp);
          }
        }
      });

  tbb::parallel_sort(merges.begin(), merges.end());
  merges.erase_till_end(std::unique(merges.begin(), merges.end()));
  tbb::parallel_sort(bundle_merges.begin(), bundle_merges.end());
  bundle_merges.erase_till_end(
      std::unique(bundle_merges.begin(), bundle_merges.end()));
}

} // namespace tf::cut
