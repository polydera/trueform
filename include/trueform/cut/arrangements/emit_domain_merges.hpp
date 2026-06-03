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
/// @ref tf::topology::domains::emit_domain_merges. Side detection uses
/// the directed-edge search over `vertex_t` (the loop's native vertex
/// type); component labels come from @ref tf::arrangement_graph.
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

  // Local variant of tf::directed_edge_id_in_face for vertex_t faces.
  // Returns the slot index (Index) of the directed edge (v0 -> v1) in
  // the face, or face.size() if absent.
  auto directed_edge_id = [](const vertex_t &v0, const vertex_t &v1,
                              const auto &face) -> Index {
    Index size = Index(face.size());
    Index prev = size - 1;
    for (Index i = 0; i < size; prev = i++) {
      if (char(face[prev] == v0) & char(face[i] == v1))
        return prev;
    }
    return size;
  };

  struct local_state_t {
    tf::hash_set<std::array<Index, 2>, tf::array_hash<Index, 2>> seen;
    tf::hash_set<std::array<Index, 2>, tf::array_hash<Index, 2>> bundle_seen;
  };

  tf::generic_generate(
      reps, std::tie(merges, bundle_merges), local_state_t{},
      [&](Index rep, auto &out_buffers, local_state_t &state) {
        auto edge = fans.edges[rep];
        const vertex_t vi = edge[0];
        const vertex_t vj = edge[1];
        auto face_block = fans.faces[rep];
        Index K = Index(face_block.size());

        auto &out_merges = std::get<0>(out_buffers);
        auto &out_bundles = std::get<1>(out_buffers);

        for (Index r = 0; r < K; ++r) {
          Index Fa = face_block[r];
          Index Fb = face_block[tf::circular_increment(r, K)];
          const auto &face_a = loops[Fa];
          const auto &face_b = loops[Fb];

          Index sa = directed_edge_id(vi, vj, face_a) == Index(face_a.size())
                         ? 1
                         : 0;
          Index sb = directed_edge_id(vi, vj, face_b) == Index(face_b.size())
                         ? 1
                         : 0;
          sb ^= 1;

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
