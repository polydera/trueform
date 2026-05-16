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
#include "../../core/algorithm/circular_increment.hpp"
#include "../../core/algorithm/generic_generate.hpp"
#include "../../core/array_hash.hpp"
#include "../../core/buffer.hpp"
#include "../../core/hash_set.hpp"
#include "../../core/views/enumerate.hpp"
#include "../directed_edge_id_in_face.hpp"
#include "tbb/parallel_sort.h"
#include <algorithm>
#include <array>

namespace tf::topology::domains {

/// @ingroup topology_components
/// @brief For each valid NM edge, walk its incident faces cyclically in
/// the canonical (majority) order written by Phase D, and emit:
///   1. one fragment-side merge pair per cyclic-adjacent face pair to
///      `merges` (drives the domain partition);
///   2. one fragment-only merge pair per cyclic-adjacent face pair to
///      `bundle_merges` (drives the 3D-connected-component / bundle
///      partition).
///
/// The fragment-side merge encodes "these two fragment-sides bound the
/// same volumetric wedge". The fragment-only merge encodes "these two
/// fragments touch at this non-manifold edge", which is the union-find
/// primitive for 3D-connected bundles.
///
/// Both buffers receive unordered `{min, max}` pairs.
template <typename Polygons, typename FragLabels, typename Edges,
          typename Faces, typename Index>
void emit_domain_merges(const Polygons &polygons,
                        const FragLabels &fragment_labels,
                        const Edges &nm_edges, const Faces &nm_edge_faces,
                        const tf::buffer<char> &is_valid,
                        tf::buffer<std::array<Index, 2>> &merges,
                        tf::buffer<std::array<Index, 2>> &bundle_merges) {
  struct local_state_t {
    tf::hash_set<std::array<Index, 2>, tf::array_hash<Index, 2>> seen;
    tf::hash_set<std::array<Index, 2>, tf::array_hash<Index, 2>> bundle_seen;
  };
  tf::generic_generate(
      tf::enumerate(nm_edges), std::tie(merges, bundle_merges), local_state_t{},
      [&](const auto &pair, auto &out_buffers, local_state_t &state) {
        const auto &[k, edge] = pair;
        if (!is_valid[k])
          return;
        Index i = edge[0];
        Index j = edge[1];
        auto face_block = nm_edge_faces[k];
        Index K = Index(face_block.size());

        auto &out_merges = std::get<0>(out_buffers);
        auto &out_bundles = std::get<1>(out_buffers);

        for (Index r = 0; r < K; ++r) {
          Index Fa = face_block[r];
          Index Fb = face_block[tf::circular_increment(r, K)];
          const auto &face_a = polygons.faces()[Fa];
          const auto &face_b = polygons.faces()[Fb];

          Index sa = tf::directed_edge_id_in_face(i, j, face_a) ==
                             Index(face_a.size())
                         ? 1
                         : 0;
          Index sb = tf::directed_edge_id_in_face(i, j, face_b) ==
                             Index(face_b.size())
                         ? 1
                         : 0;
          sb ^= 1;

          Index frag_a = fragment_labels.labels[Fa];
          Index frag_b = fragment_labels.labels[Fb];
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

} // namespace tf::topology::domains
