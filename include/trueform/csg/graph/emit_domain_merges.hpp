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
#include "../../core/algorithm/generic_generate.hpp"
#include "../../core/array_hash.hpp"
#include "../../core/buffer.hpp"
#include "../../core/hash_set.hpp"
#include "../../core/range.hpp"
#include "../../core/views/sequence_range.hpp"
#include "./for_each_ring_sector.hpp"
#include "./make_plane_radial_fans.hpp"
#include "tbb/parallel_sort.h"
#include <algorithm>
#include <array>
#include <cstddef>
#include <tuple>

namespace tf::csg::graph {

/// @ingroup csg_graph_internals
/// @brief For each radial fan, walk its pages cyclically and emit:
///   1. one component-side merge pair per occurrence pair of two
///      cyclic-adjacent pages to `merges` (drives the domain partition);
///   2. one component-only merge pair per such occurrence pair to
///      `bundle_merges` (drives the 3D-connected-component partition).
///
/// A ring its emission does not certify emits the second alone.
///
/// The fan is the arrangement's: one per fan piece, its pages — one
/// carrier plane on one side — already in radial order, each carrying the
/// occurrences that sit on it with their traversal bits. Two adjacent
/// pages bound one sector, so every occurrence of the one meets every
/// occurrence of the other there. Sides come from those bits — a slit
/// carrier's two pages take opposite sides, which a per-carrier
/// containment test cannot express.
///
/// @param labels          Per exposed triangle: its component.
/// @param exposed_of_row  Arrangement row -> exposed triangle; an
///                        occurrence names a row, the partition names
///                        the stream. An uncut sheet's occurrence is
///                        negative — `-(component + 1)` — and states
///                        its component directly, since no stream row
///                        names it.
template <typename Index, typename Labels, typename ExposedOfRow>
void emit_domain_merges(const Labels &labels,
                        const tf::csg::graph::plane_radial_fans<Index> &fans,
                        const ExposedOfRow &exposed_of_row,
                        const tf::buffer<char> &is_valid,
                        tf::buffer<std::array<Index, 2>> &merges,
                        tf::buffer<std::array<Index, 2>> &bundle_merges) {
  struct local_state_t {
    tf::hash_set<std::array<Index, 2>, tf::array_hash<Index, 2>> seen;
    tf::hash_set<std::array<Index, 2>, tf::array_hash<Index, 2>> bundle_seen;
  };

  tf::generic_generate(
      tf::make_sequence_range(Index(fans.pieces.size())),
      std::tie(merges, bundle_merges),
      [&](Index fan, auto &out_buffers, local_state_t &state) {
        if (!is_valid[std::size_t(fan)])
          return;
        auto &out_merges = std::get<0>(out_buffers);
        auto &out_bundles = std::get<1>(out_buffers);
        // an uncertified ring states that its components touch, not which
        // of their sides face each other
        const bool certified = fans.verdicts[std::size_t(fan)] ==
                               plane_radial_ring_verdict::certified;
        for_each_ring_sector(
            labels, fans, exposed_of_row, fan,
            [&](Index node_a, Index node_b, Index frag_a, Index frag_b) {
              std::array<Index, 2> p = {std::min(node_a, node_b),
                                        std::max(node_a, node_b)};
              if (certified && state.seen.insert(p).second)
                out_merges.push_back(p);
              if (frag_a != frag_b) {
                std::array<Index, 2> bp = {std::min(frag_a, frag_b),
                                           std::max(frag_a, frag_b)};
                if (state.bundle_seen.insert(bp).second)
                  out_bundles.push_back(bp);
              }
            });
      },
      local_state_t{});

  tbb::parallel_sort(merges.begin(), merges.end());
  merges.erase_till_end(std::unique(merges.begin(), merges.end()));
  tbb::parallel_sort(bundle_merges.begin(), bundle_merges.end());
  bundle_merges.erase_till_end(
      std::unique(bundle_merges.begin(), bundle_merges.end()));
}

} // namespace tf::csg::graph
