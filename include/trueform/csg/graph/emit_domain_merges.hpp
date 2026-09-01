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
#include "../../core/range.hpp"
#include "../../core/views/offset_block_range.hpp"
#include "../../core/views/sequence_range.hpp"
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
///                        the stream.
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

  auto dirs_view = tf::make_offset_block_range(fans.rows.offsets_buffer(),
                                               tf::make_range(fans.dirs));

  tf::generic_generate(
      tf::make_sequence_range(Index(fans.pieces.size())),
      std::tie(merges, bundle_merges),
      [&](Index fan, auto &out_buffers, local_state_t &state) {
        if (!is_valid[std::size_t(fan)])
          return;
        const Index first = fans.page_offsets[std::size_t(fan)];
        const Index K = fans.page_offsets[std::size_t(fan) + 1] - first;

        auto &out_merges = std::get<0>(out_buffers);
        auto &out_bundles = std::get<1>(out_buffers);

        for (Index r = 0; r < K; ++r) {
          const Index rn = tf::circular_increment(r, K);
          auto page = fans.rows[std::size_t(first + r)];
          auto page_dirs = dirs_view[std::size_t(first + r)];
          auto next = fans.rows[std::size_t(first + rn)];
          auto next_dirs = dirs_view[std::size_t(first + rn)];

          for (std::size_t a = 0; a < page.size(); ++a) {
            const Index Fa = exposed_of_row[std::size_t(page[a] / Index(3))];
            if (Fa == Index(-1))
              continue; // an occurrence the exposure did not carry
            // Sides come from the occurrence's own traversal direction —
            // a slit carrier's two pages take opposite sides, which a
            // per-carrier containment test cannot express.
            const Index sa = page_dirs[a] ? Index(0) : Index(1);
            const Index frag_a = labels[Fa];

            for (std::size_t b = 0; b < next.size(); ++b) {
              const Index Fb = exposed_of_row[std::size_t(next[b] / Index(3))];
              if (Fb == Index(-1))
                continue;
              const Index sb =
                  (next_dirs[b] ? Index(0) : Index(1)) ^ Index(1);
              const Index frag_b = labels[Fb];
              const Index node_a = 2 * frag_a + sa;
              const Index node_b = 2 * frag_b + sb;
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
          }
        }
      },
          local_state_t{});

  tbb::parallel_sort(merges.begin(), merges.end());
  merges.erase_till_end(std::unique(merges.begin(), merges.end()));
  tbb::parallel_sort(bundle_merges.begin(), bundle_merges.end());
  bundle_merges.erase_till_end(
      std::unique(bundle_merges.begin(), bundle_merges.end()));
}

} // namespace tf::csg::graph
