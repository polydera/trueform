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
#include "../../../../../core/algorithm/parallel_for_each.hpp"
#include "../../../../../core/buffer.hpp"
#include "../../../../../core/views/sequence_range.hpp"
#include <cstddef>

namespace tf::topology::cdt::dt3 {

/// Claim for patch `p`, in `patch_of`, the piece of its range holding
/// `seeds[p]`: breadth first from the seed through `applier(c, f)`, which
/// calls `f` on each neighbour of carrier `c`, over the carriers
/// `range_of(c)` names `p` that `patch_of` holds unclaimed (`-2`). The
/// ranges are disjoint, so every patch walks in parallel on its own
/// carriers and the claims are the same at any worker count.
template <typename Seeds, typename RangeOf, typename Applier, typename Index>
auto claim_tet_patch_ranges(const Seeds &seeds, const RangeOf &range_of,
                            const Applier &applier, tf::buffer<Index> &patch_of)
    -> void {
  struct local_t {
    tf::buffer<Index> queue;
  };
  tf::parallel_for_each(
      tf::make_sequence_range(seeds.size()),
      [&seeds, &range_of, &applier, &patch_of](std::size_t p, local_t &local) {
        auto &queue = local.queue;
        queue.clear();
        const Index seed = Index(seeds[p]);
        patch_of[std::size_t(seed)] = Index(p);
        queue.push_back(seed);
        for (std::size_t at = 0; at < queue.size(); ++at)
          applier(queue[at], [&](Index u) {
            if (patch_of[std::size_t(u)] == Index(-2) &&
                Index(range_of(u)) == Index(p)) {
              patch_of[std::size_t(u)] = Index(p);
              queue.push_back(u);
            }
          });
      },
      local_t{});
}

} // namespace tf::topology::cdt::dt3
