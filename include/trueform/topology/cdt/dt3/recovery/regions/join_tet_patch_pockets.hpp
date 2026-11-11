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
#include "../../../../../core/algorithm/parallel_fill.hpp"
#include "../../../../../core/algorithm/sequenced_generate.hpp"
#include "../../../../../core/buffer.hpp"
#include "../../../../../core/checked.hpp"
#include <array>
#include <cstddef>

namespace tf::topology::cdt::dt3 {

/// Join every carrier of `order` that `patch_of` holds unclaimed (`-2`) — a
/// pocket, the parts of a range its seed's piece does not reach — to the
/// claimed patch nearest it through `applier(c, f)`, in hops, the lower
/// patch on a tie, whatever that patch's size: so every patch stays
/// connected. A piece of the surface no claim reaches takes, whole, the
/// range `range_of` gives its first carrier along `order`. The pockets'
/// borders are found in parallel and in `order`, the hops walked serially
/// over the pockets alone; `queue` and `hops` are the pass's scratch.
template <typename Order, typename RangeOf, typename Applier, typename Index>
auto join_tet_patch_pockets(const Order &order, const RangeOf &range_of,
                            const Applier &applier, tf::buffer<Index> &patch_of,
                            tf::buffer<std::array<Index, 2>> &border,
                            tf::buffer<Index> &queue, tf::buffer<Index> &hops)
    -> void {
  border.clear();
  tf::sequenced_generate(
      order, border,
      [&applier, &patch_of](Index c, auto &out) {
        if (patch_of[std::size_t(c)] != Index(-2))
          return;
        Index nearest = Index(-1);
        applier(c, [&patch_of, &nearest](Index u) {
          const Index q = patch_of[std::size_t(u)];
          if (q >= 0 && (nearest < 0 || q < nearest))
            nearest = q;
        });
        if (nearest >= 0)
          out.push_back(std::array<Index, 2>{c, nearest});
      },
      tf::checked);
  hops.allocate(patch_of.size());
  tf::parallel_fill(hops, Index(0));
  queue.clear();
  for (const auto &[c, p] : border) {
    patch_of[std::size_t(c)] = p;
    hops[std::size_t(c)] = Index(1);
    queue.push_back(c);
  }
  std::size_t at = 0, scan = 0;
  for (;;) {
    for (; at < queue.size(); ++at) {
      const Index c = queue[at];
      const Index p = patch_of[std::size_t(c)];
      const Index next = hops[std::size_t(c)] + 1;
      applier(c, [&](Index u) {
        Index &q = patch_of[std::size_t(u)];
        if (q == Index(-2)) {
          q = p;
          hops[std::size_t(u)] = next;
          queue.push_back(u);
        } else if (q > p && hops[std::size_t(u)] == next)
          q = p;
      });
    }
    while (scan < order.size() &&
           patch_of[std::size_t(order[scan])] != Index(-2))
      ++scan;
    if (scan == order.size())
      return;
    const Index c = Index(order[scan]);
    patch_of[std::size_t(c)] = Index(range_of(c));
    hops[std::size_t(c)] = Index(1);
    queue.push_back(c);
  }
}

} // namespace tf::topology::cdt::dt3
